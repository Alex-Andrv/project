from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from llmq.kernels.bf16_matmul import bf16_linear_triton
from llmq.kernels.int4_quantize import quantize_fp16_to_int4
from llmq.kernels.w4a16_matmul import (
    dequantize_int4_weight_torch,
    dequantize_int4_weight_triton,
    w4a16_linear_bf16,
)
from llmq.modules.quant_linear import QuantLinearW4A16, replace_linear_with_w4a16

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton kernels")


SMALL_LLAMA_LINEAR_SPECS = [
    ("q_proj", 64, 64),
    ("k_proj", 16, 64),
    ("v_proj", 16, 64),
    ("o_proj", 64, 64),
    ("gate_proj", 128, 64),
    ("up_proj", 128, 64),
    ("down_proj", 64, 128),
]

@pytest.fixture(autouse=True)
def _seed() -> None:
    torch.manual_seed(0)


@pytest.mark.parametrize(("layer_name", "n_out", "n_in"), SMALL_LLAMA_LINEAR_SPECS)
def test_w4a16_kernel_matches_torch_dequant_for_all_llama_layer_types(
    layer_name: str,
    n_out: int,
    n_in: int,
) -> None:
    del layer_name
    device = torch.device("cuda")
    group_size = 16
    x = torch.randn(9, n_in, device=device, dtype=torch.bfloat16)
    w = torch.randn(n_out, n_in, device=device, dtype=torch.bfloat16) * 0.02

    qw = quantize_fp16_to_int4(w, group_size=group_size, prefer_triton=True)
    y_triton = w4a16_linear_bf16(x, qw)
    y_ref = x @ dequantize_int4_weight_torch(qw, dtype=torch.bfloat16).T

    torch.testing.assert_close(y_triton, y_ref, atol=1e-2, rtol=1e-2)
    assert qw.storage_bytes_weights_only() < w.numel() * w.element_size()


@pytest.mark.parametrize(("layer_name", "n_out", "n_in"), SMALL_LLAMA_LINEAR_SPECS)
def test_quant_linear_matches_fused_kernel_for_all_llama_layer_types(
    layer_name: str,
    n_out: int,
    n_in: int,
) -> None:
    del layer_name
    device = torch.device("cuda")
    linear = nn.Linear(n_in, n_out, bias=True, dtype=torch.bfloat16, device=device)
    x = torch.randn(2, 5, n_in, device=device, dtype=torch.bfloat16)

    quant = QuantLinearW4A16.from_linear(linear, group_size=16, prefer_triton_quant=True)
    y_layer = quant(x)
    y_kernel = w4a16_linear_bf16(x.reshape(-1, n_in), quant._qweight())
    y_kernel = (y_kernel + linear.bias.to(y_kernel.dtype)).reshape(2, 5, n_out)

    torch.testing.assert_close(y_layer, y_kernel, atol=0, rtol=0)


def test_bf16_triton_matmul_matches_torch() -> None:
    device = torch.device("cuda")
    x = torch.randn(17, 96, device=device, dtype=torch.bfloat16)
    w = torch.randn(80, 96, device=device, dtype=torch.bfloat16)

    y_triton = bf16_linear_triton(x, w)
    y_torch = x @ w.T

    torch.testing.assert_close(y_triton, y_torch, atol=2e-2, rtol=2e-2)


def test_triton_dequant_matches_torch_dequant() -> None:
    device = torch.device("cuda")
    w = torch.randn(80, 96, device=device, dtype=torch.bfloat16) * 0.02
    qw = quantize_fp16_to_int4(w, group_size=16, prefer_triton=True)

    w_triton = dequantize_int4_weight_triton(qw)
    w_torch = dequantize_int4_weight_torch(qw, dtype=torch.bfloat16)

    torch.testing.assert_close(w_triton, w_torch, atol=0, rtol=0)


class _TinyLlamaBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        hidden = 64
        kv = 16
        intermediate = 128
        self.q_proj = nn.Linear(hidden, hidden, bias=False, dtype=torch.bfloat16)
        self.k_proj = nn.Linear(hidden, kv, bias=False, dtype=torch.bfloat16)
        self.v_proj = nn.Linear(hidden, kv, bias=False, dtype=torch.bfloat16)
        self.o_proj = nn.Linear(hidden, hidden, bias=False, dtype=torch.bfloat16)
        self.mlp = nn.ModuleDict(
            {
                "gate_proj": nn.Linear(hidden, intermediate, bias=False, dtype=torch.bfloat16),
                "up_proj": nn.Linear(hidden, intermediate, bias=False, dtype=torch.bfloat16),
                "down_proj": nn.Linear(intermediate, hidden, bias=False, dtype=torch.bfloat16),
            }
        )
        self.lm_head = nn.Linear(hidden, 256, bias=False, dtype=torch.bfloat16)


def test_replace_linear_with_w4a16_replaces_all_llama_layer_types_and_skips_lm_head() -> None:
    model = _TinyLlamaBlock().cuda()

    stats = replace_linear_with_w4a16(
        model,
        group_size=16,
        prefer_triton_quant=True,
    )

    assert stats["replaced"] == 7
    assert stats["skipped"] == 1
    assert isinstance(model.q_proj, QuantLinearW4A16)
    assert isinstance(model.k_proj, QuantLinearW4A16)
    assert isinstance(model.v_proj, QuantLinearW4A16)
    assert isinstance(model.o_proj, QuantLinearW4A16)
    assert isinstance(model.mlp["gate_proj"], QuantLinearW4A16)
    assert isinstance(model.mlp["up_proj"], QuantLinearW4A16)
    assert isinstance(model.mlp["down_proj"], QuantLinearW4A16)
    assert isinstance(model.lm_head, nn.Linear)
