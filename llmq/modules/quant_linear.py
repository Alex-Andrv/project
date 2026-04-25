from __future__ import annotations

import torch
import torch.nn as nn
from llmq.kernels.int4_quantize import QuantizedInt4Weight, quantize_fp16_to_int4
from llmq.kernels.w4a16_matmul import w4a16_linear_bf16


class QuantLinearW4A16(nn.Module):
    """Quantized linear layer with packed int4 weights and optional dense bias."""

    def __init__(
        self,
        qweight: QuantizedInt4Weight,
        bias: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.group_size = qweight.group_size
        self.out_features, self.in_features = qweight.orig_shape

        self.register_buffer("packed", qweight.packed)
        self.register_buffer("scales", qweight.scales)

        if bias is not None:
            self.register_buffer("bias", bias.contiguous())
        else:
            self.bias = None

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        group_size: int = 128,
        prefer_triton_quant: bool = False,
    ) -> QuantLinearW4A16:
        qweight = quantize_fp16_to_int4(
            linear.weight.data,
            group_size=group_size,
            prefer_triton=prefer_triton_quant,
        )
        bias = linear.bias.data if linear.bias is not None else None
        return cls(qweight=qweight, bias=bias)

    def _qweight(self) -> QuantizedInt4Weight:
        return QuantizedInt4Weight(
            packed=self.packed,
            scales=self.scales,
            group_size=self.group_size,
            orig_shape=(self.out_features, self.in_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        assert orig_shape[-1] == self.in_features

        x2d = x.reshape(-1, self.in_features)
        if x2d.dtype != torch.bfloat16:
            x2d = x2d.to(torch.bfloat16)

        y2d = w4a16_linear_bf16(x2d, self._qweight())

        if self.bias is not None:
            y2d = y2d + self.bias.to(y2d.dtype)

        return y2d.reshape(*orig_shape[:-1], self.out_features)


def replace_linear_with_w4a16(
    module: nn.Module,
    *,
    group_size: int = 128,
    prefer_triton_quant: bool = True,
    prefix: str = "",
) -> dict[str, int]:
    """
    Recursively replace eligible nn.Linear modules with QuantLinearW4A16.

    Returns simple stats useful for benchmark logs.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("QuantLinearW4A16 requires CUDA tensors.")

    stats = {"replaced": 0, "skipped": 0, "params_replaced": 0}

    for child_name, child in list(module.named_children()):
        full_name = f"{prefix}.{child_name}" if prefix else child_name
        if isinstance(child, QuantLinearW4A16):
            stats["skipped"] += 1
            continue

        if isinstance(child, nn.Linear):
            if child_name == "lm_head" or child.in_features % 2 != 0:
                stats["skipped"] += 1
                continue

            quant = QuantLinearW4A16.from_linear(
                child,
                group_size=group_size,
                prefer_triton_quant=prefer_triton_quant,
            )
            setattr(module, child_name, quant)
            stats["replaced"] += 1
            stats["params_replaced"] += child.weight.numel()
            continue

        child_stats = replace_linear_with_w4a16(
            child,
            group_size=group_size,
            prefer_triton_quant=prefer_triton_quant,
            prefix=full_name,
        )
        for key, value in child_stats.items():
            stats[key] += value

    return stats
