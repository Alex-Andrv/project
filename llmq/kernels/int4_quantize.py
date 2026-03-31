"""
FP16/BF16 -> symmetric int4 with per-group scales; pack 2 nibbles per uint8.
Веса fp16: N*K*2 байт; packed uint8: N*(K//2) — в 4 раза меньше по сырым весам
(без учёта scales).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@triton.jit
def _quantize_pack_row_kernel(
    w_ptr,
    w_stride_m,
    w_stride_k,
    packed_ptr,
    packed_stride_m,
    packed_stride_k,
    scales_ptr,
    scales_stride_m,
    scales_stride_g,
    n_cols,
    GROUP_SIZE: tl.constexpr,
    HALF_GROUP_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    g = tl.program_id(1)

    k0 = g * GROUP_SIZE
    offs = tl.arange(0, GROUP_SIZE)
    mask_k = (k0 + offs) < n_cols

    w = tl.load(
        w_ptr + row * w_stride_m + (k0 + offs) * w_stride_k,
        mask=mask_k,
        other=0.0,
    ).to(tl.float32)

    absmax = tl.max(tl.abs(w))
    scale_safe = tl.where(absmax > 1e-8, absmax / 7.0, 1.0)

    idx_pair = tl.arange(0, HALF_GROUP_SIZE)
    k_a = k0 + idx_pair * 2
    k_b = k0 + idx_pair * 2 + 1

    mask_a = k_a < n_cols
    mask_b = k_b < n_cols

    w0 = tl.load(
        w_ptr + row * w_stride_m + k_a * w_stride_k,
        mask=mask_a,
        other=0.0,
    ).to(tl.float32)
    w1 = tl.load(
        w_ptr + row * w_stride_m + k_b * w_stride_k,
        mask=mask_b,
        other=0.0,
    ).to(tl.float32)

    q0f = tl.maximum(-8.0, tl.minimum(7.0, tl.extra.cuda.libdevice.round(w0 / scale_safe)))
    q1f = tl.maximum(-8.0, tl.minimum(7.0, tl.extra.cuda.libdevice.round(w1 / scale_safe)))

    b0 = (q0f.to(tl.int32) + 8) & 0xF
    b1 = (q1f.to(tl.int32) + 8) & 0xF
    byte = (b0 | (b1 << 4)).to(tl.uint8)

    pack_col = (k0 // 2) + idx_pair

    tl.store(
        scales_ptr + row * scales_stride_m + g * scales_stride_g,
        scale_safe.to(tl.float32),
    )
    tl.store(
        packed_ptr + row * packed_stride_m + pack_col * packed_stride_k,
        byte,
        mask=mask_a,
    )


def _quantize_triton(w: torch.Tensor, group_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    assert w.is_cuda
    assert w.dim() == 2
    n_rows, k = w.shape
    assert k % 2 == 0
    assert group_size > 0 and group_size % 2 == 0

    num_groups = (k + group_size - 1) // group_size
    w_c = w.contiguous()
    if w_c.dtype not in (torch.float16, torch.bfloat16):
        w_c = w_c.to(torch.float16)

    packed = torch.empty((n_rows, k // 2), device=w.device, dtype=torch.uint8)
    scales = torch.empty((n_rows, num_groups), device=w.device, dtype=torch.float32)

    grid = (n_rows, num_groups)
    _quantize_pack_row_kernel[grid](
        w_c,
        w_c.stride(0),
        w_c.stride(1),
        packed,
        packed.stride(0),
        packed.stride(1),
        scales,
        scales.stride(0),
        scales.stride(1),
        k,
        GROUP_SIZE=group_size,
        HALF_GROUP_SIZE=group_size // 2,
        num_warps=4,
    )
    return packed, scales


def _quantize_torch_reference(w: torch.Tensor, group_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Векторизованный torch-reference без Python-циклов по элементам.
    """
    assert w.dim() == 2
    n_rows, k = w.shape
    assert k % 2 == 0
    assert group_size > 0 and group_size % 2 == 0

    wf = w.float()
    num_groups = (k + group_size - 1) // group_size
    padded_k = num_groups * group_size
    pad = padded_k - k

    if pad > 0:
        wf_pad = torch.nn.functional.pad(wf, (0, pad))
    else:
        wf_pad = wf

    wg = wf_pad.view(n_rows, num_groups, group_size)  # (N, G, GS)

    absmax = wg.abs().amax(dim=-1).clamp_min(1e-8)
    scales = absmax / 7.0

    q = torch.round(wg / scales.unsqueeze(-1)).clamp(-8, 7).to(torch.int32)
    q = q.view(n_rows, padded_k)[:, :k]

    q_off = q + 8
    q_even = q_off[:, 0::2]
    q_odd = q_off[:, 1::2]

    packed = ((q_even & 0xF) | ((q_odd & 0xF) << 4)).to(torch.uint8)
    return packed.contiguous(), scales.contiguous()


@dataclass
class QuantizedInt4Weight:
    packed: torch.Tensor
    scales: torch.Tensor
    group_size: int
    orig_shape: tuple[int, int]

    def storage_bytes_weights_only(self) -> int:
        return self.packed.numel() + self.scales.numel() * 4


def quantize_fp16_to_int4(
    w: torch.Tensor,
    group_size: int = 128,
    prefer_triton: bool = False,
) -> QuantizedInt4Weight:
    """
    W: (N, K) как в nn.Linear: out_features × in_features.
    """
    assert w.dim() == 2
    orig = tuple(w.shape)
    wc = w.contiguous()

    if not wc.is_cuda:
        packed, scales = _quantize_torch_reference(wc, group_size)
    elif prefer_triton:
        packed, scales = _quantize_triton(wc, group_size)
    else:
        packed, scales = _quantize_torch_reference(wc, group_size)

    return QuantizedInt4Weight(
        packed=packed,
        scales=scales,
        group_size=group_size,
        orig_shape=orig,
    )