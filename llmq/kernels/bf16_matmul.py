"""Tuned Triton BF16 baseline: Y = X @ W^T."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_BF16_AUTOTUNE_RAW: list[tuple[int, int, int, int, int]] = [
    (32, 32, 64, 4, 3),
    (32, 64, 64, 4, 3),
    (32, 64, 128, 4, 3),
    (64, 32, 64, 4, 3),
    (64, 64, 64, 4, 3),
    (64, 64, 128, 4, 3),
    (64, 128, 64, 4, 3),
    (64, 128, 128, 4, 3),
    (128, 32, 64, 4, 3),
    (128, 64, 64, 4, 3),
    (128, 64, 128, 4, 3),
    (128, 128, 64, 4, 3),
    (128, 128, 128, 4, 3),
]

_BF16_AUTOTUNE_CONFIGS = [
    triton.Config(
        {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk},
        num_warps=nw,
        num_stages=ns,
    )
    for bm, bn, bk, nw, ns in _BF16_AUTOTUNE_RAW
]


@triton.autotune(configs=_BF16_AUTOTUNE_CONFIGS, key=["M", "N", "K"])
@triton.jit
def _bf16_matmul_kernel(
    x_ptr,
    x_stride_m,
    x_stride_k,
    w_ptr,
    w_stride_n,
    w_stride_k,
    y_ptr,
    y_stride_m,
    y_stride_n,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    num_k_blocks = tl.cdiv(K, BLOCK_K)
    for kb in range(0, num_k_blocks):
        k0 = kb * BLOCK_K
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        x = tl.load(
            x_ptr + offs_m[:, None] * x_stride_m + offs_k[None, :] * x_stride_k,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )
        w = tl.load(
            w_ptr + offs_n[:, None] * w_stride_n + offs_k[None, :] * w_stride_k,
            mask=mask_n[:, None] & mask_k[None, :],
            other=0.0,
        )
        acc += tl.dot(x, tl.trans(w), out_dtype=tl.float32)

    tl.store(
        y_ptr + offs_m[:, None] * y_stride_m + offs_n[None, :] * y_stride_n,
        acc.to(tl.bfloat16),
        mask=mask_m[:, None] & mask_n[None, :],
    )


def bf16_linear_triton(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Return X @ W.T for contiguous BF16 tensors."""
    assert x.is_cuda and w.is_cuda
    assert x.dtype == torch.bfloat16 and w.dtype == torch.bfloat16
    assert x.dim() == 2 and w.dim() == 2
    m, k = x.shape
    n, kw = w.shape
    assert k == kw

    x_c = x.contiguous()
    w_c = w.contiguous()
    y = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)

    def grid(meta):
        return (triton.cdiv(m, meta["BLOCK_M"]), triton.cdiv(n, meta["BLOCK_N"]))

    _bf16_matmul_kernel[grid](
        x_c,
        x_c.stride(0),
        x_c.stride(1),
        w_c,
        w_c.stride(0),
        w_c.stride(1),
        y,
        y.stride(0),
        y.stride(1),
        m,
        n,
        k,
    )
    return y
