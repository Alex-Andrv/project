"""Y = X @ W^T, X — bf16 (M,K), W — int4 packed row-wise as in quantize."""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from .int4_quantize import QuantizedInt4Weight


@triton.jit
def _w4a16_matmul_kernel(
    x_ptr,
    x_stride_m,
    x_stride_k,
    p_ptr,
    p_stride_n,
    p_stride_pk,
    s_ptr,
    s_stride_n,
    s_stride_g,
    y_ptr,
    y_stride_m,
    y_stride_n,
    M,
    N,
    K,
    GROUP_SIZE: tl.constexpr,
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
        ).to(tl.bfloat16)

        pack_idx = offs_k // 2
        shift = ((offs_k & 1) * 4).to(tl.uint32)

        packed = tl.load(
            p_ptr + offs_n[:, None] * p_stride_n + pack_idx[None, :] * p_stride_pk,
            mask=mask_n[:, None] & mask_k[None, :],
            other=0,
        ).to(tl.uint32)

        nib = (packed >> shift[None, :]) & 0xF
        q = nib.to(tl.int32) - 8

        g_idx = offs_k // GROUP_SIZE
        scales = tl.load(
            s_ptr + offs_n[:, None] * s_stride_n + g_idx[None, :] * s_stride_g,
            mask=mask_n[:, None] & mask_k[None, :],
            other=0.0,
        ).to(tl.bfloat16)

        w = q.to(tl.bfloat16) * scales
        acc += tl.dot(x, tl.trans(w), out_dtype=tl.float32)

    tl.store(
        y_ptr + offs_m[:, None] * y_stride_m + offs_n[None, :] * y_stride_n,
        acc.to(tl.bfloat16),
        mask=mask_m[:, None] & mask_n[None, :],
    )


def w4a16_linear_bf16(
    x: torch.Tensor,
    qw: QuantizedInt4Weight,
    *,
    BLOCK_M: int = 64,
    BLOCK_N: int = 64,
    BLOCK_K: int = 128,
) -> torch.Tensor:
    """
    x: (M, K) bf16, qw для W.shape = (N, K).
    Возвращает (M, N) bf16.
    """
    assert x.is_cuda and x.dtype == torch.bfloat16
    M, K = x.shape
    N, K_w = qw.orig_shape
    assert K == K_w
    assert BLOCK_K % 2 == 0

    y = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)
    assert qw.packed.is_cuda and qw.scales.is_cuda

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _w4a16_matmul_kernel[grid](
        x,
        x.stride(0),
        x.stride(1),
        qw.packed,
        qw.packed.stride(0),
        qw.packed.stride(1),
        qw.scales,
        qw.scales.stride(0),
        qw.scales.stride(1),
        y,
        y.stride(0),
        y.stride(1),
        M,
        N,
        K,
        GROUP_SIZE=qw.group_size,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=3,
    )
    return y


def dequantize_int4_weight_torch(
    qw: QuantizedInt4Weight,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Быстрая векторизованная распаковка int4 -> float.
    Возвращает W.shape = (N, K).
    """
    N, K = qw.orig_shape
    p = qw.packed

    low = (p & 0xF).to(torch.int16) - 8
    high = ((p >> 4) & 0xF).to(torch.int16) - 8

    q = torch.empty((N, K), device=p.device, dtype=torch.int16)
    q[:, 0::2] = low
    q[:, 1::2] = high

    scales_full = torch.repeat_interleave(qw.scales, repeats=qw.group_size, dim=1)[:, :K]
    w = q.to(torch.float32) * scales_full.to(torch.float32)

    if dtype != torch.float32:
        w = w.to(dtype)
    return w


def w4a16_matmul_reference(x_bf16: torch.Tensor, qw: QuantizedInt4Weight) -> torch.Tensor:
    """
    Быстрый reference без Python-циклов:
    dequantize -> dense matmul.
    """
    N, K = qw.orig_shape
    M, Kx = x_bf16.shape
    assert K == Kx

    w = dequantize_int4_weight_torch(qw, dtype=torch.float32)
    return x_bf16.float() @ w.T