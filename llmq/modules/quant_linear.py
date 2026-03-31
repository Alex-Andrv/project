from __future__ import annotations

import torch
import torch.nn as nn

from llmq.kernels.int4_quantize import QuantizedInt4Weight, quantize_fp16_to_int4
from llmq.kernels.w4a16_matmul import w4a16_linear_bf16


class QuantLinearW4A16(nn.Module):
    """
    Простейший quantized linear layer.
    Веса хранятся как int4+scales, bias опционально остаётся dense.
    """

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
    ) -> "QuantLinearW4A16":
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