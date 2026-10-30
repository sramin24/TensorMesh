from dataclasses import dataclass

import numpy as np
import torch

INT8_MAX = 127.0
FP8_E4M3_MAX = 448.0


@dataclass
class QuantizedTensor:
    shape: list[int]
    dtype: str
    tensor_data: bytes
    quant_scales: bytes = b""


def _to_bytes(t: torch.Tensor) -> bytes:
    # numpy has no float8, so reinterpret 1-byte types as raw uint8 before export.
    if t.element_size() == 1:
        t = t.view(torch.uint8)
    return t.contiguous().numpy().tobytes()


def _from_bytes(buf: bytes, np_dtype: type, shape: list[int]) -> torch.Tensor:
    return torch.from_numpy(np.frombuffer(buf, dtype=np_dtype).reshape(shape).copy())


def _per_token_scales(xf: torch.Tensor, qmax: float) -> torch.Tensor:
    # One float32 scale per row of the hidden dimension. An all-zero token (e.g.
    # padding) would give scale 0 and 0/0 = NaN, so fall back to 1.0 there.
    scales = xf.abs().amax(dim=-1, keepdim=True) / qmax
    return torch.where(scales > 0, scales, torch.ones_like(scales))


def quantize(x: torch.Tensor, dtype: str) -> QuantizedTensor:
    """Encode an activation tensor into the byte layout ActivationFrame carries."""
    x = x.detach().to("cpu")
    shape = list(x.shape)
    if dtype == "FP16":
        return QuantizedTensor(shape, dtype, _to_bytes(x.to(torch.float16)))

    xf = x.to(torch.float32)
    if dtype == "INT8":
        scales = _per_token_scales(xf, INT8_MAX)
        q = torch.round(xf / scales).clamp(-INT8_MAX, INT8_MAX).to(torch.int8)
    elif dtype == "FP8_E4M3":
        scales = _per_token_scales(xf, FP8_E4M3_MAX)
        q = (xf / scales).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    else:
        raise ValueError(f"unsupported activation dtype: {dtype!r}")
    return QuantizedTensor(shape, dtype, _to_bytes(q), _to_bytes(scales))


def dequantize(qt: QuantizedTensor) -> torch.Tensor:
    """Decode a QuantizedTensor back to an FP16 activation tensor."""
    if qt.dtype == "FP16":
        return _from_bytes(qt.tensor_data, np.float16, qt.shape)

    if qt.dtype == "INT8":
        q = _from_bytes(qt.tensor_data, np.int8, qt.shape)
    elif qt.dtype == "FP8_E4M3":
        q = _from_bytes(qt.tensor_data, np.uint8, qt.shape).view(torch.float8_e4m3fn)
    else:
        raise ValueError(f"unsupported activation dtype: {qt.dtype!r}")
    scales = _from_bytes(qt.quant_scales, np.float32, [*qt.shape[:-1], 1])
    return (q.to(torch.float32) * scales).to(torch.float16)
