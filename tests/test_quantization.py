import pytest
import torch

from tensormesh.network.quant import dequantize, quantize


def test_fp16_round_trip_is_lossless_for_fp16_input():
    x = torch.tensor([[1.5, -2.25, 0.0], [65504.0, -0.001, 3.0]], dtype=torch.float16)

    qt = quantize(x, "FP16")

    assert qt.dtype == "FP16"
    assert qt.shape == [2, 3]
    assert torch.equal(dequantize(qt), x)


def test_int8_uses_per_token_absmax_scale():
    # Token 0: absmax 254 -> scale 2.0; -127/2 = -63.5 and 1/2 = 0.5 round half-to-even.
    # Token 1: absmax 12.7 -> scale 0.1; every value lands on the grid exactly.
    x = torch.tensor([[254.0, -127.0, 1.0], [12.7, -6.3, 0.0]])

    qt = quantize(x, "INT8")

    assert qt.dtype == "INT8"
    assert len(qt.tensor_data) == 6  # one byte per element
    expected = torch.tensor([[254.0, -128.0, 0.0], [12.7, -6.3, 0.0]])
    torch.testing.assert_close(dequantize(qt).float(), expected, atol=1e-2, rtol=0)


@pytest.mark.parametrize("dtype", ["INT8", "FP8_E4M3"])
def test_all_zero_token_round_trips_to_zero(dtype):
    x = torch.zeros(2, 4)

    out = dequantize(quantize(x, dtype))

    assert torch.equal(out.float(), x)


def test_fp8_e4m3_scales_each_token_to_the_format_max_of_448():
    # Token 0: absmax 448 -> scale 1.0. Token 1: absmax 896 -> scale 2.0, so 3 -> 1.5.
    # All scaled values are exactly representable in E4M3, so the round trip is exact.
    x = torch.tensor([[448.0, 1.0, -224.0], [896.0, 3.0, -0.0]])

    qt = quantize(x, "FP8_E4M3")

    assert qt.dtype == "FP8_E4M3"
    assert len(qt.tensor_data) == 6  # one byte per element
    assert torch.equal(dequantize(qt).float(), x)


@pytest.mark.parametrize("dtype, atol", [("FP16", 1e-3), ("FP8_E4M3", 0.15), ("INT8", 0.05)])
def test_activation_survives_an_activation_frame_round_trip(dtype, atol):
    from tensormesh.network.proto import protocol_pb2
    from tensormesh.network.quant import QuantizedTensor

    torch.manual_seed(0)
    x = torch.randn(2, 5, 64, dtype=torch.float16)  # [batch, seq, hidden]

    qt = quantize(x, dtype)
    wire = protocol_pb2.ActivationFrame(
        session_id="s", shape=qt.shape, dtype=qt.dtype,
        tensor_data=qt.tensor_data, quant_scales=qt.quant_scales,
    ).SerializeToString()
    frame = protocol_pb2.ActivationFrame.FromString(wire)
    out = dequantize(QuantizedTensor(list(frame.shape), frame.dtype, frame.tensor_data, frame.quant_scales))

    assert out.shape == x.shape
    torch.testing.assert_close(out, x, atol=atol, rtol=0.07)


def test_unknown_dtype_is_rejected():
    with pytest.raises(ValueError, match="unsupported activation dtype"):
        quantize(torch.ones(1, 2), "FP4")
