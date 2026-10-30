import time
from collections.abc import AsyncIterable, AsyncIterator, Callable, Iterable
from dataclasses import dataclass

import grpc
import torch

from tensormesh.network.proto import protocol_pb2, protocol_pb2_grpc
from tensormesh.network.quant import QuantizedTensor, dequantize, quantize


@dataclass
class Activation:
    """One pipeline step's hidden states plus the metadata that travels with them."""

    session_id: str
    seq_step: int
    tensor: torch.Tensor
    is_prefill: bool = False
    is_last_stage: bool = False


# A pipeline stage: takes the incoming hidden states (and their metadata) and
# returns the hidden states to send back.
StageFn = Callable[[torch.Tensor, Activation], torch.Tensor]


def encode_frame(act: Activation, wire_dtype: str) -> protocol_pb2.ActivationFrame:
    qt = quantize(act.tensor, wire_dtype)
    return protocol_pb2.ActivationFrame(
        session_id=act.session_id,
        seq_step=act.seq_step,
        shape=qt.shape,
        dtype=qt.dtype,
        tensor_data=qt.tensor_data,
        quant_scales=qt.quant_scales,
        is_prefill=act.is_prefill,
        is_last_stage=act.is_last_stage,
    )


def decode_frame(frame: protocol_pb2.ActivationFrame) -> Activation:
    qt = QuantizedTensor(list(frame.shape), frame.dtype, frame.tensor_data, frame.quant_scales)
    return Activation(frame.session_id, frame.seq_step, dequantize(qt), frame.is_prefill, frame.is_last_stage)


class NodeServicer(protocol_pb2_grpc.NodeServiceServicer):
    def __init__(self, stage_fn: StageFn, cluster_id: str):
        self._stage_fn = stage_fn
        self._cluster_id = cluster_id
        self._ranks: dict[str, int] = {}

    async def Handshake(self, request, context):
        # Ranks follow join order; a node that reconnects keeps the rank it had.
        rank = self._ranks.setdefault(request.node_id, len(self._ranks))
        return protocol_pb2.HandshakeResponse(accepted=True, cluster_id=self._cluster_id, assigned_rank=rank)

    async def ProbeInterconnect(self, request, context):
        return protocol_pb2.PongPayload(
            client_timestamp_ns=request.client_timestamp_ns,
            server_timestamp_ns=time.time_ns(),
        )

    async def ForwardActivation(self, request_iterator, context):
        async for frame in request_iterator:
            act = decode_frame(frame)
            try:
                act.tensor = self._stage_fn(act.tensor, act)
            except Exception as e:
                # Tell the upstream node why its step died instead of a bare UNKNOWN.
                await context.abort(
                    grpc.StatusCode.INTERNAL,
                    f"stage failed at session={act.session_id} step={act.seq_step}: {type(e).__name__}: {e}",
                )
            # Reply in the same wire dtype the sender chose.
            yield encode_frame(act, frame.dtype)


async def start_node_server(stage_fn: StageFn, cluster_id: str, address: str = "0.0.0.0:50051"):
    """Start a NodeService on `address`. Returns (server, bound_port)."""
    server = grpc.aio.server()
    protocol_pb2_grpc.add_NodeServiceServicer_to_server(NodeServicer(stage_fn, cluster_id), server)
    port = server.add_insecure_port(address)
    await server.start()
    return server, port


class NodeClient:
    def __init__(self, address: str, wire_dtype: str = "FP16"):
        self._channel = grpc.aio.insecure_channel(address)
        self._stub = protocol_pb2_grpc.NodeServiceStub(self._channel)
        self._wire_dtype = wire_dtype

    async def probe(self, payload_bytes: int = 0) -> int:
        """Round-trip a ping of `payload_bytes` and return the RTT in nanoseconds."""
        start = time.perf_counter_ns()
        await self._stub.ProbeInterconnect(
            protocol_pb2.PingPayload(client_timestamp_ns=time.time_ns(), test_payload=bytes(payload_bytes))
        )
        return time.perf_counter_ns() - start

    async def handshake(
        self,
        node_id: str,
        device_type: str,
        total_vram_bytes: int = 0,
        available_vram_bytes: int = 0,
        compute_tflops_estimate: float = 0.0,
    ) -> protocol_pb2.HandshakeResponse:
        return await self._stub.Handshake(
            protocol_pb2.HandshakeRequest(
                node_id=node_id,
                device_type=device_type,
                total_vram_bytes=total_vram_bytes,
                available_vram_bytes=available_vram_bytes,
                compute_tflops_estimate=compute_tflops_estimate,
            )
        )

    async def forward(self, steps: Iterable[Activation] | AsyncIterable[Activation]) -> AsyncIterator[Activation]:
        """Stream activations to the remote stage and yield its outputs in order."""

        async def frames():
            if isinstance(steps, AsyncIterable):
                async for act in steps:
                    yield encode_frame(act, self._wire_dtype)
            else:
                for act in steps:
                    yield encode_frame(act, self._wire_dtype)

        async for frame in self._stub.ForwardActivation(frames()):
            yield decode_frame(frame)

    async def close(self) -> None:
        await self._channel.close()
