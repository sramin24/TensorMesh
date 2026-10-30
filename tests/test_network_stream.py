import grpc
import pytest
import torch

from tensormesh.network.transport import Activation, NodeClient, start_node_server


def identity_stage(x, frame):
    return x


@pytest.fixture
async def connect():
    opened = []

    async def _connect(stage_fn=identity_stage, wire_dtype="FP16"):
        server, port = await start_node_server(stage_fn, cluster_id="c1", address="127.0.0.1:0")
        client = NodeClient(f"127.0.0.1:{port}", wire_dtype=wire_dtype)
        opened.append((server, client))
        return client

    yield _connect
    for server, client in opened:
        await client.close()
        await server.stop(grace=None)


@pytest.fixture
async def node(connect):
    return await connect()


async def test_probe_reports_positive_round_trip_time(node):
    rtt_ns = await node.probe(payload_bytes=1024)

    assert rtt_ns > 0


async def test_handshake_assigns_ranks_in_join_order_and_is_idempotent(node):
    a = await node.handshake(node_id="node-a", device_type="cuda", total_vram_bytes=24 << 30)
    b = await node.handshake(node_id="node-b", device_type="mps", total_vram_bytes=16 << 30)
    a_again = await node.handshake(node_id="node-a", device_type="cuda", total_vram_bytes=24 << 30)

    assert (a.accepted, a.cluster_id, a.assigned_rank) == (True, "c1", 0)
    assert b.assigned_rank == 1
    assert a_again.assigned_rank == 0


async def test_forward_stream_runs_the_stage_on_each_step_in_order(connect):
    client = await connect(stage_fn=lambda x, act: x * 2)
    steps = [
        Activation("sess", 0, torch.full((1, 4, 8), 1.0, dtype=torch.float16), is_prefill=True),
        Activation("sess", 1, torch.full((1, 1, 8), 2.0, dtype=torch.float16)),
        Activation("sess", 2, torch.full((1, 1, 8), 3.0, dtype=torch.float16)),
    ]

    replies = [r async for r in client.forward(steps)]

    assert [(r.session_id, r.seq_step, r.is_prefill) for r in replies] == [
        ("sess", 0, True), ("sess", 1, False), ("sess", 2, False),
    ]
    assert [r.tensor.shape for r in replies] == [(1, 4, 8), (1, 1, 8), (1, 1, 8)]
    assert [r.tensor.unique().item() for r in replies] == [2.0, 4.0, 6.0]


async def test_quantized_wire_dtype_round_trips_within_quantization_error(connect):
    seen = []
    client = await connect(stage_fn=lambda x, act: seen.append(x) or x + 1, wire_dtype="INT8")
    torch.manual_seed(0)
    x = torch.randn(1, 3, 64, dtype=torch.float16)

    [reply] = [r async for r in client.forward([Activation("s", 0, x)])]

    torch.testing.assert_close(seen[0], x, atol=0.05, rtol=0.05)
    torch.testing.assert_close(reply.tensor, x + 1, atol=0.1, rtol=0.05)


async def test_stage_failure_surfaces_as_an_rpc_error(connect):
    def broken_stage(x, act):
        raise RuntimeError("out of memory")

    client = await connect(stage_fn=broken_stage)

    with pytest.raises(grpc.aio.AioRpcError) as exc:
        [r async for r in client.forward([Activation("s", 0, torch.ones(1, 1, 4))])]
    assert exc.value.code() == grpc.StatusCode.INTERNAL
    assert "out of memory" in exc.value.details()
