from __future__ import annotations

import asyncio
import sys
from contextlib import suppress
from pathlib import Path
from typing import Any

from fastmcp import Client
from fastmcp.client.transports import StdioTransport

from colab_codex_adapter.broker import (
    broker_is_healthy,
    broker_process_is_alive,
    read_broker_state,
    stop_broker,
)


def _stdio_client(
    project: Path,
    runtime_dir: Path,
    broker_port: int,
    shim_index: int,
) -> Client[Any]:
    args = [
        "-m",
        "colab_codex_adapter",
        "--broker-port",
        str(broker_port),
        "--broker-state-file",
        str(runtime_dir / "broker.json"),
        "--broker-lock-file",
        str(runtime_dir / "broker.lock"),
        "--broker-launch-lock-file",
        str(runtime_dir / "broker-launch.lock"),
        "--log",
        str(runtime_dir / "logs"),
        "--pid-file",
        str(runtime_dir / f"shim-{shim_index}.pid"),
        "--state-file",
        str(runtime_dir / f"shim-{shim_index}.json"),
    ]
    return Client(
        StdioTransport(sys.executable, args, cwd=str(project)), init_timeout=30
    )


async def _close_clients(clients: list[Client[Any]]) -> None:
    results = await asyncio.gather(
        *(client.__aexit__(None, None, None) for client in clients),
        return_exceptions=True,
    )
    for result in results:
        if isinstance(result, BaseException):
            raise result


async def test_many_stdio_shims_share_one_service_and_survive_shim_exit(
    tmp_path: Path, unused_tcp_port: int
) -> None:
    project = Path(__file__).parents[1]
    broker_state_file = tmp_path / "broker.json"
    clients = [
        _stdio_client(project, tmp_path, unused_tcp_port, index)
        for index in range(8)
    ]
    entered: list[Client[Any]] = []

    async def enter(client: Client[Any]) -> None:
        await client.__aenter__()
        entered.append(client)

    try:
        await asyncio.gather(*(enter(client) for client in clients))
        results = await asyncio.gather(
            *(client.call_tool("colab_adapter_info", {}) for client in clients)
        )
        infos = [result.data for result in results]
        first = infos[0]

        assert len({info["service_instance_id"] for info in infos}) == 1
        assert len({info["service_pid"] for info in infos}) == 1
        assert len({info["service_owner_id"] for info in infos}) == 1
        assert len({info["service_generation"] for info in infos}) == 1
        assert len({info["service_started_at"] for info in infos}) == 1
        assert len({info["connection"]["connection_id"] for info in infos}) == 1
        assert first["service_instance_id"]
        assert first["service_pid"] > 0
        assert first["service_owner_id"]
        assert first["service_generation"] > 0
        assert first["service_started_at"] > 0
        assert all(info["service_status"] == "ready" for info in infos)
        assert all(info["service_healthy"] is True for info in infos)
        assert all(info["instance_scope"] == "user" for info in infos)
        assert all(info["transport"] == "stdio" for info in infos)

        shim_pids = {
            int((tmp_path / f"shim-{index}.pid").read_text(encoding="utf-8"))
            for index in range(len(clients))
        }
        assert len(shim_pids) == len(clients)
        assert first["service_pid"] not in shim_pids
        assert all(info["adapter_pid"] == first["service_pid"] for info in infos)
        assert all(info["broker_pid"] == first["service_pid"] for info in infos)
        assert all(
            info["broker_owner_id"] == first["service_owner_id"] for info in infos
        )
        assert all(
            info["broker_generation"] == first["service_generation"]
            for info in infos
        )

        closing = entered[:4]
        await _close_clients(closing)
        del entered[:4]
        after_partial_close = (
            await entered[0].call_tool("colab_adapter_info", {})
        ).data
        assert after_partial_close["service_instance_id"] == first[
            "service_instance_id"
        ]
        assert after_partial_close["service_pid"] == first["service_pid"]

        await _close_clients(entered)
        entered.clear()
        state = read_broker_state(broker_state_file)
        assert state is not None
        assert broker_process_is_alive(state.owner_pid)
        assert await broker_is_healthy(state)

        replacement_shim = _stdio_client(
            project, tmp_path, unused_tcp_port, len(clients)
        )
        await replacement_shim.__aenter__()
        entered.append(replacement_shim)
        after_all_closed = (
            await replacement_shim.call_tool("colab_adapter_info", {})
        ).data
        assert after_all_closed["service_instance_id"] == first[
            "service_instance_id"
        ]
        assert after_all_closed["service_pid"] == first["service_pid"]
        assert after_all_closed["service_owner_id"] == first["service_owner_id"]
        assert after_all_closed["service_generation"] == first[
            "service_generation"
        ]
    finally:
        with suppress(BaseException):
            await _close_clients(entered)
        state = read_broker_state(broker_state_file)
        if state is not None and state.status not in {"failed", "stopped"}:
            assert await stop_broker(
                state_file=broker_state_file,
                expected_owner_id=state.owner_id,
            )
