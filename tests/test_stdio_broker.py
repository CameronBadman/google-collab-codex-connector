from __future__ import annotations

import sys
from pathlib import Path

from fastmcp import Client
from fastmcp.client.transports import StdioTransport

async def test_two_stdio_adapters_share_broker_and_connection(
    tmp_path: Path, unused_tcp_port: int
) -> None:
    project = Path(__file__).parents[1]
    args = [
        "-m",
        "colab_codex_adapter",
        "--broker-port",
        str(unused_tcp_port),
        "--broker-state-file",
        str(tmp_path / "broker.json"),
        "--broker-lock-file",
        str(tmp_path / "broker.lock"),
        "--log",
        str(tmp_path / "logs"),
        "--pid-file",
        str(tmp_path / "adapter.pid"),
        "--state-file",
        str(tmp_path / "adapter-state.json"),
    ]
    first = Client(
        StdioTransport(sys.executable, args, cwd=str(project)), init_timeout=15
    )
    second = Client(
        StdioTransport(sys.executable, args, cwd=str(project)), init_timeout=15
    )

    async with first:
        first_info = (await first.call_tool("colab_adapter_info", {})).data
        async with second:
            second_info = (await second.call_tool("colab_adapter_info", {})).data
            assert len(await first.list_tools()) == len(await second.list_tools())

        assert first_info["adapter_pid"] == second_info["adapter_pid"]
        assert first_info["broker_pid"] == second_info["broker_pid"]
        assert (
            first_info["connection"]["connection_id"]
            == second_info["connection"]["connection_id"]
        )
