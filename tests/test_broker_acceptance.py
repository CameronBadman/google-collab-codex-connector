from __future__ import annotations

import asyncio
import os
import signal
import sys
from collections.abc import Callable
from pathlib import Path

from fastmcp import Client
from fastmcp.client.transports import StdioTransport

from colab_codex_adapter.broker import (
    BrokerState,
    read_broker_state,
    stop_broker,
)


async def _wait_until(
    predicate: Callable[[], bool], *, timeout: float = 15.0
) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.05)


async def _stop_discovered_broker(state_file: Path) -> None:
    state = read_broker_state(state_file)
    if state is None or state.status in {"failed", "stopped"}:
        return
    await stop_broker(
        state_file=state_file,
        expected_owner_id=state.owner_id,
        timeout=10.0,
    )


async def test_existing_stdio_clients_converge_on_one_replacement_broker(
    tmp_path: Path,
    unused_tcp_port: int,
    monkeypatch,
) -> None:
    project = Path(__file__).parents[1]
    state_file = tmp_path / "broker.json"
    browser_probe = tmp_path / "browser-opened"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_xdg_open = fake_bin / "xdg-open"
    fake_xdg_open.write_text(
        "#!/bin/sh\n: > \"$COLAB_BROWSER_PROBE\"\n",
        encoding="utf-8",
    )
    fake_xdg_open.chmod(0o755)
    monkeypatch.setenv("COLAB_BROWSER_PROBE", str(browser_probe))
    monkeypatch.setenv(
        "PATH", os.pathsep.join((str(fake_bin), os.environ.get("PATH", "")))
    )
    monkeypatch.delenv("COLAB_CODEX_OPEN_NATIVE_BROWSER", raising=False)

    args = [
        "-m",
        "colab_codex_adapter",
        "--broker-port",
        str(unused_tcp_port),
        "--broker-state-file",
        str(state_file),
        "--broker-lock-file",
        str(tmp_path / "broker.lock"),
        "--broker-launch-lock-file",
        str(tmp_path / "broker-launch.lock"),
        "--log",
        str(tmp_path / "logs"),
        "--pid-file",
        str(tmp_path / "adapter.pid"),
        "--state-file",
        str(tmp_path / "adapter-state.json"),
    ]
    first: Client | None = Client(
        StdioTransport(sys.executable, args, cwd=str(project)), init_timeout=30
    )
    second: Client | None = Client(
        StdioTransport(sys.executable, args, cwd=str(project)), init_timeout=30
    )
    first_entered = False
    second_entered = False
    observed_states: list[BrokerState] = []

    try:
        await first.__aenter__()
        first_entered = True
        await second.__aenter__()
        second_entered = True

        first_info, second_info = await asyncio.gather(
            first.call_tool("colab_adapter_info", {}),
            second.call_tool("colab_adapter_info", {}),
        )
        initial = read_broker_state(state_file)
        assert initial is not None
        observed_states.append(initial)
        assert initial.status == "ready"
        assert initial.generation == 1
        assert {
            first_info.data["broker_owner_id"],
            second_info.data["broker_owner_id"],
        } == {initial.owner_id}
        initial_connection_id = first_info.data["connection"]["connection_id"]
        assert (
            second_info.data["connection"]["connection_id"]
            == initial_connection_id
        )
        assert browser_probe.exists() is False

        os.kill(initial.owner_pid, signal.SIGKILL)

        recovered_results = await asyncio.gather(
            first.call_tool("colab_adapter_info", {}),
            second.call_tool("colab_adapter_info", {}),
        )
        await _wait_until(
            lambda: (
                (state := read_broker_state(state_file)) is not None
                and state.status == "ready"
                and state.generation == initial.generation + 1
            ),
            timeout=30.0,
        )
        recovered = read_broker_state(state_file)
        assert recovered is not None
        observed_states.append(recovered)

        assert recovered.owner_id != initial.owner_id
        assert recovered.owner_pid != initial.owner_pid
        assert recovered.generation == initial.generation + 1
        assert recovered.previous_owner_id == initial.owner_id
        assert recovered.previous_generation == initial.generation
        assert recovered.token == initial.token
        assert {
            result.data["broker_owner_id"] for result in recovered_results
        } == {recovered.owner_id}
        assert {
            result.data["broker_pid"] for result in recovered_results
        } == {recovered.owner_pid}
        assert {
            result.data["broker_generation"] for result in recovered_results
        } == {recovered.generation}
        assert {
            result.data["connection"]["connection_id"]
            for result in recovered_results
        } == {initial_connection_id}

        await asyncio.sleep(0.2)
        stable = read_broker_state(state_file)
        assert stable is not None
        assert stable.owner_id == recovered.owner_id
        assert stable.owner_pid == recovered.owner_pid
        assert stable.generation == recovered.generation
        assert browser_probe.exists() is False
    finally:
        if first is not None and first_entered:
            await first.__aexit__(None, None, None)
            first = None
        if second is not None and second_entered:
            await second.__aexit__(None, None, None)
            second = None
        await _stop_discovered_broker(state_file)

        for state in observed_states:
            if state.owner_pid == os.getpid():
                continue
            try:
                os.kill(state.owner_pid, 0)
            except ProcessLookupError:
                continue
            current = read_broker_state(state_file)
            if current is not None and current.owner_id == state.owner_id:
                await stop_broker(
                    state_file=state_file,
                    expected_owner_id=state.owner_id,
                    timeout=5.0,
                )
