from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import time
from pathlib import Path

import pytest
from fastmcp import Client, FastMCP
from fastmcp.server.auth import StaticTokenVerifier

from colab_codex_adapter import broker as broker_module
from colab_codex_adapter.broker import (
    BROKER_PROTOCOL_VERSION,
    BrokerClientFactory,
    BrokerLaunchConfig,
    BrokerLauncher,
    BrokerProtocolMismatchError,
    BrokerState,
    broker_is_healthy,
    read_broker_state,
    stop_broker,
    write_broker_state,
)


async def test_stop_broker_rejects_reused_process_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_file = tmp_path / "broker.json"
    state = BrokerState(
        endpoint="http://127.0.0.1:1/mcp",
        token="private",
        owner_pid=os.getpid(),
        owner_id="stale-owner",
        owner_process_start="not-this-process",
        started_at=time.time(),
    )
    write_broker_state(state_file, state)
    signals: list[int] = []

    def record_signal(pid: int, signum: int) -> None:
        assert pid == os.getpid()
        signals.append(signum)

    monkeypatch.setattr(broker_module.os, "kill", record_signal)

    assert not await stop_broker(
        state_file=state_file, expected_owner_id=state.owner_id
    )
    assert signals == [0]


async def _start_test_backend(
    state: BrokerState, *, port: int
) -> asyncio.Task[None]:
    verifier = StaticTokenVerifier(
        {state.token: {"client_id": "test-proxy", "scopes": []}}
    )
    backend = FastMCP("broker-test", auth=verifier)

    @backend.tool()
    async def shared_identity() -> dict[str, int]:
        return {"owner_pid": os.getpid()}

    server_task: asyncio.Task[None] = asyncio.create_task(
        backend.run_http_async(
            show_banner=False,
            host="127.0.0.1",
            port=port,
            log_level="critical",
            uvicorn_config={"access_log": False, "log_config": None},
            json_response=True,
            stateless_http=True,
        )
    )
    async with asyncio.timeout(5.0):
        while not await broker_is_healthy(state, timeout=0.2):
            if server_task.done():
                await server_task
            await asyncio.sleep(0.05)
    return server_task


async def test_healthy_discovered_endpoint_is_authoritative(
    tmp_path: Path, unused_tcp_port: int
) -> None:
    state_file = tmp_path / "broker.json"
    state = BrokerState(
        endpoint=f"http://127.0.0.1:{unused_tcp_port}/mcp",
        token="private-token",
        owner_pid=os.getpid(),
        started_at=time.time(),
        service_instance_id="shared-service",
        owner_id="existing-owner",
        protocol_version=BROKER_PROTOCOL_VERSION,
    )
    write_broker_state(state_file, state)
    server_task = await _start_test_backend(state, port=unused_tcp_port)

    def unexpected_spawn(*args, **kwargs):
        raise AssertionError("a healthy discovered service must not be replaced")

    launcher = BrokerLauncher(
        BrokerLaunchConfig(
            port=1,
            state_file=state_file,
            lock_file=tmp_path / "broker.lock",
            launch_lock_file=tmp_path / "launch.lock",
        ),
        popen_factory=unexpected_spawn,
    )
    try:
        discovered = await launcher.ensure_running()
        assert discovered == state
        assert discovered.endpoint != launcher.config.endpoint
    finally:
        server_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await server_task


async def test_healthy_incompatible_protocol_fails_without_replacement(
    tmp_path: Path, unused_tcp_port: int
) -> None:
    state_file = tmp_path / "broker.json"
    state = BrokerState(
        endpoint=f"http://127.0.0.1:{unused_tcp_port}/mcp",
        token="private-token",
        owner_pid=os.getpid(),
        started_at=time.time(),
        service_instance_id="future-service",
        owner_id="future-owner",
        protocol_version="future-protocol",
    )
    write_broker_state(state_file, state)
    server_task = await _start_test_backend(state, port=unused_tcp_port)

    def unexpected_spawn(*args, **kwargs):
        raise AssertionError("an incompatible healthy service must not be replaced")

    launcher = BrokerLauncher(
        BrokerLaunchConfig(
            port=1,
            state_file=state_file,
            lock_file=tmp_path / "broker.lock",
            launch_lock_file=tmp_path / "launch.lock",
        ),
        popen_factory=unexpected_spawn,
    )
    try:
        with pytest.raises(
            BrokerProtocolMismatchError, match="future-protocol"
        ):
            await launcher.ensure_running()
    finally:
        server_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await server_task


async def test_live_incompatible_owner_fails_when_endpoint_is_unhealthy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_file = tmp_path / "broker.json"
    state = BrokerState(
        endpoint="http://127.0.0.1:1/mcp",
        token="private-token",
        owner_pid=123,
        owner_process_start="known-process",
        started_at=time.time(),
        service_instance_id="future-service",
        owner_id="future-owner",
        protocol_version="future-protocol",
    )
    write_broker_state(state_file, state)
    monkeypatch.setattr(broker_module, "broker_process_is_alive", lambda pid: True)
    monkeypatch.setattr(
        broker_module,
        "broker_process_start_identity",
        lambda pid: "known-process",
    )

    def unexpected_spawn(*args, **kwargs):
        raise AssertionError("a live incompatible service must not be replaced")

    launcher = BrokerLauncher(
        BrokerLaunchConfig(
            state_file=state_file,
            lock_file=tmp_path / "broker.lock",
            launch_lock_file=tmp_path / "launch.lock",
        ),
        popen_factory=unexpected_spawn,
    )

    with pytest.raises(BrokerProtocolMismatchError, match="future-protocol"):
        await launcher.ensure_running()


def test_broker_state_is_atomic_private_and_legacy_compatible(tmp_path: Path) -> None:
    state_file = tmp_path / "private" / "broker.json"
    state = BrokerState.from_dict(
        {
            "endpoint": "http://127.0.0.1:8765/mcp",
            "token": "private-token",
            "owner_pid": 123,
            "started_at": 10,
        }
    )

    assert state.owner_id == "legacy-123"
    assert state.service_instance_id == "legacy-123"
    assert state.generation == 1
    assert state.protocol_version == "1"
    assert "token" not in state.public_dict()

    write_broker_state(state_file, state)

    assert read_broker_state(state_file) == state
    assert state_file.stat().st_mode & 0o777 == 0o600
    assert state_file.parent.stat().st_mode & 0o777 == 0o700
    assert not list(state_file.parent.glob("*.tmp"))


class _FakeProcess:
    def __init__(self, pid: int, *, terminate_exits: bool = True) -> None:
        self.pid = pid
        self.terminate_exits = terminate_exits
        self.return_code: int | None = None
        self.terminated = False
        self.killed = False
        self.wait_calls = 0

    def poll(self) -> int | None:
        return self.return_code

    def terminate(self) -> None:
        self.terminated = True
        if self.terminate_exits:
            self.return_code = -signal.SIGTERM

    def kill(self) -> None:
        self.killed = True
        self.return_code = -signal.SIGKILL

    def wait(self, timeout: float | None = None) -> int:
        self.wait_calls += 1
        if self.return_code is None:
            raise subprocess.TimeoutExpired(["fake-broker"], timeout)
        return self.return_code


async def test_replacement_accepts_recovered_existing_owner_and_reaps_child(
    tmp_path: Path,
) -> None:
    existing = BrokerState(
        endpoint="http://127.0.0.1:8765/mcp",
        token="existing-token",
        owner_pid=101,
        owner_id="existing-owner",
        generation=7,
        started_at=1,
    )
    process = _FakeProcess(202)
    launcher = BrokerLauncher(
        BrokerLaunchConfig(
            state_file=tmp_path / "broker.json",
            lock_file=tmp_path / "broker.lock",
            launch_lock_file=tmp_path / "launch.lock",
        ),
        popen_factory=lambda *args, **kwargs: process,  # type: ignore[arg-type]
    )

    async def healthy_state() -> BrokerState:
        return existing

    launcher._healthy_state = healthy_state  # type: ignore[method-assign]
    recovered = await launcher._start_replacement(time.monotonic() + 1)

    assert recovered is existing
    assert process.terminated is True
    assert process.killed is False
    assert process.wait_calls == 1


async def test_cancelled_replacement_terminates_and_reaps_child(
    tmp_path: Path,
) -> None:
    process = _FakeProcess(303)
    health_started = asyncio.Event()
    never_ready = asyncio.Event()
    launcher = BrokerLauncher(
        BrokerLaunchConfig(
            state_file=tmp_path / "broker.json",
            lock_file=tmp_path / "broker.lock",
            launch_lock_file=tmp_path / "launch.lock",
        ),
        popen_factory=lambda *args, **kwargs: process,  # type: ignore[arg-type]
    )

    async def blocked_health() -> None:
        health_started.set()
        await never_ready.wait()

    launcher._healthy_state = blocked_health  # type: ignore[method-assign]
    task = asyncio.create_task(
        launcher._start_replacement(time.monotonic() + 10)
    )
    await health_started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert process.terminated is True
    assert process.killed is False
    assert process.wait_calls == 1


async def test_replacement_timeout_kills_and_reaps_stuck_child(
    tmp_path: Path,
) -> None:
    process = _FakeProcess(404, terminate_exits=False)
    launcher = BrokerLauncher(
        BrokerLaunchConfig(
            state_file=tmp_path / "broker.json",
            lock_file=tmp_path / "broker.lock",
            launch_lock_file=tmp_path / "launch.lock",
        ),
        popen_factory=lambda *args, **kwargs: process,  # type: ignore[arg-type]
    )

    with pytest.raises(TimeoutError, match="Timed out starting"):
        await launcher._start_replacement(time.monotonic())

    assert process.terminated is True
    assert process.killed is True
    assert process.wait_calls == 2


def _write_test_broker_runtime(path: Path) -> None:
    path.write_text(
        """
import os

from fastmcp import FastMCP
from fastmcp.server.auth import StaticTokenVerifier


async def serve(state):
    verifier = StaticTokenVerifier(
        {state.token: {"client_id": "detached-test", "scopes": []}}
    )
    backend = FastMCP("detached-broker-test", auth=verifier)

    @backend.tool()
    async def broker_identity():
        return {
            "owner_id": state.owner_id,
            "owner_pid": os.getpid(),
            "generation": state.generation,
        }

    await backend.run_http_async(
        show_banner=False,
        host="127.0.0.1",
        port=int(state.endpoint.split(":")[2].split("/")[0]),
        log_level="critical",
        uvicorn_config={"access_log": False, "log_config": None},
        json_response=True,
        stateless_http=True,
    )
""".lstrip(),
        encoding="utf-8",
    )


async def test_concurrent_launchers_elect_one_replacement_daemon(
    tmp_path: Path,
    unused_tcp_port: int,
    monkeypatch,
) -> None:
    runtime_module = tmp_path / "detached_test_runtime.py"
    _write_test_broker_runtime(runtime_module)
    project = Path(__file__).parents[1]
    existing_pythonpath = os.environ.get("PYTHONPATH")
    pythonpath = os.pathsep.join(
        part
        for part in (str(tmp_path), str(project), existing_pythonpath)
        if part
    )
    monkeypatch.setenv("PYTHONPATH", pythonpath)

    config = BrokerLaunchConfig(
        port=unused_tcp_port,
        state_file=tmp_path / "broker.json",
        lock_file=tmp_path / "broker.lock",
        launch_lock_file=tmp_path / "broker-launch.lock",
        factory="detached_test_runtime:serve",
        startup_timeout=10.0,
        health_timeout=1.0,
    )
    spawned: list[subprocess.Popen[bytes]] = []
    commands: list[list[str]] = []

    def tracked_popen(command, **kwargs):
        commands.append(command)
        process = subprocess.Popen(command, **kwargs)
        spawned.append(process)
        return process

    launchers = [
        BrokerLauncher(config, popen_factory=tracked_popen) for _ in range(8)
    ]

    try:
        initial_states = await asyncio.gather(
            *(launcher.ensure_running() for launcher in launchers)
        )
        first = initial_states[0]

        assert len(spawned) == 1
        assert {state.owner_id for state in initial_states} == {first.owner_id}
        assert {state.owner_pid for state in initial_states} == {first.owner_pid}
        assert first.generation == 1
        assert first.service_instance_id
        assert first.protocol_version == BROKER_PROTOCOL_VERSION
        assert config.state_file.stat().st_mode & 0o777 == 0o600
        assert all(first.token not in argument for argument in commands[0])

        factory = BrokerClientFactory(launchers[0], timeout=2.0, init_timeout=2.0)
        async with factory.client() as client:
            result = await client.call_tool("broker_identity", {})
        assert result.data == {
            "owner_id": first.owner_id,
            "owner_pid": first.owner_pid,
            "generation": 1,
        }

        os.kill(first.owner_pid, signal.SIGKILL)
        spawned[0].wait(timeout=5)

        recovered_states = await asyncio.gather(
            *(launcher.ensure_running() for launcher in launchers)
        )
        recovered = recovered_states[0]

        assert len(spawned) == 2
        assert recovered.owner_id != first.owner_id
        assert recovered.owner_pid != first.owner_pid
        assert recovered.generation == 2
        assert recovered.service_instance_id == first.service_instance_id
        assert recovered.token == first.token
        assert {state.owner_id for state in recovered_states} == {
            recovered.owner_id
        }
        assert all(recovered.token not in argument for argument in commands[1])

        assert await stop_broker(
            state_file=config.state_file,
            expected_owner_id=recovered.owner_id,
        )
        spawned[1].wait(timeout=5)
        stopped = read_broker_state(config.state_file)
        assert stopped is not None
        assert stopped.owner_id == recovered.owner_id
        assert stopped.status == "stopped"
    finally:
        current = read_broker_state(config.state_file)
        if current is not None and current.status not in {"failed", "stopped"}:
            await stop_broker(
                state_file=config.state_file,
                expected_owner_id=current.owner_id,
            )
        for process in spawned:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
