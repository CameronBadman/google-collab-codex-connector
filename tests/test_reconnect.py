from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any

import pytest
import websockets
from websockets.exceptions import ConnectionClosed

from colab_codex_adapter import session as session_module
from colab_codex_adapter.bridge import (
    COLAB,
    HARD_MAX_WEBSOCKET_FRAME_BYTES,
    ColabWebSocketServer,
)
from colab_codex_adapter.session import (
    ConnectionTransition,
    ColabSessionManager,
    TransportDisconnected,
)


def _websocket_url(bridge: ColabWebSocketServer) -> str:
    return f"ws://127.0.0.1:{bridge.port}/?access_token={bridge.token}"


async def _wait_until(
    predicate: Callable[[], bool], timeout: float = 2.0
) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.01)


class FakeColabFrontend:
    def __init__(self, url: str) -> None:
        self.url = url
        self.connected = asyncio.Event()
        self.websocket: Any = None
        self.task: asyncio.Task[None] | None = None

    def start(self) -> None:
        self.task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        async with websockets.connect(
            self.url,
            origin=COLAB,
            subprotocols=["mcp"],
            proxy=None,
            max_size=HARD_MAX_WEBSOCKET_FRAME_BYTES,
        ) as websocket:
            self.websocket = websocket
            self.connected.set()
            async for raw_message in websocket:
                message = json.loads(raw_message)
                request_id = message.get("id")
                if request_id is None:
                    continue
                method = message.get("method")
                if method == "initialize":
                    result = {
                        "protocolVersion": message["params"]["protocolVersion"],
                        "capabilities": {"tools": {"listChanged": False}},
                        "serverInfo": {"name": "fake-colab", "version": "1.0"},
                    }
                elif method == "tools/list":
                    result = {"tools": []}
                else:
                    result = {}
                await websocket.send(
                    json.dumps(
                        {"jsonrpc": "2.0", "id": request_id, "result": result}
                    )
                )

    async def close(self, code: int = 1001, reason: str = "test reconnect") -> None:
        if self.websocket is not None:
            await self.websocket.close(code=code, reason=reason)
        if self.task is not None:
            await self.task


async def test_bridge_accepts_large_frame_and_reuses_token_with_fresh_streams() -> None:
    async with ColabWebSocketServer(
        host="127.0.0.1", max_frame_bytes=2 * 1024 * 1024
    ) as bridge:
        websocket_url = _websocket_url(bridge)
        async with websockets.connect(
            websocket_url,
            origin=COLAB,
            subprotocols=["mcp"],
            proxy=None,
            max_size=2 * 1024 * 1024,
        ) as first_websocket:
            first_connection = await bridge.wait_for_connection()
            payload = json.dumps(
                {
                    "jsonrpc": "2.0",
                    "method": "notifications/message",
                    "params": {"level": "info", "data": "x" * (1024 * 1024)},
                }
            )
            await first_websocket.send(payload)
            await first_connection.read_stream.receive()
            assert first_connection.diagnostics.received_bytes > 1024 * 1024

        await _wait_until(first_connection.closed.is_set)

        async with websockets.connect(
            websocket_url,
            origin=COLAB,
            subprotocols=["mcp"],
            proxy=None,
        ):
            second_connection = await bridge.wait_for_connection(
                first_connection.generation
            )
            assert second_connection.generation == first_connection.generation + 1
            assert second_connection.read_stream is not first_connection.read_stream
            assert second_connection.write_stream is not first_connection.write_stream


async def test_bridge_rejects_frames_above_configured_limit_with_1009() -> None:
    async with ColabWebSocketServer(
        host="127.0.0.1", max_frame_bytes=1024
    ) as bridge:
        async with websockets.connect(
            _websocket_url(bridge),
            origin=COLAB,
            subprotocols=["mcp"],
            proxy=None,
        ) as websocket:
            await bridge.wait_for_connection()
            await websocket.send(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "method": "notifications/message",
                        "params": {"data": "x" * 2048},
                    }
                )
            )
            with pytest.raises(ConnectionClosed) as closed:
                await websocket.recv()

            assert closed.value.rcvd is not None
            assert closed.value.rcvd.code == 1009
        await _wait_until(lambda: bridge.active_connection is None)
        diagnostics = bridge.diagnostics()
        assert diagnostics["last_browser_close_code"] == 1009
        assert diagnostics["browser_rejected_frame_bytes"] > 2048
        assert diagnostics["browser_largest_received_frame_bytes"] > 2048


async def test_second_browser_is_rejected_without_disrupting_active_tab() -> None:
    async with ColabWebSocketServer(host="127.0.0.1") as bridge:
        websocket_url = _websocket_url(bridge)
        async with websockets.connect(
            websocket_url,
            origin=COLAB,
            subprotocols=["mcp"],
            proxy=None,
        ) as active_websocket:
            active_connection = await bridge.wait_for_connection()
            async with websockets.connect(
                websocket_url,
                origin=COLAB,
                subprotocols=["mcp"],
                proxy=None,
            ) as rejected_websocket:
                with pytest.raises(ConnectionClosed) as closed:
                    await rejected_websocket.recv()
                assert closed.value.rcvd is not None
                assert closed.value.rcvd.code == 1013

            assert bridge.active_connection is active_connection
            assert bridge.connection_live.is_set()
            await active_websocket.send(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "method": "notifications/message",
                        "params": {"level": "info", "data": "still active"},
                    }
                )
            )
            await active_connection.read_stream.receive()
            assert bridge.rejected_connections == 1


async def test_session_supervisor_reinitializes_same_url_after_disconnect() -> None:
    transitions: list[ConnectionTransition] = []
    manager = ColabSessionManager(bridge_host="127.0.0.1")
    manager.add_connection_listener(transitions.append)
    await manager.start()
    initial = await manager.status()
    websocket_url = _websocket_url(manager.bridge)
    first_frontend = FakeColabFrontend(websocket_url)
    second_frontend: FakeColabFrontend | None = None
    try:
        first_frontend.start()
        first_status = await manager.connect(wait_seconds=2)
        assert first_status.connected is True
        assert first_status.browser_generation == 1
        first_stream = manager.bridge.active_connection.read_stream  # type: ignore[union-attr]

        await first_frontend.close(code=1001, reason="temporary network loss")
        await _wait_until(lambda: not manager.browser_ws_connected())
        with pytest.raises(TransportDisconnected) as disconnected:
            manager.require_client()
        assert disconnected.value.browser_generation == 1

        second_frontend = FakeColabFrontend(websocket_url)
        second_frontend.start()
        second_status = await manager.connect(wait_seconds=2)
        assert second_status.connected is True
        assert second_status.browser_generation == 2
        assert second_status.runtime_generation == 2
        assert second_status.connection_id == initial.connection_id
        assert second_status.url == initial.url
        assert manager.bridge.active_connection.read_stream is not first_stream  # type: ignore[union-attr]
        assert [transition.kind for transition in transitions] == [
            "browser_connected",
            "runtime_ready",
            "browser_disconnected",
            "browser_connected",
            "runtime_ready",
        ]
    finally:
        await manager.close()
        if second_frontend is not None and second_frontend.task is not None:
            await second_frontend.task


async def test_default_connect_and_reset_never_open_browser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_browser(*args: object, **kwargs: object) -> object:
        raise AssertionError("default connection path opened a browser")

    monkeypatch.setattr(session_module.subprocess, "Popen", unexpected_browser)
    manager = ColabSessionManager(bridge_host="127.0.0.1")
    try:
        first = await manager.connect(wait_seconds=0.01)
        second = await manager.reset(wait_seconds=0.01)
    finally:
        await manager.close()

    assert first.browser_launch_attempted is False
    assert second.browser_launch_attempted is False
    assert second.connection_id != first.connection_id
    assert second.url != first.url


async def test_concurrent_resets_serialize_session_lifecycle() -> None:
    active_resets = 0
    max_active_resets = 0

    async def listener(transition: ConnectionTransition) -> None:
        nonlocal active_resets, max_active_resets
        if transition.kind != "reset":
            return
        active_resets += 1
        max_active_resets = max(max_active_resets, active_resets)
        await asyncio.sleep(0.02)
        active_resets -= 1

    manager = ColabSessionManager(bridge_host="127.0.0.1")
    manager.add_connection_listener(listener)
    try:
        await manager.start()
        results = await asyncio.gather(
            manager.reset(wait_seconds=0.01),
            manager.reset(wait_seconds=0.01),
        )
        current = await manager.status()
    finally:
        await manager.close()

    assert max_active_resets == 1
    assert active_resets == 0
    assert all(result.server_listening for result in results)
    assert current.server_listening is True


def test_bridge_configuration_is_bounded_and_colab_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        ColabWebSocketServer(
            max_frame_bytes=HARD_MAX_WEBSOCKET_FRAME_BYTES + 1
        )
    with pytest.raises(ValueError, match="must use HTTPS"):
        ColabWebSocketServer(notebook_url="http://colab.research.google.com/test")
    with pytest.raises(ValueError, match="must use HTTPS"):
        ColabWebSocketServer(notebook_url="https://example.com/notebook.ipynb")

    monkeypatch.setenv("COLAB_CODEX_REMOTE_INIT_TIMEOUT_SECONDS", "12")
    monkeypatch.setenv("COLAB_CODEX_REMOTE_TOOL_LIST_TIMEOUT_SECONDS", "17")
    manager = ColabSessionManager()
    assert manager.remote_init_timeout_seconds == 12
    assert manager.remote_tool_list_timeout_seconds == 17


async def test_configured_notebook_url_and_private_recovery_state() -> None:
    notebook_url = (
        "https://colab.research.google.com/github/example/project/blob/main/demo.ipynb"
        "?authuser=0"
    )
    manager = ColabSessionManager(
        bridge_host="127.0.0.1",
        bridge_token="persisted-test-token",
        notebook_url=notebook_url,
        connection_id="persisted-connection-id",
    )
    try:
        await manager.start()
        state = manager.private_bridge_state()
        status = await manager.status()
    finally:
        await manager.close()

    assert status.url.startswith(notebook_url + "#")
    assert status.connection_id == "persisted-connection-id"
    assert state == {
        "token": "persisted-test-token",
        "port": status.port,
        "notebook_url": notebook_url,
        "max_frame_bytes": HARD_MAX_WEBSOCKET_FRAME_BYTES,
        "connection_id": "persisted-connection-id",
    }
