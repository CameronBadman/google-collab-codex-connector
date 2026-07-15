from __future__ import annotations

import socket
import sys
from pathlib import Path

import pytest
from fastmcp import Client

from colab_codex_adapter import session as session_module
from colab_codex_adapter.private_state import read_private_json, write_private_json
from colab_codex_adapter.server import (
    _start_recovered_session,
    create_mcp,
    parse_args,
)
from colab_codex_adapter.session import (
    NATIVE_BROWSER_ENV,
    ColabSessionManager,
)


async def test_static_tool_list_is_available_without_colab_browser() -> None:
    mcp = create_mcp()
    async with Client(mcp) as client:
        names = {tool.name for tool in await client.list_tools()}

    assert {
        "colab_adapter_info",
        "colab_connect",
        "colab_connection_url",
        "colab_reset_connection",
        "colab_status",
        "colab_list_remote_tools",
        "colab_call_remote_tool",
        "colab_run_python",
        "colab_run_python_async",
        "colab_job_status",
        "colab_wait_job",
        "colab_run_python_wait",
        "colab_list_jobs",
        "colab_read_artifact",
    }.issubset(names)


async def test_connect_emits_progress_without_exposing_context_schema() -> None:
    mcp = create_mcp()
    progress: list[tuple[float, float | None, str | None]] = []

    async def progress_handler(
        current: float,
        total: float | None,
        message: str | None,
    ) -> None:
        progress.append((current, total, message))

    async with Client(mcp) as client:
        tools = {tool.name: tool for tool in await client.list_tools()}
        result = await client.call_tool(
            "colab_connect",
            {"wait_seconds": 0.01, "open_browser": False},
            progress_handler=progress_handler,
        )

    assert result.is_error is False
    assert "ctx" not in tools["colab_connect"].inputSchema.get("properties", {})
    assert progress == [
        (1.0, None, "Waiting for Colab browser"),
        (2.0, None, "Browser connection not ready"),
    ]


def test_stdio_shim_diagnostic_files_are_opt_in(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["colab-codex-adapter"])

    args = parse_args()

    assert args.pid_file is None
    assert args.state_file is None


async def test_status_reports_phases_without_remote_tools() -> None:
    manager = ColabSessionManager()
    try:
        status = await manager.status(include_remote_tools=True)
    finally:
        await manager.close()

    assert status.server_listening is True
    assert status.browser_ws_connected is False
    assert status.remote_mcp_initialized is False
    assert status.remote_tool_count is None
    assert status.adapter_pid > 0
    assert status.connection_id
    assert status.token_prefix
    assert status.open_url_path == "/tmp/colab-codex-adapter/open-url"


async def test_reset_connection_rotates_connection_metadata() -> None:
    manager = ColabSessionManager()
    try:
        first = await manager.status()
        second = await manager.reset(wait_seconds=0.01, open_browser=False)
    finally:
        await manager.close()

    assert first.connection_id != second.connection_id
    assert first.url != second.url
    assert second.server_listening is True
    assert second.browser_ws_connected is False


async def test_recovered_session_rebinds_when_persisted_port_is_busy(
    tmp_path: Path,
) -> None:
    occupied = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    occupied.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    occupied.bind(("127.0.0.1", 0))
    occupied.listen()
    occupied_port = occupied.getsockname()[1]
    state_file = tmp_path / "browser.json"
    write_private_json(
        state_file,
        {
            "port": occupied_port,
            "token": "stable-test-token",
            "connection_id": "stable-connection-id",
        },
    )

    session = await _start_recovered_session(state_file)
    try:
        status = await session.status()
        persisted = read_private_json(state_file)
        assert status.port != occupied_port
        assert status.token_prefix == "stable-t"
        assert status.connection_id == "stable-connection-id"
        assert persisted is not None
        assert persisted["port"] == status.port
        assert persisted["token"] == "stable-test-token"
    finally:
        await session.close()
        occupied.close()


async def test_connect_writes_private_url_file_when_browser_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    open_url = tmp_path / "private" / "open-url"
    victim = tmp_path / "victim.txt"
    victim.write_text("unchanged", encoding="utf-8")
    open_url.parent.mkdir()
    open_url.symlink_to(victim)
    monkeypatch.setattr(session_module, "OPEN_URL_PATH", str(open_url))
    manager = ColabSessionManager()
    monkeypatch.setenv(NATIVE_BROWSER_ENV, "0")
    try:
        status = await manager.connect(wait_seconds=0.01, open_browser=True)
        assert open_url.exists()
        assert open_url.is_symlink() is False
        assert open_url.read_text(encoding="utf-8").strip() == status.url
        assert open_url.stat().st_mode & 0o777 == 0o600
        assert open_url.parent.stat().st_mode & 0o777 == 0o700
        assert victim.read_text(encoding="utf-8") == "unchanged"
        assert status.open_url_path == str(open_url)
        assert status.browser_launch_attempted is False
    finally:
        await manager.close()


async def test_connect_opens_default_browser_with_isolated_file_descriptors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    launched: list[tuple[list[str], dict[str, object]]] = []

    def fake_popen(command: list[str], **kwargs: object) -> object:
        launched.append((command, kwargs))
        return object()

    monkeypatch.delenv(NATIVE_BROWSER_ENV, raising=False)
    monkeypatch.setenv("BROWSER", "/tmp/obsolete-browser-shim")
    monkeypatch.setattr(
        session_module, "OPEN_URL_PATH", str(tmp_path / "private" / "open-url")
    )
    monkeypatch.setattr(session_module.subprocess, "Popen", fake_popen)
    manager = ColabSessionManager()
    try:
        status = await manager.connect(wait_seconds=0.01, open_browser=True)
    finally:
        await manager.close()

    assert launched[0][0] == ["xdg-open", status.url]
    options = launched[0][1]
    assert options["stdin"] is session_module.subprocess.DEVNULL
    assert options["stdout"] is session_module.subprocess.DEVNULL
    assert options["stderr"] is session_module.subprocess.DEVNULL
    assert options["close_fds"] is True
    assert options["start_new_session"] is True
    assert "BROWSER" not in options["env"]  # type: ignore[operator]
    assert status.browser_launch_attempted is True
    assert status.browser_launch_succeeded is True
    assert status.browser_launch_error is None
