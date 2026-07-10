from __future__ import annotations

from pathlib import Path

import pytest
from fastmcp import Client

from colab_codex_adapter import session as session_module
from colab_codex_adapter.server import create_mcp
from colab_codex_adapter.session import (
    NATIVE_BROWSER_ENV,
    ColabSessionManager,
    OPEN_URL_PATH,
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
    }.issubset(names)


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
    assert status.open_url_path == "/tmp/colab-mcp-open-url"


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


async def test_connect_writes_private_url_file_when_browser_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = ColabSessionManager()
    open_url = Path(OPEN_URL_PATH)
    monkeypatch.setenv(NATIVE_BROWSER_ENV, "0")
    try:
        if open_url.exists():
            open_url.unlink()
        status = await manager.connect(wait_seconds=0.01, open_browser=True)
        assert open_url.exists()
        assert open_url.read_text(encoding="utf-8").strip() == status.url
        assert open_url.stat().st_mode & 0o777 == 0o600
        assert status.browser_launch_attempted is False
    finally:
        await manager.close()
        if open_url.exists():
            open_url.unlink()


async def test_connect_opens_default_browser_with_isolated_file_descriptors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launched: list[tuple[list[str], dict[str, object]]] = []

    def fake_popen(command: list[str], **kwargs: object) -> object:
        launched.append((command, kwargs))
        return object()

    monkeypatch.delenv(NATIVE_BROWSER_ENV, raising=False)
    monkeypatch.setenv("BROWSER", "/tmp/obsolete-browser-shim")
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
