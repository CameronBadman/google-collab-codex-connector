from __future__ import annotations

from fastmcp import Client

from colab_codex_adapter.server import create_mcp
from colab_codex_adapter.session import ColabSessionManager


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
