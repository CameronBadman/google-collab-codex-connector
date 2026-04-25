from __future__ import annotations

from fastmcp import Client

from colab_codex_adapter.server import create_mcp
from colab_codex_adapter.session import ColabSessionManager


async def test_static_tool_list_is_available_without_colab_browser() -> None:
    mcp = create_mcp()
    async with Client(mcp) as client:
        names = {tool.name for tool in await client.list_tools()}

    assert {
        "colab_connect",
        "colab_status",
        "colab_list_remote_tools",
        "colab_call_remote_tool",
        "colab_run_python",
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
