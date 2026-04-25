from __future__ import annotations

from fastmcp import Client

from colab_codex_adapter.server import create_mcp


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
