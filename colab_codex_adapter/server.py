from __future__ import annotations

import argparse
import asyncio
import logging
import tempfile
from typing import Any

from fastmcp import FastMCP
from fastmcp.utilities import logging as fastmcp_logger

from .session import ColabSessionManager, NotConnectedError
from .tools import call_resolved_tool, serialize_tool_result, tool_summary


def create_mcp(manager: ColabSessionManager | None = None) -> FastMCP:
    session = manager or ColabSessionManager()
    mcp = FastMCP(
        name="ColabCodexAdapter",
        instructions=(
            "Static-tool adapter for connecting Codex to a Google Colab browser "
            "session. Call colab_connect first, then use the colab_* tools."
        ),
    )

    @mcp.tool()
    async def colab_connect(
        wait_seconds: float = 60.0, open_browser: bool = True
    ) -> dict[str, Any]:
        """Open the Colab connection URL and wait for the browser session."""
        status = await session.connect(
            wait_seconds=wait_seconds, open_browser=open_browser
        )
        return status.__dict__

    @mcp.tool()
    async def colab_status(include_remote_tools: bool = False) -> dict[str, Any]:
        """Return current browser connection state and the connection URL."""
        status = await session.status(include_remote_tools=include_remote_tools)
        return status.__dict__

    @mcp.tool()
    async def colab_list_remote_tools() -> dict[str, Any]:
        """List tools exposed by the connected Colab browser frontend."""
        tools = await session.list_tools()
        return {"tools": [tool_summary(tool) for tool in tools]}

    @mcp.tool()
    async def colab_call_remote_tool(
        name: str, arguments: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Call a raw tool exposed by the connected Colab browser frontend."""
        result = await session.call_tool(name, arguments or {})
        return serialize_tool_result(result)

    @mcp.tool()
    async def colab_get_notebook(remote_tool_name: str | None = None) -> dict[str, Any]:
        """Read notebook/cell state from the connected Colab session."""
        return await call_resolved_tool(
            session,
            [
                "get_notebook",
                "read_notebook",
                "list_cells",
                "get_cells",
                "notebook_info",
            ],
            {},
            remote_tool_name,
        )

    @mcp.tool()
    async def colab_add_cell(
        code: str,
        cell_type: str = "code",
        cell_index: int | None = None,
        after_cell_id: str | None = None,
        remote_tool_name: str | None = None,
    ) -> dict[str, Any]:
        """Add a code or markdown cell to the connected Colab notebook."""
        return await call_resolved_tool(
            session,
            [
                "add_cell",
                "add_code_cell",
                "add_text_cell",
                "insert_cell",
                "create_cell",
            ],
            {
                "code": code,
                "cell_type": cell_type,
                "cell_index": cell_index,
                "after_cell_id": after_cell_id,
            },
            remote_tool_name,
        )

    @mcp.tool()
    async def colab_update_cell(
        code: str,
        cell_id: str | None = None,
        cell_index: int | None = None,
        remote_tool_name: str | None = None,
    ) -> dict[str, Any]:
        """Replace the source of an existing Colab notebook cell."""
        return await call_resolved_tool(
            session,
            ["update_cell", "replace_cell", "edit_cell", "set_cell", "write_cell"],
            {"code": code, "cell_id": cell_id, "cell_index": cell_index},
            remote_tool_name,
        )

    @mcp.tool()
    async def colab_run_cell(
        cell_id: str | None = None,
        cell_index: int | None = None,
        remote_tool_name: str | None = None,
    ) -> dict[str, Any]:
        """Execute an existing Colab notebook cell."""
        return await call_resolved_tool(
            session,
            ["run_code_cell", "execute_cell", "run_cell", "run_selected_cell"],
            {"cell_id": cell_id, "cell_index": cell_index},
            remote_tool_name,
        )

    @mcp.tool()
    async def colab_run_python(
        code: str, remote_tool_name: str | None = None
    ) -> dict[str, Any]:
        """Execute Python code in the connected Colab runtime."""
        return await call_resolved_tool(
            session,
            [
                "run_code_cell",
                "execute_code",
                "run_code",
                "execute_python",
                "runtime_execute_code",
                "run_python",
            ],
            {"code": code},
            remote_tool_name,
        )

    @mcp.tool()
    async def colab_install_package(
        packages: list[str] | str, remote_tool_name: str | None = None
    ) -> dict[str, Any]:
        """Install one or more Python packages into the connected Colab runtime."""
        package_value = packages if isinstance(packages, list) else [packages]
        return await call_resolved_tool(
            session,
            [
                "install_package",
                "pip_install",
                "install_packages",
                "run_pip",
                "run_code_cell",
                "execute_code",
                "run_code",
            ],
            {
                "packages": package_value,
                "code": f"%pip install {' '.join(package_value)}",
            },
            remote_tool_name,
        )

    return mcp


def init_logger(logdir: str) -> None:
    log_filename = f"{logdir}/colab-codex-adapter.log"
    logging.basicConfig(
        format="%(asctime)s %(levelname)s:%(message)s",
        filename=log_filename,
        level=logging.INFO,
    )
    fastmcp_logger.get_logger("colab-codex-adapter").info(
        "logging to %s", log_filename
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Codex-compatible Colab MCP adapter")
    parser.add_argument(
        "-l",
        "--log",
        default=tempfile.mkdtemp(prefix="colab-codex-adapter-logs-"),
        help="directory for adapter logs",
    )
    return parser.parse_args()


async def main_async() -> None:
    args = parse_args()
    init_logger(args.log)
    manager = ColabSessionManager()
    mcp = create_mcp(manager)
    await manager.start()
    try:
        await mcp.run_async()
    finally:
        await manager.close()


def main() -> None:
    try:
        asyncio.run(main_async())
    except NotConnectedError as exc:
        raise SystemExit(str(exc)) from exc
