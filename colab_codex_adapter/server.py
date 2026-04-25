from __future__ import annotations

import argparse
import asyncio
import logging
import tempfile
from typing import Any

from fastmcp import FastMCP
from fastmcp.utilities import logging as fastmcp_logger

from .session import ColabSessionManager, NotConnectedError
from .tools import (
    call_resolved_tool,
    first_json_object,
    serialize_tool_result,
    tool_summary,
)


def _result_data(result: Any) -> dict[str, Any]:
    data = getattr(result, "structured_content", None)
    if isinstance(data, dict):
        return data
    data = getattr(result, "data", None)
    if isinstance(data, dict):
        return data
    data = first_json_object(result)
    return data if isinstance(data, dict) else {}


async def _remote_tool_names(session: ColabSessionManager) -> set[str]:
    return {tool.name for tool in await session.list_tools()}


async def _get_cells(
    session: ColabSessionManager, include_outputs: bool = False
) -> list[dict[str, Any]]:
    result = await session.call_tool("get_cells", {"includeOutputs": include_outputs})
    cells = _result_data(result).get("cells", [])
    return cells if isinstance(cells, list) else []


async def _cell_id_at_index(session: ColabSessionManager, cell_index: int) -> str:
    cells = await _get_cells(session)
    try:
        cell = cells[cell_index]
    except IndexError as exc:
        raise ValueError(f"No Colab cell exists at index {cell_index}") from exc
    cell_id = cell.get("id")
    if not isinstance(cell_id, str):
        raise ValueError(f"Colab cell at index {cell_index} has no string id")
    return cell_id


async def _append_and_run_code(
    session: ColabSessionManager, code: str, language: str = "python"
) -> dict[str, Any]:
    cells = await _get_cells(session)
    add_result = await session.call_tool(
        "add_code_cell",
        {"cellIndex": len(cells), "language": language, "code": code},
    )
    new_cell_id = _result_data(add_result).get("newCellId")
    if not isinstance(new_cell_id, str):
        raise ValueError("Colab did not return a newCellId from add_code_cell")
    run_result = await session.call_tool("run_code_cell", {"cellId": new_cell_id})
    return {
        "cell_id": new_cell_id,
        "add_result": serialize_tool_result(add_result),
        "run_result": serialize_tool_result(run_result),
        "outputs": _result_data(run_result).get("outputs"),
    }


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
        if remote_tool_name is None and "get_cells" in await _remote_tool_names(session):
            result = await session.call_tool("get_cells", {"includeOutputs": True})
            return {
                "remote_tool": "get_cells",
                "result": serialize_tool_result(result),
                "cells": _result_data(result).get("cells", []),
            }
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
        if remote_tool_name is None:
            names = await _remote_tool_names(session)
            if {"add_code_cell", "add_text_cell"} & names:
                if cell_index is None:
                    if "get_cells" not in names:
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
                    cell_index = len(await _get_cells(session))
                if (
                    cell_type.lower() in {"markdown", "text"}
                    and "add_text_cell" in names
                ):
                    result = await session.call_tool(
                        "add_text_cell", {"cellIndex": cell_index, "content": code}
                    )
                    return {
                        "remote_tool": "add_text_cell",
                        "result": serialize_tool_result(result),
                        "cell_id": _result_data(result).get("newCellId"),
                    }
                if "add_code_cell" in names:
                    result = await session.call_tool(
                        "add_code_cell",
                        {
                            "cellIndex": cell_index,
                            "language": "python",
                            "code": code,
                        },
                    )
                    return {
                        "remote_tool": "add_code_cell",
                        "result": serialize_tool_result(result),
                        "cell_id": _result_data(result).get("newCellId"),
                    }
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
        if remote_tool_name is None and "update_cell" in await _remote_tool_names(session):
            if cell_id is None:
                if cell_index is None:
                    raise ValueError("colab_update_cell requires cell_id or cell_index")
                cell_id = await _cell_id_at_index(session, cell_index)
            result = await session.call_tool(
                "update_cell", {"cellId": cell_id, "content": code}
            )
            return {
                "remote_tool": "update_cell",
                "cell_id": cell_id,
                "result": serialize_tool_result(result),
            }
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
        if remote_tool_name is None and "run_code_cell" in await _remote_tool_names(session):
            if cell_id is None:
                if cell_index is None:
                    raise ValueError("colab_run_cell requires cell_id or cell_index")
                cell_id = await _cell_id_at_index(session, cell_index)
            result = await session.call_tool("run_code_cell", {"cellId": cell_id})
            return {
                "remote_tool": "run_code_cell",
                "cell_id": cell_id,
                "result": serialize_tool_result(result),
                "outputs": _result_data(result).get("outputs"),
            }
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
        if remote_tool_name is None:
            names = await _remote_tool_names(session)
            if {"add_code_cell", "run_code_cell", "get_cells"}.issubset(names):
                return await _append_and_run_code(session, code)
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
        if remote_tool_name is None:
            names = await _remote_tool_names(session)
            if {"add_code_cell", "run_code_cell", "get_cells"}.issubset(names):
                return await _append_and_run_code(
                    session, f"%pip install {' '.join(package_value)}"
                )
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
