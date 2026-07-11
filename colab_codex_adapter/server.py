from __future__ import annotations

import argparse
import asyncio
import errno
import logging
import os
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext
from fastmcp.server.proxy import FastMCPProxy, ProxyToolManager
from fastmcp.server.auth import StaticTokenVerifier
from fastmcp.tools.tool import ToolResult
from fastmcp.utilities import logging as fastmcp_logger
from mcp.types import CallToolRequestParams
from pydantic import ValidationError

from . import __version__
from .artifacts import (
    DEFAULT_ARTIFACT_DIR,
    ArtifactStore,
    get_default_artifact_store,
)
from .broker import (
    DEFAULT_BROKER_HOST,
    DEFAULT_BROKER_LAUNCH_LOCK_FILE,
    DEFAULT_BROKER_LOCK_FILE,
    DEFAULT_BROKER_PORT,
    DEFAULT_BROKER_STATE_FILE,
    BrokerClientFactory,
    BrokerLaunchConfig,
    BrokerLauncher,
    BrokerState,
    broker_process_is_alive,
    broker_process_start_identity,
)
from .diagnostics import (
    DEFAULT_LOG_DIR,
    DEFAULT_RUNTIME_DIR,
    adapter_info,
    write_pid,
    write_state,
)
from .private_state import read_private_json, write_private_json
from .jobs import (
    DEFAULT_CELL_METADATA_PAGE_SIZE,
    DEFAULT_EXECUTION_TIMEOUT_SECONDS,
    DEFAULT_KERNEL_PROBE_TIMEOUT_SECONDS,
    DEFAULT_WAIT_TIMEOUT_SECONDS,
    MAX_WAIT_TIMEOUT_SECONDS,
    ColabJobManager,
    result_data,
    validate_submitted_code,
)
from .session import ColabSessionManager, ConnectionTransition, NotConnectedError
from .tools import (
    bounded_tool_result,
    tool_summary,
)

DEFAULT_BROWSER_STATE_FILE = DEFAULT_RUNTIME_DIR / "browser.json"
DEFAULT_JOB_JOURNAL_FILE = DEFAULT_RUNTIME_DIR / "jobs.json"
DEFAULT_JOB_RECONCILIATION_POLL_SECONDS = float(
    os.environ.get("COLAB_CODEX_JOB_RECONCILIATION_POLL_SECONDS", "15")
)
_READ_ONLY_PROXY_TOOLS = frozenset(
    {
        "colab_adapter_info",
        "colab_connection_url",
        "colab_get_notebook",
        "colab_job_status",
        "colab_list_jobs",
        "colab_list_remote_tools",
        "colab_read_artifact",
        "colab_status",
        "colab_wait_job",
    }
)


class _RedactToolExceptionTrace(logging.Filter):
    """Keep tool identity in diagnostics without logging user or remote payloads."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.exc_info is not None:
            record.exc_info = None
            record.exc_text = None
        return True


class _BoundedToolErrors(Middleware):
    async def on_call_tool(
        self,
        context: MiddlewareContext[CallToolRequestParams],
        call_next: CallNext[CallToolRequestParams, ToolResult],
    ) -> ToolResult:
        try:
            return await call_next(context)
        except ValidationError as exc:
            raise ToolError("Invalid tool arguments") from exc


def _install_sensitive_log_filters() -> None:
    logger = logging.getLogger("fastmcp.tools.tool_manager")
    if not any(isinstance(item, _RedactToolExceptionTrace) for item in logger.filters):
        logger.addFilter(_RedactToolExceptionTrace())


def _result_data(result: Any) -> dict[str, Any]:
    return result_data(result)


async def _remote_tool_names(session: ColabSessionManager) -> set[str]:
    return {tool.name for tool in await session.list_tools()}


async def _get_cells(
    session: ColabSessionManager,
    include_outputs: bool = False,
    *,
    start: int | None = None,
    end: int | None = None,
) -> list[dict[str, Any]]:
    arguments: dict[str, Any] = {"includeOutputs": include_outputs}
    if start is not None:
        arguments["cellIndexStart"] = start
    if end is not None:
        if start is not None and end <= start:
            raise ValueError("cell metadata end must be greater than start")
        arguments["cellIndexEnd"] = end - 1
    result = await session.call_tool("get_cells", arguments)
    cells = _result_data(result).get("cells", [])
    if not isinstance(cells, list):
        return []
    if start is not None and end is not None and len(cells) > end - start:
        raise RuntimeError(
            "Colab frontend did not honor the bounded cell metadata range"
        )
    return cells


async def _cell_pages(
    session: ColabSessionManager,
) -> AsyncIterator[tuple[int, list[dict[str, Any]]]]:
    offset = 0
    seen_ids: set[str] = set()
    while True:
        cells = await _get_cells(
            session,
            start=offset,
            end=offset + DEFAULT_CELL_METADATA_PAGE_SIZE,
        )
        if len(cells) > DEFAULT_CELL_METADATA_PAGE_SIZE:
            raise RuntimeError(
                "Colab frontend did not honor bounded cell metadata pagination"
            )
        if not cells:
            return
        page_ids = {
            cell_id
            for cell in cells
            if isinstance((cell_id := cell.get("id")), str)
        }
        if len(page_ids) != len(cells):
            raise RuntimeError(
                "Colab frontend returned invalid cell metadata identifiers"
            )
        if offset and page_ids & seen_ids:
            raise RuntimeError(
                "Colab frontend returned overlapping cell metadata pages"
            )
        yield offset, cells
        seen_ids.update(page_ids)
        offset += len(cells)
        if len(cells) < DEFAULT_CELL_METADATA_PAGE_SIZE:
            return


async def _all_cells(session: ColabSessionManager) -> list[dict[str, Any]]:
    all_cells: list[dict[str, Any]] = []
    async for _, cells in _cell_pages(session):
        all_cells.extend(cells)
    return all_cells


async def _cell_count(session: ColabSessionManager) -> int:
    count = 0
    async for offset, cells in _cell_pages(session):
        count = offset + len(cells)
    return count


async def _cell_index_by_id(
    session: ColabSessionManager, cell_id: str
) -> int:
    async for offset, cells in _cell_pages(session):
        for relative_index, cell in enumerate(cells):
            if cell.get("id") == cell_id:
                return offset + relative_index
    raise ValueError("Requested Colab cell was not found")


async def _cell_id_at_index(session: ColabSessionManager, cell_index: int) -> str:
    if cell_index < 0:
        raise ValueError("cell_index must be greater than or equal to zero")
    cells = await _get_cells(session, start=cell_index, end=cell_index + 1)
    if not cells:
        raise ValueError(f"No Colab cell exists at index {cell_index}")
    cell = cells[0]
    cell_id = cell.get("id")
    if not isinstance(cell_id, str):
        raise ValueError(f"Colab cell at index {cell_index} has no string id")
    return cell_id


async def _cell_source(
    session: ColabSessionManager,
    *,
    cell_id: str | None,
    cell_index: int | None,
) -> tuple[str, str]:
    cell: dict[str, Any] | None = None
    if cell_id is not None:
        async for _, cells in _cell_pages(session):
            cell = next((item for item in cells if item.get("id") == cell_id), None)
            if cell is not None:
                break
    elif cell_index is not None:
        if cell_index < 0:
            raise ValueError("cell_index must be greater than or equal to zero")
        cells = await _get_cells(session, start=cell_index, end=cell_index + 1)
        cell = cells[0] if cells else None
    else:
        raise ValueError("colab_run_cell requires cell_id or cell_index")
    if cell is None or not isinstance(cell.get("id"), str):
        raise ValueError("Requested Colab cell was not found")
    source = cell.get("source", cell.get("content", cell.get("code")))
    if isinstance(source, list):
        source = "".join(str(part) for part in source)
    if not isinstance(source, str):
        raise ValueError("Requested Colab cell has no executable source")
    return cell["id"], source


def _scoped_runtime_path(path: Path, broker_port: int) -> Path:
    if broker_port == DEFAULT_BROKER_PORT:
        return path
    return path.with_name(f"{path.stem}-{broker_port}{path.suffix}")


def _restored_session(
    browser_state_file: Path, *, force_ephemeral_port: bool = False
) -> ColabSessionManager:
    state = read_private_json(browser_state_file) or {}
    try:
        return ColabSessionManager(
            bridge_port=(0 if force_ephemeral_port else int(state.get("port", 0))),
            bridge_token=(
                str(state["token"]) if isinstance(state.get("token"), str) else None
            ),
            notebook_url=(
                str(state["notebook_url"])
                if isinstance(state.get("notebook_url"), str)
                else None
            ),
            websocket_max_frame_bytes=(
                int(state["max_frame_bytes"])
                if isinstance(state.get("max_frame_bytes"), int)
                else None
            ),
            connection_id=(
                str(state["connection_id"])
                if isinstance(state.get("connection_id"), str)
                else None
            ),
        )
    except (TypeError, ValueError):
        logging.warning("Ignoring invalid private browser recovery state")
        return ColabSessionManager()


def _persist_session(session: ColabSessionManager, path: Path) -> None:
    write_private_json(path, session.private_bridge_state())


async def _start_recovered_session(
    browser_state_file: Path,
) -> ColabSessionManager:
    session = _restored_session(browser_state_file)
    try:
        await session.start()
    except OSError as exc:
        if exc.errno != errno.EADDRINUSE:
            raise
        logging.warning(
            "Recovered browser port is unavailable; selecting a new port"
        )
        await session.close()
        session = _restored_session(
            browser_state_file, force_ephemeral_port=True
        )
        await session.start()
    _persist_session(session, browser_state_file)
    return session


def _wire_job_transitions(
    session: ColabSessionManager,
    jobs: ColabJobManager,
    background_tasks: set[asyncio.Task[Any]],
    reconciliation_poll_seconds: float,
) -> None:
    reconciliation_task: asyncio.Task[None] | None = None

    async def reconcile(expected_generation: int) -> None:
        while session.runtime_generation == expected_generation:
            try:
                probe_kernel = getattr(jobs, "probe_kernel", None)
                readiness = (
                    await probe_kernel(timeout=DEFAULT_KERNEL_PROBE_TIMEOUT_SECONDS)
                    if probe_kernel is not None
                    else {"kernel_execution_ready": True}
                )
                if readiness["kernel_execution_ready"] is True:
                    await jobs.reconcile_detached()
                else:
                    logging.warning(
                        "Colab kernel readiness probe failed generation=%s",
                        expected_generation,
                    )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logging.warning(
                    "Colab job reconciliation failed error_type=%s",
                    type(exc).__name__,
                )
            await asyncio.sleep(reconciliation_poll_seconds)

    async def listener(transition: ConnectionTransition) -> None:
        nonlocal reconciliation_task
        if transition.kind in {"browser_disconnected", "reset"}:
            mark_kernel_unknown = getattr(jobs, "mark_kernel_unknown", None)
            if mark_kernel_unknown is not None:
                mark_kernel_unknown("Colab browser transport is not connected")
            if reconciliation_task is not None and not reconciliation_task.done():
                reconciliation_task.cancel()
                await asyncio.gather(
                    reconciliation_task, return_exceptions=True
                )
            reconciliation_task = None
        if transition.kind == "browser_disconnected":
            reason = (
                "Browser transport disconnected "
                f"generation={transition.browser_generation} "
                f"code={transition.close_code}"
            )
            await jobs.on_browser_disconnect(reason)
        elif transition.kind == "runtime_ready":
            if reconciliation_task is not None and not reconciliation_task.done():
                reconciliation_task.cancel()
                await asyncio.gather(
                    reconciliation_task, return_exceptions=True
                )
            task = asyncio.create_task(
                reconcile(transition.browser_generation),
                name=(
                    "colab-job-reconciliation-"
                    f"{transition.browser_generation}"
                ),
            )
            reconciliation_task = task
            background_tasks.add(task)
            task.add_done_callback(background_tasks.discard)

    session.add_connection_listener(listener)


def create_mcp(
    manager: ColabSessionManager | None = None,
    runtime_info: dict[str, Any] | None = None,
    job_manager: ColabJobManager | None = None,
    artifact_store: ArtifactStore | None = None,
    background_tasks: set[asyncio.Task[Any]] | None = None,
    browser_state_file: Path | None = None,
    reconciliation_poll_seconds: float = DEFAULT_JOB_RECONCILIATION_POLL_SECONDS,
    auth: Any = None,
) -> FastMCP:
    _install_sensitive_log_filters()
    session = manager or ColabSessionManager()
    artifacts = artifact_store or get_default_artifact_store()
    jobs = job_manager or ColabJobManager(session, artifact_store=artifacts)
    managed_tasks = background_tasks if background_tasks is not None else set()
    if not 0.1 <= reconciliation_poll_seconds <= 300:
        raise ValueError(
            "reconciliation_poll_seconds must be between 0.1 and 300"
        )
    _wire_job_transitions(
        session, jobs, managed_tasks, reconciliation_poll_seconds
    )
    runtime_info = runtime_info or {}
    mcp = FastMCP(
        name="ColabCodexAdapter",
        instructions=(
            "Static-tool adapter for connecting Codex to a Google Colab browser "
            "session. Call colab_connect first, then use the colab_* tools."
        ),
        auth=auth,
        middleware=[_BoundedToolErrors()],
        mask_error_details=True,
    )

    def response(data: dict[str, Any], summary: str) -> ToolResult:
        return bounded_tool_result(
            data,
            summary=summary,
            artifact_store=artifacts,
        )

    def status_data(status: Any) -> dict[str, Any]:
        readiness = getattr(jobs, "kernel_readiness", lambda: {})()
        return {
            **status.__dict__,
            "frontend_mcp_ready": status.remote_mcp_initialized,
            **readiness,
        }

    async def connection_data() -> dict[str, Any]:
        connection = await session.connection_url()
        readiness = getattr(jobs, "kernel_readiness", lambda: {})()
        return {
            **connection,
            "frontend_mcp_ready": connection["runtime_alive"],
            **readiness,
        }

    @mcp.tool()
    async def colab_connect(
        wait_seconds: float = 60.0, open_browser: bool = False
    ) -> ToolResult:
        """Open the Colab connection URL and wait for the browser session."""
        status = await session.connect(
            wait_seconds=wait_seconds, open_browser=open_browser
        )
        if browser_state_file is not None:
            _persist_session(session, browser_state_file)
        return response(status_data(status), "Colab connection status")

    @mcp.tool()
    async def colab_status(include_remote_tools: bool = False) -> ToolResult:
        """Return current browser connection state and the connection URL."""
        status = await session.status(include_remote_tools=include_remote_tools)
        return response(status_data(status), "Colab status")

    @mcp.tool()
    async def colab_adapter_info() -> ToolResult:
        """Return shared service, transport, and connection metadata."""
        connection = await connection_data()
        pid_file = runtime_info.get("pid_file")
        state_file = runtime_info.get("state_file")
        data = adapter_info(
            log_dir=Path(runtime_info.get("log_dir", DEFAULT_LOG_DIR)),
            log_file=(
                Path(runtime_info["log_file"]) if runtime_info.get("log_file") else None
            ),
            pid_file=Path(pid_file) if pid_file else None,
            state_file=Path(state_file) if state_file else None,
            extra={
                **runtime_info,
                "adapter_version": __version__,
                "connection": connection,
            },
        )
        return response(data, "Colab adapter information")

    @mcp.tool()
    async def colab_connection_url() -> ToolResult:
        """Return the current Colab connection URL without remote calls."""
        return response(await connection_data(), "Colab connection URL")

    @mcp.tool()
    async def colab_reset_connection(
        wait_seconds: float = 1.0, open_browser: bool = False
    ) -> ToolResult:
        """Restart the local Colab websocket bridge with a fresh token and URL."""
        await jobs.mark_stale("Colab connection was reset")
        status = await session.reset(
            wait_seconds=wait_seconds, open_browser=open_browser
        )
        if browser_state_file is not None:
            _persist_session(session, browser_state_file)
        return response(status_data(status), "Colab connection reset")

    @mcp.tool()
    async def colab_list_remote_tools() -> ToolResult:
        """List tools exposed by the connected Colab browser frontend."""
        tools = await session.list_tools()
        return response(
            {"tools": [tool_summary(tool) for tool in tools]},
            "Colab remote tools",
        )

    @mcp.tool()
    async def colab_call_remote_tool(
        name: str, arguments: dict[str, Any] | None = None
    ) -> ToolResult:
        """Reject unbounded raw frontend calls on the managed-safe surface."""
        del name, arguments
        raise ToolError(
            "Raw remote tool calls are disabled because their output cannot be "
            "bounded before it crosses the browser transport"
        )

    @mcp.tool()
    async def colab_get_notebook(
        include_outputs: bool = False,
        remote_tool_name: str | None = None,
    ) -> ToolResult:
        """Read notebook/cell state from the connected Colab session."""
        if include_outputs:
            raise ValueError(
                "Notebook-wide output reads are disabled; use tracked job output "
                "or an artifact reference"
            )
        if remote_tool_name is not None:
            raise ValueError("Remote notebook tool overrides are disabled")
        if "get_cells" in await _remote_tool_names(session):
            cells = await _all_cells(session)
            return response(
                {
                    "remote_tool": "get_cells",
                    "include_outputs": include_outputs,
                    "cells": cells,
                },
                "Colab notebook cells",
            )
        raise ValueError(
            "Paged notebook metadata is unavailable in this Colab frontend"
        )

    @mcp.tool()
    async def colab_add_cell(
        code: str,
        cell_type: str = "code",
        cell_index: int | None = None,
        after_cell_id: str | None = None,
        remote_tool_name: str | None = None,
    ) -> ToolResult:
        """Add a code or markdown cell to the connected Colab notebook."""
        validate_submitted_code(code)
        if remote_tool_name is not None:
            raise ValueError("Remote cell tool overrides are disabled")
        if cell_index is not None and after_cell_id is not None:
            raise ValueError("Specify cell_index or after_cell_id, not both")
        if remote_tool_name is None:
            names = await _remote_tool_names(session)
            if {"add_code_cell", "add_text_cell"} & names:
                if after_cell_id is not None:
                    if "get_cells" not in names:
                        raise ValueError(
                            "Paged cell placement is unavailable in this Colab "
                            "frontend"
                        )
                    cell_index = (
                        await _cell_index_by_id(session, after_cell_id)
                    ) + 1
                if cell_index is None:
                    if "get_cells" not in names:
                        raise ValueError(
                            "Paged cell placement is unavailable in this Colab "
                            "frontend"
                        )
                    cell_index = await _cell_count(session)
                if (
                    cell_type.lower() in {"markdown", "text"}
                    and "add_text_cell" in names
                ):
                    result = await session.call_tool(
                        "add_text_cell", {"cellIndex": cell_index, "content": code}
                    )
                    return response(
                        {
                            "remote_tool": "add_text_cell",
                            "cell_id": _result_data(result).get("newCellId"),
                        },
                        "Colab text cell added",
                    )
                if "add_code_cell" in names:
                    result = await session.call_tool(
                        "add_code_cell",
                        {
                            "cellIndex": cell_index,
                            "language": "python",
                            "code": code,
                        },
                    )
                    return response(
                        {
                            "remote_tool": "add_code_cell",
                            "cell_id": _result_data(result).get("newCellId"),
                        },
                        "Colab code cell added",
                    )
        raise ValueError(
            "Exact Colab cell-add tools are unavailable in this frontend"
        )

    @mcp.tool()
    async def colab_update_cell(
        code: str,
        cell_id: str | None = None,
        cell_index: int | None = None,
        remote_tool_name: str | None = None,
    ) -> ToolResult:
        """Replace the source of an existing Colab notebook cell."""
        validate_submitted_code(code)
        if remote_tool_name is not None:
            raise ValueError("Remote cell tool overrides are disabled")
        if "update_cell" in await _remote_tool_names(session):
            if cell_id is None:
                if cell_index is None:
                    raise ValueError("colab_update_cell requires cell_id or cell_index")
                cell_id = await _cell_id_at_index(session, cell_index)
            result = await session.call_tool(
                "update_cell", {"cellId": cell_id, "content": code}
            )
            return response(
                {"remote_tool": "update_cell", "cell_id": cell_id},
                "Colab cell updated",
            )
        raise ValueError(
            "Exact Colab cell-update support is unavailable in this frontend"
        )

    @mcp.tool()
    async def colab_run_cell(
        cell_id: str | None = None,
        cell_index: int | None = None,
        remote_tool_name: str | None = None,
        timeout_seconds: float = DEFAULT_WAIT_TIMEOUT_SECONDS,
        execution_timeout_seconds: float = DEFAULT_EXECUTION_TIMEOUT_SECONDS,
    ) -> ToolResult:
        """Execute existing CPython source through the bounded tracked runner."""
        if remote_tool_name is not None:
            raise ValueError("Unbounded remote cell execution is disabled")
        source_cell_id, source = await _cell_source(
            session, cell_id=cell_id, cell_index=cell_index
        )
        data = await jobs.run_python_wait(
            source,
            timeout_seconds=timeout_seconds,
            execution_timeout_seconds=execution_timeout_seconds,
        )
        data["source_cell_id"] = source_cell_id
        return response(data, "Colab cell source execution")

    @mcp.tool()
    async def colab_run_python(
        code: str,
        timeout_seconds: float = DEFAULT_WAIT_TIMEOUT_SECONDS,
        execution_timeout_seconds: float = DEFAULT_EXECUTION_TIMEOUT_SECONDS,
        remote_tool_name: str | None = None,
    ) -> ToolResult:
        """Execute Python code in the connected Colab runtime."""
        if remote_tool_name is not None:
            raise ValueError("Unbounded remote Python execution is disabled")
        names = await _remote_tool_names(session)
        if not {
            "add_code_cell",
            "run_code_cell",
            "get_cells",
            "update_cell",
        }.issubset(names):
            raise ValueError(
                "Tracked Python execution is unavailable in this Colab frontend"
            )
        data = await jobs.run_python_wait(
            code,
            timeout_seconds=timeout_seconds,
            execution_timeout_seconds=execution_timeout_seconds,
        )
        return response(data, "Colab Python execution")

    @mcp.tool()
    async def colab_run_python_async(
        code: str,
        execution_timeout_seconds: float = DEFAULT_EXECUTION_TIMEOUT_SECONDS,
    ) -> ToolResult:
        """Start a tracked Python cell job and return its job id."""
        return response(
            await jobs.start_python(
                code, execution_timeout_seconds=execution_timeout_seconds
            ),
            "Colab Python job started",
        )

    @mcp.tool()
    async def colab_job_status(job_id: str) -> ToolResult:
        """Return the current state and outputs for a tracked Colab job."""
        return response(await jobs.status(job_id), "Colab job status")

    @mcp.tool()
    async def colab_wait_job(
        job_id: str, timeout_seconds: float = DEFAULT_WAIT_TIMEOUT_SECONDS
    ) -> ToolResult:
        """Wait until a tracked Colab job finishes or the timeout expires."""
        return response(
            await jobs.wait(job_id, timeout_seconds), "Colab job wait result"
        )

    @mcp.tool()
    async def colab_run_python_wait(
        code: str,
        timeout_seconds: float = DEFAULT_WAIT_TIMEOUT_SECONDS,
        execution_timeout_seconds: float = DEFAULT_EXECUTION_TIMEOUT_SECONDS,
    ) -> ToolResult:
        """Run Python in a tracked cell job and wait for its outputs."""
        return response(
            await jobs.run_python_wait(
                code,
                timeout_seconds,
                execution_timeout_seconds=execution_timeout_seconds,
            ),
            "Colab Python job result",
        )

    @mcp.tool()
    async def colab_list_jobs() -> ToolResult:
        """List tracked Colab jobs for this adapter process."""
        return response({"jobs": jobs.list_jobs()}, "Colab jobs")

    @mcp.tool()
    async def colab_read_artifact(
        artifact_id: str,
        offset: int = 0,
        limit_bytes: int = 64 * 1024,
    ) -> ToolResult:
        """Read one bounded chunk from an opaque connector artifact."""
        try:
            data = await jobs.read_artifact(
                artifact_id, offset=offset, limit_bytes=limit_bytes
            )
        except TimeoutError as exc:
            raise ToolError(
                "Colab runtime is busy; retry the artifact read"
            ) from exc
        return response(data, "Colab artifact chunk")

    @mcp.tool()
    async def colab_install_package(
        packages: list[str] | str,
        timeout_seconds: float = MAX_WAIT_TIMEOUT_SECONDS,
        execution_timeout_seconds: float = 3600.0,
        remote_tool_name: str | None = None,
    ) -> ToolResult:
        """Install one or more Python packages into the connected Colab runtime."""
        package_value = packages if isinstance(packages, list) else [packages]
        if remote_tool_name is None:
            names = await _remote_tool_names(session)
            if {
                "add_code_cell",
                "run_code_cell",
                "get_cells",
                "update_cell",
            }.issubset(names):
                install_code = (
                    "import subprocess, sys\n"
                    f"subprocess.run([sys.executable, '-m', 'pip', 'install', "
                    f"*{package_value!r}], check=True)"
                )
                data = await jobs.run_python_wait(
                    install_code,
                    timeout_seconds=timeout_seconds,
                    execution_timeout_seconds=execution_timeout_seconds,
                )
                return response(data, "Colab package installation")
        raise ValueError(
            "Tracked package installation is unavailable in this Colab frontend"
        )

    return mcp


def init_logger(logdir: Path) -> Path:
    logdir.mkdir(parents=True, exist_ok=True)
    log_filename = logdir / "colab-codex-adapter.log"
    logging.basicConfig(
        format="%(asctime)s %(levelname)s:%(message)s",
        filename=str(log_filename),
        level=logging.INFO,
        force=True,
    )
    _install_sensitive_log_filters()
    fastmcp_logger.get_logger("colab-codex-adapter").info(
        "logging to %s", log_filename
    )
    return log_filename


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Codex-compatible Colab MCP adapter")
    parser.add_argument(
        "-l",
        "--log",
        type=Path,
        default=DEFAULT_LOG_DIR,
        help="directory for adapter logs",
    )
    parser.add_argument(
        "--pid-file",
        type=Path,
        default=None,
        help="optional per-shim diagnostic PID file (disabled by default)",
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        default=None,
        help="optional per-shim diagnostic state file (disabled by default)",
    )
    parser.add_argument(
        "--broker-port",
        type=int,
        default=int(os.environ.get("COLAB_CODEX_BROKER_PORT", DEFAULT_BROKER_PORT)),
        help="loopback port used by the shared multi-agent broker",
    )
    parser.add_argument(
        "--broker-state-file",
        type=Path,
        default=DEFAULT_BROKER_STATE_FILE,
        help="private shared-broker endpoint and token state",
    )
    parser.add_argument(
        "--broker-lock-file",
        type=Path,
        default=DEFAULT_BROKER_LOCK_FILE,
        help="daemon lifetime ownership lock for the shared broker",
    )
    parser.add_argument(
        "--broker-launch-lock-file",
        type=Path,
        default=DEFAULT_BROKER_LAUNCH_LOCK_FILE,
        help="short-held lock used to elect one detached broker launcher",
    )
    return parser.parse_args()


async def run_broker_daemon_backend(state: BrokerState) -> None:
    """Run the shared authenticated backend inside the detached daemon."""
    endpoint = urlsplit(state.endpoint)
    host = endpoint.hostname or DEFAULT_BROKER_HOST
    port = endpoint.port or DEFAULT_BROKER_PORT
    log_file = init_logger(DEFAULT_LOG_DIR)
    logging.info(
        "Colab broker owner transition previous_owner_id=%s "
        "previous_generation=%s owner_id=%s generation=%s",
        state.previous_owner_id or None,
        state.previous_generation,
        state.owner_id,
        state.generation,
    )
    browser_state_file = _scoped_runtime_path(DEFAULT_BROWSER_STATE_FILE, port)
    journal_file = _scoped_runtime_path(DEFAULT_JOB_JOURNAL_FILE, port)
    artifact_root = _scoped_runtime_path(DEFAULT_ARTIFACT_DIR, port)
    session: ColabSessionManager | None = None
    background_tasks: set[asyncio.Task[Any]] = set()
    jobs: ColabJobManager | None = None
    try:
        session = await _start_recovered_session(browser_state_file)
        artifacts = ArtifactStore(artifact_root)
        jobs = ColabJobManager(
            session,
            artifact_store=artifacts,
            journal_path=journal_file,
        )
        runtime_info = {
            "adapter_version": __version__,
            "service_instance_id": state.service_instance_id,
            "service_pid": state.owner_pid,
            "service_owner_id": state.owner_id,
            "service_generation": state.generation,
            "service_started_at": state.started_at,
            "service_status": "ready",
            "service_healthy": True,
            "instance_scope": "user",
            "transport": "stdio",
            "broker_alive": True,
            "broker_pid": state.owner_pid,
            "broker_owner_id": state.owner_id,
            "broker_generation": state.generation,
            "broker_status": "serving",
            "broker_endpoint": state.endpoint,
            "log_dir": str(DEFAULT_LOG_DIR),
            "log_file": str(log_file),
        }
        verifier = StaticTokenVerifier(
            {
                state.token: {
                    "client_id": "colab-codex-adapter",
                    "scopes": [],
                }
            }
        )
        mcp = create_mcp(
            session,
            runtime_info,
            job_manager=jobs,
            artifact_store=artifacts,
            background_tasks=background_tasks,
            browser_state_file=browser_state_file,
            auth=verifier,
        )
        await mcp.run_http_async(
            show_banner=False,
            host=host,
            port=port,
            log_level="critical",
            uvicorn_config={"access_log": False, "log_config": None},
            json_response=True,
            stateless_http=True,
        )
    finally:
        for task in tuple(background_tasks):
            task.cancel()
        if background_tasks:
            await asyncio.gather(*background_tasks, return_exceptions=True)
        if jobs is not None:
            await jobs.detach_for_shutdown("Broker owner stopped")
        if session is not None:
            await session.close()


def _proxy_server(launcher: BrokerLauncher) -> FastMCPProxy:
    clients = BrokerClientFactory(launcher, timeout=1200.0, init_timeout=30.0)

    class BrokerOwnerLost(RuntimeError):
        def __init__(self, state: BrokerState) -> None:
            self.state = state
            super().__init__("Broker owner stopped during proxy operation")

    class ProxyOperationFailed(RuntimeError):
        def __init__(self, error: Exception, state: BrokerState | None) -> None:
            self.error = error
            self.state = state
            super().__init__("Broker proxy operation failed")

    async def owner_watched(
        operation: Any,
    ) -> tuple[Any, BrokerState | None]:
        task = asyncio.create_task(operation)
        try:
            while True:
                done, _ = await asyncio.wait({task}, timeout=0.25)
                if done:
                    state = clients.state_for(task)
                    try:
                        return await task, state
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        raise ProxyOperationFailed(exc, state) from exc
                state = clients.state_for(task)
                if state is None:
                    continue
                identity_changed = bool(
                    state.owner_process_start
                    and broker_process_start_identity(state.owner_pid)
                    != state.owner_process_start
                )
                if not broker_process_is_alive(state.owner_pid) or identity_changed:
                    task.cancel()
                    try:
                        await asyncio.wait_for(
                            asyncio.gather(task, return_exceptions=True), timeout=2.0
                        )
                    except asyncio.TimeoutError:
                        logging.warning(
                            "Timed out cleaning failed broker client generation=%s",
                            state.generation,
                        )
                    raise BrokerOwnerLost(state)
        except asyncio.CancelledError:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            raise
        finally:
            clients.clear(task)

    class RecoveringProxyToolManager(ProxyToolManager):
        async def replay_read_only(
            self, key: str, arguments: dict[str, Any]
        ) -> ToolResult:
            try:
                result, _ = await owner_watched(
                    super().call_tool(key, arguments)
                )
                return result
            except ProxyOperationFailed as failure:
                raise failure.error
            except BrokerOwnerLost as failure:
                raise ToolError(
                    "Broker owner changed again during the read-only retry"
                ) from failure

        async def get_tools(self) -> dict[str, Any]:
            try:
                result, state = await owner_watched(super().get_tools())
            except asyncio.CancelledError:
                raise
            except (BrokerOwnerLost, ProxyOperationFailed):
                # Discovery is read-only, so replaying it once is unambiguous.
                await launcher.ensure_running()
                try:
                    result, state = await owner_watched(super().get_tools())
                except ProxyOperationFailed as failure:
                    raise failure.error
            current = asyncio.current_task()
            if current is not None and state is not None:
                clients.remember(current, state)
            return result

        async def call_tool(
            self, key: str, arguments: dict[str, Any]
        ) -> ToolResult:
            try:
                result, _ = await owner_watched(
                    super().call_tool(key, arguments)
                )
                return result
            except asyncio.CancelledError:
                raise
            except BrokerOwnerLost as exc:
                try:
                    await launcher.ensure_running()
                except Exception as recovery_exc:
                    raise ToolError(
                        "Broker connection failed during the tool call; the outcome "
                        "is unknown and broker recovery did not complete"
                    ) from recovery_exc
                if key in _READ_ONLY_PROXY_TOOLS:
                    return await self.replay_read_only(key, arguments)
                raise ToolError(
                    "Broker owner changed during the tool call; the outcome is "
                    "unknown. Check connector and job status before retrying"
                ) from exc
            except ProxyOperationFailed as failure:
                try:
                    current = await launcher.ensure_running()
                except Exception as recovery_exc:
                    raise ToolError(
                        "Broker connection failed during the tool call; the outcome "
                        "is unknown and broker recovery did not complete"
                    ) from recovery_exc
                used = failure.state
                if used is not None and (
                    used.owner_id == current.owner_id
                    and used.owner_pid == current.owner_pid
                    and used.generation == current.generation
                ):
                    raise failure.error
                if key in _READ_ONLY_PROXY_TOOLS:
                    return await self.replay_read_only(key, arguments)
                raise ToolError(
                    "Broker owner changed during the tool call; the outcome is "
                    "unknown. Check connector and job status before retrying"
                ) from failure.error

    proxy = FastMCPProxy(
        client_factory=clients,
        name="ColabCodexAdapter",
        mask_error_details=True,
    )
    proxy._tool_manager = RecoveringProxyToolManager(
        client_factory=clients,
        transformations=proxy._tool_manager.transformations,
        mask_error_details=True,
    )
    return proxy


async def main_async() -> None:
    args = parse_args()
    log_file = init_logger(args.log)
    started_at = time.time()
    config = BrokerLaunchConfig(
        host=DEFAULT_BROKER_HOST,
        port=args.broker_port,
        state_file=args.broker_state_file,
        lock_file=args.broker_lock_file,
        launch_lock_file=args.broker_launch_lock_file,
    )
    launcher = BrokerLauncher(config)
    broker_state = await launcher.ensure_running()
    runtime_info = {
        "adapter_version": __version__,
        "service_instance_id": broker_state.service_instance_id,
        "service_pid": broker_state.owner_pid,
        "service_owner_id": broker_state.owner_id,
        "service_generation": broker_state.generation,
        "service_started_at": broker_state.started_at,
        "service_status": broker_state.status,
        "service_healthy": True,
        "instance_scope": "user",
        "transport": "stdio",
        "stdio_proxy_pid": os.getpid(),
        "stdio_proxy_started_at": started_at,
        "broker_owner": False,
        "broker_pid": broker_state.owner_pid,
        "broker_owner_id": broker_state.owner_id,
        "broker_generation": broker_state.generation,
        "broker_status": broker_state.status,
        "broker_endpoint": broker_state.endpoint,
        "log_dir": str(args.log),
        "log_file": str(log_file),
    }
    if args.pid_file is not None:
        runtime_info["pid_file"] = str(args.pid_file)
        write_pid(args.pid_file, os.getpid())
    if args.state_file is not None:
        runtime_info["state_file"] = str(args.state_file)
        write_state(args.state_file, {**runtime_info, "state": "running"})
    proxy = _proxy_server(launcher)
    try:
        await proxy.run_async(show_banner=False)
    except Exception as exc:
        logging.exception("Colab Codex adapter exited with an unhandled exception")
        if args.state_file is not None:
            write_state(
                args.state_file,
                {
                    **runtime_info,
                    "state": "crashed",
                    "error_type": type(exc).__name__,
                },
            )
        raise
    finally:
        if args.state_file is not None:
            write_state(args.state_file, {**runtime_info, "state": "stopped"})


def main() -> None:
    try:
        asyncio.run(main_async())
    except NotConnectedError as exc:
        raise SystemExit(str(exc)) from exc
