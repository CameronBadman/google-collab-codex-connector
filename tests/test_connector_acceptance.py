from __future__ import annotations

import asyncio
import json
from contextlib import suppress
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import websockets
from fastmcp import Client
from mcp.types import CallToolResult as McpCallToolResult
from mcp.types import TextContent
from mcp.types import Tool
from websockets.exceptions import ConnectionClosed

from colab_codex_adapter.artifacts import ArtifactStore
from colab_codex_adapter.bridge import COLAB, HARD_MAX_WEBSOCKET_FRAME_BYTES
from colab_codex_adapter.server import create_mcp
from colab_codex_adapter.session import (
    ConnectionTransition,
    ColabSessionManager,
)
from colab_codex_adapter.tools import DEFAULT_MAX_TOOL_RESPONSE_BYTES


async def _wait_until(predicate: Callable[[], bool], timeout: float = 2.0) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.01)


class RecordingRemoteSession:
    def __init__(
        self,
        result: McpCallToolResult | None = None,
        tools: list[Tool] | None = None,
    ) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.listeners: list[Any] = []
        self.runtime_generation: int | None = None
        self.result = result or McpCallToolResult(
            content=[TextContent(type="text", text="ok")],
            structuredContent={"ok": True},
        )
        self.tools = tools or []

    def add_connection_listener(self, listener: Any) -> None:
        self.listeners.append(listener)

    async def list_tools(self) -> list[Tool]:
        return self.tools

    async def call_tool(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> McpCallToolResult:
        self.calls.append((name, arguments or {}))
        return self.result


class NoopJobs:
    async def reconcile_detached(self) -> list[dict[str, Any]]:
        return []

    async def on_browser_disconnect(self, reason: str) -> None:
        del reason


async def test_large_remote_result_is_bounded_on_fastmcp_client_envelope(
    tmp_path: Path,
) -> None:
    large_value = "x" * (2 * 1024 * 1024)
    remote_result = McpCallToolResult(
        content=[TextContent(type="text", text=large_value)],
        structuredContent={"cells": [{"id": "large", "source": large_value}]},
    )
    session = RecordingRemoteSession(
        remote_result,
        tools=[Tool(name="get_cells", inputSchema={"type": "object"})],
    )
    artifacts = ArtifactStore(
        tmp_path / "artifacts",
        max_artifact_bytes=8 * 1024 * 1024,
        max_total_bytes=16 * 1024 * 1024,
    )
    mcp = create_mcp(
        manager=session,  # type: ignore[arg-type]
        job_manager=NoopJobs(),  # type: ignore[arg-type]
        artifact_store=artifacts,
    )

    async with Client(mcp) as client:
        result = await client.call_tool(
            "colab_get_notebook",
            {},
        )

    protocol_result = McpCallToolResult(
        content=result.content,
        structuredContent=result.structured_content,
        isError=result.is_error,
        _meta=result.meta,
    )
    result_json = protocol_result.model_dump_json(
        by_alias=True, exclude_none=True
    ).encode("utf-8")
    jsonrpc_frame = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "result": json.loads(result_json),
        },
        separators=(",", ":"),
    ).encode("utf-8")

    assert len(result_json) <= DEFAULT_MAX_TOOL_RESPONSE_BYTES
    assert len(jsonrpc_frame) <= DEFAULT_MAX_TOOL_RESPONSE_BYTES
    assert large_value.encode("utf-8") not in result_json
    assert result.structured_content is not None
    bounded_remote = result.structured_content
    assert bounded_remote["response_truncated"] is True
    artifact_id = bounded_remote["response_artifact"]["artifact_id"]
    assert artifacts.get_ref(artifact_id).size_bytes > 2 * 1024 * 1024


async def test_raw_remote_calls_are_disabled_before_browser_transport() -> None:
    session = RecordingRemoteSession()
    mcp = create_mcp(
        manager=session,  # type: ignore[arg-type]
        job_manager=NoopJobs(),  # type: ignore[arg-type]
    )

    raw_calls = [
        ("get_cells", None),
        ("GET.CELLS", {"include_outputs": False}),
        ("Get-Cells", {"Include-Outputs": False}),
        ("GET-CELLS", {"Include_Outputs": True}),
        ("getCells", {"includeOutputs": None}),
        ("List.Cells", {"includeOutputs": False}),
        ("Read/Notebook", {}),
        ("NOTEBOOK-INFO", {}),
        ("run_code_cell", {"cellId": "cell-1"}),
    ]

    async with Client(mcp) as client:
        for name, arguments in raw_calls:
            request: dict[str, Any] = {"name": name}
            if arguments is not None:
                request["arguments"] = arguments
            result = await client.call_tool(
                "colab_call_remote_tool",
                request,
                raise_on_error=False,
            )
            assert result.is_error is True

        direct_result = await client.call_tool(
            "colab_get_notebook",
            {"include_outputs": True},
            raise_on_error=False,
        )

    assert direct_result.is_error is True
    assert session.calls == []


async def test_notebook_remote_override_is_rejected_before_browser_transport() -> None:
    session = RecordingRemoteSession(
        tools=[
            Tool(
                name="get_cells",
                inputSchema={"type": "object", "properties": {}},
            )
        ]
    )
    mcp = create_mcp(
        manager=session,  # type: ignore[arg-type]
        job_manager=NoopJobs(),  # type: ignore[arg-type]
    )

    async with Client(mcp) as client:
        result = await client.call_tool(
            "colab_get_notebook",
            {"remote_tool_name": "get_cells"},
            raise_on_error=False,
        )

    assert result.is_error is True
    assert session.calls == []


async def test_cell_remote_overrides_cannot_invoke_untracked_execution() -> None:
    session = RecordingRemoteSession(
        tools=[
            Tool(
                name="run_code_cell",
                inputSchema={
                    "type": "object",
                    "properties": {"cellId": {"type": "string"}},
                },
            )
        ]
    )
    mcp = create_mcp(
        manager=session,  # type: ignore[arg-type]
        job_manager=NoopJobs(),  # type: ignore[arg-type]
    )

    async with Client(mcp) as client:
        added = await client.call_tool(
            "colab_add_cell",
            {"code": "pass", "remote_tool_name": "run_code_cell"},
            raise_on_error=False,
        )
        updated = await client.call_tool(
            "colab_update_cell",
            {
                "code": "pass",
                "cell_id": "cell-7",
                "remote_tool_name": "run_code_cell",
            },
            raise_on_error=False,
        )

    assert added.is_error is True
    assert updated.is_error is True
    assert session.calls == []


async def test_fuzzy_cell_mutation_aliases_fail_closed() -> None:
    session = RecordingRemoteSession(
        tools=[
            Tool(name="create_cell", inputSchema={"type": "object"}),
            Tool(name="replace_cell", inputSchema={"type": "object"}),
        ]
    )
    mcp = create_mcp(
        manager=session,  # type: ignore[arg-type]
        job_manager=NoopJobs(),  # type: ignore[arg-type]
    )

    async with Client(mcp) as client:
        added = await client.call_tool(
            "colab_add_cell", {"code": "pass"}, raise_on_error=False
        )
        updated = await client.call_tool(
            "colab_update_cell",
            {"code": "pass", "cell_id": "cell-1"},
            raise_on_error=False,
        )

    assert added.is_error is True
    assert updated.is_error is True
    assert session.calls == []


class PagedRemoteSession(RecordingRemoteSession):
    def __init__(self, cells: list[dict[str, Any]]) -> None:
        super().__init__(
            tools=[Tool(name="get_cells", inputSchema={"type": "object"})]
        )
        self.cells = cells

    async def call_tool(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> McpCallToolResult:
        arguments = arguments or {}
        self.calls.append((name, arguments))
        assert name == "get_cells"
        assert arguments.get("includeOutputs") is False
        assert isinstance(arguments.get("cellIndexStart"), int)
        assert isinstance(arguments.get("cellIndexEnd"), int)
        cells = [
            {key: value for key, value in cell.items() if key != "outputs"}
            for cell in self.cells[
                arguments["cellIndexStart"] : arguments["cellIndexEnd"] + 1
            ]
        ]
        return McpCallToolResult(
            content=[TextContent(type="text", text=json.dumps({"cells": cells}))],
            structuredContent={"cells": cells},
        )


async def test_notebook_metadata_is_fetched_in_bounded_pages() -> None:
    cells = [
        {"id": f"cell-{index}", "source": ["pass"], "outputs": ["omitted"]}
        for index in range(17)
    ]
    session = PagedRemoteSession(cells)
    mcp = create_mcp(
        manager=session,  # type: ignore[arg-type]
        job_manager=NoopJobs(),  # type: ignore[arg-type]
    )

    async with Client(mcp) as client:
        result = await client.call_tool("colab_get_notebook", {})

    assert result.is_error is False
    assert result.structured_content is not None
    assert len(result.structured_content["cells"]) == 17
    assert all("outputs" not in cell for cell in result.structured_content["cells"])
    assert session.calls == [
        (
            "get_cells",
            {"includeOutputs": False, "cellIndexStart": 0, "cellIndexEnd": 7},
        ),
        (
            "get_cells",
            {"includeOutputs": False, "cellIndexStart": 8, "cellIndexEnd": 15},
        ),
        (
            "get_cells",
            {"includeOutputs": False, "cellIndexStart": 16, "cellIndexEnd": 23},
        ),
    ]


async def test_invalid_arguments_have_a_bounded_masked_wire_error() -> None:
    secret = "PRIVATE_CORPUS_VALUE" * 100_000
    mcp = create_mcp(
        manager=RecordingRemoteSession(),  # type: ignore[arg-type]
        job_manager=NoopJobs(),  # type: ignore[arg-type]
    )

    async with Client(mcp) as client:
        result = await client.call_tool(
            "colab_wait_job",
            {"job_id": "0" * 32, "timeout_seconds": secret},
            raise_on_error=False,
        )

    protocol_result = McpCallToolResult(
        content=result.content,
        structuredContent=result.structured_content,
        isError=result.is_error,
        _meta=result.meta,
    )
    payload = protocol_result.model_dump_json(
        by_alias=True, exclude_none=True
    ).encode("utf-8")
    assert result.is_error is True
    assert len(payload) < 1024
    assert b"PRIVATE_CORPUS_VALUE" not in payload


class BusyArtifactJobs(NoopJobs):
    async def read_artifact(
        self, artifact_id: str, *, offset: int, limit_bytes: int
    ) -> dict[str, Any]:
        del artifact_id, offset, limit_bytes
        raise asyncio.TimeoutError


async def test_busy_runtime_artifact_read_returns_retryable_bounded_error() -> None:
    mcp = create_mcp(
        manager=RecordingRemoteSession(),  # type: ignore[arg-type]
        job_manager=BusyArtifactJobs(),  # type: ignore[arg-type]
    )

    async with Client(mcp) as client:
        result = await client.call_tool(
            "colab_read_artifact",
            {"artifact_id": "a" * 32},
            raise_on_error=False,
        )

    payload = json.dumps(
        [item.model_dump() for item in result.content], separators=(",", ":")
    ).encode("utf-8")
    assert result.is_error is True
    assert b"runtime is busy" in payload.lower()
    assert len(payload) < 1024


class TransitionSession:
    def __init__(self) -> None:
        self.runtime_generation: int | None = None
        self.listeners: list[Any] = []

    def add_connection_listener(self, listener: Any) -> None:
        self.listeners.append(listener)

    async def emit(self, transition: ConnectionTransition) -> None:
        self.runtime_generation = (
            transition.browser_generation
            if transition.kind == "runtime_ready"
            else None
        )
        for listener in self.listeners:
            await listener(transition)


class BlockingReconciliationJobs:
    def __init__(self) -> None:
        self.started = [asyncio.Event(), asyncio.Event()]
        self.release = [asyncio.Event(), asyncio.Event()]
        self.cancelled: list[int] = []
        self.completed: list[int] = []
        self.calls = 0

    async def reconcile_detached(self) -> list[dict[str, Any]]:
        index = self.calls
        self.calls += 1
        self.started[index].set()
        try:
            await self.release[index].wait()
        except asyncio.CancelledError:
            self.cancelled.append(index)
            raise
        self.completed.append(index)
        return []

    async def on_browser_disconnect(self, reason: str) -> None:
        del reason


def _transition(kind: Any, generation: int) -> ConnectionTransition:
    return ConnectionTransition(
        kind=kind,
        connection_id="acceptance-connection",
        browser_generation=generation,
        occurred_at=1.0,
        browser_alive=kind == "runtime_ready",
        runtime_alive=kind == "runtime_ready",
    )


async def test_new_runtime_generation_cancels_previous_reconciliation() -> None:
    session = TransitionSession()
    jobs = BlockingReconciliationJobs()
    background_tasks: set[asyncio.Task[Any]] = set()
    create_mcp(
        manager=session,  # type: ignore[arg-type]
        job_manager=jobs,  # type: ignore[arg-type]
        background_tasks=background_tasks,
    )

    try:
        await session.emit(_transition("runtime_ready", 1))
        await asyncio.wait_for(jobs.started[0].wait(), timeout=1.0)

        await session.emit(_transition("runtime_ready", 2))
        await asyncio.wait_for(jobs.started[1].wait(), timeout=1.0)
        await _wait_until(lambda: jobs.cancelled == [0])

        jobs.release[1].set()
        await _wait_until(lambda: jobs.completed == [1])
        await session.emit(_transition("browser_disconnected", 2))
    finally:
        for task in tuple(background_tasks):
            task.cancel()
        if background_tasks:
            await asyncio.gather(*background_tasks, return_exceptions=True)

    assert jobs.calls == 2
    assert jobs.cancelled == [0]
    assert jobs.completed == [1]


class LateDetachJobs:
    def __init__(self) -> None:
        self.first_scan = asyncio.Event()
        self.reconciled = asyncio.Event()
        self.detached = False
        self.calls = 0

    async def reconcile_detached(self) -> list[dict[str, Any]]:
        self.calls += 1
        self.first_scan.set()
        if not self.detached:
            return []
        self.reconciled.set()
        return [{"state": "finished", "tracking_state": "complete"}]

    async def on_browser_disconnect(self, reason: str) -> None:
        del reason


async def test_runtime_reconciler_observes_job_detached_after_initial_scan() -> None:
    session = TransitionSession()
    jobs = LateDetachJobs()
    background_tasks: set[asyncio.Task[Any]] = set()
    create_mcp(
        manager=session,  # type: ignore[arg-type]
        job_manager=jobs,  # type: ignore[arg-type]
        background_tasks=background_tasks,
        reconciliation_poll_seconds=0.1,
    )

    try:
        await session.emit(_transition("runtime_ready", 1))
        await asyncio.wait_for(jobs.first_scan.wait(), timeout=1.0)
        jobs.detached = True
        await asyncio.wait_for(jobs.reconciled.wait(), timeout=1.0)
        await session.emit(_transition("browser_disconnected", 1))
    finally:
        for task in tuple(background_tasks):
            task.cancel()
        if background_tasks:
            await asyncio.gather(*background_tasks, return_exceptions=True)

    assert jobs.calls >= 2


class TransientReconciliationJobs:
    def __init__(self) -> None:
        self.calls = 0
        self.retried = asyncio.Event()

    async def reconcile_detached(self) -> list[dict[str, Any]]:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("transient probe failure")
        self.retried.set()
        return []

    async def on_browser_disconnect(self, reason: str) -> None:
        del reason


async def test_runtime_reconciler_retries_after_transient_probe_error() -> None:
    session = TransitionSession()
    jobs = TransientReconciliationJobs()
    background_tasks: set[asyncio.Task[Any]] = set()
    create_mcp(
        manager=session,  # type: ignore[arg-type]
        job_manager=jobs,  # type: ignore[arg-type]
        background_tasks=background_tasks,
        reconciliation_poll_seconds=0.1,
    )

    try:
        await session.emit(_transition("runtime_ready", 1))
        await asyncio.wait_for(jobs.retried.wait(), timeout=1.0)
        await session.emit(_transition("browser_disconnected", 1))
    finally:
        for task in tuple(background_tasks):
            task.cancel()
        if background_tasks:
            await asyncio.gather(*background_tasks, return_exceptions=True)

    assert jobs.calls >= 2


class ScriptedFrontend:
    def __init__(
        self,
        url: str,
        *,
        delayed_method: str | None = None,
        delay_seconds: float = 0.0,
    ) -> None:
        self.url = url
        self.delayed_method = delayed_method
        self.delay_seconds = delay_seconds
        self.connected = asyncio.Event()
        self.websocket: Any = None
        self.task: asyncio.Task[None] | None = None

    def start(self) -> None:
        self.task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        try:
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
                    if method == self.delayed_method:
                        await asyncio.sleep(self.delay_seconds)
                    if method == "initialize":
                        result = {
                            "protocolVersion": message["params"]["protocolVersion"],
                            "capabilities": {"tools": {"listChanged": False}},
                            "serverInfo": {
                                "name": "acceptance-frontend",
                                "version": "1.0",
                            },
                        }
                    elif method == "tools/list":
                        result = {"tools": []}
                    else:
                        result = {}
                    await websocket.send(
                        json.dumps(
                            {
                                "jsonrpc": "2.0",
                                "id": request_id,
                                "result": result,
                            }
                        )
                    )
        except ConnectionClosed:
            pass

    async def close(self) -> None:
        if self.websocket is not None:
            with suppress(ConnectionClosed):
                await self.websocket.close(code=1001, reason="test complete")
        if self.task is not None:
            await self.task


def _websocket_url(manager: ColabSessionManager) -> str:
    bridge = manager.bridge
    return f"ws://127.0.0.1:{bridge.port}/?access_token={bridge.token}"


@pytest.mark.parametrize(
    ("delayed_method", "expected_stage"),
    [("initialize", "initialization"), ("tools/list", "tool discovery")],
)
async def test_configured_frontend_timeout_recovers_on_same_url(
    delayed_method: str, expected_stage: str
) -> None:
    manager = ColabSessionManager(
        bridge_host="127.0.0.1",
        remote_init_timeout_seconds=0.05,
        remote_tool_list_timeout_seconds=0.05,
    )
    await manager.start()
    initial_status = await manager.status()
    websocket_url = _websocket_url(manager)
    slow_frontend = ScriptedFrontend(
        websocket_url,
        delayed_method=delayed_method,
        delay_seconds=0.15,
    )
    recovered_frontend: ScriptedFrontend | None = None

    try:
        slow_frontend.start()
        timed_out = await manager.connect(wait_seconds=0.25)
        await _wait_until(lambda: not manager.browser_ws_connected())

        assert timed_out.connected is False
        assert timed_out.browser_generation == 1
        assert timed_out.last_error == f"Timed out during Colab frontend {expected_stage}"
        assert timed_out.last_browser_close_code == 1011

        recovered_frontend = ScriptedFrontend(websocket_url)
        recovered_frontend.start()
        recovered = await manager.connect(wait_seconds=1.0)

        assert recovered.connected is True
        assert recovered.browser_generation == 2
        assert recovered.runtime_generation == 2
        assert recovered.connection_id == initial_status.connection_id
        assert recovered.url == initial_status.url
        assert recovered.last_error is None
    finally:
        await manager.close()
        await slow_frontend.close()
        if recovered_frontend is not None:
            await recovered_frontend.close()
