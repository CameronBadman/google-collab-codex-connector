from __future__ import annotations

import asyncio
import time
from typing import Any

from mcp.types import CallToolResult, TextContent, Tool

from colab_codex_adapter.jobs import ColabJobManager


class FakeSession:
    def __init__(
        self,
        run_outputs: list[dict[str, Any]] | None = None,
        run_gate: asyncio.Event | None = None,
        run_error: Exception | None = None,
    ) -> None:
        self.cells: list[dict[str, Any]] = []
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.run_outputs = run_outputs
        self.run_gate = run_gate
        self.run_error = run_error

    async def list_tools(self) -> list[Tool]:
        return [
            Tool(name="add_code_cell", inputSchema={"type": "object"}),
            Tool(name="run_code_cell", inputSchema={"type": "object"}),
            Tool(name="get_cells", inputSchema={"type": "object"}),
        ]

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        timeout: float = 300.0,
    ) -> CallToolResult:
        del timeout
        arguments = arguments or {}
        self.calls.append((name, arguments))
        if name == "get_cells":
            cells = self.cells
            if not arguments.get("includeOutputs"):
                cells = [
                    {key: value for key, value in cell.items() if key != "outputs"}
                    for cell in self.cells
                ]
            return result({"cells": cells})
        if name == "add_code_cell":
            cell_id = f"cell-{len(self.cells)}"
            self.cells.append(
                {
                    "id": cell_id,
                    "cell_type": "code",
                    "source": [arguments["code"]],
                    "outputs": [],
                }
            )
            return result({"newCellId": cell_id})
        if name == "run_code_cell":
            if self.run_gate is not None:
                await self.run_gate.wait()
            if self.run_error is not None:
                raise self.run_error
            for cell in self.cells:
                if cell["id"] == arguments["cellId"]:
                    if self.run_outputs is None:
                        self.run_outputs = [
                            {
                                "output_type": "stream",
                                "name": "stdout",
                                "text": ["ok\n"],
                            }
                        ]
                    cell["outputs"] = self.run_outputs
                    return result({"outputs": cell["outputs"]})
        raise AssertionError(f"unexpected tool call: {name}")


def result(data: dict[str, Any]) -> CallToolResult:
    import json

    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(data))],
        structuredContent=data,
    )


async def test_run_python_async_returns_before_execution_finishes() -> None:
    gate = asyncio.Event()
    manager = ColabJobManager(FakeSession(run_gate=gate))  # type: ignore[arg-type]

    started = await manager.start_python("print('ok')")
    assert started["state"] == "running"
    assert started["task_alive"] is True
    assert started["updated_at"] == started["started_at"]
    assert started["last_output_at"] is None
    gate.set()
    status = await manager.wait(started["job_id"], timeout_seconds=1.0)

    assert started["cell_id"] == "cell-0"
    assert status["outputs"][0]["text"] == ["ok\n"]
    assert status["state"] == "finished"
    assert status["task_alive"] is False
    assert status["updated_at"] >= status["started_at"]
    assert status["last_output_at"] is not None


async def test_wait_job_returns_existing_finished_job() -> None:
    manager = ColabJobManager(FakeSession())  # type: ignore[arg-type]
    started = await manager.start_python("print('ok')")

    waited = await manager.wait(started["job_id"], timeout_seconds=1.0)

    assert waited["state"] == "finished"
    assert waited["timed_out"] is False


async def test_list_jobs_returns_tracked_jobs() -> None:
    manager = ColabJobManager(FakeSession())  # type: ignore[arg-type]
    started = await manager.start_python("print('ok')")

    assert manager.list_jobs()[0]["job_id"] == started["job_id"]


async def test_wait_job_times_out_without_cancelling_execution() -> None:
    gate = asyncio.Event()
    manager = ColabJobManager(
        FakeSession(run_outputs=[], run_gate=gate)  # type: ignore[arg-type]
    )
    started = await manager.start_python("while True: pass")

    waited = await manager.wait(started["job_id"], timeout_seconds=1.0)

    assert waited["state"] == "running"
    assert waited["timed_out"] is True
    assert waited["wait_timed_out"] is True
    assert waited["waited_seconds"] >= 1.0
    gate.set()
    finished = await manager.wait(started["job_id"], timeout_seconds=1.0)
    assert finished["state"] == "finished"
    assert finished["outputs"] == []


async def test_wait_returns_as_soon_as_completion_event_fires() -> None:
    gate = asyncio.Event()
    session = FakeSession(run_gate=gate)
    manager = ColabJobManager(session)  # type: ignore[arg-type]
    started = await manager.start_python("train()")
    get_cells_before = sum(call[0] == "get_cells" for call in session.calls)
    wait_started = time.monotonic()
    waiting = asyncio.create_task(
        manager.wait(started["job_id"], timeout_seconds=900.0)
    )

    await asyncio.sleep(0.02)
    gate.set()
    finished = await waiting

    assert finished["state"] == "finished"
    assert finished["wait_timed_out"] is False
    assert time.monotonic() - wait_started < 0.5
    assert sum(call[0] == "get_cells" for call in session.calls) - get_cells_before == 1


async def test_wait_timeout_bounds_are_enforced() -> None:
    manager = ColabJobManager(FakeSession())  # type: ignore[arg-type]
    started = await manager.start_python("print('ok')")

    await manager.wait(started["job_id"], timeout_seconds=1.0)
    await manager.wait(started["job_id"], timeout_seconds=900.0)

    for invalid in (0.0, 901.0, float("inf")):
        try:
            await manager.wait(started["job_id"], timeout_seconds=invalid)
        except ValueError as exc:
            assert "between 1 and 900" in str(exc)
        else:
            raise AssertionError(f"timeout {invalid} should have been rejected")


async def test_empty_outputs_mark_completed_execution_finished() -> None:
    manager = ColabJobManager(
        FakeSession(run_outputs=[])  # type: ignore[arg-type]
    )

    started = await manager.start_python("x = 1")
    finished = await manager.wait(started["job_id"], timeout_seconds=1.0)

    assert finished["state"] == "finished"
    assert finished["outputs"] == []


async def test_error_output_marks_job_error() -> None:
    manager = ColabJobManager(
        FakeSession(
            run_outputs=[
                {
                    "output_type": "error",
                    "ename": "ValueError",
                    "evalue": "bad",
                }
            ]
        )
    )  # type: ignore[arg-type]

    started = await manager.start_python("raise ValueError('bad')")
    finished = await manager.wait(started["job_id"], timeout_seconds=1.0)

    assert finished["state"] == "error"
    assert finished["error"] == "ValueError: bad"


async def test_mark_stale_marks_running_jobs_only() -> None:
    gate = asyncio.Event()
    manager = ColabJobManager(
        FakeSession(run_outputs=[], run_gate=gate)  # type: ignore[arg-type]
    )
    started = await manager.start_python("while True: pass")

    await manager.mark_stale("reset")

    status = await manager.status(started["job_id"])
    assert status["state"] == "stale"
    assert status["error"] == "reset"


async def test_execution_timeout_marks_job_timed_out() -> None:
    manager = ColabJobManager(
        FakeSession(run_error=asyncio.TimeoutError())  # type: ignore[arg-type]
    )

    started = await manager.start_python("train()", execution_timeout_seconds=12)
    finished = await manager.wait(started["job_id"], timeout_seconds=1.0)

    assert finished["state"] == "timed_out"
    assert "12 seconds" in finished["error"]


async def test_remote_exception_marks_job_error() -> None:
    manager = ColabJobManager(
        FakeSession(run_error=RuntimeError("disconnected"))  # type: ignore[arg-type]
    )

    started = await manager.start_python("train()")
    finished = await manager.wait(started["job_id"], timeout_seconds=1.0)

    assert finished["state"] == "error"
    assert finished["error"] == "RuntimeError: disconnected"
