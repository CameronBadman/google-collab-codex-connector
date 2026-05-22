from __future__ import annotations

from typing import Any

from mcp.types import CallToolResult, TextContent, Tool

from colab_codex_adapter.jobs import ColabJobManager


class FakeSession:
    def __init__(self, run_outputs: list[dict[str, Any]] | None = None) -> None:
        self.cells: list[dict[str, Any]] = []
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.run_outputs = run_outputs

    async def list_tools(self) -> list[Tool]:
        return [
            Tool(name="add_code_cell", inputSchema={"type": "object"}),
            Tool(name="run_code_cell", inputSchema={"type": "object"}),
            Tool(name="get_cells", inputSchema={"type": "object"}),
        ]

    async def call_tool(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> CallToolResult:
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


async def test_run_python_async_tracks_finished_job() -> None:
    manager = ColabJobManager(FakeSession())  # type: ignore[arg-type]

    started = await manager.start_python("print('ok')")
    status = await manager.status(started["job_id"])

    assert started["state"] == "finished"
    assert started["cell_id"] == "cell-0"
    assert status["outputs"][0]["text"] == ["ok\n"]


async def test_wait_job_returns_existing_finished_job() -> None:
    manager = ColabJobManager(FakeSession())  # type: ignore[arg-type]
    started = await manager.start_python("print('ok')")

    waited = await manager.wait(started["job_id"], timeout_seconds=0.01)

    assert waited["state"] == "finished"
    assert waited["timed_out"] is False


async def test_list_jobs_returns_tracked_jobs() -> None:
    manager = ColabJobManager(FakeSession())  # type: ignore[arg-type]
    started = await manager.start_python("print('ok')")

    assert manager.list_jobs()[0]["job_id"] == started["job_id"]


async def test_wait_job_times_out_when_outputs_are_empty() -> None:
    manager = ColabJobManager(FakeSession(run_outputs=[]))  # type: ignore[arg-type]
    started = await manager.start_python("while True: pass")

    waited = await manager.wait(started["job_id"], timeout_seconds=0.01)

    assert waited["state"] == "running"
    assert waited["timed_out"] is True


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

    assert started["state"] == "error"
    assert started["error"] == "ValueError: bad"


async def test_mark_stale_marks_running_jobs_only() -> None:
    manager = ColabJobManager(FakeSession(run_outputs=[]))  # type: ignore[arg-type]
    started = await manager.start_python("while True: pass")

    manager.mark_stale("reset")

    status = await manager.status(started["job_id"])
    assert status["state"] == "stale"
    assert status["error"] == "reset"
