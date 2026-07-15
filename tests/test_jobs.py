from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest
from mcp.types import CallToolResult, TextContent, Tool

from colab_codex_adapter import jobs as jobs_module
from colab_codex_adapter.activity import ActivityEvent, ActivityPhase
from colab_codex_adapter.jobs import ColabJob, ColabJobManager
from colab_codex_adapter.artifacts import ArtifactNotFoundError, ArtifactStore


class FakeSession:
    def __init__(
        self,
        run_outputs: list[dict[str, Any]] | None = None,
        run_gate: asyncio.Event | None = None,
        run_error: Exception | None = None,
        run_data: dict[str, Any] | None = None,
        connection_id: str = "fake-connection",
        manifest_output_excerpt: str = "ok\n",
        manifest_output_bytes: int | None = None,
        manifest_artifact_size: int = 0,
        manifest_state: str = "finished",
        manifest_error: str | None = None,
    ) -> None:
        self.cells: list[dict[str, Any]] = []
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.run_outputs = run_outputs
        self.run_gate = run_gate
        self.run_error = run_error
        self.run_data = run_data
        self.connection_id = connection_id
        self.manifest_output_excerpt = manifest_output_excerpt
        self.manifest_output_bytes = manifest_output_bytes
        self.manifest_artifact_size = manifest_artifact_size
        self.manifest_state = manifest_state
        self.manifest_error = manifest_error

    async def list_tools(self) -> list[Tool]:
        return [
            Tool(name="add_code_cell", inputSchema={"type": "object"}),
            Tool(name="run_code_cell", inputSchema={"type": "object"}),
            Tool(name="get_cells", inputSchema={"type": "object"}),
            Tool(name="update_cell", inputSchema={"type": "object"}),
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
            start = arguments.get("cellIndexStart")
            end = arguments.get("cellIndexEnd")
            if isinstance(start, int) or isinstance(end, int):
                cells = cells[
                    start if isinstance(start, int) else 0 :
                    end + 1 if isinstance(end, int) else None
                ]
            return result({"cells": cells})
        if name == "add_code_cell":
            cell_id = f"cell-{len(self.cells)}"
            cell = {
                "id": cell_id,
                "cell_type": "code",
                "source": [arguments["code"]],
                "outputs": [],
            }
            self.cells.insert(arguments["cellIndex"], cell)
            return result({"newCellId": cell_id})
        if name == "update_cell":
            for cell in self.cells:
                if cell["id"] == arguments["cellId"]:
                    cell["source"] = [arguments["content"]]
                    cell["outputs"] = []
                    return result({"cellId": cell["id"]})
        if name == "run_code_cell":
            if self.run_gate is not None:
                await self.run_gate.wait()
            if self.run_error is not None:
                raise self.run_error
            if self.run_data is not None:
                return result(self.run_data)
            for cell in self.cells:
                if cell["id"] == arguments["cellId"]:
                    if self.run_outputs is None:
                        source = "".join(cell["source"])
                        config_line = next(
                            (
                                line
                                for line in source.splitlines()
                                if line.strip().startswith("_cc_config = {")
                            ),
                            None,
                        )
                        if config_line is None:
                            outputs = [
                                {
                                    "output_type": "stream",
                                    "name": "stdout",
                                    "text": ["ok\n"],
                                }
                            ]
                        else:
                            config = json.loads(config_line.split("=", 1)[1].strip())
                            excerpt = self.manifest_output_excerpt
                            output_bytes = (
                                self.manifest_output_bytes
                                if self.manifest_output_bytes is not None
                                else len(excerpt.encode("utf-8"))
                            )
                            manifest = {
                                "job_id": config["job_id"],
                                "state": self.manifest_state,
                                "error": self.manifest_error,
                                "output_bytes": output_bytes,
                                "output_excerpt": excerpt,
                                "output_truncated": output_bytes
                                > len(excerpt.encode("utf-8")),
                                "artifact_size_bytes": self.manifest_artifact_size,
                                "artifact_sha256": "a" * 64,
                                "artifact_truncated": False,
                            }
                            outputs = [
                                {
                                    "output_type": "stream",
                                    "name": "stdout",
                                    "text": [
                                        jobs_module._JOB_SENTINEL
                                        + json.dumps(manifest, separators=(",", ":"))
                                        + "\n"
                                    ],
                                }
                            ]
                    else:
                        outputs = self.run_outputs
                    cell["outputs"] = outputs
                    return result({"outputs": cell["outputs"]})
        raise AssertionError(f"unexpected tool call: {name}")


class AmbiguousAddSession(FakeSession):
    def __init__(self) -> None:
        super().__init__()
        self.fail_after_next_add = True

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        timeout: float = 300.0,
    ) -> CallToolResult:
        result_value = await super().call_tool(name, arguments, timeout)
        if name == "add_code_cell" and self.fail_after_next_add:
            self.fail_after_next_add = False
            raise RuntimeError("add response lost")
        return result_value


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
    assert started["tracking_state"] == "active"
    assert started["execution_alive"] is True
    assert "code" not in started
    assert "add_result" not in started
    gate.set()
    status = await manager.wait(started["job_id"], timeout_seconds=1.0)

    assert started["cell_id"] == "cell-0"
    assert status["outputs"][0]["text"] == ["ok\n"]
    assert status["state"] == "finished"
    assert status["task_alive"] is False
    assert status["updated_at"] >= status["started_at"]
    assert status["last_output_at"] is not None
    assert status["tracking_state"] == "complete"
    assert status["execution_alive"] is False


async def test_run_python_wait_reports_ordered_activity() -> None:
    manager = ColabJobManager(FakeSession())  # type: ignore[arg-type]
    events: list[ActivityEvent] = []

    async def reporter(event: ActivityEvent) -> None:
        events.append(event)

    result_value = await manager.run_python_wait(
        "print('ok')",
        timeout_seconds=1.0,
        reporter=reporter,
    )

    assert result_value["state"] == "finished"
    assert [event.phase for event in events] == [
        ActivityPhase.INITIALIZING_RUNTIME,
        ActivityPhase.PREPARING_CELL,
        ActivityPhase.PREPARING_CELL,
        ActivityPhase.EXECUTING,
        ActivityPhase.WAITING,
        ActivityPhase.FINISHED,
    ]
    assert [event.message for event in events] == [
        "Inspecting Colab runtime",
        "Preparing tracked cell",
        "Adding tracked cell",
        "Starting cell execution",
        "Waiting for cell completion",
        "Cell execution finished",
    ]
    assert all("print('ok')" not in event.message for event in events)


async def test_activity_reporter_failure_does_not_break_execution() -> None:
    manager = ColabJobManager(FakeSession())  # type: ignore[arg-type]

    async def failing_reporter(event: ActivityEvent) -> None:
        del event
        raise RuntimeError("notification transport unavailable")

    result_value = await manager.run_python_wait(
        "print('ok')",
        timeout_seconds=1.0,
        reporter=failing_reporter,
    )

    assert result_value["state"] == "finished"


async def test_concurrent_job_starts_allocate_distinct_cell_indexes() -> None:
    gate = asyncio.Event()
    session = FakeSession(run_gate=gate)
    manager = ColabJobManager(session)  # type: ignore[arg-type]

    first, second = await asyncio.gather(
        manager.start_python("first()"),
        manager.start_python("second()"),
    )

    assert {first["cell_index"], second["cell_index"]} == {0, 1}
    assert {first["cell_id"], second["cell_id"]} == {"cell-0", "cell-1"}
    gate.set()
    await asyncio.gather(
        manager.wait(first["job_id"], timeout_seconds=1.0),
        manager.wait(second["job_id"], timeout_seconds=1.0),
    )


async def test_job_start_counts_existing_cells_in_bounded_metadata_pages() -> None:
    gate = asyncio.Event()
    session = FakeSession(run_gate=gate)
    session.cells.extend(
        {
            "id": f"existing-{index}",
            "cell_type": "code",
            "source": ["pass"],
            "outputs": [],
        }
        for index in range(17)
    )
    manager = ColabJobManager(session)  # type: ignore[arg-type]

    started = await manager.start_python("train()")

    assert started["cell_index"] == 17
    page_calls = [args for name, args in session.calls if name == "get_cells"]
    assert page_calls[:3] == [
        {"includeOutputs": False, "cellIndexStart": 0, "cellIndexEnd": 7},
        {"includeOutputs": False, "cellIndexStart": 8, "cellIndexEnd": 15},
        {"includeOutputs": False, "cellIndexStart": 16, "cellIndexEnd": 23},
    ]
    assert all(
        "cellIndexStart" in args and "cellIndexEnd" in args for args in page_calls
    )
    gate.set()
    await manager.wait(started["job_id"], timeout_seconds=1.0)


async def test_status_finds_job_after_notebook_cell_reorder() -> None:
    gate = asyncio.Event()
    session = FakeSession(run_gate=gate)
    manager = ColabJobManager(session)  # type: ignore[arg-type]
    started = await manager.start_python("train()")
    session.cells.insert(
        0,
        {
            "id": "inserted-before-job",
            "cell_type": "code",
            "source": ["pass"],
            "outputs": [],
        },
    )

    status = await manager.status(started["job_id"])

    assert status["state"] == "running"
    assert status["cell_index"] == 1
    gate.set()
    await manager.wait(started["job_id"], timeout_seconds=1.0)


async def test_status_finds_reordered_job_beyond_first_metadata_page() -> None:
    gate = asyncio.Event()
    session = FakeSession(run_gate=gate)
    session.cells.extend(
        {
            "id": f"existing-{index}",
            "cell_type": "code",
            "source": ["pass"],
            "outputs": [],
        }
        for index in range(17)
    )
    manager = ColabJobManager(session)  # type: ignore[arg-type]
    started = await manager.start_python("train()")
    session.cells.insert(
        0,
        {
            "id": "inserted-before-job",
            "cell_type": "code",
            "source": ["pass"],
            "outputs": [],
        },
    )
    session.calls.clear()

    status = await manager.status(started["job_id"])

    assert status["state"] == "running"
    assert status["cell_index"] == 18
    get_cells = [args for name, args in session.calls if name == "get_cells"]
    assert get_cells[0] == {
        "includeOutputs": False,
        "cellIndexStart": 17,
        "cellIndexEnd": 17,
    }
    assert all(
        "cellIndexStart" in args and "cellIndexEnd" in args for args in get_cells
    )
    gate.set()
    await manager.wait(started["job_id"], timeout_seconds=1.0)


async def test_wait_job_returns_existing_finished_job() -> None:
    manager = ColabJobManager(FakeSession())  # type: ignore[arg-type]
    started = await manager.start_python("print('ok')")

    waited = await manager.wait(started["job_id"], timeout_seconds=1.0)

    assert waited["state"] == "finished"
    assert waited["timed_out"] is False


async def test_list_jobs_returns_tracked_jobs() -> None:
    manager = ColabJobManager(FakeSession())  # type: ignore[arg-type]
    started = await manager.start_python("print('ok')")

    listed = manager.list_jobs()[0]
    assert listed["job_id"] == started["job_id"]
    assert "outputs" not in listed
    assert "code" not in listed


async def test_completed_job_registry_is_bounded() -> None:
    manager = ColabJobManager(
        FakeSession(),  # type: ignore[arg-type]
        max_tracked_jobs=2,
    )
    first = await manager.start_python("first()")
    await manager.wait(first["job_id"], timeout_seconds=1.0)
    second = await manager.start_python("second()")
    await manager.wait(second["job_id"], timeout_seconds=1.0)
    third = await manager.start_python("third()")
    await manager.wait(third["job_id"], timeout_seconds=1.0)

    assert len(manager.jobs) == 2
    assert first["job_id"] not in manager.jobs
    assert {second["job_id"], third["job_id"]} == set(manager.jobs)


async def test_sequential_jobs_reuse_a_bounded_connector_cell_pool() -> None:
    session = FakeSession()
    manager = ColabJobManager(
        session,  # type: ignore[arg-type]
        max_tracked_jobs=2,
        max_job_cells=2,
    )

    for index in range(12):
        started = await manager.start_python(f"result = {index}")
        await manager.wait(started["job_id"], timeout_seconds=1.0)

    assert len(session.cells) == 1
    assert sum(name == "add_code_cell" for name, _ in session.calls) == 1
    assert sum(name == "update_cell" for name, _ in session.calls) == 11
    assert len(manager._job_cells) == 1


async def test_job_cell_pool_bounds_concurrent_unfinished_jobs() -> None:
    gate = asyncio.Event()
    manager = ColabJobManager(
        FakeSession(run_gate=gate),  # type: ignore[arg-type]
        max_tracked_jobs=4,
        max_job_cells=2,
    )
    first = await manager.start_python("first()")
    second = await manager.start_python("second()")

    with pytest.raises(RuntimeError, match="Maximum concurrent Colab job cells"):
        await manager.start_python("third()")

    gate.set()
    await asyncio.gather(
        manager.wait(first["job_id"], timeout_seconds=1.0),
        manager.wait(second["job_id"], timeout_seconds=1.0),
    )


async def test_job_cell_pool_survives_manager_recovery(tmp_path: Path) -> None:
    session = FakeSession()
    journal = tmp_path / "jobs.json"
    first_manager = ColabJobManager(
        session,  # type: ignore[arg-type]
        journal_path=journal,
        max_tracked_jobs=2,
        max_job_cells=2,
    )
    first = await first_manager.start_python("first()")
    await first_manager.wait(first["job_id"], timeout_seconds=1.0)

    recovered = ColabJobManager(
        session,  # type: ignore[arg-type]
        journal_path=journal,
        max_tracked_jobs=2,
        max_job_cells=2,
    )
    second = await recovered.start_python("second()")
    await recovered.wait(second["job_id"], timeout_seconds=1.0)

    assert len(session.cells) == 1
    assert sum(name == "add_code_cell" for name, _ in session.calls) == 1
    assert sum(name == "update_cell" for name, _ in session.calls) == 1


async def test_stale_pooled_cell_is_not_reused_in_a_new_notebook(
    tmp_path: Path,
) -> None:
    session = FakeSession()
    journal = tmp_path / "jobs.json"
    first_manager = ColabJobManager(session, journal_path=journal)  # type: ignore[arg-type]
    first = await first_manager.start_python("first()")
    await first_manager.wait(first["job_id"], timeout_seconds=1.0)
    stale_cell_id = first["cell_id"]

    session.cells = [
        {
            "id": "current-notebook-cell",
            "cell_type": "code",
            "source": ["pass"],
            "outputs": [],
        }
    ]
    session.calls.clear()
    recovered = ColabJobManager(session, journal_path=journal)  # type: ignore[arg-type]
    second = await recovered.start_python("second()")
    await recovered.wait(second["job_id"], timeout_seconds=1.0)

    assert second["cell_id"] != stale_cell_id
    assert not any(
        name == "update_cell" and args["cellId"] == stale_cell_id
        for name, args in session.calls
    )


def test_cell_pool_journal_is_scoped_to_connection_id(tmp_path: Path) -> None:
    journal = tmp_path / "jobs.json"
    first = ColabJobManager(
        FakeSession(connection_id="notebook-a"),  # type: ignore[arg-type]
        journal_path=journal,
    )
    first._job_cells["cell-from-a"] = 4
    first._probe_cell_id = "probe-from-a"
    first._persist_journal()

    recovered = ColabJobManager(
        FakeSession(connection_id="notebook-b"),  # type: ignore[arg-type]
        journal_path=journal,
    )

    assert recovered._job_cells == {}
    assert recovered._probe_cell_id is None


async def test_legacy_finished_journal_without_manifest_proof_is_invalidated(
    tmp_path: Path,
) -> None:
    journal = tmp_path / "jobs.json"
    session = FakeSession()
    first = ColabJobManager(session, journal_path=journal)  # type: ignore[arg-type]
    started = await first.start_python("x = 1")
    await first.wait(started["job_id"], timeout_seconds=1.0)
    data = json.loads(journal.read_text(encoding="utf-8"))
    data["version"] = 2
    data.pop("connection_id", None)
    for record in data["jobs"]:
        record.pop("terminal_manifest_found", None)
        record.pop("completion_source", None)
    journal.write_text(json.dumps(data), encoding="utf-8")

    recovered = ColabJobManager(session, journal_path=journal)  # type: ignore[arg-type]
    status = await recovered.status(started["job_id"])

    assert status["state"] == "interrupted"
    assert status["terminal_manifest_found"] is False
    assert "lacked terminal-manifest proof" in status["output_unavailable_reason"]
    migrated = json.loads(journal.read_text(encoding="utf-8"))
    assert migrated["version"] == 3
    assert migrated["connection_id"] == session.connection_id


async def test_lost_job_cell_add_response_recovers_without_pool_growth() -> None:
    session = AmbiguousAddSession()
    manager = ColabJobManager(
        session,  # type: ignore[arg-type]
        max_tracked_jobs=1,
        max_job_cells=1,
    )

    with pytest.raises(RuntimeError, match="add response lost"):
        await manager.start_python("first()")

    started = await manager.start_python("second()")
    await manager.wait(started["job_id"], timeout_seconds=1.0)

    assert len(session.cells) == 1
    assert len(manager._job_cells) == 1
    assert sum(name == "add_code_cell" for name, _ in session.calls) == 1
    assert sum(name == "update_cell" for name, _ in session.calls) == 1


async def test_running_job_registry_limit_rejects_new_work() -> None:
    gate = asyncio.Event()
    manager = ColabJobManager(
        FakeSession(run_gate=gate),  # type: ignore[arg-type]
        max_tracked_jobs=1,
    )
    started = await manager.start_python("first()")

    with pytest.raises(RuntimeError, match="Maximum tracked"):
        await manager.start_python("second()")

    gate.set()
    await manager.wait(started["job_id"], timeout_seconds=1.0)


def test_tracked_job_limit_cannot_exceed_recoverable_journal_limit() -> None:
    with pytest.raises(ValueError, match="between 1 and 1024"):
        ColabJobManager(
            FakeSession(),  # type: ignore[arg-type]
            max_tracked_jobs=1025,
        )


async def test_submitted_code_size_is_bounded_before_browser_call() -> None:
    session = FakeSession()
    manager = ColabJobManager(session)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="size limit"):
        await manager.start_python(
            "x" * (jobs_module.MAX_SUBMITTED_CODE_BYTES + 1)
        )

    assert session.calls == []


async def test_execution_timeout_has_a_finite_upper_bound() -> None:
    session = FakeSession()
    manager = ColabJobManager(session)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="no more than 86400"):
        await manager.start_python("train()", execution_timeout_seconds=86_401)

    assert session.calls == []


async def test_wait_job_times_out_without_cancelling_execution() -> None:
    gate = asyncio.Event()
    manager = ColabJobManager(
        FakeSession(run_gate=gate)  # type: ignore[arg-type]
    )
    started = await manager.start_python("while True: pass")

    await asyncio.sleep(0)
    calls_before_wait = list(manager.session.calls)  # type: ignore[attr-defined]
    waited = await manager.wait(started["job_id"], timeout_seconds=1.0)

    assert waited["state"] == "running"
    assert waited["timed_out"] is True
    assert waited["wait_timed_out"] is True
    assert waited["waited_seconds"] >= 1.0
    assert manager.session.calls == calls_before_wait  # type: ignore[attr-defined]
    gate.set()
    finished = await manager.wait(started["job_id"], timeout_seconds=1.0)
    assert finished["state"] == "finished"
    assert finished["terminal_manifest_found"] is True


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
    assert sum(call[0] == "get_cells" for call in session.calls) - get_cells_before == 0


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


async def test_empty_outputs_leave_execution_ambiguous() -> None:
    manager = ColabJobManager(
        FakeSession(run_outputs=[])  # type: ignore[arg-type]
    )

    started = await manager.start_python("x = 1")
    finished = await manager.wait(started["job_id"], timeout_seconds=1.0)

    assert finished["state"] == "running"
    assert finished["tracking_state"] == "detached"
    assert finished["completion_source"] is None
    assert finished["terminal_manifest_found"] is False
    assert finished["captured_runtime_output_bytes"] is None
    assert finished["remote_output_count"] == 0
    assert finished["outputs"] == []


@pytest.mark.parametrize(
    "run_data",
    [
        {},
        {"outputs": None},
        {"outputs": "not-a-list"},
        {"outputs": []},
        {
            "outputs": [
                {"output_type": "stream", "name": "stdout", "text": ["unproven"]}
            ]
        },
    ],
)
async def test_unproven_execution_responses_never_finish(
    run_data: dict[str, Any],
) -> None:
    manager = ColabJobManager(
        FakeSession(run_data=run_data)  # type: ignore[arg-type]
    )

    started = await manager.start_python("x = 1")
    result_value = await manager.wait(started["job_id"], timeout_seconds=1.0)

    assert result_value["state"] == "running"
    assert result_value["tracking_state"] == "detached"
    assert result_value["terminal_manifest_found"] is False
    assert result_value["completion_source"] is None


async def test_controlled_exception_finishes_through_matching_manifest() -> None:
    manager = ColabJobManager(
        FakeSession(
            manifest_output_excerpt="before failure\ntraceback\n",
            manifest_state="error",
            manifest_error="RuntimeError: controlled",
        )  # type: ignore[arg-type]
    )

    started = await manager.start_python("raise RuntimeError('controlled')")
    finished = await manager.wait(started["job_id"], timeout_seconds=1.0)

    assert finished["state"] == "error"
    assert finished["error"] == "RuntimeError: controlled"
    assert "before failure" in finished["outputs"][0]["text"][0]
    assert finished["terminal_manifest_found"] is True
    assert finished["completion_source"] == "terminal_manifest"


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


async def test_mark_stale_invalidates_notebook_scoped_cell_state() -> None:
    manager = ColabJobManager(FakeSession())  # type: ignore[arg-type]
    manager._job_cells["job-cell"] = 1
    manager._probe_cell_id = "probe-cell"
    manager._pending_job_cell_sha256 = "a" * 64
    manager._pending_probe_cell_sha256 = "b" * 64

    await manager.mark_stale("reset")

    assert manager._job_cells == {}
    assert manager._probe_cell_id is None
    assert manager._pending_job_cell_sha256 is None
    assert manager._pending_probe_cell_sha256 is None


async def test_execution_timeout_marks_job_timed_out() -> None:
    manager = ColabJobManager(
        FakeSession(run_error=asyncio.TimeoutError())  # type: ignore[arg-type]
    )

    started = await manager.start_python("train()", execution_timeout_seconds=12)
    finished = await manager.wait(started["job_id"], timeout_seconds=1.0)

    assert finished["state"] == "timed_out"
    assert "12 seconds" in finished["error"]


async def test_remote_execution_exception_detaches_for_reconciliation() -> None:
    manager = ColabJobManager(
        FakeSession(run_error=RuntimeError("disconnected"))  # type: ignore[arg-type]
    )

    started = await manager.start_python("train()")
    finished = await manager.wait(started["job_id"], timeout_seconds=1.0)

    assert finished["state"] == "running"
    assert finished["tracking_state"] == "detached"
    assert finished["execution_alive"] is None
    assert finished["error"] is None
    assert "RuntimeError" in finished["tracking_error"]


async def test_status_never_downloads_accumulated_notebook_outputs() -> None:
    gate = asyncio.Event()
    session = FakeSession(run_gate=gate)
    session.cells.append(
        {
            "id": "unrelated",
            "cell_type": "code",
            "source": ["print('old')"],
            "outputs": [
                {
                    "output_type": "stream",
                    "name": "stdout",
                    "text": ["x" * (2 * 1024 * 1024)],
                }
            ],
        }
    )
    manager = ColabJobManager(session)  # type: ignore[arg-type]
    started = await manager.start_python("train()")

    status = await manager.status(started["job_id"])

    assert status["state"] == "running"
    get_cells_calls = [args for name, args in session.calls if name == "get_cells"]
    assert get_cells_calls
    assert all(args.get("includeOutputs") is False for args in get_cells_calls)
    status_call = get_cells_calls[-1]
    assert status_call["cellIndexStart"] == started["cell_index"]
    assert status_call["cellIndexEnd"] == started["cell_index"]
    assert len(json.dumps(status).encode("utf-8")) < 16 * 1024
    gate.set()
    await manager.wait(started["job_id"], timeout_seconds=1.0)


async def test_large_target_output_is_bounded_and_stored_as_artifact(
    tmp_path: Path,
) -> None:
    large_text = "training-log\n" * 180_000
    large_bytes = len(large_text.encode("utf-8"))
    session = FakeSession(
        manifest_output_excerpt=large_text[: 24 * 1024],
        manifest_output_bytes=large_bytes,
        manifest_artifact_size=large_bytes,
    )
    store = ArtifactStore(tmp_path / "artifacts")
    manager = ColabJobManager(
        session,  # type: ignore[arg-type]
        artifact_store=store,
        output_excerpt_bytes=32 * 1024,
    )

    started = await manager.start_python("train()")
    finished = await manager.wait(started["job_id"], timeout_seconds=1.0)

    assert finished["state"] == "finished"
    assert finished["output_bytes"] == large_bytes
    assert finished["output_truncated"] is True
    assert finished["output_artifact"]["storage"] == "colab_runtime"
    assert len(json.dumps(finished).encode("utf-8")) < 64 * 1024


async def test_large_binary_display_is_not_echoed_or_duplicated(
    tmp_path: Path,
) -> None:
    binary_payload_bytes = 2 * 1024 * 1024
    manager = ColabJobManager(
        FakeSession(
            manifest_output_excerpt=(
                "plot\n[binary/rich display omitted: image/png]\n"
            ),
            manifest_output_bytes=binary_payload_bytes,
            manifest_artifact_size=binary_payload_bytes,
        ),  # type: ignore[arg-type]
        artifact_store=ArtifactStore(tmp_path / "artifacts"),
        output_excerpt_bytes=16 * 1024,
    )

    started = await manager.start_python("display(plot)")
    finished = await manager.wait(started["job_id"], timeout_seconds=1.0)
    serialized = json.dumps(finished)

    assert "[binary/rich display omitted: image/png]" in serialized
    assert serialized.count("plot") == 1
    assert finished["output_artifact"] is not None


async def test_public_job_data_never_contains_submitted_code() -> None:
    gate = asyncio.Event()
    session = FakeSession(run_gate=gate)
    manager = ColabJobManager(session)  # type: ignore[arg-type]
    secret_code = "SECRET_CORPUS_VALUE = '" + ("private" * 100_000) + "'"

    started = await manager.start_python(secret_code)
    status = await manager.status(started["job_id"])
    listed = manager.list_jobs()[0]

    assert "SECRET_CORPUS_VALUE" not in json.dumps(started)
    assert "SECRET_CORPUS_VALUE" not in json.dumps(status)
    assert "SECRET_CORPUS_VALUE" not in json.dumps(listed)
    assert started["code_bytes"] == len(secret_code.encode("utf-8"))
    assert len(started["code_sha256"]) == 64
    added_code = next(
        arguments["code"]
        for name, arguments in session.calls
        if name == "add_code_cell"
    )
    assert "/content/.colab_codex/jobs" in added_code
    assert "_cc_builtins.exec(" in added_code
    assert added_code.count("_cc_user_globals") >= 3
    assert secret_code not in added_code
    gate.set()
    await manager.wait(started["job_id"], timeout_seconds=1.0)


async def test_tracked_jobs_reject_non_python_languages() -> None:
    manager = ColabJobManager(FakeSession())  # type: ignore[arg-type]
    try:
        await manager.start_python("console.log(1)", language="javascript")
    except ValueError as exc:
        assert "CPython only" in str(exc)
    else:
        raise AssertionError("non-Python tracked jobs should be rejected")


async def test_private_job_journal_excludes_code_outputs_and_errors(
    tmp_path: Path,
) -> None:
    gate = asyncio.Event()
    session = FakeSession(run_gate=gate)
    journal = tmp_path / "private" / "jobs.json"
    manager = ColabJobManager(
        session, journal_path=journal  # type: ignore[arg-type]
    )
    secret = "TOP_SECRET_CORPUS_VALUE"

    started = await manager.start_python(f"raise RuntimeError('{secret}')")
    journal_text = journal.read_text(encoding="utf-8")

    assert journal.stat().st_mode & 0o777 == 0o600
    assert journal.parent.stat().st_mode & 0o777 == 0o700
    assert secret not in journal_text
    assert '"code"' not in journal_text
    assert '"outputs"' not in journal_text
    assert '"error"' not in journal_text
    assert started["job_id"] in journal_text
    gate.set()
    await manager.wait(started["job_id"], timeout_seconds=1.0)


async def test_job_journal_restores_running_job_as_detached(
    tmp_path: Path,
) -> None:
    gate = asyncio.Event()
    journal = tmp_path / "jobs.json"
    first = ColabJobManager(
        FakeSession(run_gate=gate),  # type: ignore[arg-type]
        journal_path=journal,
    )
    started = await first.start_python("train()")
    await first.on_browser_disconnect("forced disconnect")

    recovered = ColabJobManager(
        FakeSession(),  # type: ignore[arg-type]
        journal_path=journal,
    )
    restored = recovered.jobs[started["job_id"]]
    status = await recovered.wait(started["job_id"], timeout_seconds=900.0)

    assert restored.state == "running"
    assert status["tracking_state"] == "detached"
    assert status["execution_alive"] is None
    assert status["task_alive"] is False
    assert status["waited_seconds"] < 0.1
    assert status["outputs"] == []


async def test_detached_job_reconciles_without_rerunning_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = asyncio.Event()
    session = FakeSession(run_gate=gate)
    manager = ColabJobManager(session)  # type: ignore[arg-type]
    started = await manager.start_python("train_once()")
    await asyncio.sleep(0)

    await manager.on_browser_disconnect("forced websocket loss")
    run_calls_before = sum(name == "run_code_cell" for name, _ in session.calls)

    async def recovered_markers(jobs: list[Any]) -> dict[str, Any]:
        assert [job.job_id for job in jobs] == [started["job_id"]]
        return {
            started["job_id"]: {
                "job_id": started["job_id"],
                "state": "finished",
                "output_excerpt": "epoch complete\n",
                "output_bytes": 15,
                "artifact_size_bytes": 15,
                "artifact_sha256": "a" * 64,
                "output_truncated": False,
                "artifact_truncated": False,
            }
        }

    monkeypatch.setattr(manager, "_read_runtime_markers", recovered_markers)
    reconciled = await manager.reconcile_detached()

    assert reconciled[0]["state"] == "finished"
    assert reconciled[0]["tracking_state"] == "complete"
    assert reconciled[0]["outputs"][0]["text"] == ["epoch complete\n"]
    assert sum(name == "run_code_cell" for name, _ in session.calls) == run_calls_before


async def test_stale_running_marker_becomes_terminal_interrupted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = asyncio.Event()
    manager = ColabJobManager(
        FakeSession(run_gate=gate)  # type: ignore[arg-type]
    )
    started = await manager.start_python(
        "train_once()", execution_timeout_seconds=1
    )
    await asyncio.sleep(0)
    await manager.on_browser_disconnect("forced disconnect")
    stale_started = (
        time.time() - jobs_module.DEFAULT_RUNTIME_STALE_GRACE_SECONDS - 2
    )
    manager.jobs[started["job_id"]].started_at = stale_started

    probe_called = False

    async def stale_marker(jobs: list[Any]) -> dict[str, Any]:
        nonlocal probe_called
        probe_called = True
        return {
            jobs[0].job_id: {
                "job_id": jobs[0].job_id,
                "state": "running",
                "started_at": stale_started,
            }
        }

    monkeypatch.setattr(manager, "_read_runtime_markers", stale_marker)
    reconciled = await manager.reconcile_detached()

    assert reconciled[0]["state"] == "interrupted"
    assert reconciled[0]["tracking_state"] == "complete"
    assert "recovery deadline" in reconciled[0]["output_unavailable_reason"]
    assert probe_called is False


async def test_marker_probe_timeout_is_capped_to_recovery_deadline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manager = ColabJobManager(FakeSession())  # type: ignore[arg-type]
    now = time.time()
    job = ColabJob(
        job_id="d" * 32,
        cell_id="test-cell",
        cell_index=0,
        code_bytes=1,
        code_sha256=hashlib.sha256(b"x").hexdigest(),
        state="timed_out",
        tracking_state="detached",
        started_at=(
            now - jobs_module.DEFAULT_RUNTIME_STALE_GRACE_SECONDS + 4
        ),
        updated_at=now,
        execution_timeout_seconds=1,
        runtime_marker_path=str(tmp_path / "deadline.json"),
        runtime_output_path=str(tmp_path / "deadline.output"),
        runtime_artifact_id="e" * 32,
    )
    observed_timeout = 0.0

    async def capture_probe(code: str, *, timeout: float) -> list[Any]:
        nonlocal observed_timeout
        del code
        observed_timeout = timeout
        return [
            {
                "text": [
                    jobs_module._RECONCILE_SENTINEL + "{}"
                ]
            }
        ]

    monkeypatch.setattr(manager, "_append_and_run_probe", capture_probe)
    await manager._read_runtime_markers([job])

    assert 0.1 <= observed_timeout <= 5.1


async def test_shutdown_detaches_running_job_and_preserves_recovery_journal(
    tmp_path: Path,
) -> None:
    gate = asyncio.Event()
    journal = tmp_path / "jobs.json"
    manager = ColabJobManager(
        FakeSession(run_gate=gate),  # type: ignore[arg-type]
        journal_path=journal,
    )
    started = await manager.start_python("train_once()")
    await asyncio.sleep(0)

    await manager.close()

    status = await manager.status(started["job_id"])
    assert status["state"] == "running"
    assert status["tracking_state"] == "detached"
    assert status["execution_alive"] is None
    assert status["task_alive"] is False

    recovered = ColabJobManager(
        FakeSession(),  # type: ignore[arg-type]
        journal_path=journal,
    )
    restored = await recovered.status(started["job_id"])
    assert restored["state"] == "running"
    assert restored["tracking_state"] == "detached"
    assert restored["execution_alive"] is None


async def test_reconciliation_does_not_overwrite_concurrent_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = asyncio.Event()
    probe_started = asyncio.Event()
    probe_release = asyncio.Event()
    manager = ColabJobManager(
        FakeSession(run_gate=gate)  # type: ignore[arg-type]
    )
    started = await manager.start_python("train_once()")
    await asyncio.sleep(0)
    await manager.on_browser_disconnect("forced disconnect")

    async def delayed_markers(jobs: list[Any]) -> dict[str, Any]:
        probe_started.set()
        await probe_release.wait()
        return {
            jobs[0].job_id: {
                "job_id": jobs[0].job_id,
                "state": "finished",
                "output_excerpt": "stale result",
                "output_bytes": 12,
                "artifact_size_bytes": 12,
                "artifact_sha256": "a" * 64,
                "output_truncated": False,
                "artifact_truncated": False,
            }
        }

    monkeypatch.setattr(manager, "_read_runtime_markers", delayed_markers)
    reconciliation = asyncio.create_task(manager.reconcile_detached())
    await probe_started.wait()
    await manager.mark_stale("explicit reset")
    probe_release.set()
    await reconciliation

    status = await manager.status(started["job_id"])
    assert status["state"] == "stale"
    assert status["tracking_state"] == "complete"
    assert status["error"] == "explicit reset"
    assert status["outputs"] == []


async def test_cancelled_reconciliation_detaches_and_can_be_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = asyncio.Event()
    probe_started = asyncio.Event()
    manager = ColabJobManager(
        FakeSession(run_gate=gate)  # type: ignore[arg-type]
    )
    started = await manager.start_python("train_once()")
    await asyncio.sleep(0)
    await manager.on_browser_disconnect("forced disconnect")

    async def blocked_markers(jobs: list[Any]) -> dict[str, Any]:
        probe_started.set()
        await asyncio.Event().wait()
        return {}

    monkeypatch.setattr(manager, "_read_runtime_markers", blocked_markers)
    reconciliation = asyncio.create_task(manager.reconcile_detached())
    await probe_started.wait()
    reconciliation.cancel()
    with pytest.raises(asyncio.CancelledError):
        await reconciliation

    detached = await manager.status(started["job_id"])
    assert detached["state"] == "running"
    assert detached["tracking_state"] == "detached"

    async def finished_markers(jobs: list[Any]) -> dict[str, Any]:
        return {
            jobs[0].job_id: {
                "job_id": jobs[0].job_id,
                "state": "finished",
                "output_excerpt": "complete",
                "output_bytes": 8,
                "artifact_size_bytes": 8,
                "artifact_sha256": "a" * 64,
                "output_truncated": False,
                "artifact_truncated": False,
            }
        }

    monkeypatch.setattr(manager, "_read_runtime_markers", finished_markers)
    retried = await manager.reconcile_detached()
    assert retried[0]["state"] == "finished"


async def test_status_does_not_overwrite_concurrent_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = asyncio.Event()
    lookup_started = asyncio.Event()
    lookup_release = asyncio.Event()
    manager = ColabJobManager(
        FakeSession(run_gate=gate)  # type: ignore[arg-type]
    )
    started = await manager.start_python("train_once()")

    async def delayed_missing(job: ColabJob) -> bool:
        lookup_started.set()
        await lookup_release.wait()
        return False

    monkeypatch.setattr(manager, "_job_cell_exists", delayed_missing)
    status_task = asyncio.create_task(manager.status(started["job_id"]))
    await lookup_started.wait()
    await manager.mark_stale("explicit reset")
    lookup_release.set()
    status = await status_task

    assert status["state"] == "stale"
    assert status["tracking_state"] == "complete"
    assert status["error"] == "explicit reset"


async def test_disconnect_batches_job_journal_persistence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = asyncio.Event()
    manager = ColabJobManager(
        FakeSession(run_gate=gate)  # type: ignore[arg-type]
    )
    await asyncio.gather(
        manager.start_python("first()"),
        manager.start_python("second()"),
        manager.start_python("third()"),
    )
    persists = 0

    def record_persist() -> None:
        nonlocal persists
        persists += 1

    monkeypatch.setattr(manager, "_persist_journal", record_persist)
    await manager.on_browser_disconnect("forced disconnect")

    assert persists == 1


async def test_job_completion_persists_journal_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = asyncio.Event()
    manager = ColabJobManager(
        FakeSession(run_gate=gate)  # type: ignore[arg-type]
    )
    started = await manager.start_python("train_once()")
    persists = 0

    def record_persist() -> None:
        nonlocal persists
        persists += 1

    monkeypatch.setattr(manager, "_persist_journal", record_persist)
    gate.set()
    await manager.wait(started["job_id"], timeout_seconds=1.0)

    assert persists == 1


async def test_timed_out_execution_can_reconcile_to_finished(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    journal = tmp_path / "jobs.json"
    manager = ColabJobManager(
        FakeSession(run_error=asyncio.TimeoutError()),  # type: ignore[arg-type]
        journal_path=journal,
    )
    started = await manager.start_python(
        "train_once()", execution_timeout_seconds=12
    )
    timed_out = await manager.wait(started["job_id"], timeout_seconds=1.0)
    assert timed_out["state"] == "timed_out"
    assert timed_out["tracking_state"] == "detached"

    recovered = ColabJobManager(
        FakeSession(),  # type: ignore[arg-type]
        journal_path=journal,
    )
    restored = await recovered.status(started["job_id"])
    assert restored["state"] == "timed_out"
    assert restored["tracking_state"] == "detached"
    assert restored["execution_alive"] is None

    async def completed_marker(jobs: list[Any]) -> dict[str, Any]:
        job_id = jobs[0].job_id
        return {
            job_id: {
                "job_id": job_id,
                "state": "finished",
                "output_excerpt": "completed after timeout\n",
                "output_bytes": 24,
                "artifact_size_bytes": 24,
                "artifact_sha256": "b" * 64,
                "output_truncated": False,
                "artifact_truncated": False,
            }
        }

    monkeypatch.setattr(recovered, "_read_runtime_markers", completed_marker)
    reconciled = await recovered.reconcile_detached()

    assert reconciled[0]["state"] == "finished"
    assert reconciled[0]["tracking_state"] == "complete"
    assert reconciled[0]["error"] is None
    assert reconciled[0]["outputs"][0]["text"] == [
        "completed after timeout\n"
    ]


async def test_unknown_job_and_artifact_ids_are_not_echoed() -> None:
    manager = ColabJobManager(FakeSession())  # type: ignore[arg-type]
    supplied = "../../PRIVATE_JOB_OR_ARTIFACT"

    with pytest.raises(ValueError) as status_error:
        await manager.status(supplied)
    with pytest.raises(ValueError) as wait_error:
        await manager.wait(supplied, timeout_seconds=1.0)
    with pytest.raises(ArtifactNotFoundError) as artifact_error:
        await manager.read_artifact(supplied)

    assert str(status_error.value) == "Unknown Colab job id"
    assert str(wait_error.value) == "Unknown Colab job id"
    assert str(artifact_error.value) == "Unknown or expired artifact id"
    assert supplied not in (
        str(status_error.value) + str(wait_error.value) + str(artifact_error.value)
    )


async def test_runtime_artifact_reference_expires_and_read_fails_closed(
    tmp_path: Path,
) -> None:
    session = FakeSession()
    manager = ColabJobManager(
        session,  # type: ignore[arg-type]
        artifact_store=ArtifactStore(tmp_path / "local-artifacts"),
        runtime_artifact_ttl_seconds=60,
    )
    now = time.time()
    artifact_id = "6" * 32
    job = ColabJob(
        job_id="5" * 32,
        cell_id="test-cell",
        cell_index=0,
        code_bytes=1,
        code_sha256=hashlib.sha256(b"x").hexdigest(),
        state="running",
        tracking_state="active",
        started_at=now,
        updated_at=now,
        execution_timeout_seconds=30,
        runtime_marker_path=str(tmp_path / "runtime.json"),
        runtime_output_path=str(tmp_path / "runtime.output"),
        runtime_artifact_id=artifact_id,
    )
    manager.jobs[job.job_id] = job
    manager._completion_events[job.job_id] = asyncio.Event()
    manager._finish_from_manifest(
        job,
        {
            "state": "finished",
            "output_excerpt": "x" * 1024,
            "output_bytes": 2048,
            "artifact_size_bytes": 2048,
            "artifact_sha256": "a" * 64,
            "output_truncated": True,
            "artifact_truncated": False,
        },
    )

    assert job.output_artifact is not None
    assert job.output_artifact["expires_at"] > now
    job.output_artifact["expires_at"] = time.time() - 1

    with pytest.raises(ArtifactNotFoundError, match="Unknown or expired"):
        await manager.read_artifact(artifact_id)
    assert artifact_id not in manager._runtime_artifacts
    assert session.calls == []


async def test_runtime_artifact_probe_uses_short_bounded_timeout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manager = ColabJobManager(
        FakeSession(),  # type: ignore[arg-type]
        artifact_store=ArtifactStore(tmp_path / "local-artifacts"),
    )
    now = time.time()
    artifact_id = "8" * 32
    job = ColabJob(
        job_id="7" * 32,
        cell_id="test-cell",
        cell_index=0,
        code_bytes=1,
        code_sha256=hashlib.sha256(b"x").hexdigest(),
        state="finished",
        tracking_state="complete",
        started_at=now,
        updated_at=now,
        execution_timeout_seconds=30,
        output_artifact={
            "artifact_id": artifact_id,
            "storage": "colab_runtime",
            "media_type": "text/plain; charset=utf-8",
            "size_bytes": 1024,
            "sha256": "a" * 64,
            "created_at": now,
            "expires_at": now + 60,
            "truncated": False,
        },
        runtime_marker_path=str(tmp_path / "runtime.json"),
        runtime_output_path=str(tmp_path / "runtime.output"),
        runtime_artifact_id=artifact_id,
    )
    manager.jobs[job.job_id] = job
    manager._runtime_artifacts[artifact_id] = job.runtime_output_path
    observed_timeout = 0.0

    async def busy_probe(code: str, *, timeout: float) -> list[Any]:
        nonlocal observed_timeout
        del code
        observed_timeout = timeout
        raise asyncio.TimeoutError

    monkeypatch.setattr(manager, "_append_and_run_probe", busy_probe)

    with pytest.raises(asyncio.TimeoutError):
        await manager.read_artifact(artifact_id)
    assert 0.1 <= observed_timeout <= 30


def test_runtime_wrapper_bounds_subprocess_and_native_output(tmp_path: Path) -> None:
    manager = ColabJobManager(
        FakeSession(),  # type: ignore[arg-type]
        output_excerpt_bytes=1024,
        runtime_artifact_bytes=64 * 1024,
    )
    source = """
import os
import subprocess
import sys

subprocess.run(
    [
        sys.executable,
        "-c",
        "import sys; "
        "sys.stdout.buffer.write(b'x' * 700000); sys.stdout.buffer.flush(); "
        "sys.stderr.buffer.write(b'y' * 700000); sys.stderr.buffer.flush()",
    ],
    check=True,
)
remaining = 200000
while remaining:
    written = os.write(1, b'z' * min(remaining, 65536))
    remaining -= written
print("python stream output")
"""
    source_bytes = source.encode("utf-8")
    job = ColabJob(
        job_id="1" * 32,
        cell_id="test-cell",
        cell_index=0,
        code_bytes=len(source_bytes),
        code_sha256=hashlib.sha256(source_bytes).hexdigest(),
        state="running",
        tracking_state="active",
        started_at=time.time(),
        updated_at=time.time(),
        execution_timeout_seconds=30,
        runtime_marker_path=str(tmp_path / "job.json"),
        runtime_output_path=str(tmp_path / "job.output"),
        runtime_artifact_id="2" * 32,
    )

    completed = subprocess.run(
        [sys.executable, "-c", manager._runtime_wrapper(job, source)],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )

    sentinel = "__COLAB_CODEX_JOB__"
    lines = [line for line in completed.stdout.splitlines() if line.startswith(sentinel)]
    assert len(lines) == 1
    manifest = json.loads(lines[0][len(sentinel) :])
    artifact = Path(job.runtime_output_path).read_bytes()
    marker = json.loads(Path(job.runtime_marker_path).read_text(encoding="utf-8"))

    assert manifest["state"] == "finished"
    assert manifest["output_bytes"] > 1024 * 1024
    assert manifest["output_truncated"] is True
    assert manifest["artifact_truncated"] is True
    assert manifest["artifact_size_bytes"] == 64 * 1024
    assert len(manifest["output_excerpt"].encode("utf-8")) <= 1024
    assert len(completed.stdout.encode("utf-8")) < 16 * 1024
    assert completed.stderr == ""
    assert len(artifact) == 64 * 1024
    assert hashlib.sha256(artifact).hexdigest() == manifest["artifact_sha256"]
    assert marker["state"] == "finished"
    assert "output_excerpt" not in marker


async def test_reusable_probe_cell_update_and_run_are_serialized() -> None:
    run_gate = asyncio.Event()
    session = FakeSession(run_gate=run_gate)
    manager = ColabJobManager(session)  # type: ignore[arg-type]

    first = asyncio.create_task(manager._append_and_run_probe("print('first')"))
    second = asyncio.create_task(manager._append_and_run_probe("print('second')"))
    while not any(name == "run_code_cell" for name, _ in session.calls):
        await asyncio.sleep(0)
    await asyncio.sleep(0.02)

    assert sum(name == "add_code_cell" for name, _ in session.calls) == 1
    assert sum(name == "update_cell" for name, _ in session.calls) == 0
    assert sum(name == "run_code_cell" for name, _ in session.calls) == 1

    run_gate.set()
    await asyncio.gather(first, second)
    mutation_sequence = [
        name
        for name, _ in session.calls
        if name in {"add_code_cell", "update_cell", "run_code_cell"}
    ]
    assert mutation_sequence == [
        "add_code_cell",
        "run_code_cell",
        "update_cell",
        "run_code_cell",
    ]


async def test_recovery_probe_append_index_uses_bounded_metadata_pages() -> None:
    session = FakeSession()
    session.cells.extend(
        {
            "id": f"existing-{index}",
            "cell_type": "code",
            "source": ["pass"],
            "outputs": [],
        }
        for index in range(17)
    )
    manager = ColabJobManager(session)  # type: ignore[arg-type]

    await manager._append_and_run_probe("print('probe')")

    page_calls = [args for name, args in session.calls if name == "get_cells"]
    assert page_calls == [
        {"includeOutputs": False, "cellIndexStart": 0, "cellIndexEnd": 7},
        {"includeOutputs": False, "cellIndexStart": 8, "cellIndexEnd": 15},
        {"includeOutputs": False, "cellIndexStart": 16, "cellIndexEnd": 23},
    ]
    add_call = next(args for name, args in session.calls if name == "add_code_cell")
    assert add_call["cellIndex"] == 17


async def test_reconciliation_probe_requires_matching_sentinel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = ColabJobManager(FakeSession())  # type: ignore[arg-type]
    now = time.time()
    job = ColabJob(
        job_id="a" * 32,
        cell_id="cell",
        cell_index=0,
        code_bytes=1,
        code_sha256="b" * 64,
        state="running",
        tracking_state="detached",
        started_at=now,
        updated_at=now,
        execution_timeout_seconds=60,
        runtime_marker_path="/tmp/marker",
        runtime_output_path="/tmp/output",
        runtime_artifact_id="c" * 32,
    )

    async def empty_probe(code: str, *, timeout: float) -> list[Any]:
        del code, timeout
        return []

    monkeypatch.setattr(manager, "_append_and_run_probe", empty_probe)

    with pytest.raises(RuntimeError, match="reconciliation sentinel"):
        await manager._read_runtime_markers([job])


async def test_kernel_readiness_requires_matching_nonce(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = ColabJobManager(FakeSession())  # type: ignore[arg-type]

    class FixedUuid:
        hex = "kernel-nonce"

    monkeypatch.setattr(jobs_module.uuid, "uuid4", lambda: FixedUuid())

    async def proven_probe(code: str, *, timeout: float) -> list[Any]:
        del code, timeout
        return [
            {
                "output_type": "stream",
                "name": "stdout",
                "text": [
                    jobs_module._KERNEL_PROBE_SENTINEL
                    + '{"nonce":"kernel-nonce"}\n'
                ],
            }
        ]

    monkeypatch.setattr(manager, "_append_and_run_probe", proven_probe)

    readiness = await manager.probe_kernel()

    assert readiness["kernel_execution_ready"] is True
    assert readiness["kernel_probe_at"] is not None
    assert readiness["kernel_probe_latency_ms"] is not None
    assert readiness["kernel_probe_error"] is None


async def test_kernel_readiness_records_failed_execution_probe() -> None:
    manager = ColabJobManager(
        FakeSession(run_outputs=[])  # type: ignore[arg-type]
    )

    readiness = await manager.probe_kernel(timeout=1)

    assert readiness["kernel_execution_ready"] is False
    assert "kernel sentinel" in readiness["kernel_probe_error"]


async def test_probe_timeout_includes_waiting_for_probe_lock() -> None:
    manager = ColabJobManager(FakeSession())  # type: ignore[arg-type]
    await manager._probe_lock.acquire()
    try:
        with pytest.raises(asyncio.TimeoutError):
            await manager._append_and_run_probe("print('probe')", timeout=0.05)
    finally:
        manager._probe_lock.release()


async def test_recovery_probe_cell_id_survives_manager_recovery(
    tmp_path: Path,
) -> None:
    session = FakeSession()
    journal = tmp_path / "jobs.json"
    first = ColabJobManager(
        session,  # type: ignore[arg-type]
        journal_path=journal,
    )
    await first._append_and_run_probe("print('first probe')")
    probe_cell_id = first._probe_cell_id

    recovered = ColabJobManager(
        session,  # type: ignore[arg-type]
        journal_path=journal,
    )
    await recovered._append_and_run_probe("print('second probe')")

    assert recovered._probe_cell_id == probe_cell_id
    assert len(session.cells) == 1
    assert sum(name == "add_code_cell" for name, _ in session.calls) == 1
    assert sum(name == "update_cell" for name, _ in session.calls) == 1


async def test_lost_probe_cell_add_response_recovers_without_growth() -> None:
    session = AmbiguousAddSession()
    manager = ColabJobManager(session)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="add response lost"):
        await manager._append_and_run_probe("print('first probe')", timeout=2)

    await manager._append_and_run_probe("print('second probe')", timeout=2)

    assert len(session.cells) == 1
    assert manager._probe_cell_id == session.cells[0]["id"]
    assert sum(name == "add_code_cell" for name, _ in session.calls) == 1
    assert sum(name == "update_cell" for name, _ in session.calls) == 1


def test_runtime_wrapper_does_not_wait_for_background_child(tmp_path: Path) -> None:
    manager = ColabJobManager(
        FakeSession(),  # type: ignore[arg-type]
        output_excerpt_bytes=1024,
        runtime_artifact_bytes=64 * 1024,
    )
    source = """
import subprocess
import sys

subprocess.Popen(
    [
        sys.executable,
        "-c",
        "import sys, time; "
        "sys.stdout.write('background-started\\n'); sys.stdout.flush(); "
        "time.sleep(2)",
    ]
)
print("foreground-finished")
"""
    source_bytes = source.encode("utf-8")
    now = time.time()
    job = ColabJob(
        job_id="3" * 32,
        cell_id="test-cell",
        cell_index=0,
        code_bytes=len(source_bytes),
        code_sha256=hashlib.sha256(source_bytes).hexdigest(),
        state="running",
        tracking_state="active",
        started_at=now,
        updated_at=now,
        execution_timeout_seconds=30,
        runtime_marker_path=str(tmp_path / "background.json"),
        runtime_output_path=str(tmp_path / "background.output"),
        runtime_artifact_id="4" * 32,
    )

    started_at = time.monotonic()
    completed = subprocess.run(
        [sys.executable, "-c", manager._runtime_wrapper(job, source)],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    elapsed = time.monotonic() - started_at
    sentinel = "__COLAB_CODEX_JOB__"
    manifest_line = next(
        line for line in completed.stdout.splitlines() if line.startswith(sentinel)
    )
    manifest = json.loads(manifest_line[len(sentinel) :])

    assert elapsed < 1.5
    assert manifest["state"] == "finished"
    assert Path(job.runtime_marker_path).is_file()


def test_runtime_wrapper_suppresses_lingering_python_thread_output(
    tmp_path: Path,
) -> None:
    manager = ColabJobManager(FakeSession())  # type: ignore[arg-type]
    display_bypass = tmp_path / "background-display-bypass"
    source = """
import threading
import time
import os

def background_log():
    time.sleep(0.05)
    print("BACKGROUND-BYPASS")
    os.write(1, b"BACKGROUND-NATIVE-BYPASS\\n")
    get_ipython().display_pub.publish({"text/plain": "BACKGROUND-DISPLAY"})
    def child_log():
        time.sleep(0.1)
        print("CHILD-BYPASS")
        os.write(1, b"CHILD-NATIVE-BYPASS\\n")
        get_ipython().display_pub.publish({"text/plain": "CHILD-DISPLAY"})
    threading.Thread(target=child_log).start()

threading.Thread(target=background_log).start()
print("foreground-finished")
"""
    prefix = f"""
import builtins
from pathlib import Path
class _BackgroundPublisher:
    def publish(self, data, metadata=None, **kwargs):
        Path({str(display_bypass)!r}).touch()
class _BackgroundShell:
    display_pub = _BackgroundPublisher()
builtins.get_ipython = lambda: _BackgroundShell()
"""
    raw = source.encode("utf-8")
    now = time.time()
    job = ColabJob(
        job_id="0" * 32,
        cell_id="test-cell",
        cell_index=0,
        code_bytes=len(raw),
        code_sha256=hashlib.sha256(raw).hexdigest(),
        state="running",
        tracking_state="active",
        started_at=now,
        updated_at=now,
        execution_timeout_seconds=30,
        runtime_marker_path=str(tmp_path / "thread.json"),
        runtime_output_path=str(tmp_path / "thread.output"),
        runtime_artifact_id="1" * 32,
    )

    completed = subprocess.run(
        [sys.executable, "-c", prefix + manager._runtime_wrapper(job, source)],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )

    output_lines = completed.stdout.splitlines()
    assert len(output_lines) == 1
    assert output_lines[0].startswith("__COLAB_CODEX_JOB__")
    manifest = json.loads(output_lines[0][len("__COLAB_CODEX_JOB__") :])
    assert "BACKGROUND-BYPASS" in manifest["output_excerpt"]
    assert "BACKGROUND-NATIVE-BYPASS" in manifest["output_excerpt"]
    assert "BACKGROUND-DISPLAY" in manifest["output_excerpt"]
    assert "CHILD-BYPASS" not in completed.stdout
    assert "CHILD-NATIVE-BYPASS" not in completed.stdout
    assert "CHILD-DISPLAY" not in completed.stdout
    assert not display_bypass.exists()
    marker = json.loads(Path(job.runtime_marker_path).read_text(encoding="utf-8"))
    assert marker["state"] == "finished"


def test_runtime_wrapper_thread_tracking_is_race_safe(tmp_path: Path) -> None:
    manager = ColabJobManager(FakeSession())  # type: ignore[arg-type]
    source = """
import threading

def noop():
    return None

def fanout():
    for _ in range(300):
        threading.Thread(target=noop).start()

threading.Thread(target=fanout).start()
"""
    raw = source.encode("utf-8")
    now = time.time()
    job = ColabJob(
        job_id="6" * 32,
        cell_id="test-cell",
        cell_index=0,
        code_bytes=len(raw),
        code_sha256=hashlib.sha256(raw).hexdigest(),
        state="running",
        tracking_state="active",
        started_at=now,
        updated_at=now,
        execution_timeout_seconds=30,
        runtime_marker_path=str(tmp_path / "thread-race.json"),
        runtime_output_path=str(tmp_path / "thread-race.output"),
        runtime_artifact_id="7" * 32,
    )

    completed = subprocess.run(
        [sys.executable, "-c", manager._runtime_wrapper(job, source)],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.stdout.count("__COLAB_CODEX_JOB__") == 1
    marker = json.loads(Path(job.runtime_marker_path).read_text(encoding="utf-8"))
    assert marker["state"] == "finished"


def test_runtime_wrapper_evicts_old_output_to_enforce_total_quota(
    tmp_path: Path,
) -> None:
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    old_id = "7" * 32
    old_output = runtime_root / f"{old_id}.output"
    old_marker = runtime_root / f"{old_id}.json"
    old_output.write_bytes(b"o" * 2048)
    old_marker.write_text('{"state":"finished"}', encoding="utf-8")
    manager = ColabJobManager(
        FakeSession(),  # type: ignore[arg-type]
        output_excerpt_bytes=1024,
        runtime_artifact_bytes=2048,
        runtime_artifact_total_bytes=34 * 1024,
        runtime_root=str(runtime_root),
    )
    source = "print('n' * 1500)"
    raw = source.encode("utf-8")
    now = time.time()
    job = ColabJob(
        job_id="8" * 32,
        cell_id="test-cell",
        cell_index=0,
        code_bytes=len(raw),
        code_sha256=hashlib.sha256(raw).hexdigest(),
        state="running",
        tracking_state="active",
        started_at=now,
        updated_at=now,
        execution_timeout_seconds=30,
        runtime_marker_path=str(runtime_root / f"{'8' * 32}.json"),
        runtime_output_path=str(runtime_root / f"{'8' * 32}.output"),
        runtime_artifact_id="9" * 32,
    )

    subprocess.run(
        [sys.executable, "-c", manager._runtime_wrapper(job, source)],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )

    assert not old_output.exists()
    assert not old_marker.exists()
    assert Path(job.runtime_output_path).stat().st_size <= 2048
    total = sum(path.stat().st_size for path in runtime_root.iterdir())
    assert total <= 34 * 1024


def test_runtime_wrapper_bounds_marker_only_pairs_and_temp_files(
    tmp_path: Path,
) -> None:
    runtime_root = tmp_path / "runtime-pairs"
    runtime_root.mkdir()
    for stem, state in (("c" * 32, "running"), ("d" * 32, "finished")):
        (runtime_root / f"{stem}.json").write_text(
            json.dumps({"state": state}), encoding="utf-8"
        )
    stale_temp = runtime_root / f"{'e' * 32}.json.tmp"
    stale_temp.write_text("partial", encoding="utf-8")
    manager = ColabJobManager(
        FakeSession(),  # type: ignore[arg-type]
        output_excerpt_bytes=1024,
        runtime_artifact_bytes=1024,
        runtime_artifact_total_bytes=32 * 1024,
        runtime_root=str(runtime_root),
        max_tracked_jobs=2,
    )
    source = "result = 1"
    raw = source.encode("utf-8")
    now = time.time()
    job = ColabJob(
        job_id="f" * 32,
        cell_id="test-cell",
        cell_index=0,
        code_bytes=len(raw),
        code_sha256=hashlib.sha256(raw).hexdigest(),
        state="running",
        tracking_state="active",
        started_at=now,
        updated_at=now,
        execution_timeout_seconds=30,
        runtime_marker_path=str(runtime_root / f"{'f' * 32}.json"),
        runtime_output_path=str(runtime_root / f"{'f' * 32}.output"),
        runtime_artifact_id="1" * 32,
    )

    subprocess.run(
        [sys.executable, "-c", manager._runtime_wrapper(job, source)],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )

    stems = {
        path.name.removesuffix(".output").removesuffix(".json")
        for path in runtime_root.iterdir()
        if path.suffix in {".output", ".json"}
    }
    assert len(stems) <= 2
    assert "f" * 32 in stems
    assert not stale_temp.exists()


def test_runtime_wrapper_terminal_marker_fits_minimum_total_quota(
    tmp_path: Path,
) -> None:
    manager = ColabJobManager(
        FakeSession(),  # type: ignore[arg-type]
        output_excerpt_bytes=1024,
        runtime_artifact_bytes=1024,
        runtime_artifact_total_bytes=32 * 1024,
        runtime_root=str(tmp_path),
    )
    source = "raise RuntimeError('\\x01' * 4096)"
    raw = source.encode("utf-8")
    now = time.time()
    job = ColabJob(
        job_id="2" * 32,
        cell_id="test-cell",
        cell_index=0,
        code_bytes=len(raw),
        code_sha256=hashlib.sha256(raw).hexdigest(),
        state="running",
        tracking_state="active",
        started_at=now,
        updated_at=now,
        execution_timeout_seconds=30,
        runtime_marker_path=str(tmp_path / f"{'2' * 32}.json"),
        runtime_output_path=str(tmp_path / f"{'2' * 32}.output"),
        runtime_artifact_id="3" * 32,
    )

    completed = subprocess.run(
        [sys.executable, "-c", manager._runtime_wrapper(job, source)],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    marker = json.loads(Path(job.runtime_marker_path).read_text(encoding="utf-8"))
    total = sum(path.stat().st_size for path in tmp_path.iterdir())

    assert marker["state"] == "error"
    assert total <= 32 * 1024
    assert "__COLAB_CODEX_JOB__" in completed.stdout


def test_runtime_wrapper_fails_closed_when_display_capture_is_immutable(
    tmp_path: Path,
) -> None:
    manager = ColabJobManager(FakeSession())  # type: ignore[arg-type]
    user_side_effect = tmp_path / "user-code-ran"
    source = f"from pathlib import Path\nPath({str(user_side_effect)!r}).touch()"
    source_bytes = source.encode("utf-8")
    now = time.time()
    job = ColabJob(
        job_id="5" * 32,
        cell_id="test-cell",
        cell_index=0,
        code_bytes=len(source_bytes),
        code_sha256=hashlib.sha256(source_bytes).hexdigest(),
        state="running",
        tracking_state="active",
        started_at=now,
        updated_at=now,
        execution_timeout_seconds=30,
        runtime_marker_path=str(tmp_path / "display.json"),
        runtime_output_path=str(tmp_path / "display.output"),
        runtime_artifact_id="6" * 32,
    )
    immutable_publisher = """
class _LockedPublisher:
    @property
    def publish(self):
        return lambda *args, **kwargs: None
    @publish.setter
    def publish(self, value):
        raise RuntimeError("locked")
class _Shell:
    display_pub = _LockedPublisher()
def get_ipython():
    return _Shell()
"""

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            immutable_publisher + manager._runtime_wrapper(job, source),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    sentinel = "__COLAB_CODEX_JOB__"
    manifest_line = next(
        line for line in completed.stdout.splitlines() if line.startswith(sentinel)
    )
    manifest = json.loads(manifest_line[len(sentinel) :])

    assert manifest["state"] == "error"
    assert "Unable to install bounded display capture" in manifest["error"]
    assert not user_side_effect.exists()


def test_runtime_wrapper_emits_manifest_after_user_shadows_print(
    tmp_path: Path,
) -> None:
    manager = ColabJobManager(FakeSession())  # type: ignore[arg-type]
    source = "print('before shadow')\ndef print(*args, **kwargs): pass"
    raw = source.encode("utf-8")
    now = time.time()
    job = ColabJob(
        job_id="a" * 32,
        cell_id="test-cell",
        cell_index=0,
        code_bytes=len(raw),
        code_sha256=hashlib.sha256(raw).hexdigest(),
        state="running",
        tracking_state="active",
        started_at=now,
        updated_at=now,
        execution_timeout_seconds=30,
        runtime_marker_path=str(tmp_path / "shadow.json"),
        runtime_output_path=str(tmp_path / "shadow.output"),
        runtime_artifact_id="b" * 32,
    )

    completed = subprocess.run(
        [sys.executable, "-c", manager._runtime_wrapper(job, source)],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    sentinel = "__COLAB_CODEX_JOB__"
    manifest_line = next(
        line for line in completed.stdout.splitlines() if line.startswith(sentinel)
    )
    manifest = json.loads(manifest_line[len(sentinel) :])

    assert manifest["state"] == "finished"
    assert "before shadow" in manifest["output_excerpt"]


def test_runtime_wrapper_survives_shadowed_builtins_across_jobs(
    tmp_path: Path,
) -> None:
    manager = ColabJobManager(FakeSession())  # type: ignore[arg-type]

    def make_job(stem: str, source: str) -> ColabJob:
        raw = source.encode("utf-8")
        now = time.time()
        return ColabJob(
            job_id=stem * 32,
            cell_id=f"cell-{stem}",
            cell_index=0,
            code_bytes=len(raw),
            code_sha256=hashlib.sha256(raw).hexdigest(),
            state="running",
            tracking_state="active",
            started_at=now,
            updated_at=now,
            execution_timeout_seconds=30,
            runtime_marker_path=str(tmp_path / f"{stem * 32}.json"),
            runtime_output_path=str(tmp_path / f"{stem * 32}.output"),
            runtime_artifact_id={"8": "a", "9": "b", "7": "c"}[stem] * 32,
        )

    first_source = (
        "print('first')\nprint = 0\nlen = 0\nbytes = 0\ntype = 0\nstr = 0"
    )
    second_source = "second_job_completed = True"
    third_source = (
        "len = 0\nbytes = 0\ntype = 0\nstr = 0\n"
        "raise ValueError('expected failure')"
    )
    first = make_job("8", first_source)
    second = make_job("9", second_source)
    third = make_job("7", third_source)

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            manager._runtime_wrapper(first, first_source)
            + manager._runtime_wrapper(second, second_source)
            + manager._runtime_wrapper(third, third_source),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    sentinel = "__COLAB_CODEX_JOB__"
    manifests = [
        json.loads(line[len(sentinel) :])
        for line in completed.stdout.splitlines()
        if line.startswith(sentinel)
    ]

    assert [manifest["state"] for manifest in manifests] == [
        "finished",
        "finished",
        "error",
    ]
    assert manifests[2]["error"] == "ValueError: expected failure"
    assert json.loads(Path(second.runtime_marker_path).read_text())["state"] == "finished"


def test_runtime_wrapper_captures_display_when_get_ipython_is_builtin_only(
    tmp_path: Path,
) -> None:
    manager = ColabJobManager(FakeSession())  # type: ignore[arg-type]
    source = "get_ipython().display_pub.publish({'text/plain': 'bounded-display'})"
    raw = source.encode("utf-8")
    now = time.time()
    job = ColabJob(
        job_id="4" * 32,
        cell_id="display-cell",
        cell_index=0,
        code_bytes=len(raw),
        code_sha256=hashlib.sha256(raw).hexdigest(),
        state="running",
        tracking_state="active",
        started_at=now,
        updated_at=now,
        execution_timeout_seconds=30,
        runtime_marker_path=str(tmp_path / "builtin-display.json"),
        runtime_output_path=str(tmp_path / "builtin-display.output"),
        runtime_artifact_id="5" * 32,
    )
    prefix = """
import builtins
class _Publisher:
    def publish(self, data, metadata=None, **kwargs):
        print("UNBOUNDED-DISPLAY")
class _Shell:
    display_pub = _Publisher()
builtins.get_ipython = lambda: _Shell()
"""

    completed = subprocess.run(
        [sys.executable, "-c", prefix + manager._runtime_wrapper(job, source)],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    sentinel = "__COLAB_CODEX_JOB__"
    manifest_line = next(
        line for line in completed.stdout.splitlines() if line.startswith(sentinel)
    )
    manifest = json.loads(manifest_line[len(sentinel) :])

    assert manifest["state"] == "finished"
    assert "[display:text/plain] bounded-display" in manifest["output_excerpt"]
    assert "UNBOUNDED-DISPLAY" not in completed.stdout
