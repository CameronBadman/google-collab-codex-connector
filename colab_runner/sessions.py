from __future__ import annotations

import asyncio
import tempfile
import time
import uuid
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from .cli import ColabCli
from .notebooks import NotebookController, SelectedCell
from .orchestrator import (
    ActivityReporter,
    artifact_local_path,
    new_session_arguments,
    normalize_session_prefix,
    process_result_data,
    validate_accelerator,
    validate_dependencies,
    validate_remote_artifact,
    validate_session_name,
    validate_workload_path,
)
from .process import (
    ProcessExecutionError,
    ProcessExecutionTimeout,
    ProcessResult,
)


DEFAULT_IDLE_TIMEOUT_SECONDS: Final = 600.0
MIN_IDLE_TIMEOUT_SECONDS: Final = 60.0
MAX_IDLE_TIMEOUT_SECONDS: Final = 21_600.0
_MAX_SETUP_SECONDS: Final = 3_600.0
_MAX_OPERATION_SECONDS: Final = 86_400.0
_CLEANUP_RETRY_SECONDS: Final = 60.0
_DEFAULT_EXECUTION_RECORD_LIMIT: Final = 512
_CELL_OUTPUT_EXCERPT_BYTES: Final = 8 * 1024
_TERMINAL_EXECUTION_STATES: Final = frozenset(
    {"finished", "failed", "timed_out", "interrupted"}
)


@dataclass(slots=True)
class _SessionLease:
    session_name: str
    accelerator: str
    idle_timeout_seconds: float
    created_at_epoch: float
    last_activity_epoch: float
    expires_at_epoch: float
    expires_at_monotonic: float
    state: str = "ready"
    cleanup_error: str | None = None
    stop_requested: bool = False
    active_command_task: asyncio.Task[ProcessResult] | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    wake_reaper: asyncio.Event = field(default_factory=asyncio.Event)
    reaper_task: asyncio.Task[None] | None = None


@dataclass(slots=True)
class _CellExecutionRecord:
    execution_id: str
    session_name: str
    selected: SelectedCell
    timeout_seconds: float
    background: bool
    write_output_to_notebook: bool
    queued_at_epoch: float
    state: str = "queued"
    started_at_epoch: float | None = None
    completed_at_epoch: float | None = None
    elapsed_seconds: float | None = None
    ok: bool | None = None
    error: str | None = None
    stdout: str = ""
    stderr: str = ""
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    returncode: int | None = None
    output_writeback: dict[str, Any] | None = None


class ColabSessionManager:
    """Own reusable Colab sessions and release them after bounded idle time."""

    def __init__(
        self,
        cli: ColabCli | Any | None = None,
        *,
        cleanup_timeout_seconds: float = 60,
        minimum_idle_timeout_seconds: float = MIN_IDLE_TIMEOUT_SECONDS,
        maximum_idle_timeout_seconds: float = MAX_IDLE_TIMEOUT_SECONDS,
        notebook_controller: NotebookController | None = None,
        execution_record_limit: int = _DEFAULT_EXECUTION_RECORD_LIMIT,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        if minimum_idle_timeout_seconds <= 0:
            raise ValueError("minimum_idle_timeout_seconds must be positive")
        if maximum_idle_timeout_seconds < minimum_idle_timeout_seconds:
            raise ValueError(
                "maximum_idle_timeout_seconds must not be below the minimum"
            )
        if (
            isinstance(execution_record_limit, bool)
            or not isinstance(execution_record_limit, int)
            or execution_record_limit < 16
        ):
            raise ValueError("execution_record_limit must be at least 16")
        self.cli = cli or ColabCli()
        self.cleanup_timeout_seconds = cleanup_timeout_seconds
        self.minimum_idle_timeout_seconds = minimum_idle_timeout_seconds
        self.maximum_idle_timeout_seconds = maximum_idle_timeout_seconds
        self.notebooks = notebook_controller or NotebookController()
        self.execution_record_limit = execution_record_limit
        self._clock = clock
        self._wall_clock = wall_clock
        self._leases: dict[str, _SessionLease] = {}
        self._cell_executions: OrderedDict[str, _CellExecutionRecord] = (
            OrderedDict()
        )
        self._background_tasks: dict[str, asyncio.Task[None]] = {}
        self._closed = False

    async def start_session(
        self,
        *,
        accelerator: str = "CPU",
        packages: Sequence[str] = (),
        requirements_file: str | None = None,
        idle_timeout_seconds: float = DEFAULT_IDLE_TIMEOUT_SECONDS,
        setup_timeout_seconds: float = 900,
        session_name_prefix: str = "codex-live",
        reporter: ActivityReporter | None = None,
    ) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("Colab session manager is closed")
        validate_accelerator(accelerator)
        package_values, requirements = validate_dependencies(
            packages, requirements_file
        )
        idle_timeout = self._validate_idle_timeout(idle_timeout_seconds)
        if not 30 <= setup_timeout_seconds <= _MAX_SETUP_SECONDS:
            raise ValueError("setup_timeout_seconds must be between 30 and 3600")

        prefix = normalize_session_prefix(session_name_prefix)
        session_name = f"{prefix}-{uuid.uuid4().hex[:8]}"
        started = self._clock()
        deadline = asyncio.get_running_loop().time() + setup_timeout_seconds
        allocation_attempted = False
        failed_step: str | None = None
        failure_result: ProcessResult | None = None
        error: str | None = None
        state = "starting"

        try:
            failed_step = "provision"
            allocation_attempted = True
            await _report(
                reporter,
                f"Provisioning reusable {accelerator.replace('_', ' ')} session",
            )
            await self.cli.run(
                new_session_arguments(session_name, accelerator),
                timeout_seconds=_remaining(deadline),
            )

            if package_values or requirements is not None:
                failed_step = "install"
                await _report(reporter, "Installing reusable session dependencies")
                arguments = ["install", "-s", session_name]
                if requirements is not None:
                    arguments.extend(["-r", str(requirements)])
                else:
                    arguments.extend(package_values)
                await self.cli.run(
                    arguments,
                    timeout_seconds=_remaining(deadline),
                )
        except asyncio.CancelledError:
            if allocation_attempted:
                await self._cleanup_unmanaged_session(session_name)
            raise
        except ProcessExecutionTimeout as exc:
            state = "timed_out"
            error = str(exc)
            failure_result = exc.result
        except asyncio.TimeoutError:
            state = "timed_out"
            error = (
                f"Session setup exceeded its {setup_timeout_seconds:g}s deadline"
            )
        except ProcessExecutionError as exc:
            state = "failed"
            error = str(exc)
            failure_result = exc.result
        except FileNotFoundError as exc:
            state = "failed"
            error = str(exc)
        except Exception as exc:
            state = "failed"
            error = f"{type(exc).__name__}: {exc}"
        else:
            now_monotonic = self._clock()
            now_epoch = self._wall_clock()
            lease = _SessionLease(
                session_name=session_name,
                accelerator=accelerator,
                idle_timeout_seconds=idle_timeout,
                created_at_epoch=now_epoch,
                last_activity_epoch=now_epoch,
                expires_at_epoch=now_epoch + idle_timeout,
                expires_at_monotonic=now_monotonic + idle_timeout,
            )
            self._leases[session_name] = lease
            lease.reaper_task = asyncio.create_task(
                self._reap_idle_session(lease),
                name=f"colab-idle-reaper-{session_name}",
            )
            await _report(reporter, "Reusable Colab session is ready")
            return {
                "ok": True,
                "state": "ready",
                "session_name": session_name,
                "accelerator": accelerator,
                "elapsed_seconds": self._clock() - started,
                "failed_step": None,
                "error": None,
                "lease": self._lease_data(lease),
            }

        cleanup = (
            await self._cleanup_unmanaged_session(session_name)
            if allocation_attempted
            else {"attempted": False, "succeeded": False, "error": None}
        )
        result: dict[str, Any] = {
            "ok": False,
            "state": state,
            "session_name": session_name,
            "accelerator": accelerator,
            "elapsed_seconds": self._clock() - started,
            "failed_step": failed_step,
            "error": error,
            "cleanup": cleanup,
        }
        if failure_result is not None:
            result["command"] = process_result_data(failure_result)
        return result

    async def execute(
        self,
        *,
        session_name: str,
        script_path: str,
        cell_index: int | None = None,
        cell_id: str | None = None,
        timeout_seconds: float = 1800,
        background: bool = False,
        write_output_to_notebook: bool = False,
        reporter: ActivityReporter | None = None,
    ) -> dict[str, Any]:
        timeout = _validate_operation_timeout(timeout_seconds)
        source_path = validate_workload_path(script_path)
        lease = self._require_lease(session_name)
        has_cell_selector = cell_index is not None or cell_id is not None
        if has_cell_selector and source_path.suffix != ".ipynb":
            raise ValueError("cell selection is only valid for .ipynb workloads")
        if background and not has_cell_selector:
            raise ValueError(
                "background execution requires a selected notebook cell"
            )
        if write_output_to_notebook and not has_cell_selector:
            raise ValueError(
                "write_output_to_notebook requires a selected notebook cell"
            )

        selected = (
            await self.notebooks.select_code_cell(
                source_path,
                cell_index=cell_index,
                cell_id=cell_id,
            )
            if has_cell_selector
            else None
        )
        self._ensure_current(lease)
        execution_path, temporary_directory = _execution_path(
            source_path,
            selected,
        )
        if selected is not None:
            selected = selected.reference()
        started = self._clock()
        record = (
            self._new_cell_execution(
                lease,
                selected,
                timeout_seconds=timeout,
                background=background,
                write_output_to_notebook=write_output_to_notebook,
            )
            if selected is not None
            else None
        )

        if background:
            assert record is not None
            try:
                await _report(
                    reporter,
                    f"Queued notebook cell {selected.cell_index}",
                )
            except asyncio.CancelledError:
                self._interrupt_cell_execution(
                    record,
                    "The background execution request was cancelled before queueing",
                    started=started,
                )
                if temporary_directory is not None:
                    temporary_directory.cleanup()
                raise
            task = asyncio.create_task(
                self._run_background_cell(
                    lease,
                    source_path=source_path,
                    execution_path=execution_path,
                    temporary_directory=temporary_directory,
                    selected=selected,
                    record=record,
                    timeout_seconds=timeout,
                    started=started,
                ),
                name=f"colab-cell-{record.execution_id}",
            )
            self._background_tasks[record.execution_id] = task
            return {
                "ok": True,
                "state": "queued",
                "session_name": session_name,
                "source_path": str(source_path),
                "selected_cell": selected.data(),
                "execution_id": record.execution_id,
                "cell_execution": self._record_data(record),
                "lease": self._lease_data(lease),
                "error": None,
            }

        try:
            result = await self._execute_prepared(
                lease,
                source_path=source_path,
                execution_path=execution_path,
                selected=selected,
                timeout_seconds=timeout,
                started=started,
                record=record,
                reporter=reporter,
            )
            if record is not None:
                await self._finish_cell_execution(record, result)
                result["execution_id"] = record.execution_id
                result["cell_execution"] = self._record_data(record)
            return result
        except asyncio.CancelledError:
            if record is not None:
                self._interrupt_cell_execution(
                    record,
                    "The execution request was cancelled",
                    started=started,
                )
            raise
        finally:
            if temporary_directory is not None:
                temporary_directory.cleanup()

    async def _execute_prepared(
        self,
        lease: _SessionLease,
        *,
        source_path: Path,
        execution_path: Path,
        selected: SelectedCell | None,
        timeout_seconds: float,
        started: float,
        record: _CellExecutionRecord | None,
        reporter: ActivityReporter | None,
    ) -> dict[str, Any]:
        selected_data = selected.data() if selected is not None else None
        async with lease.lock:
            self._ensure_current(lease)
            lease.state = "executing"
            lease.cleanup_error = None
            self._touch(lease)
            if record is not None:
                self._start_cell_execution(record)
            description = (
                f"Executing notebook cell {selected.cell_index}"
                if selected is not None
                else f"Executing {source_path.name}"
            )
            try:
                await _report(reporter, description)
            except asyncio.CancelledError:
                lease.state = "ready"
                self._touch(lease)
                raise
            if lease.stop_requested:
                cleanup = await self._release_locked(lease)
                return {
                    "ok": False,
                    "state": "interrupted",
                    "session_name": lease.session_name,
                    "source_path": str(source_path),
                    "selected_cell": selected_data,
                    "elapsed_seconds": self._clock() - started,
                    "error": "The reusable session was stopped before execution",
                    "cleanup": cleanup,
                }
            command_task = asyncio.create_task(
                self.cli.run(
                    [
                        "exec",
                        "-s",
                        lease.session_name,
                        "--timeout",
                        f"{timeout_seconds:.3f}",
                        "-f",
                        str(execution_path),
                    ],
                    timeout_seconds=timeout_seconds,
                )
            )
            lease.active_command_task = command_task
            try:
                command = await command_task
            except asyncio.CancelledError:
                lease.active_command_task = None
                lease.state = "stopping"
                cleanup = await self._release_locked(lease)
                if lease.stop_requested:
                    return {
                        "ok": False,
                        "state": "interrupted",
                        "session_name": lease.session_name,
                        "source_path": str(source_path),
                        "selected_cell": selected_data,
                        "elapsed_seconds": self._clock() - started,
                        "error": "The reusable session was explicitly stopped",
                        "cleanup": cleanup,
                    }
                raise
            except ProcessExecutionTimeout as exc:
                lease.active_command_task = None
                lease.state = "stopping"
                cleanup = await self._release_locked(lease)
                return {
                    "ok": False,
                    "state": "timed_out",
                    "session_name": lease.session_name,
                    "source_path": str(source_path),
                    "selected_cell": selected_data,
                    "elapsed_seconds": self._clock() - started,
                    "error": str(exc),
                    "execution": process_result_data(exc.result),
                    "cleanup": cleanup,
                }
            except ProcessExecutionError as exc:
                lease.active_command_task = None
                if lease.stop_requested:
                    lease.state = "stopping"
                    cleanup = await self._release_locked(lease)
                    return {
                        "ok": False,
                        "state": "interrupted",
                        "session_name": lease.session_name,
                        "source_path": str(source_path),
                        "selected_cell": selected_data,
                        "elapsed_seconds": self._clock() - started,
                        "error": "The reusable session was explicitly stopped",
                        "execution": process_result_data(exc.result),
                        "cleanup": cleanup,
                    }
                lease.state = "ready"
                self._touch(lease)
                return {
                    "ok": False,
                    "state": "failed",
                    "session_name": lease.session_name,
                    "source_path": str(source_path),
                    "selected_cell": selected_data,
                    "elapsed_seconds": self._clock() - started,
                    "error": str(exc),
                    "execution": process_result_data(exc.result),
                    "lease": self._lease_data(lease),
                    **_notebook_output_data(source_path, selected),
                }
            except FileNotFoundError as exc:
                lease.active_command_task = None
                lease.state = "ready"
                self._touch(lease)
                return {
                    "ok": False,
                    "state": "failed",
                    "session_name": lease.session_name,
                    "source_path": str(source_path),
                    "selected_cell": selected_data,
                    "elapsed_seconds": self._clock() - started,
                    "error": str(exc),
                    "lease": self._lease_data(lease),
                }
            except Exception as exc:
                lease.active_command_task = None
                lease.state = "ready"
                self._touch(lease)
                return {
                    "ok": False,
                    "state": "failed",
                    "session_name": lease.session_name,
                    "source_path": str(source_path),
                    "selected_cell": selected_data,
                    "elapsed_seconds": self._clock() - started,
                    "error": f"{type(exc).__name__}: {exc}",
                    "lease": self._lease_data(lease),
                }

            lease.active_command_task = None
            if lease.stop_requested:
                lease.state = "stopping"
                cleanup = await self._release_locked(lease)
                return {
                    "ok": False,
                    "state": "interrupted",
                    "session_name": lease.session_name,
                    "source_path": str(source_path),
                    "selected_cell": selected_data,
                    "elapsed_seconds": self._clock() - started,
                    "error": "The reusable session was explicitly stopped",
                    "execution": process_result_data(command),
                    "cleanup": cleanup,
                }
            lease.state = "ready"
            self._touch(lease)
            return {
                "ok": True,
                "state": "finished",
                "session_name": lease.session_name,
                "source_path": str(source_path),
                "selected_cell": selected_data,
                "elapsed_seconds": self._clock() - started,
                "error": None,
                "execution": process_result_data(command),
                "lease": self._lease_data(lease),
                **_notebook_output_data(source_path, selected),
            }

    async def _run_background_cell(
        self,
        lease: _SessionLease,
        *,
        source_path: Path,
        execution_path: Path,
        temporary_directory: tempfile.TemporaryDirectory[str] | None,
        selected: SelectedCell,
        record: _CellExecutionRecord,
        timeout_seconds: float,
        started: float,
    ) -> None:
        try:
            result = await self._execute_prepared(
                lease,
                source_path=source_path,
                execution_path=execution_path,
                selected=selected,
                timeout_seconds=timeout_seconds,
                started=started,
                record=record,
                reporter=None,
            )
            await self._finish_cell_execution(record, result)
        except asyncio.CancelledError:
            self._interrupt_cell_execution(
                record,
                "The background execution was cancelled",
                started=started,
            )
        except Exception as exc:
            self._fail_cell_execution(
                record,
                f"{type(exc).__name__}: {exc}",
                started=started,
            )
        finally:
            if temporary_directory is not None:
                temporary_directory.cleanup()
            self._background_tasks.pop(record.execution_id, None)
            self._trim_cell_executions()

    def cell_status(self, execution_id: str) -> dict[str, Any]:
        if (
            not isinstance(execution_id, str)
            or len(execution_id) != 32
            or any(character not in "0123456789abcdef" for character in execution_id)
        ):
            raise ValueError("execution_id must be a 32-character lowercase hex ID")
        record = self._cell_executions.get(execution_id)
        if record is None:
            raise ValueError(f"cell execution {execution_id!r} was not found")
        return self._record_data(record)

    def latest_cell_execution(
        self,
        *,
        notebook_path: str,
        cell_index: int,
        cell_id: str | None,
        session_name: str | None = None,
    ) -> dict[str, Any] | None:
        resolved = str(Path(notebook_path).expanduser().resolve())
        for record in reversed(self._cell_executions.values()):
            if session_name is not None and record.session_name != session_name:
                continue
            if str(record.selected.notebook_path) != resolved:
                continue
            if cell_id is not None and record.selected.cell_id == cell_id:
                return self._record_summary(record)
            if cell_id is None and record.selected.cell_index == cell_index:
                return self._record_summary(record)
        return None

    def _new_cell_execution(
        self,
        lease: _SessionLease,
        selected: SelectedCell,
        *,
        timeout_seconds: float,
        background: bool,
        write_output_to_notebook: bool,
    ) -> _CellExecutionRecord:
        record = _CellExecutionRecord(
            execution_id=uuid.uuid4().hex,
            session_name=lease.session_name,
            selected=selected.reference(),
            timeout_seconds=timeout_seconds,
            background=background,
            write_output_to_notebook=write_output_to_notebook,
            queued_at_epoch=self._wall_clock(),
        )
        self._cell_executions[record.execution_id] = record
        self._trim_cell_executions()
        return record

    def _start_cell_execution(self, record: _CellExecutionRecord) -> None:
        record.state = "running"
        record.started_at_epoch = self._wall_clock()

    async def _finish_cell_execution(
        self,
        record: _CellExecutionRecord,
        result: dict[str, Any],
    ) -> None:
        state = str(result.get("state", "failed"))
        if state not in _TERMINAL_EXECUTION_STATES:
            state = "failed"
        record.ok = bool(result.get("ok", False))
        record.error = _bounded_error(result.get("error"))
        elapsed = result.get("elapsed_seconds")
        record.elapsed_seconds = (
            float(elapsed)
            if isinstance(elapsed, int | float) and not isinstance(elapsed, bool)
            else None
        )
        completed_at_epoch = self._wall_clock()
        execution = result.get("execution")
        if isinstance(execution, dict):
            stdout, stdout_truncated = _bounded_text(
                execution.get("stdout"),
                _CELL_OUTPUT_EXCERPT_BYTES,
            )
            stderr, stderr_truncated = _bounded_text(
                execution.get("stderr"),
                _CELL_OUTPUT_EXCERPT_BYTES,
            )
            record.stdout = stdout
            record.stderr = stderr
            record.stdout_truncated = bool(
                execution.get("stdout_truncated", False) or stdout_truncated
            )
            record.stderr_truncated = bool(
                execution.get("stderr_truncated", False) or stderr_truncated
            )
            returncode = execution.get("returncode")
            if isinstance(returncode, int) and not isinstance(returncode, bool):
                record.returncode = returncode

        if not record.write_output_to_notebook:
            record.completed_at_epoch = completed_at_epoch
            record.state = state
            self._trim_cell_executions()
            return
        if state not in {"finished", "failed"} or not isinstance(execution, dict):
            record.output_writeback = {
                "requested": True,
                "state": "skipped",
                "written": False,
                "error": "Output write-back requires a completed cell response",
            }
            record.completed_at_epoch = completed_at_epoch
            record.state = state
            self._trim_cell_executions()
            return

        record.output_writeback = {
            "requested": True,
            "state": "pending",
            "written": False,
            "error": None,
        }
        try:
            record.output_writeback = await self.notebooks.write_execution_outputs(
                record.selected,
                session_name=record.session_name,
                execution_id=record.execution_id,
                state=state,
                executed_at=_timestamp(completed_at_epoch),
                stdout=record.stdout,
                stderr=record.stderr,
                stdout_truncated=record.stdout_truncated,
                stderr_truncated=record.stderr_truncated,
            )
        except Exception as exc:
            record.output_writeback = {
                "requested": True,
                "state": "failed",
                "written": False,
                "error": _bounded_error(f"{type(exc).__name__}: {exc}"),
            }
        record.completed_at_epoch = completed_at_epoch
        record.state = state
        self._trim_cell_executions()

    def _interrupt_cell_execution(
        self,
        record: _CellExecutionRecord,
        error: str,
        *,
        started: float,
    ) -> None:
        if record.state in _TERMINAL_EXECUTION_STATES:
            return
        record.state = "interrupted"
        record.ok = False
        record.error = _bounded_error(error)
        record.elapsed_seconds = self._clock() - started
        record.completed_at_epoch = self._wall_clock()
        if record.write_output_to_notebook:
            record.output_writeback = {
                "requested": True,
                "state": "skipped",
                "written": False,
                "error": "Interrupted execution output is not written",
            }
        self._trim_cell_executions()

    def _fail_cell_execution(
        self,
        record: _CellExecutionRecord,
        error: str,
        *,
        started: float,
    ) -> None:
        if record.state in _TERMINAL_EXECUTION_STATES:
            return
        record.state = "failed"
        record.ok = False
        record.error = _bounded_error(error)
        record.elapsed_seconds = self._clock() - started
        record.completed_at_epoch = self._wall_clock()
        if record.write_output_to_notebook:
            record.output_writeback = {
                "requested": True,
                "state": "skipped",
                "written": False,
                "error": "No completed cell response was available",
            }
        self._trim_cell_executions()

    def _record_data(self, record: _CellExecutionRecord) -> dict[str, Any]:
        return {
            "execution_id": record.execution_id,
            "session_name": record.session_name,
            "state": record.state,
            "terminal": record.state in _TERMINAL_EXECUTION_STATES,
            "ok": record.ok,
            "background": record.background,
            "notebook_path": str(record.selected.notebook_path),
            "selected_cell": record.selected.data(),
            "timeout_seconds": record.timeout_seconds,
            "queued_at": _timestamp(record.queued_at_epoch),
            "started_at": (
                _timestamp(record.started_at_epoch)
                if record.started_at_epoch is not None
                else None
            ),
            "completed_at": (
                _timestamp(record.completed_at_epoch)
                if record.completed_at_epoch is not None
                else None
            ),
            "elapsed_seconds": record.elapsed_seconds,
            "error": record.error,
            "execution": {
                "returncode": record.returncode,
                "stdout": record.stdout,
                "stderr": record.stderr,
                "stdout_bytes": len(record.stdout.encode("utf-8")),
                "stderr_bytes": len(record.stderr.encode("utf-8")),
                "stdout_truncated": record.stdout_truncated,
                "stderr_truncated": record.stderr_truncated,
            },
            "output_writeback": (
                record.output_writeback
                if record.write_output_to_notebook
                else {"requested": False}
            ),
        }

    def _record_summary(self, record: _CellExecutionRecord) -> dict[str, Any]:
        return {
            "execution_id": record.execution_id,
            "session_name": record.session_name,
            "state": record.state,
            "terminal": record.state in _TERMINAL_EXECUTION_STATES,
            "ok": record.ok,
            "background": record.background,
            "selected_cell": record.selected.data(),
            "queued_at": _timestamp(record.queued_at_epoch),
            "started_at": (
                _timestamp(record.started_at_epoch)
                if record.started_at_epoch is not None
                else None
            ),
            "completed_at": (
                _timestamp(record.completed_at_epoch)
                if record.completed_at_epoch is not None
                else None
            ),
            "error": record.error,
            "output_writeback": (
                record.output_writeback
                if record.write_output_to_notebook
                else {"requested": False}
            ),
        }

    def _trim_cell_executions(self) -> None:
        while len(self._cell_executions) > self.execution_record_limit:
            removable = next(
                (
                    execution_id
                    for execution_id, record in self._cell_executions.items()
                    if record.state in _TERMINAL_EXECUTION_STATES
                    and execution_id not in self._background_tasks
                ),
                None,
            )
            if removable is None:
                return
            del self._cell_executions[removable]

    async def renew_session(
        self,
        *,
        session_name: str,
        idle_timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        lease = self._require_lease(session_name)
        replacement = (
            None
            if idle_timeout_seconds is None
            else self._validate_idle_timeout(idle_timeout_seconds)
        )
        async with lease.lock:
            self._ensure_current(lease)
            if replacement is not None:
                lease.idle_timeout_seconds = replacement
            self._touch(lease)
            return {
                "ok": True,
                "state": lease.state,
                "session_name": session_name,
                "lease": self._lease_data(lease),
            }

    async def download_artifact(
        self,
        *,
        session_name: str,
        remote_path: str,
        artifact_dir: str,
        timeout_seconds: float = 300,
        reporter: ActivityReporter | None = None,
    ) -> dict[str, Any]:
        timeout = _validate_operation_timeout(timeout_seconds)
        validate_remote_artifact(remote_path)
        destination = Path(artifact_dir).expanduser()
        if not destination.is_absolute():
            raise ValueError("artifact_dir must be an absolute local path")
        local_path = artifact_local_path(destination.resolve(), remote_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        lease = self._require_lease(session_name)
        await _report(reporter, "Downloading artifact from reusable session")
        return await self._run_auxiliary_command(
            lease,
            [
                "download",
                "-s",
                session_name,
                remote_path,
                str(local_path),
            ],
            timeout_seconds=timeout,
            success_data={
                "remote_path": remote_path,
                "local_path": str(local_path),
            },
        )

    async def export_log(
        self,
        *,
        session_name: str,
        output_path: str,
        timeout_seconds: float = 300,
        reporter: ActivityReporter | None = None,
    ) -> dict[str, Any]:
        timeout = _validate_operation_timeout(timeout_seconds)
        destination = Path(output_path).expanduser()
        if not destination.is_absolute() or destination.suffix != ".ipynb":
            raise ValueError("output_path must be an absolute .ipynb path")
        destination = destination.resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)

        lease = self._require_lease(session_name)
        await _report(reporter, "Exporting reusable session notebook log")
        return await self._run_auxiliary_command(
            lease,
            ["log", "-s", session_name, "-o", str(destination)],
            timeout_seconds=timeout,
            success_data={"log_path": str(destination)},
        )

    async def stop_session(self, session_name: str) -> dict[str, Any]:
        lease = self._require_lease(session_name)
        lease.stop_requested = True
        lease.state = "stopping"
        background_tasks = self._background_tasks_for_session(session_name)
        for task in background_tasks:
            task.cancel()
        active_command = lease.active_command_task
        if active_command is not None:
            active_command.cancel()
        if background_tasks:
            await asyncio.gather(*background_tasks, return_exceptions=True)
        async with lease.lock:
            if self._leases.get(session_name) is not lease:
                cleanup = {"attempted": True, "succeeded": True, "error": None}
                return {
                    "ok": True,
                    "state": "stopped",
                    "session_name": session_name,
                    "error": None,
                    "cleanup": cleanup,
                }
            lease.state = "stopping"
            cleanup = await self._release_locked(lease)
        return {
            "ok": cleanup["succeeded"],
            "state": "stopped" if cleanup["succeeded"] else "cleanup_failed",
            "session_name": session_name,
            "error": cleanup["error"],
            "cleanup": cleanup,
        }

    def is_managed(self, session_name: str) -> bool:
        return session_name in self._leases

    def lease_status(self, session_name: str) -> dict[str, Any] | None:
        lease = self._leases.get(session_name)
        return None if lease is None else self._lease_data(lease)

    def leases(self) -> list[dict[str, Any]]:
        return [
            self._lease_data(lease)
            for lease in sorted(
                self._leases.values(), key=lambda item: item.session_name
            )
        ]

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        leases = list(self._leases.values())
        background_tasks = list(self._background_tasks.values())
        for lease in leases:
            lease.stop_requested = True
            lease.state = "stopping"
            if lease.active_command_task is not None:
                lease.active_command_task.cancel()
            if lease.reaper_task is not None:
                lease.reaper_task.cancel()
        for task in background_tasks:
            task.cancel()
        await asyncio.gather(
            *(
                lease.reaper_task
                for lease in leases
                if lease.reaper_task is not None
            ),
            return_exceptions=True,
        )
        await asyncio.gather(*background_tasks, return_exceptions=True)

        async def release(lease: _SessionLease) -> None:
            async with lease.lock:
                if self._leases.get(lease.session_name) is lease:
                    lease.state = "stopping"
                    await self._release_locked(lease)

        await asyncio.gather(*(release(lease) for lease in leases))

    async def _run_auxiliary_command(
        self,
        lease: _SessionLease,
        arguments: list[str],
        *,
        timeout_seconds: float,
        success_data: dict[str, Any],
    ) -> dict[str, Any]:
        started = self._clock()
        async with lease.lock:
            self._ensure_current(lease)
            self._touch(lease)
            try:
                command = await self.cli.run(
                    arguments,
                    timeout_seconds=timeout_seconds,
                )
            except ProcessExecutionTimeout as exc:
                self._touch(lease)
                return {
                    "ok": False,
                    "state": "timed_out",
                    "session_name": lease.session_name,
                    "elapsed_seconds": self._clock() - started,
                    "error": str(exc),
                    "command": process_result_data(exc.result),
                    "lease": self._lease_data(lease),
                }
            except ProcessExecutionError as exc:
                self._touch(lease)
                return {
                    "ok": False,
                    "state": "failed",
                    "session_name": lease.session_name,
                    "elapsed_seconds": self._clock() - started,
                    "error": str(exc),
                    "command": process_result_data(exc.result),
                    "lease": self._lease_data(lease),
                }

            self._touch(lease)
            return {
                "ok": True,
                "state": "finished",
                "session_name": lease.session_name,
                "elapsed_seconds": self._clock() - started,
                "error": None,
                **success_data,
                "command": process_result_data(command),
                "lease": self._lease_data(lease),
            }

    async def _reap_idle_session(self, lease: _SessionLease) -> None:
        try:
            while self._leases.get(lease.session_name) is lease:
                delay = max(0.0, lease.expires_at_monotonic - self._clock())
                try:
                    await asyncio.wait_for(lease.wake_reaper.wait(), timeout=delay)
                    lease.wake_reaper.clear()
                    continue
                except asyncio.TimeoutError:
                    pass

                async with lease.lock:
                    if self._leases.get(lease.session_name) is not lease:
                        return
                    if self._clock() < lease.expires_at_monotonic:
                        continue
                    lease.state = "stopping"
                    cleanup = await self._release_locked(lease)
                    if cleanup["succeeded"]:
                        return
        except asyncio.CancelledError:
            return

    async def _release_locked(self, lease: _SessionLease) -> dict[str, Any]:
        cancelled = False
        stop_task = asyncio.create_task(
            self.cli.run(
                ["stop", "-s", lease.session_name],
                timeout_seconds=self.cleanup_timeout_seconds,
            )
        )
        try:
            try:
                await asyncio.shield(stop_task)
            except asyncio.CancelledError:
                cancelled = True
                await stop_task
        except (ProcessExecutionError, ProcessExecutionTimeout) as exc:
            result = exc.result
            detail = f"{result.stdout}\n{result.stderr}".lower()
            if "not found" in detail:
                self._drop_lease(lease)
                cleanup = {"attempted": True, "succeeded": True, "error": None}
            else:
                lease.state = "cleanup_failed"
                lease.cleanup_error = f"{type(exc).__name__}: {exc}"
                self._schedule_cleanup_retry(lease)
                cleanup = {
                    "attempted": True,
                    "succeeded": False,
                    "error": lease.cleanup_error,
                }
        except Exception as exc:
            lease.state = "cleanup_failed"
            lease.cleanup_error = f"{type(exc).__name__}: {exc}"
            self._schedule_cleanup_retry(lease)
            cleanup = {
                "attempted": True,
                "succeeded": False,
                "error": lease.cleanup_error,
            }
        else:
            self._drop_lease(lease)
            cleanup = {"attempted": True, "succeeded": True, "error": None}

        if cancelled:
            raise asyncio.CancelledError
        return cleanup

    async def _cleanup_unmanaged_session(
        self, session_name: str
    ) -> dict[str, Any]:
        cleanup_task = asyncio.create_task(
            self.cli.run(
                ["stop", "-s", session_name],
                timeout_seconds=self.cleanup_timeout_seconds,
            )
        )
        try:
            await asyncio.shield(cleanup_task)
        except asyncio.CancelledError:
            try:
                await cleanup_task
            except Exception:
                pass
            raise
        except (ProcessExecutionError, ProcessExecutionTimeout) as exc:
            detail = f"{exc.result.stdout}\n{exc.result.stderr}".lower()
            if "not found" in detail:
                return {"attempted": True, "succeeded": True, "error": None}
            return {
                "attempted": True,
                "succeeded": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        except Exception as exc:
            return {
                "attempted": True,
                "succeeded": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        return {"attempted": True, "succeeded": True, "error": None}

    def _require_lease(self, session_name: str) -> _SessionLease:
        validate_session_name(session_name)
        lease = self._leases.get(session_name)
        if lease is None:
            raise ValueError(
                f"session {session_name!r} is not managed by this connector instance"
            )
        return lease

    def _ensure_current(self, lease: _SessionLease) -> None:
        if self._leases.get(lease.session_name) is not lease:
            raise ValueError(
                f"session {lease.session_name!r} is no longer managed"
            )

    def _touch(self, lease: _SessionLease) -> None:
        now_monotonic = self._clock()
        now_epoch = self._wall_clock()
        lease.last_activity_epoch = now_epoch
        lease.expires_at_epoch = now_epoch + lease.idle_timeout_seconds
        lease.expires_at_monotonic = now_monotonic + lease.idle_timeout_seconds
        lease.wake_reaper.set()

    def _schedule_cleanup_retry(self, lease: _SessionLease) -> None:
        retry_seconds = min(
            _CLEANUP_RETRY_SECONDS,
            lease.idle_timeout_seconds,
        )
        now_monotonic = self._clock()
        now_epoch = self._wall_clock()
        lease.expires_at_monotonic = now_monotonic + retry_seconds
        lease.expires_at_epoch = now_epoch + retry_seconds
        lease.wake_reaper.set()

    def _drop_lease(self, lease: _SessionLease) -> None:
        if self._leases.get(lease.session_name) is lease:
            del self._leases[lease.session_name]
        current_task = asyncio.current_task()
        for task in self._background_tasks_for_session(lease.session_name):
            if task is not current_task:
                task.cancel()
        lease.wake_reaper.set()
        task = lease.reaper_task
        if task is not None and task is not asyncio.current_task():
            task.cancel()

    def _background_tasks_for_session(
        self, session_name: str
    ) -> list[asyncio.Task[None]]:
        return [
            task
            for execution_id, task in self._background_tasks.items()
            if self._cell_executions[execution_id].session_name == session_name
        ]

    def _validate_idle_timeout(self, value: float) -> float:
        if not self.minimum_idle_timeout_seconds <= value <= (
            self.maximum_idle_timeout_seconds
        ):
            raise ValueError(
                "idle_timeout_seconds must be between "
                f"{self.minimum_idle_timeout_seconds:g} and "
                f"{self.maximum_idle_timeout_seconds:g}"
            )
        return float(value)

    def _lease_data(self, lease: _SessionLease) -> dict[str, Any]:
        return {
            "session_name": lease.session_name,
            "accelerator": lease.accelerator,
            "state": lease.state,
            "idle_timeout_seconds": lease.idle_timeout_seconds,
            "idle_seconds_remaining": max(
                0.0, lease.expires_at_monotonic - self._clock()
            ),
            "created_at": _timestamp(lease.created_at_epoch),
            "last_activity_at": _timestamp(lease.last_activity_epoch),
            "expires_at": _timestamp(lease.expires_at_epoch),
            "cleanup_error": lease.cleanup_error,
        }


def _execution_path(
    source_path: Path,
    selected: SelectedCell | None,
) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    if selected is None:
        return source_path, None
    temporary_directory = tempfile.TemporaryDirectory(
        prefix="colab-runner-cell-"
    )
    execution_path = Path(temporary_directory.name) / "cell.py"
    execution_path.write_text(selected.source + "\n", encoding="utf-8")
    return execution_path, temporary_directory


def _validate_operation_timeout(value: float) -> float:
    if not 1 <= value <= _MAX_OPERATION_SECONDS:
        raise ValueError("timeout_seconds must be between 1 and 86400")
    return float(value)


def _notebook_output_data(
    source_path: Path,
    selected_cell: SelectedCell | None,
) -> dict[str, str]:
    if source_path.suffix != ".ipynb" or selected_cell is not None:
        return {}
    output_path = source_path.with_name(f"{source_path.stem}_output.ipynb")
    return (
        {"notebook_output_path": str(output_path)}
        if output_path.is_file()
        else {}
    )


def _bounded_text(value: Any, limit: int) -> tuple[str, bool]:
    text = value if isinstance(value, str) else ""
    encoded = text.encode("utf-8")
    if len(encoded) <= limit:
        return text, False
    marker = "\n... output excerpt truncated by colab-runner ...\n"
    marker_bytes = marker.encode("utf-8")
    available = max(0, limit - len(marker_bytes))
    head_limit = available // 2
    tail_limit = available - head_limit
    head = encoded[:head_limit].decode("utf-8", errors="ignore")
    tail = (
        encoded[-tail_limit:].decode("utf-8", errors="ignore")
        if tail_limit
        else ""
    )
    return f"{head}{marker}{tail}", True


def _bounded_error(value: Any) -> str | None:
    if value is None:
        return None
    excerpt, _ = _bounded_text(str(value), 2048)
    return excerpt


def _remaining(deadline: float) -> float:
    remaining = deadline - asyncio.get_running_loop().time()
    if remaining <= 0:
        raise asyncio.TimeoutError
    return min(remaining, _MAX_OPERATION_SECONDS)


def _timestamp(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, UTC).isoformat()


async def _report(
    reporter: Callable[[str], Awaitable[None]] | None,
    message: str,
) -> None:
    if reporter is None:
        return
    try:
        await reporter(message)
    except asyncio.CancelledError:
        raise
    except Exception:
        return
