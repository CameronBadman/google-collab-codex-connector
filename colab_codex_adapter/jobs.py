from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import math
import os
import tempfile
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .artifacts import (
    DEFAULT_ARTIFACT_TTL_SECONDS,
    DEFAULT_MAX_ARTIFACT_TOTAL_BYTES,
    MAX_ARTIFACT_READ_BYTES,
    ArtifactNotFoundError,
    ArtifactRef,
    ArtifactStore,
    get_default_artifact_store,
    json_bytes,
    valid_opaque_id,
)
from .session import ColabSessionManager, NotConnectedError
from .tools import first_json_object

DEFAULT_EXECUTION_TIMEOUT_SECONDS = 43_200.0
MAX_EXECUTION_TIMEOUT_SECONDS = 86_400.0
DEFAULT_WAIT_TIMEOUT_SECONDS = 300.0
MIN_WAIT_TIMEOUT_SECONDS = 1.0
MAX_WAIT_TIMEOUT_SECONDS = 900.0
DEFAULT_OUTPUT_EXCERPT_BYTES = int(
    os.environ.get("COLAB_CODEX_JOB_OUTPUT_EXCERPT_BYTES", 64 * 1024)
)
DEFAULT_RUNTIME_ARTIFACT_BYTES = int(
    os.environ.get("COLAB_CODEX_MAX_ARTIFACT_BYTES", 32 * 1024 * 1024)
)
RUNTIME_MARKER_RESERVE_BYTES = 32 * 1024
MAX_SUBMITTED_CODE_BYTES = int(
    os.environ.get("COLAB_CODEX_MAX_SUBMITTED_CODE_BYTES", 1024 * 1024)
)
if MAX_SUBMITTED_CODE_BYTES <= 0:
    raise ValueError("COLAB_CODEX_MAX_SUBMITTED_CODE_BYTES must be positive")
DEFAULT_JOB_JOURNAL_PATH = Path(
    os.environ.get(
        "COLAB_CODEX_JOB_JOURNAL_PATH", "/tmp/colab-codex-adapter/jobs.json"
    )
)
DEFAULT_COLAB_RUNTIME_ROOT = "/content/.colab_codex/jobs"
MAX_JOURNAL_JOBS = 1024
DEFAULT_MAX_TRACKED_JOBS = int(
    os.environ.get("COLAB_CODEX_MAX_TRACKED_JOBS", MAX_JOURNAL_JOBS)
)
DEFAULT_MAX_JOB_CELLS = int(os.environ.get("COLAB_CODEX_MAX_JOB_CELLS", 16))
if not 1 <= DEFAULT_MAX_JOB_CELLS <= MAX_JOURNAL_JOBS:
    raise ValueError(
        f"COLAB_CODEX_MAX_JOB_CELLS must be between 1 and {MAX_JOURNAL_JOBS}"
    )
DEFAULT_CELL_METADATA_PAGE_SIZE = int(
    os.environ.get("COLAB_CODEX_CELL_METADATA_PAGE_SIZE", 8)
)
if not 1 <= DEFAULT_CELL_METADATA_PAGE_SIZE <= 256:
    raise ValueError("COLAB_CODEX_CELL_METADATA_PAGE_SIZE must be between 1 and 256")
DEFAULT_RUNTIME_STALE_GRACE_SECONDS = float(
    os.environ.get("COLAB_CODEX_RUNTIME_STALE_GRACE_SECONDS", 300)
)
if (
    not math.isfinite(DEFAULT_RUNTIME_STALE_GRACE_SECONDS)
    or DEFAULT_RUNTIME_STALE_GRACE_SECONDS < 0
):
    raise ValueError(
        "COLAB_CODEX_RUNTIME_STALE_GRACE_SECONDS must be finite and nonnegative"
    )
DEFAULT_ARTIFACT_PROBE_TIMEOUT_SECONDS = float(
    os.environ.get("COLAB_CODEX_ARTIFACT_PROBE_TIMEOUT_SECONDS", 30)
)
if (
    not math.isfinite(DEFAULT_ARTIFACT_PROBE_TIMEOUT_SECONDS)
    or not 1 <= DEFAULT_ARTIFACT_PROBE_TIMEOUT_SECONDS <= 300
):
    raise ValueError(
        "COLAB_CODEX_ARTIFACT_PROBE_TIMEOUT_SECONDS must be between 1 and 300"
    )
DEFAULT_KERNEL_PROBE_TIMEOUT_SECONDS = float(
    os.environ.get("COLAB_CODEX_KERNEL_PROBE_TIMEOUT_SECONDS", 30)
)
if (
    not math.isfinite(DEFAULT_KERNEL_PROBE_TIMEOUT_SECONDS)
    or not 1 <= DEFAULT_KERNEL_PROBE_TIMEOUT_SECONDS <= 300
):
    raise ValueError(
        "COLAB_CODEX_KERNEL_PROBE_TIMEOUT_SECONDS must be between 1 and 300"
    )
RECONCILIATION_BATCH_SIZE = 64
_JOB_SENTINEL = "__COLAB_CODEX_JOB__"
_RECONCILE_SENTINEL = "__COLAB_CODEX_RECONCILE__"
_ARTIFACT_SENTINEL = "__COLAB_CODEX_ARTIFACT__"
_KERNEL_PROBE_SENTINEL = "__COLAB_CODEX_KERNEL_PROBE__"
_PROBE_CELL_MARKER = "# colab-codex-managed-probe"
_UNKNOWN_JOB_MESSAGE = "Unknown Colab job id"
_MISSING = object()
_TERMINAL_STATES = {
    "finished",
    "error",
    "timed_out",
    "missing",
    "stale",
    "interrupted",
}


def validate_submitted_code(code: str) -> None:
    if len(code.encode("utf-8")) > MAX_SUBMITTED_CODE_BYTES:
        raise ValueError(
            "Submitted code exceeds the configured connector size limit"
        )


def result_data(result: Any) -> dict[str, Any]:
    data = getattr(result, "structured_content", None)
    if isinstance(data, dict):
        return data
    data = getattr(result, "structuredContent", None)
    if isinstance(data, dict):
        return data
    data = getattr(result, "data", None)
    if isinstance(data, dict):
        return data
    data = first_json_object(result)
    return data if isinstance(data, dict) else {}


def cell_outputs(cell: dict[str, Any]) -> list[Any]:
    outputs = cell.get("outputs", [])
    return outputs if isinstance(outputs, list) else []


def output_has_error(outputs: list[Any]) -> bool:
    return any(
        isinstance(output, dict) and output.get("output_type") == "error"
        for output in outputs
    )


def output_error(outputs: list[Any]) -> str | None:
    for output in outputs:
        if not isinstance(output, dict) or output.get("output_type") != "error":
            continue
        ename = output.get("ename")
        evalue = output.get("evalue")
        if ename and evalue:
            return f"{ename}: {evalue}"
        if ename:
            return str(ename)
        if evalue:
            return str(evalue)
        return "Cell execution returned an error output"
    return None


def _safe_error(value: Any, max_bytes: int = 4096) -> str | None:
    if value is None:
        return None
    encoded = str(value).encode("utf-8", errors="replace")
    if len(encoded) > max_bytes:
        encoded = encoded[: max_bytes - 3] + b"..."
    return encoded.decode("utf-8", errors="replace")


def _output_text(outputs: list[Any]) -> str:
    parts: list[str] = []
    for output in outputs:
        if not isinstance(output, dict):
            parts.append(str(output))
            continue
        output_type = output.get("output_type")
        if output_type == "stream":
            text = output.get("text", "")
            parts.append("".join(str(item) for item in text) if isinstance(text, list) else str(text))
        elif output_type == "error":
            traceback_value = output.get("traceback")
            if isinstance(traceback_value, list):
                parts.append("\n".join(str(item) for item in traceback_value))
            else:
                parts.append(output_error([output]) or "Cell execution failed")
        else:
            data = output.get("data")
            if isinstance(data, dict):
                plain = data.get("text/plain")
                if plain is not None:
                    parts.append(
                        "".join(str(item) for item in plain)
                        if isinstance(plain, list)
                        else str(plain)
                    )
                omitted = [key for key in data if key != "text/plain"]
                if omitted:
                    parts.append(
                        "[binary/rich display omitted: " + ", ".join(sorted(omitted)) + "]"
                    )
    return "".join(parts)


def _excerpt_outputs(outputs: list[Any], max_bytes: int) -> list[Any]:
    raw = json_bytes(outputs)
    if len(raw) <= max_bytes:
        return outputs
    text = _output_text(outputs).encode("utf-8", errors="replace")
    budget = max(0, max_bytes - 160)
    excerpt = text[:budget].decode("utf-8", errors="ignore")
    if len(text) > budget:
        excerpt += "..."
    return [
        {
            "output_type": "stream",
            "name": "stdout",
            "text": [excerpt],
        }
    ]


def _is_transport_disconnect(exc: BaseException) -> bool:
    if isinstance(exc, (NotConnectedError, ConnectionError, EOFError)):
        return True
    if getattr(exc, "transport_disconnected", False):
        return True
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    return any(term in name for term in ("connectionclosed", "brokenresource", "endofstream")) or any(
        term in message
        for term in (
            "connection closed",
            "websocket closed",
            "transport closed",
            "browser disconnected",
        )
    )


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value
    )


@dataclass
class ColabJob:
    job_id: str
    cell_id: str
    cell_index: int
    code_bytes: int
    code_sha256: str
    state: str
    tracking_state: str
    started_at: float
    updated_at: float
    execution_timeout_seconds: float
    execution_alive: bool | None = True
    finished_at: float | None = None
    last_output_at: float | None = None
    outputs: list[Any] = field(default_factory=list)
    output_bytes: int = 0
    output_excerpt_bytes: int = 0
    output_truncated: bool = False
    output_artifact: dict[str, Any] | None = None
    output_unavailable_reason: str | None = None
    error: str | None = None
    tracking_error: str | None = None
    remote_response_bytes: int = 0
    remote_output_count: int | None = None
    captured_runtime_output_bytes: int | None = None
    terminal_manifest_found: bool = False
    completion_source: str | None = None
    runtime_marker_path: str = field(default="", repr=False)
    runtime_output_path: str = field(default="", repr=False)
    runtime_artifact_id: str = field(default="", repr=False)


class ColabJobManager:
    def __init__(
        self,
        session: ColabSessionManager,
        *,
        artifact_store: ArtifactStore | None = None,
        output_excerpt_bytes: int = DEFAULT_OUTPUT_EXCERPT_BYTES,
        runtime_artifact_bytes: int = DEFAULT_RUNTIME_ARTIFACT_BYTES,
        runtime_artifact_total_bytes: int = DEFAULT_MAX_ARTIFACT_TOTAL_BYTES,
        runtime_artifact_ttl_seconds: float = DEFAULT_ARTIFACT_TTL_SECONDS,
        runtime_root: str = DEFAULT_COLAB_RUNTIME_ROOT,
        max_tracked_jobs: int = DEFAULT_MAX_TRACKED_JOBS,
        max_job_cells: int = DEFAULT_MAX_JOB_CELLS,
        journal_path: Path | str | None = DEFAULT_JOB_JOURNAL_PATH,
    ) -> None:
        if (
            isinstance(output_excerpt_bytes, bool)
            or not isinstance(output_excerpt_bytes, int)
            or output_excerpt_bytes < 1024
        ):
            raise ValueError("output_excerpt_bytes must be at least 1024")
        if (
            isinstance(runtime_artifact_bytes, bool)
            or not isinstance(runtime_artifact_bytes, int)
            or runtime_artifact_bytes < output_excerpt_bytes
        ):
            raise ValueError(
                "runtime_artifact_bytes must be at least output_excerpt_bytes"
            )
        if (
            isinstance(runtime_artifact_total_bytes, bool)
            or not isinstance(runtime_artifact_total_bytes, int)
            or runtime_artifact_total_bytes < runtime_artifact_bytes
            or runtime_artifact_total_bytes < RUNTIME_MARKER_RESERVE_BYTES
        ):
            raise ValueError(
                "runtime_artifact_total_bytes must be greater than or equal to "
                "runtime_artifact_bytes and the runtime marker reserve"
            )
        if (
            isinstance(runtime_artifact_ttl_seconds, bool)
            or not isinstance(runtime_artifact_ttl_seconds, int | float)
            or not math.isfinite(runtime_artifact_ttl_seconds)
            or runtime_artifact_ttl_seconds <= 0
        ):
            raise ValueError(
                "runtime_artifact_ttl_seconds must be finite and positive"
            )
        if (
            isinstance(max_tracked_jobs, bool)
            or not isinstance(max_tracked_jobs, int)
            or max_tracked_jobs < 1
            or max_tracked_jobs > MAX_JOURNAL_JOBS
        ):
            raise ValueError(
                f"max_tracked_jobs must be between 1 and {MAX_JOURNAL_JOBS}"
            )
        if (
            isinstance(max_job_cells, bool)
            or not isinstance(max_job_cells, int)
            or max_job_cells < 1
            or max_job_cells > MAX_JOURNAL_JOBS
        ):
            raise ValueError(
                f"max_job_cells must be between 1 and {MAX_JOURNAL_JOBS}"
            )
        if not runtime_root.startswith("/") or "\x00" in runtime_root:
            raise ValueError("runtime_root must be an absolute runtime path")
        self.session = session
        self.artifact_store = artifact_store or get_default_artifact_store()
        self.output_excerpt_bytes = output_excerpt_bytes
        self.runtime_artifact_bytes = runtime_artifact_bytes
        self.runtime_artifact_total_bytes = runtime_artifact_total_bytes
        self.runtime_artifact_ttl_seconds = runtime_artifact_ttl_seconds
        self.runtime_root = runtime_root.rstrip("/")
        self.max_tracked_jobs = max_tracked_jobs
        self.max_job_cells = min(max_job_cells, max_tracked_jobs)
        self.jobs: dict[str, ColabJob] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._completion_events: dict[str, asyncio.Event] = {}
        self._runtime_artifacts: dict[str, str] = {}
        self._job_cells: dict[str, int] = {}
        self._pending_job_cell_sha256: str | None = None
        self._probe_cell_id: str | None = None
        self._pending_probe_cell_sha256: str | None = None
        self._probe_lock = asyncio.Lock()
        self._reconciliation_attempts: dict[str, object] = {}
        self._deferred_task_persistence: set[str] = set()
        self._lock = asyncio.Lock()
        self._kernel_execution_ready: bool | None = None
        self._kernel_probe_at: float | None = None
        self._kernel_probe_latency_ms: float | None = None
        self._kernel_probe_error: str | None = None
        # Lightweight fake sessions used by callers/tests should not unexpectedly
        # share the production journal unless an explicit non-default path is set.
        if (
            journal_path is not None
            and Path(journal_path) == DEFAULT_JOB_JOURNAL_PATH
            and not isinstance(session, ColabSessionManager)
        ):
            journal_path = None
        self.journal_path = Path(journal_path) if journal_path is not None else None
        self._load_journal()

    def _session_connection_id(self) -> str | None:
        value = getattr(self.session, "connection_id", None)
        return value if isinstance(value, str) and value else None

    def _journal_record(self, job: ColabJob) -> dict[str, Any]:
        artifact = job.output_artifact
        if isinstance(artifact, dict):
            artifact = {
                key: artifact.get(key)
                for key in (
                    "artifact_id",
                    "storage",
                    "media_type",
                    "size_bytes",
                    "sha256",
                    "created_at",
                    "expires_at",
                    "truncated",
                )
            }
        return {
            "job_id": job.job_id,
            "cell_id": job.cell_id,
            "cell_index": job.cell_index,
            "code_bytes": job.code_bytes,
            "code_sha256": job.code_sha256,
            "state": job.state,
            "tracking_state": job.tracking_state,
            "started_at": job.started_at,
            "updated_at": job.updated_at,
            "execution_timeout_seconds": job.execution_timeout_seconds,
            "finished_at": job.finished_at,
            "last_output_at": job.last_output_at,
            "output_bytes": job.output_bytes,
            "output_excerpt_bytes": job.output_excerpt_bytes,
            "output_truncated": job.output_truncated,
            "output_artifact": artifact,
            "remote_response_bytes": job.remote_response_bytes,
            "remote_output_count": job.remote_output_count,
            "captured_runtime_output_bytes": job.captured_runtime_output_bytes,
            "terminal_manifest_found": job.terminal_manifest_found,
            "completion_source": job.completion_source,
            "runtime_artifact_id": job.runtime_artifact_id,
        }

    def _persist_journal(self) -> None:
        if self.journal_path is None:
            return
        path = self.journal_path
        data = {
            "version": 3,
            "updated_at": time.time(),
            "connection_id": self._session_connection_id(),
            "probe_cell_id": self._probe_cell_id,
            "pending_job_cell_sha256": self._pending_job_cell_sha256,
            "pending_probe_cell_sha256": self._pending_probe_cell_sha256,
            "job_cells": [
                {"cell_id": cell_id, "cell_index": cell_index}
                for cell_id, cell_index in self._job_cells.items()
            ],
            "jobs": [
                self._journal_record(job)
                for job in list(self.jobs.values())[-MAX_JOURNAL_JOBS:]
            ],
        }
        try:
            path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            os.chmod(path.parent, 0o700)
            descriptor, temporary_name = tempfile.mkstemp(
                dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
            )
            temporary_path = Path(temporary_name)
            try:
                os.fchmod(descriptor, 0o600)
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    json.dump(data, handle, ensure_ascii=True, separators=(",", ":"))
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary_path, path)
                os.chmod(path, 0o600)
            except BaseException:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
                temporary_path.unlink(missing_ok=True)
                raise
        except OSError as exc:
            logging.warning(
                "Failed to persist Colab job metadata error_type=%s", type(exc).__name__
            )

    def _load_journal(self) -> None:
        if self.journal_path is None or not self.journal_path.exists():
            return
        try:
            raw = json.loads(self.journal_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            logging.warning(
                "Ignoring invalid Colab job journal error_type=%s", type(exc).__name__
            )
            return
        records = raw.get("jobs") if isinstance(raw, dict) else None
        if not isinstance(records, list):
            return
        journal_needs_migration = raw.get("version") != 3
        for record in records[-MAX_JOURNAL_JOBS:]:
            job = self._restore_record(record)
            if job is None:
                continue
            self.jobs[job.job_id] = job
            event = asyncio.Event()
            event.set()
            self._completion_events[job.job_id] = event
            if job.output_artifact and job.output_artifact.get("storage") == "colab_runtime":
                self._runtime_artifacts[job.runtime_artifact_id] = job.runtime_output_path
            if job.state == "interrupted" and record.get("state") == "finished":
                journal_needs_migration = True
        journal_connection_id = raw.get("connection_id")
        if (
            raw.get("version") != 3
            or not isinstance(journal_connection_id, str)
            or journal_connection_id != self._session_connection_id()
        ):
            # Historical jobs are portable broker metadata. Notebook cell ids are
            # not, so legacy or differently scoped journals must not restore them.
            if journal_needs_migration:
                self._persist_journal()
            return
        probe_cell_id = raw.get("probe_cell_id")
        if isinstance(probe_cell_id, str) and 1 <= len(probe_cell_id) <= 256:
            self._probe_cell_id = probe_cell_id
        pending_job_cell_sha256 = raw.get("pending_job_cell_sha256")
        if _valid_sha256(pending_job_cell_sha256):
            self._pending_job_cell_sha256 = pending_job_cell_sha256
        pending_probe_cell_sha256 = raw.get("pending_probe_cell_sha256")
        if _valid_sha256(pending_probe_cell_sha256):
            self._pending_probe_cell_sha256 = pending_probe_cell_sha256
        pool_records = raw.get("job_cells")
        if isinstance(pool_records, list):
            for pool_record in pool_records[-self.max_job_cells :]:
                if not isinstance(pool_record, dict):
                    continue
                cell_id = pool_record.get("cell_id")
                cell_index = pool_record.get("cell_index")
                if (
                    isinstance(cell_id, str)
                    and 1 <= len(cell_id) <= 256
                    and isinstance(cell_index, int)
                    and not isinstance(cell_index, bool)
                    and cell_index >= 0
                ):
                    self._job_cells[cell_id] = cell_index

    def _restore_record(self, record: Any) -> ColabJob | None:
        if not isinstance(record, dict):
            return None
        job_id = record.get("job_id")
        cell_id = record.get("cell_id")
        runtime_artifact_id = record.get("runtime_artifact_id")
        code_sha256 = record.get("code_sha256")
        if (
            not valid_opaque_id(job_id)
            or not valid_opaque_id(runtime_artifact_id)
            or not _valid_sha256(code_sha256)
            or not isinstance(cell_id, str)
        ):
            return None
        try:
            state = str(record.get("state", "running"))
            unproven_finished = state == "finished" and not (
                record.get("terminal_manifest_found") is True
                and record.get("completion_source") == "terminal_manifest"
            )
            if unproven_finished:
                state = "interrupted"
            recoverable = state not in _TERMINAL_STATES or (
                state == "timed_out"
                and record.get("tracking_state") == "detached"
            )
            job = ColabJob(
                job_id=job_id,
                cell_id=cell_id,
                cell_index=int(record.get("cell_index", 0)),
                code_bytes=max(0, int(record.get("code_bytes", 0))),
                code_sha256=code_sha256,
                state=state,
                tracking_state="detached" if recoverable else "complete",
                started_at=float(record.get("started_at", 0)),
                updated_at=float(record.get("updated_at", 0)),
                execution_timeout_seconds=float(
                    record.get(
                        "execution_timeout_seconds",
                        DEFAULT_EXECUTION_TIMEOUT_SECONDS,
                    )
                ),
                execution_alive=None if recoverable else False,
                finished_at=(
                    float(record["finished_at"])
                    if record.get("finished_at") is not None
                    else None
                ),
                last_output_at=(
                    float(record["last_output_at"])
                    if record.get("last_output_at") is not None
                    else None
                ),
                output_bytes=max(0, int(record.get("output_bytes", 0))),
                output_excerpt_bytes=0,
                output_truncated=bool(record.get("output_truncated")),
                output_artifact=(
                    record.get("output_artifact")
                    if isinstance(record.get("output_artifact"), dict)
                    else None
                ),
                output_unavailable_reason=(
                    "Legacy job completion lacked terminal-manifest proof"
                    if unproven_finished
                    else (
                        "Output excerpt is unavailable after broker recovery"
                        if int(record.get("output_excerpt_bytes", 0)) > 0
                        else None
                    )
                ),
                tracking_error=(
                    "Broker owner changed; awaiting runtime reconciliation"
                    if recoverable
                    else None
                ),
                remote_response_bytes=max(
                    0, int(record.get("remote_response_bytes", 0))
                ),
                remote_output_count=(
                    max(0, int(record["remote_output_count"]))
                    if record.get("remote_output_count") is not None
                    else None
                ),
                captured_runtime_output_bytes=(
                    max(0, int(record["captured_runtime_output_bytes"]))
                    if record.get("captured_runtime_output_bytes") is not None
                    else None
                ),
                terminal_manifest_found=bool(
                    record.get("terminal_manifest_found", False)
                ),
                completion_source=(
                    str(record["completion_source"])
                    if record.get("completion_source") is not None
                    else None
                ),
                runtime_marker_path=f"{self.runtime_root}/{job_id}.json",
                runtime_output_path=f"{self.runtime_root}/{job_id}.output",
                runtime_artifact_id=runtime_artifact_id,
            )
        except (TypeError, ValueError, OverflowError):
            return None
        if (
            not math.isfinite(job.execution_timeout_seconds)
            or not 0 < job.execution_timeout_seconds <= MAX_EXECUTION_TIMEOUT_SECONDS
            or not math.isfinite(job.started_at)
            or not math.isfinite(job.updated_at)
        ):
            return None
        return job

    def _get_job(self, job_id: object) -> ColabJob:
        if not valid_opaque_id(job_id):
            raise ValueError(_UNKNOWN_JOB_MESSAGE)
        job = self.jobs.get(job_id)
        if job is None:
            raise ValueError(_UNKNOWN_JOB_MESSAGE)
        return job

    def _owns_reconciliation(self, job: ColabJob, attempt: object) -> bool:
        return (
            self.jobs.get(job.job_id) is job
            and self._reconciliation_attempts.get(job.job_id) is attempt
            and job.state in {"running", "timed_out"}
            and job.tracking_state == "recovering"
        )

    def _job_dict(
        self, job: ColabJob, *, include_outputs: bool = True
    ) -> dict[str, Any]:
        task = self._tasks.get(job.job_id)
        data: dict[str, Any] = {
            "job_id": job.job_id,
            "cell_id": job.cell_id,
            "cell_index": job.cell_index,
            "code_bytes": job.code_bytes,
            "code_sha256": job.code_sha256,
            "state": job.state,
            "tracking_state": job.tracking_state,
            "started_at": job.started_at,
            "updated_at": job.updated_at,
            "execution_timeout_seconds": job.execution_timeout_seconds,
            "execution_alive": job.execution_alive,
            "task_alive": (
                job.state == "running" and task is not None and not task.done()
            ),
            "finished_at": job.finished_at,
            "last_output_at": job.last_output_at,
            "output_bytes": job.output_bytes,
            "output_excerpt_bytes": job.output_excerpt_bytes,
            "output_truncated": job.output_truncated,
            "output_artifact": job.output_artifact,
            "output_unavailable_reason": job.output_unavailable_reason,
            "error": job.error,
            "tracking_error": job.tracking_error,
            "remote_response_bytes": job.remote_response_bytes,
            "remote_output_count": job.remote_output_count,
            "captured_runtime_output_bytes": job.captured_runtime_output_bytes,
            "terminal_manifest_found": job.terminal_manifest_found,
            "completion_source": job.completion_source,
        }
        if include_outputs:
            data["outputs"] = job.outputs
        return data

    async def _remote_tool_names(self) -> set[str]:
        return {tool.name for tool in await self.session.list_tools()}

    async def _get_cells(
        self, *, start: int | None = None, end: int | None = None
    ) -> list[dict[str, Any]]:
        arguments: dict[str, Any] = {"includeOutputs": False}
        if start is not None:
            arguments["cellIndexStart"] = start
        if end is not None:
            if start is not None and end <= start:
                raise ValueError("cell metadata end must be greater than start")
            # The frontend schema uses an inclusive end. Connector internals keep
            # Python's conventional exclusive end so page-size accounting is clear.
            arguments["cellIndexEnd"] = end - 1
        result = await self.session.call_tool("get_cells", arguments)
        cells = result_data(result).get("cells", [])
        if not isinstance(cells, list):
            return []
        if start is not None and end is not None and len(cells) > end - start:
            raise RuntimeError(
                "Colab frontend did not honor the bounded cell metadata range"
            )
        return cells

    async def _cell_pages(
        self,
    ) -> AsyncIterator[tuple[int, list[dict[str, Any]]]]:
        offset = 0
        seen_ids: set[str] = set()
        while True:
            cells = await self._get_cells(
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

    async def _cell_count(self) -> int:
        count = 0
        async for offset, cells in self._cell_pages():
            count = offset + len(cells)
        return count

    async def _find_cell(
        self, cell_id: str, cell_index: int | None = None
    ) -> tuple[dict[str, Any], int] | None:
        if cell_index is not None and cell_index >= 0:
            cells = await self._get_cells(start=cell_index, end=cell_index + 1)
            if cells and cells[0].get("id") == cell_id:
                return cells[0], cell_index
        async for offset, cells in self._cell_pages():
            for relative_index, cell in enumerate(cells):
                if cell.get("id") == cell_id:
                    return cell, offset + relative_index
        return None

    async def _find_cell_index(self, cell_id: str) -> int | None:
        found = await self._find_cell(cell_id)
        return found[1] if found is not None else None

    @classmethod
    def _is_owned_job_cell(cls, cell: dict[str, Any]) -> bool:
        source = cls._cell_source_text(cell)
        return bool(
            source
            and "def _cc_run_connector_job():" in source
            and _JOB_SENTINEL in source
        )

    @classmethod
    def _is_owned_probe_cell(cls, cell: dict[str, Any]) -> bool:
        source = cls._cell_source_text(cell)
        return bool(source and source.startswith(_PROBE_CELL_MARKER + "\n"))

    async def _next_reusable_job_cell(
        self, protected_cell_ids: set[str]
    ) -> tuple[str, int] | None:
        changed = False
        for cell_id, cell_index in list(self._job_cells.items()):
            if cell_id in protected_cell_ids:
                continue
            found = await self._find_cell(cell_id, cell_index)
            if found is None or not self._is_owned_job_cell(found[0]):
                self._job_cells.pop(cell_id, None)
                changed = True
                continue
            current_index = found[1]
            if current_index != cell_index:
                self._job_cells[cell_id] = current_index
                changed = True
            if changed:
                self._persist_journal()
            return cell_id, current_index
        if changed:
            self._persist_journal()
        return None

    @staticmethod
    def _cell_source_text(cell: dict[str, Any]) -> str | None:
        source = cell.get("source", cell.get("content", cell.get("code")))
        if isinstance(source, list):
            source = "".join(str(part) for part in source)
        return source if isinstance(source, str) else None

    async def _find_cell_by_source_sha256(
        self, source_sha256: str
    ) -> tuple[str, int] | None:
        async for offset, cells in self._cell_pages():
            for relative_index, cell in enumerate(cells):
                source = self._cell_source_text(cell)
                cell_id = cell.get("id")
                if (
                    source is not None
                    and isinstance(cell_id, str)
                    and hashlib.sha256(source.encode("utf-8")).hexdigest()
                    == source_sha256
                ):
                    return cell_id, offset + relative_index
        return None

    async def _recover_pending_job_cell(
        self, *, clear_if_missing: bool = True
    ) -> None:
        source_sha256 = self._pending_job_cell_sha256
        if source_sha256 is None:
            return
        recovered = await self._find_cell_by_source_sha256(source_sha256)
        if recovered is not None:
            cell_id, cell_index = recovered
            self._job_cells[cell_id] = cell_index
        if recovered is not None or clear_if_missing:
            self._pending_job_cell_sha256 = None
            self._persist_journal()

    async def _recover_pending_probe_cell(
        self, *, clear_if_missing: bool = True
    ) -> None:
        source_sha256 = self._pending_probe_cell_sha256
        if source_sha256 is None:
            return
        recovered = await self._find_cell_by_source_sha256(source_sha256)
        if recovered is not None:
            self._probe_cell_id = recovered[0]
        if recovered is not None or clear_if_missing:
            self._pending_probe_cell_sha256 = None
            self._persist_journal()

    async def _job_cell_exists(self, job: ColabJob) -> bool:
        cells = await self._get_cells(start=job.cell_index, end=job.cell_index + 1)
        if any(cell.get("id") == job.cell_id for cell in cells):
            return True
        # Cells may be inserted or reordered while a job runs. Fall back to an
        # output-free paged ID scan before declaring the stable cell id missing.
        index = await self._find_cell_index(job.cell_id)
        if index is None:
            # External notebook edits can move a cell behind the page cursor.
            # Retry once before making the job irreversibly missing.
            index = await self._find_cell_index(job.cell_id)
        if index is not None:
            job.cell_index = index
            if job.cell_id in self._job_cells:
                self._job_cells[job.cell_id] = index
            job.updated_at = time.time()
            self._persist_journal()
            return True
        return False

    def _runtime_wrapper(self, job: ColabJob, code: str) -> str:
        encoded_code = base64.b64encode(code.encode("utf-8")).decode("ascii")
        values = {
            "job_id": job.job_id,
            "artifact_id": job.runtime_artifact_id,
            "marker_path": job.runtime_marker_path,
            "output_path": job.runtime_output_path,
            "code_b64": encoded_code,
            "artifact_limit": self.runtime_artifact_bytes,
            "artifact_total_limit": self.runtime_artifact_total_bytes,
            "artifact_ttl_seconds": self.runtime_artifact_ttl_seconds,
            "artifact_max_pairs": self.max_tracked_jobs,
            "artifact_marker_reserve": RUNTIME_MARKER_RESERVE_BYTES,
            "excerpt_limit": self.output_excerpt_bytes,
            "sentinel": _JOB_SENTINEL,
        }
        return _RUNTIME_WRAPPER_TEMPLATE.replace(
            "__COLAB_JOB_CONFIG__",
            json.dumps(values, ensure_ascii=True, separators=(",", ":")),
        )

    def _prune_completed_jobs(self, *, reserve: int = 0) -> None:
        target = max(0, self.max_tracked_jobs - reserve)
        if len(self.jobs) <= target:
            return
        completed = sorted(
            (
                job
                for job in self.jobs.values()
                if job.tracking_state == "complete"
                and job.state in _TERMINAL_STATES
            ),
            key=lambda job: (job.finished_at or job.updated_at, job.started_at),
        )
        for job in completed:
            if len(self.jobs) <= target:
                break
            self.jobs.pop(job.job_id, None)
            self._completion_events.pop(job.job_id, None)
            self._tasks.pop(job.job_id, None)
            if job.runtime_artifact_id:
                self._runtime_artifacts.pop(job.runtime_artifact_id, None)

    async def start_python(
        self,
        code: str,
        language: str = "python",
        execution_timeout_seconds: float = DEFAULT_EXECUTION_TIMEOUT_SECONDS,
    ) -> dict[str, Any]:
        if language.lower() not in {"python", "py", "python3"}:
            raise ValueError("Tracked Colab jobs support CPython only")
        validate_submitted_code(code)
        if (
            not math.isfinite(execution_timeout_seconds)
            or not 0 < execution_timeout_seconds <= MAX_EXECUTION_TIMEOUT_SECONDS
        ):
            raise ValueError(
                "execution_timeout_seconds must be greater than zero and no more "
                f"than {MAX_EXECUTION_TIMEOUT_SECONDS:g}"
            )
        names = await self._remote_tool_names()
        required = {"add_code_cell", "run_code_cell", "get_cells", "update_cell"}
        if not required.issubset(names):
            missing = ", ".join(sorted(required - names))
            raise ValueError(f"Colab remote tools missing for jobs: {missing}")

        job_id = uuid.uuid4().hex
        started_at = time.time()
        code_raw = code.encode("utf-8")
        async with self._lock:
            await self._recover_pending_job_cell()
            self._prune_completed_jobs(reserve=1)
            if len(self.jobs) >= self.max_tracked_jobs:
                raise RuntimeError(
                    "Maximum tracked Colab jobs reached; wait for a running job "
                    "to finish before starting another"
                )
            protected_cell_ids = {
                existing.cell_id
                for existing in self.jobs.values()
                if existing.state in {"running", "timed_out"}
                and existing.tracking_state != "complete"
            }
            reusable = await self._next_reusable_job_cell(protected_cell_ids)
            if reusable is None:
                if len(self._job_cells) >= self.max_job_cells:
                    raise RuntimeError(
                        "Maximum concurrent Colab job cells reached; wait for an "
                        "unfinished job to become terminal"
                    )
                cell_index = await self._cell_count()
                cell_id = ""
            else:
                cell_id, cell_index = reusable
            job = ColabJob(
                job_id=job_id,
                cell_id=cell_id,
                cell_index=cell_index,
                code_bytes=len(code_raw),
                code_sha256=hashlib.sha256(code_raw).hexdigest(),
                state="running",
                tracking_state="active",
                started_at=started_at,
                updated_at=started_at,
                execution_timeout_seconds=execution_timeout_seconds,
                runtime_marker_path=f"{self.runtime_root}/{job_id}.json",
                runtime_output_path=f"{self.runtime_root}/{job_id}.output",
                runtime_artifact_id=uuid.uuid4().hex,
            )
            wrapped_code = self._runtime_wrapper(job, code)
            if job.cell_id:
                try:
                    update_result = await self.session.call_tool(
                        "update_cell",
                        {"cellId": job.cell_id, "content": wrapped_code},
                    )
                    updated_cell_id = result_data(update_result).get("cellId")
                    if updated_cell_id is not None and updated_cell_id != job.cell_id:
                        raise RuntimeError(
                            "Colab updated a different cell than the requested job cell"
                        )
                except Exception:
                    try:
                        current_index = await self._find_cell_index(job.cell_id)
                    except Exception:
                        pass
                    else:
                        if current_index is None:
                            self._job_cells.pop(job.cell_id, None)
                            self._persist_journal()
                        else:
                            self._job_cells[job.cell_id] = current_index
                    raise
            else:
                self._pending_job_cell_sha256 = hashlib.sha256(
                    wrapped_code.encode("utf-8")
                ).hexdigest()
                self._persist_journal()
                try:
                    add_result = await self.session.call_tool(
                        "add_code_cell",
                        {
                            "cellIndex": job.cell_index,
                            "language": "python",
                            "code": wrapped_code,
                        },
                    )
                    cell_id = result_data(add_result).get("newCellId")
                    if not isinstance(cell_id, str):
                        raise ValueError(
                            "Colab did not return a newCellId from add_code_cell"
                        )
                except Exception:
                    try:
                        await self._recover_pending_job_cell(clear_if_missing=False)
                    except Exception:
                        pass
                    raise
                job.cell_id = cell_id
                self._job_cells[cell_id] = job.cell_index
                self._pending_job_cell_sha256 = None
            self.jobs[job.job_id] = job
            self._completion_events[job.job_id] = asyncio.Event()
            self._tasks[job.job_id] = asyncio.create_task(
                self._execute(job), name=f"colab-job-{job.job_id}"
            )
            self._persist_journal()
        return self._job_dict(job)

    async def _execute(self, job: ColabJob) -> None:
        try:
            run_result = await self.session.call_tool(
                "run_code_cell",
                {"cellId": job.cell_id},
                timeout=job.execution_timeout_seconds,
            )
            data = result_data(run_result)
            job.remote_response_bytes = len(json_bytes(data))
            outputs = data.get("outputs", _MISSING)
            if job.state == "running":
                if not isinstance(outputs, list):
                    self._detach_job(
                        job,
                        "Malformed run_code_cell response: outputs must be a list",
                        persist=False,
                    )
                    return
                job.remote_output_count = len(outputs)
                manifest = self._manifest_from_outputs(job, outputs)
                if manifest is not None:
                    self._finish_from_manifest(job, manifest, persist=False)
                elif output_has_error(outputs):
                    self._finish_from_outputs(job, outputs, persist=False)
                else:
                    self._detach_job(
                        job,
                        "run_code_cell returned no matching terminal manifest",
                        persist=False,
                    )
        except asyncio.TimeoutError:
            if job.state == "running":
                job.state = "timed_out"
                job.tracking_state = "detached"
                job.execution_alive = None
                job.error = (
                    "Colab execution exceeded "
                    f"{job.execution_timeout_seconds:g} seconds; execution state is unknown"
                )
                job.finished_at = job.updated_at = time.time()
        except asyncio.CancelledError:
            if job.state == "running" and job.tracking_state != "detached":
                self._detach_job(
                    job,
                    "Colab execution tracking was cancelled",
                    persist=False,
                )
            raise
        except Exception as exc:
            if job.state == "running":
                # A raised run_code_cell request doesn't prove whether execution
                # started or finished. Only a returned manifest/output can safely
                # make this job terminal; otherwise reconcile it after reconnect.
                self._detach_job(
                    job,
                    "Colab execution response was lost "
                    f"({type(exc).__name__})",
                    persist=False,
                )
        finally:
            self._tasks.pop(job.job_id, None)
            event = self._completion_events.get(job.job_id)
            if event is not None:
                event.set()
            if job.job_id not in self._deferred_task_persistence:
                self._persist_journal()

    def _manifest_from_outputs(
        self, job: ColabJob, outputs: list[Any]
    ) -> dict[str, Any] | None:
        for output in reversed(outputs):
            if not isinstance(output, dict):
                continue
            text = output.get("text")
            values = text if isinstance(text, list) else [text]
            combined = "".join(str(value) for value in values if value is not None)
            for line in reversed(combined.splitlines()):
                if not line.startswith(_JOB_SENTINEL):
                    continue
                try:
                    manifest = json.loads(line[len(_JOB_SENTINEL) :])
                except json.JSONDecodeError:
                    continue
                if isinstance(manifest, dict) and manifest.get("job_id") == job.job_id:
                    return manifest
        return None

    def _finish_from_manifest(
        self,
        job: ColabJob,
        manifest: dict[str, Any],
        *,
        persist: bool = True,
    ) -> None:
        now = time.time()
        state = manifest.get("state")
        job.state = state if state in {"finished", "error"} else "error"
        job.tracking_state = "complete"
        job.execution_alive = False
        job.terminal_manifest_found = True
        job.completion_source = "terminal_manifest"
        job.error = _safe_error(manifest.get("error"))
        excerpt = manifest.get("output_excerpt", "")
        if not isinstance(excerpt, str):
            excerpt = str(excerpt)
        excerpt_raw = excerpt.encode("utf-8", errors="replace")[: self.output_excerpt_bytes]
        excerpt = excerpt_raw.decode("utf-8", errors="ignore")
        job.outputs = (
            [{"output_type": "stream", "name": "stdout", "text": [excerpt]}]
            if excerpt
            else []
        )
        job.output_bytes = max(0, int(manifest.get("output_bytes", len(excerpt_raw))))
        job.captured_runtime_output_bytes = job.output_bytes
        job.output_excerpt_bytes = len(json_bytes(job.outputs))
        artifact_size = max(0, int(manifest.get("artifact_size_bytes", 0)))
        truncated = bool(manifest.get("output_truncated")) or job.output_bytes > len(
            excerpt_raw
        )
        job.output_truncated = truncated
        if truncated and artifact_size:
            artifact_id = job.runtime_artifact_id
            self._runtime_artifacts[artifact_id] = job.runtime_output_path
            job.output_artifact = ArtifactRef(
                artifact_id=artifact_id,
                storage="colab_runtime",
                media_type="text/plain; charset=utf-8",
                size_bytes=artifact_size,
                sha256=str(manifest.get("artifact_sha256", "")),
                created_at=job.started_at,
                expires_at=now + self.runtime_artifact_ttl_seconds,
                truncated=bool(manifest.get("artifact_truncated")),
            ).to_dict()
        job.last_output_at = now if job.output_bytes else None
        job.finished_at = job.updated_at = now
        if persist:
            self._persist_journal()

    def _finish_from_outputs(
        self,
        job: ColabJob,
        outputs: list[Any],
        *,
        persist: bool = True,
    ) -> None:
        if not output_has_error(outputs):
            raise ValueError(
                "Non-manifest execution output cannot prove tracked job completion"
            )
        raw = json_bytes(outputs)
        job.output_bytes = len(raw)
        job.error = output_error(outputs)
        job.state = "error"
        job.tracking_state = "complete"
        job.execution_alive = False
        job.completion_source = "explicit_cell_error"
        job.outputs = _excerpt_outputs(outputs, self.output_excerpt_bytes)
        job.output_excerpt_bytes = len(json_bytes(job.outputs))
        job.output_truncated = len(raw) > self.output_excerpt_bytes
        if job.output_truncated:
            job.output_artifact = self.artifact_store.put_bytes(
                raw, media_type="application/json; charset=utf-8"
            ).to_dict()
        now = time.time()
        job.last_output_at = now if outputs else None
        job.finished_at = job.updated_at = now
        if persist:
            self._persist_journal()

    def _detach_job(
        self, job: ColabJob, reason: str, *, persist: bool = True
    ) -> None:
        job.tracking_state = "detached"
        job.execution_alive = None
        job.tracking_error = _safe_error(reason)
        job.updated_at = time.time()
        event = self._completion_events.get(job.job_id)
        if event is not None:
            event.set()
        if persist:
            self._persist_journal()

    async def status(self, job_id: str) -> dict[str, Any]:
        job = self._get_job(job_id)

        if job.state == "running" and job.tracking_state == "active":
            try:
                exists = await self._job_cell_exists(job)
            except Exception as exc:
                if (
                    _is_transport_disconnect(exc)
                    and self.jobs.get(job_id) is job
                    and job.state == "running"
                    and job.tracking_state == "active"
                ):
                    self._detach_job(
                        job,
                        "Colab cell status response was lost "
                        f"({type(exc).__name__})",
                    )
                else:
                    raise
            else:
                if (
                    not exists
                    and self.jobs.get(job_id) is job
                    and job.state == "running"
                    and job.tracking_state == "active"
                ):
                    job.state = "missing"
                    job.tracking_state = "complete"
                    job.execution_alive = False
                    job.error = "Job cell no longer exists in the notebook"
                    job.finished_at = job.updated_at = time.time()
                    task = self._tasks.get(job.job_id)
                    if task is not None:
                        task.cancel()
                    self._persist_journal()

        return self._job_dict(job)

    async def wait(
        self,
        job_id: str,
        timeout_seconds: float = DEFAULT_WAIT_TIMEOUT_SECONDS,
    ) -> dict[str, Any]:
        if not math.isfinite(timeout_seconds) or not (
            MIN_WAIT_TIMEOUT_SECONDS <= timeout_seconds <= MAX_WAIT_TIMEOUT_SECONDS
        ):
            raise ValueError(
                "timeout_seconds must be between "
                f"{MIN_WAIT_TIMEOUT_SECONDS:g} and {MAX_WAIT_TIMEOUT_SECONDS:g}"
            )
        job = self._get_job(job_id)

        wait_started = time.monotonic()
        wait_timed_out = False
        event = self._completion_events[job_id]
        if job.state == "running" and not event.is_set():
            try:
                await asyncio.wait_for(event.wait(), timeout=timeout_seconds)
            except asyncio.TimeoutError:
                wait_timed_out = True
        return {
            **self._job_dict(job),
            "timed_out": wait_timed_out,
            "wait_timed_out": wait_timed_out,
            "waited_seconds": time.monotonic() - wait_started,
        }

    async def run_python_wait(
        self,
        code: str,
        timeout_seconds: float = DEFAULT_WAIT_TIMEOUT_SECONDS,
        execution_timeout_seconds: float = DEFAULT_EXECUTION_TIMEOUT_SECONDS,
    ) -> dict[str, Any]:
        started = await self.start_python(
            code, execution_timeout_seconds=execution_timeout_seconds
        )
        return await self.wait(started["job_id"], timeout_seconds)

    def list_jobs(self) -> list[dict[str, Any]]:
        return [self._job_dict(job, include_outputs=False) for job in self.jobs.values()]

    async def on_browser_disconnect(self, reason: str) -> None:
        tasks: list[tuple[str, asyncio.Task[None]]] = []
        for job in self.jobs.values():
            if job.state != "running":
                continue
            self._detach_job(job, reason, persist=False)
            task = self._tasks.get(job.job_id)
            if task is not None and not task.done():
                tasks.append((job.job_id, task))
        await self._cancel_job_tasks(tasks)
        self._persist_journal()

    async def _cancel_job_tasks(
        self, tasks: list[tuple[str, asyncio.Task[None]]]
    ) -> None:
        if not tasks:
            return
        job_ids = {job_id for job_id, _ in tasks}
        self._deferred_task_persistence.update(job_ids)
        try:
            for _, task in tasks:
                task.cancel()
            await asyncio.gather(
                *(task for _, task in tasks), return_exceptions=True
            )
        finally:
            self._deferred_task_persistence.difference_update(job_ids)

    async def reconcile_detached(self) -> list[dict[str, Any]]:
        all_detached = [
            job
            for job in self.jobs.values()
            if job.state in {"running", "timed_out"}
            and job.tracking_state == "detached"
        ]
        now = time.time()
        detached: list[ColabJob] = []
        for job in all_detached:
            recovery_deadline = (
                job.started_at
                + job.execution_timeout_seconds
                + DEFAULT_RUNTIME_STALE_GRACE_SECONDS
            )
            if now < recovery_deadline:
                detached.append(job)
                continue
            job.state = "interrupted"
            job.tracking_state = "complete"
            job.execution_alive = False
            job.output_unavailable_reason = (
                "Runtime job did not produce a terminal marker before "
                "its recovery deadline"
            )
            job.finished_at = job.updated_at = now
            event = self._completion_events.get(job.job_id)
            if event is not None:
                event.set()
        if not detached:
            if all_detached:
                self._persist_journal()
            return [self._job_dict(job) for job in all_detached]
        attempts: dict[str, object] = {}
        for job in detached:
            attempt = object()
            attempts[job.job_id] = attempt
            self._reconciliation_attempts[job.job_id] = attempt
            job.tracking_state = "recovering"
            job.tracking_error = None
            job.updated_at = time.time()
            self._completion_events[job.job_id].clear()
        self._persist_journal()
        try:
            markers: dict[str, Any] = {}
            for start in range(0, len(detached), RECONCILIATION_BATCH_SIZE):
                batch = detached[start : start + RECONCILIATION_BATCH_SIZE]
                markers.update(await self._read_runtime_markers(batch))
        except asyncio.CancelledError:
            for job in detached:
                attempt = attempts[job.job_id]
                if self._owns_reconciliation(job, attempt):
                    self._detach_job(
                        job,
                        "Colab job reconciliation was superseded",
                        persist=False,
                    )
            self._persist_journal()
            raise
        except Exception as exc:
            for job in detached:
                attempt = attempts[job.job_id]
                if self._owns_reconciliation(job, attempt):
                    self._detach_job(
                        job,
                        "Colab job reconciliation response was lost "
                        f"({type(exc).__name__})",
                        persist=False,
                    )
            self._persist_journal()
            raise
        else:
            for job in detached:
                attempt = attempts[job.job_id]
                if not self._owns_reconciliation(job, attempt):
                    continue
                marker = markers.get(job.job_id)
                if (
                    not isinstance(marker, dict)
                    or marker.get("job_id") != job.job_id
                ):
                    job.state = "interrupted"
                    job.tracking_state = "complete"
                    job.execution_alive = False
                    job.output_unavailable_reason = (
                        "Runtime completion marker was not found"
                    )
                    job.finished_at = job.updated_at = time.time()
                elif marker.get("state") in {"finished", "error"}:
                    try:
                        self._finish_from_manifest(job, marker, persist=False)
                    except (TypeError, ValueError, OverflowError):
                        self._detach_job(
                            job,
                            "Runtime completion marker was invalid",
                            persist=False,
                        )
                    else:
                        if job.output_bytes and not job.outputs:
                            job.output_unavailable_reason = (
                                "Browser disconnected before the final output response; "
                                "read the runtime artifact instead"
                            )
                else:
                    stale_after = (
                        job.started_at
                        + job.execution_timeout_seconds
                        + DEFAULT_RUNTIME_STALE_GRACE_SECONDS
                    )
                    if time.time() > stale_after:
                        job.state = "interrupted"
                        job.tracking_state = "complete"
                        job.execution_alive = False
                        job.output_unavailable_reason = (
                            "Runtime job did not produce a terminal marker before "
                            "its recovery deadline"
                        )
                        job.finished_at = job.updated_at = time.time()
                    else:
                        self._detach_job(
                            job,
                            "Runtime job has not produced a terminal marker",
                            persist=False,
                        )
                self._completion_events[job.job_id].set()
            self._persist_journal()
            return [self._job_dict(job) for job in all_detached]
        finally:
            for job_id, attempt in attempts.items():
                if self._reconciliation_attempts.get(job_id) is attempt:
                    self._reconciliation_attempts.pop(job_id, None)

    async def _read_runtime_markers(
        self, jobs: list[ColabJob]
    ) -> dict[str, Any]:
        paths = {job.job_id: job.runtime_marker_path for job in jobs}
        code = (
            "import builtins as _cc_builtins\n"
            "import json, os\n"
            f"_paths = {paths!r}\n"
            "_markers = {}\n"
            "for _job_id, _path in _paths.items():\n"
            "    try:\n"
            "        with _cc_builtins.open(_path, 'r', encoding='utf-8') as _handle:\n"
            "            _markers[_job_id] = json.load(_handle)\n"
            "    except (_cc_builtins.OSError, _cc_builtins.ValueError):\n"
            "        _markers[_job_id] = None\n"
            f"_cc_builtins.print({_RECONCILE_SENTINEL!r} + json.dumps(_markers, separators=(',', ':')))\n"
        )
        now = time.time()
        probe_timeout = max(
            0.1,
            min(
                job.started_at
                + job.execution_timeout_seconds
                + DEFAULT_RUNTIME_STALE_GRACE_SECONDS
                - now
                for job in jobs
            ),
        )
        result = await self._append_and_run_probe(code, timeout=probe_timeout)
        parsed = _sentinel_json(result, _RECONCILE_SENTINEL)
        if not isinstance(parsed, dict):
            raise RuntimeError(
                "Colab runtime probe returned no matching reconciliation sentinel"
            )
        return parsed

    def kernel_readiness(self) -> dict[str, Any]:
        return {
            "kernel_execution_ready": self._kernel_execution_ready,
            "kernel_probe_at": self._kernel_probe_at,
            "kernel_probe_latency_ms": self._kernel_probe_latency_ms,
            "kernel_probe_error": self._kernel_probe_error,
        }

    def mark_kernel_unknown(self, reason: str | None = None) -> None:
        self._kernel_execution_ready = None
        self._kernel_probe_at = None
        self._kernel_probe_latency_ms = None
        self._kernel_probe_error = _safe_error(reason)

    async def probe_kernel(
        self, timeout: float = DEFAULT_KERNEL_PROBE_TIMEOUT_SECONDS
    ) -> dict[str, Any]:
        nonce = uuid.uuid4().hex
        code = (
            "import builtins as _cc_builtins\n"
            "import json\n"
            f"_cc_builtins.print({_KERNEL_PROBE_SENTINEL!r} + "
            f"json.dumps({{'nonce': {nonce!r}}}, separators=(',', ':')))\n"
        )
        started = time.monotonic()
        try:
            outputs = await self._append_and_run_probe(code, timeout=timeout)
            proof = _sentinel_json(outputs, _KERNEL_PROBE_SENTINEL)
            if not isinstance(proof, dict) or proof.get("nonce") != nonce:
                raise RuntimeError(
                    "Colab runtime probe returned no matching kernel sentinel"
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._kernel_execution_ready = False
            self._kernel_probe_error = _safe_error(
                f"{type(exc).__name__}: {exc}"
            )
        else:
            self._kernel_execution_ready = True
            self._kernel_probe_error = None
        self._kernel_probe_at = time.time()
        self._kernel_probe_latency_ms = (time.monotonic() - started) * 1000
        return self.kernel_readiness()

    async def _append_and_run_probe(
        self, code: str, *, timeout: float = DEFAULT_EXECUTION_TIMEOUT_SECONDS
    ) -> list[Any]:
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("probe timeout must be finite and positive")
        async with asyncio.timeout(timeout):
            async with self._probe_lock:
                return await self._append_and_run_probe_locked(
                    code, timeout=timeout
                )

    async def _append_and_run_probe_locked(
        self, code: str, *, timeout: float
    ) -> list[Any]:
        code = f"{_PROBE_CELL_MARKER}\n{code}"
        await self._recover_pending_probe_cell()
        cell_id = self._probe_cell_id
        if cell_id is not None:
            found = await self._find_cell(cell_id)
            if found is None or not self._is_owned_probe_cell(found[0]):
                cell_id = None
                self._probe_cell_id = None
                self._persist_journal()
        if cell_id is not None:
            try:
                update_result = await self.session.call_tool(
                    "update_cell", {"cellId": cell_id, "content": code}
                )
                updated_cell_id = result_data(update_result).get("cellId")
                if updated_cell_id is not None and updated_cell_id != cell_id:
                    raise RuntimeError(
                        "Colab updated a different cell than the requested probe cell"
                    )
            except Exception:
                try:
                    current_index = await self._find_cell_index(cell_id)
                except Exception:
                    raise
                if current_index is not None:
                    raise
                cell_id = None
                self._probe_cell_id = None
                self._persist_journal()
        if cell_id is None:
            cell_count = await self._cell_count()
            self._pending_probe_cell_sha256 = hashlib.sha256(
                code.encode("utf-8")
            ).hexdigest()
            self._persist_journal()
            try:
                added = await self.session.call_tool(
                    "add_code_cell",
                    {"cellIndex": cell_count, "language": "python", "code": code},
                )
                cell_id = result_data(added).get("newCellId")
                if not isinstance(cell_id, str):
                    raise ValueError(
                        "Colab did not return a cell id for the recovery probe"
                    )
            except Exception:
                try:
                    await self._recover_pending_probe_cell(clear_if_missing=False)
                except Exception:
                    pass
                raise
            self._probe_cell_id = cell_id
            self._pending_probe_cell_sha256 = None
            self._persist_journal()
        run = await self.session.call_tool(
            "run_code_cell", {"cellId": cell_id}, timeout=timeout
        )
        data = result_data(run)
        outputs = data.get("outputs", _MISSING)
        if not isinstance(outputs, list):
            raise RuntimeError(
                "Malformed run_code_cell probe response: outputs must be a list"
            )
        return outputs

    async def read_artifact(
        self,
        artifact_id: str,
        *,
        offset: int = 0,
        limit_bytes: int = MAX_ARTIFACT_READ_BYTES,
    ) -> dict[str, Any]:
        if not valid_opaque_id(artifact_id):
            raise ArtifactNotFoundError("Unknown or expired artifact id")
        try:
            return self.artifact_store.read_chunk(
                artifact_id, offset=offset, limit_bytes=limit_bytes
            )
        except ArtifactNotFoundError:
            pass
        runtime_path = self._runtime_artifacts.get(artifact_id)
        if runtime_path is None:
            raise ArtifactNotFoundError("Unknown or expired artifact id")
        if offset < 0:
            raise ValueError("offset must be greater than or equal to zero")
        if not 1 <= limit_bytes <= MAX_ARTIFACT_READ_BYTES:
            raise ValueError(
                f"limit_bytes must be between 1 and {MAX_ARTIFACT_READ_BYTES}"
            )
        artifact_ref = next(
            (
                job.output_artifact
                for job in self.jobs.values()
                if isinstance(job.output_artifact, dict)
                and job.output_artifact.get("artifact_id") == artifact_id
            ),
            None,
        )
        expires_at = (
            artifact_ref.get("expires_at")
            if isinstance(artifact_ref, dict)
            else None
        )
        if isinstance(expires_at, int | float) and time.time() >= expires_at:
            self._runtime_artifacts.pop(artifact_id, None)
            raise ArtifactNotFoundError("Unknown or expired artifact id")
        expected_sha256 = (
            str(artifact_ref.get("sha256", ""))
            if isinstance(artifact_ref, dict)
            else ""
        )
        code = (
            "import builtins as _cc_builtins\n"
            "import base64, json, os\n"
            f"_path = {runtime_path!r}\n"
            f"_offset = {offset}\n"
            f"_limit = {limit_bytes}\n"
            "_size = os.path.getsize(_path)\n"
            "if _offset > _size: raise _cc_builtins.ValueError('offset is beyond the artifact')\n"
            "with _cc_builtins.open(_path, 'rb') as _handle:\n"
            "    _handle.seek(_offset)\n"
            "    _chunk = _handle.read(_limit)\n"
            "_next = _offset + _cc_builtins.len(_chunk)\n"
            "_value = {'artifact_id': "
            f"{artifact_id!r}, 'size_bytes': _size, 'sha256': {expected_sha256!r}, "
            "'offset': _offset, 'next_offset': _next, 'eof': _next >= _size, "
            "'encoding': 'base64', 'data': base64.b64encode(_chunk).decode('ascii')}\n"
            f"_cc_builtins.print({_ARTIFACT_SENTINEL!r} + json.dumps(_value, separators=(',', ':')))\n"
        )
        artifact_probe_timeout = DEFAULT_ARTIFACT_PROBE_TIMEOUT_SECONDS
        if isinstance(expires_at, int | float):
            artifact_probe_timeout = min(
                artifact_probe_timeout,
                max(0.1, expires_at - time.time()),
            )
        outputs = await self._append_and_run_probe(
            code, timeout=artifact_probe_timeout
        )
        if isinstance(expires_at, int | float) and time.time() >= expires_at:
            self._runtime_artifacts.pop(artifact_id, None)
            raise ArtifactNotFoundError("Unknown or expired artifact id")
        parsed = _sentinel_json(outputs, _ARTIFACT_SENTINEL)
        if not isinstance(parsed, dict):
            raise ValueError("Colab runtime did not return an artifact chunk")
        return parsed

    async def mark_stale(self, reason: str) -> None:
        now = time.time()
        tasks: list[tuple[str, asyncio.Task[None]]] = []
        for job in self.jobs.values():
            if job.state in _TERMINAL_STATES and job.tracking_state == "complete":
                continue
            job.state = "stale"
            job.tracking_state = "complete"
            job.execution_alive = None
            job.error = _safe_error(reason)
            job.finished_at = job.updated_at = now
            task = self._tasks.get(job.job_id)
            if task is not None:
                tasks.append((job.job_id, task))
        await self._cancel_job_tasks(tasks)
        for job in self.jobs.values():
            if job.state == "stale":
                event = self._completion_events.get(job.job_id)
                if event is not None:
                    event.set()
        self._job_cells.clear()
        self._probe_cell_id = None
        self._pending_job_cell_sha256 = None
        self._pending_probe_cell_sha256 = None
        self.mark_kernel_unknown("Colab connection was reset")
        self._persist_journal()

    async def detach_for_shutdown(
        self, reason: str = "Colab adapter shut down"
    ) -> None:
        """Stop local tracking while preserving jobs for later reconciliation."""

        tasks: list[tuple[str, asyncio.Task[None]]] = []
        for job in self.jobs.values():
            if job.state in {"running", "timed_out"} and job.tracking_state != "complete":
                self._detach_job(job, reason, persist=False)
            task = self._tasks.get(job.job_id)
            if task is not None and not task.done():
                tasks.append((job.job_id, task))
        await self._cancel_job_tasks(tasks)
        self._persist_journal()

    async def close(self) -> None:
        await self.detach_for_shutdown()


def _sentinel_json(outputs: list[Any], sentinel: str) -> Any:
    for output in reversed(outputs):
        if not isinstance(output, dict):
            continue
        text = output.get("text", "")
        combined = "".join(str(item) for item in text) if isinstance(text, list) else str(text)
        for line in reversed(combined.splitlines()):
            if line.startswith(sentinel):
                try:
                    return json.loads(line[len(sentinel) :])
                except json.JSONDecodeError:
                    continue
    return None


_RUNTIME_WRAPPER_TEMPLATE = r'''
def _cc_run_connector_job():
    import base64 as _cc_base64
    import builtins as _cc_builtins
    import hashlib as _cc_hashlib
    import io as _cc_io
    import json as _cc_json
    import os as _cc_os
    import sys as _cc_sys
    import threading as _cc_threading
    import time as _cc_time
    import traceback as _cc_traceback

    _cc_config = __COLAB_JOB_CONFIG__
    _cc_user_globals = _cc_builtins.globals()
    _cc_os.makedirs(_cc_os.path.dirname(_cc_config["marker_path"]), exist_ok=True)

    def _cc_state_threads_snapshot(_cc_state_value):
        _cc_lock = _cc_state_value.get("lock")
        if _cc_lock is None:
            return _cc_builtins.tuple(_cc_state_value.get("threads", ()))
        with _cc_lock:
            return _cc_builtins.tuple(_cc_state_value.get("threads", ()))

    def _cc_state_add_thread(_cc_state_value, _cc_thread):
        _cc_lock = _cc_state_value.get("lock")
        if _cc_lock is None:
            _cc_state_value["threads"].add(_cc_thread)
            return
        with _cc_lock:
            _cc_state_value["threads"].add(_cc_thread)

    def _cc_state_clear_threads(_cc_state_value):
        _cc_lock = _cc_state_value.get("lock")
        if _cc_lock is None:
            _cc_state_value["threads"].clear()
            return
        with _cc_lock:
            _cc_state_value["threads"].clear()

    def _cc_unwrap_idle_guard(_cc_value):
        while _cc_builtins.getattr(
            _cc_value, "_cc_colab_output_guard", False
        ):
            _cc_old_state = _cc_builtins.getattr(
                _cc_value, "_cc_guard_state", None
            )
            if not _cc_builtins.isinstance(_cc_old_state, _cc_builtins.dict):
                break
            if _cc_old_state.get("active") or _cc_builtins.any(
                _cc_thread.is_alive()
                for _cc_thread in _cc_state_threads_snapshot(_cc_old_state)
            ):
                break
            _cc_value = _cc_builtins.getattr(
                _cc_value, "_cc_guard_original", _cc_value
            )
        return _cc_value

    def _cc_valid_stem(_cc_stem):
        return _cc_builtins.len(_cc_stem) == 32 and not _cc_builtins.any(
            _cc_char not in "0123456789abcdef" for _cc_char in _cc_stem
        )

    def _cc_remove_runtime_pair(_cc_output, _cc_marker):
        for _cc_path in (_cc_output, _cc_marker, _cc_marker + ".tmp"):
            try:
                _cc_os.unlink(_cc_path)
            except _cc_builtins.FileNotFoundError:
                pass

    def _cc_apply_runtime_quota():
        _cc_root = _cc_os.path.dirname(_cc_config["output_path"])
        _cc_candidates = []
        _cc_total = 0
        _cc_pair_count = 0
        _cc_now = _cc_time.time()
        _cc_current_stem = _cc_config["job_id"]
        _cc_stems = _cc_builtins.set()
        for _cc_name in _cc_os.listdir(_cc_root):
            if _cc_name.endswith(".json.tmp"):
                _cc_stem = _cc_name[:-9]
                if _cc_valid_stem(_cc_stem) and _cc_stem != _cc_current_stem:
                    try:
                        _cc_os.unlink(_cc_os.path.join(_cc_root, _cc_name))
                    except _cc_builtins.FileNotFoundError:
                        pass
                continue
            if _cc_name.endswith(".output"):
                _cc_stem = _cc_name[:-7]
            elif _cc_name.endswith(".json"):
                _cc_stem = _cc_name[:-5]
            else:
                continue
            if _cc_valid_stem(_cc_stem) and _cc_stem != _cc_current_stem:
                _cc_stems.add(_cc_stem)
        for _cc_stem in _cc_stems:
            _cc_pair_count += 1
            _cc_output = _cc_os.path.join(_cc_root, _cc_stem + ".output")
            _cc_marker = _cc_os.path.join(_cc_root, _cc_stem + ".json")
            _cc_size = 0
            _cc_mtime = 0
            for _cc_path in (_cc_output, _cc_marker):
                try:
                    _cc_stat = _cc_os.stat(_cc_path)
                except _cc_builtins.OSError:
                    continue
                _cc_size += _cc_stat.st_size
                _cc_mtime = _cc_builtins.max(_cc_mtime, _cc_stat.st_mtime)
            if _cc_now - _cc_mtime > _cc_config["artifact_ttl_seconds"]:
                _cc_remove_runtime_pair(_cc_output, _cc_marker)
                _cc_pair_count -= 1
                continue
            _cc_total += _cc_size
            # Colab executes one cell at a time. A different marker still marked
            # running when this wrapper starts is an orphan, not an active peer.
            _cc_candidates.append((_cc_mtime, _cc_size, _cc_output, _cc_marker))
        for _, _cc_size, _cc_output, _cc_marker in _cc_builtins.sorted(
            _cc_candidates
        ):
            if (
                _cc_total
                + _cc_config["artifact_limit"]
                + _cc_config["artifact_marker_reserve"]
                <= _cc_config["artifact_total_limit"]
                and _cc_pair_count + 1 <= _cc_config["artifact_max_pairs"]
            ):
                break
            _cc_remove_runtime_pair(_cc_output, _cc_marker)
            _cc_total -= _cc_size
            _cc_pair_count -= 1
        _cc_available = _cc_builtins.max(
            0,
            _cc_config["artifact_total_limit"]
            - _cc_total
            - _cc_config["artifact_marker_reserve"],
        )
        _cc_config["artifact_limit"] = _cc_builtins.min(
            _cc_config["artifact_limit"], _cc_available
        )

    _cc_apply_runtime_quota()

    def _cc_write_marker(_cc_value):
        _cc_value["job_id"] = _cc_config["job_id"]
        _cc_value["artifact_id"] = _cc_config["artifact_id"]
        _cc_temp = _cc_config["marker_path"] + ".tmp"
        with _cc_builtins.open(_cc_temp, "w", encoding="utf-8") as _cc_handle:
            _cc_json.dump(
                _cc_value,
                _cc_handle,
                ensure_ascii=True,
                separators=(",", ":"),
            )
            _cc_handle.flush()
            _cc_os.fsync(_cc_handle.fileno())
        _cc_os.replace(_cc_temp, _cc_config["marker_path"])

    class _cc_Capture:
        def __init__(self):
            self.total = 0
            self.stored = 0
            self.excerpt = _cc_builtins.bytearray()
            self.digest = _cc_hashlib.sha256()
            self.handle = _cc_builtins.open(_cc_config["output_path"], "wb")
            self.lock = _cc_threading.Lock()

        def write_bytes(self, raw):
            if not _cc_builtins.isinstance(raw, _cc_builtins.bytes):
                raw = _cc_builtins.bytes(raw)
            with self.lock:
                self.total += _cc_builtins.len(raw)
                remaining = _cc_builtins.max(
                    0, _cc_config["artifact_limit"] - self.stored
                )
                stored = raw[:remaining]
                if stored:
                    self.handle.write(stored)
                    self.digest.update(stored)
                    self.stored += _cc_builtins.len(stored)
                excerpt_remaining = _cc_builtins.max(
                    0,
                    _cc_config["excerpt_limit"]
                    - _cc_builtins.len(self.excerpt),
                )
                if excerpt_remaining:
                    self.excerpt.extend(raw[:excerpt_remaining])

        def write(self, channel, value):
            if not _cc_builtins.isinstance(value, _cc_builtins.str):
                value = _cc_builtins.str(value)
            raw = ("[" + channel + "] " + value).encode(
                "utf-8", errors="replace"
            )
            self.write_bytes(raw)
            return _cc_builtins.len(value)

        def flush(self):
            with self.lock:
                self.handle.flush()

        def close(self):
            with self.lock:
                self.handle.flush()
                _cc_os.fsync(self.handle.fileno())
                self.handle.close()

    class _cc_ThreadAwareStream(_cc_io.TextIOBase):
        def __init__(self, delegate, capture, channel, guard_state):
            self.delegate = delegate
            self.capture = capture
            self.channel = channel
            self.guard_state = guard_state
            self.capture_all = True
            self._cc_colab_output_guard = True
            self._cc_guard_state = guard_state

        def write(self, value):
            _cc_threads = _cc_state_threads_snapshot(self.guard_state)
            if _cc_threads and not _cc_builtins.any(
                _cc_thread.is_alive() for _cc_thread in _cc_threads
            ):
                _cc_state_clear_threads(self.guard_state)
                _cc_threads = ()
            if (
                self.capture_all
                or _cc_threading.current_thread() in _cc_threads
            ):
                if self.capture is None:
                    return _cc_builtins.len(_cc_builtins.str(value))
                return self.capture.write(self.channel, value)
            return self.delegate.write(value)

        def flush(self):
            if self.capture_all and self.capture is not None:
                self.capture.flush()
            else:
                self.delegate.flush()

        def fileno(self):
            return self.delegate.fileno()

        def isatty(self):
            return self.delegate.isatty()

        def finish(self):
            self.capture_all = False

        def detach_capture(self):
            self.capture = None

    _cc_capture = _cc_Capture()
    _cc_native_read, _cc_native_write = _cc_os.pipe()
    _cc_os.set_blocking(_cc_native_read, False)
    _cc_saved_stdout = _cc_os.dup(1)
    _cc_saved_stderr = _cc_os.dup(2)
    _cc_native_stop = _cc_threading.Event()

    def _cc_drain_native_output():
        try:
            while not _cc_native_stop.is_set():
                try:
                    _cc_chunk = _cc_os.read(_cc_native_read, 65536)
                except _cc_builtins.BlockingIOError:
                    _cc_native_stop.wait(0.01)
                    continue
                if not _cc_chunk:
                    return
                _cc_capture.write_bytes(_cc_chunk)
            # Drain a bounded amount already queued when the user code returned.
            for _cc_unused in _cc_builtins.range(16):
                try:
                    _cc_chunk = _cc_os.read(_cc_native_read, 65536)
                except _cc_builtins.BlockingIOError:
                    break
                if not _cc_chunk:
                    break
                _cc_capture.write_bytes(_cc_chunk)
        finally:
            _cc_os.close(_cc_native_read)

    _cc_native_thread = _cc_threading.Thread(
        target=_cc_drain_native_output,
        name="colab-codex-output-capture",
        daemon=True,
    )
    _cc_native_thread.start()
    _cc_os.dup2(_cc_native_write, 1)
    _cc_os.dup2(_cc_native_write, 2)
    _cc_os.close(_cc_native_write)
    _cc_original_stdout = _cc_sys.stdout
    while _cc_builtins.getattr(
        _cc_original_stdout, "_cc_colab_output_guard", False
    ) and not _cc_builtins.any(
        _cc_thread.is_alive()
        for _cc_thread in _cc_state_threads_snapshot(
            _cc_original_stdout._cc_guard_state
        )
    ):
        _cc_original_stdout = _cc_original_stdout.delegate
    _cc_original_stderr = _cc_sys.stderr
    while _cc_builtins.getattr(
        _cc_original_stderr, "_cc_colab_output_guard", False
    ) and not _cc_builtins.any(
        _cc_thread.is_alive()
        for _cc_thread in _cc_state_threads_snapshot(
            _cc_original_stderr._cc_guard_state
        )
    ):
        _cc_original_stderr = _cc_original_stderr.delegate
    _cc_print_guard_state = {
        "active": True,
        "threads": _cc_builtins.set(),
        "capture": _cc_capture,
        "lock": _cc_threading.Lock(),
    }
    _cc_stdout_guard = _cc_ThreadAwareStream(
        _cc_original_stdout, _cc_capture, "stdout", _cc_print_guard_state
    )
    _cc_stderr_guard = _cc_ThreadAwareStream(
        _cc_original_stderr, _cc_capture, "stderr", _cc_print_guard_state
    )
    _cc_sys.stdout = _cc_stdout_guard
    _cc_sys.stderr = _cc_stderr_guard
    _cc_original_print = _cc_unwrap_idle_guard(_cc_builtins.print)

    def _cc_guarded_print(*args, **kwargs):
        _cc_tracked_threads = _cc_state_threads_snapshot(
            _cc_print_guard_state
        )
        if (
            not _cc_print_guard_state["active"]
            and _cc_tracked_threads
            and not _cc_builtins.any(
                _cc_thread.is_alive()
                for _cc_thread in _cc_tracked_threads
            )
        ):
            _cc_state_clear_threads(_cc_print_guard_state)
            _cc_tracked_threads = ()
        if (
            _cc_print_guard_state["active"]
            or _cc_threading.current_thread()
            in _cc_tracked_threads
        ):
            if _cc_stdout_guard.capture is None:
                return None
        return _cc_original_print(*args, **kwargs)

    _cc_guarded_print._cc_colab_output_guard = True
    _cc_guarded_print._cc_guard_state = _cc_print_guard_state
    _cc_guarded_print._cc_guard_original = _cc_original_print

    _cc_original_os_write = _cc_unwrap_idle_guard(_cc_os.write)

    def _cc_guarded_os_write(fd, data):
        _cc_tracked_threads = _cc_state_threads_snapshot(
            _cc_print_guard_state
        )
        if (
            not _cc_print_guard_state["active"]
            and _cc_tracked_threads
            and not _cc_builtins.any(
                _cc_thread.is_alive()
                for _cc_thread in _cc_tracked_threads
            )
        ):
            _cc_state_clear_threads(_cc_print_guard_state)
            _cc_tracked_threads = ()
        if (
            _cc_print_guard_state["active"]
            or _cc_threading.current_thread()
            in _cc_tracked_threads
        ) and _cc_print_guard_state["capture"] is None:
            return _cc_builtins.len(data)
        return _cc_original_os_write(fd, data)

    _cc_guarded_os_write._cc_colab_output_guard = True
    _cc_guarded_os_write._cc_guard_state = _cc_print_guard_state
    _cc_guarded_os_write._cc_guard_original = _cc_original_os_write

    _cc_original_thread_start = _cc_unwrap_idle_guard(
        _cc_threading.Thread.start
    )

    def _cc_guarded_thread_start(thread, *args, **kwargs):
        _cc_tracked_threads = _cc_state_threads_snapshot(
            _cc_print_guard_state
        )
        if (
            _cc_print_guard_state["active"]
            or _cc_threading.current_thread()
            in _cc_tracked_threads
        ):
            _cc_state_add_thread(_cc_print_guard_state, thread)
        return _cc_original_thread_start(thread, *args, **kwargs)

    _cc_guarded_thread_start._cc_colab_output_guard = True
    _cc_guarded_thread_start._cc_guard_state = _cc_print_guard_state
    _cc_guarded_thread_start._cc_guard_original = _cc_original_thread_start

    _cc_builtins.print = _cc_guarded_print
    _cc_os.write = _cc_guarded_os_write
    _cc_threading.Thread.start = _cc_guarded_thread_start
    _cc_emit_result = _cc_original_print
    _cc_started = _cc_time.time()
    _cc_write_marker({"state": "running", "started_at": _cc_started})
    _cc_state = "finished"
    _cc_error = None
    _cc_display_pub = None
    _cc_original_publish = None
    try:
        try:
            _cc_get_ipython = _cc_user_globals.get("get_ipython")
            if not _cc_builtins.callable(_cc_get_ipython):
                _cc_get_ipython = _cc_builtins.getattr(
                    _cc_builtins, "get_ipython", None
                )
            _cc_ipython = (
                _cc_get_ipython()
                if _cc_builtins.callable(_cc_get_ipython)
                else None
            )
            _cc_display_pub = _cc_builtins.getattr(
                _cc_ipython, "display_pub", None
            )
            _cc_original_publish = _cc_builtins.getattr(
                _cc_display_pub, "publish", None
            )
            if _cc_original_publish is not None:
                _cc_original_publish = _cc_unwrap_idle_guard(
                    _cc_original_publish
                )
            if _cc_original_publish is not None:
                def _cc_publish(data, metadata=None, **kwargs):
                    if not _cc_print_guard_state["active"]:
                        _cc_tracked_threads = _cc_state_threads_snapshot(
                            _cc_print_guard_state
                        )
                        if (
                            _cc_tracked_threads
                            and not _cc_builtins.any(
                                _cc_thread.is_alive()
                                for _cc_thread in _cc_tracked_threads
                            )
                        ):
                            _cc_state_clear_threads(_cc_print_guard_state)
                            _cc_tracked_threads = ()
                        if (
                            _cc_threading.current_thread()
                            in _cc_tracked_threads
                        ):
                            return None
                        return _cc_original_publish(
                            data, metadata=metadata, **kwargs
                        )
                    _cc_display_capture = _cc_print_guard_state["capture"]
                    if _cc_display_capture is None:
                        return None
                    for _cc_mime, _cc_value in (data or {}).items():
                        if _cc_builtins.isinstance(
                            _cc_value, _cc_builtins.bytes
                        ):
                            _cc_value = _cc_base64.b64encode(_cc_value).decode(
                                "ascii"
                            )
                        _cc_display_capture.write(
                            "display:" + _cc_builtins.str(_cc_mime),
                            _cc_value,
                        )
                _cc_publish._cc_colab_output_guard = True
                _cc_publish._cc_guard_state = _cc_print_guard_state
                _cc_publish._cc_guard_original = _cc_original_publish
                _cc_display_pub.publish = _cc_publish
        except _cc_builtins.Exception as _cc_display_exc:
            _cc_display_capture_required = (
                _cc_display_pub is not None
                and _cc_original_publish is not None
            )
            _cc_display_pub = None
            _cc_original_publish = None
            if _cc_display_capture_required:
                raise _cc_builtins.RuntimeError(
                    "Unable to install bounded display capture"
                ) from _cc_display_exc
        _cc_source = _cc_base64.b64decode(
            _cc_config["code_b64"]
        ).decode("utf-8")
        _cc_builtins.exec(
            _cc_builtins.compile(
                _cc_source,
                "<colab-codex-job:" + _cc_config["job_id"] + ">",
                "exec",
            ),
            _cc_user_globals,
            _cc_user_globals,
        )
    except _cc_builtins.BaseException as _cc_exc:
        _cc_state = "error"
        _cc_error = (
            _cc_builtins.type(_cc_exc).__name__
            + ": "
            + _cc_builtins.str(_cc_exc)
        ).encode("utf-8", errors="replace")[:4096].decode(
            "utf-8", errors="replace"
        )
        _cc_capture.write("traceback", _cc_traceback.format_exc())
    finally:
        _cc_drain_deadline = _cc_time.monotonic() + 0.1
        while _cc_time.monotonic() < _cc_drain_deadline:
            _cc_live_threads = [
                _cc_thread
                for _cc_thread in _cc_state_threads_snapshot(
                    _cc_print_guard_state
                )
                if _cc_thread.is_alive()
            ]
            if not _cc_live_threads:
                break
            _cc_remaining = _cc_drain_deadline - _cc_time.monotonic()
            for _cc_thread in _cc_live_threads:
                _cc_thread.join(
                    timeout=_cc_builtins.max(
                        0, _cc_remaining / _cc_builtins.len(_cc_live_threads)
                    )
                )
        with _cc_print_guard_state["lock"]:
            _cc_lingering_threads = {
                _cc_thread
                for _cc_thread in _cc_print_guard_state["threads"]
                if _cc_thread.is_alive()
            }
            _cc_print_guard_state["active"] = False
            _cc_print_guard_state["threads"] = _cc_lingering_threads
        _cc_stdout_guard.finish()
        _cc_stderr_guard.finish()

        def _cc_restore_output_guards():
            if _cc_sys.stdout is _cc_stdout_guard:
                _cc_sys.stdout = _cc_original_stdout
            if _cc_sys.stderr is _cc_stderr_guard:
                _cc_sys.stderr = _cc_original_stderr
            if _cc_builtins.print is _cc_guarded_print:
                _cc_builtins.print = _cc_original_print
            if _cc_os.write is _cc_guarded_os_write:
                _cc_os.write = _cc_original_os_write
            if _cc_threading.Thread.start is _cc_guarded_thread_start:
                _cc_threading.Thread.start = _cc_original_thread_start
            if (
                _cc_display_pub is not None
                and _cc_original_publish is not None
                and _cc_builtins.getattr(_cc_display_pub, "publish", None)
                is _cc_publish
            ):
                _cc_display_pub.publish = _cc_original_publish

        if not _cc_lingering_threads:
            _cc_restore_output_guards()
        _cc_native_stop.set()
        _cc_native_thread.join(timeout=1.0)
        _cc_capture.close()
        _cc_print_guard_state["capture"] = None
        _cc_stdout_guard.detach_capture()
        _cc_stderr_guard.detach_capture()
        _cc_os.dup2(_cc_saved_stdout, 1)
        _cc_os.dup2(_cc_saved_stderr, 2)
        _cc_os.close(_cc_saved_stdout)
        _cc_os.close(_cc_saved_stderr)

    _cc_manifest = {
        "state": _cc_state,
        "error": _cc_error,
        "started_at": _cc_started,
        "finished_at": _cc_time.time(),
        "output_bytes": _cc_capture.total,
        "output_excerpt": _cc_builtins.bytes(_cc_capture.excerpt).decode(
            "utf-8", errors="replace"
        ),
        "output_truncated": _cc_capture.total
        > _cc_builtins.len(_cc_capture.excerpt),
        "artifact_size_bytes": _cc_capture.stored,
        "artifact_sha256": _cc_capture.digest.hexdigest(),
        "artifact_truncated": _cc_capture.total > _cc_capture.stored,
    }
    _cc_marker_manifest = {**_cc_manifest}
    _cc_marker_manifest.pop("output_excerpt", None)
    _cc_write_marker(_cc_marker_manifest)
    _cc_emit_result(
        _cc_config["sentinel"]
        + _cc_json.dumps(
            {**_cc_manifest, "job_id": _cc_config["job_id"]},
            ensure_ascii=True,
            separators=(",", ":"),
        )
    )

_cc_run_connector_job()
'''
