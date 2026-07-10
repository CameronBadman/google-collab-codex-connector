from __future__ import annotations

import asyncio
import math
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from .session import ColabSessionManager
from .tools import first_json_object, serialize_tool_result

DEFAULT_EXECUTION_TIMEOUT_SECONDS = 43_200.0
DEFAULT_WAIT_TIMEOUT_SECONDS = 300.0
MIN_WAIT_TIMEOUT_SECONDS = 1.0
MAX_WAIT_TIMEOUT_SECONDS = 900.0


def result_data(result: Any) -> dict[str, Any]:
    data = getattr(result, "structured_content", None)
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


@dataclass
class ColabJob:
    job_id: str
    cell_id: str
    code: str
    state: str
    started_at: float
    updated_at: float
    execution_timeout_seconds: float
    finished_at: float | None = None
    last_output_at: float | None = None
    outputs: list[Any] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ColabJobManager:
    def __init__(self, session: ColabSessionManager):
        self.session = session
        self.jobs: dict[str, ColabJob] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._completion_events: dict[str, asyncio.Event] = {}
        self._lock = asyncio.Lock()

    def _job_dict(self, job: ColabJob) -> dict[str, Any]:
        task = self._tasks.get(job.job_id)
        return {
            **job.to_dict(),
            "task_alive": (
                job.state == "running" and task is not None and not task.done()
            ),
        }

    def _update_outputs(self, job: ColabJob, outputs: list[Any]) -> None:
        if outputs == job.outputs:
            return
        now = time.time()
        job.outputs = outputs
        job.updated_at = now
        if outputs:
            job.last_output_at = now

    async def _remote_tool_names(self) -> set[str]:
        return {tool.name for tool in await self.session.list_tools()}

    async def _get_cells(self, include_outputs: bool = False) -> list[dict[str, Any]]:
        result = await self.session.call_tool(
            "get_cells", {"includeOutputs": include_outputs}
        )
        cells = result_data(result).get("cells", [])
        return cells if isinstance(cells, list) else []

    async def _job_cell(self, job: ColabJob) -> dict[str, Any] | None:
        for cell in await self._get_cells(include_outputs=True):
            if cell.get("id") == job.cell_id:
                return cell
        return None

    async def start_python(
        self,
        code: str,
        language: str = "python",
        execution_timeout_seconds: float = DEFAULT_EXECUTION_TIMEOUT_SECONDS,
    ) -> dict[str, Any]:
        if execution_timeout_seconds <= 0:
            raise ValueError("execution_timeout_seconds must be greater than zero")
        names = await self._remote_tool_names()
        required = {"add_code_cell", "run_code_cell", "get_cells"}
        if not required.issubset(names):
            missing = ", ".join(sorted(required - names))
            raise ValueError(f"Colab remote tools missing for jobs: {missing}")

        cells = await self._get_cells(include_outputs=False)
        add_result = await self.session.call_tool(
            "add_code_cell",
            {"cellIndex": len(cells), "language": language, "code": code},
        )
        cell_id = result_data(add_result).get("newCellId")
        if not isinstance(cell_id, str):
            raise ValueError("Colab did not return a newCellId from add_code_cell")

        started_at = time.time()
        job = ColabJob(
            job_id=uuid.uuid4().hex,
            cell_id=cell_id,
            code=code,
            state="running",
            started_at=started_at,
            updated_at=started_at,
            execution_timeout_seconds=execution_timeout_seconds,
        )
        async with self._lock:
            self.jobs[job.job_id] = job
            self._completion_events[job.job_id] = asyncio.Event()
            self._tasks[job.job_id] = asyncio.create_task(
                self._execute(job), name=f"colab-job-{job.job_id}"
            )
        return {
            **self._job_dict(job),
            "add_result": serialize_tool_result(add_result),
        }

    async def _execute(self, job: ColabJob) -> None:
        try:
            run_result = await self.session.call_tool(
                "run_code_cell",
                {"cellId": job.cell_id},
                timeout=job.execution_timeout_seconds,
            )
            outputs = result_data(run_result).get("outputs", [])
            if not isinstance(outputs, list):
                outputs = []
            if job.state == "running":
                self._finish_from_outputs(job, outputs)
        except asyncio.TimeoutError:
            if job.state == "running":
                job.state = "timed_out"
                job.error = (
                    "Colab execution exceeded "
                    f"{job.execution_timeout_seconds:g} seconds"
                )
                job.finished_at = job.updated_at = time.time()
        except asyncio.CancelledError:
            if job.state == "running":
                job.state = "stale"
                job.error = "Colab execution tracking was cancelled"
                job.finished_at = job.updated_at = time.time()
            raise
        except Exception as exc:
            if job.state == "running":
                job.state = "error"
                job.error = f"{type(exc).__name__}: {exc}"
                job.finished_at = job.updated_at = time.time()
        finally:
            self._tasks.pop(job.job_id, None)
            event = self._completion_events.get(job.job_id)
            if event is not None:
                event.set()

    def _finish_from_outputs(self, job: ColabJob, outputs: list[Any]) -> None:
        self._update_outputs(job, outputs)
        job.error = output_error(outputs)
        job.state = "error" if output_has_error(outputs) else "finished"
        job.finished_at = job.updated_at = time.time()

    async def status(self, job_id: str) -> dict[str, Any]:
        job = self.jobs.get(job_id)
        if job is None:
            raise ValueError(f"Unknown Colab job id: {job_id}")

        if job.state == "running":
            cell = await self._job_cell(job)
            if cell is None:
                job.state = "missing"
                job.error = "Job cell no longer exists in the notebook"
                job.finished_at = job.updated_at = time.time()
                task = self._tasks.get(job.job_id)
                if task is not None:
                    task.cancel()
            else:
                outputs = cell_outputs(cell)
                if outputs:
                    self._update_outputs(job, outputs)

        return self._job_dict(job)

    async def wait(
        self,
        job_id: str,
        timeout_seconds: float = DEFAULT_WAIT_TIMEOUT_SECONDS,
    ) -> dict[str, Any]:
        if not math.isfinite(timeout_seconds) or not (
            MIN_WAIT_TIMEOUT_SECONDS
            <= timeout_seconds
            <= MAX_WAIT_TIMEOUT_SECONDS
        ):
            raise ValueError(
                "timeout_seconds must be between "
                f"{MIN_WAIT_TIMEOUT_SECONDS:g} and "
                f"{MAX_WAIT_TIMEOUT_SECONDS:g}"
            )
        wait_started = time.monotonic()
        last_status = await self.status(job_id)
        wait_timed_out = False
        if last_status["state"] == "running":
            event = self._completion_events[job_id]
            try:
                await asyncio.wait_for(event.wait(), timeout=timeout_seconds)
            except asyncio.TimeoutError:
                wait_timed_out = True
            last_status = await self.status(job_id)
            if last_status["state"] != "running":
                wait_timed_out = False
        waited_seconds = time.monotonic() - wait_started
        return {
            **last_status,
            "timed_out": wait_timed_out,
            "wait_timed_out": wait_timed_out,
            "waited_seconds": waited_seconds,
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
        waited = await self.wait(started["job_id"], timeout_seconds)
        return {**waited, "start_result": started}

    def list_jobs(self) -> list[dict[str, Any]]:
        return [self._job_dict(job) for job in self.jobs.values()]

    async def mark_stale(self, reason: str) -> None:
        now = time.time()
        tasks: list[asyncio.Task[None]] = []
        for job in self.jobs.values():
            if job.state in {
                "finished",
                "error",
                "timed_out",
                "missing",
                "stale",
            }:
                continue
            job.state = "stale"
            job.error = reason
            job.finished_at = job.updated_at = now
            task = self._tasks.get(job.job_id)
            if task is not None:
                task.cancel()
                tasks.append(task)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        for job in self.jobs.values():
            if job.state == "stale":
                event = self._completion_events.get(job.job_id)
                if event is not None:
                    event.set()

    async def close(self) -> None:
        await self.mark_stale("Colab adapter shut down")
