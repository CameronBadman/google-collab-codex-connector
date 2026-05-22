from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from .session import ColabSessionManager
from .tools import first_json_object, serialize_tool_result

POLL_INTERVAL_SECONDS = 1.0


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
    finished_at: float | None = None
    outputs: list[Any] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ColabJobManager:
    def __init__(self, session: ColabSessionManager):
        self.session = session
        self.jobs: dict[str, ColabJob] = {}
        self._lock = asyncio.Lock()

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

    async def start_python(self, code: str, language: str = "python") -> dict[str, Any]:
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

        job = ColabJob(
            job_id=uuid.uuid4().hex,
            cell_id=cell_id,
            code=code,
            state="running",
            started_at=time.time(),
        )
        async with self._lock:
            self.jobs[job.job_id] = job

        run_result = await self.session.call_tool("run_code_cell", {"cellId": cell_id})
        run_data = result_data(run_result)
        outputs = run_data.get("outputs", [])
        if isinstance(outputs, list) and outputs:
            self._finish_from_outputs(job, outputs)
        return {
            **job.to_dict(),
            "add_result": serialize_tool_result(add_result),
            "run_result": serialize_tool_result(run_result),
        }

    def _finish_from_outputs(self, job: ColabJob, outputs: list[Any]) -> None:
        job.outputs = outputs
        job.error = output_error(outputs)
        job.state = "error" if output_has_error(outputs) else "finished"
        job.finished_at = time.time()

    async def status(self, job_id: str) -> dict[str, Any]:
        job = self.jobs.get(job_id)
        if job is None:
            raise ValueError(f"Unknown Colab job id: {job_id}")

        if job.state == "running":
            cell = await self._job_cell(job)
            if cell is None:
                job.state = "missing"
                job.error = "Job cell no longer exists in the notebook"
                job.finished_at = time.time()
            else:
                outputs = cell_outputs(cell)
                if outputs:
                    self._finish_from_outputs(job, outputs)

        return job.to_dict()

    async def wait(self, job_id: str, timeout_seconds: float = 300.0) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        last_status = await self.status(job_id)
        while last_status["state"] == "running":
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return {**last_status, "timed_out": True}
            await asyncio.sleep(min(POLL_INTERVAL_SECONDS, remaining))
            last_status = await self.status(job_id)
        return {**last_status, "timed_out": False}

    async def run_python_wait(
        self, code: str, timeout_seconds: float = 300.0
    ) -> dict[str, Any]:
        started = await self.start_python(code)
        waited = await self.wait(started["job_id"], timeout_seconds)
        return {**waited, "start_result": started}

    def list_jobs(self) -> list[dict[str, Any]]:
        return [job.to_dict() for job in self.jobs.values()]
