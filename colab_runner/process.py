from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final


DEFAULT_CAPTURE_BYTES: Final = 64 * 1024
_TRUNCATION_MARKER: Final = b"\n... output truncated by colab-runner ...\n"


@dataclass(frozen=True)
class ProcessResult:
    returncode: int
    stdout: str
    stderr: str
    stdout_truncated: bool
    stderr_truncated: bool
    elapsed_seconds: float


class ProcessExecutionError(RuntimeError):
    def __init__(self, result: ProcessResult) -> None:
        self.result = result
        detail = result.stderr.strip() or result.stdout.strip()
        if detail:
            detail = detail[-1000:]
            message = f"Colab CLI exited with code {result.returncode}: {detail}"
        else:
            message = f"Colab CLI exited with code {result.returncode}"
        super().__init__(message)


class ProcessExecutionTimeout(RuntimeError):
    def __init__(self, timeout_seconds: float, result: ProcessResult) -> None:
        self.timeout_seconds = timeout_seconds
        self.result = result
        super().__init__(f"Colab CLI exceeded the {timeout_seconds:g}s deadline")


class ProcessRunner:
    """Run argv-only subprocesses with hard deadlines and bounded output."""

    def __init__(self, capture_bytes: int = DEFAULT_CAPTURE_BYTES) -> None:
        if capture_bytes < 1024:
            raise ValueError("capture_bytes must be at least 1024")
        self.capture_bytes = capture_bytes

    async def run(
        self,
        argv: list[str],
        *,
        timeout_seconds: float,
        cwd: Path | None = None,
    ) -> ProcessResult:
        if not argv or any(not isinstance(item, str) or not item for item in argv):
            raise ValueError("argv must contain non-empty strings")
        if not 0 < timeout_seconds <= 86_400:
            raise ValueError("timeout_seconds must be between 0 and 86400")

        started = time.monotonic()
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                cwd=str(cwd) if cwd is not None else None,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Required executable was not found: {argv[0]}"
            ) from exc

        assert process.stdout is not None
        assert process.stderr is not None
        stdout_task = asyncio.create_task(
            _read_bounded(process.stdout, self.capture_bytes)
        )
        stderr_task = asyncio.create_task(
            _read_bounded(process.stderr, self.capture_bytes)
        )

        timed_out = False
        try:
            await asyncio.wait_for(process.wait(), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            timed_out = True
            await _terminate(process)
        except asyncio.CancelledError:
            await asyncio.shield(_terminate(process))
            await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
            raise

        stdout_data, stderr_data = await asyncio.gather(stdout_task, stderr_task)
        result = ProcessResult(
            returncode=process.returncode if process.returncode is not None else -1,
            stdout=stdout_data[0],
            stderr=stderr_data[0],
            stdout_truncated=stdout_data[1],
            stderr_truncated=stderr_data[1],
            elapsed_seconds=time.monotonic() - started,
        )
        if timed_out:
            raise ProcessExecutionTimeout(timeout_seconds, result)
        if result.returncode != 0:
            raise ProcessExecutionError(result)
        return result


async def _terminate(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    try:
        process.terminate()
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(process.wait(), timeout=3)
    except asyncio.TimeoutError:
        try:
            process.kill()
        except ProcessLookupError:
            return
        await process.wait()


async def _read_bounded(
    stream: asyncio.StreamReader, limit: int
) -> tuple[str, bool]:
    head_limit = limit // 2
    tail_limit = limit - head_limit
    head = bytearray()
    tail = bytearray()
    total = 0

    while chunk := await stream.read(8192):
        total += len(chunk)
        if len(head) < head_limit:
            take = min(head_limit - len(head), len(chunk))
            head.extend(chunk[:take])
            chunk = chunk[take:]
        if chunk:
            tail.extend(chunk)
            if len(tail) > tail_limit:
                del tail[: len(tail) - tail_limit]

    truncated = total > len(head) + len(tail)
    raw = bytes(head)
    if truncated:
        raw += _TRUNCATION_MARKER
    raw += bytes(tail)
    return raw.decode("utf-8", errors="replace"), truncated
