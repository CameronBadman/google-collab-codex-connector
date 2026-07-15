from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum


class ActivityPhase(StrEnum):
    WAITING_FOR_BROWSER = "waiting_for_browser"
    INITIALIZING_RUNTIME = "initializing_runtime"
    INSPECTING_NOTEBOOK = "inspecting_notebook"
    PREPARING_CELL = "preparing_cell"
    EXECUTING = "executing"
    WAITING = "waiting"
    RECOVERING = "recovering"
    READING_ARTIFACT = "reading_artifact"
    FINISHED = "finished"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    INTERRUPTED = "interrupted"


@dataclass(frozen=True)
class ActivityEvent:
    phase: ActivityPhase
    message: str
    job_id: str | None = None


ActivityReporter = Callable[[ActivityEvent], Awaitable[None]]


async def report_activity(
    reporter: ActivityReporter | None,
    phase: ActivityPhase,
    message: str,
    *,
    job_id: str | None = None,
) -> None:
    """Publish optional telemetry without allowing it to break execution."""
    if reporter is None:
        return
    try:
        await reporter(ActivityEvent(phase=phase, message=message, job_id=job_id))
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logging.debug(
            "Colab activity notification failed error_type=%s",
            type(exc).__name__,
        )
