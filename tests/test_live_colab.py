from __future__ import annotations

import asyncio
import os
import uuid
from pathlib import Path

import pytest

from colab_runner.cli import ColabCli
from colab_runner.orchestrator import ColabJobOrchestrator


pytestmark = pytest.mark.live


def _live_enabled(variable: str) -> bool:
    return os.environ.get(variable, "").strip().lower() in {"1", "true", "yes"}


@pytest.mark.skipif(
    not _live_enabled("COLAB_RUNNER_LIVE_TEST"),
    reason="set COLAB_RUNNER_LIVE_TEST=1 to allocate a real Colab CPU runtime",
)
@pytest.mark.asyncio
async def test_authenticated_cpu_job_downloads_artifact_exports_log_and_stops(
    tmp_path: Path,
) -> None:
    script = tmp_path / "live_smoke.py"
    script.write_text(
        "from pathlib import Path\n"
        "target = Path('/content/colab-runner-live/result.txt')\n"
        "target.parent.mkdir(parents=True, exist_ok=True)\n"
        "target.write_text('colab-runner-live-ok\\n', encoding='utf-8')\n"
        "print('COLAB_RUNNER_LIVE_OK')\n",
        encoding="utf-8",
    )
    artifact_dir = tmp_path / "artifacts"
    cli = ColabCli()

    result = await ColabJobOrchestrator(cli).run_job(
        script_path=str(script),
        artifact_dir=str(artifact_dir),
        remote_artifacts=["/content/colab-runner-live/result.txt"],
        max_runtime_seconds=600,
        session_name_prefix="live-smoke",
    )

    assert result["ok"] is True, result
    assert result["state"] == "finished"
    assert result["cleanup"]["succeeded"] is True
    assert "COLAB_RUNNER_LIVE_OK" in result["execution"]["stdout"]
    assert (
        artifact_dir / "colab-runner-live" / "result.txt"
    ).read_text(encoding="utf-8") == "colab-runner-live-ok\n"
    assert Path(result["log_path"]).is_file()

    sessions = await cli.run(["sessions"], timeout_seconds=30)
    assert result["session_name"] not in sessions.stdout


@pytest.mark.skipif(
    not _live_enabled("COLAB_RUNNER_LIVE_CANCEL_TEST"),
    reason=(
        "set COLAB_RUNNER_LIVE_CANCEL_TEST=1 to allocate and cancel a real "
        "Colab CPU runtime"
    ),
)
@pytest.mark.asyncio
async def test_cancelled_live_job_releases_its_generated_session(
    tmp_path: Path,
) -> None:
    script = tmp_path / "live_cancel.py"
    script.write_text(
        "import time\n"
        "print('COLAB_RUNNER_CANCEL_READY', flush=True)\n"
        "time.sleep(300)\n",
        encoding="utf-8",
    )
    prefix = f"live-cancel-{uuid.uuid4().hex[:6]}"
    executing = asyncio.Event()
    cli = ColabCli()

    async def reporter(message: str) -> None:
        if message.startswith("Executing "):
            executing.set()

    task = asyncio.create_task(
        ColabJobOrchestrator(cli).run_job(
            script_path=str(script),
            artifact_dir=str(tmp_path / "artifacts"),
            export_log=False,
            max_runtime_seconds=600,
            session_name_prefix=prefix,
            reporter=reporter,
        )
    )
    event_wait = asyncio.create_task(executing.wait())
    try:
        done, _ = await asyncio.wait(
            {task, event_wait},
            timeout=180,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if task in done:
            pytest.fail(f"live job ended before cancellation: {task.result()}")
        assert event_wait in done, "live job did not reach execution within 180 seconds"

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        event_wait.cancel()
        if not task.done():
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    sessions = await cli.run(["sessions"], timeout_seconds=30)
    assert prefix not in sessions.stdout
