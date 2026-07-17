from __future__ import annotations

import asyncio
import os
import uuid
from pathlib import Path

import nbformat
import pytest

from colab_runner.cli import ColabCli
from colab_runner.orchestrator import ColabJobOrchestrator
from colab_runner.sessions import ColabSessionManager


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


@pytest.mark.skipif(
    not _live_enabled("COLAB_RUNNER_LIVE_SESSION_TEST"),
    reason=(
        "set COLAB_RUNNER_LIVE_SESSION_TEST=1 to allocate a reusable real "
        "Colab CPU runtime"
    ),
)
@pytest.mark.asyncio
async def test_reusable_live_session_preserves_state_and_runs_selected_cell(
    tmp_path: Path,
) -> None:
    setup_script = tmp_path / "state_setup.py"
    setup_script.write_text(
        "live_value = 40\nprint('COLAB_RUNNER_SETUP_READY')\n",
        encoding="utf-8",
    )
    notebook_path = tmp_path / "state_steps.ipynb"
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_markdown_cell("State check"),
            nbformat.v4.new_code_cell(
                "from pathlib import Path\n"
                "result = live_value + 2\n"
                "target = Path('/content/colab-runner-live/state.txt')\n"
                "target.parent.mkdir(parents=True, exist_ok=True)\n"
                "target.write_text(f'{result}\\n', encoding='utf-8')\n"
                "print(f'COLAB_RUNNER_STATE={result}')\n",
                id="state-check",
            ),
        ]
    )
    nbformat.write(notebook, notebook_path)

    prefix = f"live-state-{uuid.uuid4().hex[:6]}"
    artifact_dir = tmp_path / "artifacts"
    log_path = tmp_path / "session.ipynb"
    cli = ColabCli()
    manager = ColabSessionManager(cli)
    session_name: str | None = None
    try:
        started = await manager.start_session(
            idle_timeout_seconds=300,
            setup_timeout_seconds=600,
            session_name_prefix=prefix,
        )
        assert started["ok"] is True, started
        session_name = started["session_name"]

        setup = await manager.execute(
            session_name=session_name,
            script_path=str(setup_script),
            timeout_seconds=300,
        )
        assert setup["ok"] is True, setup
        assert "COLAB_RUNNER_SETUP_READY" in setup["execution"]["stdout"]

        selected = await manager.execute(
            session_name=session_name,
            script_path=str(notebook_path),
            cell_id="state-check",
            timeout_seconds=300,
            background=True,
            write_output_to_notebook=True,
        )
        assert selected["ok"] is True, selected
        execution_id = selected["execution_id"]
        deadline = asyncio.get_running_loop().time() + 300
        while True:
            cell_status = manager.cell_status(execution_id)
            if cell_status["terminal"]:
                break
            if asyncio.get_running_loop().time() >= deadline:
                pytest.fail("selected-cell execution did not finish within 300 seconds")
            await asyncio.sleep(0.25)
        assert cell_status["state"] == "finished", cell_status
        assert "COLAB_RUNNER_STATE=42" in cell_status["execution"]["stdout"]
        assert cell_status["output_writeback"]["written"] is True
        written_cell = nbformat.read(notebook_path, as_version=4).cells[1]
        assert "COLAB_RUNNER_STATE=42" in written_cell.outputs[0].text

        downloaded = await manager.download_artifact(
            session_name=session_name,
            remote_path="/content/colab-runner-live/state.txt",
            artifact_dir=str(artifact_dir),
        )
        assert downloaded["ok"] is True, downloaded
        assert Path(downloaded["local_path"]).read_text(encoding="utf-8") == "42\n"

        exported = await manager.export_log(
            session_name=session_name,
            output_path=str(log_path),
        )
        assert exported["ok"] is True, exported
        assert log_path.is_file()
    finally:
        if session_name is not None and manager.is_managed(session_name):
            await manager.stop_session(session_name)
        await manager.close()

    sessions = await cli.run(["sessions"], timeout_seconds=30)
    assert prefix not in sessions.stdout
