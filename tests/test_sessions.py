from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from typing import Any

import nbformat
import pytest

from colab_runner.process import (
    ProcessExecutionError,
    ProcessExecutionTimeout,
    ProcessResult,
)
from colab_runner.sessions import ColabSessionManager


class FakeCli:
    def __init__(
        self,
        *,
        fail_exec: bool = False,
        timeout_exec: bool = False,
        fail_stop: bool = False,
        fail_command: str | None = None,
        exec_delay: float = 0,
        exec_stdout: str = "cell complete\n",
    ) -> None:
        self.calls: list[list[str]] = []
        self.executed_sources: list[str] = []
        self.fail_exec = fail_exec
        self.timeout_exec = timeout_exec
        self.fail_stop = fail_stop
        self.fail_command = fail_command
        self.exec_delay = exec_delay
        self.exec_stdout = exec_stdout
        self.active_execs = 0
        self.max_active_execs = 0

    async def run(
        self,
        arguments: list[str],
        *,
        timeout_seconds: float,
        cwd: Path | None = None,
    ) -> ProcessResult:
        del cwd
        self.calls.append(list(arguments))
        command = arguments[0]
        if command == self.fail_command:
            raise ProcessExecutionError(
                _result(returncode=1, stderr=f"{command} failed\n")
            )
        if command == "exec":
            source_path = Path(arguments[-1])
            self.executed_sources.append(source_path.read_text(encoding="utf-8"))
            self.active_execs += 1
            self.max_active_execs = max(self.max_active_execs, self.active_execs)
            try:
                if self.exec_delay:
                    await asyncio.sleep(self.exec_delay)
                if self.timeout_exec:
                    raise ProcessExecutionTimeout(
                        timeout_seconds,
                        _result(stderr="execution timed out\n"),
                    )
                if self.fail_exec:
                    raise ProcessExecutionError(
                        _result(returncode=1, stderr="user code failed\n")
                    )
                return _result(stdout=self.exec_stdout)
            finally:
                self.active_execs -= 1
        if command == "download":
            Path(arguments[-1]).write_text("artifact\n", encoding="utf-8")
        if command == "log":
            Path(arguments[-1]).write_text("{}\n", encoding="utf-8")
        if command == "stop" and self.fail_stop:
            raise ProcessExecutionError(
                _result(returncode=1, stderr="cleanup failed\n")
            )
        return _result()


@pytest.mark.asyncio
async def test_session_setup_failure_releases_partial_allocation(
    tmp_path: Path,
) -> None:
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("numpy\n", encoding="utf-8")
    cli = FakeCli(fail_command="install")
    manager = _manager(cli)

    result = await manager.start_session(
        requirements_file=str(requirements),
        idle_timeout_seconds=5,
    )

    assert result["ok"] is False
    assert result["failed_step"] == "install"
    assert result["cleanup"]["succeeded"] is True
    assert cli.calls[-1] == ["stop", "-s", result["session_name"]]
    assert manager.leases() == []
    await manager.close()


@pytest.mark.asyncio
async def test_reusable_session_runs_script_and_selected_notebook_cell(
    tmp_path: Path,
) -> None:
    script = tmp_path / "setup.py"
    script.write_text("shared_value = 40\n", encoding="utf-8")
    notebook = tmp_path / "steps.ipynb"
    _write_notebook(notebook)
    cli = FakeCli()
    manager = _manager(cli)

    started = await manager.start_session(
        accelerator="T4",
        packages=["numpy"],
        idle_timeout_seconds=5,
        session_name_prefix="stateful-test",
    )
    session_name = started["session_name"]

    first = await manager.execute(
        session_name=session_name,
        script_path=str(script),
        timeout_seconds=30,
    )
    second = await manager.execute(
        session_name=session_name,
        script_path=str(notebook),
        cell_id="target-cell",
        timeout_seconds=30,
    )

    assert first["ok"] is True
    assert second["ok"] is True
    assert second["selected_cell"] == {
        "cell_index": 2,
        "cell_id": "target-cell",
        "source_bytes": len(b"print(shared_value + 2)"),
        "source_sha256": hashlib.sha256(
            b"print(shared_value + 2)"
        ).hexdigest(),
    }
    assert cli.executed_sources == [
        "shared_value = 40\n",
        "print(shared_value + 2)\n",
    ]
    assert all(
        call[2] == session_name
        for call in cli.calls
        if call[0] in {"install", "exec"}
    )
    assert cli.calls[0] == ["new", "-s", session_name, "--gpu", "T4"]
    assert cli.calls[1] == ["install", "-s", session_name, "numpy"]

    stopped = await manager.stop_session(session_name)
    assert stopped["ok"] is True
    assert manager.leases() == []
    await manager.close()


@pytest.mark.asyncio
async def test_background_cell_execution_is_observable_and_writes_output(
    tmp_path: Path,
) -> None:
    notebook = tmp_path / "steps.ipynb"
    _write_notebook(notebook)
    cli = FakeCli(exec_delay=0.03, exec_stdout="answer=42\n")
    manager = _manager(cli)
    started = await manager.start_session(idle_timeout_seconds=5)
    session_name = started["session_name"]

    queued = await manager.execute(
        session_name=session_name,
        script_path=str(notebook),
        cell_id="target-cell",
        timeout_seconds=30,
        background=True,
        write_output_to_notebook=True,
    )
    execution_id = queued["execution_id"]

    assert queued["state"] == "queued"
    assert queued["cell_execution"]["terminal"] is False
    await _wait_until(lambda: manager.cell_status(execution_id)["terminal"])
    status = manager.cell_status(execution_id)
    latest = manager.latest_cell_execution(
        notebook_path=str(notebook),
        cell_index=2,
        cell_id="target-cell",
        session_name=session_name,
    )

    assert status["state"] == "finished"
    assert status["execution"]["stdout"] == "answer=42\n"
    assert status["output_writeback"]["written"] is True
    assert latest is not None
    assert latest["execution_id"] == execution_id
    saved = nbformat.read(notebook, as_version=4).cells[2]
    assert saved.outputs[0].text == "answer=42\n"
    assert saved.metadata["colab_runner"]["execution_id"] == execution_id

    await manager.stop_session(session_name)
    await manager.close()


@pytest.mark.asyncio
async def test_changed_cell_rejects_background_output_writeback(
    tmp_path: Path,
) -> None:
    notebook = tmp_path / "steps.ipynb"
    _write_notebook(notebook)
    cli = FakeCli(exec_delay=0.05)
    manager = _manager(cli)
    started = await manager.start_session(idle_timeout_seconds=5)
    session_name = started["session_name"]
    queued = await manager.execute(
        session_name=session_name,
        script_path=str(notebook),
        cell_id="target-cell",
        background=True,
        write_output_to_notebook=True,
    )
    await _wait_until(lambda: cli.active_execs == 1)
    cells = await manager.notebooks.inspect(str(notebook))
    await manager.notebooks.update_cell(
        str(notebook),
        cell_id="target-cell",
        cell_index=None,
        source="print('new source')",
        expected_source_sha256=cells["cells"][2]["source_sha256"],
    )
    await _wait_until(
        lambda: manager.cell_status(queued["execution_id"])["terminal"]
    )

    status = manager.cell_status(queued["execution_id"])
    assert status["state"] == "finished"
    assert status["output_writeback"]["state"] == "conflict"
    assert status["output_writeback"]["written"] is False
    saved = nbformat.read(notebook, as_version=4).cells[2]
    assert saved.source == "print('new source')"
    assert saved.outputs == []

    await manager.stop_session(session_name)
    await manager.close()


@pytest.mark.asyncio
async def test_stopping_session_interrupts_background_cell(tmp_path: Path) -> None:
    notebook = tmp_path / "steps.ipynb"
    _write_notebook(notebook)
    cli = FakeCli(exec_delay=10)
    manager = _manager(cli)
    started = await manager.start_session(idle_timeout_seconds=5)
    session_name = started["session_name"]
    queued = await manager.execute(
        session_name=session_name,
        script_path=str(notebook),
        cell_id="target-cell",
        background=True,
        write_output_to_notebook=True,
    )
    await _wait_until(lambda: cli.active_execs == 1)

    stopped = await asyncio.wait_for(
        manager.stop_session(session_name),
        timeout=1,
    )
    status = manager.cell_status(queued["execution_id"])

    assert stopped["ok"] is True
    assert status["state"] == "interrupted"
    assert status["output_writeback"]["state"] == "skipped"
    assert not manager.is_managed(session_name)
    await manager.close()


@pytest.mark.asyncio
async def test_cell_status_bounds_large_execution_output(tmp_path: Path) -> None:
    notebook = tmp_path / "steps.ipynb"
    _write_notebook(notebook)
    cli = FakeCli(exec_stdout="x" * 20_000)
    manager = _manager(cli)
    started = await manager.start_session(idle_timeout_seconds=5)
    session_name = started["session_name"]

    result = await manager.execute(
        session_name=session_name,
        script_path=str(notebook),
        cell_id="target-cell",
    )
    status = manager.cell_status(result["execution_id"])

    assert status["state"] == "finished"
    assert status["execution"]["stdout_bytes"] <= 8 * 1024
    assert status["execution"]["stdout_truncated"] is True
    await manager.stop_session(session_name)
    await manager.close()


@pytest.mark.asyncio
async def test_background_execution_requires_a_selected_notebook_cell(
    tmp_path: Path,
) -> None:
    script = tmp_path / "step.py"
    script.write_text("print('step')\n", encoding="utf-8")
    manager = _manager(FakeCli())
    started = await manager.start_session(idle_timeout_seconds=5)

    with pytest.raises(ValueError, match="selected notebook cell"):
        await manager.execute(
            session_name=started["session_name"],
            script_path=str(script),
            background=True,
        )

    await manager.close()


@pytest.mark.asyncio
async def test_same_session_serializes_concurrent_execution(tmp_path: Path) -> None:
    first_script = tmp_path / "first.py"
    second_script = tmp_path / "second.py"
    first_script.write_text("print('first')\n", encoding="utf-8")
    second_script.write_text("print('second')\n", encoding="utf-8")
    cli = FakeCli(exec_delay=0.03)
    manager = _manager(cli)
    started = await manager.start_session(idle_timeout_seconds=5)
    session_name = started["session_name"]

    await asyncio.gather(
        manager.execute(
            session_name=session_name,
            script_path=str(first_script),
            timeout_seconds=30,
        ),
        manager.execute(
            session_name=session_name,
            script_path=str(second_script),
            timeout_seconds=30,
        ),
    )

    assert cli.max_active_execs == 1
    await manager.stop_session(session_name)
    await manager.close()


@pytest.mark.asyncio
async def test_execution_timeout_releases_reusable_session(tmp_path: Path) -> None:
    script = tmp_path / "slow.py"
    script.write_text("while True: pass\n", encoding="utf-8")
    cli = FakeCli(timeout_exec=True)
    manager = _manager(cli)
    started = await manager.start_session(idle_timeout_seconds=5)

    result = await manager.execute(
        session_name=started["session_name"],
        script_path=str(script),
        timeout_seconds=1,
    )

    assert result["ok"] is False
    assert result["state"] == "timed_out"
    assert result["cleanup"]["succeeded"] is True
    assert manager.leases() == []
    assert cli.calls[-1] == ["stop", "-s", started["session_name"]]
    await manager.close()


@pytest.mark.asyncio
async def test_user_code_failure_keeps_session_available(tmp_path: Path) -> None:
    script = tmp_path / "fail.py"
    script.write_text("raise RuntimeError('expected')\n", encoding="utf-8")
    cli = FakeCli(fail_exec=True)
    manager = _manager(cli)
    started = await manager.start_session(idle_timeout_seconds=5)
    session_name = started["session_name"]

    result = await manager.execute(
        session_name=session_name,
        script_path=str(script),
        timeout_seconds=30,
    )

    assert result["ok"] is False
    assert result["state"] == "failed"
    assert result["lease"]["state"] == "ready"
    assert manager.is_managed(session_name)
    await manager.stop_session(session_name)
    await manager.close()


@pytest.mark.asyncio
async def test_idle_lease_automatically_stops_session() -> None:
    cli = FakeCli()
    manager = _manager(cli)
    started = await manager.start_session(idle_timeout_seconds=0.03)
    session_name = started["session_name"]

    await _wait_until(lambda: not manager.is_managed(session_name))

    assert ["stop", "-s", session_name] in cli.calls
    await manager.close()


@pytest.mark.asyncio
async def test_failed_idle_cleanup_is_retried() -> None:
    cli = FakeCli(fail_stop=True)
    manager = _manager(cli)
    started = await manager.start_session(idle_timeout_seconds=0.03)
    session_name = started["session_name"]

    await _wait_until(
        lambda: any(call[:2] == ["stop", "-s"] for call in cli.calls)
    )
    assert manager.lease_status(session_name)["state"] == "cleanup_failed"

    cli.fail_stop = False
    await _wait_until(lambda: not manager.is_managed(session_name))

    stop_calls = [call for call in cli.calls if call[:2] == ["stop", "-s"]]
    assert len(stop_calls) >= 2
    await manager.close()


@pytest.mark.asyncio
async def test_cancelled_execution_releases_reusable_session(
    tmp_path: Path,
) -> None:
    script = tmp_path / "slow.py"
    script.write_text("import time; time.sleep(300)\n", encoding="utf-8")
    cli = FakeCli(exec_delay=10)
    manager = _manager(cli)
    started = await manager.start_session(idle_timeout_seconds=5)
    session_name = started["session_name"]
    execution = asyncio.create_task(
        manager.execute(
            session_name=session_name,
            script_path=str(script),
            timeout_seconds=30,
        )
    )
    await _wait_until(lambda: cli.active_execs == 1)

    execution.cancel()
    with pytest.raises(asyncio.CancelledError):
        await execution

    assert not manager.is_managed(session_name)
    assert cli.calls[-1] == ["stop", "-s", session_name]
    await manager.close()


@pytest.mark.asyncio
async def test_explicit_stop_preempts_active_execution(tmp_path: Path) -> None:
    script = tmp_path / "slow.py"
    script.write_text("import time; time.sleep(300)\n", encoding="utf-8")
    cli = FakeCli(exec_delay=10)
    manager = _manager(cli)
    started = await manager.start_session(idle_timeout_seconds=5)
    session_name = started["session_name"]
    execution = asyncio.create_task(
        manager.execute(
            session_name=session_name,
            script_path=str(script),
            timeout_seconds=30,
        )
    )
    await _wait_until(lambda: cli.active_execs == 1)

    stopped = await asyncio.wait_for(
        manager.stop_session(session_name),
        timeout=1,
    )
    executed = await execution

    assert stopped["ok"] is True
    assert executed["state"] == "interrupted"
    assert executed["cleanup"]["succeeded"] is True
    assert not manager.is_managed(session_name)
    await manager.close()


@pytest.mark.asyncio
async def test_shutdown_releases_all_owned_sessions() -> None:
    cli = FakeCli()
    manager = _manager(cli)
    first = await manager.start_session(idle_timeout_seconds=5)
    second = await manager.start_session(idle_timeout_seconds=5)

    await manager.close()

    stopped = {
        call[2]
        for call in cli.calls
        if call[:2] == ["stop", "-s"]
    }
    assert stopped == {first["session_name"], second["session_name"]}
    assert manager.leases() == []


@pytest.mark.asyncio
async def test_renew_download_and_export_keep_lease_active(tmp_path: Path) -> None:
    cli = FakeCli()
    manager = _manager(cli)
    started = await manager.start_session(idle_timeout_seconds=5)
    session_name = started["session_name"]

    renewed = await manager.renew_session(
        session_name=session_name,
        idle_timeout_seconds=10,
    )
    downloaded = await manager.download_artifact(
        session_name=session_name,
        remote_path="/content/results/data.txt",
        artifact_dir=str(tmp_path / "artifacts"),
    )
    exported = await manager.export_log(
        session_name=session_name,
        output_path=str(tmp_path / "logs" / "session.ipynb"),
    )

    assert renewed["lease"]["idle_timeout_seconds"] == 10
    assert Path(downloaded["local_path"]).read_text(encoding="utf-8") == "artifact\n"
    assert Path(exported["log_path"]).is_file()
    assert manager.is_managed(session_name)
    await manager.stop_session(session_name)
    await manager.close()


@pytest.mark.parametrize(
    ("cell_index", "cell_id", "message"),
    [
        (0, "target-cell", "not both"),
        (-1, None, "non-negative"),
        (99, None, "outside"),
        (None, "missing", "not found"),
        (0, None, "not code"),
    ],
)
@pytest.mark.asyncio
async def test_invalid_notebook_cell_selection_is_rejected(
    tmp_path: Path,
    cell_index: int | None,
    cell_id: str | None,
    message: str,
) -> None:
    notebook = tmp_path / "steps.ipynb"
    _write_notebook(notebook)
    manager = _manager(FakeCli())
    started = await manager.start_session(idle_timeout_seconds=5)

    with pytest.raises(ValueError, match=message):
        await manager.execute(
            session_name=started["session_name"],
            script_path=str(notebook),
            cell_index=cell_index,
            cell_id=cell_id,
        )

    await manager.close()


def _manager(cli: FakeCli) -> ColabSessionManager:
    return ColabSessionManager(
        cli,
        cleanup_timeout_seconds=1,
        minimum_idle_timeout_seconds=0.01,
        maximum_idle_timeout_seconds=60,
    )


def _write_notebook(path: Path) -> None:
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_markdown_cell("Setup"),
            nbformat.v4.new_code_cell("shared_value = 40", id="setup-cell"),
            nbformat.v4.new_code_cell(
                "print(shared_value + 2)",
                id="target-cell",
            ),
        ]
    )
    nbformat.write(notebook, path)


async def _wait_until(predicate: Any, timeout: float = 1) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError("condition was not met before timeout")
        await asyncio.sleep(0.01)


def _result(
    *,
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
) -> ProcessResult:
    return ProcessResult(
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        stdout_truncated=False,
        stderr_truncated=False,
        elapsed_seconds=0.01,
    )
