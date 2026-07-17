from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from colab_runner.orchestrator import ColabJobOrchestrator, validate_session_name
from colab_runner.process import ProcessExecutionError, ProcessResult


class FakeCli:
    def __init__(
        self,
        *,
        fail_command: str | None = None,
        fail_stop: bool = False,
    ) -> None:
        self.calls: list[list[str]] = []
        self.fail_command = fail_command
        self.fail_stop = fail_stop

    async def run(
        self,
        arguments: list[str],
        *,
        timeout_seconds: float,
        cwd: Path | None = None,
    ) -> ProcessResult:
        del timeout_seconds, cwd
        self.calls.append(list(arguments))
        command = arguments[0]
        if command == self.fail_command or (command == "stop" and self.fail_stop):
            raise ProcessExecutionError(_result(returncode=1, stderr="boom"))
        if command == "exec":
            return _result(stdout="training complete\n")
        return _result()


@pytest.mark.asyncio
async def test_successful_job_uses_official_workflow_and_cleans_up(
    tmp_path: Path,
) -> None:
    script = tmp_path / "train.py"
    script.write_text("print('ok')\n", encoding="utf-8")
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("numpy\n", encoding="utf-8")
    artifacts = tmp_path / "artifacts"
    cli = FakeCli()
    events: list[str] = []

    result = await ColabJobOrchestrator(cli).run_job(
        script_path=str(script),
        accelerator="T4",
        requirements_file=str(requirements),
        remote_artifacts=["/content/checkpoints/model.bin"],
        artifact_dir=str(artifacts),
        max_runtime_seconds=120,
        session_name_prefix="unit-test",
        reporter=_report_to(events),
    )

    session = result["session_name"]
    assert result["ok"] is True
    assert result["state"] == "finished"
    assert result["cleanup"] == {
        "attempted": True,
        "succeeded": True,
        "error": None,
    }
    assert result["execution"]["stdout"] == "training complete\n"
    assert result["artifacts"] == [
        {
            "remote_path": "/content/checkpoints/model.bin",
            "local_path": str(artifacts / "checkpoints" / "model.bin"),
        }
    ]
    assert result["log_path"] == str(artifacts / f"{session}.ipynb")
    assert cli.calls[0] == ["new", "-s", session, "--gpu", "T4"]
    assert cli.calls[1] == [
        "install",
        "-s",
        session,
        "-r",
        str(requirements),
    ]
    assert cli.calls[2][0:3] == ["exec", "-s", session]
    assert cli.calls[3] == [
        "download",
        "-s",
        session,
        "/content/checkpoints/model.bin",
        str(artifacts / "checkpoints" / "model.bin"),
    ]
    assert cli.calls[-1] == ["stop", "-s", session]
    assert events == [
        "Provisioning T4 Colab session",
        "Installing declared workload dependencies",
        "Executing train.py",
        "Retrieving artifact 1/1",
        "Exporting replayable notebook log",
        "Stopping Colab session",
        "Colab job finished",
    ]


@pytest.mark.asyncio
async def test_execution_failure_still_stops_generated_session(tmp_path: Path) -> None:
    script = tmp_path / "fail.py"
    script.write_text("raise RuntimeError\n", encoding="utf-8")
    artifacts = tmp_path / "artifacts"
    cli = FakeCli(fail_command="exec")

    result = await ColabJobOrchestrator(cli).run_job(
        script_path=str(script),
        artifact_dir=str(artifacts),
        max_runtime_seconds=120,
    )

    assert result["ok"] is False
    assert result["state"] == "failed"
    assert result["failed_step"] == "execute"
    assert result["execution"]["stderr"] == "boom"
    assert result["cleanup"]["succeeded"] is True
    assert cli.calls[-1] == ["stop", "-s", result["session_name"]]


@pytest.mark.asyncio
async def test_cleanup_failure_is_never_reported_as_success(tmp_path: Path) -> None:
    script = tmp_path / "work.py"
    script.write_text("print('ok')\n", encoding="utf-8")
    cli = FakeCli(fail_stop=True)

    result = await ColabJobOrchestrator(cli).run_job(
        script_path=str(script),
        artifact_dir=str(tmp_path / "artifacts"),
        max_runtime_seconds=120,
    )

    assert result["ok"] is False
    assert result["state"] == "cleanup_failed"
    assert result["cleanup"]["attempted"] is True
    assert result["cleanup"]["succeeded"] is False
    assert "ProcessExecutionError" in result["cleanup"]["error"]


@pytest.mark.asyncio
async def test_tpu_value_is_mapped_to_official_cli_flag(tmp_path: Path) -> None:
    script = tmp_path / "work.py"
    script.write_text("print('ok')\n", encoding="utf-8")
    cli = FakeCli()

    result = await ColabJobOrchestrator(cli).run_job(
        script_path=str(script),
        accelerator="TPU_V6E1",
        artifact_dir=str(tmp_path / "artifacts"),
        export_log=False,
        max_runtime_seconds=120,
    )

    assert cli.calls[0] == [
        "new",
        "-s",
        result["session_name"],
        "--tpu",
        "v6e1",
    ]


@pytest.mark.parametrize(
    ("remote_artifact", "message"),
    [
        ("relative.txt", "beneath /content"),
        ("/etc/passwd", "beneath /content"),
        ("/content/../secret", "beneath /content"),
    ],
)
@pytest.mark.asyncio
async def test_remote_artifacts_are_restricted_to_content(
    tmp_path: Path, remote_artifact: str, message: str
) -> None:
    script = tmp_path / "work.py"
    script.write_text("print('ok')\n", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        await ColabJobOrchestrator(FakeCli()).run_job(
            script_path=str(script),
            remote_artifacts=[remote_artifact],
            artifact_dir=str(tmp_path / "artifacts"),
            max_runtime_seconds=120,
        )


@pytest.mark.parametrize("session_name", ["trainer_1", "Team.Run-2", "9-job"])
def test_existing_official_cli_session_names_are_accepted(session_name: str) -> None:
    assert validate_session_name(session_name) == session_name


@pytest.mark.asyncio
async def test_cancelled_job_attempts_cleanup(tmp_path: Path) -> None:
    script = tmp_path / "work.py"
    script.write_text("print('ok')\n", encoding="utf-8")

    class CancellingCli(FakeCli):
        async def run(self, arguments: list[str], **kwargs: Any) -> ProcessResult:
            self.calls.append(list(arguments))
            if arguments[0] == "exec":
                raise asyncio.CancelledError
            return _result()

    cli = CancellingCli()
    with pytest.raises(asyncio.CancelledError):
        await ColabJobOrchestrator(cli).run_job(
            script_path=str(script),
            artifact_dir=str(tmp_path / "artifacts"),
            max_runtime_seconds=120,
        )

    assert cli.calls[-1][0] == "stop"


@pytest.mark.asyncio
async def test_cancelled_cleanup_progress_does_not_skip_stop(tmp_path: Path) -> None:
    script = tmp_path / "work.py"
    script.write_text("print('ok')\n", encoding="utf-8")
    cli = FakeCli()

    async def cancelling_reporter(message: str) -> None:
        if message == "Stopping Colab session":
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await ColabJobOrchestrator(cli).run_job(
            script_path=str(script),
            artifact_dir=str(tmp_path / "artifacts"),
            max_runtime_seconds=120,
            reporter=cancelling_reporter,
        )

    assert cli.calls[-1][0] == "stop"


def _report_to(events: list[str]):
    async def report(message: str) -> None:
        events.append(message)

    return report


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
