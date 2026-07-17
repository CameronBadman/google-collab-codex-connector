from __future__ import annotations

import asyncio
import re
import time
import uuid
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Final

from .cli import ColabCli
from .process import (
    ProcessExecutionError,
    ProcessExecutionTimeout,
    ProcessResult,
)


ActivityReporter = Callable[[str], Awaitable[None]]

SUPPORTED_ACCELERATORS: Final = frozenset(
    {"CPU", "T4", "L4", "G4", "H100", "A100", "TPU_V5E1", "TPU_V6E1"}
)
_SESSION_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_PACKAGE_LIMIT: Final = 64
_ARTIFACT_LIMIT: Final = 32


class ColabJobOrchestrator:
    """Provision, execute, collect, and always release a Colab session."""

    def __init__(
        self,
        cli: ColabCli | Any | None = None,
        *,
        cleanup_timeout_seconds: float = 60,
    ) -> None:
        self.cli = cli or ColabCli()
        self.cleanup_timeout_seconds = cleanup_timeout_seconds

    async def run_job(
        self,
        *,
        script_path: str,
        accelerator: str = "CPU",
        packages: Sequence[str] = (),
        requirements_file: str | None = None,
        remote_artifacts: Sequence[str] = (),
        artifact_dir: str | None = None,
        export_log: bool = True,
        max_runtime_seconds: float = 1800,
        session_name_prefix: str = "codex",
        reporter: ActivityReporter | None = None,
    ) -> dict[str, Any]:
        request = _validate_request(
            script_path=script_path,
            accelerator=accelerator,
            packages=packages,
            requirements_file=requirements_file,
            remote_artifacts=remote_artifacts,
            artifact_dir=artifact_dir,
            export_log=export_log,
            max_runtime_seconds=max_runtime_seconds,
            session_name_prefix=session_name_prefix,
        )
        session_name = f"{request['session_prefix']}-{uuid.uuid4().hex[:8]}"
        started = time.monotonic()
        deadline = asyncio.get_running_loop().time() + max_runtime_seconds
        state = "running"
        failed_step: str | None = None
        error: str | None = None
        execution_result: ProcessResult | None = None
        downloaded: list[dict[str, str]] = []
        log_path: str | None = None
        cleanup_attempted = False
        cleanup_succeeded = False
        cleanup_error: str | None = None
        allocation_attempted = False
        cancelled = False

        try:
            failed_step = "provision"
            allocation_attempted = True
            await _report(
                reporter,
                f"Provisioning {accelerator.replace('_', ' ')} Colab session",
            )
            await self.cli.run(
                _new_arguments(session_name, accelerator),
                timeout_seconds=_remaining(deadline),
            )

            if request["packages"] or request["requirements_file"] is not None:
                failed_step = "install"
                await _report(reporter, "Installing declared workload dependencies")
                install_arguments = ["install", "-s", session_name]
                if request["requirements_file"] is not None:
                    install_arguments.extend(
                        ["-r", str(request["requirements_file"])]
                    )
                else:
                    install_arguments.extend(request["packages"])
                await self.cli.run(
                    install_arguments,
                    timeout_seconds=_remaining(deadline),
                )

            failed_step = "execute"
            await _report(reporter, f"Executing {request['script_path'].name}")
            execution_result = await self.cli.run(
                [
                    "exec",
                    "-s",
                    session_name,
                    "--timeout",
                    f"{_remaining(deadline):.3f}",
                    "-f",
                    str(request["script_path"]),
                ],
                timeout_seconds=_remaining(deadline),
            )

            for index, remote_path in enumerate(request["remote_artifacts"], start=1):
                failed_step = "download"
                await _report(
                    reporter,
                    f"Retrieving artifact {index}/{len(request['remote_artifacts'])}",
                )
                local_path = _artifact_local_path(
                    request["artifact_dir"], remote_path
                )
                local_path.parent.mkdir(parents=True, exist_ok=True)
                await self.cli.run(
                    [
                        "download",
                        "-s",
                        session_name,
                        remote_path,
                        str(local_path),
                    ],
                    timeout_seconds=_remaining(deadline),
                )
                downloaded.append(
                    {"remote_path": remote_path, "local_path": str(local_path)}
                )

            if export_log:
                failed_step = "export_log"
                await _report(reporter, "Exporting replayable notebook log")
                destination = request["artifact_dir"] / f"{session_name}.ipynb"
                destination.parent.mkdir(parents=True, exist_ok=True)
                await self.cli.run(
                    ["log", "-s", session_name, "-o", str(destination)],
                    timeout_seconds=_remaining(deadline),
                )
                log_path = str(destination)

            state = "finished"
            failed_step = None
        except asyncio.CancelledError:
            state = "interrupted"
            error = "The Colab job was interrupted"
            cancelled = True
        except ProcessExecutionTimeout as exc:
            state = "timed_out"
            error = str(exc)
            if failed_step == "execute":
                execution_result = exc.result
        except asyncio.TimeoutError:
            state = "timed_out"
            error = f"The job exceeded its {max_runtime_seconds:g}s total deadline"
        except ProcessExecutionError as exc:
            state = "failed"
            error = str(exc)
            if failed_step == "execute":
                execution_result = exc.result
        except FileNotFoundError as exc:
            state = "failed"
            error = str(exc)
        except Exception as exc:
            state = "failed"
            error = f"{type(exc).__name__}: {exc}"
        finally:
            if allocation_attempted:
                cleanup_attempted = True
                try:
                    await _report(reporter, "Stopping Colab session")
                except asyncio.CancelledError:
                    cancelled = True

                cleanup_task = asyncio.create_task(
                    self.cli.run(
                        ["stop", "-s", session_name],
                        timeout_seconds=self.cleanup_timeout_seconds,
                    )
                )
                try:
                    await asyncio.shield(cleanup_task)
                    cleanup_succeeded = True
                except asyncio.CancelledError:
                    cancelled = True
                    try:
                        await cleanup_task
                        cleanup_succeeded = True
                    except asyncio.CancelledError:
                        cleanup_error = "Colab cleanup was cancelled"
                    except Exception as exc:
                        cleanup_error = f"{type(exc).__name__}: {exc}"
                except Exception as exc:
                    cleanup_error = f"{type(exc).__name__}: {exc}"

        if cancelled:
            raise asyncio.CancelledError()

        if state == "finished" and not cleanup_succeeded:
            state = "cleanup_failed"
            error = "Workload finished, but the Colab session could not be released"

        await _report(
            reporter,
            "Colab job finished" if state == "finished" else f"Colab job {state}",
        )
        result: dict[str, Any] = {
            "ok": state == "finished",
            "state": state,
            "session_name": session_name,
            "accelerator": accelerator,
            "script_path": str(request["script_path"]),
            "elapsed_seconds": time.monotonic() - started,
            "failed_step": failed_step,
            "error": error,
            "artifacts": downloaded,
            "log_path": log_path,
            "cleanup": {
                "attempted": cleanup_attempted,
                "succeeded": cleanup_succeeded,
                "error": cleanup_error,
            },
        }
        if execution_result is not None:
            result["execution"] = _process_result_data(execution_result)

        notebook_output = request["script_path"].with_name(
            f"{request['script_path'].stem}_output.ipynb"
        )
        if request["script_path"].suffix == ".ipynb" and notebook_output.exists():
            result["notebook_output_path"] = str(notebook_output)
        return result


def validate_session_name(session_name: str) -> str:
    if not _SESSION_RE.fullmatch(session_name):
        raise ValueError(
            "session_name must start with a letter or digit and contain only "
            "letters, digits, dots, underscores, or hyphens (maximum 64 characters)"
        )
    return session_name


def _validate_request(
    *,
    script_path: str,
    accelerator: str,
    packages: Sequence[str],
    requirements_file: str | None,
    remote_artifacts: Sequence[str],
    artifact_dir: str | None,
    export_log: bool,
    max_runtime_seconds: float,
    session_name_prefix: str,
) -> dict[str, Any]:
    script = Path(script_path).expanduser().resolve()
    if not script.is_file() or script.suffix not in {".py", ".ipynb"}:
        raise ValueError("script_path must be an existing .py or .ipynb file")
    if accelerator not in SUPPORTED_ACCELERATORS:
        supported = ", ".join(sorted(SUPPORTED_ACCELERATORS))
        raise ValueError(f"accelerator must be one of: {supported}")
    if not 30 <= max_runtime_seconds <= 86_400:
        raise ValueError("max_runtime_seconds must be between 30 and 86400")

    normalized_prefix = re.sub(r"[^a-z0-9-]+", "-", session_name_prefix.lower())
    normalized_prefix = normalized_prefix.strip("-")[:30]
    if not normalized_prefix or not normalized_prefix[0].isalpha():
        raise ValueError("session_name_prefix must begin with a letter")

    package_values = tuple(packages)
    if len(package_values) > _PACKAGE_LIMIT:
        raise ValueError(f"at most {_PACKAGE_LIMIT} packages may be installed")
    for package in package_values:
        if not package or len(package) > 256 or package.startswith("-"):
            raise ValueError(f"invalid package requirement: {package!r}")

    requirements: Path | None = None
    if requirements_file is not None:
        requirements = Path(requirements_file).expanduser().resolve()
        if not requirements.is_file():
            raise ValueError("requirements_file must be an existing file")
    if requirements is not None and package_values:
        raise ValueError("use packages or requirements_file, not both")

    remote_values = tuple(remote_artifacts)
    if len(remote_values) > _ARTIFACT_LIMIT:
        raise ValueError(f"at most {_ARTIFACT_LIMIT} artifacts may be downloaded")
    for remote_path in remote_values:
        _validate_remote_artifact(remote_path)

    destination: Path | None = None
    if artifact_dir is not None:
        destination = Path(artifact_dir).expanduser()
        if not destination.is_absolute():
            raise ValueError("artifact_dir must be an absolute local path")
        destination = destination.resolve()
    if (remote_values or export_log) and destination is None:
        raise ValueError(
            "artifact_dir is required when downloading artifacts or exporting a log"
        )

    return {
        "script_path": script,
        "accelerator": accelerator,
        "packages": package_values,
        "requirements_file": requirements,
        "remote_artifacts": remote_values,
        "artifact_dir": destination,
        "export_log": export_log,
        "max_runtime_seconds": max_runtime_seconds,
        "session_prefix": normalized_prefix,
    }


def _new_arguments(session_name: str, accelerator: str) -> list[str]:
    arguments = ["new", "-s", session_name]
    if accelerator.startswith("TPU_"):
        arguments.extend(["--tpu", accelerator.removeprefix("TPU_").lower()])
    elif accelerator != "CPU":
        arguments.extend(["--gpu", accelerator])
    return arguments


def _validate_remote_artifact(remote_path: str) -> None:
    path = PurePosixPath(remote_path)
    relative_parts = remote_path.split("/")[2:]
    if (
        not remote_path.startswith("/content/")
        or not relative_parts
        or any(part in {"", ".", ".."} for part in relative_parts)
        or path == PurePosixPath("/content")
    ):
        raise ValueError(
            "remote artifact paths must be absolute files beneath /content"
        )


def _artifact_local_path(destination: Path | None, remote_path: str) -> Path:
    if destination is None:
        raise RuntimeError("artifact destination was not configured")
    relative = PurePosixPath(remote_path).relative_to("/content")
    return destination.joinpath(*relative.parts)


def _remaining(deadline: float) -> float:
    remaining = deadline - asyncio.get_running_loop().time()
    if remaining <= 0:
        raise asyncio.TimeoutError
    return min(remaining, 86_400)


def _process_result_data(result: ProcessResult) -> dict[str, Any]:
    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "stdout_truncated": result.stdout_truncated,
        "stderr_truncated": result.stderr_truncated,
        "elapsed_seconds": result.elapsed_seconds,
    }


async def _report(reporter: ActivityReporter | None, message: str) -> None:
    if reporter is None:
        return
    try:
        await reporter(message)
    except asyncio.CancelledError:
        raise
    except Exception:
        # Progress is optional telemetry and must never break a workload.
        return
