from __future__ import annotations

import asyncio
import sys

import pytest

from colab_runner.process import (
    ProcessExecutionError,
    ProcessExecutionTimeout,
    ProcessRunner,
)


@pytest.mark.asyncio
async def test_process_runner_captures_stdout_and_stderr() -> None:
    result = await ProcessRunner().run(
        [
            sys.executable,
            "-c",
            "import sys; print('hello'); print('warning', file=sys.stderr)",
        ],
        timeout_seconds=5,
    )

    assert result.returncode == 0
    assert result.stdout == "hello\n"
    assert result.stderr == "warning\n"
    assert result.stdout_truncated is False
    assert result.stderr_truncated is False


@pytest.mark.asyncio
async def test_process_runner_never_inherits_mcp_stdin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = asyncio.create_subprocess_exec
    observed_stdin: object | None = None

    async def create_subprocess_exec(*argv: str, **kwargs: object):
        nonlocal observed_stdin
        observed_stdin = kwargs.get("stdin")
        return await original(*argv, **kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess_exec)

    await ProcessRunner().run(
        [sys.executable, "-c", "print('ok')"],
        timeout_seconds=5,
    )

    assert observed_stdin == asyncio.subprocess.DEVNULL


@pytest.mark.asyncio
async def test_interactive_prompt_fails_immediately_on_eof() -> None:
    with pytest.raises(ProcessExecutionError) as captured:
        await ProcessRunner().run(
            [sys.executable, "-c", "input('authorization code: ')"],
            timeout_seconds=5,
        )

    assert captured.value.result.elapsed_seconds < 2
    assert "EOFError" in captured.value.result.stderr


@pytest.mark.asyncio
async def test_process_runner_enforces_hard_timeout() -> None:
    with pytest.raises(ProcessExecutionTimeout):
        await ProcessRunner().run(
            [sys.executable, "-c", "import time; time.sleep(5)"],
            timeout_seconds=0.05,
        )


@pytest.mark.asyncio
async def test_process_runner_bounds_noisy_output() -> None:
    result = await ProcessRunner(capture_bytes=1024).run(
        [
            sys.executable,
            "-c",
            "import sys; sys.stdout.write('A' * 5000 + 'Z' * 5000)",
        ],
        timeout_seconds=5,
    )

    assert result.stdout_truncated is True
    assert result.stdout.startswith("A" * 100)
    assert result.stdout.endswith("Z" * 100)
    assert "output truncated by colab-runner" in result.stdout
