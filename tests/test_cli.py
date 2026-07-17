from __future__ import annotations

from pathlib import Path

import pytest

from colab_runner.cli import ColabCli
from colab_runner.process import ProcessResult


class RecordingProcessRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], float, Path | None]] = []

    async def run(
        self,
        argv: list[str],
        *,
        timeout_seconds: float,
        cwd: Path | None = None,
    ) -> ProcessResult:
        self.calls.append((argv, timeout_seconds, cwd))
        return _result()


@pytest.mark.asyncio
async def test_cli_uses_official_binary_and_global_options(tmp_path: Path) -> None:
    process = RecordingProcessRunner()
    config = tmp_path / "sessions.json"
    cli = ColabCli(
        process,
        executable="official-colab",
        auth="adc",
        config_path=config,
    )

    await cli.run(["status", "-s", "demo"], timeout_seconds=12)

    assert process.calls == [
        (
            [
                "official-colab",
                "--auth=adc",
                "--config",
                str(config),
                "status",
                "-s",
                "demo",
            ],
            12,
            None,
        )
    ]


def test_cli_auth_mode_can_be_selected_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("COLAB_RUNNER_AUTH", "oauth2")

    assert ColabCli().auth == "oauth2"


def test_cli_rejects_invalid_environment_auth_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("COLAB_RUNNER_AUTH", "interactive")

    with pytest.raises(ValueError, match="adc.*oauth2"):
        ColabCli()


@pytest.mark.asyncio
async def test_pinned_official_cli_is_callable() -> None:
    result = await ColabCli().run(["version"], timeout_seconds=10)

    assert result.returncode == 0
    assert "0.6.0" in result.stdout


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("new", ("--session", "--gpu", "--tpu")),
        ("exec", ("--session", "--file", "--timeout")),
        ("download", ("REMOTE_PATH", "LOCAL_PATH")),
        ("log", ("--session", "--output")),
        ("stop", ("--session",)),
        ("whoami", ("active credentials", "scopes", "expiry")),
    ],
)
@pytest.mark.asyncio
async def test_pinned_official_cli_command_contract(
    command: str,
    expected: tuple[str, ...],
) -> None:
    result = await ColabCli().run(["help", command], timeout_seconds=10)

    assert result.returncode == 0
    for value in expected:
        assert value in result.stdout


def _result() -> ProcessResult:
    return ProcessResult(
        returncode=0,
        stdout="",
        stderr="",
        stdout_truncated=False,
        stderr_truncated=False,
        elapsed_seconds=0.01,
    )
