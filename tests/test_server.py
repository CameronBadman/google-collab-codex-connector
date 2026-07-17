from __future__ import annotations

import pytest
from fastmcp import Client

from colab_runner.process import ProcessResult
from colab_runner.server import REQUIRED_COLAB_SCOPES, create_mcp


@pytest.mark.asyncio
async def test_server_exposes_only_the_official_cli_surface() -> None:
    mcp = create_mcp()
    tools = await mcp.get_tools()

    assert set(tools) == {
        "colab_cli_doctor",
        "colab_run_job",
        "colab_session_status",
        "colab_sessions",
        "colab_stop_session",
    }
    assert not any("browser" in name or "notebook_cell" in name for name in tools)


class DoctorCli:
    def __init__(self, scopes: tuple[str, ...]) -> None:
        self.scopes = scopes
        self.calls: list[list[str]] = []

    async def run(
        self,
        arguments: list[str],
        *,
        timeout_seconds: float,
    ) -> ProcessResult:
        del timeout_seconds
        self.calls.append(arguments)
        if arguments == ["version"]:
            return _result(stdout="Version: 0.6.0\n")
        if arguments == ["whoami"]:
            scope_lines = "\n".join(f"  - {scope}" for scope in self.scopes)
            return _result(
                stdout=(
                    "Auth provider: adc\n"
                    "Email: private@example.com\n"
                    "Scopes:\n"
                    f"{scope_lines}\n"
                )
            )
        if arguments == ["sessions"]:
            return _result(stdout="[colab] No active sessions found on server.\n")
        raise AssertionError(f"unexpected command: {arguments}")


@pytest.mark.asyncio
async def test_doctor_rejects_adc_missing_colaboratory_scope() -> None:
    cli = DoctorCli(REQUIRED_COLAB_SCOPES[:-1])

    async with Client(create_mcp(cli)) as client:
        result = await client.call_tool("colab_cli_doctor", {})

    data = result.structured_content
    assert data is not None
    assert data["ok"] is False
    assert data["credentials"]["missing_scopes"] == [
        "https://www.googleapis.com/auth/colaboratory"
    ]
    assert data["sessions"] is None
    assert "private@example.com" not in str(data)
    assert cli.calls == [["version"], ["whoami"]]


@pytest.mark.asyncio
async def test_doctor_accepts_complete_adc_scope_set() -> None:
    cli = DoctorCli(REQUIRED_COLAB_SCOPES)

    async with Client(create_mcp(cli)) as client:
        result = await client.call_tool("colab_cli_doctor", {})

    data = result.structured_content
    assert data is not None
    assert data["ok"] is True
    assert data["credentials"]["missing_scopes"] == []
    assert data["sessions"]["ok"] is True
    assert cli.calls == [["version"], ["whoami"], ["sessions"]]


def _result(*, stdout: str = "") -> ProcessResult:
    return ProcessResult(
        returncode=0,
        stdout=stdout,
        stderr="",
        stdout_truncated=False,
        stderr_truncated=False,
        elapsed_seconds=0.01,
    )
