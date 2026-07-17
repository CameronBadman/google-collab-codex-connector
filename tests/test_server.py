from __future__ import annotations

import asyncio
from pathlib import Path

import nbformat
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
        "colab_cell_status",
        "colab_download_artifact",
        "colab_execute",
        "colab_export_log",
        "colab_notebook_cells",
        "colab_renew_session",
        "colab_run_job",
        "colab_session_status",
        "colab_sessions",
        "colab_start_session",
        "colab_stop_session",
        "colab_update_notebook_cell",
    }
    assert not any("browser" in name for name in tools)


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


@pytest.mark.asyncio
async def test_server_wires_reusable_session_lifecycle(tmp_path: Path) -> None:
    script = tmp_path / "step.py"
    script.write_text("print('step')\n", encoding="utf-8")
    cli = StatefulCli()

    async with Client(create_mcp(cli)) as client:
        started = await client.call_tool(
            "colab_start_session",
            {
                "accelerator": "CPU",
                "idle_timeout_seconds": 60,
                "session_name_prefix": "server-test",
            },
        )
        started_data = started.structured_content
        assert started_data is not None
        session_name = started_data["session_name"]

        executed = await client.call_tool(
            "colab_execute",
            {
                "session_name": session_name,
                "script_path": str(script),
                "timeout_seconds": 30,
            },
        )
        executed_data = executed.structured_content
        assert executed_data is not None
        assert executed_data["ok"] is True

        stopped = await client.call_tool(
            "colab_stop_session",
            {"session_name": session_name},
        )
        stopped_data = stopped.structured_content
        assert stopped_data is not None
        assert stopped_data["state"] == "stopped"

    assert cli.calls[0][0] == "new"
    assert any(call[0] == "exec" for call in cli.calls)
    assert cli.calls[-1] == ["stop", "-s", session_name]


@pytest.mark.asyncio
async def test_server_wires_notebook_cell_management(tmp_path: Path) -> None:
    notebook_path = tmp_path / "managed.ipynb"
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell("print('old')", id="target"),
        ]
    )
    nbformat.write(notebook, notebook_path)
    cli = StatefulCli()

    async with Client(create_mcp(cli)) as client:
        started = await client.call_tool(
            "colab_start_session",
            {
                "idle_timeout_seconds": 60,
                "session_name_prefix": "cell-tools",
            },
        )
        session_name = started.structured_content["session_name"]
        inspected = await client.call_tool(
            "colab_notebook_cells",
            {
                "notebook_path": str(notebook_path),
                "include_source": True,
                "session_name": session_name,
            },
        )
        inspected_data = inspected.structured_content
        assert inspected_data is not None
        cell = inspected_data["cells"][0]
        assert cell["source"] == "print('old')"
        assert cell["latest_execution"] is None

        updated = await client.call_tool(
            "colab_update_notebook_cell",
            {
                "notebook_path": str(notebook_path),
                "cell_id": "target",
                "source": "print('new')",
                "expected_source_sha256": cell["source_sha256"],
            },
        )
        assert updated.structured_content["state"] == "updated"

        queued = await client.call_tool(
            "colab_execute",
            {
                "session_name": session_name,
                "script_path": str(notebook_path),
                "cell_id": "target",
                "background": True,
                "write_output_to_notebook": True,
                "timeout_seconds": 30,
            },
        )
        execution_id = queued.structured_content["execution_id"]
        for _ in range(100):
            status = await client.call_tool(
                "colab_cell_status",
                {"execution_id": execution_id},
            )
            status_data = status.structured_content
            assert status_data is not None
            if status_data["terminal"]:
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("background cell did not finish")

        assert status_data["state"] == "finished"
        assert status_data["output_writeback"]["written"] is True
        cells_after = await client.call_tool(
            "colab_notebook_cells",
            {
                "notebook_path": str(notebook_path),
                "session_name": session_name,
            },
        )
        latest = cells_after.structured_content["cells"][0]["latest_execution"]
        assert latest["execution_id"] == execution_id
        assert latest["source_matches"] is True
        await client.call_tool(
            "colab_stop_session",
            {"session_name": session_name},
        )

    saved = nbformat.read(notebook_path, as_version=4).cells[0]
    assert saved.source == "print('new')"
    assert saved.outputs[0].text == "step complete\n"


class StatefulCli:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def run(
        self,
        arguments: list[str],
        *,
        timeout_seconds: float,
    ) -> ProcessResult:
        del timeout_seconds
        self.calls.append(arguments)
        return _result(stdout="step complete\n" if arguments[0] == "exec" else "")


def _result(*, stdout: str = "") -> ProcessResult:
    return ProcessResult(
        returncode=0,
        stdout=stdout,
        stderr="",
        stdout_truncated=False,
        stderr_truncated=False,
        elapsed_seconds=0.01,
    )
