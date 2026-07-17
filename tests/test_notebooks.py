from __future__ import annotations

import stat
from pathlib import Path

import nbformat
import pytest

from colab_runner.notebooks import MAX_CELL_SOURCE_BYTES, NotebookController


@pytest.mark.asyncio
async def test_notebook_inspection_is_paginated_and_bounds_source(
    tmp_path: Path,
) -> None:
    path = tmp_path / "work.ipynb"
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_markdown_cell("# Overview", id="overview"),
            nbformat.v4.new_code_cell("x = 1\n" + "print(x)\n" * 100, id="code-1"),
            nbformat.v4.new_code_cell("print('done')", id="code-2"),
        ]
    )
    notebook.cells[1]["outputs"] = [
        nbformat.v4.new_output("stream", name="stdout", text="1\n")
    ]
    nbformat.write(notebook, path)
    controller = NotebookController()

    result = await controller.inspect(
        str(path),
        start=1,
        limit=1,
        include_source=True,
        source_excerpt_bytes=128,
    )

    assert result["ok"] is True
    assert result["total_cells"] == 3
    assert result["returned_cells"] == 1
    assert result["next_start"] == 2
    cell = result["cells"][0]
    assert cell["cell_id"] == "code-1"
    assert cell["source_bytes"] > 128
    assert len(cell["source"].encode("utf-8")) <= 128
    assert cell["source_truncated"] is True
    assert cell["output_count"] == 1
    assert cell["output_bytes"] > 0


@pytest.mark.asyncio
async def test_cell_update_uses_source_hash_and_preserves_file_mode(
    tmp_path: Path,
) -> None:
    path = tmp_path / "work.ipynb"
    _write_notebook(path)
    path.chmod(0o640)
    controller = NotebookController()
    inspected = await controller.inspect(str(path))
    original = inspected["cells"][1]

    updated = await controller.update_cell(
        str(path),
        cell_id="target",
        cell_index=None,
        source="value = 42",
        expected_source_sha256=original["source_sha256"],
    )

    assert updated["ok"] is True
    assert updated["state"] == "updated"
    assert updated["outputs_cleared"] is True
    assert stat.S_IMODE(path.stat().st_mode) == 0o640
    notebook = nbformat.read(path, as_version=4)
    assert notebook.cells[1].source == "value = 42"
    assert notebook.cells[1].outputs == []
    assert notebook.cells[1].execution_count is None

    conflict = await controller.update_cell(
        str(path),
        cell_id="target",
        cell_index=None,
        source="value = 99",
        expected_source_sha256=original["source_sha256"],
    )
    assert conflict["ok"] is False
    assert conflict["state"] == "conflict"
    assert nbformat.read(path, as_version=4).cells[1].source == "value = 42"


@pytest.mark.asyncio
async def test_execution_output_writeback_detects_source_conflict(
    tmp_path: Path,
) -> None:
    path = tmp_path / "work.ipynb"
    _write_notebook(path)
    controller = NotebookController()
    selected = await controller.select_code_cell(
        path,
        cell_id="target",
        cell_index=None,
    )

    written = await controller.write_execution_outputs(
        selected,
        session_name="session-1",
        execution_id="execution-1",
        state="finished",
        executed_at="2026-07-17T00:00:00+00:00",
        stdout="answer=42\n",
        stderr="",
        stdout_truncated=False,
        stderr_truncated=False,
    )

    assert written["written"] is True
    notebook = nbformat.read(path, as_version=4)
    cell = notebook.cells[1]
    assert cell.outputs[0].text == "answer=42\n"
    assert cell.metadata["colab_runner"]["execution_id"] == "execution-1"

    current = await controller.inspect(str(path))
    await controller.update_cell(
        str(path),
        cell_id="target",
        cell_index=None,
        source="value = 43",
        expected_source_sha256=current["cells"][1]["source_sha256"],
    )
    conflict = await controller.write_execution_outputs(
        selected,
        session_name="session-1",
        execution_id="execution-1",
        state="finished",
        executed_at="2026-07-17T00:00:00+00:00",
        stdout="stale\n",
        stderr="",
        stdout_truncated=False,
        stderr_truncated=False,
    )

    assert conflict["written"] is False
    assert conflict["state"] == "conflict"
    assert nbformat.read(path, as_version=4).cells[1].outputs == []


def _write_notebook(path: Path) -> None:
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_markdown_cell("Notes", id="notes"),
            nbformat.v4.new_code_cell("value = 1", id="target"),
        ]
    )
    notebook.cells[1]["execution_count"] = 1
    notebook.cells[1]["outputs"] = [
        nbformat.v4.new_output("stream", name="stdout", text="old\n")
    ]
    nbformat.write(notebook, path)


@pytest.mark.asyncio
async def test_selected_cell_source_size_is_bounded(tmp_path: Path) -> None:
    path = tmp_path / "oversized.ipynb"
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell(
                "x" * (MAX_CELL_SOURCE_BYTES + 1),
                id="oversized",
            )
        ]
    )
    nbformat.write(notebook, path)

    with pytest.raises(ValueError, match="selected cell source exceeds"):
        await NotebookController().select_code_cell(
            path,
            cell_id="oversized",
            cell_index=None,
        )
