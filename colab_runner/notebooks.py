from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import re
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import nbformat


MAX_CELL_SOURCE_BYTES: Final = 256 * 1024
MAX_SOURCE_EXCERPT_BYTES: Final = 16 * 1024
MAX_INSPECTION_SOURCE_BYTES: Final = 64 * 1024
MAX_CELL_PAGE_SIZE: Final = 100
_SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_RUNNER_METADATA_KEYS: Final = frozenset(
    {
        "session_name",
        "execution_id",
        "state",
        "executed_at",
        "source_sha256",
        "stdout_truncated",
        "stderr_truncated",
    }
)


@dataclass(frozen=True, slots=True)
class SelectedCell:
    notebook_path: Path
    cell_index: int
    cell_id: str | None
    source: str
    source_bytes: int
    source_sha256: str

    def data(self) -> dict[str, Any]:
        return {
            "cell_index": self.cell_index,
            "cell_id": self.cell_id,
            "source_bytes": self.source_bytes,
            "source_sha256": self.source_sha256,
        }

    def reference(self) -> SelectedCell:
        return SelectedCell(
            notebook_path=self.notebook_path,
            cell_index=self.cell_index,
            cell_id=self.cell_id,
            source="",
            source_bytes=self.source_bytes,
            source_sha256=self.source_sha256,
        )


class NotebookController:
    """Bounded, atomic access to local notebook documents."""

    def __init__(self) -> None:
        self._locks: dict[Path, asyncio.Lock] = {}

    async def inspect(
        self,
        notebook_path: str,
        *,
        start: int = 0,
        limit: int = 50,
        include_source: bool = False,
        source_excerpt_bytes: int = 2048,
    ) -> dict[str, Any]:
        path = resolve_notebook_path(notebook_path)
        if isinstance(start, bool) or not isinstance(start, int) or start < 0:
            raise ValueError("start must be a non-negative integer")
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= MAX_CELL_PAGE_SIZE
        ):
            raise ValueError(
                f"limit must be between 1 and {MAX_CELL_PAGE_SIZE}"
            )
        if (
            isinstance(source_excerpt_bytes, bool)
            or not isinstance(source_excerpt_bytes, int)
            or not 128 <= source_excerpt_bytes <= MAX_SOURCE_EXCERPT_BYTES
        ):
            raise ValueError(
                "source_excerpt_bytes must be between 128 and "
                f"{MAX_SOURCE_EXCERPT_BYTES}"
            )
        async with self._lock_for(path):
            return await asyncio.to_thread(
                _inspect_notebook,
                path,
                start=start,
                limit=limit,
                include_source=include_source,
                source_excerpt_bytes=source_excerpt_bytes,
            )

    async def select_code_cell(
        self,
        notebook_path: str | Path,
        *,
        cell_index: int | None,
        cell_id: str | None,
    ) -> SelectedCell:
        path = resolve_notebook_path(str(notebook_path))
        async with self._lock_for(path):
            return await asyncio.to_thread(
                _select_code_cell_from_path,
                path,
                cell_index=cell_index,
                cell_id=cell_id,
            )

    async def update_cell(
        self,
        notebook_path: str,
        *,
        source: str,
        expected_source_sha256: str,
        cell_index: int | None,
        cell_id: str | None,
        clear_outputs: bool = True,
    ) -> dict[str, Any]:
        path = resolve_notebook_path(notebook_path)
        if not isinstance(source, str):
            raise ValueError("source must be a string")
        source_bytes = len(source.encode("utf-8"))
        if source_bytes > MAX_CELL_SOURCE_BYTES:
            raise ValueError(
                f"source must not exceed {MAX_CELL_SOURCE_BYTES} UTF-8 bytes"
            )
        _validate_sha256(expected_source_sha256)
        async with self._lock_for(path):
            return await asyncio.to_thread(
                _update_notebook_cell,
                path,
                source=source,
                expected_source_sha256=expected_source_sha256,
                cell_index=cell_index,
                cell_id=cell_id,
                clear_outputs=clear_outputs,
            )

    async def write_execution_outputs(
        self,
        selected: SelectedCell,
        *,
        session_name: str,
        execution_id: str,
        state: str,
        executed_at: str,
        stdout: str,
        stderr: str,
        stdout_truncated: bool,
        stderr_truncated: bool,
    ) -> dict[str, Any]:
        path = selected.notebook_path
        async with self._lock_for(path):
            return await asyncio.to_thread(
                _write_execution_outputs,
                selected,
                session_name=session_name,
                execution_id=execution_id,
                state=state,
                executed_at=executed_at,
                stdout=stdout,
                stderr=stderr,
                stdout_truncated=stdout_truncated,
                stderr_truncated=stderr_truncated,
            )

    def _lock_for(self, path: Path) -> asyncio.Lock:
        lock = self._locks.get(path)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[path] = lock
        return lock


def resolve_notebook_path(notebook_path: str) -> Path:
    path = Path(notebook_path).expanduser().resolve()
    if not path.is_file() or path.suffix != ".ipynb":
        raise ValueError("notebook_path must be an existing .ipynb file")
    return path


def source_sha256(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _inspect_notebook(
    path: Path,
    *,
    start: int,
    limit: int,
    include_source: bool,
    source_excerpt_bytes: int,
) -> dict[str, Any]:
    notebook = nbformat.read(path, as_version=4)
    total = len(notebook.cells)
    page = notebook.cells[start : start + limit]
    excerpt_bytes = (
        min(
            source_excerpt_bytes,
            MAX_INSPECTION_SOURCE_BYTES // max(1, len(page)),
        )
        if include_source
        else source_excerpt_bytes
    )
    cells = [
        _cell_data(
            cell,
            index=start + offset,
            include_source=include_source,
            source_excerpt_bytes=excerpt_bytes,
        )
        for offset, cell in enumerate(page)
    ]
    next_start = start + len(cells)
    return {
        "ok": True,
        "notebook_path": str(path),
        "notebook_sha256": _file_sha256(path),
        "notebook_bytes": path.stat().st_size,
        "total_cells": total,
        "start": start,
        "returned_cells": len(cells),
        "next_start": next_start if next_start < total else None,
        "cells": cells,
    }


def _cell_data(
    cell: Any,
    *,
    index: int,
    include_source: bool,
    source_excerpt_bytes: int,
) -> dict[str, Any]:
    source = str(cell.get("source", ""))
    outputs = cell.get("outputs", []) if cell.get("cell_type") == "code" else []
    data: dict[str, Any] = {
        "cell_index": index,
        "cell_id": cell.get("id"),
        "cell_type": cell.get("cell_type"),
        "title": _cell_title(source),
        "source_bytes": len(source.encode("utf-8")),
        "source_sha256": source_sha256(source),
        "output_count": len(outputs) if isinstance(outputs, list) else 0,
        "output_bytes": _json_bytes(outputs),
        "execution_count": (
            cell.get("execution_count")
            if cell.get("cell_type") == "code"
            else None
        ),
        "runner_execution": _runner_metadata(cell),
    }
    if include_source:
        excerpt, truncated = _utf8_excerpt(source, source_excerpt_bytes)
        data["source"] = excerpt
        data["source_truncated"] = truncated
    return data


def _select_code_cell_from_path(
    path: Path,
    *,
    cell_index: int | None,
    cell_id: str | None,
) -> SelectedCell:
    notebook = nbformat.read(path, as_version=4)
    index, cell = _select_cell(
        notebook,
        cell_index=cell_index,
        cell_id=cell_id,
    )
    if cell.get("cell_type") != "code":
        raise ValueError(
            f"notebook cell {index} is {cell.get('cell_type')}, not code"
        )
    source = str(cell.get("source", ""))
    source_bytes = len(source.encode("utf-8"))
    if source_bytes > MAX_CELL_SOURCE_BYTES:
        raise ValueError(
            f"selected cell source exceeds {MAX_CELL_SOURCE_BYTES} UTF-8 bytes"
        )
    return SelectedCell(
        notebook_path=path,
        cell_index=index,
        cell_id=cell.get("id"),
        source=source,
        source_bytes=source_bytes,
        source_sha256=source_sha256(source),
    )


def _update_notebook_cell(
    path: Path,
    *,
    source: str,
    expected_source_sha256: str,
    cell_index: int | None,
    cell_id: str | None,
    clear_outputs: bool,
) -> dict[str, Any]:
    notebook = nbformat.read(path, as_version=4)
    index, cell = _select_cell(
        notebook,
        cell_index=cell_index,
        cell_id=cell_id,
    )
    current_source = str(cell.get("source", ""))
    current_sha256 = source_sha256(current_source)
    if not hmac.compare_digest(current_sha256, expected_source_sha256):
        return {
            "ok": False,
            "state": "conflict",
            "notebook_path": str(path),
            "cell_index": index,
            "cell_id": cell.get("id"),
            "expected_source_sha256": expected_source_sha256,
            "current_source_sha256": current_sha256,
            "error": "The notebook cell changed after it was inspected",
        }

    cell["source"] = source
    outputs_cleared = False
    if cell.get("cell_type") == "code" and clear_outputs:
        cell["outputs"] = []
        cell["execution_count"] = None
        outputs_cleared = True
    metadata = cell.get("metadata")
    if isinstance(metadata, dict):
        metadata.pop("colab_runner", None)
    _atomic_write_notebook(path, notebook)
    return {
        "ok": True,
        "state": "updated",
        "notebook_path": str(path),
        "notebook_sha256": _file_sha256(path),
        "cell_index": index,
        "cell_id": cell.get("id"),
        "previous_source_sha256": current_sha256,
        "source_sha256": source_sha256(source),
        "source_bytes": len(source.encode("utf-8")),
        "outputs_cleared": outputs_cleared,
        "error": None,
    }


def _write_execution_outputs(
    selected: SelectedCell,
    *,
    session_name: str,
    execution_id: str,
    state: str,
    executed_at: str,
    stdout: str,
    stderr: str,
    stdout_truncated: bool,
    stderr_truncated: bool,
) -> dict[str, Any]:
    path = selected.notebook_path
    notebook = nbformat.read(path, as_version=4)
    selector = (
        {"cell_id": selected.cell_id, "cell_index": None}
        if selected.cell_id is not None
        else {"cell_index": selected.cell_index, "cell_id": None}
    )
    index, cell = _select_cell(notebook, **selector)
    if cell.get("cell_type") != "code":
        return {
            "requested": True,
            "state": "conflict",
            "written": False,
            "error": "The selected cell is no longer a code cell",
        }

    current_source = str(cell.get("source", ""))
    current_sha256 = source_sha256(current_source)
    if not hmac.compare_digest(current_sha256, selected.source_sha256):
        return {
            "requested": True,
            "state": "conflict",
            "written": False,
            "cell_index": index,
            "cell_id": cell.get("id"),
            "executed_source_sha256": selected.source_sha256,
            "current_source_sha256": current_sha256,
            "error": "Cell output was not written because its source changed",
        }

    outputs = []
    if stdout:
        outputs.append(
            nbformat.v4.new_output("stream", name="stdout", text=stdout)
        )
    if stderr:
        outputs.append(
            nbformat.v4.new_output("stream", name="stderr", text=stderr)
        )
    cell["outputs"] = outputs
    cell["execution_count"] = None
    metadata = cell.setdefault("metadata", {})
    metadata["colab_runner"] = {
        "session_name": session_name,
        "execution_id": execution_id,
        "state": state,
        "executed_at": executed_at,
        "source_sha256": selected.source_sha256,
        "stdout_truncated": stdout_truncated,
        "stderr_truncated": stderr_truncated,
    }
    _atomic_write_notebook(path, notebook)
    return {
        "requested": True,
        "state": "written",
        "written": True,
        "notebook_path": str(path),
        "notebook_sha256": _file_sha256(path),
        "cell_index": index,
        "cell_id": cell.get("id"),
        "output_count": len(outputs),
        "error": None,
    }


def _select_cell(
    notebook: Any,
    *,
    cell_index: int | None,
    cell_id: str | None,
) -> tuple[int, Any]:
    if cell_index is not None and cell_id is not None:
        raise ValueError("use cell_index or cell_id, not both")
    if cell_index is None and cell_id is None:
        raise ValueError("cell_index or cell_id is required")
    if cell_index is not None and (
        isinstance(cell_index, bool)
        or not isinstance(cell_index, int)
        or cell_index < 0
    ):
        raise ValueError("cell_index must be a non-negative integer")
    if cell_id is not None and (
        not isinstance(cell_id, str)
        or not cell_id
        or len(cell_id) > 256
    ):
        raise ValueError("cell_id must be between 1 and 256 characters")

    if cell_index is not None:
        if cell_index >= len(notebook.cells):
            raise ValueError(
                f"cell_index {cell_index} is outside the notebook cell range"
            )
        return cell_index, notebook.cells[cell_index]

    matches = [
        index
        for index, cell in enumerate(notebook.cells)
        if cell.get("id") == cell_id
    ]
    if not matches:
        raise ValueError(f"cell_id {cell_id!r} was not found")
    if len(matches) > 1:
        raise ValueError(f"cell_id {cell_id!r} is not unique")
    index = matches[0]
    return index, notebook.cells[index]


def _atomic_write_notebook(path: Path, notebook: Any) -> None:
    original_mode = stat.S_IMODE(path.stat().st_mode)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        nbformat.write(notebook, temporary_path)
        os.chmod(temporary_path, original_mode)
        with temporary_path.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _runner_metadata(cell: Any) -> dict[str, Any] | None:
    metadata = cell.get("metadata")
    if not isinstance(metadata, dict):
        return None
    runner = metadata.get("colab_runner")
    if not isinstance(runner, dict):
        return None
    return {key: runner.get(key) for key in _RUNNER_METADATA_KEYS if key in runner}


def _cell_title(source: str) -> str | None:
    for line in source.splitlines():
        title = line.strip()
        if title:
            return title[:120]
    return None


def _utf8_excerpt(value: str, limit: int) -> tuple[str, bool]:
    encoded = value.encode("utf-8")
    if len(encoded) <= limit:
        return value, False
    excerpt = encoded[:limit].decode("utf-8", errors="ignore")
    return excerpt, True


def _json_bytes(value: Any) -> int:
    try:
        return len(
            json.dumps(value, ensure_ascii=False, default=str).encode("utf-8")
        )
    except (TypeError, ValueError):
        return 0


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_sha256(value: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError("expected_source_sha256 must be 64 lowercase hex characters")
