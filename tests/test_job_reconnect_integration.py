from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
import sys
from contextlib import suppress
from pathlib import Path
from typing import Any

import websockets
from websockets.exceptions import ConnectionClosed

from colab_codex_adapter.artifacts import ArtifactStore
from colab_codex_adapter.bridge import COLAB, HARD_MAX_WEBSOCKET_FRAME_BYTES
from colab_codex_adapter.jobs import ColabJobManager
from colab_codex_adapter.server import create_mcp
from colab_codex_adapter.session import ColabSessionManager


async def _wait_until(predicate: Any, timeout: float = 5.0) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.02)


def _tool_result(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "content": [{"type": "text", "text": json.dumps(data)}],
        "structuredContent": data,
        "isError": False,
    }


class LocalNotebookRuntime:
    def __init__(self) -> None:
        self.cells: list[dict[str, Any]] = []
        self.execution_started = asyncio.Event()
        self.execution_finished = asyncio.Event()
        self.allow_disconnect = asyncio.Event()
        self.get_cells_arguments: list[dict[str, Any]] = []
        self.background_tasks: set[asyncio.Task[Any]] = set()

    def add_cell(self, arguments: dict[str, Any]) -> str:
        cell_id = f"cell-{len(self.cells)}"
        cell = {
            "id": cell_id,
            "cell_type": "code",
            "source": [arguments["code"]],
            "outputs": [],
        }
        self.cells.insert(arguments["cellIndex"], cell)
        return cell_id

    def update_cell(self, arguments: dict[str, Any]) -> None:
        cell = next(
            item for item in self.cells if item["id"] == arguments["cellId"]
        )
        cell["source"] = [arguments["content"]]
        cell["outputs"] = []

    async def run_probe(self, cell: dict[str, Any]) -> list[dict[str, Any]]:
        source = "".join(cell["source"])

        def execute() -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                [sys.executable, "-c", source],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )

        completed = await asyncio.to_thread(execute)
        outputs: list[dict[str, Any]] = []
        if completed.stdout:
            outputs.append(
                {
                    "output_type": "stream",
                    "name": "stdout",
                    "text": [completed.stdout],
                }
            )
        if completed.returncode != 0:
            outputs.append(
                {
                    "output_type": "error",
                    "ename": "ProbeError",
                    "evalue": completed.stderr[-4096:],
                }
            )
        cell["outputs"] = outputs
        return outputs

    def start_tracked_wrapper(self, cell: dict[str, Any]) -> None:
        source = "".join(cell["source"])

        async def execute() -> None:
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-c",
                source,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            self.execution_started.set()
            stdout, stderr = await process.communicate()
            assert process.returncode == 0, stderr.decode("utf-8", errors="replace")
            cell["outputs"] = [
                {
                    "output_type": "stream",
                    "name": "stdout",
                    "text": [stdout.decode("utf-8")],
                }
            ]
            self.execution_finished.set()

        task = asyncio.create_task(execute(), name="local-colab-wrapper")
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)

    async def close(self) -> None:
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks)


class NotebookFrontend:
    def __init__(
        self,
        url: str,
        runtime: LocalNotebookRuntime,
        *,
        disconnect_tracked_run: bool,
    ) -> None:
        self.url = url
        self.runtime = runtime
        self.disconnect_tracked_run = disconnect_tracked_run
        self.connected = asyncio.Event()
        self.websocket: Any = None
        self.task: asyncio.Task[None] | None = None
        self.pending_tasks: set[asyncio.Task[Any]] = set()

    def start(self) -> None:
        self.task = asyncio.create_task(self._run())

    async def _send_result(
        self, websocket: Any, request_id: Any, result: dict[str, Any]
    ) -> None:
        await websocket.send(
            json.dumps(
                {"jsonrpc": "2.0", "id": request_id, "result": result}
            )
        )

    async def _run(self) -> None:
        try:
            async with websockets.connect(
                self.url,
                origin=COLAB,
                subprotocols=["mcp"],
                proxy=None,
                max_size=HARD_MAX_WEBSOCKET_FRAME_BYTES,
            ) as websocket:
                self.websocket = websocket
                self.connected.set()
                async for raw in websocket:
                    message = json.loads(raw)
                    request_id = message.get("id")
                    if request_id is None:
                        continue
                    method = message.get("method")
                    if method == "initialize":
                        await self._send_result(
                            websocket,
                            request_id,
                            {
                                "protocolVersion": message["params"][
                                    "protocolVersion"
                                ],
                                "capabilities": {"tools": {"listChanged": False}},
                                "serverInfo": {
                                    "name": "local-notebook-runtime",
                                    "version": "1.0",
                                },
                            },
                        )
                    elif method == "tools/list":
                        tools = [
                            {
                                "name": name,
                                "inputSchema": {"type": "object"},
                            }
                            for name in (
                                "add_code_cell",
                                "run_code_cell",
                                "get_cells",
                                "update_cell",
                            )
                        ]
                        await self._send_result(
                            websocket, request_id, {"tools": tools}
                        )
                    elif method == "tools/call":
                        params = message["params"]
                        name = params["name"]
                        arguments = params.get("arguments") or {}
                        if name == "get_cells":
                            self.runtime.get_cells_arguments.append(arguments)
                            assert arguments.get("includeOutputs") is False
                            cells = [
                                {
                                    key: value
                                    for key, value in cell.items()
                                    if key != "outputs"
                                }
                                for cell in self.runtime.cells
                            ]
                            start = arguments.get("start", 0)
                            end = arguments.get("end")
                            cells = cells[start:end]
                            result = {"cells": cells}
                        elif name == "add_code_cell":
                            result = {
                                "newCellId": self.runtime.add_cell(arguments)
                            }
                        elif name == "update_cell":
                            self.runtime.update_cell(arguments)
                            result = {"cellId": arguments["cellId"]}
                        elif name == "run_code_cell":
                            cell = next(
                                item
                                for item in self.runtime.cells
                                if item["id"] == arguments["cellId"]
                            )
                            source = "".join(cell["source"])
                            if (
                                self.disconnect_tracked_run
                                and "__COLAB_CODEX_JOB__" in source
                            ):
                                self.disconnect_tracked_run = False
                                self.runtime.start_tracked_wrapper(cell)

                                async def disconnect_after_status() -> None:
                                    await self.runtime.execution_started.wait()
                                    await self.runtime.allow_disconnect.wait()
                                    await websocket.close(
                                        code=1012,
                                        reason="forced integration disconnect",
                                    )

                                task = asyncio.create_task(
                                    disconnect_after_status(),
                                    name="forced-browser-disconnect",
                                )
                                self.pending_tasks.add(task)
                                task.add_done_callback(
                                    self.pending_tasks.discard
                                )
                                continue
                            result = {
                                "outputs": await self.runtime.run_probe(cell)
                            }
                        else:  # pragma: no cover - fixture contract
                            raise AssertionError(name)
                        await self._send_result(
                            websocket, request_id, _tool_result(result)
                        )
        except ConnectionClosed:
            pass

    async def close(self) -> None:
        if self.websocket is not None:
            with suppress(ConnectionClosed):
                await self.websocket.close(code=1001, reason="test complete")
        if self.task is not None:
            await self.task
        if self.pending_tasks:
            await asyncio.gather(
                *self.pending_tasks, return_exceptions=True
            )


def _websocket_url(manager: ColabSessionManager) -> str:
    bridge = manager.bridge
    return f"ws://127.0.0.1:{bridge.port}/?access_token={bridge.token}"


async def test_long_job_survives_real_websocket_reconnect_and_artifact_read(
    tmp_path: Path,
) -> None:
    runtime = LocalNotebookRuntime()
    runtime.cells.append(
        {
            "id": "unrelated-large-output",
            "cell_type": "code",
            "source": ["print('old output')"],
            "outputs": [
                {
                    "output_type": "stream",
                    "name": "stdout",
                    "text": ["z" * (2 * 1024 * 1024)],
                }
            ],
        }
    )
    session = ColabSessionManager(bridge_host="127.0.0.1")
    await session.start()
    initial = await session.status()
    jobs = ColabJobManager(
        session,
        artifact_store=ArtifactStore(tmp_path / "artifacts"),
        output_excerpt_bytes=1024,
        runtime_artifact_bytes=512 * 1024,
        runtime_root=str(tmp_path / "runtime-jobs"),
        journal_path=tmp_path / "jobs.json",
    )
    background_tasks: set[asyncio.Task[Any]] = set()
    create_mcp(
        session,
        job_manager=jobs,
        artifact_store=jobs.artifact_store,
        background_tasks=background_tasks,
        reconciliation_poll_seconds=0.1,
    )
    first = NotebookFrontend(
        _websocket_url(session), runtime, disconnect_tracked_run=True
    )
    second: NotebookFrontend | None = None

    try:
        first.start()
        connected = await session.connect(wait_seconds=2.0)
        assert connected.runtime_alive is True
        started = await jobs.start_python(
            "import time\ntime.sleep(0.35)\nprint('x' * 200000)"
        )
        await runtime.execution_started.wait()
        running = await jobs.status(started["job_id"])
        assert running["state"] == "running"
        assert all(
            arguments.get("includeOutputs") is False
            for arguments in runtime.get_cells_arguments
        )
        runtime.allow_disconnect.set()
        await _wait_until(lambda: not session.browser_ws_connected())
        detached = await jobs.status(started["job_id"])
        assert detached["state"] == "running"
        assert detached["tracking_state"] == "detached"

        second = NotebookFrontend(
            _websocket_url(session), runtime, disconnect_tracked_run=False
        )
        second.start()
        reconnected = await session.connect(wait_seconds=2.0)
        assert reconnected.runtime_alive is True
        assert reconnected.connection_id == initial.connection_id
        assert reconnected.url == initial.url

        await runtime.execution_finished.wait()
        await _wait_until(
            lambda: jobs.jobs[started["job_id"]].state == "finished",
            timeout=5.0,
        )
        finished = await jobs.status(started["job_id"])
        artifact = finished["output_artifact"]
        assert finished["output_truncated"] is True
        assert artifact["storage"] == "colab_runtime"

        chunks = bytearray()
        offset = 0
        while True:
            chunk = await jobs.read_artifact(
                artifact["artifact_id"], offset=offset, limit_bytes=64 * 1024
            )
            assert chunk["encoding"] == "base64"
            import base64

            chunks.extend(base64.b64decode(chunk["data"]))
            if chunk["eof"]:
                break
            offset = chunk["next_offset"]
        assert hashlib.sha256(chunks).hexdigest() == artifact["sha256"]
        assert len(chunks) >= 200000
    finally:
        for task in tuple(background_tasks):
            task.cancel()
        if background_tasks:
            await asyncio.gather(*background_tasks, return_exceptions=True)
        await session.close()
        await first.close()
        if second is not None:
            await second.close()
        await runtime.close()
