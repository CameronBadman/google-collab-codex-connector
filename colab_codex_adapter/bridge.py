# Copyright 2026 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted from googlecolab/colab-mcp's websocket bridge. This module keeps
# the browser-facing protocol compatible while the public MCP server exposes
# static tools for clients that do not consume tools/list_changed updates.

from __future__ import annotations

import asyncio
import logging
import secrets
from typing import Any

import anyio
import mcp.types as types
import websockets
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from mcp.shared.message import SessionMessage
from pydantic_core import ValidationError
from websockets.asyncio.server import ServerConnection
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Request, Response
from websockets.typing import Subprotocol

COLAB = "https://colab.research.google.com"
COLAB_ALT_DOMAIN = "https://colab.google.com"
SCRATCH_PATH = "/notebooks/empty.ipynb"


class ColabWebSocketServer:
    """Accepts one MCP websocket connection from a Google Colab browser tab."""

    def __init__(self, host: str = "localhost") -> None:
        self.host = host
        self.port = 0
        self.connection_lock = asyncio.Lock()
        self.connection_live = asyncio.Event()
        self.allowed_origins = [COLAB, COLAB_ALT_DOMAIN]
        self.token = secrets.token_urlsafe(16)

        self._server: websockets.Server | None = None
        self._read_stream_writer: MemoryObjectSendStream[SessionMessage | Exception]
        self.read_stream: MemoryObjectReceiveStream[SessionMessage | Exception]
        self.write_stream: MemoryObjectSendStream[SessionMessage]
        self._write_stream_reader: MemoryObjectReceiveStream[SessionMessage]

        self._read_stream_writer, self.read_stream = anyio.create_memory_object_stream(
            0
        )
        self.write_stream, self._write_stream_reader = (
            anyio.create_memory_object_stream(0)
        )

    @property
    def browser_url(self) -> str:
        return (
            f"{COLAB}{SCRATCH_PATH}"
            f"#mcpProxyToken={self.token}&mcpProxyPort={self.port}"
        )

    async def _read_from_socket(self, websocket: ServerConnection) -> None:
        async for msg in websocket:
            try:
                client_message = types.JSONRPCMessage.model_validate_json(msg)
            except ValidationError as exc:
                await self._read_stream_writer.send(exc)
                continue
            await self._read_stream_writer.send(SessionMessage(client_message))

    async def _write_to_socket(self, websocket: ServerConnection) -> None:
        try:
            while True:
                msg = await self._write_stream_reader.receive()
                try:
                    payload = msg.message.model_dump_json(
                        by_alias=True, exclude_none=True
                    )
                    await websocket.send(payload)
                except ConnectionClosed:
                    break
        except (anyio.ClosedResourceError, anyio.EndOfStream):
            pass

    def _validate_authorization(
        self, websocket: ServerConnection, request: Request
    ) -> Response | None:
        del websocket
        if request.path.find(f"access_token={self.token}") != -1:
            return None
        try:
            headers: Headers = request.headers
            auth_header = headers.get("Authorization")
            if not auth_header:
                return Response(401, "Missing authorization", Headers([]))
            scheme, token = auth_header.split(None, 1)
            if scheme.lower() != "bearer":
                return Response(400, "Invalid authorization header", Headers([]))
        except ValueError:
            return Response(400, "Invalid header format", Headers([]))
        if token == self.token:
            return None
        return Response(403, "Bad authorization token", Headers([]))

    async def _connection_handler(self, websocket: ServerConnection) -> None:
        if self.connection_lock.locked():
            logging.warning("Rejected second Colab websocket connection")
            await websocket.close(code=1013, reason="Server is busy")
            return

        async with self.connection_lock:
            try:
                self.connection_live.set()
                reading_task = asyncio.create_task(self._read_from_socket(websocket))
                writing_task = asyncio.create_task(self._write_to_socket(websocket))
                _, pending = await asyncio.wait(
                    [reading_task, writing_task], return_when=asyncio.FIRST_COMPLETED
                )
                for task in pending:
                    task.cancel()
            except websockets.exceptions.ConnectionClosed as exc:
                logging.info("Colab websocket closed: %s - %s", exc.code, exc.reason)
                await self._read_stream_writer.send(
                    Exception("Colab frontend disconnected")
                )
            except Exception:
                logging.exception("Unexpected Colab websocket error")
            finally:
                self.connection_live.clear()

    async def __aenter__(self) -> "ColabWebSocketServer":
        self._server = await websockets.serve(
            self._connection_handler,
            host=self.host,
            port=0,
            subprotocols=[Subprotocol("mcp")],
            origins=self.allowed_origins,
            process_request=self._validate_authorization,
        )
        self.port = self._server.sockets[0].getsockname()[1]
        logging.info(
            "Started Colab websocket server on ws://%s:%s", self.host, self.port
        )
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        logging.info("Closing Colab websocket server")
        if self._server:
            self._server.close()
            self.write_stream.close()
            self.read_stream.close()
            await self._server.wait_closed()
