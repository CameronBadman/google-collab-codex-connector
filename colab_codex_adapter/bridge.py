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
import contextlib
import logging
import os
import re
import secrets
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode, urlsplit, urlunsplit

import anyio
import mcp.types as types
import websockets
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from mcp.shared.message import SessionMessage
from pydantic_core import ValidationError
from websockets.asyncio.server import ServerConnection
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosed, PayloadTooBig
from websockets.http11 import Request, Response
from websockets.typing import Subprotocol

COLAB = "https://colab.research.google.com"
COLAB_ALT_DOMAIN = "https://colab.google.com"
SCRATCH_PATH = "/notebooks/empty.ipynb"
DEFAULT_NOTEBOOK_URL = f"{COLAB}{SCRATCH_PATH}"
NOTEBOOK_URL_ENV = "COLAB_CODEX_NOTEBOOK_URL"
WEBSOCKET_MAX_FRAME_BYTES_ENV = "COLAB_CODEX_WS_MAX_FRAME_BYTES"
DEFAULT_WEBSOCKET_MAX_FRAME_BYTES = 32 * 1024 * 1024
HARD_MAX_WEBSOCKET_FRAME_BYTES = 32 * 1024 * 1024
ALLOWED_COLAB_HOSTS = frozenset({"colab.research.google.com", "colab.google.com"})

_TOKEN_PATTERN = re.compile(
    r"(?i)(access_token|mcpProxyToken)(?:=|%3D)[^&#\s]+"
)


def _configured_frame_limit(value: int | None = None) -> int:
    if value is None:
        raw_value = os.environ.get(WEBSOCKET_MAX_FRAME_BYTES_ENV)
        if raw_value is None:
            return DEFAULT_WEBSOCKET_MAX_FRAME_BYTES
        try:
            value = int(raw_value)
        except ValueError as exc:
            raise ValueError(
                f"{WEBSOCKET_MAX_FRAME_BYTES_ENV} must be an integer"
            ) from exc
    if isinstance(value, bool) or value <= 0:
        raise ValueError("WebSocket frame limit must be a positive integer")
    if value > HARD_MAX_WEBSOCKET_FRAME_BYTES:
        raise ValueError(
            "WebSocket frame limit cannot exceed "
            f"{HARD_MAX_WEBSOCKET_FRAME_BYTES} bytes"
        )
    return value


def _configured_notebook_url(value: str | None = None) -> str:
    url = value or os.environ.get(NOTEBOOK_URL_ENV, DEFAULT_NOTEBOOK_URL)
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname not in ALLOWED_COLAB_HOSTS
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port not in (None, 443)
    ):
        raise ValueError(
            "Colab notebook URL must use HTTPS on colab.research.google.com "
            "or colab.google.com"
        )
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, parsed.query, ""))


def _safe_close_reason(reason: str | None) -> str | None:
    if not reason:
        return None
    redacted = _TOKEN_PATTERN.sub(r"\1=[REDACTED]", reason)
    printable = "".join(char for char in redacted if char.isprintable())
    return printable[:256]


def _message_size(message: str | bytes) -> int:
    return len(message.encode("utf-8")) if isinstance(message, str) else len(message)


@dataclass
class BrowserConnectionDiagnostics:
    generation: int
    connected_at: float
    closed_at: float | None = None
    close_code: int | None = None
    close_reason: str | None = None
    received_frames: int = 0
    received_bytes: int = 0
    largest_received_frame_bytes: int = 0
    rejected_frame_bytes: int = 0
    sent_frames: int = 0
    sent_bytes: int = 0
    largest_sent_frame_bytes: int = 0


class BrowserConnection:
    """One browser WebSocket and the MCP streams bound to that generation."""

    def __init__(self, websocket: ServerConnection, generation: int) -> None:
        self.websocket = websocket
        self.generation = generation
        self.closed = asyncio.Event()
        self.diagnostics = BrowserConnectionDiagnostics(
            generation=generation,
            connected_at=time.time(),
        )
        self._read_stream_writer: MemoryObjectSendStream[SessionMessage | Exception]
        self.read_stream: MemoryObjectReceiveStream[SessionMessage | Exception]
        self.write_stream: MemoryObjectSendStream[SessionMessage]
        self._write_stream_reader: MemoryObjectReceiveStream[SessionMessage]
        self._read_stream_writer, self.read_stream = anyio.create_memory_object_stream(1)
        self.write_stream, self._write_stream_reader = anyio.create_memory_object_stream(1)

    async def read_from_socket(self) -> None:
        async for message in self.websocket:
            size = _message_size(message)
            self.diagnostics.received_frames += 1
            self.diagnostics.received_bytes += size
            self.diagnostics.largest_received_frame_bytes = max(
                self.diagnostics.largest_received_frame_bytes, size
            )
            try:
                client_message = types.JSONRPCMessage.model_validate_json(message)
            except ValidationError as exc:
                await self._read_stream_writer.send(exc)
                continue
            await self._read_stream_writer.send(SessionMessage(client_message))

    async def write_to_socket(self) -> None:
        try:
            while True:
                message = await self._write_stream_reader.receive()
                payload = message.message.model_dump_json(
                    by_alias=True, exclude_none=True
                )
                size = _message_size(payload)
                self.diagnostics.sent_frames += 1
                self.diagnostics.sent_bytes += size
                self.diagnostics.largest_sent_frame_bytes = max(
                    self.diagnostics.largest_sent_frame_bytes, size
                )
                await self.websocket.send(payload)
        except (anyio.ClosedResourceError, anyio.EndOfStream, ConnectionClosed):
            pass

    async def request_close(self, code: int = 1011, reason: str = "MCP reset") -> None:
        if not self.closed.is_set():
            await self.websocket.close(code=code, reason=reason[:123])

    async def finish(self) -> None:
        with contextlib.suppress(Exception):
            await self.websocket.wait_closed()
        self.diagnostics.closed_at = time.time()
        self.diagnostics.close_code = self.websocket.close_code
        self.diagnostics.close_reason = _safe_close_reason(
            self.websocket.close_reason
        )
        parser_error = getattr(self.websocket.protocol, "parser_exc", None)
        if isinstance(parser_error, PayloadTooBig):
            rejected_size = (parser_error.current_size or 0) + (
                parser_error.size or 0
            )
            if rejected_size == 0:
                rejected_size = parser_error.max_size + 1
            self.diagnostics.rejected_frame_bytes = rejected_size
            self.diagnostics.largest_received_frame_bytes = max(
                self.diagnostics.largest_received_frame_bytes,
                rejected_size,
            )
            self.diagnostics.close_code = 1009
            self.diagnostics.close_reason = _safe_close_reason(
                str(parser_error)
            )
        await self._read_stream_writer.aclose()
        await self._write_stream_reader.aclose()
        self.closed.set()


class ColabWebSocketServer:
    """Accept one reusable Colab browser connection at a time."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 0,
        *,
        token: str | None = None,
        notebook_url: str | None = None,
        max_frame_bytes: int | None = None,
    ) -> None:
        self.host = host
        self.port = 0
        self._requested_port = port
        self.connection_lock = asyncio.Lock()
        self.connection_live = asyncio.Event()
        self.allowed_origins = [COLAB, COLAB_ALT_DOMAIN]
        self.token = token or secrets.token_urlsafe(16)
        if not self.token:
            raise ValueError("WebSocket token cannot be empty")
        self.notebook_url = _configured_notebook_url(notebook_url)
        self.max_frame_bytes = _configured_frame_limit(max_frame_bytes)

        self._server: websockets.Server | None = None
        self._connection_condition = asyncio.Condition()
        self._active_connection: BrowserConnection | None = None
        self._generation = 0
        self._last_connection: BrowserConnectionDiagnostics | None = None
        self._rejected_connections = 0

    @property
    def browser_url(self) -> str:
        parsed = urlsplit(self.notebook_url)
        fragment = urlencode(
            {"mcpProxyToken": self.token, "mcpProxyPort": str(self.port)}
        )
        return urlunsplit(
            (parsed.scheme, parsed.netloc, parsed.path, parsed.query, fragment)
        )

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def active_connection(self) -> BrowserConnection | None:
        connection = self._active_connection
        if connection is not None and not connection.closed.is_set():
            return connection
        return None

    @property
    def last_connection(self) -> BrowserConnectionDiagnostics | None:
        return self._last_connection

    @property
    def rejected_connections(self) -> int:
        return self._rejected_connections

    async def wait_for_connection(
        self, after_generation: int = 0
    ) -> BrowserConnection:
        async with self._connection_condition:
            await self._connection_condition.wait_for(
                lambda: (
                    self.active_connection is not None
                    and self.active_connection.generation > after_generation
                )
            )
            connection = self.active_connection
            if connection is None:  # pragma: no cover - guarded by condition
                raise RuntimeError("Browser connection disappeared")
            return connection

    def diagnostics(self) -> dict[str, Any]:
        active = self.active_connection
        current = active.diagnostics if active else self._last_connection
        return {
            "browser_alive": active is not None,
            "browser_generation": self._generation,
            "websocket_max_frame_bytes": self.max_frame_bytes,
            "rejected_browser_connections": self._rejected_connections,
            "last_browser_close_code": current.close_code if current else None,
            "last_browser_close_reason": current.close_reason if current else None,
            "browser_received_frames": current.received_frames if current else 0,
            "browser_received_bytes": current.received_bytes if current else 0,
            "browser_largest_received_frame_bytes": (
                current.largest_received_frame_bytes if current else 0
            ),
            "browser_rejected_frame_bytes": (
                current.rejected_frame_bytes if current else 0
            ),
            "browser_sent_frames": current.sent_frames if current else 0,
            "browser_sent_bytes": current.sent_bytes if current else 0,
            "browser_largest_sent_frame_bytes": (
                current.largest_sent_frame_bytes if current else 0
            ),
        }

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
        if secrets.compare_digest(token, self.token):
            return None
        return Response(403, "Bad authorization token", Headers([]))

    async def _set_active(self, connection: BrowserConnection | None) -> None:
        async with self._connection_condition:
            self._active_connection = connection
            if connection is None:
                self.connection_live.clear()
            else:
                self.connection_live.set()
            self._connection_condition.notify_all()

    async def _connection_handler(self, websocket: ServerConnection) -> None:
        if self.connection_lock.locked():
            self._rejected_connections += 1
            logging.warning("Rejected concurrent Colab websocket connection")
            await websocket.close(code=1013, reason="Another Colab tab is active")
            return

        async with self.connection_lock:
            self._generation += 1
            connection = BrowserConnection(websocket, self._generation)
            await self._set_active(connection)
            logging.info(
                "Colab websocket connected generation=%s", connection.generation
            )
            reading_task = asyncio.create_task(connection.read_from_socket())
            writing_task = asyncio.create_task(connection.write_to_socket())
            try:
                done, pending = await asyncio.wait(
                    [reading_task, writing_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                for task in done:
                    with contextlib.suppress(ConnectionClosed):
                        task.result()
            except Exception as exc:
                logging.error(
                    "Unexpected Colab websocket error generation=%s error_type=%s",
                    connection.generation,
                    type(exc).__name__,
                )
            finally:
                await connection.finish()
                self._last_connection = connection.diagnostics
                if self._active_connection is connection:
                    await self._set_active(None)
                logging.info(
                    "Colab websocket disconnected generation=%s code=%s "
                    "reason_length=%s received_bytes=%s sent_bytes=%s",
                    connection.generation,
                    connection.diagnostics.close_code,
                    len(connection.diagnostics.close_reason or ""),
                    connection.diagnostics.received_bytes,
                    connection.diagnostics.sent_bytes,
                )
                if connection.diagnostics.rejected_frame_bytes:
                    logging.warning(
                        "Rejected oversized Colab websocket frame generation=%s "
                        "payload_bytes=%s limit_bytes=%s",
                        connection.generation,
                        connection.diagnostics.rejected_frame_bytes,
                        self.max_frame_bytes,
                    )

    async def __aenter__(self) -> "ColabWebSocketServer":
        self._server = await websockets.serve(
            self._connection_handler,
            host=self.host,
            port=self._requested_port,
            subprotocols=[Subprotocol("mcp")],
            origins=self.allowed_origins,
            process_request=self._validate_authorization,
            max_size=self.max_frame_bytes,
            compression=None,
        )
        self.port = self._server.sockets[0].getsockname()[1]
        logging.info(
            "Started Colab websocket server host=%s port=%s max_frame_bytes=%s",
            self.host,
            self.port,
            self.max_frame_bytes,
        )
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        logging.info("Closing Colab websocket server")
        if self._server:
            active = self.active_connection
            if active is not None:
                await active.request_close(code=1001, reason="Bridge stopped")
            self._server.close()
            await self._server.wait_closed()
            self._server = None
            self.port = 0
