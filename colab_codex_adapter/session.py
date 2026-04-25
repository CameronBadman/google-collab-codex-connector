from __future__ import annotations

import asyncio
import contextlib
import logging
import webbrowser
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any

from fastmcp import Client
from fastmcp.client.transports import ClientTransport
from mcp.client.session import ClientSession
from mcp.types import CallToolResult, Tool

from .bridge import ColabWebSocketServer


class ColabTransport(ClientTransport):
    def __init__(self, bridge: ColabWebSocketServer):
        self.bridge = bridge

    @contextlib.asynccontextmanager
    async def connect_session(
        self, **session_kwargs: Any
    ) -> AsyncIterator[ClientSession]:
        async with ClientSession(
            self.bridge.read_stream, self.bridge.write_stream, **session_kwargs
        ) as session:
            yield session

    def __repr__(self) -> str:
        return "<ColabCodexAdapterTransport>"


@dataclass
class ConnectionStatus:
    connected: bool
    connecting: bool
    url: str
    port: int
    remote_tool_count: int | None = None
    last_error: str | None = None


class NotConnectedError(RuntimeError):
    pass


class ColabSessionManager:
    def __init__(self) -> None:
        self._exit_stack = AsyncExitStack()
        self._bridge: ColabWebSocketServer | None = None
        self._client: Client | None = None
        self._connect_task: asyncio.Task[None] | None = None
        self._last_error: str | None = None
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        async with self._lock:
            if self._bridge is not None:
                return
            self._bridge = await self._exit_stack.enter_async_context(
                ColabWebSocketServer()
            )
            self._start_connect_task()

    async def close(self) -> None:
        if self._connect_task:
            self._connect_task.cancel()
        await self._exit_stack.aclose()
        self._bridge = None
        self._client = None
        self._connect_task = None

    @property
    def bridge(self) -> ColabWebSocketServer:
        if self._bridge is None:
            raise RuntimeError("Colab session manager has not started")
        return self._bridge

    def is_connected(self) -> bool:
        return (
            self._bridge is not None
            and self._bridge.connection_live.is_set()
            and self._client is not None
        )

    def _is_connecting(self) -> bool:
        return self._connect_task is not None and not self._connect_task.done()

    def _start_connect_task(self) -> None:
        if self._is_connecting():
            return
        self._connect_task = asyncio.create_task(self._connect_client())

    async def _connect_client(self) -> None:
        if self._bridge is None:
            return
        try:
            await self._bridge.connection_live.wait()
            self._client = await self._exit_stack.enter_async_context(
                Client(ColabTransport(self._bridge), init_timeout=None)
            )
            self._last_error = None
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._client = None
            self._last_error = f"{type(exc).__name__}: {exc}"
            logging.exception("Failed to initialize Colab frontend MCP client")

    async def ensure_connecting(self) -> None:
        await self.start()
        if not self.is_connected():
            self._start_connect_task()

    async def connect(
        self, wait_seconds: float = 60.0, open_browser: bool = True
    ) -> ConnectionStatus:
        await self.ensure_connecting()
        if open_browser:
            webbrowser.open_new(self.bridge.browser_url)

        if self.is_connected():
            return await self.status(include_remote_tools=True)

        with contextlib.suppress(asyncio.TimeoutError):
            if self._connect_task:
                await asyncio.wait_for(
                    asyncio.shield(self._connect_task), timeout=wait_seconds
                )

        return await self.status(include_remote_tools=True)

    async def status(self, include_remote_tools: bool = False) -> ConnectionStatus:
        await self.start()
        remote_tool_count: int | None = None
        if include_remote_tools and self.is_connected():
            try:
                remote_tool_count = len(await self.list_tools())
            except Exception as exc:
                self._last_error = f"{type(exc).__name__}: {exc}"
        return ConnectionStatus(
            connected=self.is_connected(),
            connecting=self._is_connecting(),
            url=self.bridge.browser_url,
            port=self.bridge.port,
            remote_tool_count=remote_tool_count,
            last_error=self._last_error,
        )

    def require_client(self) -> Client:
        if not self.is_connected() or self._client is None:
            raise NotConnectedError(
                "Colab browser session is not connected. Call colab_connect first "
                "and open the returned URL."
            )
        return self._client

    async def list_tools(self) -> list[Tool]:
        return await self.require_client().list_tools()

    async def call_tool(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> CallToolResult:
        return await self.require_client().call_tool(name, arguments or {})
