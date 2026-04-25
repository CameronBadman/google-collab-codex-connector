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

REMOTE_INIT_TIMEOUT_SECONDS = 5.0
REMOTE_TOOL_LIST_TIMEOUT_SECONDS = 5.0
REMOTE_TOOL_CALL_TIMEOUT_SECONDS = 300.0


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
    server_listening: bool
    browser_ws_connected: bool
    remote_mcp_initialized: bool
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
        self._remote_tools: list[Tool] | None = None
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
        self._remote_tools = None

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
            and self._remote_tools is not None
        )

    def browser_ws_connected(self) -> bool:
        return self._bridge is not None and self._bridge.connection_live.is_set()

    def _is_connecting(self) -> bool:
        return self._connect_task is not None and not self._connect_task.done()

    def _start_connect_task(self) -> None:
        if self._is_connecting():
            return
        self._client = None
        self._remote_tools = None
        self._connect_task = asyncio.create_task(self._connect_client())

    async def _connect_client(self) -> None:
        if self._bridge is None:
            return
        try:
            await self._bridge.connection_live.wait()
            client = await asyncio.wait_for(
                self._exit_stack.enter_async_context(
                    Client(
                        ColabTransport(self._bridge),
                        init_timeout=REMOTE_INIT_TIMEOUT_SECONDS,
                    )
                ),
                timeout=REMOTE_INIT_TIMEOUT_SECONDS,
            )
            tools = await asyncio.wait_for(
                client.list_tools(), timeout=REMOTE_TOOL_LIST_TIMEOUT_SECONDS
            )
            self._client = client
            self._remote_tools = tools
            self._last_error = None
        except asyncio.TimeoutError:
            self._client = None
            self._remote_tools = None
            self._last_error = "Timed out initializing Colab frontend MCP session"
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._client = None
            self._remote_tools = None
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
        remote_tool_count = len(self._remote_tools) if self._remote_tools else None
        if include_remote_tools and self.is_connected():
            try:
                remote_tool_count = len(
                    await self.list_tools(timeout=REMOTE_TOOL_LIST_TIMEOUT_SECONDS)
                )
            except asyncio.TimeoutError:
                self._last_error = "Timed out listing Colab frontend tools"
            except Exception as exc:
                self._last_error = f"{type(exc).__name__}: {exc}"
        return ConnectionStatus(
            connected=self.is_connected(),
            connecting=self._is_connecting(),
            server_listening=self._bridge is not None and self.bridge.port != 0,
            browser_ws_connected=self.browser_ws_connected(),
            remote_mcp_initialized=self.is_connected(),
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

    async def list_tools(
        self, timeout: float = REMOTE_TOOL_LIST_TIMEOUT_SECONDS
    ) -> list[Tool]:
        tools = await asyncio.wait_for(self.require_client().list_tools(), timeout)
        self._remote_tools = tools
        return tools

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        timeout: float = REMOTE_TOOL_CALL_TIMEOUT_SECONDS,
    ) -> CallToolResult:
        return await asyncio.wait_for(
            self.require_client().call_tool(name, arguments or {}, timeout=timeout),
            timeout,
        )
