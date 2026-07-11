from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import math
import os
import subprocess
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AsyncExitStack
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal

from fastmcp.client.transports import ClientTransport
from mcp.client.session import ClientSession
from mcp.types import CallToolResult, Tool

from .bridge import BrowserConnection, ColabWebSocketServer

REMOTE_INIT_TIMEOUT_ENV = "COLAB_CODEX_REMOTE_INIT_TIMEOUT_SECONDS"
REMOTE_TOOL_LIST_TIMEOUT_ENV = "COLAB_CODEX_REMOTE_TOOL_LIST_TIMEOUT_SECONDS"
REMOTE_INIT_TIMEOUT_SECONDS = 30.0
REMOTE_TOOL_LIST_TIMEOUT_SECONDS = 30.0
REMOTE_TOOL_CALL_TIMEOUT_SECONDS = 300.0
CLIENT_CLEANUP_TIMEOUT_SECONDS = 0.5
OPEN_URL_PATH = "/tmp/colab-codex-adapter/open-url"
NATIVE_BROWSER_ENV = "COLAB_CODEX_OPEN_NATIVE_BROWSER"
FALSE_ENV_VALUES = {"0", "false", "no", "off"}

TransitionKind = Literal[
    "browser_connected",
    "runtime_ready",
    "browser_disconnected",
    "reset",
]


def _configured_timeout(env_name: str, default: float, value: float | None) -> float:
    if value is None:
        raw_value = os.environ.get(env_name)
        if raw_value is None:
            return default
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise ValueError(f"{env_name} must be a number") from exc
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{env_name} must be a finite positive number")
    return value


class ColabTransport(ClientTransport):
    """FastMCP transport bound to exactly one browser generation."""

    def __init__(self, connection: BrowserConnection):
        self.connection = connection

    @contextlib.asynccontextmanager
    async def connect_session(
        self, **session_kwargs: Any
    ) -> AsyncIterator[ClientSession]:
        async with ClientSession(
            self.connection.read_stream,
            self.connection.write_stream,
            **session_kwargs,
        ) as session:
            yield session

    def __repr__(self) -> str:
        return f"<ColabCodexAdapterTransport generation={self.connection.generation}>"


@dataclass(frozen=True)
class ConnectionTransition:
    kind: TransitionKind
    connection_id: str
    browser_generation: int
    occurred_at: float
    browser_alive: bool
    runtime_alive: bool
    close_code: int | None = None
    close_reason: str | None = None


ConnectionListener = Callable[
    [ConnectionTransition], Awaitable[None] | None
]


@dataclass
class ConnectionStatus:
    connected: bool
    connecting: bool
    server_listening: bool
    browser_ws_connected: bool
    remote_mcp_initialized: bool
    browser_alive: bool
    runtime_alive: bool
    url: str
    port: int
    adapter_pid: int
    adapter_started_at: float
    connection_id: str
    browser_generation: int
    runtime_generation: int | None
    last_state_change: float
    token_prefix: str
    open_url_path: str
    browser_launch_attempted: bool
    browser_launch_succeeded: bool | None
    browser_launch_error: str | None
    websocket_max_frame_bytes: int
    rejected_browser_connections: int
    last_browser_close_code: int | None
    last_browser_close_reason: str | None
    browser_received_frames: int
    browser_received_bytes: int
    browser_largest_received_frame_bytes: int
    browser_rejected_frame_bytes: int
    browser_sent_frames: int
    browser_sent_bytes: int
    browser_largest_sent_frame_bytes: int
    remote_tool_count: int | None = None
    last_error: str | None = None


class NotConnectedError(RuntimeError):
    pass


class TransportDisconnected(NotConnectedError):
    """A request lost the browser transport and must not be replayed blindly."""

    def __init__(self, connection_id: str, browser_generation: int) -> None:
        self.connection_id = connection_id
        self.browser_generation = browser_generation
        super().__init__(
            "Colab browser transport disconnected "
            f"(connection_id={connection_id}, generation={browser_generation})"
        )


class ColabSessionManager:
    def __init__(
        self,
        *,
        bridge_host: str = "localhost",
        bridge_port: int = 0,
        bridge_token: str | None = None,
        notebook_url: str | None = None,
        websocket_max_frame_bytes: int | None = None,
        connection_id: str | None = None,
        remote_init_timeout_seconds: float | None = None,
        remote_tool_list_timeout_seconds: float | None = None,
    ) -> None:
        self._exit_stack = AsyncExitStack()
        self._bridge: ColabWebSocketServer | None = None
        self._client: ClientSession | None = None
        self._supervisor_task: asyncio.Task[None] | None = None
        self._remote_tools: list[Tool] | None = None
        self._runtime_generation: int | None = None
        self._initializing_generation: int | None = None
        self._last_error: str | None = None
        self._browser_launch_attempted = False
        self._browser_launch_succeeded: bool | None = None
        self._browser_launch_error: str | None = None
        self._lock = asyncio.Lock()
        self._ready = asyncio.Event()
        self._listeners: set[ConnectionListener] = set()
        self._adapter_started_at = time.time()
        self._connection_id = connection_id or uuid.uuid4().hex
        self._last_state_change = self._adapter_started_at
        self._remote_init_timeout_seconds = _configured_timeout(
            REMOTE_INIT_TIMEOUT_ENV,
            REMOTE_INIT_TIMEOUT_SECONDS,
            remote_init_timeout_seconds,
        )
        self._remote_tool_list_timeout_seconds = _configured_timeout(
            REMOTE_TOOL_LIST_TIMEOUT_ENV,
            REMOTE_TOOL_LIST_TIMEOUT_SECONDS,
            remote_tool_list_timeout_seconds,
        )
        self._bridge_options: dict[str, Any] = {
            "host": bridge_host,
            "port": bridge_port,
            "token": bridge_token,
            "notebook_url": notebook_url,
            "max_frame_bytes": websocket_max_frame_bytes,
        }

    @property
    def remote_init_timeout_seconds(self) -> float:
        return self._remote_init_timeout_seconds

    @property
    def remote_tool_list_timeout_seconds(self) -> float:
        return self._remote_tool_list_timeout_seconds

    @property
    def connection_id(self) -> str:
        return self._connection_id

    @property
    def browser_generation(self) -> int:
        return self._bridge.generation if self._bridge else 0

    @property
    def runtime_generation(self) -> int | None:
        return self._runtime_generation

    def add_connection_listener(self, listener: ConnectionListener) -> None:
        self._listeners.add(listener)

    def remove_connection_listener(self, listener: ConnectionListener) -> None:
        self._listeners.discard(listener)

    async def _emit_transition(self, transition: ConnectionTransition) -> None:
        for listener in tuple(self._listeners):
            try:
                result = listener(transition)
                if inspect.isawaitable(result):
                    await result
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logging.error(
                    "Colab connection listener failed event=%s generation=%s "
                    "error_type=%s",
                    transition.kind,
                    transition.browser_generation,
                    type(exc).__name__,
                )

    def _transition(
        self,
        kind: TransitionKind,
        generation: int,
        *,
        browser_alive: bool,
        runtime_alive: bool,
        close_code: int | None = None,
        close_reason: str | None = None,
    ) -> ConnectionTransition:
        return ConnectionTransition(
            kind=kind,
            connection_id=self._connection_id,
            browser_generation=generation,
            occurred_at=time.time(),
            browser_alive=browser_alive,
            runtime_alive=runtime_alive,
            close_code=close_code,
            close_reason=close_reason,
        )

    async def start(self) -> None:
        async with self._lock:
            await self._start_locked()

    async def _start_locked(self) -> None:
        if self._bridge is not None:
            self._ensure_supervisor()
            return
        self._bridge = await self._exit_stack.enter_async_context(
            ColabWebSocketServer(**self._bridge_options)
        )
        self._last_state_change = time.time()
        self._ensure_supervisor()

    async def close(self) -> None:
        async with self._lock:
            await self._close_locked()

    async def _close_locked(self) -> None:
        supervisor = self._supervisor_task
        self._supervisor_task = None
        if supervisor:
            supervisor.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await supervisor
        self._clear_runtime()
        await self._exit_stack.aclose()
        self._exit_stack = AsyncExitStack()
        self._bridge = None
        self._last_state_change = time.time()

    async def reset(
        self, wait_seconds: float = 1.0, open_browser: bool = False
    ) -> ConnectionStatus:
        async with self._lock:
            old_generation = self.browser_generation
            await self._emit_transition(
                self._transition(
                    "reset",
                    old_generation,
                    browser_alive=self.browser_ws_connected(),
                    runtime_alive=self.is_connected(),
                )
            )
            await self._close_locked()
            self._connection_id = uuid.uuid4().hex
            self._bridge_options["token"] = None
            self._bridge_options["port"] = 0
            self._last_error = None
            self._browser_launch_attempted = False
            self._browser_launch_succeeded = None
            self._browser_launch_error = None
            await self._start_locked()
        return await self.connect(wait_seconds=wait_seconds, open_browser=open_browser)

    @property
    def bridge(self) -> ColabWebSocketServer:
        if self._bridge is None:
            raise RuntimeError("Colab session manager has not started")
        return self._bridge

    def private_bridge_state(self) -> dict[str, Any]:
        """Return restart state for a private, permission-protected broker file."""
        bridge = self.bridge
        return {
            "token": bridge.token,
            "port": bridge.port,
            "notebook_url": bridge.notebook_url,
            "max_frame_bytes": bridge.max_frame_bytes,
            "connection_id": self._connection_id,
        }

    def is_connected(self) -> bool:
        active = self._bridge.active_connection if self._bridge else None
        return (
            active is not None
            and self._ready.is_set()
            and self._client is not None
            and self._remote_tools is not None
            and self._runtime_generation == active.generation
        )

    def browser_ws_connected(self) -> bool:
        return self._bridge is not None and self._bridge.active_connection is not None

    def _is_connecting(self) -> bool:
        return self.browser_ws_connected() and not self.is_connected()

    def _ensure_supervisor(self) -> None:
        if self._supervisor_task is None or self._supervisor_task.done():
            self._supervisor_task = asyncio.create_task(self._supervise_connections())

    def _clear_runtime(self, generation: int | None = None) -> None:
        if generation is not None and self._runtime_generation != generation:
            return
        self._client = None
        self._remote_tools = None
        self._runtime_generation = None
        self._initializing_generation = None
        self._ready.clear()

    async def _supervise_connections(self) -> None:
        generation = 0
        try:
            while self._bridge is not None:
                bridge = self._bridge
                connection = await bridge.wait_for_connection(generation)
                generation = connection.generation
                self._initializing_generation = generation
                self._last_state_change = time.time()
                await self._emit_transition(
                    self._transition(
                        "browser_connected",
                        generation,
                        browser_alive=True,
                        runtime_alive=False,
                    )
                )

                client_context = ClientSession(
                    connection.read_stream,
                    connection.write_stream,
                )
                client: ClientSession | None = None
                stage = "initialization"
                try:
                    client = await client_context.__aenter__()
                    await asyncio.wait_for(
                        client.initialize(),
                        timeout=self._remote_init_timeout_seconds,
                    )
                    stage = "tool discovery"
                    tool_result = await asyncio.wait_for(
                        client.list_tools(),
                        timeout=self._remote_tool_list_timeout_seconds,
                    )
                    tools = tool_result.tools
                    if connection.closed.is_set() or bridge.active_connection is not connection:
                        raise TransportDisconnected(
                            self._connection_id, connection.generation
                        )
                    self._client = client
                    self._remote_tools = tools
                    self._runtime_generation = generation
                    self._initializing_generation = None
                    self._last_error = None
                    self._last_state_change = time.time()
                    self._ready.set()
                    await self._emit_transition(
                        self._transition(
                            "runtime_ready",
                            generation,
                            browser_alive=True,
                            runtime_alive=True,
                        )
                    )
                    await connection.closed.wait()
                except asyncio.TimeoutError:
                    self._last_error = f"Timed out during Colab frontend {stage}"
                    self._last_state_change = time.time()
                    logging.warning(
                        "Colab frontend timeout stage=%s generation=%s",
                        stage,
                        generation,
                    )
                    await connection.request_close(
                        code=1011, reason=f"MCP {stage} timeout"
                    )
                except asyncio.CancelledError:
                    raise
                except TransportDisconnected:
                    self._last_error = "Colab frontend transport disconnected"
                    self._last_state_change = time.time()
                except Exception as exc:
                    self._last_error = (
                        f"{type(exc).__name__} during Colab frontend {stage}"
                    )
                    self._last_state_change = time.time()
                    logging.warning(
                        "Failed Colab frontend MCP stage=%s generation=%s "
                        "error_type=%s",
                        stage,
                        generation,
                        type(exc).__name__,
                    )
                    await connection.request_close(
                        code=1011, reason=f"MCP {stage} failed"
                    )
                finally:
                    self._clear_runtime(generation)
                    if client is not None:
                        with contextlib.suppress(Exception, asyncio.TimeoutError):
                            await asyncio.wait_for(
                                client_context.__aexit__(None, None, None),
                                timeout=CLIENT_CLEANUP_TIMEOUT_SECONDS,
                            )
                    if not connection.closed.is_set():
                        with contextlib.suppress(Exception):
                            await connection.request_close(
                                code=1001, reason="Session stopped"
                            )
                    if not connection.closed.is_set():
                        with contextlib.suppress(asyncio.TimeoutError):
                            await asyncio.wait_for(connection.closed.wait(), timeout=2)
                    diagnostics = connection.diagnostics
                    await self._emit_transition(
                        self._transition(
                            "browser_disconnected",
                            generation,
                            browser_alive=False,
                            runtime_alive=False,
                            close_code=diagnostics.close_code,
                            close_reason=diagnostics.close_reason,
                        )
                    )
                    self._last_state_change = time.time()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._last_error = f"{type(exc).__name__} in connection supervisor"
            self._last_state_change = time.time()
            logging.error(
                "Colab connection supervisor stopped error_type=%s",
                type(exc).__name__,
            )

    async def ensure_connecting(self) -> None:
        await self.start()

    async def connect(
        self, wait_seconds: float = 60.0, open_browser: bool = False
    ) -> ConnectionStatus:
        await self.ensure_connecting()
        if open_browser:
            async with self._lock:
                self._publish_browser_url()

        if not self.is_connected() and wait_seconds > 0:
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(
                    self._ready.wait(),
                    timeout=wait_seconds,
                )

        return await self.status(include_remote_tools=True)

    def _publish_browser_url(self) -> None:
        path = Path(OPEN_URL_PATH)
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if path.parent.is_symlink():
            raise OSError("Colab URL directory cannot be a symbolic link")
        os.chmod(path.parent, 0o700)
        temporary = path.with_name(
            f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        descriptor = os.open(temporary, flags, 0o600)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(self.bridge.browser_url + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            os.chmod(path, 0o600)
        finally:
            temporary.unlink(missing_ok=True)

        enabled = os.environ.get(NATIVE_BROWSER_ENV, "1").strip().lower()
        if enabled in FALSE_ENV_VALUES:
            return

        self._browser_launch_attempted = True
        environment = os.environ.copy()
        environment.pop("BROWSER", None)
        try:
            subprocess.Popen(
                ["xdg-open", self.bridge.browser_url],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
                start_new_session=True,
                env=environment,
            )
        except OSError as exc:
            self._browser_launch_succeeded = False
            self._browser_launch_error = f"{type(exc).__name__}"
            logging.warning(
                "Failed to open the Colab URL error_type=%s", type(exc).__name__
            )
        else:
            self._browser_launch_succeeded = True
            self._browser_launch_error = None

    async def status(self, include_remote_tools: bool = False) -> ConnectionStatus:
        await self.start()
        remote_tool_count = (
            len(self._remote_tools) if self._remote_tools is not None else None
        )
        if include_remote_tools and self.is_connected():
            try:
                remote_tool_count = len(
                    await self.list_tools(
                        timeout=self._remote_tool_list_timeout_seconds
                    )
                )
            except asyncio.TimeoutError:
                self._last_error = "Timed out listing Colab frontend tools"
            except TransportDisconnected:
                self._last_error = "Colab frontend transport disconnected"
            except Exception as exc:
                self._last_error = (
                    f"{type(exc).__name__} while listing Colab frontend tools"
                )
        async with self._lock:
            await self._start_locked()
            bridge = self.bridge
            diagnostics = bridge.diagnostics()
            connected = self.is_connected()
            browser_connected = self.browser_ws_connected()
            return ConnectionStatus(
                connected=connected,
                connecting=self._is_connecting(),
                server_listening=bridge.port != 0,
                browser_ws_connected=browser_connected,
                remote_mcp_initialized=connected,
                browser_alive=browser_connected,
                runtime_alive=connected,
                url=bridge.browser_url,
                port=bridge.port,
                adapter_pid=os.getpid(),
                adapter_started_at=self._adapter_started_at,
                connection_id=self._connection_id,
                browser_generation=diagnostics["browser_generation"],
                runtime_generation=self._runtime_generation,
                last_state_change=self._last_state_change,
                token_prefix=bridge.token[:8],
                open_url_path=OPEN_URL_PATH,
                browser_launch_attempted=self._browser_launch_attempted,
                browser_launch_succeeded=self._browser_launch_succeeded,
                browser_launch_error=self._browser_launch_error,
                websocket_max_frame_bytes=diagnostics["websocket_max_frame_bytes"],
                rejected_browser_connections=diagnostics[
                    "rejected_browser_connections"
                ],
                last_browser_close_code=diagnostics["last_browser_close_code"],
                last_browser_close_reason=diagnostics["last_browser_close_reason"],
                browser_received_frames=diagnostics["browser_received_frames"],
                browser_received_bytes=diagnostics["browser_received_bytes"],
                browser_largest_received_frame_bytes=diagnostics[
                    "browser_largest_received_frame_bytes"
                ],
                browser_rejected_frame_bytes=diagnostics[
                    "browser_rejected_frame_bytes"
                ],
                browser_sent_frames=diagnostics["browser_sent_frames"],
                browser_sent_bytes=diagnostics["browser_sent_bytes"],
                browser_largest_sent_frame_bytes=diagnostics[
                    "browser_largest_sent_frame_bytes"
                ],
                remote_tool_count=remote_tool_count,
                last_error=self._last_error,
            )

    async def connection_url(self) -> dict[str, Any]:
        status = await self.status(include_remote_tools=False)
        return {
            "url": status.url,
            "port": status.port,
            "token_prefix": status.token_prefix,
            "connection_id": status.connection_id,
            "adapter_pid": status.adapter_pid,
            "adapter_started_at": status.adapter_started_at,
            "last_state_change": status.last_state_change,
            "browser_generation": status.browser_generation,
            "runtime_generation": status.runtime_generation,
            "browser_alive": status.browser_alive,
            "runtime_alive": status.runtime_alive,
            "last_browser_close_code": status.last_browser_close_code,
            "last_browser_close_reason": status.last_browser_close_reason,
            "open_url_path": status.open_url_path,
            "browser_launch_attempted": status.browser_launch_attempted,
            "browser_launch_succeeded": status.browser_launch_succeeded,
            "browser_launch_error": status.browser_launch_error,
        }

    def require_client(self) -> ClientSession:
        if self.is_connected() and self._client is not None:
            return self._client
        if self.browser_generation > 0 and not self.browser_ws_connected():
            raise TransportDisconnected(
                self._connection_id, self.browser_generation
            )
        raise NotConnectedError(
            "Colab browser session is not connected. Call colab_connect first "
            "and open the returned URL."
        )

    async def list_tools(
        self, timeout: float | None = None
    ) -> list[Tool]:
        if timeout is None:
            timeout = self._remote_tool_list_timeout_seconds
        client = self.require_client()
        generation = self._runtime_generation or self.browser_generation
        try:
            result = await asyncio.wait_for(client.list_tools(), timeout)
        except Exception as exc:
            if not self.is_connected():
                raise TransportDisconnected(
                    self._connection_id, generation
                ) from exc
            raise
        self._remote_tools = result.tools
        return result.tools

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        timeout: float = REMOTE_TOOL_CALL_TIMEOUT_SECONDS,
    ) -> CallToolResult:
        client = self.require_client()
        generation = self._runtime_generation or self.browser_generation
        try:
            return await asyncio.wait_for(
                client.call_tool(
                    name,
                    arguments or {},
                    read_timeout_seconds=timedelta(seconds=timeout),
                ),
                timeout,
            )
        except Exception as exc:
            if not self.is_connected():
                raise TransportDisconnected(
                    self._connection_id, generation
                ) from exc
            raise
