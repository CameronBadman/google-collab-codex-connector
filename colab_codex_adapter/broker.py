from __future__ import annotations

import asyncio
import fcntl
import json
import os
import secrets
import signal
import subprocess
import sys
import threading
import time
import weakref
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import IO, Any, AsyncIterator, Callable

from fastmcp import Client
from fastmcp.server.proxy import ProxyClient

from .diagnostics import DEFAULT_RUNTIME_DIR

DEFAULT_BROKER_HOST = "127.0.0.1"
DEFAULT_BROKER_PORT = 8765
DEFAULT_BROKER_STATE_FILE = DEFAULT_RUNTIME_DIR / "broker.json"
DEFAULT_BROKER_LOCK_FILE = DEFAULT_RUNTIME_DIR / "broker.lock"
DEFAULT_BROKER_LAUNCH_LOCK_FILE = DEFAULT_RUNTIME_DIR / "broker-launch.lock"
DEFAULT_BROKER_FACTORY = "colab_codex_adapter.server:run_broker_daemon_backend"
BROKER_START_TIMEOUT_SECONDS = 30.0
BROKER_HEALTH_TIMEOUT_SECONDS = 2.0
BROKER_CHILD_STOP_TIMEOUT_SECONDS = 2.0
BROKER_STATE_VERSION = 4
BROKER_PROTOCOL_VERSION = "3"


class BrokerProtocolMismatchError(RuntimeError):
    """A healthy discovered service uses an incompatible broker protocol."""

    def __init__(self, *, discovered: str, expected: str, endpoint: str) -> None:
        self.discovered = discovered
        self.expected = expected
        self.endpoint = endpoint
        super().__init__(
            "The shared Colab service at "
            f"{endpoint} uses broker protocol {discovered!r}, but this adapter "
            f"requires {expected!r}. Stop or upgrade the existing service before "
            "retrying"
        )


@dataclass(frozen=True)
class BrokerState:
    endpoint: str
    token: str
    owner_pid: int
    started_at: float
    service_instance_id: str = ""
    owner_id: str = ""
    owner_process_start: str = ""
    previous_owner_id: str = ""
    previous_generation: int | None = None
    generation: int = 1
    version: int = BROKER_STATE_VERSION
    protocol_version: str = BROKER_PROTOCOL_VERSION
    status: str = "ready"
    updated_at: float = 0.0

    def __post_init__(self) -> None:
        if not self.service_instance_id:
            object.__setattr__(
                self,
                "service_instance_id",
                self.owner_id or f"legacy-{self.owner_pid}",
            )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "BrokerState":
        owner_pid = int(value["owner_pid"])
        started_at = float(value["started_at"])
        owner_id = str(value.get("owner_id") or f"legacy-{owner_pid}")
        return cls(
            endpoint=str(value["endpoint"]),
            token=str(value["token"]),
            owner_pid=owner_pid,
            started_at=started_at,
            service_instance_id=str(value.get("service_instance_id") or owner_id),
            owner_id=owner_id,
            owner_process_start=str(value.get("owner_process_start") or ""),
            previous_owner_id=str(value.get("previous_owner_id") or ""),
            previous_generation=(
                int(value["previous_generation"])
                if value.get("previous_generation") is not None
                else None
            ),
            generation=max(1, int(value.get("generation", 1))),
            version=int(value.get("version", value.get("state_version", 1))),
            protocol_version=str(
                value.get("protocol_version", value.get("protocol", "1"))
            ),
            status=str(value.get("status", "ready")),
            updated_at=float(value.get("updated_at", started_at)),
        )

    def public_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("token")
        return value


def _prepare_private_file(path: Path) -> None:
    parent = path.parent
    existed = parent.exists()
    parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if (
        not existed
        or parent == DEFAULT_RUNTIME_DIR
        or DEFAULT_RUNTIME_DIR in parent.parents
    ):
        os.chmod(parent, 0o700)


def write_broker_state(path: Path, state: BrokerState) -> None:
    """Atomically publish private broker discovery state with mode 0600."""

    _prepare_private_file(path)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(asdict(state), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        temporary.unlink(missing_ok=True)


def read_broker_state(path: Path = DEFAULT_BROKER_STATE_FILE) -> BrokerState | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            return None
        return BrokerState.from_dict(value)
    except (
        FileNotFoundError,
        OSError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ):
        return None


async def broker_is_healthy(
    state: BrokerState,
    *,
    timeout: float = BROKER_HEALTH_TIMEOUT_SECONDS,
) -> bool:
    if state.status in {"failed", "stopped"}:
        return False
    try:
        client = Client(
            state.endpoint,
            auth=state.token,
            timeout=timeout,
            init_timeout=timeout,
        )
        async with client:
            await client.list_tools()
        return True
    except Exception:
        return False


def broker_process_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    proc_root = Path("/proc")
    if not proc_root.exists():
        return True
    try:
        stat = (proc_root / str(pid) / "stat").read_text(encoding="utf-8")
        fields = stat[stat.rfind(")") + 2 :].split()
        if fields and fields[0] == "Z":
            return False
    except FileNotFoundError:
        return False
    except OSError:
        return True
    return True


def broker_process_start_identity(pid: int) -> str | None:
    """Return Linux start ticks so stale state cannot target a reused PID."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        fields = stat[stat.rfind(")") + 2 :].split()
        return fields[19]
    except (FileNotFoundError, OSError, IndexError):
        return None


def _try_lock(path: Path) -> IO[str] | None:
    _prepare_private_file(path)
    handle = path.open("a+", encoding="utf-8")
    os.chmod(path, 0o600)
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    return handle


def _unlock(handle: IO[str] | None) -> None:
    if handle is None:
        return
    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    handle.close()


def _next_generation(previous: BrokerState | None) -> int:
    return 1 if previous is None else previous.generation + 1


@dataclass(frozen=True)
class BrokerLaunchConfig:
    host: str = DEFAULT_BROKER_HOST
    port: int = DEFAULT_BROKER_PORT
    state_file: Path = DEFAULT_BROKER_STATE_FILE
    lock_file: Path = DEFAULT_BROKER_LOCK_FILE
    launch_lock_file: Path = DEFAULT_BROKER_LAUNCH_LOCK_FILE
    factory: str = DEFAULT_BROKER_FACTORY
    startup_timeout: float = BROKER_START_TIMEOUT_SECONDS
    health_timeout: float = BROKER_HEALTH_TIMEOUT_SECONDS
    protocol_version: str = BROKER_PROTOCOL_VERSION
    python_executable: str = sys.executable

    @property
    def endpoint(self) -> str:
        return f"http://{self.host}:{self.port}/mcp"

    def daemon_command(self) -> list[str]:
        # Authentication material is intentionally absent. The daemon creates or
        # recovers its token after it starts.
        return [
            self.python_executable,
            "-m",
            "colab_codex_adapter.broker_daemon",
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--state-file",
            str(self.state_file),
            "--lock-file",
            str(self.lock_file),
            "--factory",
            self.factory,
            "--startup-timeout",
            str(self.startup_timeout),
            "--health-timeout",
            str(self.health_timeout),
            "--protocol-version",
            self.protocol_version,
        ]


PopenFactory = Callable[..., subprocess.Popen[bytes]]


async def _terminate_and_reap(process: subprocess.Popen[bytes]) -> None:
    """Stop a child that did not become the published broker owner."""

    if process.poll() is not None:
        return
    process.terminate()
    try:
        await asyncio.to_thread(
            process.wait, timeout=BROKER_CHILD_STOP_TIMEOUT_SECONDS
        )
        return
    except subprocess.TimeoutExpired:
        pass

    if process.poll() is None:
        process.kill()
    await asyncio.to_thread(process.wait, timeout=BROKER_CHILD_STOP_TIMEOUT_SECONDS)


def _reap_published_child(process: subprocess.Popen[bytes]) -> None:
    """Reap a detached owner without coupling its lifetime to the launcher."""

    threading.Thread(
        target=process.wait,
        name=f"colab-broker-reaper-{process.pid}",
        daemon=True,
    ).start()


class BrokerLauncher:
    """Discover a healthy broker or start exactly one detached replacement."""

    def __init__(
        self,
        config: BrokerLaunchConfig | None = None,
        *,
        popen_factory: PopenFactory = subprocess.Popen,
    ) -> None:
        self.config = config or BrokerLaunchConfig()
        self._popen_factory = popen_factory

    @staticmethod
    def _same_identity(first: BrokerState, second: BrokerState) -> bool:
        return (
            first.owner_id == second.owner_id
            and first.owner_pid == second.owner_pid
            and first.generation == second.generation
            and first.endpoint == second.endpoint
            and first.service_instance_id == second.service_instance_id
            and first.protocol_version == second.protocol_version
            and secrets.compare_digest(first.token, second.token)
        )

    async def _healthy_state(self) -> BrokerState | None:
        state = read_broker_state(self.config.state_file)
        if state is None:
            return None
        if (
            state.protocol_version != self.config.protocol_version
            and state.status not in {"failed", "stopped"}
            and state.owner_process_start
            and broker_process_is_alive(state.owner_pid)
            and broker_process_start_identity(state.owner_pid)
            == state.owner_process_start
        ):
            raise BrokerProtocolMismatchError(
                discovered=state.protocol_version,
                expected=self.config.protocol_version,
                endpoint=state.endpoint,
            )
        if await broker_is_healthy(state, timeout=self.config.health_timeout):
            # Token reuse permits an old state snapshot to authenticate against a
            # replacement that came up during the probe. Re-read discovery state
            # so callers never remain pinned to the dead owner's generation.
            current = read_broker_state(self.config.state_file)
            if (
                current is not None
                and current.status == "ready"
                and self._same_identity(state, current)
            ):
                if current.protocol_version != self.config.protocol_version:
                    raise BrokerProtocolMismatchError(
                        discovered=current.protocol_version,
                        expected=self.config.protocol_version,
                        endpoint=current.endpoint,
                    )
                return current
        return None

    async def ensure_running(self) -> BrokerState:
        healthy = await self._healthy_state()
        if healthy is not None:
            return healthy

        deadline = time.monotonic() + self.config.startup_timeout
        while time.monotonic() < deadline:
            launch_handle = _try_lock(self.config.launch_lock_file)
            if launch_handle is None:
                healthy = await self._healthy_state()
                if healthy is not None:
                    return healthy
                await asyncio.sleep(0.05)
                continue

            try:
                # A different launcher may have completed between our first
                # health probe and acquisition of the launch lock.
                healthy = await self._healthy_state()
                if healthy is not None:
                    return healthy
                return await self._start_replacement(deadline)
            finally:
                _unlock(launch_handle)

        raise TimeoutError("Timed out waiting for the detached Colab broker")

    async def _start_replacement(self, deadline: float) -> BrokerState:
        command = self.config.daemon_command()
        process = self._popen_factory(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            start_new_session=True,
        )
        published_owner = False
        try:
            while time.monotonic() < deadline:
                state = await self._healthy_state()
                if state is not None:
                    published_owner = state.owner_pid == process.pid
                    if published_owner and isinstance(process, subprocess.Popen):
                        _reap_published_child(process)
                    return state
                return_code = process.poll()
                if return_code is not None:
                    raise RuntimeError(
                        "Detached Colab broker exited during startup "
                        f"with status {return_code}"
                    )
                await asyncio.sleep(0.05)

            raise TimeoutError("Timed out starting the detached Colab broker")
        finally:
            if not published_owner:
                await _terminate_and_reap(process)


class BrokerClientFactory:
    """Create a fresh authenticated client for the current broker generation."""

    def __init__(
        self,
        launcher: BrokerLauncher | None = None,
        *,
        timeout: float = 1200.0,
        init_timeout: float = 30.0,
        proxy_progress: bool = False,
    ) -> None:
        self.launcher = launcher or BrokerLauncher()
        self.timeout = timeout
        self.init_timeout = init_timeout
        self.proxy_progress = proxy_progress
        self._used_states: weakref.WeakKeyDictionary[
            asyncio.Task[Any], BrokerState
        ] = weakref.WeakKeyDictionary()

    async def state(self) -> BrokerState:
        return await self.launcher.ensure_running()

    def _client_for(self, state: BrokerState) -> Client[Any]:
        client_type = ProxyClient if self.proxy_progress else Client
        return client_type(
            state.endpoint,
            auth=state.token,
            timeout=self.timeout,
            init_timeout=self.init_timeout,
        )

    async def __call__(self) -> Client[Any]:
        state = await self.state()
        task = asyncio.current_task()
        if task is not None:
            self.remember(task, state)
        return self._client_for(state)

    def remember(self, task: asyncio.Task[Any], state: BrokerState) -> None:
        self._used_states[task] = state

    def state_for(self, task: asyncio.Task[Any]) -> BrokerState | None:
        return self._used_states.get(task)

    def clear(self, task: asyncio.Task[Any]) -> None:
        self._used_states.pop(task, None)

    @asynccontextmanager
    async def client(self) -> AsyncIterator[Client[Any]]:
        state = await self.state()
        client = self._client_for(state)
        async with client:
            yield client


async def stop_broker(
    *,
    state_file: Path = DEFAULT_BROKER_STATE_FILE,
    expected_owner_id: str | None = None,
    timeout: float = 5.0,
) -> bool:
    """Stop the discovered daemon after verifying its current owner identity."""

    state = read_broker_state(state_file)
    if state is None:
        return False
    if expected_owner_id is not None and state.owner_id != expected_owner_id:
        return False
    if not broker_process_is_alive(state.owner_pid):
        return False
    if (
        state.owner_process_start
        and broker_process_start_identity(state.owner_pid)
        != state.owner_process_start
    ):
        return False

    os.kill(state.owner_pid, signal.SIGTERM)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        current = read_broker_state(state_file)
        if not broker_process_is_alive(state.owner_pid):
            return True
        if (
            current is not None
            and current.owner_id == state.owner_id
            and current.status == "stopped"
        ):
            return True
        await asyncio.sleep(0.05)
    return False


def stopped_state(state: BrokerState, status: str = "stopped") -> BrokerState:
    """Return a terminal state while preserving recovery credentials."""

    return replace(state, status=status, updated_at=time.time())
