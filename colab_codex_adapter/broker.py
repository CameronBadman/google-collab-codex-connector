from __future__ import annotations

import asyncio
import fcntl
import json
import os
import secrets
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import IO, Any

from fastmcp import Client

from .diagnostics import DEFAULT_RUNTIME_DIR

DEFAULT_BROKER_HOST = "127.0.0.1"
DEFAULT_BROKER_PORT = 8765
DEFAULT_BROKER_STATE_FILE = DEFAULT_RUNTIME_DIR / "broker.json"
DEFAULT_BROKER_LOCK_FILE = DEFAULT_RUNTIME_DIR / "broker.lock"
BROKER_START_TIMEOUT_SECONDS = 10.0


@dataclass(frozen=True)
class BrokerState:
    endpoint: str
    token: str
    owner_pid: int
    started_at: float

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "BrokerState":
        return cls(
            endpoint=str(value["endpoint"]),
            token=str(value["token"]),
            owner_pid=int(value["owner_pid"]),
            started_at=float(value["started_at"]),
        )

    def public_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("token")
        return value


def read_broker_state(path: Path = DEFAULT_BROKER_STATE_FILE) -> BrokerState | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            return None
        return BrokerState.from_dict(value)
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


async def broker_is_healthy(state: BrokerState) -> bool:
    try:
        client = Client(
            state.endpoint,
            auth=state.token,
            timeout=1.0,
            init_timeout=1.0,
        )
        async with client:
            await client.list_tools()
        return True
    except Exception:
        return False


class BrokerCoordinator:
    """Elect one stdio adapter as broker owner and connect later adapters to it."""

    def __init__(
        self,
        *,
        host: str = DEFAULT_BROKER_HOST,
        port: int = DEFAULT_BROKER_PORT,
        state_file: Path = DEFAULT_BROKER_STATE_FILE,
        lock_file: Path = DEFAULT_BROKER_LOCK_FILE,
        startup_timeout: float = BROKER_START_TIMEOUT_SECONDS,
    ) -> None:
        self.host = host
        self.port = port
        self.state_file = state_file
        self.lock_file = lock_file
        self.startup_timeout = startup_timeout
        self.is_owner = False
        self._lock_handle: IO[str] | None = None

    async def claim(self) -> BrokerState:
        runtime_dir = self.state_file.parent
        runtime_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(runtime_dir, 0o700)
        deadline = time.monotonic() + self.startup_timeout

        while True:
            handle = self.lock_file.open("a+", encoding="utf-8")
            os.chmod(self.lock_file, 0o600)
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close()
                state = read_broker_state(self.state_file)
                if state is not None and await broker_is_healthy(state):
                    return state
            else:
                self.is_owner = True
                self._lock_handle = handle
                self.state_file.unlink(missing_ok=True)
                return BrokerState(
                    endpoint=f"http://{self.host}:{self.port}/mcp",
                    token=secrets.token_urlsafe(32),
                    owner_pid=os.getpid(),
                    started_at=time.time(),
                )

            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "Timed out waiting for the shared Colab broker to start"
                )
            await asyncio.sleep(0.1)

    async def wait_until_healthy(
        self,
        state: BrokerState,
        server_task: asyncio.Task[None] | None = None,
    ) -> None:
        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            if server_task is not None and server_task.done():
                await server_task
                raise RuntimeError("Shared Colab broker stopped during startup")
            if await broker_is_healthy(state):
                return
            await asyncio.sleep(0.1)
        raise TimeoutError(
            f"Timed out starting shared Colab broker at {state.endpoint}"
        )

    def publish(self, state: BrokerState) -> None:
        if not self.is_owner:
            raise RuntimeError("Only the broker owner can publish broker state")
        temporary = self.state_file.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(asdict(state), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.chmod(temporary, 0o600)
        temporary.replace(self.state_file)

    def release(self, state: BrokerState | None = None) -> None:
        if self.is_owner:
            current = read_broker_state(self.state_file)
            if state is None or current is None or current.owner_pid == state.owner_pid:
                self.state_file.unlink(missing_ok=True)
        if self._lock_handle is not None:
            fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_UN)
            self._lock_handle.close()
            self._lock_handle = None
        self.is_owner = False
