from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from . import __version__

DEFAULT_RUNTIME_DIR = Path("/tmp/colab-codex-adapter")
DEFAULT_LOG_DIR = DEFAULT_RUNTIME_DIR / "logs"
DEFAULT_PID_FILE = DEFAULT_RUNTIME_DIR / "adapter.pid"
DEFAULT_STATE_FILE = DEFAULT_RUNTIME_DIR / "adapter-state.json"
DEFAULT_BROKER_STATE_FILE = DEFAULT_RUNTIME_DIR / "broker.json"


def pid_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")


def write_pid(path: Path, pid: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{pid}\n")


def adapter_info(
    *,
    log_dir: Path = DEFAULT_LOG_DIR,
    log_file: Path | None = None,
    pid_file: Path = DEFAULT_PID_FILE,
    state_file: Path = DEFAULT_STATE_FILE,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pid = os.getpid()
    info: dict[str, Any] = {
        "adapter_version": __version__,
        "adapter_pid": pid,
        "pid_running": pid_is_running(pid),
        "log_dir": str(log_dir),
        "log_file": str(log_file) if log_file else None,
        "pid_file": str(pid_file),
        "state_file": str(state_file),
        "reported_at": time.time(),
    }
    if extra:
        info.update(extra)
    return info


def doctor(
    pid_file: Path = DEFAULT_PID_FILE,
    state_file: Path = DEFAULT_STATE_FILE,
    broker_state_file: Path = DEFAULT_BROKER_STATE_FILE,
) -> dict[str, Any]:
    pid_text: str | None = None
    pid: int | None = None
    try:
        pid_text = pid_file.read_text().strip()
        pid = int(pid_text)
    except (FileNotFoundError, ValueError):
        pass

    state = read_json(state_file)
    broker_state = read_json(broker_state_file)
    if broker_state is not None:
        broker_state.pop("token", None)
    running = pid_is_running(pid) if pid is not None else False
    return {
        "pid_file": str(pid_file),
        "state_file": str(state_file),
        "pid": pid,
        "pid_file_raw": pid_text,
        "adapter_process_running": running,
        "state": state,
        "broker_state_file": str(broker_state_file),
        "broker": broker_state,
        "diagnosis": diagnosis(running, state, broker_state),
    }


def diagnosis(
    running: bool,
    state: dict[str, Any] | None,
    broker_state: dict[str, Any] | None = None,
) -> str:
    if running:
        if broker_state is None:
            return (
                "adapter_process_running_without_shared_broker; restart Codex "
                "to load the multi-agent connector"
            )
        return "adapter_process_running"
    if state:
        return (
            "adapter_process_not_running; Codex likely needs to relaunch the MCP "
            "server because stdio transports cannot be reattached from this process"
        )
    return "no_adapter_state_found; Codex has probably not launched this adapter yet"


def main() -> None:
    print(json.dumps(doctor(), indent=2, sort_keys=True))
