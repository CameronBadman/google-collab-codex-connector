from __future__ import annotations

import argparse
import asyncio
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from . import __version__

DEFAULT_RUNTIME_DIR = Path("/tmp/colab-codex-adapter")
DEFAULT_LOG_DIR = DEFAULT_RUNTIME_DIR / "logs"
DEFAULT_PID_FILE = DEFAULT_RUNTIME_DIR / "adapter.pid"
DEFAULT_STATE_FILE = DEFAULT_RUNTIME_DIR / "adapter-state.json"
DEFAULT_BROKER_STATE_FILE = DEFAULT_RUNTIME_DIR / "broker.json"

_SENSITIVE_KEYS = {
    "args",
    "arguments",
    "checkpoint",
    "checkpoints",
    "code",
    "content",
    "contents",
    "corpus",
    "output",
    "outputs",
    "payload",
    "prompt",
    "result",
    "results",
    "source_code",
    "submitted_code",
    "traceback",
}
_CREDENTIAL_KEY_PARTS = (
    "authorization",
    "bearer",
    "password",
    "secret",
    "token",
)


def pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _safe_url(value: str) -> str:
    """Remove query and fragment credentials while keeping a useful location."""

    try:
        parts = urlsplit(value)
    except ValueError:
        return "[redacted-url]"
    if not parts.scheme or not parts.netloc:
        if parts.path and not parts.query and not parts.fragment:
            return parts.path
        return "[redacted-url]"
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def _is_sensitive_key(key: str) -> bool:
    normalized = key.casefold().replace("-", "_")
    return normalized in _SENSITIVE_KEYS or any(
        part in normalized for part in _CREDENTIAL_KEY_PARTS
    )


def redact_diagnostic_value(value: Any, *, key: str = "") -> Any:
    """Return operational metadata without credentials or user/corpus payloads."""

    if _is_sensitive_key(key):
        return None
    if isinstance(value, dict):
        return {
            str(item_key): safe_value
            for item_key, item_value in value.items()
            if (safe_value := redact_diagnostic_value(item_value, key=str(item_key)))
            is not None
        }
    if isinstance(value, (list, tuple)):
        return [
            safe_value
            for item in value
            if (safe_value := redact_diagnostic_value(item, key=key)) is not None
        ]
    if isinstance(value, str) and (
        key.casefold().endswith("url") or value.startswith(("http://", "https://"))
    ):
        return _safe_url(value)
    return value


def _prepare_state_parent(path: Path) -> None:
    parent = path.parent
    existed = parent.exists()
    parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if (
        not existed
        or parent == DEFAULT_RUNTIME_DIR
        or DEFAULT_RUNTIME_DIR in parent.parents
    ):
        os.chmod(parent, 0o700)


def _write_private_text(path: Path, value: str) -> None:
    _prepare_state_parent(path)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise
    finally:
        temporary.unlink(missing_ok=True)


def write_state(path: Path, state: dict[str, Any]) -> None:
    """Persist diagnostic metadata after removing credentials and workload data."""

    safe_state = redact_diagnostic_value(state)
    _write_private_text(
        path, json.dumps(safe_state, indent=2, sort_keys=True) + "\n"
    )


def write_pid(path: Path, pid: int) -> None:
    _write_private_text(path, f"{pid}\n")


def adapter_info(
    *,
    log_dir: Path = DEFAULT_LOG_DIR,
    log_file: Path | None = None,
    pid_file: Path | None = None,
    state_file: Path | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe the shared service without exposing connection secrets."""

    pid = os.getpid()
    info: dict[str, Any] = {
        "adapter_version": __version__,
        "log_dir": str(log_dir),
        "log_file": str(log_file) if log_file else None,
        "pid_file": str(pid_file) if pid_file is not None else None,
        "state_file": str(state_file) if state_file is not None else None,
        "reported_at": time.time(),
    }
    if extra:
        safe_extra = redact_diagnostic_value(extra)
        if isinstance(safe_extra, dict):
            info.update(safe_extra)

    try:
        service_pid = int(info.get("service_pid", info.get("broker_pid", pid)))
    except (TypeError, ValueError):
        service_pid = pid
    service_owner_id = info.get(
        "service_owner_id", info.get("broker_owner_id")
    )
    service_instance_id = (
        info.get("service_instance_id")
        or info.get("instance_id")
        or service_owner_id
    )
    service_generation = info.get(
        "service_generation", info.get("broker_generation")
    )
    service_started_at = info.get(
        "service_started_at",
        info.get("broker_started_at", info.get("adapter_started_at")),
    )
    service_status = info.get(
        "service_status", info.get("broker_status", "ready")
    )
    service_healthy = bool(
        info.get(
            "service_healthy",
            info.get("broker_healthy", info.get("broker_alive", True)),
        )
    )

    info.update(
        {
            "service_instance_id": service_instance_id,
            "service_pid": service_pid,
            "service_owner_id": service_owner_id,
            "service_generation": service_generation,
            "service_started_at": service_started_at,
            "service_status": service_status,
            "service_healthy": service_healthy,
            "instance_scope": info.get("instance_scope", "user"),
            "transport": info.get("transport", "stdio"),
            # Compatibility aliases now consistently describe the service.
            "adapter_pid": service_pid,
            "adapter_started_at": service_started_at,
            "adapter_process_running": pid_is_running(service_pid),
            "pid_running": pid_is_running(service_pid),
            "broker_pid": service_pid,
            "broker_owner_id": service_owner_id,
            "broker_generation": service_generation,
            "broker_status": service_status,
            "broker_alive": service_healthy,
            "broker_healthy": service_healthy,
        }
    )
    return info


def _read_pid(path: Path | None) -> tuple[str | None, int | None]:
    if path is None:
        return None, None
    try:
        raw = path.read_text(encoding="utf-8").strip()
        return raw, int(raw)
    except (FileNotFoundError, OSError, ValueError):
        return None, None


def _broker_health(
    broker_state: dict[str, Any] | None,
    owner_running: bool,
    *,
    endpoint_probed: bool,
    endpoint_healthy: bool,
) -> tuple[bool, str]:
    if broker_state is None:
        return False, "not_discovered"
    status = str(broker_state.get("status", "unknown"))
    if not owner_running:
        return False, "owner_not_running"
    if status == "ready":
        if not endpoint_probed:
            return False, "endpoint_not_probed"
        if endpoint_healthy:
            return True, "ready"
        return False, "endpoint_unreachable"
    if status in {"failed", "stopped"}:
        return False, status
    return False, f"owner_running_{status}"


def doctor(
    pid_file: Path | None = None,
    state_file: Path | None = None,
    broker_state_file: Path = DEFAULT_BROKER_STATE_FILE,
    *,
    probe_broker_endpoint: bool = True,
    broker_probe_timeout: float = 0.5,
) -> dict[str, Any]:
    """Report shared-service and optional instrumented-shim lifecycles."""

    pid_text, proxy_pid = _read_pid(pid_file)
    adapter_state = redact_diagnostic_value(
        read_json(state_file) if state_file is not None else None
    )
    broker_private_state = read_json(broker_state_file)
    broker_state = redact_diagnostic_value(broker_private_state)
    if not isinstance(adapter_state, dict):
        adapter_state = None
    if not isinstance(broker_state, dict):
        broker_state = None

    proxy_running = pid_is_running(proxy_pid) if proxy_pid is not None else False
    broker_owner_pid: int | None = None
    if broker_state is not None:
        try:
            broker_owner_pid = int(broker_state["owner_pid"])
        except (KeyError, TypeError, ValueError):
            pass
    if broker_owner_pid is not None:
        from .broker import broker_process_is_alive

        broker_owner_running = broker_process_is_alive(broker_owner_pid)
    else:
        broker_owner_running = False
    owner_identity_matches: bool | None = None
    if broker_owner_running and isinstance(broker_private_state, dict):
        expected_start = broker_private_state.get("owner_process_start")
        if isinstance(expected_start, str) and expected_start:
            from .broker import broker_process_start_identity

            owner_identity_matches = (
                broker_process_start_identity(broker_owner_pid or 0)
                == expected_start
            )
            broker_owner_running = owner_identity_matches

    endpoint_probed = False
    endpoint_healthy = False
    if (
        probe_broker_endpoint
        and broker_owner_running
        and isinstance(broker_private_state, dict)
        and broker_private_state.get("status") == "ready"
    ):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            from .broker import BrokerState, broker_is_healthy

            try:
                private_state = BrokerState.from_dict(broker_private_state)
                endpoint_healthy = asyncio.run(
                    broker_is_healthy(
                        private_state, timeout=broker_probe_timeout
                    )
                )
            except (KeyError, OSError, OverflowError, TypeError, ValueError):
                endpoint_healthy = False
            endpoint_probed = True
    broker_healthy, broker_health = _broker_health(
        broker_state,
        broker_owner_running,
        endpoint_probed=endpoint_probed,
        endpoint_healthy=endpoint_healthy,
    )
    broker_generation = broker_state.get("generation") if broker_state else None
    broker_status = broker_state.get("status") if broker_state else None
    broker_owner_id = broker_state.get("owner_id") if broker_state else None
    service_instance_id = (
        (
            broker_state.get("service_instance_id")
            or broker_state.get("instance_id")
            or broker_owner_id
        )
        if broker_state
        else None
    )
    service_started_at = broker_state.get("started_at") if broker_state else None

    broker_public = dict(broker_state) if broker_state is not None else None
    if broker_public is not None:
        broker_public.update(
            {
                "owner_process_running": broker_owner_running,
                "owner_process_identity_matches": owner_identity_matches,
                "endpoint_probed": endpoint_probed,
                "endpoint_healthy": endpoint_healthy,
                "healthy": broker_healthy,
                "health": broker_health,
            }
        )

    return {
        "adapter_version": __version__,
        "reported_at": time.time(),
        "service_instance_id": service_instance_id,
        "service_pid": broker_owner_pid,
        "service_owner_id": broker_owner_id,
        "service_generation": broker_generation,
        "service_started_at": service_started_at,
        "service_status": broker_status,
        "service_healthy": broker_healthy,
        "instance_scope": "user",
        "transport": "stdio",
        # Existing names remain for scripts written against 0.2.x.
        "pid_file": str(pid_file) if pid_file is not None else None,
        "state_file": str(state_file) if state_file is not None else None,
        "pid": proxy_pid,
        "pid_file_raw": pid_text,
        "adapter_pid": broker_owner_pid,
        "adapter_started_at": service_started_at,
        "adapter_process_running": broker_owner_running,
        "state": adapter_state,
        "broker_state_file": str(broker_state_file),
        "broker": broker_public,
        # Explicit 0.3.x detached-process fields.
        "stdio_proxy_pid": proxy_pid,
        "stdio_proxy_process_running": proxy_running,
        "broker_owner_pid": broker_owner_pid,
        "broker_owner_id": broker_owner_id,
        "broker_owner_process_running": broker_owner_running,
        "broker_pid": broker_owner_pid,
        "broker_generation": broker_generation,
        "broker_status": broker_status,
        "broker_alive": broker_healthy,
        "broker_healthy": broker_healthy,
        "broker_health": broker_health,
        "diagnosis": diagnosis(
            proxy_running,
            adapter_state,
            broker_state,
            shim_probed=pid_file is not None,
            broker_owner_running=broker_owner_running,
            broker_healthy=broker_healthy,
        ),
    }


def diagnosis(
    proxy_running: bool,
    state: dict[str, Any] | None,
    broker_state: dict[str, Any] | None = None,
    *,
    shim_probed: bool = False,
    broker_owner_running: bool | None = None,
    broker_healthy: bool | None = None,
) -> str:
    if broker_state is not None:
        if broker_owner_running is None:
            try:
                from .broker import broker_process_is_alive

                broker_owner_running = broker_process_is_alive(
                    int(broker_state["owner_pid"])
                )
            except (KeyError, TypeError, ValueError):
                broker_owner_running = False
        if broker_healthy is None:
            broker_healthy, _ = _broker_health(
                broker_state,
                broker_owner_running,
                endpoint_probed=False,
                endpoint_healthy=False,
            )
        if broker_healthy and proxy_running:
            return "shared_service_and_instrumented_stdio_shim_running"
        if broker_healthy:
            if not shim_probed:
                return "shared_service_running; stdio_shim_not_probed"
            return (
                "shared_service_running_without_instrumented_stdio_shim; Codex "
                "can relaunch the shim without replacing the service"
            )
        if not broker_owner_running:
            return (
                "service_owner_not_running; the next stdio shim request should "
                "elect one replacement owner"
            )
        return f"service_owner_running_with_status_{broker_state.get('status', 'unknown')}"
    if proxy_running:
        return (
            "instrumented_stdio_shim_running_without_service; the next request "
            "should start a detached service owner"
        )
    if state:
        return "instrumented_stdio_shim_not_running; Codex can relaunch the shim"
    return "no_service_or_instrumented_shim_state_found"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect the shared Colab connector service"
    )
    parser.add_argument(
        "--pid-file",
        type=Path,
        help="optional PID file for one explicitly instrumented stdio shim",
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        help="optional state file for the same instrumented stdio shim",
    )
    parser.add_argument(
        "--broker-state-file",
        type=Path,
        default=DEFAULT_BROKER_STATE_FILE,
        help="private shared-service discovery state",
    )
    parser.add_argument("--probe-timeout", type=float, default=0.5)
    parser.add_argument("--no-probe", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            doctor(
                args.pid_file,
                args.state_file,
                args.broker_state_file,
                probe_broker_endpoint=not args.no_probe,
                broker_probe_timeout=args.probe_timeout,
            ),
            indent=2,
            sort_keys=True,
        )
    )


def broker_stop_main() -> None:
    """Stop the discovered detached broker after owner-identity verification."""

    parser = argparse.ArgumentParser(description="Stop the detached Colab MCP broker")
    parser.add_argument(
        "--state-file", type=Path, default=DEFAULT_BROKER_STATE_FILE
    )
    parser.add_argument(
        "--owner-id",
        help="stop only when the discovered broker has this owner identity",
    )
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args()

    # Imported lazily because broker.py imports DEFAULT_RUNTIME_DIR from this module.
    from .broker import read_broker_state, stop_broker

    before = read_broker_state(args.state_file)
    stopped = asyncio.run(
        stop_broker(
            state_file=args.state_file,
            expected_owner_id=args.owner_id,
            timeout=args.timeout,
        )
    )
    result = {
        "stopped": stopped,
        "state_file": str(args.state_file),
        "owner_id": before.owner_id if before is not None else None,
        "owner_pid": before.owner_pid if before is not None else None,
        "generation": before.generation if before is not None else None,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if not stopped:
        raise SystemExit(1)
