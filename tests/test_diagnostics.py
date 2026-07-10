from __future__ import annotations

import json
import os
from pathlib import Path

from colab_codex_adapter import broker as broker_module
from colab_codex_adapter.diagnostics import adapter_info, doctor, write_state


def test_doctor_does_not_assume_shared_stdio_shim_files(tmp_path: Path) -> None:
    result = doctor(
        broker_state_file=tmp_path / "missing-broker.json",
        probe_broker_endpoint=False,
    )

    assert result["pid_file"] is None
    assert result["state_file"] is None
    assert result["stdio_proxy_pid"] is None
    assert result["stdio_proxy_process_running"] is False
    assert result["diagnosis"] == "no_service_or_instrumented_shim_state_found"


def test_doctor_redacts_tokens_and_reports_owner_fields(
    tmp_path: Path, monkeypatch
) -> None:
    async def healthy(*args, **kwargs) -> bool:
        return True

    monkeypatch.setattr(broker_module, "broker_is_healthy", healthy)
    pid_file = tmp_path / "adapter.pid"
    state_file = tmp_path / "adapter.json"
    broker_file = tmp_path / "broker.json"
    pid_file.write_text(f"{os.getpid()}\n", encoding="utf-8")
    state_file.write_text(
        json.dumps(
            {
                "state": "running",
                "browser": {
                    "token_prefix": "should-not-leak",
                    "url": (
                        "https://colab.research.google.com/notebooks/empty.ipynb"
                        "?ignored=yes#mcpProxyToken=browser-secret"
                    ),
                    "last_close_code": 1009,
                },
                "corpus": "private notebook text",
            }
        ),
        encoding="utf-8",
    )
    broker_file.write_text(
        json.dumps(
            {
                "endpoint": "http://127.0.0.1:8765/mcp",
                "owner_pid": os.getpid(),
                "owner_id": "owner-a",
                "service_instance_id": "service-a",
                "generation": 4,
                "status": "ready",
                "started_at": 1,
                "token": "broker-secret",
            }
        ),
        encoding="utf-8",
    )

    result = doctor(pid_file, state_file, broker_file)

    assert result["stdio_proxy_pid"] == os.getpid()
    assert result["stdio_proxy_process_running"] is True
    assert result["broker_owner_pid"] == os.getpid()
    assert result["broker_owner_id"] == "owner-a"
    assert result["broker_owner_process_running"] is True
    assert result["broker_generation"] == 4
    assert result["broker_status"] == "ready"
    assert result["broker_healthy"] is True
    assert result["service_instance_id"] == "service-a"
    assert result["service_pid"] == os.getpid()
    assert result["service_owner_id"] == "owner-a"
    assert result["service_generation"] == 4
    assert result["service_started_at"] == 1
    assert result["service_status"] == "ready"
    assert result["service_healthy"] is True
    assert result["instance_scope"] == "user"
    assert result["transport"] == "stdio"
    assert result["adapter_pid"] == result["service_pid"]
    assert result["broker_pid"] == result["service_pid"]
    assert result["broker"]["owner_id"] == "owner-a"
    assert result["broker"]["health"] == "ready"
    assert result["state"]["browser"]["last_close_code"] == 1009
    assert result["state"]["browser"]["url"].endswith("empty.ipynb")
    serialized = json.dumps(result)
    assert "broker-secret" not in serialized
    assert "browser-secret" not in serialized
    assert "should-not-leak" not in serialized
    assert "private notebook text" not in serialized
    assert (
        result["diagnosis"]
        == "shared_service_and_instrumented_stdio_shim_running"
    )


def test_doctor_distinguishes_live_broker_from_dead_stdio_proxy(
    tmp_path: Path, monkeypatch
) -> None:
    async def healthy(*args, **kwargs) -> bool:
        return True

    monkeypatch.setattr(broker_module, "broker_is_healthy", healthy)
    broker_file = tmp_path / "broker.json"
    broker_file.write_text(
        json.dumps(
            {
                "endpoint": "http://127.0.0.1:8765/mcp",
                "owner_pid": os.getpid(),
                "owner_id": "owner-b",
                "generation": 2,
                "status": "ready",
                "started_at": 1,
                "token": "secret",
            }
        ),
        encoding="utf-8",
    )

    result = doctor(
        tmp_path / "missing-adapter.pid",
        tmp_path / "missing-adapter.json",
        broker_file,
    )

    assert result["stdio_proxy_process_running"] is False
    assert result["broker_owner_process_running"] is True
    assert result["broker_healthy"] is True
    assert result["service_instance_id"] == "owner-b"
    assert "without_instrumented_stdio_shim" in result["diagnosis"]


def test_doctor_does_not_report_an_unprobed_shim_as_missing(
    tmp_path: Path, monkeypatch
) -> None:
    async def healthy(*args, **kwargs) -> bool:
        return True

    monkeypatch.setattr(broker_module, "broker_is_healthy", healthy)
    broker_file = tmp_path / "broker.json"
    broker_file.write_text(
        json.dumps(
            {
                "endpoint": "http://127.0.0.1:8765/mcp",
                "owner_pid": os.getpid(),
                "owner_id": "owner-unprobed",
                "service_instance_id": "service-unprobed",
                "generation": 1,
                "status": "ready",
                "started_at": 1,
                "token": "secret",
            }
        ),
        encoding="utf-8",
    )

    result = doctor(broker_state_file=broker_file)

    assert result["service_healthy"] is True
    assert result["stdio_proxy_pid"] is None
    assert result["diagnosis"] == "shared_service_running; stdio_shim_not_probed"


def test_doctor_marks_dead_broker_owner_for_replacement(tmp_path: Path) -> None:
    broker_file = tmp_path / "broker.json"
    broker_file.write_text(
        json.dumps(
            {
                "endpoint": "http://127.0.0.1:8765/mcp",
                "owner_pid": 999_999_999,
                "owner_id": "dead-owner",
                "generation": 9,
                "status": "ready",
                "started_at": 1,
                "token": "secret",
            }
        ),
        encoding="utf-8",
    )

    result = doctor(
        tmp_path / "missing-adapter.pid",
        tmp_path / "missing-adapter.json",
        broker_file,
    )

    assert result["broker_owner_process_running"] is False
    assert result["broker_healthy"] is False
    assert result["broker_health"] == "owner_not_running"
    assert "replacement owner" in result["diagnosis"]


def test_doctor_rejects_reused_broker_pid_identity(tmp_path: Path) -> None:
    broker_file = tmp_path / "broker.json"
    broker_file.write_text(
        json.dumps(
            {
                "endpoint": "http://127.0.0.1:8765/mcp",
                "owner_pid": os.getpid(),
                "owner_id": "stale-owner",
                "owner_process_start": "different-process",
                "generation": 3,
                "status": "ready",
                "started_at": 1,
                "token": "secret",
            }
        ),
        encoding="utf-8",
    )

    result = doctor(
        tmp_path / "missing.pid",
        tmp_path / "missing.json",
        broker_file,
    )

    assert result["broker_owner_process_running"] is False
    assert result["broker_healthy"] is False
    assert result["broker"]["owner_process_identity_matches"] is False
    assert result["broker"]["endpoint_probed"] is False


def test_adapter_info_redacts_connection_credentials() -> None:
    info = adapter_info(
        extra={
            "broker_token": "broker-secret",
            "connection": {
                "url": "https://colab.research.google.com/#mcpProxyToken=browser-secret",
                "token_prefix": "browser",
                "connection_id": "connection-a",
            },
        }
    )

    serialized = json.dumps(info)
    assert "broker-secret" not in serialized
    assert "browser-secret" not in serialized
    assert "token_prefix" not in serialized
    assert info["connection"]["url"] == "https://colab.research.google.com/"
    assert info["connection"]["connection_id"] == "connection-a"


def test_adapter_info_reports_service_as_canonical_process() -> None:
    info = adapter_info(
        extra={
            "service_instance_id": "service-a",
            "service_pid": os.getpid(),
            "service_owner_id": "owner-a",
            "service_generation": 7,
            "service_started_at": 123.0,
            "service_status": "ready",
            "service_healthy": True,
        }
    )

    assert info["service_instance_id"] == "service-a"
    assert info["instance_scope"] == "user"
    assert info["transport"] == "stdio"
    assert info["adapter_pid"] == info["service_pid"] == os.getpid()
    assert info["adapter_started_at"] == info["service_started_at"] == 123.0
    assert info["broker_pid"] == info["service_pid"]
    assert info["broker_owner_id"] == info["service_owner_id"] == "owner-a"
    assert info["broker_generation"] == info["service_generation"] == 7
    assert info["broker_status"] == info["service_status"] == "ready"
    assert info["broker_healthy"] is info["service_healthy"] is True
    assert info["pid_file"] is None
    assert info["state_file"] is None


def test_write_state_is_private_and_payload_free(tmp_path: Path) -> None:
    path = tmp_path / "runtime" / "adapter-state.json"

    write_state(
        path,
        {
            "state": "running",
            "broker_token": "secret",
            "outputs": ["notebook payload"],
            "last_browser_close_code": 1009,
        },
    )

    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted == {"last_browser_close_code": 1009, "state": "running"}
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700
