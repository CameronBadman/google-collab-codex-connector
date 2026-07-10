from __future__ import annotations

import json
import os
from pathlib import Path

from colab_codex_adapter.diagnostics import doctor


def test_doctor_redacts_broker_token(tmp_path: Path) -> None:
    pid_file = tmp_path / "adapter.pid"
    state_file = tmp_path / "adapter.json"
    broker_file = tmp_path / "broker.json"
    pid_file.write_text(f"{os.getpid()}\n", encoding="utf-8")
    state_file.write_text('{"state": "running"}\n', encoding="utf-8")
    broker_file.write_text(
        json.dumps(
            {
                "endpoint": "http://127.0.0.1:8765/mcp",
                "owner_pid": os.getpid(),
                "started_at": 1,
                "token": "secret",
            }
        ),
        encoding="utf-8",
    )

    result = doctor(pid_file, state_file, broker_file)

    assert result["broker"]["endpoint"].endswith("/mcp")
    assert "token" not in result["broker"]
    assert result["diagnosis"] == "adapter_process_running"


def test_doctor_requests_restart_for_legacy_running_adapter(tmp_path: Path) -> None:
    pid_file = tmp_path / "adapter.pid"
    state_file = tmp_path / "adapter.json"
    pid_file.write_text(f"{os.getpid()}\n", encoding="utf-8")
    state_file.write_text('{"state": "running"}\n', encoding="utf-8")

    result = doctor(pid_file, state_file, tmp_path / "missing-broker.json")

    assert "restart Codex" in result["diagnosis"]
