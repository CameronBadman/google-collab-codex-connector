from __future__ import annotations

import subprocess
import tomllib
from pathlib import Path

import pytest

from colab_codex_adapter import agent_setup
from colab_codex_adapter.agent_setup import (
    colab_mcp_settings,
    install_agent_profile,
    render_agent_profile,
)


def init_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(path)], check=True)


def test_install_agent_profile_creates_project_scoped_worker(tmp_path: Path) -> None:
    init_repo(tmp_path)

    destination = install_agent_profile(
        project=tmp_path,
        model="configured-worker-model",
        reasoning_effort="high",
    )
    profile = tomllib.loads(destination.read_text(encoding="utf-8"))

    assert destination == tmp_path / ".codex" / "agents" / "colab-worker.toml"
    assert profile["name"] == "colab_worker"
    assert profile["model"] == "configured-worker-model"
    assert profile["model_reasoning_effort"] == "high"
    assert profile["sandbox_mode"] == "workspace-write"
    assert "Checkpoints are gated" in profile["developer_instructions"]
    assert "You own the polling timer" in profile["developer_instructions"]
    assert "between 1 and 900 seconds" in profile["developer_instructions"]
    assert "poll_interval_seconds" in profile["developer_instructions"]


def test_install_agent_profile_requires_force_to_replace(tmp_path: Path) -> None:
    init_repo(tmp_path)
    destination = install_agent_profile(project=tmp_path, model="first")

    with pytest.raises(FileExistsError):
        install_agent_profile(project=tmp_path, model="second")

    replaced = install_agent_profile(project=tmp_path, model="second", force=True)
    assert replaced == destination
    assert tomllib.loads(replaced.read_text(encoding="utf-8"))["model"] == "second"


def test_render_agent_profile_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        render_agent_profile("  ")
    with pytest.raises(ValueError, match="Unsupported reasoning effort"):
        render_agent_profile("model", "extreme")


def test_install_agent_profile_requires_git_worktree(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not inside a Git worktree"):
        install_agent_profile(project=tmp_path, model="model")


def test_colab_mcp_settings_resolve_project_timeout_over_global(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    codex_home = tmp_path / "home" / ".codex"
    (project / ".codex").mkdir(parents=True)
    codex_home.mkdir(parents=True)
    (codex_home / "config.toml").write_text(
        '[mcp_servers.colab]\ncommand = "uv"\ntool_timeout_sec = 300\n',
        encoding="utf-8",
    )

    configured, timeout = colab_mcp_settings(project, codex_home)
    assert configured is True
    assert timeout == 300

    (project / ".codex" / "config.toml").write_text(
        "[mcp_servers.colab]\ntool_timeout_sec = 1200\n",
        encoding="utf-8",
    )
    configured, timeout = colab_mcp_settings(project, codex_home)
    assert configured is True
    assert timeout == 1200


def test_colab_mcp_settings_reports_absent_and_ignores_boolean_timeout(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    codex_home = tmp_path / "home" / ".codex"

    assert colab_mcp_settings(project, codex_home) == (False, None)

    (project / ".codex").mkdir(parents=True)
    (project / ".codex" / "config.toml").write_text(
        "[mcp_servers.colab]\ntool_timeout_sec = true\n",
        encoding="utf-8",
    )
    assert colab_mcp_settings(project, codex_home) == (True, None)


def test_agent_init_warns_when_tool_timeout_is_too_short(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    init_repo(tmp_path)
    monkeypatch.setattr(agent_setup, "colab_mcp_settings", lambda project: (True, 300.0))

    agent_setup.main(["--project", str(tmp_path), "--model", "worker-model"])

    assert "set tool_timeout_sec = 1200" in capsys.readouterr().err


def test_agent_init_accepts_sufficient_tool_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    init_repo(tmp_path)
    monkeypatch.setattr(agent_setup, "colab_mcp_settings", lambda project: (True, 1200.0))

    agent_setup.main(["--project", str(tmp_path), "--model", "worker-model"])

    assert "warning:" not in capsys.readouterr().err
