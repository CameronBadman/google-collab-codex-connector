from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

AGENT_FILENAME = "colab-worker.toml"
REASONING_EFFORTS = ("low", "medium", "high", "xhigh", "max", "ultra")
REQUIRED_TOOL_TIMEOUT_SECONDS = 1200.0

DEVELOPER_INSTRUCTIONS = """You are the lower-cost execution worker for Colab and repository experiments.

Work only on the assignment delegated by the parent agent. You may edit the repository and the connected Colab notebook, but you own those mutations exclusively while the assignment is active. Do not spawn other agents.

Start by checking colab_status and validating the notebook/runtime. If a human must open a connection URL, report that blocker to the parent instead of inventing a workaround.

Use colab_run_python_async for long execution. You own the polling timer: choose the next useful observation interval and call colab_wait_job instead of ending your turn merely to sleep. Each wait must be between 1 and 900 seconds. Normally use 10-60 seconds for setup or short commands, 60-300 seconds for evaluation or package operations, and 300-900 seconds for training phases. When a wait times out, inspect the returned job state and outputs, then choose a new interval from the current evidence. Keep experiment inputs, outputs, metrics, and relevant paths explicit.

Checkpoints are gated. Report after environment setup and after every completed experiment or evaluation, then end your turn and wait for a follow-up from the parent. Also report immediately on an unrecoverable failure or a decision that changes scope, cost, or the experimental hypothesis. Do not continue past a checkpoint without parent direction.

Every report must contain these fields:
- kind: checkpoint, blocked, complete, or failed
- summary
- changes
- evidence
- metrics
- interpretation
- recommendation
- decision_needed
- proposed_next_step
- poll_interval_seconds
- poll_reason
- expected_signal
- hard_deadline
"""


def project_root(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"Project directory does not exist: {path}")
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ValueError(f"Project is not inside a Git worktree: {path}")
    return Path(result.stdout.strip()).resolve()


def render_agent_profile(model: str, reasoning_effort: str = "medium") -> str:
    if not model.strip():
        raise ValueError("model must not be empty")
    if reasoning_effort not in REASONING_EFFORTS:
        raise ValueError(f"Unsupported reasoning effort: {reasoning_effort}")
    profile = (
        'name = "colab_worker"\n'
        'description = "Execution worker for Colab/GPU experiments with gated parent checkpoints."\n'
        f"model = {json.dumps(model.strip())}\n"
        f"model_reasoning_effort = {json.dumps(reasoning_effort)}\n"
        'sandbox_mode = "workspace-write"\n'
        f"developer_instructions = {json.dumps(DEVELOPER_INSTRUCTIONS)}\n"
    )
    tomllib.loads(profile)
    return profile


def _read_config(path: Path) -> dict[str, object]:
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, tomllib.TOMLDecodeError):
        return {}
    return data


def _colab_server(path: Path) -> dict[str, object] | None:
    data = _read_config(path)
    servers = data.get("mcp_servers")
    if not isinstance(servers, dict):
        return None
    server = servers.get("colab")
    return server if isinstance(server, dict) else None


def colab_mcp_settings(
    project: Path,
    codex_home: Path | None = None,
) -> tuple[bool, float | None]:
    codex_home = codex_home or Path.home() / ".codex"
    global_server = _colab_server(codex_home / "config.toml")
    project_server = _colab_server(project / ".codex" / "config.toml")
    configured = global_server is not None or project_server is not None

    timeout: object | None = None
    if project_server is not None:
        timeout = project_server.get("tool_timeout_sec")
    if timeout is None and global_server is not None:
        timeout = global_server.get("tool_timeout_sec")
    if not isinstance(timeout, bool) and isinstance(timeout, int | float):
        return configured, float(timeout)
    return configured, None


def colab_mcp_is_configured(project: Path) -> bool:
    configured, _ = colab_mcp_settings(project)
    return configured


def install_agent_profile(
    *,
    project: Path,
    model: str,
    reasoning_effort: str = "medium",
    force: bool = False,
) -> Path:
    root = project_root(project)
    codex_dir = root / ".codex"
    if codex_dir.exists() and not codex_dir.is_dir():
        raise ValueError(f"Cannot create Codex configuration: {codex_dir} is a file")
    agents_dir = codex_dir / "agents"
    agents_dir.mkdir(parents=True, exist_ok=True)
    destination = agents_dir / AGENT_FILENAME
    if destination.exists() and not force:
        raise FileExistsError(
            f"Agent profile already exists: {destination}; pass --force to replace it"
        )

    content = render_agent_profile(model, reasoning_effort)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=agents_dir,
            prefix=f".{AGENT_FILENAME}.",
            delete=False,
        ) as handle:
            handle.write(content)
            temporary_path = Path(handle.name)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return destination


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install a project-scoped Codex Colab worker profile"
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=Path.cwd(),
        help="project directory (defaults to the current Git worktree)",
    )
    parser.add_argument("--model", required=True, help="Codex model for the worker")
    parser.add_argument(
        "--reasoning-effort",
        choices=REASONING_EFFORTS,
        default="medium",
    )
    parser.add_argument(
        "--force", action="store_true", help="replace an existing worker profile"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        destination = install_agent_profile(
            project=args.project,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            force=args.force,
        )
    except (FileExistsError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    root = destination.parents[2]
    print(destination)
    configured, tool_timeout = colab_mcp_settings(root)
    if not configured:
        print(
            "warning: no mcp_servers.colab configuration was found; configure the "
            "connector before starting the worker",
            file=sys.stderr,
        )
    elif tool_timeout is None or tool_timeout < REQUIRED_TOOL_TIMEOUT_SECONDS:
        current = "not set" if tool_timeout is None else f"{tool_timeout:g}"
        print(
            "warning: mcp_servers.colab.tool_timeout_sec is "
            f"{current}; set tool_timeout_sec = "
            f"{REQUIRED_TOOL_TIMEOUT_SECONDS:g} and restart Codex to allow "
            "15-minute worker waits",
            file=sys.stderr,
        )
    print("Start a new Codex session before using the colab_worker agent.")
