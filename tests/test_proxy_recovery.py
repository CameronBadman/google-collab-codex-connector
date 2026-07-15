from __future__ import annotations

import os
import sys
from pathlib import Path

from fastmcp import Client

from colab_codex_adapter.broker import (
    BrokerLaunchConfig,
    BrokerLauncher,
    read_broker_state,
    stop_broker,
)
from colab_codex_adapter.server import _proxy_server


def _write_failure_runtime(path: Path) -> None:
    path.write_text(
        """
import os
import signal

from fastmcp import Context, FastMCP
from fastmcp.server.auth import StaticTokenVerifier
from fastmcp.server.middleware import Middleware


class KillOneList(Middleware):
    async def on_list_tools(self, context, call_next):
        marker = os.environ.get("COLAB_TEST_KILL_LIST_MARKER")
        if marker and os.path.exists(marker):
            os.unlink(marker)
            os.kill(os.getpid(), signal.SIGKILL)
        return await call_next(context)


async def serve(state):
    verifier = StaticTokenVerifier(
        {state.token: {"client_id": "proxy-recovery-test", "scopes": []}}
    )
    backend = FastMCP(
        "proxy-recovery-test",
        auth=verifier,
        middleware=[KillOneList()],
        mask_error_details=True,
    )

    @backend.tool()
    async def broker_identity():
        return {
            "owner_id": state.owner_id,
            "owner_pid": os.getpid(),
            "generation": state.generation,
        }

    @backend.tool()
    async def die_during_call():
        os.kill(os.getpid(), signal.SIGKILL)

    @backend.tool()
    async def colab_status():
        marker = os.environ.get("COLAB_TEST_KILL_READ_MARKER")
        if marker and os.path.exists(marker):
            os.unlink(marker)
            os.kill(os.getpid(), signal.SIGKILL)
        return {
            "owner_id": state.owner_id,
            "owner_pid": os.getpid(),
            "generation": state.generation,
        }

    @backend.tool()
    async def progress_probe(ctx: Context):
        await ctx.report_progress(1.0, message="Backend progress reached proxy")
        return {"ok": True}

    await backend.run_http_async(
        show_banner=False,
        host="127.0.0.1",
        port=int(state.endpoint.split(":")[2].split("/")[0]),
        log_level="critical",
        uvicorn_config={"access_log": False, "log_config": None},
        json_response=False,
        stateless_http=True,
    )
""".lstrip(),
        encoding="utf-8",
    )


async def test_proxy_recovers_inflight_discovery_without_replaying_mutation(
    tmp_path: Path,
    unused_tcp_port: int,
    monkeypatch,
) -> None:
    runtime_module = tmp_path / "proxy_failure_runtime.py"
    _write_failure_runtime(runtime_module)
    project = Path(__file__).parents[1]
    pythonpath = os.pathsep.join(
        part
        for part in (
            str(tmp_path),
            str(project),
            os.environ.get("PYTHONPATH"),
        )
        if part
    )
    marker = tmp_path / "kill-next-list"
    read_marker = tmp_path / "kill-next-read"
    monkeypatch.setenv("PYTHONPATH", pythonpath)
    monkeypatch.setenv("COLAB_TEST_KILL_LIST_MARKER", str(marker))
    monkeypatch.setenv("COLAB_TEST_KILL_READ_MARKER", str(read_marker))

    config = BrokerLaunchConfig(
        port=unused_tcp_port,
        state_file=tmp_path / "broker.json",
        lock_file=tmp_path / "broker.lock",
        launch_lock_file=tmp_path / "broker-launch.lock",
        factory="proxy_failure_runtime:serve",
        startup_timeout=15.0,
        health_timeout=1.0,
        python_executable=sys.executable,
    )

    class ArmedLauncher(BrokerLauncher):
        arm_after_health_check = False

        async def ensure_running(self):
            state = await super().ensure_running()
            if self.arm_after_health_check:
                self.arm_after_health_check = False
                marker.touch()
            return state

    launcher = ArmedLauncher(config)
    proxy = _proxy_server(launcher)

    try:
        async with Client(proxy) as client:
            await client.list_tools()
            initial = read_broker_state(config.state_file)
            assert initial is not None

            launcher.arm_after_health_check = True
            names = {tool.name for tool in await client.list_tools()}
            after_discovery = read_broker_state(config.state_file)
            assert after_discovery is not None
            assert after_discovery.generation == initial.generation + 1
            assert {
                "broker_identity",
                "colab_status",
                "die_during_call",
                "progress_probe",
            }.issubset(names)

            progress = []

            async def progress_handler(current, total, message):
                progress.append((current, total, message))

            progress_result = await client.call_tool(
                "progress_probe", {}, progress_handler=progress_handler
            )
            assert progress_result.data == {"ok": True}
            assert progress == [
                (1.0, None, "Backend progress reached proxy")
            ]

            ambiguous = await client.call_tool(
                "die_during_call", {}, raise_on_error=False
            )
            assert ambiguous.is_error is True
            assert "outcome is unknown" in ambiguous.content[0].text

            identity = await client.call_tool("broker_identity", {})
            recovered = read_broker_state(config.state_file)
            assert recovered is not None
            assert recovered.generation == after_discovery.generation + 1
            assert identity.data["owner_id"] == recovered.owner_id

            read_marker.touch()
            status = await client.call_tool("colab_status", {})
            after_read_retry = read_broker_state(config.state_file)
            assert after_read_retry is not None
            assert after_read_retry.generation == recovered.generation + 1
            assert status.data["owner_id"] == after_read_retry.owner_id
    finally:
        state = read_broker_state(config.state_file)
        if state is not None and state.status not in {"failed", "stopped"}:
            await stop_broker(
                state_file=config.state_file,
                expected_owner_id=state.owner_id,
                timeout=10.0,
            )
