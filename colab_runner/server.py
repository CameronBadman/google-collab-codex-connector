import asyncio
from collections.abc import Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastmcp import Context, FastMCP

from .cli import ColabCli
from .orchestrator import ColabJobOrchestrator, validate_session_name
from .process import ProcessExecutionError, ProcessExecutionTimeout, ProcessResult
from .sessions import ColabSessionManager


Accelerator = Literal[
    "CPU", "T4", "L4", "G4", "H100", "A100", "TPU_V5E1", "TPU_V6E1"
]
REQUIRED_COLAB_SCOPES = (
    "openid",
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/colaboratory",
)


def create_mcp(
    cli: ColabCli | Any | None = None,
    session_manager: ColabSessionManager | None = None,
) -> FastMCP:
    official_cli = cli or ColabCli()
    jobs = ColabJobOrchestrator(official_cli)
    managed_sessions = session_manager or ColabSessionManager(official_cli)

    @asynccontextmanager
    async def lifespan(_: FastMCP):
        try:
            yield
        finally:
            await managed_sessions.close()

    mcp = FastMCP(
        name="ColabRunner",
        instructions=(
            "Run safe, observable workloads through Google's official Colab CLI. "
            "Use colab_cli_doctor before the first allocation. Use colab_run_job "
            "for isolated work, or a leased session for repeated stateful "
            "execution. Stop leased sessions promptly; idle leases and server "
            "shutdown trigger automatic cleanup."
        ),
        lifespan=lifespan,
        mask_error_details=True,
    )

    def reporter(ctx: Context) -> Callable[[str], Awaitable[None]]:
        sequence = 0

        async def publish(message: str) -> None:
            nonlocal sequence
            sequence += 1
            await ctx.report_progress(float(sequence), total=None, message=message)

        return publish

    @mcp.tool()
    async def colab_cli_doctor(ctx: Context) -> dict[str, Any]:
        """Check the official CLI installation and non-mutating session access."""
        publish = reporter(ctx)
        await publish("Checking official Colab CLI")
        version = await _safe_command(official_cli, ["version"], timeout_seconds=30)
        if not version["ok"]:
            return {
                "ok": False,
                "version": version,
                "credentials": None,
                "sessions": None,
            }
        await publish("Checking Colab credential scopes")
        credentials = await _credential_scope_status(official_cli)
        if not credentials["ok"]:
            return {
                "ok": False,
                "version": version,
                "credentials": credentials,
                "sessions": None,
            }
        await publish("Checking Colab session access")
        sessions = await _safe_command(
            official_cli, ["sessions"], timeout_seconds=30
        )
        return {
            "ok": sessions["ok"],
            "version": version,
            "credentials": credentials,
            "sessions": sessions,
        }

    @mcp.tool()
    async def colab_sessions(ctx: Context) -> dict[str, Any]:
        """List active official Colab CLI sessions without changing them."""
        await reporter(ctx)("Listing Colab sessions")
        result = await _safe_command(
            official_cli, ["sessions"], timeout_seconds=30
        )
        result["managed_leases"] = managed_sessions.leases()
        return result

    @mcp.tool()
    async def colab_session_status(
        session_name: str, ctx: Context
    ) -> dict[str, Any]:
        """Read status for one named official Colab CLI session."""
        validate_session_name(session_name)
        await reporter(ctx)("Inspecting Colab session")
        result = await _safe_command(
            official_cli,
            ["status", "-s", session_name],
            timeout_seconds=30,
        )
        result["found"] = result["ok"] and "not found" not in result.get(
            "stdout", ""
        ).lower()
        result["managed_lease"] = managed_sessions.lease_status(session_name)
        return result

    @mcp.tool()
    async def colab_start_session(
        ctx: Context,
        accelerator: Accelerator = "CPU",
        packages: Sequence[str] = (),
        requirements_file: str | None = None,
        idle_timeout_seconds: float = 600,
        setup_timeout_seconds: float = 900,
        session_name_prefix: str = "codex-live",
    ) -> dict[str, Any]:
        """Allocate a reusable Colab kernel protected by an idle lease."""
        return await managed_sessions.start_session(
            accelerator=accelerator,
            packages=packages,
            requirements_file=requirements_file,
            idle_timeout_seconds=idle_timeout_seconds,
            setup_timeout_seconds=setup_timeout_seconds,
            session_name_prefix=session_name_prefix,
            reporter=reporter(ctx),
        )

    @mcp.tool()
    async def colab_execute(
        session_name: str,
        script_path: str,
        ctx: Context,
        cell_index: int | None = None,
        cell_id: str | None = None,
        timeout_seconds: float = 1800,
    ) -> dict[str, Any]:
        """Execute a file or selected notebook cell on a reusable session."""
        return await managed_sessions.execute(
            session_name=session_name,
            script_path=script_path,
            cell_index=cell_index,
            cell_id=cell_id,
            timeout_seconds=timeout_seconds,
            reporter=reporter(ctx),
        )

    @mcp.tool()
    async def colab_renew_session(
        session_name: str,
        ctx: Context,
        idle_timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Renew a reusable session lease, optionally changing its idle limit."""
        await reporter(ctx)("Renewing reusable Colab session lease")
        return await managed_sessions.renew_session(
            session_name=session_name,
            idle_timeout_seconds=idle_timeout_seconds,
        )

    @mcp.tool()
    async def colab_download_artifact(
        session_name: str,
        remote_path: str,
        artifact_dir: str,
        ctx: Context,
        timeout_seconds: float = 300,
    ) -> dict[str, Any]:
        """Download one declared /content file from a reusable session."""
        return await managed_sessions.download_artifact(
            session_name=session_name,
            remote_path=remote_path,
            artifact_dir=artifact_dir,
            timeout_seconds=timeout_seconds,
            reporter=reporter(ctx),
        )

    @mcp.tool()
    async def colab_export_log(
        session_name: str,
        output_path: str,
        ctx: Context,
        timeout_seconds: float = 300,
    ) -> dict[str, Any]:
        """Export a reusable session's execution history as a notebook."""
        return await managed_sessions.export_log(
            session_name=session_name,
            output_path=output_path,
            timeout_seconds=timeout_seconds,
            reporter=reporter(ctx),
        )

    @mcp.tool()
    async def colab_run_job(
        script_path: str,
        artifact_dir: str,
        ctx: Context,
        accelerator: Accelerator = "CPU",
        packages: Sequence[str] = (),
        requirements_file: str | None = None,
        remote_artifacts: Sequence[str] = (),
        export_log: bool = True,
        max_runtime_seconds: float = 1800,
        session_name_prefix: str = "codex",
    ) -> dict[str, Any]:
        """Run a local Python script or notebook on a fresh, auto-cleaned session."""
        return await jobs.run_job(
            script_path=script_path,
            accelerator=accelerator,
            packages=packages,
            requirements_file=requirements_file,
            remote_artifacts=remote_artifacts,
            artifact_dir=artifact_dir,
            export_log=export_log,
            max_runtime_seconds=max_runtime_seconds,
            session_name_prefix=session_name_prefix,
            reporter=reporter(ctx),
        )

    @mcp.tool()
    async def colab_stop_session(
        session_name: str, ctx: Context
    ) -> dict[str, Any]:
        """Explicitly stop and release a named Colab session."""
        validate_session_name(session_name)
        await reporter(ctx)("Stopping Colab session")
        if managed_sessions.is_managed(session_name):
            return await managed_sessions.stop_session(session_name)
        result = await _safe_command(
            official_cli,
            ["stop", "-s", session_name],
            timeout_seconds=60,
        )
        result["already_absent"] = result["ok"] and "not found" in result.get(
            "stdout", ""
        ).lower()
        return result

    return mcp


async def _credential_scope_status(cli: ColabCli | Any) -> dict[str, Any]:
    result = await _safe_command(cli, ["whoami"], timeout_seconds=30)
    auth_mode = getattr(cli, "auth", None)
    if not result["ok"]:
        return {
            "ok": False,
            "auth_mode": auth_mode,
            "required_scopes": list(REQUIRED_COLAB_SCOPES),
            "missing_scopes": list(REQUIRED_COLAB_SCOPES),
            "error": result["error"],
        }

    present = _parse_whoami_scopes(result.get("stdout", ""))
    missing = [scope for scope in REQUIRED_COLAB_SCOPES if scope not in present]
    return {
        "ok": not missing,
        "auth_mode": auth_mode,
        "required_scopes": list(REQUIRED_COLAB_SCOPES),
        "missing_scopes": missing,
        "error": (
            None
            if not missing
            else "Colab credentials are missing required scopes"
        ),
    }


def _parse_whoami_scopes(output: str) -> set[str]:
    scopes: set[str] = set()
    in_scopes = False
    for line in output.splitlines():
        if line.strip() == "Scopes:":
            in_scopes = True
            continue
        if not in_scopes:
            continue
        stripped = line.strip()
        if stripped.startswith("- "):
            scopes.add(stripped.removeprefix("- ").strip())
        elif stripped:
            break
    return scopes


async def _safe_command(
    cli: ColabCli | Any,
    arguments: list[str],
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    try:
        result = await cli.run(arguments, timeout_seconds=timeout_seconds)
    except ProcessExecutionError as exc:
        return {"ok": False, "error": str(exc), **_result_data(exc.result)}
    except ProcessExecutionTimeout as exc:
        return {"ok": False, "error": str(exc), **_result_data(exc.result)}
    except FileNotFoundError as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "error": None, **_result_data(result)}


def _result_data(result: ProcessResult) -> dict[str, Any]:
    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "stdout_truncated": result.stdout_truncated,
        "stderr_truncated": result.stderr_truncated,
        "elapsed_seconds": result.elapsed_seconds,
    }


async def main_async() -> None:
    await create_mcp().run_async(show_banner=False)


def main() -> None:
    asyncio.run(main_async())
