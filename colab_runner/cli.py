from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

from .process import ProcessResult, ProcessRunner


class ColabCli:
    """Typed subprocess boundary around Google's official ``colab`` command."""

    def __init__(
        self,
        process_runner: ProcessRunner | None = None,
        *,
        executable: str | None = None,
        auth: str | None = None,
        config_path: Path | None = None,
    ) -> None:
        selected_auth = (
            auth or os.environ.get("COLAB_RUNNER_AUTH", "adc")
        ).strip().lower()
        if selected_auth not in {"adc", "oauth2"}:
            raise ValueError("auth must be 'adc' or 'oauth2'")
        self.process_runner = process_runner or ProcessRunner()
        self.executable = executable or os.environ.get("COLAB_RUNNER_CLI", "colab")
        self.auth = selected_auth
        configured_path = os.environ.get("COLAB_RUNNER_CONFIG")
        self.config_path = config_path or (
            Path(configured_path) if configured_path else None
        )

    async def run(
        self,
        arguments: Sequence[str],
        *,
        timeout_seconds: float,
        cwd: Path | None = None,
    ) -> ProcessResult:
        argv = [self.executable, f"--auth={self.auth}"]
        if self.config_path is not None:
            argv.extend(["--config", str(self.config_path)])
        argv.extend(arguments)
        return await self.process_runner.run(
            argv,
            timeout_seconds=timeout_seconds,
            cwd=cwd,
        )
