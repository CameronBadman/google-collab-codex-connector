---
name: colab-runner
description: Run local Python scripts or notebooks on Google Colab through Google's official CLI with bounded execution, live status, artifact retrieval, replayable logs, and automatic runtime cleanup. Use when Codex should offload a Python or ML workload to Colab CPU, GPU, or TPU; inspect active Colab CLI sessions; recover outputs from /content; or stop a leaked session.
---

# Colab Runner

Use the plugin's MCP tools instead of invoking interactive Colab commands. The
tools wrap Google's official CLI, publish lifecycle progress, bound output, and
clean up generated sessions.

## Workflow

1. Call `colab_cli_doctor` before the first job in a thread. If installation or
   authentication fails, read [official-cli.md](references/official-cli.md).
2. Inspect the local workload and determine its script/notebook path,
   dependencies, expected `/content` outputs, and maximum runtime.
3. Default to `CPU`. Select a GPU or TPU only when the user explicitly requests
   an accelerator or the workload clearly requires one and the user approves.
4. Give `colab_run_job` absolute paths for the local workload and artifact
   directory. Declare every remote artifact to retrieve before execution.
5. Pass `packages` or `requirements_file` only when the user requested those
   dependencies or the repository already declares them. Never pass both.
6. Check `state`, captured execution output, artifact paths, log path, and
   `cleanup.succeeded` in the result. Treat failed cleanup as unfinished work.
7. Summarize the runtime, accelerator, elapsed time, outputs, artifacts, and
   cleanup outcome.

## Safety

- Keep `max_runtime_seconds` finite and proportional to the task.
- Never use GPU or TPU merely because one may be available.
- Do not invoke `colab auth`, `colab drivemount`, `colab repl`, or
  `colab console` through the non-interactive MCP server.
- Do not download undeclared paths or paths outside `/content`.
- Do not retry a failed workload automatically when it may have side effects.
- Use `colab_stop_session` only for a session the user placed in scope or a
  generated session reported by a failed cleanup.
- If cancellation or cleanup status is ambiguous, inspect with
  `colab_sessions` and `colab_session_status`; never assume compute was released.

## Tool choice

- `colab_cli_doctor`: installation and authentication preflight; no allocation.
- `colab_sessions`: read-only inventory of assigned sessions.
- `colab_session_status`: read-only inspection of one session.
- `colab_run_job`: fresh bounded job with artifact/log collection and cleanup.
- `colab_stop_session`: explicit emergency release.
