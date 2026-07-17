---
name: colab-runner
description: Run isolated workloads or reusable leased Python kernels on Google Colab through Google's official CLI with guarded local notebook-cell edits, background selected-cell execution, bounded status, artifact retrieval, replayable logs, and automatic runtime cleanup. Use when Codex should offload Python or ML work to Colab CPU, GPU, or TPU; preserve state across several execution steps; inspect or edit local notebook cells; retrieve outputs from /content; or release compute.
---

# Colab Runner

Use the plugin's MCP tools instead of invoking interactive Colab commands. The
tools wrap Google's official CLI, publish lifecycle progress, bound output, and
clean up generated or leased sessions.

## Workflow

1. Call `colab_cli_doctor` before the first job in a thread. If installation or
   authentication fails, read [official-cli.md](references/official-cli.md).
2. Inspect the local workload and determine its script/notebook paths,
   dependencies, expected `/content` outputs, and maximum runtime.
3. Choose `colab_run_job` for one isolated workload. Choose a leased session
   only when multiple steps benefit from preserving imports, variables, loaded
   models, files, or GPU memory.
4. Default to `CPU`. Select a GPU or TPU only when the user explicitly requests
   an accelerator or the workload clearly requires one and the user approves.
5. Give `colab_run_job` absolute paths for the local workload and artifact
   directory. Declare every remote artifact to retrieve before execution.
6. Pass `packages` or `requirements_file` only when the user requested those
   dependencies or the repository already declares them. Never pass both.
7. Check `state`, captured execution output, artifact paths, log path, and
   `cleanup.succeeded` in the result. Treat failed cleanup as unfinished work.
8. Summarize the runtime, accelerator, elapsed time, outputs, artifacts, and
   cleanup outcome.

## Reusable sessions

1. Call `colab_start_session` once with the shortest practical idle timeout.
   The 600-second default is appropriate for normal iterative work.
2. Call `colab_execute` with an absolute `.py` or `.ipynb` path. To run one
   notebook code cell, pass exactly one of `cell_index` or `cell_id`.
   `cell_index` is zero-based across markdown and code cells.
3. Reuse the returned `session_name`. Calls on one session are serialized, and
   kernel state carries across successful executions.
4. Use `colab_renew_session` only when the user is intentionally retaining
   state without executing another operation.
5. Download required `/content` files with `colab_download_artifact`, then use
   `colab_export_log` when a replayable notebook record is useful.
6. Call `colab_stop_session` immediately after the last stateful operation.
   Confirm its result or inspect `colab_sessions`; do not wait for idle expiry
   as the normal cleanup path.

## Managed notebook cells

1. Treat the local `.ipynb` as the source of truth and the leased Colab resource
   as a stateful Python kernel.
2. Call `colab_notebook_cells` before editing. Retain the target cell ID or
   index and its `source_sha256`.
3. Pass that hash as `expected_source_sha256` to
   `colab_update_notebook_cell`. On conflict, inspect again; do not overwrite
   the newer source.
4. For long work, call `colab_execute` with a selected cell and
   `background=true`. Poll the returned ID with `colab_cell_status`.
5. Use `write_output_to_notebook=true` only when local output persistence is
   useful. A source-hash conflict prevents stale output from being written.
6. Never infer completion from an empty status response. Require a terminal
   execution state.

## Safety

- Keep `max_runtime_seconds` finite and proportional to the task.
- Keep reusable-session idle leases short. An idle GPU session still consumes
  Colab account capacity.
- Never use GPU or TPU merely because one may be available.
- Do not invoke `colab auth`, `colab drivemount`, `colab repl`, or
  `colab console` through the non-interactive MCP server.
- Do not download undeclared paths or paths outside `/content`.
- Do not retry a failed workload automatically when it may have side effects.
- Do not automatically replay a timed-out or cancelled persistent execution.
  The connector releases that session because remote execution is ambiguous.
- Use `colab_stop_session` only for a session the user placed in scope or a
  generated session reported by a failed cleanup.
- If cancellation or cleanup status is ambiguous, inspect with
  `colab_sessions` and `colab_session_status`; never assume compute was released.

## Tool choice

- `colab_cli_doctor`: installation and authentication preflight; no allocation.
- `colab_sessions`: read-only inventory of assigned sessions.
- `colab_session_status`: read-only inspection of one session.
- `colab_start_session`: reusable kernel with an automatic idle lease.
- `colab_notebook_cells`: bounded local cell inspection and source identities.
- `colab_update_notebook_cell`: guarded atomic local cell update.
- `colab_execute`: whole local workload or one selected notebook code cell on a
  connector-managed session; selected cells support background mode.
- `colab_cell_status`: bounded state and output for a selected-cell execution.
- `colab_renew_session`: intentional lease extension.
- `colab_download_artifact`: one file beneath `/content` from a leased session.
- `colab_export_log`: replayable notebook history for a leased session.
- `colab_run_job`: fresh bounded job with artifact/log collection and cleanup.
- `colab_stop_session`: explicit emergency release.
