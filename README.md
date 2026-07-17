# Colab Runner for Codex

Colab Runner lets Codex execute bounded, observable workloads or reuse a
stateful kernel through
[Google's official Colab CLI](https://github.com/googlecolab/google-colab-cli).
It is a Codex plugin with a small MCP orchestration layer and an agent skill.

The project no longer contains a browser bridge, notebook-tab proxy, WebSocket
broker, or custom Colab transport. Google owns authentication, runtime
provisioning, kernel communication, file transfer, and keep-alive behavior;
Colab Runner adds the agent-facing safety and lifecycle UX around those
official primitives.

```text
Codex
  └─ Colab Runner MCP tools
       ├─ isolated job
       │    └─ create → execute → collect → stop
       ├─ reusable leased session
       │    ├─ create → execute file/cell → execute again
       │    ├─ preserve imports, variables, models, and GPU memory
       │    ├─ retrieve /content artifacts and export a log
       │    └─ explicit stop, idle expiry, or MCP shutdown
       └─ official Google Colab CLI
            └─ Colab runtime
```

## What it provides

- A valid Codex plugin manifest and `colab-runner` skill.
- Live MCP progress for provisioning, dependency setup, execution, downloads,
  log export, and cleanup.
- CPU by default; GPU or TPU allocation must be selected explicitly.
- A finite total deadline between 30 seconds and 24 hours.
- Bounded subprocess output so a noisy workload cannot flood MCP transport.
- Reusable named kernels for iterative work with a 10-minute default idle
  lease.
- Execution of one local notebook code cell by zero-based index or exact cell
  ID without running the rest of the notebook.
- Per-session execution serialization so multiple agents cannot concurrently
  mutate one kernel.
- Artifact downloads restricted to declared paths beneath `/content`.
- A generated session name per job, avoiding collisions with existing sessions.
- Cleanup in a `finally` path on success, failure, timeout, or interruption.
- Automatic cleanup of leased sessions on idle expiry and normal MCP shutdown;
  failed idle cleanup is retried.
- No imports from private Colab CLI internals. Every operation crosses the
  official `colab` command boundary; the doctor additionally uses the pinned
  CLI's read-only `whoami` diagnostic.

## Requirements

- Linux or macOS. The official CLI does not currently support Windows.
- Python 3.12 or newer.
- [`uv`](https://docs.astral.sh/uv/).
- A Google account with Colab access and available compute entitlement.
- Application Default Credentials or a cached Colab CLI OAuth2 login with the
  required scopes described below.

## Setup

Install the pinned official CLI and the local MCP server:

```bash
uv sync --locked
```

Authenticate once for headless agent use:

```bash
gcloud auth application-default login \
  --scopes=openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory
```

Verify the official CLI without allocating a runtime:

```bash
uv run colab --auth=adc version
uv run colab --auth=adc whoami
uv run colab --auth=adc sessions
```

Keep `--auth=adc` before the command. The pinned CLI otherwise defaults to its
interactive OAuth flow, while the MCP server deliberately uses ADC and provides
no interactive stdin. Missing or expired ADC therefore fails promptly with the
CLI's remediation instead of blocking the MCP transport for an authorization
code.

`colab_cli_doctor` performs the same read-only checks. It verifies the installed
version, uses `whoami` to require `openid`, `cloud-platform`, `userinfo.email`,
and `colaboratory`, and only then checks session access. Missing scopes are
reported without returning the account email.

On systems without `gcloud`, use the official CLI's remote copy/paste OAuth2
flow once in a user-controlled terminal:

```bash
uv run colab --auth=oauth2 whoami
export COLAB_RUNNER_AUTH=oauth2
uv run colab --auth=oauth2 sessions
```

Persist `COLAB_RUNNER_AUTH=oauth2` in the environment that launches Codex. The
MCP server defaults to `adc`; it never starts an interactive login itself.

The plugin's `.mcp.json` starts `colab-runner-mcp` with `uv` from the plugin
root. For direct development before installing the plugin, point Codex at this
checkout:

```toml
[mcp_servers.colab-runner]
command = "uv"
args = ["run", "--project", "/absolute/path/to/this/repo", "--locked", "colab-runner-mcp"]
startup_timeout_sec = 120
tool_timeout_sec = 86400

[mcp_servers.colab-runner.env]
COLAB_RUNNER_AUTH = "oauth2" # omit this table to keep the ADC default
```

Restart Codex after changing MCP configuration. Call `colab_cli_doctor` before
the first workload.

## Tools

| Tool | Behavior |
| --- | --- |
| `colab_cli_doctor` | Checks the CLI, required credential scopes, and read-only session access. |
| `colab_sessions` | Lists active sessions and connector-managed lease metadata. |
| `colab_session_status` | Reads one session's hardware, state, and managed lease. |
| `colab_start_session` | Allocates a reusable kernel with a bounded idle lease. |
| `colab_execute` | Executes a `.py`, all code cells in an `.ipynb`, or one selected notebook code cell. |
| `colab_renew_session` | Renews a lease and optionally changes its idle timeout. |
| `colab_download_artifact` | Downloads one file beneath `/content` from a reusable session. |
| `colab_export_log` | Exports a reusable session's execution history as `.ipynb`. |
| `colab_run_job` | Provisions, executes, retrieves declared outputs, exports a log, and cleans up. |
| `colab_stop_session` | Emergency release for a specifically named session. |

`colab_run_job` accepts an existing local `.py` or `.ipynb`, an absolute local
artifact directory, an optional accelerator, optional declared dependencies,
optional remote files beneath `/content`, and a hard total deadline. It never
leaves a session running intentionally.

## Execution modes

Use `colab_run_job` for isolated work. It provisions a fresh runtime and stops
it before returning, including after failure, timeout, or cancellation. This is
the lowest-risk option for a single training run, evaluation, conversion, or
artifact build.

Use a reusable session when multiple steps benefit from the same live Python
kernel:

1. Call `colab_start_session` with an accelerator and idle timeout.
2. Call `colab_execute` repeatedly with local `.py` or `.ipynb` paths.
3. For one notebook cell, pass either `cell_index` or `cell_id`. The index is
   zero-based across all notebook cells, including markdown. The selected cell
   must be a code cell.
4. Retrieve required files with `colab_download_artifact` and optionally export
   the full history with `colab_export_log`.
5. Call `colab_stop_session` as soon as state is no longer needed.

The connector extracts a selected cell locally with `nbformat`, sends only that
cell's source through the official `colab exec` boundary, and deletes the
temporary local file afterwards. Imports, variables, loaded models, files under
`/content`, and GPU allocations from earlier calls remain in the same remote
kernel.

Reusable sessions reduce repeated provisioning, dependency installation,
workload transmission, and setup output. That can reduce agent context usage
and Colab startup overhead. They do not make idle GPU time free: an allocated
runtime continues consuming account capacity while it waits. The default lease
expires after 600 idle seconds, the accepted range is 60 to 21,600 seconds, and
each execution, download, log export, or explicit renewal resets it. Execution
is serialized per session. Execution timeout or cancellation releases the
session because the remote execution state can no longer be proven safe.

Example request to Codex:

```text
Use Colab Runner to execute ./train.py on a T4 for at most 20 minutes.
Install from requirements.txt, download /content/model.safetensors into
./artifacts, export the notebook log, and confirm the runtime was released.
```

Stateful example:

```text
Start a T4 Colab session with a 10-minute idle lease. Run ./load_model.py,
then execute cell ID evaluate-batch in ./evaluation.ipynb on the same kernel.
Download /content/results/metrics.json, export the session log, and stop the
session.
```

## Development

```bash
uv run pytest
```

Tests use fake Colab CLI implementations or local subprocesses. They do not
allocate remote runtimes.

The default suite also checks the pinned binary's version and command help
without authenticating or allocating compute. After configuring ADC, run the
opt-in CPU acceptance test to verify execution, artifact download, notebook-log
export, and cleanup against Colab:

```bash
COLAB_RUNNER_LIVE_TEST=1 uv run pytest -q tests/test_live_colab.py
```

Cancellation cleanup is a separate opt-in because it allocates another runtime:

```bash
COLAB_RUNNER_LIVE_CANCEL_TEST=1 \
  uv run pytest -q tests/test_live_colab.py
```

The reusable-session acceptance test verifies kernel state carry-over,
selected-cell execution, artifact download, log export, and explicit release:

```bash
COLAB_RUNNER_LIVE_SESSION_TEST=1 \
  uv run pytest -q tests/test_live_colab.py
```

## Upstream boundary

The official CLI is pinned to `google-colab-cli==0.6.0`. Upgrade deliberately,
run the contract tests, and check Google's command documentation before changing
the pin. Colab Runner does not patch or vendor the official client.
