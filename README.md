# Google Colab Codex Connector

An MCP connector that lets Codex inspect, edit, and execute Google Colab
notebooks through the browser-side Colab MCP frontend. It exposes a stable
`colab_*` tool surface before the browser connects. One detached connector
service owns the notebook session for the local user; parent agents and native
Codex workers reach that service through disposable stdio shims.

> **Project status:** experimental. Version `0.3.0` targets one trusted local
> user, one active Colab browser tab, and one notebook runtime. It is not a
> hosted or multi-tenant service.

## Reliability Model

- One detached loopback service owns the browser bridge, notebook session, job
  registry, and artifact store independently of any Codex stdio shim.
- Stdio shims discover the current service generation and automatically elect
  exactly one replacement owner when the active owner dies.
- Browser disconnects discard per-connection MCP streams. The same Colab tab,
  URL, and token can reconnect without resetting the bridge.
- Connect and reset operations default to `open_browser=False`; recovery never
  opens a surprise browser tab.
- Job status checks never download accumulated notebook outputs. Timed waits
  use a local completion event and make no Colab request on timeout.
- Reconnected jobs are rechecked through bounded runtime markers every 15
  seconds by default until they become terminal.
- Final MCP results are capped at 256 KiB by default. Larger data is exposed by
  an opaque, chunk-readable artifact reference.
- WebSocket frames have a finite 32 MiB ceiling. This is transport headroom,
  not permission to return unbounded notebook data.

The connector does not add Google Drive synchronization, Drive authentication,
or Drive-specific workflow features.

## Architecture

```text
                       one Google Colab browser tab
                                   |
                       authenticated WebSocket
                                   |
                     +-------------v-------------+
                     | shared connector service |
                     | bridge, session, jobs,    |
                     | bounded artifact service  |
                     +------+------+-------------+
                            |      |
                 authenticated    authenticated
                 loopback MCP     loopback MCP
                            |      |
                     +------v--+ +-v------------+
                     | parent  | | Codex worker |
                     | stdio   | | stdio        |
                     +---------+ +--------------+
```

The singleton boundary is the stateful connector service, not an individual MCP
transport. Codex may start a lightweight stdio shim for each parent or worker
session; all shims discover and use the same service instance. Before each
request, a shim rechecks protected discovery state so it does not remain pinned
to a dead owner. A launch lock serializes replacement, while a lifetime lock
identifies the active daemon. Closing one or every stdio shim does not terminate
the service or an active notebook job. A later shim reconnects to the same
instance.

### Service Identity

`colab_adapter_info` reports the shared service through canonical fields:

| Field | Meaning |
| --- | --- |
| `service_instance_id` | Stable logical identity retained across compatible owner replacement. |
| `service_pid` | PID of the process currently hosting the service. |
| `service_owner_id` | Process-specific owner identity used to reject stale state. |
| `service_generation` | Monotonic owner generation. |
| `service_started_at` | Start time of the current service owner. |
| `service_status` / `service_healthy` | Published lifecycle state and endpoint health. |
| `instance_scope` | `user`; trusted local conversations intentionally share the service. |
| `transport` | `stdio`; describes the Codex-facing shim transport. |

The existing `adapter_*` and `broker_*` fields remain compatibility aliases.
They describe the shared service and must not be interpreted as the identity of
the calling stdio shim.

## Requirements

- Linux or another platform with `fcntl` file locking.
- Python 3.13 or newer.
- [`uv`](https://docs.astral.sh/uv/).
- A local Codex client with MCP support.
- A signed-in Google Colab browser session.

## Installation

```bash
git clone https://github.com/CameronBadman/google-collab-codex-connector.git
cd google-collab-codex-connector
uv --cache-dir /tmp/uv-cache sync --group dev
```

Add the connector to `~/.codex/config.toml`, replacing `cwd` with the checkout:

```toml
[mcp_servers.colab]
command = "uv"
args = ["--cache-dir", "/tmp/uv-cache", "run", "colab-codex-adapter"]
cwd = "/absolute/path/to/google-collab-codex-connector"
startup_timeout_sec = 30
tool_timeout_sec = 1200

[mcp_servers.colab.env]
UV_CACHE_DIR = "/tmp/uv-cache"
UV_TOOL_DIR = "/tmp/uv-tools"
COLAB_CODEX_BROKER_PORT = "8765"
```

After changing MCP configuration, restart Codex. After upgrading connector
code, stop the detached service first so the next shim starts the matching
backend, then restart Codex:

```bash
uv --cache-dir /tmp/uv-cache run colab-codex-broker-stop
```

Stopping the service interrupts active connector-side job tracking, so complete
or checkpoint important work before upgrading.

## Quick Start

1. Call `colab_status`.
2. Call `colab_connect(open_browser=False)`. This starts or discovers the
   bridge but does not launch a browser.
3. Use `colab_connection_url` and open its URL in the intended Colab tab, or
   explicitly call `colab_connect(open_browser=True)` once to use `xdg-open`.
4. Confirm `browser_alive` and `runtime_alive` are both `true` in
   `colab_status`.
5. Inspect the notebook with `colab_get_notebook` or run code with
   `colab_run_python`.

The URL contains `mcpProxyToken` and `mcpProxyPort` in its fragment; do not
remove them. When explicit desktop launching is requested, the URL is also
written to `/tmp/colab-codex-adapter/open-url` with mode `0600`. Disable native launching
entirely with `COLAB_CODEX_OPEN_NATIVE_BROWSER=0`.

Set `COLAB_CODEX_NOTEBOOK_URL` to an existing notebook URL to avoid opening the
default `empty.ipynb`. Only HTTPS URLs on `colab.research.google.com` and
`colab.google.com` are accepted.

### Reconnection

When the browser WebSocket drops, the bridge returns to its listening state and
clears the disconnected frontend client and tool cache. Reopening or refreshing
the same tab reuses the token, URL, port, and connector identity while creating
fresh MCP streams. No reset is needed. A second simultaneous browser is rejected
without disconnecting the active tab.

Use `colab_reset_connection` only for an explicit identity/token rotation. It
also defaults to `open_browser=False`.

## Tool Reference

| Area | Tools | Purpose |
| --- | --- | --- |
| Connection | `colab_connect`, `colab_status`, `colab_connection_url`, `colab_reset_connection` | Establish, inspect, or deliberately replace the browser bridge. |
| Diagnostics | `colab_adapter_info`, `colab_list_remote_tools` | Inspect independent broker, browser, runtime, and connection state. |
| Notebook | `colab_get_notebook`, `colab_add_cell`, `colab_update_cell`, `colab_run_cell` | Read or modify cells; execute existing CPython source through a bounded tracked job. Notebook-wide output reads are disabled. |
| Python | `colab_run_python`, `colab_install_package` | Execute short synchronous code or install packages. |
| Jobs | `colab_run_python_async`, `colab_job_status`, `colab_wait_job`, `colab_run_python_wait`, `colab_list_jobs` | Run and observe long CPython workloads. |
| Artifacts | `colab_read_artifact` | Read an issued artifact in bounded chunks. |
| Safety guard | `colab_call_remote_tool` | Reject raw frontend calls whose output cannot be bounded before crossing WebSocket transport. |

Use remote notebook tools only after `runtime_alive` becomes `true`.
`colab_connection_url` is local-only and remains available while the frontend
is disconnected.

## Background Jobs

Start a tracked CPython job without holding the initiating tool call open:

```text
colab_run_python_async(
  code="train_model()",
  execution_timeout_seconds=43200
)
```

Managed execution accepts CPython source only. Raw frontend execution, IPython
magics, and shell syntax are excluded because their output cannot be bounded
before it crosses the browser transport. Execution deadlines must be finite and
no greater than 86,400 seconds.

The wrapper records a compact completion marker and bounded output artifact
under `/content/.colab_codex/jobs/`. The blocking `run_code_cell` request is the
only normal source of final target-cell output. While a job runs:

- `colab_job_status` calls `get_cells(includeOutputs=False)` only to confirm
  that the target cell still exists; bounded range requests are used for
  notebook scans and unrelated notebook outputs are not read.
- `colab_wait_job` accepts `timeout_seconds` from `1` through `900`, waits on a
  local event, and returns cached metadata on timeout without a remote request.
- `colab_list_jobs` returns metadata only and omits output excerpts.
- Submitted source and raw cell-creation results are never echoed. Responses
  expose `code_bytes` and `code_sha256` instead.

Completed executions release their notebook cell back to a connector-owned
pool. The pool defaults to 16 cells, bounds persistent wrapper source growth,
and rejects additional concurrent work when every pooled cell is still owned by
an unfinished job. The reusable recovery-probe cell is persisted across broker
replacement rather than appended again.

A browser disconnect never causes `run_code_cell` to be replayed. The job moves
to `tracking_state="detached"`, with `execution_alive=null` because execution
cannot be known from transport state alone. After the browser reconnects,
bounded reconciliation probes read connector-owned completion markers. Running
markers are polled at the configured interval until terminal or until the job's
execution timeout plus recovery grace has elapsed. A terminal marker completes
the original job; a missing or stale marker produces `state="interrupted"`.
The connector never guesses by rerunning user code.

Bounded metadata for up to 1,024 jobs is journaled atomically to the private
local state directory. The journal contains no submitted code, errors, output
excerpts, or corpus data. A replacement broker restores unfinished entries as
detached and reconciles them against their runtime markers after reconnect.

Useful job fields include:

- `state`: `running`, `finished`, `error`, `timed_out`, `missing`, `stale`, or
  `interrupted`.
- `tracking_state`: `active`, `detached`, `recovering`, or `complete`.
- `task_alive`: whether the broker still has a local tracking task.
- `execution_alive`: `true`, `false`, or `null` when Colab execution is unknown.
- `output_bytes`, `output_excerpt_bytes`, and `output_truncated`.
- `output_artifact` and `output_unavailable_reason`.
- `wait_timed_out` and `waited_seconds` on wait responses.

## Bounded Results and Artifacts

The default job excerpt is 64 KiB and the final serialized MCP response budget
is 256 KiB. Complete arrays, checkpoints, rich display payloads, large logs,
and notebook histories are never inlined. If a result exceeds the budget, the
response contains compact size, checksum, truncation, and artifact metadata.

Read issued artifacts with:

```text
colab_read_artifact(artifact_id="...", offset=0, limit_bytes=65536)
```

Chunks are at most 64 KiB and are returned as UTF-8 when valid or base64
otherwise. Each response includes the next offset, EOF flag, stored size, and
SHA-256 metadata. IDs are opaque and only connector-issued IDs are accepted.

Runtime output artifacts remain in the current Colab runtime. Runtime and local
artifacts default to a 32 MiB individual limit, 256 MiB total quota, 24-hour
expiry, and oldest-first eviction. Local storage uses `0700` directories and
`0600` files. The broker retains at most 1,024 tracked job records by default.

## Broker Recovery

Broker discovery state is stored under `/tmp/colab-codex-adapter` with user-only
permissions. It includes the stable service instance ID, owner PID/UUID,
protocol version, generation, endpoint, and a private bearer token. Tokens are
not passed in daemon command line arguments or returned by diagnostics.

When a proxy detects a dead endpoint, it rereads discovery state and contends
for the launch lock. Exactly one proxy starts a detached replacement and
publishes the next generation; the others attach after its health check passes.
Read-only discovery is retried once after recovery. Mutating requests with an
ambiguous outcome are not replayed; the proxy recovers the broker and returns an
explicit outcome-unknown error so callers can inspect job/status state first.

For an intentional upgrade or test teardown, stop the detached daemon with:

```bash
uv --cache-dir /tmp/uv-cache run colab-codex-broker-stop
```

Use `--owner-id` to require a specific discovered owner before signaling it.
The next adapter request starts a replacement.

## Configuration

| Variable | Default | Meaning |
| --- | ---: | --- |
| `COLAB_CODEX_BROKER_PORT` | `8765` | Authenticated loopback broker port. |
| `COLAB_CODEX_NOTEBOOK_URL` | Colab `empty.ipynb` | Validated browser launch target. |
| `COLAB_CODEX_WS_MAX_FRAME_BYTES` | `33554432` | Finite WebSocket frame limit; cannot exceed 32 MiB. |
| `COLAB_CODEX_MAX_TOOL_RESPONSE_BYTES` | `262144` | Final serialized MCP result budget. |
| `COLAB_CODEX_JOB_OUTPUT_EXCERPT_BYTES` | `65536` | Maximum cached job output excerpt. |
| `COLAB_CODEX_MAX_SUBMITTED_CODE_BYTES` | `1048576` | Maximum UTF-8 source bytes sent through the browser connector. |
| `COLAB_CODEX_CELL_METADATA_PAGE_SIZE` | `8` | Maximum cells requested in one output-free notebook metadata frame. |
| `COLAB_CODEX_MAX_TRACKED_JOBS` | `1024` | In-memory job records; oldest completed jobs are evicted first. |
| `COLAB_CODEX_MAX_JOB_CELLS` | `16` | Maximum reusable connector-owned execution cells; also bounded by the tracked-job limit. |
| `COLAB_CODEX_JOB_JOURNAL_PATH` | `/tmp/colab-codex-adapter/jobs.json` | Private bounded job-metadata journal. |
| `COLAB_CODEX_JOB_RECONCILIATION_POLL_SECONDS` | `15` | Poll interval for detached jobs whose runtime marker is still running. |
| `COLAB_CODEX_RUNTIME_STALE_GRACE_SECONDS` | `300` | Recovery grace after a job's execution deadline before a running marker is considered interrupted. |
| `COLAB_CODEX_REMOTE_INIT_TIMEOUT_SECONDS` | `30` | Browser frontend MCP initialization timeout. |
| `COLAB_CODEX_REMOTE_TOOL_LIST_TIMEOUT_SECONDS` | `30` | Browser tool-discovery timeout. |
| `COLAB_CODEX_ARTIFACT_DIR` | `/tmp/colab-codex-adapter/artifacts` | Private local artifact directory. |
| `COLAB_CODEX_MAX_ARTIFACT_BYTES` | `33554432` | Maximum bytes stored per local/runtime artifact. |
| `COLAB_CODEX_MAX_ARTIFACT_TOTAL_BYTES` | `268435456` | Local and Colab-runtime artifact quota. |
| `COLAB_CODEX_ARTIFACT_TTL_SECONDS` | `86400` | Local and Colab-runtime artifact lifetime. |
| `COLAB_CODEX_ARTIFACT_PROBE_TIMEOUT_SECONDS` | `30` | Maximum time a runtime artifact read waits behind a busy kernel. |

Limits must remain finite. Raising the WebSocket ceiling is not a substitute
for keeping tool and job responses bounded.

## Diagnostics

Run the doctor outside Codex:

```bash
uv --cache-dir /tmp/uv-cache run colab-codex-doctor
```

The report diagnoses the shared service independently of any stdio shim. It
distinguishes:

- the stable service instance ID, current owner PID, process-start identity,
  owner UUID, generation, endpoint health, and owner transition;
- an optional per-shim PID only when explicit per-shim diagnostic paths were
  supplied; no shim is treated as the canonical connector process;
- browser and runtime liveness reported by connector tools;
- the protected state paths needed for local troubleshooting.

WebSocket diagnostics include the last close code and sanitized reason, browser
generation, accepted/rejected connection counts, rejected oversized-frame
bytes, and frame/byte maxima. Job
diagnostics include connection and job IDs plus independent tracking and
execution liveness. Tokens, token-bearing URL fragments, submitted code,
outputs, prompts, corpus contents, tool arguments, and tool results are removed
from persistent diagnostic state.

| Symptom | Action |
| --- | --- |
| `browser_alive` is `false` | Refresh or reopen the same configured notebook URL; do not reset first. |
| Browser is alive but `runtime_alive` is `false` | Allow up to the configured 30-second initialization/tool-list stages, then inspect close diagnostics. |
| A second tab cannot connect | Close the unintended tab; only one active browser is allowed. |
| Broker owner is not running | Make any connector request; one proxy will elect a replacement. |
| Worker waits fail near five minutes | Set `mcp_servers.colab.tool_timeout_sec = 1200` and restart Codex. |
| A job is `detached` | Reconnect the same browser tab and allow marker reconciliation; do not rerun it manually. |
| An artifact ID is unknown | It expired, was evicted, belonged to a previous runtime, or was not connector-issued. |
| An artifact read reports a busy runtime | Retry after the currently executing cell yields the kernel. |

Adapter logs default to
`/tmp/colab-codex-adapter/logs/colab-codex-adapter.log`.

## Native Codex Worker

Install a project-scoped `colab_worker` profile with an explicitly selected
model:

```bash
uv --cache-dir /tmp/uv-cache run colab-codex-agent-init \
  --project /absolute/path/to/your/project \
  --model YOUR_WORKER_MODEL \
  --reasoning-effort medium
```

The generated worker inherits the project's Colab tools, owns repository and
notebook mutations for its assignment, reports gated checkpoints, and chooses
its own `1-900` second wait interval. It normally waits `10-60` seconds for
setup, `60-300` for evaluation, and `300-900` for training. A wait timeout does
not wake the parent or cancel execution.

## Security and Limitations

- Browser and broker tokens are independent. Runtime state and artifacts are
  private to the local OS user.
- The broker binds only to `127.0.0.1` and requires bearer authentication.
- Notebook URLs are restricted to approved HTTPS Colab hosts.
- One trusted local workflow and one active browser tab are supported. There is
  no tenant isolation or remote security boundary.
- Colab hardware selection and runtime reconnection remain browser operations.
- Runtime artifacts disappear when the Colab runtime is replaced.
- Raw remote-tool calls and direct unwrapped cell execution are disabled on the
  managed-safe surface. Remote tool-name overrides are also rejected.
- Notebook metadata is read in bounded pages. A single pre-existing cell whose
  source alone exceeds the configured WebSocket frame limit still cannot be
  inspected until the frontend supports chunked cell-source reads.
- Managed code should not intentionally leave permanent Python background
  threads. Short-lived and descendant-thread output is captured or suppressed,
  but a permanently live thread retains its output guard for the life of that
  thread.

Do not expose the loopback broker through a public proxy.

## Development

```bash
uv --cache-dir /tmp/uv-cache sync --group dev
uv --cache-dir /tmp/uv-cache run pytest -q
uv --cache-dir /tmp/uv-cache build
```

The test suite covers bounded large-output behavior, WebSocket frame limits,
same-token browser reconnection, second-tab rejection, zero-I/O job waits,
in-flight and concurrent broker replacement, artifact permissions and quotas,
diagnostic redaction, agent installation, eight concurrent stdio shims sharing
one service, shim-independent service lifetime, and an integrated long-job
reconnect with 2 MiB of unrelated notebook output plus artifact checksum
verification.

The automated suite runs that reconnect workflow against a protocol-real local
frontend. Manual release verification should repeat it in a signed-in Colab tab
to cover browser and kernel behavior. No Drive workflow is required.

Issues and pull requests are welcome at
[`CameronBadman/google-collab-codex-connector`](https://github.com/CameronBadman/google-collab-codex-connector).
