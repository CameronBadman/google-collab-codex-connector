# Google Colab Codex Connector

A stable Model Context Protocol (MCP) adapter that lets Codex inspect, edit,
and execute Google Colab notebooks. It also supports native Codex workers using
a separately configured model while sharing the parent's Colab session.

The upstream `googlecolab/colab-mcp` integration discovers notebook tools after
the browser connects and announces them through `notifications/tools/list_changed`.
Codex does not always refresh that dynamic tool list. This connector exposes a
static `colab_*` tool surface from startup and resolves each operation against
the live Colab frontend after connection.

> **Project status:** functional and tested, but still experimental. The current
> release is designed for one trusted user, one local Codex run, and one Colab
> notebook. It is not a hosted or multi-tenant service.

## Highlights

- Stable tools visible to Codex before the browser connects.
- Notebook inspection, cell creation, editing, movement through raw tools, and
  execution.
- Immediate background job IDs for long-running Python and training cells.
- One shared Colab session and job registry across native Codex subagents.
- Project-scoped worker profiles with an explicitly selected Codex model.
- Connection diagnostics, stale-session recovery, and a raw-tool escape hatch
  for upstream schema changes.
- Authenticated loopback broker with private runtime state.

## Architecture

```text
                         Google Colab browser tab
                                  |
                         authenticated WebSocket
                                  |
                     +------------v-------------+
                     | shared connector broker  |
                     | session, tools, and jobs |
                     +------+-------------+-----+
                            |             |
                    loopback MCP     loopback MCP
                            |             |
                  +---------v--+     +----v----------+
                  | parent     |     | Codex worker  |
                  | stdio MCP  |     | stdio MCP     |
                  +------------+     +---------------+
```

Codex starts a separate stdio MCP process for a spawned agent. The first
adapter process becomes the broker owner; later processes proxy to it. Both
agents therefore observe the same `connection_id`, notebook state, and tracked
jobs instead of opening competing Colab bridges.

## Requirements

- Linux or another platform with `fcntl` file locking.
- Python 3.13 or newer.
- [`uv`](https://docs.astral.sh/uv/).
- A local Codex client with MCP support.
- A Google Colab browser session.

## Installation

```bash
git clone https://github.com/CameronBadman/google-collab-codex-connector.git
cd google-collab-codex-connector
uv --cache-dir /tmp/uv-cache sync
```

Configure the connector in `~/.codex/config.toml`. Replace `cwd` with the path
to your checkout:

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

Restart Codex after changing MCP configuration or connector code. Codex loads
MCP servers when a session starts.

## Quick Start

1. Ask Codex to call `colab_status`.
2. If it is disconnected, call `colab_connect`.
3. The connector opens the URL in the desktop default browser. If desktop
   launching is unavailable, open `/tmp/colab-mcp-open-url` manually.
4. Call `colab_status` again and confirm `remote_mcp_initialized` is `true`.
5. Inspect the notebook with `colab_get_notebook` or execute code with
   `colab_run_python`.

The connection URL includes the `mcpProxyToken` and `mcpProxyPort` fragment;
open it without removing that fragment. The launcher is detached with all
standard file descriptors redirected away from MCP stdio. To disable automatic
browser launching:

```bash
COLAB_CODEX_OPEN_NATIVE_BROWSER=0
```

## Native Codex Worker

The connector can install a project-scoped `colab_worker` custom agent. You
choose the worker model explicitly; the connector does not silently select or
fall back to a model.

From the connector checkout, run:

```bash
uv --cache-dir /tmp/uv-cache run colab-codex-agent-init \
  --project /absolute/path/to/your/project \
  --model YOUR_WORKER_MODEL \
  --reasoning-effort medium
```

This writes:

```text
<project>/.codex/agents/colab-worker.toml
```

The generated worker:

- Uses `workspace-write` access for the repository.
- Inherits the project's `colab` MCP tools.
- Owns repository and notebook mutations while its assignment is active.
- Reports a gated checkpoint after environment setup and each completed
  experiment or evaluation.
- Chooses its own `1-900` second job-wait interval from the expected next useful
  signal instead of waking the parent on a fixed schedule.
- Waits for the parent to review evidence and issue the next instruction.
- Does not create nested agents.

Start a new Codex session after installing or changing the profile. Delegate
explicitly:

```text
Use the colab_worker agent for this experiment. Do not edit while it owns the
assignment. Review each checkpoint and send the next instruction to the same
worker thread.
```

At a checkpoint the worker reports its changes, evidence, metrics,
interpretation, recommendation, decision needed, and proposed next step. The
parent remains the supervisor; the worker remains a native Codex agent rather
than a separate Responses API loop.

While a job is running, the worker stays inside its active turn. It normally
uses `10-60` second waits for setup, `60-300` seconds for evaluation, and
`300-900` seconds for training. After a wait expires it inspects the new job
state and selects the next interval from the evidence. The parent is not
invoked merely because a polling timer elapsed.

## Tool Reference

| Area | Tools | Purpose |
| --- | --- | --- |
| Connection | `colab_connect`, `colab_status`, `colab_connection_url`, `colab_reset_connection` | Establish, inspect, or replace the browser bridge. |
| Diagnostics | `colab_adapter_info`, `colab_list_remote_tools` | Inspect broker, connection, and upstream tool metadata. |
| Notebook | `colab_get_notebook`, `colab_add_cell`, `colab_update_cell`, `colab_run_cell` | Read and modify notebook cells. |
| Python | `colab_run_python`, `colab_install_package` | Execute code or install packages synchronously. |
| Jobs | `colab_run_python_async`, `colab_job_status`, `colab_wait_job`, `colab_run_python_wait`, `colab_list_jobs` | Run and monitor long-lived execution. |
| Escape hatch | `colab_call_remote_tool` | Call an exact browser-side tool when a wrapper no longer matches Colab. |

Use notebook tools only after `remote_mcp_initialized` becomes `true`.
`colab_connection_url` is local-only and remains safe to call when the browser
side is stale or only partially connected.

## Background Jobs

Start long-running execution without holding a Codex tool call open:

```text
colab_run_python_async(
  code="train_model()",
  execution_timeout_seconds=43200
)
```

The tool creates a notebook cell, schedules its blocking `run_code_cell`
request in the broker, and returns a `job_id` immediately. Poll with
`colab_job_status` or wait in bounded intervals with `colab_wait_job`.

`colab_wait_job` accepts `timeout_seconds` from `1` through `900`. It waits on a
broker completion event and returns immediately when execution finishes; it
does not query Colab once per second. A timeout performs one status refresh and
returns `wait_timed_out`, `waited_seconds`, `updated_at`, `last_output_at`, and
`task_alive` so the worker can choose its next observation interval.

Job states are:

- `running`: the browser-side execution request is active.
- `finished`: execution completed, including successful cells with no output.
- `error`: Colab returned an error output or the remote request failed.
- `timed_out`: execution exceeded `execution_timeout_seconds`.
- `missing`: the tracked notebook cell was removed.
- `stale`: the connection was reset or the broker shut down.

`colab_wait_job(..., timeout_seconds=300)` timing out does not cancel the job.
The execution continues until it reaches a terminal state or its independent
execution timeout.

Jobs are in memory, owned by the root broker, and shared by agent proxies. They
do not survive the root Codex session ending.

## Connection and Broker Details

The browser-facing bridge accepts one authenticated Colab WebSocket. The
agent-facing broker listens only on `127.0.0.1`, uses a random bearer token,
and stores its state under `/tmp/colab-codex-adapter` with user-only
permissions. The token is never returned by `colab_adapter_info` or the doctor.

The default broker port is `8765`. Set the same
`COLAB_CODEX_BROKER_PORT` value for every connector instance if it must be
changed. A stale broker state file is removed automatically when a new owner
acquires the process lock.

Useful connection fields include:

- `server_listening`: the local Colab WebSocket bridge has bound a port.
- `browser_ws_connected`: a Colab tab opened the bridge.
- `remote_mcp_initialized`: browser-side MCP initialization completed.
- `remote_tool_count`: number of discovered Colab frontend tools.
- `connection_id`: identity shared by the parent and its workers.
- `broker_pid`: process that owns the Colab session and jobs.

## Diagnostics

Run the doctor outside Codex:

```bash
uv --cache-dir /tmp/uv-cache run colab-codex-doctor
```

It reports the owner PID, process state, log path, connection metadata, and a
redacted broker record. If it reports
`adapter_process_running_without_shared_broker`, an older adapter is still
running; restart Codex to load the multi-agent connector.

The worker installer also checks the effective project/global Codex MCP
configuration. If `tool_timeout_sec` is missing or below `1200`, it prints the
exact required change. It never rewrites Codex configuration automatically.

| Symptom | Action |
| --- | --- |
| Only bootstrap or no `colab_*` tools appear | Confirm the MCP `cwd`, then restart Codex. |
| `browser_ws_connected` is `false` | Call `colab_connect`; if automatic launch fails, open `/tmp/colab-mcp-open-url`. |
| Browser connected but MCP is not initialized | Close stale Colab tabs, call `colab_reset_connection`, and open the new URL. |
| Broker port is already in use | Choose another `COLAB_CODEX_BROKER_PORT` and restart all related Codex sessions. |
| Worker waits fail near five minutes | Set `mcp_servers.colab.tool_timeout_sec = 1200` and restart Codex. |
| A job becomes `stale` | Reconnect Colab and start a new job; stale jobs cannot be resumed. |
| A wrapper fails after a Colab update | Inspect `colab_list_remote_tools`, then use `colab_call_remote_tool` with the exact schema. |

Adapter logs default to:

```text
/tmp/colab-codex-adapter/logs/colab-codex-adapter.log
```

## Current Limitations

- Colab does not expose CPU/GPU/TPU selection or runtime reconnection through
  this MCP surface. Change hardware manually through **Runtime -> Change
  runtime type** in Colab.
- The broker and job registry are process-local and end with the root Codex
  adapter.
- V1 assumes one trusted local user, one active root Codex workflow, one worker,
  and one connected notebook.
- Exclusive parent/worker edit ownership is an orchestration rule; MCP calls do
  not carry a reliable agent identity that the connector can lock against.
- Resetting the browser connection invalidates running job tracking.
- The raw remote-tool escape hatch is intentionally powerful and should be used
  only after inspecting the live schema.

## Security Model

- The broker binds to loopback and requires a random bearer token.
- Runtime directories use mode `0700`; broker state and lock files use `0600`.
- Browser and broker tokens are separate.
- Diagnostic output removes the broker token.
- No credentials are stored in the generated worker profile.

The connector does not provide tenant isolation, remote authentication, or a
hosted security boundary. Do not expose the loopback broker through a public
proxy.

## Development

Install development dependencies and run the suite:

```bash
uv --cache-dir /tmp/uv-cache sync --group dev
uv --cache-dir /tmp/uv-cache run pytest -q
```

Build distributable artifacts:

```bash
uv --cache-dir /tmp/uv-cache build
```

The tests cover static tool discovery, browser connection state, tool-schema
adaptation, real background job transitions, broker authentication and owner
election, stale-state recovery, agent-profile installation, diagnostics
redaction, and two independent stdio adapters sharing one broker.

When submitting a change, include focused tests for altered tool contracts,
connection lifecycle behavior, or job-state transitions.

## Roadmap

- Durable worker and job recovery across Codex restarts.
- A supported cancellation path when Colab exposes an interrupt primitive.
- Broader platform support beyond `fcntl`-based local ownership.
- Compatibility tracking as the Colab frontend MCP schema evolves.

Issues and pull requests are welcome at
[`CameronBadman/google-collab-codex-connector`](https://github.com/CameronBadman/google-collab-codex-connector).
