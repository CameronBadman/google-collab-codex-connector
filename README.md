# Colab Runner for Codex

Colab Runner lets Codex execute bounded, observable workloads through
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
       ├─ progress and hard deadline
       ├─ fresh unique session
       ├─ execute local .py or .ipynb
       ├─ retrieve declared /content artifacts
       ├─ export replayable notebook log
       └─ always attempt colab stop
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
- Artifact downloads restricted to declared paths beneath `/content`.
- A generated session name per job, avoiding collisions with existing sessions.
- Cleanup in a `finally` path on success, failure, timeout, or interruption.
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
| `colab_cli_doctor` | Checks the CLI, required ADC scopes, and read-only session access. |
| `colab_sessions` | Lists active sessions without changing them. |
| `colab_session_status` | Reads one named session's hardware and state. |
| `colab_run_job` | Provisions, executes, retrieves declared outputs, exports a log, and cleans up. |
| `colab_stop_session` | Emergency release for a specifically named session. |

`colab_run_job` accepts an existing local `.py` or `.ipynb`, an absolute local
artifact directory, an optional accelerator, optional declared dependencies,
optional remote files beneath `/content`, and a hard total deadline. It never
leaves a session running intentionally.

Example request to Codex:

```text
Use Colab Runner to execute ./train.py on a T4 for at most 20 minutes.
Install from requirements.txt, download /content/model.safetensors into
./artifacts, export the notebook log, and confirm the runtime was released.
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

## Upstream boundary

The official CLI is pinned to `google-colab-cli==0.6.0`. Upgrade deliberately,
run the contract tests, and check Google's command documentation before changing
the pin. Colab Runner does not patch or vendor the official client.
