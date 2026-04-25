# google-collab-codex-connector

Codex-compatible MCP adapter for Google Colab's session proxy.

The upstream `googlecolab/colab-mcp` server exposes only a bootstrap tool at
startup, then relies on `notifications/tools/list_changed` after the Colab
browser connects. Codex currently sees the bootstrap tool but does not reliably
refresh into the proxied notebook tools.

This adapter keeps Codex's MCP tool list static from process startup. The
browser/websocket protocol remains compatible with Colab's session proxy, while
Codex calls stable `colab_*` wrapper tools.

## Run

```bash
uv run colab-codex-adapter
```

## Codex config

Replace the upstream Colab MCP server entry in `~/.codex/config.toml` with:

```toml
[mcp_servers.colab]
command = "uv"
args = ["run", "colab-codex-adapter"]
cwd = "/home/cameron/projects/google-collab-codex-con"
startup_timeout_sec = 30
tool_timeout_sec = 300

[mcp_servers.colab.env]
UV_CACHE_DIR = "/tmp/uv-cache"
UV_TOOL_DIR = "/tmp/uv-tools"
BROWSER = "/tmp/colab-mcp-browser-shim"
```

The existing browser shim can stay as:

```sh
#!/bin/sh
printf '%s\n' "$@" > /tmp/colab-mcp-open-url
```

## Tools

- `colab_connect`
- `colab_status`
- `colab_list_remote_tools`
- `colab_call_remote_tool`
- `colab_get_notebook`
- `colab_add_cell`
- `colab_update_cell`
- `colab_run_cell`
- `colab_run_python`
- `colab_install_package`

The wrapper tools resolve likely Colab frontend tool names automatically. If
Colab changes names or exposes a different surface, call
`colab_list_remote_tools` and pass `remote_tool_name` to the wrapper, or use
`colab_call_remote_tool` directly.

## Test

```bash
uv run pytest
```
