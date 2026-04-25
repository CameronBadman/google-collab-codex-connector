# google-collab-codex-connector

Codex-compatible MCP adapter for Google Colab.

The upstream `googlecolab/colab-mcp` server starts in Codex, but it depends on
dynamic `notifications/tools/list_changed` updates after the browser connects.
Codex loads the initial bootstrap tool, then does not reliably refresh into the
proxied notebook tool set.

This repo provides a local MCP adapter with a static tool list from startup. It
keeps the Colab websocket/session bridge compatible with the upstream browser
flow, while exposing stable `colab_*` tools that Codex can see immediately.

## What Works

- Connect Codex to a Colab browser tab through the Colab MCP websocket bridge.
- Read notebook cells.
- Add code or markdown cells.
- Update cells.
- Run existing code cells.
- Run arbitrary Python by adding a code cell and executing it.
- Install packages by adding and running a `%pip install ...` cell.
- Fall back to raw remote tool calls when Colab changes its frontend tool names.

The live Colab frontend currently exposes these remote tools:

- `add_code_cell`
- `add_text_cell`
- `delete_cell`
- `get_cells`
- `move_cell`
- `run_code_cell`
- `update_cell`

## Limitations

Colab does not currently expose a runtime-management tool through this MCP
surface. Changing CPU/GPU/TPU, reconnecting runtime hardware, or selecting a
different runtime type must still be done manually in the Colab UI:

`Runtime` -> `Change runtime type`

After changing runtime settings, keep or reopen the MCP connection URL so Codex
can continue using the notebook tools.

## Codex Config

Use this MCP server entry in `~/.codex/config.toml`:

```toml
[mcp_servers.colab]
command = "uv"
args = ["--cache-dir", "/tmp/uv-cache", "run", "colab-codex-adapter"]
cwd = "/home/cameron/projects/google-collab-codex-con"
startup_timeout_sec = 30
tool_timeout_sec = 300

[mcp_servers.colab.env]
UV_CACHE_DIR = "/tmp/uv-cache"
UV_TOOL_DIR = "/tmp/uv-tools"
BROWSER = "/tmp/colab-mcp-browser-shim"
```

Restart Codex after changing this config or after changing adapter code. MCP
servers are loaded when the Codex session starts.

## Browser Shim

The adapter uses Python's `webbrowser.open_new()` to open the Colab connection
URL. In a Codex-managed background process, launching Firefox directly is not as
reliable as writing the URL to a file and opening it yourself.

Use this shim at `/tmp/colab-mcp-browser-shim`:

```sh
#!/bin/sh
printf '%s\n' "$@" > /tmp/colab-mcp-open-url
```

Make it executable:

```bash
chmod +x /tmp/colab-mcp-browser-shim
```

When `colab_connect` runs, open the full URL written to:

```text
/tmp/colab-mcp-open-url
```

The URL must include the fragment with `mcpProxyToken` and `mcpProxyPort`.

## Usage

After Codex starts with the adapter configured:

1. Call `colab_status`.
2. If not connected, call `colab_connect`.
3. Open the generated URL from `/tmp/colab-mcp-open-url` in Firefox.
4. Call `colab_status` again and confirm `connected` is `true`.
5. Use the notebook tools.

`colab_status` reports the connection phases separately:

- `server_listening`: the local websocket server is bound to a port.
- `browser_ws_connected`: the Colab browser tab opened the websocket.
- `remote_mcp_initialized`: the browser-side MCP session completed startup and
  tool discovery.
- `remote_tool_count`: cached count of discovered Colab frontend tools.

Notebook tools should only be used after `remote_mcp_initialized` is `true`.
Status calls are intentionally non-blocking around remote tool discovery; if the
browser connects but remote MCP startup stalls, status returns the partial state
instead of hanging.

Primary tools:

- `colab_connect`
- `colab_status`
- `colab_list_remote_tools`
- `colab_get_notebook`
- `colab_add_cell`
- `colab_update_cell`
- `colab_run_cell`
- `colab_run_python`
- `colab_install_package`

Diagnostic escape hatch:

- `colab_call_remote_tool`

If a wrapper does not match Colab's current frontend schema, call
`colab_list_remote_tools` and then use `colab_call_remote_tool` with the exact
remote tool name and arguments.

## Local Development

Run tests:

```bash
uv --cache-dir /tmp/uv-cache run pytest
```

Run the adapter directly:

```bash
UV_CACHE_DIR=/tmp/uv-cache \
UV_TOOL_DIR=/tmp/uv-tools \
BROWSER=/tmp/colab-mcp-browser-shim \
uv --cache-dir /tmp/uv-cache run colab-codex-adapter
```

The direct adapter command starts an MCP server on stdio, so it is usually most
useful when launched by Codex rather than from an interactive terminal.
