from __future__ import annotations

import pytest
from mcp.types import CallToolResult, TextContent, Tool

from colab_codex_adapter.tools import build_args, pick_tool, serialize_tool_result


def tool(name: str, properties: dict | None = None, description: str = "") -> Tool:
    return Tool(
        name=name,
        description=description,
        inputSchema={"type": "object", "properties": properties or {}},
    )


def test_pick_tool_prefers_exact_name() -> None:
    selected = pick_tool(
        [tool("runtime_execute_code"), tool("execute_code")],
        None,
        ["execute_code"],
    )
    assert selected.name == "execute_code"


def test_pick_tool_accepts_explicit_remote_name() -> None:
    selected = pick_tool([tool("custom")], "custom", ["execute_code"])
    assert selected.name == "custom"


def test_pick_tool_reports_available_tools() -> None:
    with pytest.raises(ValueError, match="Available remote tools: custom"):
        pick_tool([tool("custom")], None, ["execute_code"])


def test_build_args_maps_common_schema_names() -> None:
    selected = tool(
        "add_cell",
        {
            "source": {"type": "string"},
            "cellType": {"type": "string"},
            "position": {"type": "integer"},
        },
    )
    args = build_args(
        selected, {"code": "print(1)", "cell_type": "code", "cell_index": 2}
    )
    assert args == {"source": "print(1)", "cellType": "code", "position": 2}


def test_serialize_tool_result_uses_json_aliases() -> None:
    result = CallToolResult(
        content=[TextContent(type="text", text="ok")],
        structuredContent={"value": 1},
    )
    assert serialize_tool_result(result)["structuredContent"] == {"value": 1}
