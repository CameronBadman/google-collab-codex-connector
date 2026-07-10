from __future__ import annotations

import json
from pathlib import Path

import pytest
from mcp.types import CallToolResult, TextContent, Tool

from colab_codex_adapter.artifacts import ArtifactStore
from colab_codex_adapter.tools import (
    bounded_tool_result,
    build_args,
    pick_tool,
    serialize_tool_result,
)


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


def test_pick_tool_matches_real_colab_run_code_cell_name() -> None:
    selected = pick_tool(
        [tool("add_code_cell"), tool("run_code_cell"), tool("update_cell")],
        None,
        ["run_code_cell", "execute_code", "run_code"],
    )
    assert selected.name == "run_code_cell"


def test_pick_tool_does_not_echo_available_tools() -> None:
    with pytest.raises(ValueError, match="Could not resolve") as exc_info:
        pick_tool([tool("custom")], None, ["execute_code"])
    assert "custom" not in str(exc_info.value)


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


def test_serialize_tool_result_prefers_structured_content_without_duplication() -> None:
    result = CallToolResult(
        content=[TextContent(type="text", text="ok")],
        structuredContent={"value": 1},
    )
    serialized = serialize_tool_result(result)
    assert serialized["structured_content"] == {"value": 1}
    assert "content" not in serialized


def test_bounded_tool_result_uses_short_text_and_single_structured_payload() -> None:
    data = {"state": "finished", "outputs": [{"text": "ok"}]}
    result = bounded_tool_result(data, summary="Job finished")

    assert result.structured_content == data
    assert len(result.content) == 1
    assert result.content[0].text == "Job finished"  # type: ignore[union-attr]
    assert json.dumps(data) not in result.content[0].text  # type: ignore[union-attr]


def test_large_final_tool_result_is_externalized_within_exact_budget(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path, max_artifact_bytes=8192, max_total_bytes=16384)
    result = bounded_tool_result(
        {"value": "x" * 4096},
        summary="Large result",
        artifact_store=store,
        max_response_bytes=1024,
    )
    mcp_result = CallToolResult(
        content=result.content,
        structuredContent=result.structured_content,
    )

    assert len(mcp_result.model_dump_json(by_alias=True).encode("utf-8")) <= 1024
    assert result.structured_content["response_truncated"] is True
    artifact_id = result.structured_content["response_artifact"]["artifact_id"]
    assert store.get_ref(artifact_id).size_bytes > 4096


def test_large_serialized_remote_result_does_not_echo_binary_payload(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path, max_artifact_bytes=8192, max_total_bytes=16384)
    binary = "A" * 4096
    remote = CallToolResult(
        content=[TextContent(type="text", text=binary)],
        structuredContent={"image": binary},
    )

    serialized = serialize_tool_result(
        remote, artifact_store=store, max_bytes=1024
    )

    assert serialized["response_truncated"] is True
    assert binary not in json.dumps(serialized)
