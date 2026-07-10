from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable
from typing import Any

from fastmcp.tools.tool import ToolResult
from mcp.types import CallToolResult, Tool
from mcp.types import TextContent

from .artifacts import ArtifactStore, get_default_artifact_store, json_bytes
from .session import ColabSessionManager

DEFAULT_MAX_TOOL_RESPONSE_BYTES = int(
    os.environ.get("COLAB_CODEX_MAX_TOOL_RESPONSE_BYTES", 256 * 1024)
)
MAX_RESULT_SUMMARY_BYTES = 1024
MCP_PROTOCOL_HEADROOM_BYTES = 1024


def model_to_dict(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(by_alias=True, exclude_none=True)
    if isinstance(value, list):
        return [model_to_dict(item) for item in value]
    if isinstance(value, dict):
        return {key: model_to_dict(item) for key, item in value.items()}
    return value


def _bounded_text(value: str, max_bytes: int) -> str:
    encoded = value.encode("utf-8")
    if len(encoded) <= max_bytes:
        return value
    suffix = b"..."
    return encoded[: max(0, max_bytes - len(suffix))].decode(
        "utf-8", errors="ignore"
    ) + suffix.decode("ascii")


def bounded_payload(
    value: Any,
    *,
    artifact_store: ArtifactStore | None = None,
    max_bytes: int = DEFAULT_MAX_TOOL_RESPONSE_BYTES,
    artifact_field: str = "response_artifact",
) -> Any:
    """Return JSON data no larger than max_bytes, externalizing overflow."""
    raw = json_bytes(value)
    if len(raw) <= max_bytes:
        return value
    store = artifact_store or get_default_artifact_store()
    ref = store.put_bytes(raw, media_type="application/json; charset=utf-8")
    return {
        "response_truncated": True,
        "response_bytes": len(raw),
        artifact_field: ref.to_dict(),
    }


def serialize_tool_result(
    result: CallToolResult,
    *,
    artifact_store: ArtifactStore | None = None,
    max_bytes: int = DEFAULT_MAX_TOOL_RESPONSE_BYTES,
) -> dict[str, Any]:
    structured = getattr(result, "structured_content", None)
    if not isinstance(structured, dict):
        structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict):
        serialized: dict[str, Any] = {
            "structured_content": model_to_dict(structured),
            "is_error": bool(getattr(result, "isError", False)),
        }
    else:
        content: list[dict[str, Any]] = []
        for item in result.content:
            if getattr(item, "type", None) == "text":
                content.append({"type": "text", "text": getattr(item, "text", "")})
            else:
                store = artifact_store or get_default_artifact_store()
                ref = store.put_bytes(
                    json_bytes(model_to_dict(item)),
                    media_type="application/json; charset=utf-8",
                )
                content.append(
                    {
                        "type": str(getattr(item, "type", "rich")),
                        "omitted": True,
                        "artifact": ref.to_dict(),
                    }
                )
        serialized = {
            "content": content,
            "is_error": bool(getattr(result, "isError", False)),
        }
    bounded = bounded_payload(
        serialized, artifact_store=artifact_store, max_bytes=max_bytes
    )
    if not isinstance(bounded, dict):
        raise TypeError("Serialized MCP tool result must be an object")
    return bounded


def _mcp_result_size(content: list[TextContent], data: dict[str, Any]) -> int:
    result = CallToolResult(content=content, structuredContent=data)
    return len(result.model_dump_json(by_alias=True, exclude_none=True).encode("utf-8"))


def bounded_tool_result(
    data: dict[str, Any],
    *,
    summary: str,
    artifact_store: ArtifactStore | None = None,
    max_response_bytes: int = DEFAULT_MAX_TOOL_RESPONSE_BYTES,
) -> ToolResult:
    """Build a ToolResult whose final MCP envelope fits a strict byte budget.

    The text content is deliberately only a short summary. Structured data appears
    once, avoiding the common JSON-in-text plus structuredContent duplication.
    """
    if max_response_bytes < 1024:
        raise ValueError("max_response_bytes must be at least 1024")
    protocol_headroom = min(
        MCP_PROTOCOL_HEADROOM_BYTES, max_response_bytes // 4
    )
    envelope_budget = max_response_bytes - protocol_headroom
    summary = _bounded_text(summary, MAX_RESULT_SUMMARY_BYTES)
    content = [TextContent(type="text", text=summary)]
    if _mcp_result_size(content, data) <= envelope_budget:
        return ToolResult(content=content, structured_content=data)

    raw = json_bytes(data)
    store = artifact_store or get_default_artifact_store()
    ref = store.put_bytes(raw, media_type="application/json; charset=utf-8")
    bounded = {
        "response_truncated": True,
        "response_bytes": len(raw),
        "response_artifact": ref.to_dict(),
    }
    if _mcp_result_size(content, bounded) > envelope_budget:
        content = [TextContent(type="text", text="Result stored as an artifact.")]
    if _mcp_result_size(content, bounded) > envelope_budget:
        raise ValueError("max_response_bytes is too small for artifact metadata")
    return ToolResult(content=content, structured_content=bounded)


def text_from_result(result: CallToolResult) -> str:
    parts: list[str] = []
    for item in result.content:
        if getattr(item, "type", None) == "text":
            parts.append(getattr(item, "text", ""))
    return "\n".join(part for part in parts if part)


def first_json_object(result: CallToolResult) -> Any:
    text = text_from_result(result).strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def tool_summary(tool: Tool) -> dict[str, Any]:
    data = model_to_dict(tool)
    return {
        "name": data.get("name"),
        "description": data.get("description"),
        "input_schema": data.get("inputSchema"),
        "output_schema": data.get("outputSchema"),
    }


def _normal(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _score_tool(tool: Tool, terms: Iterable[str]) -> int:
    name = _normal(tool.name)
    description = _normal(tool.description or "")
    score = 0
    for term in terms:
        term = _normal(term)
        if name == term:
            score += 100
        elif name.endswith(term) or name.startswith(term):
            score += 40
        elif term in name:
            score += 25
        elif term in description:
            score += 5
    return score


def pick_tool(
    tools: list[Tool], preferred_name: str | None, candidate_terms: list[str]
) -> Tool:
    if preferred_name:
        if not re.fullmatch(r"[A-Za-z0-9_.:/-]{1,128}", preferred_name):
            raise ValueError("Invalid remote tool name")
        for tool in tools:
            if tool.name == preferred_name:
                return tool
        raise ValueError("Requested remote Colab tool was not found")

    ranked = sorted(
        ((_score_tool(tool, candidate_terms), tool) for tool in tools),
        key=lambda item: item[0],
        reverse=True,
    )
    if ranked and ranked[0][0] > 0:
        return ranked[0][1]
    raise ValueError("Could not resolve a matching Colab remote tool")


def _schema_properties(tool: Tool) -> dict[str, Any]:
    schema = model_to_dict(tool).get("inputSchema") or {}
    return schema.get("properties") or {}


def _add_if_present(
    args: dict[str, Any], props: dict[str, Any], names: Iterable[str], value: Any
) -> bool:
    if value is None:
        return False
    prop_names = {_normal(name): name for name in props}
    for name in names:
        actual = prop_names.get(_normal(name))
        if actual:
            args[actual] = value
            return True
    return False


def build_args(tool: Tool, logical: dict[str, Any]) -> dict[str, Any]:
    props = _schema_properties(tool)
    if not props:
        return {key: value for key, value in logical.items() if value is not None}

    args: dict[str, Any] = {}
    _add_if_present(
        args, props, ["code", "source", "content", "text"], logical.get("code")
    )
    _add_if_present(
        args, props, ["cell_type", "type", "kind"], logical.get("cell_type")
    )
    _add_if_present(
        args, props, ["cell_id", "cellId", "id"], logical.get("cell_id")
    )
    _add_if_present(
        args, props, ["cell_index", "cellIndex", "index", "position"],
        logical.get("cell_index"),
    )
    _add_if_present(
        args, props, ["after_cell_id", "afterCellId", "after_id"],
        logical.get("after_cell_id"),
    )
    _add_if_present(
        args, props, ["package", "packages", "package_names", "packageNames"],
        logical.get("packages"),
    )
    _add_if_present(
        args,
        props,
        ["include_outputs", "includeOutputs"],
        logical.get("include_outputs"),
    )

    for key, value in logical.items():
        if value is not None and key in props and key not in args:
            args[key] = value
    return args


async def call_resolved_tool(
    manager: ColabSessionManager,
    candidate_terms: list[str],
    logical_args: dict[str, Any],
    preferred_remote_tool: str | None = None,
    *,
    artifact_store: ArtifactStore | None = None,
) -> dict[str, Any]:
    tools = await manager.list_tools()
    tool = pick_tool(tools, preferred_remote_tool, candidate_terms)
    result = await manager.call_tool(tool.name, build_args(tool, logical_args))
    structured = getattr(result, "structured_content", None)
    if not isinstance(structured, dict):
        structured = getattr(result, "structuredContent", None)
    if not isinstance(structured, dict):
        parsed = first_json_object(result)
        structured = parsed if isinstance(parsed, dict) else None
    payload: dict[str, Any] = {
        "remote_tool": tool.name,
        "result": serialize_tool_result(result, artifact_store=artifact_store),
    }
    if structured is None:
        text = text_from_result(result)
        if text:
            payload["text"] = _bounded_text(text, MAX_RESULT_SUMMARY_BYTES)
    bounded = bounded_payload(payload, artifact_store=artifact_store)
    if not isinstance(bounded, dict):
        raise TypeError("Resolved tool result must be an object")
    return bounded
