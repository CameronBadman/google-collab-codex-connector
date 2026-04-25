from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Any

from mcp.types import CallToolResult, Tool

from .session import ColabSessionManager


def model_to_dict(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(by_alias=True, exclude_none=True)
    if isinstance(value, list):
        return [model_to_dict(item) for item in value]
    if isinstance(value, dict):
        return {key: model_to_dict(item) for key, item in value.items()}
    return value


def serialize_tool_result(result: CallToolResult) -> dict[str, Any]:
    return model_to_dict(result)


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
        for tool in tools:
            if tool.name == preferred_name:
                return tool
        raise ValueError(f"Remote Colab tool not found: {preferred_name}")

    ranked = sorted(
        ((_score_tool(tool, candidate_terms), tool) for tool in tools),
        key=lambda item: item[0],
        reverse=True,
    )
    if ranked and ranked[0][0] > 0:
        return ranked[0][1]
    available = ", ".join(tool.name for tool in tools)
    raise ValueError(
        "Could not resolve a matching Colab remote tool. "
        f"Available remote tools: {available}"
    )


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

    for key, value in logical.items():
        if value is not None and key in props and key not in args:
            args[key] = value
    return args


async def call_resolved_tool(
    manager: ColabSessionManager,
    candidate_terms: list[str],
    logical_args: dict[str, Any],
    preferred_remote_tool: str | None = None,
) -> dict[str, Any]:
    tools = await manager.list_tools()
    tool = pick_tool(tools, preferred_remote_tool, candidate_terms)
    result = await manager.call_tool(tool.name, build_args(tool, logical_args))
    return {
        "remote_tool": tool.name,
        "result": serialize_tool_result(result),
        "text": text_from_result(result),
        "json": first_json_object(result),
    }
