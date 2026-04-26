#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

import json
from collections.abc import Callable
from datetime import datetime as _dt
from datetime import timezone as _tz
from pathlib import Path as _Path
from typing import Any

from src.tools import count_items, date_diff

from .common import _api_model, _get_client, call_oai
from .librarian import detect_source_language
from .prompt_routing.mapping import (
    DEFAULT_INFERENCE_LEAF_PLUGIN_STATE,
    INFERENCE_LEAF_PLUGINS,
    InferenceLeafPlugin,
    resolve_inference_prompt_key,
    retrieval_to_prompt_type,
)

# ── Backward-compat aliases (loaded from .md files) ──
# These were inline strings until the provider refactor moved them to src/prompts/inference/*.md.
# Benchmark/eval code still imports them by name.

def _lazy_prompt(name: str) -> str:
    path = _Path(__file__).parent / "prompts" / "inference" / f"{name}.md"
    return path.read_text(encoding="utf-8")

INF_PROMPT = _lazy_prompt("lookup")
INF_PROMPT_TEMP = _lazy_prompt("temporal")
INF_PROMPT_ADV = _lazy_prompt("lookup")
INF_PROMPT_TEMPORAL = _lazy_prompt("temporal")
INF_PROMPT_TEMPORAL_NOTOOL = _lazy_prompt("temporal")
INF_PROMPT_COUNTING = _lazy_prompt("aggregate")
INF_PROMPT_SYNTHESIS = _lazy_prompt("synthesize")

# ── Tool definitions ──

TEMPORAL_TOOLS = [
    {
        "name": "date_diff",
        "description": "Compute exact difference between two dates. Always use this for duration questions — never compute in your head.",
        "input_schema": {
            "type": "object",
            "properties": {
                "date1": {"type": "string", "description": "First date — ISO (2021-03-15) or natural language (March 2021, summer 2023, early 2022)"},
                "date2": {"type": "string", "description": "Second date — same formats"},
                "unit":  {"type": "string", "enum": ["days", "weeks", "months", "years"]}
            },
            "required": ["date1", "date2", "unit"]
        }
    }
]

COUNTING_TOOLS = [
    {
        "name": "count_items",
        "description": "Count a list of items exactly. Always use this for counting questions — never count in your head.",
        "input_schema": {
            "type": "object",
            "properties": {
                "items": {"type": "array", "items": {"type": "string"}, "description": "List of items to count"}
            },
            "required": ["items"]
        }
    }
]

# ── get_more_context tool (Unit 9, run_23p.py lines 248-284) ──

GET_CONTEXT_TOOL = {
    "name": "get_more_context",
    "description": "Retrieve full raw text of a specific session. Use ONLY if facts and raw context don't contain enough detail.",
    "input_schema": {
        "type": "object",
        "properties": {
            "session_id": {"type": "integer", "description": "Session number (e.g. 12 for S12)"}
        },
        "required": ["session_id"]
    }
}


def get_more_context(session_id: int, raw_sessions: list = None) -> dict:
    """Return full text of a session. Truncated at 15K chars. (Unit 9, run_23p.py)"""
    if raw_sessions is None:
        raw_sessions = []
    if 0 < session_id <= len(raw_sessions):
        rs = raw_sessions[session_id - 1]
        # Check raw_session visibility (status field)
        if isinstance(rs, dict) and rs.get("status", "active") != "active":
            return {"result": f"Session {session_id} not found."}
        # Check TTL expiry on raw session
        if isinstance(rs, dict) and rs.get("retention_ttl") is not None:
            _now = _dt.now(_tz.utc)
            created = rs.get("stored_at") or rs.get("created_at")
            if created:
                try:
                    created_dt = _dt.fromisoformat(created.replace("Z", "+00:00"))
                    elapsed = (_now - created_dt).total_seconds()
                    if elapsed > rs["retention_ttl"]:
                        return {"result": f"Session {session_id} not found."}
                except (ValueError, TypeError):
                    pass
        # Reference mode degradation (Unit 12): if storage_mode is "reference"
        # and content is empty, return a degradation message.
        if isinstance(rs, dict):
            mode = rs.get("storage_mode", "inline")
            semantic_ready = rs.get("semantic_ready")
            canonical_text = str(rs.get("canonical_en") or "")
            if isinstance(semantic_ready, bool):
                text = canonical_text if semantic_ready else ""
            else:
                text = canonical_text or (
                    rs.get("content", "")
                    if detect_source_language(str(rs.get("content") or "")) in {"en", "und"}
                    else ""
                )
            if semantic_ready is False and not text.strip():
                return {"result": f"Session {session_id} semantic context unavailable "
                        f"(English canonicalization failed)."}
            if mode == "reference" and not text.strip():
                return {"result": f"Session {session_id} content unavailable "
                        f"(reference mode — original source not inline)."}
        else:
            text = str(rs)
        if len(text) > 15000:
            text = text[:15000] + "\n[...truncated]"
        return {"result": f"Full text of Session {session_id}:\n{text}"}
    return {"result": f"Session {session_id} not found."}


# ── Tool execution ──

TOOL_REGISTRY: dict[str, Callable[..., dict[str, Any]]] = {
    "date_diff":   date_diff,
    "count_items": count_items,
    "get_more_context": get_more_context,
}


def execute_tool(tool_name: str, tool_input: dict, context: dict = None) -> str:
    """Execute a tool call and return result as JSON string.

    context: extra kwargs injected by the caller (not from model).
    Used for get_more_context which needs raw_sessions from MemoryServer.
    """
    fn = TOOL_REGISTRY.get(tool_name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
    merged = {**tool_input}
    if context:
        merged.update(context)
    result = fn(**merged)  # type: ignore[operator]
    return json.dumps(result)


async def call_inference_with_tools(
    client,
    model: str,
    prompt: str,
    context: str,
    question: str,
    tools: list,
    max_tool_rounds: int = 2,
    reference_date: str = "2023-01-01",
    tool_context: dict = None,
    max_tokens: int = 512,
    return_metadata: bool = False,
    format_kwargs: dict = None,
) -> str | dict:
    """Run inference with tool round-trip support.

    If model issues tool_use → execute locally → send tool_result → get final answer.
    Max 2 tool rounds per question (safety limit).

    return_metadata: if True, return dict with "answer", "tool_called", "tool_results".
    format_kwargs: extra format kwargs for prompt (merged with context/question/reference_date).
    """
    fmt = {"context": context, "question": question, "reference_date": reference_date}
    if format_kwargs:
        fmt.update(format_kwargs)
    formatted_prompt = prompt.format(**fmt)

    # Non-Anthropic models: fall back to call_oai (no tool-use)
    if not model.startswith("anthropic/") and not hasattr(client, "messages"):
        answer = await call_oai(model, formatted_prompt, max_tokens=max_tokens)
        if return_metadata:
            return {"answer": answer, "tool_called": False, "tool_results": []}
        return answer

    messages: list[dict[str, Any]] = [{"role": "user", "content": formatted_prompt}]

    text_blocks = []
    tool_called = False
    all_tool_results = []
    for _ in range(max_tool_rounds + 1):
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            tools=tools,
            messages=messages,
        )

        tool_uses = [b for b in response.content if b.type == "tool_use"]
        text_blocks = [b for b in response.content if b.type == "text"]

        if not tool_uses:
            break

        # Execute all tool calls
        tool_called = True
        tool_results = []
        for tu in tool_uses:
            result_str = execute_tool(tu.name, tu.input, context=tool_context)
            all_tool_results.append({"tool": tu.name, "input": tu.input, "result": result_str})
            # Audit raw-text access via tool_context
            if tu.name == "get_more_context" and tool_context and tool_context.get("audit"):
                tool_context["audit"].log("get_more_context",
                    caller_id=tool_context.get("caller_id", "unknown"),
                    details={"session_id": tu.input.get("session_id")})
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": result_str,
            })

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})  # type: ignore[dict-item]

    answer = text_blocks[0].text.strip() if text_blocks else ""
    if return_metadata:
        return {"answer": answer, "tool_called": tool_called, "tool_results": all_tool_results}
    return answer


# ── Context building (for sprint-9 style artifact retrieval) ──

def build_context(top5, art_map):
    """Build context string from top-5 retrieval results.

    Args:
        top5: list of {"id": ..., "s": ...}
        art_map: dict mapping id -> artifact dict
    """
    parts = []
    for i, t in enumerate(top5, 1):
        art = art_map.get(t["id"])
        if art:
            date_str = art.get("updated_at", "")[:10]
            body = art.get("body", art.get("summary", ""))
            parts.append(f"[{i}] ({date_str}) {body}")
    return "\n\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
# Query-type-adaptive inference prompts (from sprint 23d)
# ═══════════════════════════════════════════════════════════════════════════
# All 8 query types loaded from src/prompts/inference/*.md.
# Each prompt expects {context} and {question} format placeholders.

_INF_PROMPT_DIR = _Path(__file__).parent / "prompts" / "inference"

_INF_PROMPT_TYPES = [
    "lookup", "temporal", "aggregate", "current",
    "synthesize", "procedural", "prospective", "summarize", "icl",
    "hybrid", "tool", "summarize_with_metadata", "list_set", "slot_query", "compositional",
    "codebase", "codebase_mixed", "code_slot", "code_chain", "risk_review", "container_exact_copy",
]


def _load_inf_prompt(name: str) -> str:
    """Load inference prompt from .md file."""
    path = _INF_PROMPT_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(
            f"Inference prompt '{name}' not found at {path}. "
            f"Expected src/prompts/inference/{name}.md"
        )
    return path.read_text(encoding="utf-8")


INF_PROMPTS = {name: _load_inf_prompt(name) for name in _INF_PROMPT_TYPES}

def get_inf_prompt(query_type: str) -> str:
    """Return the inference prompt for a given query type.

    Maps retrieval query_type names to inference prompt names, then
    falls back to the 'lookup' prompt for unknown query types.
    """
    inf_name = retrieval_to_prompt_type(query_type)
    return INF_PROMPTS.get(inf_name, INF_PROMPTS["lookup"])
