#!/usr/bin/env python3
# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

from __future__ import annotations

from typing import Any

from .codebase_semantic_sidecars import CodebaseSemanticSidecarStore
from .query_lexicon import (
    CODE_EVIDENCE_QUERY_MARKERS as _CODE_EVIDENCE_MARKERS,
)
from .query_lexicon import (
    HYDRATION_QUERY_MARKERS as _HYDRATION_QUERY_MARKERS,
)
from .query_lexicon import (
    PROSE_EVIDENCE_QUERY_MARKERS as _PROSE_EVIDENCE_MARKERS,
)

_WHOLE_FILE_MAX_LINES = 2000
_WINDOW_RADIUS = 250
_MAX_HIGHLIGHT_SPANS = 3


def query_requests_precise_code(query: str) -> bool:
    lowered = str(query or "").lower()
    return any(marker in lowered for marker in _HYDRATION_QUERY_MARKERS)


def query_requests_code_evidence(query: str) -> bool:
    lowered = str(query or "").lower()
    return any(marker in lowered for marker in _CODE_EVIDENCE_MARKERS)


def query_requests_mixed_code_plus_prose(query: str) -> bool:
    lowered = str(query or "").lower()
    return query_requests_code_evidence(lowered) and any(marker in lowered for marker in _PROSE_EVIDENCE_MARKERS)


def classify_codebase_query_mode(query: str) -> str:
    if query_requests_mixed_code_plus_prose(query):
        return "mixed_code_plus_prose"
    if query_requests_precise_code(query):
        return "precise_code"
    return "non_code"


def _fact_source_family(fact: dict[str, Any]) -> str:
    return str(fact.get("source_family") or ((fact.get("metadata") or {}).get("source_family") or "")).lower()


def _fact_semantic_kind(fact: dict[str, Any]) -> str:
    return str(fact.get("semantic_kind") or ((fact.get("metadata") or {}).get("semantic_kind") or "")).lower()


def _fact_semantic_type(fact: dict[str, Any]) -> str:
    return str(fact.get("semantic_type") or ((fact.get("metadata") or {}).get("semantic_type") or "")).lower()


def _fact_file_path(fact: dict[str, Any]) -> str:
    return str(fact.get("file_path") or ((fact.get("metadata") or {}).get("file_path") or "")).strip()


def _fact_span(fact: dict[str, Any]) -> dict[str, int | None] | None:
    span = fact.get("span")
    if not isinstance(span, dict):
        return None
    start_line = span.get("start_line")
    end_line = span.get("end_line")
    if not isinstance(start_line, int) or not isinstance(end_line, int):
        return None
    return {
        "start_line": start_line,
        "end_line": end_line,
        "start_col": span.get("start_col") if isinstance(span.get("start_col"), int) else None,
        "end_col": span.get("end_col") if isinstance(span.get("end_col"), int) else None,
    }


def _span_label(span: dict[str, int | None]) -> str:
    start_line = span.get("start_line")
    end_line = span.get("end_line")
    if start_line == end_line:
        return f"L{start_line}"
    return f"L{start_line}-L{end_line}"


def _span_key(span: dict[str, int | None]) -> tuple[int, int, int | None, int | None]:
    return (
        int(span.get("start_line") or 0),
        int(span.get("end_line") or 0),
        span.get("start_col") if isinstance(span.get("start_col"), int) else None,
        span.get("end_col") if isinstance(span.get("end_col"), int) else None,
    )


def _string_contains(query_lower: str, candidate: Any) -> bool:
    value = str(candidate or "").strip().lower()
    return bool(value) and value in query_lower


def _fact_match_bonus(query_lower: str, fact: dict[str, Any]) -> float:
    payload = fact.get("semantic_payload") or {}
    file_path = _fact_file_path(fact).lower()
    semantic_kind = _fact_semantic_kind(fact)
    semantic_type = _fact_semantic_type(fact)
    bonus = 0.0
    if semantic_kind == "object" and semantic_type != "import":
        if _string_contains(query_lower, payload.get("qualified_name")):
            bonus += 20.0
        if _string_contains(query_lower, payload.get("name")):
            bonus += 10.0
    if _string_contains(query_lower, payload.get("callee_name")):
        bonus += 4.0
    if semantic_kind == "object" and semantic_type == "import" and _string_contains(query_lower, payload.get("import_path")):
        bonus += 4.0
    if file_path and file_path in query_lower:
        bonus += 8.0
    if file_path:
        file_name = file_path.rsplit("/", 1)[-1]
        if file_name and file_name in query_lower:
            bonus += 4.0
    return bonus


def _fact_is_defining_anchor(fact: dict[str, Any]) -> bool:
    semantic_kind = _fact_semantic_kind(fact)
    semantic_type = _fact_semantic_type(fact)
    if semantic_kind == "object":
        return semantic_type in {"callable", "class", "module", "field", "parameter", "test_case"}
    if semantic_kind == "relation":
        return semantic_type == "declares"
    return False


def _fact_priority_score(query_lower: str, fact: dict[str, Any], rank: int) -> float:
    semantic_kind = _fact_semantic_kind(fact)
    semantic_type = _fact_semantic_type(fact)
    score = max(0.0, 40.0 - float(rank))
    if isinstance(fact.get("file_sidecar_ref"), dict):
        score += 12.0
    if isinstance(fact.get("sidecar_ref"), dict):
        score += 6.0
    if semantic_kind == "object":
        score += {
            "module": 12.0,
            "callable": 20.0,
            "class": 16.0,
            "test_case": 14.0,
            "field": 6.0,
            "parameter": 6.0,
            "import": -10.0,
            "callsite": -12.0,
        }.get(semantic_type, 0.0)
    elif semantic_kind == "relation":
        score += {
            "declares": 8.0,
            "imports": -6.0,
            "calls": -2.0,
            "test_covers": 1.0,
        }.get(semantic_type, 0.0)
    score += _fact_match_bonus(query_lower, fact)
    if _fact_is_defining_anchor(fact):
        score += 6.0
    return score


def _highlight_spans(scored_facts: list[dict[str, Any]]) -> list[dict[str, int | None]]:
    candidate_spans: list[dict[str, int | None]] = []
    seen = set()
    for item in scored_facts:
        span = _fact_span(item["fact"])
        if span is None:
            continue
        key = _span_key(span)
        if key in seen:
            continue
        seen.add(key)
        candidate_spans.append(span)
        if len(candidate_spans) >= (_MAX_HIGHLIGHT_SPANS * 3):
            break
    if not candidate_spans:
        return []

    def _covers(outer: dict[str, int | None], inner: dict[str, int | None]) -> bool:
        return (
            int(outer.get("start_line") or 0) <= int(inner.get("start_line") or 0)
            and int(outer.get("end_line") or 0) >= int(inner.get("end_line") or 0)
        )

    filtered: list[dict[str, int | None]] = []
    for span in candidate_spans:
        if any(
            _covers(span, other)
            and _span_key(span) != _span_key(other)
            and (int(span.get("end_line") or 0) - int(span.get("start_line") or 0))
            > (int(other.get("end_line") or 0) - int(other.get("start_line") or 0))
            for other in candidate_spans
        ):
            continue
        filtered.append(span)
        if len(filtered) >= _MAX_HIGHLIGHT_SPANS:
            break
    return filtered or candidate_spans[:_MAX_HIGHLIGHT_SPANS]


def _format_source_lines(lines: list[str], *, start_line: int) -> str:
    return "\n".join(f"{idx:>4}: {line}" for idx, line in enumerate(lines, start=start_line))


def _merge_windows(windows: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not windows:
        return []
    merged = [windows[0]]
    for start, end in windows[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + 1:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _render_source_file_text(
    *,
    file_path: str,
    language: str,
    code: str,
    highlight_spans: list[dict[str, int | None]],
) -> tuple[str, str, int]:
    lines = code.splitlines()
    if not lines:
        lines = [""]
    line_count = len(lines)
    highlights_label = ", ".join(_span_label(span) for span in highlight_spans) if highlight_spans else "none"
    if line_count <= _WHOLE_FILE_MAX_LINES:
        header = (
            f"[File: {file_path}] [Language: {language}] [Mode: whole_file]\n"
            f"[Highlights: {highlights_label}]"
        )
        body = _format_source_lines(lines, start_line=1)
        return f"{header}\n{body}".rstrip(), "whole_file", line_count

    windows = []
    if highlight_spans:
        for span in highlight_spans:
            start_line = int(span.get("start_line") or 1)
            end_line = int(span.get("end_line") or start_line)
            windows.append((max(1, start_line - _WINDOW_RADIUS), min(line_count, end_line + _WINDOW_RADIUS)))
    else:
        windows.append((1, min(line_count, 500)))
    windows = _merge_windows(sorted(windows))
    rendered_windows: list[str] = []
    for window_start, window_end in windows:
        window_lines = lines[window_start - 1:window_end]
        rendered_windows.append(
            f"[Window: L{window_start}-L{window_end}]\n"
            + _format_source_lines(window_lines, start_line=window_start)
        )
    header = (
        f"[File: {file_path}] [Language: {language}] [Mode: windowed_file]\n"
        f"[Highlights: {highlights_label}]\n"
        f"[Truncation: large_file_window]"
    )
    return f"{header}\n" + "\n\n".join(rendered_windows), "windowed_file", line_count


def _needs_code_hydration(query: str, retrieved_facts: list[dict[str, Any]]) -> bool:
    if not retrieved_facts:
        return False
    return classify_codebase_query_mode(query) in {"precise_code", "mixed_code_plus_prose"}

def promote_defining_code_facts(
    *,
    query: str,
    retrieved_facts: list[dict[str, Any]],
    candidate_facts: list[dict[str, Any]],
    limit: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    mode = classify_codebase_query_mode(query)
    if mode != "precise_code":
        return list(retrieved_facts), {"mode": "inactive", "reason": "query_not_precise_code"}

    query_lower = str(query or "").lower()
    promoted: list[dict[str, Any]] = []
    for fact in candidate_facts:
        if _fact_source_family(fact) != "codebase":
            continue
        if not _fact_is_defining_anchor(fact):
            continue
        if _fact_match_bonus(query_lower, fact) <= 0.0:
            continue
        promoted.append(fact)

    if not promoted:
        return list(retrieved_facts), {"mode": "inactive", "reason": "no_defining_candidates"}

    promoted = sorted(
        promoted,
        key=lambda fact: (
            -_fact_priority_score(query_lower, fact, 0),
            str(fact.get("id") or ""),
        ),
    )

    merged: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    promoted_ids: list[str] = []
    for fact in [*promoted, *retrieved_facts]:
        fact_id = str(fact.get("id") or "")
        if fact_id and fact_id in seen_ids:
            continue
        if fact_id:
            seen_ids.add(fact_id)
        merged.append(fact)
        if fact in promoted[:limit] and fact_id:
            promoted_ids.append(fact_id)
        if len(merged) >= max(1, int(limit)):
            break
    return merged, {
        "mode": "promoted" if promoted_ids else "inactive",
        "promoted_fact_ids": promoted_ids[:limit],
    }


def augment_codebase_context(
    *,
    query: str,
    retrieved_facts: list[dict[str, Any]],
    data_dir: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    codebase_facts = [fact for fact in retrieved_facts if _fact_source_family(fact) == "codebase"]
    if not codebase_facts:
        return [], {"mode": "inactive", "reason": "no_codebase_facts"}
    if not _needs_code_hydration(query, codebase_facts):
        return [], {"mode": "hot_only", "reason": "query_not_code_hydration"}

    query_lower = str(query or "").lower()
    grouped: dict[str, dict[str, Any]] = {}
    for rank, fact in enumerate(codebase_facts):
        file_path = _fact_file_path(fact)
        if not file_path:
            continue
        score = _fact_priority_score(query_lower, fact, rank)
        entry = grouped.setdefault(
            file_path,
            {
                "file_path": file_path,
                "language": str(fact.get("language") or ((fact.get("metadata") or {}).get("language") or "code")).lower(),
                "file_sidecar_ref": None,
                "scored_facts": [],
                "best_rank": rank,
                "best_score": score,
            },
        )
        entry["scored_facts"].append({"fact": fact, "score": score, "rank": rank})
        entry["best_rank"] = min(entry["best_rank"], rank)
        entry["best_score"] = max(entry["best_score"], score)
        if entry["file_sidecar_ref"] is None and isinstance(fact.get("file_sidecar_ref"), dict):
            entry["file_sidecar_ref"] = fact["file_sidecar_ref"]

    if not grouped:
        return [], {"mode": "hot_only", "reason": "no_codebase_files"}

    candidates: list[dict[str, Any]] = []
    for entry in grouped.values():
        scored_facts = sorted(
            entry["scored_facts"],
            key=lambda row: (-float(row["score"]), int(row["rank"]), str(row["fact"].get("id") or "")),
        )
        aggregate_score = sum(max(0.0, float(row["score"])) for row in scored_facts[:4])
        defining_scores = [
            float(row["score"])
            for row in scored_facts
            if _fact_is_defining_anchor(row["fact"])
        ]
        candidates.append(
            {
                **entry,
                "scored_facts": scored_facts,
                "aggregate_score": aggregate_score,
                "best_defining_score": max(defining_scores) if defining_scores else float("-inf"),
                "defining_count": len(defining_scores),
            }
        )

    candidates.sort(
        key=lambda row: (
            -float(row["best_defining_score"]),
            -int(row["defining_count"]),
            -float(row["aggregate_score"]),
            -float(row["best_score"]),
            0 if isinstance(row.get("file_sidecar_ref"), dict) else 1,
            int(row["best_rank"]),
            str(row["file_path"]),
        )
    )
    selected = candidates[0]
    file_sidecar_ref = selected.get("file_sidecar_ref")
    if not isinstance(file_sidecar_ref, dict):
        return [], {
            "mode": "hot_only",
            "reason": "no_file_sidecars",
            "selected_file": selected["file_path"],
        }

    store = CodebaseSemanticSidecarStore(data_dir)
    sidecar_id = str(file_sidecar_ref.get("sidecar_id") or "")
    try:
        payload = store.hydrate_sidecar(file_sidecar_ref)
    except Exception as exc:
        return [], {
            "mode": "hot_only",
            "reason": "file_hydration_failed",
            "selected_file": selected["file_path"],
            "failed_sidecars": [{"sidecar_id": sidecar_id, "error": exc.__class__.__name__}],
        }

    code = str(payload.get("code") or "")
    if not code.strip():
        return [], {
            "mode": "hot_only",
            "reason": "empty_file_payload",
            "selected_file": selected["file_path"],
            "failed_sidecars": [{"sidecar_id": sidecar_id, "error": "EmptyPayload"}],
        }

    highlight_spans = _highlight_spans(selected["scored_facts"])
    rendered_text, mode, line_count = _render_source_file_text(
        file_path=str(payload.get("file_path") or selected["file_path"]),
        language=str(payload.get("language") or selected["language"] or "code"),
        code=code,
        highlight_spans=highlight_spans,
    )
    trace = {
        "mode": mode,
        "selected_file": str(payload.get("file_path") or selected["file_path"]),
        "line_count": line_count,
        "highlight_spans": highlight_spans,
        "file_sidecar_id": sidecar_id,
        "selected_fact_ids": [
            str(item["fact"].get("id") or "")
            for item in selected["scored_facts"][:4]
            if str(item["fact"].get("id") or "")
        ],
    }
    if mode == "windowed_file":
        trace["truncation_mode"] = "large_file_window"

    return [
        {
            "text": rendered_text,
            "rank": int(selected["best_rank"]),
            "source": "code",
            "file_path": str(payload.get("file_path") or selected["file_path"]),
            "file_sidecar_id": sidecar_id,
            "highlight_spans": highlight_spans,
            "mode": mode,
        }
    ], trace
