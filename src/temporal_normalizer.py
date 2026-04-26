#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from typing import Any

from .temporal import add_event, empty_temporal_index, finalize_temporal_index

ORDINAL_PATTERNS = {
    "step": re.compile(r"\bstep\s+(\d+)\b", re.I),
    "turn": re.compile(r"\bturn\s+(\d+)\b", re.I),
    "message": re.compile(r"\bmessage\s+(\d+)\b", re.I),
}
ISO_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
TEXT_DATE_RE = re.compile(
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
    r"\s+\d{1,2},?\s+\d{4}\b",
    re.I,
)
YEAR_RE = re.compile(r"\bin\s+(\d{4})\b", re.I)
AGO_RE = re.compile(r"\b(\d+)\s+(years?|months?|weeks?)\s+ago\b", re.I)
LAST_RE = re.compile(r"\blast\s+(week|month|year)\b", re.I)
FOR_RE = re.compile(r"\bfor\s+(\d+)\s+(years?|months?|weeks?)\b", re.I)
ORDINAL_HEADER_RE = re.compile(r"^\s*\[?(step|turn|message)\s+\d+\]?:?\s*", re.I)
ACTION_RE = re.compile(
    r"(?:^|\n)Action:\s*(.+?)(?=(?:\nObservation:|\n[A-Z][A-Za-z _-]+:|\Z))",
    re.S,
)
OBSERVATION_RE = re.compile(
    r"(?:^|\n)Observation:\s*(.+?)(?=(?:\n[A-Z][A-Za-z _-]+:|\Z))",
    re.S,
)
PATH_RE = re.compile(r"(?:/[A-Za-z0-9._@%+=:,\\-]+)+")
ID_RE = re.compile(r"\b\d{4,}\b")
RESULT_BLOCK_RE = re.compile(r"^\s*```", re.S)
EXECUTED_RESULTS_RE = re.compile(r"^\s*>?\s*Executed Results?:", re.I | re.M)


def _source_span(provenance: dict | None) -> dict | None:
    if not isinstance(provenance, dict):
        return None
    if isinstance(provenance.get("start_char"), int) and isinstance(provenance.get("end_char"), int):
        span: dict[str, int | str] = {
            "start_char": int(provenance["start_char"]),
            "end_char": int(provenance["end_char"]),
        }
        source_field = str(provenance.get("source_field") or "").strip()
        episode_id = str(provenance.get("episode_id") or "").strip()
        if source_field:
            span["source_field"] = source_field
        if episode_id:
            span["episode_id"] = episode_id
        return span
    raw_span = provenance.get("raw_span")
    if isinstance(raw_span, (list, tuple)) and len(raw_span) == 2:
        try:
            raw_span_dict: dict[str, int | str] = {
                "start_char": int(raw_span[0]),
                "end_char": int(raw_span[1]),
            }
            source_field = str(provenance.get("source_field") or "").strip()
            episode_id = str(provenance.get("episode_id") or "").strip()
            if source_field:
                raw_span_dict["source_field"] = source_field
            if episode_id:
                raw_span_dict["episode_id"] = episode_id
            return raw_span_dict
        except Exception:
            return None
    return None


def _parse_anchor_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        pass
    for pattern in (
        "%I:%M %p on %d %B, %Y",
        "%I:%M %p on %d %b, %Y",
        "%H:%M on %d %B, %Y",
        "%H:%M on %d %b, %Y",
        "%d %B %Y",
        "%d %b %Y",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(text, pattern)
        except Exception:
            continue
    return None


def _iso_date(dt: datetime) -> str:
    return dt.date().isoformat()


def _approx_shift(anchor: datetime, amount: int, unit: str) -> datetime:
    unit = unit.lower()
    if unit.startswith("week"):
        return anchor - timedelta(weeks=amount)
    if unit.startswith("month"):
        return anchor - timedelta(days=30 * amount)
    return anchor - timedelta(days=365 * amount)


def _calendar_payload(
    text: str,
    timestamp: str | None,
) -> tuple[str | None, str | None, str | None, str | None, str | None]:
    lower = text.lower()
    match = ISO_DATE_RE.search(text)
    if match:
        date_str = match.group(1)
        return (date_str, "point", date_str, date_str, "day")
    match = TEXT_DATE_RE.search(text)
    if match:
        normalized = match.group(0).title()
        for pattern in ("%B %d, %Y", "%B %d %Y", "%b %d, %Y", "%b %d %Y"):
            try:
                dt = datetime.strptime(normalized, pattern)
                date_str = _iso_date(dt)
                return (match.group(0), "point", date_str, date_str, "day")
            except Exception:
                continue
    match = YEAR_RE.search(lower)
    if match:
        year = int(match.group(1))
        return (match.group(0), "interval", f"{year:04d}-01-01", f"{year:04d}-12-31", "year")
    anchor = _parse_anchor_datetime(timestamp)
    match = AGO_RE.search(lower)
    if match and anchor is not None:
        amount = int(match.group(1))
        unit = match.group(2)
        resolved = _approx_shift(anchor, amount, unit)
        date_str = _iso_date(resolved)
        granularity = "year" if unit.lower().startswith("year") else "month" if unit.lower().startswith("month") else "day"
        return (match.group(0), "point", date_str, date_str, granularity)
    match = LAST_RE.search(lower)
    if match and anchor is not None:
        unit = match.group(1)
        if unit == "week":
            end = anchor - timedelta(days=1)
            start = end - timedelta(days=6)
            granularity = "day"
        elif unit == "month":
            end = anchor - timedelta(days=1)
            start = end - timedelta(days=29)
            granularity = "month"
        else:
            end = anchor - timedelta(days=1)
            start = end - timedelta(days=364)
            granularity = "year"
        return (match.group(0), "interval", _iso_date(start), _iso_date(end), granularity)
    match = FOR_RE.search(lower)
    if match and anchor is not None:
        amount = int(match.group(1))
        unit = match.group(2)
        start = _approx_shift(anchor, amount, unit)
        end = anchor
        granularity = "year" if unit.lower().startswith("year") else "month" if unit.lower().startswith("month") else "day"
        return (match.group(0), "duration", _iso_date(start), _iso_date(end), granularity)
    return (None, None, None, None, None)


def _extract_ordinal_matches(text: str) -> list[tuple[int, int, str, int]]:
    matches: list[tuple[int, int, str, int]] = []
    seen: set[tuple[int, int, str, int]] = set()
    for kind, pattern in ORDINAL_PATTERNS.items():
        for match in pattern.finditer(text):
            record = (match.start(), match.end(), kind, int(match.group(1)))
            if record in seen:
                continue
            seen.add(record)
            matches.append(record)
    matches.sort()
    return matches


def _ordinal_marker_position(text: str, start: int, end: int) -> str:
    leading = text[:start].strip("[](): \t\r\n")
    trailing = text[end:].strip("[](): \t\r\n")
    if not leading:
        return "prefix"
    if not trailing:
        return "suffix"
    return "embedded"


def _shift_source_span(
    provenance: dict | None,
    *,
    local_start: int,
    local_end: int,
) -> dict | None:
    span = _source_span(provenance)
    if not span:
        return None
    start_char = span.get("start_char")
    end_char = span.get("end_char")
    if not isinstance(start_char, int) or not isinstance(end_char, int):
        return span
    base_len = max(0, end_char - start_char)
    if base_len <= 0:
        return span
    shifted = dict(span)
    shifted["start_char"] = start_char + max(0, int(local_start))
    shifted["end_char"] = min(end_char, start_char + max(0, int(local_end)))
    return shifted


def _extract_paths(text: str) -> list[str]:
    seen: set[str] = set()
    paths: list[str] = []
    for match in PATH_RE.finditer(text):
        value = match.group(0).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        paths.append(value)
    return paths


def _extract_ids(text: str) -> list[str]:
    seen: set[str] = set()
    ids: list[str] = []
    for match in ID_RE.finditer(text):
        value = match.group(0).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        ids.append(value)
    return ids


def _coerce_tool_args(tool_name: str | None, raw: str | None) -> dict | None:
    raw = str(raw or "").strip()
    if not raw:
        return None
    if raw.startswith("{") or raw.startswith("["):
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = None
        if isinstance(parsed, dict):
            return parsed
    lower_tool = str(tool_name or "").strip().lower()
    if lower_tool == "execute_bash":
        return {"command": raw}
    if lower_tool == "execute_snowflake_sql":
        return {"sql": raw}
    return {"value": raw}


def extract_step_payload(block_text: str, base_payload: dict | None = None) -> dict:
    payload = dict(base_payload or {})
    body = ORDINAL_HEADER_RE.sub("", str(block_text or ""), count=1).strip()
    payload["raw_step_block"] = str(block_text or "").strip()
    if body:
        payload["step_body"] = body

    action_match = ACTION_RE.search(body)
    if action_match:
        action_raw = action_match.group(1).strip()
        payload["action_raw"] = action_raw
        tool_name = None
        tool_args_raw = None
        tool_match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)(?::\s*(.*))?$", action_raw, re.S)
        if tool_match:
            tool_name = tool_match.group(1).strip()
            tool_args_raw = str(tool_match.group(2) or "").strip()
            payload["action_name"] = tool_name
            payload["tool_name"] = tool_name
            if tool_args_raw:
                payload["tool_args_raw"] = tool_args_raw
                tool_args = _coerce_tool_args(tool_name, tool_args_raw)
                if tool_args:
                    payload["tool_args"] = tool_args
        else:
            payload["action_name"] = action_raw

    observation_match = OBSERVATION_RE.search(body)
    if observation_match:
        payload["observation_raw"] = observation_match.group(1).strip()

    combined = "\n".join(
        part
        for part in (
            str(payload.get("action_raw") or "").strip(),
            str(payload.get("observation_raw") or "").strip(),
            str(payload.get("step_body") or "").strip(),
        )
        if part
    )
    paths = _extract_paths(combined)
    ids = _extract_ids(combined)
    if paths:
        payload["paths"] = paths
    if ids:
        payload["ids"] = ids
    return payload


def _make_event(
    *,
    event_id: str,
    source_id: str,
    timeline_id: str,
    source_span: dict | None,
    kind: str | None,
    value: int | None,
    time_raw: str | None,
    time_kind: str | None,
    time_start: str | None,
    time_end: str | None,
    time_granularity: str | None,
    label: str,
    support_fact_ids: list[str],
    payload: dict,
    confidence: float,
    source_order: int,
) -> dict:
    return {
        "event_id": event_id,
        "source_id": source_id,
        "timeline_id": timeline_id,
        "source_span": source_span,
        "ordinal_kind": kind,
        "ordinal_index": value,
        "ordinal_start": value,
        "ordinal_end": value,
        "time_raw": time_raw,
        "time_kind": time_kind,
        "time_start": time_start,
        "time_end": time_end,
        "time_granularity": time_granularity,
        "label": label,
        "support_fact_ids": support_fact_ids,
        "payload": payload,
        "confidence": confidence,
        "_source_order": source_order,
    }


def _ordinal_sort_key(value: object) -> int:
    return int(value) if isinstance(value, int) else 10**9


def _payload_has_action_data(payload: dict | None) -> bool:
    if not isinstance(payload, dict):
        return False
    return any(
        str(payload.get(key) or "").strip()
        for key in ("action_raw", "tool_name", "tool_args_raw", "observation_raw", "step_body")
    )


def _looks_like_result_text(text: str) -> bool:
    value = str(text or "").strip()
    if not value:
        return False
    if ACTION_RE.search(value):
        return False
    if RESULT_BLOCK_RE.search(value):
        return True
    if EXECUTED_RESULTS_RE.search(value):
        return True
    if value.startswith("{") or value.startswith("["):
        return True
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    if len(lines) >= 2:
        structured = sum(
            1
            for line in lines[:4]
            if ("," in line or "\t" in line or "|" in line)
        )
        if structured >= 2:
            return True
    return False


def _impute_pre_marker_action_companions(index: dict, span_records: list[dict]) -> None:
    spans_by_timeline: dict[str, list[dict]] = {}
    for record in span_records:
        timeline_id = str(record.get("timeline_id") or "").strip()
        if not timeline_id:
            continue
        spans_by_timeline.setdefault(timeline_id, []).append(record)

    for timeline_id, spans in spans_by_timeline.items():
        spans.sort(key=lambda row: (row["source_order"], row["span_id"]))
        for idx in range(1, len(spans)):
            prev_span = spans[idx - 1]
            span = spans[idx]
            prev_explicit = list(prev_span.get("explicit_events") or [])
            current_explicit = list(span.get("explicit_events") or [])
            if not prev_explicit or not current_explicit:
                continue

            prev_event = prev_explicit[-1]
            current_event = current_explicit[0]
            kind = str(prev_event.get("ordinal_kind") or "").strip().lower()
            if kind not in {"step", "turn", "message"}:
                continue
            if kind != str(current_event.get("ordinal_kind") or "").strip().lower():
                continue

            prev_value = prev_event.get("ordinal_value")
            current_value = current_event.get("ordinal_value")
            if not isinstance(prev_value, int) or not isinstance(current_value, int):
                continue
            if current_value != prev_value + 1:
                continue

            prev_payload = prev_event.get("payload") or {}
            if _payload_has_action_data(prev_payload):
                continue

            first_start = current_event.get("local_start")
            if not isinstance(first_start, int) or first_start <= 0:
                continue
            pre_marker_text = str(span.get("text") or "")[:first_start].strip()
            if not pre_marker_text or not ACTION_RE.search(pre_marker_text):
                continue

            event_id = f"{span['span_id']}:imputed:{kind}:{prev_value}:boundary"
            payload = extract_step_payload(
                pre_marker_text,
                {
                    **(span.get("payload") or {}),
                    "ordinal_imputed": True,
                    "ordinal_marker_position": "imputed",
                    "imputed_from_event_id": str(prev_event.get("event_id") or ""),
                },
            )
            add_event(
                index,
                _make_event(
                    event_id=event_id,
                    source_id=str(span.get("source_id") or ""),
                    timeline_id=timeline_id,
                    source_span=_shift_source_span(
                        span.get("provenance"),
                        local_start=0,
                        local_end=first_start,
                    ),
                    kind=kind,
                    value=prev_value,
                    time_raw=None,
                    time_kind=None,
                    time_start=None,
                    time_end=None,
                    time_granularity=None,
                    label=f"{kind} {prev_value}",
                    support_fact_ids=list(span.get("support_fact_ids") or []),
                    payload=payload,
                    confidence=0.82,
                    source_order=int(span.get("source_order") or 0),
                ),
            )


def _impute_pre_marker_result_companions(index: dict, span_records: list[dict]) -> None:
    spans_by_timeline: dict[str, list[dict]] = {}
    for record in span_records:
        timeline_id = str(record.get("timeline_id") or "").strip()
        if not timeline_id:
            continue
        spans_by_timeline.setdefault(timeline_id, []).append(record)

    for timeline_id, spans in spans_by_timeline.items():
        spans.sort(key=lambda row: (row["source_order"], row["span_id"]))
        for idx in range(1, len(spans)):
            prev_span = spans[idx - 1]
            span = spans[idx]
            prev_explicit = list(prev_span.get("explicit_events") or [])
            current_explicit = list(span.get("explicit_events") or [])
            if not prev_explicit or not current_explicit:
                continue

            prev_event = prev_explicit[-1]
            current_event = current_explicit[0]
            kind = str(prev_event.get("ordinal_kind") or "").strip().lower()
            if kind not in {"step", "turn", "message"}:
                continue
            if kind != str(current_event.get("ordinal_kind") or "").strip().lower():
                continue

            prev_value = prev_event.get("ordinal_value")
            current_value = current_event.get("ordinal_value")
            if not isinstance(prev_value, int) or not isinstance(current_value, int):
                continue
            if current_value != prev_value + 1:
                continue

            first_start = current_event.get("local_start")
            if not isinstance(first_start, int) or first_start <= 0:
                continue
            pre_marker_text = str(span.get("text") or "")[:first_start].strip()
            if not _looks_like_result_text(pre_marker_text):
                continue

            event_id = f"{span['span_id']}:imputed:{kind}:{prev_value}:result"
            payload = extract_step_payload(
                pre_marker_text,
                {
                    **(span.get("payload") or {}),
                    "ordinal_imputed": True,
                    "ordinal_marker_position": "imputed_result",
                    "result_companion": True,
                    "imputed_from_event_id": str(prev_event.get("event_id") or ""),
                },
            )
            add_event(
                index,
                _make_event(
                    event_id=event_id,
                    source_id=str(span.get("source_id") or ""),
                    timeline_id=timeline_id,
                    source_span=_shift_source_span(
                        span.get("provenance"),
                        local_start=0,
                        local_end=first_start,
                    ),
                    kind=kind,
                    value=prev_value,
                    time_raw=None,
                    time_kind=None,
                    time_start=None,
                    time_end=None,
                    time_granularity=None,
                    label=f"{kind} {prev_value}",
                    support_fact_ids=list(span.get("support_fact_ids") or []),
                    payload=payload,
                    confidence=0.82,
                    source_order=int(span.get("source_order") or 0),
                ),
            )


def _impute_action_only_ordinal_events(index: dict, span_records: list[dict]) -> None:
    spans_by_timeline: dict[str, list[dict]] = {}
    for record in span_records:
        timeline_id = str(record.get("timeline_id") or "").strip()
        if not timeline_id:
            continue
        spans_by_timeline.setdefault(timeline_id, []).append(record)

    for timeline_id, spans in spans_by_timeline.items():
        spans.sort(key=lambda row: (row["source_order"], row["span_id"]))
        explicit_events: list[dict] = []
        for span in spans:
            for explicit in span.get("explicit_events", []):
                explicit_events.append(
                    {
                        **explicit,
                        "source_order": span["source_order"],
                        "span_id": span["span_id"],
                        "payload": explicit.get("payload") or {},
                    }
                )
        explicit_events.sort(
            key=lambda row: (
                row["source_order"],
                _ordinal_sort_key(row.get("ordinal_value")),
                str(row.get("event_id") or ""),
            )
        )
        if len(explicit_events) < 2:
            continue
        span_by_order = {span["source_order"]: span for span in spans}
        for idx in range(len(explicit_events) - 1):
            prev_event = explicit_events[idx]
            next_event = explicit_events[idx + 1]
            if prev_event.get("ordinal_kind") != next_event.get("ordinal_kind"):
                continue
            kind = str(prev_event.get("ordinal_kind") or "").strip().lower()
            if kind not in {"step", "turn", "message"}:
                continue
            prev_value = prev_event.get("ordinal_value")
            next_value = next_event.get("ordinal_value")
            if not isinstance(prev_value, int) or not isinstance(next_value, int) or next_value <= prev_value:
                continue
            action_spans: list[dict] = []
            for source_order in range(prev_event["source_order"] + 1, next_event["source_order"]):
                span_record = span_by_order.get(source_order)
                if not span_record:
                    continue
                if span_record.get("explicit_events"):
                    continue
                if not span_record.get("has_action"):
                    continue
                action_spans.append(span_record)
            if not action_spans:
                continue
            prev_payload = prev_event.get("payload") or {}
            start_offset = (
                0
                if prev_payload.get("ordinal_marker_position") == "suffix"
                or not _payload_has_action_data(prev_payload)
                else 1
            )
            candidate_values = [prev_value + start_offset + offset for offset in range(len(action_spans))]
            if any(value >= next_value for value in candidate_values):
                continue
            for span, value in zip(action_spans, candidate_values):
                payload = extract_step_payload(
                    span.get("text") or "",
                    {
                        **(span.get("payload") or {}),
                        "ordinal_imputed": True,
                        "ordinal_marker_position": "imputed",
                        "imputed_from_event_id": str(prev_event.get("event_id") or ""),
                    },
                )
                event_id = f"{span['span_id']}:imputed:{kind}:{value}"
                add_event(
                    index,
                    _make_event(
                        event_id=event_id,
                        source_id=str(span.get("source_id") or ""),
                        timeline_id=timeline_id,
                        source_span=_source_span(span.get("provenance")),
                        kind=kind,
                        value=value,
                        time_raw=None,
                        time_kind=None,
                        time_start=None,
                        time_end=None,
                        time_granularity=None,
                        label=f"{kind} {value}",
                        support_fact_ids=list(span.get("support_fact_ids") or []),
                        payload=payload,
                        confidence=0.8,
                        source_order=int(span.get("source_order") or 0),
                    ),
                )


def normalize_temporal_index(text_spans: list[dict]) -> dict:
    index = empty_temporal_index()
    span_records: list[dict] = []
    for source_order, span in enumerate(text_spans):
        text = str(span.get("text") or "")
        if not text:
            continue
        span_id = str(span.get("span_id") or f"span_{source_order:04d}")
        source_id = str(span.get("source_id") or "")
        timeline_id = str(span.get("timeline_id") or source_id or "timeline:unknown")
        timestamp = span.get("timestamp")
        support_fact_ids = [
            fact_id
            for fact_id in (span.get("support_fact_ids") or [])
            if isinstance(fact_id, str) and fact_id
        ]
        payload = dict(span.get("payload") or {})
        ordinal_matches = _extract_ordinal_matches(text)
        span_record: dict[str, Any] = {
            "span_id": span_id,
            "source_id": source_id,
            "timeline_id": timeline_id,
            "text": text,
            "timestamp": timestamp,
            "provenance": span.get("provenance"),
            "support_fact_ids": support_fact_ids,
            "payload": payload,
            "source_order": source_order,
            "explicit_events": [],
            "has_action": bool(ACTION_RE.search(text)),
        }
        for ordinal_pos, (_start, _end, kind, value) in enumerate(ordinal_matches):
            next_start = (
                ordinal_matches[ordinal_pos + 1][0]
                if ordinal_pos + 1 < len(ordinal_matches)
                else len(text)
            )
            marker_position = _ordinal_marker_position(text, _start, _end)
            if marker_position == "suffix":
                block_text = text[_start:_end].strip()
            else:
                block_text = text[_start:next_start].strip()
            event_payload = extract_step_payload(block_text, payload)
            event_payload["ordinal_marker_position"] = marker_position
            event_id = f"{span_id}:{kind}:{value}:{ordinal_pos}"
            event = _make_event(
                event_id=event_id,
                source_id=source_id,
                timeline_id=timeline_id,
                source_span=_shift_source_span(
                    span.get("provenance"),
                    local_start=_start,
                    local_end=next_start,
                ),
                kind=kind,
                value=value,
                time_raw=None,
                time_kind=None,
                time_start=None,
                time_end=None,
                time_granularity=None,
                label=f"{kind} {value}",
                support_fact_ids=support_fact_ids,
                payload=event_payload,
                confidence=1.0,
                source_order=source_order,
            )
            add_event(index, event)
            span_record["explicit_events"].append(
                {
                    "event_id": event_id,
                    "ordinal_kind": kind,
                    "ordinal_value": value,
                    "local_start": _start,
                    "local_end": _end,
                    "payload": event_payload,
                }
            )
        span_records.append(span_record)
        if ordinal_matches:
            continue
        time_raw, time_kind, time_start, time_end, time_granularity = _calendar_payload(text, timestamp)
        if not time_start:
            continue
        event_id = f"{span_id}:time"
        add_event(
            index,
            _make_event(
                event_id=event_id,
                source_id=source_id,
                timeline_id=timeline_id,
                source_span=_source_span(span.get("provenance")),
                kind=None,
                value=None,
                time_raw=time_raw,
                time_kind=time_kind,
                time_start=time_start,
                time_end=time_end,
                time_granularity=time_granularity,
                label=time_raw or time_start,
                support_fact_ids=support_fact_ids,
                payload=payload,
                confidence=1.0,
                source_order=source_order,
            ),
        )
    _impute_pre_marker_action_companions(index, span_records)
    _impute_pre_marker_result_companions(index, span_records)
    _impute_action_only_ordinal_events(index, span_records)
    return finalize_temporal_index(index)
