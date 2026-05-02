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

import re
from collections import defaultdict
from datetime import date, datetime
from typing import Any

from dateutil import parser as date_parser

from .temporal_planner import classify_temporal_query, resolve_calendar_query_interval

_MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "between",
    "by",
    "did",
    "do",
    "does",
    "during",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "in",
    "is",
    "it",
    "kind",
    "last",
    "many",
    "month",
    "of",
    "on",
    "or",
    "since",
    "the",
    "this",
    "time",
    "times",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "year",
}

_DATE_WORDS = set(_MONTHS) | {
    "today",
    "tonight",
    "tomorrow",
    "yesterday",
    "morning",
    "afternoon",
    "evening",
    "night",
    "week",
    "weeks",
    "month",
    "months",
    "year",
    "years",
}

_TOKEN_RE = re.compile(r"[a-z0-9_./:-]+", re.I)
_CAPITALIZED_SPAN_RE = re.compile(r"\b[A-Z][A-Za-z0-9]*(?:[\s_-]+[A-Z][A-Za-z0-9]*)*\b")
_PATH_OR_CODE_RE = re.compile(r"(?:/[A-Za-z0-9._@%+=:,\\-]+)+|\b[A-Z]+-\d+[A-Z0-9_-]*\b|\b[A-Za-z]+_\d+\b")
_MONTH_YEAR_RE = re.compile(
    r"\b("
    + "|".join(sorted(_MONTHS, key=len, reverse=True))
    + r")\s+(\d{4})\b",
    re.I,
)
_YEAR_RE = re.compile(r"\b(?:in|during|throughout|for)\s+(\d{4})\b", re.I)


def _parse_iso_date(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value).strip()
    if not text:
        return None
    iso_match = re.match(r"^(\d{4}-\d{2}-\d{2})(?:[T\s].*)?$", text)
    if iso_match:
        try:
            return datetime.fromisoformat(iso_match.group(1)).date().isoformat()
        except ValueError:
            return None
    month_name = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
    if not (
        re.search(r"\b\d{1,2}(?:st|nd|rd|th)?\s+" + month_name + r"\b.*\b\d{4}\b", text, re.I)
        or re.search(month_name + r"\s+\d{1,2}(?:st|nd|rd|th)?(?:,)?\s+\d{4}\b", text, re.I)
    ):
        return None
    try:
        return date_parser.parse(text, fuzzy=False).date().isoformat()
    except (TypeError, ValueError, OverflowError):
        return None


def _month_end(year: int, month: int) -> str:
    if month == 12:
        return f"{year:04d}-12-31"
    next_month = datetime(year, month + 1, 1)
    end = next_month.toordinal() - 1
    return date.fromordinal(end).isoformat()


def _fallback_query_interval(question: str) -> dict | None:
    text = str(question or "")
    match = _MONTH_YEAR_RE.search(text)
    if match:
        month = _MONTHS[match.group(1).lower()]
        year = int(match.group(2))
        return {
            "time_raw": match.group(0),
            "time_kind": "interval",
            "time_start": f"{year:04d}-{month:02d}-01",
            "time_end": _month_end(year, month),
            "time_granularity": "month",
            "source": "explicit",
            "confidence": 0.9,
        }
    match = _YEAR_RE.search(text)
    if match:
        year = int(match.group(1))
        return {
            "time_raw": match.group(0),
            "time_kind": "interval",
            "time_start": f"{year:04d}-01-01",
            "time_end": f"{year:04d}-12-31",
            "time_granularity": "year",
            "source": "explicit",
            "confidence": 0.8,
        }
    return None


def _query_interval(question: str, temporal_trace: dict | None) -> dict | None:
    temporal_trace = temporal_trace if isinstance(temporal_trace, dict) else {}
    for key in ("query_interval", "query"):
        value = temporal_trace.get(key)
        if isinstance(value, dict) and value.get("time_start") and value.get("time_end"):
            out = dict(value)
            out.setdefault("source", "calendar_executor")
            out.setdefault("confidence", 1.0)
            return out
    interval = resolve_calendar_query_interval(question)
    if interval:
        out = dict(interval)
        out.setdefault("source", "explicit")
        out.setdefault("confidence", 0.9)
        return out
    return _fallback_query_interval(question)


def _operator_class(question: str, interval: dict | None) -> str:
    text = str(question or "").lower()
    if "how long" in text or "duration" in text:
        return "duration_state"
    if "as of" in text:
        return "as_of_state"
    if re.search(r"\bhow many\b|\bcount\b|\bnumber of\b", text):
        return "count_completeness"
    if re.search(r"\bwhich\b|\bwhat\b|\blist\b", text) and re.search(r"\bcities\b|\bitems\b|\bevents\b|\bperformances\b|\bincidents\b", text):
        return "list_enumeration"
    if classify_temporal_query(question) == "ordinal":
        return "ordinal_interval"
    if not interval:
        return "unknown"
    granularity = str(interval.get("time_granularity") or "").lower()
    if granularity in {"month", "year"}:
        return "month_year_lookup"
    if str(interval.get("source") or "") == "relative":
        return "relative_interval"
    if interval.get("time_start") == interval.get("time_end"):
        return "direct_lookup"
    return "relative_interval" if str(interval.get("time_kind") or "") == "duration" else "month_year_lookup"


def _query_tokens(question: str) -> list[str]:
    tokens: list[str] = []
    seen: set[str] = set()
    for raw in _TOKEN_RE.findall(str(question or "").lower()):
        token = raw.strip("._:-/")
        if not token or token in _STOPWORDS or token in _DATE_WORDS:
            continue
        if token.isdigit() or len(token) <= 2:
            continue
        if token not in seen:
            tokens.append(token)
            seen.add(token)
    return tokens


def _named_slots(question: str) -> list[str]:
    slots: list[str] = []
    seen: set[str] = set()
    for quoted in re.findall(r"['\"]([^'\"]{2,80})['\"]", str(question or "")):
        key = quoted.lower().strip()
        if key and key not in seen:
            slots.append(key)
            seen.add(key)
    for match in _PATH_OR_CODE_RE.finditer(str(question or "")):
        key = match.group(0).lower().strip()
        if key and key not in seen:
            slots.append(key)
            seen.add(key)
    for match in _CAPITALIZED_SPAN_RE.finditer(str(question or "")):
        key = re.sub(r"\s+", " ", match.group(0)).lower().strip()
        if key in _DATE_WORDS or key in seen:
            continue
        slots.append(key)
        seen.add(key)
    return slots[:16]


def _event_episode_id(event: dict, fact_to_episode: dict[str, str]) -> str | None:
    payload_value = event.get("payload")
    payload = payload_value if isinstance(payload_value, dict) else {}
    for value in (event.get("episode_id"), payload.get("episode_id")):
        text = str(value or "").strip()
        if text:
            return text
    for fact_id in event.get("support_fact_ids") or []:
        ep_id = fact_to_episode.get(str(fact_id))
        if ep_id:
            return ep_id
    return None


def _event_fact_ids(event: dict) -> list[str]:
    fact_ids: list[str] = []
    for value in event.get("support_fact_ids") or []:
        text = str(value or "").strip()
        if text and text not in fact_ids:
            fact_ids.append(text)
    payload_value = event.get("payload")
    payload = payload_value if isinstance(payload_value, dict) else {}
    for value in payload.get("support_fact_ids") or []:
        text = str(value or "").strip()
        if text and text not in fact_ids:
            fact_ids.append(text)
    return fact_ids


def _fact_dates(facts: list[dict]) -> tuple[str | None, str | None, str]:
    event_date: str | None = None
    source_date: str | None = None
    provenance = "unknown"
    for fact in facts:
        metadata_value = fact.get("metadata")
        metadata = metadata_value if isinstance(metadata_value, dict) else {}
        raw_event = metadata.get("event_date") or fact.get("event_date")
        raw_source_fallback = metadata.get("source_date_fallback")
        raw_source = metadata.get("source_date") or fact.get("source_date")
        parsed_event = _parse_iso_date(raw_event)
        parsed_source_fallback = _parse_iso_date(raw_source_fallback)
        parsed_source = _parse_iso_date(raw_source)
        if parsed_event and event_date is None:
            event_date = parsed_event
            provenance = str(metadata.get("event_date_provenance") or "event_date")
        if parsed_source_fallback and source_date is None:
            source_date = parsed_source_fallback
            if event_date is None:
                provenance = "source_date_fallback"
        elif parsed_source and source_date is None:
            source_date = parsed_source
    return event_date, source_date, provenance


def _episode_text(ep: dict, facts: list[dict]) -> str:
    fact_text = " ".join(str(fact.get("fact") or "") for fact in facts[:8])
    return " ".join([
        str(ep.get("topic_key") or ""),
        str(ep.get("raw_text") or ""),
        fact_text,
    ])


def _date_relation(candidate_date: str | None, start: str | None, end: str | None, operator_class: str) -> str:
    if not candidate_date or not start or not end:
        return "unknown"
    if candidate_date < start:
        return "before_cutoff"
    if candidate_date > end:
        return "post_cutoff" if operator_class in {"as_of_state", "duration_state"} else "outside_range"
    return "inside_range"


def _candidate_sort_key(row: dict) -> tuple[int, int, str]:
    relation_rank = {
        "already_selected": 0,
        "available_not_selected": 1,
        "date_match_content_weak": 2,
        "post_cutoff": 3,
        "missing_required_support": 4,
        "unknown": 5,
    }
    return (
        relation_rank.get(str(row.get("selection_status") or ""), 9),
        0 if row.get("date_relation") == "inside_range" else 1,
        str(row.get("episode_id") or ""),
    )


def build_temporal_evidence_candidates_trace(
    *,
    question: str,
    packet: dict,
    episode_lookup: dict[str, dict],
    facts_by_episode: dict[str, list[dict]],
    temporal_index: dict | None = None,
    temporal_trace: dict | None = None,
    max_candidates: int = 20,
) -> dict | None:
    """Build bounded temporal evidence-candidate telemetry.

    Returns ``None`` for non-temporal queries to avoid runtime trace noise.
    """

    question_text = str(question or packet.get("question") or packet.get("query") or "")
    interval = _query_interval(question_text, temporal_trace)
    operator_class = _operator_class(question_text, interval)
    if operator_class == "unknown" and not interval:
        return None

    selected_ids = [str(ep_id) for ep_id in packet.get("retrieved_episode_ids") or [] if str(ep_id)]
    injected_ids = [str(ep_id) for ep_id in packet.get("actual_injected_episode_ids") or [] if str(ep_id)]
    fact_pool_ids = [str(ep_id) for ep_id in packet.get("fact_episode_ids") or [] if str(ep_id)]
    selected_set = set(selected_ids)
    injected_set = set(injected_ids)
    query_terms = _query_tokens(question_text)
    named_slots = _named_slots(question_text)
    fact_to_episode = {
        str(fact.get("id") or ""): ep_id
        for ep_id, facts in facts_by_episode.items()
        for fact in facts
        if str(fact.get("id") or "")
    }

    episode_sources: dict[str, set[str]] = defaultdict(set)
    index_events_by_episode: dict[str, list[dict]] = defaultdict(list)
    for ep_id in [*selected_ids, *injected_ids, *fact_pool_ids]:
        if ep_id in episode_lookup:
            episode_sources[ep_id].add("packet")

    start = str(interval.get("time_start") or "") if interval else ""
    end = str(interval.get("time_end") or "") if interval else ""
    if temporal_index and start and end:
        for event in (temporal_index.get("events") or {}).values():
            if not isinstance(event, dict):
                continue
            event_start = _parse_iso_date(event.get("time_start"))
            event_end = _parse_iso_date(event.get("time_end")) or event_start
            if not event_start:
                continue
            overlaps_range = bool(event_end and event_start <= end and event_end >= start)
            is_post_cutoff = operator_class in {"as_of_state", "duration_state"} and event_start > end
            if overlaps_range or is_post_cutoff:
                event_ep_id = _event_episode_id(event, fact_to_episode)
                if event_ep_id and event_ep_id in episode_lookup:
                    episode_sources[event_ep_id].add(
                        "temporal_index_post_cutoff" if is_post_cutoff and not overlaps_range else "temporal_index_overlap"
                    )
                    index_events_by_episode[event_ep_id].append({
                        "event_date": event_start,
                        "event_end": event_end,
                        "fact_ids": _event_fact_ids(event),
                    })

    candidates: list[dict] = []
    for ep_id, sources in episode_sources.items():
        ep = episode_lookup.get(ep_id) or {}
        facts = facts_by_episode.get(ep_id) or []
        event_date, fact_source_date, provenance = _fact_dates(facts)
        index_event_rows = index_events_by_episode.get(ep_id) or []
        index_event_date = next(
            (row.get("event_date") for row in index_event_rows if row.get("event_date")),
            None,
        )
        episode_source_date = _parse_iso_date(ep.get("source_date") or ep.get("timestamp"))
        source_date = fact_source_date or episode_source_date
        session_date = _parse_iso_date(ep.get("session_date"))
        if event_date is None and index_event_date is not None:
            event_date = index_event_date
            provenance = "event_date"
        candidate_date = event_date or source_date or session_date
        date_provenance = provenance
        if event_date:
            date_provenance = "event_date" if provenance == "unknown" else provenance
        elif source_date:
            date_provenance = "source_date_fallback" if provenance == "source_date_fallback" else "source_date"
        elif session_date:
            date_provenance = "session_date"
        text = _episode_text(ep, facts).lower()
        overlap_terms = [token for token in query_terms if token in text]
        named_overlap = [slot for slot in named_slots if slot in text]
        required_missing = [slot for slot in named_slots if slot not in text][:8]
        answer_support = "present" if named_overlap or len(overlap_terms) >= 2 else "absent" if query_terms or named_slots else "unknown"
        relation = _date_relation(candidate_date, start or None, end or None, operator_class)
        fact_ids = [str(fact.get("id") or "") for fact in facts if str(fact.get("id") or "")]
        for row in index_event_rows:
            for fact_id in row.get("fact_ids") or []:
                if fact_id not in fact_ids:
                    fact_ids.append(fact_id)
        selected = ep_id in selected_set or ep_id in injected_set
        if selected:
            status = "already_selected"
            reason = "selected_context_contains_candidate"
        elif relation == "post_cutoff":
            status = "post_cutoff"
            reason = "candidate_after_as_of_cutoff"
        elif relation == "inside_range" and answer_support == "present":
            status = "available_not_selected"
            reason = "date_and_content_match_available"
        elif relation == "inside_range":
            status = "date_match_content_weak"
            reason = "date_match_but_content_support_weak"
        elif required_missing:
            status = "missing_required_support"
            reason = "required_terms_missing"
        else:
            status = "unknown"
            reason = "candidate_date_relation_unknown_or_outside"
        candidates.append({
            "episode_id": ep_id,
            "fact_ids": fact_ids[:8],
            "event_date": event_date,
            "source_date": source_date,
            "session_date": session_date,
            "date_provenance": date_provenance,
            "date_relation": relation,
            "content_support": {
                "query_token_overlap": len(overlap_terms),
                "named_slot_overlap": len(named_overlap),
                "answer_object_support": answer_support,
                "required_terms_missing": required_missing,
            },
            "selection_status": status,
            "reason": reason,
            "source": sorted(sources),
        })

    candidates.sort(key=_candidate_sort_key)
    omitted_inside = [
        row["episode_id"]
        for row in candidates
        if row.get("date_relation") == "inside_range" and row.get("episode_id") not in injected_set
    ]
    warnings: list[str] = []
    if len(candidates) > max_candidates:
        warnings.append("candidate_rows_truncated")
    selected_inside = [
        row for row in candidates
        if row.get("date_relation") == "inside_range" and row.get("episode_id") in injected_set
    ]
    available_inside = [
        row for row in candidates
        if row.get("date_relation") == "inside_range" and row.get("episode_id") not in injected_set
    ]
    if operator_class in {"count_completeness", "list_enumeration"} and available_inside:
        warnings.append("temporal_candidate_pool_exceeds_selected_context")
    if operator_class in {"as_of_state", "duration_state"} and any(row.get("date_relation") == "post_cutoff" for row in candidates):
        warnings.append("post_cutoff_candidates_present")

    return {
        "trace_version": 1,
        "operator_class": operator_class,
        "query_range": {
            "start": start or None,
            "end": end or None,
            "source": (interval or {}).get("source") or ("explicit" if interval else "unknown"),
            "confidence": float((interval or {}).get("confidence") or (0.0 if not interval else 0.8)),
        },
        "candidate_count": len(candidates),
        "selected_episode_ids": selected_ids,
        "selected_inside_range_count": len(selected_inside),
        "available_inside_range_count": len(available_inside),
        "candidates": candidates[:max_candidates],
        "omitted_date_matching_episode_ids": omitted_inside[:max_candidates],
        "warnings": warnings,
    }
