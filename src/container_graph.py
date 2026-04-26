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

import hashlib
import json
import re
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

from .common import normalize_term_token
from .document_span import (
    ARTIFACT_MARKER_RE,
    BARE_ARTIFACT_MARKER_RE,
    assign_document_artifact_span_ids,
    extract_artifact_markers,
)

CONTAINER_GRAPH_KEYS = (
    "graph_revisions",
    "containers",
    "relations",
    "anchors",
    "evidence",
    "refs",
    "ref_lookup",
    "ref_ranges",
    "render_refs",
    "state",
    "contracts",
    "artifacts",
)

CONTAINER_GRAPH_SCHEMA_VERSION = "container-graph-v1"
DOCUMENT_GRAPH_ADAPTER = "document_artifact_container_graph"
DOCUMENT_GRAPH_ADAPTER_VERSION = "1"
REF_SCHEMA_VERSION = "container-ref-v1"
RENDER_REF_SCHEMA_VERSION = "container-render-ref-v1"
EXACT_COPY_RENDER_SOURCE = "raw_doc_marker_span"
DEGRADED_EPISODE_JOIN_RENDER_SOURCE = "episode_join_fallback"
CONTAINER_PROOF_SOURCE_FIELDS = (
    "traits_json.instruction_text",
    "render_ref_json.instruction_text",
    "render_ref_json.artifact_id",
)

_ARTIFACT_KEY_RE = re.compile(r"::artifact::(.+)$")
_RESPONSE_LABEL_RE = re.compile(r"(?im)^[ \t]*(?:response|answer|output)[ \t]*:[ \t]*")
_INSTRUCTION_LABEL_RE = re.compile(r"(?im)^[ \t]*(?:instruction|prompt|request|question)[ \t]*:[ \t]*")
_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'_-]*")

_STRUCTURAL_QUERY_STOP_TOKENS = {
    "the",
    "this",
    "that",
    "these",
    "those",
    "indexed",
    "index",
    "response",
    "answer",
    "text",
    "content",
    "include",
    "other",
    "your",
    "only",
    "prepend",
    "with",
    "from",
    "into",
}

_ORDINAL_TOKENS = {
    "first",
    "second",
    "third",
    "fourth",
    "fifth",
    "sixth",
    "seventh",
    "eighth",
    "ninth",
    "tenth",
    "last",
    "final",
    "penultimate",
}

_ORDINAL_PREFIX_RE = re.compile(
    r"^\s*(?:the\s+)?(?:\d+(?:st|nd|rd|th)?|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|last|final|penultimate)\b",
    flags=re.IGNORECASE,
)


def _normalized_token_variants(token: str) -> set[str]:
    raw = str(token or "").strip().lower()
    if not raw:
        return set()
    normalized = normalize_term_token(raw)
    variants = {raw}
    if normalized:
        variants.add(normalized)
        normalized_again = normalize_term_token(normalized)
        if normalized_again:
            variants.add(normalized_again)
    return {variant for variant in variants if variant}


def empty_container_graph() -> dict[str, list[dict]]:
    return {key: [] for key in CONTAINER_GRAPH_KEYS}


def normalize_container_graph(raw: Any) -> dict[str, list[dict]]:
    graph = empty_container_graph()
    if not isinstance(raw, dict):
        return graph
    for key in CONTAINER_GRAPH_KEYS:
        rows = raw.get(key) or []
        if isinstance(rows, list):
            graph[key] = [dict(row) for row in rows if isinstance(row, dict)]
    return graph


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_hash(*parts: Any, prefix: str = "cg") -> str:
    payload = _json_dumps(parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]
    return f"{prefix}_{digest}"


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_json_dumps(value).encode("utf-8")).hexdigest()


def _episode_sort_key(episode: dict) -> tuple[int, str]:
    episode_id = str(episode.get("episode_id") or "")
    match = re.search(r"_e(\d+)\b", episode_id)
    if match:
        return int(match.group(1)), episode_id
    session_num = episode.get("session_num")
    if isinstance(session_num, int) and session_num > 0:
        return session_num, episode_id
    return 10**9, episode_id


def _source_id_from_doc(doc: dict, episodes: list[dict]) -> str:
    for episode in episodes:
        source_id = str((episode or {}).get("source_id") or "").strip()
        if source_id:
            return source_id
    doc_id = str((doc or {}).get("doc_id") or "").strip()
    return doc_id.removeprefix("document:")


def _logical_source_id(source_id: str, record: dict | None) -> str:
    source_meta = dict((record or {}).get("source_meta") or {})
    return str(source_meta.get("logical_source_id") or source_id).strip() or source_id


def _revision_id(source_id: str, record: dict | None) -> str:
    return str((record or {}).get("version_id") or (record or {}).get("content_hash") or source_id).strip() or source_id


def _content_revision_id(record: dict | None) -> str | None:
    value = str((record or {}).get("content_hash") or "").strip()
    return value or None


def _artifact_key(span_id: str) -> str:
    match = _ARTIFACT_KEY_RE.search(str(span_id or ""))
    return match.group(1) if match else str(span_id or "")


def _iter_artifact_marker_matches(text: str) -> list[re.Match]:
    matches: list[re.Match] = list(ARTIFACT_MARKER_RE.finditer(text or ""))
    matches.extend(BARE_ARTIFACT_MARKER_RE.finditer(text or ""))
    return sorted(matches, key=lambda match: match.start())


def _artifact_end_before_next_marker(raw_doc: str, next_marker_start: int | None) -> int:
    if next_marker_start is None:
        return len(raw_doc)
    end = int(next_marker_start)
    # A marker delimiter commonly contributes one separator newline between
    # artifacts. Exclude that structural separator without stripping response
    # body edge formatting.
    if end >= 2 and raw_doc[end - 1] == "\n" and raw_doc[end - 2] == "\n":
        end -= 1
    elif end >= 4 and raw_doc[end - 2 : end] == "\r\n" and raw_doc[end - 4 : end - 2] == "\r\n":
        end -= 2
    return end


def _marker_value(match: re.Match) -> str:
    try:
        return str(match.group(1) or "").strip()
    except IndexError:
        return ""


def _normalize_marker(value: str) -> str:
    marker = re.sub(r"[^0-9A-Za-z._-]+", "_", str(value or "").strip()).strip("._-")
    return marker


def extract_artifact_text_from_raw_doc(
    raw_doc: str,
    *,
    artifact_span_id: str,
    fallback_text: str,
) -> tuple[str, int | None, int | None, str]:
    """Return exact artifact text and raw-doc range when a marker can resolve.

    Marker-based extraction is generic document-structure behavior. If the
    source has no resolvable artifact marker, the caller gets the episode-based
    fallback with an explicit provenance mode.
    """

    raw_doc = str(raw_doc or "")
    artifact_key = _normalize_marker(_artifact_key(artifact_span_id))
    if raw_doc and artifact_key:
        matches = _iter_artifact_marker_matches(raw_doc)
        for idx, match in enumerate(matches):
            if _normalize_marker(_marker_value(match)).lower() != artifact_key.lower():
                continue
            start = match.start()
            end = _artifact_end_before_next_marker(
                raw_doc,
                matches[idx + 1].start() if idx + 1 < len(matches) else None,
            )
            return raw_doc[start:end], start, end, EXACT_COPY_RENDER_SOURCE
    return str(fallback_text or ""), None, None, DEGRADED_EPISODE_JOIN_RENDER_SOURCE


def split_prompt_response_artifact(artifact_text: str) -> tuple[str, str]:
    """Split a prompt/response artifact into instruction and renderable body."""

    text = str(artifact_text or "")
    response_match = _RESPONSE_LABEL_RE.search(text)
    if not response_match:
        return "", text
    instruction_text = text[: response_match.start()]
    response_text = text[response_match.end() :]
    next_marker = _iter_artifact_marker_matches(response_text)
    if next_marker:
        response_text = response_text[: next_marker[0].start()]
    instruction_text = _INSTRUCTION_LABEL_RE.sub(" ", instruction_text)
    if response_text.startswith("\r\n"):
        response_text = response_text[2:]
    elif response_text[:1] in {"\n", "\r"}:
        response_text = response_text[1:]
    return instruction_text.strip(), response_text


def _raw_doc_artifact_specs(raw_doc: str, source_id: str) -> list[dict]:
    specs: list[dict] = []
    matches = _iter_artifact_marker_matches(raw_doc)
    for idx, match in enumerate(matches):
        artifact_key = _normalize_marker(_marker_value(match))
        if not artifact_key:
            continue
        start = match.start()
        end = _artifact_end_before_next_marker(raw_doc, matches[idx + 1].start() if idx + 1 < len(matches) else None)
        artifact_text = raw_doc[start:end]
        specs.append(
            {
                "span_id": f"{source_id}::artifact::{artifact_key}",
                "artifact_key": artifact_key,
                "artifact_text": artifact_text,
                "raw_start": start,
                "raw_end": end,
                "render_source": "raw_doc_marker_span",
            }
        )
    return specs


def _episode_marker_span_ids(episode: dict, source_id: str) -> list[str]:
    spans: list[str] = []
    for value in (
        episode.get("raw_original"),
        episode.get("raw_text"),
        (episode.get("metadata") or {}).get("source_section_path"),
        (episode.get("provenance") or {}).get("source_section_path"),
        episode.get("section_path"),
    ):
        text = str(value or "")
        if not text:
            continue
        for marker in extract_artifact_markers(text):
            span_id = f"{source_id}::artifact::{marker}"
            if span_id not in spans:
                spans.append(span_id)
    return spans


def _row_by_id(rows: list[dict], key: str) -> dict[str, dict]:
    return {str(row.get(key) or ""): row for row in rows if str(row.get(key) or "")}


def _preserve_non_document_graph_rows(existing_graph: dict | None) -> dict[str, list[dict]]:
    preserved = empty_container_graph()
    if not existing_graph:
        return preserved
    existing = normalize_container_graph(existing_graph)
    preserved_revision_ids = {
        str(row.get("container_graph_revision_id") or "")
        for row in existing["graph_revisions"]
        if str(row.get("adapter_name") or "") != DOCUMENT_GRAPH_ADAPTER
    }
    preserved_container_ids = {
        str(row.get("container_id") or "")
        for row in existing["containers"]
        if str(row.get("adapter_name") or "") != DOCUMENT_GRAPH_ADAPTER
        or str(row.get("container_graph_revision_id") or "") in preserved_revision_ids
    }
    preserved_render_ref_ids = {
        str(row.get("render_ref_id") or "")
        for row in existing["render_refs"]
        if str(row.get("container_id") or "") in preserved_container_ids
        or str(row.get("container_graph_revision_id") or "") in preserved_revision_ids
    }
    preserved_ref_ids = {
        str(row.get("ref_id") or "")
        for row in existing["refs"]
        if str(row.get("container_id") or "") in preserved_container_ids
        or str(row.get("container_graph_revision_id") or "") in preserved_revision_ids
    }

    def _row_revision_ids(row: dict) -> set[str]:
        ids = {str(row.get("container_graph_revision_id") or "")}
        ids.update(str(value or "") for value in row.get("container_graph_revision_ids_json") or [])
        return {value for value in ids if value}

    def _preserve_contract(row: dict) -> bool:
        subject_kind = str(row.get("subject_kind") or "").lower()
        if subject_kind in {"global", "profile", "operator", "system"}:
            return True
        if not str(row.get("subject_id") or "") and not str(row.get("container_id") or ""):
            return True
        return (
            str(row.get("container_id") or "") in preserved_container_ids
            or bool(_row_revision_ids(row) & preserved_revision_ids)
        )

    def _preserve_artifact(row: dict) -> bool:
        if str(row.get("container_id") or "") in preserved_container_ids:
            return True
        if str(row.get("subject_id") or "") in preserved_container_ids:
            return True
        if _row_revision_ids(row) & preserved_revision_ids:
            return True
        families = {str(value or "").lower() for value in row.get("families_json") or []}
        return bool(families and "document" not in families)

    preserved["graph_revisions"] = [
        row for row in existing["graph_revisions"]
        if str(row.get("container_graph_revision_id") or "") in preserved_revision_ids
    ]
    preserved["containers"] = [
        row for row in existing["containers"]
        if str(row.get("container_id") or "") in preserved_container_ids
    ]
    preserved["relations"] = [
        row for row in existing["relations"]
        if str(row.get("container_graph_revision_id") or "") in preserved_revision_ids
        or str(row.get("src_container_id") or "") in preserved_container_ids
        or str(row.get("dst_container_id") or "") in preserved_container_ids
    ]
    preserved["anchors"] = [
        row for row in existing["anchors"]
        if str(row.get("container_id") or "") in preserved_container_ids
        or str(row.get("container_graph_revision_id") or "") in preserved_revision_ids
    ]
    preserved["evidence"] = [
        row for row in existing["evidence"]
        if str(row.get("container_graph_revision_id") or "") in preserved_revision_ids
        or str(row.get("subject_id") or "") in preserved_container_ids
        or str(row.get("subject_id") or "") in preserved_render_ref_ids
    ]
    preserved["render_refs"] = [
        row for row in existing["render_refs"]
        if str(row.get("render_ref_id") or "") in preserved_render_ref_ids
    ]
    preserved["state"] = [
        row for row in existing["state"]
        if str(row.get("container_id") or "") in preserved_container_ids
        or str(row.get("container_graph_revision_id") or "") in preserved_revision_ids
    ]
    preserved["refs"] = [
        row for row in existing["refs"]
        if str(row.get("ref_id") or "") in preserved_ref_ids
    ]
    preserved["ref_lookup"] = [
        row for row in existing["ref_lookup"]
        if str(row.get("ref_id") or "") in preserved_ref_ids
    ]
    preserved["ref_ranges"] = [
        row for row in existing["ref_ranges"]
        if str(row.get("ref_id") or "") in preserved_ref_ids
    ]
    preserved["contracts"] = [
        row for row in existing["contracts"]
        if _preserve_contract(row)
    ]
    preserved["artifacts"] = [
        row for row in existing["artifacts"]
        if _preserve_artifact(row)
    ]
    return preserved


def _ref_row(
    *,
    graph_revision_id: str,
    container_id: str | None,
    ref_role: str,
    ref_type: str,
    ref_json: dict,
    source_id: str,
    logical_source_id: str,
    revision_id: str,
    coverage_required: bool = False,
) -> dict:
    ref_fingerprint = _fingerprint({"ref_type": ref_type, **ref_json})
    ref_id = stable_hash(
        REF_SCHEMA_VERSION,
        graph_revision_id,
        container_id or "unmapped",
        ref_role,
        ref_type,
        ref_fingerprint,
        prefix="ref",
    )
    return {
        "ref_id": ref_id,
        "container_graph_revision_id": graph_revision_id,
        "container_id": container_id,
        "ref_role": ref_role,
        "ref_type": ref_type,
        "ref_json": ref_json,
        "ref_fingerprint": ref_fingerprint,
        "source_id": source_id,
        "logical_source_id": logical_source_id,
        "revision_id": revision_id,
        "coverage_required": 1 if coverage_required else 0,
        "coverage_status": "mapped",
        "status": "active",
        "created_at": _utcnow_iso(),
    }


def _lookup_row(ref_id: str, lookup_ns: str, lookup_key: str, *, value_text: str | None = None, value_int: int | None = None) -> dict:
    value_hash = _fingerprint(
        {
            "lookup_ns": lookup_ns,
            "lookup_key": lookup_key,
            "value_text": value_text,
            "value_int": value_int,
        }
    )
    return {
        "lookup_id": stable_hash("container-ref-lookup-v1", ref_id, lookup_ns, lookup_key, value_hash, prefix="lookup"),
        "ref_id": ref_id,
        "lookup_ns": lookup_ns,
        "lookup_key": lookup_key,
        "value_text": value_text,
        "value_int": value_int,
        "value_hash": value_hash,
        "status": "active",
    }


def _range_row(ref_id: str, range_kind: str, range_scope: str, start_value: int, end_value: int) -> dict:
    return {
        "range_id": stable_hash(
            "container-ref-range-v1",
            ref_id,
            range_kind,
            range_scope,
            int(start_value),
            int(end_value),
            prefix="range",
        ),
        "ref_id": ref_id,
        "range_kind": range_kind,
        "range_scope": range_scope,
        "start_value": int(start_value),
        "end_value": int(end_value),
        "status": "active",
    }


def _render_ref_row(
    *,
    graph_revision_id: str,
    container_id: str,
    render_kind: str,
    render_mode: str,
    ref_json: dict,
    fidelity: str = "exact",
    language: str = "source",
    fmt: str = "text",
) -> dict:
    ref_fingerprint = _fingerprint({"render_kind": render_kind, "render_mode": render_mode, **ref_json})
    return {
        "render_ref_id": stable_hash(
            RENDER_REF_SCHEMA_VERSION,
            graph_revision_id,
            container_id,
            render_kind,
            render_mode,
            ref_fingerprint,
            prefix="render",
        ),
        "container_id": container_id,
        "container_graph_revision_id": graph_revision_id,
        "render_kind": render_kind,
        "render_mode": render_mode,
        "ref_json": ref_json,
        "ref_fingerprint": ref_fingerprint,
        "fidelity": fidelity,
        "language": language,
        "format": fmt,
        "token_estimate": max(1, len(str(ref_json.get("text") or "")) // 4),
        "status": "active",
        "created_at": _utcnow_iso(),
    }


def _evidence_row(
    *,
    graph_revision_id: str,
    subject_type: str,
    subject_id: str,
    evidence_kind: str,
    evidence_ref_json: dict,
    role: str,
    score_name: str | None = None,
    score: float | None = None,
    trace: dict | None = None,
) -> dict:
    evidence_id = stable_hash(
        "container-evidence-v1",
        subject_type,
        subject_id,
        evidence_kind,
        _fingerprint(evidence_ref_json),
        role,
        score_name or "",
        prefix="ev",
    )
    return {
        "evidence_id": evidence_id,
        "container_graph_revision_id": graph_revision_id,
        "subject_type": subject_type,
        "subject_id": subject_id,
        "evidence_kind": evidence_kind,
        "evidence_ref_json": evidence_ref_json,
        "role": role,
        "score_name": score_name,
        "score": score,
        "status": "active",
        "trace_json": trace or {},
        "created_at": _utcnow_iso(),
    }


def _relation_row(
    *,
    graph_revision_id: str,
    src_container_id: str,
    dst_container_id: str,
    relation_kind: str,
    order_key: dict | None = None,
) -> dict:
    return {
        "relation_id": stable_hash(
            "container-relation-v1",
            src_container_id,
            dst_container_id,
            "core",
            relation_kind,
            prefix="rel",
        ),
        "container_graph_revision_id": graph_revision_id,
        "src_container_id": src_container_id,
        "dst_container_id": dst_container_id,
        "src_container_graph_revision_id": graph_revision_id,
        "dst_container_graph_revision_id": graph_revision_id,
        "relation_ns": "core",
        "relation_kind": relation_kind,
        "order_key_json": order_key or {},
        "traits_json": {},
        "relation_score": 1.0,
        "status": "active",
        "created_at": _utcnow_iso(),
        "updated_at": _utcnow_iso(),
    }


def build_document_container_graph(
    *,
    raw_docs: dict[str, str],
    episode_corpus: dict,
    source_records: dict[str, dict],
    existing_graph: dict | None = None,
) -> dict[str, list[dict]]:
    """Build active document containers from current raw/episode substrate."""

    graph = _preserve_non_document_graph_rows(existing_graph)

    graph_revision_ids_seen: set[str] = set()
    doc_groups: list[dict[str, Any]] = []
    for docs in (episode_corpus or {}).values():
        if isinstance(docs, list):
            doc_groups.extend(doc for doc in docs if isinstance(doc, dict))

    for doc in doc_groups:
        episodes = [dict(ep) for ep in (doc.get("episodes") or []) if isinstance(ep, dict)]
        if not episodes:
            continue
        episodes.sort(key=_episode_sort_key)
        source_id = _source_id_from_doc(doc, episodes)
        if not source_id:
            continue
        record = source_records.get(source_id) or {}
        if str(record.get("family") or "document") != "document":
            continue
        logical_source_id = _logical_source_id(source_id, record)
        revision_id = _revision_id(source_id, record)
        content_revision_id = _content_revision_id(record)
        input_fingerprint = _fingerprint(
            {
                "source_id": source_id,
                "revision_id": revision_id,
                "raw_hash": hashlib.sha256(str((raw_docs or {}).get(source_id) or "").encode("utf-8")).hexdigest(),
                "episode_ids": [ep.get("episode_id") for ep in episodes],
            }
        )
        graph_revision_id = stable_hash(
            "container-graph-revision-v1",
            source_id,
            revision_id,
            "document",
            DOCUMENT_GRAPH_ADAPTER,
            DOCUMENT_GRAPH_ADAPTER_VERSION,
            input_fingerprint,
            prefix="cgr",
        )
        raw_doc_present = source_id in (raw_docs or {}) and bool(str((raw_docs or {}).get(source_id) or ""))
        raw_doc = str((raw_docs or {}).get(source_id) or "")
        revision_status = "active" if raw_doc_present else "failed_closed"
        if graph_revision_id not in graph_revision_ids_seen:
            graph["graph_revisions"].append(
                {
                    "container_graph_revision_id": graph_revision_id,
                    "source_id": source_id,
                    "logical_source_id": logical_source_id,
                    "family": "document",
                    "revision_id": revision_id,
                    "revision_scope": "source_revision",
                    "content_revision_id": content_revision_id,
                    "external_revision_id": None,
                    "adapter_name": DOCUMENT_GRAPH_ADAPTER,
                    "adapter_version": DOCUMENT_GRAPH_ADAPTER_VERSION,
                    "inference_version": None,
                    "input_fingerprint": input_fingerprint,
                    "graph_fingerprint": None,
                    "profile_ids_json": ["document:v1"],
                    "coverage_report_ids_json": [],
                    "status": revision_status,
                    "parent_graph_revision_id": None,
                    "created_at": _utcnow_iso(),
                    "completed_at": _utcnow_iso(),
                }
            )
            graph_revision_ids_seen.add(graph_revision_id)
        if not raw_doc_present:
            continue

        raw_source_provenance = "original_source"

        source_render_ref = {
            "ref_type": "raw_doc",
            "source_id": source_id,
            "logical_source_id": logical_source_id,
            "revision_id": revision_id,
            "text": raw_doc,
            "render_source": "raw_doc_full" if raw_doc_present else raw_source_provenance,
            "raw_source_present": raw_doc_present,
            "raw_source_provenance": raw_source_provenance,
        }
        source_span_refs = [
            {
                "ref_type": "raw_doc_char_span",
                "source_id": source_id,
                "revision_id": revision_id,
                "start": 0,
                "end": len(raw_doc),
                "text_hash": hashlib.sha256(raw_doc.encode("utf-8")).hexdigest(),
            }
        ]
        source_container_id = stable_hash(
            "container-id-v1",
            "source_revision",
            source_id,
            revision_id,
            "document",
            "document:source",
            source_span_refs,
            source_render_ref,
            prefix="ctr",
        )
        source_render = _render_ref_row(
            graph_revision_id=graph_revision_id,
            container_id=source_container_id,
            render_kind="original_text",
            render_mode="exact_copy" if raw_doc_present else "degraded_episode_join",
            ref_json=source_render_ref,
        )
        graph["render_refs"].append(source_render)
        graph["containers"].append(
            {
                "container_id": source_container_id,
                "id_origin": "stable_hash",
                "id_schema_version": "container-id-v1",
                "identity_scope": "source_revision",
                "source_id": source_id,
                "logical_source_id": logical_source_id,
                "family": "document",
                "revision_id": revision_id,
                "revision_scope": "source_revision",
                "content_revision_id": content_revision_id,
                "external_revision_id": None,
                "container_graph_revision_id": graph_revision_id,
                "kind_ns": "document",
                "kind": "source",
                "kind_fq": "document:source",
                "kind_version": "v1",
                "traits_json": {"capabilities": ["source_unit", "renderable_unit"]},
                "order_key_json": {
                    "basis": "source_order",
                    "scope": "source_revision",
                    "scope_id": f"{source_id}:{revision_id}",
                    "segments": [0],
                    "unit": "container",
                },
                "order_basis": "source_order",
                "order_scope": "source_revision",
                "order_scope_id": f"{source_id}:{revision_id}",
                "span_refs_json": source_span_refs,
                "episode_ids_json": [ep.get("episode_id") for ep in episodes if ep.get("episode_id")],
                "primary_render_ref_id": source_render["render_ref_id"],
                "primary_render_ref_fingerprint": source_render["ref_fingerprint"],
                "render_ref_json": source_render_ref,
                "status": "active",
                "supersedes_container_id": None,
                "superseded_by_container_id": None,
                "boundary_score": 1.0,
                "kind_score": 1.0,
                "render_score": 1.0,
                "acl_inherit_source": 1,
                "owner_id": record.get("owner_id"),
                "scope": record.get("scope"),
                "agent_id": record.get("agent_id"),
                "swarm_id": record.get("swarm_id"),
                "read_json": list(record.get("read") or []),
                "write_json": list(record.get("write") or []),
                "acl_source_id": source_id,
                "acl_policy": "inherit_source_record",
                "adapter_name": DOCUMENT_GRAPH_ADAPTER,
                "adapter_version": DOCUMENT_GRAPH_ADAPTER_VERSION,
                "inference_version": None,
                "created_at": _utcnow_iso(),
                "updated_at": _utcnow_iso(),
            }
        )
        source_ref = _ref_row(
            graph_revision_id=graph_revision_id,
            container_id=source_container_id,
            ref_role="source",
            ref_type="source_document",
            ref_json={"source_id": source_id, "revision_id": revision_id},
            source_id=source_id,
            logical_source_id=logical_source_id,
            revision_id=revision_id,
            coverage_required=True,
        )
        graph["refs"].append(source_ref)
        graph["ref_lookup"].append(_lookup_row(source_ref["ref_id"], "source", "source_id", value_text=source_id))

        assign_document_artifact_span_ids(episodes, source_id=source_id)
        episodes_by_span: dict[str, list[dict]] = defaultdict(list)
        for episode in episodes:
            marker_span_ids = _episode_marker_span_ids(episode, source_id)
            span_ids = marker_span_ids or [str(episode.get("artifact_span_id") or "").strip()]
            for span_id in span_ids:
                if span_id and episode not in episodes_by_span[span_id]:
                    episodes_by_span[span_id].append(episode)

        raw_artifact_specs = _raw_doc_artifact_specs(raw_doc, source_id) if raw_doc_present else []
        if raw_artifact_specs:
            artifact_specs = [
                {
                    **spec,
                    "episodes": sorted(episodes_by_span.get(str(spec["span_id"])) or [], key=_episode_sort_key),
                }
                for spec in raw_artifact_specs
            ]
        else:
            # Episode/block joins are derived working text and must not become
            # exact-copy render refs. Without raw marker spans, structural
            # artifact rendering fails closed at plan time.
            artifact_specs = []

        previous_container_id: str | None = None
        for order_index, artifact_spec in enumerate(artifact_specs, start=1):
            span_id = str(artifact_spec["span_id"])
            span_episodes = list(artifact_spec.get("episodes") or [])
            episode_ids = [str(ep.get("episode_id") or "") for ep in span_episodes if str(ep.get("episode_id") or "")]
            artifact_text = str(artifact_spec.get("artifact_text") or "")
            raw_start = artifact_spec.get("raw_start")
            raw_end = artifact_spec.get("raw_end")
            render_source = str(artifact_spec.get("render_source") or "")
            instruction_text, response_text = split_prompt_response_artifact(artifact_text)
            render_text = response_text or artifact_text
            artifact_key = str(artifact_spec.get("artifact_key") or _artifact_key(span_id))
            span_refs = []
            if raw_start is not None and raw_end is not None:
                span_refs.append(
                    {
                        "ref_type": "raw_doc_char_span",
                        "source_id": source_id,
                        "revision_id": revision_id,
                        "start": raw_start,
                        "end": raw_end,
                        "text_hash": hashlib.sha256(artifact_text.encode("utf-8")).hexdigest(),
                    }
                )
            span_refs.extend(
                {
                    "ref_type": "episode",
                    "doc_id": str(doc.get("doc_id") or f"document:{source_id}"),
                    "episode_id": episode_id,
                    "source_id": source_id,
                    "revision_id": revision_id,
                }
                for episode_id in episode_ids
            )
            render_ref = {
                "ref_type": "document_artifact_response_text",
                "source_id": source_id,
                "logical_source_id": logical_source_id,
                "revision_id": revision_id,
                "artifact_span_id": span_id,
                "artifact_id": artifact_key,
                "render_source": render_source,
                "raw_source_present": raw_doc_present,
                "raw_source_provenance": raw_source_provenance,
                "raw_start": raw_start,
                "raw_end": raw_end,
                "text": render_text,
                "artifact_text": artifact_text,
                "instruction_text": instruction_text,
            }
            container_id = stable_hash(
                "container-id-v1",
                "source_revision",
                source_id,
                revision_id,
                "document",
                "document:artifact",
                span_refs,
                {
                    key: value
                    for key, value in render_ref.items()
                    if key not in {"text", "artifact_text", "instruction_text"}
                },
                prefix="ctr",
            )
            render_row = _render_ref_row(
                graph_revision_id=graph_revision_id,
                container_id=container_id,
                render_kind="original_text",
                render_mode="exact_copy" if render_source == EXACT_COPY_RENDER_SOURCE else "degraded_episode_join",
                ref_json=render_ref,
            )
            graph["render_refs"].append(render_row)
            order_key = {
                "basis": "source_order",
                "scope": "source_revision",
                "scope_id": f"{source_id}:{revision_id}",
                "segments": [order_index],
                "unit": "container",
            }
            graph["containers"].append(
                {
                    "container_id": container_id,
                    "id_origin": "stable_hash",
                    "id_schema_version": "container-id-v1",
                    "identity_scope": "source_revision",
                    "source_id": source_id,
                    "logical_source_id": logical_source_id,
                    "family": "document",
                    "revision_id": revision_id,
                    "revision_scope": "source_revision",
                    "content_revision_id": content_revision_id,
                    "external_revision_id": None,
                    "container_graph_revision_id": graph_revision_id,
                    "kind_ns": "document",
                    "kind": "artifact",
                    "kind_fq": "document:artifact",
                    "kind_version": "v1",
                    "traits_json": {
                        "capabilities": ["addressable_unit", "ordered_unit", "renderable_unit"],
                        "artifact_id": artifact_key,
                        "instruction_text": instruction_text,
                    },
                    "order_key_json": order_key,
                    "order_basis": "source_order",
                    "order_scope": "source_revision",
                    "order_scope_id": f"{source_id}:{revision_id}",
                    "span_refs_json": span_refs,
                    "episode_ids_json": episode_ids,
                    "primary_render_ref_id": render_row["render_ref_id"],
                    "primary_render_ref_fingerprint": render_row["ref_fingerprint"],
                    "render_ref_json": render_ref,
                    "status": "active",
                    "supersedes_container_id": None,
                    "superseded_by_container_id": None,
                    "boundary_score": 1.0 if render_source == "raw_doc_marker_span" else 0.75,
                    "kind_score": 1.0,
                    "render_score": 1.0 if render_text else 0.0,
                    "acl_inherit_source": 1,
                    "owner_id": record.get("owner_id"),
                    "scope": record.get("scope"),
                    "agent_id": record.get("agent_id"),
                    "swarm_id": record.get("swarm_id"),
                    "read_json": list(record.get("read") or []),
                    "write_json": list(record.get("write") or []),
                    "acl_source_id": source_id,
                    "acl_policy": "inherit_source_record",
                    "adapter_name": DOCUMENT_GRAPH_ADAPTER,
                    "adapter_version": DOCUMENT_GRAPH_ADAPTER_VERSION,
                    "inference_version": None,
                    "created_at": _utcnow_iso(),
                    "updated_at": _utcnow_iso(),
                }
            )
            artifact_ref = _ref_row(
                graph_revision_id=graph_revision_id,
                container_id=container_id,
                ref_role="structural_unit",
                ref_type="document_artifact",
                ref_json={
                    "source_id": source_id,
                    "revision_id": revision_id,
                    "artifact_span_id": span_id,
                    "artifact_id": artifact_key,
                    "source_order": order_index,
                },
                source_id=source_id,
                logical_source_id=logical_source_id,
                revision_id=revision_id,
                coverage_required=True,
            )
            graph["refs"].append(artifact_ref)
            graph["ref_lookup"].extend(
                [
                    _lookup_row(artifact_ref["ref_id"], "document", "artifact_span_id", value_text=span_id),
                    _lookup_row(artifact_ref["ref_id"], "document", "artifact_id", value_text=artifact_key),
                    _lookup_row(artifact_ref["ref_id"], "document", "source_id", value_text=source_id),
                    _lookup_row(artifact_ref["ref_id"], "document", "source_order", value_int=order_index),
                ]
            )
            graph["ref_ranges"].append(
                _range_row(
                    artifact_ref["ref_id"],
                    "source_order",
                    f"{source_id}:{revision_id}",
                    order_index,
                    order_index,
                )
            )
            if raw_start is not None and raw_end is not None:
                graph["ref_ranges"].append(
                    _range_row(
                        artifact_ref["ref_id"],
                        "raw_doc_char_span",
                        f"{source_id}:{revision_id}",
                        raw_start,
                        raw_end,
                    )
                )
            for episode_id in episode_ids:
                episode_ref = _ref_row(
                    graph_revision_id=graph_revision_id,
                    container_id=container_id,
                    ref_role="retrievable_unit",
                    ref_type="episode",
                    ref_json={
                        "doc_id": str(doc.get("doc_id") or f"document:{source_id}"),
                        "episode_id": episode_id,
                        "source_id": source_id,
                        "revision_id": revision_id,
                    },
                    source_id=source_id,
                    logical_source_id=logical_source_id,
                    revision_id=revision_id,
                    coverage_required=True,
                )
                graph["refs"].append(episode_ref)
                graph["ref_lookup"].extend(
                    [
                        _lookup_row(episode_ref["ref_id"], "episode", "episode_id", value_text=episode_id),
                        _lookup_row(episode_ref["ref_id"], "episode", "source_id", value_text=source_id),
                    ]
                )
            graph["evidence"].extend(
                [
                    _evidence_row(
                        graph_revision_id=graph_revision_id,
                        subject_type="container",
                        subject_id=container_id,
                        evidence_kind="boundary",
                        evidence_ref_json={
                            "artifact_span_id": span_id,
                            "artifact_id": artifact_key,
                            "episode_ids": episode_ids,
                            "render_source": render_source,
                        },
                        role="boundary_detection",
                        score_name="boundary_score",
                        score=1.0 if render_source == "raw_doc_marker_span" else 0.75,
                    ),
                    _evidence_row(
                        graph_revision_id=graph_revision_id,
                        subject_type="container_render_ref",
                        subject_id=render_row["render_ref_id"],
                        evidence_kind="render_ref",
                        evidence_ref_json={"artifact_span_id": span_id, "render_kind": "original_text"},
                        role="exact_copy_render",
                        score_name="render_score",
                        score=1.0 if render_text else 0.0,
                    ),
                ]
            )
            graph["relations"].append(
                _relation_row(
                    graph_revision_id=graph_revision_id,
                    src_container_id=source_container_id,
                    dst_container_id=container_id,
                    relation_kind="contains",
                    order_key=order_key,
                )
            )
            graph["relations"].append(
                _relation_row(
                    graph_revision_id=graph_revision_id,
                    src_container_id=container_id,
                    dst_container_id=source_container_id,
                    relation_kind="contained_by",
                    order_key=order_key,
                )
            )
            if previous_container_id:
                graph["relations"].append(
                    _relation_row(
                        graph_revision_id=graph_revision_id,
                        src_container_id=previous_container_id,
                        dst_container_id=container_id,
                        relation_kind="reading_order_next",
                        order_key=order_key,
                    )
                )
                graph["relations"].append(
                    _relation_row(
                        graph_revision_id=graph_revision_id,
                        src_container_id=previous_container_id,
                        dst_container_id=container_id,
                        relation_kind="precedes",
                        order_key=order_key,
                    )
                )
            previous_container_id = container_id

    for key in CONTAINER_GRAPH_KEYS:
        id_key = {
            "graph_revisions": "container_graph_revision_id",
            "containers": "container_id",
            "relations": "relation_id",
            "anchors": "anchor_id",
            "evidence": "evidence_id",
            "refs": "ref_id",
            "ref_lookup": "lookup_id",
            "ref_ranges": "range_id",
            "render_refs": "render_ref_id",
            "state": "state_id",
            "contracts": "contract_id",
            "artifacts": "artifact_id",
        }[key]
        deduped = _row_by_id(graph[key], id_key)
        graph[key] = [deduped[row_id] for row_id in sorted(deduped)]
    return graph


def graph_indexes(graph: dict) -> dict[str, dict]:
    graph = normalize_container_graph(graph)
    containers = _row_by_id(graph["containers"], "container_id")
    render_refs = _row_by_id(graph["render_refs"], "render_ref_id")
    refs = _row_by_id(graph["refs"], "ref_id")
    by_episode: dict[str, list[str]] = defaultdict(list)
    by_episode_ref: dict[str, list[str]] = defaultdict(list)
    by_source_artifact: dict[str, list[dict]] = defaultdict(list)
    by_order_scope_artifact: dict[str, list[dict]] = defaultdict(list)
    source_order_scopes: dict[str, set[str]] = defaultdict(set)
    for ref in graph["refs"]:
        if ref.get("status") != "active":
            continue
        container_id = str(ref.get("container_id") or "")
        if not container_id:
            continue
        if ref.get("ref_type") == "episode":
            ref_json = dict(ref.get("ref_json") or {})
            episode_id = str(ref_json.get("episode_id") or "")
            if episode_id:
                by_episode[episode_id].append(container_id)
                doc_id = str(ref_json.get("doc_id") or "")
                if doc_id:
                    by_episode_ref[f"{doc_id}\0{episode_id}"].append(container_id)
    for container in graph["containers"]:
        if container.get("status") != "active" or container.get("kind_fq") != "document:artifact":
            continue
        source_id = str(container.get("source_id") or "")
        order_scope_id = str(container.get("order_scope_id") or "")
        by_source_artifact[source_id].append(container)
        if order_scope_id:
            by_order_scope_artifact[order_scope_id].append(container)
            if source_id:
                source_order_scopes[source_id].add(order_scope_id)
    for rows in by_source_artifact.values():
        rows.sort(key=lambda row: tuple((row.get("order_key_json") or {}).get("segments") or [10**9]))
    for rows in by_order_scope_artifact.values():
        rows.sort(key=lambda row: tuple((row.get("order_key_json") or {}).get("segments") or [10**9]))
    return {
        "containers": containers,
        "render_refs": render_refs,
        "refs": refs,
        "by_episode": dict(by_episode),
        "by_episode_ref": dict(by_episode_ref),
        "by_source_artifact": dict(by_source_artifact),
        "by_order_scope_artifact": dict(by_order_scope_artifact),
        "source_order_scopes": {source_id: sorted(scope_ids) for source_id, scope_ids in source_order_scopes.items()},
    }


def lift_episode_ids_to_container_ids(graph: dict, episode_ids: list[str]) -> list[str]:
    indexes = graph_indexes(graph)
    lifted: list[str] = []
    for episode_id in episode_ids:
        for container_id in indexes["by_episode"].get(str(episode_id or ""), []):
            if container_id not in lifted:
                lifted.append(container_id)
    return lifted


def lift_episode_refs_to_container_ids(graph: dict, episode_refs: list[dict]) -> list[str]:
    indexes = graph_indexes(graph)
    lifted: list[str] = []
    for episode_ref in episode_refs:
        episode_id = str((episode_ref or {}).get("episode_id") or "")
        doc_id = str((episode_ref or {}).get("doc_id") or "")
        if doc_id and episode_id:
            candidates = indexes["by_episode_ref"].get(f"{doc_id}\0{episode_id}", [])
        else:
            candidates = indexes["by_episode"].get(episode_id, [])
        for container_id in candidates:
            if container_id not in lifted:
                lifted.append(container_id)
    return lifted


def _container_search_tokens(container: dict) -> set[str]:
    traits = dict(container.get("traits_json") or {})
    render_ref = dict(container.get("render_ref_json") or {})
    text = " ".join(
        str(value or "")
        for value in (
            traits.get("instruction_text"),
            render_ref.get("instruction_text"),
            render_ref.get("artifact_id"),
        )
    )
    tokens: set[str] = set()
    for match in _WORD_RE.finditer(text):
        tokens.update(_normalized_token_variants(match.group(0)))
    return tokens


def _selector_surface_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for word_match in _WORD_RE.finditer(text):
        word = word_match.group(0).lower()
        if word in _STRUCTURAL_QUERY_STOP_TOKENS or word in _ORDINAL_TOKENS:
            continue
        if word not in tokens:
            tokens.append(word)
    return tokens


def _surface_anchor_match(surface_tokens: list[str], selected_tokens: set[str]) -> tuple[list[str], list[str]]:
    matched: list[str] = []
    missing: list[str] = []
    for token in surface_tokens:
        variants = _normalized_token_variants(token)
        if variants & selected_tokens:
            matched.append(token)
        else:
            missing.append(token)
    return matched, missing


def _about_topic_anchor_tokens(query_features: dict) -> set[str]:
    retrieval_target = str(query_features.get("retrieval_target") or query_features.get("raw") or "")
    match = re.search(r"\babout\s+(.+?)(?:[.?!]|$)", retrieval_target, flags=re.IGNORECASE)
    if not match:
        return set()
    topic = match.group(1)
    topic = re.sub(r"\([^)]*\)", " ", topic)
    topic = re.sub(r"\b(?:do not|in your response|include any other text|return only)\b.*$", " ", topic, flags=re.I)
    tokens: list[str] = []
    for word_match in _WORD_RE.finditer(topic):
        word = word_match.group(0)
        lowered = word.lower()
        if lowered in {"the", "a", "an", "about"} or lowered in _ORDINAL_TOKENS:
            continue
        variants = _normalized_token_variants(word)
        tokens.extend(sorted(variants))
    return {
        token
        for token in tokens
        if token and token not in _STRUCTURAL_QUERY_STOP_TOKENS and token not in _ORDINAL_TOKENS
    }


def _selector_text_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for word_match in _WORD_RE.finditer(text):
        word = word_match.group(0)
        lowered = word.lower()
        if lowered in _STRUCTURAL_QUERY_STOP_TOKENS or lowered in _ORDINAL_TOKENS:
            continue
        tokens.update(
            token
            for token in _normalized_token_variants(word)
            if token and token not in _STRUCTURAL_QUERY_STOP_TOKENS and token not in _ORDINAL_TOKENS
        )
    return tokens


def _selector_target_proof(query_features: dict) -> dict:
    retrieval_target = str(query_features.get("retrieval_target") or query_features.get("raw") or "")
    cleaned = re.sub(r"\([^)]*\)", " ", retrieval_target)
    cleaned = re.sub(
        r"\b(?:do not|include any other text|return only|in your response)\b.*$",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    topic_text = ""
    topic_match = re.search(r"\babout\s+(.+?)(?:[.?!]|$)", cleaned, flags=re.IGNORECASE)
    if topic_match:
        topic_text = topic_match.group(1).strip()
        kind_text = cleaned[: topic_match.start()]
    else:
        kind_text = cleaned
    kind_text = re.sub(r"^\s*prepend\s+\S+\s+to\s+", " ", kind_text, flags=re.IGNORECASE)
    kind_text = _ORDINAL_PREFIX_RE.sub(" ", kind_text)
    kind_text = re.sub(r"^\s*(?:the|a|an)\b", " ", kind_text, flags=re.IGNORECASE)
    kind_text = re.sub(r"\s+", " ", kind_text).strip(" .,:;")
    topic_text = re.sub(r"\s+", " ", topic_text).strip(" .,:;")
    target_anchor_text = " ".join(
        part for part in (kind_text, f"about {topic_text}" if topic_text else "") if part
    )
    return {
        "target_kind": kind_text or None,
        "target_topic": topic_text or None,
        "target_anchor_text": target_anchor_text or None,
        "target_kind_surface_tokens": _selector_surface_tokens(kind_text),
        "target_topic_surface_tokens": _selector_surface_tokens(topic_text),
        "target_kind_tokens": sorted(_selector_text_tokens(kind_text)),
        "target_topic_tokens": sorted(_selector_text_tokens(topic_text)),
    }


def _query_anchor_tokens(query_features: dict) -> set[str]:
    words = set(query_features.get("words") or set())
    normalized: set[str] = set()
    for token in words:
        normalized.update(_normalized_token_variants(str(token or "")))
    normalized.update(_about_topic_anchor_tokens(query_features))
    anchors = {
        token
        for token in normalized
        if token and token not in _STRUCTURAL_QUERY_STOP_TOKENS and token not in _ORDINAL_TOKENS
    }
    # The container graph path is for selecting the requested answer unit. The
    # answer-unit kind is useful ("song"), but the post-"about" topic should not
    # disappear just because it is a normally interrogative word ("The Who").
    return anchors


def _order_segments(container: dict) -> tuple:
    order_key = dict(container.get("order_key_json") or {})
    return tuple(order_key.get("segments") or [10**9])


def _render_text_for_container(container: dict, indexes: dict) -> tuple[str | None, dict | None, str | None]:
    render_ref_id = str(container.get("primary_render_ref_id") or "")
    render_ref = indexes["render_refs"].get(render_ref_id)
    if not render_ref or render_ref.get("status") != "active":
        return None, None, "render_ref_unresolved"
    ref_json = dict(render_ref.get("ref_json") or {})
    if str(ref_json.get("ref_type") or "") == "document_artifact_response_text":
        render_source = str(ref_json.get("render_source") or "")
        if render_source != EXACT_COPY_RENDER_SOURCE or str(render_ref.get("render_mode") or "") != "exact_copy":
            return None, render_ref, "EXACT_COPY_SOURCE_DEGRADED"
    text = str(ref_json.get("text") or "")
    if not text:
        return None, render_ref, "EXACT_RENDER_REF_UNRESOLVED"
    return text, render_ref, None


def validate_container_exact_copy_render_refs(graph: dict) -> list[dict]:
    """Return validation errors for exact-copy render refs not backed by raw source spans."""

    errors: list[dict] = []
    for row in normalize_container_graph(graph).get("render_refs") or []:
        ref_json = dict(row.get("ref_json") or {})
        if str(ref_json.get("ref_type") or "") != "document_artifact_response_text":
            continue
        render_source = str(ref_json.get("render_source") or "")
        render_mode = str(row.get("render_mode") or "")
        render_ref_id = str(row.get("render_ref_id") or "")
        if render_source != EXACT_COPY_RENDER_SOURCE or render_mode != "exact_copy":
            errors.append(
                {
                    "code": "EXACT_COPY_SOURCE_DEGRADED",
                    "render_ref_id": render_ref_id,
                    "container_id": row.get("container_id"),
                    "render_source": render_source or None,
                    "render_mode": render_mode or None,
                }
            )
            continue
        if ref_json.get("raw_start") is None or ref_json.get("raw_end") is None or not str(ref_json.get("text") or ""):
            errors.append(
                {
                    "code": "EXACT_RENDER_REF_UNRESOLVED",
                    "render_ref_id": render_ref_id,
                    "container_id": row.get("container_id"),
                    "render_source": render_source,
                }
            )
    return errors


def plan_document_structural_exact_copy(
    *,
    graph: dict,
    query: str,
    query_features: dict,
    seed_episode_ids: list[str],
    seed_episode_refs: list[dict] | None = None,
    fallback_source_ids: list[str] | None = None,
    explicit_order_scope_ids: list[str] | None = None,
) -> dict | None:
    """Plan structural document exact-copy retrieval over full operator domain."""

    operator_plan = dict(query_features.get("operator_plan") or {})
    ordinal = dict(operator_plan.get("ordinal") or {})
    output_constraints = dict(query_features.get("output_constraints") or {})
    if not ordinal.get("enabled") or not int(ordinal.get("index") or 0) > 0:
        return None
    if not (output_constraints.get("return_only") or output_constraints.get("prepend_prefix")):
        return None

    graph = normalize_container_graph(graph)
    indexes = graph_indexes(graph)
    seed_container_ids = (
        lift_episode_refs_to_container_ids(graph, seed_episode_refs)
        if seed_episode_refs is not None
        else lift_episode_ids_to_container_ids(graph, seed_episode_ids)
    )
    seed_containers = [
        indexes["containers"][container_id]
        for container_id in seed_container_ids
        if container_id in indexes["containers"]
    ]
    scope_ids = list(dict.fromkeys(str(row.get("order_scope_id") or "") for row in seed_containers if row.get("order_scope_id")))
    source_ids = list(dict.fromkeys(str(row.get("source_id") or "") for row in seed_containers if row.get("source_id")))
    if not scope_ids and explicit_order_scope_ids:
        scope_ids = [
            str(value or "")
            for value in dict.fromkeys(explicit_order_scope_ids)
            if str(value or "") in indexes["by_order_scope_artifact"]
        ]
    if not scope_ids and fallback_source_ids:
        source_ids = list(dict.fromkeys(str(value or "") for value in fallback_source_ids if str(value or "")))
        for source_id in source_ids:
            scope_ids.extend(indexes["source_order_scopes"].get(source_id, []))
        scope_ids = list(dict.fromkeys(scope_ids))
    ordinal_index = int(ordinal.get("index") or 0)
    target_proof = _selector_target_proof(query_features)
    anchor_tokens = _query_anchor_tokens(query_features)
    surface_anchor_tokens = list(
        dict.fromkeys(
            [
                *(target_proof.get("target_kind_surface_tokens") or []),
                *(target_proof.get("target_topic_surface_tokens") or []),
            ]
        )
    )
    target_kind_tokens = set(target_proof.get("target_kind_tokens") or [])
    target_topic_tokens = set(target_proof.get("target_topic_tokens") or [])
    trace: dict[str, Any] = {
        "container_path_attempted": True,
        "query_type": "exact_copy",
        "planner": "container_graph",
        "planner_contract_version": "container-query-plan-v1",
        "planner_implementation": "document_artifact_planner",
        "planner_implementation_scope": "document_artifact_first_implementation",
        "operator_kind": "nth",
        "requested_index": ordinal_index,
        "indexing": "one_indexed",
        "target_kind": target_proof.get("target_kind"),
        "target_topic": target_proof.get("target_topic"),
        "target_anchor_text": target_proof.get("target_anchor_text"),
        "target_kind_tokens": list(target_proof.get("target_kind_tokens") or []),
        "target_topic_tokens": list(target_proof.get("target_topic_tokens") or []),
        "target_kind_surface_tokens": list(target_proof.get("target_kind_surface_tokens") or []),
        "target_topic_surface_tokens": list(target_proof.get("target_topic_surface_tokens") or []),
        "anchor_tokens": sorted(anchor_tokens),
        "surface_anchor_tokens_requested": surface_anchor_tokens,
        "surface_anchor_tokens_matched": [],
        "normalized_anchor_tokens_requested": sorted(anchor_tokens),
        "normalized_anchor_tokens_matched": [],
        "proof_source_fields": list(CONTAINER_PROOF_SOURCE_FIELDS),
        "candidate_family": "document",
        "candidate_kind_fq": "document:artifact",
        "candidate_domain_policy": "enumerate_matching_containers_in_scope",
        "selected_index_in_matching_domain": None,
        "matching_domain_count": 0,
        "ordinal_satisfied": False,
        "kind_satisfied": not target_kind_tokens,
        "topic_satisfied": not target_topic_tokens,
        "anchor_tokens_matched": [],
        "anchor_tokens_missing": sorted(anchor_tokens),
        "render_ref_validated": False,
        "query_plan": {
            "planner_contract_version": "container-query-plan-v1",
            "intent": {
                "intent_kind": "document.structural_exact_copy",
                "family_any": ["document"],
                "answer_unit_capability": "renderable_unit",
            },
            "candidate_domain": {
                "seed_policy": "retrieval_hits",
                "lift_policy": "container_refs_first",
                "scope_policy": "same_source_revision_as_seed",
                "domain_policy": "enumerate_matching_containers_in_scope",
                "required_capabilities": ["ordered_unit", "renderable_unit"],
                "allowed_kinds": ["document:artifact"],
                "excluded_statuses": ["deleted", "superseded", "rejected"],
            },
            "operators": [
                {
                    "operator_kind": "nth",
                    "k": ordinal_index,
                    "indexing": "one_indexed",
                    "operator_domain": "all_matching_containers_in_scope",
                    "order_basis": "source_order",
                    "order_scope": "source_revision",
                    "sort_direction": "asc",
                }
            ],
            "render": {"mode": "exact_copy", "include_render_refs": True, "include_trace": True},
            "fallback_policy": {"default": "fail_closed", "allowed_fallbacks": []},
        },
        "seed_container_ids": seed_container_ids,
        "scope_domain": [
            {
                "source_id": str((indexes["by_order_scope_artifact"].get(scope_id) or [{}])[0].get("source_id") or ""),
                "order_scope_id": scope_id,
            }
            for scope_id in scope_ids
        ],
        "operator_domain_container_ids": [],
        "ordered_candidate_container_ids": [],
        "selected_container_ids": [],
        "selected_render_ref_ids": [],
        "rejected_container_ids": [],
        "order_basis": "source_order",
        "order_scope": "source_revision",
        "order_scope_id": None,
        "render_mode": "exact_copy",
        "fallback_allowed": False,
        "fallback_reason": None,
        "raw_source_present": None,
        "raw_source_provenance": None,
        "render_source": None,
        "exact_copy_validated": False,
        "degraded_render_source": None,
        "no_cross_container_contamination": True,
    }
    if not scope_ids:
        trace["fallback_reason"] = "no_scope_domain"
        return {"status": "failed_closed", "trace": trace, "reason": "no_scope_domain"}
    if len(scope_ids) != 1:
        trace["fallback_reason"] = "ambiguous_order_scope"
        return {"status": "failed_closed", "trace": trace, "reason": "ambiguous_order_scope"}

    all_scope_candidates: list[dict] = list(indexes["by_order_scope_artifact"].get(scope_ids[0], []))
    all_scope_candidates = [
        row
        for row in all_scope_candidates
        if row.get("status") == "active" and row.get("kind_fq") == "document:artifact"
    ]
    all_scope_candidates.sort(key=lambda row: (str(row.get("order_scope_id") or ""), _order_segments(row)))
    trace["operator_domain_container_ids"] = [str(row.get("container_id") or "") for row in all_scope_candidates]

    if not all_scope_candidates:
        trace["fallback_reason"] = "empty_operator_domain"
        return {"status": "failed_closed", "trace": trace, "reason": "empty_operator_domain"}

    if anchor_tokens:
        filtered: list[dict] = []
        for container in all_scope_candidates:
            tokens = _container_search_tokens(container)
            missing = sorted(anchor_tokens - tokens)
            if missing:
                trace["rejected_container_ids"].append(
                    {
                        "container_id": container.get("container_id"),
                        "reason": "anchor_tokens_missing",
                        "missing": missing,
                    }
                )
                continue
            filtered.append(container)
    else:
        filtered = list(all_scope_candidates)

    if not filtered:
        trace["fallback_reason"] = "empty_anchor_filtered_domain"
        return {"status": "failed_closed", "trace": trace, "reason": "empty_anchor_filtered_domain"}

    trace["order_scope_id"] = scope_ids[0]
    trace["query_plan"]["operators"][0]["order_scope_id"] = scope_ids[0]
    ordered = sorted(filtered, key=_order_segments)
    trace["ordered_candidate_container_ids"] = [str(row.get("container_id") or "") for row in ordered]
    trace["matching_domain_count"] = len(ordered)

    index = ordinal_index
    if index < 1 or index > len(ordered):
        trace["fallback_reason"] = "ordinal_out_of_range"
        return {"status": "failed_closed", "trace": trace, "reason": "ordinal_out_of_range"}

    selected = ordered[index - 1]
    selected_tokens = _container_search_tokens(selected)
    surface_matched, surface_missing = _surface_anchor_match(surface_anchor_tokens, selected_tokens)
    trace.update(
        {
            "candidate_family": str(selected.get("family") or "document"),
            "candidate_kind_fq": str(selected.get("kind_fq") or "document:artifact"),
            "selected_index_in_matching_domain": index,
            "ordinal_satisfied": index == ordinal_index,
            "kind_satisfied": not bool(target_kind_tokens - selected_tokens),
            "topic_satisfied": not bool(target_topic_tokens - selected_tokens),
            "anchor_tokens_matched": sorted(anchor_tokens & selected_tokens),
            "anchor_tokens_missing": sorted(anchor_tokens - selected_tokens),
            "surface_anchor_tokens_matched": surface_matched,
            "surface_anchor_tokens_missing": surface_missing,
            "normalized_anchor_tokens_matched": sorted(anchor_tokens & selected_tokens),
            "normalized_anchor_tokens_missing": sorted(anchor_tokens - selected_tokens),
        }
    )
    render_text, render_ref, render_error = _render_text_for_container(selected, indexes)
    render_ref_json = dict((render_ref or {}).get("ref_json") or {})
    if render_ref_json:
        trace["raw_source_present"] = bool(render_ref_json.get("raw_source_present"))
        trace["raw_source_provenance"] = render_ref_json.get("raw_source_provenance")
        trace["render_source"] = render_ref_json.get("render_source")
    if render_text is None or render_ref is None:
        trace["fallback_reason"] = render_error or "render_ref_unresolved"
        trace["selected_container_ids"] = [str(selected.get("container_id") or "")]
        if render_error == "EXACT_COPY_SOURCE_DEGRADED":
            trace["degraded_render_source"] = render_ref_json.get("render_source")
        return {"status": "failed_closed", "trace": trace, "reason": render_error or "render_ref_unresolved"}

    trace["selected_container_ids"] = [str(selected.get("container_id") or "")]
    trace["selected_render_ref_ids"] = [str(render_ref.get("render_ref_id") or "")]
    trace["selected_episode_ids"] = list(selected.get("episode_ids_json") or [])
    trace["selected_artifact_span_ids"] = [str((selected.get("render_ref_json") or {}).get("artifact_span_id") or "")]
    trace["exact_copy_validated"] = True
    trace["render_ref_validated"] = True
    return {
        "status": "rendered",
        "trace": trace,
        "selected_container": deepcopy(selected),
        "selected_render_ref": deepcopy(render_ref),
        "render_text": render_text,
        "context": (
            "--- CONTAINER GRAPH EXACT RENDER ---\n"
            f"[Document Span: {(selected.get('render_ref_json') or {}).get('artifact_span_id')}] "
            f"[Container: {selected.get('container_id')}] "
            f"[Render Ref: {render_ref.get('render_ref_id')}]\n"
            f"{render_text}"
        ),
    }
