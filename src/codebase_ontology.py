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
from typing import Any


def _compact_mapping(values: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key, value in values.items():
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if isinstance(value, (list, tuple, set, dict)) and not value:
            continue
        if isinstance(value, dict):
            compact[key] = dict(value)
        elif isinstance(value, list):
            compact[key] = list(value)
        elif isinstance(value, tuple):
            compact[key] = tuple(value)
        elif isinstance(value, set):
            compact[key] = set(value)
        else:
            compact[key] = value
    return compact


def _dedupe_strings(values: list[str] | tuple[str, ...] | None) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values or []:
        item = str(value or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def build_codebase_anchor(kind: str, **kwargs: Any) -> dict[str, Any]:
    return _compact_mapping({"kind": str(kind or "").strip(), **kwargs})


def _stable_relation_fact_id(relation_type: str, from_id: str, to_id: str, anchor: dict[str, Any] | None) -> str:
    payload = json.dumps(
        {
            "relation_type": relation_type,
            "from_id": from_id,
            "to_id": to_id,
            "anchor": anchor or {},
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha1(payload.encode("utf-8"), usedforsecurity=False).hexdigest()[:20]
    return f"rel_{relation_type}_{digest}"


def build_codebase_relation_fact(
    *,
    relation_type: str,
    from_id: str,
    to_id: str,
    fact_text: str,
    anchor: dict[str, Any] | None = None,
    entities: list[str] | None = None,
    tags: list[str] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = _compact_mapping(
        {
            "codebase": _compact_mapping(
                {
                    "relation_type": relation_type,
                    "from_id": from_id,
                    "to_id": to_id,
                    "anchor": anchor or {},
                    **(extra_metadata or {}),
                }
            )
        }
    )
    return {
        "id": _stable_relation_fact_id(relation_type, from_id, to_id, anchor),
        "fact": str(fact_text or "").strip(),
        "kind": "codebase_relation",
        "entities": _dedupe_strings(entities or []),
        "tags": _dedupe_strings(["codebase", relation_type, *(tags or [])]),
        "metadata": metadata,
    }


def build_codebase_object_fact(
    *,
    object_type: str,
    object_id: str,
    anchor: dict[str, Any] | None = None,
    fact_text: str,
    entities: list[str] | None = None,
    tags: list[str] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = _compact_mapping(
        {
            "codebase": _compact_mapping(
                {
                    "object_type": object_type,
                    "object_id": object_id,
                    "anchor": anchor or {},
                    **(extra_metadata or {}),
                }
            )
        }
    )
    return {
        "id": f"obj_{object_id}",
        "fact": str(fact_text or "").strip(),
        "kind": "codebase_object",
        "entities": _dedupe_strings([object_id, *(entities or [])]),
        "tags": _dedupe_strings(["codebase", object_type, *(tags or [])]),
        "metadata": metadata,
    }


def build_codebase_object_row(
    *,
    object_type: str,
    object_id: str,
    anchor: dict[str, Any] | None = None,
    fact_text: str,
    entities: list[str] | None = None,
    tags: list[str] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    object_type_name = str(object_type or "").strip()
    extra = dict(extra_metadata or {})
    anchor_row = dict(anchor or {})
    if object_type_name == "diff":
        return _compact_mapping(
            {
                "__codebase_object_row_v1__": "diff",
                "i": object_id,
                "c": str(extra.get("commit_sha") or "").strip(),
                "p": str(extra.get("path") or anchor_row.get("path") or "").strip(),
                "s": str(extra.get("status") or anchor_row.get("status") or "").strip(),
                "rf": str(extra.get("rename_from") or "").strip(),
                "rt": str(extra.get("rename_to") or "").strip(),
                "hs": str(extra.get("hydration_state") or "").strip(),
                "pr": str(extra.get("patch_ref") or "").strip(),
                "hc": int(extra.get("hunk_count") or 0),
            }
        )
    if object_type_name == "hunk":
        old_range = list(extra.get("old_range") or [])
        new_range = list(extra.get("new_range") or [])
        return _compact_mapping(
            {
                "__codebase_object_row_v1__": "hunk",
                "i": object_id,
                "c": str(extra.get("commit_sha") or "").strip(),
                "p": str(extra.get("path") or anchor_row.get("path") or "").strip(),
                "hh": str(extra.get("hunk_header") or "").strip(),
                "os": old_range[0] if len(old_range) > 0 else None,
                "oc": old_range[1] if len(old_range) > 1 else None,
                "ns": new_range[0] if len(new_range) > 0 else None,
                "nc": new_range[1] if len(new_range) > 1 else None,
                "s": str(extra.get("status") or (tags[1] if tags and len(tags) > 1 else "")).strip(),
                "hs": str(extra.get("hydration_state") or "").strip(),
                "pr": str(extra.get("patch_ref") or "").strip(),
            }
        )
    return _compact_mapping(
        {
            "__codebase_object_row_v1__": True,
            "object_type": object_type_name,
            "object_id": object_id,
            "fact_text": str(fact_text or "").strip(),
            "anchor": anchor or {},
            "entities": _dedupe_strings(entities or []),
            "tags": _dedupe_strings(tags or []),
            "extra_metadata": extra_metadata or {},
        }
    )


def hydrate_codebase_object_row(row: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(row, dict):
        return {}
    row_type = row.get("__codebase_object_row_v1__")
    if not row_type:
        return dict(row)
    if row_type == "diff":
        commit_sha = str(row.get("c") or "").strip()
        path = str(row.get("p") or "").strip()
        status = str(row.get("s") or "modified").strip() or "modified"
        hunk_count = int(row.get("hc") or 0)
        return build_codebase_object_fact(
            object_type="diff",
            object_id=str(row.get("i") or "").strip(),
            anchor=build_codebase_anchor("diff", commit_sha=commit_sha, path=path, status=status),
            fact_text=f"Diff for commit {commit_sha} changes file {path} with {hunk_count} hunks.",
            entities=[commit_sha, path],
            tags=["diff", status, "git"],
            extra_metadata={
                "commit_sha": commit_sha,
                "path": path,
                "status": status,
                "rename_from": str(row.get("rf") or "").strip(),
                "rename_to": str(row.get("rt") or "").strip(),
                "hydration_state": str(row.get("hs") or "deferred").strip() or "deferred",
                "patch_ref": str(row.get("pr") or "").strip(),
                "hunk_count": hunk_count,
            },
        )
    if row_type == "hunk":
        commit_sha = str(row.get("c") or "").strip()
        path = str(row.get("p") or "").strip()
        header = str(row.get("hh") or "").strip()
        hydration_state = str(row.get("hs") or "").strip() or "deferred"
        patch_ref = str(row.get("pr") or "").strip()
        if hydration_state == "deferred" and not header:
            fact_text = f"Hunk details for commit {commit_sha} and file {path} are deferred to {patch_ref or '(none)'}."
        else:
            fact_text = f"Hunk {header} belongs to commit {commit_sha} and file {path}."
        return build_codebase_object_fact(
            object_type="hunk",
            object_id=str(row.get("i") or "").strip(),
            anchor=build_codebase_anchor(
                "hunk",
                commit_sha=commit_sha,
                path=path,
                hunk_header=header,
                old_range=[row.get("os"), row.get("oc")],
                new_range=[row.get("ns"), row.get("nc")],
                patch_ref=patch_ref,
                hydration_state=hydration_state,
            ),
            fact_text=fact_text,
            entities=[commit_sha, path, header],
            tags=["hunk", str(row.get("s") or "modified").strip() or "modified", "git", *(["deferred"] if hydration_state == "deferred" else [])],
            extra_metadata={
                "commit_sha": commit_sha,
                "path": path,
                "hunk_header": header,
                "old_range": [row.get("os"), row.get("oc")],
                "new_range": [row.get("ns"), row.get("nc")],
                "hydration_state": hydration_state,
                "patch_ref": patch_ref,
            },
        )
    return build_codebase_object_fact(
        object_type=str(row.get("object_type") or "").strip(),
        object_id=str(row.get("object_id") or "").strip(),
        anchor=dict(row.get("anchor") or {}),
        fact_text=str(row.get("fact_text") or "").strip(),
        entities=list(row.get("entities") or []),
        tags=list(row.get("tags") or []),
        extra_metadata=dict(row.get("extra_metadata") or {}),
    )


def build_codebase_object_unit(
    *,
    object_type: str,
    object_id: str,
    raw_text: str,
    fact_text: str,
    anchor: dict[str, Any] | None = None,
    source_date: str = "",
    currentness: str = "unknown",
    topic_key: str | None = None,
    state_label: str | None = None,
    notes: str | None = None,
    entities: list[str] | None = None,
    tags: list[str] | None = None,
    extra_metadata: dict[str, Any] | None = None,
    relation_facts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    rendered = str(raw_text or "").strip()
    object_fact = build_codebase_object_fact(
        object_type=object_type,
        object_id=object_id,
        anchor=anchor,
        fact_text=fact_text,
        entities=entities,
        tags=tags,
        extra_metadata=extra_metadata,
    )
    metadata = dict(object_fact.get("metadata") or {})
    facts = [object_fact]
    if relation_facts:
        facts.extend(list(relation_facts))
    return {
        "unit_key": object_id,
        "object_type": object_type,
        "episode": {
            "topic_key": str(topic_key or object_type),
            "state_label": str(state_label or object_type),
            "source_date": str(source_date or ""),
            "currentness": str(currentness or "unknown"),
            "raw_text": rendered,
            "provenance": {
                "source_section_path": str(object_id),
                "raw_span": [0, len(rendered)],
            },
            "notes": str(notes or "") if notes else None,
            "metadata": metadata,
        },
        "facts": facts,
    }


def codebase_metadata_from_fact(fact: dict[str, Any]) -> dict[str, Any]:
    metadata = fact.get("metadata") or {}
    if not isinstance(metadata, dict):
        return {}
    codebase = metadata.get("codebase") or {}
    return codebase if isinstance(codebase, dict) else {}


def codebase_object_id_from_fact(fact: dict[str, Any]) -> str:
    codebase = codebase_metadata_from_fact(fact)
    object_id = str(codebase.get("object_id") or "").strip()
    if object_id:
        return object_id
    from_id = str(codebase.get("from_id") or "").strip()
    if from_id:
        return from_id
    to_id = str(codebase.get("to_id") or "").strip()
    return to_id


def codebase_relation_endpoints(fact: dict[str, Any]) -> tuple[str, str]:
    codebase = codebase_metadata_from_fact(fact)
    return (
        str(codebase.get("from_id") or "").strip(),
        str(codebase.get("to_id") or "").strip(),
    )
