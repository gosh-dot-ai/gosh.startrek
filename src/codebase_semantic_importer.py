#!/usr/bin/env python3
# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .codebase_semantic_bundle import normalize_semantic_bundle
from .codebase_semantic_sidecars import CodebaseSemanticSidecarStore


def _span_label(span: dict[str, Any]) -> str:
    start_line = span.get("start_line")
    end_line = span.get("end_line")
    if start_line == end_line:
        return f"L{start_line}"
    return f"L{start_line}-L{end_line}"


def _payload_name(payload: dict[str, Any], fallback: str) -> str:
    for key in ("qualified_name", "name", "callee_name", "import_path", "signature"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return fallback


def _object_fact_text(row: dict[str, Any]) -> str:
    payload = row["payload"]
    language = str(row["language"]).title()
    name = _payload_name(payload, row["id"])
    file_path = row["file_path"]
    object_type = row["object_type"]
    if object_type == "module":
        return f"{language} module {name} is declared in {file_path}."
    if object_type == "class":
        return f"{language} class {name} is declared in {file_path}."
    if object_type == "callable":
        signature = str(payload.get("signature") or "").strip()
        if signature:
            return f"{language} callable {name} is declared in {file_path} with signature {signature}."
        return f"{language} callable {name} is declared in {file_path}."
    if object_type == "test_case":
        return f"{language} test case {name} is declared in {file_path}."
    if object_type == "parameter":
        owner = str(payload.get("callable_id") or "").strip()
        return f"{language} parameter {name} belongs to callable {owner} in {file_path}."
    if object_type == "field":
        owner = str(payload.get("owner_id") or "").strip()
        return f"{language} field {name} belongs to type {owner} in {file_path}."
    if object_type == "import":
        import_path = str(payload.get("import_path") or name)
        return f"{language} import {import_path} is declared in {file_path}."
    if object_type == "callsite":
        callee_name = str(payload.get("callee_name") or name)
        return f"{language} callsite for {callee_name} appears in {file_path}."
    return f"{language} {object_type} {name} is declared in {file_path}."


def _relation_fact_text(row: dict[str, Any], object_lookup: dict[str, dict[str, Any]]) -> str:
    from_row = object_lookup.get(row["from_id"]) or {}
    to_row = object_lookup.get(row["to_id"]) or {}
    from_name = _payload_name(from_row.get("payload") or {}, row["from_id"])
    to_name = _payload_name(to_row.get("payload") or {}, row["to_id"])
    from_type = str(from_row.get("object_type") or "object")
    to_type = str(to_row.get("object_type") or "object")
    language = str(row["language"]).title()
    relation_type = row["relation_type"]
    file_path = row["file_path"]
    if relation_type == "declares":
        return f"{language} {from_type} {from_name} declares {to_type} {to_name} in {file_path}."
    if relation_type == "imports":
        return f"{language} module {from_name} imports {to_name} in {file_path}."
    if relation_type == "calls":
        return f"{language} callable {from_name} calls {to_name} in {file_path}."
    if relation_type == "test_covers":
        return f"{language} test case {from_name} covers callable {to_name} in {file_path}."
    return f"{language} relation {relation_type} links {from_name} to {to_name} in {file_path}."


def _base_metadata(
    row: dict[str, Any],
    semantic_kind: str,
    semantic_type: str,
    sidecar_ref: dict[str, Any] | None,
    file_sidecar_ref: dict[str, Any] | None,
    *,
    skip_embedding: bool = False,
) -> dict[str, Any]:
    metadata = {
        "source_family": "codebase",
        "codebase_stage": "codebase_semantic",
        "semantic_kind": semantic_kind,
        "semantic_type": semantic_type,
        "repo_id": str(row["repo_id"]),
        "revision": str(row["revision"]),
        "file_path": str(row["file_path"]),
        "span": _span_label(row["span"]),
        "language": str(row["language"]),
        "analyzer_id": str(row["analyzer_id"]),
        "analyzer_version": str(row["analyzer_version"]),
        "derivation_type": str(row["derivation_type"]),
        "codebase": {
            "stage": "codebase_semantic",
            "semantic_kind": semantic_kind,
            "semantic_type": semantic_type,
            "object_type": semantic_type if semantic_kind == "object" else "",
            "relation_type": semantic_type if semantic_kind == "relation" else "",
            "skip_embedding": bool(skip_embedding),
        },
    }
    if sidecar_ref is not None:
        metadata["sidecar_id"] = str(sidecar_ref["sidecar_id"])
        metadata["sidecar_kind"] = str(sidecar_ref["sidecar_kind"])
    if file_sidecar_ref is not None:
        metadata["file_sidecar_id"] = str(file_sidecar_ref["sidecar_id"])
    return metadata


def _base_tags(row: dict[str, Any], semantic_kind: str, semantic_type: str) -> list[str]:
    return [
        "codebase",
        "codebase_semantic",
        str(row["language"]).lower(),
        semantic_kind,
        semantic_type,
    ]


def _entities(row: dict[str, Any], fallback: list[str] | None = None) -> list[str]:
    payload = row.get("payload") or {}
    values = []
    semantic_type = str(row.get("object_type") or row.get("relation_type") or "").strip().lower()
    entity_keys: tuple[str, ...] = ("qualified_name", "name", "callee_name")
    if semantic_type == "callsite":
        entity_keys = ("callee_name", "name")
    elif semantic_type == "import":
        entity_keys = ("import_path", "alias")
    for key in entity_keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            values.append(value.strip())
    if fallback:
        values.extend(str(item) for item in fallback if item)
    return list(dict.fromkeys(values))


def _select_node_sidecar(sidecars: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not sidecars:
        return None
    prioritized = sorted(
        sidecars,
        key=lambda row: (
            0 if str(row.get("sidecar_kind") or "") in {"ast", "semantic_snapshot"} else 1,
            str(row.get("sidecar_id") or ""),
        ),
    )
    return deepcopy(prioritized[0])


def _select_file_sidecar_refs(
    *,
    normalized_objects: list[dict[str, Any]],
    persisted_sidecars: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    object_lookup = {str(row["id"]): row for row in normalized_objects}
    refs: dict[str, dict[str, Any]] = {}
    for sidecar in persisted_sidecars:
        node_id = str(sidecar.get("node_id") or "")
        object_row = object_lookup.get(node_id) or {}
        if str(object_row.get("object_type") or "") != "module":
            continue
        file_path = str(sidecar.get("file_path") or object_row.get("file_path") or "").strip()
        if not file_path or file_path in refs:
            continue
        refs[file_path] = deepcopy(sidecar)
    return refs


_SEED_OBJECT_TYPES = {
    "module",
    "file",
    "directory",
    "class",
    "callable",
    "function",
    "method",
    "type",
    "interface",
    "schema",
    "config",
    "dependency_manifest",
    "command",
    "workflow",
    "workflow_job",
    "test_suite",
    "test_case",
    "test_run",
    "test_failure",
    "stack_trace",
    "stack_frame",
    "patch_attempt",
    "verification_result",
    "callsite",
}


def _is_hot_seed_fact(semantic_kind: str, semantic_type: str) -> bool:
    """Keep hot facts as query seeds, not a duplicate of the full graph.

    Repository context materializes the authoritative relation/ref/render substrate into
    the container graph. The fact tier stays as a compatibility seed index for
    repository scope discovery, so high-cardinality details such as parameters,
    imports, and graph relations should not be duplicated there.
    """
    if semantic_kind != "object":
        return False
    return semantic_type in _SEED_OBJECT_TYPES


def import_semantic_bundle(
    bundle: dict[str, Any],
    *,
    source_id: str,
    data_dir: str,
    hot_fact_policy: str = "all",
) -> dict[str, Any]:
    if hot_fact_policy not in {"all", "seed"}:
        raise ValueError("hot_fact_policy must be 'all' or 'seed'")
    normalized = normalize_semantic_bundle(bundle)
    sidecar_store = CodebaseSemanticSidecarStore(data_dir)
    persisted_sidecars: list[dict[str, Any]] = []
    sidecars_by_node: dict[str, list[dict[str, Any]]] = {}
    for sidecar in normalized["sidecars"]:
        persisted = sidecar_store.persist_sidecar(sidecar)
        persisted_sidecars.append(persisted)
        sidecars_by_node.setdefault(str(persisted["node_id"]), []).append(persisted)
    file_sidecars = _select_file_sidecar_refs(
        normalized_objects=normalized["objects"],
        persisted_sidecars=persisted_sidecars,
    )

    object_lookup = {row["id"]: row for row in normalized["objects"]}
    facts: list[dict[str, Any]] = []

    for row in normalized["objects"]:
        semantic_type = str(row["object_type"])
        if hot_fact_policy == "seed" and not _is_hot_seed_fact("object", semantic_type):
            continue
        sidecar_ref = _select_node_sidecar(sidecars_by_node.get(row["id"]) or [])
        file_sidecar_ref = file_sidecars.get(str(row["file_path"]))
        fact = {
            "id": row["id"],
            "fact": _object_fact_text(row),
            "kind": "fact",
            "entities": _entities(row),
            "tags": _base_tags(row, "object", row["object_type"]),
            "source_id": source_id,
            "source_family": "codebase",
            "semantic_kind": "object",
            "semantic_type": row["object_type"],
            "repo_id": row["repo_id"],
            "revision": row["revision"],
            "file_path": row["file_path"],
            "span": deepcopy(row["span"]),
            "language": row["language"],
            "analyzer_id": row["analyzer_id"],
            "analyzer_version": row["analyzer_version"],
            "derivation_type": row["derivation_type"],
            "semantic_payload": deepcopy(row["payload"]),
            "metadata": _base_metadata(row, "object", row["object_type"], sidecar_ref, file_sidecar_ref),
        }
        if sidecar_ref is not None:
            fact["sidecar_ref"] = deepcopy(sidecar_ref)
        if file_sidecar_ref is not None:
            fact["file_sidecar_ref"] = deepcopy(file_sidecar_ref)
        facts.append(fact)

    for row in normalized["relations"]:
        semantic_type = str(row["relation_type"])
        if hot_fact_policy == "seed" and not _is_hot_seed_fact("relation", semantic_type):
            continue
        file_sidecar_ref = file_sidecars.get(str(row["file_path"]))
        fact = {
            "id": row["id"],
            "fact": _relation_fact_text(row, object_lookup),
            "kind": "fact",
            "entities": _entities(
                row,
                fallback=[
                    _payload_name((object_lookup.get(row["from_id"]) or {}).get("payload") or {}, row["from_id"]),
                    _payload_name((object_lookup.get(row["to_id"]) or {}).get("payload") or {}, row["to_id"]),
                ],
            ),
            "tags": _base_tags(row, "relation", row["relation_type"]),
            "source_id": source_id,
            "source_family": "codebase",
            "semantic_kind": "relation",
            "semantic_type": row["relation_type"],
            "repo_id": row["repo_id"],
            "revision": row["revision"],
            "file_path": row["file_path"],
            "span": deepcopy(row["span"]),
            "language": row["language"],
            "analyzer_id": row["analyzer_id"],
            "analyzer_version": row["analyzer_version"],
            "derivation_type": row["derivation_type"],
            "semantic_payload": deepcopy(row["payload"]),
            "metadata": _base_metadata(row, "relation", row["relation_type"], None, file_sidecar_ref),
        }
        if file_sidecar_ref is not None:
            fact["file_sidecar_ref"] = deepcopy(file_sidecar_ref)
        facts.append(fact)

    source_meta = {
        "codebase_context": {
            "repo_id": normalized["provenance"]["repo_id"],
            "revision": normalized["provenance"]["revision"],
            "generated_at": normalized["provenance"]["generated_at"],
            "object_count": len(normalized["objects"]),
            "relation_count": len(normalized["relations"]),
            "sidecar_count": len(persisted_sidecars),
            "languages": sorted({row["language"] for row in normalized["objects"] + normalized["relations"]}),
            "analyzers": sorted({row["analyzer_id"] for row in normalized["objects"] + normalized["relations"]}),
            "capability_report": deepcopy(normalized["capability_report"]),
            "gap_report": deepcopy(normalized["gap_report"]),
            "semantic_sidecars": persisted_sidecars,
        }
    }
    return {
        "bundle": normalized,
        "facts": facts,
        "sidecar_refs": persisted_sidecars,
        "source_meta": source_meta,
    }
