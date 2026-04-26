#!/usr/bin/env python3
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
from copy import deepcopy
from typing import Any

SEMANTIC_OBJECT_TYPES = {
    "module",
    "directory",
    "file",
    "symbol",
    "callable",
    "class",
    "method",
    "interface",
    "protocol",
    "field",
    "parameter",
    "local_variable",
    "type",
    "import",
    "export",
    "callsite",
    "route",
    "command",
    "job",
    "event_handler",
    "guard",
    "policy",
    "role",
    "capability",
    "scope",
    "resource",
    "schema",
    "config",
    "dependency_manifest",
    "lockfile",
    "workflow",
    "workflow_job",
    "test_suite",
    "test_case",
    "fixture",
    "mock",
    "test_run",
    "test_failure",
    "stack_trace",
    "stack_frame",
    "patch_attempt",
    "diff_hunk",
    "verification_result",
}

SEMANTIC_RELATION_TYPES = {
    "declares",
    "imports",
    "exports",
    "resolves_to",
    "calls",
    "extends",
    "implements",
    "overrides",
    "has_type",
    "route_targets",
    "command_targets",
    "job_targets",
    "event_handler_targets",
    "protected_by",
    "uses_policy",
    "requires_role",
    "requires_capability",
    "requires_scope",
    "uses_resource",
    "depends_on_package",
    "configures",
    "workflow_runs",
    "workflow_depends_on",
    "test_covers",
    "fixture_for",
    "mocks",
    "failure_points_to",
    "stack_frame_points_to",
    "hunk_touches_symbol",
    "commit_touches_symbol",
    "pr_touches_symbol",
    "patch_touches",
    "patch_changes_behavior_of",
    "verification_confirms",
    "verification_refutes",
    "issue_mentions",
    "issue_requires",
    "historical_fix_similar_to",
}

SEMANTIC_DERIVATION_TYPES = {"observed", "resolved", "approximate"}

SIDECAR_KINDS = {
    "ast",
    "cfg",
    "dataflow",
    "ssa",
    "scip_index",
    "findings",
    "analyzer_trace",
    "test_trace",
    "semantic_snapshot",
    "codeql_db",
}

REQUIRED_BUNDLE_KEYS = {
    "objects",
    "relations",
    "provenance",
    "capability_report",
    "gap_report",
    "sidecars",
}


def stable_semantic_id(namespace: str, *parts: Any) -> str:
    seed = json.dumps(
        {
            "namespace": namespace,
            "parts": [str(part) for part in parts],
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    digest = hashlib.sha1(seed.encode("utf-8"), usedforsecurity=False).hexdigest()
    return f"{namespace}_{digest[:20]}"


def _require_non_empty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _normalize_span(span: Any, label: str) -> dict[str, int | None]:
    if not isinstance(span, dict):
        raise ValueError(f"{label} must be an object")
    start_line = span.get("start_line")
    end_line = span.get("end_line")
    start_col = span.get("start_col")
    end_col = span.get("end_col")
    if not isinstance(start_line, int) or start_line <= 0:
        raise ValueError(f"{label}.start_line must be a positive integer")
    if not isinstance(end_line, int) or end_line < start_line:
        raise ValueError(f"{label}.end_line must be an integer >= start_line")
    for key, value in (("start_col", start_col), ("end_col", end_col)):
        if value is None:
            continue
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"{label}.{key} must be null or a non-negative integer")
    return {
        "start_line": start_line,
        "end_line": end_line,
        "start_col": start_col if isinstance(start_col, int) else None,
        "end_col": end_col if isinstance(end_col, int) else None,
    }


def _normalize_payload(payload: Any, label: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be an object")
    return deepcopy(payload)


def _normalize_string_list(values: Any, label: str) -> list[str]:
    if values is None:
        return []
    if not isinstance(values, list):
        raise ValueError(f"{label} must be a list of strings")
    normalized: list[str] = []
    for idx, value in enumerate(values):
        if not isinstance(value, str):
            raise ValueError(f"{label}[{idx}] must be a string")
        stripped = value.strip()
        if not stripped:
            raise ValueError(f"{label}[{idx}] must be non-empty")
        normalized.append(stripped)
    return normalized


def _normalize_provenance(provenance: Any) -> dict[str, Any]:
    if not isinstance(provenance, dict):
        raise ValueError("bundle.provenance must be an object")
    normalized = deepcopy(provenance)
    normalized["repo_id"] = _require_non_empty_string(normalized.get("repo_id"), "bundle.provenance.repo_id")
    normalized["revision"] = _require_non_empty_string(normalized.get("revision"), "bundle.provenance.revision")
    normalized["generated_at"] = _require_non_empty_string(
        normalized.get("generated_at"),
        "bundle.provenance.generated_at",
    )
    if "source_root" in normalized and normalized["source_root"] is not None:
        normalized["source_root"] = _require_non_empty_string(
            normalized.get("source_root"),
            "bundle.provenance.source_root",
        )
    return normalized


def _normalize_report(report: Any, label: str) -> dict[str, Any]:
    if not isinstance(report, dict):
        raise ValueError(f"{label} must be an object")
    return deepcopy(report)


def _normalize_object(row: Any, index: int) -> dict[str, Any]:
    label = f"bundle.objects[{index}]"
    if not isinstance(row, dict):
        raise ValueError(f"{label} must be an object")
    object_type = _require_non_empty_string(row.get("object_type"), f"{label}.object_type")
    if object_type not in SEMANTIC_OBJECT_TYPES:
        raise ValueError(f"{label}.object_type '{object_type}' is not supported")
    derivation_type = _require_non_empty_string(row.get("derivation_type"), f"{label}.derivation_type")
    if derivation_type not in SEMANTIC_DERIVATION_TYPES:
        raise ValueError(f"{label}.derivation_type '{derivation_type}' is not supported")
    return {
        "id": _require_non_empty_string(row.get("id"), f"{label}.id"),
        "object_type": object_type,
        "repo_id": _require_non_empty_string(row.get("repo_id"), f"{label}.repo_id"),
        "revision": _require_non_empty_string(row.get("revision"), f"{label}.revision"),
        "file_path": _require_non_empty_string(row.get("file_path"), f"{label}.file_path"),
        "span": _normalize_span(row.get("span"), f"{label}.span"),
        "language": _require_non_empty_string(row.get("language"), f"{label}.language"),
        "analyzer_id": _require_non_empty_string(row.get("analyzer_id"), f"{label}.analyzer_id"),
        "analyzer_version": _require_non_empty_string(row.get("analyzer_version"), f"{label}.analyzer_version"),
        "derivation_type": derivation_type,
        "payload": _normalize_payload(row.get("payload"), f"{label}.payload"),
    }


def _normalize_relation(row: Any, index: int) -> dict[str, Any]:
    label = f"bundle.relations[{index}]"
    if not isinstance(row, dict):
        raise ValueError(f"{label} must be an object")
    relation_type = _require_non_empty_string(row.get("relation_type"), f"{label}.relation_type")
    if relation_type not in SEMANTIC_RELATION_TYPES:
        raise ValueError(f"{label}.relation_type '{relation_type}' is not supported")
    derivation_type = _require_non_empty_string(row.get("derivation_type"), f"{label}.derivation_type")
    if derivation_type not in SEMANTIC_DERIVATION_TYPES:
        raise ValueError(f"{label}.derivation_type '{derivation_type}' is not supported")
    return {
        "id": _require_non_empty_string(row.get("id"), f"{label}.id"),
        "relation_type": relation_type,
        "from_id": _require_non_empty_string(row.get("from_id"), f"{label}.from_id"),
        "to_id": _require_non_empty_string(row.get("to_id"), f"{label}.to_id"),
        "repo_id": _require_non_empty_string(row.get("repo_id"), f"{label}.repo_id"),
        "revision": _require_non_empty_string(row.get("revision"), f"{label}.revision"),
        "file_path": _require_non_empty_string(row.get("file_path"), f"{label}.file_path"),
        "span": _normalize_span(row.get("span"), f"{label}.span"),
        "language": _require_non_empty_string(row.get("language"), f"{label}.language"),
        "analyzer_id": _require_non_empty_string(row.get("analyzer_id"), f"{label}.analyzer_id"),
        "analyzer_version": _require_non_empty_string(row.get("analyzer_version"), f"{label}.analyzer_version"),
        "derivation_type": derivation_type,
        "payload": _normalize_payload(row.get("payload"), f"{label}.payload"),
    }


def _normalize_sidecar(row: Any, index: int) -> dict[str, Any]:
    label = f"bundle.sidecars[{index}]"
    if not isinstance(row, dict):
        raise ValueError(f"{label} must be an object")
    sidecar_kind = _require_non_empty_string(row.get("sidecar_kind"), f"{label}.sidecar_kind")
    if sidecar_kind not in SIDECAR_KINDS:
        raise ValueError(f"{label}.sidecar_kind '{sidecar_kind}' is not supported")
    byte_size = row.get("byte_size")
    if not isinstance(byte_size, int) or byte_size < 0:
        raise ValueError(f"{label}.byte_size must be a non-negative integer")
    normalized = {
        "sidecar_id": _require_non_empty_string(row.get("sidecar_id"), f"{label}.sidecar_id"),
        "sidecar_kind": sidecar_kind,
        "format_family": _require_non_empty_string(row.get("format_family"), f"{label}.format_family"),
        "format_name": _require_non_empty_string(row.get("format_name"), f"{label}.format_name"),
        "format_version": _require_non_empty_string(row.get("format_version"), f"{label}.format_version"),
        "encoding": _require_non_empty_string(row.get("encoding"), f"{label}.encoding"),
        "compression": _require_non_empty_string(row.get("compression"), f"{label}.compression"),
        "repo_id": _require_non_empty_string(row.get("repo_id"), f"{label}.repo_id"),
        "revision": _require_non_empty_string(row.get("revision"), f"{label}.revision"),
        "file_path": _require_non_empty_string(row.get("file_path"), f"{label}.file_path"),
        "span": _normalize_span(row.get("span"), f"{label}.span"),
        "node_id": _require_non_empty_string(row.get("node_id"), f"{label}.node_id"),
        "storage_ref": _require_non_empty_string(row.get("storage_ref"), f"{label}.storage_ref"),
        "content_hash": _require_non_empty_string(row.get("content_hash"), f"{label}.content_hash"),
        "byte_size": byte_size,
        "producer": _require_non_empty_string(row.get("producer"), f"{label}.producer"),
        "metadata": _normalize_payload(row.get("metadata") or {}, f"{label}.metadata"),
    }
    if "payload" in row:
        normalized["payload"] = deepcopy(row["payload"])
    return normalized


def normalize_semantic_bundle(bundle: Any) -> dict[str, Any]:
    if not isinstance(bundle, dict):
        raise ValueError("semantic bundle must be an object")
    missing = REQUIRED_BUNDLE_KEYS - set(bundle)
    if missing:
        raise ValueError(f"semantic bundle missing required keys: {', '.join(sorted(missing))}")

    objects_raw = bundle.get("objects")
    relations_raw = bundle.get("relations")
    sidecars_raw = bundle.get("sidecars")
    if not isinstance(objects_raw, list):
        raise ValueError("bundle.objects must be a list")
    if not isinstance(relations_raw, list):
        raise ValueError("bundle.relations must be a list")
    if not isinstance(sidecars_raw, list):
        raise ValueError("bundle.sidecars must be a list")

    objects = [_normalize_object(row, index) for index, row in enumerate(objects_raw)]
    relations = [_normalize_relation(row, index) for index, row in enumerate(relations_raw)]
    sidecars = [_normalize_sidecar(row, index) for index, row in enumerate(sidecars_raw)]

    object_ids = {row["id"] for row in objects}
    relation_ids = set()
    for relation in relations:
        if relation["id"] in relation_ids:
            raise ValueError(f"duplicate relation id: {relation['id']}")
        relation_ids.add(relation["id"])
        if relation["from_id"] not in object_ids:
            raise ValueError(f"relation {relation['id']} references unknown from_id {relation['from_id']}")
        if relation["to_id"] not in object_ids:
            raise ValueError(f"relation {relation['id']} references unknown to_id {relation['to_id']}")

    sidecar_ids = set()
    for sidecar in sidecars:
        if sidecar["sidecar_id"] in sidecar_ids:
            raise ValueError(f"duplicate sidecar id: {sidecar['sidecar_id']}")
        sidecar_ids.add(sidecar["sidecar_id"])
        if sidecar["node_id"] not in object_ids:
            raise ValueError(f"sidecar {sidecar['sidecar_id']} references unknown node_id {sidecar['node_id']}")

    return {
        "objects": objects,
        "relations": relations,
        "provenance": _normalize_provenance(bundle.get("provenance")),
        "capability_report": _normalize_report(bundle.get("capability_report"), "bundle.capability_report"),
        "gap_report": _normalize_report(bundle.get("gap_report"), "bundle.gap_report"),
        "sidecars": sidecars,
    }


def merge_semantic_bundles(bundles: list[dict[str, Any]]) -> dict[str, Any]:
    if not bundles:
        raise ValueError("at least one semantic bundle is required")
    normalized_bundles = [normalize_semantic_bundle(bundle) for bundle in bundles]
    objects: dict[str, dict[str, Any]] = {}
    relations: dict[str, dict[str, Any]] = {}
    sidecars: dict[str, dict[str, Any]] = {}
    languages: set[str] = set()
    analyzers: set[str] = set()
    capability_languages: set[str] = set()
    capability_values: set[str] = set()
    skipped_files: list[str] = []
    notes: list[str] = []

    for bundle in normalized_bundles:
        for row in bundle["objects"]:
            objects[row["id"]] = row
            languages.add(row["language"])
            analyzers.add(row["analyzer_id"])
        for row in bundle["relations"]:
            relations[row["id"]] = row
            languages.add(row["language"])
            analyzers.add(row["analyzer_id"])
        for row in bundle["sidecars"]:
            sidecars[row["sidecar_id"]] = row
        capability_report = bundle.get("capability_report") or {}
        capability_languages.update(_normalize_string_list(capability_report.get("supported_languages"), "capability_report.supported_languages"))
        capability_values.update(_normalize_string_list(capability_report.get("supported_capabilities"), "capability_report.supported_capabilities"))
        gap_report = bundle.get("gap_report") or {}
        skipped_files.extend(_normalize_string_list(gap_report.get("skipped_files"), "gap_report.skipped_files"))
        notes.extend(_normalize_string_list(gap_report.get("notes"), "gap_report.notes"))

    base = normalized_bundles[0]
    provenance = dict(base["provenance"])
    provenance["languages"] = sorted(languages)
    provenance["analyzers"] = sorted(analyzers)

    capability_report = dict(base.get("capability_report") or {})
    capability_report["supported_languages"] = sorted(capability_languages or languages)
    capability_report["supported_capabilities"] = sorted(capability_values)

    gap_report = dict(base.get("gap_report") or {})
    gap_report["skipped_files"] = list(dict.fromkeys(skipped_files))
    gap_report["notes"] = list(dict.fromkeys(notes))

    return normalize_semantic_bundle(
        {
            "objects": list(objects.values()),
            "relations": list(relations.values()),
            "provenance": provenance,
            "capability_report": capability_report,
            "gap_report": gap_report,
            "sidecars": list(sidecars.values()),
        }
    )
