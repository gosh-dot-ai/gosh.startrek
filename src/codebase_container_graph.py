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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .codebase_semantic_bundle import normalize_semantic_bundle
from .container_graph import empty_container_graph, normalize_container_graph, stable_hash

CODEBASE_GRAPH_ADAPTER = "codebase_semantic_container_graph"
CODEBASE_GRAPH_ADAPTER_VERSION = "1"
CODEBASE_GRAPH_PROFILE_ID = "repo_task:v1"
CODEBASE_GRAPH_CONTRACT_VERSION = "repo-container-contract-v1"
CODEBASE_REF_SCHEMA_VERSION = "repo-container-ref-v1"
CODEBASE_RENDER_REF_SCHEMA_VERSION = "repo-container-render-ref-v1"
CONTEXT_PACK_CODE_CANDIDATE_LIMIT = 256
CONTEXT_PACK_TEST_LIMIT = 128
CONTEXT_PACK_COMMAND_LIMIT = 64
CONTEXT_PACK_WORKFLOW_LIMIT = 64
CONTEXT_PACK_RELATION_LIMIT = 512

OBJECT_KIND_MAP = {
    "directory": ("code", "directory"),
    "file": ("code", "file"),
    "module": ("code", "file"),
    "symbol": ("code", "symbol"),
    "callable": ("code", "function"),
    "class": ("code", "class"),
    "method": ("code", "method"),
    "interface": ("code", "interface"),
    "protocol": ("code", "interface"),
    "field": ("code", "type"),
    "parameter": ("code", "symbol"),
    "local_variable": ("code", "symbol"),
    "type": ("code", "type"),
    "import": ("code", "import"),
    "export": ("code", "export"),
    "callsite": ("code", "callsite"),
    "route": ("code", "route"),
    "command": ("code", "command"),
    "job": ("code", "workflow_job"),
    "event_handler": ("code", "symbol"),
    "guard": ("code", "symbol"),
    "policy": ("code", "config"),
    "role": ("code", "config"),
    "capability": ("code", "config"),
    "scope": ("code", "config"),
    "resource": ("code", "dependency_manifest"),
    "schema": ("code", "schema"),
    "config": ("code", "config"),
    "dependency_manifest": ("code", "dependency_manifest"),
    "lockfile": ("code", "lockfile"),
    "workflow": ("code", "workflow"),
    "workflow_job": ("code", "workflow_job"),
    "test_suite": ("code", "test_suite"),
    "test_case": ("code", "test_case"),
    "fixture": ("code", "fixture"),
    "mock": ("code", "mock"),
    "test_run": ("operation", "test_run"),
    "test_failure": ("operation", "test_failure"),
    "stack_trace": ("operation", "stack_trace"),
    "stack_frame": ("operation", "stack_frame"),
    "patch_attempt": ("operation", "patch_attempt"),
    "diff_hunk": ("operation", "diff_hunk"),
    "verification_result": ("operation", "verification_result"),
}

RELATION_CAPABILITY_MAP = {
    "declares": "declares",
    "imports": "imports",
    "exports": "exports",
    "resolves_to": "depends_on",
    "calls": "calls",
    "extends": "depends_on",
    "implements": "depends_on",
    "overrides": "depends_on",
    "has_type": "depends_on",
    "route_targets": "command_targets",
    "command_targets": "command_targets",
    "job_targets": "workflow_runs",
    "event_handler_targets": "calls",
    "protected_by": "depends_on",
    "uses_policy": "depends_on",
    "requires_role": "depends_on",
    "requires_capability": "depends_on",
    "requires_scope": "depends_on",
    "uses_resource": "depends_on",
    "depends_on_package": "depends_on_package",
    "configures": "configures",
    "workflow_runs": "workflow_runs",
    "workflow_depends_on": "workflow_depends_on",
    "test_covers": "test_covers",
    "fixture_for": "fixture_for",
    "mocks": "mocks",
    "failure_points_to": "failure_points_to",
    "stack_frame_points_to": "stack_frame_points_to",
    "hunk_touches_symbol": "patch_touches",
    "commit_touches_symbol": "patch_touches",
    "pr_touches_symbol": "patch_touches",
    "patch_touches": "patch_touches",
    "patch_changes_behavior_of": "patch_changes_behavior_of",
    "verification_confirms": "verification_confirms",
    "verification_refutes": "verification_refutes",
    "issue_mentions": "issue_mentions",
    "issue_requires": "issue_requires",
    "historical_fix_similar_to": "historical_fix_similar_to",
    "has_patch_attempt": "has_patch_attempt",
    "verified_by": "verified_by",
    "runs_test": "runs_test",
}

UNIVERSAL_RELATION_CAPABILITIES = sorted(
    {
        "declares",
        "imports",
        "exports",
        "calls",
        "called_by",
        "owns",
        "owns_file",
        "owns_symbol",
        "depends_on",
        "depends_on_package",
        "configures",
        "generates",
        "generated_by",
        "command_targets",
        "workflow_runs",
        "workflow_depends_on",
        "test_covers",
        "fixture_for",
        "mocks",
        "failure_points_to",
        "stack_frame_points_to",
        "patch_touches",
        "patch_changes_behavior_of",
        "verification_confirms",
        "verification_refutes",
        "issue_mentions",
        "issue_requires",
        "historical_fix_similar_to",
        "has_patch_attempt",
        "verified_by",
        "runs_test",
    }
)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_json_dumps(value).encode("utf-8")).hexdigest()


def _payload_name(payload: dict[str, Any], fallback: str) -> str:
    for key in ("qualified_name", "name", "callee_name", "import_path", "signature"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return fallback


def _source_record_work_item(source_record: dict[str, Any]) -> dict[str, Any]:
    metadata_raw = source_record.get("metadata")
    metadata: dict[str, Any] = metadata_raw if isinstance(metadata_raw, dict) else {}
    source_meta_raw = source_record.get("source_meta")
    source_meta: dict[str, Any] = source_meta_raw if isinstance(source_meta_raw, dict) else {}
    work_item_raw = metadata.get("work_item")
    raw: dict[str, Any] = work_item_raw if isinstance(work_item_raw, dict) else {}
    if not raw:
        source_work_item_raw = source_meta.get("work_item")
        raw = source_work_item_raw if isinstance(source_work_item_raw, dict) else {}
    if not raw:
        raw = {
            key: metadata.get(f"work_item_{key}")
            for key in (
                "work_item_id",
                "work_item_kind",
                "repo",
                "base_commit",
                "problem_statement",
                "requirements",
                "interface",
                "selected_test_files_to_run",
                "before_repo_set_cmd",
            )
            if metadata.get(f"work_item_{key}") is not None
        }

    allowed_fields = {
        "work_item_id",
        "work_item_kind",
        "repo",
        "base_commit",
        "problem_statement",
        "requirements",
        "interface",
        "selected_test_files_to_run",
        "before_repo_set_cmd",
    }
    work_item: dict[str, Any] = {}
    for key in allowed_fields:
        value = raw.get(key)
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            rendered = str(value).strip()
            if rendered:
                work_item[key] = rendered
        elif isinstance(value, list):
            normalized = [str(item).strip() for item in value if str(item).strip()]
            if normalized:
                work_item[key] = normalized
    return work_item


def _line_scope(repo_id: str, revision: str, path: str) -> str:
    return f"{repo_id}:{revision}:{path}"


def _content_hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_file_text(repo_root: Path | None, file_path: str) -> tuple[str | None, str | None, str | None]:
    if repo_root is None:
        return None, None, None
    try:
        path = (repo_root / file_path).resolve()
        root = repo_root.resolve()
        path.relative_to(root)
        raw = path.read_bytes()
    except Exception:
        return None, None, None
    newline_mode = "crlf" if b"\r\n" in raw else "lf"
    try:
        text = raw.decode("utf-8")
        encoding = "utf-8"
    except UnicodeDecodeError:
        text = raw.decode("utf-8", errors="replace")
        encoding = "utf-8-replacement"
    return text, encoding, newline_mode


def _line_offsets(text: str) -> list[int]:
    offsets = [0]
    for idx, char in enumerate(text):
        if char == "\n":
            offsets.append(idx + 1)
    return offsets


def _line_span_to_char_offsets(text: str, span: dict[str, Any]) -> tuple[int, int]:
    offsets = _line_offsets(text)
    start_line = max(1, int(span.get("start_line") or 1))
    end_line = max(start_line, int(span.get("end_line") or start_line))
    start_col = span.get("start_col")
    end_col = span.get("end_col")
    start = offsets[start_line - 1] if start_line - 1 < len(offsets) else len(text)
    if isinstance(start_col, int) and start_col > 0:
        start += start_col
    if end_line < len(offsets):
        end = offsets[end_line] - (1 if offsets[end_line] > 0 and text[offsets[end_line] - 1] == "\n" else 0)
    else:
        end = len(text)
    if isinstance(end_col, int) and end_col > 0 and end_line - 1 < len(offsets):
        end = offsets[end_line - 1] + end_col
    return max(0, min(start, len(text))), max(0, min(max(start, end), len(text)))


def _container_row(
    *,
    graph_revision_id: str,
    source_id: str,
    logical_source_id: str,
    revision_id: str,
    repo_id: str,
    content_revision_id: str | None,
    kind_ns: str,
    kind: str,
    traits: dict[str, Any],
    span_refs: list[dict[str, Any]],
    order_segments: list[Any],
    primary_render_ref_id: str | None,
    primary_render_ref_fingerprint: str | None,
    render_ref_json: dict[str, Any],
    record: dict[str, Any],
    boundary_score: float = 1.0,
    render_score: float = 1.0,
) -> dict[str, Any]:
    kind_fq = f"{kind_ns}:{kind}"
    container_id = stable_hash(
        "codebase-container-id-v1",
        repo_id,
        revision_id,
        kind_fq,
        traits.get("semantic_id") or traits.get("path") or traits.get("name") or span_refs,
        prefix="ctr",
    )
    order_scope_id = f"{repo_id}:{revision_id}"
    order_key = {
        "basis": "repo_source_order",
        "scope": "repo_revision",
        "scope_id": order_scope_id,
        "segments": order_segments,
        "unit": "container",
    }
    return {
        "container_id": container_id,
        "id_origin": "stable_hash",
        "id_schema_version": "container-id-v1",
        "identity_scope": "repo_revision",
        "source_id": source_id,
        "logical_source_id": logical_source_id,
        "family": "codebase",
        "revision_id": revision_id,
        "revision_scope": "repo_revision",
        "content_revision_id": content_revision_id,
        "external_revision_id": revision_id,
        "container_graph_revision_id": graph_revision_id,
        "kind_ns": kind_ns,
        "kind": kind,
        "kind_fq": kind_fq,
        "kind_version": "v1",
        "traits_json": traits,
        "order_key_json": order_key,
        "order_basis": "repo_source_order",
        "order_scope": "repo_revision",
        "order_scope_id": order_scope_id,
        "span_refs_json": span_refs,
        "episode_ids_json": [],
        "primary_render_ref_id": primary_render_ref_id,
        "primary_render_ref_fingerprint": primary_render_ref_fingerprint,
        "render_ref_json": render_ref_json,
        "status": "active",
        "supersedes_container_id": None,
        "superseded_by_container_id": None,
        "boundary_score": boundary_score,
        "kind_score": 1.0,
        "render_score": render_score,
        "acl_inherit_source": 1,
        "owner_id": record.get("owner_id"),
        "scope": record.get("scope"),
        "agent_id": record.get("agent_id"),
        "swarm_id": record.get("swarm_id"),
        "read_json": list(record.get("read") or []),
        "write_json": list(record.get("write") or []),
        "acl_source_id": source_id,
        "acl_policy": "inherit_source_record",
        "adapter_name": CODEBASE_GRAPH_ADAPTER,
        "adapter_version": CODEBASE_GRAPH_ADAPTER_VERSION,
        "inference_version": None,
        "created_at": _utcnow_iso(),
        "updated_at": _utcnow_iso(),
    }


def _render_ref_row(
    *,
    graph_revision_id: str,
    container_id: str,
    render_kind: str,
    render_mode: str,
    ref_json: dict[str, Any],
    fidelity: str,
    language: str,
    fmt: str = "text",
) -> dict[str, Any]:
    ref_fingerprint = _fingerprint({"render_kind": render_kind, "render_mode": render_mode, **ref_json})
    return {
        "render_ref_id": stable_hash(
            CODEBASE_RENDER_REF_SCHEMA_VERSION,
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


def _ref_row(
    *,
    graph_revision_id: str,
    container_id: str | None,
    ref_role: str,
    ref_type: str,
    ref_json: dict[str, Any],
    source_id: str,
    logical_source_id: str,
    revision_id: str,
    coverage_required: bool,
    coverage_status: str = "mapped",
) -> dict[str, Any]:
    ref_fingerprint = _fingerprint({"ref_type": ref_type, **ref_json})
    return {
        "ref_id": stable_hash(
            CODEBASE_REF_SCHEMA_VERSION,
            graph_revision_id,
            container_id or "unmapped",
            ref_role,
            ref_type,
            ref_fingerprint,
            prefix="ref",
        ),
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
        "coverage_status": coverage_status,
        "status": "active",
        "created_at": _utcnow_iso(),
    }


def _lookup_row(ref_id: str, lookup_ns: str, lookup_key: str, *, value_text: str | None = None, value_int: int | None = None) -> dict[str, Any]:
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


def _range_row(ref_id: str, range_kind: str, range_scope: str, start_value: int, end_value: int) -> dict[str, Any]:
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


def _relation_row(
    *,
    graph_revision_id: str,
    src_container_id: str,
    dst_container_id: str,
    relation_kind: str,
    traits: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "relation_id": stable_hash(
            "container-relation-v1",
            graph_revision_id,
            src_container_id,
            dst_container_id,
            "codebase",
            relation_kind,
            prefix="rel",
        ),
        "container_graph_revision_id": graph_revision_id,
        "src_container_id": src_container_id,
        "dst_container_id": dst_container_id,
        "src_container_graph_revision_id": graph_revision_id,
        "dst_container_graph_revision_id": graph_revision_id,
        "relation_ns": "codebase",
        "relation_kind": relation_kind,
        "order_key_json": {},
        "traits_json": traits or {},
        "relation_score": 1.0,
        "status": "active",
        "created_at": _utcnow_iso(),
        "updated_at": _utcnow_iso(),
    }


def _contract_row(
    *,
    contract_kind: str,
    payload_schema_id: str,
    payload: dict[str, Any],
    family: str = "codebase",
    profile_id: str = CODEBASE_GRAPH_PROFILE_ID,
    subject_kind: str | None = None,
    subject_id: str | None = None,
    status: str = "active",
) -> dict[str, Any]:
    payload_fingerprint = _fingerprint(payload)
    return {
        "contract_id": stable_hash(
            "container-contract-v1",
            contract_kind,
            family,
            profile_id,
            subject_kind or "",
            subject_id or "",
            payload_fingerprint,
            prefix="contract",
        ),
        "contract_kind": contract_kind,
        "payload_schema_id": payload_schema_id,
        "family": family,
        "profile_id": profile_id,
        "profile_version": "v1",
        "contract_version": CODEBASE_GRAPH_CONTRACT_VERSION,
        "subject_kind": subject_kind,
        "subject_id": subject_id,
        "payload_json": payload,
        "payload_fingerprint": payload_fingerprint,
        "status": status,
        "created_at": _utcnow_iso(),
        "updated_at": _utcnow_iso(),
    }


def _artifact_row(
    *,
    artifact_kind: str,
    graph_revision_id: str,
    payload: dict[str, Any],
    subject_type: str | None = None,
    subject_id: str | None = None,
    query_id: str | None = None,
    profile_id: str = CODEBASE_GRAPH_PROFILE_ID,
    status: str = "active",
) -> dict[str, Any]:
    payload_fingerprint = _fingerprint(payload)
    return {
        "artifact_id": stable_hash(
            "container-artifact-v1",
            artifact_kind,
            graph_revision_id,
            subject_type or "",
            subject_id or "",
            query_id or "",
            payload_fingerprint,
            prefix="artifact",
        ),
        "artifact_kind": artifact_kind,
        "artifact_schema_version": "v1",
        "container_graph_revision_id": graph_revision_id,
        "container_graph_revision_ids_json": [graph_revision_id],
        "families_json": ["codebase", "repo", "operation"],
        "profile_ids_json": [profile_id],
        "query_id": query_id,
        "profile_id": profile_id,
        "subject_type": subject_type,
        "subject_id": subject_id,
        "payload_json": payload,
        "payload_fingerprint": payload_fingerprint,
        "status": status,
        "created_at": _utcnow_iso(),
    }


def _semantic_ref_type(row: dict[str, Any], kind_fq: str) -> str:
    object_type = str(row.get("object_type") or "")
    if kind_fq == "code:file":
        return "file_line_range"
    if object_type in {"class", "callable", "symbol", "interface", "protocol", "type"}:
        return "lsp_symbol"
    if object_type == "test_case":
        return "test_case"
    if object_type == "command":
        return "command"
    if object_type in {"job", "workflow_job"}:
        return "workflow_job"
    return "file_line_range"


def _semantic_ref_json(row: dict[str, Any], ref_type: str) -> dict[str, Any]:
    payload = dict(row.get("payload") or {})
    repo_id = str(row["repo_id"])
    revision = str(row["revision"])
    path = str(row["file_path"])
    span = dict(row["span"])
    base = {
        "schema_version": "v1",
        "repo_id": repo_id,
        "commit_id": revision,
        "path": path,
        "span": span,
    }
    if ref_type == "lsp_symbol":
        name = _payload_name(payload, str(row["id"]))
        base.update(
            {
                "fully_qualified_name": str(payload.get("qualified_name") or name),
                "lsp_symbol_uri": f"{repo_id}:{revision}:{path}:{name}",
            }
        )
    elif ref_type == "test_case":
        base["test_case_id"] = str(row["id"])
    elif ref_type == "workflow_job":
        base["workflow_id"] = str(payload.get("workflow_id") or path)
        base["workflow_job_id"] = str(row["id"])
    elif ref_type == "command":
        base["command_id"] = str(row["id"])
        base["command"] = str(payload.get("command") or payload.get("name") or "")
        base["cwd"] = str(payload.get("cwd") or "")
    return base


def _file_char_ref_json(row: dict[str, Any], char_range: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "v1",
        "repo_id": str(row["repo_id"]),
        "commit_id": str(row["revision"]),
        "path": str(row["file_path"]),
        "span": dict(row["span"]),
        "char_start": int(char_range["start"]),
        "char_end": int(char_range["end"]),
    }


def _ast_node_ref_json(row: dict[str, Any]) -> dict[str, Any]:
    payload = dict(row.get("payload") or {})
    return {
        "schema_version": "v1",
        "repo_id": str(row["repo_id"]),
        "commit_id": str(row["revision"]),
        "path": str(row["file_path"]),
        "ast_node_id": str(row["id"]),
        "node_kind": str(row.get("object_type") or ""),
        "fully_qualified_name": str(payload.get("qualified_name") or _payload_name(payload, str(row["id"]))),
        "span": dict(row["span"]),
    }


def _add_lookup_projection(graph: dict[str, list[dict]], ref: dict[str, Any], ref_json: dict[str, Any], ref_type: str) -> None:
    repo_id = str(ref_json.get("repo_id") or "")
    commit_id = str(ref_json.get("commit_id") or "")
    path = str(ref_json.get("path") or "")
    if repo_id:
        graph["ref_lookup"].append(_lookup_row(ref["ref_id"], ref_type, "repo_id", value_text=repo_id))
    if commit_id:
        graph["ref_lookup"].append(_lookup_row(ref["ref_id"], ref_type, "commit_id", value_text=commit_id))
    if path:
        graph["ref_lookup"].append(_lookup_row(ref["ref_id"], ref_type, "path", value_text=path))
    for key in (
        "fully_qualified_name",
        "lsp_symbol_uri",
        "test_case_id",
        "workflow_id",
        "workflow_job_id",
        "command_id",
        "command",
        "ast_node_id",
    ):
        value = str(ref_json.get(key) or "").strip()
        if value:
            graph["ref_lookup"].append(_lookup_row(ref["ref_id"], ref_type, key, value_text=value))
    span = ref_json.get("span") or {}
    if isinstance(span, dict) and path:
        start_line = int(span.get("start_line") or 1)
        end_line = int(span.get("end_line") or start_line)
        graph["ref_ranges"].append(_range_row(ref["ref_id"], "line", _line_scope(repo_id, commit_id, path), start_line, end_line))


def _render_ref_for_object(
    *,
    graph_revision_id: str,
    container_id: str,
    row: dict[str, Any],
    repo_root: Path | None,
    file_text_cache: dict[str, tuple[str | None, str | None, str | None]],
) -> tuple[dict[str, Any] | None, dict[str, Any], dict[str, Any] | None]:
    file_path = str(row["file_path"])
    if file_path not in file_text_cache:
        file_text_cache[file_path] = _read_file_text(repo_root, file_path)
    text, encoding, newline_mode = file_text_cache[file_path]
    if text is None:
        return None, {"render_source": "repo_file_missing", "raw_source_present": False}, None
    start, end = _line_span_to_char_offsets(text, dict(row["span"]))
    render_text = text[start:end]
    ref_json = {
        "ref_type": "repo_file_span",
        "repo_id": str(row["repo_id"]),
        "commit_id": str(row["revision"]),
        "path": str(row["file_path"]),
        "span": dict(row["span"]),
        "char_start": start,
        "char_end": end,
        "encoding": encoding,
        "newline_mode": newline_mode,
        "byte_exact": encoding == "utf-8",
        "text_exact": True,
        "render_source": "repo_file_blob_span",
        "raw_source_present": True,
        "raw_source_provenance": "repo_worktree_original_file",
        "text_hash": _content_hash_text(render_text),
        "text": render_text,
    }
    render = _render_ref_row(
        graph_revision_id=graph_revision_id,
        container_id=container_id,
        render_kind="source_text",
        render_mode="exact_copy",
        ref_json=ref_json,
        fidelity="text_exact" if encoding == "utf-8" else "text_lossy_decode",
        language=str(row.get("language") or "source"),
    )
    return render, ref_json, {"start": start, "end": end}


def _build_coverage_payload(
    *,
    normalized: dict[str, Any],
    graph: dict[str, list[dict]],
    missing_render_objects: list[str],
    relation_capabilities_used: list[str],
) -> dict[str, Any]:
    object_count = len(normalized["objects"])
    relation_count = len(normalized["relations"])
    required_refs = [row for row in graph["refs"] if int(row.get("coverage_required") or 0)]
    unresolved_refs = [row for row in required_refs if row.get("coverage_status") != "mapped"]
    supported_languages = list((normalized.get("capability_report") or {}).get("supported_languages") or [])
    supported_capabilities = list((normalized.get("capability_report") or {}).get("supported_capabilities") or [])
    gap_report = deepcopy(normalized.get("gap_report") or {})
    skipped_files = [
        str(path)
        for path in (gap_report.get("skipped_files") or [])
        if str(path or "").strip()
    ]
    gap_notes = [
        str(note)
        for note in (gap_report.get("notes") or [])
        if str(note or "").strip()
    ]
    represented_gap_files = [
        path
        for path in skipped_files
        if any(path in note for note in gap_notes)
    ]
    unrepresented_gap_files = sorted(set(skipped_files) - set(represented_gap_files))
    explicitly_absent = []
    for capability in ("workflow_semantics", "command_execution_history", "patch_attempt_history", "verification_history"):
        if capability not in supported_capabilities:
            explicitly_absent.append({"capability": capability, "status": "explicitly_absent"})
    gap_coverage = {
        "unsupported_file_count": len(skipped_files),
        "represented_gap_count": len(represented_gap_files),
        "unrepresented_gap_count": len(unrepresented_gap_files),
        "unrepresented_gap_files": unrepresented_gap_files,
        "status": "explicitly_reported" if not unrepresented_gap_files else "failed_closed",
        "policy": "unsupported files must be represented by gap_report notes, not silently ignored",
    }
    silent_absence_count = len(unrepresented_gap_files)
    status = "ok"
    if unresolved_refs or missing_render_objects or silent_absence_count:
        status = "failed_closed"
    return {
        "coverage_report_kind": "repo_container_coverage_report",
        "profile_id": CODEBASE_GRAPH_PROFILE_ID,
        "object_count": object_count,
        "relation_count": relation_count,
        "container_count": len(graph["containers"]),
        "required_ref_count": len(required_refs),
        "unresolved_required_ref_count": len(unresolved_refs),
        "render_ref_count": len(graph["render_refs"]),
        "missing_render_object_ids": missing_render_objects,
        "supported_languages": supported_languages,
        "supported_capabilities": supported_capabilities,
        "relation_capabilities_used": sorted(set(relation_capabilities_used)),
        "gap_report": gap_report,
        "gap_coverage": gap_coverage,
        "explicitly_absent_capabilities": explicitly_absent,
        "silent_absence_count": silent_absence_count,
        "status": status,
    }


def _validator_payload(coverage_payload: dict[str, Any]) -> dict[str, Any]:
    errors = []
    if int(coverage_payload.get("unresolved_required_ref_count") or 0):
        errors.append(
            {
                "code": "UNRESOLVED_REQUIRED_REFS",
                "count": coverage_payload.get("unresolved_required_ref_count"),
            }
        )
    if coverage_payload.get("missing_render_object_ids"):
        errors.append(
            {
                "code": "MISSING_EXACT_RENDER_REFS",
                "object_ids": list(coverage_payload.get("missing_render_object_ids") or []),
            }
        )
    if int(coverage_payload.get("silent_absence_count") or 0):
        errors.append({"code": "SILENT_ABSENCE", "count": coverage_payload.get("silent_absence_count")})
    return {
        "validator_report_kind": "repo_container_validator_report",
        "profile_id": CODEBASE_GRAPH_PROFILE_ID,
        "ok": not errors,
        "errors": errors,
        "coverage_status": coverage_payload.get("status"),
    }


def _context_pack_payload(
    *,
    repo_container_id: str,
    work_item_container_id: str,
    graph_revision_id: str,
    coverage_artifact_id: str,
    containers: list[dict[str, Any]],
    relation_rows: list[dict[str, Any]],
    work_item: dict[str, Any] | None = None,
) -> dict[str, Any]:
    code_candidates: list[dict[str, Any]] = []
    tests: list[dict[str, Any]] = []
    commands: list[dict[str, Any]] = []
    workflows: list[dict[str, Any]] = []
    rejected_candidates: list[dict[str, Any]] = []
    total_containers = 0
    total_relations = 0
    for container in containers:
        if container.get("container_graph_revision_id") != graph_revision_id:
            continue
        total_containers += 1
        traits = dict(container.get("traits_json") or {})
        item = {
            "container_id": container["container_id"],
            "kind_fq": container["kind_fq"],
            "name": traits.get("name") or traits.get("qualified_name") or traits.get("path"),
            "path": traits.get("path"),
            "trace": {
                "reason": "codebase_semantic_container",
                "source": "container_graph",
                "capabilities": traits.get("capabilities") or [],
            },
        }
        if container["kind_fq"] == "code:test_case":
            if len(tests) < CONTEXT_PACK_TEST_LIMIT:
                tests.append(item)
        elif container["kind_fq"] == "code:command":
            if len(commands) < CONTEXT_PACK_COMMAND_LIMIT:
                commands.append(item)
        elif container["kind_fq"] == "code:workflow_job":
            if len(workflows) < CONTEXT_PACK_WORKFLOW_LIMIT:
                workflows.append(item)
        elif str(container["kind_fq"]).startswith("code:"):
            if len(code_candidates) < CONTEXT_PACK_CODE_CANDIDATE_LIMIT:
                code_candidates.append(item)
        else:
            if len(rejected_candidates) < 32:
                rejected_candidates.append(
                    {
                        **item,
                        "trace": {
                            "reason": "not_selected_for_default_repo_task_pack",
                            "source": "container_graph",
                        },
                    }
                )
    dependency_neighborhood: list[dict[str, Any]] = []
    for row in relation_rows:
        if row.get("container_graph_revision_id") != graph_revision_id:
            continue
        total_relations += 1
        if len(dependency_neighborhood) >= CONTEXT_PACK_RELATION_LIMIT:
            continue
        dependency_neighborhood.append(
            {
                "relation_id": row["relation_id"],
                "relation_capability": (row.get("traits_json") or {}).get("relation_capability"),
                "src_container_id": row["src_container_id"],
                "dst_container_id": row["dst_container_id"],
                "trace": {"reason": "typed_relation_expansion"},
            }
        )
    if total_containers > len(code_candidates) + len(tests) + len(commands) + len(workflows) + len(rejected_candidates):
        rejected_candidates.append(
            {
                "trace": {
                    "reason": "not_materialized_into_initial_context_pack_due_to_budget",
                    "source": "container_graph",
                    "total_containers": total_containers,
                    "selected_code_candidates": len(code_candidates),
                    "selected_tests": len(tests),
                    "selected_commands": len(commands),
                    "selected_workflows": len(workflows),
                },
            }
        )
    work_item_payload = dict(work_item or {})
    requirement_items: list[dict[str, Any]] = []
    for field in ("problem_statement", "requirements", "interface", "selected_test_files_to_run", "before_repo_set_cmd"):
        value = work_item_payload.get(field)
        if not value:
            continue
        if isinstance(value, list):
            rendered = "\n".join(str(item).strip() for item in value if str(item).strip())
        else:
            rendered = str(value).strip()
        if rendered:
            requirement_items.append(
                {
                    "kind": field,
                    "text": rendered,
                    "trace": {"reason": "work_item_metadata", "source": "source_record.metadata.work_item"},
                }
            )

    return {
        "context_pack_kind": "repo_task_context_pack",
        "work_item": {
            "container_id": work_item_container_id,
            "repo_container_id": repo_container_id,
            "graph_revision_id": graph_revision_id,
            "work_item_id": work_item_payload.get("work_item_id"),
            "work_item_kind": work_item_payload.get("work_item_kind", "repository_memory_scope"),
            "trace": {"reason": "repository_work_item_scope" if work_item_payload else "default_repository_work_item_scope"},
        },
        "sections": {
            "requirements": requirement_items,
            "mentioned_code": [],
            "semantic_code_candidates": code_candidates,
            "dependency_neighborhood": dependency_neighborhood,
            "tests": tests,
            "commands": commands,
            "workflows": workflows,
            "failures": [],
            "docs": [],
            "conversation": [],
            "history": [],
            "patch_attempts": [],
            "verification": [],
            "rejected_candidates": rejected_candidates,
        },
        "coverage_report_refs": [coverage_artifact_id],
        "risk_report": {"status": "not_generated", "reason": "no_patch_attempt"},
        "completeness_report": {"status": "initial_pack", "silent_absence": False},
        "trace": {
            "seed_domain_policy": "facts_and_query_are_seed_only",
            "operator_domain_policy": "full_typed_container_domain_in_scope",
            "planner": "universal_repo_task_context_pack",
            "candidate_domain_policy": "bounded_initial_materialized_pack",
            "total_operator_domain_containers": total_containers,
            "total_operator_domain_relations": total_relations,
            "context_pack_limits": {
                "semantic_code_candidates": CONTEXT_PACK_CODE_CANDIDATE_LIMIT,
                "tests": CONTEXT_PACK_TEST_LIMIT,
                "commands": CONTEXT_PACK_COMMAND_LIMIT,
                "workflows": CONTEXT_PACK_WORKFLOW_LIMIT,
                "dependency_neighborhood": CONTEXT_PACK_RELATION_LIMIT,
            },
        },
    }


def build_codebase_container_graph(
    bundle: dict[str, Any],
    *,
    source_id: str,
    source_record: dict[str, Any],
    repo_root: str | Path | None = None,
    existing_graph: dict | None = None,
) -> dict[str, list[dict]]:
    """Materialize a normalized Repository context bundle into generic container rows."""

    normalized = normalize_semantic_bundle(bundle)
    graph = normalize_container_graph(existing_graph)
    graph = _drop_existing_codebase_source_rows(graph, source_id)
    repo_id = str(normalized["provenance"]["repo_id"])
    revision = str(normalized["provenance"]["revision"])
    logical_source_id = str((source_record.get("source_meta") or {}).get("logical_source_id") or source_id)
    content_revision_id = str(source_record.get("content_hash") or "") or None
    input_fingerprint = _fingerprint(
        {
            "repo_id": repo_id,
            "revision": revision,
            "object_ids": [row["id"] for row in normalized["objects"]],
            "relation_ids": [row["id"] for row in normalized["relations"]],
            "capability_report": normalized.get("capability_report") or {},
            "gap_report": normalized.get("gap_report") or {},
        }
    )
    graph_revision_id = stable_hash(
        "container-graph-revision-v1",
        source_id,
        revision,
        "codebase",
        CODEBASE_GRAPH_ADAPTER,
        CODEBASE_GRAPH_ADAPTER_VERSION,
        input_fingerprint,
        prefix="cgr",
    )
    repo_root_path = Path(repo_root).expanduser().resolve() if repo_root else None
    now = _utcnow_iso()

    graph["graph_revisions"].append(
        {
            "container_graph_revision_id": graph_revision_id,
            "source_id": source_id,
            "logical_source_id": logical_source_id,
            "family": "codebase",
            "revision_id": revision,
            "revision_scope": "repo_revision",
            "content_revision_id": content_revision_id,
            "external_revision_id": revision,
            "adapter_name": CODEBASE_GRAPH_ADAPTER,
            "adapter_version": CODEBASE_GRAPH_ADAPTER_VERSION,
            "inference_version": None,
            "input_fingerprint": input_fingerprint,
            "graph_fingerprint": None,
            "profile_ids_json": [CODEBASE_GRAPH_PROFILE_ID],
            "coverage_report_ids_json": [],
            "status": "building",
            "parent_graph_revision_id": None,
            "created_at": now,
            "completed_at": None,
        }
    )

    record = source_record
    repo_render_ref_json = {
        "ref_type": "repo_revision",
        "repo_id": repo_id,
        "commit_id": revision,
        "render_source": "repo_metadata",
        "raw_source_present": bool(repo_root_path),
        "raw_source_provenance": "repo_worktree",
    }
    repo_container = _container_row(
        graph_revision_id=graph_revision_id,
        source_id=source_id,
        logical_source_id=logical_source_id,
        revision_id=revision,
        repo_id=repo_id,
        content_revision_id=content_revision_id,
        kind_ns="repo",
        kind="repository",
        traits={
            "semantic_id": f"{repo_id}:{revision}",
            "repo_id": repo_id,
            "revision": revision,
            "capabilities": ["scope_unit", "operator_domain_root"],
        },
        span_refs=[],
        order_segments=[0],
        primary_render_ref_id=None,
        primary_render_ref_fingerprint=None,
        render_ref_json=repo_render_ref_json,
        record=record,
    )
    graph["containers"].append(repo_container)

    work_item_payload = _source_record_work_item(source_record)
    work_item_kind = str(work_item_payload.get("work_item_kind") or "repository_memory_scope")
    work_item_container = _container_row(
        graph_revision_id=graph_revision_id,
        source_id=source_id,
        logical_source_id=logical_source_id,
        revision_id=revision,
        repo_id=repo_id,
        content_revision_id=content_revision_id,
        kind_ns="repo",
        kind="work_item",
        traits={
            "semantic_id": f"{repo_id}:{revision}:{work_item_payload.get('work_item_id') or 'default_work_item'}",
            "work_item_kind": work_item_kind,
            "work_item": work_item_payload,
            "capabilities": ["work_item", "context_pack_root"],
        },
        span_refs=[],
        order_segments=[1],
        primary_render_ref_id=None,
        primary_render_ref_fingerprint=None,
        render_ref_json={"ref_type": "repo_work_item", "repo_id": repo_id, "commit_id": revision, "work_item": work_item_payload},
        record=record,
    )
    graph["containers"].append(work_item_container)
    graph["relations"].append(
        _relation_row(
            graph_revision_id=graph_revision_id,
            src_container_id=repo_container["container_id"],
            dst_container_id=work_item_container["container_id"],
            relation_kind="owns",
            traits={"relation_capability": "owns"},
        )
    )

    container_by_object_id: dict[str, dict[str, Any]] = {}
    missing_render_objects: list[str] = []
    file_text_cache: dict[str, tuple[str | None, str | None, str | None]] = {}
    path_order = {path: idx for idx, path in enumerate(sorted({str(row["file_path"]) for row in normalized["objects"]}), start=1)}
    for index, row in enumerate(normalized["objects"], start=1):
        kind_ns, kind = OBJECT_KIND_MAP.get(str(row["object_type"]), ("code", "symbol"))
        kind_fq = f"{kind_ns}:{kind}"
        payload = dict(row.get("payload") or {})
        name = _payload_name(payload, str(row["id"]))
        span = dict(row["span"])
        ref_type = _semantic_ref_type(row, kind_fq)
        ref_json = _semantic_ref_json(row, ref_type)
        span_refs = [{"ref_type": ref_type, **ref_json}]
        capabilities: list[str] = ["addressable_unit", "typed_ref"]
        traits: dict[str, Any] = {
            "semantic_id": str(row["id"]),
            "semantic_object_type": str(row["object_type"]),
            "name": name,
            "qualified_name": str(payload.get("qualified_name") or name),
            "path": str(row["file_path"]),
            "language": str(row["language"]),
            "analyzer_id": str(row["analyzer_id"]),
            "analyzer_version": str(row["analyzer_version"]),
            "derivation_type": str(row["derivation_type"]),
            "payload": payload,
            "capabilities": capabilities,
        }
        if kind_fq in {"code:file", "code:function", "code:method", "code:class", "code:test_case"}:
            capabilities.append("renderable_unit")
        if kind_fq in {"code:file", "code:test_case"}:
            capabilities.append("operator_domain_member")
        placeholder = _container_row(
            graph_revision_id=graph_revision_id,
            source_id=source_id,
            logical_source_id=logical_source_id,
            revision_id=revision,
            repo_id=repo_id,
            content_revision_id=content_revision_id,
            kind_ns=kind_ns,
            kind=kind,
            traits=traits,
            span_refs=span_refs,
            order_segments=[path_order.get(str(row["file_path"]), index), span.get("start_line") or 1, index],
            primary_render_ref_id=None,
            primary_render_ref_fingerprint=None,
            render_ref_json=ref_json,
            record=record,
        )
        render_row, render_ref_json, char_range = _render_ref_for_object(
            graph_revision_id=graph_revision_id,
            container_id=placeholder["container_id"],
            row=row,
            repo_root=repo_root_path,
            file_text_cache=file_text_cache,
        )
        if render_row:
            placeholder["primary_render_ref_id"] = render_row["render_ref_id"]
            placeholder["primary_render_ref_fingerprint"] = render_row["ref_fingerprint"]
            placeholder["render_ref_json"] = render_ref_json
            graph["render_refs"].append(render_row)
        else:
            missing_render_objects.append(str(row["id"]))
            placeholder["render_score"] = 0.0
        graph["containers"].append(placeholder)
        container_by_object_id[str(row["id"])] = placeholder
        ref = _ref_row(
            graph_revision_id=graph_revision_id,
            container_id=placeholder["container_id"],
            ref_role="semantic_object",
            ref_type=ref_type,
            ref_json=ref_json,
            source_id=source_id,
            logical_source_id=logical_source_id,
            revision_id=revision,
            coverage_required=True,
        )
        graph["refs"].append(ref)
        _add_lookup_projection(graph, ref, ref_json, ref_type)
        if char_range:
            char_ref_json = _file_char_ref_json(row, char_range)
            char_ref = _ref_row(
                graph_revision_id=graph_revision_id,
                container_id=placeholder["container_id"],
                ref_role="source_char_range",
                ref_type="file_char_range",
                ref_json=char_ref_json,
                source_id=source_id,
                logical_source_id=logical_source_id,
                revision_id=revision,
                coverage_required=True,
            )
            graph["refs"].append(char_ref)
            _add_lookup_projection(graph, char_ref, char_ref_json, "file_char_range")
            graph["ref_ranges"].append(
                _range_row(
                    ref["ref_id"],
                    "char",
                    _line_scope(str(row["repo_id"]), str(row["revision"]), str(row["file_path"])),
                    int(char_range["start"]),
                    int(char_range["end"]),
                )
            )
            graph["ref_ranges"].append(
                _range_row(
                    char_ref["ref_id"],
                    "char",
                    _line_scope(str(row["repo_id"]), str(row["revision"]), str(row["file_path"])),
                    int(char_range["start"]),
                    int(char_range["end"]),
                )
            )
        ast_ref_json = _ast_node_ref_json(row)
        ast_ref = _ref_row(
            graph_revision_id=graph_revision_id,
            container_id=placeholder["container_id"],
            ref_role="semantic_ast_node",
            ref_type="ast_node",
            ref_json=ast_ref_json,
            source_id=source_id,
            logical_source_id=logical_source_id,
            revision_id=revision,
            coverage_required=True,
        )
        graph["refs"].append(ast_ref)
        _add_lookup_projection(graph, ast_ref, ast_ref_json, "ast_node")
        graph["relations"].append(
            _relation_row(
                graph_revision_id=graph_revision_id,
                src_container_id=repo_container["container_id"],
                dst_container_id=placeholder["container_id"],
                relation_kind="owns",
                traits={"relation_capability": "owns"},
            )
        )

    relation_capabilities_used: list[str] = []
    for row in normalized["relations"]:
        src = container_by_object_id.get(str(row["from_id"]))
        dst = container_by_object_id.get(str(row["to_id"]))
        capability = RELATION_CAPABILITY_MAP.get(str(row["relation_type"]))
        if not src or not dst or not capability:
            continue
        relation_capabilities_used.append(capability)
        graph["relations"].append(
            _relation_row(
                graph_revision_id=graph_revision_id,
                src_container_id=src["container_id"],
                dst_container_id=dst["container_id"],
                relation_kind=capability,
                traits={
                    "relation_capability": capability,
                    "semantic_relation_type": str(row["relation_type"]),
                    "semantic_relation_id": str(row["id"]),
                    "derivation_type": str(row["derivation_type"]),
                    "payload": deepcopy(row.get("payload") or {}),
                },
            )
        )

    graph["contracts"].extend(
        [
            _contract_row(
                contract_kind="family_profile_contract",
                payload_schema_id="repo_family_profile_contract_v1",
                payload={
                    "family": "codebase",
                    "profile_id": CODEBASE_GRAPH_PROFILE_ID,
                    "scope": "universal_repository_memory",
                    "planner_domain_policy": "seed_then_full_typed_operator_domain",
                },
            ),
            _contract_row(
                contract_kind="relation_capability_contract",
                payload_schema_id="repo_relation_capability_contract_v1",
                payload={
                    "relation_capabilities": UNIVERSAL_RELATION_CAPABILITIES,
                    "mapped_relations": RELATION_CAPABILITY_MAP,
                    "expansion_targets_capabilities": True,
                },
            ),
            _contract_row(
                contract_kind="render_contract",
                payload_schema_id="repo_render_contract_v1",
                payload={
                    "exact_render_source": "repo_file_blob_span",
                    "snippet_display_is_not_exact_copy": True,
                    "line_numbered_display_mode": "exact_structured_or_grounded_context",
                },
            ),
            _contract_row(
                contract_kind="coverage_policy_contract",
                payload_schema_id="repo_coverage_policy_contract_v1",
                payload={
                    "silent_absence": "forbidden",
                    "missing_capabilities": ["not_applicable", "explicitly_absent", "failed_closed"],
                },
            ),
        ]
    )

    coverage_payload = _build_coverage_payload(
        normalized=normalized,
        graph=graph,
        missing_render_objects=missing_render_objects,
        relation_capabilities_used=relation_capabilities_used,
    )
    validator_payload = _validator_payload(coverage_payload)
    coverage_artifact = _artifact_row(
        artifact_kind="coverage_report",
        graph_revision_id=graph_revision_id,
        subject_type="container_graph_revision",
        subject_id=graph_revision_id,
        payload=coverage_payload,
    )
    validator_artifact = _artifact_row(
        artifact_kind="validator_report",
        graph_revision_id=graph_revision_id,
        subject_type="container_graph_revision",
        subject_id=graph_revision_id,
        payload=validator_payload,
    )
    context_pack = _artifact_row(
        artifact_kind="context_pack",
        graph_revision_id=graph_revision_id,
        subject_type="container",
        subject_id=work_item_container["container_id"],
        payload=_context_pack_payload(
            repo_container_id=repo_container["container_id"],
            work_item_container_id=work_item_container["container_id"],
            graph_revision_id=graph_revision_id,
            coverage_artifact_id=coverage_artifact["artifact_id"],
            containers=graph["containers"],
            relation_rows=graph["relations"],
            work_item=work_item_payload,
        ),
    )
    graph["artifacts"].extend([coverage_artifact, validator_artifact, context_pack])

    active = bool(validator_payload.get("ok"))
    for revision_row in graph["graph_revisions"]:
        if revision_row.get("container_graph_revision_id") == graph_revision_id:
            revision_row["coverage_report_ids_json"] = [coverage_artifact["artifact_id"]]
            revision_row["status"] = "active" if active else "failed_closed"
            revision_row["completed_at"] = _utcnow_iso()
            revision_row["graph_fingerprint"] = _fingerprint(
                {
                    "containers": [row["container_id"] for row in graph["containers"] if row.get("container_graph_revision_id") == graph_revision_id],
                    "relations": [row["relation_id"] for row in graph["relations"] if row.get("container_graph_revision_id") == graph_revision_id],
                    "artifacts": [coverage_artifact["artifact_id"], validator_artifact["artifact_id"], context_pack["artifact_id"]],
                }
            )
            break

    return _dedupe_graph(graph)


def _drop_existing_codebase_source_rows(graph: dict[str, list[dict]], source_id: str) -> dict[str, list[dict]]:
    graph = normalize_container_graph(graph)
    revision_ids = {
        str(row.get("container_graph_revision_id") or "")
        for row in graph["graph_revisions"]
        if str(row.get("source_id") or "") == source_id
        and str(row.get("adapter_name") or "") == CODEBASE_GRAPH_ADAPTER
    }
    container_ids = {
        str(row.get("container_id") or "")
        for row in graph["containers"]
        if str(row.get("container_graph_revision_id") or "") in revision_ids
    }
    ref_ids = {
        str(row.get("ref_id") or "")
        for row in graph["refs"]
        if str(row.get("container_graph_revision_id") or "") in revision_ids
        or str(row.get("container_id") or "") in container_ids
    }
    render_ref_ids = {
        str(row.get("render_ref_id") or "")
        for row in graph["render_refs"]
        if str(row.get("container_graph_revision_id") or "") in revision_ids
        or str(row.get("container_id") or "") in container_ids
    }
    filtered = empty_container_graph()
    filtered["graph_revisions"] = [row for row in graph["graph_revisions"] if str(row.get("container_graph_revision_id") or "") not in revision_ids]
    filtered["containers"] = [row for row in graph["containers"] if str(row.get("container_id") or "") not in container_ids]
    filtered["relations"] = [
        row
        for row in graph["relations"]
        if str(row.get("container_graph_revision_id") or "") not in revision_ids
        and str(row.get("src_container_id") or "") not in container_ids
        and str(row.get("dst_container_id") or "") not in container_ids
    ]
    filtered["anchors"] = [
        row
        for row in graph["anchors"]
        if str(row.get("container_graph_revision_id") or "") not in revision_ids
        and str(row.get("container_id") or "") not in container_ids
    ]
    filtered["evidence"] = [
        row
        for row in graph["evidence"]
        if str(row.get("container_graph_revision_id") or "") not in revision_ids
        and str(row.get("subject_id") or "") not in container_ids
        and str(row.get("subject_id") or "") not in render_ref_ids
    ]
    filtered["refs"] = [row for row in graph["refs"] if str(row.get("ref_id") or "") not in ref_ids]
    filtered["ref_lookup"] = [row for row in graph["ref_lookup"] if str(row.get("ref_id") or "") not in ref_ids]
    filtered["ref_ranges"] = [row for row in graph["ref_ranges"] if str(row.get("ref_id") or "") not in ref_ids]
    filtered["render_refs"] = [row for row in graph["render_refs"] if str(row.get("render_ref_id") or "") not in render_ref_ids]
    filtered["state"] = [
        row
        for row in graph["state"]
        if str(row.get("container_graph_revision_id") or "") not in revision_ids
        and str(row.get("container_id") or "") not in container_ids
    ]
    filtered["contracts"] = list(graph["contracts"])
    filtered["artifacts"] = [
        row
        for row in graph["artifacts"]
        if str(row.get("container_graph_revision_id") or "") not in revision_ids
        and not (set(str(value or "") for value in (row.get("container_graph_revision_ids_json") or [])) & revision_ids)
        and str(row.get("subject_id") or "") not in container_ids
    ]
    return filtered


def _dedupe_graph(graph: dict[str, list[dict]]) -> dict[str, list[dict]]:
    graph = normalize_container_graph(graph)
    id_keys = {
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
    }
    for key, id_key in id_keys.items():
        rows = {str(row.get(id_key) or ""): row for row in graph[key] if str(row.get(id_key) or "")}
        graph[key] = [rows[row_id] for row_id in sorted(rows)]
    return graph
