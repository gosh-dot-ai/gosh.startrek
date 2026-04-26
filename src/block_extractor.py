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

import copy
import json
import logging
from pathlib import Path
from typing import Any

from .block_segmenter import Block
from .extraction_repair import (
    apply_patch_operations,
    build_issue,
    drop_target_root,
    group_issues_by_target_root,
    parse_json_object,
    parse_patch_operations,
    path_matches_root,
)
from .object_flags import build_object_flag, normalize_object_flags_field, validate_object_flags
from .object_reports import build_report, build_report_entry
from .prompt_safety import render_data_block, render_kv_block

_log = logging.getLogger(__name__)
_PROMPT_DIR = Path(__file__).parent / "prompts" / "extraction"

_EMPTY_RESULT: dict[str, list[dict]] = {"facts": [], "temporal_links": []}
_ALLOWED_TOP_LEVEL_KEYS = {"facts", "temporal_links"}
_ALLOWED_TEMPORAL_RELATIONS = {"before", "after", "during", "same_time", "overlaps"}
_BLOCK_ROOT_CONTRACT: dict[str, list[Any]] = {"facts": [], "temporal_links": []}
_BLOCK_SCHEMA_SNIPPETS = {
    "facts": {
        "type": "list[fact]",
        "item_fields": [
            "local_id",
            "fact",
            "kind",
            "entities",
            "tags",
            "depends_on",
            "speaker",
            "speaker_role",
            "supersedes_topic",
            "confidence",
            "event_date",
            "flags",
        ],
    },
    "temporal_links": {
        "type": "list[temporal_link]",
        "item_fields": ["before", "after", "signal", "relation"],
    },
}

_FAMILY_PROMPT = {
    "PROSE": "prose_block.md",
    "LIST": "list_block.md",
    "TABLE": "table_block.md",
    "UNKNOWN": "fallback_block.md",
    "KV": "prose_block.md",
    "CODE": "fallback_block.md",
    "OBJECT": "fallback_block.md",
}


def _load_prompt(name: str) -> str:
    filename = name if name.endswith(".md") else f"{name}.md"
    path = _PROMPT_DIR / filename
    return path.read_text(encoding="utf-8")


def _format_prompt(template: str, block: Block, session_metadata: dict) -> str:
    return template.format(
        container_kind=session_metadata.get("container_kind", "conversation"),
        speaker=block.speaker or "unknown",
        lead_in=block.lead_in or "none",
        session_date=session_metadata.get("session_date", "unknown"),
        session_num=session_metadata.get("session_num", 0),
        section_path=block.section_path or "none",
    )


def _normalize_fact_item(item: dict[str, Any]) -> dict[str, Any]:
    fact = copy.deepcopy(item)
    if "id" in fact and "local_id" not in fact:
        fact["local_id"] = fact["id"]
    fact.setdefault("kind", "fact")
    normalize_object_flags_field(fact)
    return fact


def _build_block_user_payload(block: Block, session_metadata: dict) -> str:
    metadata = {
        "container_kind": session_metadata.get("container_kind", "conversation"),
        "block_family": block.family or "UNKNOWN",
        "section_path": block.section_path or "none",
        "speaker": block.speaker or "unknown",
        "speaker_role": block.speaker_role or "unknown",
        "lead_in": block.lead_in or "none",
        "session_date": session_metadata.get("session_date", "unknown"),
        "session_num": session_metadata.get("session_num", 0),
        "block_order": block.order,
    }
    return "\n\n".join(
        [
            render_kv_block("BLOCK_METADATA", metadata),
            render_data_block("SOURCE_TEXT", block.text),
        ]
    )


def _normalize_temporal_link_item(item: dict[str, Any]) -> dict[str, Any]:
    link = copy.deepcopy(item)
    normalize_object_flags_field(link)
    return link


def _normalize_result(obj: Any) -> dict | None:
    if not isinstance(obj, dict):
        return None
    normalized = copy.deepcopy(obj)
    if isinstance(normalized.get("facts"), list):
        normalized["facts"] = [
            _normalize_fact_item(f) if isinstance(f, dict) else copy.deepcopy(f)
            for f in normalized["facts"]
        ]
    if isinstance(normalized.get("temporal_links"), list):
        normalized["temporal_links"] = [
            _normalize_temporal_link_item(link) if isinstance(link, dict) else copy.deepcopy(link)
            for link in normalized["temporal_links"]
        ]
    return normalized


def _parse_block_payload(raw: Any, *, family: str) -> tuple[dict | None, list[dict[str, Any]]]:
    parsed, issues = parse_json_object(raw, layer="top_level", family=family)
    if parsed is None:
        return None, issues
    normalized = _normalize_result(parsed)
    return normalized if normalized is not None else parsed, []


def _context_payload(block: Block, session_metadata: dict) -> str:
    payload = {
        "block_family": block.family,
        "block_order": block.order,
        "block_text": block.text,
        "speaker": block.speaker,
        "speaker_role": block.speaker_role,
        "lead_in": block.lead_in,
        "section_path": block.section_path,
        "session_metadata": session_metadata,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _issue_payload(issues: list[dict]) -> str:
    return json.dumps(issues, ensure_ascii=False, indent=2)


def _target_schema_payload(roots: list[str]) -> str:
    schema: dict[str, Any] = {}
    for root in roots:
        base = root.split("[", 1)[0]
        schema[root] = _BLOCK_SCHEMA_SNIPPETS.get(base, {"type": "unknown"})
    return json.dumps(schema, ensure_ascii=False, indent=2)


def _parse_failure_report_entries(issues: list[dict[str, Any]], *, repair_attempted: bool) -> list[dict]:
    entries: list[dict] = []
    for issue in issues:
        entries.append(
            build_report_entry(
                target_path="root",
                status="dropped",
                issue=issue,
                repair_attempted=repair_attempted,
            )
        )
    return entries


def _render_root_repair_prompt(
    *,
    previous_raw: Any,
    issues: list[dict],
    block: Block,
    session_metadata: dict,
    prompt_overrides: dict[str, str] | None,
) -> str:
    override = prompt_overrides.get("structured_json_root_repair") if prompt_overrides else None
    template = override if isinstance(override, str) else _load_prompt("structured_json_root_repair")
    return template.format(
        root_contract_json=json.dumps(_BLOCK_ROOT_CONTRACT, ensure_ascii=False, indent=2),
        previous_raw=previous_raw if isinstance(previous_raw, str) else json.dumps(previous_raw, ensure_ascii=False, indent=2),
        issues_json=_issue_payload(issues),
        context_payload=_context_payload(block, session_metadata),
    )


def _render_patch_repair_prompt(
    *,
    payload: dict,
    issues: list[dict],
    roots: list[str],
    block: Block,
    session_metadata: dict,
    prompt_overrides: dict[str, str] | None,
) -> str:
    override = prompt_overrides.get("structured_item_patch_repair") if prompt_overrides else None
    template = override if isinstance(override, str) else _load_prompt("structured_item_patch_repair")
    return template.format(
        allowed_paths_json=json.dumps(sorted(roots), ensure_ascii=False, indent=2),
        issues_json=_issue_payload(issues),
        current_payload_json=json.dumps(payload, ensure_ascii=False, indent=2),
        target_schema_json=_target_schema_payload(roots),
        context_payload=_context_payload(block, session_metadata),
    )


def _append_top_level_issues(body: dict, *, issues: list[dict]) -> None:
    keys = set(body)
    for key in sorted(_ALLOWED_TOP_LEVEL_KEYS - keys):
        issues.append(
            build_issue(
                normalized_code="shape.missing_top_level_key",
                raw_code="missing_top_level_key",
                layer="top_level",
                family=None,
                path=key,
                object_id=None,
                expected="list",
                actual=None,
                message=f"Top-level key {key!r} is required.",
                repair_hint=f"Add the missing {key} list.",
                repairable=True,
            )
        )
    for key in sorted(keys - _ALLOWED_TOP_LEVEL_KEYS):
        issues.append(
            build_issue(
                normalized_code="shape.unexpected_top_level_key",
                raw_code="unexpected_top_level_key",
                layer="top_level",
                family=None,
                path=key,
                object_id=None,
                expected="only supported top-level keys",
                actual=key,
                message=f"Unexpected top-level key {key!r} is not allowed.",
                repair_hint=f"Remove the unexpected key {key}.",
                repairable=True,
            )
        )
    for key in sorted(_ALLOWED_TOP_LEVEL_KEYS & keys):
        if not isinstance(body.get(key), list):
            issues.append(
                build_issue(
                    normalized_code="shape.invalid_top_level_type",
                    raw_code="invalid_top_level_type",
                    layer="top_level",
                    family=None,
                    path=key,
                    object_id=None,
                    expected="list",
                    actual=type(body.get(key)).__name__,
                    message=f"Top-level key {key!r} must be a list.",
                    repair_hint=f"Replace {key} with a list.",
                    repairable=True,
                )
            )


def _fact_object_id(item: dict[str, Any]) -> str | None:
    for key in ("local_id", "id"):
        value = str(item.get(key) or "").strip()
        if value:
            return value
    return None


def _link_object_id(item: dict[str, Any]) -> str | None:
    before = str(item.get("before") or "").strip()
    after = str(item.get("after") or "").strip()
    if before or after:
        return f"{before}->{after}"
    return None


def _append_required_string_issue(item: dict, path: str, key: str, object_id: str | None, issues: list[dict]) -> None:
    if key not in item:
        issues.append(
            build_issue(
                normalized_code="field.missing_required",
                raw_code="missing_required_field",
                layer="fact_layer",
                family=None,
                path=f"{path}.{key}",
                object_id=object_id,
                expected="non-empty string",
                actual=None,
                message=f"{path}.{key} is required.",
                repair_hint=f"Populate {key} with a grounded string.",
                repairable=True,
            )
        )
        return
    value = item.get(key)
    if not isinstance(value, str):
        issues.append(
            build_issue(
                normalized_code="shape.type_error",
                raw_code="invalid_field_type",
                layer="fact_layer",
                family=None,
                path=f"{path}.{key}",
                object_id=object_id,
                expected="string",
                actual=type(value).__name__,
                message=f"{path}.{key} must be a string.",
                repair_hint=f"Set {key} to a string value.",
                repairable=True,
            )
        )
        return
    if not value.strip():
        issues.append(
            build_issue(
                normalized_code="field.empty_required",
                raw_code="empty_required_field",
                layer="fact_layer",
                family=None,
                path=f"{path}.{key}",
                object_id=object_id,
                expected="non-empty string",
                actual="",
                message=f"{path}.{key} must not be empty.",
                repair_hint=f"Populate {key} with a grounded string or remove the fact.",
                repairable=True,
            )
        )


def _append_optional_string_issue(item: dict, path: str, key: str, object_id: str | None, issues: list[dict], *, layer: str) -> None:
    if key not in item or item.get(key) is None:
        return
    value = item.get(key)
    if not isinstance(value, str):
        issues.append(
            build_issue(
                normalized_code="shape.type_error",
                raw_code="invalid_field_type",
                layer=layer,
                family=None,
                path=f"{path}.{key}",
                object_id=object_id,
                expected="string or null",
                actual=type(value).__name__,
                message=f"{path}.{key} must be a string or null.",
                repair_hint=f"Set {key} to a string or null.",
                repairable=True,
            )
        )


def _append_string_list_issue(item: dict, path: str, key: str, object_id: str | None, issues: list[dict]) -> None:
    if key not in item:
        return
    value = item.get(key)
    if value is None:
        return
    if not isinstance(value, list):
        issues.append(
            build_issue(
                normalized_code="shape.type_error",
                raw_code="invalid_field_type",
                layer="fact_layer",
                family=None,
                path=f"{path}.{key}",
                object_id=object_id,
                expected="list",
                actual=type(value).__name__,
                message=f"{path}.{key} must be a list.",
                repair_hint=f"Set {key} to a list of strings.",
                repairable=True,
            )
        )
        return
    for idx, entry in enumerate(value):
        if not isinstance(entry, str):
            issues.append(
                build_issue(
                    normalized_code="shape.list_item_type_error",
                    raw_code="invalid_list_item_type",
                    layer="fact_layer",
                    family=None,
                    path=f"{path}.{key}[{idx}]",
                    object_id=object_id,
                    expected="string",
                    actual=type(entry).__name__,
                    message=f"{path}.{key}[{idx}] must be a string.",
                    repair_hint=f"Replace {key}[{idx}] with a string or remove it.",
                    repairable=True,
                )
            )


def _append_fact_issues(payload: dict, issues: list[dict]) -> None:
    facts = payload.get("facts", [])
    fact_ids: set[str] = set()
    for idx, item in enumerate(facts):
        path = f"facts[{idx}]"
        if not isinstance(item, dict):
            issues.append(
                build_issue(
                    normalized_code="shape.list_item_type_error",
                    raw_code="invalid_fact_item_type",
                    layer="fact_layer",
                    family=None,
                    path=path,
                    object_id=None,
                    expected="object",
                    actual=type(item).__name__,
                    message=f"{path} must be an object.",
                    repair_hint="Replace the malformed item with a valid fact object or remove it.",
                    repairable=True,
                )
            )
            continue
        object_id = _fact_object_id(item)
        _append_required_string_issue(item, path, "local_id", object_id, issues)
        _append_required_string_issue(item, path, "fact", object_id, issues)
        _append_required_string_issue(item, path, "kind", object_id, issues)
        for key in ("speaker", "speaker_role", "supersedes_topic", "event_date"):
            _append_optional_string_issue(item, path, key, object_id, issues, layer="fact_layer")
        for key in ("entities", "tags", "depends_on"):
            _append_string_list_issue(item, path, key, object_id, issues)
        if "confidence" in item and item.get("confidence") is not None:
            value = item.get("confidence")
            if isinstance(value, str):
                if not value.strip():
                    issues.append(
                        build_issue(
                            normalized_code="field.empty_required",
                            raw_code="empty_confidence_string",
                            layer="fact_layer",
                            family=None,
                            path=f"{path}.confidence",
                            object_id=object_id,
                            expected="non-empty string, number in [0, 1], or null",
                            actual="",
                            message=f"{path}.confidence must not be empty.",
                            repair_hint="Set confidence to a non-empty string, a number in [0, 1], or null.",
                            repairable=True,
                        )
                    )
            elif not isinstance(value, (int, float)):
                issues.append(
                    build_issue(
                        normalized_code="shape.type_error",
                        raw_code="invalid_confidence_type",
                        layer="fact_layer",
                        family=None,
                        path=f"{path}.confidence",
                        object_id=object_id,
                        expected="string|number|null",
                        actual=type(value).__name__,
                        message=f"{path}.confidence must be a string, number, or null.",
                        repair_hint="Set confidence to a non-empty string, a number in [0, 1], or null.",
                        repairable=True,
                    )
                )
            elif float(value) < 0.0 or float(value) > 1.0:
                issues.append(
                    build_issue(
                        normalized_code="field.invalid_range",
                        raw_code="invalid_confidence_range",
                        layer="fact_layer",
                        family=None,
                        path=f"{path}.confidence",
                        object_id=object_id,
                        expected="0 <= confidence <= 1",
                        actual=str(value),
                        message=f"{path}.confidence must be between 0 and 1.",
                        repair_hint="Clamp numeric confidence into [0, 1], or use a non-empty string/null.",
                        repairable=True,
                    )
                )
        flag_err = validate_object_flags(item.get("flags"))
        if flag_err:
            issues.append(
                build_issue(
                    normalized_code="shape.type_error",
                    raw_code="invalid_fact_flags",
                    layer="fact_layer",
                    family=None,
                    path=f"{path}.flags",
                    object_id=object_id,
                    expected="valid object.flags[] schema",
                    actual=flag_err,
                    message=f"{path}.flags is invalid: {flag_err}",
                    repair_hint="Return a valid flags[] list or remove the field.",
                    repairable=True,
                )
            )
        if object_id:
            if object_id in fact_ids:
                issues.append(
                    build_issue(
                        normalized_code="id.duplicate",
                        raw_code="duplicate_fact_ids",
                        layer="fact_layer",
                        family=None,
                        path=f"{path}.local_id",
                        object_id=object_id,
                        expected="unique local_id",
                        actual=object_id,
                        message=f"Duplicate local_id {object_id!r} detected in facts.",
                        repair_hint="Rename or remove the duplicate fact.",
                        repairable=True,
                    )
                )
            fact_ids.add(object_id)


def _append_temporal_link_issues(payload: dict, issues: list[dict]) -> None:
    facts = payload.get("facts", [])
    fact_ids = {
        _fact_object_id(item)
        for item in facts
        if isinstance(item, dict) and _fact_object_id(item)
    }
    for idx, item in enumerate(payload.get("temporal_links", [])):
        path = f"temporal_links[{idx}]"
        if not isinstance(item, dict):
            issues.append(
                build_issue(
                    normalized_code="shape.list_item_type_error",
                    raw_code="invalid_temporal_link_item_type",
                    layer="temporal_layer",
                    family=None,
                    path=path,
                    object_id=None,
                    expected="object",
                    actual=type(item).__name__,
                    message=f"{path} must be an object.",
                    repair_hint="Replace the malformed temporal link with a valid object or remove it.",
                    repairable=True,
                )
            )
            continue
        object_id = _link_object_id(item)
        _append_required_string_issue(item, path, "before", object_id, issues)
        _append_required_string_issue(item, path, "after", object_id, issues)
        _append_optional_string_issue(item, path, "signal", object_id, issues, layer="temporal_layer")
        _append_optional_string_issue(item, path, "relation", object_id, issues, layer="temporal_layer")
        before = item.get("before")
        after = item.get("after")
        if isinstance(before, str) and isinstance(after, str):
            if before.strip() and before == after:
                issues.append(
                    build_issue(
                        normalized_code="ref.self_reference_forbidden",
                        raw_code="self_temporal_reference",
                        layer="temporal_layer",
                        family=None,
                        path=f"{path}.after",
                        object_id=object_id,
                        expected="different fact id",
                        actual=after,
                        message=f"{path} must not link a fact to itself.",
                        repair_hint="Point the temporal link at a different fact or remove it.",
                        repairable=True,
                    )
                )
            if fact_ids and before.strip() and before not in fact_ids:
                issues.append(
                    build_issue(
                        normalized_code="ref.unknown_id",
                        raw_code="unknown_temporal_fact_id",
                        layer="temporal_layer",
                        family=None,
                        path=f"{path}.before",
                        object_id=object_id,
                        expected="existing local_id",
                        actual=before,
                        message=f"{path}.before references unknown fact id {before!r}.",
                        repair_hint="Replace before with an existing local_id or remove the link.",
                        repairable=True,
                    )
                )
            if fact_ids and after.strip() and after not in fact_ids:
                issues.append(
                    build_issue(
                        normalized_code="ref.unknown_id",
                        raw_code="unknown_temporal_fact_id",
                        layer="temporal_layer",
                        family=None,
                        path=f"{path}.after",
                        object_id=object_id,
                        expected="existing local_id",
                        actual=after,
                        message=f"{path}.after references unknown fact id {after!r}.",
                        repair_hint="Replace after with an existing local_id or remove the link.",
                        repairable=True,
                    )
                )
        relation = item.get("relation")
        if isinstance(relation, str) and relation.strip() and relation not in _ALLOWED_TEMPORAL_RELATIONS:
            issues.append(
                build_issue(
                    normalized_code="enum.invalid_value",
                    raw_code="invalid_temporal_relation",
                    layer="temporal_layer",
                    family=None,
                    path=f"{path}.relation",
                    object_id=object_id,
                    expected="before|after|during|same_time|overlaps",
                    actual=relation,
                    message=f"{path}.relation has unsupported value {relation!r}.",
                    repair_hint="Use a supported temporal relation or remove the link.",
                    repairable=True,
                )
            )
        flag_err = validate_object_flags(item.get("flags"))
        if flag_err:
            issues.append(
                build_issue(
                    normalized_code="shape.type_error",
                    raw_code="invalid_temporal_link_flags",
                    layer="temporal_layer",
                    family=None,
                    path=f"{path}.flags",
                    object_id=object_id,
                    expected="valid object.flags[] schema",
                    actual=flag_err,
                    message=f"{path}.flags is invalid: {flag_err}",
                    repair_hint="Return a valid flags[] list or remove the field.",
                    repairable=True,
                )
            )


def _collect_block_issues(body: dict) -> tuple[dict, list[dict]]:
    issues: list[dict] = []
    _append_top_level_issues(body, issues=issues)
    payload = {
        "facts": copy.deepcopy(body.get("facts")) if isinstance(body.get("facts"), list) else [],
        "temporal_links": copy.deepcopy(body.get("temporal_links")) if isinstance(body.get("temporal_links"), list) else [],
    }
    _append_fact_issues(payload, issues)
    _append_temporal_link_issues(payload, issues)
    return payload, issues


def _object_id_for_root(payload: dict, root: str) -> str | None:
    if not root.endswith("]") or "[" not in root:
        return None
    collection, index_s = root[:-1].split("[", 1)
    if not index_s.isdigit():
        return None
    index = int(index_s)
    items = payload.get(collection) or []
    if not isinstance(items, list) or index < 0 or index >= len(items):
        return None
    item = items[index]
    if not isinstance(item, dict):
        return None
    if collection == "facts":
        return _fact_object_id(item)
    if collection == "temporal_links":
        return _link_object_id(item)
    return None


def _repair_outcomes(
    *,
    original_payload: dict,
    candidate_payload: dict,
    original_issues: list[dict],
    candidate_issues: list[dict],
) -> tuple[dict, list[dict], dict[str, dict]]:
    grouped = group_issues_by_target_root(original_issues)
    report_entries: list[dict] = []
    outcomes: dict[str, dict] = {}
    working_payload = candidate_payload
    for root, root_issues in grouped.items():
        remaining = [issue for issue in candidate_issues if path_matches_root(issue.get("path"), root)]
        if not remaining:
            outcomes[root] = {
                "status": "resolved",
                "issues": root_issues,
                "object_id": _object_id_for_root(candidate_payload, root),
            }
            report_entries.append(
                build_report_entry(
                    target_path=root,
                    status="resolved",
                    issue=root_issues[0],
                    repair_attempted=True,
                )
            )
            continue
        outcomes[root] = {
            "status": "dropped",
            "issues": remaining,
            "object_id": _object_id_for_root(candidate_payload, root) or _object_id_for_root(original_payload, root),
        }
        report_entries.append(
            build_report_entry(
                target_path=root,
                status="dropped",
                issue=remaining[0],
                repair_attempted=True,
            )
        )
        working_payload = drop_target_root(working_payload, root)
    return working_payload, report_entries, outcomes


def _attach_repair_flags(objects: list[dict], outcomes: dict[str, dict], *, collection: str) -> None:
    flags_by_id: dict[str, list[dict]] = {}
    for root, outcome in outcomes.items():
        if outcome.get("status") != "resolved" or not root.startswith(f"{collection}["):
            continue
        object_id = outcome.get("object_id")
        if not object_id:
            continue
        flags = [
            build_object_flag(
                producer="block_extractor",
                category="extraction",
                severity=issue["severity"],
                message=issue["message"],
                status="resolved",
                code=issue.get("normalized_code"),
                path=issue.get("path"),
                object_id=issue.get("object_id"),
                resolution="repaired",
                repair_attempted=True,
                details={
                    "raw_code": issue.get("raw_code"),
                    "layer": issue.get("layer"),
                    "family": issue.get("family"),
                    "expected": issue.get("expected"),
                    "actual": issue.get("actual"),
                    "repair_hint": issue.get("repair_hint"),
                },
            )
            for issue in outcome.get("issues", [])
        ]
        if flags:
            flags_by_id.setdefault(object_id, []).extend(flags)
    for obj in objects:
        object_id = _fact_object_id(obj) if collection == "facts" else _link_object_id(obj)
        if object_id and object_id in flags_by_id:
            existing = list(obj.get("flags") or [])
            obj["flags"] = existing + flags_by_id[object_id]


def _build_extraction_report(entries: list[dict], *, status: str | None = None) -> dict:
    dropped = sum(1 for entry in entries if isinstance(entry, dict) and entry.get("status") == "dropped")
    resolved = sum(1 for entry in entries if isinstance(entry, dict) and entry.get("status") == "resolved")
    final_status = status or ("partial" if entries else "ok")
    return build_report(
        report_kind="extraction",
        producer="block_extractor",
        status=final_status,
        entries=entries,
        summary={
            "entry_count": len(entries),
            "dropped_count": dropped,
            "resolved_count": resolved,
        },
    )


def _finalize_result(payload: dict, report_entries: list[dict], outcomes: dict[str, dict] | None = None, *, report_status: str | None = None) -> dict:
    facts = copy.deepcopy(payload.get("facts") or [])
    temporal_links = copy.deepcopy(payload.get("temporal_links") or [])
    if outcomes:
        _attach_repair_flags(facts, outcomes, collection="facts")
        _attach_repair_flags(temporal_links, outcomes, collection="temporal_links")
    return {
        "facts": facts,
        "temporal_links": temporal_links,
        "extraction_report": _build_extraction_report(copy.deepcopy(report_entries), status=report_status),
    }


async def extract_block(
    block: Block,
    session_metadata: dict,
    model: str | None = None,
    call_extract_fn=None,
    prompt_overrides: dict[str, str] | None = None,
) -> dict:
    """Extract facts from a single block."""
    if call_extract_fn is None:
        from .common import call_extract

        call_extract_fn = call_extract

    prompt_file = _FAMILY_PROMPT.get(block.family, "fallback_block.md")
    prompt_name = prompt_file.replace(".md", "")
    if prompt_overrides and prompt_name in prompt_overrides:
        template = prompt_overrides[prompt_name]
    else:
        template = _load_prompt(prompt_file)
    system_prompt = _format_prompt(template, block, session_metadata)
    user_msg = _build_block_user_payload(block, session_metadata)
    raw = await call_extract_fn(model, system_prompt, user_msg, max_tokens=4096)
    parsed, parse_issues = _parse_block_payload(raw, family=block.family)

    if parsed is None:
        repair_prompt = _render_root_repair_prompt(
            previous_raw=raw,
            issues=parse_issues,
            block=block,
            session_metadata=session_metadata,
            prompt_overrides=prompt_overrides,
        )
        repaired_raw = await call_extract_fn(model, repair_prompt, "", max_tokens=4096)
        parsed, parse_issues = _parse_block_payload(repaired_raw, family=block.family)
        if parsed is None:
            report_entries = _parse_failure_report_entries(parse_issues, repair_attempted=True)
            _log.warning("Block extraction parse failed after deterministic root repair (block order=%d)", block.order)
            return _finalize_result(dict(_EMPTY_RESULT), report_entries, report_status="failed")

    payload, issues = _collect_block_issues(parsed)
    if not issues:
        return _finalize_result(payload, report_entries=[])

    roots = sorted(group_issues_by_target_root(issues))
    patch_prompt = _render_patch_repair_prompt(
        payload=parsed,
        issues=issues,
        roots=roots,
        block=block,
        session_metadata=session_metadata,
        prompt_overrides=prompt_overrides,
    )
    patch_raw = await call_extract_fn(model, patch_prompt, "", max_tokens=4096)
    operations, patch_issues = parse_patch_operations(
        patch_raw,
        producer="block_extractor",
        family=block.family,
        layer="top_level",
    )

    candidate_body = parsed
    if operations is not None:
        candidate_body, apply_issues = apply_patch_operations(
            parsed,
            operations,
            allowed_roots=set(roots),
            family=block.family,
            layer="top_level",
        )
        candidate_body = _normalize_result(candidate_body) or candidate_body
        patch_issues.extend(apply_issues)

    candidate_payload, candidate_issues = _collect_block_issues(candidate_body)
    candidate_issues.extend(patch_issues)
    repaired_payload, report_entries, outcomes = _repair_outcomes(
        original_payload=payload,
        candidate_payload=candidate_payload,
        original_issues=issues,
        candidate_issues=candidate_issues,
    )

    final_payload, final_issues = _collect_block_issues(repaired_payload)
    if final_issues:
        for root, root_issues in group_issues_by_target_root(final_issues).items():
            report_entries.append(build_report_entry(target_path=root, status="dropped", issue=root_issues[0], repair_attempted=True))
            final_payload = drop_target_root(final_payload, root)
        final_payload, final_issues = _collect_block_issues(final_payload)
        if final_issues:
            _log.warning(
                "Block extraction validation failed after deterministic repair (block order=%d): %s",
                block.order,
                ", ".join(issue["raw_code"] for issue in final_issues),
            )
            return _finalize_result(dict(_EMPTY_RESULT), report_entries, report_status="failed")

    return _finalize_result(final_payload, report_entries=report_entries, outcomes=outcomes)
