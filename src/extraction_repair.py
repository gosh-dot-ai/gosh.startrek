# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

from __future__ import annotations

import copy
import hashlib
import json
import re
from typing import Any

from .object_flags import ALLOWED_FLAG_STATUSES, build_object_flag, validate_object_flags
from .object_reports import build_report_entry

ALLOWED_ISSUE_CATEGORIES = frozenset(
    {
        "parse",
        "shape",
        "field",
        "enum",
        "id",
        "ref",
        "grounding",
        "temporal",
        "semantic",
        "consistency",
        "security",
        "repair",
        "degrade",
    }
)
ALLOWED_ISSUE_SEVERITIES = frozenset({"critical", "high", "medium", "low", "info"})
ALLOWED_FLAG_RESOLUTIONS = frozenset(
    {"repaired", "accepted_with_warning", "dropped_after_failed_repair"}
)
NORMALIZED_ISSUE_CODES = frozenset(
    {
        "parse.invalid_json",
        "parse.non_object_root",
        "shape.missing_top_level_key",
        "shape.unexpected_top_level_key",
        "shape.invalid_top_level_type",
        "shape.type_error",
        "shape.list_item_type_error",
        "shape.object_item_type_error",
        "field.missing_required",
        "field.empty_required",
        "field.invalid_format",
        "field.invalid_range",
        "field.invalid_length",
        "field.invalid_precision",
        "field.invalid_charset",
        "enum.invalid_value",
        "id.missing",
        "id.duplicate",
        "id.cross_layer_collision",
        "ref.unknown_id",
        "ref.type_mismatch",
        "ref.invalid_direction",
        "ref.self_reference_forbidden",
        "ref.cardinality_violation",
        "ref.orphan_node",
        "ref.orphan_edge",
        "ref.dangling_relation",
        "grounding.span_not_found",
        "grounding.offset_mismatch",
        "grounding.value_not_supported",
        "grounding.date_not_supported",
        "grounding.number_unit_not_supported",
        "grounding.anchor_not_supported",
        "grounding.entity_not_supported",
        "grounding.locality_not_supported",
        "temporal.invalid_order",
        "temporal.invalid_interval",
        "temporal.currentness_conflict",
        "temporal.revision_conflict",
        "semantic.empty_higher_order_success",
        "semantic.incomplete_object",
        "semantic.membership_conflict",
        "semantic.object_relation_mismatch",
        "semantic.mutually_exclusive_fields",
        "semantic.unsupported_combination",
        "consistency.locality_mismatch",
        "consistency.source_scope_mismatch",
        "consistency.payload_scope_mismatch",
        "consistency.cross_object_conflict",
        "security.secret_exposed",
        "security.forbidden_field_populated",
        "repair.exhausted",
        "repair.patch_invalid",
        "repair.path_not_allowed",
        "repair.untouched_required_issue",
        "degrade.item_dropped",
        "degrade.layer_dropped",
    }
)

_SEVERITY_BY_CATEGORY = {
    "parse": "critical",
    "shape": "high",
    "field": "medium",
    "enum": "medium",
    "id": "high",
    "ref": "high",
    "grounding": "medium",
    "temporal": "medium",
    "semantic": "medium",
    "consistency": "high",
    "security": "critical",
    "repair": "medium",
    "degrade": "low",
}
_PATH_SEGMENT_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)(?:\[(\d+)\])?")


def _clean_raw_text(raw: Any) -> str:
    if isinstance(raw, str):
        text = raw
    else:
        try:
            text = json.dumps(raw, ensure_ascii=False)
        except TypeError:
            text = str(raw)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    return text.strip()


def build_issue(
    *,
    normalized_code: str,
    raw_code: str,
    layer: str | None,
    family: str | None,
    path: str | None,
    object_id: str | None,
    expected: str | None,
    actual: str | None,
    message: str,
    repair_hint: str | None,
    repairable: bool,
    severity: str | None = None,
) -> dict[str, Any]:
    if normalized_code not in NORMALIZED_ISSUE_CODES:
        raise ValueError(f"unsupported normalized_code: {normalized_code}")
    category = normalized_code.split(".", 1)[0]
    if category not in ALLOWED_ISSUE_CATEGORIES:
        raise ValueError(f"unsupported category: {category}")
    final_severity = severity or _SEVERITY_BY_CATEGORY[category]
    if final_severity not in ALLOWED_ISSUE_SEVERITIES:
        raise ValueError(f"unsupported severity: {final_severity}")
    return {
        "normalized_code": normalized_code,
        "raw_code": raw_code,
        "category": category,
        "layer": layer,
        "family": family,
        "path": path,
        "object_id": object_id,
        "expected": expected,
        "actual": actual,
        "message": message,
        "repair_hint": repair_hint,
        "repairable": bool(repairable),
        "severity": final_severity,
    }


def parse_json_object(
    raw: Any,
    *,
    layer: str | None = None,
    family: str | None = None,
) -> tuple[dict | None, list[dict[str, Any]]]:
    if isinstance(raw, dict):
        return copy.deepcopy(raw), []

    cleaned = _clean_raw_text(raw)
    if not cleaned:
        return None, [
            build_issue(
                normalized_code="parse.invalid_json",
                raw_code="empty_output",
                layer=layer,
                family=family,
                path=None,
                object_id=None,
                expected="valid JSON object",
                actual="empty output",
                message="Extractor returned empty output instead of a JSON object.",
                repair_hint="Return exactly one strict JSON object.",
                repairable=True,
            )
        ]

    candidates = [cleaned]
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if fenced:
        candidates.append(fenced.group(1))
    braces = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if braces:
        candidates.append(braces.group(0))

    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj, []
        return None, [
            build_issue(
                normalized_code="parse.non_object_root",
                raw_code="parsed_non_object_root",
                layer=layer,
                family=family,
                path=None,
                object_id=None,
                expected="JSON object",
                actual=type(obj).__name__,
                message=f"Parsed JSON root must be an object, got {type(obj).__name__}.",
                repair_hint="Return exactly one JSON object at the root.",
                repairable=True,
            )
        ]

    return None, [
        build_issue(
            normalized_code="parse.invalid_json",
            raw_code="json_decode_error",
            layer=layer,
            family=family,
            path=None,
            object_id=None,
            expected="valid JSON object",
            actual=cleaned[:200],
            message="Extractor output could not be parsed as a JSON object.",
            repair_hint="Return strict JSON only with no markdown or commentary.",
            repairable=True,
        )
    ]


def issue_target_root(path: str | None) -> str | None:
    if not path:
        return None
    match = re.match(r"^[A-Za-z_][A-Za-z0-9_]*(?:\[\d+\])?", path)
    return match.group(0) if match else path


def group_issues_by_target_root(issues: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for issue in issues:
        root = issue_target_root(issue.get("path"))
        if not root:
            continue
        grouped.setdefault(root, []).append(issue)
    return grouped


def path_matches_root(path: str | None, root: str) -> bool:
    if not path:
        return False
    return path == root or path.startswith(f"{root}.")


def _parse_path(path: str) -> list[str | int]:
    tokens: list[str | int] = []
    for segment in path.split("."):
        match = _PATH_SEGMENT_RE.fullmatch(segment)
        if not match:
            raise ValueError(f"unsupported patch path segment: {segment}")
        key, index = match.groups()
        tokens.append(key)
        if index is not None:
            tokens.append(int(index))
    return tokens


def _resolve_parent(container: Any, tokens: list[str | int]) -> tuple[Any, str | int]:
    if not tokens:
        raise ValueError("patch path must not be empty")
    current = container
    for token in tokens[:-1]:
        if isinstance(token, str):
            if not isinstance(current, dict) or token not in current:
                raise ValueError(f"missing path segment: {token}")
            current = current[token]
        else:
            if not isinstance(current, list) or token < 0 or token >= len(current):
                raise ValueError(f"list index out of range: {token}")
            current = current[token]
    return current, tokens[-1]


def _path_allowed(path: str, action: str, allowed_roots: set[str]) -> bool:
    if action == "append":
        return path in allowed_roots
    return any(path == root or path.startswith(f"{root}.") for root in allowed_roots)


def parse_patch_operations(
    raw: Any,
    *,
    producer: str,
    family: str | None,
    layer: str | None,
) -> tuple[list[dict[str, Any]] | None, list[dict[str, Any]]]:
    payload, parse_issues = parse_json_object(raw, layer=layer, family=family)
    if payload is None:
        return None, [
            build_issue(
                normalized_code="repair.patch_invalid",
                raw_code=issue["raw_code"],
                layer=layer,
                family=family,
                path=None,
                object_id=None,
                expected="operations JSON object",
                actual=issue.get("actual"),
                message=f"{producer} patch repair output was not a valid operations object.",
                repair_hint="Return only {\"operations\": []}",
                repairable=False,
            )
            for issue in parse_issues
        ]
    operations = payload.get("operations")
    if not isinstance(operations, list):
        return None, [
            build_issue(
                normalized_code="repair.patch_invalid",
                raw_code="missing_operations_list",
                layer=layer,
                family=family,
                path="operations",
                object_id=None,
                expected="list",
                actual=type(operations).__name__,
                message="Patch repair output must include an operations list.",
                repair_hint="Return only {\"operations\": []}",
                repairable=False,
            )
        ]
    validated: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    for idx, op in enumerate(operations):
        op_path = f"operations[{idx}]"
        if not isinstance(op, dict):
            issues.append(
                build_issue(
                    normalized_code="repair.patch_invalid",
                    raw_code="operation_not_object",
                    layer=layer,
                    family=family,
                    path=op_path,
                    object_id=None,
                    expected="object",
                    actual=type(op).__name__,
                    message="Patch operation must be an object.",
                    repair_hint="Return an object with path/action/value or path/action/reason.",
                    repairable=False,
                )
            )
            continue
        action = op.get("action")
        path = op.get("path")
        if not isinstance(path, str) or not path.strip():
            issues.append(
                build_issue(
                    normalized_code="repair.patch_invalid",
                    raw_code="operation_missing_path",
                    layer=layer,
                    family=family,
                    path=f"{op_path}.path",
                    object_id=None,
                    expected="non-empty string",
                    actual=type(path).__name__,
                    message="Patch operation path must be a non-empty string.",
                    repair_hint="Provide the exact target path for the operation.",
                    repairable=False,
                )
            )
            continue
        if action not in {"set", "replace", "remove", "append"}:
            issues.append(
                build_issue(
                    normalized_code="repair.patch_invalid",
                    raw_code="unsupported_operation_action",
                    layer=layer,
                    family=family,
                    path=path,
                    object_id=None,
                    expected="set|replace|remove|append",
                    actual=str(action),
                    message=f"Unsupported patch action {action!r}.",
                    repair_hint="Use only set, replace, remove, or append.",
                    repairable=False,
                )
            )
            continue
        validated.append(copy.deepcopy(op))
    if issues:
        return None, issues
    return validated, []


def apply_patch_operations(
    payload: dict[str, Any],
    operations: list[dict[str, Any]],
    *,
    allowed_roots: set[str],
    family: str | None,
    layer: str | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    updated = copy.deepcopy(payload)
    issues: list[dict[str, Any]] = []
    for op in operations:
        path = str(op["path"])
        action = str(op["action"])
        if not _path_allowed(path, action, allowed_roots):
            issues.append(
                build_issue(
                    normalized_code="repair.path_not_allowed",
                    raw_code="operation_outside_allowed_roots",
                    layer=layer,
                    family=family,
                    path=path,
                    object_id=None,
                    expected=", ".join(sorted(allowed_roots)),
                    actual=path,
                    message=f"Patch operation touches disallowed path {path!r}.",
                    repair_hint="Only touch allowed target paths for broken roots.",
                    repairable=False,
                )
            )
            continue
        try:
            tokens = _parse_path(path)
            if action == "append":
                current = updated
                for token in tokens:
                    if isinstance(token, str):
                        if not isinstance(current, dict) or token not in current:
                            raise ValueError(f"missing path segment: {token}")
                        current = current[token]
                    else:
                        if not isinstance(current, list) or token < 0 or token >= len(current):
                            raise ValueError(f"list index out of range: {token}")
                        current = current[token]
                if not isinstance(current, list):
                    raise ValueError("append target must be a list")
                current.append(copy.deepcopy(op.get("value")))
                continue

            parent, leaf = _resolve_parent(updated, tokens)
            if action in {"set", "replace"}:
                value = copy.deepcopy(op.get("value"))
                if isinstance(leaf, int):
                    if not isinstance(parent, list) or leaf < 0 or leaf >= len(parent):
                        raise ValueError(f"list index out of range: {leaf}")
                    parent[leaf] = value
                else:
                    if not isinstance(parent, dict):
                        raise ValueError(f"path parent for {path!r} is not an object")
                    parent[leaf] = value
            elif action == "remove":
                if isinstance(leaf, int):
                    if not isinstance(parent, list) or leaf < 0 or leaf >= len(parent):
                        raise ValueError(f"list index out of range: {leaf}")
                    del parent[leaf]
                else:
                    if not isinstance(parent, dict):
                        raise ValueError(f"path parent for {path!r} is not an object")
                    parent.pop(leaf, None)
        except ValueError as exc:
            issues.append(
                build_issue(
                    normalized_code="repair.patch_invalid",
                    raw_code="operation_apply_failed",
                    layer=layer,
                    family=family,
                    path=path,
                    object_id=None,
                    expected="valid patch application",
                    actual=str(exc),
                    message=f"Patch operation failed to apply at {path!r}: {exc}",
                    repair_hint="Return only valid operations targeting existing schema paths.",
                    repairable=False,
                )
            )
    return updated if not issues else copy.deepcopy(payload), issues


def drop_target_root(payload: dict[str, Any], root: str) -> dict[str, Any]:
    updated = copy.deepcopy(payload)
    tokens = _parse_path(root)
    try:
        parent, leaf = _resolve_parent(updated, tokens)
    except ValueError:
        return updated
    if isinstance(leaf, int):
        if isinstance(parent, list) and 0 <= leaf < len(parent):
            del parent[leaf]
        return updated
    if not isinstance(parent, dict):
        return updated
    current = parent.get(leaf)
    if isinstance(current, list):
        parent[leaf] = []
    else:
        parent.pop(leaf, None)
    return updated


def build_fact_flag(
    issue: dict[str, Any],
    *,
    producer: str,
    status: str,
    resolution: str | None,
    repair_attempted: bool,
) -> dict[str, Any]:
    if status not in ALLOWED_FLAG_STATUSES:
        raise ValueError(f"unsupported flag status: {status}")
    if resolution is not None and resolution not in ALLOWED_FLAG_RESOLUTIONS:
        raise ValueError(f"unsupported flag resolution: {resolution}")
    material = "|".join(
        [
            producer,
            issue.get("normalized_code") or "",
            issue.get("raw_code") or "",
            issue.get("path") or "",
            issue.get("object_id") or "",
            issue.get("message") or "",
        ]
    )
    flag_id = hashlib.sha1(material.encode("utf-8"), usedforsecurity=False).hexdigest()[:12]
    return build_object_flag(
        flag_id=f"flag_{flag_id}",
        producer=producer,
        category="extraction",
        severity=issue["severity"],
        message=issue["message"],
        status=status,
        code=issue.get("normalized_code"),
        path=issue.get("path"),
        object_id=issue.get("object_id"),
        resolution=resolution,
        repair_attempted=bool(repair_attempted),
        details={
            "raw_code": issue.get("raw_code"),
            "layer": issue.get("layer"),
            "family": issue.get("family"),
            "expected": issue.get("expected"),
            "actual": issue.get("actual"),
            "repair_hint": issue.get("repair_hint"),
        },
    )


def build_diagnostic(
    *,
    target_path: str,
    issue: dict[str, Any],
    repair_attempted: bool,
    status: str = "dropped",
) -> dict[str, Any]:
    return build_report_entry(
        target_path=target_path,
        status=status,
        issue=copy.deepcopy(issue),
        repair_attempted=bool(repair_attempted),
    )


def validate_fact_flags(value: Any) -> str | None:
    return validate_object_flags(value)
