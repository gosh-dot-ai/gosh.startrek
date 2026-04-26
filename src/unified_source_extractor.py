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
from pathlib import Path
from typing import Any

from .extraction_repair import (
    apply_patch_operations,
    build_issue,
    drop_target_root,
    group_issues_by_target_root,
    parse_json_object,
    parse_patch_operations,
    path_matches_root,
)
from .extraction_substrate import (
    SubstrateValidationError,
    build_fact_lookup,
    run_source_aggregation_validation_pipeline,
    support_text_from_fact_ids,
    validate_edge,
    validate_edge_grounding,
    validate_event,
    validate_event_grounding,
    validate_record,
    validate_record_grounding_source_level,
    validate_revision,
    validate_revision_grounding,
    validate_source_aggregation_payload,
    validate_structural_edge_direction,
)
from .object_flags import build_object_flag
from .object_reports import build_report, build_report_entry
from .prompt_safety import render_data_block

_PROMPT_DIR = Path(__file__).parent / "prompts" / "extraction"
_LAYER_ORDER = [
    "atomic_fact_layer",
    "locality_layer",
    "revision_currentness_layer",
    "event_layer",
    "record_layer",
    "edge_layer",
]
_SOURCE_ROOT_CONTRACT: dict[str, list[Any]] = {
    "revision_currentness": [],
    "events": [],
    "records": [],
    "edges": [],
}
_IGNORED_TOP_LEVEL_KEYS = {"atomic_facts"}
_ALLOWED_TOP_LEVEL_KEYS = set(_SOURCE_ROOT_CONTRACT)

_SOURCE_SCHEMA_SNIPPETS = {
    "revision_currentness": {
        "type": "list[revision]",
        "item_fields": [
            "revision_id",
            "topic_key",
            "old_fact_id",
            "new_fact_id",
            "link_type",
            "current_fact_id",
            "effective_date",
            "revision_source_fact_ids",
        ],
    },
    "events": {
        "type": "list[event]",
        "item_fields": [
            "event_id",
            "event_type",
            "participants",
            "object",
            "time",
            "location",
            "parameters",
            "outcome",
            "status",
            "support_fact_ids",
        ],
    },
    "records": {
        "type": "list[record]",
        "item_fields": [
            "record_id",
            "record_type",
            "item_id",
            "status",
            "date",
            "qualifier",
            "owner",
            "source_section",
            "support_fact_ids",
        ],
    },
    "edges": {
        "type": "list[edge]",
        "item_fields": [
            "edge_id",
            "edge_type",
            "from_id",
            "to_id",
            "edge_evidence_text",
            "anchor_key",
            "anchor_basis_fact_ids",
            "support_fact_ids",
        ],
    },
}


def _load_prompt(name: str) -> str:
    path = _PROMPT_DIR / f"{name}.md"
    return path.read_text(encoding="utf-8")


def _collection(body: dict, key: str) -> list:
    value = body.get(key, [])
    return value if isinstance(value, list) else []


def _deterministic_locality(source_id: str, episode: dict, episode_ids: list[str]) -> dict:
    episode_id = episode["episode_id"]
    idx = episode_ids.index(episode_id)
    neighbors = []
    if idx > 0:
        neighbors.append(episode_ids[idx - 1])
    if idx + 1 < len(episode_ids):
        neighbors.append(episode_ids[idx + 1])
    section_path = ((episode.get("metadata") or {}).get("source_section_path") or "").strip() or None
    return {
        "source_id": source_id,
        "episode_id": episode_id,
        "section_id": section_path,
        "heading": section_path,
        "table_id": None,
        "list_id": None,
        "paragraph_cluster_id": f"{episode_id}_p01",
        "neighbor_episode_ids": neighbors,
    }


def _episode_descriptor_block(episodes: list[dict]) -> str:
    lines: list[str] = []
    for ep in episodes:
        lines.append(f"[EPISODE {ep['episode_id']}]")
        source_date = (ep.get("source_date") or "").strip() or "unknown"
        lines.append(f"DATE: {source_date}")
        lines.append("TEXT:")
        lines.append(ep.get("raw_text", ""))
        lines.append("")
    return "\n".join(lines).strip()


def _grounded_fact_catalog(source_facts: list[dict]) -> list[dict]:
    catalog: list[dict] = []
    seen_ids: set[str] = set()
    for fact in source_facts:
        if not isinstance(fact, dict):
            continue
        fact_id = str(fact.get("id") or "").strip()
        fact_text = str(fact.get("fact") or "").strip()
        if not fact_id or not fact_text or fact_id in seen_ids:
            continue
        metadata = fact.get("metadata") or {}
        episode_id = str(
            metadata.get("episode_id")
            or metadata.get("episode_source_id")
            or fact.get("episode_id")
            or ""
        ).strip()
        entities = [
            entity
            for entity in (fact.get("entities") or [])
            if isinstance(entity, str) and entity.strip()
        ]
        catalog.append(
            {
                "fact_id": fact_id,
                "episode_id": episode_id or None,
                "fact_text": fact_text,
                "entity_ids": entities,
            }
        )
        seen_ids.add(fact_id)
    return catalog


def _base_atomic_facts(source_facts: list[dict]) -> list[dict]:
    atomic_facts: list[dict] = []
    for row in _grounded_fact_catalog(source_facts):
        fact_text = row["fact_text"]
        entity_ids = list(row.get("entity_ids") or [])
        subject = entity_ids[0] if entity_ids else (row.get("episode_id") or row["fact_id"])
        atomic_facts.append(
            {
                "fact_id": row["fact_id"],
                "subject": subject,
                "relation": "grounded_support",
                "object": fact_text,
                "value_text": fact_text,
                "value_number": None,
                "value_unit": None,
                "polarity": "positive",
                "confidence": 1.0,
                "source_span": fact_text,
                "source_span_start": None,
                "source_span_end": None,
                "asserted_at": None,
                "entity_ids": entity_ids,
                "episode_id": row.get("episode_id"),
            }
        )
    return atomic_facts


def _grounded_fact_payload(source_facts: list[dict]) -> str:
    return json.dumps(_grounded_fact_catalog(source_facts), ensure_ascii=False, indent=2)


def _base_user_payload(episodes: list[dict], source_facts: list[dict]) -> str:
    return "\n\n".join(
        [
            render_data_block("GROUNDED_FACT_CATALOG", _grounded_fact_payload(source_facts)),
            render_data_block("EPISODE_TEXTS", _episode_descriptor_block(episodes)),
        ]
    )


def _base_prompt(
    source_id: str,
    source_kind: str,
    episodes: list[dict],
    source_facts: list[dict],
    prompt_overrides: dict[str, str] | None = None,
) -> str:
    if prompt_overrides and "unified_source_aggregation" in prompt_overrides:
        prompt = prompt_overrides["unified_source_aggregation"]
    else:
        prompt = _load_prompt("unified_source_aggregation")
    return prompt.format(
        source_id=source_id,
        source_kind=source_kind,
    )


def _build_payload_envelope(
    source_id: str,
    source_kind: str,
    episodes: list[dict],
    source_facts: list[dict],
    body: dict,
) -> tuple[dict, dict[str, dict], dict[str, str]]:
    episode_ids = [ep["episode_id"] for ep in episodes]
    locality_by_episode = {
        ep["episode_id"]: _deterministic_locality(source_id, ep, episode_ids)
        for ep in episodes
    }
    source_text_by_episode = {
        ep["episode_id"]: ep.get("raw_text", "")
        for ep in episodes
    }
    payload = {
        "schema": "extraction_substrate",
        "payload_scope": "source_aggregation",
        "source_id": source_id,
        "source_kind": source_kind,
        "episode_ids": episode_ids,
        "locality_by_episode": locality_by_episode,
        "atomic_facts": _base_atomic_facts(source_facts),
        "revision_currentness": _collection(body, "revision_currentness"),
        "events": _collection(body, "events"),
        "records": _collection(body, "records"),
        "edges": _collection(body, "edges"),
    }
    return payload, locality_by_episode, source_text_by_episode


def _atomic_lookup(payload: dict) -> dict[str, dict]:
    return {fact["fact_id"]: fact for fact in payload.get("atomic_facts", []) if isinstance(fact, dict) and fact.get("fact_id")}


def _event_lookup(payload: dict) -> dict[str, dict]:
    return {event["event_id"]: event for event in payload.get("events", []) if isinstance(event, dict) and event.get("event_id")}


def _record_lookup(payload: dict) -> dict[str, dict]:
    return {record["record_id"]: record for record in payload.get("records", []) if isinstance(record, dict) and record.get("record_id")}


def _support_entity_ids(payload: dict, support_fact_ids: list[str]) -> list[str]:
    fact_lookup = _atomic_lookup(payload)
    out: list[str] = []
    seen: set[str] = set()
    for fact_id in support_fact_ids or []:
        fact = fact_lookup.get(fact_id) or {}
        for entity in fact.get("entity_ids", []) or []:
            if isinstance(entity, str) and entity and entity not in seen:
                seen.add(entity)
                out.append(entity)
    return out


def _event_summary(event: dict) -> str:
    parts: list[str] = []
    participants = event.get("participants") or []
    if participants:
        parts.append(", ".join(participants))
    event_type = (event.get("event_type") or "event").replace("_", " ")
    if event_type:
        parts.append(event_type)
    if event.get("object"):
        parts.append(f"for {event['object']}")
    if event.get("time"):
        parts.append(f"on {event['time']}")
    if event.get("location"):
        parts.append(f"at {event['location']}")
    params = []
    for param in event.get("parameters", []) or []:
        value_text = param.get("value_text")
        if value_text:
            params.append(f"{param.get('name')}: {value_text}")
    if params:
        parts.append(f"with {', '.join(params)}")
    if event.get("outcome"):
        parts.append(f"outcome {event['outcome']}")
    if event.get("status"):
        parts.append(f"status {event['status']}")
    return " ".join(parts).strip().rstrip(".") + "."


def _record_summary(record: dict) -> str:
    parts = [f"{record.get('record_type', 'record').replace('_', ' ')} {record.get('item_id', '')}".strip()]
    if record.get("status"):
        parts.append(f"status {record['status']}")
    if record.get("date"):
        parts.append(f"date {record['date']}")
    if record.get("qualifier"):
        parts.append(f"qualifier {record['qualifier']}")
    if record.get("owner"):
        parts.append(f"owner {record['owner']}")
    if record.get("source_section"):
        parts.append(f"section {record['source_section']}")
    return " ".join(parts).strip().rstrip(".") + "."


def _fact_summary(fact: dict) -> str:
    obj = fact.get("value_text") or fact.get("object") or ""
    return f"{fact.get('subject', '').strip()} {fact.get('relation', '').strip().replace('_', ' ')} {obj}".strip()


def _episode_id_from_fact(payload: dict, fact_id: str) -> str | None:
    fact = _atomic_lookup(payload).get(fact_id) or {}
    episode_id = str(fact.get("episode_id") or "").strip()
    if episode_id:
        return episode_id
    return _episode_id_from_fact_id(fact_id)


def _episode_metadata(payload: dict, fact_ids: list[str]) -> dict[str, Any]:
    episode_ids: set[str] = set()
    for fact_id in fact_ids:
        episode_id = _episode_id_from_fact(payload, fact_id)
        if episode_id:
            episode_ids.add(episode_id)
    metadata: dict[str, Any] = {
        "source_aggregation": True,
        "episode_ids": sorted(episode_ids),
    }
    if len(metadata["episode_ids"]) == 1:
        metadata["episode_id"] = metadata["episode_ids"][0]
    return metadata


def _node_summary(payload: dict, node_id: str) -> str:
    event = _event_lookup(payload).get(node_id)
    if event:
        return _event_summary(event).rstrip(".")
    record = _record_lookup(payload).get(node_id)
    if record:
        return _record_summary(record).rstrip(".")
    fact = _atomic_lookup(payload).get(node_id)
    if fact:
        return _fact_summary(fact)
    return node_id


def flatten_source_aggregation_payload(payload: dict) -> list[dict]:
    facts: list[dict] = []
    fact_lookup = _atomic_lookup(payload)

    for revision in payload.get("revision_currentness", []) or []:
        old_fact = fact_lookup.get(revision["old_fact_id"], {})
        new_fact = fact_lookup.get(revision["new_fact_id"], {})
        text = (
            f"Current value for {revision['topic_key'].replace('_', ' ')} is "
            f"{new_fact.get('value_text') or new_fact.get('object') or _fact_summary(new_fact)}; "
            f"this supersedes {old_fact.get('value_text') or old_fact.get('object') or _fact_summary(old_fact)}"
        )
        if revision.get("effective_date"):
            text += f" effective {revision['effective_date']}"
        facts.append(
            {
                "id": revision["revision_id"],
                "fact": text.rstrip(".") + ".",
                "kind": "fact",
                "entities": _support_entity_ids(payload, revision.get("revision_source_fact_ids", [])),
                "tags": ["substrate", "revision_currentness"],
                "source_ids": list(revision.get("revision_source_fact_ids", [])),
                "metadata": {
                    "substrate_layer": "revision_currentness_layer",
                    **_episode_metadata(payload, list(revision.get("revision_source_fact_ids", []))),
                },
            }
        )

    for event in payload.get("events", []) or []:
        facts.append(
            {
                "id": event["event_id"],
                "fact": _event_summary(event),
                "kind": "fact",
                "entities": list(dict.fromkeys((event.get("participants") or []) + _support_entity_ids(payload, event.get("support_fact_ids", [])))),
                "tags": ["substrate", "event", event.get("event_type", "event")],
                "source_ids": list(event.get("support_fact_ids", [])),
                "metadata": {
                    "substrate_layer": "event_layer",
                    **_episode_metadata(payload, list(event.get("support_fact_ids", []))),
                },
            }
        )

    for record in payload.get("records", []) or []:
        facts.append(
            {
                "id": record["record_id"],
                "fact": _record_summary(record),
                "kind": "fact",
                "entities": _support_entity_ids(payload, record.get("support_fact_ids", [])),
                "tags": ["substrate", "record", record.get("record_type", "record")],
                "source_ids": list(record.get("support_fact_ids", [])),
                "metadata": {
                    "substrate_layer": "record_layer",
                    **_episode_metadata(payload, list(record.get("support_fact_ids", []))),
                },
            }
        )

    for edge in payload.get("edges", []) or []:
        edge_type = edge.get("edge_type")
        if edge_type in {"belongs_to_event", "belongs_to_record"}:
            continue
        if edge_type == "same_anchor":
            text = (
                f"{_node_summary(payload, edge['from_id'])} and {_node_summary(payload, edge['to_id'])} "
                f"share the same anchor: {edge.get('anchor_key')}."
            )
        else:
            text = (
                f"{_node_summary(payload, edge['from_id'])} {edge_type.replace('_', ' ')} "
                f"{_node_summary(payload, edge['to_id'])}."
            )
            if edge.get("edge_evidence_text"):
                text = text.rstrip(".") + f" Evidence: {edge['edge_evidence_text']}."
        facts.append(
            {
                "id": edge["edge_id"],
                "fact": text,
                "kind": "fact",
                "entities": _support_entity_ids(payload, edge.get("support_fact_ids", [])),
                "tags": ["substrate", "edge", edge_type],
                "source_ids": list(edge.get("support_fact_ids", [])),
                "metadata": {
                    "substrate_layer": "edge_layer",
                    **_episode_metadata(payload, list(edge.get("support_fact_ids", []))),
                },
            }
        )

    return facts


def _episode_id_from_fact_id(fact_id: str) -> str | None:
    match = re.match(r"^ep_(.+?)_f(?:_|$)", fact_id or "")
    if not match:
        return None
    return match.group(1)



def _issue_payload(
    issues: list[dict]) -> str:
    return json.dumps(issues, ensure_ascii=False, indent=2)


def _context_payload(episodes: list[dict], source_facts: list[dict]) -> str:
    return (
        "Grounded fact catalog:\n"
        + _grounded_fact_payload(source_facts)
        + "\n\nEpisode texts:\n"
        + _episode_descriptor_block(episodes)
    )


def _render_root_repair_prompt(
    *,
    previous_raw: Any,
    issues: list[dict],
    episodes: list[dict],
    source_facts: list[dict],
    prompt_overrides: dict[str, str] | None,
) -> str:
    override = prompt_overrides.get("structured_json_root_repair") if prompt_overrides else None
    template = override if isinstance(override, str) else _load_prompt("structured_json_root_repair")
    return template.format(
        root_contract_json=json.dumps(_SOURCE_ROOT_CONTRACT, ensure_ascii=False, indent=2),
        previous_raw=str(previous_raw),
        issues_json=_issue_payload(issues),
        context_payload=_context_payload(episodes, source_facts),
    )


def _target_schema_payload(roots: list[str]) -> str:
    payload: dict[str, Any] = {}
    for root in roots:
        collection = root.split("[", 1)[0]
        payload[root] = _SOURCE_SCHEMA_SNIPPETS.get(collection, {"type": "unknown"})
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _render_patch_repair_prompt(
    *,
    payload: dict,
    issues: list[dict],
    roots: list[str],
    episodes: list[dict],
    source_facts: list[dict],
    prompt_overrides: dict[str, str] | None,
) -> str:
    override = prompt_overrides.get("structured_item_patch_repair") if prompt_overrides else None
    template = override if isinstance(override, str) else _load_prompt("structured_item_patch_repair")
    return template.format(
        allowed_paths_json=json.dumps(sorted(roots), ensure_ascii=False, indent=2),
        issues_json=_issue_payload(issues),
        current_payload_json=json.dumps(payload, ensure_ascii=False, indent=2),
        target_schema_json=_target_schema_payload(roots),
        context_payload=_context_payload(episodes, source_facts),
    )


def _relative_path_value(obj: Any, relative_path: str) -> Any:
    if relative_path in {"event", "record", "revision_currentness", "edge"}:
        return obj
    current = obj
    for segment in relative_path.split("."):
        match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)(?:\[(\d+)\])?", segment)
        if not match:
            return None
        key, index = match.groups()
        if not isinstance(current, dict):
            return None
        current = current.get(key)
        if index is not None:
            if not isinstance(current, list):
                return None
            idx = int(index)
            if idx < 0 or idx >= len(current):
                return None
            current = current[idx]
    return current


def _prefixed_path(root_path: str, relative_path: str | None) -> str:
    if not relative_path or relative_path in {"event", "record", "revision_currentness", "edge"}:
        return root_path
    return f"{root_path}.{relative_path}"


def _shape_issue_from_validation_error(
    *,
    raw_code: str,
    message: str,
    root_path: str,
    obj: dict,
    family: str,
    object_id: str | None,
    layer: str,
) -> dict:
    relative_path = message.split(":", 1)[0].strip() if ":" in message else None
    path = _prefixed_path(root_path, relative_path)
    actual_value = _relative_path_value(obj, relative_path or "")
    actual = type(actual_value).__name__ if actual_value is not None else None
    normalized = "shape.type_error"
    expected = None
    repair_hint = None
    if "unknown id reference" in message:
        normalized = "ref.unknown_id"
        expected = "known referenced id"
        repair_hint = "Replace the unknown reference with an existing grounded id or remove the object."
    elif "must reference an event_id" in message or "must reference a record_id" in message:
        normalized = "ref.type_mismatch"
        expected = "reference of the required object type"
        repair_hint = "Point the relation to the correct object type."
    elif "must not reference" in message:
        normalized = "ref.invalid_direction"
        expected = "valid structural edge direction"
        repair_hint = "Flip the edge direction or remove the invalid edge."
    elif "requires non-empty" in message:
        normalized = "ref.cardinality_violation"
        expected = "non-empty supporting reference list"
        repair_hint = "Populate the required support ids or remove the object."
    elif "expected one of" in message:
        normalized = "enum.invalid_value"
        expected = message.split("expected ", 1)[-1]
        repair_hint = "Set the field to one of the allowed enum values."
    elif "expected ISO date" in message:
        normalized = "field.invalid_format"
        expected = "ISO date YYYY-MM-DD"
        repair_hint = "Use an ISO date string or null if unsupported."
    elif "expected float in [0, 1]" in message:
        normalized = "field.invalid_range"
        expected = "float in [0, 1]"
        repair_hint = "Clamp confidence into [0, 1] or set it to null."
    elif "expected object" in message and relative_path in {"event", "record", "revision_currentness", "edge"}:
        normalized = "shape.object_item_type_error"
        expected = "object"
        repair_hint = "Replace the malformed item with a valid object or remove it."
    elif "expected non-empty string" in message:
        if actual_value is None:
            normalized = "field.missing_required"
            expected = "non-empty string"
            repair_hint = "Populate the required field with a grounded string or remove the object."
        elif isinstance(actual_value, str) and not actual_value.strip():
            normalized = "field.empty_required"
            expected = "non-empty string"
            repair_hint = "Populate the field with a grounded string or remove the object."
        else:
            normalized = "shape.type_error"
            expected = "non-empty string"
            repair_hint = "Set the field to the correct scalar type."
    elif "expected list" in message:
        normalized = "shape.type_error"
        expected = "list"
        repair_hint = "Set the field to a list of supported values."
    elif "expected int" in message or "expected int or float" in message:
        normalized = "shape.type_error"
        expected = "numeric"
        repair_hint = "Set the field to a numeric value or null."

    return build_issue(
        normalized_code=normalized,
        raw_code=raw_code,
        layer=layer,
        family=family,
        path=path,
        object_id=object_id,
        expected=expected,
        actual=actual,
        message=message,
        repair_hint=repair_hint,
        repairable=True,
    )


def _grounding_issue_from_validation_error(
    *,
    raw_code: str,
    message: str,
    root_path: str,
    family: str,
    object_id: str | None,
    layer: str,
) -> dict:
    relative_path = message.split(":", 1)[0].strip() if ":" in message else None
    path = _prefixed_path(root_path, relative_path)
    normalized = "grounding.value_not_supported"
    expected = "grounded value from source support"
    repair_hint = "Remove unsupported fields or remove the object if the field is required."
    if "source_span not found" in message:
        normalized = "grounding.span_not_found"
        expected = "source span present in support text"
    elif "source_span does not match" in message or "invalid source span offsets" in message:
        normalized = "grounding.offset_mismatch"
        expected = "consistent span offsets"
    elif "date" in message and "not grounded" in message:
        normalized = "grounding.date_not_supported"
        expected = "grounded date in support text"
    elif "numeric value" in message:
        normalized = "grounding.number_unit_not_supported"
        expected = "grounded numeric value and unit"
    elif "anchor_key" in message:
        normalized = "grounding.anchor_not_supported"
        expected = "grounded anchor key"
    elif relative_path and relative_path.startswith("participants"):
        normalized = "grounding.entity_not_supported"
        expected = "grounded participant/entity"
    elif "locality" in message or "source_section" in message:
        normalized = "grounding.locality_not_supported"
        expected = "grounded locality metadata"
    return build_issue(
        normalized_code=normalized,
        raw_code=raw_code,
        layer=layer,
        family=family,
        path=path,
        object_id=object_id,
        expected=expected,
        actual=None,
        message=message,
        repair_hint=repair_hint,
        repairable=True,
    )


def _append_top_level_shape_issues(body: dict, *, family: str, issues: list[dict]) -> None:
    keys = set(body)
    for key in sorted(_ALLOWED_TOP_LEVEL_KEYS - keys):
        issues.append(
            build_issue(
                normalized_code="shape.missing_top_level_key",
                raw_code="missing_top_level_key",
                layer="top_level",
                family=family,
                path=key,
                object_id=None,
                expected="list",
                actual=None,
                message=f"Top-level key {key!r} is required.",
                repair_hint=f"Add the missing {key} list.",
                repairable=True,
            )
        )
    for key in sorted(keys - _ALLOWED_TOP_LEVEL_KEYS - _IGNORED_TOP_LEVEL_KEYS):
        issues.append(
            build_issue(
                normalized_code="shape.unexpected_top_level_key",
                raw_code="unexpected_top_level_key",
                layer="top_level",
                family=family,
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
                    family=family,
                    path=key,
                    object_id=None,
                    expected="list",
                    actual=type(body.get(key)).__name__,
                    message=f"Top-level key {key!r} must be a list.",
                    repair_hint=f"Replace {key} with a list.",
                    repairable=True,
                )
            )


def _append_duplicate_id_issues(items: list, *, list_name: str, id_field: str, layer: str, family: str, issues: list[dict]) -> None:
    seen: dict[str, int] = {}
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        object_id = str(item.get(id_field) or "").strip()
        if not object_id:
            continue
        if object_id in seen:
            issues.append(
                build_issue(
                    normalized_code="id.duplicate",
                    raw_code="duplicate_ids",
                    layer=layer,
                    family=family,
                    path=f"{list_name}[{idx}].{id_field}",
                    object_id=object_id,
                    expected="unique id",
                    actual=object_id,
                    message=f"Duplicate id {object_id!r} detected in {list_name}.",
                    repair_hint="Rename or remove the duplicate object.",
                    repairable=True,
                )
            )
        else:
            seen[object_id] = idx


def _append_cross_layer_collision_issues(payload: dict, *, family: str, issues: list[dict]) -> None:
    atomic_ids = {fact.get("fact_id") for fact in payload.get("atomic_facts", []) if isinstance(fact, dict)}
    seen: dict[str, str] = {str(value): "atomic_facts" for value in atomic_ids if value}
    for list_name, id_field, layer in (
        ("events", "event_id", "event_layer"),
        ("records", "record_id", "record_layer"),
    ):
        for idx, item in enumerate(payload.get(list_name, [])):
            if not isinstance(item, dict):
                continue
            object_id = str(item.get(id_field) or "").strip()
            if not object_id:
                continue
            owner = seen.get(object_id)
            if owner and owner != list_name:
                issues.append(
                    build_issue(
                        normalized_code="id.cross_layer_collision",
                        raw_code="cross_layer_id_collision",
                        layer=layer,
                        family=family,
                        path=f"{list_name}[{idx}].{id_field}",
                        object_id=object_id,
                        expected="id unique across layers",
                        actual=object_id,
                        message=f"Id {object_id!r} collides across layers ({owner} vs {list_name}).",
                        repair_hint="Rename or remove the colliding object.",
                        repairable=True,
                    )
                )
            seen[object_id] = list_name


def _object_id_for_root(payload: dict, root: str) -> str | None:
    match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)\[(\d+)\]", root)
    if not match:
        return None
    collection, index_s = match.groups()
    index = int(index_s)
    items = payload.get(collection) or []
    if not isinstance(items, list) or index < 0 or index >= len(items):
        return None
    item = items[index]
    if not isinstance(item, dict):
        return None
    for key in ("revision_id", "event_id", "record_id", "edge_id"):
        value = str(item.get(key) or "").strip()
        if value:
            return value
    return None


def _collect_source_aggregation_issues(
    *,
    source_id: str,
    source_kind: str,
    episodes: list[dict],
    source_facts: list[dict],
    body: dict,
) -> tuple[dict, dict[str, dict], dict[str, str], list[dict]]:
    payload, locality_by_episode, source_text_by_episode = _build_payload_envelope(
        source_id,
        source_kind,
        episodes,
        source_facts,
        body,
    )
    issues: list[dict] = []
    _append_top_level_shape_issues(body, family=source_kind, issues=issues)

    fact_lookup = build_fact_lookup(payload["atomic_facts"])
    fact_ids = set(fact_lookup)

    _append_duplicate_id_issues(payload.get("revision_currentness", []), list_name="revision_currentness", id_field="revision_id", layer="revision_currentness_layer", family=source_kind, issues=issues)
    _append_duplicate_id_issues(payload.get("events", []), list_name="events", id_field="event_id", layer="event_layer", family=source_kind, issues=issues)
    _append_duplicate_id_issues(payload.get("records", []), list_name="records", id_field="record_id", layer="record_layer", family=source_kind, issues=issues)
    _append_duplicate_id_issues(payload.get("edges", []), list_name="edges", id_field="edge_id", layer="edge_layer", family=source_kind, issues=issues)
    _append_cross_layer_collision_issues(payload, family=source_kind, issues=issues)

    for list_name, layer, validator, grounding in (
        ("revision_currentness", "revision_currentness_layer", validate_revision, validate_revision_grounding),
        ("events", "event_layer", validate_event, validate_event_grounding),
        ("records", "record_layer", validate_record, None),
    ):
        for idx, item in enumerate(payload.get(list_name, [])):
            root_path = f"{list_name}[{idx}]"
            if not isinstance(item, dict):
                issues.append(
                    build_issue(
                        normalized_code="shape.list_item_type_error",
                        raw_code=f"invalid_{list_name}_item_type",
                        layer=layer,
                        family=source_kind,
                        path=root_path,
                        object_id=None,
                        expected="object",
                        actual=type(item).__name__,
                        message=f"{root_path} must be an object.",
                        repair_hint="Replace the malformed item with a valid object or remove it.",
                        repairable=True,
                    )
                )
                continue
            object_id = _object_id_for_root(payload, root_path)
            try:
                validator(item, fact_ids)
            except SubstrateValidationError as exc:
                issues.append(
                    _shape_issue_from_validation_error(
                        raw_code=exc.code,
                        message=str(exc),
                        root_path=root_path,
                        obj=item,
                        family=source_kind,
                        object_id=object_id,
                        layer=layer,
                    )
                )
                continue
            try:
                if list_name == "records":
                    validate_record_grounding_source_level(item, fact_lookup, locality_by_episode)
                elif grounding is not None:
                    grounding(item, fact_lookup)
            except SubstrateValidationError as exc:
                issues.append(
                    _grounding_issue_from_validation_error(
                        raw_code=exc.code,
                        message=str(exc),
                        root_path=root_path,
                        family=source_kind,
                        object_id=object_id,
                        layer=layer,
                    )
                )

    event_ids: set[str] = {str(obj.get("event_id")) for obj in payload.get("events", []) if isinstance(obj, dict) and isinstance(obj.get("event_id"), str) and str(obj.get("event_id")).strip()}
    record_ids: set[str] = {str(obj.get("record_id")) for obj in payload.get("records", []) if isinstance(obj, dict) and isinstance(obj.get("record_id"), str) and str(obj.get("record_id")).strip()}
    all_node_ids: set[str] = fact_ids | event_ids | record_ids
    for idx, item in enumerate(payload.get("edges", [])):
        root_path = f"edges[{idx}]"
        if not isinstance(item, dict):
            issues.append(
                build_issue(
                    normalized_code="shape.list_item_type_error",
                    raw_code="invalid_edges_item_type",
                    layer="edge_layer",
                    family=source_kind,
                    path=root_path,
                    object_id=None,
                    expected="object",
                    actual=type(item).__name__,
                    message=f"{root_path} must be an object.",
                    repair_hint="Replace the malformed item with a valid object or remove it.",
                    repairable=True,
                )
            )
            continue
        object_id = _object_id_for_root(payload, root_path)
        try:
            validate_edge(item, all_node_ids, fact_ids)
            validate_structural_edge_direction(item, event_ids, record_ids)
        except SubstrateValidationError as exc:
            issues.append(
                _shape_issue_from_validation_error(
                    raw_code=exc.code,
                    message=str(exc),
                    root_path=root_path,
                    obj=item,
                    family=source_kind,
                    object_id=object_id,
                    layer="edge_layer",
                )
            )
            continue
        try:
            validate_edge_grounding(item, fact_lookup)
        except SubstrateValidationError as exc:
            issues.append(
                _grounding_issue_from_validation_error(
                    raw_code=exc.code,
                    message=str(exc),
                    root_path=root_path,
                    family=source_kind,
                    object_id=object_id,
                    layer="edge_layer",
                )
            )

    higher_order_count = sum(len(payload.get(key, []) or []) for key in _ALLOWED_TOP_LEVEL_KEYS)
    if higher_order_count <= 0:
        issues.append(
            build_issue(
                normalized_code="semantic.empty_higher_order_success",
                raw_code="empty_source_aggregation",
                layer="top_level",
                family=source_kind,
                path="events",
                object_id=None,
                expected="at least one grounded higher-order object",
                actual="0 objects",
                message="source_aggregation: expected at least one revision_currentness/event/record/edge object",
                repair_hint="Append a grounded higher-order object or keep the payload empty only when no grounded structure exists.",
                repairable=True,
            )
        )

    return payload, locality_by_episode, source_text_by_episode, issues


def _attach_repair_flags(derived_facts: list[dict], outcomes: dict[str, dict]) -> None:
    flags_by_id: dict[str, list[dict]] = {}
    for outcome in outcomes.values():
        if outcome.get("status") != "resolved":
            continue
        object_id = outcome.get("object_id")
        if not object_id:
            continue
        flags = [
            build_object_flag(
                producer="unified_source_extractor",
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
    for fact in derived_facts:
        object_id = str(fact.get("id") or "")
        if object_id and object_id in flags_by_id:
            existing = list(fact.get("flags") or [])
            fact["flags"] = existing + list(flags_by_id[object_id])


def _merge_failure_reasons(*groups: list[str]) -> list[str]:
    out: list[str] = []
    for group in groups:
        for value in group:
            if value and value not in out:
                out.append(value)
    return out


def _base_validation(payload: dict) -> dict:
    return {
        "payload": payload,
        "aggregation_status": "accepted",
        "accepted_layers": list(_LAYER_ORDER),
        "dropped_layers": [],
        "failure_reasons": [],
    }


def _failed_validation(failure_reasons: list[str]) -> dict:
    return {
        "payload": None,
        "aggregation_status": "failed",
        "accepted_layers": [],
        "dropped_layers": [],
        "failure_reasons": failure_reasons,
    }


def _build_source_aggregation_report(entries: list[dict], *, validation: dict | None = None, status: str | None = None) -> dict:
    dropped = sum(1 for entry in entries if isinstance(entry, dict) and entry.get("status") == "dropped")
    resolved = sum(1 for entry in entries if isinstance(entry, dict) and entry.get("status") == "resolved")
    final_status = status or ("partial" if entries else "ok")
    summary: dict[str, Any] = {
        "entry_count": len(entries),
        "dropped_count": dropped,
        "resolved_count": resolved,
    }
    if isinstance(validation, dict) and validation:
        summary["validation"] = validation
    return build_report(
        report_kind="source_aggregation",
        producer="unified_source_extractor",
        status=final_status,
        entries=entries,
        summary=summary,
    )


def _guardrail_failure_reasons(report_entries: list[dict], final_issues: list[dict], guardrail_result: dict | None) -> list[str]:
    diagnostic_codes = [entry.get("issue", {}).get("raw_code") for entry in report_entries if isinstance(entry, dict) and isinstance(entry.get("issue"), dict)]
    issue_codes = [issue["raw_code"] for issue in final_issues]
    guardrail_codes = list((guardrail_result or {}).get("failure_reasons", []))
    return _merge_failure_reasons(diagnostic_codes, issue_codes, guardrail_codes)


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


def _repair_outcomes(
    *,
    original_payload: dict,
    candidate_payload: dict,
    original_issues: list[dict],
    candidate_issues: list[dict],
) -> tuple[dict, list[dict], dict]:
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


async def extract_source_aggregation(
    *,
    source_id: str,
    source_kind: str,
    episodes: list[dict],
    source_facts: list[dict],
    model: str,
    call_extract_fn,
    prompt_overrides: dict[str, str] | None = None,
) -> dict | None:
    if not episodes or not source_facts:
        return None

    system_prompt = _base_prompt(source_id, source_kind, episodes, source_facts, prompt_overrides=prompt_overrides)
    user_msg = _base_user_payload(episodes, source_facts)
    raw = await call_extract_fn(model, system_prompt, user_msg, max_tokens=8192)
    parsed, parse_issues = parse_json_object(raw, layer="top_level", family=source_kind)
    if parsed is None:
        repair_prompt = _render_root_repair_prompt(
            previous_raw=raw,
            issues=parse_issues,
            episodes=episodes,
            source_facts=source_facts,
            prompt_overrides=prompt_overrides,
        )
        repaired_raw = await call_extract_fn(model, repair_prompt, "", max_tokens=8192)
        parsed, parse_issues = parse_json_object(repaired_raw, layer="top_level", family=source_kind)
        if parsed is None:
            reasons = _merge_failure_reasons([issue["normalized_code"] for issue in parse_issues], ["source_aggregation_retry_exhausted"])
            return {
                "validation": _failed_validation(reasons),
                "derived_facts": [],
                "source_aggregation_report": _build_source_aggregation_report(_parse_failure_report_entries(parse_issues, repair_attempted=True), validation=_failed_validation(reasons), status="failed"),
            }

    payload, locality_by_episode, source_text_by_episode, issues = _collect_source_aggregation_issues(
        source_id=source_id,
        source_kind=source_kind,
        episodes=episodes,
        source_facts=source_facts,
        body=parsed,
    )
    if not issues:
        return {
            "validation": _base_validation(payload),
            "derived_facts": flatten_source_aggregation_payload(payload),
            "source_aggregation_report": _build_source_aggregation_report([], validation=_base_validation(payload), status="ok"),
        }

    roots = sorted(group_issues_by_target_root(issues))
    patch_prompt = _render_patch_repair_prompt(
        payload=parsed,
        issues=issues,
        roots=roots,
        episodes=episodes,
        source_facts=source_facts,
        prompt_overrides=prompt_overrides,
    )
    patch_raw = await call_extract_fn(model, patch_prompt, "", max_tokens=8192)
    operations, patch_issues = parse_patch_operations(
        patch_raw,
        producer="unified_source_extractor",
        family=source_kind,
        layer="top_level",
    )
    candidate_body = parsed
    if operations is not None:
        candidate_body, apply_issues = apply_patch_operations(
            parsed,
            operations,
            allowed_roots=set(roots),
            family=source_kind,
            layer="top_level",
        )
        patch_issues.extend(apply_issues)

    candidate_payload, _, _, candidate_issues = _collect_source_aggregation_issues(
        source_id=source_id,
        source_kind=source_kind,
        episodes=episodes,
        source_facts=source_facts,
        body=candidate_body,
    )
    candidate_issues.extend(patch_issues)
    repaired_payload, report_entries, outcomes = _repair_outcomes(
        original_payload=parsed,
        candidate_payload=candidate_body,
        original_issues=issues,
        candidate_issues=candidate_issues,
    )

    final_payload, _, _, final_issues = _collect_source_aggregation_issues(
        source_id=source_id,
        source_kind=source_kind,
        episodes=episodes,
        source_facts=source_facts,
        body=repaired_payload,
    )

    if final_issues:
        guardrail_result = run_source_aggregation_validation_pipeline(
            [final_payload],
            locality_metadata_by_episode=locality_by_episode,
            episode_count=len(episodes),
            source_text_by_episode=source_text_by_episode,
        )
        validated_payload = guardrail_result.get("payload")
        if not validated_payload:
            guardrail_result["failure_reasons"] = _guardrail_failure_reasons(report_entries, final_issues, guardrail_result)
            return {
                "validation": guardrail_result,
                "derived_facts": [],
                "source_aggregation_report": _build_source_aggregation_report(report_entries, validation=guardrail_result, status="failed"),
            }
        derived_facts = flatten_source_aggregation_payload(validated_payload)
        _attach_repair_flags(derived_facts, outcomes)
        guardrail_result["failure_reasons"] = _guardrail_failure_reasons(report_entries, final_issues, guardrail_result)
        return {
            "validation": guardrail_result,
            "derived_facts": derived_facts,
            "source_aggregation_report": _build_source_aggregation_report(report_entries, validation=guardrail_result, status="partial"),
        }

    try:
        validate_source_aggregation_payload(
            final_payload,
            locality_metadata_by_episode=locality_by_episode,
            source_text_by_episode=source_text_by_episode,
        )
    except SubstrateValidationError as exc:
        failure_reasons = _merge_failure_reasons([exc.code], ["source_aggregation_retry_exhausted"])
        return {
            "validation": _failed_validation(failure_reasons),
            "derived_facts": [],
            "source_aggregation_report": _build_source_aggregation_report(report_entries, validation=_failed_validation(failure_reasons), status="failed"),
        }

    derived_facts = flatten_source_aggregation_payload(final_payload)
    _attach_repair_flags(derived_facts, outcomes)
    return {
        "validation": _base_validation(final_payload),
        "derived_facts": derived_facts,
        "source_aggregation_report": _build_source_aggregation_report(report_entries, validation=_base_validation(final_payload), status="partial" if report_entries else "ok"),
    }
