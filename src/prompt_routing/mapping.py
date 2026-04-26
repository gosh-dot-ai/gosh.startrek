# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from ..query_lexicon import (
    PROMPT_ROUTING_CODE_CHAIN_MARKERS as _CODE_CHAIN_MARKERS,
)
from ..query_lexicon import (
    PROMPT_ROUTING_CODE_SLOT_MARKERS as _CODE_SLOT_MARKERS,
)
from ..query_lexicon import (
    PROMPT_ROUTING_MIXED_CODE_PROSE_MARKERS as _MIXED_CODE_PROSE_MARKERS,
)
from ..query_lexicon import (
    PROMPT_ROUTING_RISK_REVIEW_MARKERS as _RISK_REVIEW_MARKERS,
)

_RETRIEVAL_TO_INF = {
    "default": "lookup",
    "counting": "aggregate",
    "temporal": "temporal",
    "current": "current",
    "rule": "procedural",
    "synthesis": "synthesize",
    "prospective": "prospective",
    "summarize": "summarize",
    "icl": "icl",
    "supersession": "current",
    "exact_copy": "container_exact_copy",
}

_PROSE_FAMILIES = {"conversation", "document"}
_SUPPORTED_FAMILIES = _PROSE_FAMILIES | {"codebase"}

LeafMatcher = Callable[[str, dict[str, Any], str, Mapping[str, Any]], bool]


def _normalize_family(value: Any) -> str | None:
    normalized = str(value or "").strip().lower()
    if normalized in _SUPPORTED_FAMILIES:
        return normalized
    return None


def _retrieval_families(recall_result: Mapping[str, Any] | None) -> tuple[str, ...]:
    result = recall_result or {}
    families: list[str] = []

    for family in result.get("retrieval_families") or []:
        normalized = _normalize_family(family)
        if normalized is not None:
            families.append(normalized)

    mixed_trace = ((result.get("runtime_trace") or {}).get("mixed_family_merge") or {})
    for family in mixed_trace.get("merged_families") or []:
        normalized = _normalize_family(family)
        if normalized is not None:
            families.append(normalized)

    return tuple(dict.fromkeys(families))


def _explicit_non_code_family(recall_result: Mapping[str, Any] | None) -> bool:
    search_family = _normalize_family((recall_result or {}).get("search_family"))
    return search_family in _PROSE_FAMILIES


def _has_codebase_evidence(recall_result: Mapping[str, Any] | None) -> bool:
    families = set(_retrieval_families(recall_result))
    if "codebase" in families:
        return True
    runtime_trace = (recall_result or {}).get("runtime_trace") or {}
    code_trace = (runtime_trace.get("codebase_augmentation") or {})
    codebase_trace = (runtime_trace.get("codebase_context") or {})
    return (
        str(code_trace.get("mode") or "").strip().lower() in {"whole_file", "windowed_file", "hydrated", "hot_only"}
        or str(codebase_trace.get("mode") or "").strip().lower() == "active"
    )


def _has_prose_evidence(recall_result: Mapping[str, Any] | None) -> bool:
    families = set(_retrieval_families(recall_result))
    return bool(families & _PROSE_FAMILIES)


def _is_codebase_only_recall(recall_result: Mapping[str, Any] | None) -> bool:
    families = set(_retrieval_families(recall_result))
    if families:
        return families == {"codebase"}
    return _has_codebase_evidence(recall_result) and not _has_prose_evidence(recall_result)


def _is_mixed_codebase_recall(recall_result: Mapping[str, Any] | None) -> bool:
    families = set(_retrieval_families(recall_result))
    return "codebase" in families and bool(families & _PROSE_FAMILIES)


def _runtime_code_query_mode(recall_result: Mapping[str, Any] | None) -> str | None:
    mode = ((((recall_result or {}).get("runtime_trace") or {}).get("query") or {}).get("code_query_mode"))
    normalized = str(mode or "").strip().lower()
    if normalized in {"non_code", "precise_code", "mixed_code_plus_prose"}:
        return normalized
    return None


def _query_has_any(query: str, markers: tuple[str, ...]) -> bool:
    lowered = str(query or "").lower()
    return any(marker in lowered for marker in markers)


def _classify_code_query_mode(query: str, recall_result: Mapping[str, Any] | None) -> str:
    runtime_mode = _runtime_code_query_mode(recall_result)
    if runtime_mode is not None:
        return runtime_mode
    if _query_has_any(query, _MIXED_CODE_PROSE_MARKERS) and _query_has_any(
        query,
        _CODE_SLOT_MARKERS + _CODE_CHAIN_MARKERS + _RISK_REVIEW_MARKERS,
    ):
        return "mixed_code_plus_prose"
    if _query_has_any(query, _CODE_SLOT_MARKERS + _CODE_CHAIN_MARKERS + _RISK_REVIEW_MARKERS):
        return "precise_code"
    return "non_code"


def _is_code_slot_query(query: str, recall_result: Mapping[str, Any] | None) -> bool:
    if _classify_code_query_mode(query, recall_result) != "precise_code":
        return False
    return _query_has_any(query, _CODE_SLOT_MARKERS)


def _is_code_chain_query(query: str, recall_result: Mapping[str, Any] | None) -> bool:
    if _classify_code_query_mode(query, recall_result) == "non_code":
        return False
    return _query_has_any(query, _CODE_CHAIN_MARKERS)


def _is_risk_review_query(query: str, recall_result: Mapping[str, Any] | None) -> bool:
    if _classify_code_query_mode(query, recall_result) == "non_code":
        return False
    return _query_has_any(query, _RISK_REVIEW_MARKERS)


def _match_codebase_mixed(
    prompt_type: str,
    operator_plan: dict[str, Any],
    query: str,
    recall_result: Mapping[str, Any],
) -> bool:
    del prompt_type, operator_plan
    return (
        not _explicit_non_code_family(recall_result)
        and _is_mixed_codebase_recall(recall_result)
        and _classify_code_query_mode(query, recall_result) != "non_code"
    )


def _match_risk_review(
    prompt_type: str,
    operator_plan: dict[str, Any],
    query: str,
    recall_result: Mapping[str, Any],
) -> bool:
    del prompt_type, operator_plan
    return (
        not _explicit_non_code_family(recall_result)
        and _is_codebase_only_recall(recall_result)
        and _has_codebase_evidence(recall_result)
        and _is_risk_review_query(query, recall_result)
    )


def _match_code_chain(
    prompt_type: str,
    operator_plan: dict[str, Any],
    query: str,
    recall_result: Mapping[str, Any],
) -> bool:
    del prompt_type, operator_plan
    return (
        not _explicit_non_code_family(recall_result)
        and _is_codebase_only_recall(recall_result)
        and _has_codebase_evidence(recall_result)
        and _is_code_chain_query(query, recall_result)
    )


def _match_code_slot(
    prompt_type: str,
    operator_plan: dict[str, Any],
    query: str,
    recall_result: Mapping[str, Any],
) -> bool:
    del prompt_type, operator_plan
    return (
        not _explicit_non_code_family(recall_result)
        and _is_codebase_only_recall(recall_result)
        and _has_codebase_evidence(recall_result)
        and _is_code_slot_query(query, recall_result)
    )


def _match_codebase(
    prompt_type: str,
    operator_plan: dict[str, Any],
    query: str,
    recall_result: Mapping[str, Any],
) -> bool:
    del prompt_type, operator_plan, query
    return (
        not _explicit_non_code_family(recall_result)
        and _is_codebase_only_recall(recall_result)
        and _has_codebase_evidence(recall_result)
    )


def _match_container_exact_copy(
    prompt_type: str,
    operator_plan: dict[str, Any],
    query: str,
    recall_result: Mapping[str, Any],
) -> bool:
    del prompt_type, operator_plan, query
    return bool(recall_result.get("terminal_render_candidate"))


@dataclass(frozen=True)
class InferenceLeafPlugin:
    name: str
    prompt_name: str
    base_prompt_types: tuple[str, ...]
    requires_enabled: tuple[str, ...] = ()
    blocked_by: tuple[str, ...] = ()
    priority: int = 0
    matcher: LeafMatcher | None = None

    def matches(
        self,
        prompt_type: str,
        operator_plan: dict[str, Any],
        *,
        query: str = "",
        recall_result: Mapping[str, Any] | None = None,
    ) -> bool:
        if prompt_type not in self.base_prompt_types:
            return False
        for op_name in self.requires_enabled:
            if not operator_plan.get(op_name, {}).get("enabled", False):
                return False
        for op_name in self.blocked_by:
            if operator_plan.get(op_name, {}).get("enabled", False):
                return False
        if self.matcher is not None:
            return self.matcher(prompt_type, operator_plan, query, recall_result or {})
        return True


INFERENCE_LEAF_PLUGINS: tuple[InferenceLeafPlugin, ...] = (
    InferenceLeafPlugin(
        name="container_exact_copy",
        prompt_name="container_exact_copy",
        base_prompt_types=("lookup", "hybrid", "synthesis", "synthesize", "exact_copy"),
        priority=130,
        matcher=_match_container_exact_copy,
    ),
    InferenceLeafPlugin(
        name="codebase_mixed",
        prompt_name="codebase_mixed",
        base_prompt_types=("lookup", "hybrid", "synthesis", "synthesize"),
        priority=120,
        matcher=_match_codebase_mixed,
    ),
    InferenceLeafPlugin(
        name="risk_review",
        prompt_name="risk_review",
        base_prompt_types=("lookup", "hybrid", "synthesis", "synthesize", "procedural", "prospective"),
        priority=84,
        matcher=_match_risk_review,
    ),
    InferenceLeafPlugin(
        name="code_chain",
        prompt_name="code_chain",
        base_prompt_types=("lookup", "hybrid", "synthesis", "synthesize"),
        priority=83,
        matcher=_match_code_chain,
    ),
    InferenceLeafPlugin(
        name="code_slot",
        prompt_name="code_slot",
        base_prompt_types=("lookup", "hybrid", "synthesis", "synthesize"),
        priority=82,
        matcher=_match_code_slot,
    ),
    InferenceLeafPlugin(
        name="codebase",
        prompt_name="codebase",
        base_prompt_types=("lookup", "hybrid", "synthesis", "synthesize"),
        priority=81,
        matcher=_match_codebase,
    ),
    InferenceLeafPlugin(
        name="slot_query",
        prompt_name="slot_query",
        base_prompt_types=("lookup", "hybrid", "synthesize", "synthesis"),
        requires_enabled=("slot_query",),
        blocked_by=("ordinal", "commonality", "compare_diff", "list_set", "bounded_chain", "local_anchor", "temporal_grounding"),
        priority=110,
    ),
    InferenceLeafPlugin(
        name="list_set",
        prompt_name="list_set",
        base_prompt_types=("lookup", "hybrid"),
        requires_enabled=("list_set",),
        blocked_by=("ordinal", "commonality", "compare_diff", "bounded_chain"),
        priority=100,
    ),
    InferenceLeafPlugin(
        name="compositional",
        prompt_name="compositional",
        base_prompt_types=("lookup", "hybrid", "synthesis", "synthesize"),
        requires_enabled=("compositional",),
        blocked_by=("list_set", "ordinal", "commonality", "compare_diff"),
        priority=90,
    ),
)

DEFAULT_INFERENCE_LEAF_PLUGIN_STATE = {
    plugin.name: True for plugin in INFERENCE_LEAF_PLUGINS
}


def resolve_inference_prompt_key(
    prompt_type: str,
    operator_plan: dict | None = None,
    *,
    plugin_state: dict[str, bool] | None = None,
    query: str = "",
    recall_result: Mapping[str, Any] | None = None,
) -> str:
    operator_plan = operator_plan or {}
    canonical_prompt_type = _RETRIEVAL_TO_INF.get(prompt_type, prompt_type)
    state = dict(DEFAULT_INFERENCE_LEAF_PLUGIN_STATE)
    if plugin_state:
        state.update({str(name): bool(enabled) for name, enabled in plugin_state.items()})
    for plugin in sorted(INFERENCE_LEAF_PLUGINS, key=lambda item: item.priority, reverse=True):
        if not state.get(plugin.name, True):
            continue
        if plugin.matches(
            canonical_prompt_type,
            operator_plan,
            query=query,
            recall_result=recall_result,
        ):
            return plugin.prompt_name
    return canonical_prompt_type


def retrieval_to_prompt_type(query_type: str) -> str:
    return _RETRIEVAL_TO_INF.get(query_type, query_type)
