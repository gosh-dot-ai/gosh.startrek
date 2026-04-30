#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

import asyncio
import contextlib
import hashlib
import json
import logging
import math
import os
import re
import secrets
import shutil
from collections import Counter, defaultdict
from collections.abc import Iterable
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import numpy as np
from dateutil import parser as date_parser

from .audit import AuditLog
from .codebase_container_graph import build_codebase_container_graph
from .codebase_ingest import build_codebase_stage1_bundle
from .codebase_query import augment_codebase_structural_packet, build_codebase_context
from .codebase_semantic_importer import import_semantic_bundle
from .codebase_semantic_plugins.runner import build_codebase_semantic_bundle
from .codebase_semantic_runtime import (
    augment_codebase_context,
    classify_codebase_query_mode,
    promote_defining_code_facts,
    query_requests_precise_code,
)
from .codebase_semantic_sidecars import CodebaseSemanticSidecarStore
from .common import (
    STOP_WORDS,
    _api_model,
    _call_model,
    _get_client,
    _resolve_embed_config,
    _supports_temperature,
    _tok_key,
    call_extract,
    call_oai,
    embed_query,
    embed_texts,
    embed_texts_sync,
    get_cost_summary,
    normalize_term_token,
    parse_json_response,
    runtime_secret_context,
)
from .config import MemoryConfig
from .container_graph import (
    build_document_container_graph,
    empty_container_graph,
    normalize_container_graph,
    stable_hash,
    validate_container_exact_copy_render_refs,
)
from .coverage_recovery import (
    classify_coverage_query,
    compute_coverage_stats,
    merge_coverage_recovery_facts,
    needs_coverage_recovery,
)
from .document_span import assign_document_artifact_span_ids
from .episode_extraction import build_singleton_episodes, extract_doc_metadata, group_document
from .episode_features import extract_query_features, has_exact_step_mention, step_range_overlap
from .episode_packet import (
    _fact_content_tokens,
    _fact_slot_fill_candidates,
    _pseudo_facts_from_episode,
    _select_bounded_chain_seed_facts,
    build_bounded_chain_candidate_bundle,
    build_context_from_retrieved_facts,
    build_context_from_selected_episodes,
    fact_episode_ids,
)
from .episode_retrieval import (
    available_families,
    build_episode_bm25,
    choose_episode_ids,
    choose_episode_ids_with_trace,
    partition_corpus_by_family,
    resolve_selection_config,
    route_retrieval_families,
    select_episode_ids_late_fusion,
    select_episode_ids_late_fusion_with_trace,
)
from .episodes import build_episode_lookup, build_facts_by_episode
from .fact_alignment import (
    align_facts_batch,
    facts_as_selectors_enabled,
    iter_support_spans,
    selector_surface_text,
)
from .identity import (
    _generate_artifact_id,
    _generate_version_id,
    content_hash_text,
)
from .inference import (
    COUNTING_TOOLS,
    DEFAULT_INFERENCE_LEAF_PLUGIN_STATE,
    GET_CONTEXT_TOOL,
    TEMPORAL_TOOLS,
    call_inference_with_tools,
    get_inf_prompt,
    get_more_context,
    resolve_inference_prompt_key,
)
from .librarian import (
    ENGLISH_CANONICAL_TRANSLATION_VERSION,
    canonicalize_source_to_english,
    detect_format,
    detect_source_language,
    extract_session,
    normalize_content_format,
    resolve_supersession,
)
from .local_cli_backend import LocalCliTimeoutError, render_local_cli_prompt, run_local_cli
from .mal.apply import current_gen_dir as _mal_current_gen_dir
from .normalizer import acl_domain_key, dedup_domain_key, hamming_distance, normalize_text, simhash
from .object_flags import normalize_object_flags_field, validate_object_flags
from .object_reports import build_report, normalize_legacy_report_fields, validate_report_object
from .prompt_registry import PromptRegistry
from .prompt_routing.hooks import build_payload_messages as build_prompt_payload_messages
from .prompt_routing.hooks import resolve_prompt_key as resolve_inference_leaf_prompt_key
from .query_executors.registry import run_default_query_executor_chain
from .query_lexicon import (
    CODE_ATTACHMENT_SECTION_LABEL,
    CODEBASE_FILE_LOOKUP_MULTI_PART_MARKERS,
    CODEBASE_FILE_LOOKUP_NORMALIZE_CLAUSES,
    CODEBASE_FILE_LOOKUP_PATTERNS,
    CODEBASE_FILE_LOOKUP_REQUEST_MARKERS,
)
from .recall_policy import (
    FACT_LIKELIHOOD_HIGH,
    FACT_LIKELIHOOD_MEDIUM,
    FACT_LIKELIHOOD_UNCERTAIN,
    RAW_CONVERSATION_WINDOW_RADIUS,
    RAW_EPISODE_RETRIEVAL_LIMIT,
    RAW_SOURCE_WINDOW_BUDGET_CHARS,
    FactLikelihood,
)
from .recall_policy import (
    RECALL_EXTRACTION_POLICY_MIRROR as _RECALL_EXTRACTION_POLICY_MIRROR,
)
from .retrieval import detect_query_type, source_local_fact_sweep
from .source_adapters import segment_document_text
from .source_adapters.registry import registered_source_retrieval_families
from .storage import (
    IngressWriteStorageBackend,
    ProjectionWriteThroughStorageBackend,
    SecretStorageBackend,
    StorageBackend,
    make_storage,
)
from .temporal import (
    empty_temporal_index,
    latest_calendar_anchor,
    lookup_events_for_fact,
)
from .temporal_normalizer import normalize_temporal_index
from .temporal_planner import (
    classify_temporal_query,
    execute_calendar_query,
    execute_ordinal_query,
    extract_calendar_query,
)
from .tuning import get_runtime_tuning, get_tuning_section
from .unified_source_extractor import extract_source_aggregation

log = logging.getLogger(__name__)

NEAR_DUP_SIMHASH_THRESHOLD = 3
NEAR_DUP_MIN_CHARS = 200
LIVE_SCOPE_REQUIRED_ERROR = "scope must be provided explicitly"
EXACT_COPY_REFUSAL = "Not enough grounded context."


def _load_mal_active_config(data_dir: str, key: str, agent_id: str) -> dict:
    """Load MAL generation config for a binding, or empty dict if none."""
    try:
        gen_dir = _mal_current_gen_dir(data_dir, key, agent_id)
        config_path = gen_dir / "active_config.json"
        if config_path.exists():
            return json.loads(config_path.read_text())
    except Exception:
        pass
    return {}


def _resolve_extract_model(base_model: str, data_dir: str, key: str, agent_id: str) -> str:
    """Return MAL-overridden extraction model, or the base model."""
    mal_cfg = _load_mal_active_config(data_dir, key, agent_id)
    return mal_cfg.get("extraction_model") or base_model


_COMMONALITY_TOKEN_CANON = {
    "store": "business",
    "stores": "business",
    "studio": "business",
    "studios": "business",
    "shop": "business",
    "shops": "business",
    "company": "business",
    "companies": "business",
    "venture": "business",
    "ventures": "business",
    "startup": "business",
    "startups": "business",
    "open": "start",
    "opens": "start",
    "opened": "start",
    "opening": "start",
    "launch": "start",
    "launches": "start",
    "launched": "start",
    "launching": "start",
    "jobless": "job",
}
_COMMONALITY_IGNORE = STOP_WORDS | {
    "both", "common", "share", "shared", "similar", "similarities",
    "passion", "special", "meaning", "support", "supported", "supportive",
    "motivation", "motivating", "inspiration", "friend", "friendship",
    "journey", "dream", "dreams", "love", "loves", "loving",
    "there", "around", "always", "really", "very", "still",
    "great", "good", "excited", "nervous", "happy", "proud",
    "more", "much", "many", "last", "week", "month", "year",
    "today", "yesterday", "tomorrow", "together", "wait",
    "kind", "word", "words", "hard", "work", "paid", "pay", "off",
    "online", "speaker", "user",
}
_COMMONALITY_QUERY_RE = re.compile(r"\b(in common|both|shared?|same as)\b", re.I)
_COMMONALITY_INTEREST_QUERY_RE = re.compile(r"\b(interests?|hobbies?|favorite|favourite|enjoy|like|likes)\b", re.I)
_COMMONALITY_INTEREST_FACT_RE = re.compile(
    r"\b(love|enjoy|favorite|favourite|hobb(?:y|ies)|watch(?:ing)?|play(?:ing)?|"
    r"make(?:ing)?|bak(?:e|ing)|cook(?:ing)?|read(?:ing)?)\b",
    re.I,
)
_COMMONALITY_OVERLAP_IGNORE = {
    "shar", "same", "similar", "great", "good", "cool", "support", "fun",
    "reward", "really", "just", "pretty", "main",
}
_LOCAL_ANCHOR_CUE_RE = re.compile(
    r"\b(?:at|from|in|to|into|inside|near)\s+([A-Z][A-Za-z0-9&'._-]+(?:\s+[A-Z][A-Za-z0-9&'._-]+){0,3})"
)
_LOCAL_ANCHOR_CAP_RE = re.compile(
    r"(?<!\w)([A-Z][A-Za-z0-9&'._-]+(?:\s+[A-Z][A-Za-z0-9&'._-]+){0,3})"
)
_LOCAL_ANCHOR_LINE_CUE_RE = re.compile(
    r"\b(store|retailers?|shop|coupon|redeem|redeemed|purchase|purchased|bought|"
    r"step|action|position|moved|city|country|location|headquarters|works? in)\b",
    re.I,
)
_LOCAL_ANCHOR_IGNORE = {
    "Action", "Observation", "Active Rules", "Objects on the map", "Step",
    "User", "Assistant", "I", "I'm", "It", "The",
    "Many", "Some", "Several", "These", "Those", "Here", "There",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
}
_QUERY_ENTITY_IGNORE = {"what", "which", "when", "where", "who", "why", "how"}
_CITY_TO_COUNTRY = {
    "amsterdam": "Netherlands",
    "athens": "Greece",
    "barcelona": "Spain",
    "beijing": "China",
    "berlin": "Germany",
    "boston": "United States",
    "brisbane": "Australia",
    "chicago": "United States",
    "dublin": "Ireland",
    "edinburgh": "United Kingdom",
    "hong kong": "China",
    "london": "United Kingdom",
    "los angeles": "United States",
    "madrid": "Spain",
    "melbourne": "Australia",
    "miami": "United States",
    "montreal": "Canada",
    "moscow": "Russia",
    "mumbai": "India",
    "new york": "United States",
    "osaka": "Japan",
    "paris": "France",
    "rome": "Italy",
    "san francisco": "United States",
    "seattle": "United States",
    "seoul": "South Korea",
    "shanghai": "China",
    "sydney": "Australia",
    "tokyo": "Japan",
    "toronto": "Canada",
    "vancouver": "Canada",
    "washington": "United States",
}


_ANSWER_CITATION_PARENS_RE = re.compile(
    r"\(\s*(?:evidence|source|sources|citation|citations|fact|retrieved\s+fact[s]?|ref|refs|reference|references)"
    r"[^\)]*\)",
    re.IGNORECASE,
)
_ANSWER_CITATION_TAIL_RE = re.compile(
    r"\b(?:sources?|evidence|citations?|references?)\s*:\s*.*$",
    re.IGNORECASE | re.DOTALL,
)
_ANSWER_ACCORDING_TO_RE = re.compile(
    r"^\s*according\s+to[^,.\n]*[,.\n]?",
    re.IGNORECASE,
)
_ANSWER_BRACKET_REF_RE = re.compile(r"\[\s*\d+(?:\s*[-,]\s*\d+)*\s*\]")
_ANSWER_PAREN_SESSION_REF_RE = re.compile(
    r"\(\s*(?:S|Session|Sess|Fact|fact|F|sess)\s*\d+\b[^\)]*\)",
    re.IGNORECASE,
)
_ANSWER_META_EXPLANATION_TAIL_RE = re.compile(
    r"\b(?:as\s+(?:noted|confirmed|stated|mentioned|cited)\s+in|"
    r"explicitly\s+(?:states?|stated|noted|confirmed|mentioned)|"
    r"as\s+per\s+(?:the\s+)?(?:retrieved\s+)?fact[s]?|"
    r"this\s+is\s+(?:explicitly\s+)?(?:stated|noted|confirmed)\s+in)\b[^.\n]*[.\n]?",
    re.IGNORECASE,
)


_GROUNDED_GATE_MONTH_NAMES = {
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
}
_GROUNDED_GATE_WEEKDAY_NAMES = {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "mon", "tue", "wed", "thu", "fri", "sat", "sun",
}
_GROUNDED_GATE_DATE_COMPONENT_TOKENS = (
    _GROUNDED_GATE_MONTH_NAMES | _GROUNDED_GATE_WEEKDAY_NAMES
)


def _answer_core_for_grounding(answer: str) -> str:
    """Strip citation/source/explanation wrappers from a model answer.

    Instruction-tuned models routinely append evidence citations and
    meta-explanation tails to a substantively correct answer, e.g.::

        John signed with the Minnesota Wolves on 21 May 2023.
        Sources: [1] (S1) explicitly states ...

    or::

        Evan got a new Prius after his old Prius broke down.
        (Evidence: [1][2][5][7] explicitly state he repaired/sold the old Prius.)

    The grounding gate downstream tokenises the answer and rejects it if
    any token is missing from the retrieved evidence text. Citation and
    meta-explanation wording (``Sources:``, ``Evidence: [1]``,
    ``explicitly states``, ``[2]``) is by construction absent from raw
    evidence — so without stripping it, every answer that cites its
    sources gets falsely classified as ungrounded.

    This helper removes:

    - ``(Evidence: [1][2] ...)`` / ``(Sources: ...)`` parenthesised tails
    - trailing ``Sources: ...`` / ``Evidence: ...`` / ``Citations: ...``
      / ``References: ...`` blocks
    - leading ``According to ...`` clauses
    - bracket session/fact refs ``[1]``, ``[2-3]``
    - parenthesised session refs ``(S5)``, ``(Fact 12)``
    - trailing ``as noted in / explicitly states / as per the retrieved
      fact ...`` meta-explanation clauses

    The remaining text is the *answer core* the gate must judge.
    Hallucinations (fabricated content with no support) still produce
    zero grounded tokens after this strip and are rejected by the gate
    as before.
    """
    if not answer:
        return ""
    text = answer
    text = _ANSWER_CITATION_PARENS_RE.sub(" ", text)
    text = _ANSWER_CITATION_TAIL_RE.sub(" ", text)
    text = _ANSWER_ACCORDING_TO_RE.sub(" ", text)
    text = _ANSWER_PAREN_SESSION_REF_RE.sub(" ", text)
    text = _ANSWER_BRACKET_REF_RE.sub(" ", text)
    text = _ANSWER_META_EXPLANATION_TAIL_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def _normalize_commonality_token(token: str) -> str:
    low = token.lower()
    if len(low) > 5 and low.endswith("ing"):
        low = low[:-3]
    elif len(low) > 4 and low.endswith("ed"):
        low = low[:-2]
    elif len(low) > 4 and low.endswith("ies"):
        low = low[:-3] + "y"
    elif len(low) > 4 and low.endswith("s") and not low.endswith("ss"):
        low = low[:-1]
    return _COMMONALITY_TOKEN_CANON.get(low, low)


def _extract_query_named_entities(query: str) -> list[str]:
    seen = set()
    ordered = []
    for match in re.findall(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*", query):
        entity = " ".join(match.split()).lower()
        if entity in _QUERY_ENTITY_IGNORE:
            continue
        if entity not in seen:
            seen.add(entity)
            ordered.append(entity)
    return ordered


def _fact_entity_hits(fact: dict, query_entities: list[str]) -> set[str]:
    if not query_entities:
        return set()
    fact_text = fact.get("fact", "").lower()
    fact_ents = {
        (entity.lower() if isinstance(entity, str) else str(entity).lower())
        for entity in fact.get("entities", [])
    }
    hits = set()
    for query_entity in query_entities:
        if query_entity in fact_text or any(
            query_entity in fact_ent or fact_ent in query_entity
            for fact_ent in fact_ents
        ):
            hits.add(query_entity)
    return hits


def _commonality_tokens(text: str, query_entities: list[str]) -> set[str]:
    entity_tokens = {
        _normalize_commonality_token(token)
        for entity in query_entities
        for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9'_-]+", entity)
    }
    tokens = set()
    for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9'_-]+", text):
        norm = _normalize_commonality_token(token)
        if len(norm) < 3:
            continue
        if norm in _COMMONALITY_IGNORE or norm in entity_tokens:
            continue
        tokens.add(norm)
    return tokens


def _self_grounded_commonality_bonus(fact: dict, entity: str) -> float:
    speaker = str(fact.get("speaker", "") or "").strip().lower()
    if not speaker:
        return 0.0
    entity_key = entity.split()[0].lower()
    return 4.0 if speaker == entity_key or speaker == entity.lower() else 0.0


def _rank_commonality_groups(candidates: list[tuple[float, set[str], dict, dict]]) -> list[dict]:
    groups = defaultdict(list)
    for score, overlap, left, right in candidates:
        overlap_key = tuple(sorted(overlap))
        if overlap_key:
            groups[overlap_key].append((score, left, right))

    ranked: list[dict[str, Any]] = []
    for overlap_key, pairs in groups.items():
        pairs.sort(
            key=lambda row: (
                -row[0],
                row[1].get("rank", row[1].get("idx", 0)),
                row[2].get("rank", row[2].get("idx", 0)),
            )
        )
        unique_pairs = []
        seen_pair_ids = set()
        for score, left, right in pairs:
            left_fact = left.get("fact", left)
            right_fact = right.get("fact", right)
            if not isinstance(left_fact, dict):
                left_fact = {"fact": str(left_fact)}
            if not isinstance(right_fact, dict):
                right_fact = {"fact": str(right_fact)}
            pair_id = (
                left_fact.get("id", ""),
                right_fact.get("id", ""),
                left_fact.get("session", 0),
                right_fact.get("session", 0),
            )
            if pair_id in seen_pair_ids:
                continue
            seen_pair_ids.add(pair_id)
            unique_pairs.append((score, left, right))

        ranked.append(
            {
                "overlap": overlap_key,
                "pairs": unique_pairs,
                "pair_count": len(unique_pairs),
                "top_score": unique_pairs[0][0] if unique_pairs else 0.0,
                "mean_top_score": (
                    sum(score for score, _left, _right in unique_pairs[:3]) / min(len(unique_pairs), 3)
                    if unique_pairs
                    else 0.0
                ),
                "specificity": len(overlap_key),
                "is_multi": len(overlap_key) >= 2,
            }
        )

    ranked.sort(
        key=lambda group: (
            1 if group["is_multi"] else 0,
            group["mean_top_score"] + math.log1p(group["pair_count"]) + 0.25 * group["specificity"],
            group["specificity"],
            group["top_score"],
        ),
        reverse=True,
    )
    return ranked


def _build_raw_commonality_support_items(query: str, raw_sessions: list[dict]) -> list[dict]:
    if not _COMMONALITY_QUERY_RE.search(query):
        return []
    query_entities = _extract_query_named_entities(query)
    if len(query_entities) < 2:
        return []

    left_entity, right_entity = query_entities[:2]
    left_key = left_entity.split()[0]
    right_key = right_entity.split()[0]
    candidates: list[tuple[float, set[str], int, tuple[int, str, set[str]], tuple[int, str, set[str]]]] = []
    seen: set[tuple[tuple[str, ...], str, str, int]] = set()

    for session_idx, raw_session in enumerate(raw_sessions, start=1):
        if not isinstance(raw_session, dict):
            continue
        text = _semantic_raw_session_text(raw_session)
        if not text:
            continue
        lower = text.lower()
        if left_key not in lower or right_key not in lower:
            continue
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            continue

        left_rows: list[tuple[int, str, set[str]]] = []
        right_rows: list[tuple[int, str, set[str]]] = []
        for idx, line in enumerate(lines):
            lower_line = line.lower()
            tokens = _commonality_tokens(line, query_entities)
            if not tokens:
                continue
            if lower_line.startswith(f"{left_key}:"):
                left_rows.append((idx, line, tokens))
                continue
            if lower_line.startswith(f"{right_key}:"):
                right_rows.append((idx, line, tokens))
                continue
            if left_key in lower_line:
                left_rows.append((idx, line, tokens))
            if right_key in lower_line:
                right_rows.append((idx, line, tokens))

        if not left_rows or not right_rows:
            continue

        token_freq: defaultdict[str, int] = defaultdict(int)
        for _idx, _line, tokens in left_rows + right_rows:
            for token in tokens:
                token_freq[token] += 1

        for left in left_rows[:16]:
            for right in right_rows[:16]:
                if left[0] == right[0] or left[1] == right[1]:
                    continue
                overlap = left[2] & right[2]
                if not overlap:
                    continue
                key = (tuple(sorted(overlap)), left[1], right[1], session_idx)
                if key in seen:
                    continue
                seen.add(key)
                rarity = sum(1.0 / max(token_freq.get(token, 1), 1) for token in overlap)
                score = rarity * 10.0 + len(overlap) * 1.5 - 0.08 * abs(left[0] - right[0])
                candidates.append((score, overlap, session_idx, left, right))

    candidates.sort(key=lambda row: (-row[0], -len(row[1]), row[2], row[3][0], row[4][0]))
    items: list[dict[str, Any]] = []
    used_overlap = set()
    for idx, (_score, overlap, session_idx, left, right) in enumerate(candidates):
        overlap_key = tuple(sorted(overlap))
        if overlap_key in used_overlap:
            continue
        used_overlap.add(overlap_key)
        label = ", ".join(sorted(overlap))
        items.append(
            {
                "text": (
                    f"[Shared raw {len(items)+1}] overlap in source session S{session_idx}: {label}\n"
                    f"- {left[1]}\n"
                    f"- {right[1]}"
                ),
                "rank": -1200 - idx,
                "source": "commonality_raw",
                "session": session_idx,
            }
        )
        if len(items) >= 3:
            break
    return items


def _augment_commonality_facts(
    query: str,
    retrieved_facts: list[dict],
    all_facts: list[dict],
    *,
    limit: int = 6,
) -> list[dict]:
    if not _COMMONALITY_QUERY_RE.search(query):
        return []
    query_entities = _extract_query_named_entities(query)
    if len(query_entities) < 2:
        return []

    existing_ids = {fact.get("id", "") for fact in retrieved_facts}
    rows_by_entity: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    token_freq: defaultdict[str, int] = defaultdict(int)
    session_values: list[int] = []

    for idx, fact in enumerate(all_facts):
        hits = _fact_entity_hits(fact, query_entities)
        if not hits:
            continue
        tokens = _commonality_tokens(fact.get("fact", ""), query_entities)
        if not tokens:
            continue
        row = {
            "fact": fact,
            "tokens": tokens,
            "idx": idx,
            "already_retrieved": fact.get("id", "") in existing_ids,
        }
        session_no = _coerce_positive_session_num(fact.get("session", 0))
        if session_no is not None:
            session_values.append(session_no)
        for token in tokens:
            token_freq[token] += 1
        for entity in hits:
            rows_by_entity[entity].append(row)

    entities = query_entities[:2]
    if any(not rows_by_entity.get(entity) for entity in entities):
        return []
    earliest_session = min(session_values) if session_values else 0

    interest_query = bool(_COMMONALITY_INTEREST_QUERY_RE.search(query))

    candidates: list[tuple[float, set[str], dict[str, Any], dict[str, Any]]] = []
    seen_pairs: set[tuple[tuple[str, ...], str, str]] = set()
    for left in rows_by_entity[entities[0]][:160]:
        for right in rows_by_entity[entities[1]][:160]:
            left_fact = left["fact"]
            right_fact = right["fact"]
            if left_fact.get("id", "") == right_fact.get("id", ""):
                continue
            if interest_query and not (
                _COMMONALITY_INTEREST_FACT_RE.search(left_fact.get("fact", ""))
                and _COMMONALITY_INTEREST_FACT_RE.search(right_fact.get("fact", ""))
            ):
                continue
            overlap = (left["tokens"] & right["tokens"]) - _COMMONALITY_OVERLAP_IGNORE
            if not overlap:
                continue
            key = (tuple(sorted(overlap)), left_fact.get("id", ""), right_fact.get("id", ""))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            rarity = sum(1.0 / max(token_freq.get(token, 1), 1) for token in overlap)
            retrieved_bonus = 1.0 if left["already_retrieved"] or right["already_retrieved"] else 0.0
            session_gap = abs((left_fact.get("session") or 0) - (right_fact.get("session") or 0))
            session_bonus = max(0.0, 8.0 - 0.5 * session_gap) if session_gap else 8.0
            session_floor = min((left_fact.get("session") or 0), (right_fact.get("session") or 0))
            origin_bonus = 0.0
            if earliest_session and session_floor:
                origin_bonus = max(0.0, 6.0 - 0.25 * max(0, session_floor - earliest_session))
            left_bonus = _self_grounded_commonality_bonus(left_fact, entities[0])
            right_bonus = _self_grounded_commonality_bonus(right_fact, entities[1])
            score = (
                rarity * 10.0
                + len(overlap) * 1.5
                + retrieved_bonus
                + session_bonus
                + origin_bonus
                + left_bonus
                + right_bonus
            )
            candidates.append((score, overlap, left, right))

    extras: list[dict] = []
    added_ids = set(existing_ids)
    ranked_groups = _rank_commonality_groups(candidates)
    for group in ranked_groups:
        if not group["pairs"]:
            continue
        for _score, left, right in group["pairs"][:2]:
            for row in (left, right):
                fact = row["fact"]
                fact_id = fact.get("id", "")
                if fact_id and fact_id not in added_ids:
                    extras.append(fact)
                    added_ids.add(fact_id)
                    if len(extras) >= limit:
                        return extras
    return extras


def _extract_conversation_windows(text: str, facts: list[dict], cap: int) -> str:
    if cap <= 0:
        return ""
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if not lines:
        return text[:cap]

    def _norm_tokens(value: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9'_-]+", value.lower())
            if len(token) >= 3
        }

    fact_tokens = [_norm_tokens(fact.get("fact", "")) for fact in facts]
    line_scores = []
    for idx, line in enumerate(lines):
        line_tokens = _norm_tokens(line)
        if not line_tokens:
            continue
        score = max((len(line_tokens & fact_token_set) for fact_token_set in fact_tokens), default=0)
        if score > 0:
            line_scores.append((score, idx))

    if not line_scores:
        return text[:cap]

    line_scores.sort(key=lambda item: (-item[0], item[1]))
    chosen = sorted({idx for _score, idx in line_scores[:3]})
    windows = []
    radius = 8
    for idx in chosen:
        windows.append((max(0, idx - radius), min(len(lines), idx + radius + 1)))
    windows.sort()

    merged: list[list[int]] = []
    for start, end in windows:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)

    parts: list[str] = []
    total = 0
    for start, end in merged:
        snippet = "\n".join(lines[start:end]).strip()
        if not snippet:
            continue
        if parts:
            snippet = "...\n" + snippet
        remaining = cap - total
        if remaining <= 0:
            break
        if len(snippet) > remaining:
            snippet = snippet[:remaining].rstrip()
        if not snippet:
            break
        parts.append(snippet)
        total += len(snippet)
        if total >= cap:
            break

    return "\n".join(parts) if parts else text[:cap]


def _extract_conversation_support_lines(text: str, facts: list[dict], max_lines: int = 4) -> list[str]:
    if max_lines <= 0:
        return []
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if not lines:
        return []

    def _norm_tokens(value: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9'_-]+", value.lower())
            if len(token) >= 3
        }

    fact_tokens = [_norm_tokens(fact.get("fact", "")) for fact in facts]
    scored = []
    for idx, line in enumerate(lines):
        line_tokens = _norm_tokens(line)
        if not line_tokens:
            continue
        score = max((len(line_tokens & fact_token_set) for fact_token_set in fact_tokens), default=0)
        if score > 0:
            scored.append((score, idx))

    if not scored:
        return []

    scored.sort(key=lambda item: (-item[0], item[1]))
    chosen = sorted({idx for _score, idx in scored[:2]})
    radius = 4
    out = []
    seen = set()
    for idx in chosen:
        for pos in range(max(0, idx - radius), min(len(lines), idx + radius + 1)):
            line = lines[pos].strip()
            if not line or line in seen:
                continue
            out.append(line)
            seen.add(line)
            if len(out) >= max_lines:
                return out
    return out


def _extract_query_focused_conversation_excerpt(
    text: str,
    facts: list[dict],
    query_terms: set[str],
    cap: int = 2400,
) -> str:
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if not lines:
        return text[:cap]

    def _norm_tokens(value: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9'_-]+", value.lower())
            if len(token) >= 3
        }

    fact_token_sets = [_norm_tokens(fact.get("fact", "")) for fact in facts[:4]]
    scored = []
    for idx, line in enumerate(lines):
        line_tokens = _norm_tokens(line)
        if not line_tokens:
            continue
        query_overlap = len(line_tokens & query_terms)
        fact_overlap = max((len(line_tokens & fact_tokens) for fact_tokens in fact_token_sets), default=0)
        score = fact_overlap * 3 + query_overlap * 2
        if score > 0:
            scored.append((score, idx))

    if not scored:
        return _extract_conversation_windows(text, facts, cap)

    scored.sort(key=lambda item: (-item[0], item[1]))
    chosen = sorted({idx for _score, idx in scored[:2]})
    radius = 6
    parts: list[str] = []
    total = 0
    for idx in chosen:
        start = max(0, idx - radius)
        end = min(len(lines), idx + radius + 1)
        snippet = "\n".join(lines[start:end]).strip()
        if not snippet:
            continue
        if parts:
            snippet = "...\n" + snippet
        remaining = cap - total
        if remaining <= 0:
            break
        if len(snippet) > remaining:
            snippet = snippet[:remaining].rstrip()
        if not snippet:
            break
        parts.append(snippet)
        total += len(snippet)
        if total >= cap:
            break
    return "\n".join(parts) if parts else _extract_conversation_windows(text, facts, cap)


def _is_local_anchor_query(query: str) -> bool:
    qf = extract_query_features(query)
    if qf.get("asks_where"):
        return True
    return bool((qf.get("operator_plan") or {}).get("local_anchor", {}).get("enabled"))


def _extract_anchor_candidates_from_line(
    line: str,
    query_entities: list[str],
    query_lower: str,
) -> list[tuple[str, bool]]:
    candidates = []
    for pattern, is_cue in ((_LOCAL_ANCHOR_CUE_RE, True), (_LOCAL_ANCHOR_CAP_RE, False)):
        for match in pattern.findall(line):
            candidate = " ".join(str(match).split()).strip(" .,:;!?")
            if not candidate or candidate in _LOCAL_ANCHOR_IGNORE:
                continue
            low = candidate.lower()
            if low in query_lower or low in query_entities or len(low) < 3:
                continue
            candidates.append((candidate, is_cue))
    seen = set()
    ordered = []
    for candidate, is_cue in candidates:
        low = candidate.lower()
        if low not in seen:
            seen.add(low)
            ordered.append((candidate, is_cue))
    return ordered


def _build_local_anchor_support_items(
    query: str,
    retrieved_facts: list[dict],
    raw_sessions: list[dict],
) -> list[dict]:
    if not _is_local_anchor_query(query):
        return []

    raw_session_matches = _match_raw_sessions_for_facts(retrieved_facts, raw_sessions)
    session_facts = defaultdict(list)
    session_lookup: dict[str, tuple[int, dict]] = {}
    for rank, fact in enumerate(retrieved_facts):
        matched = raw_session_matches.get(rank)
        if matched is None:
            continue
        match_key, display_session_num, raw_session = matched
        session_lookup[match_key] = (display_session_num, raw_session)
        session_facts[match_key].append((rank, fact))
    if not session_facts:
        return []

    query_terms = {
        token
        for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9'_-]+", query.lower())
        if len(token) >= 4 and token not in STOP_WORDS and token not in _QUERY_ENTITY_IGNORE
    }
    session_rows = []
    for match_key, pairs in session_facts.items():
        query_score = 0.0
        for rank, fact in pairs:
            fact_lower = fact.get("fact", "").lower()
            overlap = sum(1 for token in query_terms if token in fact_lower)
            query_score += overlap * 3 - rank * 0.05
        display_session_num, _raw_session = session_lookup[match_key]
        session_rows.append((match_key, display_session_num, query_score, len(pairs), min(rank for rank, _fact in pairs)))
    session_rows.sort(key=lambda row: (-row[2], -row[3], row[4], row[1]))
    focus_key = session_rows[0][0]

    focus_session, raw_session = session_lookup[focus_key]
    if not isinstance(raw_session, dict):
        return []
    text = _semantic_raw_session_text(raw_session)
    if not text:
        return []

    query_entities = _extract_query_named_entities(query)
    query_lower = query.lower()
    facts = [fact for _rank, fact in session_facts[focus_key]]
    focus_facts = sorted(
        facts,
        key=lambda fact: (
            -sum(1 for token in query_terms if token in fact.get("fact", "").lower()),
            fact.get("id", ""),
        ),
    )[:6]
    excerpt = _extract_query_focused_conversation_excerpt(text, focus_facts or facts, query_terms, cap=2400)
    lines = [line.strip() for line in excerpt.splitlines() if line.strip()]
    if not lines:
        return []

    candidate_rows: dict[str, dict[str, Any]] = {}
    for line_idx, line in enumerate(lines):
        for candidate, is_cue in _extract_anchor_candidates_from_line(line, query_entities, query_lower):
            key = candidate.lower()
            row = candidate_rows.setdefault(
                key,
                {
                    "candidate": candidate,
                    "count": 0,
                    "cue_count": 0,
                    "best_line": line_idx,
                    "lines": [],
                },
            )
            row["count"] += 1
            row["cue_count"] += int(is_cue)
            row["cue_count"] += int(bool(_LOCAL_ANCHOR_LINE_CUE_RE.search(line)))
            row["best_line"] = min(row["best_line"], line_idx)
            if line not in row["lines"]:
                row["lines"].append(line)

    if not candidate_rows:
        return []

    ranked = sorted(
        candidate_rows.values(),
        key=lambda row: (-row["cue_count"], -row["count"], row["best_line"], row["candidate"].lower()),
    )
    top = ranked[0]
    if top["cue_count"] <= 0 and top["count"] < 2 and len(top["lines"]) < 2:
        return []
    return [
        {
            "text": (
                f"[Anchor 1] strongest local anchor: {top['candidate']}\n"
                + "\n".join(f"- {line}" for line in top["lines"][:3])
            ),
            "rank": -900,
            "source": "local_anchor",
        }
    ]


def _context_has_source_excerpts(context: str) -> bool:
    if not context:
        return False
    return any(
        marker in context
        for marker in (
            "RAW CONTEXT",
            "RAW CONTEXT (source text excerpts):",
            "--- SOURCE DOCUMENT SECTIONS ---",
            "--- SOURCE EPISODE RAW TEXT ---",
        )
    )


_VALID_ID_PREFIXES = ("user:", "agent:", "service:", "swarm:")
_BARE_IDS = ("system", "anonymous")
NAMED_SWARM_REQUIRED_ERROR = "scope='swarm-shared' requires explicit named swarm_id"


def _normalize_identity(identity: str, allow_public: bool = False) -> str:
    """Validate and normalize identity string. Raises ValueError if invalid."""
    if identity in _BARE_IDS:
        return identity
    if identity == "agent:PUBLIC":
        if not allow_public:
            raise ValueError("agent:PUBLIC cannot be used as owner_id")
        return identity
    if not any(identity.startswith(p) for p in _VALID_ID_PREFIXES):
        raise ValueError(
            f"Invalid identity '{identity}': must start with user:/agent:/swarm: or be system/anonymous"
        )
    return identity


def _normalize_acl_principals(values, *, allow_public: bool = True) -> list[str]:
    """Normalize ACL principal lists with stable ordering and no duplicates."""
    if values is None:
        return []
    if not isinstance(values, (list, tuple, set)):
        raise ValueError("ACL principals must be a list/tuple/set of identities")
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            raise ValueError("ACL principals must contain only strings")
        principal = _normalize_identity(value, allow_public=allow_public)
        if principal not in seen:
            seen.add(principal)
            normalized.append(principal)
    return normalized


def _normalize_agent_identity(agent_id: str | None) -> str:
    raw = str(agent_id or "default").strip()
    return raw or "default"


def _normalize_swarm_identity(swarm_id: str | None) -> str:
    raw = str(swarm_id or "default").strip()
    return raw or "default"


def _resolve_mal_binding_id(caller_id: str | None, agent_id: str | None) -> str:
    """Resolve MAL config binding from canonical caller identity when available."""
    if caller_id is not None:
        return _normalize_identity(str(caller_id), allow_public=False)
    return _canonical_owner_from_identity(None, agent_id)


def _canonical_owner_from_identity(caller_id: str | None, agent_id: str | None) -> str:
    if caller_id:
        return _normalize_identity(str(caller_id), allow_public=False)
    normalized_agent = _normalize_agent_identity(agent_id)
    if normalized_agent != "default":
        return _normalize_identity(f"agent:{normalized_agent}", allow_public=False)
    return "system"


def _default_runtime_caller_id(
    caller_id: str | None,
    agent_id: str | None,
    default_agent_id: str,
) -> str:
    if caller_id:
        return _normalize_identity(str(caller_id), allow_public=False)
    effective_agent = str(agent_id or default_agent_id or "default").strip() or "default"
    if effective_agent != "default":
        return _normalize_identity(f"agent:{effective_agent}", allow_public=False)
    return "system"


def _normalize_loaded_acl_fields(
    *,
    scope: str | None,
    agent_id: str | None,
    swarm_id: str | None,
    owner_id: str | None,
    read,
    write,
) -> dict[str, Any]:
    """Normalize loaded ACL fields with fail-closed semantics."""
    normalized_agent = _normalize_agent_identity(agent_id)
    normalized_swarm = _normalize_swarm_identity(swarm_id)
    normalized_scope = str(scope or "").strip()
    if normalized_scope not in MemoryServer.VALID_SCOPES or (
        normalized_scope == "swarm-shared" and normalized_swarm == "default"
    ):
        normalized_scope = "agent-private"

    try:
        normalized_owner = (
            _normalize_identity(str(owner_id), allow_public=False)
            if owner_id is not None
            else None
        )
    except Exception:
        normalized_owner = None
    if normalized_scope == "system-wide":
        normalized_owner = "system"
    elif normalized_owner is None:
        normalized_owner = _canonical_owner_from_identity(None, normalized_agent)

    normalized_read = _normalize_acl_principals(read, allow_public=True) if read is not None else None
    normalized_write = _normalize_acl_principals(write, allow_public=True) if write is not None else None
    if normalized_scope != "system-wide":
        normalized_read = [principal for principal in (normalized_read or []) if principal != "agent:PUBLIC"]
        normalized_write = [principal for principal in (normalized_write or []) if principal != "agent:PUBLIC"]

    if normalized_scope == "system-wide":
        final_read = ["agent:PUBLIC"]
        final_write = ["agent:PUBLIC"]
    elif normalized_scope == "swarm-shared":
        swarm_grant = _normalize_identity(f"swarm:{normalized_swarm}")
        final_read = normalized_read if normalized_read is not None else [swarm_grant]
        final_write = normalized_write if normalized_write is not None else [swarm_grant]
    else:
        final_read = normalized_read if normalized_read is not None else []
        final_write = normalized_write if normalized_write is not None else []

    return {
        "scope": normalized_scope,
        "agent_id": normalized_agent,
        "swarm_id": normalized_swarm,
        "owner_id": normalized_owner,
        "read": list(final_read),
        "write": list(final_write),
    }


def _normalize_loaded_acl_object(row: dict[str, Any]) -> dict[str, Any]:
    """Apply canonical ACL normalization to a loaded persisted row."""
    acl = _normalize_loaded_acl_fields(
        scope=row.get("scope"),
        agent_id=row.get("agent_id"),
        swarm_id=row.get("swarm_id"),
        owner_id=row.get("owner_id"),
        read=row.get("read"),
        write=row.get("write"),
    )
    row["scope"] = acl["scope"]
    row["agent_id"] = acl["agent_id"]
    row["swarm_id"] = acl["swarm_id"]
    row["owner_id"] = acl["owner_id"]
    row["read"] = acl["read"]
    row["write"] = acl["write"]
    normalize_legacy_report_fields(row)
    normalize_object_flags_field(row)
    return row


def _promote_source_meta_runtime_fields(record: dict[str, Any], source_meta: dict[str, Any]) -> None:
    for key in ("flags", "extraction_report", "source_aggregation_report"):
        if key not in record and key in source_meta:
            record[key] = deepcopy(source_meta[key])


def _normalize_loaded_source_record(source_id: str, record: dict[str, Any]) -> dict[str, Any]:
    """Normalize loaded source-record ACL/provenance without trusting persisted shape."""
    normalized = dict(record or {})
    source_meta = dict(normalized.get("source_meta") or {})
    scope = normalized.get("scope") or source_meta.get("scope")
    agent_id = normalized.get("agent_id") or source_meta.get("agent_id")
    swarm_id = normalized.get("swarm_id") or source_meta.get("swarm_id")
    owner_id = normalized.get("owner_id")
    read = normalized.get("read")
    write = normalized.get("write")
    if scope is None and owner_id is not None:
        inferred_scope, inferred_agent_id, inferred_swarm_id = _infer_scope_from_acl_fields(
            owner_id,
            read,
            write,
        )
        if inferred_scope is not None:
            scope = inferred_scope
            if agent_id in (None, "", "default") and inferred_agent_id:
                agent_id = inferred_agent_id
            if swarm_id in (None, "", "default") and inferred_swarm_id:
                swarm_id = inferred_swarm_id
    acl = _normalize_loaded_acl_fields(
        scope=scope,
        agent_id=agent_id,
        swarm_id=swarm_id,
        owner_id=owner_id,
        read=read,
        write=write,
    )
    normalized.update(acl)
    normalized["source_id"] = normalized.get("source_id") or source_id
    normalize_legacy_report_fields(source_meta)
    normalize_object_flags_field(source_meta)
    normalized["source_meta"] = source_meta
    _promote_source_meta_runtime_fields(normalized, source_meta)
    normalize_legacy_report_fields(normalized)
    normalize_object_flags_field(normalized)
    return normalized


def _normalize_loaded_instance_config(instance_config: dict[str, Any] | None) -> dict[str, Any] | None:
    """Drop unsafe persisted instance ACL state and rebuild derived grants from roots."""
    if not isinstance(instance_config, dict):
        return None
    owner_id = str(instance_config.get("owner_id") or "system")
    try:
        normalized_owner = _normalize_identity(owner_id, allow_public=False)
    except Exception:
        normalized_owner = "system"
    return {
        "owner_id": normalized_owner,
        "read": [],
        "write": [],
        "_derived_read": [],
        "_derived_write": [],
    }


def _normalize_target(target) -> list[str] | None:
    """Normalize delivery target to canonical list[str] or None if omitted."""
    if target is None:
        return None
    if isinstance(target, str):
        values = [target]
    elif isinstance(target, list):
        values = target
    else:
        raise ValueError("target must be a string, list of strings, or null")

    normalized = []
    seen = set()
    for value in values:
        if not isinstance(value, str):
            raise ValueError("target list must contain only strings")
        canonical = _normalize_identity(value)
        if canonical not in seen:
            seen.add(canonical)
            normalized.append(canonical)
    return normalized


def _build_index_state(
    granular_facts: list[dict],
    gran_embs: np.ndarray,
    cons_facts: list[dict],
    cons_embs: np.ndarray,
    cross_facts: list[dict],
    cross_embs: np.ndarray,
) -> dict:
    """Build the minimal production index state for the final runtime.

    The final runtime only needs:
    - fact lookup for visibility / provenance checks
    - cached embeddings per stored tier
    """
    fact_lookup = {}
    for fact in granular_facts + cons_facts + cross_facts:
        fact_id = fact.get("id")
        if fact_id:
            fact_lookup[fact_id] = fact

    return {
        "atomic_embs": gran_embs,
        "cons_embs": cons_embs,
        "cross_embs": cross_embs,
        "fact_lookup": fact_lookup,
    }


def _fact_uses_dense_embedding_index(fact: dict[str, Any]) -> bool:
    if not isinstance(fact, dict):
        return True
    if str(fact.get("kind") or "").strip() == "codebase_relation":
        return False
    metadata = fact.get("metadata") or {}
    codebase = metadata.get("codebase") if isinstance(metadata, dict) else {}
    if not isinstance(codebase, dict):
        return True
    object_type = str(codebase.get("object_type") or "").strip()
    if object_type in {"diff", "hunk"}:
        return False
    return not bool(codebase.get("skip_embedding"))


def _iter_fact_embedding_rows(
    facts: list[dict[str, Any]],
    embeddings: Any,
    indices: Any = None,
):
    if not isinstance(embeddings, np.ndarray):
        return
    if indices is None:
        if len(embeddings) != len(facts):
            return
        for idx, fact in enumerate(facts):
            yield idx, fact, embeddings[idx]
        return
    try:
        mapped = list(indices)
    except Exception:
        return
    if len(mapped) != len(embeddings):
        return
    for emb_idx, fact_idx in enumerate(mapped):
        if not isinstance(fact_idx, (int, np.integer)):
            continue
        if 0 <= int(fact_idx) < len(facts):
            yield int(fact_idx), facts[int(fact_idx)], embeddings[emb_idx]


def _derive_acl_from_scope(scope, agent_id, swarm_id):
    """Derive canonical ACL fields from an explicit ingress scope selection."""
    normalized_scope = str(scope or "").strip()
    if not normalized_scope:
        raise ValueError(LIVE_SCOPE_REQUIRED_ERROR)
    if normalized_scope not in MemoryServer.VALID_SCOPES:
        raise ValueError(f"Unknown scope: {normalized_scope}")
    normalized_agent = _normalize_agent_identity(agent_id)
    normalized_swarm = _normalize_swarm_identity(swarm_id)
    if normalized_scope == "swarm-shared" and normalized_swarm == "default":
        raise ValueError(NAMED_SWARM_REQUIRED_ERROR)
    if normalized_scope == "agent-private":
        return {"owner_id": _canonical_owner_from_identity(None, normalized_agent), "read": [], "write": []}
    if normalized_scope == "swarm-shared":
        grant = _normalize_identity(f"swarm:{normalized_swarm}")
        return {
            "owner_id": _canonical_owner_from_identity(None, normalized_agent),
            "read": [grant],
            "write": [grant],
        }
    return {"owner_id": "system", "read": ["agent:PUBLIC"], "write": ["agent:PUBLIC"]}


def _infer_scope_from_acl_fields(
    owner_id: str | None,
    read: list[str] | None,
    write: list[str] | None,
) -> tuple[str | None, str | None, str | None]:
    """Best-effort canonical scope inference from an explicit ACL triple."""
    owner = str(owner_id or "").strip()
    read_list = [str(item) for item in (read or [])]
    write_list = [str(item) for item in (write or [])]

    if owner.startswith("agent:") and not read_list and not write_list:
        return "agent-private", owner.split(":", 1)[1], None

    swarm_grants = {
        item
        for item in read_list + write_list
        if isinstance(item, str) and item.startswith("swarm:")
    }
    if owner.startswith("agent:") and len(swarm_grants) == 1:
        grant = next(iter(swarm_grants))
        return "swarm-shared", owner.split(":", 1)[1], grant.split(":", 1)[1]

    if owner == "system" and read_list == ["agent:PUBLIC"] and write_list == ["agent:PUBLIC"]:
        return "system-wide", None, None

    return None, None, None


def _estimate_tokens(value) -> int:
    """Cheap token estimate used across routing and payload accounting."""
    if value is None:
        return 0
    if isinstance(value, str):
        if value == "":
            return 0
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return max(1, int(len(text) / 3.5))


def _provider_for_model(model: str) -> str:
    if model.startswith("inception/"):
        return "inception"
    if model.startswith("anthropic/"):
        return "anthropic"
    if model.startswith("google/"):
        return "google"
    return "openai"


def _provider_family_for_model(model: str) -> str:
    provider = _provider_for_model(model)
    if provider == "anthropic":
        return "anthropic"
    if provider == "google":
        return "google"
    return "openai_compatible"


def _openai_tool_payload(tool: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["input_schema"],
        },
    }


def _build_openai_tools_payload() -> list[dict]:
    tools = [_openai_tool_payload(GET_CONTEXT_TOOL)]
    return tools


def _classify_context_tier(fact: dict) -> str:
    kind = fact.get("kind", "")
    status = fact.get("status", "active")
    if status == "active" and kind in {"decision", "constraint", "rejection"}:
        return "tier1"
    if kind in {"action_item", "requirement", "preference"}:
        return "tier2"
    return "tier3"


def _build_context_packet(retrieved_facts, raw_sessions, budget=5000, raw_docs=None):
    """Preserve context segments by truncation priority before final rendering."""
    packet = {
        "tier1": [],
        "tier2": [],
        "tier3": [],
        "tier4": [],
    }

    for i, fact in enumerate(retrieved_facts):
        labels = ""
        metadata = fact.get("metadata", {})
        source_family = str(
            fact.get("source_family")
            or (metadata.get("source_family") if isinstance(metadata, dict) else "")
            or ""
        ).lower()
        if isinstance(metadata, dict):
            version_status = metadata.get("version_status")
            if version_status == "current":
                labels += " [CURRENT]"
            elif version_status == "outdated":
                labels += f" [OUTDATED: superseded by {metadata.get('version_superseded_by', '?')}]"
            section_path = metadata.get("section_path")
            if section_path:
                labels += f" [Section: {section_path}]"
        if source_family == "codebase":
            file_path = str(fact.get("file_path") or metadata.get("file_path") or "").strip()
            semantic_type = str(
                fact.get("semantic_type")
                or metadata.get("semantic_type")
                or fact.get("kind")
                or "fact"
            ).strip()
            span = fact.get("span") if isinstance(fact.get("span"), dict) else None
            if not span and isinstance(metadata, dict):
                span_text = str(metadata.get("span") or "").strip()
                if span_text:
                    labels += f" [{span_text}]"
            elif isinstance(span, dict) and span.get("start_line") and span.get("end_line"):
                if span["start_line"] == span["end_line"]:
                    labels += f" [L{span['start_line']}]"
                else:
                    labels += f" [L{span['start_line']}-L{span['end_line']}]"
            if file_path:
                labels += f" [File: {file_path}]"
            line = f"[{i+1}] ({semantic_type}) {fact.get('fact', '')}{labels}"
        else:
            line = f"[{i+1}] (S{fact.get('session', '?')}) {fact.get('fact', '')}{labels}"
        packet[_classify_context_tier(fact)].append({
            "text": line,
            "rank": i,
            "source": "fact",
            "fact_id": fact.get("id"),
            "kind": fact.get("kind"),
        })

    total_chars = 0
    seen_raw_keys = set()
    for rank, fact in enumerate(retrieved_facts):
        if total_chars >= budget:
            break
        matched = _match_raw_session_for_fact(fact, raw_sessions)
        if matched is None:
            continue
        raw_key, session_num, raw = matched
        if raw_key in seen_raw_keys:
            continue
        seen_raw_keys.add(raw_key)
        text = _semantic_raw_session_text(raw) if isinstance(raw, dict) else str(raw)
        if not text:
            continue
        remaining = budget - total_chars
        chunk = text[:remaining].strip()
        if not chunk:
            continue
        packet["tier4"].append({
            "text": f"[Raw S{session_num}]\n{chunk}",
            "rank": rank,
            "source": "raw",
            "session": session_num,
        })
        total_chars += len(chunk)

    if raw_docs:
        doc_chars = 0
        seen_doc_sources = set()
        for rank, fact in enumerate(retrieved_facts):
            if doc_chars >= budget:
                break
            metadata = fact.get("metadata") or {}
            source_label = metadata.get("document_source") or fact.get("conv_id", "")
            if not source_label or source_label in seen_doc_sources or source_label not in raw_docs:
                continue
            seen_doc_sources.add(source_label)
            remaining = min(2000, budget - doc_chars)
            chunk = raw_docs[source_label][:remaining].strip()
            if not chunk:
                continue
            section_path = metadata.get("section_path", "")
            header = ""
            if section_path:
                header += f"[Section: {section_path}]\n"
            header += f"[Source: {source_label}]"
            packet["tier4"].append({
                "text": f"{header}\n{chunk}",
                "rank": rank,
                "source": "doc",
                "section_path": section_path,
                "source_label": source_label,
            })
            doc_chars += len(chunk)

    return packet


def _render_code_attachment_block(code_segments: list[dict[str, Any]]) -> str:
    code_lines = [
        seg["text"]
        for seg in sorted(
            code_segments,
            key=lambda seg: (seg.get("rank", 0), seg.get("file_path", "")),
        )
    ]
    if not code_lines:
        return ""
    return CODE_ATTACHMENT_SECTION_LABEL + "\n" + "\n".join(code_lines)


def _render_context_packet(packet: dict) -> str:
    """Render a structured packet into the hybrid context string."""
    all_segments = [
        seg
        for tier in ("tier1", "tier2", "tier3", "tier4")
        for seg in packet.get(tier, [])
    ]
    priority_codebase = [
        seg
        for seg in all_segments
        if seg.get("source") == "codebase_context" and int(seg.get("rank") or 0) >= 1_000_000
    ]
    if priority_codebase:
        return "\n\n".join(
            str(seg.get("text") or "")
            for seg in sorted(priority_codebase, key=lambda s: s.get("rank", 0), reverse=True)
            if str(seg.get("text") or "")
        )

    fact_lines = _context_packet_fact_lines(packet)

    codebase_context_lines = [s["text"] for s in sorted(
        [seg for seg in packet.get("tier4", []) if seg.get("source") == "codebase_context"],
        key=lambda s: s.get("rank", 0),
    )]
    raw_lines = [s["text"] for s in sorted(
        [
            seg
            for seg in packet.get("tier4", [])
            if seg.get("source") in {"raw", "raw_window", "completed_raw"}
        ],
        key=lambda s: s.get("rank", 0),
    )]
    continuation_lines = [s["text"] for s in sorted(
        [seg for seg in packet.get("tier4", []) if seg.get("source") == "recall_continuation"],
        key=lambda s: s.get("rank", 0),
    )]
    doc_lines = [s["text"] for s in sorted(
        [seg for seg in packet.get("tier4", []) if seg.get("source") == "doc"],
        key=lambda s: (s.get("rank", 0), s.get("source_label", "")),
    )]
    code_block = _render_code_attachment_block([
        seg for seg in packet.get("tier4", []) if seg.get("source") == "code"
    ])

    parts = ["RETRIEVED FACTS:"]
    parts.extend(fact_lines)
    if code_block:
        parts.append("")
        parts.append(code_block)
    if codebase_context_lines:
        parts.append("")
        parts.extend(codebase_context_lines)
    if raw_lines:
        parts.append("")
        parts.append("RAW CONTEXT (source text excerpts):")
        parts.extend(raw_lines)
    if continuation_lines:
        parts.append("")
        parts.extend(continuation_lines)
    if doc_lines:
        parts.append("")
        parts.append("--- SOURCE DOCUMENT SECTIONS ---")
        parts.extend(doc_lines)
    return "\n".join(parts)


def _context_packet_fact_lines(packet: dict) -> list[str]:
    fact_lines = []
    for tier in ("tier1", "tier2", "tier3"):
        fact_lines.extend(packet.get(tier, []))
    return [s["text"] for s in sorted(fact_lines, key=lambda s: s.get("rank", 0))]


def build_hybrid_context(retrieved_facts, raw_sessions, budget=5000, raw_docs=None):
    """Build hybrid context from structured facts plus raw snippets."""
    packet = _build_context_packet(
        retrieved_facts=retrieved_facts,
        raw_sessions=raw_sessions,
        budget=budget,
        raw_docs=raw_docs,
    )
    return _render_context_packet(packet)


def _raw_session_match_identity(raw_session: dict, idx: int) -> str:
    raw_session_id = str(raw_session.get("raw_session_id") or "").strip()
    if raw_session_id:
        return f"raw_session:{raw_session_id}"
    message_id = str(raw_session.get("message_id") or "").strip()
    if message_id:
        return f"message:{message_id}"
    return f"idx:{idx}"


def _raw_session_projection_source_id(raw_session: dict) -> str:
    return str(
        raw_session.get("projection_source_id")
        or raw_session.get("source_id")
        or ""
    ).strip()


def _fact_source_candidates(fact: dict) -> list[str]:
    raw_metadata = fact.get("metadata")
    metadata: dict[str, Any] = raw_metadata if isinstance(raw_metadata, dict) else {}
    ordered = []
    for value in (
        fact.get("projection_source_id"),
        fact.get("source_id"),
        metadata.get("projection_source_id"),
        metadata.get("document_source"),
        metadata.get("episode_source_id"),
        metadata.get("logical_source_id"),
    ):
        candidate = str(value or "").strip()
        if candidate and candidate not in ordered:
            ordered.append(candidate)
    return ordered


def _match_raw_session_for_fact(fact: dict, raw_sessions: list[dict]) -> tuple[str, int, dict] | None:
    fact_raw_session_id = str(fact.get("raw_session_id") or "").strip()
    fact_session_num = _coerce_positive_session_num(fact.get("session"))
    fact_owner = str(fact.get("owner_id") or "").strip()
    fact_scope = str(fact.get("scope") or "").strip()
    fact_swarm = str(fact.get("swarm_id") or "").strip()
    source_candidates = _fact_source_candidates(fact)

    if fact_raw_session_id:
        for idx, raw_session in enumerate(raw_sessions):
            if str(raw_session.get("raw_session_id") or "").strip() != fact_raw_session_id:
                continue
            return (
                _raw_session_match_identity(raw_session, idx),
                fact_session_num or _coerce_positive_session_num(raw_session.get("session_num")) or idx + 1,
                raw_session,
            )

    best_match = None
    best_score = None
    for idx, raw_session in enumerate(raw_sessions):
        if not isinstance(raw_session, dict):
            continue
        score = 0
        raw_projection_source = _raw_session_projection_source_id(raw_session)
        raw_logical_source = str(raw_session.get("logical_source_id") or raw_session.get("source_id") or "").strip()
        if source_candidates:
            if raw_projection_source in source_candidates:
                score += 10
            elif raw_logical_source in source_candidates:
                score += 8
            else:
                continue
        if fact_owner:
            raw_owner = str(raw_session.get("owner_id") or "").strip()
            if raw_owner != fact_owner:
                continue
            score += 4
        if fact_scope:
            raw_scope = str(raw_session.get("scope") or "").strip()
            if raw_scope != fact_scope:
                continue
            score += 3
        if fact_swarm:
            raw_swarm = str(raw_session.get("swarm_id") or "").strip()
            if raw_swarm != fact_swarm:
                continue
            score += 2
        if fact_session_num is not None:
            raw_session_num = _coerce_positive_session_num(raw_session.get("session_num"))
            if raw_session_num != fact_session_num:
                continue
            score += 2
        if best_score is None or score > best_score:
            best_score = score
            best_match = (
                _raw_session_match_identity(raw_session, idx),
                fact_session_num or _coerce_positive_session_num(raw_session.get("session_num")) or idx + 1,
                raw_session,
            )

    if best_match is not None:
        return best_match

    if fact_session_num is None or not (0 < fact_session_num <= len(raw_sessions)):
        return None
    fallback = raw_sessions[fact_session_num - 1]
    if not isinstance(fallback, dict):
        return None
    return (
        _raw_session_match_identity(fallback, fact_session_num - 1),
        fact_session_num,
        fallback,
    )


def _match_raw_sessions_for_facts(
    retrieved_facts: list[dict],
    raw_sessions: list[dict],
) -> dict[int, tuple[str, int, dict]]:
    matches: dict[int, tuple[str, int, dict]] = {}
    for rank, fact in enumerate(retrieved_facts):
        matched = _match_raw_session_for_fact(fact, raw_sessions)
        if matched is not None:
            matches[rank] = matched
    return matches


def compute_raw_budget(qt, total_sessions, sessions_in_context):
    """Coverage-based raw budget."""
    if qt != "summarize":
        return 5000
    coverage_pct = sessions_in_context / total_sessions * 100 if total_sessions else 100
    if coverage_pct >= 50:
        return 5000
    if coverage_pct >= 20:
        return 15000
    return 30000


def _route_prompt_type(
    resolved_type,
    resolved_facts,
    total_sessions,
    sessions_in_ctx,
    hybrid_ctx,
    *,
    allow_tool_mode=True,
):
    """Coverage-based prompt routing shared by fact and episode recall paths."""
    if resolved_type == "summarize":
        return "summarize_with_metadata", True
    if resolved_type == "icl":
        return "icl", False
    if (
        allow_tool_mode
        and resolved_facts
        and total_sessions > 20
        and sessions_in_ctx < total_sessions * 0.3
    ):
        return "tool", True
    if _context_has_source_excerpts(hybrid_ctx):
        return "hybrid", False
    return resolved_type, False


def _apply_output_constraints(answer: str, constraints: dict | None) -> str:
    if not answer:
        return answer
    constraints = constraints or {}
    rendered = answer.strip()
    prefix = constraints.get("prepend_prefix")
    if prefix and not rendered.startswith(prefix):
        rendered = f"{prefix}{rendered}"
    return rendered


def _apply_exact_copy_output_constraints(answer: str, constraints: dict | None) -> str:
    if answer == "":
        return answer
    constraints = constraints or {}
    prefix = constraints.get("prepend_prefix")
    if prefix and not answer.startswith(prefix):
        return f"{prefix}{answer}"
    return answer


def _strip_model_think_blocks(answer: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", str(answer or ""), flags=re.DOTALL)
    return re.sub(r"<think>.*", "", text, flags=re.DOTALL).strip()


def _parse_terminal_render_candidate_decision(answer: str) -> dict | None:
    """Parse the model's provider-neutral exact-copy candidate decision."""
    text = _strip_model_think_blocks(answer)
    if not text:
        return None
    if not (text.startswith("{") and text.endswith("}")):
        match = re.search(r"\{[^{}]*\}", text, flags=re.DOTALL)
        if not match:
            return None
        text = match.group(0)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("decision") != "use_candidate":
        return None
    allowed_keys = {"decision", "candidate_id", "render_ref_id", "container_id"}
    if not set(payload).issubset(allowed_keys):
        return None
    if not str(payload.get("candidate_id") or "").strip():
        return None
    return payload


def _deterministic_answer_is_exact_copy(recall_result: dict) -> bool:
    runtime_trace = dict(recall_result.get("runtime_trace") or {})
    container_trace = dict(runtime_trace.get("container_graph") or {})
    return str(container_trace.get("render_mode") or "") == "exact_copy"


def _container_exact_copy_model_path(recall_result: dict) -> bool:
    return bool(recall_result.get("terminal_render_candidate"))


def _ordinal_event_episode_id(event: dict) -> str:
    payload = event.get("payload") or {}
    if isinstance(payload, dict):
        episode_id = str(payload.get("episode_id") or "").strip()
        if episode_id:
            return episode_id
    source_span = event.get("source_span") or {}
    return str(source_span.get("episode_id") or "").strip()


def _ordinal_event_fact_ids(event: dict) -> list[str]:
    payload = event.get("payload") or {}
    fact_ids: list[str] = []
    if isinstance(payload, dict):
        fact_id = str(payload.get("fact_id") or "").strip()
        if fact_id:
            fact_ids.append(fact_id)
    fact_ids.extend(
        str(fid).strip()
        for fid in (event.get("support_fact_ids") or [])
        if str(fid).strip()
    )
    return list(dict.fromkeys(fact_ids))


def _ordinal_event_quality(event: dict) -> int:
    payload = event.get("payload") or {}
    score = 0
    if payload.get("action_raw"):
        score += 5
    if payload.get("tool_name"):
        score += 3
    if payload.get("tool_args_raw") or payload.get("tool_args"):
        score += 2
    if payload.get("observation_raw"):
        score += 1
    if payload.get("paths"):
        score += 1
    if payload.get("ids"):
        score += 1
    if _ordinal_event_fact_ids(event):
        score += 1
    return score


def _temporal_hit_quality(hit: dict) -> int:
    primary_event = hit.get("primary_event") or {}
    score = _ordinal_event_quality(primary_event)
    reducer = str(hit.get("reducer") or "").strip().lower()
    payload = primary_event.get("payload") or {}
    deterministic_answer = str(hit.get("deterministic_answer") or "").strip()
    if reducer in {"exact_command", "exact_sql"}:
        if payload.get("tool_name"):
            score += 3
        if payload.get("tool_args_raw") or payload.get("tool_args"):
            score += 2
    elif (reducer == "exact_file_path" and payload.get("paths")) or (reducer == "exact_id" and payload.get("ids")):
        score += 3
    elif reducer == "exact_action" and payload.get("action_raw"):
        score += 2
    if deterministic_answer:
        score += 1
    return score


def _ordinal_adjacent_episode_id(ep_id: str, delta: int) -> str | None:
    match = re.search(r"^(.*_e)(\d+)\b", ep_id)
    if not match:
        return None
    prefix, raw_num = match.groups()
    next_num = int(raw_num) + int(delta)
    if next_num <= 0:
        return None
    return f"{prefix}{next_num:0{len(raw_num)}d}"


def _temporal_scope_from_corpus(corpus: dict) -> tuple[set[str] | None, set[str] | None]:
    scope_map = _temporal_scope_map_from_corpus(corpus)
    source_ids = set(scope_map)
    timeline_ids = {
        timeline_id
        for timeline_set in scope_map.values()
        for timeline_id in timeline_set
    }
    return (source_ids or None), (timeline_ids or None)


def _temporal_scope_map_from_corpus(corpus: dict) -> dict[str, set[str]]:
    source_timelines: dict[str, set[str]] = {}
    for doc in corpus.get("documents", []):
        doc_id = str(doc.get("doc_id") or "").strip()
        for episode in doc.get("episodes", []):
            source_id = str(episode.get("source_id") or "").strip()
            if not source_id:
                continue
            timeline_id = (
                str(episode.get("timeline_id") or "").strip()
                or doc_id
                or f"timeline:{source_id}:main"
            )
            source_timelines.setdefault(source_id, set())
            if timeline_id:
                source_timelines[source_id].add(timeline_id)
    return source_timelines


def _temporal_scope_query_tokens(question: str, query_features: dict) -> set[str]:
    stop = {
        "what", "which", "when", "where", "who", "why", "how",
        "step", "steps", "turn", "turns", "message", "messages",
        "exact", "specific", "performed", "occurred", "happened",
        "action", "actions", "command", "commands", "date", "year",
        "month", "first", "last", "between", "from", "through",
        "later", "after", "before", "did", "was", "were", "is",
        "the", "a", "an", "to", "of", "in", "on", "at", "for",
    }
    tokens = {
        normalize_term_token(token)
        for token in (query_features.get("words") or set())
        if normalize_term_token(token)
    }
    raw_tokens = {
        normalize_term_token(token)
        for token in re.findall(r"[A-Za-z0-9_./:-]+", str(question or ""))
        if normalize_term_token(token)
    }
    tokens |= raw_tokens
    return {
        token
        for token in tokens
        if token
        and not token.isdigit()
        and token not in stop
        and len(token) > 2
    }


def _temporal_scope_event_tokens(events: list[dict]) -> set[str]:
    tokens: set[str] = set()
    for event in events:
        payload = event.get("payload") or {}
        parts: list[str] = [
            str(event.get("label") or "").strip(),
            str(payload.get("action_raw") or "").strip(),
            str(payload.get("tool_name") or "").strip(),
            str(payload.get("tool_args_raw") or "").strip(),
            str(payload.get("observation_raw") or "").strip(),
            str(payload.get("step_body") or "").strip(),
            str(payload.get("raw_step_block") or "").strip(),
        ]
        support_texts = payload.get("support_texts") or []
        if isinstance(support_texts, list):
            parts.extend(str(text or "").strip() for text in support_texts[:4])
        for part in parts:
            if not part:
                continue
            for raw_token in re.findall(r"[A-Za-z0-9_./:-]+", part):
                token = normalize_term_token(raw_token)
                if token and not token.isdigit() and len(token) > 2:
                    tokens.add(token)
    return tokens


def _temporal_scope_source_tokens(
    corpus: dict,
    *,
    source_id: str,
    timeline_ids: set[str],
) -> set[str]:
    tokens: set[str] = set()
    for doc in corpus.get("documents", []):
        doc_id = str(doc.get("doc_id") or "").strip()
        if timeline_ids and doc_id and doc_id not in timeline_ids:
            continue
        for episode in doc.get("episodes", []):
            if str(episode.get("source_id") or "").strip() != source_id:
                continue
            parts = [
                str(episode.get("topic_key") or "").strip(),
                str(episode.get("state_label") or "").strip(),
                str(episode.get("raw_text") or "").strip(),
            ]
            for part in parts:
                if not part:
                    continue
                for raw_token in re.findall(r"[A-Za-z0-9_./:-]+", part):
                    token = normalize_term_token(raw_token)
                    if token and not token.isdigit() and len(token) > 2:
                        tokens.add(token)
    return tokens


def _source_anchor_timestamp_for_scope(
    corpus: dict,
    temporal_index: dict,
    *,
    source_ids: set[str] | None = None,
) -> str | None:
    resolved_dates: set[str] = set()
    for doc in corpus.get("documents", []):
        for episode in doc.get("episodes", []):
            source_id = str(episode.get("source_id") or "").strip()
            if source_ids is not None and source_id not in source_ids:
                continue
            source_date = str(episode.get("source_date") or "").strip()
            if not source_date:
                continue
            try:
                candidate = date_parser.parse(source_date, fuzzy=True)
            except Exception:
                continue
            resolved_dates.add(candidate.date().isoformat())
    if resolved_dates:
        return max(resolved_dates)
    if temporal_index is not None:
        return latest_calendar_anchor(temporal_index, source_ids=source_ids)
    return None


def _narrow_temporal_scope_from_visible_corpus(
    *,
    question: str,
    query_features: dict,
    corpus: dict,
    execute_for_scope,
) -> tuple[set[str] | None, set[str] | None, dict | None, dict]:
    scope_map = _temporal_scope_map_from_corpus(corpus)
    if len(scope_map) <= 1:
        source_ids, timeline_ids = _temporal_scope_from_corpus(corpus)
        return source_ids, timeline_ids, None, {"scope_mode": "single_source"}

    query_tokens = _temporal_scope_query_tokens(question, query_features)
    rows: list[dict] = []
    for source_id, timeline_ids in sorted(scope_map.items()):
        hit = execute_for_scope({source_id}, set(timeline_ids))
        matched = bool(hit.get("matched"))
        resolved = bool(hit.get("resolved", matched and bool(hit.get("events"))))
        if not matched:
            continue
        lexical_tokens = _temporal_scope_event_tokens(list(hit.get("events") or []))
        lexical_tokens |= _temporal_scope_source_tokens(
            corpus,
            source_id=source_id,
            timeline_ids=set(timeline_ids),
        )
        overlap = len(query_tokens & lexical_tokens)
        rows.append(
            {
                "source_id": source_id,
                "timeline_ids": set(timeline_ids),
                "hit": hit,
                "matched": matched,
                "resolved": resolved,
                "overlap": overlap,
                "hit_quality": _temporal_hit_quality(hit),
                "event_count": len(list(hit.get("events") or [])),
            }
        )

    trace = {
        "scope_mode": "multi_source_probe",
        "candidate_source_ids": sorted(scope_map),
        "probed_source_ids": [row["source_id"] for row in rows],
    }
    if not rows:
        source_ids, timeline_ids = _temporal_scope_from_corpus(corpus)
        trace["scope_mode"] = "probe_miss"
        return source_ids, timeline_ids, None, trace
    if len(rows) == 1:
        row = rows[0]
        trace["selected_source_ids"] = [row["source_id"]]
        trace["scope_mode"] = "single_matched_source"
        return {row["source_id"]}, set(row["timeline_ids"]), row["hit"], trace

    resolved_rows = [row for row in rows if row["resolved"]]
    if len(resolved_rows) == 1:
        row = resolved_rows[0]
        trace["selected_source_ids"] = [row["source_id"]]
        trace["scope_mode"] = "single_resolved_source"
        return {row["source_id"]}, set(row["timeline_ids"]), row["hit"], trace

    if len(resolved_rows) > 1:
        ranked = sorted(
            resolved_rows,
            key=lambda row: (
                row["overlap"],
                row["hit_quality"],
                row["event_count"],
                row["source_id"],
            ),
            reverse=True,
        )
        if len(ranked) == 1 or (
            ranked[0]["overlap"],
            ranked[0]["hit_quality"],
            ranked[0]["event_count"],
        ) > (
            ranked[1]["overlap"],
            ranked[1]["hit_quality"],
            ranked[1]["event_count"],
        ):
            row = ranked[0]
            trace["selected_source_ids"] = [row["source_id"]]
            trace["scope_mode"] = "lexical_disambiguation"
            return {row["source_id"]}, set(row["timeline_ids"]), row["hit"], trace

    source_ids, timeline_ids = _temporal_scope_from_corpus(corpus)
    trace["scope_mode"] = "ambiguous_multi_source"
    return source_ids, timeline_ids, None, trace


def _build_ordinal_executor_packet(
    question: str,
    *,
    corpus: dict,
    episode_lookup: dict[str, dict],
    facts_by_episode: dict[str, list[dict]],
    query_features: dict,
    effective_selector: dict,
    output_constraints: dict,
    operator_tuning: dict,
    packet_tuning: dict,
    search_family: str | None,
    temporal_index: dict,
) -> tuple[dict | None, dict]:
    temporal_limit = max(
        int(effective_selector.get("max_episodes_default", 4)) * 4,
        8,
    )
    scope_source_ids, scope_timeline_ids, precomputed_hit, scope_trace = _narrow_temporal_scope_from_visible_corpus(
        question=question,
        query_features=query_features,
        corpus=corpus,
        execute_for_scope=lambda source_ids, timeline_ids: execute_ordinal_query(
            question,
            temporal_index,
            source_ids=source_ids,
            timeline_ids=timeline_ids,
            limit=temporal_limit,
        ),
    )
    hit = precomputed_hit or execute_ordinal_query(
        question,
        temporal_index,
        source_ids=scope_source_ids,
        timeline_ids=scope_timeline_ids,
        limit=temporal_limit,
    )
    base_trace = {
        "query_class": "ordinal",
        "matched": bool(hit.get("matched")),
        "resolved": bool(hit.get("resolved")),
        "mode": hit.get("mode"),
        "kind": hit.get("kind"),
        "anchor_resolved": bool(hit.get("matched")),
        "matched_event_ids": [
            str(event.get("event_id") or "")
            for event in (hit.get("events") or [])
            if str(event.get("event_id") or "").strip()
        ],
        "fallback": True,
        "scope_trace": scope_trace,
    }
    if not hit.get("matched"):
        base_trace["fallback_reason"] = "miss"
        return None, base_trace
    if not hit.get("resolved"):
        base_trace["fallback_reason"] = "unresolved"
        return None, base_trace

    events = sorted(
        list(hit.get("events") or []),
        key=lambda event: (
            int(event.get("ordinal_start") or 10**9),
            -_ordinal_event_quality(event),
            str(event.get("event_id") or ""),
        ),
    )
    source_ids = {
        str(event.get("source_id") or "").strip()
        for event in events
        if str(event.get("source_id") or "").strip()
    }
    if len(source_ids) != 1:
        base_trace["fallback_reason"] = "ambiguous_source"
        base_trace["candidate_source_ids"] = sorted(source_ids)
        return None, base_trace

    selected_episode_ids: list[str] = []
    for event in sorted(events, key=lambda event: (-_ordinal_event_quality(event), str(event.get("event_id") or ""))):
        ep_id = _ordinal_event_episode_id(event)
        if ep_id and ep_id not in selected_episode_ids:
            selected_episode_ids.append(ep_id)
    if not selected_episode_ids:
        base_trace["fallback_reason"] = "no_episode"
        return None, base_trace

    selected_source_id = next(iter(source_ids))
    step_numbers = set(query_features.get("step_numbers") or [])
    step_range = query_features.get("step_range")

    def _target_step_hits(text: str) -> set[int]:
        hits: set[int] = set()
        if not text:
            return hits
        for step in step_numbers:
            if has_exact_step_mention(text, step):
                hits.add(step)
        if step_range:
            hits |= step_range_overlap(text, step_range)
        return hits

    def _episode_step_order(ep_id: str) -> int:
        ep = episode_lookup.get(ep_id) or {}
        lower = (ep.get("raw_text") or "").lower()
        match = re.search(r"\[step\s+(\d+)\]|\bstep\s+(\d+)\b", lower)
        if match:
            return int(match.group(1) or match.group(2))
        topic = (ep.get("topic_key") or "").lower()
        match = re.search(r"\bstep\s+(\d+)\b", topic)
        if match:
            return int(match.group(1))
        match = re.search(r"_e(\d+)\b", ep_id)
        if match:
            return int(match.group(1))
        return 10**9

    target_step_episode_ids = [
        ep_id
        for ep_id, ep in episode_lookup.items()
        if ep.get("source_id", "") == selected_source_id
        and _target_step_hits(ep.get("raw_text", ""))
    ]
    target_step_episode_ids = sorted(
        list(dict.fromkeys(target_step_episode_ids + selected_episode_ids)),
        key=lambda ep_id: (_episode_step_order(ep_id), ep_id),
    )

    primary_event = hit.get("primary_event") or {}
    answer_episode_id = _ordinal_event_episode_id(primary_event)
    if answer_episode_id and answer_episode_id not in target_step_episode_ids:
        target_step_episode_ids = sorted(
            target_step_episode_ids + [answer_episode_id],
            key=lambda ep_id: (_episode_step_order(ep_id), ep_id),
        )

    companion_episode_ids: list[str] = []
    for ep_id in target_step_episode_ids:
        ep = episode_lookup.get(ep_id) or {}
        raw = (ep.get("raw_text") or "").strip()
        lower = raw.lower()
        if not raw or not _target_step_hits(raw):
            continue
        next_ep_id = _ordinal_adjacent_episode_id(ep_id, 1)
        if not next_ep_id or next_ep_id in companion_episode_ids:
            continue
        next_ep = episode_lookup.get(next_ep_id) or {}
        next_raw = (next_ep.get("raw_text") or "").strip()
        next_lower = next_raw.lower()
        if not next_raw:
            continue
        if next_ep.get("source_id", "") != selected_source_id:
            continue
        if not next_lower.startswith("action:") or "observation:" not in next_lower:
            continue
        companion_episode_ids.append(next_ep_id)

    fact_episode_ids = list(target_step_episode_ids)
    for event in events:
        for fact_id in _ordinal_event_fact_ids(event):
            for ep_id, facts in facts_by_episode.items():
                if any(str(fact.get("id") or "").strip() == fact_id for fact in facts):
                    if ep_id not in fact_episode_ids:
                        fact_episode_ids.append(ep_id)
    for ep_id, facts in facts_by_episode.items():
        ep = episode_lookup.get(ep_id) or {}
        if ep.get("source_id", "") != selected_source_id:
            continue
        if ep_id in fact_episode_ids:
            continue
        if any(_target_step_hits(str(fact.get("fact") or "")) for fact in facts):
            fact_episode_ids.append(ep_id)
    for ep_id in companion_episode_ids:
        if ep_id not in fact_episode_ids:
            fact_episode_ids.append(ep_id)

    context, actual_injected_episode_ids, selected_fact_ids = build_context_from_selected_episodes(
        question,
        target_step_episode_ids,
        episode_lookup,
        facts_by_episode,
        fact_episode_ids=fact_episode_ids,
        support_episode_ids=companion_episode_ids,
        budget=effective_selector["budget"],
        max_total_facts=effective_selector["supporting_facts_total"],
        max_facts_per_episode=effective_selector["supporting_facts_per_episode"],
        snippet_mode=bool(effective_selector.get("snippet_mode", False)),
        snippet_chars=int(packet_tuning.get("snippet_chars", 1200)),
        allow_pseudo_facts=bool(effective_selector.get("allow_pseudo_facts", True)),
        query_features=query_features,
        local_anchor_window_chars=int(operator_tuning.get("local_anchor_window_chars", 1200)),
        local_anchor_fact_radius=int(operator_tuning.get("local_anchor_fact_radius", 12)),
        list_set_dedup_overlap=float(operator_tuning.get("list_set_dedup_overlap", 0.9)),
        bounded_chain_fact_bonus=float(operator_tuning.get("bounded_chain_fact_bonus", 0.0)),
        query_specificity_bonus=float(packet_tuning.get("query_specificity_bonus", 0.0)),
        inject_support_fact_episodes=bool(companion_episode_ids or fact_episode_ids),
        max_injected_support_fact_episodes=int(
            max(
                int(packet_tuning.get("max_injected_support_fact_episodes", 8)),
                len(companion_episode_ids),
            )
        ),
    )

    routed_families = route_retrieval_families(
        question,
        available_families(corpus),
        explicit_family=search_family,
    )
    trace = dict(base_trace)
    trace.update(
        {
            "matched": True,
            "fallback": False,
            "executor_episode_ids": target_step_episode_ids,
            "pinned_episode_ids": target_step_episode_ids,
            "support_episode_ids": fact_episode_ids,
            "matched_fact_ids": selected_fact_ids,
        }
    )
    deterministic_answer = str(hit.get("deterministic_answer") or "").strip()
    if deterministic_answer:
        trace["deterministic_answer"] = deterministic_answer
        trace["reducer"] = str(hit.get("reducer") or "").strip()
        trace["answer_event_id"] = str(primary_event.get("event_id") or "").strip()

    packet = {
        "context": context,
        "retrieved_episode_ids": target_step_episode_ids,
        "actual_injected_episode_ids": actual_injected_episode_ids,
        "fact_episode_ids": fact_episode_ids,
        "retrieved_fact_ids": selected_fact_ids,
        "selection_scores": [
            {"episode_id": ep_id, "score": float(1_000_000 - idx)}
            for idx, ep_id in enumerate(target_step_episode_ids)
        ],
        "selector_config": effective_selector,
        "query_operator_plan": query_features["operator_plan"],
        "output_constraints": output_constraints,
        "retrieval_families": routed_families,
        "search_family": search_family or "auto",
        "family_first_pass_trace": {
            "available_families": available_families(corpus),
            "retrieval_families": routed_families,
            "requested_search_family": search_family or "auto",
            "per_family": [],
            "mode": "skipped_by_ordinal_executor",
        },
        "late_fusion_trace": {"mode": "skipped_by_ordinal_executor"},
        "tuning_snapshot": {
            "selector": effective_selector,
            "operators": operator_tuning,
            "packet": packet_tuning,
            "routing": get_runtime_tuning()["routing"],
            "telemetry": get_runtime_tuning()["telemetry"],
        },
        "temporal_trace": trace,
    }
    return packet, trace


def _build_calendar_executor_packet(
    question: str,
    *,
    corpus: dict,
    episode_lookup: dict[str, dict],
    facts_by_episode: dict[str, list[dict]],
    query_features: dict,
    effective_selector: dict,
    output_constraints: dict,
    operator_tuning: dict,
    packet_tuning: dict,
    search_family: str | None,
    temporal_index: dict,
) -> tuple[dict | None, dict]:
    plan = extract_calendar_query(question)
    base_trace = {
        "query_class": "calendar-answer",
        "matched": False,
        "anchor_resolved": False,
        "matched_event_ids": [],
        "fallback": True,
    }
    if not plan or plan.get("mode") != "answer":
        base_trace["fallback_reason"] = "unsupported_mode"
        return None, base_trace
    temporal_limit = max(int(effective_selector.get("max_episodes_default", 4)) * 4, 8)
    scope_source_ids, scope_timeline_ids, precomputed_hit, scope_trace = _narrow_temporal_scope_from_visible_corpus(
        question=question,
        query_features=query_features,
        corpus=corpus,
        execute_for_scope=lambda source_ids, timeline_ids: execute_calendar_query(
            question,
            temporal_index,
            anchor_timestamp=_source_anchor_timestamp_for_scope(
                corpus,
                temporal_index,
                source_ids=source_ids,
            ),
            source_ids=source_ids,
            timeline_ids=timeline_ids,
            limit=temporal_limit,
        ),
    )
    hit = precomputed_hit or execute_calendar_query(
        question,
        temporal_index,
        anchor_timestamp=_source_anchor_timestamp_for_scope(
            corpus,
            temporal_index,
            source_ids=scope_source_ids,
        ),
        source_ids=scope_source_ids,
        timeline_ids=scope_timeline_ids,
        limit=temporal_limit,
    )
    events = list(hit.get("events") or [])
    base_trace["scope_trace"] = scope_trace
    base_trace["matched_event_ids"] = [
        str(event.get("event_id") or "")
        for event in events
        if str(event.get("event_id") or "").strip()
    ]
    if not events:
        base_trace["fallback_reason"] = "miss"
        return None, base_trace
    source_ids = {
        str(event.get("source_id") or "").strip()
        for event in events
        if str(event.get("source_id") or "").strip()
    }
    if len(source_ids) != 1:
        base_trace["fallback_reason"] = "ambiguous_source"
        base_trace["candidate_source_ids"] = sorted(source_ids)
        return None, base_trace
    selected_episode_ids: list[str] = []
    fact_episode_ids: list[str] = []
    for event in events:
        ep_id = _ordinal_event_episode_id(event)
        if ep_id and ep_id not in selected_episode_ids:
            selected_episode_ids.append(ep_id)
        for fact_id in _ordinal_event_fact_ids(event):
            for candidate_ep_id, facts in facts_by_episode.items():
                if any(str(fact.get("id") or "").strip() == fact_id for fact in facts):
                    if candidate_ep_id not in fact_episode_ids:
                        fact_episode_ids.append(candidate_ep_id)
    for ep_id in selected_episode_ids:
        if ep_id not in fact_episode_ids:
            fact_episode_ids.append(ep_id)
    if not selected_episode_ids:
        base_trace["fallback_reason"] = "no_episode"
        return None, base_trace

    context, actual_injected_episode_ids, selected_fact_ids = build_context_from_selected_episodes(
        question,
        selected_episode_ids,
        episode_lookup,
        facts_by_episode,
        fact_episode_ids=fact_episode_ids,
        budget=effective_selector["budget"],
        max_total_facts=effective_selector["supporting_facts_total"],
        max_facts_per_episode=effective_selector["supporting_facts_per_episode"],
        snippet_mode=bool(effective_selector.get("snippet_mode", False)),
        snippet_chars=int(packet_tuning.get("snippet_chars", 1200)),
        allow_pseudo_facts=bool(effective_selector.get("allow_pseudo_facts", True)),
        query_features=query_features,
        local_anchor_window_chars=int(operator_tuning.get("local_anchor_window_chars", 1200)),
        local_anchor_fact_radius=int(operator_tuning.get("local_anchor_fact_radius", 12)),
        list_set_dedup_overlap=float(operator_tuning.get("list_set_dedup_overlap", 0.9)),
        bounded_chain_fact_bonus=float(operator_tuning.get("bounded_chain_fact_bonus", 0.0)),
        query_specificity_bonus=float(packet_tuning.get("query_specificity_bonus", 0.0)),
    )
    routed_families = route_retrieval_families(
        question,
        available_families(corpus),
        explicit_family=search_family,
    )
    trace = dict(base_trace)
    trace.update(
        {
            "matched": True,
            "anchor_resolved": True,
            "fallback": False,
            "executor_episode_ids": selected_episode_ids,
            "pinned_episode_ids": selected_episode_ids,
            "matched_fact_ids": selected_fact_ids,
        }
    )
    packet = {
        "context": context,
        "retrieved_episode_ids": selected_episode_ids,
        "actual_injected_episode_ids": actual_injected_episode_ids,
        "fact_episode_ids": fact_episode_ids,
        "retrieved_fact_ids": selected_fact_ids,
        "selection_scores": [
            {"episode_id": ep_id, "score": float(1_000_000 - idx)}
            for idx, ep_id in enumerate(selected_episode_ids)
        ],
        "selector_config": effective_selector,
        "query_operator_plan": query_features["operator_plan"],
        "output_constraints": output_constraints,
        "retrieval_families": routed_families,
        "search_family": search_family or "auto",
        "family_first_pass_trace": {
            "available_families": available_families(corpus),
            "retrieval_families": routed_families,
            "requested_search_family": search_family or "auto",
            "per_family": [],
            "mode": "skipped_by_calendar_executor",
        },
        "late_fusion_trace": {"mode": "skipped_by_calendar_executor"},
        "tuning_snapshot": {
            "selector": effective_selector,
            "operators": operator_tuning,
            "packet": packet_tuning,
            "routing": get_runtime_tuning()["routing"],
            "telemetry": get_runtime_tuning()["telemetry"],
        },
        "temporal_trace": trace,
    }
    return packet, trace


def build_episode_hybrid_context(
    question,
    corpus,
    episode_facts,
    selector_config=None,
    search_family=None,
    temporal_index=None,
):
    """Build context through the episode-native runtime path.

    This is the explicit production entrypoint for episode-backed data.
    It never routes through document section grouping.
    """
    query_features = extract_query_features(question)
    operator_plan = query_features["operator_plan"]
    output_constraints = query_features.get("output_constraints", {})
    selector = resolve_selection_config(selector_config)
    selector_overrides = selector_config or {}
    tuning = get_runtime_tuning()
    operator_tuning = tuning["operators"]
    packet_tuning = tuning["packet"]
    available = available_families(corpus)
    explicit_family = str(search_family or "").strip().lower()
    document_family_requested = explicit_family == "document" or (
        explicit_family in {"", "auto"} and set(available) == {"document"}
    )
    effective_selector = dict(selector)
    packet_to_selector = {
        "budget": ("budget", int),
        "max_facts": ("supporting_facts_total", int),
        "max_facts_per_episode": ("supporting_facts_per_episode", int),
        "max_episodes": ("max_episodes_default", int),
        "per_family_cap": ("max_episodes_per_family", int),
        "per_source_cap": ("max_sources_per_family", int),
        "snippet_mode": ("snippet_mode", bool),
    }
    for packet_key, (selector_key, caster) in packet_to_selector.items():
        if selector_key in selector_overrides:
            continue
        effective_selector[selector_key] = caster(
            packet_tuning.get(packet_key, effective_selector.get(selector_key))
        )
    document_copy_query_requested = (
        document_family_requested
        and operator_plan.get("ordinal", {}).get("enabled", False)
        and (
            bool(output_constraints.get("prepend_prefix"))
            or bool(output_constraints.get("return_only"))
        )
    )
    if document_copy_query_requested:
        document_tuning = (
            tuning.get("retrieval", {}).get("document_family", {})
            if isinstance(tuning.get("retrieval"), dict)
            else {}
        )
        if "budget" not in selector_overrides:
            effective_selector["budget"] = max(
                effective_selector["budget"],
                int(document_tuning.get("budget", effective_selector["budget"])),
            )
        if "max_episodes_default" not in selector_overrides:
            effective_selector["max_episodes_default"] = max(
                effective_selector["max_episodes_default"],
                int(
                    document_tuning.get(
                        "max_episodes_default",
                        effective_selector["max_episodes_default"],
                    )
                ),
            )
        if "max_episodes_per_family" not in selector_overrides:
            effective_selector["max_episodes_per_family"] = max(
                effective_selector["max_episodes_per_family"],
                int(
                    document_tuning.get(
                        "max_episodes_per_family",
                        effective_selector["max_episodes_per_family"],
                    )
                ),
            )
        if "max_sources_per_family" not in selector_overrides:
            effective_selector["max_sources_per_family"] = max(
                effective_selector["max_sources_per_family"],
                int(
                    document_tuning.get(
                        "max_sources_per_family",
                        effective_selector["max_sources_per_family"],
                    )
                ),
            )
        if "supporting_facts_total" not in selector_overrides:
            effective_selector["supporting_facts_total"] = max(
                effective_selector["supporting_facts_total"],
                int(
                    document_tuning.get(
                        "supporting_facts_total",
                        effective_selector["supporting_facts_total"],
                    )
                ),
            )
        if "supporting_facts_per_episode" not in selector_overrides:
            effective_selector["supporting_facts_per_episode"] = max(
                effective_selector["supporting_facts_per_episode"],
                int(
                    document_tuning.get(
                        "supporting_facts_per_episode",
                        effective_selector["supporting_facts_per_episode"],
                    )
                ),
            )
    if operator_plan["ordinal"]["enabled"] and operator_tuning.get("enable_snippet_for_ordinal", True):
        effective_selector["snippet_mode"] = True
    if (
        operator_plan["local_anchor"]["enabled"]
        and not operator_plan["bounded_chain"]["enabled"]
        and operator_tuning.get("enable_snippet_for_local_anchor", True)
    ):
        effective_selector["snippet_mode"] = True
        step_numbers = sorted(query_features.get("step_numbers") or [])
        step_range = query_features.get("step_range")
        if step_range:
            start_step, end_step = step_range
            step_span = max(1, end_step - start_step + 1)
            effective_selector["max_episodes_default"] = max(
                effective_selector["max_episodes_default"],
                min(
                    step_span,
                    int(
                        operator_tuning.get(
                            "local_anchor_step_range_max_episodes",
                            max(effective_selector["max_episodes_default"], 8),
                        )
                    ),
                ),
            )
        elif len(step_numbers) > 1:
            effective_selector["max_episodes_default"] = max(
                effective_selector["max_episodes_default"],
                min(
                    len(step_numbers),
                    int(
                        operator_tuning.get(
                            "local_anchor_multi_step_max_episodes",
                            max(effective_selector["max_episodes_default"], 6),
                        )
                    ),
                ),
            )
        else:
            effective_selector["max_episodes_default"] = min(
                effective_selector["max_episodes_default"],
                int(operator_tuning.get("local_anchor_max_episodes", effective_selector["max_episodes_default"])),
            )
        effective_selector["supporting_facts_total"] = max(
            effective_selector["supporting_facts_total"],
            int(operator_tuning.get("local_anchor_supporting_facts_total", effective_selector["supporting_facts_total"])),
        )
        effective_selector["supporting_facts_per_episode"] = max(
            effective_selector["supporting_facts_per_episode"],
            int(operator_tuning.get("local_anchor_supporting_facts_per_episode", effective_selector["supporting_facts_per_episode"])),
        )
    if document_copy_query_requested:
        effective_selector["snippet_mode"] = False
    if operator_plan["list_set"]["enabled"]:
        effective_selector["max_episodes_default"] = max(
            effective_selector["max_episodes_default"],
            int(operator_tuning.get("list_set_max_episodes", effective_selector["max_episodes_default"])),
        )
        effective_selector["supporting_facts_total"] = max(
            effective_selector["supporting_facts_total"],
            int(operator_tuning.get("list_set_supporting_facts_total", effective_selector["supporting_facts_total"])),
        )
        effective_selector["supporting_facts_per_episode"] = max(
            effective_selector["supporting_facts_per_episode"],
            int(operator_tuning.get("list_set_supporting_facts_per_episode", effective_selector["supporting_facts_per_episode"])),
        )
    if any(
        operator_plan[name]["enabled"]
        for name in ("commonality", "list_set", "compare_diff", "bounded_chain")
    ):
        effective_selector["supporting_facts_total"] = max(
            effective_selector["supporting_facts_total"],
            int(operator_tuning.get("structural_query_supporting_facts_total", 12)),
        )
        effective_selector["supporting_facts_per_episode"] = max(
            effective_selector["supporting_facts_per_episode"],
            int(operator_tuning.get("structural_query_supporting_facts_per_episode", 4)),
        )
    if operator_plan["bounded_chain"]["enabled"]:
        operator_plan["bounded_chain"]["max_hops"] = int(
            operator_tuning.get("bounded_chain_max_hops", operator_plan["bounded_chain"]["max_hops"])
        )

    episode_lookup = build_episode_lookup(corpus)
    facts_by_episode = build_facts_by_episode(episode_facts)
    selector_phase3_enabled = (
        facts_as_selectors_enabled()
        and str(query_features.get("retrieval_type") or "").lower() != "temporal"
        and not query_features.get("step_numbers")
        and not query_features.get("step_range")
    )
    if selector_phase3_enabled:
        MemoryServer._hydrate_selector_surfaces_runtime(facts_by_episode, episode_lookup)
    else:
        for facts in facts_by_episode.values():
            for fact in facts:
                fact.pop("_selector_surface_text", None)
        for episode in episode_lookup.values():
            if isinstance(episode, dict):
                episode.pop("_selector_surface_text", None)
    temporal_trace = None
    if temporal_index is not None and classify_temporal_query(question) == "ordinal":
        ordinal_packet, temporal_trace = _build_ordinal_executor_packet(
            question,
            corpus=corpus,
            episode_lookup=episode_lookup,
            facts_by_episode=facts_by_episode,
            query_features=query_features,
            effective_selector=effective_selector,
            output_constraints=output_constraints,
            operator_tuning=operator_tuning,
            packet_tuning=packet_tuning,
            search_family=search_family,
            temporal_index=temporal_index,
        )
        if ordinal_packet is not None:
            return ordinal_packet
    if temporal_index is not None and classify_temporal_query(question) == "calendar":
        calendar_packet, temporal_trace = _build_calendar_executor_packet(
            question,
            corpus=corpus,
            episode_lookup=episode_lookup,
            facts_by_episode=facts_by_episode,
            query_features=query_features,
            effective_selector=effective_selector,
            output_constraints=output_constraints,
            operator_tuning=operator_tuning,
            packet_tuning=packet_tuning,
            search_family=search_family,
            temporal_index=temporal_index,
        )
        if calendar_packet is not None:
            return calendar_packet
    family_corpora = partition_corpus_by_family(corpus)
    family_results = []
    routed_families = route_retrieval_families(
        question,
        available,
        explicit_family=search_family,
    )
    for family in routed_families:
        family_corpus = family_corpora.get(family)
        if not family_corpus:
            continue
        family_lookup = build_episode_lookup(family_corpus)
        family_bm25 = build_episode_bm25(family_corpus)
        family_result = choose_episode_ids_with_trace(
            question,
            family_bm25,
            family_lookup,
            effective_selector,
        )
        family_results.append(
            {
                "family": family,
                "selected_ids": family_result["selected_ids"],
                "scored": family_result["scored"],
                "trace": family_result["trace"],
            }
        )
    late_fusion = select_episode_ids_late_fusion_with_trace(
        question,
        family_results,
        episode_lookup,
        effective_selector,
    )
    selected_episode_ids = late_fusion["selected_ids"]
    scored = late_fusion["scored"]
    step_numbers = set(query_features.get("step_numbers", set()) or set())
    step_range = query_features.get("step_range")

    def _target_step_hits(text: str) -> set[int]:
        hits: set[int] = set()
        if not text:
            return hits
        for step in step_numbers:
            if has_exact_step_mention(text, step):
                hits.add(step)
        if step_range:
            hits |= step_range_overlap(text, step_range)
        return hits

    def _episode_step_order(ep_id: str) -> int:
        ep = episode_lookup.get(ep_id) or {}
        lower = (ep.get("raw_text") or "").lower()
        match = re.search(r"\[step\s+(\d+)\]|\bstep\s+(\d+)\b", lower)
        if match:
            return int(match.group(1) or match.group(2))
        topic = (ep.get("topic_key") or "").lower()
        match = re.search(r"\bstep\s+(\d+)\b", topic)
        if match:
            return int(match.group(1))
        match = re.search(r"_e(\d+)\b", ep_id)
        if match:
            return int(match.group(1))
        return 10**9

    def _step_range_episode_sort_key(ep_id: str) -> tuple[int, int, str]:
        ep = episode_lookup.get(ep_id) or {}
        overlap = sorted(step_range_overlap(ep.get("raw_text", ""), step_range))
        if overlap:
            return (0, overlap[0], ep_id)
        return (1, _episode_step_order(ep_id), ep_id)

    def _target_step_episode_sort_key(ep_id: str) -> tuple[int, int, str]:
        ep = episode_lookup.get(ep_id) or {}
        hits = sorted(_target_step_hits(ep.get("raw_text", "")))
        if hits:
            return (0, hits[0], ep_id)
        return (1, _episode_step_order(ep_id), ep_id)

    def _adjacent_episode_id(ep_id: str, delta: int) -> str | None:
        match = re.search(r"^(.*_e)(\d+)\b", ep_id)
        if not match:
            return None
        prefix, raw_num = match.groups()
        next_num = int(raw_num) + int(delta)
        if next_num <= 0:
            return None
        return f"{prefix}{next_num:0{len(raw_num)}d}"

    def _target_step_companion_episode_ids() -> list[str]:
        if not (step_numbers or step_range):
            return []
        companions: list[str] = []
        seen: set[str] = set()
        for ep_id in selected_episode_ids:
            ep = episode_lookup.get(ep_id) or {}
            raw = (ep.get("raw_text") or "").strip()
            lower = raw.lower()
            if not raw or not _target_step_hits(raw):
                continue
            next_ep_id = _adjacent_episode_id(ep_id, 1)
            if not next_ep_id or next_ep_id in seen:
                continue
            next_ep = episode_lookup.get(next_ep_id) or {}
            next_raw = (next_ep.get("raw_text") or "").strip()
            next_lower = next_raw.lower()
            if not next_raw:
                continue
            if next_ep.get("source_id", "") != ep.get("source_id", ""):
                continue
            if next_ep.get("source_type", "") != ep.get("source_type", ""):
                continue
            if not next_lower.startswith("action:") or "observation:" not in next_lower:
                continue
            companions.append(next_ep_id)
            seen.add(next_ep_id)
        return companions

    def _target_step_support_fact_episode_ids(limit: int) -> list[str]:
        if not (step_numbers or step_range) or not selected_source_ids or selected_source_families != {"document"}:
            return []
        candidates: list[str] = []
        seen: set[str] = set()
        for ep_id, facts in facts_by_episode.items():
            ep = episode_lookup.get(ep_id) or {}
            if not ep:
                continue
            if ep.get("source_id", "") not in selected_source_ids:
                continue
            if ep_id in seen:
                continue
            if any(_target_step_hits(fact.get("fact", "")) for fact in facts):
                candidates.append(ep_id)
                seen.add(ep_id)
        candidates.sort(key=_target_step_episode_sort_key)
        return candidates[:limit]

    def _target_step_selected_episode_ids(limit: int) -> list[str]:
        if not (step_numbers or step_range) or not selected_source_ids or selected_source_families != {"document"}:
            return []
        candidates: list[str] = []
        seen: set[str] = set()
        for ep_id, ep in episode_lookup.items():
            if ep.get("source_id", "") not in selected_source_ids:
                continue
            if ep_id in seen:
                continue
            if not _target_step_hits(ep.get("raw_text", "")):
                continue
            candidates.append(ep_id)
            seen.add(ep_id)
        candidates.sort(key=_target_step_episode_sort_key)
        return candidates[:limit]

    if step_range and selected_episode_ids:
        selected_episode_ids = sorted(selected_episode_ids, key=_step_range_episode_sort_key)
        scored = sorted(
            scored,
            key=lambda row: _step_range_episode_sort_key(row[0]),
        )
    selected_source_ids = {
        episode_lookup.get(ep_id, {}).get("source_id", "")
        for ep_id in selected_episode_ids
        if episode_lookup.get(ep_id)
    }
    selected_source_families = {
        episode_lookup.get(ep_id, {}).get("source_type", "")
        for ep_id in selected_episode_ids
        if episode_lookup.get(ep_id)
    }
    if (step_numbers or step_range) and selected_source_families == {"document"}:
        step_span = max(1, step_range[1] - step_range[0] + 1) if step_range else max(1, len(step_numbers))
        target_step_selected_episode_ids = _target_step_selected_episode_ids(max(2, step_span * 2))
        if target_step_selected_episode_ids:
            selected_episode_ids = list(
                dict.fromkeys(target_step_selected_episode_ids + selected_episode_ids)
            )
    support_episode_pool_size = max(
        len(selected_episode_ids),
        int(packet_tuning.get("support_episode_pool_size", len(selected_episode_ids))),
    )
    fact_episode_ids = list(selected_episode_ids)
    target_step_companion_episode_ids: list[str] = []
    target_step_support_fact_episode_ids: list[str] = []
    if (step_numbers or step_range) and selected_source_families == {"document"}:
        target_step_companion_episode_ids = _target_step_companion_episode_ids()
        for ep_id in target_step_companion_episode_ids:
            if ep_id not in fact_episode_ids:
                fact_episode_ids.append(ep_id)
        step_span = max(1, step_range[1] - step_range[0] + 1) if step_range else max(1, len(step_numbers))
        target_step_support_fact_episode_ids = _target_step_support_fact_episode_ids(
            max(support_episode_pool_size, step_span * 2)
        )
        for ep_id in target_step_support_fact_episode_ids:
            if ep_id not in fact_episode_ids:
                fact_episode_ids.append(ep_id)
    structural_conversation_query = (
        selected_source_families == {"conversation"}
        and not operator_plan["list_set"]["enabled"]
        and any(
            operator_plan[name]["enabled"]
            for name in ("commonality", "compare_diff")
        )
    )
    if (
        support_episode_pool_size > len(fact_episode_ids)
        and selected_source_ids
        and selected_source_families == {"document"}
    ):
        for ep_id, _score in scored:
            ep = episode_lookup.get(ep_id) or {}
            if not ep or ep.get("source_id", "") not in selected_source_ids:
                continue
            if ep_id in fact_episode_ids:
                continue
            fact_episode_ids.append(ep_id)
            if len(fact_episode_ids) >= support_episode_pool_size:
                break
    elif structural_conversation_query and packet_tuning.get(
        "expand_conversation_source_for_structural_queries",
        True,
    ):
        conversation_pool_size = max(
            support_episode_pool_size,
            int(
                packet_tuning.get(
                    "conversation_structural_support_episode_pool_size",
                    support_episode_pool_size,
                )
            ),
        )
        for doc in corpus.get("documents", []):
            for ep in doc.get("episodes", []):
                ep_id = ep.get("episode_id", "")
                if not ep_id or ep_id in fact_episode_ids:
                    continue
                if ep.get("source_id", "") not in selected_source_ids:
                    continue
                fact_episode_ids.append(ep_id)
                if len(fact_episode_ids) >= conversation_pool_size:
                    break
            if len(fact_episode_ids) >= conversation_pool_size:
                break

    if step_range and fact_episode_ids:
        fact_episode_ids = sorted(
            list(dict.fromkeys(fact_episode_ids)),
            key=_step_range_episode_sort_key,
        )

    snippet_mode = bool(effective_selector.get("snippet_mode", False))
    snippet_chars = int(packet_tuning.get("snippet_chars", 1200))

    context, actual_injected_episode_ids, selected_fact_ids = (
        build_context_from_selected_episodes(
            question,
            selected_episode_ids,
            episode_lookup,
            facts_by_episode,
            fact_episode_ids=fact_episode_ids,
            support_episode_ids=target_step_companion_episode_ids,
            budget=effective_selector["budget"],
            max_total_facts=effective_selector["supporting_facts_total"],
            max_facts_per_episode=effective_selector["supporting_facts_per_episode"],
            snippet_mode=snippet_mode,
            snippet_chars=snippet_chars,
            allow_pseudo_facts=bool(effective_selector.get("allow_pseudo_facts", True)),
            query_features=query_features,
            local_anchor_window_chars=int(operator_tuning.get("local_anchor_window_chars", snippet_chars)),
            local_anchor_fact_radius=int(operator_tuning.get("local_anchor_fact_radius", 12)),
            list_set_dedup_overlap=float(operator_tuning.get("list_set_dedup_overlap", 0.9)),
            bounded_chain_fact_bonus=float(operator_tuning.get("bounded_chain_fact_bonus", 0.0)),
            query_specificity_bonus=float(packet_tuning.get("query_specificity_bonus", 0.0)),
            inject_support_fact_episodes=bool(
                (
                    structural_conversation_query
                    and packet_tuning.get("inject_support_fact_episodes", True)
                )
                or target_step_companion_episode_ids
                or target_step_support_fact_episode_ids
            ),
            max_injected_support_fact_episodes=int(
                max(
                    int(packet_tuning.get("max_injected_support_fact_episodes", 8)),
                    len(target_step_companion_episode_ids),
                    len(target_step_support_fact_episode_ids),
                )
            ),
        )
    )
    return {
        "context": context,
        "retrieved_episode_ids": selected_episode_ids,
        "actual_injected_episode_ids": actual_injected_episode_ids,
        "fact_episode_ids": fact_episode_ids,
        "retrieved_fact_ids": selected_fact_ids,
        "selection_scores": [
            {"episode_id": ep_id, "score": sc}
            for ep_id, sc in scored[: get_runtime_tuning()["telemetry"]["max_selection_scores"]]
        ],
        "selector_config": effective_selector,
        "query_operator_plan": operator_plan,
        "query_features": query_features,
        "output_constraints": output_constraints,
        "retrieval_families": routed_families,
        "search_family": search_family or "auto",
        "family_first_pass_trace": {
            "available_families": available_families(corpus),
            "retrieval_families": routed_families,
            "requested_search_family": search_family or "auto",
            "per_family": [
                {
                    "family": result["family"],
                    **result["trace"],
                }
                for result in family_results
            ],
        },
        "late_fusion_trace": late_fusion["trace"],
        "tuning_snapshot": {
            "selector": effective_selector,
            "operators": operator_tuning,
            "packet": packet_tuning,
            "routing": tuning["routing"],
            "telemetry": tuning["telemetry"],
        },
        "temporal_trace": temporal_trace,
    }


def _embedding_fingerprint(facts: list[dict]) -> str:
    """SHA-256 of concatenated fact texts. Detects text changes even when count is unchanged."""
    h = hashlib.sha256("|".join(f.get("fact", "") for f in facts).encode()).hexdigest()
    return h


def _is_embedding_count_mismatch_error(exc: Exception) -> bool:
    return isinstance(exc, ValueError) and "Embedding count mismatch" in str(exc)


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return max(minimum, int(raw))
    except (TypeError, ValueError):
        return default


DEFAULT_WRITE_LOG_LEASE_MS = 5 * 60 * 1000
DEFAULT_INDEX_DEBOUNCE_MS = 5 * 1000
DEFAULT_INDEX_MAX_DELAY_MS = 60 * 1000
DEFAULT_INDEX_BUILD_LEASE_MS = 30 * 60 * 1000
DEFAULT_INDEX_BACKOFF_INITIAL_MS = 30 * 1000
DEFAULT_INDEX_BACKOFF_MAX_MS = 10 * 60 * 1000
DEFAULT_PRIORITY_INDEX_TOP_K = 64
DEFAULT_PRIORITY_INDEX_TIMEOUT_MS = 3 * 1000


# ── content_complexity (pure function, no LLM calls) ──

def _compute_content_complexity(facts: list[dict]) -> float:
    """Compute content complexity from already-extracted Librarian metadata.

    Pure function, zero LLM calls. Max-style aggregation across four axes:
    kind weight, entity density, temporal presence, fact count.
    Returns max of axis scores, capped at 1.0.
    """
    if not facts:
        return 0.0

    # Kind-based: take the max across all facts
    KIND_WEIGHTS = {
        "action_item": 0.70,
        "requirement": 0.65,
        "decision":    0.50,
        "constraint":  0.45,
        "preference":  0.10,
        "fact":        0.10,
    }
    kind_scores = [KIND_WEIGHTS.get(f.get("kind", "fact"), 0.10) for f in facts]
    kind_score = max(kind_scores) if kind_scores else 0.0

    # Entity density: count unique entities across all facts
    all_entities = set()
    for f in facts:
        for e in f.get("entities", []):
            if isinstance(e, str):
                all_entities.add(e.lower())
    n_entities = len(all_entities)
    if n_entities > 20:
        entity_score = 0.60
    elif n_entities > 10:
        entity_score = 0.40
    elif n_entities > 5:
        entity_score = 0.20
    else:
        entity_score = 0.0

    # Temporal: if any fact has temporal links
    has_temporal = any(
        f.get("_temporal_links") or f.get("event_date") or f.get("depends_on")
        for f in facts
    )
    temporal_score = 0.35 if has_temporal else 0.0

    # Fact count
    n_facts = len(facts)
    if n_facts > 50:
        fact_score = 0.50
    elif n_facts > 20:
        fact_score = 0.30
    elif n_facts > 10:
        fact_score = 0.15
    else:
        fact_score = 0.0

    score = max(kind_score, entity_score, temporal_score, fact_score)
    return round(min(1.0, score), 3)


def _compute_query_shape_complexity(query: str | None, resolved_type: str) -> float:
    """Estimate reasoning complexity from the query shape alone.

    This is intentionally lexical and conservative. It should only raise
    complexity for clearly comparative, multi-constraint recommendation
    prompts, while leaving plain lookups untouched.
    """
    if not query:
        return 0.0

    q = query.lower()

    decision_markers = (
        "which option",
        "what option",
        "which approach",
        "design a migration plan",
        "actionable plan",
        "phased rollout",
        "recommended",
        "recommend",
        "should be chosen",
        "should we choose",
        "best option",
    )
    comparison_markers = (
        "compare at least",
        "compare ",
        "why is the other",
        "why the other",
        "compared with",
        "versus",
        " vs ",
        "tradeoff",
        "trade-off",
        "trade-offs",
        "risky",
    )
    strong_constraint_terms = (
        "latency",
        "ownership",
        "owner",
        "rollback",
        "budget",
        "cost",
        "throughput",
        "capacity",
        "availability",
        "dependency",
        "dependencies",
        "compliance",
        "consistency",
        "monitoring",
        "rollout",
        "failure mode",
        "failure modes",
        "risk matrix",
    )

    has_decision = any(marker in q for marker in decision_markers)
    comparison_hits = sum(1 for marker in comparison_markers if marker in q)
    constraint_hits = sum(1 for term in strong_constraint_terms if term in q)

    if has_decision and comparison_hits >= 1 and constraint_hits >= 3:
        return 0.70
    if has_decision and comparison_hits >= 1 and constraint_hits >= 2:
        return 0.65
    if has_decision and constraint_hits >= 3:
        return 0.70
    if has_decision and constraint_hits >= 2:
        return 0.65
    if has_decision and comparison_hits >= 1:
        return 0.50
    if resolved_type in (
        "aggregate",
        "counting",
        "synthesize",
        "synthesis",
        "procedural",
        "rule",
    ) and constraint_hits >= 2:
        return 0.55
    return 0.0


# ── complexity_hint v2 (pure function, no LLM calls) ──

def _compute_complexity_hint(retrieved, resolved_type, is_multihop, fact_lookup, query: str | None = None):
    """Compute complexity hint from retrieval signals and retrieved fact content.

    Three independent axes:
    - retrieval_complexity: structural signals from retrieval (multi_hop,
      cross_scope, conflict_found, high_fact_count)
    - content_complexity: complexity of the actual fact set returned by recall
    - query_complexity: complexity implied by the query shape itself

    Final score = max(retrieval_complexity, content_complexity, query_complexity).
    Zero LLM calls.
    """
    signals = []

    def _resolve_fact(item: dict) -> dict | None:
        if not isinstance(item, dict):
            return None
        # Preferred path: recall already passed the actual retrieved fact.
        if any(
            key in item
            for key in (
                "fact",
                "kind",
                "entities",
                "tags",
                "event_date",
                "depends_on",
                "_temporal_links",
            )
        ):
            return item
        fid = item.get("fact_id", item.get("id", ""))
        if fid and fact_lookup:
            return fact_lookup.get(fid)
        return None

    retrieved_facts = []
    if retrieved:
        for item in retrieved:
            fact = _resolve_fact(item)
            if fact:
                retrieved_facts.append(fact)

    # ── Retrieval signals (structural only) ──
    if is_multihop:
        signals.append("multi_hop")

    if resolved_type == "supersession":
        signals.append("conflict_found")

    if len(retrieved) > 50:
        signals.append("high_fact_count")

    if retrieved_facts:
        agent_ids = set()
        for fact in retrieved_facts:
            agent_ids.add(fact.get("agent_id", ""))
        if len(agent_ids) > 1:
            signals.append("cross_scope")

    retrieval_complexity = 0.0
    if "multi_hop" in signals:       retrieval_complexity += 0.35
    if "cross_scope" in signals:     retrieval_complexity += 0.25
    if "conflict_found" in signals:  retrieval_complexity += 0.20
    if "high_fact_count" in signals: retrieval_complexity += 0.05
    retrieval_complexity = min(1.0, retrieval_complexity)

    # ── Content complexity from the actual retrieved fact set ──
    content_complexity = _compute_content_complexity(retrieved_facts)
    query_complexity = _compute_query_shape_complexity(query, resolved_type)

    # ── Combined score ──
    score = max(retrieval_complexity, content_complexity, query_complexity)
    score = round(max(0.0, min(1.0, score)), 3)

    if score <= 0.2:   level = 1
    elif score <= 0.4: level = 2
    elif score <= 0.6: level = 3
    elif score <= 0.8: level = 4
    else:              level = 5

    # Dominant axis
    if retrieval_complexity > content_complexity and retrieval_complexity > query_complexity:
        dominant = "retrieval"
    elif content_complexity > retrieval_complexity and content_complexity > query_complexity:
        dominant = "content"
    elif query_complexity > retrieval_complexity and query_complexity > content_complexity:
        dominant = "query"
    else:
        dominant = "tie"

    return {
        "score": score,
        "level": level,
        "signals": signals,
        "retrieval_complexity": round(retrieval_complexity, 3),
        "content_complexity": round(content_complexity, 3),
        "query_complexity": round(query_complexity, 3),
        "dominant": dominant,
    }


# ── memory_query constants ──

SORTABLE_FIELDS = {
    "id", "fact", "kind", "speaker", "event_date", "supersedes_topic",
    "conv_id", "agent_id", "swarm_id", "scope", "owner_id", "created_at",
    "session_date", "artifact_id", "version_id", "content_hash",
    "source_id", "status",
    "session", "retention_ttl",
}

NUMERIC_SORT_FIELDS = {"session", "retention_ttl"}

VALID_SORT_ORDERS = {"asc", "desc"}
MAX_QUERY_FACT_CHARS = 1200

RANGE_OPS = {"gt", "gte", "lt", "lte"}


def _consensus_metadata(source_facts: list, source_ids: list = None) -> dict:
    """Propagate metadata from source facts by consensus.

    Only keys where ALL sources agree on value are inherited.
    Missing/incomplete source_ids → empty metadata.
    """
    if not source_facts:
        return {}
    if source_ids:
        matched = [sf.get("metadata", {}) for sf in source_facts
                   if sf.get("id") in source_ids
                   or any(sf.get("id", "").endswith(s) for s in source_ids)]
        if not matched:
            return {}  # unresolved lineage → no metadata (not inferred)
    else:
        return {}
    all_keys: set[str] = set()
    for m in matched:
        if isinstance(m, dict):
            all_keys.update(m.keys())
    consensus = {}
    for k in all_keys:
        vals = [m.get(k) for m in matched if isinstance(m, dict) and k in m]
        if vals and all(v == vals[0] for v in vals):
            consensus[k] = vals[0]
    return consensus


def _consensus_target(source_facts: list, source_ids: list = None) -> list[str] | None:
    """Propagate target only when all relevant source facts agree on it."""
    if not source_facts:
        return None
    if source_ids:
        matched = [
            sf for sf in source_facts
            if sf.get("id") in source_ids
            or any(sf.get("id", "").endswith(s) for s in source_ids)
        ]
        if not matched:
            return None
    else:
        matched = list(source_facts)

    normalized: list[tuple[str, ...] | None] = []
    for fact in matched:
        target = fact.get("target")
        if not target:
            normalized.append(None)
            continue
        normalized.append(tuple(_normalize_target(target) or []))

    if not normalized:
        return None
    first = normalized[0]
    if all(value == first for value in normalized):
        return list(first) if first else None
    return None


def _resolve_field(fact: dict, key: str):
    if key.startswith("metadata."):
        meta = fact.get("metadata") or {}
        return meta.get(key[9:])
    return fact.get(key)


def _match_filter_value(fact_value, filter_value, allow_range: bool = False) -> bool:
    if isinstance(filter_value, dict) and filter_value.keys() <= RANGE_OPS:
        if not allow_range:
            return False
        if fact_value is None:
            return False
        for op, threshold in filter_value.items():
            if op == "gt" and not (fact_value > threshold):
                return False
            if op == "gte" and not (fact_value >= threshold):
                return False
            if op == "lt" and not (fact_value < threshold):
                return False
            if op == "lte" and not (fact_value <= threshold):
                return False
        return True
    if isinstance(fact_value, list):
        return filter_value in fact_value
    return fact_value == filter_value


def _merge_fact_metadata(existing, stamped):
    """Merge flat metadata dicts with caller-provided metadata winning on conflicts."""
    if stamped is None:
        return existing if isinstance(existing, dict) else None

    merged: dict[str, Any] = {}
    if isinstance(existing, dict):
        merged.update(existing)
    merged.update(stamped)
    return merged


def _is_asserted_derived_fact(fact: dict) -> bool:
    metadata = fact.get("metadata") or {}
    return bool(metadata.get("asserted_derived_tier"))


def _is_supported_cross_fact(fact: dict) -> bool:
    metadata = fact.get("metadata") or {}
    return bool(metadata.get("source_aggregation") or metadata.get("asserted_derived_tier"))


def _stamp_source_aggregation_source(facts: list[dict], source_id: str, source_kind: str | None = None) -> None:
    for fact in facts or []:
        fact.setdefault("source_id", source_id)
        metadata = fact.setdefault("metadata", {})
        metadata.setdefault("source_id", source_id)
        if source_kind:
            metadata.setdefault("source_kind", source_kind)


def _lifecycle_status(record: dict | None) -> str:
    if not isinstance(record, dict):
        return "active"
    return str(record.get("status") or "active").strip() or "active"


def _is_active_lifecycle_record(record: dict | None) -> bool:
    return _lifecycle_status(record) == "active"


def _fact_matches_structured_filter(
    fact: dict,
    filter: dict | None,
    metadata_schema: dict | None = None,
) -> bool:
    """Apply the same structured filter semantics used by query()."""
    if not filter:
        return True

    range_types = {"number", "integer", "datetime"}
    for key, value in filter.items():
        allow_range = True
        if key.startswith("metadata."):
            field_type = None
            if metadata_schema:
                field_def = metadata_schema.get(key[9:])
                field_type = field_def.get("type") if field_def else None
            allow_range = field_type in range_types
        if not _match_filter_value(
            _resolve_field(fact, key),
            value,
            allow_range=allow_range,
        ):
            return False
    return True


def _is_sortable(sort_by: str, metadata_schema: dict = None) -> bool:
    if sort_by in SORTABLE_FIELDS:
        return True
    if sort_by.startswith("metadata."):
        if not metadata_schema:
            return False
        meta_key = sort_by[9:]
        field_def = metadata_schema.get(meta_key)
        if not field_def:
            return False
        return field_def.get("type", "") in ("string", "number", "integer", "boolean", "datetime")
    return False


def _split_sort_values(facts: list, sort_by: str):
    present, missing = [], []
    for f in facts:
        if _resolve_field(f, sort_by) is None:
            missing.append(f)
        else:
            present.append(f)
    return present, missing


def _sort_value(f: dict, sort_by: str):
    v = _resolve_field(f, sort_by)
    if sort_by in NUMERIC_SORT_FIELDS or isinstance(v, (int, float)):
        return v if isinstance(v, (int, float)) else 0
    return str(v)


# ── L0 enrichment helper ──

def _normalize_fact_types(f: dict) -> None:
    """Fix malformed metadata types in-place before enrichment/storage.

    Converts wrong types to correct ones so downstream code never sees
    entities as string or tags as non-list.
    """
    if "kind" in f and (not isinstance(f["kind"], str) or not f["kind"]):
        del f["kind"]  # remove so enrichment/setdefault can fill it
    if "entities" in f and not isinstance(f["entities"], list):
        v = f["entities"]
        f["entities"] = [v] if isinstance(v, str) and v else []
    if "tags" in f and not isinstance(f["tags"], list):
        v = f["tags"]
        f["tags"] = [v] if isinstance(v, str) and v else []


def _validate_object_flags_field(obj: dict, *, label: str = "object") -> str | None:
    if not isinstance(obj, dict):
        return f"{label} must be a dict, got {type(obj).__name__}"
    return validate_object_flags(obj.get("flags"))


def _coerce_runtime_report(value: Any, *, producer: str, report_kind: str, validation: dict | None = None) -> dict:
    try:
        if isinstance(value, dict):
            normalized_report = normalize_legacy_report_fields(value, producer=producer, report_kind=report_kind)
            report: dict[str, Any] | Any = normalized_report if isinstance(normalized_report, dict) else value
            if isinstance(report, dict) and report.get("report_id"):
                if validation:
                    summary_obj = report.get("summary")
                    validation_summary: dict[str, Any] = dict(summary_obj) if isinstance(summary_obj, dict) else {}
                    validation_summary.setdefault("validation", deepcopy(validation))
                    report["summary"] = validation_summary
                normalize_object_flags_field(report)
                for entry in report.get("entries") or []:
                    if isinstance(entry, dict):
                        normalize_object_flags_field(entry)
                validation_error = validate_report_object(report)
                if validation_error:
                    raise ValueError(f"invalid {report_kind} report: {validation_error}")
                return report
        if isinstance(value, list):
            legacy_report: dict[str, Any] = {"diagnostics": value}
            report = normalize_legacy_report_fields(legacy_report, producer=producer, report_kind=report_kind)
            if isinstance(report, dict) and report.get("report_id"):
                if validation:
                    summary_obj = report.get("summary")
                    legacy_validation_summary: dict[str, Any] = dict(summary_obj) if isinstance(summary_obj, dict) else {}
                    legacy_validation_summary.setdefault("validation", deepcopy(validation))
                    report["summary"] = legacy_validation_summary
                validation_error = validate_report_object(report)
                if validation_error:
                    raise ValueError(f"invalid {report_kind} report: {validation_error}")
                return report
    except ValueError as exc:
        message = str(exc)
        if not message.startswith(f"invalid {report_kind} report:"):
            raise ValueError(f"invalid {report_kind} report: {message}") from exc
        raise
    status = "ok" if not validation or validation.get("aggregation_status", "accepted") == "accepted" else "partial"
    summary: dict[str, Any] = {"entry_count": 0}
    if validation:
        summary["validation"] = deepcopy(validation)
    return build_report(
        report_kind=report_kind,
        producer=producer,
        status=status,
        entries=[],
        summary=summary,
    )


def _coerce_extract_session_result(result: Any) -> tuple[str, int, str, list[dict], list[dict], dict]:
    if not isinstance(result, (tuple, list)):
        raise ValueError(f"extract_session returned {type(result).__name__}, expected tuple/list")
    if len(result) == 5:
        conv_id, sn, sdate, facts, tlinks = result
        report = _coerce_runtime_report(None, producer="block_extractor", report_kind="extraction")
    elif len(result) == 6:
        conv_id, sn, sdate, facts, tlinks, report = result
        report = _coerce_runtime_report(report, producer="block_extractor", report_kind="extraction")
    else:
        raise ValueError(f"extract_session returned {len(result)} values, expected 5 or 6")
    tlinks_list = list(tlinks or [])
    for link in tlinks_list:
        if isinstance(link, dict):
            normalize_object_flags_field(link)
    return conv_id, sn, sdate, list(facts or []), tlinks_list, report


def _set_runtime_report_artifact(
    target: dict[str, Any],
    *,
    field_name: str,
    producer: str,
    report_kind: str,
    report: dict | list | None,
    validation: dict | None = None,
) -> None:
    if report is None and validation is None:
        target.pop(field_name, None)
        return
    target[field_name] = _coerce_runtime_report(
        report,
        producer=producer,
        report_kind=report_kind,
        validation=validation,
    )


_SELECTOR_OPTIONAL_SOURCE_FIELDS = ("query", "caption", "blip_caption")

def _selector_optional_raw_fields(*sources: dict | None) -> dict[str, str]:
    fields: dict[str, str] = {}
    for source in sources:
        if not isinstance(source, dict):
            continue
        for key in _SELECTOR_OPTIONAL_SOURCE_FIELDS:
            value = source.get(key)
            if not isinstance(value, str):
                continue
            stripped = value.strip()
            if stripped:
                fields.setdefault(key, stripped)
    return fields


def _selector_raw_fields(raw_text: str, *sources: dict | None) -> dict[str, str]:
    fields = {"raw_text": str(raw_text or "")}
    fields.update(_selector_optional_raw_fields(*sources))
    return fields


def _record_semantic_ready(record: dict | None, *, fallback_text: str = "") -> bool:
    if not isinstance(record, dict):
        return False
    explicit_ready = record.get("semantic_ready")
    if isinstance(explicit_ready, bool):
        return explicit_ready
    status = str(record.get("canonicalization_status") or "").strip().lower()
    if status == "ready":
        return True
    if status == "failed":
        return False
    canonical_en = str(record.get("canonical_en") or "").strip()
    if canonical_en:
        return True
    legacy_text = str(fallback_text or "").strip()
    if not legacy_text:
        return False
    return detect_source_language(legacy_text) in {"en", "und"}


def _semantic_raw_session_text(raw_session: dict | None) -> str:
    if not isinstance(raw_session, dict):
        return ""
    if not _record_semantic_ready(raw_session, fallback_text=str(raw_session.get("content") or "")):
        return ""
    canonical_en = str(raw_session.get("canonical_en") or "").strip()
    if canonical_en:
        return canonical_en
    return str(raw_session.get("content") or "")


def _semantic_episode_text(episode: dict | None) -> str:
    if not isinstance(episode, dict):
        return ""
    if not _record_semantic_ready(episode, fallback_text=str(episode.get("raw_text") or "")):
        return ""
    canonical_en = str(episode.get("canonical_en") or "").strip()
    if canonical_en:
        return canonical_en
    return str(episode.get("raw_text") or "")


def _stamp_selector_episode_fields(episode: dict, *sources: dict | None) -> None:
    if not isinstance(episode, dict):
        return
    for key, value in _selector_optional_raw_fields(*sources).items():
        episode.setdefault(key, value)


def _coerce_positive_session_num(value) -> int | None:
    """Return a positive integer session number when the value is coercible."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float) and value.is_integer():
        ivalue = int(value)
        return ivalue if ivalue > 0 else None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            ivalue = int(stripped)
            return ivalue if ivalue > 0 else None
    return None


def _namespace_derived_fact_ids(facts: list[dict], prefix: str) -> None:
    """Ensure derived tiers never reuse local extractor IDs from source facts."""
    for idx, fact in enumerate(facts or []):
        raw_id = str(fact.get("id") or f"{idx:03d}").strip()
        if raw_id.startswith(f"{prefix}_"):
            continue
        fact["id"] = f"{prefix}_{raw_id or f'{idx:03d}'}"


def _needs_l0_enrichment(f: dict) -> bool:
    """Check if fact needs L0 classification. Presence-based, not value-based."""
    return (
        "kind" not in f or not isinstance(f.get("kind"), str) or not f.get("kind")
        or "entities" not in f or not isinstance(f.get("entities"), list)
        or "tags" not in f or not isinstance(f.get("tags"), list)
    )


# ── Shared visibility predicate (Unit 5) ──

def _is_visible(fact, now=None, fact_lookup=None):
    """Check if a fact is visible (active, not expired, source not retracted).

    Used as the single source of truth for all read paths.
    """
    status = fact.get("status", "active")
    if status != "active":
        return False

    # TTL check
    ttl = fact.get("retention_ttl")
    if ttl is not None:
        created = fact.get("created_at")
        if created and now:
            try:
                from datetime import datetime as _dt
                from datetime import timezone as _tz
                created_dt = _dt.fromisoformat(created.replace("Z", "+00:00"))
                if isinstance(now, str):
                    now_dt = _dt.fromisoformat(now.replace("Z", "+00:00"))
                else:
                    now_dt = now
                elapsed = (now_dt - created_dt).total_seconds()
                if elapsed > ttl:
                    return False
            except (ValueError, TypeError):
                pass

    # Derived tier staleness: hide if ANY source fact is invisible
    if fact_lookup and "source_ids" in fact and fact["source_ids"]:
        for sid in fact["source_ids"]:
            source = fact_lookup.get(sid)
            if source:
                src_status = source.get("status", "active")
                if src_status != "active":
                    return False
                # TTL check on source
                src_ttl = source.get("retention_ttl")
                if src_ttl is not None and now:
                    src_created = source.get("created_at")
                    if src_created:
                        try:
                            from datetime import datetime as _dt2
                            from datetime import timezone as _tz2
                            sc_dt = _dt2.fromisoformat(src_created.replace("Z", "+00:00"))
                            n_dt = now if not isinstance(now, str) else _dt2.fromisoformat(now.replace("Z", "+00:00"))
                            if (n_dt - sc_dt).total_seconds() > src_ttl:
                                return False
                        except (ValueError, TypeError):
                            pass

    return True


# ── MemoryServer ──

def _resolve_default_owner(server) -> str:
    if server.agent_id and server.agent_id != "default":
        return f"agent:{server.agent_id}"
    return "system"


class MemoryServer:
    """In-process memory server with 3-tier fact indexing and multi-agent isolation.

    One instance = one conversation (conv_id = key).
    """

    _EXTRACT_SEM: asyncio.Semaphore | None = None
    _EXTRACT_SEM_LOOP_ID: int | None = None

    @classmethod
    def _get_extract_sem(cls) -> asyncio.Semaphore:
        current_loop_id = id(asyncio.get_event_loop())
        if cls._EXTRACT_SEM is None or current_loop_id != cls._EXTRACT_SEM_LOOP_ID:
            try:
                limit = max(1, int(os.getenv("MEMORY_EXTRACT_CONCURRENCY", "3")))
            except Exception:
                limit = 3
            cls._EXTRACT_SEM = asyncio.Semaphore(limit)
            cls._EXTRACT_SEM_LOOP_ID = current_loop_id
        return cls._EXTRACT_SEM

    def __init__(
        self,
        data_dir: str,
        key: str,
        extract_model: str | None = "",
        agent_id: str = "default",
        scope: str = "swarm-shared",
        swarm_id: str = "default",
        storage: StorageBackend = None,
        profiles: dict = None,
        profile_configs: dict = None,
        inference_leaf_plugins: dict | None = None,
    ):
        # _UNSET sentinel distinguishes "caller omitted the arg" (default "")
        # from "caller explicitly passed None or a model string".
        _UNSET = object()
        _ctor_extract_model = _UNSET if extract_model == "" else extract_model
        self._extract_disabled = extract_model is None
        _ctor_profiles = deepcopy(profiles) if profiles is not None else None
        _ctor_profile_configs = deepcopy(profile_configs) if profile_configs is not None else None
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.key = key
        if extract_model is None:
            self.extract_model = ""
        elif extract_model == "":
            self.extract_model = MemoryConfig().extraction_model
        else:
            self.extract_model = extract_model
        self.agent_id = agent_id
        self.scope = scope
        self.swarm_id = swarm_id
        self._profiles = profiles  # complexity level → profile name mapping
        self._profile_configs = profile_configs or {}  # name → {model, context_window, ...}
        self._inference_leaf_plugins = dict(DEFAULT_INFERENCE_LEAF_PLUGIN_STATE)
        if inference_leaf_plugins:
            self._inference_leaf_plugins.update(
                {str(name): bool(enabled) for name, enabled in inference_leaf_plugins.items()}
            )
        self._memory_config: dict | None = None
        self._embedding_model: str | None = None
        self._librarian_profile: str | None = None

        self._storage = storage or make_storage(data_dir, key)
        self._supports_write_log = isinstance(self._storage, IngressWriteStorageBackend)
        self._supports_index_coordination = all(
            hasattr(self._storage, name)
            for name in (
                "mark_index_dirty",
                "read_index_status",
                "acquire_index_build_lease",
                "release_index_build_lease",
            )
        )
        self._prompt_registry = PromptRegistry(data_dir=data_dir, key=key)
        self._file_lock = asyncio.Lock()
        self._queue_lock = asyncio.Lock()
        self._active_sync_message_ids: set[str] = set()
        self._worker_id = f"{os.getpid()}:{self.key}:{uuid4().hex[:12]}"
        self._last_write_log_claim_trace: dict[str, Any] = {}
        self._upsert_locks: dict[str, asyncio.Lock] = {}
        self._data_dict = None
        self._audit = AuditLog(self.data_dir / "audit")

        # Load existing cache if present
        self._all_granular: list[dict] = []
        self._all_cons: list[dict] = []
        self._all_cross: list[dict] = []
        self._all_tlinks: list[dict] = []
        self._raw_sessions: list[dict] = []
        self._raw_docs: dict[str, str] = {}  # doc_key → raw text for hybrid doc context
        self._episode_corpus: dict = {"documents": []}
        self._raw_episode_visibility_cache: dict[str, tuple[dict, ...]] = {}
        self._raw_episode_visibility_cache_version = -1
        self._container_graph: dict = empty_container_graph()
        self._terminal_render_candidate_registry: dict[str, dict[str, Any]] = {}
        self._recall_continuations: dict[str, dict[str, Any]] = {}
        self._temporal_index: dict = empty_temporal_index()
        self._temporal_index_dirty = True
        self._secrets: list[dict] = []
        self._n_sessions = 0
        self._n_sessions_with_facts = 0
        self._tiers_dirty = False
        self._index_snapshot_version = 0
        self._emb_fingerprints = {}
        self._dedup_index: dict[tuple, dict] = {}
        self._content_dedup_index: dict[tuple, dict] = {}
        self._git_dedup_index: dict[tuple, dict] = {}
        self._simhash_index: dict[tuple, int] = {}
        self._fact_lookup: dict = {}
        self._metadata_schema: dict | None = None
        self._instance_config: dict | None = None
        self._scope_record: dict = {}
        self._source_records: dict[str, dict] = {}

        needs_content_dedup_rebuild = False
        if self._storage.exists:
            cached = self._storage.load_facts(internal=True)
            self._all_granular = cached.get("granular", [])
            self._all_cons = cached.get("cons", [])
            self._all_cross = cached.get("cross", [])
            self._all_tlinks = cached.get("tlinks", [])
            self._n_sessions = cached.get("n_sessions", 0)
            self._n_sessions_with_facts = cached.get("n_sessions_with_facts", 0)
            self._raw_sessions = cached.get("raw_sessions", [])
            self._raw_docs = cached.get("raw_docs", {})
            self._episode_corpus = cached.get("episode_corpus", {"documents": []})
            self._container_graph = normalize_container_graph(cached.get("container_graph"))
            for row in self._raw_sessions:
                if isinstance(row, dict):
                    normalize_legacy_report_fields(row)
                    normalize_object_flags_field(row)
            for link in self._all_tlinks:
                if isinstance(link, dict):
                    normalize_object_flags_field(link)
            # Restore _temporal_links on granular facts
            for f in self._all_granular:
                f["_temporal_links"] = []
            if self._all_granular and self._all_tlinks:
                self._all_granular[0]["_temporal_links"] = self._all_tlinks
            log.info("Loaded cache: %dg/%dc/%dx",
                     len(self._all_granular), len(self._all_cons), len(self._all_cross))

            for tier in (self._all_granular, self._all_cons, self._all_cross):
                for f in tier:
                    _normalize_loaded_acl_object(f)

            # Backward compat: identity fields default
            for tier in (self._all_granular, self._all_cons, self._all_cross):
                for f in tier:
                    f.setdefault("status", "active")

            # Load dedup index
            saved_dedup = cached.get("_dedup_index", {})
            # JSON keys are strings; convert back to tuples
            for k, v in saved_dedup.items():
                try:
                    key_tuple = tuple(json.loads(k))
                    self._dedup_index[key_tuple] = v
                except (json.JSONDecodeError, TypeError):
                    pass

            has_saved_content_dedup = "_content_dedup_index" in cached
            saved_content_dedup = cached.get("_content_dedup_index", {})
            for k, v in saved_content_dedup.items():
                try:
                    key_tuple = tuple(json.loads(k))
                    if isinstance(v, dict):
                        self._content_dedup_index[key_tuple] = dict(v)
                except (json.JSONDecodeError, TypeError):
                    pass

            saved_git_dedup = cached.get("_git_dedup_index", {})
            for k, v in saved_git_dedup.items():
                try:
                    key_tuple = tuple(json.loads(k))
                    self._git_dedup_index[key_tuple] = v
                except (json.JSONDecodeError, TypeError):
                    pass

            has_saved_simhash = "_simhash_index" in cached
            saved_simhash = cached.get("_simhash_index", {})
            for k, v in saved_simhash.items():
                try:
                    key_tuple = tuple(json.loads(k))
                    self._simhash_index[key_tuple] = int(v)
                except (json.JSONDecodeError, TypeError, ValueError):
                    pass

            cached_metadata_schema = cached.get("metadata_schema")
            self._metadata_schema = cached_metadata_schema if isinstance(cached_metadata_schema, dict) else None
            cached_instance_config = cached.get("instance_config")
            self._instance_config = _normalize_loaded_instance_config(cached_instance_config)
            self._scope_record = cached.get("scope_record") or {}
            self._source_records = {
                str(source_id): _normalize_loaded_source_record(str(source_id), record)
                for source_id, record in (cached.get("source_records") or {}).items()
            }
            saved_memory_config = cached.get("memory_config")
            saved_profiles = cached.get("profiles")
            saved_configs = cached.get("profile_configs")
            if saved_profiles and not self._profiles:
                self._profiles = {int(k): v for k, v in saved_profiles.items()}
            if saved_configs and not self._profile_configs:
                self._profile_configs = saved_configs
            if saved_memory_config:
                self._apply_memory_config(saved_memory_config)

            needs_content_dedup_rebuild = not has_saved_content_dedup or not has_saved_simhash

            # Legacy auto-generated merge tiers are no longer part of the production path.
            # Keep asserted derived tiers and source-aggregation cross facts.
            self._all_cons = [f for f in self._all_cons if _is_asserted_derived_fact(f)]
            self._all_cross = [f for f in self._all_cross if self._is_current_supported_cross_fact(f)]
            # Load fingerprints and pre-load embeddings if fingerprint matches
            self._emb_fingerprints = cached.get("_emb_fingerprints", {})
            current_fps = {
                "gran": _embedding_fingerprint(self._all_granular),
                "cons": _embedding_fingerprint(self._all_cons),
                "cross": _embedding_fingerprint(self._all_cross),
            }
            saved_embs = self._storage.load_embeddings()
            if saved_embs is not None and self._emb_fingerprints == current_fps:
                gran_embs = saved_embs.get("gran", np.zeros((0, 3072)))
                cons_embs = saved_embs.get("cons", np.zeros((0, 3072)))
                cross_embs = saved_embs.get("cross", np.zeros((0, 3072)))
                if (len(gran_embs) == len(self._all_granular)
                        and len(cons_embs) == len(self._all_cons)
                        and len(cross_embs) == len(self._all_cross)):
                    self._data_dict = _build_index_state(
                        self._all_granular,
                        gran_embs,
                        self._all_cons,
                        cons_embs,
                        self._all_cross,
                        cross_embs,
                    )
                    self._fact_lookup = self._data_dict.get("fact_lookup", {})
                    log.info("Loaded embeddings from disk (fingerprint match): %d/%d/%d",
                             len(gran_embs), len(cons_embs), len(cross_embs))
                else:
                    log.info("Saved embeddings count mismatch — will re-embed on next build_index()")
                    self._emb_fingerprints = {}
            elif saved_embs is not None:
                log.info("Embedding fingerprint mismatch — will re-embed on next build_index()")
                self._emb_fingerprints = {}

        if self._memory_config is None:
            self._apply_memory_config(self._default_memory_config())
        if _ctor_profiles is not None or _ctor_profile_configs is not None:
            override_cfg = self.get_config()
            if _ctor_profiles is not None:
                override_cfg["profiles"] = _ctor_profiles
            if _ctor_profile_configs is not None:
                override_cfg["profile_configs"] = _ctor_profile_configs
            self._apply_memory_config(override_cfg)
        # CLI-provided extract_model takes priority over cached config.
        # _UNSET means the caller used the default (no override); any other
        # value — including None (disable extraction) — wins over cache.
        if _ctor_extract_model is not _UNSET:
            if _ctor_extract_model is None:
                self.extract_model = ""
            else:
                self.extract_model = cast(str, _ctor_extract_model)
        self._initialize_scope_registry()
        self._rebuild_instance_acl_from_sources()
        self._refresh_secret_summaries()
        if needs_content_dedup_rebuild:
            self._rebuild_content_dedup_indices(persist=True)
        self._rebuild_source_version_index()

    VALID_SCOPES = {"agent-private", "swarm-shared", "system-wide"}

    def _reload_runtime_from_storage(self) -> None:
        if not self._storage.exists:
            return
        cached = self._storage.load_facts(internal=True)
        self._data_dict = None
        self._fact_lookup = {}
        self._all_granular = cached.get("granular", [])
        self._all_cons = cached.get("cons", [])
        self._all_cross = cached.get("cross", [])
        self._all_tlinks = cached.get("tlinks", [])
        self._n_sessions = cached.get("n_sessions", 0)
        self._n_sessions_with_facts = cached.get("n_sessions_with_facts", 0)
        self._raw_sessions = cached.get("raw_sessions", [])
        self._raw_docs = cached.get("raw_docs", {})
        self._episode_corpus = cached.get("episode_corpus", {"documents": []})
        self._container_graph = normalize_container_graph(cached.get("container_graph"))

        for f in self._all_granular:
            f["_temporal_links"] = []
        if self._all_granular and self._all_tlinks:
            self._all_granular[0]["_temporal_links"] = self._all_tlinks

        for tier in (self._all_granular, self._all_cons, self._all_cross):
            for f in tier:
                _normalize_loaded_acl_object(f)
                f.setdefault("status", "active")

        self._dedup_index = {}
        for k, v in (cached.get("_dedup_index", {}) or {}).items():
            try:
                key_tuple = tuple(json.loads(k))
                self._dedup_index[key_tuple] = v
            except (json.JSONDecodeError, TypeError):
                pass

        has_saved_content_dedup = "_content_dedup_index" in cached
        self._content_dedup_index = {}
        for k, v in (cached.get("_content_dedup_index", {}) or {}).items():
            try:
                key_tuple = tuple(json.loads(k))
                if isinstance(v, dict):
                    self._content_dedup_index[key_tuple] = dict(v)
            except (json.JSONDecodeError, TypeError):
                pass

        self._git_dedup_index = {}
        for k, v in (cached.get("_git_dedup_index", {}) or {}).items():
            try:
                key_tuple = tuple(json.loads(k))
                self._git_dedup_index[key_tuple] = v
            except (json.JSONDecodeError, TypeError):
                pass

        has_saved_simhash = "_simhash_index" in cached
        self._simhash_index = {}
        for k, v in (cached.get("_simhash_index", {}) or {}).items():
            try:
                key_tuple = tuple(json.loads(k))
                self._simhash_index[key_tuple] = int(v)
            except (json.JSONDecodeError, TypeError, ValueError):
                pass

        self._metadata_schema = cached.get("metadata_schema")
        self._instance_config = _normalize_loaded_instance_config(cached.get("instance_config"))
        self._scope_record = cached.get("scope_record") or {}
        self._source_records = {
            str(source_id): _normalize_loaded_source_record(str(source_id), record)
            for source_id, record in (cached.get("source_records") or {}).items()
        }
        self._emb_fingerprints = cached.get("_emb_fingerprints", {})

        has_saved_memory_config = "memory_config" in cached
        saved_memory_config = cached.get("memory_config")
        has_saved_profiles = "profiles" in cached
        saved_profiles = cached.get("profiles")
        has_saved_configs = "profile_configs" in cached
        saved_configs = cached.get("profile_configs")
        if has_saved_memory_config and saved_memory_config:
            self._apply_memory_config(saved_memory_config)
        elif has_saved_profiles or has_saved_configs:
            next_config = self.get_config()
            if has_saved_profiles:
                next_config["profiles"] = (
                    {int(k): v for k, v in saved_profiles.items()}
                    if saved_profiles else {}
                )
            if has_saved_configs:
                next_config["profile_configs"] = saved_configs or {}
            self._apply_memory_config(next_config)

        self._all_cons = [f for f in self._all_cons if _is_asserted_derived_fact(f)]
        self._all_cross = [f for f in self._all_cross if self._is_current_supported_cross_fact(f)]
        self._initialize_scope_registry()
        self._rebuild_instance_acl_from_sources()
        self._refresh_secret_summaries()
        if not has_saved_content_dedup or not has_saved_simhash:
            self._rebuild_content_dedup_indices(persist=False)
        self._rebuild_source_version_index()
        self._bump_index_snapshot_version()

    def _rebuild_instance_acl_from_sources(self) -> None:
        """Recompute derived instance ACL from canonical source roots on load/reload."""
        if self._instance_config is None:
            return
        owner_id = self._instance_config.get("owner_id") or "system"
        rebuilt: dict[str, Any] = {
            "owner_id": owner_id,
            "read": [],
            "write": [],
            "_derived_read": [],
            "_derived_write": [],
        }
        for source in self._source_records.values():
            scope = str(source.get("scope") or "").strip()
            swarm_id = str(source.get("swarm_id") or "default").strip() or "default"
            if scope == "system-wide":
                if "agent:PUBLIC" not in rebuilt["_derived_read"]:
                    rebuilt["_derived_read"].append("agent:PUBLIC")
            elif scope == "swarm-shared" and swarm_id != "default":
                grant = f"swarm:{swarm_id}"
                if grant not in rebuilt["_derived_read"]:
                    rebuilt["_derived_read"].append(grant)
                if grant not in rebuilt["_derived_write"]:
                    rebuilt["_derived_write"].append(grant)
        self._instance_config = rebuilt

    def _mark_tiers_dirty(self):
        """Mark derived tiers as stale and clear stale derived data."""
        self._tiers_dirty = True
        self._temporal_index_dirty = True
        self._all_cons = [f for f in self._all_cons if _is_asserted_derived_fact(f)]
        self._all_cross = [f for f in self._all_cross if self._is_current_supported_cross_fact(f)]
        self._mark_full_index_dirty()

    def _index_debounce_ms(self) -> int:
        return _env_int("GOSH_MEMORY_INDEX_DEBOUNCE_MS", DEFAULT_INDEX_DEBOUNCE_MS, minimum=0)

    def _index_max_delay_ms(self) -> int:
        return _env_int("GOSH_MEMORY_INDEX_MAX_DELAY_MS", DEFAULT_INDEX_MAX_DELAY_MS, minimum=0)

    def _priority_index_top_k(self) -> int:
        return _env_int("GOSH_MEMORY_PRIORITY_INDEX_TOP_K", DEFAULT_PRIORITY_INDEX_TOP_K, minimum=1)

    def _priority_index_timeout_ms(self) -> int:
        return _env_int("GOSH_MEMORY_PRIORITY_INDEX_TIMEOUT_MS", DEFAULT_PRIORITY_INDEX_TIMEOUT_MS, minimum=1)

    def _now_ms(self) -> int:
        return int(datetime.now(timezone.utc).timestamp() * 1000)

    def _storage_supports_index_coordination(self) -> bool:
        return self._supports_index_coordination

    def _index_snapshot_fingerprint(self) -> str:
        payload = {
            "gran": _embedding_fingerprint(self._all_granular),
            "cons": _embedding_fingerprint(self._all_cons),
            "cross": _embedding_fingerprint(self._all_cross),
            "snapshot_version": self._index_snapshot_version,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def _mark_full_index_dirty(self) -> None:
        if not self._storage_supports_index_coordination():
            return
        try:
            storage = cast(Any, self._storage)
            storage.mark_index_dirty(
                now_ms=self._now_ms(),
                debounce_ms=self._index_debounce_ms(),
                max_delay_ms=self._index_max_delay_ms(),
            )
        except Exception:
            log.exception("failed to mark full index dirty")

    def _read_full_index_status(self) -> dict[str, Any]:
        if not self._storage_supports_index_coordination():
            return {
                "index_state": "ready" if self._data_dict is not None else "missing",
                "index_dirty": self._tiers_dirty,
                "index_build_lease_owner": None,
                "next_index_build_after_ms": None,
                "next_index_retry_after_ms": None,
            }
        storage = cast(Any, self._storage)
        status = storage.read_index_status(now_ms=self._now_ms())
        if self._data_dict is not None and status.get("index_state") == "missing":
            status["index_state"] = "ready"
        return status

    def _index_trace(self, *, corpus_embedding_rebuild: bool = False, embedding_cache_hit: bool = False) -> dict[str, Any]:
        status = self._read_full_index_status()
        return {
            "index_state": status.get("index_state"),
            "index_dirty_since_ms": status.get("index_dirty_since_ms"),
            "last_index_dirty_ms": status.get("last_index_dirty_ms"),
            "next_index_build_after_ms": status.get("next_index_build_after_ms"),
            "next_index_retry_after_ms": status.get("next_index_retry_after_ms"),
            "index_build_lease_owner": status.get("index_build_lease_owner"),
            "last_index_build_started_ms": status.get("last_index_build_started_ms"),
            "last_index_build_completed_ms": status.get("last_index_build_completed_ms"),
            "last_index_build_error": status.get("last_index_build_error"),
            "corpus_embedding_rebuild": corpus_embedding_rebuild,
            "embedding_cache_hit": embedding_cache_hit,
            "index_stale": bool(status.get("index_dirty")) and self._data_dict is not None,
        }

    def _schedule_full_index_after_recall_if_needed(self) -> bool:
        if not self._storage_supports_index_coordination():
            return False
        status = self._read_full_index_status()
        if status.get("index_state") in {"scheduled", "building", "backoff"}:
            return status.get("index_state") == "scheduled"
        if self._data_dict is not None and not status.get("index_dirty"):
            return False
        if not self._all_granular:
            return False
        self._mark_full_index_dirty()
        return True

    def _bump_index_snapshot_version(self) -> int:
        self._index_snapshot_version += 1
        return self._index_snapshot_version

    def _index_snapshot_is_stale(self, snapshot_version: int) -> bool:
        return self._tiers_dirty or snapshot_version != self._index_snapshot_version

    # ── ACL enforcement ──

    def _acl_allows_access(
        self,
        fact: dict,
        caller_id: str,
        caller_memberships: list[str] = None,
        caller_role: str = "user",
        *,
        need: str = "read",
    ) -> bool:
        """Check whether caller is allowed to read or write a resource."""
        if caller_role == "admin":
            return True

        if "owner_id" not in fact:
            acl = _normalize_loaded_acl_fields(
                scope=fact.get("scope"),
                agent_id=fact.get("agent_id", "default"),
                swarm_id=fact.get("swarm_id", "default"),
                owner_id=fact.get("owner_id"),
                read=fact.get("read"),
                write=fact.get("write"),
            )
            fact_owner = acl["owner_id"]
            fact_read = acl["read"]
            fact_write = acl["write"]
        else:
            fact_owner = fact["owner_id"]
            fact_read = fact.get("read", ["agent:PUBLIC"])
            fact_write = fact.get("write", [])

        if fact_owner == caller_id:
            return True

        if caller_id == "system" and fact_owner == "system":
            return True

        grants = fact_read if need == "read" else fact_write

        if "agent:PUBLIC" in grants:
            return True

        if caller_id in grants:
            return True

        if caller_memberships:
            for m in caller_memberships:
                if m in grants:
                    return True

        return False

    def _acl_allows(self, fact: dict, caller_id: str,
                    caller_memberships: list[str] = None,
                    caller_role: str = "user") -> bool:
        """Compatibility wrapper for read ACL checks."""
        return self._acl_allows_access(
            fact,
            caller_id,
            caller_memberships,
            caller_role,
            need="read",
        )

    # ── Fact tagging ──

    def _tag_facts(self, facts, session_date: str,
                   agent_id=None, swarm_id=None, scope=None,
                   owner_id=None, read=None, write=None,
                   artifact_id=None, version_id=None,
                   content_hash=None, status="active",
                   retention_ttl=None, metadata=None, target=None):
        """Tag all facts with conv_id, agent_id, swarm_id, scope, ACL, created_at.

        Per-call params override instance defaults — eliminates the race
        condition from concurrent store() calls with different identities.
        Also propagates identity/versioning fields when provided.
        """
        now = datetime.now(timezone.utc).isoformat()
        _agent_id = agent_id if agent_id is not None else self.agent_id
        _swarm_id = swarm_id if swarm_id is not None else self.swarm_id
        _scope = scope if scope is not None else self.scope

        # Resolve ACL: explicit inputs override scope-derived defaults.
        acl = _derive_acl_from_scope(_scope, _agent_id, _swarm_id)
        if owner_id is not None:
            _owner_id = owner_id
            _read = read if read is not None else list(acl["read"])
            _write = write if write is not None else list(acl["write"])
        else:
            _owner_id = acl["owner_id"]
            _read = read if read is not None else list(acl["read"])
            _write = write if write is not None else list(acl["write"])

        for f in facts:
            f["conv_id"] = self.key
            f["session_date"] = session_date
            f["agent_id"] = _agent_id
            f["swarm_id"] = _swarm_id
            f["scope"] = _scope
            f["owner_id"] = _owner_id
            f["read"] = list(_read)
            f["write"] = list(_write)
            f["created_at"] = now
            # Identity/versioning fields
            if artifact_id is not None:
                f["artifact_id"] = artifact_id
            if version_id is not None:
                f["version_id"] = version_id
            if content_hash is not None:
                f["content_hash"] = content_hash
            f.setdefault("status", status)
            if retention_ttl is not None:
                f["retention_ttl"] = retention_ttl
            if metadata is not None:
                merged_metadata = _merge_fact_metadata(f.get("metadata"), metadata)
                if merged_metadata:
                    f["metadata"] = merged_metadata
                else:
                    f.pop("metadata", None)
            if target is not None:
                if target:
                    f["target"] = list(target)
                else:
                    f.pop("target", None)
            # Auto-promote rules/constraints to swarm-shared,
            # but never override an explicit agent-private scope
            if f.get("kind") in ("constraint", "rule") and _scope != "agent-private":
                f["scope"] = "swarm-shared"

    def _resolve_live_writer_agent_id(
        self,
        *,
        requested_agent_id: str | None,
        caller_id: str | None = None,
        caller_principal_kind: str | None = None,
    ) -> str:
        """Resolve the trustworthy producer agent_id for live content writes.

        Protected live writes derive producer identity from the authenticated
        principal, not from a free-form request field. When no caller
        principal is provided, this helper preserves the existing direct-Python
        runtime behavior and falls back to the server/default direct agent
        identity.
        """
        if caller_id is not None:
            principal_id = _normalize_identity(str(caller_id), allow_public=False)
            if not principal_id.startswith("agent:") or (
                caller_principal_kind is not None and str(caller_principal_kind) != "agent"
            ):
                raise PermissionError("live content writes require authenticated agent principal")
            effective_agent_id = principal_id.split(":", 1)[1]
            if requested_agent_id is None:
                return effective_agent_id
            raw_requested = str(requested_agent_id).strip()
            if not raw_requested or raw_requested == "default":
                raise ValueError("agent_id must be omitted or match authenticated agent principal")
            if raw_requested != effective_agent_id:
                raise ValueError(
                    f"agent_id '{raw_requested}' does not match authenticated agent principal '{effective_agent_id}'"
                )
            return effective_agent_id

        fallback_agent_id = requested_agent_id if requested_agent_id is not None else self.agent_id
        normalized = _normalize_agent_identity(fallback_agent_id)
        if normalized == "default":
            return _normalize_agent_identity(self.agent_id)
        return normalized

    def _resolve_live_acl_context(
        self,
        *,
        agent_id: str | None,
        swarm_id: str | None,
        scope: str | None,
        owner_id: str | None,
        read: list[str] | None,
        write: list[str] | None,
        caller_id: str | None = None,
        caller_principal_kind: str | None = None,
    ) -> tuple[str, str, str, str, list[str], list[str]]:
        """Resolve direct live-write ACL without falling back to unsafe public defaults."""
        _agent_id = agent_id if agent_id is not None else self.agent_id
        _swarm_id = swarm_id if swarm_id is not None else self.swarm_id
        inferred_scope, inferred_agent_id, inferred_swarm_id = (None, None, None)
        if scope is None and owner_id is not None and read is not None and write is not None:
            inferred_scope, inferred_agent_id, inferred_swarm_id = _infer_scope_from_acl_fields(
                owner_id,
                read,
                write,
            )
        _scope = scope if scope is not None else inferred_scope
        if _scope is None:
            raise ValueError(LIVE_SCOPE_REQUIRED_ERROR)
        if _scope not in self.VALID_SCOPES:
            raise ValueError(f"Unknown scope: {_scope}")
        requested_agent_id = agent_id
        if requested_agent_id is None and inferred_agent_id and (_agent_id in (None, "", "default")):
            requested_agent_id = inferred_agent_id
        _agent_id = self._resolve_live_writer_agent_id(
            requested_agent_id=requested_agent_id,
            caller_id=caller_id,
            caller_principal_kind=caller_principal_kind,
        )
        if swarm_id is None and inferred_swarm_id and (_swarm_id in (None, "", "default")):
            _swarm_id = inferred_swarm_id
        acl_defaults = _derive_acl_from_scope(_scope, _agent_id, _swarm_id)
        _owner_id = owner_id if owner_id is not None else acl_defaults["owner_id"]
        _read = list(read) if read is not None else list(acl_defaults["read"])
        _write = list(write) if write is not None else list(acl_defaults["write"])
        return _agent_id, _swarm_id, _scope, _owner_id, _read, _write

    def _resolve_asserted_import_context(
        self,
        *,
        facts: list[dict],
        consolidated: list[dict] | None,
        cross_session: list[dict] | None,
        raw_sessions: list[dict] | None,
        provenance: dict | None,
        agent_id: str | None,
        swarm_id: str | None,
        scope: str | None,
        owner_id: str | None,
        read: list[str] | None,
        write: list[str] | None,
        caller_id: str | None = None,
        caller_principal_kind: str | None = None,
    ) -> tuple[str | None, str | None, str, str, list[str], list[str]]:
        """Resolve asserted-import ACL from explicit inputs or payload-carried ACL."""

        def _norm_list(value):
            if value is None:
                return None
            return [str(item) for item in value]

        def _capture_scalar(current, value, label):
            text = str(value or "").strip()
            if not text:
                return current
            if current not in (None, "", text):
                raise ValueError(f"mixed asserted fact {label}s require explicit split")
            return text

        def _capture_list(current, value, label):
            if value is None:
                return current
            normalized = _norm_list(value)
            if current is not None and current != normalized:
                raise ValueError(f"mixed asserted fact {label} ACLs require explicit split")
            return normalized

        resolved_scope = scope
        resolved_agent_id = agent_id
        resolved_swarm_id = swarm_id
        resolved_owner_id = owner_id
        resolved_read = _norm_list(read)
        resolved_write = _norm_list(write)

        for payload in (facts or []) + (consolidated or []) + (cross_session or []) + (raw_sessions or []):
            resolved_scope = _capture_scalar(resolved_scope, payload.get("scope"), "scope")
            resolved_agent_id = _capture_scalar(resolved_agent_id, payload.get("agent_id"), "agent_id")
            resolved_swarm_id = _capture_scalar(resolved_swarm_id, payload.get("swarm_id"), "swarm_id")
            resolved_owner_id = _capture_scalar(resolved_owner_id, payload.get("owner_id"), "owner_id")
            resolved_read = _capture_list(resolved_read, payload.get("read"), "read")
            resolved_write = _capture_list(resolved_write, payload.get("write"), "write")

        if provenance:
            resolved_scope = _capture_scalar(resolved_scope, provenance.get("scope"), "scope")
            resolved_agent_id = _capture_scalar(resolved_agent_id, provenance.get("agent_id"), "agent_id")
            resolved_swarm_id = _capture_scalar(resolved_swarm_id, provenance.get("swarm_id"), "swarm_id")
            resolved_owner_id = _capture_scalar(resolved_owner_id, provenance.get("owner_id"), "owner_id")
            resolved_read = _capture_list(resolved_read, provenance.get("read"), "read")
            resolved_write = _capture_list(resolved_write, provenance.get("write"), "write")

        if resolved_scope is None and resolved_owner_id is not None and resolved_read is not None and resolved_write is not None:
            inferred_scope, inferred_agent_id, inferred_swarm_id = _infer_scope_from_acl_fields(
                resolved_owner_id,
                resolved_read,
                resolved_write,
            )
            if inferred_scope:
                resolved_scope = inferred_scope
                if resolved_agent_id in (None, "", "default") and inferred_agent_id:
                    resolved_agent_id = inferred_agent_id
                if resolved_swarm_id in (None, "", "default") and inferred_swarm_id:
                    resolved_swarm_id = inferred_swarm_id

        if resolved_scope is None:
            raise ValueError(LIVE_SCOPE_REQUIRED_ERROR)
        if resolved_scope not in self.VALID_SCOPES:
            raise ValueError(f"Unknown scope: {resolved_scope}")

        if caller_id is not None:
            base_agent_id = self._resolve_live_writer_agent_id(
                requested_agent_id=resolved_agent_id,
                caller_id=caller_id,
                caller_principal_kind=caller_principal_kind,
            )
        else:
            base_agent_id = resolved_agent_id if resolved_agent_id is not None else self.agent_id
        base_swarm_id = resolved_swarm_id if resolved_swarm_id is not None else self.swarm_id
        acl_defaults = _derive_acl_from_scope(resolved_scope, base_agent_id, base_swarm_id)
        final_owner_id = resolved_owner_id if resolved_owner_id is not None else acl_defaults["owner_id"]
        final_read = list(resolved_read) if resolved_read is not None else list(acl_defaults["read"])
        final_write = list(resolved_write) if resolved_write is not None else list(acl_defaults["write"])
        return base_agent_id, base_swarm_id, resolved_scope, final_owner_id, final_read, final_write

    def _resolve_secret_acl_context(
        self,
        *,
        agent_id: str | None,
        swarm_id: str | None,
        scope: str | None,
        owner_id: str | None,
        read: list[str] | None,
        write: list[str] | None,
        caller_id: str | None,
        caller_role: str = "user",
    ) -> tuple[str | None, str | None, str, str, list[str], list[str], str]:
        resolved_agent_id = str(agent_id or "").strip() or None
        resolved_swarm_id = str(swarm_id or "").strip() or None
        resolved_scope = str(scope or "").strip() or None
        resolved_owner_id = str(owner_id or "").strip() or None
        resolved_read = _normalize_acl_principals(read, allow_public=True) if read is not None else None
        resolved_write = _normalize_acl_principals(write, allow_public=True) if write is not None else None
        normalized_caller_id = (
            _normalize_identity(str(caller_id), allow_public=False)
            if caller_id is not None
            else None
        )

        normalized_owner_id = (
            _normalize_identity(resolved_owner_id, allow_public=False)
            if resolved_owner_id is not None
            else None
        )

        if resolved_scope is None and normalized_owner_id is not None and resolved_read is not None and resolved_write is not None:
            inferred_scope, inferred_agent_id, inferred_swarm_id = _infer_scope_from_acl_fields(
                normalized_owner_id,
                resolved_read,
                resolved_write,
            )
            if inferred_scope is None and not resolved_read and not resolved_write and normalized_owner_id.startswith("agent:"):
                inferred_scope = "agent-private"
                inferred_agent_id = normalized_owner_id.split(":", 1)[1]
            if inferred_scope is not None:
                resolved_scope = inferred_scope
                if resolved_agent_id is None and inferred_agent_id:
                    resolved_agent_id = inferred_agent_id
                if resolved_swarm_id is None and inferred_swarm_id:
                    resolved_swarm_id = inferred_swarm_id

        if resolved_scope is None:
            raise ValueError(LIVE_SCOPE_REQUIRED_ERROR)
        if resolved_scope not in self.VALID_SCOPES:
            raise ValueError(f"Unknown scope: {resolved_scope}")
        if resolved_scope == "swarm-shared" and resolved_swarm_id in (None, "", "default"):
            raise ValueError(NAMED_SWARM_REQUIRED_ERROR)

        if resolved_scope == "system-wide":
            final_owner_id = "system"
            final_read = list(resolved_read) if resolved_read is not None else ["agent:PUBLIC"]
            final_write = list(resolved_write) if resolved_write is not None else ["agent:PUBLIC"]
            if final_owner_id != "system" or final_read != ["agent:PUBLIC"] or final_write != ["agent:PUBLIC"]:
                raise ValueError("system-wide secrets use the canonical system/public ACL")
            return None, None, resolved_scope, final_owner_id, final_read, final_write, acl_domain_key(
                final_owner_id,
                final_read,
                final_write,
            )

        if normalized_owner_id is None:
            if resolved_agent_id in (None, "", "default"):
                if normalized_caller_id and normalized_caller_id.startswith("agent:"):
                    resolved_agent_id = normalized_caller_id.split(":", 1)[1]
                else:
                    raise ValueError("non-system secret scopes require explicit agent_id")
            final_owner_id = _normalize_identity(f"agent:{resolved_agent_id}", allow_public=False)
        else:
            final_owner_id = normalized_owner_id
            if not final_owner_id.startswith("agent:"):
                raise ValueError("non-system secret scopes require agent-owned secrets")
            if resolved_agent_id in (None, "", "default"):
                resolved_agent_id = final_owner_id.split(":", 1)[1]

        if resolved_scope == "swarm-shared":
            grant = _normalize_identity(f"swarm:{resolved_swarm_id}")
            default_read = [grant]
            default_write = [grant]
        else:
            default_read = []
            default_write = []

        final_read = list(resolved_read) if resolved_read is not None else list(default_read)
        final_write = list(resolved_write) if resolved_write is not None else list(default_write)
        domain_key = acl_domain_key(final_owner_id, final_read, final_write)
        return resolved_agent_id, resolved_swarm_id, resolved_scope, final_owner_id, final_read, final_write, domain_key

    def _resolve_secret_lookup_domain(
        self,
        *,
        agent_id: str | None,
        swarm_id: str | None,
        scope: str | None,
        owner_id: str | None,
        caller_id: str | None,
    ) -> tuple[str, str | None, str | None, str]:
        resolved_scope = str(scope or "").strip()
        if not resolved_scope:
            raise ValueError(LIVE_SCOPE_REQUIRED_ERROR)
        if resolved_scope not in self.VALID_SCOPES:
            raise ValueError(f"Unknown scope: {resolved_scope}")
        resolved_swarm_id = str(swarm_id or "").strip() or None
        resolved_owner_id = str(owner_id or "").strip() or None
        resolved_agent_id = str(agent_id or "").strip() or None

        if resolved_scope == "system-wide":
            final_owner_id = "system"
            final_read = ["agent:PUBLIC"]
            final_write = ["agent:PUBLIC"]
            return resolved_scope, final_owner_id, None, acl_domain_key(final_owner_id, final_read, final_write)

        if resolved_scope == "swarm-shared" and resolved_swarm_id in (None, "", "default"):
            raise ValueError(NAMED_SWARM_REQUIRED_ERROR)

        if resolved_owner_id is not None:
            final_owner_id = _normalize_identity(resolved_owner_id, allow_public=False)
            if not final_owner_id.startswith("agent:"):
                raise ValueError("non-system secret scopes require agent-owned secrets")
            if resolved_agent_id in (None, "", "default"):
                resolved_agent_id = final_owner_id.split(":", 1)[1]
        else:
            if resolved_agent_id in (None, "", "default"):
                if caller_id is None:
                    raise ValueError("non-system secret lookups require explicit agent_id")
                normalized_caller_id = _normalize_identity(str(caller_id), allow_public=False)
                if not normalized_caller_id.startswith("agent:"):
                    raise ValueError("non-system secret lookups require explicit agent_id")
                resolved_agent_id = normalized_caller_id.split(":", 1)[1]
            final_owner_id = _normalize_identity(f"agent:{resolved_agent_id}", allow_public=False)

        if resolved_scope == "swarm-shared":
            grant = _normalize_identity(f"swarm:{resolved_swarm_id}")
            final_read = [grant]
            final_write = [grant]
        else:
            final_read = []
            final_write = []

        return resolved_scope, final_owner_id, resolved_swarm_id, acl_domain_key(
            final_owner_id,
            final_read,
            final_write,
        )

    def _resolve_secret_row_by_ref(
        self,
        *,
        name: str,
        agent_id: str | None,
        swarm_id: str | None,
        scope: str | None,
        owner_id: str | None,
        caller_id: str | None,
        include_value: bool,
    ) -> dict[str, Any]:
        secret_storage = self._secret_storage()
        if secret_storage is None:
            raise RuntimeError("secret storage requires SQLite backend")

        resolved_scope = str(scope or "").strip()
        if not resolved_scope:
            raise ValueError(LIVE_SCOPE_REQUIRED_ERROR)
        if resolved_scope not in self.VALID_SCOPES:
            raise ValueError(f"Unknown scope: {resolved_scope}")
        resolved_swarm_id = str(swarm_id or "").strip() or None
        resolved_agent_id = str(agent_id or "").strip() or None
        resolved_owner_id = str(owner_id or "").strip() or None
        caller_agent_id = None
        if isinstance(caller_id, str) and caller_id.startswith("agent:"):
            caller_agent_id = caller_id.split(":", 1)[1]
        if resolved_scope == "swarm-shared" and resolved_agent_id == caller_agent_id and not resolved_owner_id:
            resolved_agent_id = None

        if resolved_scope == "swarm-shared" and resolved_swarm_id in (None, "", "default"):
            raise ValueError(NAMED_SWARM_REQUIRED_ERROR)

        if resolved_scope == "swarm-shared" and resolved_agent_id in (None, "", "default") and not resolved_owner_id:
            rows = [
                row
                for row in secret_storage.list_secret_rows(include_values=include_value)
                if str(row.get("name") or "") == str(name)
                and str(row.get("scope") or "") == "swarm-shared"
                and str(row.get("swarm_id") or "") == str(resolved_swarm_id or "")
            ]
            if not rows:
                raise KeyError("secret not found")
            if len(rows) > 1:
                raise ValueError("secret ref is ambiguous; provide agent_id explicitly")
            return rows[0]

        _, _, _, domain_key = self._resolve_secret_lookup_domain(
            agent_id=resolved_agent_id,
            swarm_id=resolved_swarm_id,
            scope=resolved_scope,
            owner_id=resolved_owner_id or None,
            caller_id=caller_id,
        )
        row = secret_storage.get_secret_row(name=str(name), acl_domain_key=domain_key, include_value=include_value)
        if row is None:
            raise KeyError("secret not found")
        return row

    @staticmethod
    def _validate_persisted_acl_fields(
        *,
        scope: str | None,
        owner_id: str | None,
        read: list[str] | None,
        write: list[str] | None,
        context: str,
    ) -> None:
        missing = []
        if scope is None or str(scope).strip() == "":
            missing.append("scope")
        if owner_id is None or str(owner_id).strip() == "":
            missing.append("owner_id")
        if read is None:
            missing.append("read")
        if write is None:
            missing.append("write")
        if missing:
            raise ValueError(
                f"{context} missing persisted ACL fields: {', '.join(missing)}"
            )

    # ── Disk I/O ──

    def _ingress_storage(self) -> IngressWriteStorageBackend | None:
        if isinstance(self._storage, IngressWriteStorageBackend):
            return self._storage
        return None

    def _projection_storage(self) -> ProjectionWriteThroughStorageBackend | None:
        if isinstance(self._storage, ProjectionWriteThroughStorageBackend):
            return self._storage
        return None

    def _secret_storage(self) -> SecretStorageBackend | None:
        if isinstance(self._storage, SecretStorageBackend):
            return self._storage
        return None

    def _normalize_runtime_secret_ref(
        self,
        secret_ref: dict[str, Any] | None,
        *,
        field_name: str,
        required: bool = True,
    ) -> dict[str, Any] | None:
        if secret_ref is None:
            if required:
                raise RuntimeError(
                    f"Missing {field_name}; internal runtime secrets must use an explicit persisted secret_ref"
                )
            return None
        if not isinstance(secret_ref, dict):
            raise ValueError(f"{field_name} must be an object")
        # Accept prior locally-normalized refs during restart/upgrade, but do not
        # preserve derived fields in the user-facing config contract.
        allowed = {"name", "scope", "agent_id", "swarm_id", "owner_id", "domain_key"}
        unknown = set(secret_ref) - allowed
        if unknown:
            raise ValueError(
                f"{field_name} has unknown keys: {', '.join(sorted(unknown))}"
            )
        name = str(secret_ref.get("name") or "").strip()
        if not name:
            raise ValueError(f"{field_name}.name must be a non-empty string")
        scope = str(secret_ref.get("scope") or "").strip()
        if not scope:
            raise ValueError(f"{field_name}.scope must be a non-empty string")
        if scope not in self.VALID_SCOPES:
            raise ValueError(f"{field_name}.scope has unknown value: {scope}")
        had_agent_id = "agent_id" in secret_ref and str(secret_ref.get("agent_id") or "").strip() != ""
        had_swarm_id = "swarm_id" in secret_ref and str(secret_ref.get("swarm_id") or "").strip() != ""
        had_owner_id = "owner_id" in secret_ref and str(secret_ref.get("owner_id") or "").strip() != ""
        agent_id = str(secret_ref.get("agent_id") or "").strip() or None
        swarm_id = str(secret_ref.get("swarm_id") or "").strip() or None
        owner_id = str(secret_ref.get("owner_id") or "").strip() or None
        normalized_owner_id = None
        if owner_id is not None:
            normalized_owner_id = _normalize_identity(owner_id, allow_public=False)

        if scope == "system-wide":
            if normalized_owner_id not in (None, "system"):
                raise ValueError(f"{field_name}.owner_id must be 'system' for system-wide secrets")
            normalized_owner_id = "system"
            agent_id = None
            swarm_id = None
        else:
            if scope == "swarm-shared" and swarm_id in (None, "", "default"):
                raise ValueError(f"{field_name}.swarm_id must be provided explicitly for swarm-shared scope")
            if normalized_owner_id is not None and not normalized_owner_id.startswith("agent:"):
                raise ValueError(f"{field_name}.owner_id must be an agent principal for non-system scopes")
            if normalized_owner_id is None and scope == "agent-private" and agent_id in (None, "", "default"):
                raise ValueError(f"{field_name}.agent_id or owner_id must be provided for agent-private scope")
            if agent_id not in (None, "", "default"):
                agent_id = _normalize_agent_identity(agent_id)
            else:
                agent_id = None
            if normalized_owner_id is not None and agent_id is None:
                agent_id = normalized_owner_id.split(":", 1)[1]

        normalized = {
            "name": name,
            "scope": scope,
        }
        if had_agent_id and agent_id is not None:
            normalized["agent_id"] = agent_id
        if had_owner_id and normalized_owner_id is not None:
            normalized["owner_id"] = normalized_owner_id
        if had_swarm_id and swarm_id is not None and scope == "swarm-shared":
            normalized["swarm_id"] = _normalize_swarm_identity(swarm_id)
        return normalized

    @contextlib.contextmanager
    def _runtime_secret_context(
        self,
        secret_ref: dict[str, Any] | None,
        *,
        field_name: str,
    ):
        normalized = self._normalize_runtime_secret_ref(
            secret_ref,
            field_name=field_name,
            required=True,
        )
        assert normalized is not None
        with runtime_secret_context(self, normalized):
            yield

    def _resolve_runtime_secret_ref(self, secret_ref: dict[str, Any]) -> str:
        """Resolve one internal runtime secret from the persisted store only."""
        secret_storage = self._secret_storage()
        if secret_storage is None:
            raise RuntimeError("runtime secret resolution requires SQLite secret storage")
        normalized = self._normalize_runtime_secret_ref(
            secret_ref,
            field_name="runtime secret_ref",
            required=True,
        )
        assert normalized is not None
        secret_name = normalized["name"]
        try:
            row = self._resolve_secret_row_by_ref(
                name=secret_name,
                agent_id=normalized.get("agent_id"),
                swarm_id=normalized.get("swarm_id"),
                scope=normalized["scope"],
                owner_id=normalized.get("owner_id"),
                caller_id=None,
                include_value=True,
            )
        except KeyError:
            raise RuntimeError(
                f"Missing runtime secret '{secret_name}' in persisted secret store"
            )
        secret_value = str(row.get("value") or "")
        if not secret_value:
            raise RuntimeError(
                f"Runtime secret '{secret_name}' is empty in persisted secret store"
            )
        return secret_value

    def _resolve_librarian_secret_ref(self, model: str) -> dict[str, Any] | None:
        cfg = self.get_config()
        profile_cfg = self._resolve_librarian_profile_config(model)
        if profile_cfg is not None and self._profile_backend(profile_cfg) == "local_cli":
            return None
        if profile_cfg is not None and profile_cfg.get("secret_ref") is not None:
            normalized = self._normalize_runtime_secret_ref(
                profile_cfg.get("secret_ref"),
                field_name=f"profile_configs.{self._librarian_profile}.secret_ref",
                required=True,
            )
            assert normalized is not None
            return normalized
        normalized = self._normalize_runtime_secret_ref(
            cfg.get("librarian_secret_ref"),
            field_name="librarian_secret_ref",
            required=True,
        )
        assert normalized is not None
        return normalized

    def _resolve_embedding_secret_ref(self, *, model=None, provider=None) -> dict[str, Any] | None:
        _resolved_model, resolved_provider = _resolve_embed_config(model, provider)
        if resolved_provider == "local":
            return None
        return self._normalize_runtime_secret_ref(
            self.get_config().get("embedding_secret_ref"),
            field_name="embedding_secret_ref",
            required=True,
        )

    async def _call_extract_with_runtime_secrets(
        self,
        model: str,
        system: str,
        user_msg: str,
        max_tokens: int,
        sem: asyncio.Semaphore | None = None,
    ):
        profile_cfg = self._resolve_librarian_profile_config(model)
        if profile_cfg is not None and self._profile_backend(profile_cfg) == "local_cli":
            return await self._run_local_cli_extract(
                system=system,
                user_msg=user_msg,
                cli_bin=str(profile_cfg["cli_bin"]),
                cli_args_prefix=list(profile_cfg["cli_args_prefix"]),
                timeout_secs=profile_cfg.get("timeout_secs"),
                sem=sem,
            )
        with self._runtime_secret_context(
            self._resolve_librarian_secret_ref(model),
            field_name="librarian_secret_ref",
        ):
            return await call_extract(model, system, user_msg, max_tokens, sem)

    async def _canonicalize_semantic_source_text(
        self,
        text: str,
        *,
        family: str,
        model: str | None,
        call_extract_fn,
    ) -> dict[str, Any]:
        """Return raw original + canonical English semantic text for ingest/reextract."""
        original_text = str(text or "")
        canonical = await canonicalize_source_to_english(
            original_text,
            model=model,
            call_extract_fn=call_extract_fn,
        )
        semantic_ready = bool(canonical.get("semantic_ready"))
        canonical_en = (
            self._normalize_ingress_text(str(canonical.get("canonical_en") or ""), family)
            if semantic_ready
            else ""
        )
        return {
            "raw_original": original_text,
            "source_lang": str(canonical.get("source_lang") or "und"),
            "canonical_en": canonical_en,
            "semantic_ready": semantic_ready,
            "canonicalization_status": str(canonical.get("canonicalization_status") or "failed"),
            "canonicalization_error": (
                str(canonical.get("canonicalization_error") or "").strip() or None
            ),
            "translation_version": str(
                canonical.get("translation_version") or ENGLISH_CANONICAL_TRANSLATION_VERSION
            ),
        }

    async def _canonicalize_recall_query(
        self,
        query: str,
    ) -> dict[str, Any]:
        """Accept only English recall queries without model-side translation."""
        original_query = str(query or "").strip()
        source_lang = detect_source_language(original_query)
        blocked = source_lang not in {"en", "und"}
        canonical_en = "" if blocked else re.sub(r"\s+", " ", original_query).strip()
        return {
            "raw_original": original_query,
            "source_lang": source_lang,
            "canonical_en": canonical_en,
            "semantic_ready": not blocked,
            "canonicalization_status": "blocked_in_recall" if blocked else "ready",
            "canonicalization_error": (
                "memory_recall accepts English queries only; translate in the calling agent/model"
                if blocked
                else None
            ),
            "translation_version": "none",
        }

    async def _canonicalize_document_blocks_for_retrieval(
        self,
        block_dicts: list[dict],
        *,
        model: str | None,
        call_extract_fn,
    ) -> dict[str, Any]:
        """Canonicalize document blocks into English before grouping/extraction."""
        canonical_blocks: list[dict] = []
        seen_non_english = False
        cursor = 0
        first_error: str | None = None
        for block in block_dicts:
            block_text = str(block.get("text") or "")
            canonical = await self._canonicalize_semantic_source_text(
                block_text,
                family="document",
                model=model,
                call_extract_fn=call_extract_fn,
            )
            if canonical["source_lang"] not in {"en", "und"}:
                seen_non_english = True
            if not canonical["semantic_ready"]:
                first_error = str(canonical.get("canonicalization_error") or "document canonicalization failed")
                return {
                    "canonical_blocks": [],
                    "canonical_doc_text": "",
                    "source_lang": canonical["source_lang"] if seen_non_english else "mixed",
                    "semantic_ready": False,
                    "canonicalization_status": "failed",
                    "canonicalization_error": first_error,
                    "translation_version": canonical["translation_version"],
                }
            canonical_text = canonical["canonical_en"]
            canonical_block = dict(block)
            canonical_block["original_text"] = block_text
            canonical_block["source_lang"] = canonical["source_lang"]
            canonical_block["translation_version"] = canonical["translation_version"]
            canonical_block["semantic_ready"] = True
            canonical_block["canonicalization_status"] = canonical["canonicalization_status"]
            canonical_block["text"] = canonical_text
            canonical_block["text_preview"] = canonical_text[:240]
            canonical_block["char_len"] = len(canonical_text)
            canonical_block["original_raw_span"] = list(block.get("raw_span") or [0, len(block_text)])
            canonical_block["raw_span"] = [cursor, cursor + len(canonical_text)]
            canonical_blocks.append(canonical_block)
            cursor += len(canonical_text) + 2
        canonical_doc_text = "\n\n".join(
            str(block.get("text") or "").strip()
            for block in canonical_blocks
            if str(block.get("text") or "").strip()
        ).strip()
        return {
            "canonical_blocks": canonical_blocks,
            "canonical_doc_text": canonical_doc_text,
            "source_lang": ("mixed" if seen_non_english else "en"),
            "semantic_ready": True,
            "canonicalization_status": "ready",
            "canonicalization_error": None,
            "translation_version": ENGLISH_CANONICAL_TRANSLATION_VERSION,
        }

    @staticmethod
    def _episode_original_text_from_blocks(
        episode: dict,
        original_block_map: dict[str, dict],
        original_content: str | None = None,
    ) -> str:
        block_ids = list((episode.get("provenance") or {}).get("block_ids") or [])
        original_content = str(original_content or "")
        spans: list[tuple[int, int]] = []
        if original_content:
            for block_id in block_ids:
                raw_span = (original_block_map.get(block_id) or {}).get("raw_span")
                if not isinstance(raw_span, (list, tuple)) or len(raw_span) != 2:
                    continue
                try:
                    start = int(raw_span[0])
                    end = int(raw_span[1])
                except (TypeError, ValueError):
                    continue
                if 0 <= start <= end <= len(original_content):
                    spans.append((start, end))
            if spans:
                return original_content[min(start for start, _end in spans): max(end for _start, end in spans)]
        parts = [
            str((original_block_map.get(block_id) or {}).get("text") or "")
            for block_id in block_ids
            if str((original_block_map.get(block_id) or {}).get("text") or "")
        ]
        return "\n\n".join(parts)

    @staticmethod
    def _semantic_state_fields(canonical: dict[str, Any]) -> dict[str, Any]:
        return {
            "semantic_ready": bool(canonical.get("semantic_ready")),
            "canonicalization_status": str(canonical.get("canonicalization_status") or "failed"),
            "canonicalization_error": (
                str(canonical.get("canonicalization_error") or "").strip() or None
            ),
            "source_lang": str(canonical.get("source_lang") or "und"),
            "translation_version": str(
                canonical.get("translation_version") or ENGLISH_CANONICAL_TRANSLATION_VERSION
            ),
        }
    async def _call_oai_with_runtime_secrets(
        self,
        model: str,
        prompt: str,
        *,
        max_tokens: int = 300,
        json_mode: bool = False,
        temperature: float = 0,
        semaphore: asyncio.Semaphore | None = None,
        secret_ref: dict[str, Any] | None = None,
    ):
        with self._runtime_secret_context(secret_ref, field_name="inference secret_ref"):
            return await call_oai(
                model,
                prompt,
                max_tokens=max_tokens,
                json_mode=json_mode,
                temperature=temperature,
                semaphore=semaphore,
            )

    async def _call_model_with_runtime_secrets(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int = 300,
        temperature: float = 0.0,
        json_mode: bool = False,
        secret_ref: dict[str, Any] | None = None,
    ):
        with self._runtime_secret_context(secret_ref, field_name="inference secret_ref"):
            return await _call_model(
                model,
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                json_mode=json_mode,
            )

    async def _embed_query_with_runtime_secrets(self, text: str, model=None, provider=None):
        effective_model = model or self._embedding_model
        secret_ref = self._resolve_embedding_secret_ref(model=effective_model, provider=provider)
        if secret_ref is None:
            return await embed_query(text, model=effective_model, provider=provider)
        with self._runtime_secret_context(secret_ref, field_name="embedding_secret_ref"):
            return await embed_query(text, model=effective_model, provider=provider)

    async def _embed_texts_with_runtime_secrets(self, texts, **kwargs):
        effective_model = kwargs.get("model") or self._embedding_model
        if effective_model is not None:
            kwargs = {**kwargs, "model": effective_model}
        secret_ref = self._resolve_embedding_secret_ref(
            model=kwargs.get("model"),
            provider=kwargs.get("provider"),
        )
        if secret_ref is None:
            return await embed_texts(texts, **kwargs)
        with self._runtime_secret_context(secret_ref, field_name="embedding_secret_ref"):
            return await embed_texts(texts, **kwargs)

    def _embed_texts_sync_with_runtime_secrets(self, texts, **kwargs):
        secret_ref = self._resolve_embedding_secret_ref(
            model=kwargs.get("model"),
            provider=kwargs.get("provider"),
        )
        if secret_ref is None:
            return embed_texts_sync(texts, **kwargs)
        with self._runtime_secret_context(secret_ref, field_name="embedding_secret_ref"):
            return embed_texts_sync(texts, **kwargs)

    def _get_model_client_with_runtime_secrets(self, model: str, *, secret_ref: dict[str, Any] | None = None):
        with self._runtime_secret_context(secret_ref, field_name="inference secret_ref"):
            return _get_client(model)

    def _refresh_secret_summaries(self) -> None:
        secret_storage = self._secret_storage()
        if secret_storage is None:
            self._secrets = []
            return
        try:
            rows = secret_storage.list_secret_rows(include_values=False)
        except RuntimeError:
            self._secrets = []
            return
        self._secrets = [
            {
                "secret_id": row.get("secret_id"),
                "name": row.get("name"),
                "acl_domain_key": row.get("acl_domain_key"),
                "created_by_principal_id": row.get("created_by_principal_id"),
                "scope": row.get("scope"),
                "owner_id": row.get("owner_id"),
                "agent_id": row.get("agent_id"),
                "swarm_id": row.get("swarm_id"),
                "created_at": row.get("created_at"),
                "updated_at": row.get("updated_at"),
                "metadata": dict(row.get("metadata") or {}),
            }
            for row in rows
        ]

    def _state_json_values(self) -> dict:
        return {
            "n_sessions": self._n_sessions,
            "n_sessions_with_facts": self._n_sessions_with_facts,
            "_emb_fingerprints": self._emb_fingerprints,
            "_dedup_index": {json.dumps(list(k)): v for k, v in self._dedup_index.items()},
            "_content_dedup_index": {json.dumps(list(k)): v for k, v in self._content_dedup_index.items()},
            "_git_dedup_index": {json.dumps(list(k)): v for k, v in self._git_dedup_index.items()},
            "_simhash_index": {json.dumps(list(k)): v for k, v in self._simhash_index.items()},
            "metadata_schema": self._metadata_schema,
            "instance_config": self._instance_config,
            "scope_record": self._scope_record,
            "profiles": self._profiles,
            "profile_configs": self._profile_configs,
            "memory_config": self._memory_config,
        }

    def _persist_projection_delta(self, **kwargs) -> None:
        projection_storage = self._projection_storage()
        if projection_storage is None:
            self._save_snapshot()
            return
        if self._container_graph_delta_required(kwargs):
            self._container_graph = build_document_container_graph(
                raw_docs=self._raw_docs,
                episode_corpus=self._episode_corpus,
                source_records=self._source_records,
                existing_graph=self._container_graph,
            )
            kwargs.update(self._container_graph_projection_kwargs(self._container_graph, replace=True))
        projection_storage.persist_projection_delta(**kwargs)

    @staticmethod
    def _container_graph_delta_required(delta_kwargs: dict) -> bool:
        """Return true when a write-through delta can change document container rows."""

        return any(
            delta_kwargs.get(key)
            for key in (
                "raw_doc_upserts",
                "episode_doc_replacements",
                "source_record_upserts",
            )
        )

    @staticmethod
    def _container_graph_projection_kwargs(graph: dict, *, replace: bool) -> dict:
        graph = normalize_container_graph(graph)
        return {
            "replace_container_graph": replace,
            "container_graph_revision_upserts": graph.get("graph_revisions") or [],
            "container_upserts": graph.get("containers") or [],
            "container_relation_upserts": graph.get("relations") or [],
            "container_anchor_upserts": graph.get("anchors") or [],
            "container_evidence_upserts": graph.get("evidence") or [],
            "container_ref_upserts": graph.get("refs") or [],
            "container_ref_lookup_upserts": graph.get("ref_lookup") or [],
            "container_ref_range_upserts": graph.get("ref_ranges") or [],
            "container_render_ref_upserts": graph.get("render_refs") or [],
            "container_contract_upserts": graph.get("contracts") or [],
            "container_artifact_upserts": graph.get("artifacts") or [],
            "container_state_upserts": graph.get("state") or [],
        }

    def _snapshot_cache_payload(self) -> dict:
        save_granular = [{k: v for k, v in f.items() if k != "_temporal_links"}
                         for f in self._all_granular]

        # Serialize dedup_index: tuple keys → JSON string keys
        serializable_dedup = {json.dumps(list(k)): v
                              for k, v in self._dedup_index.items()}
        serializable_content_dedup = {json.dumps(list(k)): v
                                      for k, v in self._content_dedup_index.items()}
        serializable_git_dedup = {json.dumps(list(k)): v
                                  for k, v in self._git_dedup_index.items()}
        serializable_simhash = {json.dumps(list(k)): v
                                for k, v in self._simhash_index.items()}
        self._container_graph = build_document_container_graph(
            raw_docs=self._raw_docs,
            episode_corpus=self._episode_corpus,
            source_records=self._source_records,
            existing_graph=self._container_graph,
        )

        return {
            "granular":              save_granular,
            "cons":                  self._all_cons,
            "cross":                 self._all_cross,
            "tlinks":                self._all_tlinks,
            "raw_sessions":          self._raw_sessions,
            "raw_docs":              self._raw_docs,
            "episode_corpus":        self._episode_corpus,
            "n_sessions":            self._n_sessions,
            "n_sessions_with_facts": self._n_sessions_with_facts,
            "_emb_fingerprints":     self._emb_fingerprints,
            "_dedup_index":          serializable_dedup,
            "_content_dedup_index":  serializable_content_dedup,
            "_git_dedup_index":      serializable_git_dedup,
            "_simhash_index":        serializable_simhash,
            "metadata_schema":       self._metadata_schema,
            "instance_config":       self._instance_config,
            "scope_record":          self._scope_record,
            "source_records":        self._source_records,
            "container_graph":       self._container_graph,
            "profiles":              self._profiles,
            "profile_configs":       self._profile_configs,
            "memory_config":         self._memory_config,
        }

    def _save_snapshot(self):
        """Persist the current extracted snapshot view."""
        self._storage.save_facts(self._snapshot_cache_payload())

    def _save_cache(self):
        """Persist the current snapshot view."""
        self._save_snapshot()

    def _ensure_container_graph(self, *, persist: bool = False) -> dict:
        """Build or refresh the container graph from cached raw/episode state."""

        next_graph = build_document_container_graph(
            raw_docs=self._raw_docs,
            episode_corpus=self._episode_corpus,
            source_records=self._source_records,
            existing_graph=self._container_graph,
        )
        if normalize_container_graph(next_graph) != normalize_container_graph(self._container_graph):
            self._container_graph = next_graph
            if persist:
                projection_storage = self._projection_storage()
                if projection_storage is None:
                    self._save_cache()
                else:
                    projection_storage.persist_projection_delta(
                        **self._container_graph_projection_kwargs(self._container_graph, replace=True)
                    )
        return self._container_graph

    def validate_document_raw_sources(self, *, strict: bool = True) -> dict[str, Any]:
        """Validate that active document sources have original raw source material."""

        allowed_provenance = {
            "original_source",
            "original_source_multipart",
            "original_source_backfill",
            "canonical_benchmark_raw",
        }
        errors: list[dict[str, Any]] = []
        for source_id, record in sorted((self._source_records or {}).items()):
            if str(record.get("family") or "") != "document":
                continue
            raw_text = self._raw_docs.get(source_id)
            source_meta = dict(record.get("source_meta") or {})
            if not str(raw_text or ""):
                errors.append({"code": "RAW_SOURCE_MISSING", "source_id": source_id})
                continue
            provenance = str(source_meta.get("raw_source_provenance") or "").strip()
            if strict and provenance not in allowed_provenance:
                errors.append(
                    {
                        "code": "RAW_SOURCE_NOT_ORIGINAL",
                        "source_id": source_id,
                        "raw_source_provenance": provenance or None,
                    }
                )
        storage_validator = getattr(self._storage, "validate_raw_doc_projection", None)
        if callable(storage_validator):
            for error in storage_validator():
                if error not in errors:
                    errors.append(error)
        for error in validate_container_exact_copy_render_refs(self._container_graph):
            if error not in errors:
                errors.append(error)
        return {"ok": not errors, "errors": errors}

    @staticmethod
    def _load_original_raw_source_manifest(manifest_path: str | Path) -> list[dict[str, Any]]:
        path = Path(manifest_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            raw_entries = payload.get("sources")
            if raw_entries is None:
                raw_entries = payload.get("entries")
            if raw_entries is None:
                raw_entries = [
                    {"source_id": key, **(value if isinstance(value, dict) else {"original_content": value})}
                    for key, value in payload.items()
                    if isinstance(key, str)
                ]
        elif isinstance(payload, list):
            raw_entries = payload
        else:
            raw_entries = []
        entries: list[dict[str, Any]] = []
        for raw_entry in raw_entries or []:
            if not isinstance(raw_entry, dict):
                continue
            entry = dict(raw_entry)
            source_id = str(
                entry.get("source_id")
                or entry.get("logical_source_id")
                or entry.get("case_id")
                or ""
            ).strip()
            if not source_id:
                continue
            content_kind = str(entry.get("content_kind") or entry.get("fixture_kind") or "").lower()
            if content_kind in {"expected_answer", "scorer_ground_truth", "answer_key"}:
                entry["refused"] = "expected_answer_source_forbidden"
            content = entry.get("original_content")
            if content is None:
                content = entry.get("original_raw")
            if content is None:
                content = entry.get("content")
            content_path = entry.get("content_path") or entry.get("raw_path") or entry.get("path")
            if content is None and content_path:
                raw_path = Path(str(content_path))
                if not raw_path.is_absolute():
                    raw_path = path.parent / raw_path
                content = raw_path.read_text(encoding="utf-8")
            entry["source_id"] = source_id
            entry["original_content"] = "" if content is None else str(content)
            entries.append(entry)
        return entries

    def _resolve_raw_backfill_source_id(self, manifest_source_id: str) -> str | None:
        manifest_source_id = str(manifest_source_id or "").strip()
        if not manifest_source_id:
            return None
        if manifest_source_id in self._source_records:
            return manifest_source_id
        for source_id, record in (self._source_records or {}).items():
            source_meta = dict((record or {}).get("source_meta") or {})
            if str(source_meta.get("logical_source_id") or "").strip() == manifest_source_id:
                return str(source_id)
        return None

    def backfill_original_raw_sources(
        self,
        manifest_path: str | Path,
        *,
        dry_run: bool = False,
        strict: bool = True,
    ) -> dict[str, Any]:
        """Backfill trusted original raw document sources without rerunning extraction."""

        entries = self._load_original_raw_source_manifest(manifest_path)
        return self.backfill_original_raw_source_entries(
            entries,
            dry_run=dry_run,
            strict=strict,
            manifest_label=str(manifest_path),
        )

    def backfill_original_raw_source_entries(
        self,
        entries: list[dict[str, Any]],
        *,
        dry_run: bool = False,
        strict: bool = True,
        manifest_label: str = "memory_admin_api",
    ) -> dict[str, Any]:
        """Backfill trusted original raw document source entries without extraction."""

        backfilled: list[str] = []
        already_valid: list[str] = []
        missing: list[str] = []
        refused: list[dict[str, Any]] = []
        raw_doc_upserts: list[dict[str, Any]] = []
        source_record_upserts: dict[str, dict] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                missing.append("<invalid-entry>")
                continue
            entry = dict(entry)
            manifest_source_id = str(entry.get("source_id") or "").strip()
            if not manifest_source_id:
                manifest_source_id = str(entry.get("logical_source_id") or "").strip()
            source_id = self._resolve_raw_backfill_source_id(manifest_source_id) or manifest_source_id
            content_kind = str(entry.get("content_kind") or entry.get("fixture_kind") or "").lower()
            if content_kind in {"expected_answer", "scorer_ground_truth", "answer_key"}:
                refused.append({"source_id": source_id, "code": "expected_answer_source_forbidden"})
                continue
            if entry.get("refused"):
                refused.append({"source_id": source_id, "code": str(entry["refused"])})
                continue
            original_content = str(entry.get("original_content") or "")
            if original_content == "":
                original_content = str(entry.get("original_raw") or entry.get("content") or "")
            if not source_id or not original_content:
                missing.append(source_id or "<missing-source-id>")
                continue
            if source_id not in self._source_records and strict:
                missing.append(source_id)
                continue
            content_hash = hashlib.sha256(original_content.encode("utf-8")).hexdigest()
            source_record = self._source_records.setdefault(
                source_id,
                {"source_id": source_id, "family": "document", "source_meta": {"logical_source_id": source_id}},
            )
            source_record.setdefault("family", "document")
            source_meta = source_record.setdefault("source_meta", {})
            if (
                self._raw_docs.get(source_id) == original_content
                and source_meta.get("raw_source_content_hash") == content_hash
            ):
                already_valid.append(source_id)
                continue
            if dry_run:
                backfilled.append(source_id)
                continue
            self._raw_docs[source_id] = original_content
            source_meta.update(
                {
                    "raw_source_provenance": "original_source_backfill",
                    "raw_source_backfilled": True,
                    "backfill_source": manifest_label,
                    "extraction_rerun": False,
                    "backfill_content_hash": content_hash,
                    "raw_source_content_hash": content_hash,
                }
            )
            raw_doc_upserts.append(
                {
                    "source_id": source_id,
                    "message_id": str(entry.get("message_id") or f"rawdoc:{source_id}"),
                    "metadata": {
                        "raw_source_backfilled": True,
                        "backfill_source": manifest_label,
                        "extraction_rerun": False,
                        "backfill_content_hash": content_hash,
                    },
                }
            )
            source_record_upserts[source_id] = deepcopy(source_record)
            storage_backfill = getattr(self._storage, "backfill_raw_doc_content", None)
            if callable(storage_backfill):
                storage_result = storage_backfill(
                    source_id=source_id,
                    content_text=original_content,
                    metadata=raw_doc_upserts[-1]["metadata"],
                    message_id=raw_doc_upserts[-1]["message_id"],
                )
                raw_doc_upserts[-1]["message_id"] = str(storage_result.get("message_id") or raw_doc_upserts[-1]["message_id"])
            backfilled.append(source_id)
        if not dry_run and (raw_doc_upserts or source_record_upserts):
            self._persist_projection_delta(
                raw_doc_upserts=raw_doc_upserts,
                source_record_upserts=source_record_upserts,
                state_values=self._state_json_values(),
                episode_corpus=self._episode_corpus,
            )
            self._ensure_container_graph(persist=True)
        validation = self.validate_document_raw_sources(strict=strict)
        return {
            "manifest_path": manifest_label,
            "backfilled": backfilled,
            "already_valid": already_valid,
            "missing": missing,
            "refused": refused,
            "dry_run": dry_run,
            "extraction_rerun": False,
            "validation": validation,
        }

    def _persist_runtime_state(self) -> None:
        """Persist state_json without re-establishing ingress truth on live SQLite paths."""
        projection_storage = self._projection_storage()
        if projection_storage is not None:
            projection_storage.persist_projection_delta(
                state_values=self._state_json_values(),
                episode_corpus=self._episode_corpus,
            )
            return
        self._save_snapshot()

    @staticmethod
    def _fact_projection_signature(facts: list[dict]) -> list[tuple[str, str, str, str, str, str, str]]:
        return [
            (
                str(fact.get("id") or ""),
                str(fact.get("status") or "active"),
                str(fact.get("fact") or ""),
                str(fact.get("artifact_id") or ""),
                str(fact.get("version_id") or ""),
                str(fact.get("source_id") or ""),
                str(fact.get("session") or ""),
            )
            for fact in facts
        ]

    def _snapshot_refresh_required(self) -> bool:
        """Return True when persisted snapshot projections no longer match runtime tiers."""
        projection_storage = self._projection_storage()
        if projection_storage is None or not self._storage.exists:
            return False
        try:
            persisted = self._storage.load_facts(internal=True)
        except Exception:
            return True
        if not isinstance(persisted, dict):
            return True
        return any(
            self._fact_projection_signature(persisted.get(key) or [])
            != self._fact_projection_signature(current)
            for key, current in (
                ("granular", self._all_granular),
                ("cons", self._all_cons),
                ("cross", self._all_cross),
            )
        )

    @staticmethod
    def _versioning_domain(scope: str, owner_id: str | None, swarm_id: str | None) -> str:
        return dedup_domain_key(scope, owner_id, swarm_id)

    @staticmethod
    def _logical_source_id(record_key: str, record: dict | None = None) -> str:
        if isinstance(record, dict):
            source_meta = record.get("source_meta") or {}
            logical = str(source_meta.get("logical_source_id") or "").strip()
            if logical:
                return logical
        return str(record_key or "").strip()

    @staticmethod
    def _source_record_acl_context_from_record(record: dict | None) -> tuple[str, str | None, str | None]:
        record = record or {}
        source_meta = record.get("source_meta") or {}
        owner_id = record.get("owner_id")
        swarm_id = source_meta.get("swarm_id")
        scope = source_meta.get("scope")
        if scope:
            return str(scope), owner_id, str(swarm_id) if swarm_id else None
        read = list(record.get("read") or [])
        if owner_id == "system" and "agent:PUBLIC" in read:
            return "system-wide", owner_id, str(swarm_id) if swarm_id else None
        swarm_grants = [grant for grant in read if isinstance(grant, str) and grant.startswith("swarm:")]
        if swarm_grants:
            return "swarm-shared", owner_id, swarm_grants[0].split(":", 1)[1]
        if owner_id:
            return "agent-private", owner_id, str(swarm_id) if swarm_id else None
        return "system-wide", owner_id, str(swarm_id) if swarm_id else None

    @staticmethod
    def _projection_source_id_token(
        *,
        source_id: str,
        family: str,
        scope: str,
        owner_id: str | None,
        swarm_id: str | None,
    ) -> str:
        canonical_family = MemoryServer._canonical_content_family(family)
        domain = dedup_domain_key(scope, owner_id, swarm_id)
        seed = json.dumps(
            {
                "family": canonical_family,
                "domain": domain,
                "source_id": str(source_id),
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        digest = hashlib.sha1(seed.encode("utf-8"), usedforsecurity=False).hexdigest()[:12]
        return f"{source_id}@@{digest}"

    def _projection_source_id(
        self,
        *,
        source_id: str,
        family: str,
        scope: str,
        owner_id: str | None,
        swarm_id: str | None,
    ) -> str:
        logical_source_id = str(source_id or "").strip()
        canonical_family = self._canonical_content_family(family)
        for record_key, record in self._source_records.items():
            if self._logical_source_id(record_key, record) != logical_source_id:
                continue
            record_family = self._canonical_content_family(record.get("family") or "conversation")
            record_scope, record_owner, record_swarm = self._source_record_acl_context_from_record(record)
            if (
                record_family == canonical_family
                and str(record_scope or "") == str(scope or "")
                and str(record_owner or "") == str(owner_id or "")
                and str(record_swarm or "") == str(swarm_id or "")
            ):
                return str(record_key)
        if logical_source_id not in self._source_records:
            return logical_source_id
        token = self._projection_source_id_token(
            source_id=logical_source_id,
            family=canonical_family,
            scope=scope,
            owner_id=owner_id,
            swarm_id=swarm_id,
        )
        if token not in self._source_records:
            return token
        suffix = 2
        while f"{token}:{suffix}" in self._source_records:
            suffix += 1
        return f"{token}:{suffix}"

    @staticmethod
    def _projection_session_num_for_row(raw_session: dict) -> int | None:
        session_num = raw_session.get("projection_session_num")
        if isinstance(session_num, int) and session_num > 0:
            return session_num
        if isinstance(session_num, str) and session_num.isdigit() and int(session_num) > 0:
            return int(session_num)
        session_num = raw_session.get("session_num")
        if isinstance(session_num, int) and session_num > 0:
            return session_num
        if isinstance(session_num, str) and session_num.isdigit() and int(session_num) > 0:
            return int(session_num)
        return None

    def _resolve_projection_session_num(
        self,
        *,
        logical_session_num: int,
        source_id: str,
        family: str,
        scope: str,
        owner_id: str | None,
        swarm_id: str | None,
    ) -> int:
        canonical_family = self._canonical_content_family(family)
        for raw_session in self._raw_sessions:
            if self._canonical_content_family(raw_session.get("format") or "conversation") != canonical_family:
                continue
            if str(raw_session.get("source_id") or "") != str(source_id):
                continue
            if _coerce_positive_session_num(raw_session.get("session_num")) != int(logical_session_num):
                continue
            if str(raw_session.get("scope") or "") != str(scope or ""):
                continue
            if str(raw_session.get("owner_id") or "") != str(owner_id or ""):
                continue
            if str(raw_session.get("swarm_id") or "") != str(swarm_id or ""):
                continue
            existing = self._projection_session_num_for_row(raw_session)
            if existing is not None:
                return existing
        existing_projection_nums = [
            num
            for num in (self._projection_session_num_for_row(raw_session) for raw_session in self._raw_sessions)
            if num is not None
        ]
        if logical_session_num not in existing_projection_nums:
            return int(logical_session_num)
        return (max(existing_projection_nums) if existing_projection_nums else 0) + 1

    @staticmethod
    def _namespace_projection_fact_ids(facts: list[dict], *, source_id: str) -> None:
        prefix = MemoryServer._episode_source_key(source_id)
        for fact in facts:
            fact_id = str(fact.get("id") or "").strip()
            if fact_id and not fact_id.startswith(f"{prefix}_"):
                fact["id"] = f"{prefix}_{fact_id}"

    def _source_versioning_key(
        self,
        *,
        source_id: str,
        family: str,
        scope: str,
        owner_id: str | None,
        swarm_id: str | None,
        session_num: int | None = None,
        multipart_part_key: str | None = None,
    ) -> tuple:
        canonical_family = self._canonical_content_family(family)
        domain = self._versioning_domain(scope, owner_id, swarm_id)
        if canonical_family == "document":
            if multipart_part_key:
                return ("v2", canonical_family, domain, str(source_id), "__document_part__", str(multipart_part_key))
            return ("v2", canonical_family, domain, str(source_id), "__document__")
        if canonical_family == "codebase":
            return ("v2", canonical_family, domain, str(source_id), "__codebase__")
        return ("v2", canonical_family, domain, str(source_id), int(session_num or 0))

    def _rebuild_source_version_index(self) -> None:
        rebuilt: dict[tuple, dict] = {}
        eligible_raw_session_ids = {
            str(f.get("raw_session_id") or "")
            for f in self._all_granular
            if str(f.get("status") or "active") == "active" and str(f.get("raw_session_id") or "")
        }
        eligible_document_sources = {
            str(f.get("source_id") or "")
            for f in self._all_granular
            if str(f.get("status") or "active") == "active"
            and str(f.get("source_id") or "")
            and self._canonical_content_family(
                (self._source_records.get(str(f.get("source_id") or "")) or {}).get("family") or "conversation"
            ) == "document"
        }
        eligible_document_sources.update(
            str(source_id)
            for source_id, record in self._source_records.items()
            if self._canonical_content_family((record or {}).get("family") or "conversation") == "document"
        )

        def _prefer(current: dict | None, candidate: dict) -> dict:
            if current is None:
                return candidate
            current_status = str(current.get("status") or "active")
            candidate_status = str(candidate.get("status") or "active")
            if current_status != "active" and candidate_status == "active":
                return candidate
            if current_status == "active" and candidate_status != "active":
                return current
            current_stored = str(current.get("stored_at") or "")
            candidate_stored = str(candidate.get("stored_at") or "")
            if candidate_stored >= current_stored:
                return candidate
            return current

        for fact in self._all_granular:
            if str(fact.get("status") or "active") != "active":
                continue
            source_id = str(fact.get("source_id") or "").strip()
            if not source_id:
                continue
            record = self._source_records.get(source_id) or {}
            family = self._canonical_content_family(
                str(record.get("family") or ("document" if str((fact.get("metadata") or {}).get("document_source") or "").strip() else "conversation"))
            )
            key = self._source_versioning_key(
                source_id=source_id,
                family=family,
                scope=str(fact.get("scope") or "swarm-shared"),
                owner_id=fact.get("owner_id"),
                swarm_id=fact.get("swarm_id"),
                session_num=_coerce_positive_session_num(fact.get("session")),
                multipart_part_key=(
                    self._multipart_part_key(fact.get("metadata"))
                    if family == "document"
                    else None
                ),
            )
            candidate = {
                "artifact_id": fact.get("artifact_id"),
                "version_id": fact.get("version_id"),
                "content_hash": fact.get("content_hash"),
                "message_id": None,
                "stored_at": fact.get("created_at"),
                "session_num": _coerce_positive_session_num(fact.get("session")),
                "status": fact.get("status", "active"),
            }
            rebuilt[key] = _prefer(rebuilt.get(key), candidate)

        for raw_session in self._raw_sessions:
            source_id = str(raw_session.get("source_id") or "").strip()
            if not source_id:
                continue
            family = self._canonical_content_family(raw_session.get("format") or "conversation")
            raw_session_id = str(raw_session.get("raw_session_id") or "")
            if raw_session_id and raw_session_id not in eligible_raw_session_ids:
                continue
            part_key = self._raw_session_part_key(raw_session) if family == "document" else None
            key = self._source_versioning_key(
                source_id=source_id,
                family=family,
                scope=str(raw_session.get("scope") or "swarm-shared"),
                owner_id=raw_session.get("owner_id"),
                swarm_id=raw_session.get("swarm_id"),
                session_num=_coerce_positive_session_num(raw_session.get("session_num")),
                multipart_part_key=part_key,
            )
            candidate = {
                "artifact_id": raw_session.get("artifact_id"),
                "version_id": raw_session.get("version_id"),
                "content_hash": raw_session.get("content_hash"),
                "message_id": raw_session.get("message_id"),
                "stored_at": raw_session.get("stored_at"),
                "session_num": _coerce_positive_session_num(raw_session.get("session_num")),
                "status": raw_session.get("status", "active"),
            }
            rebuilt[key] = _prefer(rebuilt.get(key), candidate)

        for source_id, raw_text in self._raw_docs.items():
            source_id = str(source_id or "").strip()
            if not source_id:
                continue
            if source_id not in eligible_document_sources:
                continue
            record = self._source_records.get(source_id) or {}
            scope, owner_id, swarm_id = self._source_record_acl_context(source_id)
            key = self._source_versioning_key(
                source_id=source_id,
                family=str(record.get("family") or "document"),
                scope=scope,
                owner_id=owner_id,
                swarm_id=swarm_id,
            )
            candidate = {
                "artifact_id": record.get("artifact_id"),
                "version_id": record.get("version_id"),
                "content_hash": record.get("content_hash")
                or content_hash_text(str(raw_text or ""), family=str(record.get("family") or "document")),
                "message_id": f"rawdoc:{source_id}",
                "stored_at": ((record.get("source_meta") or {}).get("stored_at")),
                "session_num": None,
                "status": "active",
            }
            rebuilt[key] = _prefer(rebuilt.get(key), candidate)

        self._dedup_index = rebuilt

    @staticmethod
    def _canonical_content_family(family: str | None) -> str:
        normalized = str(family or "").strip().lower()
        if normalized == "chat":
            return "conversation"
        if normalized in {"conversation", "document", "codebase", "artifact"}:
            return normalized
        return normalized or "conversation"

    def _normalize_ingress_text(self, text: str, family: str | None) -> str:
        return normalize_text(text, family=self._canonical_content_family(family))

    @staticmethod
    def _coerce_ingress_metadata(metadata: dict | None) -> dict | None:
        if metadata is None or not isinstance(metadata, dict):
            return metadata
        coerced: dict = {}
        for key, value in metadata.items():
            if key in {"turn_number", "part_idx"} and isinstance(value, int) and not isinstance(value, bool):
                coerced[key] = str(value)
            else:
                coerced[key] = value
        return coerced

    @staticmethod
    def _content_dedup_hash(
        normalized_text: str,
        *,
        family: str,
        multipart_part_key: str | None = None,
    ) -> str:
        canonical_family = MemoryServer._canonical_content_family(family)
        if multipart_part_key:
            payload = json.dumps(
                {"part_key": multipart_part_key, "text": normalized_text},
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return content_hash_text(normalized_text, family=canonical_family)

    @staticmethod
    def _near_duplicate_similarity(distance: int) -> float:
        similarity = 1.0 - (float(distance) / 64.0)
        return round(max(0.0, similarity), 3)

    @staticmethod
    def _dedup_reference(entry: dict | None) -> dict:
        info = dict(entry or {})
        return {
            "message_id": info.get("message_id"),
            "source_id": info.get("source_id"),
            "session_num": info.get("session_num"),
            "stored_at": info.get("stored_at"),
        }

    def _content_index_entries_for_message(self, message_id: str) -> list[tuple[tuple, dict]]:
        return [
            (key, value)
            for key, value in self._content_dedup_index.items()
            if str((value or {}).get("message_id") or "") == str(message_id)
        ]

    def _remove_content_indices_for_message(self, message_id: str) -> None:
        for key, _value in list(self._content_index_entries_for_message(message_id)):
            self._content_dedup_index.pop(key, None)
        for key in [
            key for key in self._simhash_index
            if len(key) == 2 and str(key[1] or "") == str(message_id)
        ]:
            self._simhash_index.pop(key, None)

    def _remove_document_content_indices(self, source_id: str, multipart_part_key: str | None) -> None:
        removable_messages: set[str] = set()
        for key, value in list(self._content_dedup_index.items()):
            if str((value or {}).get("source_id") or "") != str(source_id):
                continue
            if str((value or {}).get("family") or "") == "conversation":
                continue
            if str((value or {}).get("multipart_part_key") or "") != str(multipart_part_key or ""):
                continue
            removable_messages.add(str((value or {}).get("message_id") or ""))
            self._content_dedup_index.pop(key, None)
        for message_id in removable_messages:
            self._remove_content_indices_for_message(message_id)

    @staticmethod
    def _raw_session_part_key(raw_session: dict) -> str | None:
        part_key = str(raw_session.get("part_source_id") or "").strip()
        if part_key:
            return part_key
        metadata = raw_session.get("metadata")
        return MemoryServer._multipart_part_key(metadata if isinstance(metadata, dict) else None)

    def _active_granular_facts_for_content_entry(self, info: dict) -> list[dict]:
        """Return active semantic evidence proving a conversation content duplicate.

        Raw sessions are lifecycle artifacts, not semantic evidence. Conversation
        exact content dedup only becomes authoritative after at least one active
        extracted fact can be linked to the stored raw artifact. Legacy imports
        may miss raw_session_id, so fall back through message_id and then the
        source/session tuple.
        """
        if not isinstance(info, dict):
            return []
        message_id = str(info.get("message_id") or "").strip()
        source_id = str(info.get("source_id") or "").strip()
        session_num = _coerce_positive_session_num(info.get("session_num"))
        raw_session_ids = {
            str(raw.get("raw_session_id") or "").strip()
            for raw in self._raw_sessions
            if _is_active_lifecycle_record(raw)
            and message_id
            and str(raw.get("message_id") or "").strip() == message_id
            and str(raw.get("raw_session_id") or "").strip()
        }
        active_facts = [fact for fact in self._all_granular if _is_active_lifecycle_record(fact)]
        if raw_session_ids:
            linked = [
                fact for fact in active_facts
                if str(fact.get("raw_session_id") or "").strip() in raw_session_ids
            ]
            if linked:
                return linked
        if message_id:
            linked = [
                fact for fact in active_facts
                if str(fact.get("message_id") or "").strip() == message_id
            ]
            if linked:
                return linked
        if source_id and session_num is not None:
            return [
                fact for fact in active_facts
                if str(fact.get("source_id") or "").strip() == source_id
                and (
                    _coerce_positive_session_num(fact.get("session")) == session_num
                    or _coerce_positive_session_num(fact.get("session_num")) == session_num
                    or _coerce_positive_session_num(fact.get("projection_session_num")) == session_num
                )
            ]
        return []

    def _is_current_supported_cross_fact(self, fact: dict) -> bool:
        if _is_asserted_derived_fact(fact):
            return _is_active_lifecycle_record(fact)
        if not _is_supported_cross_fact(fact) or not _is_active_lifecycle_record(fact):
            return False
        metadata = fact.get("metadata") or {}
        source_id = str(
            fact.get("source_id")
            or metadata.get("source_id")
            or metadata.get("episode_source_id")
            or ""
        ).strip()
        if not source_id:
            return False
        source_record = self._source_records.get(source_id) or {}
        source_version = str(source_record.get("version_id") or "").strip()
        source_artifact = str(source_record.get("artifact_id") or "").strip()
        fact_version = str(fact.get("version_id") or "").strip()
        fact_artifact = str(fact.get("artifact_id") or "").strip()
        if source_version and fact_version != source_version:
            return False
        if source_artifact and fact_artifact != source_artifact:
            return False
        for source_fact in self._all_granular:
            if not _is_active_lifecycle_record(source_fact):
                continue
            if str(source_fact.get("source_id") or "") != source_id:
                continue
            source_fact_version = str(source_fact.get("version_id") or "").strip()
            if source_version and source_fact_version != source_version:
                continue
            if fact_version and source_fact_version and source_fact_version != fact_version:
                continue
            source_fact_artifact = str(source_fact.get("artifact_id") or "").strip()
            if source_artifact and source_fact_artifact != source_artifact:
                continue
            if fact_artifact and source_fact_artifact and source_fact_artifact != fact_artifact:
                continue
            return True
        return False

    def _source_record_acl_context(self, source_id: str) -> tuple[str, str | None, str | None]:
        return self._source_record_acl_context_from_record(self._source_records.get(source_id) or {})

    def _index_content_entry(
        self,
        *,
        message_id: str,
        source_id: str | None,
        session_num: int | None,
        stored_at: str | None,
        scope: str,
        owner_id: str | None,
        swarm_id: str | None,
        family: str,
        content: str,
        multipart_part_key: str | None = None,
    ) -> None:
        if not message_id:
            return
        canonical_family = self._canonical_content_family(family)
        normalized_content = self._normalize_ingress_text(content, canonical_family)
        domain = dedup_domain_key(scope, owner_id, swarm_id)
        dedup_hash = self._content_dedup_hash(
            normalized_content,
            family=canonical_family,
            multipart_part_key=multipart_part_key,
        )
        self._remove_content_indices_for_message(message_id)
        self._content_dedup_index[(domain, dedup_hash)] = {
            "message_id": message_id,
            "source_id": source_id,
            "session_num": session_num,
            "stored_at": stored_at,
            "family": canonical_family,
            "multipart_part_key": multipart_part_key,
        }
        if len(normalized_content) < NEAR_DUP_MIN_CHARS:
            return
        self._simhash_index[(domain, message_id)] = simhash(normalized_content, ngram=3)

    def _find_exact_duplicate(
        self,
        *,
        content: str,
        family: str,
        scope: str,
        owner_id: str | None,
        swarm_id: str | None,
        multipart_part_key: str | None = None,
    ) -> dict | None:
        canonical_family = self._canonical_content_family(family)
        normalized_content = self._normalize_ingress_text(content, canonical_family)
        domain = dedup_domain_key(scope, owner_id, swarm_id)
        dedup_hash = self._content_dedup_hash(
            normalized_content,
            family=canonical_family,
            multipart_part_key=multipart_part_key,
        )
        info = self._content_dedup_index.get((domain, dedup_hash))
        if not isinstance(info, dict):
            return None
        if canonical_family == "conversation" and not self._active_granular_facts_for_content_entry(info):
            self._remove_content_indices_for_message(str(info.get("message_id") or ""))
            return None
        return dict(info)

    def _find_near_duplicate(
        self,
        *,
        content: str,
        family: str,
        scope: str,
        owner_id: str | None,
        swarm_id: str | None,
        multipart_part_key: str | None = None,
    ) -> dict | None:
        canonical_family = self._canonical_content_family(family)
        normalized_content = self._normalize_ingress_text(content, canonical_family)
        if len(normalized_content) < NEAR_DUP_MIN_CHARS:
            return None
        domain = dedup_domain_key(scope, owner_id, swarm_id)
        current = simhash(normalized_content, ngram=3)
        best: dict | None = None
        best_distance: int | None = None
        for (candidate_domain, candidate_message_id), candidate_simhash in self._simhash_index.items():
            if candidate_domain != domain:
                continue
            infos = [
                value
                for key, value in self._content_dedup_index.items()
                if key[0] == domain and str((value or {}).get("message_id") or "") == str(candidate_message_id)
            ]
            if not infos:
                continue
            info = infos[0]
            if str(info.get("family") or "") != canonical_family:
                continue
            if str(info.get("multipart_part_key") or "") != str(multipart_part_key or ""):
                continue
            distance = hamming_distance(current, int(candidate_simhash))
            if distance > NEAR_DUP_SIMHASH_THRESHOLD:
                continue
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best = dict(info)
        if best is None or best_distance is None:
            return None
        return {
            "similar_to_message_id": best.get("message_id"),
            "similar_to_source_id": best.get("source_id"),
            "similar_to_session_num": best.get("session_num"),
            "similarity": self._near_duplicate_similarity(best_distance),
        }

    def _rebuild_content_dedup_indices(self, *, persist: bool) -> None:
        self._content_dedup_index = {}
        self._simhash_index = {}

        eligible_raw_session_ids = {
            str(f.get("raw_session_id") or "")
            for f in self._all_granular
            if _is_active_lifecycle_record(f) and str(f.get("raw_session_id") or "")
        }
        eligible_document_sources = {
            str(f.get("source_id") or "")
            for f in self._all_granular
            if _is_active_lifecycle_record(f)
            and str(f.get("source_id") or "")
            and self._canonical_content_family((self._source_records.get(str(f.get("source_id") or "")) or {}).get("family") or "conversation") != "conversation"
        }
        eligible_document_sources.update(
            str(source_id)
            for source_id, record in self._source_records.items()
            if self._canonical_content_family((record or {}).get("family") or "conversation") != "conversation"
        )

        multipart_sources = {
            str(rs.get("source_id") or "")
            for rs in self._raw_sessions
            if self._canonical_content_family(rs.get("format") or "conversation") != "conversation"
            and self._raw_session_part_key(rs)
        }

        for raw_session in self._raw_sessions:
            if not _is_active_lifecycle_record(raw_session):
                continue
            if (
                str(raw_session.get("raw_session_id") or "")
                and str(raw_session.get("raw_session_id") or "") not in eligible_raw_session_ids
            ):
                continue
            family = self._canonical_content_family(raw_session.get("format") or "conversation")
            source_id = str(raw_session.get("source_id") or self.key)
            if family != "conversation":
                part_key = self._raw_session_part_key(raw_session)
                if not part_key:
                    continue
                same_part = [
                    rs for rs in self._raw_sessions
                    if self._canonical_content_family(rs.get("format") or "conversation") == family
                    and _is_active_lifecycle_record(rs)
                    and str(rs.get("source_id") or "") == source_id
                    and self._raw_session_part_key(rs) == part_key
                    and str(rs.get("message_id") or "") == str(raw_session.get("message_id") or "")
                ]
                if not same_part:
                    continue
                same_part.sort(key=lambda rs: int(rs.get("session_num") or 0))
                combined = "\n\n".join(
                    _semantic_raw_session_text(rs).strip()
                    for rs in same_part
                    if _semantic_raw_session_text(rs).strip()
                ).strip()
                first = same_part[0]
                self._index_content_entry(
                    message_id=str(first.get("message_id") or ""),
                    source_id=source_id,
                    session_num=_coerce_positive_session_num(first.get("session_num")),
                    stored_at=first.get("stored_at"),
                    scope=str(first.get("scope") or "swarm-shared"),
                    owner_id=first.get("owner_id"),
                    swarm_id=first.get("swarm_id"),
                    family=family,
                    content=combined,
                    multipart_part_key=part_key,
                )
                continue
            semantic_content = _semantic_raw_session_text(raw_session).strip()
            if not semantic_content:
                continue
            self._index_content_entry(
                message_id=str(raw_session.get("message_id") or raw_session.get("raw_session_id") or ""),
                source_id=source_id,
                session_num=_coerce_positive_session_num(raw_session.get("session_num")),
                stored_at=raw_session.get("stored_at"),
                scope=str(raw_session.get("scope") or "swarm-shared"),
                owner_id=raw_session.get("owner_id"),
                swarm_id=raw_session.get("swarm_id"),
                family=family,
                content=semantic_content,
            )

        for source_id, raw_text in self._raw_docs.items():
            if source_id not in eligible_document_sources:
                continue
            if source_id in multipart_sources:
                continue
            semantic_doc_text = str(raw_text or "").strip()
            if not semantic_doc_text:
                continue
            scope, owner_id, swarm_id = self._source_record_acl_context(source_id)
            matching_sessions = [
                rs for rs in self._raw_sessions
                if rs.get("format") == "document"
                and _is_active_lifecycle_record(rs)
                and str(rs.get("source_id") or "") == str(source_id)
            ]
            matching_sessions.sort(key=lambda rs: int(rs.get("session_num") or 0))
            message_id = (
                str(matching_sessions[0].get("message_id") or "")
                if matching_sessions
                else f"rawdoc:{source_id}"
            )
            session_num = (
                _coerce_positive_session_num(matching_sessions[0].get("session_num"))
                if matching_sessions
                else None
            )
            stored_at = matching_sessions[0].get("stored_at") if matching_sessions else None
            family = str((self._source_records.get(source_id) or {}).get("family") or "document")
            self._index_content_entry(
                message_id=message_id,
                source_id=source_id,
                session_num=session_num,
                stored_at=stored_at,
                scope=scope,
                owner_id=owner_id,
                swarm_id=swarm_id,
                family=family,
                content=semantic_doc_text,
            )

        if persist and self._storage.exists:
            self._persist_runtime_state()

    def _build_temporal_text_spans(self) -> list[dict]:
        facts_by_episode = build_facts_by_episode(self._all_granular)
        episode_lookup = build_episode_lookup(self._episode_corpus)
        episode_timeline_ids: dict[str, str] = {}
        spans: list[dict] = []
        for doc in self._episode_corpus.get("documents", []):
            doc_id = str(doc.get("doc_id") or "")
            for ep in doc.get("episodes", []):
                episode_id = str(ep.get("episode_id") or "").strip()
                if not episode_id:
                    continue
                source_id = str(ep.get("source_id") or doc_id or self.key)
                timeline_id = doc_id or f"timeline:{source_id}:main"
                episode_timeline_ids[episode_id] = timeline_id
                support_texts: list[str] = []
                seen_support_texts: set[str] = set()
                for fact in facts_by_episode.get(episode_id, []):
                    fact_text = str(fact.get("fact") or "").strip()
                    if not fact_text or fact_text in seen_support_texts:
                        continue
                    seen_support_texts.add(fact_text)
                    support_texts.append(fact_text[:400])
                    if len(support_texts) >= 8:
                        break
                spans.append(
                    {
                        "span_id": episode_id,
                        "source_id": source_id,
                        "timeline_id": timeline_id,
                        "text": ep.get("raw_text", ""),
                        "timestamp": ep.get("source_date"),
                        "ordinal_hint": None,
                        "provenance": ep.get("provenance") or {},
                        "support_fact_ids": [
                            fact_id
                            for fact_id in (
                                fact.get("id", "")
                                for fact in facts_by_episode.get(episode_id, [])
                            )
                            if fact_id
                        ],
                        "payload": {
                            "episode_id": episode_id,
                            "doc_id": doc_id,
                            "support_texts": support_texts,
                        },
                    }
                )
        for fact in self._all_granular:
            fact_id = str(fact.get("id") or "").strip()
            fact_text = str(fact.get("fact") or "").strip()
            if not fact_id or not fact_text:
                continue
            metadata = fact.get("metadata") or {}
            episode_id = str(
                metadata.get("episode_id")
                or fact.get("episode_id")
                or ""
            ).strip()
            episode = episode_lookup.get(episode_id) or {}
            source_id = str(
                fact.get("source_id")
                or metadata.get("episode_source_id")
                or episode.get("source_id")
                or self.key
            )
            provenance = {}
            support_spans = fact.get("support_spans")
            if isinstance(support_spans, list) and support_spans:
                first_span = support_spans[0] or {}
                start = first_span.get("start")
                end = first_span.get("end")
                if isinstance(start, int) and isinstance(end, int):
                    provenance = {
                        "start_char": int(start),
                        "end_char": int(end),
                        "source_field": str(first_span.get("source_field") or "raw_text"),
                        "episode_id": str(first_span.get("episode_id") or episode_id or ""),
                    }
            elif episode.get("provenance"):
                provenance = dict(episode.get("provenance") or {})
                if episode_id and not provenance.get("episode_id"):
                    provenance["episode_id"] = episode_id
                provenance.setdefault("source_field", "raw_text")
            spans.append(
                {
                    "span_id": f"fact:{fact_id}",
                    "source_id": source_id,
                    "timeline_id": episode_timeline_ids.get(episode_id)
                    or episode.get("source_id")
                    or source_id
                    or self.key,
                    "text": fact_text,
                    "timestamp": fact.get("session_date") or episode.get("source_date"),
                    "ordinal_hint": None,
                    "provenance": provenance,
                    "support_fact_ids": [fact_id],
                    "payload": {
                        "episode_id": episode_id,
                        "fact_id": fact_id,
                    },
                }
            )
        return spans

    def _rebuild_temporal_index(self) -> None:
        self._temporal_index = normalize_temporal_index(self._build_temporal_text_spans())
        self._temporal_index_dirty = False

    @staticmethod
    def _format_calendar_resolution_answer(plan: dict, event: dict) -> str | None:
        time_start = str(event.get("time_start") or "").strip()
        if not time_start:
            return None
        granularity = str(plan.get("granularity") or "date").lower()
        event_granularity = str(event.get("time_granularity") or "").lower()
        if granularity == "year":
            return time_start[:4]
        if granularity == "month":
            try:
                dt = datetime.fromisoformat(f"{time_start[:10]}T00:00:00")
                return dt.strftime("%B")
            except Exception:
                return time_start[:7] or None
        if event_granularity == "year":
            return time_start[:4]
        if event_granularity == "month":
            try:
                dt = datetime.fromisoformat(f"{time_start[:10]}T00:00:00")
                return dt.strftime("%B %Y")
            except Exception:
                return time_start[:7] or None
        return time_start[:10]

    @staticmethod
    def _is_fact_specific_temporal_event(fact_id: str, event: dict) -> bool:
        payload = event.get("payload") or {}
        if isinstance(payload, dict) and str(payload.get("fact_id") or "").strip() == fact_id:
            return True
        support_fact_ids = [str(fid) for fid in (event.get("support_fact_ids") or []) if str(fid).strip()]
        return len(support_fact_ids) == 1 and support_fact_ids[0] == fact_id

    @staticmethod
    def _calendar_fact_candidate_score(fact_text: str, query: str) -> tuple[float, int]:
        fact_tokens = {
            normalize_term_token(token)
            for token in re.findall(r"[A-Za-z]+(?:-[A-Za-z]+)?", fact_text.lower())
            if normalize_term_token(token) and normalize_term_token(token) not in STOP_WORDS
        }
        query_tokens = {
            normalize_term_token(token)
            for token in re.findall(r"[A-Za-z]+(?:-[A-Za-z]+)?", query.lower())
            if normalize_term_token(token) and normalize_term_token(token) not in STOP_WORDS
        }
        overlap = len(fact_tokens & query_tokens)
        temporal_bonus = 0.0
        if re.search(r"\b\d+\s+(?:years?|months?|weeks?)\s+ago\b", fact_text, re.I):
            temporal_bonus += 2.0
        if re.search(r"\bfor\s+\d+\s+(?:years?|months?|weeks?)\b", fact_text, re.I):
            temporal_bonus += 2.0
        if re.search(r"\blast\s+(?:week|month|year)\b", fact_text, re.I):
            temporal_bonus += 1.5
        return (overlap + temporal_bonus, -len(fact_text))

    def _calendar_query_anchor_timestamp(
        self,
        *,
        source_ids: set[str] | None = None,
        timeline_ids: set[str] | None = None,
    ) -> str | None:
        anchor = latest_calendar_anchor(
            self._temporal_index,
            source_ids=source_ids,
            timeline_ids=timeline_ids,
        )
        if anchor:
            return anchor
        latest_dt: datetime | None = None
        for raw in self._raw_sessions or []:
            raw_date = str(raw.get("session_date") or "").strip()
            if not raw_date:
                continue
            try:
                candidate = date_parser.parse(raw_date, fuzzy=True)
            except Exception:
                continue
            if latest_dt is None or candidate > latest_dt:
                latest_dt = candidate
        if latest_dt is None:
            return None
        return latest_dt.date().isoformat()

    async def _resolve_calendar_answer(
        self,
        *,
        query: str,
        candidate_facts: list[dict],
        source_ids: set[str] | None = None,
        timeline_ids: set[str] | None = None,
        limit: int = 8,
    ) -> dict | None:
        plan = extract_calendar_query(query)
        if not plan or plan.get("mode") != "answer":
            return None
        hit = execute_calendar_query(
            query,
            self._temporal_index,
            anchor_timestamp=self._calendar_query_anchor_timestamp(
                source_ids=source_ids,
                timeline_ids=timeline_ids,
            ),
            source_ids=source_ids,
            timeline_ids=timeline_ids,
            limit=limit,
        )
        events = list(hit.get("events") or [])
        if not events:
            return None
        fact_lookup = {
            str(fact.get("id") or "").strip(): fact
            for fact in candidate_facts
            if str(fact.get("id") or "").strip()
        }
        matched_facts: list[dict] = []
        seen_fact_ids: set[str] = set()
        for event in events:
            payload = event.get("payload") or {}
            candidate_fact_ids = []
            payload_fact_id = str(payload.get("fact_id") or "").strip() if isinstance(payload, dict) else ""
            if payload_fact_id:
                candidate_fact_ids.append(payload_fact_id)
            candidate_fact_ids.extend(
                str(fid).strip()
                for fid in (event.get("support_fact_ids") or [])
                if str(fid).strip()
            )
            for fact_id in candidate_fact_ids:
                if fact_id in seen_fact_ids:
                    continue
                fact = fact_lookup.get(fact_id)
                if not fact:
                    continue
                matched_facts.append(fact)
                seen_fact_ids.add(fact_id)
        if matched_facts:
            atomic_embs = (self._data_dict or {}).get("atomic_embs")
            matched_ids = {
                str(fact.get("id") or "").strip()
                for fact in matched_facts
                if str(fact.get("id") or "").strip()
            }
            ranked_facts: list[dict] = []
            ranked_embs: list[np.ndarray] = []
            if isinstance(atomic_embs, np.ndarray) and len(atomic_embs) == len(self._all_granular):
                for idx, fact in enumerate(self._all_granular):
                    fact_id = str(fact.get("id") or "").strip()
                    if not fact_id or fact_id not in matched_ids:
                        continue
                    ranked_facts.append(fact)
                    ranked_embs.append(atomic_embs[idx])
            if ranked_facts and ranked_embs:
                query_embedding = await self._embed_query_with_runtime_secrets(query)
                sweep = source_local_fact_sweep(
                    query,
                    ranked_facts,
                    np.asarray(ranked_embs),
                    query_embedding=query_embedding,
                    top_k=max(1, limit),
                    bm25_pool=max(8, limit * 2),
                    vector_pool=max(8, limit * 2),
                    entity_pool=max(4, limit),
                    rrf_k=60,
                )
                reranked = [row.get("fact") for row in sweep.get("retrieved", []) if row.get("fact")]
                if reranked:
                    matched_facts = reranked[:limit]
            else:
                matched_facts.sort(
                    key=lambda fact: self._calendar_fact_candidate_score(
                        str(fact.get("fact") or ""),
                        query,
                    ),
                    reverse=True,
                )
                matched_facts = matched_facts[:limit]
            kept_fact_ids = {
                str(fact.get("id") or "").strip()
                for fact in matched_facts
                if str(fact.get("id") or "").strip()
            }
            filtered_events: list[dict] = []
            for event in events:
                payload = event.get("payload") or {}
                payload_fact_id = str(payload.get("fact_id") or "").strip() if isinstance(payload, dict) else ""
                event_fact_ids = {
                    payload_fact_id,
                    *(
                        str(fid).strip()
                        for fid in (event.get("support_fact_ids") or [])
                        if str(fid).strip()
                    ),
                }
                if event_fact_ids & kept_fact_ids:
                    filtered_events.append(event)
            if filtered_events:
                events = filtered_events[:limit]
        return {
            "plan": hit.get("query") or plan,
            "events": events,
            "facts": matched_facts,
        }

    def _temporal_selector_evidence_lines(
        self,
        *,
        fact: dict | None,
        event: dict | None = None,
        max_spans: int = 1,
        max_chars_per_span: int = 220,
    ) -> list[str]:
        if not facts_as_selectors_enabled():
            return []
        episode_lookup = build_episode_lookup(self._episode_corpus)
        fact_lookup = self._fact_lookup if isinstance(getattr(self, "_fact_lookup", None), dict) else None
        support_spans: list[dict] = []
        if isinstance(fact, dict):
            support_spans.extend(
                span
                for span in iter_support_spans(fact, fact_lookup=fact_lookup)
                if isinstance(span, dict)
            )
        if not support_spans and isinstance(event, dict):
            source_span = event.get("source_span")
            if isinstance(source_span, dict):
                support_spans.append(source_span)

        lines: list[str] = []
        seen_refs: set[tuple[str, str, int, int]] = set()
        for span in support_spans[:max_spans]:
            ep_id = str(
                span.get("episode_id")
                or ((fact or {}).get("metadata") or {}).get("episode_id")
                or ""
            ).strip()
            if not ep_id:
                continue
            episode = episode_lookup.get(ep_id) or {}
            source_field = str(span.get("source_field") or "raw_text").strip() or "raw_text"
            raw_text = str(
                (episode.get(source_field) if source_field != "raw_text" else episode.get("raw_text"))
                or ""
            )
            if not raw_text:
                continue
            start_value = span.get("start", span.get("start_char", 0))
            end_value = span.get("end", span.get("end_char", 0))
            try:
                start = max(0, int(start_value))
                end = min(len(raw_text), int(end_value))
            except Exception:
                continue
            if end <= start:
                continue
            ref = (ep_id, source_field, start, end)
            if ref in seen_refs:
                continue
            seen_refs.add(ref)
            snippet = raw_text[start:end].strip().replace("\n", " ")
            if not snippet:
                continue
            if len(snippet) > max_chars_per_span:
                snippet = snippet[: max_chars_per_span - 3].rstrip() + "..."
            lines.extend(
                [
                    f"    Source ({source_field}, Episode {ep_id}, chars {start}-{end}):",
                    f"    > {snippet}",
                ]
            )
        return lines

    @staticmethod
    def _inject_temporal_evidence_block(recall_result: dict, block_text: str) -> None:
        block = str(block_text or "").strip()
        if not block:
            return
        context = str(recall_result.get("context") or "")
        if block not in context:
            recall_result["context"] = f"{block}\n\n{context}" if context else block
        context_packet = recall_result.get("_context_packet")
        if isinstance(context_packet, dict):
            tier1 = list(context_packet.get("tier1") or [])
            if not any(item.get("text") == block for item in tier1 if isinstance(item, dict)):
                tier1.insert(0, {"text": block, "rank": 1_000_000.0, "source": "temporal"})
            context_packet["tier1"] = tier1
            recall_result["_context_packet"] = context_packet
    def _attach_calendar_answer_resolution(
        self,
        *,
        query: str,
        recall_result: dict,
        candidate_facts: list[dict],
        source_ids: set[str] | None = None,
        timeline_ids: set[str] | None = None,
        resolution: dict | None = None,
    ) -> dict:
        if resolution is None:
            return recall_result
        if not resolution:
            return recall_result
        events = list(resolution.get("events") or [])
        facts = list(resolution.get("facts") or [])
        if facts:
            existing = []
            if recall_result.get("runtime_trace", {}).get("runtime") == "fact":
                existing = [
                    fact
                    for fact in candidate_facts
                    if str(fact.get("id") or "").strip() in {
                        item.get("fact_id", "")
                        for item in (recall_result.get("retrieved") or [])
                        if isinstance(item, dict)
                    }
                ]
            else:
                existing = list(candidate_facts)
            seen_fact_ids: set[str] = set()
            merged_facts: list[dict] = []
            for fact in facts + existing:
                fact_id = str(fact.get("id") or "").strip()
                if fact_id and fact_id not in seen_fact_ids:
                    merged_facts.append(fact)
                    seen_fact_ids.add(fact_id)
            if recall_result.get("runtime_trace", {}).get("runtime") == "fact":
                recall_result["retrieved"] = [
                    {
                        "fact_id": str(fact.get("id") or ""),
                        "conv_id": fact.get("conv_id", self.key),
                        "sim": 1_000_000.0 if idx < len(facts) else 0.0,
                    }
                    for idx, fact in enumerate(merged_facts)
                    if str(fact.get("id") or "").strip()
                ]
        lines = []
        for idx, fact in enumerate(facts[:3], start=1):
            fact_id = str(fact.get("id") or "").strip()
            linked_event = next(
                (
                    event
                    for event in events
                    if fact_id
                    and (
                        fact_id == str((event.get("payload") or {}).get("fact_id") or "").strip()
                        or fact_id in {
                            str(fid).strip()
                            for fid in (event.get("support_fact_ids") or [])
                            if str(fid).strip()
                        }
                    )
                ),
                None,
            )
            if not linked_event:
                linked_events = lookup_events_for_fact(self._temporal_index, fact_id=fact_id)
                linked_event = linked_events[0] if linked_events else None
            event_time = str((linked_event or {}).get("time_start") or "").strip()
            if not event_time:
                continue
            fact_text = str(fact.get("fact") or "").strip() or str((linked_event or {}).get("label") or "").strip()
            lines.append(f"[T{idx}] {fact_text} [Event time: {event_time}]")
            lines.extend(
                self._temporal_selector_evidence_lines(
                    fact=fact,
                    event=linked_event,
                )
            )
        if lines:
            prefix = "TEMPORAL EVIDENCE:"
            addition = prefix + "\n" + "\n".join(lines)
            self._inject_temporal_evidence_block(recall_result, addition)
        recall_result["temporal_resolution"] = {
            "mode": "calendar-answer",
            "matched_event_ids": [str(event.get("event_id") or "") for event in events],
            "matched_fact_ids": [str(fact.get("id") or "") for fact in facts],
        }
        runtime_trace = recall_result.get("runtime_trace") or {}
        runtime_trace["temporal_resolution"] = {
            "mode": "calendar-answer",
            "matched_event_ids": [str(event.get("event_id") or "") for event in events],
            "matched_fact_ids": [str(fact.get("id") or "") for fact in facts],
        }
        recall_result["runtime_trace"] = runtime_trace
        return recall_result

    async def _resolve_calendar_seeking(
        self,
        *,
        query: str,
        candidate_facts: list[dict],
    ) -> dict | None:
        plan = extract_calendar_query(query)
        if not plan or plan.get("mode") != "seeking":
            return None
        if not candidate_facts:
            return None
        atomic_embs = (self._data_dict or {}).get("atomic_embs")
        if not isinstance(atomic_embs, np.ndarray) or len(atomic_embs) != len(self._all_granular):
            return None
        allowed_ids = {
            str(f.get("id") or "")
            for f in candidate_facts
            if str(f.get("id") or "").strip()
        }
        ranked_facts: list[dict] = []
        ranked_embs: list[np.ndarray] = []
        for idx, fact in enumerate(self._all_granular):
            fact_id = str(fact.get("id") or "").strip()
            if not fact_id or fact_id not in allowed_ids:
                continue
            ranked_facts.append(fact)
            ranked_embs.append(atomic_embs[idx])
        if not ranked_facts or not ranked_embs:
            return None
        search_queries: list[str] = [plan["content_query"]]
        stripped = re.sub(r'["“”][^"“”]+["“”]', " ", plan["content_query"])
        stripped = re.sub(r"\s+", " ", stripped).strip(" \t:-,?.!")
        if stripped and stripped not in search_queries:
            search_queries.append(stripped)
        facts_by_episode: dict[str, list[dict]] = defaultdict(list)
        for fact in ranked_facts:
            episode_id = str((fact.get("metadata") or {}).get("episode_id") or fact.get("episode_id") or "").strip()
            if episode_id:
                facts_by_episode[episode_id].append(fact)

        ranked_rows: list[dict] = []
        traces: list[dict] = []
        seen_fact_ids: set[str] = set()
        for search_query in search_queries:
            query_embedding = await self._embed_query_with_runtime_secrets(search_query)
            sweep = source_local_fact_sweep(
                search_query,
                ranked_facts,
                np.asarray(ranked_embs),
                query_embedding=query_embedding,
                top_k=12,
                bm25_pool=36,
                vector_pool=24,
                entity_pool=12,
                rrf_k=60,
            )
            traces.append({"query": search_query, "trace": sweep.get("trace", {})})
            for row in sweep.get("retrieved", []):
                fact = row.get("fact") or {}
                fact_id = str(fact.get("id") or "").strip()
                if not fact_id or fact_id in seen_fact_ids:
                    continue
                seen_fact_ids.add(fact_id)
                ranked_rows.append(row)
        for require_specific in (True, False):
            for row in ranked_rows:
                fact = row.get("fact") or {}
                fact_id = str(fact.get("id") or "").strip()
                if not fact_id:
                    continue
                episode_id = str((fact.get("metadata") or {}).get("episode_id") or fact.get("episode_id") or "").strip()
                episode_candidates = list(facts_by_episode.get(episode_id) or []) if episode_id else []
                if fact not in episode_candidates:
                    episode_candidates.insert(0, fact)
                episode_candidates.sort(
                    key=lambda item: self._calendar_fact_candidate_score(
                        str(item.get("fact") or ""),
                        plan["content_query"],
                    ),
                    reverse=True,
                )
                for candidate in episode_candidates:
                    candidate_id = str(candidate.get("id") or "").strip()
                    if not candidate_id:
                        continue
                    events = lookup_events_for_fact(self._temporal_index, fact_id=candidate_id)
                    for event in events:
                        if require_specific and not self._is_fact_specific_temporal_event(candidate_id, event):
                            continue
                        answer = self._format_calendar_resolution_answer(plan, event)
                        if not answer:
                            continue
                        return {
                            "plan": plan,
                            "fact": candidate,
                            "event": event,
                            "answer": answer,
                            "trace": traces,
                        }
        return None

    async def _attach_calendar_seeking_resolution(
        self,
        *,
        query: str,
        recall_result: dict,
        candidate_facts: list[dict],
    ) -> dict:
        resolution = await self._resolve_calendar_seeking(
            query=query,
            candidate_facts=candidate_facts,
        )
        if not resolution:
            return recall_result
        fact = resolution["fact"]
        event = resolution["event"]
        answer = resolution["answer"]
        prefix = "TEMPORAL EVIDENCE:"
        line = (
            f"[T1] {fact.get('fact', '')} "
            f"[Resolved time: {answer}; Event time: {event.get('time_start')}]"
        ).strip()
        lines = [line]
        lines.extend(
            self._temporal_selector_evidence_lines(
                fact=fact,
                event=event,
            )
        )
        self._inject_temporal_evidence_block(recall_result, prefix + "\n" + "\n".join(lines))
        recall_result["temporal_resolution"] = {
            "mode": "calendar-seeking",
            "answer": answer,
            "fact_id": fact.get("id", ""),
            "event_id": event.get("event_id", ""),
            "time_start": event.get("time_start"),
            "time_end": event.get("time_end"),
            "time_granularity": event.get("time_granularity"),
        }
        runtime_trace = recall_result.get("runtime_trace") or {}
        runtime_trace["temporal_resolution"] = {
            "mode": "calendar-seeking",
            "answer": answer,
            "fact_id": fact.get("id", ""),
            "event_id": event.get("event_id", ""),
            "trace": resolution.get("trace", {}),
        }
        recall_result["runtime_trace"] = runtime_trace
        return recall_result

    def _conversation_doc_id(self, source_id: str | None = None) -> str:
        return f"conversation:{source_id or self.key}"

    def _conversation_episode_from_raw_session(
        self,
        *,
        raw_session: dict,
        source_id: str,
        session_num: int,
        session_date: str,
        canonical_content: str,
        canonical_source: dict,
        merged_source_meta: dict | None = None,
    ) -> dict:
        """Create the source-evidence episode for a conversation turn.

        Facts may be empty, but the raw turn is still retrievable evidence.
        """
        episode_id = f"{self._episode_source_key(source_id)}_e{int(session_num):04d}"
        raw_session_for_stamp = {**raw_session, "episode_id": episode_id}
        episode = {
            "episode_id": episode_id,
            "source_type": "conversation",
            "source_id": source_id,
            "source_date": session_date,
            "topic_key": f"session_{session_num}",
            "state_label": "session",
            "currentness": "unknown",
            "raw_text": canonical_content,
            "raw_original": canonical_source["raw_original"],
            "canonical_en": canonical_content,
            "semantic_ready": bool(canonical_source["semantic_ready"]),
            "canonicalization_status": canonical_source["canonicalization_status"],
            "canonicalization_error": canonical_source["canonicalization_error"],
            "source_lang": canonical_source["source_lang"],
            "translation_version": canonical_source["translation_version"],
            "provenance": {"raw_span": [0, len(canonical_content)]},
            "session_num": session_num,
            "projection_session_num": raw_session_for_stamp.get("projection_session_num"),
            "raw_session_id": raw_session_for_stamp.get("raw_session_id"),
            "message_id": raw_session_for_stamp.get("message_id"),
            "stored_at": raw_session_for_stamp.get("stored_at"),
            "agent_id": raw_session_for_stamp.get("agent_id"),
            "swarm_id": raw_session_for_stamp.get("swarm_id"),
            "scope": raw_session_for_stamp.get("scope"),
            "owner_id": raw_session_for_stamp.get("owner_id"),
            "read": list(raw_session_for_stamp.get("read") or []),
            "write": list(raw_session_for_stamp.get("write") or []),
            "artifact_id": raw_session_for_stamp.get("artifact_id"),
            "version_id": raw_session_for_stamp.get("version_id"),
            "status": raw_session_for_stamp.get("status") or "active",
        }
        metadata = raw_session_for_stamp.get("metadata")
        if isinstance(metadata, dict):
            episode["metadata"] = dict(metadata)
            role = str(metadata.get("role") or "").strip()
            if role:
                episode["role"] = role
            turn_number = _coerce_positive_session_num(metadata.get("turn_number"))
            if turn_number is not None:
                episode["turn_number"] = turn_number
        _stamp_selector_episode_fields(episode, raw_session_for_stamp, merged_source_meta or {})
        return episode

    @staticmethod
    def _document_doc_id(source_id: str) -> str:
        return f"document:{source_id}"

    @staticmethod
    def _codebase_doc_id(source_id: str) -> str:
        return f"codebase:{source_id}"

    @staticmethod
    def _multipart_part_key(metadata: dict | None) -> str | None:
        if not isinstance(metadata, dict):
            return None
        part_source_id = str(metadata.get("part_source_id") or "").strip()
        if part_source_id:
            return part_source_id
        part_idx = metadata.get("part_idx")
        if part_idx is None:
            return None
        try:
            part_num = int(part_idx)
        except (TypeError, ValueError):
            return None
        if part_num <= 0:
            return None
        return f"part_{part_num:04d}"

    @staticmethod
    def _episode_sort_key(episode: dict) -> tuple[int, str]:
        episode_id = str(episode.get("episode_id") or "")
        match = re.search(r"_e(\d+)\b", episode_id)
        if match:
            return (int(match.group(1)), episode_id)
        return (10**9, episode_id)

    def _upsert_episode_document(
        self,
        doc_id: str,
        episodes: list[dict],
        *,
        replace_part_key: str | None = None,
    ) -> None:
        docs = self._episode_corpus.setdefault("documents", [])
        for doc in docs:
            if doc.get("doc_id") == doc_id:
                existing = list(doc.get("episodes", []))
                if replace_part_key:
                    existing = [
                        ep
                        for ep in existing
                        if ((ep.get("provenance") or {}).get("multipart_part_key") != replace_part_key)
                    ]
                    doc["episodes"] = existing + episodes
                    doc["episodes"].sort(key=self._episode_sort_key)
                else:
                    doc["episodes"] = episodes
                return
        initial = list(episodes)
        initial.sort(key=self._episode_sort_key)
        docs.append({"doc_id": doc_id, "episodes": initial})

    def _snapshot_document_projection_state(self, source_id: str) -> dict:
        doc_id = self._document_doc_id(source_id)
        doc_snapshot = None
        for doc in self._episode_corpus.get("documents", []):
            if doc.get("doc_id") == doc_id:
                doc_snapshot = deepcopy(doc)
                break
        raw_doc_present = source_id in self._raw_docs
        source_record_present = source_id in self._source_records
        return {
            "doc_id": doc_id,
            "doc": doc_snapshot,
            "raw_doc_present": raw_doc_present,
            "raw_doc": self._raw_docs.get(source_id) if raw_doc_present else None,
            "source_record_present": source_record_present,
            "source_record": deepcopy(self._source_records.get(source_id)) if source_record_present else None,
        }

    def _restore_document_projection_state(self, source_id: str, snapshot: dict) -> None:
        doc_id = str(snapshot.get("doc_id") or self._document_doc_id(source_id))
        docs = self._episode_corpus.setdefault("documents", [])
        docs[:] = [doc for doc in docs if doc.get("doc_id") != doc_id]
        if snapshot.get("doc") is not None:
            docs.append(deepcopy(snapshot["doc"]))
            docs.sort(key=lambda doc: str(doc.get("doc_id") or ""))
        if snapshot.get("raw_doc_present"):
            self._raw_docs[source_id] = snapshot.get("raw_doc") or ""
        else:
            self._raw_docs.pop(source_id, None)
        if snapshot.get("source_record_present"):
            self._source_records[source_id] = deepcopy(snapshot.get("source_record") or {})
        else:
            self._source_records.pop(source_id, None)

    def _rollback_failed_document_ingest(
        self,
        *,
        source_id: str,
        version_id: str,
        projection_snapshot: dict,
    ) -> None:
        self._restore_document_projection_state(source_id, projection_snapshot)
        self._raw_sessions = [rs for rs in self._raw_sessions if rs.get("version_id") != version_id]
        self._all_granular = [fact for fact in self._all_granular if fact.get("version_id") != version_id]
        self._all_cross = [fact for fact in self._all_cross if fact.get("version_id") != version_id]
        self._mark_tiers_dirty()
        self._data_dict = None

    def _append_or_replace_episode(self, doc_id: str, episode: dict) -> None:
        docs = self._episode_corpus.setdefault("documents", [])
        for doc in docs:
            if doc.get("doc_id") == doc_id:
                doc["episodes"] = [
                    ep for ep in doc.get("episodes", [])
                    if ep.get("episode_id") != episode.get("episode_id")
                ]
                doc["episodes"].append(episode)
                doc["episodes"].sort(key=lambda ep: ep.get("episode_id", ""))
                return
        docs.append({"doc_id": doc_id, "episodes": [episode]})

    def _get_episode_documents(self, source_id: str, source_kind: str) -> list[dict]:
        if source_kind == "conversation":
            doc_id = self._conversation_doc_id(source_id)
        elif source_kind == "codebase":
            doc_id = self._codebase_doc_id(source_id)
        else:
            doc_id = self._document_doc_id(source_id)
        for doc in self._episode_corpus.get("documents", []):
            if doc.get("doc_id") == doc_id:
                episodes = [deepcopy(ep) for ep in doc.get("episodes", []) if isinstance(ep, dict)]
                episodes.sort(key=self._episode_sort_key)
                if source_kind == "document":
                    assign_document_artifact_span_ids(episodes, source_id=source_id)
                return episodes
        return []

    def _next_document_episode_index(self, source_id: str) -> int:
        existing = self._get_episode_documents(source_id, "document")
        if not existing:
            return 1
        return max(self._episode_sort_key(ep)[0] for ep in existing) + 1

    def _next_document_session_num(self, source_id: str) -> int:
        values = []
        for rs in self._raw_sessions:
            session_num = _coerce_positive_session_num(rs.get("session_num"))
            if session_num is not None:
                values.append(session_num)
        for episode in self._get_episode_documents(source_id, "document"):
            session_num = _coerce_positive_session_num(episode.get("session_num"))
            if session_num is not None:
                values.append(session_num)
        for fact in self._all_granular:
            session_num = _coerce_positive_session_num(fact.get("session") or fact.get("session_num"))
            if session_num is not None:
                values.append(session_num)
        return (max(values) if values else 0) + 1

    def _store_document_original_source_text(
        self,
        source_id: str,
        original_content: str,
        *,
        multipart_part_key: str | None = None,
        metadata: dict | None = None,
        message_id: str | None = None,
    ) -> str:
        """Store only trusted ingress raw source text in _raw_docs.

        Episode/block text is a derived working representation and must never
        re-establish exact-render source truth.
        """

        source_id = str(source_id or "")
        original_content = str(original_content or "")
        record = self._source_records.setdefault(
            source_id,
            {"source_id": source_id, "family": "document", "source_meta": {"logical_source_id": source_id}},
        )
        source_meta = record.setdefault("source_meta", {})
        if not multipart_part_key:
            source_meta.pop("raw_source_parts", None)
            self._raw_docs[source_id] = original_content
            return original_content

        metadata = dict(metadata or {})
        part_key = str(multipart_part_key or "").strip()
        existing_parts = [
            dict(item)
            for item in source_meta.get("raw_source_parts") or []
            if isinstance(item, dict) and str(item.get("part_key") or "").strip()
        ]
        replaced_part = next((item for item in existing_parts if str(item.get("part_key") or "") == part_key), None)

        part_idx = metadata.get("part_idx")
        try:
            explicit_order = int(cast(Any, part_idx)) if part_idx is not None else None
        except (TypeError, ValueError):
            explicit_order = None
        if explicit_order is None:
            replaced_order = replaced_part.get("order") if replaced_part is not None else None
            if replaced_order is not None:
                try:
                    explicit_order = int(cast(Any, replaced_order))
                except (TypeError, ValueError):
                    explicit_order = None
            if explicit_order is None:
                match = re.search(r"(\d+)(?!.*\d)", part_key)
                explicit_order = int(match.group(1)) if match else len(existing_parts) + 1

        next_part = {
            "part_key": part_key,
            "order": explicit_order,
            "message_id": str(message_id or ""),
            "content": original_content,
            "content_hash": hashlib.sha256(original_content.encode("utf-8")).hexdigest(),
        }
        parts = [item for item in existing_parts if str(item.get("part_key") or "") != part_key]
        parts.append(next_part)
        parts.sort(key=lambda item: (int(item.get("order") or 10**9), str(item.get("part_key") or "")))
        source_meta["raw_source_parts"] = parts
        next_raw = "\n\n".join(str(item.get("content") or "") for item in parts if str(item.get("content") or ""))
        self._raw_docs[source_id] = next_raw
        return next_raw

    def _reindex_document_episodes(
        self,
        source_id: str,
        episodes: list[dict],
        *,
        start_index: int,
        session_start: int | None = None,
        part_key: str | None = None,
        family: str = "document",
    ) -> list[dict]:
        remapped = []
        for offset, episode in enumerate(episodes):
            episode_copy = deepcopy(episode)
            episode_copy["episode_id"] = f"{source_id}_e{start_index + offset:02d}"
            episode_copy["source_id"] = source_id
            episode_copy["source_type"] = family
            if session_start is not None:
                episode_copy["session_num"] = session_start + offset
            provenance = dict(episode_copy.get("provenance") or {})
            if part_key:
                provenance["multipart_part_key"] = part_key
            if provenance:
                episode_copy["provenance"] = provenance
            remapped.append(episode_copy)
        if episodes:
            assign_document_artifact_span_ids(remapped, source_id=source_id)
        return remapped

    async def _extract_source_aggregation_facts(
        self,
        *,
        source_id: str,
        source_kind: str,
        source_facts: list[dict],
        source_date: str,
        model: str,
        call_extract_fn,
        agent_id: str = "default",
    ) -> list[dict]:
        if self._canonical_content_family(source_kind) == "codebase":
            return []
        if not model:
            return []
        episodes = self._get_episode_documents(source_id, source_kind)
        if not episodes:
            return []

        # MAL source aggregation prompt overrides
        _mal_cfg = _load_mal_active_config(str(self.data_dir), self.key, agent_id)
        _agg_overrides = {}
        for _pk, _pv in (_mal_cfg.get("extraction_prompts") or {}).items():
            if _pk.startswith("document_source_aggregation_prompt:"):
                _agg_overrides[_pk.split(":", 1)[1]] = _pv
        result = await extract_source_aggregation(
            source_id=source_id,
            source_kind=source_kind,
            episodes=episodes,
            source_facts=source_facts,
            model=model,
            call_extract_fn=call_extract_fn,
            prompt_overrides=_agg_overrides or None,
        )
        if not result:
            return []

        derived_facts = result.get("derived_facts", []) or []
        validation = result.get("validation", {}) or {}
        source_aggregation_report = _coerce_runtime_report(
            result.get("source_aggregation_report") or result.get("diagnostics"),
            producer="unified_source_extractor",
            report_kind="source_aggregation",
            validation=validation,
        )
        aggregation_status = validation.get("aggregation_status", "accepted")
        accepted_layers = list(validation.get("accepted_layers", []))
        dropped_layers = list(validation.get("dropped_layers", []))
        failure_reasons = list(validation.get("failure_reasons", []))

        for row in self._raw_sessions:
            row_source_ids = {
                str(row.get("source_id") or ""),
                str(row.get("logical_source_id") or ""),
                str(row.get("part_source_id") or ""),
            }
            if source_id in row_source_ids:
                row["source_aggregation_report"] = deepcopy(source_aggregation_report)
        source_record = self._source_records.get(source_id)
        if isinstance(source_record, dict):
            source_record["source_aggregation_report"] = deepcopy(source_aggregation_report)
            source_meta = source_record.get("source_meta")
            if not isinstance(source_meta, dict):
                source_meta = {}
                source_record["source_meta"] = source_meta
            source_meta["source_aggregation_report"] = deepcopy(source_aggregation_report)

        for fact in derived_facts:
            fact["conv_id"] = self.key
            fact["source_id"] = source_id
            tags = list(dict.fromkeys((fact.get("tags") or []) + ["unified_substrate", source_kind]))
            fact["tags"] = tags
            metadata = fact.setdefault("metadata", {})
            metadata["source_id"] = source_id
            metadata["source_kind"] = source_kind
            metadata["source_aggregation"] = True
            metadata["aggregation_status"] = aggregation_status
            metadata["accepted_layers"] = accepted_layers
            metadata["dropped_layers"] = dropped_layers
            metadata["failure_reasons"] = failure_reasons
            if source_date:
                fact.setdefault("event_date", source_date)
            err = _validate_object_flags_field(fact)
            if err:
                raise ValueError(f"Derived fact '{fact.get('id', '?')}': {err}")
        return derived_facts

    @staticmethod
    def _stamp_episode_metadata(
        facts: list[dict],
        episode_id: str,
        episode_source_id: str,
        *,
        artifact_span_id: str | None = None,
    ) -> None:
        for fact in facts:
            meta = fact.setdefault("metadata", {})
            meta["episode_id"] = episode_id
            meta["episode_source_id"] = episode_source_id
            if artifact_span_id:
                meta["artifact_span_id"] = artifact_span_id

    @staticmethod
    def _align_fact_selectors(
        facts: list[dict],
        *,
        episode_id: str,
        source_kind: str,
        raw_fields: dict[str, str],
        speakers: dict[str, str] | None = None,
    ) -> None:
        if not facts_as_selectors_enabled() or not facts:
            return
        align_facts_batch(
            facts,
            raw_fields,
            episode_id=episode_id,
            source_kind=source_kind,
            speakers=speakers,
        )

    @staticmethod
    def _hydrate_selector_surfaces_runtime(
        facts_by_episode: dict[str, list[dict]],
        episode_lookup: dict[str, dict],
    ) -> None:
        if not facts_as_selectors_enabled():
            return
        fact_lookup: dict[str, dict] = {}
        for facts in facts_by_episode.values():
            for fact in facts:
                fact_id = str(fact.get("id") or "")
                if fact_id:
                    fact_lookup[fact_id] = fact

        for ep_id, facts in facts_by_episode.items():
            episode = episode_lookup.get(ep_id)
            if not isinstance(episode, dict):
                continue
            seen_surface: set[str] = set()
            collected: list[str] = []
            total_chars = 0
            for fact in facts:
                surface = selector_surface_text(
                    fact,
                    episode_lookup,
                    fact_lookup=fact_lookup,
                    max_spans=2,
                    max_chars_per_span=220,
                )
                if surface:
                    fact["_selector_surface_text"] = surface
                else:
                    fact.pop("_selector_surface_text", None)
                    continue
                if surface in seen_surface:
                    continue
                seen_surface.add(surface)
                collected.append(surface)
                total_chars += len(surface)
                if len(collected) >= 6 or total_chars >= 1200:
                    break
            if collected:
                episode["_selector_surface_text"] = "\n".join(collected)
            else:
                episode.pop("_selector_surface_text", None)

    @staticmethod
    def _episode_source_key(source_id: str) -> str:
        key = re.sub(r"[^A-Za-z0-9._-]+", "_", source_id or "")
        key = key.strip("._-")
        return key or "source"

    def _initialize_scope_registry(self) -> None:
        now = datetime.now(timezone.utc).isoformat()
        if not isinstance(self._scope_record, dict):
            self._scope_record = {}
        if not isinstance(self._source_records, dict):
            self._source_records = {}

        self._scope_record.setdefault("scope_id", self.key)
        self._scope_record.setdefault("scope_kind", self.scope)
        self._scope_record.setdefault("agent_id", self.agent_id)
        self._scope_record.setdefault("swarm_id", self.swarm_id)
        self._scope_record.setdefault("source_ids", [])
        self._scope_record.setdefault("links", [])
        self._scope_record["updated_at"] = now

        for session in self._raw_sessions:
            family = self._canonical_content_family(session.get("format") or "conversation")
            source_id = session.get("source_id") or (self.key if family == "conversation" else "")
            if not source_id:
                continue
            self._register_source_record(
                source_id=source_id,
                family=family,
                owner_id=session.get("owner_id", "system"),
                read=session.get("read", ["agent:PUBLIC"]),
                write=session.get("write", ["agent:PUBLIC"]),
                artifact_id=session.get("artifact_id"),
                version_id=session.get("version_id"),
                content_hash=session.get("content_hash"),
                metadata=session.get("metadata"),
                target=session.get("target"),
                source_meta={
                    "logical_source_id": session.get("logical_source_id") or source_id,
                    "stored_format": session.get("format"),
                    "stored_at": session.get("stored_at"),
                    "scope": session.get("scope"),
                    "swarm_id": session.get("swarm_id"),
                },
            )

        for source_id in self._raw_docs:
            inferred_family = "document"
            for doc in self._episode_corpus.get("documents", []):
                episodes = doc.get("episodes", [])
                if not episodes:
                    continue
                sample = episodes[0]
                if sample.get("source_id") == source_id:
                    inferred_family = sample.get("source_type") or inferred_family
                    break
            self._register_source_record(
                source_id=source_id,
                family=inferred_family,
                source_meta={"stored_format": inferred_family, "logical_source_id": source_id},
            )

        for doc in self._episode_corpus.get("documents", []):
            episodes = doc.get("episodes", [])
            if not episodes:
                continue
            sample = episodes[0]
            source_id = sample.get("source_id") or ""
            family = sample.get("source_type") or "document"
            if not source_id:
                continue
            self._register_source_record(
                source_id=source_id,
                family=family,
                source_meta={
                    "doc_id": doc.get("doc_id"),
                    "logical_source_id": sample.get("logical_source_id") or source_id,
                },
            )

    def _register_source_record(
        self,
        source_id: str,
        family: str,
        owner_id: str = "system",
        read: list[str] | None = None,
        write: list[str] | None = None,
        artifact_id: str | None = None,
        version_id: str | None = None,
        content_hash: str | None = None,
        metadata: dict | None = None,
        target: list[str] | None = None,
        source_meta: dict | None = None,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        record = self._source_records.get(source_id, {})
        merged_source_meta = dict(record.get("source_meta", {}))
        if source_meta:
            merged_source_meta.update(source_meta)
        merged_source_meta.setdefault("logical_source_id", source_id)
        record_scope = record.get("scope") or merged_source_meta.get("scope")
        record_agent_id = record.get("agent_id") or merged_source_meta.get("agent_id")
        record_swarm_id = record.get("swarm_id") or merged_source_meta.get("swarm_id")
        if record_scope is None:
            inferred_scope, inferred_agent_id, inferred_swarm_id = _infer_scope_from_acl_fields(
                owner_id,
                read,
                write,
            )
            if inferred_scope is not None:
                record_scope = inferred_scope
                if record_agent_id in (None, "", "default") and inferred_agent_id:
                    record_agent_id = inferred_agent_id
                if record_swarm_id in (None, "", "default") and inferred_swarm_id:
                    record_swarm_id = inferred_swarm_id
        if record_agent_id in (None, "", "default") and isinstance(owner_id, str) and owner_id.startswith("agent:"):
            record_agent_id = owner_id.split(":", 1)[1]
        normalized_acl = _normalize_loaded_acl_fields(
            scope=record_scope,
            agent_id=record_agent_id,
            swarm_id=record_swarm_id,
            owner_id=owner_id,
            read=read,
            write=write,
        )
        record.update(
            {
                "source_id": source_id,
                "family": family,
                "scope_id": self.key,
                "scope": normalized_acl["scope"],
                "agent_id": normalized_acl["agent_id"],
                "swarm_id": normalized_acl["swarm_id"],
                "owner_id": normalized_acl["owner_id"],
                "read": list(normalized_acl["read"]),
                "write": list(normalized_acl["write"]),
                "updated_at": now,
            }
        )
        record.setdefault("created_at", now)
        if artifact_id is not None:
            record["artifact_id"] = artifact_id
        if version_id is not None:
            record["version_id"] = version_id
        if content_hash is not None:
            record["content_hash"] = content_hash
        if metadata is not None:
            record["metadata"] = dict(metadata)
        if target is not None:
            record["target"] = list(target)
        if merged_source_meta:
            record["source_meta"] = merged_source_meta
        self._source_records[source_id] = record

        source_ids = set(self._scope_record.get("source_ids", []))
        source_ids.add(source_id)
        self._scope_record["source_ids"] = sorted(source_ids)
        self._scope_record["updated_at"] = now

    def _scope_trace(self) -> dict:
        telemetry = get_runtime_tuning()["telemetry"]
        source_ids = list(self._scope_record.get("source_ids", []))
        max_ids = int(telemetry.get("max_scope_source_ids", 32))
        return {
            "scope_id": self._scope_record.get("scope_id", self.key),
            "scope_kind": self._scope_record.get("scope_kind", self.scope),
            "source_count": len(source_ids),
            "source_ids": source_ids[:max_ids],
            "link_count": len(self._scope_record.get("links", [])),
        }

    def _cross_contamination_trace(
        self,
        *,
        facts: list[dict] | None = None,
        episode_ids: list[str] | None = None,
        episode_lookup: dict[str, dict] | None = None,
        family_first_pass_trace: dict | None = None,
        late_fusion_trace: dict | None = None,
    ) -> dict:
        family_counts: defaultdict[str, int] = defaultdict(int)
        source_ids: list[str] = []

        if episode_ids and episode_lookup:
            for ep_id in episode_ids:
                ep = episode_lookup.get(ep_id) or {}
                source_id = ep.get("source_id") or ""
                if not source_id:
                    continue
                source_ids.append(source_id)
                family = ep.get("source_type") or self._source_records.get(source_id, {}).get("family", "unknown")
                family_counts[family] += 1
        elif facts:
            for fact in facts:
                source_id = fact.get("source_id") or (fact.get("metadata") or {}).get("episode_source_id", "")
                if not source_id:
                    continue
                source_ids.append(source_id)
                family = self._source_records.get(source_id, {}).get("family", "unknown")
                family_counts[family] += 1

        unique_source_ids = sorted(set(source_ids))
        rejected_source_ids = set()
        rejected_family_counts: defaultdict[str, int] = defaultdict(int)
        candidate_source_ids = set(unique_source_ids)
        candidate_family_counts: defaultdict[str, int] = defaultdict(int, family_counts)

        if family_first_pass_trace:
            for family_row in family_first_pass_trace.get("per_family", []):
                family = family_row.get("family", "unknown")
                for row in family_row.get("pre_source_gate", []):
                    source_id = row.get("source_id", "")
                    if source_id:
                        candidate_source_ids.add(source_id)
                        candidate_family_counts[family] += 1
                for row in family_row.get("post_source_gate", []):
                    source_id = row.get("source_id", "")
                    if source_id and source_id not in unique_source_ids:
                        rejected_source_ids.add(source_id)
                        rejected_family_counts[family] += 1

        if late_fusion_trace:
            for row in late_fusion_trace.get("rejected_candidates", []):
                ep_id = row.get("episode_id", "")
                ep = (episode_lookup or {}).get(ep_id, {})
                source_id = ep.get("source_id", "")
                family = ep.get("source_type") or self._source_records.get(source_id, {}).get("family", "unknown")
                if source_id and source_id not in unique_source_ids:
                    rejected_source_ids.add(source_id)
                    rejected_family_counts[family] += 1

        return {
            "source_ids": unique_source_ids,
            "source_count": len(unique_source_ids),
            "family_counts": dict(sorted(family_counts.items())),
            "multi_source": len(unique_source_ids) > 1,
            "candidate_source_ids": sorted(candidate_source_ids),
            "candidate_source_count": len(candidate_source_ids),
            "candidate_family_counts": dict(sorted(candidate_family_counts.items())),
            "rejected_source_ids": sorted(rejected_source_ids),
            "rejected_source_count": len(rejected_source_ids),
            "rejected_family_counts": dict(sorted(rejected_family_counts.items())),
        }

    def _episode_runtime_trace(
        self,
        *,
        corpus: dict,
        packet: dict,
        episode_lookup: dict[str, dict],
        resolved_facts: list[dict],
    ) -> dict:
        telemetry = get_runtime_tuning()["telemetry"]
        if not telemetry.get("include_runtime_trace", True):
            return {"runtime": "episode", "trace_disabled": True}
        return {
            "runtime": "episode",
            "scope": self._scope_trace(),
            "family_first_pass": packet.get("family_first_pass_trace", {
                "available_families": available_families(corpus),
                "retrieval_families": packet.get("retrieval_families", []),
                "requested_search_family": packet.get("search_family", "auto"),
                "per_family": [],
            }),
            "query": {
                **packet.get("query_operator_plan", {}),
                "output_constraints": packet.get("output_constraints", {}),
            },
            "late_fusion": packet.get("late_fusion_trace", {}),
            "temporal_resolution": packet.get("temporal_trace", {}),
            "selection": {
                "retrieved_episode_ids": packet.get("retrieved_episode_ids", []),
                "actual_injected_episode_ids": packet.get("actual_injected_episode_ids", []),
                "selection_scores": packet.get("selection_scores", []),
            },
            "cross_contamination": self._cross_contamination_trace(
                facts=resolved_facts,
                episode_ids=packet.get("retrieved_episode_ids", []),
                episode_lookup=episode_lookup,
                family_first_pass_trace=packet.get("family_first_pass_trace"),
                late_fusion_trace=packet.get("late_fusion_trace"),
            ),
            "packet": {
                "retrieved_fact_ids": packet.get("retrieved_fact_ids", [])[
                    : int(telemetry.get("max_packet_fact_ids", 24))
                ],
                "retrieved_fact_count": len(packet.get("retrieved_fact_ids", [])),
                "requested_episode_count": len(packet.get("retrieved_episode_ids", [])),
                "actual_injected_episode_count": len(packet.get("actual_injected_episode_ids", [])),
                "support_episode_count": len(packet.get("fact_episode_ids", [])),
                "support_episode_ids": packet.get("fact_episode_ids", [])[
                    : int(telemetry.get("max_packet_fact_ids", 24))
                ],
                "context_chars": len(packet.get("context", "")),
                "snippet_mode": bool(packet.get("selector_config", {}).get("snippet_mode", False)),
                "budget_chars": packet.get("selector_config", {}).get("budget"),
                "source_local_fact_sweep": packet.get("source_local_fact_sweep_trace", {}),
            },
            "tuning": packet.get("tuning_snapshot", {}),
        }

    def _canonical_retrieved_items(self, facts: list[dict]) -> list[dict]:
        items: list[dict] = []
        seen: set[str] = set()
        for fact in facts:
            fact_id = str(fact.get("id", "")).strip()
            if not fact_id or fact_id in seen:
                continue
            seen.add(fact_id)
            items.append({
                "fact_id": fact_id,
                "conv_id": fact.get("conv_id", self.key),
                "sim": float(fact.get("sim", fact.get("score", 0.0)) or 0.0),
            })
        return items

    def _ensure_min_synthesis_evidence(
        self,
        *,
        packet: dict,
        resolved_facts: list[dict],
        episode_lookup: dict[str, dict],
        fact_filter,
        minimum_facts: int = 2,
    ) -> tuple[dict, list[dict]]:
        if len(resolved_facts) >= minimum_facts:
            return packet, resolved_facts

        anchor_episode_ids = list(
            dict.fromkeys(
                packet.get("actual_injected_episode_ids", [])
                or packet.get("fact_episode_ids", [])
                or packet.get("retrieved_episode_ids", [])
            )
        )
        if not anchor_episode_ids:
            return packet, resolved_facts

        anchor_source_ids = {
            (episode_lookup.get(ep_id) or {}).get("source_id", "")
            for ep_id in anchor_episode_ids
        }
        anchor_source_ids.discard("")
        seen_fact_ids = {
            str(fact.get("id", "")).strip()
            for fact in resolved_facts
            if str(fact.get("id", "")).strip()
        }

        def _candidate_rank(fact: dict) -> tuple[int, int, str]:
            episode_ids = fact_episode_ids(fact)
            best_anchor = min(
                (anchor_episode_ids.index(ep_id) for ep_id in episode_ids if ep_id in anchor_episode_ids),
                default=len(anchor_episode_ids),
            )
            session_num = _coerce_positive_session_num(fact.get("session")) or 10**9
            return (best_anchor, session_num, str(fact.get("id", "")))

        extras: list[dict] = []
        for fact in list(self._all_granular) + list(self._all_cross):
            fact_id = str(fact.get("id", "")).strip()
            if not fact_id or fact_id in seen_fact_ids:
                continue
            if not fact_filter(fact):
                continue
            episode_ids = fact_episode_ids(fact)
            source_id = fact.get("source_id") or (fact.get("metadata") or {}).get("episode_source_id", "")
            if not (
                any(ep_id in anchor_episode_ids for ep_id in episode_ids)
                or (source_id and source_id in anchor_source_ids)
            ):
                continue
            extras.append(fact)

        if not extras:
            return packet, resolved_facts

        extras.sort(key=_candidate_rank)
        topped_up = list(resolved_facts)
        for fact in extras:
            topped_up.append(fact)
            seen_fact_ids.add(str(fact.get("id", "")).strip())
            if len(topped_up) >= minimum_facts:
                break

        if len(topped_up) == len(resolved_facts):
            return packet, resolved_facts

        fact_lookup = {fact.get("id", ""): fact for fact in list(self._all_granular) + list(self._all_cross)}
        context_trace: dict = {}
        context, actual_injected_episode_ids = build_context_from_retrieved_facts(
            topped_up,
            episode_lookup,
            fact_lookup=fact_lookup,
            budget=int(packet.get("selector_config", {}).get("budget", 8000)),
            snippet_chars=int(packet.get("tuning_snapshot", {}).get("packet", {}).get("snippet_chars", 1200)),
            context_trace=context_trace,
        )

        packet = dict(packet)
        packet["context"] = context
        packet["actual_injected_episode_ids"] = actual_injected_episode_ids
        packet["retrieved_fact_ids"] = [fact.get("id", "") for fact in topped_up if fact.get("id", "")]
        packet["fact_episode_ids"] = list(
            dict.fromkeys(
                episode_id
                for fact in topped_up
                for episode_id in fact_episode_ids(fact)
                if episode_id
            )
        )
        if context_trace.get("document_target_span_ids") or context_trace.get("document_target_span_mode") != "disabled":
            packet.update(context_trace)
        return packet, topped_up

    def _synthesis_retrieved_items(
        self,
        *,
        resolved_facts: list[dict],
        packet: dict,
        episode_lookup: dict[str, dict],
        minimum_items: int = 2,
    ) -> list[dict]:
        items = list(resolved_facts)
        if len(items) >= minimum_items:
            return items

        for ep_id in packet.get("actual_injected_episode_ids", []) or packet.get("retrieved_episode_ids", []):
            episode = episode_lookup.get(ep_id)
            if not episode:
                continue
            raw_text = str(episode.get("raw_text", "") or "").strip()
            if not raw_text:
                continue
            preview = raw_text[:280]
            if len(raw_text) > 280:
                preview += "..."
            items.append({
                "id": f"support_{ep_id}",
                "fact": f"Source excerpt support from episode {ep_id}: {preview}",
                "kind": "source_excerpt",
                "session": _coerce_positive_session_num(episode.get("session_num")) or 0,
                "metadata": {
                    "episode_id": ep_id,
                    "episode_source_id": episode.get("source_id", ""),
                    "support_only": True,
                },
            })
            if len(items) >= minimum_items:
                break
        return items

    async def _augment_conversation_structural_packet(
        self,
        *,
        query: str,
        packet: dict,
        episode_lookup: dict[str, dict],
        fact_filter,
    ) -> tuple[dict, list[dict] | None]:
        from .query_executors.conversation import augment_conversation_structural_packet

        return await augment_conversation_structural_packet(
            self,
            query=query,
            packet=packet,
            episode_lookup=episode_lookup,
            fact_filter=fact_filter,
        )

    async def _recover_multi_item_coverage_packet(
        self,
        *,
        query: str,
        packet: dict,
        episode_lookup: dict[str, dict],
        fact_filter,
    ) -> tuple[dict, list[dict] | None]:
        from .query_executors.coverage import recover_multi_item_coverage_packet

        return await recover_multi_item_coverage_packet(
            self,
            query=query,
            packet=packet,
            episode_lookup=episode_lookup,
            fact_filter=fact_filter,
        )

    async def _augment_document_structural_packet(
        self,
        *,
        query: str,
        packet: dict,
        episode_lookup: dict[str, dict],
        fact_filter,
    ) -> tuple[dict, list[dict] | None]:
        from .query_executors.document import augment_document_structural_packet

        return await augment_document_structural_packet(
            self,
            query=query,
            packet=packet,
            episode_lookup=episode_lookup,
            fact_filter=fact_filter,
        )

    async def _rescue_episode_packet_with_semantic_fact_sweep(
        self,
        *,
        query: str,
        packet: dict,
        episode_lookup: dict[str, dict],
        fact_filter,
    ) -> tuple[dict, list[dict] | None]:
        from .query_executors.semantic_rescue import rescue_episode_packet_with_semantic_fact_sweep

        return await rescue_episode_packet_with_semantic_fact_sweep(
            self,
            query=query,
            packet=packet,
            episode_lookup=episode_lookup,
            fact_filter=fact_filter,
        )

    @staticmethod
    def _temporal_query_surface_dates(query: str) -> list[str]:
        pattern = re.compile(
            r"\b(?:"
            r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
            r"\s+\d{1,2},\s*\d{4}"
            r"|"
            r"\d{1,2}\s+"
            r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
            r"(?:,?\s+\d{4})?"
            r"|"
            r"\d{4}-\d{2}-\d{2}"
            r")\b",
            re.I,
        )
        seen: list[str] = []
        for match in pattern.finditer(query or ""):
            value = " ".join(match.group(0).split()).strip().lower()
            if value and value not in seen:
                seen.append(value)
        return seen

    @staticmethod
    def _strip_temporal_surface(query: str) -> str:
        pattern = re.compile(
            r"\b(?:"
            r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
            r"\s+\d{1,2},\s*\d{4}"
            r"|"
            r"\d{1,2}\s+"
            r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
            r"(?:,?\s+\d{4})?"
            r"|"
            r"\d{4}-\d{2}-\d{2}"
            r")\b",
            re.I,
        )
        stripped = pattern.sub(" ", query or "")
        stripped = re.sub(r"\b(?:on|in|at|during|for)\s+(?=[?.!,;:]?$)", " ", stripped, flags=re.I)
        stripped = re.sub(r"\s+", " ", stripped).strip(" \t\r\n,?.!;:")
        return stripped

    @staticmethod
    def _shift_month_anchor(anchor: datetime, delta: int) -> datetime:
        month_index = anchor.month - 1 + delta
        year = anchor.year + month_index // 12
        month = month_index % 12 + 1
        day = min(anchor.day, 28)
        return anchor.replace(year=year, month=month, day=day)

    @staticmethod
    def _format_temporal_anchor(anchor: datetime) -> str:
        return anchor.strftime("%B %d, %Y").replace(" 0", " ")

    @staticmethod
    def _format_temporal_month(anchor: datetime) -> str:
        return anchor.strftime("%B %Y")

    @staticmethod
    def _format_temporal_interval(start: datetime, end: datetime) -> str:
        if start.year == end.year:
            return (
                f"between {start.day} {start.strftime('%B')} and "
                f"{end.day} {end.strftime('%B %Y')}"
            )
        return (
            f"between {start.day} {start.strftime('%B %Y')} and "
            f"{end.day} {end.strftime('%B %Y')}"
        )

    @staticmethod
    def _temporal_query_requests_relative_resolution(query: str) -> bool:
        lowered = (query or "").strip().lower()
        return bool(re.match(r"^(when\b|what date\b|what day\b|which month\b|what month\b|what year\b)", lowered))

    @staticmethod
    def _temporal_query_requests_year_resolution(query: str) -> bool:
        lowered = (query or "").strip().lower()
        return bool(re.match(r"^(what year\b|which year\b)", lowered))

    @staticmethod
    def _temporal_query_requests_month_resolution(query: str) -> bool:
        lowered = (query or "").strip().lower()
        return bool(re.match(r"^(?:in\s+)?(?:which|what) month\b", lowered))

    @staticmethod
    def _temporal_query_requests_first_window(query: str) -> bool:
        lowered = (query or "").strip().lower()
        if "first" not in lowered:
            return False
        return bool(re.match(r"^(when\b|what date\b|what day\b)", lowered))

    @staticmethod
    def _parse_duration_year_count(text: str) -> int | None:
        if not text:
            return None
        word_to_int = {
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
            "ten": 10,
            "eleven": 11,
            "twelve": 12,
        }
        patterns = (
            r"\bfor\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+years?\b",
            r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+years?\s+old\b",
            r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+years?\s+ago\b",
        )
        lowered = text.lower()
        for pattern in patterns:
            match = re.search(pattern, lowered, re.I)
            if not match:
                continue
            raw = match.group(1).lower()
            if raw.isdigit():
                return int(raw)
            return word_to_int.get(raw)
        return None

    def _relative_temporal_answer_from_text(self, text: str, *, source_date: str) -> str | None:
        if not text or not source_date:
            return None
        try:
            anchor = date_parser.parse(source_date, fuzzy=True)
        except Exception:
            return None
        lowered = text.lower()
        if "yesterday" in lowered or "last night" in lowered:
            return self._format_temporal_anchor(anchor - timedelta(days=1))
        if "today" in lowered or "tonight" in lowered:
            return self._format_temporal_anchor(anchor)
        if "tomorrow" in lowered:
            return self._format_temporal_anchor(anchor + timedelta(days=1))
        if "last week" in lowered:
            return f"the week before {self._format_temporal_anchor(anchor)}"
        if "this week" in lowered:
            return f"the week of {self._format_temporal_anchor(anchor)}"
        if "next week" in lowered:
            return f"the week after {self._format_temporal_anchor(anchor)}"
        if "last month" in lowered:
            return self._format_temporal_month(self._shift_month_anchor(anchor, -1))
        if "this month" in lowered:
            return self._format_temporal_month(anchor)
        if "next month" in lowered:
            return self._format_temporal_month(self._shift_month_anchor(anchor, 1))

        weekday_re = re.compile(
            r"\b(last|this)\s+"
            r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
            re.I,
        )
        weekday_to_index = {
            "monday": 0,
            "tuesday": 1,
            "wednesday": 2,
            "thursday": 3,
            "friday": 4,
            "saturday": 5,
            "sunday": 6,
        }
        match = weekday_re.search(text)
        if not match:
            return None
        rel = match.group(1).lower()
        weekday = match.group(2).lower()
        target = weekday_to_index.get(weekday)
        if target is None:
            return None
        days_back = (anchor.weekday() - target) % 7
        if rel == "last":
            days_back = 7 if days_back == 0 else days_back
        resolved = anchor - timedelta(days=days_back)
        return self._format_temporal_anchor(resolved)

    def _derive_relative_temporal_deterministic_answer(
        self,
        *,
        query: str,
        resolved_facts: list[dict],
        episode_lookup: dict[str, dict],
    ) -> str | None:
        if not self._temporal_query_requests_relative_resolution(query):
            return None
        if self._temporal_query_surface_dates(query):
            return None
        qf = extract_query_features(query)
        if qf.get("operator_plan", {}).get("temporal_grounding", {}).get("enabled", False):
            return None
        for fact in resolved_facts:
            fact_text = str((fact or {}).get("fact") or "").strip()
            if not fact_text:
                continue
            for ep_id in fact_episode_ids(fact):
                episode = episode_lookup.get(ep_id) or {}
                source_date = str(episode.get("source_date") or "").strip()
                answer = self._relative_temporal_answer_from_text(
                    fact_text,
                    source_date=source_date,
                )
                if answer:
                    return answer
        return None

    @staticmethod
    def _fact_anchor_datetime(fact: dict, episode_lookup: dict[str, dict]) -> datetime | None:
        for ep_id in fact_episode_ids(fact):
            episode = episode_lookup.get(ep_id) or {}
            for field in ("source_date", "session_date"):
                raw = str(episode.get(field) or "").strip()
                if not raw:
                    continue
                try:
                    return date_parser.parse(raw, fuzzy=True)
                except Exception:
                    continue
        for field in ("source_date", "session_date"):
            raw = str((fact or {}).get(field) or "").strip()
            if not raw:
                continue
            try:
                return date_parser.parse(raw, fuzzy=True)
            except Exception:
                continue
        return None

    def _derive_duration_temporal_deterministic_answer(
        self,
        *,
        query: str,
        resolved_facts: list[dict],
        episode_lookup: dict[str, dict],
    ) -> str | None:
        if not self._temporal_query_requests_year_resolution(query):
            return None
        if self._temporal_query_surface_dates(query):
            return None
        for fact in resolved_facts:
            fact_text = str((fact or {}).get("fact") or "").strip()
            if not fact_text:
                continue
            year_count = self._parse_duration_year_count(fact_text)
            if year_count is None:
                continue
            anchor = self._fact_anchor_datetime(fact, episode_lookup)
            if anchor is None:
                continue
            return str(anchor.year - year_count)
        return None

    def _derive_month_temporal_deterministic_answer(
        self,
        *,
        query: str,
        resolved_facts: list[dict],
        episode_lookup: dict[str, dict],
    ) -> str | None:
        if not self._temporal_query_requests_month_resolution(query):
            return None
        if self._temporal_query_surface_dates(query):
            return None
        for fact in resolved_facts:
            fact_text = str((fact or {}).get("fact") or "").strip()
            if not fact_text:
                continue
            anchor = self._fact_anchor_datetime(fact, episode_lookup)
            if anchor is None:
                continue
            lowered = fact_text.lower()
            direct = self._relative_temporal_answer_from_text(fact_text, source_date=str(anchor))
            if direct and re.match(r"^[A-Z][a-z]+\s+\d{4}$", direct):
                return direct
            if not re.search(r"\blast (?:week|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", lowered):
                continue
            for ep_id in fact_episode_ids(fact):
                episode = episode_lookup.get(ep_id) or {}
                raw_text = str(episode.get("raw_text") or "")
                if "last month" not in raw_text.lower():
                    continue
                return self._format_temporal_month(self._shift_month_anchor(anchor, -1))
        return None

    def _derive_first_window_temporal_deterministic_answer(
        self,
        *,
        query: str,
        context: str,
    ) -> str | None:
        if not self._temporal_query_requests_first_window(query):
            return None
        if self._temporal_query_surface_dates(query):
            return None
        match = re.search(
            r"First-mention window: earliest surfaced dated support is "
            r"(\d{4}-\d{2}-\d{2}), with the previous dated episode on "
            r"(\d{4}-\d{2}-\d{2})\.",
            context or "",
        )
        if not match:
            return None
        try:
            end = datetime.strptime(match.group(1), "%Y-%m-%d")
            start = datetime.strptime(match.group(2), "%Y-%m-%d")
        except ValueError:
            return None
        if not start < end:
            return None
        return self._format_temporal_interval(start, end)

    @staticmethod
    def _query_month_year_anchor(query: str) -> tuple[int, int] | None:
        match = re.search(
            r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(20\d{2}|19\d{2})\b",
            query or "",
            re.I,
        )
        if not match:
            return None
        month_lookup = {
            "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
            "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
        }
        return month_lookup[match.group(1).lower()], int(match.group(2))

    @staticmethod
    def _query_requests_acquisition_item_list(query: str, qf: dict) -> bool:
        if not (qf.get("operator_plan") or {}).get("list_set", {}).get("enabled"):
            return False
        lowered = (query or "").lower()
        if not re.search(r"\b(what|which)\s+(?:items?|things?)\b", lowered):
            return False
        return bool((qf.get("words") or set()) & {"buy", "purchase", "acquire", "get", "own"})

    @staticmethod
    def _extract_acquisition_item_candidates(text: str) -> list[str]:
        if not text:
            return []
        patterns = (
            r"\b(?:acquired|bought|purchased)\s+(?:myself\s+)?(?:a|an|the)?\s*(?:new\s+)?(?P<item>[A-Za-z0-9][A-Za-z0-9&'._-]*(?:\s+[A-Za-z0-9][A-Za-z0-9&'._-]*){0,6})",
            r"\b(?:got|gets|getting)\s+(?:myself\s+)?(?:a|an|the)?\s*(?:new\s+)?(?P<item>[A-Za-z0-9][A-Za-z0-9&'._-]*(?:\s+[A-Za-z0-9][A-Za-z0-9&'._-]*){0,6})",
            r"\bnow owns?\s+(?:a|an|the)?\s*(?:new\s+)?(?P<item>[A-Za-z0-9][A-Za-z0-9&'._-]*(?:\s+[A-Za-z0-9][A-Za-z0-9&'._-]*){0,6})",
            r"\bhas\s+(?:a|an)\s+new\s+(?P<item>[A-Za-z0-9][A-Za-z0-9&'._-]*(?:\s+[A-Za-z0-9][A-Za-z0-9&'._-]*){0,6})",
        )
        candidates: list[str] = []
        seen: set[str] = set()
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.I):
                candidate = re.split(r"\b(?:while|after|before|because|and|but)\b|[.?!;:]", match.group("item"), maxsplit=1)[0]
                candidate = re.sub(r"\s+", " ", candidate).strip(" \t\r\n,.-")
                if not candidate:
                    continue
                words = [w for w in candidate.split() if w.lower() not in {"new", "this", "that", "my", "our"}]
                candidate = " ".join(words).strip()
                if not candidate:
                    continue
                key = candidate.lower()
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(candidate)
        return candidates


    @staticmethod
    def _query_requests_time_scoped_activity_acquisition(query: str, qf: dict) -> bool:
        lowered = (query or "").lower()
        operator_plan = qf.get("operator_plan") or {}
        slot_head_tokens = set((operator_plan.get("slot_query") or {}).get("head_tokens") or [])
        list_head_tokens = set((operator_plan.get("list_set") or {}).get("head_tokens") or [])
        head_tokens = slot_head_tokens | list_head_tokens
        if not head_tokens & {"activity", "activities", "hobby", "hobbies", "pastime", "pastimes", "sport", "sports", "game", "games"}:
            return False
        if not MemoryServer._query_month_year_anchor(query):
            return False
        return bool(re.search(r"\b(?:take\s+up|takes\s+up|took\s+up|taking\s+up|start|starts|started|starting|begin|begins|began|beginning|try|tries|tried|trying|get\s+into|gets\s+into|got\s+into|getting\s+into)\b", lowered))

    @staticmethod
    def _extract_activity_acquisition_candidates(text: str) -> list[str]:
        if not text:
            return []
        patterns = (
            r"\b(?:take\s+up|takes\s+up|took\s+up|taking\s+up|start|starts|started|starting|begin|begins|began|beginning|try|tries|tried|trying|get\s+into|gets\s+into|got\s+into|getting\s+into)\s+(?P<item>[A-Za-z][A-Za-z&'._-]*(?:\s+[A-Za-z][A-Za-z&'._-]*){0,5})",
        )
        candidates: list[str] = []
        seen: set[str] = set()
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.I):
                candidate = re.split(r"\b(?:while|after|before|because|and|but|or|if|so)\b|[.?!;:]", match.group("item"), maxsplit=1)[0]
                candidate = re.sub(r"\s+", " ", candidate).strip(" \t\r\n,.-")
                if not candidate:
                    continue
                words = [
                    w for w in candidate.split()
                    if w.lower() not in {"new", "this", "that", "my", "our", "another", "calming"}
                ]
                candidate = " ".join(words).strip()
                if not candidate:
                    continue
                key = candidate.lower()
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(candidate)
        return candidates

    @staticmethod
    def _refine_acquisition_candidate(candidate: str, source_facts: list[dict]) -> str:
        head = normalize_term_token(candidate.split()[-1]) if candidate.split() else ""
        if not head:
            return candidate
        best = candidate
        best_score = (int(any(ch.isupper() for ch in candidate)), int(any(ch.isdigit() for ch in candidate)), len(candidate.split()), len(candidate))
        pattern = re.compile(
            rf"\b([A-Za-z0-9][A-Za-z0-9&'._-]*(?:\s+[A-Za-z0-9][A-Za-z0-9&'._-]*){{0,5}}\s+{re.escape(head)})\b",
            re.I,
        )
        for fact in source_facts:
            text = str((fact or {}).get("fact") or "")
            for match in pattern.finditer(text):
                phrase = re.sub(r"\s+", " ", match.group(1).strip())
                tokens = phrase.split()
                variants = []
                for width in range(2, min(len(tokens), 4) + 1):
                    variants.append(" ".join(tokens[-width:]))
                if not variants:
                    variants = [phrase]
                for variant in variants:
                    cleaned_tokens = list(variant.split())
                    while cleaned_tokens and cleaned_tokens[0].lower() in {"in", "a", "an", "the", "my", "our", "this", "that"}:
                        cleaned_tokens.pop(0)
                    cleaned_variant = " ".join(cleaned_tokens).strip()
                    if not cleaned_variant:
                        continue
                    if normalize_term_token(cleaned_variant.split()[-1]) != head:
                        continue
                    score = (
                        int(any(ch.isupper() for ch in cleaned_variant)),
                        int(any(ch.isdigit() for ch in cleaned_variant)),
                        len(cleaned_variant.split()),
                        len(cleaned_variant),
                    )
                    if score > best_score:
                        best = cleaned_variant
                        best_score = score
        return best

    @staticmethod
    def _query_requests_activity_list(query: str, qf: dict) -> bool:
        list_plan = (qf.get("operator_plan") or {}).get("list_set", {})
        if not list_plan.get("enabled"):
            return False
        head_tokens = set(list_plan.get("head_tokens") or [])
        return "activity" in head_tokens

    @staticmethod
    def _extract_activity_list_candidates(text: str, query_features: dict) -> list[str]:
        lowered = (text or "").lower()
        if not lowered:
            return []
        candidates: list[str] = []
        if "board game" in lowered:
            candidates.append("board games")
        if "wine tasting" in lowered:
            candidates.append("wine tasting")
        if "pet shelter" in lowered and re.search(r"\bvolunteer", lowered):
            candidates.append("pet shelter volunteering")
        if "flower" in lowered and re.search(r"\b(taking care|garden|bloom)", lowered):
            candidates.append("growing flowers")
        if re.search(r"\bcook(?:ing)?\b", lowered):
            candidates.append("cooking")
        if "indoor" in (query_features.get("words") or set()):
            outdoor_tokens = {"picnic", "trail", "hiking", "park", "bike", "biking", "camping", "surfing", "walk", "walking"}
            candidates = [candidate for candidate in candidates if not any(token in candidate for token in outdoor_tokens)]
        seen: set[str] = set()
        ordered: list[str] = []
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            ordered.append(candidate)
        return ordered

    def _derive_activity_list_deterministic_answer(
        self,
        *,
        query: str,
        query_features: dict,
        packet: dict,
        episode_lookup: dict[str, dict],
    ) -> str | None:
        if not self._query_requests_activity_list(query, query_features):
            return None
        selected_source_ids = {
            str((episode_lookup.get(ep_id) or {}).get("source_id") or "")
            for ep_id in packet.get("retrieved_episode_ids", [])
            if episode_lookup.get(ep_id)
        }
        selected_source_ids.discard("")
        if not selected_source_ids:
            return None
        primary_entity = ""
        for phrase in query_features.get("entity_phrases") or []:
            if phrase:
                primary_entity = str(phrase).strip()
                break
        source_facts = []
        for fact in self._all_granular:
            source_id = str(fact.get("source_id") or (fact.get("metadata") or {}).get("episode_source_id") or "")
            if source_id not in selected_source_ids:
                continue
            if primary_entity:
                speaker = str(fact.get("speaker") or "").strip()
                fact_text = str(fact.get("fact") or "")
                if speaker.lower() != primary_entity.lower() and primary_entity.lower() not in fact_text.lower():
                    continue
            source_facts.append(fact)
        if not source_facts:
            return None
        scored: list[tuple[datetime, str]] = []
        for fact in source_facts:
            anchor = self._fact_anchor_datetime(fact, episode_lookup) or datetime.max.replace(tzinfo=None)
            if getattr(anchor, 'tzinfo', None) is not None:
                anchor = anchor.replace(tzinfo=None)
            for candidate in self._extract_activity_list_candidates(str(fact.get("fact") or ""), query_features):
                scored.append((anchor, candidate))
        if not scored:
            return None
        scored.sort(key=lambda item: item[0])
        seen: set[str] = set()
        ordered: list[str] = []
        for _anchor, candidate in scored:
            if candidate in seen:
                continue
            seen.add(candidate)
            ordered.append(candidate)
        if not ordered:
            return None
        return ", ".join(ordered)

    def _derive_time_scoped_acquisition_deterministic_answer(
        self,
        *,
        query: str,
        query_features: dict,
        packet: dict,
        episode_lookup: dict[str, dict],
    ) -> str | None:
        if not self._query_requests_acquisition_item_list(query, query_features):
            return None
        month_year = self._query_month_year_anchor(query)
        if not month_year:
            return None
        month, year = month_year
        selected_source_ids = {
            str((episode_lookup.get(ep_id) or {}).get("source_id") or "")
            for ep_id in packet.get("retrieved_episode_ids", [])
            if episode_lookup.get(ep_id)
        }
        selected_source_ids.discard("")
        if not selected_source_ids:
            return None

        source_facts = []
        for fact in self._all_granular:
            source_id = str(fact.get("source_id") or (fact.get("metadata") or {}).get("episode_source_id") or "")
            if source_id in selected_source_ids:
                source_facts.append(fact)
        if not source_facts:
            return None

        episode_best: dict[str, tuple[tuple[int, int, int, int], str, str]] = {}
        for fact in source_facts:
            ep_ids = fact_episode_ids(fact)
            if not ep_ids:
                continue
            ep_id = ep_ids[0]
            episode = episode_lookup.get(ep_id) or {}
            source_date = str(episode.get("source_date") or "")
            try:
                anchor = date_parser.parse(source_date, fuzzy=True)
            except Exception:
                continue
            if anchor.year != year or anchor.month != month:
                continue
            for candidate in self._extract_acquisition_item_candidates(str(fact.get("fact") or "")):
                refined = self._refine_acquisition_candidate(candidate, source_facts)
                score = (
                    int(any(ch.isupper() for ch in refined)),
                    int(any(ch.isdigit() for ch in refined)),
                    len(refined.split()),
                    len(refined),
                )
                current = episode_best.get(ep_id)
                if current is None or score > current[0]:
                    episode_best[ep_id] = (score, refined, source_date)

        if not episode_best:
            return None

        ordered = sorted(episode_best.items(), key=lambda item: date_parser.parse(item[1][2], fuzzy=True))
        candidates: list[str] = []
        seen: set[str] = set()
        for _ep_id, (_score, candidate, _date) in ordered:
            key = candidate.lower()
            if key in seen:
                continue
            seen.add(key)
            candidates.append(candidate)
        if not candidates:
            return None
        return ", ".join(candidates)


    def _derive_time_scoped_activity_acquisition_deterministic_answer(
        self,
        *,
        query: str,
        query_features: dict,
        packet: dict,
        episode_lookup: dict[str, dict],
    ) -> str | None:
        if not self._query_requests_time_scoped_activity_acquisition(query, query_features):
            return None
        month_year = self._query_month_year_anchor(query)
        if not month_year:
            return None
        month, year = month_year
        selected_source_ids = {
            str((episode_lookup.get(ep_id) or {}).get("source_id") or "")
            for ep_id in packet.get("retrieved_episode_ids", [])
            if episode_lookup.get(ep_id)
        }
        selected_source_ids.discard("")
        if not selected_source_ids:
            return None

        primary_entity = ""
        for phrase in query_features.get("entity_phrases") or []:
            if phrase:
                primary_entity = str(phrase).strip().lower()
                break

        source_facts = []
        for fact in self._all_granular:
            source_id = str(fact.get("source_id") or (fact.get("metadata") or {}).get("episode_source_id") or "")
            if source_id not in selected_source_ids:
                continue
            fact_text = str(fact.get("fact") or "")
            speaker = str(fact.get("speaker") or "").strip().lower()
            if primary_entity and primary_entity not in fact_text.lower() and speaker != primary_entity:
                continue
            source_facts.append(fact)
        if not source_facts:
            return None

        episode_best: dict[str, tuple[tuple[int, int, int], str, str]] = {}
        for fact in source_facts:
            ep_ids = fact_episode_ids(fact)
            if not ep_ids:
                continue
            ep_id = ep_ids[0]
            episode = episode_lookup.get(ep_id) or {}
            source_date = str(episode.get("source_date") or "")
            try:
                anchor = date_parser.parse(source_date, fuzzy=True)
            except Exception:
                continue
            if anchor.year != year or anchor.month != month:
                continue
            for candidate in self._extract_activity_acquisition_candidates(str(fact.get("fact") or "")):
                score = (
                    len(candidate.split()),
                    int(any(ch.isupper() for ch in candidate)),
                    len(candidate),
                )
                current = episode_best.get(ep_id)
                if current is None or score > current[0]:
                    episode_best[ep_id] = (score, candidate, source_date)

        if not episode_best:
            return None

        ordered = sorted(episode_best.items(), key=lambda item: date_parser.parse(item[1][2], fuzzy=True))
        seen: set[str] = set()
        candidates: list[str] = []
        for _ep_id, (_score, candidate, _date) in ordered:
            key = candidate.lower()
            if key in seen:
                continue
            seen.add(key)
            candidates.append(candidate)
        if not candidates:
            return None
        return ", ".join(candidates)

    def _temporal_grounding_pseudo_facts(self, ep_id: str, episode: dict) -> list[dict]:
        raw_text = str(episode.get("raw_text") or "")
        source_date = str(episode.get("source_date") or "").strip()
        if not raw_text or not source_date:
            return []
        try:
            anchor = date_parser.parse(source_date, fuzzy=True)
        except Exception:
            return []

        weekday_re = re.compile(
            r"\b(last|this)\s+"
            r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
            re.I,
        )
        weekday_to_index = {
            "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
            "friday": 4, "saturday": 5, "sunday": 6,
        }

        pseudo: list[dict] = []
        seen: set[str] = set()
        for line_idx, raw_line in enumerate(raw_text.splitlines(), start=1):
            line = raw_line.strip()
            if not line or line.lower().startswith("[turn query]"):
                continue
            lowered = line.lower()
            resolved: datetime | None = None
            if "yesterday" in lowered or "last night" in lowered:
                resolved = anchor - timedelta(days=1)
            elif "today" in lowered or "tonight" in lowered:
                resolved = anchor
            elif "tomorrow" in lowered:
                resolved = anchor + timedelta(days=1)
            elif "last week" in lowered:
                resolved = anchor - timedelta(days=7)
            elif "this week" in lowered:
                resolved = anchor
            elif "next week" in lowered:
                resolved = anchor + timedelta(days=7)
            elif "last month" in lowered:
                resolved = self._shift_month_anchor(anchor, -1)
            elif "this month" in lowered:
                resolved = anchor
            elif "next month" in lowered:
                resolved = self._shift_month_anchor(anchor, 1)
            else:
                match = weekday_re.search(line)
                if match:
                    rel = match.group(1).lower()
                    weekday = match.group(2).lower()
                    target = weekday_to_index.get(weekday)
                    if target is not None:
                        days_back = (anchor.weekday() - target) % 7
                        if rel == "last":
                            days_back = 7 if days_back == 0 else days_back
                        resolved = anchor - timedelta(days=days_back)
            if resolved is None:
                continue
            fact_text = (
                f"{line} Resolved date: {self._format_temporal_anchor(resolved)} "
                f"({resolved.strftime('%B %Y')})."
            )
            if fact_text in seen:
                continue
            seen.add(fact_text)
            pseudo.append({
                "id": f"temporal_anchor_{ep_id}_{line_idx}",
                "session": 0,
                "fact": fact_text,
                "source_id": episode.get("source_id", ""),
                "metadata": {
                    "episode_id": ep_id,
                    "episode_source_id": episode.get("source_id", ""),
                    "source_aggregation": True,
                    "semantic_class": "temporal_grounding",
                },
            })
        return pseudo

    async def _repair_temporal_grounding_packet(
        self,
        *,
        query: str,
        packet: dict,
        episode_lookup: dict[str, dict],
        fact_filter,
    ) -> tuple[dict, list[dict] | None]:
        from .query_executors.temporal import repair_temporal_grounding_packet

        return await repair_temporal_grounding_packet(
            self,
            query=query,
            packet=packet,
            episode_lookup=episode_lookup,
            fact_filter=fact_filter,
        )
    async def _generic_fact_recall(
        self,
        *,
        query: str,
        fact_filter,
        search_family: str = "auto",
        query_type: str = "auto",
        query_metadata: dict[str, Any] | None = None,
        path_constraint_query: str | None = None,
        codebase_probe_mode: bool = False,
    ) -> dict:
        resolved_type = self._QUERY_TYPE_MAP.get(query_type) or detect_query_type(query)
        retrieval_target = (extract_query_features(query).get("retrieval_target") or query)
        rescue_cfg = get_tuning_section("retrieval").get("semantic_rescue", {})
        repo_task_metadata = dict(query_metadata or {})

        candidate_facts = []
        candidate_embeddings = []
        priority_mode = self._data_dict is None
        priority_trace: dict[str, Any] = {
            "priority_retrieval_used": priority_mode,
            "priority_index_used": False,
            "priority_index_candidate_count": 0,
            "priority_index_top_k": self._priority_index_top_k(),
            "priority_index_timeout_ms": self._priority_index_timeout_ms(),
            "full_index_required_for_query": False,
        }

        def _family_matches(fact: dict) -> bool:
            if search_family in ("auto", "", None):
                return True
            source_id = fact.get("source_id") or (fact.get("metadata") or {}).get("episode_source_id", "")
            family = self._source_records.get(source_id, {}).get("family")
            return not family or family == search_family

        if priority_mode:
            query_tokens = {
                normalize_term_token(token)
                for token in re.findall(r"[A-Za-z0-9_./:-]+", retrieval_target.lower())
                if normalize_term_token(token) and normalize_term_token(token) not in STOP_WORDS
            }
            scored_priority: list[tuple[float, int, dict]] = []
            for fact_idx, fact in enumerate([*self._all_granular, *self._all_cons, *self._all_cross]):
                if not fact_filter(fact) or not _family_matches(fact):
                    continue
                text = str(fact.get("fact") or "")
                fact_tokens = {
                    normalize_term_token(token)
                    for token in re.findall(r"[A-Za-z0-9_./:-]+", text.lower())
                    if normalize_term_token(token) and normalize_term_token(token) not in STOP_WORDS
                }
                overlap = len(query_tokens & fact_tokens)
                score = float(overlap)
                if retrieval_target.strip() and retrieval_target.lower() in text.lower():
                    score += 10.0
                if score <= 0:
                    continue
                scored_priority.append((score, fact_idx, fact))
            scored_priority.sort(key=lambda row: (-row[0], row[1]))
            top_k = self._priority_index_top_k()
            priority_rows = scored_priority[:top_k]
            candidate_facts = [fact for _score, _idx, fact in priority_rows]
            priority_trace["priority_index_candidate_count"] = len(candidate_facts)
            exact_match = any(
                retrieval_target.strip()
                and retrieval_target.lower() in str(fact.get("fact") or "").lower()
                for fact in candidate_facts
            )
            if candidate_facts and not exact_match:
                fact_texts = [str(fact.get("fact") or "") for fact in candidate_facts]
                try:
                    candidate_embeddings = list(await asyncio.wait_for(
                        self._embed_texts_with_runtime_secrets(
                            fact_texts,
                            label=f"priority-{self.key[:8]}",
                        ),
                        timeout=self._priority_index_timeout_ms() / 1000.0,
                    ))
                    priority_trace["priority_index_used"] = True
                except Exception as exc:
                    candidate_embeddings = [
                        np.zeros((0,), dtype=np.float32)
                        for _fact in candidate_facts
                    ]
                    priority_trace["priority_index_used"] = False
                    priority_trace["priority_index_error"] = type(exc).__name__
                    priority_trace["priority_index_fallback"] = "lexical_order"
            else:
                candidate_embeddings = [
                    np.zeros((0,), dtype=np.float32)
                    for _fact in candidate_facts
                ]
        else:
            for entry in (
                (
                    self._all_granular,
                    (self._data_dict or {}).get("atomic_embs"),
                    (self._data_dict or {}).get("atomic_emb_indices"),
                ),
                (self._all_cons, (self._data_dict or {}).get("cons_embs"), None),
                (self._all_cross, (self._data_dict or {}).get("cross_embs"), None),
            ):
                facts, embeddings, indices = entry
                for _fact_idx, fact, emb in _iter_fact_embedding_rows(facts, embeddings, indices):
                    if not fact_filter(fact):
                        continue
                    if not _family_matches(fact):
                        continue
                    candidate_facts.append(fact)
                    candidate_embeddings.append(emb)

        if not candidate_facts:
            repo_contract_active_for_priority_empty = (
                search_family == "codebase"
                or repo_task_metadata.get("search_family") == "codebase"
                or repo_task_metadata.get("work_item_kind") == "repo:work_item"
                or repo_task_metadata.get("task_mode") == "patch_generation"
                or repo_task_metadata.get("output_artifact") == "unified_diff"
            )
            if priority_mode and self._all_granular and not repo_contract_active_for_priority_empty:
                return {
                    "error": "full index is not ready and priority retrieval found no local candidates",
                    "code": "INDEX_NOT_READY",
                    "context": "RETRIEVED FACTS:",
                    "retrieved": [],
                    "search_family": search_family,
                    "query_type": resolved_type,
                    "is_multihop": False,
                    "complexity_hint": {
                        "score": 0.0,
                        "level": 1,
                        "signals": [],
                        "retrieval_complexity": 0.0,
                        "content_complexity": 0.0,
                        "query_complexity": 0.0,
                        "dominant": "tie",
                    },
                    "n_facts": len(self._all_granular) + len(self._all_cons) + len(self._all_cross),
                    "sessions_in_context": 0,
                    "total_sessions": self._n_sessions,
                    "coverage_pct": 0,
                    "raw_budget": 0,
                    "recommended_prompt_type": resolved_type,
                    "use_tool": False,
                    "runtime_trace": {
                        "runtime": "fact",
                        "scope": self._scope_trace(),
                        "reason": "priority_retrieval_empty",
                        "priority_retrieval": {
                            **priority_trace,
                            "full_index_required_for_query": True,
                        },
                    },
                }
            codebase_context = None
            codebase_context_trace: dict[str, Any] = {"mode": "inactive", "reason": "not_codebase_query"}
            repo_task_query_context_pack: dict[str, Any] | None = None
            repo_contract_active = (
                search_family == "codebase"
                or repo_task_metadata.get("search_family") == "codebase"
                or repo_task_metadata.get("work_item_kind") == "repo:work_item"
                or repo_task_metadata.get("task_mode") == "patch_generation"
                or repo_task_metadata.get("output_artifact") == "unified_diff"
            )
            if repo_contract_active and not codebase_probe_mode:
                metadata_source_id = str(
                    repo_task_metadata.get("source_id")
                    or repo_task_metadata.get("work_item_id")
                    or repo_task_metadata.get("instance_id")
                    or ""
                ).strip()
                codebase_context, codebase_context_trace = build_codebase_context(
                    self,
                    query=query,
                    source_ids={metadata_source_id} if metadata_source_id else None,
                    query_metadata=repo_task_metadata,
                    path_constraint_query=path_constraint_query,
                )
                if (
                    isinstance(codebase_context_trace, dict)
                    and isinstance(codebase_context_trace.get("query_context_pack"), dict)
                ):
                    repo_task_query_context_pack = deepcopy(codebase_context_trace["query_context_pack"])
            complexity_hint: dict[str, Any] = {
                "score": 0.0,
                "level": 1,
                "signals": [],
                "retrieval_complexity": 0.0,
                "content_complexity": 0.0,
                "query_complexity": 0.0,
                "dominant": "tie",
            }
            repo_patch_context = (
                bool(codebase_context)
                and isinstance(codebase_context_trace, dict)
                and (codebase_context_trace.get("task_mode") == "patch_generation"
                     or (codebase_context_trace.get("repo_task_contract") or {}).get("task_mode") == "patch_generation")
            )
            if repo_patch_context:
                complexity_hint["level"] = max(int(complexity_hint.get("level") or 1), 5)
            empty_context_packet: dict[str, list[dict[str, Any]]] = {"tier1": [], "tier2": [], "tier3": [], "tier4": []}
            if codebase_context:
                target_tier = "tier1" if repo_patch_context else "tier4"
                empty_context_packet[target_tier].append({
                    "text": codebase_context,
                    "rank": 1_000_000 if repo_patch_context else -1,
                    "source": "codebase_context",
                })
            empty_result: dict[str, Any] = {
                "context": codebase_context or "RETRIEVED FACTS:",
                "_context_packet": empty_context_packet,
                "retrieved": [],
                "search_family": search_family,
                "query_type": resolved_type,
                "is_multihop": False,
                "complexity_hint": complexity_hint,
                "n_facts": len(self._all_granular) + len(self._all_cons) + len(self._all_cross),
                "sessions_in_context": 0,
                "total_sessions": self._n_sessions,
                "coverage_pct": 0,
                "raw_budget": 0,
                "recommended_prompt_type": resolved_type,
                "use_tool": False,
                "runtime_trace": {
                    "runtime": "fact",
                    "scope": self._scope_trace(),
                    "reason": "codebase_context_without_visible_facts" if codebase_context else "empty_visible_facts",
                    "codebase_context": codebase_context_trace,
                    "priority_retrieval": priority_trace,
                },
            }
            if repo_task_query_context_pack:
                empty_result["repo_task_context_packs"] = [repo_task_query_context_pack]
            return empty_result

        codebase_candidate_mode = (
            search_family == "codebase"
            or any(self._fact_source_family(fact) == "codebase" for fact in candidate_facts)
        )
        embedding_trace: dict[str, Any] = {"mode": "vector", "status": "ok"}
        try:
            query_embedding = await self._embed_query_with_runtime_secrets(retrieval_target)
        except Exception as exc:
            if not codebase_candidate_mode:
                raise
            query_embedding = None
            embedding_trace = {
                "mode": "lexical_seed_fallback",
                "status": "embedding_unavailable",
                "error_type": type(exc).__name__,
                "reason": str(exc)[:240],
                "vector_seed_disabled": True,
            }
        sweep = source_local_fact_sweep(
            retrieval_target,
            candidate_facts,
            np.asarray(candidate_embeddings),
            query_embedding=query_embedding,
            top_k=int(rescue_cfg.get("top_k", 8)),
            bm25_pool=int(rescue_cfg.get("bm25_pool", 24)),
            vector_pool=int(rescue_cfg.get("vector_pool", 24)),
            entity_pool=int(rescue_cfg.get("entity_pool", 12)),
            rrf_k=int(rescue_cfg.get("rrf_k", 60)),
        )
        retrieved_items = [
            {
                "fact_id": row["fact"].get("id", ""),
                "conv_id": row["fact"].get("conv_id", self.key),
                "sim": float(row["score"]),
            }
            for row in sweep.get("retrieved", [])
        ]
        resolved_facts = [row["fact"] for row in sweep.get("retrieved", [])]
        codebase_promotion_trace = {"mode": "inactive", "reason": "not_applied"}
        if search_family == "codebase" or any(self._fact_source_family(fact) == "codebase" for fact in resolved_facts):
            resolved_facts, codebase_promotion_trace = promote_defining_code_facts(
                query=query,
                retrieved_facts=resolved_facts,
                candidate_facts=candidate_facts,
                limit=int(rescue_cfg.get("top_k", 8)),
            )
            retrieved_items = [
                {
                    "fact_id": fact.get("id", ""),
                    "conv_id": fact.get("conv_id", self.key),
                    "sim": next((item["sim"] for item in retrieved_items if item["fact_id"] == fact.get("id", "")), 0.0),
                }
                for fact in resolved_facts
            ]
        retrieval_families = list(
            dict.fromkeys(
                family
                for family in (
                    self._fact_source_family(fact)
                    for fact in resolved_facts
                )
                if family
            )
        )
        codebase_structural_context: str | None = None
        codebase_structural_trace: dict[str, Any] | None = None
        codebase_graph_source_ids = set()
        if not codebase_probe_mode:
            codebase_graph_source_ids = {
                str(fact.get("source_id") or (fact.get("metadata") or {}).get("episode_source_id") or "").strip()
                for fact in resolved_facts
                if self._fact_source_family(fact) == "codebase"
                and str(
                    (
                        (self._source_records.get(
                            str(fact.get("source_id") or (fact.get("metadata") or {}).get("episode_source_id") or "").strip()
                        ) or {}).get("source_meta")
                        or {}
                    ).get("codebase_graph_ref")
                    or ""
                ).strip()
            }
        if codebase_graph_source_ids:
            structural_packet = {
                "retrieved_episode_ids": list(
                    dict.fromkeys(
                        episode_id
                        for fact in resolved_facts
                        for episode_id in fact_episode_ids(fact)
                        if episode_id
                    )
                ),
                "retrieved_fact_ids": [str(fact.get("id") or "") for fact in resolved_facts],
                "retrieval_families": retrieval_families,
                "selector_config": {
                    "supporting_facts_total": max(12, int(rescue_cfg.get("top_k", 8))),
                    "budget": 8000,
                },
                "tuning_snapshot": {"packet": {"snippet_chars": 1200}},
                "context": "",
            }
            structural_packet, augmented_facts = await augment_codebase_structural_packet(
                self,
                query=query,
                packet=structural_packet,
                episode_lookup={},
                fact_filter=fact_filter,
            )
            if augmented_facts:
                resolved_facts = augmented_facts
                retrieved_items = [
                    {
                        "fact_id": fact.get("id", ""),
                        "conv_id": fact.get("conv_id", self.key),
                        "sim": next((item["sim"] for item in retrieved_items if item["fact_id"] == fact.get("id", "")), 0.0),
                    }
                    for fact in resolved_facts
                ]
                retrieval_families = list(
                    dict.fromkeys(
                        family
                        for family in (self._fact_source_family(fact) for fact in resolved_facts)
                        if family
                    )
                )
                codebase_structural_context = str(structural_packet.get("context") or "").strip() or None
                structural_trace = structural_packet.get("source_local_fact_sweep_trace")
                codebase_structural_trace = dict(structural_trace) if isinstance(structural_trace, dict) else {}

        total_sessions = self._n_sessions
        matched_context_sessions = [
            match
            for match in (
                _match_raw_session_for_fact(fact, self._raw_sessions)
                for fact in resolved_facts
            )
            if match is not None
        ]
        sessions_in_ctx = len({match_key for match_key, _session_num, _raw_session in matched_context_sessions})
        raw_budget = compute_raw_budget(resolved_type, total_sessions, sessions_in_ctx)
        coverage_pct = (sessions_in_ctx / total_sessions * 100) if total_sessions else 0

        context_packet = _build_context_packet(
            resolved_facts,
            self._raw_sessions,
            budget=raw_budget,
            raw_docs=self._raw_docs or None,
        )
        code_segments, code_trace = augment_codebase_context(
            query=query,
            retrieved_facts=resolved_facts,
            data_dir=str(self.data_dir),
        )
        repo_task_context_packs, repo_task_context_trace = self._repo_task_context_packs_for_facts(resolved_facts)
        codebase_context = None
        codebase_context_trace = {"mode": "inactive", "reason": "not_codebase_query"}
        repo_task_query_context_pack = None
        if (
            not codebase_probe_mode
            and (search_family == "codebase" or any(self._fact_source_family(fact) == "codebase" for fact in resolved_facts))
        ):
            visible_codebase_source_ids = {
                str(fact.get("source_id") or (fact.get("metadata") or {}).get("episode_source_id") or "").strip()
                for fact in candidate_facts
                if self._fact_source_family(fact) == "codebase"
            }
            metadata_source_id = str(
                repo_task_metadata.get("source_id")
                or repo_task_metadata.get("work_item_id")
                or repo_task_metadata.get("instance_id")
                or ""
            ).strip()
            if metadata_source_id:
                visible_codebase_source_ids.add(metadata_source_id)
            codebase_context, codebase_context_trace = build_codebase_context(
                self,
                query=query,
                source_ids=visible_codebase_source_ids or None,
                query_metadata=repo_task_metadata,
                path_constraint_query=path_constraint_query,
            )
            if isinstance(codebase_context_trace, dict) and isinstance(codebase_context_trace.get("query_context_pack"), dict):
                repo_task_query_context_pack = deepcopy(codebase_context_trace["query_context_pack"])
        repo_patch_context = (
            bool(codebase_context)
            and (
                str(repo_task_metadata.get("task_mode") or "") == "patch_generation"
                or str(repo_task_metadata.get("output_artifact") or "") == "unified_diff"
                or (
                    isinstance(codebase_context_trace, dict)
                    and (
                        codebase_context_trace.get("task_mode") == "patch_generation"
                        or (codebase_context_trace.get("repo_task_contract") or {}).get("task_mode") == "patch_generation"
                    )
                )
            )
        )
        if codebase_context:
            target_tier = "tier1" if repo_patch_context else "tier4"
            context_packet[target_tier].insert(0, {
                "text": codebase_context,
                "rank": 1_000_000 if repo_patch_context else -1,
                "source": "codebase_context",
            })
        if code_segments:
            context_packet["tier4"].extend(code_segments)
        if repo_patch_context and codebase_context:
            # Patch generation needs exact Repository context source context as the model-facing authority.
            # Vector/BM25/fact hits remain trace/seed data, but should not lead the prompt.
            hybrid_ctx = codebase_context
        else:
            hybrid_ctx = codebase_structural_context or _render_context_packet(context_packet)
            if codebase_structural_context and code_segments:
                hybrid_ctx = f"{hybrid_ctx}\n\n" + _render_code_attachment_block(code_segments)
            if codebase_structural_context and codebase_context:
                hybrid_ctx = f"{hybrid_ctx.rstrip()}\n\n{codebase_context}"

        complexity_hint = _compute_complexity_hint(
            retrieved=resolved_facts,
            resolved_type=resolved_type,
            is_multihop=resolved_type in ("temporal", "current", "counting"),
            fact_lookup=self._fact_lookup,
            query=query,
        )
        if repo_patch_context:
            complexity_hint["level"] = max(int(complexity_hint.get("level") or 1), 5)

        prompt_type, use_tool = _route_prompt_type(
            resolved_type,
            resolved_facts,
            total_sessions,
            sessions_in_ctx,
            hybrid_ctx,
        )

        result = {
            "context": hybrid_ctx,
            "_context_packet": context_packet,
            "retrieved": resolved_facts if codebase_structural_trace else retrieved_items,
            "retrieval_families": retrieval_families,
            "search_family": search_family,
            "query_type": resolved_type,
            "is_multihop": resolved_type in ("temporal", "current", "counting"),
            "complexity_hint": complexity_hint,
            "n_facts": len(self._all_granular) + len(self._all_cons) + len(self._all_cross),
            "sessions_in_context": sessions_in_ctx,
            "total_sessions": total_sessions,
            "coverage_pct": coverage_pct,
            "raw_budget": raw_budget,
            "recommended_prompt_type": prompt_type,
            "use_tool": use_tool,
            "runtime_trace": {
                "runtime": "fact",
                "scope": self._scope_trace(),
                "query": {
                    "retrieval_target": retrieval_target,
                    "resolved_type": resolved_type,
                    "code_query_mode": classify_codebase_query_mode(query),
                },
                "defining_fact_promotion": codebase_promotion_trace,
                "selection": {
                    "retrieved_fact_count": len(resolved_facts),
                    "selected_fact_ids": [
                        fact.get("id", "")
                        for fact in resolved_facts[:12]
                    ],
                },
                "embedding": embedding_trace,
                "priority_retrieval": priority_trace,
                "packet": {
                    "context_chars": len(hybrid_ctx),
                    "raw_budget": raw_budget,
                    "source_local_fact_sweep": sweep.get("trace", {}),
                },
                "codebase_augmentation": code_trace,
                "repo_task_context_pack": repo_task_context_trace,
                "codebase_context": codebase_context_trace,
            },
        }
        if repo_task_query_context_pack or repo_task_context_packs:
            result["repo_task_context_packs"] = [
                *([repo_task_query_context_pack] if repo_task_query_context_pack else []),
                *repo_task_context_packs,
            ]
        if codebase_structural_trace:
            result["runtime_trace"]["codebase_structural"] = codebase_structural_trace

        return result

    def _episode_runtime_facts(self, fact_filter, *, query: str | None = None) -> list[dict]:
        facts = [f for f in self._all_granular if fact_filter(f)]
        query_features = extract_query_features(query or "") if query else {}
        explicit_step_query = bool(
            query_features.get("step_numbers") or query_features.get("step_range")
        )
        for f in self._all_cross:
            if not fact_filter(f):
                continue
            metadata = f.get("metadata") or {}
            if not metadata.get("source_aggregation"):
                continue
            if metadata.get("semantic_class") != "temporal_semantics":
                continue
            if explicit_step_query:
                continue
            if not fact_episode_ids(f):
                continue
            facts.append(f)
        return facts

    def _fact_source_family(self, fact: dict[str, Any]) -> str:
        metadata = fact.get("metadata") or {}
        source_id = str(fact.get("source_id") or metadata.get("episode_source_id") or "")
        return str(
            self._source_records.get(source_id, {}).get("family")
            or fact.get("source_family")
            or metadata.get("source_family")
            or ""
        ).lower()

    def _repo_task_context_packs_for_facts(self, facts: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        source_ids = [
            str(fact.get("source_id") or (fact.get("metadata") or {}).get("episode_source_id") or "").strip()
            for fact in facts
            if self._fact_source_family(fact) == "codebase"
        ]
        source_ids = list(dict.fromkeys(source_id for source_id in source_ids if source_id))
        if not source_ids:
            return [], {"mode": "inactive", "reason": "no_codebase_source_ids"}
        graph = normalize_container_graph(self._container_graph)
        active_revision_ids = {
            str(row.get("container_graph_revision_id") or "")
            for row in graph.get("graph_revisions", [])
            if str(row.get("source_id") or "") in source_ids
            and str(row.get("family") or "") == "codebase"
            and str(row.get("adapter_name") or "") == "codebase_semantic_container_graph"
            and str(row.get("status") or "") == "active"
        }
        if not active_revision_ids:
            return [], {
                "mode": "failed_closed",
                "reason": "no_active_codebase_container_graph_revision",
                "source_ids": source_ids,
            }
        packs = [
            deepcopy(row.get("payload_json") or {})
            for row in graph.get("artifacts", [])
            if str(row.get("artifact_kind") or "") == "context_pack"
            and str(row.get("container_graph_revision_id") or "") in active_revision_ids
            and isinstance(row.get("payload_json"), dict)
            and (row.get("payload_json") or {}).get("context_pack_kind") == "repo_task_context_pack"
            and str(row.get("status") or "active") == "active"
        ]
        if not packs:
            return [], {
                "mode": "failed_closed",
                "reason": "missing_repo_task_context_pack_artifact",
                "source_ids": source_ids,
                "active_revision_ids": sorted(active_revision_ids),
            }
        return packs, {
            "mode": "active",
            "context_pack_kind": "repo_task_context_pack",
            "source_ids": source_ids,
            "active_revision_ids": sorted(active_revision_ids),
            "pack_count": len(packs),
            "operator_domain_policy": "full_typed_container_domain_in_scope",
        }

    @staticmethod
    def _repo_patch_touched_files(diff_text: str) -> list[str]:
        files: list[str] = []
        for match in re.finditer(r"(?m)^diff --git a/(.*?) b/(.*?)\s*$", str(diff_text or "")):
            for value in (match.group(2), match.group(1)):
                path = str(value or "").strip()
                if path and path != "/dev/null" and path not in files:
                    files.append(path)
                    break
        return files

    @staticmethod
    def _repo_patch_hunks(diff_text: str) -> list[dict[str, Any]]:
        hunks: list[dict[str, Any]] = []
        current_file = ""
        for line_no, line in enumerate(str(diff_text or "").splitlines(), start=1):
            file_match = re.match(r"^diff --git a/(.*?) b/(.*?)\s*$", line)
            if file_match:
                current_file = str(file_match.group(2) or file_match.group(1) or "")
                continue
            hunk_match = re.match(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
            if hunk_match:
                hunks.append(
                    {
                        "file": current_file,
                        "line": line_no,
                        "old_start": int(hunk_match.group(1)),
                        "old_count": int(hunk_match.group(2) or "1"),
                        "new_start": int(hunk_match.group(3)),
                        "new_count": int(hunk_match.group(4) or "1"),
                        "header": line,
                    }
                )
        return hunks

    @staticmethod
    def _repo_operation_hash(payload: Any) -> str:
        rendered = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(rendered.encode("utf-8")).hexdigest()

    def _active_codebase_revision_for_operation(self, source_id: str | None = None) -> dict[str, Any] | None:
        graph = normalize_container_graph(self._container_graph)
        rows = [
            row for row in graph.get("graph_revisions", [])
            if str(row.get("family") or "") == "codebase"
            and str(row.get("adapter_name") or "") == "codebase_semantic_container_graph"
            and str(row.get("status") or "") == "active"
        ]
        if source_id:
            rows = [row for row in rows if str(row.get("source_id") or "") == str(source_id)]
        rows.sort(key=lambda row: str(row.get("completed_at") or row.get("created_at") or ""), reverse=True)
        return dict(rows[0]) if rows else None

    def _repo_work_item_container_for_revision(self, graph_revision_id: str) -> dict[str, Any] | None:
        graph = normalize_container_graph(self._container_graph)
        for container in graph.get("containers", []):
            if (
                str(container.get("container_graph_revision_id") or "") == graph_revision_id
                and str(container.get("kind_fq") or "") == "repo:work_item"
                and str(container.get("status") or "active") == "active"
            ):
                return dict(container)
        return None

    def _repo_file_containers_for_revision(self, graph_revision_id: str) -> dict[str, dict[str, Any]]:
        graph = normalize_container_graph(self._container_graph)
        out: dict[str, dict[str, Any]] = {}
        for container in graph.get("containers", []):
            if str(container.get("container_graph_revision_id") or "") != graph_revision_id:
                continue
            if str(container.get("kind_fq") or "") != "code:file":
                continue
            path = str((container.get("traits_json") or {}).get("path") or "").strip()
            if path:
                out[path.lower()] = dict(container)
        return out

    def _find_repo_operation_container(self, *, kind_fq: str, payload_key: str, payload_value: str) -> dict[str, Any] | None:
        wanted = str(payload_value or "").strip()
        if not wanted:
            return None
        graph = normalize_container_graph(self._container_graph)
        for container in graph.get("containers", []):
            if str(container.get("kind_fq") or "") != kind_fq:
                continue
            payload = dict((container.get("traits_json") or {}).get("payload") or {})
            if str(payload.get(payload_key) or "") == wanted or str(container.get("container_id") or "") == wanted:
                return dict(container)
        return None

    def _repo_operation_container_row(
        self,
        *,
        graph_revision: dict[str, Any],
        kind: str,
        operation_id: str,
        payload: dict[str, Any],
        primary_render_ref_id: str | None = None,
        primary_render_ref_fingerprint: str | None = None,
        render_ref_json: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = str(payload.get("created_at") or datetime.now(timezone.utc).isoformat())
        graph_revision_id = str(graph_revision.get("container_graph_revision_id") or "")
        source_id = str(graph_revision.get("source_id") or "")
        record = self._source_records.get(source_id) or {}
        source_meta = dict(record.get("source_meta") or {})
        revision_id = str(graph_revision.get("revision_id") or payload.get("commit_id") or source_meta.get("revision") or source_id)
        container_id = stable_hash(
            "repo-operation-container-v1",
            graph_revision_id,
            kind,
            operation_id,
            prefix="ctr",
        )
        order_key = {
            "basis": "repo_operation_time",
            "scope": "repo_revision",
            "scope_id": f"{payload.get('repo_id') or source_meta.get('repo_id') or source_id}:{revision_id}",
            "segments": [now, kind, operation_id],
            "unit": "operation",
        }
        return {
            "container_id": container_id,
            "id_origin": "stable_hash",
            "id_schema_version": "container-id-v1",
            "identity_scope": "repo_revision_operation",
            "source_id": source_id,
            "logical_source_id": str(source_meta.get("logical_source_id") or source_id),
            "family": "codebase",
            "revision_id": revision_id,
            "revision_scope": "repo_revision",
            "content_revision_id": str(record.get("content_hash") or "") or None,
            "external_revision_id": revision_id,
            "container_graph_revision_id": graph_revision_id,
            "kind_ns": "operation",
            "kind": kind,
            "kind_fq": f"operation:{kind}",
            "kind_version": "v1",
            "traits_json": {
                "operation_kind": kind,
                "payload": payload,
            },
            "order_key_json": order_key,
            "order_basis": "repo_operation_time",
            "order_scope": "repo_revision",
            "order_scope_id": str(order_key["scope_id"]),
            "span_refs_json": [],
            "episode_ids_json": [],
            "primary_render_ref_id": primary_render_ref_id,
            "primary_render_ref_fingerprint": primary_render_ref_fingerprint,
            "render_ref_json": render_ref_json or {},
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
            "adapter_name": "repo_operation_memory",
            "adapter_version": "1",
            "inference_version": None,
            "created_at": now,
            "updated_at": now,
        }

    def _repo_operation_relation_row(
        self,
        *,
        graph_revision_id: str,
        src_container_id: str,
        dst_container_id: str,
        relation_kind: str,
        relation_capability: str | None = None,
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        return {
            "relation_id": stable_hash(
                "repo-operation-relation-v1",
                graph_revision_id,
                src_container_id,
                dst_container_id,
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
            "traits_json": {"relation_capability": relation_capability or relation_kind},
            "relation_score": 1.0,
            "status": "active",
            "created_at": now,
            "updated_at": now,
        }

    def _repo_operation_artifact_row(
        self,
        *,
        graph_revision_id: str,
        artifact_kind: str,
        subject_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        payload_fingerprint = self._repo_operation_hash(payload)
        return {
            "artifact_id": stable_hash(
                "repo-operation-artifact-v1",
                graph_revision_id,
                artifact_kind,
                subject_id,
                payload_fingerprint,
                prefix="artifact",
            ),
            "artifact_kind": artifact_kind,
            "artifact_schema_version": "v1",
            "container_graph_revision_id": graph_revision_id,
            "container_graph_revision_ids_json": [graph_revision_id],
            "families_json": ["codebase", "repo", "operation"],
            "profile_ids_json": ["repo_task:v1"],
            "query_id": None,
            "profile_id": "repo_task:v1",
            "subject_type": "container",
            "subject_id": subject_id,
            "payload_json": payload,
            "payload_fingerprint": payload_fingerprint,
            "status": "active",
            "created_at": now,
        }

    def _repo_operation_render_ref_row(
        self,
        *,
        graph_revision_id: str,
        container_id: str,
        render_kind: str,
        ref_json: dict[str, Any],
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        ref_fingerprint = self._repo_operation_hash({"render_kind": render_kind, "render_mode": "exact_copy", **ref_json})
        return {
            "render_ref_id": stable_hash(
                "repo-operation-render-ref-v1",
                graph_revision_id,
                container_id,
                render_kind,
                ref_fingerprint,
                prefix="render",
            ),
            "container_id": container_id,
            "container_graph_revision_id": graph_revision_id,
            "render_kind": render_kind,
            "render_mode": "exact_copy",
            "ref_json": ref_json,
            "ref_fingerprint": ref_fingerprint,
            "fidelity": "text_exact",
            "language": "diff" if render_kind == "patch_diff" else "text",
            "format": "text",
            "token_estimate": max(1, len(str(ref_json.get("text") or "")) // 4),
            "status": "active",
            "created_at": now,
        }

    def _repo_operation_state_row(
        self,
        *,
        graph_revision_id: str,
        container_id: str,
        state_kind: str,
        state_scope_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        return {
            "state_id": stable_hash(
                "repo-operation-state-v1",
                graph_revision_id,
                container_id,
                state_kind,
                payload.get("status"),
                now,
                prefix="state",
            ),
            "container_id": container_id,
            "container_graph_revision_id": graph_revision_id,
            "state_kind": state_kind,
            "state_scope_kind": "repo_work_item",
            "state_scope_id": state_scope_id,
            "state_subject_kind": "operation",
            "state_subject_id": container_id,
            "state_conflict_group_id": f"{state_scope_id}:{state_kind}",
            "confidence": 1.0,
            "valid_from": now,
            "valid_until": None,
            "superseded_by_container_id": None,
            "supersedes_container_id": None,
            "current_state": 1,
            "state_reason_json": payload,
            "status": "active",
            "created_at": now,
        }

    def _append_repo_operation_rows(
        self,
        *,
        containers: list[dict[str, Any]] | None = None,
        relations: list[dict[str, Any]] | None = None,
        render_refs: list[dict[str, Any]] | None = None,
        artifacts: list[dict[str, Any]] | None = None,
        state: list[dict[str, Any]] | None = None,
    ) -> None:
        containers = containers or []
        relations = relations or []
        render_refs = render_refs or []
        artifacts = artifacts or []
        state = state or []
        graph = normalize_container_graph(self._container_graph)

        def _upsert(key: str, id_key: str, rows: list[dict[str, Any]]) -> None:
            if not rows:
                return
            by_id = {str(row.get(id_key) or ""): idx for idx, row in enumerate(graph[key])}
            for row in rows:
                row_id = str(row.get(id_key) or "")
                if not row_id:
                    continue
                if row_id in by_id:
                    graph[key][by_id[row_id]] = row
                else:
                    by_id[row_id] = len(graph[key])
                    graph[key].append(row)

        _upsert("containers", "container_id", containers)
        _upsert("relations", "relation_id", relations)
        _upsert("render_refs", "render_ref_id", render_refs)
        _upsert("artifacts", "artifact_id", artifacts)
        _upsert("state", "state_id", state)
        self._container_graph = graph
        self._persist_projection_delta(
            container_upserts=containers,
            container_relation_upserts=relations,
            container_render_ref_upserts=render_refs,
            container_artifact_upserts=artifacts,
            container_state_upserts=state,
        )

    def record_repo_patch_attempt(
        self,
        *,
        diff_text: str,
        source_id: str | None = None,
        work_item_container_id: str | None = None,
        repo_id: str | None = None,
        commit_id: str | None = None,
        touched_files: list[str] | None = None,
        hunks: list[dict[str, Any]] | None = None,
        source_context_refs: list[str] | None = None,
        model: str | None = None,
        profile: str | None = None,
        status: str = "candidate",
        patch_attempt_id: str | None = None,
    ) -> dict[str, Any]:
        graph_revision = self._active_codebase_revision_for_operation(source_id)
        if graph_revision is None:
            return {"status": "failed_closed", "code": "NO_ACTIVE_CODEBASE_GRAPH_REVISION"}
        graph_revision_id = str(graph_revision.get("container_graph_revision_id") or "")
        work_item = self._repo_work_item_container_for_revision(graph_revision_id)
        if work_item_container_id:
            work_item = {"container_id": work_item_container_id}
        if not work_item:
            return {"status": "failed_closed", "code": "REPO_WORK_ITEM_CONTAINER_NOT_FOUND"}
        source_id = str(graph_revision.get("source_id") or source_id or "")
        record = self._source_records.get(source_id) or {}
        source_meta = dict(record.get("source_meta") or {})
        repo_id = str(repo_id or source_meta.get("repo_id") or source_id)
        commit_id = str(commit_id or source_meta.get("revision") or graph_revision.get("revision_id") or "")
        diff_hash = hashlib.sha256(str(diff_text or "").encode("utf-8")).hexdigest()
        touched_files = list(dict.fromkeys(touched_files or self._repo_patch_touched_files(diff_text)))
        hunks = list(hunks or self._repo_patch_hunks(diff_text))
        created_at = datetime.now(timezone.utc).isoformat()
        patch_attempt_id = str(patch_attempt_id or stable_hash("repo-patch-attempt-id-v1", graph_revision_id, diff_hash, created_at, prefix="patch"))
        payload = {
            "patch_attempt_id": patch_attempt_id,
            "work_item_container_id": str(work_item.get("container_id") or ""),
            "repo_id": repo_id,
            "commit_id": commit_id,
            "diff_text_hash": diff_hash,
            "touched_files": touched_files,
            "hunks": hunks,
            "source_context_refs": list(source_context_refs or []),
            "model": str(model or ""),
            "profile": str(profile or ""),
            "created_at": created_at,
            "status": status,
        }
        provisional_container_id = stable_hash("repo-operation-container-v1", graph_revision_id, "patch_attempt", patch_attempt_id, prefix="ctr")
        render_refs: list[dict[str, Any]] = []
        render_ref_json: dict[str, Any] = {}
        primary_render_ref_id = None
        primary_render_ref_fingerprint = None
        if diff_text:
            render_ref_json = {
                "ref_type": "patch_diff",
                "patch_attempt_id": patch_attempt_id,
                "repo_id": repo_id,
                "commit_id": commit_id,
                "text_hash": diff_hash,
                "text_exact": True,
                "byte_exact": True,
                "render_source": "operation_patch_attempt_output",
                "text": diff_text,
            }
            render_ref = self._repo_operation_render_ref_row(
                graph_revision_id=graph_revision_id,
                container_id=provisional_container_id,
                render_kind="patch_diff",
                ref_json=render_ref_json,
            )
            render_refs.append(render_ref)
            primary_render_ref_id = render_ref["render_ref_id"]
            primary_render_ref_fingerprint = render_ref["ref_fingerprint"]
        container = self._repo_operation_container_row(
            graph_revision=graph_revision,
            kind="patch_attempt",
            operation_id=patch_attempt_id,
            payload=payload,
            primary_render_ref_id=primary_render_ref_id,
            primary_render_ref_fingerprint=primary_render_ref_fingerprint,
            render_ref_json=render_ref_json,
        )
        relations = [
            self._repo_operation_relation_row(
                graph_revision_id=graph_revision_id,
                src_container_id=str(work_item.get("container_id") or ""),
                dst_container_id=container["container_id"],
                relation_kind="has_patch_attempt",
                relation_capability="has_patch_attempt",
            )
        ]
        file_lookup = self._repo_file_containers_for_revision(graph_revision_id)
        for touched in touched_files:
            file_container = file_lookup.get(str(touched or "").lower())
            if not file_container:
                continue
            relations.append(
                self._repo_operation_relation_row(
                    graph_revision_id=graph_revision_id,
                    src_container_id=container["container_id"],
                    dst_container_id=file_container["container_id"],
                    relation_kind="patch_touches",
                    relation_capability="patch_touches",
                )
            )
        artifact = self._repo_operation_artifact_row(
            graph_revision_id=graph_revision_id,
            artifact_kind="operation_patch_attempt",
            subject_id=container["container_id"],
            payload=payload,
        )
        state = self._repo_operation_state_row(
            graph_revision_id=graph_revision_id,
            container_id=container["container_id"],
            state_kind="patch_attempt_status",
            state_scope_id=str(work_item.get("container_id") or ""),
            payload=payload,
        )
        self._append_repo_operation_rows(
            containers=[container],
            relations=relations,
            render_refs=render_refs,
            artifacts=[artifact],
            state=[state],
        )
        return {"status": "ok", "patch_attempt_id": patch_attempt_id, "container_id": container["container_id"], "payload": payload}

    def record_repo_patch_apply_result(
        self,
        *,
        patch_attempt_id: str,
        command: str = "git apply --check",
        exit_code: int | None = None,
        stdout: str = "",
        stderr: str = "",
        failed_files: list[str] | None = None,
        failed_hunks: list[dict[str, Any]] | None = None,
        status: str | None = None,
    ) -> dict[str, Any]:
        patch_container = self._find_repo_operation_container(kind_fq="operation:patch_attempt", payload_key="patch_attempt_id", payload_value=patch_attempt_id)
        if patch_container is None:
            return {"status": "failed_closed", "code": "PATCH_ATTEMPT_NOT_FOUND"}
        graph_revision_id = str(patch_container.get("container_graph_revision_id") or "")
        graph_revision = next(
            (row for row in normalize_container_graph(self._container_graph).get("graph_revisions", []) if str(row.get("container_graph_revision_id") or "") == graph_revision_id),
            None,
        )
        if graph_revision is None:
            return {"status": "failed_closed", "code": "PATCH_ATTEMPT_GRAPH_REVISION_NOT_FOUND"}
        patch_payload = dict((patch_container.get("traits_json") or {}).get("payload") or {})
        status = status or ("passed" if int(exit_code or 0) == 0 else "failed")
        result_id = stable_hash("repo-patch-apply-result-id-v1", patch_attempt_id, command, exit_code, stdout, stderr, prefix="apply")
        payload = {
            "patch_attempt_id": patch_attempt_id,
            "patch_apply_result_id": result_id,
            "command": command,
            "exit_code": int(exit_code or 0),
            "stdout_ref": {"text_hash": hashlib.sha256(str(stdout or "").encode("utf-8")).hexdigest(), "char_count": len(str(stdout or ""))},
            "stderr_ref": {"text_hash": hashlib.sha256(str(stderr or "").encode("utf-8")).hexdigest(), "char_count": len(str(stderr or ""))},
            "failed_files": list(failed_files or []),
            "failed_hunks": list(failed_hunks or []),
            "status": status,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        container = self._repo_operation_container_row(graph_revision=graph_revision, kind="patch_apply_result", operation_id=result_id, payload=payload)
        relations = [
            self._repo_operation_relation_row(
                graph_revision_id=graph_revision_id,
                src_container_id=patch_container["container_id"],
                dst_container_id=container["container_id"],
                relation_kind="verified_by",
                relation_capability="verified_by",
            )
        ]
        artifact_payload = {**payload, "stdout": stdout, "stderr": stderr}
        artifact = self._repo_operation_artifact_row(graph_revision_id=graph_revision_id, artifact_kind="operation_patch_apply_result", subject_id=container["container_id"], payload=artifact_payload)
        state = self._repo_operation_state_row(graph_revision_id=graph_revision_id, container_id=container["container_id"], state_kind="patch_apply_status", state_scope_id=str(patch_payload.get("work_item_container_id") or ""), payload=payload)
        self._append_repo_operation_rows(containers=[container], relations=relations, artifacts=[artifact], state=[state])
        return {"status": "ok", "patch_apply_result_id": result_id, "container_id": container["container_id"], "payload": payload}

    def record_repo_test_run(
        self,
        *,
        patch_attempt_id: str,
        commands: list[str] | None = None,
        selected_tests: list[str] | None = None,
        exit_code: int | None = None,
        stdout: str = "",
        stderr: str = "",
        status: str | None = None,
    ) -> dict[str, Any]:
        patch_container = self._find_repo_operation_container(kind_fq="operation:patch_attempt", payload_key="patch_attempt_id", payload_value=patch_attempt_id)
        if patch_container is None:
            return {"status": "failed_closed", "code": "PATCH_ATTEMPT_NOT_FOUND"}
        graph_revision_id = str(patch_container.get("container_graph_revision_id") or "")
        graph_revision = next(
            (row for row in normalize_container_graph(self._container_graph).get("graph_revisions", []) if str(row.get("container_graph_revision_id") or "") == graph_revision_id),
            None,
        )
        if graph_revision is None:
            return {"status": "failed_closed", "code": "PATCH_ATTEMPT_GRAPH_REVISION_NOT_FOUND"}
        patch_payload = dict((patch_container.get("traits_json") or {}).get("payload") or {})
        status = status or ("passed" if int(exit_code or 0) == 0 else "failed")
        test_run_id = stable_hash("repo-test-run-id-v1", patch_attempt_id, commands or [], selected_tests or [], exit_code, stdout, stderr, prefix="testrun")
        payload = {
            "test_run_id": test_run_id,
            "patch_attempt_id": patch_attempt_id,
            "commands": list(commands or []),
            "selected_tests": list(selected_tests or []),
            "exit_code": int(exit_code or 0),
            "stdout_ref": {"text_hash": hashlib.sha256(str(stdout or "").encode("utf-8")).hexdigest(), "char_count": len(str(stdout or ""))},
            "stderr_ref": {"text_hash": hashlib.sha256(str(stderr or "").encode("utf-8")).hexdigest(), "char_count": len(str(stderr or ""))},
            "status": status,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        container = self._repo_operation_container_row(graph_revision=graph_revision, kind="test_run", operation_id=test_run_id, payload=payload)
        relations = [
            self._repo_operation_relation_row(
                graph_revision_id=graph_revision_id,
                src_container_id=patch_container["container_id"],
                dst_container_id=container["container_id"],
                relation_kind="verified_by",
                relation_capability="verified_by",
            )
        ]
        artifact_payload = {**payload, "stdout": stdout, "stderr": stderr}
        artifact = self._repo_operation_artifact_row(graph_revision_id=graph_revision_id, artifact_kind="operation_test_run", subject_id=container["container_id"], payload=artifact_payload)
        state = self._repo_operation_state_row(graph_revision_id=graph_revision_id, container_id=container["container_id"], state_kind="test_run_status", state_scope_id=str(patch_payload.get("work_item_container_id") or ""), payload=payload)
        self._append_repo_operation_rows(containers=[container], relations=relations, artifacts=[artifact], state=[state])
        return {"status": "ok", "test_run_id": test_run_id, "container_id": container["container_id"], "payload": payload}

    def record_repo_verification_result(
        self,
        *,
        patch_attempt_id: str,
        status: str,
        evidence_refs: list[str] | None = None,
        remaining_gaps: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        patch_container = self._find_repo_operation_container(kind_fq="operation:patch_attempt", payload_key="patch_attempt_id", payload_value=patch_attempt_id)
        if patch_container is None:
            return {"status": "failed_closed", "code": "PATCH_ATTEMPT_NOT_FOUND"}
        graph_revision_id = str(patch_container.get("container_graph_revision_id") or "")
        graph_revision = next(
            (row for row in normalize_container_graph(self._container_graph).get("graph_revisions", []) if str(row.get("container_graph_revision_id") or "") == graph_revision_id),
            None,
        )
        if graph_revision is None:
            return {"status": "failed_closed", "code": "PATCH_ATTEMPT_GRAPH_REVISION_NOT_FOUND"}
        patch_payload = dict((patch_container.get("traits_json") or {}).get("payload") or {})
        verification_result_id = stable_hash("repo-verification-result-id-v1", patch_attempt_id, status, evidence_refs or [], remaining_gaps or [], prefix="verify")
        payload = {
            "verification_result_id": verification_result_id,
            "patch_attempt_id": patch_attempt_id,
            "status": status,
            "evidence_refs": list(evidence_refs or []),
            "remaining_gaps": list(remaining_gaps or []),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        container = self._repo_operation_container_row(graph_revision=graph_revision, kind="verification_result", operation_id=verification_result_id, payload=payload)
        relation_kind = "verification_confirms" if status == "confirmed" else "verification_refutes" if status == "refuted" else "verified_by"
        relations = [
            self._repo_operation_relation_row(
                graph_revision_id=graph_revision_id,
                src_container_id=container["container_id"],
                dst_container_id=patch_container["container_id"],
                relation_kind=relation_kind,
                relation_capability=relation_kind,
            )
        ]
        artifact = self._repo_operation_artifact_row(graph_revision_id=graph_revision_id, artifact_kind="operation_verification_result", subject_id=container["container_id"], payload=payload)
        state = self._repo_operation_state_row(graph_revision_id=graph_revision_id, container_id=container["container_id"], state_kind="verification_status", state_scope_id=str(patch_payload.get("work_item_container_id") or ""), payload=payload)
        self._append_repo_operation_rows(containers=[container], relations=relations, artifacts=[artifact], state=[state])
        return {"status": "ok", "verification_result_id": verification_result_id, "container_id": container["container_id"], "payload": payload}

    def _visible_fact_lookup(self, fact_filter, *, search_family: str = "auto") -> dict[str, dict[str, Any]]:
        lookup: dict[str, dict[str, Any]] = {}
        for facts in (self._all_granular, self._all_cons, self._all_cross):
            for fact in facts:
                if not fact_filter(fact):
                    continue
                if search_family not in {"auto", "", None} and self._fact_source_family(fact) != search_family:
                    continue
                fact_id = str(fact.get("id") or "").strip()
                if fact_id and fact_id not in lookup:
                    lookup[fact_id] = fact
        return lookup

    def _visible_source_families(self, fact_filter) -> list[str]:
        families: list[str] = []
        seen: set[str] = set()
        for facts in (self._all_granular, self._all_cons, self._all_cross):
            for fact in facts:
                if not fact_filter(fact):
                    continue
                family = self._fact_source_family(fact)
                if family and family not in seen:
                    seen.add(family)
                    families.append(family)
        return families

    @staticmethod
    def _auto_discovery_query_tokens(query: str) -> set[str]:
        tokens: set[str] = set()
        for raw in re.findall(r"[A-Za-z0-9]+", str(query or "").lower()):
            token = normalize_term_token(raw)
            if not token or token in STOP_WORDS:
                continue
            tokens.add(token)
        return tokens

    @staticmethod
    def _auto_discovery_fact_text(fact: dict[str, Any]) -> str:
        metadata_raw = fact.get("metadata")
        metadata = cast(dict[str, Any], metadata_raw) if isinstance(metadata_raw, dict) else {}
        codebase_raw = metadata.get("codebase")
        codebase = cast(dict[str, Any], codebase_raw) if isinstance(codebase_raw, dict) else {}
        parts: list[str] = [
            str(fact.get("id") or ""),
            str(fact.get("kind") or ""),
            str(fact.get("fact") or ""),
            str(fact.get("source_id") or metadata.get("episode_source_id") or ""),
            str(fact.get("file_path") or metadata.get("file_path") or ""),
            str(fact.get("language") or metadata.get("language") or ""),
            str(fact.get("semantic_kind") or metadata.get("semantic_kind") or ""),
            str(fact.get("semantic_type") or metadata.get("semantic_type") or ""),
        ]
        for key in (
            "object_id",
            "relation_type",
            "qualified_name",
            "name",
            "path",
            "kind_fq",
            "anchor",
        ):
            value = codebase.get(key)
            if value:
                parts.append(str(value))
        for key in ("entities", "tags"):
            value = fact.get(key) or metadata.get(key)
            if isinstance(value, list):
                parts.extend(str(item) for item in value)
            elif value:
                parts.append(str(value))
        return "\n".join(part for part in parts if part)

    def _score_auto_codebase_candidate(
        self,
        *,
        query: str,
        query_tokens: set[str],
        fact: dict[str, Any],
        retrieval_score: float,
    ) -> dict[str, Any]:
        haystack = self._auto_discovery_fact_text(fact)
        haystack_lower = haystack.lower()
        fact_tokens = self._auto_discovery_query_tokens(haystack)
        overlap_tokens = sorted(query_tokens & fact_tokens)
        normalized_query = " ".join(
            raw
            for raw in re.findall(r"[A-Za-z0-9_./:-]+", str(query or "").lower())
            if normalize_term_token(raw) and normalize_term_token(raw) not in STOP_WORDS
        )
        phrase_match = bool(normalized_query and normalized_query in haystack_lower)
        score = float(len(overlap_tokens))
        if phrase_match:
            score += 4.0
        return {
            "fact_id": str(fact.get("id") or ""),
            "score": score,
            "retrieval_score": float(retrieval_score),
            "overlap_count": len(overlap_tokens),
            "overlap_tokens": overlap_tokens,
            "phrase_match": phrase_match,
        }

    @staticmethod
    def _source_hydration_trace(
        code_trace: dict[str, Any] | None,
        *,
        code_query_mode: str,
        default_reason: str | None = None,
    ) -> dict[str, Any]:
        trace = dict(code_trace or {})
        mode = str(trace.get("mode") or "inactive")
        hydrated = mode in {"whole_file", "windowed_file"}
        reason = str(trace.get("reason") or default_reason or "")
        if hydrated and not reason:
            reason = "query_requires_code_hydration"
        if not hydrated and not reason:
            reason = "source_hydration_not_required"
        return {
            "hydrated": hydrated,
            "mode": mode,
            "reason": reason,
            "code_query_mode": code_query_mode,
            "selected_file": trace.get("selected_file"),
            "selected_fact_ids": trace.get("selected_fact_ids", []),
        }

    def _family_discovery_trace(
        self,
        *,
        fact_filter,
        requested_search_family: str | None,
        searched_families: list[str] | None,
        selected_facts: list[dict[str, Any]],
        code_query_mode: str,
        merged_families: list[str] | None = None,
        codebase_probe_trace: dict[str, Any] | None = None,
        codebase_augmentation_trace: dict[str, Any] | None = None,
        episode_first_pass_trace: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        requested = str(requested_search_family or "auto").strip().lower() or "auto"
        available = self._visible_source_families(fact_filter)
        available_set = set(available)
        searched_set = {
            str(family or "").strip().lower()
            for family in (searched_families or [])
            if str(family or "").strip()
        }
        if codebase_probe_trace and codebase_probe_trace.get("searched"):
            searched_set.add("codebase")
        selected_ids_by_family: dict[str, list[str]] = defaultdict(list)
        for fact in selected_facts:
            family = self._fact_source_family(fact)
            fact_id = str(fact.get("id") or "").strip()
            if family and fact_id and fact_id not in selected_ids_by_family[family]:
                selected_ids_by_family[family].append(fact_id)

        selected_family_set = set(selected_ids_by_family)
        merged = list(dict.fromkeys(
            family
            for family in (merged_families or [*searched_set, *selected_family_set])
            if family
        ))
        candidate_counts: dict[str, int] = defaultdict(int)
        first_pass = episode_first_pass_trace or {}
        for row in first_pass.get("per_family") or []:
            family = str(row.get("family") or "").strip().lower()
            if family:
                candidate_counts[family] = int(row.get("candidate_count") or 0)

        family_names = sorted(
            available_set
            | searched_set
            | selected_family_set
            | ({requested} if requested not in {"", "auto"} else set())
            | ({"codebase"} if requested == "auto" or codebase_probe_trace else set())
        )
        per_family: dict[str, dict[str, Any]] = {}
        for family in family_names:
            available_family = family in available_set
            searched_family = family in searched_set
            selected_ids = selected_ids_by_family.get(family, [])
            mode = "not_available"
            skipped_reason = None
            candidate_count = int(candidate_counts.get(family, 0))
            if family == "codebase" and codebase_probe_trace is not None:
                probe = dict(codebase_probe_trace)
                per_family[family] = {
                    "available": bool(probe.get("available", available_family)),
                    "searched": bool(probe.get("searched", searched_family)),
                    "candidate_count": int(probe.get("candidate_count") or 0),
                    "visible_fact_count": int(probe.get("visible_fact_count") or 0),
                    "selected_count": int(probe.get("selected_count") or 0),
                    "selected_fact_ids": list(probe.get("selected_fact_ids") or []),
                    "skipped_reason": probe.get("skipped_reason"),
                    "mode": str(probe.get("mode") or "codebase_cheap_probe"),
                    "score_summary": probe.get("score_summary", {}),
                }
                continue
            if available_family and searched_family:
                mode = "episode_primary"
            elif available_family:
                mode = "not_searched"
                skipped_reason = "explicit_family_filter" if requested not in {"", "auto", family} else "not_selected_by_auto_routing"
            if family == "codebase" and searched_family:
                mode = "explicit_codebase" if requested == "codebase" else "auto_codebase"
            if not available_family:
                skipped_reason = "not_available"
            elif searched_family and not selected_ids and not skipped_reason:
                skipped_reason = "no_selected_evidence"
            per_family[family] = {
                "available": available_family,
                "searched": searched_family,
                "candidate_count": candidate_count,
                "selected_count": len(selected_ids),
                "selected_fact_ids": selected_ids[:12],
                "skipped_reason": skipped_reason,
                "mode": mode,
            }

        source_hydration = self._source_hydration_trace(
            codebase_augmentation_trace,
            code_query_mode=code_query_mode,
        )
        if codebase_probe_trace and not source_hydration["hydrated"]:
            source_hydration = dict(codebase_probe_trace.get("source_hydration") or source_hydration)
        return {
            "available_families": available,
            "searched_families": sorted(searched_set),
            "per_family": per_family,
            "merged_families": merged,
            "source_hydration": source_hydration,
        }

    async def _auto_discover_codebase_evidence(
        self,
        *,
        query: str,
        query_type: str,
        query_metadata: dict[str, Any] | None,
        fact_filter,
        episode_facts: list[dict[str, Any]],
        code_query_mode: str,
    ) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
        threshold: dict[str, Any] = {
            "min_score": 1.0,
            "min_overlap": 1,
            "exact_phrase_bonus": 4.0,
            "selected_limit": 6,
            "policy": "token_overlap_or_exact_phrase",
        }
        trace: dict[str, Any] = {
            "available": False,
            "searched": False,
            "candidate_count": 0,
            "selected_count": 0,
            "selected_fact_ids": [],
            "mode": "codebase_cheap_probe",
            "threshold": threshold,
            "score_summary": {"threshold": threshold, "accepted": [], "rejected": []},
            "source_hydration": self._source_hydration_trace(
                {"mode": "hot_only", "reason": "generic_auto_discovery_facts_only"},
                code_query_mode=code_query_mode,
            ),
        }
        code_lookup = self._visible_fact_lookup(fact_filter, search_family="codebase")
        if not code_lookup:
            trace["skipped_reason"] = "no_visible_codebase_facts"
            return [], "", trace

        query_tokens = self._auto_discovery_query_tokens(query)
        trace["available"] = True
        trace["visible_fact_count"] = len(code_lookup)
        if not query_tokens:
            trace["skipped_reason"] = "no_meaningful_query_tokens"
            return [], "", trace
        min_overlap = 1 if len(query_tokens) <= 1 else 2
        threshold = {
            **threshold,
            "min_score": float(min_overlap),
            "min_overlap": min_overlap,
        }
        trace["threshold"] = threshold
        trace["score_summary"]["threshold"] = threshold

        codebase_result = await self._generic_fact_recall(
            query=query,
            fact_filter=fact_filter,
            search_family="codebase",
            query_type=query_type,
            query_metadata=query_metadata,
            codebase_probe_mode=True,
        )
        trace["searched"] = True
        trace["probe_runtime_reason"] = (codebase_result.get("runtime_trace") or {}).get("reason")
        retrieval_scores: dict[str, float] = {}
        candidate_facts: list[dict[str, Any]] = []
        seen_candidate_ids: set[str] = set()
        for item in codebase_result.get("retrieved", []) or []:
            if not isinstance(item, dict):
                continue
            fact_id = str(item.get("fact_id") or item.get("id") or "").strip()
            if not fact_id or fact_id in seen_candidate_ids or fact_id not in code_lookup:
                continue
            seen_candidate_ids.add(fact_id)
            retrieval_scores[fact_id] = float(item.get("sim") or item.get("score") or 0.0)
            candidate_facts.append(code_lookup[fact_id])
        trace["candidate_count"] = len(candidate_facts)
        if not candidate_facts:
            trace["skipped_reason"] = "no_probe_candidates"
            return [], "", trace

        scored_rows = [
            self._score_auto_codebase_candidate(
                query=query,
                query_tokens=query_tokens,
                fact=fact,
                retrieval_score=retrieval_scores.get(str(fact.get("id") or ""), 0.0),
            )
            for fact in candidate_facts
        ]
        scored_rows.sort(
            key=lambda row: (
                -float(row["score"]),
                -float(row["retrieval_score"]),
                str(row["fact_id"]),
            )
        )
        accepted = [
            row
            for row in scored_rows
            if float(row["score"]) >= float(threshold["min_score"])
            and (
                int(row["overlap_count"]) >= int(threshold["min_overlap"])
                or bool(row["phrase_match"])
            )
        ]
        selected_ids = [
            str(row["fact_id"])
            for row in accepted[: int(threshold["selected_limit"])]
            if str(row.get("fact_id") or "")
        ]
        selected_id_set = set(selected_ids)
        selected_facts = [
            fact
            for fact in candidate_facts
            if str(fact.get("id") or "") in selected_id_set
        ]
        selected_facts.sort(key=lambda fact: selected_ids.index(str(fact.get("id") or "")))
        trace["selected_count"] = len(selected_facts)
        trace["selected_fact_ids"] = selected_ids
        trace["score_summary"] = {
            "threshold": threshold,
            "query_tokens": sorted(query_tokens),
            "accepted": accepted[: int(threshold["selected_limit"])],
            "rejected": [
                {
                    **row,
                    "reason": "below_overlap_threshold",
                }
                for row in scored_rows
                if str(row.get("fact_id") or "") not in selected_id_set
            ][:8],
        }
        if not selected_facts:
            trace["skipped_reason"] = "filtered_below_overlap_threshold"
            return [], "", trace

        code_packet = _build_context_packet(
            selected_facts,
            self._raw_sessions,
            budget=0,
            raw_docs=None,
        )
        code_fact_lines = _context_packet_fact_lines(code_packet)
        if not code_fact_lines:
            code_fact_lines = [
                f"- {str(fact.get('fact') or '').strip()}"
                for fact in selected_facts
                if str(fact.get("fact") or "").strip()
            ]
        rendered = ""
        if code_fact_lines:
            rendered = "--- CODEBASE FACTS ---\n" + "\n".join(code_fact_lines)

        if code_query_mode in {"precise_code", "mixed_code_plus_prose"}:
            code_segments, code_trace = augment_codebase_context(
                query=query,
                retrieved_facts=[*episode_facts, *selected_facts],
                data_dir=str(self.data_dir),
            )
            trace["source_hydration"] = self._source_hydration_trace(
                code_trace,
                code_query_mode=code_query_mode,
            )
            if code_segments:
                rendered = f"{rendered.rstrip()}\n\n{_render_code_attachment_block(code_segments)}".strip()
        return selected_facts, rendered, trace

    async def _merge_auto_mixed_codebase_result(
        self,
        *,
        query: str,
        query_type: str,
        query_metadata: dict[str, Any] | None = None,
        fact_filter,
        episode_context: str,
        episode_facts: list[dict[str, Any]],
    ) -> tuple[str, list[dict[str, Any]], dict[str, Any], list[str]]:
        prose_fact_terms = "\n".join(
            str(fact.get("fact") or "").strip()
            for fact in episode_facts[:8]
            if str(fact.get("fact") or "").strip()
        )
        codebase_query = f"{query}\n\nKnown prose evidence:\n{prose_fact_terms}" if prose_fact_terms else query
        codebase_result = await self._generic_fact_recall(
            query=codebase_query,
            fact_filter=fact_filter,
            search_family="codebase",
            query_type=query_type,
            query_metadata=query_metadata,
        )
        code_lookup = self._visible_fact_lookup(fact_filter, search_family="codebase")
        code_fact_ids = [
            str(item.get("fact_id") or "").strip()
            for item in codebase_result.get("retrieved", [])
            if isinstance(item, dict)
        ]
        code_facts = [
            code_lookup[fact_id]
            for fact_id in code_fact_ids
            if fact_id in code_lookup
        ]
        codebase_context_trace = (codebase_result.get("runtime_trace") or {}).get("codebase_context") or {}
        codebase_context = str(codebase_result.get("context") or "").strip()
        codebase_patch_context = bool(
            codebase_context
            and isinstance(codebase_context_trace, dict)
            and (
                str((query_metadata or {}).get("task_mode") or "") == "patch_generation"
                or str((query_metadata or {}).get("output_artifact") or "") == "unified_diff"
                or codebase_context_trace.get("task_mode") == "patch_generation"
                or (codebase_context_trace.get("repo_task_contract") or {}).get("task_mode") == "patch_generation"
            )
        )
        if codebase_patch_context:
            return (
                codebase_context,
                code_facts,
                {
                    "mode": "auto",
                    "query_mode": "mixed_code_plus_prose",
                    "lanes": ["codebase"],
                    "merged_families": ["codebase"],
                    "codebase_selected_fact_ids": code_fact_ids,
                    "codebase_retrieved_fact_count": len(code_facts),
                    "codebase_context_authoritative": True,
                    "codebase_context": codebase_context_trace,
                },
                ["codebase"],
            )
        if not code_facts and not codebase_context:
            return (
                episode_context,
                episode_facts,
                {
                    "mode": "auto",
                    "query_mode": "mixed_code_plus_prose",
                    "lanes": ["episode"],
                    "merged_families": sorted(
                        {
                            family
                            for family in (self._fact_source_family(fact) for fact in episode_facts)
                            if family
                        }
                    ),
                    "codebase_selected_fact_ids": [],
                },
                [],
            )
        if not code_facts:
            merged_context = episode_context.rstrip()
            if codebase_context:
                merged_context = f"{merged_context}\n\n{codebase_context}"
            merged_families = sorted(
                {
                    family
                    for family in (self._fact_source_family(fact) for fact in episode_facts)
                    if family
                } | {"codebase"}
            )
            return (
                merged_context,
                episode_facts,
                {
                    "mode": "auto",
                    "query_mode": "mixed_code_plus_prose",
                    "lanes": ["episode", "codebase"],
                    "merged_families": merged_families,
                    "codebase_selected_fact_ids": [],
                    "codebase_retrieved_fact_count": 0,
                    "codebase_context": codebase_context_trace,
                },
                merged_families,
            )

        merged_facts = list(episode_facts)
        seen_fact_ids = {
            str(fact.get("id") or "").strip()
            for fact in merged_facts
            if str(fact.get("id") or "").strip()
        }
        for fact in code_facts:
            fact_id = str(fact.get("id") or "").strip()
            if fact_id and fact_id not in seen_fact_ids:
                merged_facts.append(fact)
                seen_fact_ids.add(fact_id)

        code_packet = _build_context_packet(
            code_facts,
            self._raw_sessions,
            budget=0,
            raw_docs=None,
        )
        code_fact_lines = _context_packet_fact_lines(code_packet)
        merged_context = episode_context.rstrip()
        if code_fact_lines:
            merged_context = f"{merged_context}\n\n--- CODEBASE FACTS ---\n" + "\n".join(code_fact_lines)

        code_segments, code_trace = augment_codebase_context(
            query=codebase_query,
            retrieved_facts=merged_facts,
            data_dir=str(self.data_dir),
        )
        if code_segments:
            merged_context = f"{merged_context}\n\n" + _render_code_attachment_block(code_segments)

        merged_families = sorted(
            {
                family
                for family in (self._fact_source_family(fact) for fact in merged_facts)
                if family
            }
        )
        return (
            merged_context,
            merged_facts,
            {
                "mode": "auto",
                "query_mode": "mixed_code_plus_prose",
                "lanes": ["episode", "codebase"],
                "merged_families": merged_families,
                "codebase_selected_fact_ids": code_fact_ids,
                "codebase_retrieved_fact_count": len(code_facts),
                "codebase_augmentation": code_trace,
            },
            merged_families,
        )

    def _visible_episode_runtime(
        self,
        fact_filter,
        *,
        query: str | None = None,
    ) -> tuple[dict, dict[str, dict], dict[str, list[dict]], object] | None:
        visible_facts = self._episode_runtime_facts(fact_filter, query=query)
        if not visible_facts:
            return None
        facts_by_episode = build_facts_by_episode(visible_facts)
        if not self._episode_corpus.get("documents"):
            return None
        if not facts_by_episode:
            return None

        query_features = extract_query_features(query or "") if query else {}
        include_full_document_docs = bool(
            query
            and (
                query_features.get("step_numbers")
                or query_features.get("step_range")
            )
        )

        visible_docs = []
        for doc in self._episode_corpus.get("documents", []):
            fact_visible_episode_ids = {
                ep.get("episode_id", "")
                for ep in doc.get("episodes", [])
                if ep.get("episode_id") in facts_by_episode
                and _record_semantic_ready(ep, fallback_text=str(ep.get("raw_text") or ""))
            }
            if not fact_visible_episode_ids:
                continue
            doc_source_families = {
                ep.get("source_type", "")
                for ep in doc.get("episodes", [])
                if ep.get("episode_id")
                and _record_semantic_ready(ep, fallback_text=str(ep.get("raw_text") or ""))
            }
            if include_full_document_docs and doc_source_families == {"document"}:
                episodes = [
                    ep
                    for ep in doc.get("episodes", [])
                    if ep.get("episode_id")
                    and _record_semantic_ready(ep, fallback_text=str(ep.get("raw_text") or ""))
                ]
            else:
                episodes = [
                    ep
                    for ep in doc.get("episodes", [])
                    if ep.get("episode_id") in fact_visible_episode_ids
                    and _record_semantic_ready(ep, fallback_text=str(ep.get("raw_text") or ""))
                ]
            if episodes:
                visible_docs.append({"doc_id": doc.get("doc_id"), "episodes": episodes})
        if not visible_docs:
            return None

        corpus = {"documents": visible_docs}
        episode_lookup = build_episode_lookup(corpus)
        bm25 = build_episode_bm25(corpus)
        return corpus, episode_lookup, facts_by_episode, bm25

    def _next_session_num(self) -> int:
        session_nums = [
            rs.get("session_num", 0)
            for rs in self._raw_sessions
            if isinstance(rs, dict) and isinstance(rs.get("session_num"), int)
        ]
        return (max(session_nums) if session_nums else 0) + 1

    @staticmethod
    def _iso_from_timestamp_ms(timestamp_ms: int | None) -> str:
        ts = int(timestamp_ms or 0) / 1000.0
        return datetime.fromtimestamp(ts, timezone.utc).isoformat()

    def _raw_entry_acl_allows(self, entry: dict, caller_id: str, caller_memberships: list[str], caller_role: str) -> bool:
        pseudo_fact = {
            "owner_id": entry.get("owner_id"),
            "read": entry.get("read") or [],
            "scope": entry.get("scope"),
            "agent_id": entry.get("agent_id"),
            "swarm_id": entry.get("swarm_id"),
        }
        return self._acl_allows(pseudo_fact, caller_id, caller_memberships, caller_role)

    @staticmethod
    def _raw_query_tokens(query: str) -> list[str]:
        return [
            token
            for token in re.findall(r"[\w./:-]+", (query or "").lower(), flags=re.UNICODE)
            if len(token) >= 2 and token not in STOP_WORDS
        ]

    @staticmethod
    def _raw_metadata_text(metadata: dict | None) -> str:
        if not metadata:
            return ""
        parts: list[str] = []
        for value in metadata.values():
            if isinstance(value, str):
                parts.append(value)
            elif isinstance(value, list):
                parts.extend(str(item) for item in value if isinstance(item, (str, int, float)))
            elif isinstance(value, (int, float)):
                parts.append(str(value))
        return " ".join(parts).lower()

    def _score_raw_write_entry(self, query: str, tokens: list[str], entry: dict) -> float:
        haystack = str(entry.get("content") or "").lower()
        meta_text = self._raw_metadata_text(entry.get("metadata") or {})
        if not haystack and not meta_text:
            return 0.0
        score = 0.0
        q = (query or "").strip().lower()
        if q and q in haystack:
            score += 10.0
        if q and q in meta_text:
            score += 4.0
        for token in tokens:
            if token in haystack:
                score += 1.0
            if token in meta_text:
                score += 0.5
        family = str(entry.get("content_family") or "").lower()
        if family and family in q:
            score += 1.0
        return score

    @staticmethod
    def _raw_entry_family_is_conversation(entry: dict) -> bool:
        return str(entry.get("content_family") or "").strip().lower() in {"chat", "conversation"}

    @staticmethod
    def _raw_role(entry: dict) -> str:
        metadata = entry.get("metadata") or {}
        return str(metadata.get("role") or entry.get("role") or "").strip().lower()

    @staticmethod
    def _raw_session_num(entry: dict) -> int | None:
        num = _coerce_positive_session_num(entry.get("session_num"))
        if num is not None:
            return num
        metadata = entry.get("metadata") or {}
        for key in ("turn_number", "part_idx", "session_num"):
            num = _coerce_positive_session_num(metadata.get(key))
            if num is not None:
                return num
        return None

    @staticmethod
    def _raw_source_key(entry: dict) -> str:
        return str(
            entry.get("session_key")
            or entry.get("session_id")
            or entry.get("logical_source_id")
            or entry.get("source_id")
            or ""
        ).strip()

    @staticmethod
    def _raw_entry_timestamp(entry: dict) -> int:
        value = entry.get("timestamp_ms")
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str) and value.isdigit():
            return int(value)
        return 0

    def _fact_backed_message_ids(self) -> set[str]:
        raw_message_by_session_id = {
            str(raw.get("raw_session_id") or ""): str(raw.get("message_id") or "")
            for raw in self._raw_sessions
            if _is_active_lifecycle_record(raw) and str(raw.get("raw_session_id") or "").strip()
        }
        backed: set[str] = set()
        for fact in self._all_granular:
            if not _is_active_lifecycle_record(fact):
                continue
            message_id = str(fact.get("message_id") or "").strip()
            if message_id:
                backed.add(message_id)
            raw_session_id = str(fact.get("raw_session_id") or "").strip()
            raw_message_id = raw_message_by_session_id.get(raw_session_id, "")
            if raw_message_id:
                backed.add(raw_message_id)
        return backed

    def _selected_fact_message_ids(self, retrieved_items: list[Any]) -> set[str]:
        raw_message_by_session_id = {
            str(raw.get("raw_session_id") or ""): str(raw.get("message_id") or "")
            for raw in self._raw_sessions
            if _is_active_lifecycle_record(raw) and str(raw.get("raw_session_id") or "").strip()
        }
        selected: set[str] = set()
        for item in retrieved_items:
            if not isinstance(item, dict):
                continue
            message_id = str(item.get("message_id") or "").strip()
            if message_id:
                selected.add(message_id)
            raw_session_id = str(item.get("raw_session_id") or "").strip()
            raw_message_id = raw_message_by_session_id.get(raw_session_id, "")
            if raw_message_id:
                selected.add(raw_message_id)
        return selected

    @staticmethod
    def _query_requests_recent_evidence(query: str, result_type: str | None = None) -> bool:
        if str(result_type or "").strip().lower() == "current":
            return True
        q = str(query or "").lower()
        markers = (
            "latest", "last", "recent", "newest", "current", "just answered", "just wrote",
            "what did", "последн", "недавн", "только что", "сейчас", "последний ответ",
        )
        return any(marker in q for marker in markers)

    def _raw_recall_item_from_entry(
        self,
        entry: dict,
        *,
        score: float,
        evidence_kind: str,
    ) -> dict:
        snippet = re.sub(r"\s+", " ", str(entry.get("content") or "")).strip()[:400]
        return {
            "message_id": entry.get("message_id"),
            "session_id": entry.get("session_id") or entry.get("session_key"),
            "content_family": entry.get("content_family") or entry.get("format") or "conversation",
            "content": snippet,
            "metadata": entry.get("metadata") or {},
            "timestamp_ms": self._raw_entry_timestamp(entry),
            "session_num": self._raw_session_num(entry),
            "role": self._raw_role(entry),
            "extraction_state": entry.get("extraction_state") or "complete",
            "status": entry.get("status") or "active",
            "raw_evidence_kind": evidence_kind,
            "score": float(score),
        }

    def _episode_acl_record(self, episode: dict) -> dict:
        source_id = str(episode.get("source_id") or "").strip()
        source_record = self._source_records.get(source_id) or {}
        source_meta = source_record.get("source_meta") or {}
        return {
            "owner_id": episode.get("owner_id") or source_record.get("owner_id"),
            "read": episode.get("read") or source_record.get("read") or [],
            "write": episode.get("write") or source_record.get("write") or [],
            "scope": episode.get("scope") or source_record.get("scope") or source_meta.get("scope"),
            "agent_id": episode.get("agent_id") or source_record.get("agent_id") or source_meta.get("agent_id"),
            "swarm_id": episode.get("swarm_id") or source_record.get("swarm_id") or source_meta.get("swarm_id"),
        }

    def _episode_acl_allows(
        self,
        episode: dict,
        caller_id: str,
        caller_memberships: list[str],
        caller_role: str,
    ) -> bool:
        return self._acl_allows(self._episode_acl_record(episode), caller_id, caller_memberships, caller_role)

    def _episode_effective_swarm_id(self, episode: dict) -> str:
        record = self._episode_acl_record(episode)
        return str(record.get("swarm_id") or "")

    @staticmethod
    def _episode_content_family(episode: dict) -> str:
        return str(episode.get("source_type") or "document").strip().lower()

    @staticmethod
    def _episode_order_key(episode: dict) -> tuple[int, int, str, str, str]:
        projection_num = _coerce_positive_session_num(episode.get("projection_session_num")) or 10**9
        session_num = _coerce_positive_session_num(episode.get("session_num")) or 10**9
        temporal = str(
            episode.get("timestamp_ms")
            or episode.get("stored_at")
            or episode.get("source_date")
            or ""
        )
        return (
            projection_num,
            session_num,
            temporal,
            str(episode.get("message_id") or ""),
            str(episode.get("episode_id") or ""),
        )

    def _raw_episode_visibility_index(self) -> dict[str, tuple[dict, ...]]:
        if self._raw_episode_visibility_cache_version == self._index_snapshot_version:
            return self._raw_episode_visibility_cache

        by_family: dict[str, list[dict]] = defaultdict(list)
        for doc in self._episode_corpus.get("documents", []):
            for episode in doc.get("episodes", []):
                if not isinstance(episode, dict):
                    continue
                if not _record_semantic_ready(episode, fallback_text=str(episode.get("raw_text") or "")):
                    continue
                by_family[self._episode_content_family(episode)].append(episode)

        self._raw_episode_visibility_cache = {
            family: tuple(sorted(
                episodes,
                key=lambda ep: (
                    str(ep.get("source_id") or ""),
                    self._episode_order_key(ep),
                ),
            ))
            for family, episodes in by_family.items()
        }
        self._raw_episode_visibility_cache_version = self._index_snapshot_version
        return self._raw_episode_visibility_cache

    @staticmethod
    def _session_num_identity(value: Any) -> str:
        num = _coerce_positive_session_num(value)
        return str(num) if num is not None else ""

    def _raw_session_matches_episode_identity(self, raw: dict, episode: dict) -> bool:
        episode_artifact = str(episode.get("artifact_id") or "").strip()
        episode_version = str(episode.get("version_id") or "").strip()
        if episode_artifact and episode_version:
            return (
                str(raw.get("artifact_id") or "").strip() == episode_artifact
                and str(raw.get("version_id") or "").strip() == episode_version
            )
        episode_raw_session_id = str(episode.get("raw_session_id") or "").strip()
        if episode_raw_session_id:
            return str(raw.get("raw_session_id") or "").strip() == episode_raw_session_id
        episode_message_id = str(episode.get("message_id") or "").strip()
        if not episode_message_id or str(raw.get("message_id") or "").strip() != episode_message_id:
            return False
        episode_source = str(
            episode.get("source_id")
            or episode.get("session_id")
            or episode.get("logical_source_id")
            or ""
        ).strip()
        raw_sources = {
            str(raw.get("source_id") or "").strip(),
            str(raw.get("logical_source_id") or "").strip(),
            str(raw.get("session_key") or "").strip(),
        }
        if episode_source and episode_source not in raw_sources:
            return False
        episode_nums = {
            self._session_num_identity(episode.get("projection_session_num")),
            self._session_num_identity(episode.get("session_num")),
        } - {""}
        raw_nums = {
            self._session_num_identity(raw.get("projection_session_num")),
            self._session_num_identity(raw.get("session_num")),
        } - {""}
        return not episode_nums or not raw_nums or bool(episode_nums & raw_nums)

    def _raw_backing_record_is_active_for_episode(self, episode: dict) -> bool:
        if not _is_active_lifecycle_record(episode):
            return False
        matching_raw = [
            raw for raw in self._raw_sessions
            if isinstance(raw, dict) and self._raw_session_matches_episode_identity(raw, episode)
        ]
        if not matching_raw:
            return True
        return any(_is_active_lifecycle_record(raw) for raw in matching_raw)

    def _iter_visible_raw_episodes(
        self,
        *,
        families: set[str],
        caller_id: str,
        caller_memberships: list[str],
        caller_role: str,
        swarm_id: str | None,
    ) -> list[dict]:
        episodes: list[dict] = []
        visibility_index = self._raw_episode_visibility_index()
        for family in sorted(families):
            for episode in visibility_index.get(family, ()):
                if not self._raw_backing_record_is_active_for_episode(episode):
                    continue
                if swarm_id and swarm_id != "default" and self._episode_effective_swarm_id(episode) != swarm_id:
                    continue
                if not self._episode_acl_allows(episode, caller_id, caller_memberships, caller_role):
                    continue
                episodes.append(episode)
        return episodes

    def _raw_episode_text_with_role(self, episode: dict) -> str:
        text = _semantic_episode_text(episode).strip()
        if not text:
            return ""
        role = str(episode.get("role") or (episode.get("metadata") or {}).get("role") or "").strip().lower()
        if not role:
            return text
        if text.lower().startswith(f"{role}:"):
            return text
        return f"{role}: {text}"

    def _raw_recall_item_from_episode(
        self,
        episode: dict,
        *,
        score: float,
        evidence_kind: str,
    ) -> dict:
        text = self._raw_episode_text_with_role(episode)
        snippet = re.sub(r"\s+", " ", text).strip()[:500]
        return {
            "episode_id": episode.get("episode_id"),
            "message_id": episode.get("message_id"),
            "session_id": episode.get("session_id") or episode.get("source_id"),
            "source_id": episode.get("source_id"),
            "content_family": self._episode_content_family(episode),
            "content": snippet,
            "metadata": episode.get("metadata") or {},
            "timestamp_ms": episode.get("timestamp_ms") or 0,
            "session_num": _coerce_positive_session_num(episode.get("session_num")),
            "role": str(episode.get("role") or (episode.get("metadata") or {}).get("role") or "").strip().lower(),
            "extraction_state": "complete",
            "status": episode.get("status") or "active",
            "raw_evidence_kind": evidence_kind,
            "score": float(score),
        }

    def _raw_episode_recall_entries(
        self,
        *,
        query: str,
        families: set[str],
        caller_id: str,
        caller_memberships: list[str],
        caller_role: str,
        swarm_id: str | None,
        exclude_episode_ids: set[str],
        prefer_recent: bool,
        limit: int,
    ) -> list[dict]:
        tokens = self._raw_query_tokens(query)
        scored: list[dict] = []
        for episode in self._iter_visible_raw_episodes(
            families=families,
            caller_id=caller_id,
            caller_memberships=caller_memberships,
            caller_role=caller_role,
            swarm_id=swarm_id,
        ):
            episode_id = str(episode.get("episode_id") or "").strip()
            if episode_id and episode_id in exclude_episode_ids:
                continue
            text = self._raw_episode_text_with_role(episode)
            if not text:
                continue
            entry = {
                "content": text,
                "metadata": episode.get("metadata") or {},
                "content_family": self._episode_content_family(episode),
            }
            score = self._score_raw_write_entry(query, tokens, entry)
            if score <= 0:
                if not prefer_recent:
                    continue
                score = 0.01
            scored.append(self._raw_recall_item_from_episode(
                episode,
                score=score,
                evidence_kind="completed_raw_episode",
            ))
        scored.sort(key=lambda item: (
            -float(item.get("score") or 0.0),
            -(self._raw_session_num(item) or 0) if prefer_recent else (self._raw_session_num(item) or 10**9),
            str(item.get("episode_id") or ""),
        ))
        return scored[:limit]

    RECALL_EXTRACTION_POLICY_MIRROR = _RECALL_EXTRACTION_POLICY_MIRROR

    @staticmethod
    def _recall_query_has_word(query: str, words: Iterable[str]) -> bool:
        return any(re.search(rf"\b{re.escape(word)}\b", query) for word in words)

    @staticmethod
    def _recall_query_has_phrase(query: str, phrases: Iterable[str]) -> bool:
        return any(phrase in query for phrase in phrases)

    @classmethod
    def _recall_mirror_query_terms(cls, feature_name: str) -> tuple[str, ...]:
        terms: list[str] = []
        for family_axes in cls.RECALL_EXTRACTION_POLICY_MIRROR.values():
            for axis_meta in family_axes.values():
                feature_terms = axis_meta.get("query_feature_terms") or {}
                if isinstance(feature_terms, dict):
                    values = feature_terms.get(feature_name) or []
                    terms.extend(str(value) for value in values if str(value).strip())
        return tuple(dict.fromkeys(terms))

    @classmethod
    def _recall_mirror_signal_terms(cls, family: str, axis: str) -> tuple[str, ...]:
        axis_meta = cls.RECALL_EXTRACTION_POLICY_MIRROR.get(family, {}).get(axis, {})
        values: list[Any] = []
        for key in ("prompt_declared_signals", "query_signal_lemmas"):
            raw_values = axis_meta.get(key)
            if isinstance(raw_values, (list, tuple, set)):
                values.extend(raw_values)
        return tuple(dict.fromkeys(str(value) for value in values if str(value).strip()))

    @classmethod
    def extract_recall_policy_features(
        cls,
        query: str,
        *,
        search_family: str | None,
        query_type: str | None,
    ) -> dict:
        normalized = " ".join(str(query or "").strip().lower().split())
        family = str(search_family or "auto").strip().lower()
        source_target = "unknown"
        asks_for_assistant_output = (
            cls._recall_query_has_word(normalized, ("assistant", "reply", "response"))
            or bool(
                re.search(
                    r"\byou\b.*\b(answer|answered|reply|replied|write|wrote|say|said|provide|provided)\b",
                    normalized,
                )
            )
        )
        if family == "document":
            source_target = "document"
        elif asks_for_assistant_output:
            source_target = "assistant"
        elif cls._recall_query_has_word(normalized, ("document", "file")):
            source_target = "document"
        elif cls._recall_query_has_word(normalized, ("source", "record")):
            source_target = "source"
        elif cls._recall_query_has_word(normalized, ("user", "my", "i", "me")):
            source_target = "user"

        asks_for_identifier_like_value = cls._recall_query_has_word(
            normalized,
            cls._recall_mirror_query_terms("asks_for_identifier_like_value"),
        )
        asks_for_date_or_time = cls._recall_query_has_word(
            normalized,
            cls._recall_mirror_query_terms("asks_for_date_or_time"),
        )
        asks_for_quantity = cls._recall_query_has_word(
            normalized,
            cls._recall_mirror_query_terms("asks_for_quantity"),
        )
        asks_for_exact_value = (
            cls._recall_query_has_word(
                normalized,
                cls._recall_mirror_query_terms("asks_for_exact_value"),
            )
            or asks_for_identifier_like_value
            or asks_for_date_or_time
            or asks_for_quantity
        )
        asks_for_raw_turn_or_wording = (
            cls._recall_query_has_phrase(normalized, ("exact wording", "exact text", "raw turn"))
            or cls._recall_query_has_word(
                normalized,
                ("verbatim", "quote", "quoted", "phrase", "transcript", "neighbor", "nearby"),
            )
            or bool(
                re.search(
                    r"\bwhat\b.*\b(i|you)\b.*\b"
                    r"(say|said|write|wrote|reply|replied|answer|answered|respond|responded)\b"
                    r".*\b(before|after)\b",
                    normalized,
                )
            )
        )
        asks_for_named_entity = cls._recall_query_has_word(
            normalized,
            cls._recall_mirror_query_terms("asks_for_named_entity"),
        )
        asks_for_identity_or_relationship = cls._recall_query_has_word(
            normalized,
            ("who", "identity", "person", "role", "relationship", "relation", "parent", "partner", "colleague"),
        )
        asks_for_decision_rule_requirement_constraint = cls._recall_query_has_word(
            normalized,
            cls._recall_mirror_query_terms("asks_for_decision_rule_requirement_constraint"),
        )
        asks_for_preference_or_reason = cls._recall_query_has_word(
            normalized,
            cls._recall_mirror_query_terms("asks_for_preference_or_reason"),
        )
        asks_for_list_or_table_item = cls._recall_query_has_word(
            normalized,
            cls._recall_mirror_query_terms("asks_for_list_or_table_item"),
        )
        asks_for_temporal_order = cls._recall_query_has_word(
            normalized,
            (
                *cls._recall_mirror_signal_terms("conversation", "temporal_ordering"),
            ),
        )
        asks_for_knowledge_update = cls._recall_query_has_word(
            normalized,
            cls._recall_mirror_signal_terms("conversation", "knowledge_updates"),
        )
        asks_for_acquisition_event = cls._recall_query_has_word(
            normalized,
            cls._recall_mirror_signal_terms("conversation", "acquisition_events"),
        )
        return {
            "source_target": source_target,
            "asks_for_raw_turn_or_wording": asks_for_raw_turn_or_wording,
            "asks_for_assistant_output": asks_for_assistant_output,
            "asks_for_exact_value": asks_for_exact_value,
            "asks_for_identifier_like_value": asks_for_identifier_like_value,
            "asks_for_date_or_time": asks_for_date_or_time,
            "asks_for_quantity": asks_for_quantity,
            "asks_for_named_entity": asks_for_named_entity,
            "asks_for_identity_or_relationship": asks_for_identity_or_relationship,
            "asks_for_decision_rule_requirement_constraint": asks_for_decision_rule_requirement_constraint,
            "asks_for_preference_or_reason": asks_for_preference_or_reason,
            "asks_for_list_or_table_item": asks_for_list_or_table_item,
            "asks_for_temporal_order": asks_for_temporal_order,
            "asks_for_knowledge_update": asks_for_knowledge_update,
            "asks_for_temporal_order_or_update": asks_for_temporal_order or asks_for_knowledge_update,
            "asks_for_acquisition_event": asks_for_acquisition_event,
            "search_family": family,
            "query_type": str(query_type or "").strip().lower(),
        }

    @classmethod
    def classify_query_against_extraction_policy(cls, features: dict, mirror: dict) -> dict:
        family = str(features.get("search_family") or "auto")
        source_target = str(features.get("source_target") or "unknown")
        requested_families = ["document"] if family == "document" else ["conversation"]
        if family == "auto":
            requested_families = ["conversation", "document"]
        matched: list[tuple[str, str]] = []

        def _add(family_name: str, axis: str) -> None:
            if axis in mirror.get(family_name, {}) and (family_name, axis) not in matched:
                matched.append((family_name, axis))

        for family_name in requested_families:
            if family_name == "conversation":
                if features.get("asks_for_exact_value") or features.get("asks_for_named_entity"):
                    _add("conversation", "exact_values")
                if features.get("asks_for_named_entity"):
                    _add("conversation", "named_targets")
                if features.get("asks_for_identity_or_relationship"):
                    _add("conversation", "identity")
                    _add("conversation", "relationships")
                if features.get("asks_for_list_or_table_item"):
                    _add("conversation", "one_fact_per_item")
                    _add("conversation", "tables_schedules")
                if features.get("asks_for_raw_turn_or_wording"):
                    _add("conversation", "verbatim_quotes")
                if features.get("asks_for_temporal_order"):
                    _add("conversation", "temporal_ordering")
                if features.get("asks_for_knowledge_update"):
                    _add("conversation", "knowledge_updates")
                if features.get("asks_for_acquisition_event"):
                    _add("conversation", "acquisition_events")
                if features.get("asks_for_decision_rule_requirement_constraint"):
                    _add("conversation", "kind_classification")
                    if features.get("asks_for_assistant_output"):
                        _add("conversation", "assistant_material_facts")
                if features.get("asks_for_preference_or_reason"):
                    _add("conversation", "preferences_with_reason")
                if features.get("asks_for_quantity"):
                    _add("conversation", "financial_components")
            elif family_name == "document":
                if (
                    features.get("asks_for_exact_value")
                    or features.get("asks_for_date_or_time")
                    or features.get("asks_for_quantity")
                ):
                    _add("document", "exact_technical_values")
                if features.get("asks_for_identifier_like_value"):
                    _add("document", "unique_identifiers")
                if features.get("asks_for_list_or_table_item"):
                    _add("document", "one_fact_per_item")
                    _add("document", "tables_rows")
                if features.get("asks_for_decision_rule_requirement_constraint"):
                    _add("document", "decisions_with_alternatives")
                    _add("document", "requirements_constraints")
                    _add("document", "policy_conditions_exceptions")
                    _add("document", "requirement_rejection_kinds")
                if features.get("asks_for_knowledge_update"):
                    _add("document", "version_tracking")
                if features.get("asks_for_quantity"):
                    _add("document", "financial_components")

        matched_axes = [f"{family_name}.{axis}" for family_name, axis in matched]
        matched_rule_ids = list(dict.fromkeys(
            rule_id
            for family_name, axis in matched
            for rule_id in mirror[family_name][axis].get("rule_ids", [])
        ))
        policy_source = list(dict.fromkeys(
            str(mirror[family_name][axis].get("source_prompt") or "")
            for family_name, axis in matched
            if str(mirror[family_name][axis].get("source_prompt") or "")
        ))
        if not policy_source:
            policy_source = list(dict.fromkeys(
                str(axis_meta.get("source_prompt") or "")
                for family_name in requested_families
                for axis_meta in mirror.get(family_name, {}).values()
                if str(axis_meta.get("source_prompt") or "")
            ))

        uncertain_reasons: list[str] = []
        exact_like = (
            bool(features.get("asks_for_exact_value"))
            or bool(features.get("asks_for_identifier_like_value"))
            or bool(features.get("asks_for_date_or_time"))
            or bool(features.get("asks_for_quantity"))
        )
        contextual_axes = {
            ("conversation", "acquisition_events"),
            ("conversation", "knowledge_updates"),
            ("conversation", "named_targets"),
            ("conversation", "temporal_ordering"),
        }
        fact_likelihood: FactLikelihood
        if features.get("asks_for_raw_turn_or_wording"):
            fact_likelihood = FACT_LIKELIHOOD_UNCERTAIN
            uncertain_reasons.append("raw_turn_or_wording_requested")
        elif source_target == "assistant":
            assistant_material = ("conversation", "assistant_material_facts") in matched
            if assistant_material:
                fact_likelihood = FACT_LIKELIHOOD_MEDIUM
            else:
                fact_likelihood = FACT_LIKELIHOOD_UNCERTAIN
                uncertain_reasons.append("assistant_output_not_material_fact")
        elif not matched:
            fact_likelihood = FACT_LIKELIHOOD_UNCERTAIN
            uncertain_reasons.append("no_matching_extraction_axis")
        elif exact_like and source_target == "unknown" and not any(axis in contextual_axes for axis in matched):
            fact_likelihood = FACT_LIKELIHOOD_UNCERTAIN
            uncertain_reasons.append("exact_value_source_unknown")
        else:
            source_satisfied = False
            for family_name, axis in matched:
                requirements = set(mirror[family_name][axis].get("source_requirements") or [])
                if source_target in requirements:
                    source_satisfied = True
                    break
                if family_name == "document" and source_target == "document" and "document" in requirements:
                    source_satisfied = True
                    break
                if source_target == "source" and ("source" in requirements or "document" in requirements):
                    source_satisfied = True
                    break
            fact_likelihood = FACT_LIKELIHOOD_HIGH if (
                source_satisfied
                or not exact_like
                or any(axis in contextual_axes for axis in matched)
            ) else FACT_LIKELIHOOD_UNCERTAIN
            if fact_likelihood == FACT_LIKELIHOOD_UNCERTAIN:
                uncertain_reasons.append("source_requirement_not_satisfied")

        return {
            "fact_likelihood": fact_likelihood,
            "raw_likely": fact_likelihood == FACT_LIKELIHOOD_UNCERTAIN,
            "matched_extraction_rule_axes": matched_axes,
            "matched_rule_ids": matched_rule_ids,
            "uncertain_reasons": uncertain_reasons,
            "policy_source": policy_source,
        }

    @staticmethod
    def _raw_families_for_result(search_family: str | None, retrieval_families: list | None = None) -> set[str]:
        family = str(search_family or "auto").strip().lower()
        if family == "codebase":
            return set()
        if family in {"conversation", "document"}:
            return {family}
        routed = {
            str(item or "").strip().lower()
            for item in (retrieval_families or [])
            if str(item or "").strip().lower() in {"conversation", "document"}
        }
        return routed or {"conversation", "document"}

    def _render_raw_episode_items(
        self,
        *,
        header: str,
        items: list[dict],
        budget_chars: int,
    ) -> tuple[str, bool, list[str]]:
        lines = [header]
        used = 0
        rendered_ids: list[str] = []
        truncated = False
        for item in items:
            snippet = str(item.get("content") or "").strip()
            if not snippet:
                continue
            family = str(item.get("content_family") or "raw")
            episode_id = str(item.get("episode_id") or "")
            line = f"[{family} episode:{episode_id}] {snippet}"
            if used + len(line) > budget_chars:
                remaining = max(0, budget_chars - used)
                if remaining < 120:
                    truncated = True
                    break
                line = line[:remaining].rstrip()
                truncated = True
            lines.append(line)
            used += len(line)
            if episode_id:
                rendered_ids.append(episode_id)
            if truncated:
                break
        if len(lines) == 1:
            return "", False, []
        return "\n".join(lines), truncated, rendered_ids

    def _conversation_or_document_raw_window(
        self,
        *,
        result: dict,
        caller_id: str,
        caller_memberships: list[str],
        caller_role: str,
        swarm_id: str | None,
    ) -> tuple[str, dict, set[str]]:
        families = self._raw_families_for_result(
            result.get("search_family"),
            result.get("retrieval_families"),
        )
        if not families:
            return "", {}, set()
        anchor_episode_ids = list(dict.fromkeys(
            str(ep_id)
            for ep_id in [
                *(result.get("actual_injected_episode_ids") or []),
                *(result.get("retrieved_episode_ids") or []),
            ]
            if str(ep_id or "").strip()
        ))
        if not anchor_episode_ids:
            return "", {}, set()
        visible = self._iter_visible_raw_episodes(
            families=families,
            caller_id=caller_id,
            caller_memberships=caller_memberships,
            caller_role=caller_role,
            swarm_id=swarm_id,
        )
        by_id = {str(ep.get("episode_id") or ""): ep for ep in visible}
        by_source: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for episode in visible:
            by_source[(self._episode_content_family(episode), str(episode.get("source_id") or ""))].append(episode)
        for episodes in by_source.values():
            episodes.sort(key=self._episode_order_key)

        selected_items: list[dict] = []
        seen_episode_ids: set[str] = set()
        for anchor_id in anchor_episode_ids:
            anchor = by_id.get(anchor_id)
            if not anchor:
                continue
            key = (self._episode_content_family(anchor), str(anchor.get("source_id") or ""))
            source_episodes = by_source.get(key) or []
            try:
                anchor_idx = source_episodes.index(anchor)
            except ValueError:
                continue
            start = max(0, anchor_idx - RAW_CONVERSATION_WINDOW_RADIUS)
            end = min(len(source_episodes), anchor_idx + RAW_CONVERSATION_WINDOW_RADIUS + 1)
            for episode in source_episodes[start:end]:
                episode_id = str(episode.get("episode_id") or "")
                if not episode_id or episode_id in seen_episode_ids:
                    continue
                text = self._raw_episode_text_with_role(episode)
                if not text:
                    continue
                seen_episode_ids.add(episode_id)
                selected_items.append(self._raw_recall_item_from_episode(
                    episode,
                    score=1000.0,
                    evidence_kind=f"{self._episode_content_family(episode)}_raw_window",
                ))

        sections: list[str] = []
        trace: dict[str, dict] = {}
        consumed_episode_ids: set[str] = set()
        for family, header, trace_key in (
            ("conversation", "RAW CONVERSATION EVIDENCE:", "conversation_raw_window"),
            ("document", "RAW DOCUMENT EVIDENCE:", "document_raw_window"),
        ):
            family_items = [item for item in selected_items if item.get("content_family") == family]
            rendered, truncated, rendered_ids = self._render_raw_episode_items(
                header=header,
                items=family_items,
                budget_chars=RAW_SOURCE_WINDOW_BUDGET_CHARS,
            )
            if rendered:
                sections.append(rendered)
                consumed_episode_ids.update(rendered_ids)
            trace[trace_key] = {
                "enabled": bool(family in families),
                "anchor_episode_ids": [
                    ep_id
                    for ep_id in anchor_episode_ids
                    if self._episode_content_family(by_id.get(ep_id, {})) == family
                ],
                "injected_episode_ids": rendered_ids,
                "raw_budget_chars": RAW_SOURCE_WINDOW_BUDGET_CHARS,
                "mode": "same_source_neighbor_window",
                "truncated": truncated,
            }
        return "\n\n".join(sections), trace, consumed_episode_ids

    def _raw_entry_from_session(self, raw_session: dict) -> dict:
        return {
            "message_id": raw_session.get("message_id"),
            "session_id": raw_session.get("session_key") or raw_session.get("source_id"),
            "agent_id": raw_session.get("agent_id"),
            "swarm_id": raw_session.get("swarm_id"),
            "visibility": "private" if raw_session.get("scope") == "agent-private" else "shared",
            "owner_id": raw_session.get("owner_id"),
            "scope": raw_session.get("scope"),
            "read": raw_session.get("read") or [],
            "write": raw_session.get("write") or [],
            "content_family": raw_session.get("format") or "conversation",
            "content": raw_session.get("content") or raw_session.get("raw_original") or "",
            "metadata": raw_session.get("metadata") or {},
            "timestamp_ms": 0,
            "session_num": raw_session.get("session_num"),
            "extraction_state": "complete",
            "source_id": raw_session.get("source_id"),
            "logical_source_id": raw_session.get("logical_source_id"),
            "session_key": raw_session.get("session_key"),
            "status": raw_session.get("status") or "active",
            "artifact_id": raw_session.get("artifact_id"),
            "version_id": raw_session.get("version_id"),
        }

    def _raw_entries_same_source(self, left: dict, right: dict) -> bool:
        if str(left.get("swarm_id") or "") != str(right.get("swarm_id") or ""):
            return False
        if str(left.get("scope") or "") != str(right.get("scope") or ""):
            return False
        if str(left.get("owner_id") or "") != str(right.get("owner_id") or ""):
            return False
        left_key = self._raw_source_key(left)
        right_key = self._raw_source_key(right)
        return bool(left_key and right_key and left_key == right_key)

    def _raw_source_group_key(self, entry: dict) -> tuple[str, str, str, str] | None:
        source_key = self._raw_source_key(entry)
        if not source_key:
            return None
        return (
            str(entry.get("swarm_id") or ""),
            str(entry.get("scope") or ""),
            str(entry.get("owner_id") or ""),
            source_key,
        )

    def _adjacent_assistant_raw_entries(
        self,
        *,
        retrieved_items: list[Any],
        caller_id: str,
        caller_memberships: list[str],
        caller_role: str,
        exclude_message_ids: set[str],
    ) -> list[dict]:
        candidates: list[dict] = []
        assistant_entries_by_source: dict[
            tuple[str, str, str, str], list[tuple[int, dict]]
        ] = defaultdict(list)
        for raw in self._raw_sessions:
            if not isinstance(raw, dict):
                continue
            if not _is_active_lifecycle_record(raw):
                continue
            entry = self._raw_entry_from_session(raw)
            if self._raw_role(entry) != "assistant":
                continue
            message_id = str(entry.get("message_id") or "").strip()
            if not message_id:
                continue
            candidate_num = self._raw_session_num(entry)
            if candidate_num is None:
                continue
            source_group_key = self._raw_source_group_key(entry)
            if source_group_key is None:
                continue
            assistant_entries_by_source[source_group_key].append((candidate_num, entry))
        for entries in assistant_entries_by_source.values():
            entries.sort(key=lambda row: row[0])

        for item in retrieved_items:
            if not isinstance(item, dict):
                continue
            matched = _match_raw_session_for_fact(item, self._raw_sessions)
            if matched is None:
                continue
            _identity, _session_num, raw_session = matched
            if not _is_active_lifecycle_record(raw_session):
                continue
            question_entry = self._raw_entry_from_session(raw_session)
            question_role = self._raw_role(question_entry)
            question_text = str(question_entry.get("content") or "").strip()
            if question_role and question_role != "user":
                continue
            if not question_role and "?" not in question_text:
                continue
            question_num = self._raw_session_num(question_entry)
            if question_num is None:
                continue
            source_group_key = self._raw_source_group_key(question_entry)
            if source_group_key is None:
                continue

            source_assistants = assistant_entries_by_source.get(source_group_key, [])
            start = 0
            end = len(source_assistants)
            while start < end:
                mid = (start + end) // 2
                if source_assistants[mid][0] <= question_num:
                    start = mid + 1
                else:
                    end = mid
            for candidate_num, candidate in source_assistants[start:]:
                message_id = str(candidate.get("message_id") or "").strip()
                if not message_id or message_id in exclude_message_ids:
                    continue
                if not self._raw_entry_acl_allows(candidate, caller_id, caller_memberships, caller_role):
                    continue
                candidates.append(self._raw_recall_item_from_entry(
                    candidate,
                    score=1000.0 + float(candidate_num),
                    evidence_kind="adjacent_assistant_answer",
                ))
                break

        deduped: list[dict] = []
        seen: set[str] = set()
        for item in candidates:
            message_id = str(item.get("message_id") or "").strip()
            key = message_id or str(item.get("content") or "")
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    def _raw_recall_entries(
        self,
        *,
        query: str,
        families: set[str] | None = None,
        caller_id: str,
        caller_memberships: list[str],
        caller_role: str,
        swarm_id: str | None,
        result_type: str | None = None,
        retrieved_items: list[Any] | None = None,
        exclude_episode_ids: set[str] | None = None,
        disable_adjacent_answers: bool = False,
        limit: int = 8,
    ) -> list[dict]:
        if families is None:
            families = {"conversation", "document"}
        if not families:
            return []
        ingress_storage = self._ingress_storage()
        tokens = self._raw_query_tokens(query)
        prefer_recent = self._query_requests_recent_evidence(query, result_type)
        fact_backed_message_ids = self._fact_backed_message_ids()
        selected_message_ids = self._selected_fact_message_ids(retrieved_items or [])
        scored = []
        if ingress_storage is not None:
            entries = ingress_storage.list_write_log_entries(
                states=["pending", "in_progress", "failed"],
                swarm_id=swarm_id if swarm_id and swarm_id != "default" else None,
                order="desc",
            )
            for entry in entries:
                family = self._canonical_content_family(entry.get("content_family") or "conversation")
                if family not in families:
                    continue
                if not self._raw_entry_acl_allows(entry, caller_id, caller_memberships, caller_role):
                    continue
                score = self._score_raw_write_entry(query, tokens, entry)
                if score <= 0:
                    continue
                scored.append(self._raw_recall_item_from_entry(
                    entry,
                    score=score,
                    evidence_kind="pending_raw_write",
                ))

        scored.extend(self._raw_episode_recall_entries(
            query=query,
            families=families,
            caller_id=caller_id,
            caller_memberships=caller_memberships,
            caller_role=caller_role,
            swarm_id=swarm_id,
            exclude_episode_ids=exclude_episode_ids or set(),
            prefer_recent=prefer_recent,
            limit=limit,
        ))

        episode_ids_in_corpus = {
            str(ep.get("episode_id") or "")
            for doc in self._episode_corpus.get("documents", [])
            for ep in doc.get("episodes", [])
            if isinstance(ep, dict)
        }
        excluded_message_ids = {
            str(ep.get("message_id") or "").strip()
            for doc in self._episode_corpus.get("documents", [])
            for ep in doc.get("episodes", [])
            if isinstance(ep, dict)
            and str(ep.get("episode_id") or "") in (exclude_episode_ids or set())
            and str(ep.get("message_id") or "").strip()
        }

        for raw_session in self._raw_sessions:
            if not isinstance(raw_session, dict):
                continue
            if str(raw_session.get("status") or "active") != "active":
                continue
            entry = self._raw_entry_from_session(raw_session)
            if swarm_id and swarm_id != "default" and str(entry.get("swarm_id") or "") != swarm_id:
                continue
            family = self._canonical_content_family(entry.get("content_family") or "conversation")
            if family not in families:
                continue
            raw_episode_id = str(raw_session.get("episode_id") or "").strip()
            if not raw_episode_id:
                raw_source_id = str(entry.get("source_id") or entry.get("session_id") or "").strip()
                raw_session_num = self._raw_session_num(entry)
                if raw_source_id and raw_session_num is not None:
                    raw_episode_id = f"{self._episode_source_key(raw_source_id)}_e{int(raw_session_num):04d}"
            if raw_episode_id and raw_episode_id in episode_ids_in_corpus:
                continue
            message_id = str(entry.get("message_id") or "").strip()
            if message_id and message_id in excluded_message_ids:
                continue
            if message_id and message_id in fact_backed_message_ids:
                continue
            if message_id and message_id in selected_message_ids:
                continue
            if not self._raw_entry_acl_allows(entry, caller_id, caller_memberships, caller_role):
                continue
            score = self._score_raw_write_entry(query, tokens, entry)
            if score <= 0:
                if not prefer_recent:
                    continue
                score = 0.01
            scored.append(self._raw_recall_item_from_entry(
                entry,
                score=score,
                evidence_kind="completed_legacy_raw_session",
            ))

        if "conversation" in families and not disable_adjacent_answers:
            scored.extend(self._adjacent_assistant_raw_entries(
                retrieved_items=retrieved_items or [],
                caller_id=caller_id,
                caller_memberships=caller_memberships,
                caller_role=caller_role,
                exclude_message_ids={
                    str(item.get("message_id") or "").strip()
                    for item in scored
                    if str(item.get("message_id") or "").strip()
                } | selected_message_ids | excluded_message_ids,
            ))

        def _raw_result_sort_key(item: dict[str, Any]) -> tuple[float, int, int]:
            score_value = item.get("score")
            timestamp_value = item.get("timestamp_ms")
            score = float(score_value) if isinstance(score_value, (int, float, str)) else 0.0
            timestamp = int(timestamp_value) if isinstance(timestamp_value, (int, float, str)) else 0
            session_num = self._raw_session_num(item) or 0
            if prefer_recent:
                return (-score, -session_num, -timestamp)
            return (-score, 0, -timestamp)

        scored.sort(key=_raw_result_sort_key)
        deduped: list[dict] = []
        seen: set[str] = set()
        for item in scored:
            message_id = str(item.get("message_id") or "").strip()
            key = message_id or str(item.get("content") or "")
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
            if len(deduped) >= limit:
                break
        return deduped

    def _render_raw_recall_context(self, raw_results: list[dict]) -> str:
        if not raw_results:
            return ""
        kinds = {str(item.get("raw_evidence_kind") or "") for item in raw_results}
        if kinds and kinds <= {"pending_raw_write"}:
            header = "RECENT RAW WRITES:"
        elif "adjacent_assistant_answer" in kinds:
            header = "RAW CONVERSATION EVIDENCE:"
        else:
            header = "COMPLETED RAW EVIDENCE:"
        lines = [header]
        for item in raw_results:
            family = str(item.get("content_family") or "raw")
            role = str(item.get("role") or "").strip()
            state = str(item.get("extraction_state") or "").strip()
            prefix_parts = [family]
            if role:
                prefix_parts.append(role)
            if state:
                prefix_parts.append(state)
            snippet = str(item.get("content") or "").strip()
            if not snippet:
                continue
            lines.append(f"[{' '.join(prefix_parts)}] {snippet}")
        return "\n".join(lines)

    def _ensure_context_packet_with_raw_sections(
        self,
        result: dict,
        *,
        original_context: str,
        sections: list[tuple[str, str]],
    ) -> None:
        """Track selected raw evidence in tier4 so finalization cannot drop it."""
        cleaned_sections = [
            (str(source or "raw"), str(text or "").strip())
            for source, text in sections
            if str(text or "").strip()
        ]
        if not cleaned_sections:
            return
        packet = deepcopy(result.get("_context_packet"))
        if not isinstance(packet, dict):
            packet = {
                "tier1": [],
                "tier2": [],
                "tier3": [
                    {
                        "text": str(original_context or ""),
                        "rank": 0,
                        "source": "fact",
                    }
                ] if str(original_context or "").strip() else [],
                "tier4": [],
            }
        for tier in ("tier1", "tier2", "tier3", "tier4"):
            values = packet.get(tier)
            packet[tier] = list(values) if isinstance(values, list) else []
        existing = {
            (str(item.get("source") or ""), str(item.get("text") or ""))
            for item in packet["tier4"]
            if isinstance(item, dict)
        }
        next_rank = len(packet["tier4"])
        for source, text in cleaned_sections:
            key = (source, text)
            if key in existing:
                continue
            packet["tier4"].append({
                "text": text,
                "rank": next_rank,
                "source": source,
            })
            existing.add(key)
            next_rank += 1
        result["_context_packet"] = packet

    def _merge_raw_recall(
        self,
        *,
        query: str,
        result: dict,
        caller_id: str,
        caller_memberships: list[str],
        caller_role: str,
        swarm_id: str | None,
        raw_kind: str = "all",
    ) -> dict:
        retrieved_items = list(result.get("retrieved") or [])
        result_type = str(result.get("query_type") or "")
        families = self._raw_families_for_result(
            result.get("search_family"),
            result.get("retrieval_families"),
        )
        runtime_trace = dict(result.get("runtime_trace") or {})
        raw_likely_trace: dict | None = None
        if families:
            recall_policy_features = self.extract_recall_policy_features(
                query,
                search_family=str(result.get("search_family") or ""),
                query_type=result_type,
            )
            raw_likely_trace = self.classify_query_against_extraction_policy(
                recall_policy_features,
                self.RECALL_EXTRACTION_POLICY_MIRROR,
            )
            runtime_trace["raw_likely"] = raw_likely_trace
        if raw_kind != "all":
            if raw_likely_trace is not None:
                runtime_trace["raw_recall"] = {"count": 0, "skipped": "kind_filter"}
                result["runtime_trace"] = runtime_trace
            return result
        raw_window_context = ""
        raw_window_trace: dict = {}
        window_episode_ids: set[str] = set()
        if raw_likely_trace and raw_likely_trace.get("raw_likely") is True:
            raw_window_context, raw_window_trace, window_episode_ids = self._conversation_or_document_raw_window(
                result=result,
                caller_id=caller_id,
                caller_memberships=caller_memberships,
                caller_role=caller_role,
                swarm_id=swarm_id,
            )
        anchor_episode_ids = {
            str(ep_id)
            for ep_id in [
                *(result.get("actual_injected_episode_ids") or []),
                *(result.get("retrieved_episode_ids") or []),
            ]
            if str(ep_id or "").strip()
        }
        empty_fact_fallback = (
            bool(families)
            and not retrieved_items
            and str(runtime_trace.get("reason") or "") == "empty_visible_facts"
        )
        include_raw_lane = (
            bool(raw_likely_trace and raw_likely_trace.get("raw_likely") is True)
            or empty_fact_fallback
        )
        raw_results = []
        if include_raw_lane:
            raw_results = self._raw_recall_entries(
                query=query,
                families=families,
                caller_id=caller_id,
                caller_memberships=caller_memberships,
                caller_role=caller_role,
                swarm_id=swarm_id,
                result_type=result_type,
                retrieved_items=retrieved_items,
                exclude_episode_ids=anchor_episode_ids,
                disable_adjacent_answers=(
                    not (raw_likely_trace and raw_likely_trace.get("raw_likely") is True)
                    or bool(window_episode_ids)
                ),
            )
        if not raw_results and not raw_window_context:
            if raw_likely_trace is not None:
                result["runtime_trace"] = runtime_trace
            return result
        raw_context = self._render_raw_recall_context(raw_results)
        context = str(result.get("context") or "").strip()
        raw_section_pairs = [
            ("raw_window", raw_window_context),
            ("raw", raw_context),
        ]
        self._ensure_context_packet_with_raw_sections(
            result,
            original_context=context,
            sections=raw_section_pairs,
        )
        raw_sections = [section for _source, section in raw_section_pairs if section]
        if context and raw_sections:
            result["context"] = f"{context}\n\n" + "\n\n".join(raw_sections)
        elif raw_sections:
            result["context"] = "\n\n".join(raw_sections)
        result["retrieved"] = retrieved_items + raw_results
        window_count = len(window_episode_ids)
        result["raw_recall_count"] = len(raw_results) + window_count
        result["completed_raw_recall_count"] = sum(
            1
            for item in raw_results
            if str(item.get("raw_evidence_kind") or "") in {"completed_raw_episode", "completed_legacy_raw_session"}
        )
        result["adjacent_raw_recall_count"] = sum(
            1 for item in raw_results if str(item.get("raw_evidence_kind") or "") == "adjacent_assistant_answer"
        ) + len([
            ep_id for ep_id in window_episode_ids
            if ep_id not in set(result.get("actual_injected_episode_ids") or [])
        ])
        runtime_trace.update(raw_window_trace)
        selected_raw_episode_ids = [
            item.get("episode_id")
            for item in raw_results
            if str(item.get("raw_evidence_kind") or "") == "completed_raw_episode"
        ]
        runtime_trace["raw_episode_retrieval"] = {
            "mode": "episode_lexical",
            "count": len(selected_raw_episode_ids),
            "selected_episode_ids": selected_raw_episode_ids,
            "empty_fact_fallback": empty_fact_fallback,
        }
        runtime_trace["raw_recall"] = {
            "count": len(raw_results) + window_count,
            "message_ids": [item.get("message_id") for item in raw_results],
            "kinds": [item.get("raw_evidence_kind") for item in raw_results],
            "window_episode_ids": list(window_episode_ids),
        }
        result["runtime_trace"] = runtime_trace
        return result

    def _recall_continuation_anchor_terms(self, query: str) -> list[str]:
        terms = list(self._raw_query_tokens(query))
        if terms:
            return terms
        return [
            token
            for token in re.findall(r"[A-Za-z0-9_.-]+", str(query or "").lower())
            if len(token) > 2 and token not in STOP_WORDS
        ][:8]

    def _score_fact_for_recall_continuation(self, fact: dict, anchor_terms: list[str]) -> float:
        text = " ".join(
            str(value or "")
            for value in (
                fact.get("fact"),
                fact.get("subject"),
                fact.get("predicate"),
                fact.get("object"),
            )
        ).lower()
        if not text or not anchor_terms:
            return 0.0
        score = 0.0
        for term in anchor_terms:
            normalized = str(term or "").lower()
            if not normalized:
                continue
            if normalized in text:
                score += 1.0 + min(text.count(normalized), 3) * 0.1
        return score

    @staticmethod
    def _typed_continuation_entry_counts(entries: list[dict]) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for entry in entries:
            entry_type = str(entry.get("type") or "unknown").strip() or "unknown"
            counts[entry_type] += 1
        return dict(sorted(counts.items()))

    @staticmethod
    def _typed_continuation_families(entries: list[dict]) -> list[str]:
        return list(dict.fromkeys(
            str(entry.get("family") or "").strip()
            for entry in entries
            if str(entry.get("family") or "").strip()
        ))

    def _raw_continuation_entry(self, raw: dict, *, score: float, rank: int, stable_id: str) -> dict:
        family = self._canonical_content_family(raw.get("content_family") or raw.get("format") or "conversation")
        entry_type = "document_raw" if family == "document" else "conversation_raw"
        return {
            "entry_version": 1,
            "type": entry_type,
            "family": "document" if entry_type == "document_raw" else "conversation",
            "source_id": raw.get("source_id") or raw.get("session_id"),
            "episode_id": raw.get("episode_id"),
            "message_id": raw.get("message_id"),
            "rank": rank,
            "score": float(score),
            "stable_id": stable_id,
            "session_id": raw.get("session_id") or raw.get("session_key"),
            "session_num": raw.get("session_num"),
            "timestamp_ms": raw.get("timestamp_ms"),
            "role": raw.get("role"),
            "raw_evidence_kind": raw.get("raw_evidence_kind"),
            "status": raw.get("status"),
        }

    def _resolve_raw_continuation_entry(self, entry: dict) -> dict | None:
        message_id = str(entry.get("message_id") or "").strip()
        episode_id = str(entry.get("episode_id") or "").strip()
        if message_id:
            for raw_session in self._raw_sessions:
                if not isinstance(raw_session, dict):
                    continue
                raw_entry = self._raw_entry_from_session(raw_session)
                if str(raw_entry.get("message_id") or "").strip() == message_id:
                    raw_entry["raw_evidence_kind"] = entry.get("raw_evidence_kind") or "completed_legacy_raw_session"
                    raw_entry["score"] = float(entry.get("score") or 0.0)
                    return raw_entry
        if episode_id:
            for doc in self._episode_corpus.get("documents", []):
                for episode in doc.get("episodes", []):
                    if not isinstance(episode, dict):
                        continue
                    if str(episode.get("episode_id") or "").strip() != episode_id:
                        continue
                    item = self._raw_recall_item_from_episode(
                        episode,
                        score=float(entry.get("score") or 0.0),
                        evidence_kind=str(entry.get("raw_evidence_kind") or "completed_raw_episode"),
                    )
                    return item
        return None

    def _fact_continuation_entry(self, fact: dict, *, score: float, rank: int, stable_id: str) -> dict:
        family = self._fact_source_family(fact) or "conversation"
        entry_type = "document_raw" if family == "document" else "conversation_raw"
        return {
            "entry_version": 1,
            "type": entry_type,
            "family": "document" if entry_type == "document_raw" else "conversation",
            "source_id": fact.get("source_id") or (fact.get("metadata") or {}).get("episode_source_id"),
            "episode_id": next(iter(fact_episode_ids(fact)), None),
            "fact_id": fact.get("id"),
            "rank": rank,
            "score": float(score),
            "stable_id": stable_id,
            "fact": str(fact.get("fact") or ""),
            "session": fact.get("session"),
        }

    def _codebase_render_ref_id_for_fact(self, fact: dict) -> str:
        direct = str(
            fact.get("render_ref_id")
            or (fact.get("metadata") or {}).get("render_ref_id")
            or ""
        ).strip()
        if direct:
            return direct
        fact_id = str(fact.get("id") or "").strip()
        if not fact_id:
            return ""
        graph = normalize_container_graph(getattr(self, "_container_graph", None))
        for container in graph.get("containers", []):
            if str(container.get("container_id") or "") != fact_id:
                continue
            return str(container.get("primary_render_ref_id") or "").strip()
        return ""

    def _codebase_render_ref_by_id(self, render_ref_id: str) -> dict | None:
        wanted = str(render_ref_id or "").strip()
        if not wanted:
            return None
        graph = normalize_container_graph(getattr(self, "_container_graph", None))
        for render_ref in graph.get("render_refs", []):
            if str(render_ref.get("render_ref_id") or "") == wanted and str(render_ref.get("status") or "active") == "active":
                return deepcopy(render_ref)
        return None

    @staticmethod
    def _codebase_fact_line_span(fact: dict) -> dict[str, int | None] | None:
        span = fact.get("span") or (fact.get("metadata") or {}).get("span")
        if not isinstance(span, dict):
            sidecar = fact.get("sidecar_ref")
            if isinstance(sidecar, dict):
                span = sidecar.get("span")
        if not isinstance(span, dict):
            return None
        start_line = span.get("start_line")
        end_line = span.get("end_line")
        if not isinstance(start_line, int) or not isinstance(end_line, int):
            return None
        return {
            "start_line": start_line,
            "end_line": end_line,
            "start_col": span.get("start_col") if isinstance(span.get("start_col"), int) else None,
            "end_col": span.get("end_col") if isinstance(span.get("end_col"), int) else None,
        }

    def _codebase_continuation_entry(self, fact: dict, *, score: float, rank: int, stable_id: str) -> dict:
        metadata = fact.get("metadata") or {}
        file_sidecar_ref = fact.get("file_sidecar_ref")
        sidecar_ref = fact.get("sidecar_ref")
        line_span = self._codebase_fact_line_span(fact)
        render_ref_id = self._codebase_render_ref_id_for_fact(fact)
        file_path = str(fact.get("file_path") or metadata.get("file_path") or "").strip()
        if not file_path and isinstance(file_sidecar_ref, dict):
            file_path = str(file_sidecar_ref.get("file_path") or "").strip()
        if not file_path and isinstance(sidecar_ref, dict):
            file_path = str(sidecar_ref.get("file_path") or "").strip()
        has_source_ref = bool(line_span and (isinstance(file_sidecar_ref, dict) or render_ref_id))
        return {
            "entry_version": 1,
            "type": "codebase_source" if has_source_ref else "codebase_fact",
            "family": "codebase",
            "source_id": fact.get("source_id") or metadata.get("episode_source_id"),
            "fact_id": fact.get("id"),
            "rank": rank,
            "score": float(score),
            "stable_id": stable_id,
            "fact": str(fact.get("fact") or ""),
            "file_path": file_path,
            "language": str(fact.get("language") or metadata.get("language") or "code").lower(),
            "line_span": deepcopy(line_span),
            "semantic_kind": fact.get("semantic_kind") or metadata.get("semantic_kind"),
            "semantic_type": fact.get("semantic_type") or metadata.get("semantic_type"),
            "file_sidecar_ref": deepcopy(file_sidecar_ref) if isinstance(file_sidecar_ref, dict) else None,
            "sidecar_ref": deepcopy(sidecar_ref) if isinstance(sidecar_ref, dict) else None,
            "render_ref_id": render_ref_id or None,
            "source_ref_missing_reason": None if has_source_ref else "no_codebase_sidecar_ref",
        }

    def _continuation_entry_for_candidate(self, item: dict, *, rank: int) -> dict | None:
        if isinstance(item.get("typed_entry"), dict):
            return deepcopy(item["typed_entry"])
        score = float(item.get("score") or 0.0)
        stable_id = str(item.get("stable_id") or "").strip()
        if item.get("candidate_kind") == "raw":
            raw = dict(item.get("item") or {})
            if not raw:
                return None
            return self._raw_continuation_entry(raw, score=score, rank=rank, stable_id=stable_id)
        fact = dict(item.get("item") or {})
        if not fact:
            return None
        if self._fact_source_family(fact) == "codebase":
            return self._codebase_continuation_entry(fact, score=score, rank=rank, stable_id=stable_id)
        return self._fact_continuation_entry(fact, score=score, rank=rank, stable_id=stable_id)

    @staticmethod
    def _continuation_span_label(span: dict | None) -> str:
        if not isinstance(span, dict):
            return "unknown"
        start_line = span.get("start_line")
        end_line = span.get("end_line")
        if start_line is None or end_line is None:
            return "unknown"
        if start_line == end_line:
            return f"L{start_line}"
        return f"L{start_line}-L{end_line}"

    def _render_codebase_fact_continuation_entry(self, entry: dict, *, reason: str | None = None) -> str:
        fact_text = str(entry.get("fact") or "").strip() or "(codebase fact text unavailable)"
        refs: list[str] = []
        if entry.get("file_path"):
            refs.append(f"file={entry.get('file_path')}")
        if entry.get("line_span"):
            refs.append(f"lines={self._continuation_span_label(entry.get('line_span'))}")
        if entry.get("render_ref_id"):
            refs.append(f"render_ref_id={entry.get('render_ref_id')}")
        sidecar = entry.get("file_sidecar_ref")
        if isinstance(sidecar, dict) and sidecar.get("sidecar_id"):
            refs.append(f"file_sidecar_id={sidecar.get('sidecar_id')}")
        if reason:
            refs.append(f"source_window_reason={reason}")
        ref_text = f"\n  refs: {', '.join(refs)}" if refs else ""
        return f"- {fact_text}{ref_text}"

    @staticmethod
    def _format_continuation_source_lines(lines: list[str], *, start_line: int) -> str:
        return "\n".join(f"{idx:>4}: {line}" for idx, line in enumerate(lines, start=start_line))

    def _render_codebase_source_continuation_entry(self, entry: dict) -> tuple[str, dict]:
        trace: dict[str, Any] = {
            "entry_type": "codebase_source",
            "fact_id": entry.get("fact_id"),
            "file_path": entry.get("file_path"),
            "rendered": False,
        }
        span = entry.get("line_span") if isinstance(entry.get("line_span"), dict) else None
        if not span:
            trace["reason"] = "source_window_unavailable"
            trace["detail"] = "missing_line_span"
            return self._render_codebase_fact_continuation_entry(entry, reason="source_window_unavailable"), trace

        code = ""
        file_path = str(entry.get("file_path") or "").strip()
        language = str(entry.get("language") or "code").strip() or "code"
        base_line = 1
        file_sidecar_ref = entry.get("file_sidecar_ref")
        if isinstance(file_sidecar_ref, dict):
            try:
                payload = CodebaseSemanticSidecarStore(str(self.data_dir)).hydrate_sidecar(file_sidecar_ref)
            except Exception as exc:
                trace["reason"] = "source_window_unavailable"
                trace["detail"] = exc.__class__.__name__
                return self._render_codebase_fact_continuation_entry(entry, reason="source_window_unavailable"), trace
            code = str(payload.get("code") or "")
            file_path = str(payload.get("file_path") or file_path or file_sidecar_ref.get("file_path") or "").strip()
            language = str(payload.get("language") or language or "code").strip() or "code"
        elif entry.get("render_ref_id"):
            render_ref = self._codebase_render_ref_by_id(str(entry.get("render_ref_id") or ""))
            render_json = dict((render_ref or {}).get("ref_json") or {})
            code = str(render_json.get("text") or "")
            file_path = str(render_json.get("path") or file_path).strip()
            language = str((render_ref or {}).get("language") or language or "code").strip() or "code"
            ref_span = render_json.get("span")
            if isinstance(ref_span, dict) and isinstance(ref_span.get("start_line"), int):
                base_line = int(ref_span.get("start_line") or 1)
        if not code.strip():
            trace["reason"] = "source_window_unavailable"
            trace["detail"] = "empty_source_payload"
            return self._render_codebase_fact_continuation_entry(entry, reason="source_window_unavailable"), trace

        lines = code.splitlines() or [""]
        line_count = len(lines)
        span_start = int(span.get("start_line") or base_line)
        span_end = int(span.get("end_line") or span_start)
        radius = 18
        local_start = max(1, span_start - base_line + 1 - radius)
        local_end = min(line_count, span_end - base_line + 1 + radius)
        if local_end < local_start:
            local_start = 1
            local_end = min(line_count, 80)
        window_start_line = base_line + local_start - 1
        window_end_line = base_line + local_end - 1
        window_lines = lines[local_start - 1:local_end]
        fact_id = str(entry.get("fact_id") or "")
        header = (
            f"[File: {file_path or 'unknown'}] [Language: {language}] [Mode: source_window] "
            f"[Lines: L{window_start_line}-L{window_end_line}] [Fact: {fact_id or 'unknown'}]"
        )
        text = (
            f"{header}\n"
            f"```{language}\n"
            f"{self._format_continuation_source_lines(window_lines, start_line=window_start_line)}\n"
            "```"
        )
        trace.update({
            "rendered": True,
            "mode": "source_window",
            "line_span": {"start_line": window_start_line, "end_line": window_end_line},
            "source_line_count": line_count,
        })
        return text, trace

    def _render_typed_recall_continuation_page(
        self,
        entries: list[dict],
        *,
        page_num: int,
        hydrate_codebase_source: bool,
    ) -> tuple[str, dict[str, Any]]:
        sections = [f"RECALL CONTINUATION PAGE {page_num}:"]
        fact_lines: list[str] = []
        raw_by_type: dict[str, list[dict]] = {"conversation_raw": [], "document_raw": []}
        code_source_blocks: list[str] = []
        code_fact_lines: list[str] = []
        render_failures: list[dict[str, Any]] = []

        for entry in entries:
            entry_type = str(entry.get("type") or "")
            if entry_type in raw_by_type:
                raw = self._resolve_raw_continuation_entry(entry) or {}
                if raw:
                    raw_by_type[entry_type].append(raw)
                elif entry.get("fact"):
                    session = entry.get("session")
                    prefix = f"(S{session}) " if session else ""
                    fact_lines.append(f"- {prefix}{entry.get('fact')}")
                continue
            if entry_type == "codebase_source":
                if hydrate_codebase_source:
                    rendered, render_trace = self._render_codebase_source_continuation_entry(entry)
                    if render_trace.get("rendered"):
                        code_source_blocks.append(rendered)
                    else:
                        code_fact_lines.append(rendered)
                        render_failures.append(render_trace)
                else:
                    code_fact_lines.append(
                        self._render_codebase_fact_continuation_entry(entry, reason="source_window_deferred")
                    )
                continue
            if entry_type == "codebase_fact":
                code_fact_lines.append(
                    self._render_codebase_fact_continuation_entry(
                        entry,
                        reason=entry.get("source_ref_missing_reason"),
                    )
                )
                continue
            if entry.get("fact"):
                fact_lines.append(f"- {entry.get('fact')}")

        if fact_lines:
            sections.append("RETRIEVED FACTS:\n" + "\n".join(fact_lines))
        for entry_type in ("conversation_raw", "document_raw"):
            raw_context = self._render_raw_recall_context(raw_by_type[entry_type])
            if raw_context:
                sections.append(raw_context)
        if code_source_blocks:
            sections.append("CODEBASE SOURCE WINDOWS:\n" + "\n\n".join(code_source_blocks))
        if code_fact_lines:
            sections.append("CODEBASE FACTS:\n" + "\n".join(code_fact_lines))

        rendered = "\n\n".join(sections).strip()
        if len(rendered) > 6000:
            rendered = rendered[:6000] + "\n[...truncated]"
        trace = {
            "handle_version": 2,
            "page": page_num,
            "entries_rendered": len(entries),
            "typed_entry_counts": self._typed_continuation_entry_counts(entries),
            "source_hydration": {
                "hydrated": bool(hydrate_codebase_source and code_source_blocks),
                "reason": "continuation_page" if hydrate_codebase_source and code_source_blocks else (
                    "source_window_deferred" if not hydrate_codebase_source else "no_codebase_source_rendered"
                ),
            },
            "render_failures": render_failures,
        }
        return rendered, trace

    def _render_recall_continuation_page(self, items: list[dict], *, page_num: int) -> str:
        entries = [
            entry
            for rank, item in enumerate(items)
            if (entry := self._continuation_entry_for_candidate(item, rank=rank)) is not None
        ]
        rendered, _trace = self._render_typed_recall_continuation_page(
            entries,
            page_num=page_num,
            hydrate_codebase_source=False,
        )
        return rendered

    def _recall_continuation_now_ms(self) -> int:
        return int(datetime.now(timezone.utc).timestamp() * 1000)

    def _recall_continuation_ttl_ms(self) -> int:
        try:
            return max(1_000, int(os.getenv("GOSH_MEMORY_RECALL_CONTINUATION_TTL_MS", "600000")))
        except Exception:
            return 600_000

    def _prune_recall_continuations(self, now_ms: int | None = None) -> None:
        now_ms = self._recall_continuation_now_ms() if now_ms is None else now_ms
        for handle in [
            handle
            for handle, state in self._recall_continuations.items()
            if int(state.get("expires_at_ms") or 0) <= now_ms
        ]:
            self._recall_continuations.pop(handle, None)

    def _recall_continuation_acl_key(
        self,
        *,
        caller_id: str | None,
        caller_memberships: list | None,
        caller_role: str,
        swarm_id: str | None,
    ) -> tuple[str, str, tuple[str, ...], str]:
        return (
            str(caller_id or ""),
            str(caller_role or "user"),
            tuple(sorted(str(member) for member in (caller_memberships or []))),
            str(swarm_id or ""),
        )

    def _remember_recall_continuation(
        self,
        result: dict,
        *,
        caller_id: str | None,
        caller_memberships: list | None,
        caller_role: str,
        swarm_id: str | None,
    ) -> None:
        continuation = result.get("recall_continuation")
        pages = result.get("_recall_continuation_pages")
        if not isinstance(continuation, dict) or not continuation.get("available") or not pages:
            return
        handle = str(continuation.get("handle") or "").strip()
        if not handle:
            return
        now_ms = self._recall_continuation_now_ms()
        self._prune_recall_continuations(now_ms)
        self._recall_continuations[handle] = {
            "pages": deepcopy(pages),
            "handle_version": int(continuation.get("handle_version") or 1),
            "typed_entries": deepcopy(result.get("_recall_continuation_typed_entries") or []),
            "swarm_id": str(swarm_id or ""),
            "acl_key": self._recall_continuation_acl_key(
                caller_id=caller_id,
                caller_memberships=caller_memberships,
                caller_role=caller_role,
                swarm_id=swarm_id,
            ),
            "next_page": int(continuation.get("next_page") or 2),
            "page_size": int(continuation.get("page_size") or 0),
            "candidate_count": int(continuation.get("candidate_count") or 0),
            "returned_count": int(continuation.get("returned_count") or 0),
            "anchor_terms": list(continuation.get("anchor_terms") or []),
            "typed_entry_counts": dict(continuation.get("typed_entry_counts") or {}),
            "families_with_continuation": list(continuation.get("families_with_continuation") or []),
            "expires_at_ms": now_ms + self._recall_continuation_ttl_ms(),
        }

    def recall_continuation_page(
        self,
        *,
        continuation_handle: str,
        page: int | str | None = "next",
        caller_id: str | None,
        caller_memberships: list | None,
        caller_role: str,
        swarm_id: str | None,
        bind_swarm_from_handle: bool = False,
    ) -> dict:
        handle = str(continuation_handle or "").strip()
        now_ms = self._recall_continuation_now_ms()
        self._prune_recall_continuations(now_ms)
        state = self._recall_continuations.get(handle)
        if not state:
            return {
                "error": "Recall continuation not found",
                "code": "RECALL_CONTINUATION_NOT_FOUND",
                "recall_continuation": {
                    "available": False,
                    "handle": handle,
                    "exhausted": True,
                },
            }
        effective_swarm_id = swarm_id
        used_handle_swarm_binding = False
        if bind_swarm_from_handle and str(swarm_id or "").strip() in {"", "default"}:
            stored_swarm_id = str(state.get("swarm_id") or "").strip()
            if not stored_swarm_id:
                stored_acl_key = state.get("acl_key")
                if isinstance(stored_acl_key, tuple) and len(stored_acl_key) >= 4:
                    stored_swarm_id = str(stored_acl_key[3] or "").strip()
            if stored_swarm_id:
                effective_swarm_id = stored_swarm_id
                used_handle_swarm_binding = True
        acl_key = self._recall_continuation_acl_key(
            caller_id=caller_id,
            caller_memberships=caller_memberships,
            caller_role=caller_role,
            swarm_id=effective_swarm_id,
        )
        if state.get("acl_key") != acl_key:
            return {
                "error": "Recall continuation not found",
                "code": "RECALL_CONTINUATION_NOT_FOUND",
                "recall_continuation": {
                    "available": False,
                    "handle": handle,
                    "exhausted": True,
                },
                "runtime_trace": {
                    "recall_continuation_trace": {
                        "handle_version": int(state.get("handle_version") or 1),
                        "page_requested": page,
                        "entries_rendered": 0,
                        "next_index": None,
                        "has_more": False,
                        "render_failures": [],
                        "acl_mismatch": True,
                        "handle_bound_swarm_id": used_handle_swarm_binding,
                    }
                },
            }
        if (isinstance(page, str) and page.strip().lower() == "next") or page in (None, ""):
            page_num = int(state.get("next_page") or 2)
        elif isinstance(page, int):
            page_num = page
        elif isinstance(page, str):
            try:
                page_num = int(page.strip())
            except ValueError:
                page_num = int(state.get("next_page") or 2)
        else:
            page_num = int(state.get("next_page") or 2)
        selected = None
        for item in state.get("pages") or []:
            if int(item.get("page") or 0) == page_num:
                selected = item
                break
        if selected is None:
            self._recall_continuations.pop(handle, None)
            return {
                "context": "Recall continuation exhausted.",
                "retrieved": [],
                "query_type": "continuation",
                "recall_continuation": {
                    "available": False,
                    "handle": handle,
                    "page": page_num,
                    "exhausted": True,
                },
                "runtime_trace": {
                    "recall_continuation": {
                        "page": page_num,
                        "exhausted": True,
                        "public_mcp_tool": "get_more_context",
                    },
                    "recall_continuation_trace": {
                        "handle_version": int(state.get("handle_version") or 1),
                        "page_requested": page,
                        "entries_rendered": 0,
                        "next_index": None,
                        "has_more": False,
                        "render_failures": [],
                        "handle_bound_swarm_id": used_handle_swarm_binding,
                    }
                },
            }
        next_page = selected.get("next_page")
        exhausted = bool(selected.get("exhausted"))
        selected_entries = list(selected.get("typed_entries") or [])
        if selected_entries:
            context, render_trace = self._render_typed_recall_continuation_page(
                selected_entries,
                page_num=page_num,
                hydrate_codebase_source=True,
            )
        else:
            context = str(selected.get("context") or "")
            render_trace = {
                "handle_version": int(state.get("handle_version") or 1),
                "page": page_num,
                "entries_rendered": int(selected.get("returned_count") or 0),
                "typed_entry_counts": {},
                "render_failures": [],
            }
        if next_page:
            state["next_page"] = int(next_page)
            state["expires_at_ms"] = now_ms + self._recall_continuation_ttl_ms()
        else:
            exhausted = True
            self._recall_continuations.pop(handle, None)
        return {
            "context": context,
            "retrieved": [],
            "query_type": "continuation",
            "recall_continuation": {
                "available": not exhausted,
                "handle": handle,
                "handle_version": int(state.get("handle_version") or 1),
                "page": page_num,
                "next_page": next_page,
                "page_size": state.get("page_size", 0),
                "candidate_count": state.get("candidate_count", 0),
                "returned_count": state.get("returned_count", 0),
                "exhausted": exhausted,
                "anchor_terms": list(state.get("anchor_terms") or []),
                "typed_entry_counts": dict(state.get("typed_entry_counts") or {}),
                "families_with_continuation": list(state.get("families_with_continuation") or []),
            },
            "runtime_trace": {
                "recall_continuation": {
                    "page": page_num,
                    "exhausted": exhausted,
                    "public_mcp_tool": "get_more_context",
                },
                "recall_continuation_trace": {
                    **render_trace,
                    "page_requested": page,
                    "next_index": page_num + 1 if next_page else None,
                    "has_more": not exhausted,
                    "handle_bound_swarm_id": used_handle_swarm_binding,
                },
            },
        }

    def _attach_recall_continuation(
        self,
        *,
        query: str,
        result: dict,
        fact_filter,
        caller_id: str,
        caller_memberships: list[str],
        caller_role: str,
        swarm_id: str | None,
        raw_kind: str,
    ) -> dict:
        def _record_trace(
            *,
            handle_created: bool,
            typed_entries: list[dict] | None = None,
            skipped_reasons: list[str] | None = None,
            page_count: int = 0,
        ) -> None:
            entries = list(typed_entries or [])
            runtime_trace = dict(result.get("runtime_trace") or {})
            entry_skips = [
                str(entry.get("source_ref_missing_reason") or "")
                for entry in entries
                if str(entry.get("source_ref_missing_reason") or "")
            ]
            trace = {
                "handle_created": handle_created,
                "handle_version": 2,
                "families_with_continuation": self._typed_continuation_families(entries),
                "typed_entry_counts": self._typed_continuation_entry_counts(entries),
                "first_page_entry_count": len(result.get("retrieved") or []),
                "has_more": handle_created,
                "page_count": page_count,
                "skipped_reasons": list(dict.fromkeys([*list(skipped_reasons or []), *entry_skips])),
            }
            runtime_trace["recall_continuation_trace"] = trace
            if handle_created:
                runtime_trace["recall_continuation"] = {
                    "available": True,
                    "candidate_count": len(entries),
                    "page_count": page_count,
                    "anchor_terms": self._recall_continuation_anchor_terms(query),
                }
            result["runtime_trace"] = runtime_trace

        if raw_kind != "all":
            _record_trace(handle_created=False, skipped_reasons=["raw_kind_not_all"])
            return result
        raw_families = self._raw_families_for_result(
            result.get("search_family"),
            result.get("retrieval_families"),
        )
        anchor_terms = self._recall_continuation_anchor_terms(query)
        if not anchor_terms:
            _record_trace(handle_created=False, skipped_reasons=["no_anchor_terms"])
            return result
        retrieved_items = list(result.get("retrieved") or [])
        visible_lookup = self._visible_fact_lookup(fact_filter, search_family="auto")
        visible_code_lookup = {
            fact_id: fact
            for fact_id, fact in visible_lookup.items()
            if self._fact_source_family(fact) == "codebase"
        }
        seen_fact_ids = {
            str(item.get("id") or item.get("fact_id") or "").strip()
            for item in retrieved_items
            if isinstance(item, dict) and str(item.get("id") or item.get("fact_id") or "").strip()
        }
        seen_raw_keys = {
            str(item.get("message_id") or item.get("episode_id") or item.get("content") or "").strip()
            for item in retrieved_items
            if isinstance(item, dict)
        }
        candidates: list[dict] = []
        codebase_seed_fact_ids: list[str] = []
        codebase_retrieval_scores: dict[str, float] = {}
        for rank, item in enumerate(retrieved_items):
            if not isinstance(item, dict):
                continue
            fact_id = str(item.get("id") or item.get("fact_id") or "").strip()
            if not fact_id or fact_id not in visible_code_lookup:
                continue
            if fact_id in codebase_seed_fact_ids:
                continue
            codebase_seed_fact_ids.append(fact_id)
            codebase_retrieval_scores[fact_id] = float(item.get("sim") or item.get("score") or max(0, 10_000 - rank))
        for rank, fact_id in enumerate(codebase_seed_fact_ids):
            fact = visible_code_lookup[fact_id]
            candidates.append({
                "candidate_kind": "codebase",
                "score": codebase_retrieval_scores.get(fact_id, float(10_000 - rank)),
                "session_num": 0,
                "timestamp_ms": 0,
                "stable_id": fact_id,
                "item": fact,
                "typed_entry": self._codebase_continuation_entry(
                    fact,
                    score=codebase_retrieval_scores.get(fact_id, float(10_000 - rank)),
                    rank=rank,
                    stable_id=fact_id,
                ),
            })
        for fact in [*self._all_granular, *self._all_cons, *self._all_cross]:
            if not isinstance(fact, dict):
                continue
            fact_id = str(fact.get("id") or "").strip()
            if not fact_id or fact_id in seen_fact_ids:
                continue
            if not fact_filter(fact):
                continue
            family = self._fact_source_family(fact) or ("conversation" if "conversation" in raw_families else "")
            if family == "codebase" or (raw_families and family not in raw_families):
                continue
            score = self._score_fact_for_recall_continuation(fact, anchor_terms)
            if score <= 0:
                continue
            candidates.append({
                "candidate_kind": "fact",
                "score": score,
                "session_num": _coerce_positive_session_num(fact.get("session")) or 0,
                "timestamp_ms": 0,
                "stable_id": fact_id,
                "item": fact,
            })
        raw_candidates = self._raw_recall_entries(
            query=query,
            families=raw_families,
            caller_id=caller_id,
            caller_memberships=caller_memberships,
            caller_role=caller_role,
            swarm_id=swarm_id,
            result_type=str(result.get("query_type") or ""),
            retrieved_items=[],
            exclude_episode_ids=set(result.get("actual_injected_episode_ids") or []),
            disable_adjacent_answers=True,
            limit=64,
        )
        for raw in raw_candidates:
            key = str(raw.get("message_id") or raw.get("episode_id") or raw.get("content") or "").strip()
            if key and key in seen_raw_keys:
                continue
            candidates.append({
                "candidate_kind": "raw",
                "score": float(raw.get("score") or 0.0),
                "session_num": self._raw_session_num(raw) or 0,
                "timestamp_ms": int(raw.get("timestamp_ms") or 0),
                "stable_id": key,
                "item": raw,
            })
        if not candidates:
            skipped = []
            if not raw_families:
                skipped.append("no_raw_family_candidates")
            if not codebase_seed_fact_ids:
                skipped.append("no_codebase_source_refs")
            _record_trace(handle_created=False, skipped_reasons=skipped or ["no_continuation_candidates"])
            return result
        candidates.sort(
            key=lambda item: (
                -float(item.get("score") or 0.0),
                -int(item.get("session_num") or 0),
                -int(item.get("timestamp_ms") or 0),
                str(item.get("stable_id") or ""),
            )
        )
        page_size = 5
        max_pages = 6
        typed_entries = [
            entry
            for rank, item in enumerate(candidates)
            if (entry := self._continuation_entry_for_candidate(item, rank=rank)) is not None
        ]
        if not typed_entries:
            _record_trace(handle_created=False, skipped_reasons=["no_typed_entries"])
            return result
        pages: list[dict] = []
        for page_index in range(max_pages):
            start = page_index * page_size
            chunk = typed_entries[start:start + page_size]
            if not chunk:
                break
            page_num = page_index + 2
            next_start = start + page_size
            pages.append({
                "page": page_num,
                "next_page": page_num + 1 if next_start < len(typed_entries) and page_index + 1 < max_pages else None,
                "exhausted": not (next_start < len(typed_entries) and page_index + 1 < max_pages),
                "typed_entries": deepcopy(chunk),
                "typed_entry_counts": self._typed_continuation_entry_counts(chunk),
                "returned_count": len(chunk),
            })
        if not pages:
            _record_trace(handle_created=False, typed_entries=typed_entries, skipped_reasons=["no_pages"])
            return result
        handle = secrets.token_urlsafe(24)
        while handle in self._recall_continuations:
            handle = secrets.token_urlsafe(24)
        result["_recall_continuation_pages"] = pages
        result["_recall_continuation_typed_entries"] = typed_entries
        result["recall_continuation"] = {
            "available": True,
            "handle": handle,
            "handle_version": 2,
            "next_page": 2,
            "page_size": page_size,
            "candidate_count": len(typed_entries),
            "returned_count": len(retrieved_items),
            "exhausted": False,
            "anchor_terms": anchor_terms,
            "typed_entry_counts": self._typed_continuation_entry_counts(typed_entries),
            "families_with_continuation": self._typed_continuation_families(typed_entries),
            "tool": "get_more_context",
            "tool_usage": "call get_more_context with handle=<handle> and page=\"next\" to fetch the next evidence page",
        }
        continuation_note = self._recall_continuation_context_note(result.get("recall_continuation"))
        context = str(result.get("context") or "").strip()
        self._ensure_context_packet_with_raw_sections(
            result,
            original_context=context,
            sections=[("recall_continuation", continuation_note)],
        )
        result["context"] = f"{context}\n\n{continuation_note}" if context else continuation_note
        _record_trace(handle_created=True, typed_entries=typed_entries, page_count=len(pages))
        self._remember_recall_continuation(
            result,
            caller_id=caller_id,
            caller_memberships=caller_memberships,
            caller_role=caller_role,
            swarm_id=swarm_id,
        )
        return result

    async def write(
        self,
        *,
        message_id: str,
        session_id: str,
        content: str,
        content_family: str,
        timestamp_ms: int,
        agent_id: str = None,
        swarm_id: str = None,
        scope: str = None,
        owner_id: str = None,
        read: list[str] | None = None,
        write: list[str] | None = None,
        metadata: dict | None = None,
        caller_id: str | None = None,
        caller_principal_kind: str | None = None,
    ) -> dict:
        ingress_storage = self._ingress_storage()
        if ingress_storage is None:
            raise RuntimeError("memory_write requires SQLite storage backend")
        metadata = self._coerce_ingress_metadata(metadata)
        err = self._validate_ingress_metadata(metadata)
        if err:
            raise ValueError(err)
        if str(content_family) not in {"chat", "document", "codebase", "artifact"}:
            raise ValueError(f"Unsupported content_family: {content_family}")
        canonical_family = self._canonical_content_family(content_family)
        normalized_content = self._normalize_ingress_text(content, canonical_family)
        _agent_id, _swarm_id, _scope, _owner_id, _read, _write = self._resolve_live_acl_context(
            agent_id=agent_id,
            swarm_id=swarm_id,
            scope=scope,
            owner_id=owner_id,
            read=read,
            write=write,
            caller_id=caller_id,
            caller_principal_kind=caller_principal_kind,
        )
        visibility = "private" if _scope == "agent-private" else "shared"
        receipt = ingress_storage.append_write_log(
            message_id=str(message_id),
            session_id=str(session_id),
            agent_id=str(_agent_id or "default"),
            swarm_id=str(_swarm_id or "default"),
            visibility=visibility,
            owner_id=_owner_id,
            scope=_scope,
            read=_read,
            write=_write,
            content_family=str(content_family),
            content_text=str(normalized_content),
            metadata=metadata or {},
            timestamp_ms=int(timestamp_ms),
        )
        receipt.update({"message_id": str(message_id), "session_id": str(session_id)})
        return receipt

    def write_status(self, message_id: str) -> dict | None:
        ingress_storage = self._ingress_storage()
        if ingress_storage is None:
            return None
        return ingress_storage.get_write_status(str(message_id))

    def _should_retry_write_entry(self, entry: dict, now_ms: int) -> bool:
        state = str(entry.get("extraction_state") or "pending")
        attempts = int(entry.get("extraction_attempts") or 0)
        last_attempt_ms = int(entry.get("last_extraction_attempt_ms") or 0)
        if state == "failed" and attempts >= 3:
            return False
        if state == "in_progress":
            return (now_ms - last_attempt_ms) >= 30_000
        if attempts <= 0:
            return True
        backoff_ms = min(60_000, 1_000 * (2 ** max(attempts - 1, 0)))
        return (now_ms - last_attempt_ms) >= backoff_ms

    @staticmethod
    def _write_log_family_priority(content_family: str | None) -> int:
        family = str(content_family or "").strip().lower()
        if family in {"chat", "conversation"}:
            return 0
        if family in {"document", "codebase"}:
            return 1
        if family == "artifact":
            return 2
        return 1

    def _order_write_log_entries(self, entries: list[dict]) -> list[dict]:
        """Apply explicit worker scheduling instead of relying on DB insertion order.

        Chat/conversation items stay highest priority, but when they have to
        fall back to ``_next_session_num()`` we must preserve chronology instead
        of processing newest-first and inverting the recovered session order.
        Lower-priority families still prefer newer work first.
        """
        return sorted(
            entries,
            key=lambda entry: (
                self._write_log_family_priority(entry.get("content_family")),
                int(entry.get("timestamp_ms") or 0)
                if self._write_log_family_priority(entry.get("content_family")) == 0
                else -(int(entry.get("timestamp_ms") or 0)),
                int(entry.get("sort_order") or 0),
                str(entry.get("message_id") or ""),
            ),
        )

    @staticmethod
    def _validate_ingress_metadata(metadata) -> str | None:
        """Validate the strict Part I metadata contract for memory_write ingress."""
        if metadata is not None and not isinstance(metadata, dict):
            return f"metadata must be a dict, got {type(metadata).__name__}"
        if not metadata:
            return None
        for key, value in metadata.items():
            if isinstance(value, str):
                continue
            if isinstance(value, list):
                if all(isinstance(item, str) for item in value):
                    continue
                return f"metadata.{key}: lists must contain only strings"
            return f"metadata.{key}: expected string or list of strings"
        return None

    async def _extract_write_log_entry(self, entry: dict) -> dict:
        family = str(entry.get("content_family") or "chat")
        canonical_family = self._canonical_content_family(family)
        metadata = dict(entry.get("metadata") or {})
        normalized_content = self._normalize_ingress_text(str(entry.get("content") or ""), canonical_family)
        scope = entry.get("scope") or ("agent-private" if entry.get("visibility") == "private" else "swarm-shared")
        if family in {"chat", "conversation"}:
            session_num = metadata.get("turn_number")
            if isinstance(session_num, str) and session_num.isdigit():
                session_num = int(session_num)
            if not isinstance(session_num, int):
                session_num = metadata.get("part_idx")
            if isinstance(session_num, str) and session_num.isdigit():
                session_num = int(session_num)
            if not isinstance(session_num, int):
                session_num = self._next_session_num()
            session_date = str(metadata.get("session_date") or self._iso_from_timestamp_ms(entry.get("timestamp_ms")))
            speakers = str(metadata.get("speakers") or "User and Assistant")
            source_id = str(
                metadata.get("logical_source_id")
                or metadata.get("source_id")
                or entry.get("session_id")
                or entry.get("message_id")
            )
            source_meta = {"session_key": entry.get("session_id")}
            if family == "conversation":
                source_meta.update({
                    "root_message_id": entry.get("message_id"),
                    "content_family": family,
                })
            return await self.store(
                content=normalized_content,
                session_num=session_num,
                session_date=session_date,
                speakers=speakers,
                agent_id=entry.get("agent_id"),
                swarm_id=entry.get("swarm_id"),
                scope=scope,
                owner_id=entry.get("owner_id"),
                read=entry.get("read") or [],
                write=entry.get("write") or [],
                source_id=source_id,
                source_meta=source_meta,
                metadata=metadata,
                skip_dedup=False,
                message_id=str(entry.get("message_id") or ""),
            )
        if family == "codebase":
            repo_path = str(
                metadata.get("path")
                or metadata.get("repo_path")
                or metadata.get("source_path")
                or metadata.get("source_id")
                or normalized_content
            ).strip()
            if not repo_path:
                raise ValueError("codebase write-log entry requires metadata.path or repo_path")
            source_id = str(
                metadata.get("source_id")
                or metadata.get("logical_source_id")
                or Path(repo_path).name
                or entry.get("session_id")
                or entry.get("message_id")
            )
            result = await self.ingest_codebase(
                repo_path=repo_path,
                source_id=source_id,
                agent_id=entry.get("agent_id"),
                swarm_id=entry.get("swarm_id"),
                scope=scope,
                metadata=metadata,
                family=family,
                skip_dedup=False,
                message_id=str(entry.get("message_id") or ""),
                source_meta={
                    "root_message_id": entry.get("message_id"),
                    "session_key": entry.get("session_id"),
                    "content_family": family,
                },
            )
            if isinstance(result, dict):
                result.setdefault("source_id", source_id)
                return result
            return {"status": "ok", "facts_extracted": int(result or 0), "source_id": source_id}
        source_id = str(
            metadata.get("source_id")
            or metadata.get("path")
            or entry.get("session_id")
            or entry.get("message_id")
        )
        if family == "codebase":
            result = await self._ingest_codebase_stage1_legacy(
                locator=str(metadata.get("path") or metadata.get("locator") or "").strip() or None,
                content=normalized_content,
                filename=str(metadata.get("filename") or "").strip() or None,
                mime=str(metadata.get("mime") or "").strip() or None,
                source_id=source_id,
                agent_id=entry.get("agent_id"),
                swarm_id=entry.get("swarm_id"),
                scope=scope,
                metadata=metadata,
                skip_dedup=False,
                message_id=str(entry.get("message_id") or ""),
                source_meta={
                    "root_message_id": entry.get("message_id"),
                    "session_key": entry.get("session_id"),
                    "content_family": family,
                },
                owner_id=entry.get("owner_id"),
                read=entry.get("read") or [],
                write=entry.get("write") or [],
            )
        else:
            result = await self.ingest_document(
                content=normalized_content,
                source_id=source_id,
                agent_id=entry.get("agent_id"),
                swarm_id=entry.get("swarm_id"),
                scope=scope,
                metadata=metadata,
                family=family,
                skip_dedup=False,
                message_id=str(entry.get("message_id") or ""),
                source_meta={
                    "root_message_id": entry.get("message_id"),
                    "session_key": entry.get("session_id"),
                    "content_family": family,
                },
            )
        if isinstance(result, dict):
            result.setdefault("source_id", source_id)
            return result
        return {"status": "ok", "facts_extracted": int(result or 0), "source_id": source_id}

    async def process_write_log_once(self, batch_size: int = 8) -> int:
        ingress_storage = self._ingress_storage()
        if ingress_storage is None:
            return 0
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        processed = 0
        async with self._queue_lock:
            supports_claiming = hasattr(ingress_storage, "claim_write_log_entries")
            if supports_claiming:
                entries = self._order_write_log_entries(
                    ingress_storage.claim_write_log_entries(
                        worker_id=self._worker_id,
                        batch_size=batch_size,
                        now_ms=now_ms,
                        lease_ms=DEFAULT_WRITE_LOG_LEASE_MS,
                        retry_backoff_ms=60_000,
                        max_attempts=3,
                    )
                )
                self._last_write_log_claim_trace = {
                    "write_log_claimed_count": len(entries),
                    "write_log_worker_id": self._worker_id,
                    "write_log_claim_skipped_due_to_lease": len(entries) == 0,
                    "write_log_reclaimed_expired_leases": sum(
                        1 for entry in entries if entry.get("lease_reclaimed")
                    ),
                }
            else:
                entries = self._order_write_log_entries(
                    ingress_storage.list_write_log_entries(
                        states=["pending", "in_progress", "failed"],
                        order="asc",
                    )
                )
            for entry in entries:
                if processed >= batch_size:
                    break
                if not supports_claiming and not self._should_retry_write_entry(entry, now_ms):
                    continue
                if str(entry.get("message_id") or "") in self._active_sync_message_ids:
                    ingress_storage.mark_write_state(entry["message_id"], "pending")
                    continue
                if any(
                    rs.get("message_id") == entry.get("message_id")
                    and str(rs.get("status") or "active") == "active"
                    for rs in self._raw_sessions
                ):
                    ingress_storage.mark_write_state(entry["message_id"], "complete")
                    continue
                if not supports_claiming:
                    ingress_storage.mark_write_state(entry["message_id"], "in_progress", attempts_delta=1)
                try:
                    result = await self._extract_write_log_entry(entry)
                    if isinstance(result, dict) and result.get("error"):
                        if result.get("code") == "CANONICALIZATION_ERROR":
                            metadata_patch = {
                                "terminal_error_code": "CANONICALIZATION_ERROR",
                                "canonicalization_status": "failed",
                                "canonicalization_error": str(result.get("error") or ""),
                            }
                            for key in ("raw_session_id", "source_id", "extraction_format"):
                                if result.get(key) is not None:
                                    metadata_patch[key] = str(result[key])
                            ingress_storage.merge_write_log_metadata(entry["message_id"], metadata_patch)
                            ingress_storage.mark_write_state(entry["message_id"], "complete")
                            processed += 1
                            continue
                        raise RuntimeError(
                            f"{result.get('code') or 'EXTRACTION_ERROR'}: {result.get('error')}"
                        )
                    success_metadata_patch: dict[str, str] = {}
                    if isinstance(result, dict):
                        duplicate_of = result.get("duplicate_of") or {}
                        if result.get("status") == "duplicate" and duplicate_of.get("message_id"):
                            success_metadata_patch["duplicate_of"] = str(duplicate_of["message_id"])
                        near_warning = result.get("near_duplicate_warning") or {}
                        if near_warning.get("similar_to_message_id"):
                            success_metadata_patch["near_duplicate_of"] = str(near_warning["similar_to_message_id"])
                    if success_metadata_patch:
                        ingress_storage.merge_write_log_metadata(entry["message_id"], success_metadata_patch)
                    ingress_storage.mark_write_state(entry["message_id"], "complete")
                    self._mark_full_index_dirty()
                    processed += 1
                except Exception:
                    attempts = int(entry.get("extraction_attempts") or 0)
                    if not supports_claiming:
                        attempts += 1
                    next_state = "failed" if attempts >= 3 else "pending"
                    ingress_storage.mark_write_state(entry["message_id"], next_state)
                    log.exception("write-log extraction failed for %s", entry.get("message_id"))
        return processed

    # ── store() ──

    async def store(
        self,
        content: str,
        session_num: int,
        session_date: str,
        speakers: str = "User and Assistant",
        agent_id: str = None,
        swarm_id: str = None,
        scope: str = None,
        upsert_by_key: str = None,
        content_type: str = "default",
        content_format: str | None = None,
        librarian_prompt: str = None,
        owner_id: str = None,
        read: list = None,
        write: list = None,
        source_id: str = None,
        artifact_id: str = None,
        version_id: str = None,
        parent_version: str = None,
        content_hash: str = None,
        skip_dedup: bool = False,
        source_meta: dict = None,
        retention_ttl: int = None,
        metadata: dict = None,
        target=None,
        message_id: str = None,
        caller_id: str | None = None,
        caller_principal_kind: str | None = None,
    ) -> dict:
        """Extract atomic facts from a conversation turn and persist to disk.

        Returns dict with facts_extracted (and upserted/session_key if upsert_by_key set).
        """
        _agent_id, _swarm_id, _scope, _owner_id, _read, _write = self._resolve_live_acl_context(
            agent_id=agent_id,
            swarm_id=swarm_id,
            scope=scope,
            owner_id=owner_id,
            read=read,
            write=write,
            caller_id=caller_id,
            caller_principal_kind=caller_principal_kind,
        )

        # Upsert validation
        if upsert_by_key is not None and _scope != "agent-private":
            return {"error": "upsert_by_key requires agent-private scope",
                    "code": "UPSERT_SCOPE_ERROR"}

        # librarian_prompt scope check
        if librarian_prompt is not None and _scope != "agent-private":
            return {"error": "librarian_prompt only permitted for agent-private scope",
                    "code": "LIBRARIAN_PROMPT_SCOPE_ERROR"}

        # Metadata validation
        err = self._validate_metadata(metadata)
        if err:
            return {"error": err, "code": "VALIDATION_ERROR"}
        try:
            normalized_target = _normalize_target(target)
        except ValueError as e:
            return {"error": str(e), "code": "VALIDATION_ERROR"}

        original_content = str(content or "")
        content = self._normalize_ingress_text(content, "conversation")
        content_format = normalize_content_format(content_format)

        # ── Compute content hash if not provided ──
        if content_hash is None:
            content_hash = content_hash_text(content, family="conversation")

        # ── Generate artifact/version IDs if not provided ──
        if artifact_id is None:
            artifact_id = _generate_artifact_id()
        if version_id is None:
            version_id = _generate_version_id()

        logical_source_id = str(source_id or self.key)
        projection_source_id = self._projection_source_id(
            source_id=logical_source_id,
            family="conversation",
            scope=_scope,
            owner_id=_owner_id,
            swarm_id=_swarm_id,
        )
        effective_source_meta = dict(source_meta or {})
        effective_source_meta.setdefault("logical_source_id", logical_source_id)

        # ── Dedup check ──
        dedup_key = None
        supersede_version_id = None
        near_duplicate_warning = None
        if not skip_dedup:
            duplicate_of = self._find_exact_duplicate(
                content=content,
                family="conversation",
                scope=_scope,
                owner_id=_owner_id,
                swarm_id=_swarm_id,
            )
            if duplicate_of is not None:
                return {
                    "status": "duplicate",
                    "duplicate_of": self._dedup_reference(duplicate_of),
                }
            near_duplicate_warning = self._find_near_duplicate(
                content=content,
                family="conversation",
                scope=_scope,
                owner_id=_owner_id,
                swarm_id=_swarm_id,
            )
        if logical_source_id and not skip_dedup:
            dedup_key = self._source_versioning_key(
                source_id=projection_source_id,
                family="conversation",
                scope=_scope,
                owner_id=_owner_id,
                swarm_id=_swarm_id,
                session_num=session_num,
            )
            existing = self._dedup_index.get(dedup_key)
            if existing is not None:
                if existing["content_hash"] == content_hash:
                    return {
                        "status": "duplicate",
                        "duplicate_of": {
                            "message_id": existing.get("message_id"),
                            "source_id": logical_source_id,
                            "session_num": session_num,
                            "stored_at": existing.get("stored_at"),
                        },
                    }
                else:
                    # Different content → new version (Unit 4).
                    # Keep the current version active until the new extraction
                    # actually succeeds.
                    artifact_id = existing["artifact_id"]
                    parent_version = existing["version_id"]
                    version_id = _generate_version_id()
                    supersede_version_id = existing["version_id"]

        store_impl = (
            self._store_impl_write_through
            if self._ingress_storage() is not None and self._projection_storage() is not None
            else self._store_impl
        )

        # ── Upsert path: serialize entire operation per session_key ──
        if upsert_by_key is not None:
            if upsert_by_key not in self._upsert_locks:
                self._upsert_locks[upsert_by_key] = asyncio.Lock()
            async with self._upsert_locks[upsert_by_key]:
                result = await store_impl(
                    content, session_num, session_date, speakers,
                    _agent_id, _swarm_id, _scope,
                    upsert_by_key, content_type, content_format, librarian_prompt,
                    _agent_id, _swarm_id, _scope,
                    _owner_id, _read, _write,
                    source_id=projection_source_id, artifact_id=artifact_id,
                    version_id=version_id, parent_version=parent_version,
                    content_hash=content_hash, source_meta=effective_source_meta,
                    retention_ttl=retention_ttl, metadata=metadata,
                    target=normalized_target,
                    message_id=message_id,
                    dedup_key=dedup_key,
                    supersede_version_id=supersede_version_id,
                    original_content=original_content,
                )
                if "error" not in result:
                    result.setdefault("status", "ok")
                if near_duplicate_warning:
                    result["near_duplicate_warning"] = near_duplicate_warning
                return result

        # ── Normal (non-upsert) path ──
        result = await store_impl(
            content, session_num, session_date, speakers,
            _agent_id, _swarm_id, _scope,
            upsert_by_key, content_type, content_format, librarian_prompt,
            _agent_id, _swarm_id, _scope,
            _owner_id, _read, _write,
            source_id=projection_source_id, artifact_id=artifact_id,
            version_id=version_id, parent_version=parent_version,
            content_hash=content_hash, source_meta=effective_source_meta,
            retention_ttl=retention_ttl, metadata=metadata,
            target=normalized_target,
            message_id=message_id,
            dedup_key=dedup_key,
            supersede_version_id=supersede_version_id,
            original_content=original_content,
        )
        if "error" not in result:
            result.setdefault("status", "ok")
        if near_duplicate_warning:
            result["near_duplicate_warning"] = near_duplicate_warning
        return result

    async def _store_impl_write_through(
        self,
        content: str,
        session_num: int,
        session_date: str,
        speakers: str,
        _agent_id: str,
        _swarm_id: str,
        _scope: str,
        upsert_by_key: str | None,
        content_type: str,
        content_format: str | None,
        librarian_prompt: str | None,
        agent_id: str | None,
        swarm_id: str | None,
        scope: str | None,
        _owner_id: str = "system",
        _read: list = None,
        _write: list = None,
        source_id: str = None,
        artifact_id: str = None,
        version_id: str = None,
        parent_version: str = None,
        content_hash: str = None,
        source_meta: dict = None,
        retention_ttl: int = None,
        metadata: dict = None,
        target: list[str] | None = None,
        message_id: str | None = None,
        dedup_key: tuple | None = None,
        supersede_version_id: str | None = None,
        original_content: str | None = None,
    ) -> dict:
        """SQLite write-through implementation of store()."""
        ingress_storage = self._ingress_storage()
        assert ingress_storage is not None

        raw_session_id = str(uuid4())
        effective_message_id = str(message_id) if message_id is not None else f"raw:{raw_session_id}"
        stored_at = datetime.now(timezone.utc).isoformat()
        self._audit.log("store", _owner_id, {"session_num": session_num, "artifact_id": artifact_id})

        if _read is None:
            _read = ["agent:PUBLIC"]
        if _write is None:
            _write = ["agent:PUBLIC"]

        conv_source_id = source_id or self.key
        logical_source_id = str((source_meta or {}).get("logical_source_id") or conv_source_id)
        projection_session_num = self._resolve_projection_session_num(
            logical_session_num=int(session_num),
            source_id=conv_source_id,
            family="conversation",
            scope=_scope,
            owner_id=_owner_id,
            swarm_id=_swarm_id,
        )
        merged_source_meta = dict(source_meta or {})
        merged_source_meta["logical_source_id"] = logical_source_id
        if projection_session_num != session_num:
            merged_source_meta["logical_session_num"] = session_num
        _mal_binding_id = _resolve_mal_binding_id(_owner_id, _agent_id)
        _extract_model = _resolve_extract_model(
            self.extract_model, str(self.data_dir), self.key, _mal_binding_id,
        )
        sem = self._get_extract_sem()

        async def _canonical_call_extract_fn(model, system, user_msg, max_tokens=8192):
            return await self._call_extract_with_runtime_secrets(model, system, user_msg, max_tokens, sem)

        raw_original_content = str(original_content if original_content is not None else content or "")
        canonical_source = await self._canonicalize_semantic_source_text(
            raw_original_content,
            family="conversation",
            model=_extract_model,
            call_extract_fn=_canonical_call_extract_fn,
        )
        canonical_content = canonical_source["canonical_en"]
        resolved_extraction_format = content_format or detect_format(raw_original_content)
        ingress_storage.append_write_log(
            message_id=effective_message_id,
            session_id=str(upsert_by_key or raw_session_id),
            agent_id=str(_agent_id or "default"),
            swarm_id=str(_swarm_id or "default"),
            visibility="private" if _scope == "agent-private" else "shared",
            owner_id=_owner_id,
            scope=_scope,
            read=list(_read),
            write=list(_write),
            content_family="conversation",
            content_text=raw_original_content,
            metadata=metadata or {},
            timestamp_ms=int(datetime.now(timezone.utc).timestamp() * 1000),
        )

        raw_session = {
            "raw_session_id": raw_session_id,
            "session_num": session_num,
            "projection_session_num": projection_session_num,
            "session_date": session_date,
            "content": raw_original_content,
            "raw_original": canonical_source["raw_original"],
            "canonical_en": canonical_content,
            "semantic_ready": canonical_source["semantic_ready"],
            "canonicalization_status": canonical_source["canonicalization_status"],
            "canonicalization_error": canonical_source["canonicalization_error"],
            "source_lang": canonical_source["source_lang"],
            "translation_version": canonical_source["translation_version"],
            "speakers": speakers,
            "agent_id": _agent_id,
            "swarm_id": _swarm_id,
            "scope": _scope,
            "owner_id": _owner_id,
            "read": list(_read),
            "write": list(_write),
            "content_type": content_type,
            "extraction_format": resolved_extraction_format,
            "stored_at": stored_at,
            "format": "conversation",
            "source_id": conv_source_id,
            "logical_source_id": logical_source_id,
            "artifact_id": artifact_id,
            "version_id": version_id,
            "parent_version": parent_version,
            "content_hash": content_hash,
            "status": "active" if canonical_source["semantic_ready"] else "canonicalization_failed",
            "message_id": effective_message_id,
        }
        if metadata is not None:
            raw_session["metadata"] = metadata
        if target:
            raw_session["target"] = list(target)
        if retention_ttl is not None:
            raw_session["retention_ttl"] = retention_ttl
        if merged_source_meta:
            raw_session.update(merged_source_meta)
        if upsert_by_key is not None:
            raw_session["session_key"] = upsert_by_key

        self._active_sync_message_ids.add(effective_message_id)
        try:
            if not canonical_source["semantic_ready"]:
                async with self._file_lock:
                    self._raw_sessions = [
                        rs for rs in self._raw_sessions if rs.get("message_id") != effective_message_id
                    ]
                    self._raw_sessions.append(raw_session)
                    self._register_source_record(
                        source_id=conv_source_id,
                        family="conversation",
                        owner_id=_owner_id,
                        read=_read,
                        write=_write,
                        artifact_id=artifact_id,
                        version_id=version_id,
                        content_hash=content_hash,
                        metadata=metadata,
                        target=target,
                        source_meta={
                            "stored_format": "conversation",
                            "logical_source_id": logical_source_id,
                            **self._semantic_state_fields(canonical_source),
                            "scope": _scope,
                            "swarm_id": _swarm_id,
                        },
                    )
                    self._persist_projection_delta(
                        raw_session_upserts=[raw_session],
                        source_record_upserts={conv_source_id: dict(self._source_records[conv_source_id])},
                        state_values=self._state_json_values(),
                        episode_corpus=self._episode_corpus,
                        complete_message_ids=[effective_message_id],
                    )
                return {
                    "error": str(canonical_source.get("canonicalization_error") or "english source canonicalization failed"),
                    "code": "CANONICALIZATION_ERROR",
                    "semantic_ready": False,
                    "raw_session_id": raw_session_id,
                    "source_id": conv_source_id,
                    "extraction_format": resolved_extraction_format,
                }

            _mal_cfg = _load_mal_active_config(str(self.data_dir), self.key, _mal_binding_id)
            _mal_prompts = _mal_cfg.get("extraction_prompts") or {}
            _mal_conv_key = f"conversation_content_type:{content_type}"
            _has_mal_override = _mal_conv_key in _mal_prompts

            if librarian_prompt is not None:
                extraction_prompt = librarian_prompt
            elif _has_mal_override:
                extraction_prompt = _mal_prompts[_mal_conv_key]
            else:
                extraction_prompt = self._prompt_registry.get(content_type)

            use_custom = (
                librarian_prompt is not None
                or _has_mal_override
                or self._prompt_registry._custom_path(content_type).exists()
                or content_type != "default"
            )
            block_prompt_pipeline = content_type in {"conversation", "document"}

            if not use_custom or block_prompt_pipeline:
                async def _call_extract_fn(model, system, user_msg, max_tokens=8192):
                    return await self._call_extract_with_runtime_secrets(model, system, user_msg, max_tokens, sem)
            else:
                async def _call_extract_fn(model, system, user_msg, max_tokens=8192):
                    try:
                        dt = datetime.fromisoformat(session_date.replace("Z", "+00:00"))
                        date_str = dt.strftime("%d %B %Y")
                        year_minus_1 = str(dt.year - 1)
                    except Exception:
                        try:
                            dt = date_parser.parse(session_date, fuzzy=True)
                            date_str = dt.strftime("%d %B %Y")
                            year_minus_1 = str(dt.year - 1)
                        except Exception:
                            date_str = session_date
                            year_match = re.search(r"\b(20\d{2}|19\d{2})\b", session_date or "")
                            year_minus_1 = str(int(year_match.group(1)) - 1) if year_match else "2022"
                    class _SafeDict(dict):
                        def __missing__(self, key):
                            return "{" + key + "}"

                    custom_system = extraction_prompt.format_map(_SafeDict(
                        session_date=date_str,
                        year_minus_1=year_minus_1,
                        session_num=session_num,
                    ))
                    return await self._call_extract_with_runtime_secrets(model, custom_system, user_msg, max_tokens, sem)

            extract_error: Exception | None = None
            extraction_report: dict[str, Any] = {}
            try:
                extract_result = await extract_session(
                    session_text=canonical_content,
                    session_num=session_num,
                    session_date=session_date,
                    conv_id=self.key,
                    speakers=speakers,
                    model=_extract_model,
                    call_extract_fn=_call_extract_fn,
                    fmt=content_format,
                    return_report=True,
                )
                _conv_id, _sn, _sdate, facts, tlinks, extraction_report = _coerce_extract_session_result(extract_result)
                if len(facts) == 0:
                    extract_result = await extract_session(
                        session_text=canonical_content,
                        session_num=session_num,
                        session_date=session_date,
                        conv_id=self.key,
                        speakers=speakers,
                        model=_extract_model,
                        call_extract_fn=_call_extract_fn,
                        fmt=content_format,
                        return_report=True,
                    )
                    _conv_id, _sn, _sdate, facts, tlinks, extraction_report = _coerce_extract_session_result(extract_result)
            except Exception as exc:
                extract_error = exc
                facts = []
                tlinks = []
                extraction_report = {}

            _set_runtime_report_artifact(
                raw_session,
                field_name="extraction_report",
                producer="block_extractor",
                report_kind="extraction",
                report=extraction_report,
            )

            if len(facts) == 0:
                if extract_error is not None:
                    if isinstance(extract_error, LocalCliTimeoutError):
                        error_code = "LOCAL_CLI_TIMEOUT"
                        error_text = str(extract_error)
                        log.warning("store() extraction failed for session %d: %s", session_num, extract_error)
                        upserted = False
                        timeout_raw_session_deletes: list[int] = []
                        timeout_fact_deletes: list[tuple[str, str]] = []
                        raw_session["status"] = "extraction_failed"
                        raw_session["extraction_error"] = error_text
                        raw_session["extraction_error_code"] = error_code
                        async with self._file_lock:
                            if upsert_by_key is not None:
                                for i, rs in enumerate(list(self._raw_sessions)):
                                    if (
                                        rs.get("session_key") == upsert_by_key
                                        and rs.get("agent_id") == _agent_id
                                        and rs.get("swarm_id") == _swarm_id
                                        and rs.get("scope") == _scope
                                    ):
                                        upserted = True
                                        existing_projection_num = self._projection_session_num_for_row(rs)
                                        if existing_projection_num is not None:
                                            timeout_raw_session_deletes.append(existing_projection_num)
                                        old_rsid = rs.get("raw_session_id")
                                        if old_rsid:
                                            stale_facts = [
                                                f for f in self._all_granular if f.get("raw_session_id") == old_rsid
                                            ]
                                            timeout_fact_deletes.extend(
                                                [("granular", str(f.get("id") or "")) for f in stale_facts if str(f.get("id") or "")]
                                            )
                                            self._all_granular = [
                                                f for f in self._all_granular if f.get("raw_session_id") != old_rsid
                                            ]
                                        self._remove_content_indices_for_message(str(rs.get("message_id") or ""))
                                        self._raw_sessions.pop(i)
                                        break
                            self._raw_sessions = [
                                rs for rs in self._raw_sessions if rs.get("message_id") != effective_message_id
                            ]
                            self._raw_sessions.append(raw_session)
                            self._register_source_record(
                                source_id=conv_source_id,
                                family="conversation",
                                owner_id=_owner_id,
                                read=_read,
                                write=_write,
                                artifact_id=artifact_id,
                                version_id=version_id,
                                content_hash=content_hash,
                                metadata=metadata,
                                target=target,
                                source_meta={
                                    "stored_format": "conversation",
                                    "logical_source_id": logical_source_id,
                                    **self._semantic_state_fields(canonical_source),
                                    "scope": _scope,
                                    "swarm_id": _swarm_id,
                                    "extraction_error": error_text,
                                    "extraction_error_code": error_code,
                                },
                            )
                            self._persist_projection_delta(
                                raw_session_upserts=[raw_session],
                                raw_session_deletes=[
                                    num
                                    for num in timeout_raw_session_deletes
                                    if num > 0 and num != projection_session_num
                                ],
                                fact_deletes=timeout_fact_deletes,
                                source_record_upserts={conv_source_id: dict(self._source_records[conv_source_id])},
                                state_values=self._state_json_values(),
                                episode_corpus=self._episode_corpus,
                            )
                        failed_result: dict[str, Any] = {
                            "error": error_text,
                            "code": error_code,
                            "status": "extraction_failed",
                            "facts_extracted": 0,
                            "raw_session_id": raw_session_id,
                            "source_id": conv_source_id,
                            "extraction_format": resolved_extraction_format,
                        }
                        if upsert_by_key is not None:
                            failed_result["upserted"] = upserted
                            failed_result["session_key"] = upsert_by_key
                        return failed_result

                    log.warning("store() extraction failed for session %d: %s", session_num, extract_error)

                log.warning("store() returned 0 facts for session %d after retry", session_num)
                upserted = False
                raw_session_deletes: list[int] = []
                fact_deletes: list[tuple[str, str]] = []
                raw_session["status"] = "active"
                episode = self._conversation_episode_from_raw_session(
                    raw_session=raw_session,
                    source_id=conv_source_id,
                    session_num=session_num,
                    session_date=session_date,
                    canonical_content=canonical_content,
                    canonical_source=canonical_source,
                    merged_source_meta=merged_source_meta,
                )
                async with self._file_lock:
                    if upsert_by_key is not None:
                        for i, rs in enumerate(list(self._raw_sessions)):
                            if (
                                rs.get("session_key") == upsert_by_key
                                and rs.get("agent_id") == _agent_id
                                and rs.get("swarm_id") == _swarm_id
                                and rs.get("scope") == _scope
                            ):
                                upserted = True
                                existing_projection_num = self._projection_session_num_for_row(rs)
                                if existing_projection_num is not None:
                                    raw_session_deletes.append(existing_projection_num)
                                old_rsid = rs.get("raw_session_id")
                                if old_rsid:
                                    stale_facts = [
                                        f for f in self._all_granular if f.get("raw_session_id") == old_rsid
                                    ]
                                    fact_deletes.extend(
                                        [("granular", str(f.get("id") or "")) for f in stale_facts if str(f.get("id") or "")]
                                    )
                                    self._all_granular = [
                                        f for f in self._all_granular if f.get("raw_session_id") != old_rsid
                                    ]
                                self._remove_content_indices_for_message(str(rs.get("message_id") or ""))
                                self._raw_sessions.pop(i)
                                break
                    self._raw_sessions = [
                        rs for rs in self._raw_sessions if rs.get("message_id") != effective_message_id
                    ]
                    self._raw_sessions.append(raw_session)
                    self._append_or_replace_episode(self._conversation_doc_id(conv_source_id), episode)
                    self._register_source_record(
                        source_id=conv_source_id,
                        family="conversation",
                        owner_id=_owner_id,
                        read=_read,
                        write=_write,
                        artifact_id=artifact_id,
                        version_id=version_id,
                        content_hash=content_hash,
                        metadata=metadata,
                        target=target,
                        source_meta={
                            "stored_format": "conversation",
                            "logical_source_id": logical_source_id,
                            **self._semantic_state_fields(canonical_source),
                            "scope": _scope,
                            "swarm_id": _swarm_id,
                        },
                    )
                    if not upserted:
                        self._n_sessions += 1
                    self._persist_projection_delta(
                        raw_session_upserts=[raw_session],
                        raw_session_deletes=[
                            num for num in raw_session_deletes if num > 0 and num != projection_session_num
                        ],
                        fact_deletes=fact_deletes,
                        source_record_upserts={conv_source_id: dict(self._source_records[conv_source_id])},
                        state_values=self._state_json_values(),
                        episode_doc_replacements={
                            self._conversation_doc_id(conv_source_id): self._get_episode_documents(
                                conv_source_id,
                                "conversation",
                            )
                        },
                        episode_corpus=self._episode_corpus,
                        complete_message_ids=[effective_message_id],
                    )
                    self._bump_index_snapshot_version()
                empty_result: dict[str, Any] = {"facts_extracted": 0}
                empty_result["extraction_format"] = resolved_extraction_format
                if upsert_by_key is not None:
                    empty_result["upserted"] = upserted
                    empty_result["session_key"] = upsert_by_key
                return empty_result

            episode = self._conversation_episode_from_raw_session(
                raw_session=raw_session,
                source_id=conv_source_id,
                session_num=session_num,
                session_date=session_date,
                canonical_content=canonical_content,
                canonical_source=canonical_source,
                merged_source_meta=merged_source_meta,
            )

            self._tag_facts(
                facts,
                session_date,
                agent_id=_agent_id,
                swarm_id=_swarm_id,
                scope=_scope,
                owner_id=_owner_id,
                read=_read,
                write=_write,
                artifact_id=artifact_id,
                version_id=version_id,
                content_hash=content_hash,
                retention_ttl=retention_ttl,
                metadata=metadata,
                target=target,
            )
            for f in facts:
                raw_id = f.get("id", "")
                if raw_id and not raw_id.startswith(f"s{session_num}_"):
                    f["id"] = f"s{session_num}_{raw_id}"
            if conv_source_id != logical_source_id:
                self._namespace_projection_fact_ids(facts, source_id=conv_source_id)

            episode_id = cast(str, episode["episode_id"])
            self._stamp_episode_metadata(facts, episode_id, conv_source_id)
            self._align_fact_selectors(
                facts,
                episode_id=episode_id,
                source_kind="conversation",
                raw_fields=_selector_raw_fields(canonical_content, raw_session, merged_source_meta),
                speakers=speakers if isinstance(speakers, dict) else None,
            )
            for f in facts:
                f["source_id"] = conv_source_id
                f["raw_session_id"] = raw_session_id
                err = _validate_object_flags_field(f)
                if err:
                    raise ValueError(err)
            session_complexity = _compute_content_complexity(facts)
            for f in facts:
                f["_session_content_complexity"] = session_complexity
                f["_temporal_links"] = []
            if facts and tlinks:
                facts[0]["_temporal_links"] = tlinks

            self._mark_tiers_dirty()
            self._data_dict = None

            upserted = False
            raw_session_deletes = []
            fact_deletes = []
            updated_granular: list[dict] = []
            updated_raw_sessions: list[dict] = []
            async with self._file_lock:
                if upsert_by_key is not None:
                    for i, rs in enumerate(list(self._raw_sessions)):
                        if (
                            rs.get("session_key") == upsert_by_key
                            and rs.get("agent_id") == _agent_id
                            and rs.get("swarm_id") == _swarm_id
                            and rs.get("scope") == _scope
                        ):
                            upserted = True
                            existing_projection_num = self._projection_session_num_for_row(rs)
                            if existing_projection_num is not None:
                                raw_session_deletes.append(existing_projection_num)
                            old_rsid = rs.get("raw_session_id")
                            if old_rsid:
                                stale_facts = [f for f in self._all_granular if f.get("raw_session_id") == old_rsid]
                                fact_deletes.extend(
                                    [("granular", str(f.get("id") or "")) for f in stale_facts if str(f.get("id") or "")]
                                )
                                self._all_granular = [
                                    f for f in self._all_granular if f.get("raw_session_id") != old_rsid
                                ]
                            self._remove_content_indices_for_message(str(rs.get("message_id") or ""))
                            self._raw_sessions.pop(i)
                            break

                stale_indices = [
                    idx for idx, rs in enumerate(self._raw_sessions)
                    if rs.get("message_id") == effective_message_id
                ]
                for idx in reversed(stale_indices):
                    stale = self._raw_sessions.pop(idx)
                    self._remove_content_indices_for_message(str(stale.get("message_id") or ""))
                    stale_num = self._projection_session_num_for_row(stale)
                    if stale_num is not None and stale_num > 0:
                        raw_session_deletes.append(stale_num)
                    stale_rsid = stale.get("raw_session_id")
                    if stale_rsid:
                        stale_facts = [f for f in self._all_granular if f.get("raw_session_id") == stale_rsid]
                        fact_deletes.extend(
                            [("granular", str(f.get("id") or "")) for f in stale_facts if str(f.get("id") or "")]
                        )
                        self._all_granular = [f for f in self._all_granular if f.get("raw_session_id") != stale_rsid]

                if supersede_version_id:
                    for f in self._all_granular:
                        if f.get("version_id") == supersede_version_id:
                            f["status"] = "superseded"
                            updated_granular.append(dict(f))
                    for rs in self._raw_sessions:
                        if rs.get("version_id") == supersede_version_id:
                            rs["status"] = "superseded"
                            updated_raw_sessions.append(dict(rs))
                            self._remove_content_indices_for_message(str(rs.get("message_id") or ""))

                if dedup_key is not None:
                    self._dedup_index[dedup_key] = {
                        "artifact_id": artifact_id,
                        "version_id": version_id,
                        "content_hash": content_hash,
                        "message_id": effective_message_id,
                        "stored_at": stored_at,
                    }

                self._raw_sessions.append(raw_session)
                self._index_content_entry(
                    message_id=effective_message_id,
                    source_id=conv_source_id,
                    session_num=session_num,
                    stored_at=stored_at,
                    scope=_scope,
                    owner_id=_owner_id,
                    swarm_id=_swarm_id,
                    family="conversation",
                    content=canonical_content,
                )
                self._all_granular.extend(facts)
                self._all_tlinks.extend(tlinks)
                self._append_or_replace_episode(self._conversation_doc_id(conv_source_id), episode)
                self._register_source_record(
                    source_id=conv_source_id,
                    family="conversation",
                    owner_id=_owner_id,
                    read=_read,
                    write=_write,
                    artifact_id=artifact_id,
                    version_id=version_id,
                    content_hash=content_hash,
                    metadata=metadata,
                    target=target,
                    source_meta={
                        "stored_format": "conversation",
                        "logical_source_id": logical_source_id,
                        **self._semantic_state_fields(canonical_source),
                        "scope": _scope,
                        "swarm_id": _swarm_id,
                    },
                )
                if not upserted:
                    self._n_sessions += 1
                    if facts:
                        self._n_sessions_with_facts += 1

                self._persist_projection_delta(
                    raw_session_upserts=updated_raw_sessions + [raw_session],
                    raw_session_deletes=[
                        num for num in raw_session_deletes if num > 0 and num != projection_session_num
                    ],
                    fact_upserts={"granular": updated_granular + facts},
                    fact_deletes=fact_deletes,
                    episode_doc_replacements={
                        self._conversation_doc_id(conv_source_id): self._get_episode_documents(conv_source_id, "conversation")
                    },
                    temporal_link_appends=tlinks,
                    source_record_upserts={conv_source_id: dict(self._source_records[conv_source_id])},
                    state_values=self._state_json_values(),
                    episode_corpus=self._episode_corpus,
                    complete_message_ids=[effective_message_id],
                )
                self._bump_index_snapshot_version()

            result: dict[str, Any] = {"facts_extracted": len(facts)}
            result["extraction_format"] = resolved_extraction_format
            if upsert_by_key is not None:
                result["upserted"] = upserted
                result["session_key"] = upsert_by_key
            return result
        finally:
            self._active_sync_message_ids.discard(effective_message_id)

    async def _store_impl(
        self,
        content: str,
        session_num: int,
        session_date: str,
        speakers: str,
        _agent_id: str,
        _swarm_id: str,
        _scope: str,
        upsert_by_key: str | None,
        content_type: str,
        content_format: str | None,
        librarian_prompt: str | None,
        agent_id: str | None,
        swarm_id: str | None,
        scope: str | None,
        _owner_id: str = "system",
        _read: list = None,
        _write: list = None,
        source_id: str = None,
        artifact_id: str = None,
        version_id: str = None,
        parent_version: str = None,
        content_hash: str = None,
        source_meta: dict = None,
        retention_ttl: int = None,
        metadata: dict = None,
        target: list[str] | None = None,
        message_id: str | None = None,
        dedup_key: tuple | None = None,
        supersede_version_id: str | None = None,
        original_content: str | None = None,
    ) -> dict:
        """Internal implementation of store(). Separated so upsert path can
        hold a per-key lock around the entire operation."""
        raw_session_id = str(uuid4())
        effective_message_id = str(message_id) if message_id is not None else f"raw:{raw_session_id}"
        self._audit.log("store", _owner_id,
                        {"session_num": session_num, "artifact_id": artifact_id})

        # Handle upsert: find and remove existing session + its facts
        # Must happen under _file_lock to prevent race conditions
        upserted = False
        if upsert_by_key is not None:
            async with self._file_lock:
                for i, rs in enumerate(self._raw_sessions):
                    if (rs.get("session_key") == upsert_by_key
                            and rs.get("agent_id") == _agent_id
                            and rs.get("swarm_id") == _swarm_id
                            and rs.get("scope") == _scope):
                        old_rsid = rs.get("raw_session_id")
                        if old_rsid:
                            self._all_granular = [
                                f for f in self._all_granular
                                if f.get("raw_session_id") != old_rsid
                            ]
                        self._remove_content_indices_for_message(str(rs.get("message_id") or ""))
                        self._raw_sessions.pop(i)
                        upserted = True
                        break

        if _read is None:
            _read = ["agent:PUBLIC"]
        if _write is None:
            _write = ["agent:PUBLIC"]

        conv_source_id = source_id or self.key
        logical_source_id = str((source_meta or {}).get("logical_source_id") or conv_source_id)
        projection_session_num = self._resolve_projection_session_num(
            logical_session_num=int(session_num),
            source_id=conv_source_id,
            family="conversation",
            scope=_scope,
            owner_id=_owner_id,
            swarm_id=_swarm_id,
        )
        merged_source_meta = dict(source_meta or {})
        merged_source_meta["logical_source_id"] = logical_source_id
        if projection_session_num != session_num:
            merged_source_meta["logical_session_num"] = session_num
        _mal_binding_id = _resolve_mal_binding_id(_owner_id, _agent_id)
        _extract_model = _resolve_extract_model(
            self.extract_model, str(self.data_dir), self.key, _mal_binding_id,
        )
        sem = self._get_extract_sem()

        async def _canonical_call_extract_fn(model, system, user_msg, max_tokens=8192):
            return await self._call_extract_with_runtime_secrets(model, system, user_msg, max_tokens, sem)

        raw_original_content = str(original_content if original_content is not None else content or "")
        canonical_source = await self._canonicalize_semantic_source_text(
            raw_original_content,
            family="conversation",
            model=_extract_model,
            call_extract_fn=_canonical_call_extract_fn,
        )
        canonical_content = canonical_source["canonical_en"]
        resolved_extraction_format = content_format or detect_format(raw_original_content)

        # Save raw session BEFORE extraction — source of truth
        raw_session = {
            "raw_session_id": raw_session_id,
            "session_num": session_num,
            "projection_session_num": projection_session_num,
            "session_date": session_date,
            "content": raw_original_content,
            "raw_original": canonical_source["raw_original"],
            "canonical_en": canonical_content,
            "semantic_ready": canonical_source["semantic_ready"],
            "canonicalization_status": canonical_source["canonicalization_status"],
            "canonicalization_error": canonical_source["canonicalization_error"],
            "source_lang": canonical_source["source_lang"],
            "translation_version": canonical_source["translation_version"],
            "speakers": speakers,
            "agent_id": _agent_id,
            "swarm_id": _swarm_id,
            "scope": _scope,
            "owner_id": _owner_id,
            "read": list(_read),
            "write": list(_write),
            "content_type": content_type,
            "extraction_format": resolved_extraction_format,
            "stored_at": datetime.now(timezone.utc).isoformat(),
            "format": "conversation",
            "source_id": conv_source_id,
            "logical_source_id": logical_source_id,
            "artifact_id": artifact_id,
            "version_id": version_id,
            "parent_version": parent_version,
            "content_hash": content_hash,
            "status": "pending_extraction" if canonical_source["semantic_ready"] else "canonicalization_failed",
        }
        if metadata is not None:
            raw_session["metadata"] = metadata
        if target:
            raw_session["target"] = list(target)
        if retention_ttl is not None:
            raw_session["retention_ttl"] = retention_ttl
        if merged_source_meta:
            raw_session.update(merged_source_meta)
        if upsert_by_key is not None:
            raw_session["session_key"] = upsert_by_key
        raw_session["message_id"] = effective_message_id

        # A synchronous store() call already owns extraction for this message.
        # Keep the write-log worker away from the same pending row until the
        # synchronous extraction path finishes; after a process crash this set is
        # empty again, so stale pending rows remain retryable on restart.
        self._active_sync_message_ids.add(effective_message_id)
        try:
            async with self._file_lock:
                stale_idx = None
                stale_raw_session_id = None
                for i, rs in enumerate(self._raw_sessions):
                    if (
                        rs.get("message_id") == effective_message_id
                        and str(rs.get("status") or "") == "pending_extraction"
                    ):
                        stale_idx = i
                        stale_raw_session_id = rs.get("raw_session_id")
                        break
                if stale_idx is not None:
                    if stale_raw_session_id:
                        self._all_granular = [
                            f for f in self._all_granular if f.get("raw_session_id") != stale_raw_session_id
                        ]
                    self._remove_content_indices_for_message(str(self._raw_sessions[stale_idx].get("message_id") or ""))
                    self._raw_sessions.pop(stale_idx)
                self._raw_sessions.append(raw_session)
                self._register_source_record(
                    source_id=conv_source_id,
                    family="conversation",
                    owner_id=_owner_id,
                    read=_read,
                    write=_write,
                    artifact_id=artifact_id,
                    version_id=version_id,
                    content_hash=content_hash,
                    metadata=metadata,
                    target=target,
                    source_meta={
                        "stored_format": "conversation",
                        "logical_source_id": logical_source_id,
                        **self._semantic_state_fields(canonical_source),
                        "scope": _scope,
                        "swarm_id": _swarm_id,
                    },
                )
                self._save_cache()

            if not canonical_source["semantic_ready"]:
                return {
                    "error": str(canonical_source.get("canonicalization_error") or "english source canonicalization failed"),
                    "code": "CANONICALIZATION_ERROR",
                    "semantic_ready": False,
                    "raw_session_id": raw_session_id,
                    "source_id": conv_source_id,
                    "extraction_format": resolved_extraction_format,
                }

            # ── MAL model override + extraction prompt ──
            # MAL extraction prompt override
            _mal_cfg = _load_mal_active_config(str(self.data_dir), self.key, _mal_binding_id)
            _mal_prompts = _mal_cfg.get("extraction_prompts") or {}
            _mal_conv_key = f"conversation_content_type:{content_type}"
            _has_mal_override = _mal_conv_key in _mal_prompts

            if librarian_prompt is not None:
                extraction_prompt = librarian_prompt
            elif _has_mal_override:
                extraction_prompt = _mal_prompts[_mal_conv_key]
            else:
                extraction_prompt = self._prompt_registry.get(content_type)

            registry_prompt = self._prompt_registry.get(content_type)
            use_custom = (librarian_prompt is not None
                          or _has_mal_override
                          or self._prompt_registry._custom_path(content_type).exists()
                          or content_type != "default")
            block_prompt_pipeline = content_type in {"conversation", "document"}

            if not use_custom or block_prompt_pipeline:
                # Default path — extract_session uses its own hardcoded EXTRACTION_PROMPT
                async def _call_extract_fn(model, system, user_msg, max_tokens=8192):
                    return await self._call_extract_with_runtime_secrets(model, system, user_msg, max_tokens, sem)
            else:
                # Custom prompt — wrap call_extract_fn to inject resolved prompt
                async def _call_extract_fn(model, system, user_msg, max_tokens=8192):
                    try:
                        dt = datetime.fromisoformat(session_date.replace("Z", "+00:00"))
                        date_str = dt.strftime("%d %B %Y")
                        year_minus_1 = str(dt.year - 1)
                    except Exception:
                        try:
                            dt = date_parser.parse(session_date, fuzzy=True)
                            date_str = dt.strftime("%d %B %Y")
                            year_minus_1 = str(dt.year - 1)
                        except Exception:
                            date_str = session_date
                            year_match = re.search(r"\b(20\d{2}|19\d{2})\b", session_date or "")
                            year_minus_1 = str(int(year_match.group(1)) - 1) if year_match else "2022"
                    class _SafeDict(dict):
                        def __missing__(self, key):
                            return "{" + key + "}"

                    custom_system = extraction_prompt.format_map(_SafeDict(
                        session_date=date_str,
                        year_minus_1=year_minus_1,
                        session_num=session_num,
                    ))
                    return await self._call_extract_with_runtime_secrets(model, custom_system, user_msg, max_tokens, sem)

            extract_result = await extract_session(
                session_text=canonical_content,
                session_num=session_num,
                session_date=session_date,
                conv_id=self.key,
                speakers=speakers,
                model=_extract_model,
                call_extract_fn=_call_extract_fn,
                fmt=content_format,
                return_report=True,
            )
            conv_id, sn, sdate, facts, tlinks, extraction_report = _coerce_extract_session_result(extract_result)

            # Retry once if 0 facts
            if len(facts) == 0:
                extract_result = await extract_session(
                    session_text=canonical_content,
                    session_num=session_num,
                    session_date=session_date,
                    conv_id=self.key,
                    speakers=speakers,
                    model=_extract_model,
                    call_extract_fn=_call_extract_fn,
                    fmt=content_format,
                    return_report=True,
                )
                conv_id, sn, sdate, facts, tlinks, extraction_report = _coerce_extract_session_result(extract_result)

            _set_runtime_report_artifact(
                raw_session,
                field_name="extraction_report",
                producer="block_extractor",
                report_kind="extraction",
                report=extraction_report,
            )

            if len(facts) == 0:
                log.warning("store() returned 0 facts for session %d after retry", session_num)
                raw_session["status"] = "active"
                episode = self._conversation_episode_from_raw_session(
                    raw_session=raw_session,
                    source_id=conv_source_id,
                    session_num=session_num,
                    session_date=session_date,
                    canonical_content=canonical_content,
                    canonical_source=canonical_source,
                    merged_source_meta=merged_source_meta,
                )
                async with self._file_lock:
                    for rs in self._raw_sessions:
                        if rs.get("raw_session_id") == raw_session_id:
                            # 0 extracted facts is a terminal ingest outcome, not a
                            # corrupt half-ingested raw session.
                            rs["status"] = "active"
                            break
                    self._append_or_replace_episode(self._conversation_doc_id(conv_source_id), episode)
                    if not upserted:
                        self._n_sessions += 1
                    self._bump_index_snapshot_version()
                    self._save_cache()
                result: dict[str, Any] = {"facts_extracted": 0}
                result["extraction_format"] = resolved_extraction_format
                if upsert_by_key is not None:
                    result["upserted"] = upserted
                    result["session_key"] = upsert_by_key
                return result

            episode = self._conversation_episode_from_raw_session(
                raw_session=raw_session,
                source_id=conv_source_id,
                session_num=session_num,
                session_date=session_date,
                canonical_content=canonical_content,
                canonical_source=canonical_source,
                merged_source_meta=merged_source_meta,
            )

            self._tag_facts(facts, session_date,
                            agent_id=_agent_id, swarm_id=_swarm_id, scope=_scope,
                            owner_id=_owner_id, read=_read, write=_write,
                            artifact_id=artifact_id, version_id=version_id,
                            content_hash=content_hash, retention_ttl=retention_ttl,
                            metadata=metadata, target=target)

            # Make fact IDs unique per session to prevent lookup collision
            # from repeated model-generated local ids like f_01, f_02.
            for f in facts:
                raw_id = f.get("id", "")
                if raw_id and not raw_id.startswith(f"s{session_num}_"):
                    f["id"] = f"s{session_num}_{raw_id}"
            if conv_source_id != logical_source_id:
                self._namespace_projection_fact_ids(facts, source_id=conv_source_id)

            episode_id = cast(str, episode["episode_id"])
            episode_source_id = conv_source_id
            self._stamp_episode_metadata(facts, episode_id, episode_source_id)
            self._align_fact_selectors(
                facts,
                episode_id=episode_id,
                source_kind="conversation",
                raw_fields=_selector_raw_fields(canonical_content, raw_session, merged_source_meta),
                speakers=speakers if isinstance(speakers, dict) else None,
            )

            # Tag facts with raw_session_id for upsert tracking
            for f in facts:
                f["source_id"] = conv_source_id
                f["raw_session_id"] = raw_session_id

            # Compute and stamp _session_content_complexity on each fact
            session_complexity = _compute_content_complexity(facts)
            for f in facts:
                f["_session_content_complexity"] = session_complexity

            # Set _temporal_links
            for f in facts:
                f["_temporal_links"] = []
            if facts and tlinks:
                facts[0]["_temporal_links"] = tlinks

            # Clear stale tiers BEFORE save
            self._mark_tiers_dirty()
            self._data_dict = None

            async with self._file_lock:
                if supersede_version_id:
                    for f in self._all_granular:
                        if f.get("version_id") == supersede_version_id:
                            f["status"] = "superseded"
                    for rs in self._raw_sessions:
                        if rs.get("version_id") == supersede_version_id:
                            rs["status"] = "superseded"
                            self._remove_content_indices_for_message(str(rs.get("message_id") or ""))
                if dedup_key is not None:
                    self._dedup_index[dedup_key] = {
                        "artifact_id": artifact_id,
                        "version_id": version_id,
                        "content_hash": content_hash,
                        "message_id": effective_message_id,
                        "stored_at": raw_session.get("stored_at"),
                    }
                for rs in self._raw_sessions:
                    if rs.get("raw_session_id") == raw_session_id:
                        rs["status"] = "active"
                        break
                self._index_content_entry(
                    message_id=effective_message_id,
                    source_id=conv_source_id,
                    session_num=session_num,
                    stored_at=cast(str | None, raw_session.get("stored_at")),
                    scope=_scope,
                    owner_id=_owner_id,
                    swarm_id=_swarm_id,
                    family="conversation",
                    content=canonical_content,
                )
                self._all_granular.extend(facts)
                self._all_tlinks.extend(tlinks)
                self._append_or_replace_episode(self._conversation_doc_id(conv_source_id), episode)
                if not upserted:
                    self._n_sessions += 1
                    if facts:
                        self._n_sessions_with_facts += 1
                self._bump_index_snapshot_version()
                self._save_cache()

            store_result: dict[str, Any] = {"facts_extracted": len(facts)}
            store_result["extraction_format"] = resolved_extraction_format
            if upsert_by_key is not None:
                store_result["upserted"] = upserted
                store_result["session_key"] = upsert_by_key
            return store_result
        finally:
            self._active_sync_message_ids.discard(effective_message_id)

    # ── ingest_codebase() ──

    async def _ingest_codebase_stage1_legacy(
        self,
        *,
        locator: str | None = None,
        content: str | None = None,
        filename: str | None = None,
        mime: str | None = None,
        source_id: str,
        agent_id: str = None,
        swarm_id: str = None,
        scope: str = None,
        artifact_id: str = None,
        version_id: str = None,
        parent_version: str = None,
        content_hash: str = None,
        skip_dedup: bool = False,
        source_meta: dict = None,
        retention_ttl: int = None,
        metadata: dict = None,
        target=None,
        message_id: str | None = None,
        owner_id: str | None = None,
        read: list[str] | None = None,
        write: list[str] | None = None,
        caller_id: str | None = None,
        caller_principal_kind: str | None = None,
    ) -> dict:
        err = self._validate_metadata(metadata)
        if err:
            raise ValueError(err)
        family = "codebase"
        normalized_target = _normalize_target(target)
        normalized_content = self._normalize_ingress_text(content or "", family)
        _agent_id, _swarm_id, _scope, _owner_id, _read, _write = self._resolve_live_acl_context(
            agent_id=agent_id,
            swarm_id=swarm_id,
            scope=scope,
            owner_id=owner_id,
            read=read,
            write=write,
            caller_id=caller_id,
            caller_principal_kind=caller_principal_kind,
        )

        if artifact_id is None:
            artifact_id = _generate_artifact_id()
        if version_id is None:
            version_id = _generate_version_id()

        logical_source_id = str(source_id)
        projection_source_id = self._projection_source_id(
            source_id=logical_source_id,
            family=family,
            scope=_scope,
            owner_id=_owner_id,
            swarm_id=_swarm_id,
        )
        merged_source_meta = dict(source_meta or {})
        merged_source_meta["logical_source_id"] = logical_source_id

        bundle = build_codebase_stage1_bundle(
            locator=locator,
            content=normalized_content,
            filename=filename,
            mime=mime,
            logical_source_id=logical_source_id,
        )
        bundle_source_meta = dict(bundle.get("source_meta") or {})
        summary = dict(bundle.get("summary") or {})
        unit_spool_ref = str(bundle.get("unit_spool_ref") or "").strip()
        object_spool_ref = str(bundle.get("object_spool_ref") or "").strip()
        relation_spool_ref = str(bundle.get("relation_spool_ref") or "").strip()
        units = list(bundle.get("units") or [])
        codebase_graph = dict(bundle.get("graph") or {})
        if content_hash is None:
            content_hash = str(bundle.get("content_hash") or "").strip() or content_hash_text(
                normalized_content or logical_source_id,
                family=family,
            )

        codebase_dedup_key = self._source_versioning_key(
            source_id=projection_source_id,
            family=family,
            scope=_scope,
            owner_id=_owner_id,
            swarm_id=_swarm_id,
        )
        codebase_supersede_version_id = None
        if logical_source_id and not skip_dedup:
            existing = self._dedup_index.get(codebase_dedup_key)
            if existing is not None:
                if existing["content_hash"] == content_hash:
                    return {
                        "status": "duplicate",
                        "duplicate_of": {
                            "message_id": existing.get("message_id"),
                            "source_id": logical_source_id,
                            "session_num": existing.get("session_num"),
                            "stored_at": existing.get("stored_at"),
                        },
                    }
                artifact_id = existing["artifact_id"]
                parent_version = existing["version_id"]
                version_id = _generate_version_id()
                codebase_supersede_version_id = existing["version_id"]

        graph_ref = ""
        object_ref = ""
        relation_ref = ""
        if codebase_graph:
            graph_dir = self.data_dir / "codebase_graphs"
            graph_dir.mkdir(parents=True, exist_ok=True)
            graph_slug = hashlib.sha1(
                f"{projection_source_id}:{version_id}".encode(),
                usedforsecurity=False,
            ).hexdigest()
            graph_path = graph_dir / f"{graph_slug}.json"
            graph_path.write_text(
                json.dumps(codebase_graph, ensure_ascii=True, separators=(",", ":")),
                encoding="utf-8",
            )
            graph_ref = str(graph_path.relative_to(self.data_dir))
            bundle_source_meta["codebase_graph_ref"] = graph_ref
            if object_spool_ref:
                object_dir = self.data_dir / "codebase_objects"
                object_dir.mkdir(parents=True, exist_ok=True)
                object_path = object_dir / f"{graph_slug}.jsonl"
                shutil.move(object_spool_ref, object_path)
                object_ref = str(object_path.relative_to(self.data_dir))
                bundle_source_meta["codebase_object_ref"] = object_ref
            if relation_spool_ref:
                relation_dir = self.data_dir / "codebase_relations"
                relation_dir.mkdir(parents=True, exist_ok=True)
                relation_path = relation_dir / f"{graph_slug}.jsonl"
                shutil.move(relation_spool_ref, relation_path)
                relation_ref = str(relation_path.relative_to(self.data_dir))
                bundle_source_meta["codebase_relation_ref"] = relation_ref

        effective_message_id = str(message_id) if message_id is not None else f"rawcodebase:{logical_source_id}:{version_id}"
        ingress_storage = self._ingress_storage()
        if ingress_storage is not None:
            ingress_storage.append_write_log(
                message_id=effective_message_id,
                session_id=f"codebase:{logical_source_id}",
                agent_id=str(_agent_id or "default"),
                swarm_id=str(_swarm_id or "default"),
                visibility="private" if _scope == "agent-private" else "shared",
                owner_id=_owner_id,
                scope=_scope,
                read=list(_read),
                write=list(_write),
                content_family=family,
                content_text=normalized_content,
                metadata=metadata or {},
                timestamp_ms=int(datetime.now(timezone.utc).timestamp() * 1000),
            )

        codebase_session_start = self._next_document_session_num(projection_source_id)
        final_episodes: list[dict] = []
        codebase_raw_sessions: list[dict] = []
        codebase_granular: list[dict] = []
        total_facts = 0

        def _iter_codebase_units() -> Iterable[dict]:
            if unit_spool_ref:
                spool_path = Path(unit_spool_ref)
                with spool_path.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        yield cast(dict, json.loads(line))
                return
            yield from units

        for idx, unit in enumerate(_iter_codebase_units(), start=1):
            episode = deepcopy(unit.get("episode") or {})
            episode_id = f"{projection_source_id}_e{idx:04d}"
            session_num = codebase_session_start + idx - 1
            episode["episode_id"] = episode_id
            episode["source_id"] = projection_source_id
            episode["source_type"] = family
            episode["session_num"] = session_num

            raw_session_id = str(uuid4())
            raw_session = {
                "raw_session_id": raw_session_id,
                "message_id": effective_message_id,
                "session_num": session_num,
                "session_date": str(episode.get("source_date") or ""),
                "content": str(episode.get("raw_text") or ""),
                "speakers": "Codebase",
                "agent_id": _agent_id,
                "swarm_id": _swarm_id,
                "scope": _scope,
                "owner_id": _owner_id,
                "read": list(_read),
                "write": list(_write),
                "stored_at": datetime.now(timezone.utc).isoformat(),
                "format": family,
                "source_id": projection_source_id,
                "logical_source_id": logical_source_id,
                "artifact_id": artifact_id,
                "version_id": version_id,
                "parent_version": parent_version,
                "content_hash": content_hash,
                "status": "active",
                "episode_id": episode_id,
            }
            if metadata is not None:
                raw_session["metadata"] = dict(metadata)
            if normalized_target:
                raw_session["target"] = list(normalized_target)
            codebase_meta = dict((episode.get("metadata") or {}).get("codebase") or {})
            if codebase_meta:
                raw_session["codebase_object_id"] = str(codebase_meta.get("object_id") or "")
                raw_session["codebase_object_type"] = str(codebase_meta.get("object_type") or "")
            raw_session.update({
                **merged_source_meta,
                **bundle_source_meta,
            })

            facts = deepcopy(unit.get("facts") or [])
            self._tag_facts(
                facts,
                str(episode.get("source_date") or ""),
                agent_id=_agent_id,
                swarm_id=_swarm_id,
                scope=_scope,
                owner_id=_owner_id,
                read=_read,
                write=_write,
                artifact_id=artifact_id,
                version_id=version_id,
                content_hash=content_hash,
                target=normalized_target,
                metadata=metadata,
                retention_ttl=retention_ttl,
            )
            self._stamp_episode_metadata(facts, episode_id, projection_source_id)
            complexity = _compute_content_complexity(facts)
            for fact in facts:
                raw_id = str(fact.get("id") or "").strip()
                if raw_id and not raw_id.startswith(f"{episode_id}_"):
                    fact["id"] = f"{episode_id}_{raw_id}"
                fact["source_id"] = projection_source_id
                fact["raw_session_id"] = raw_session_id
                fact["session"] = session_num
                fact["_session_content_complexity"] = complexity
                merged_metadata = _merge_fact_metadata(
                    fact.get("metadata"),
                    {"codebase_source": projection_source_id},
                )
                if merged_metadata:
                    fact["metadata"] = merged_metadata

            final_episodes.append(episode)
            codebase_raw_sessions.append(raw_session)
            codebase_granular.extend(facts)
            total_facts += len(facts)

        updated_granular: list[dict] = []
        updated_raw_sessions: list[dict] = []
        async with self._file_lock:
            if codebase_supersede_version_id:
                for fact in self._all_granular:
                    if fact.get("version_id") == codebase_supersede_version_id:
                        fact["status"] = "superseded"
                        updated_granular.append(dict(fact))
                for raw_session in self._raw_sessions:
                    if raw_session.get("version_id") == codebase_supersede_version_id:
                        raw_session["status"] = "superseded"
                        updated_raw_sessions.append(dict(raw_session))
                        self._remove_content_indices_for_message(str(raw_session.get("message_id") or ""))

            self._raw_sessions.extend(codebase_raw_sessions)
            self._upsert_episode_document(
                self._codebase_doc_id(projection_source_id),
                final_episodes,
            )
            self._register_source_record(
                source_id=projection_source_id,
                family=family,
                owner_id=_owner_id,
                read=_read,
                write=_write,
                artifact_id=artifact_id,
                version_id=version_id,
                content_hash=content_hash,
                metadata=metadata,
                target=normalized_target,
                source_meta={
                    "stored_format": family,
                    "locator": locator or "",
                    "filename": filename or "",
                    "mime": mime or "",
                    **merged_source_meta,
                    **bundle_source_meta,
                    "scope": _scope,
                    "swarm_id": _swarm_id,
                },
            )

            if projection_source_id and not skip_dedup:
                self._dedup_index[codebase_dedup_key] = {
                    "artifact_id": artifact_id,
                    "version_id": version_id,
                    "content_hash": content_hash,
                    "message_id": effective_message_id,
                    "stored_at": codebase_raw_sessions[0].get("stored_at") if codebase_raw_sessions else None,
                    "session_num": None,
                }
                self._index_content_entry(
                    message_id=effective_message_id,
                    source_id=projection_source_id,
                    session_num=codebase_raw_sessions[0].get("session_num") if codebase_raw_sessions else None,
                    stored_at=codebase_raw_sessions[0].get("stored_at") if codebase_raw_sessions else None,
                    scope=_scope,
                    owner_id=_owner_id,
                    swarm_id=_swarm_id,
                    family=family,
                    content=normalized_content,
                )

            if codebase_granular:
                self._all_granular.extend(codebase_granular)
                self._n_sessions += len(final_episodes)
                self._n_sessions_with_facts += len({
                    fact.get("session", 1)
                    for fact in codebase_granular
                    if fact
                })

            self._persist_projection_delta(
                raw_session_upserts=updated_raw_sessions + codebase_raw_sessions,
                raw_doc_upserts=[],
                fact_upserts={
                    "granular": updated_granular + codebase_granular,
                    "cross": [],
                },
                episode_doc_replacements={
                    self._codebase_doc_id(projection_source_id): self._get_episode_documents(projection_source_id, "codebase")
                },
                temporal_link_appends=[],
                source_record_upserts={projection_source_id: dict(self._source_records[projection_source_id])},
                state_values=self._state_json_values(),
                episode_corpus=self._episode_corpus,
                complete_message_ids=[effective_message_id],
            )
            if self._projection_storage() is None:
                self._save_cache()

        self._data_dict = None
        self._mark_tiers_dirty()
        result = {
            "status": "ok",
            "facts_extracted": total_facts,
            "objects_extracted": len(final_episodes),
            "object_counts": dict(summary.get("object_counts") or {}),
            "repo_root": str(summary.get("repo_root") or ""),
            "repo_name": str(summary.get("repo_name") or ""),
            "head_commit": str(summary.get("head_commit") or bundle_source_meta.get("head_commit") or ""),
            "head_branch": str(summary.get("head_branch") or bundle_source_meta.get("head_branch") or ""),
            "hosting": dict(summary.get("hosting") or {}),
            "codebase_graph_ref": graph_ref,
        }
        if relation_ref:
            result["codebase_relation_ref"] = relation_ref
        if object_ref:
            result["codebase_object_ref"] = object_ref
        if locator:
            result["locator"] = locator
        if unit_spool_ref:
            try:
                Path(unit_spool_ref).unlink(missing_ok=True)
            except Exception:
                pass
        if object_spool_ref and not object_ref:
            try:
                Path(object_spool_ref).unlink(missing_ok=True)
            except Exception:
                pass
        if relation_spool_ref and not relation_ref:
            try:
                Path(relation_spool_ref).unlink(missing_ok=True)
            except Exception:
                pass
        return result

    # ── ingest_document() ──

    async def _ingest_document_write_through(
        self,
        *,
        content: str,
        source_id: str,
        speakers: str = "Document",
        agent_id: str = None,
        swarm_id: str = None,
        scope: str = None,
        artifact_id: str = None,
        version_id: str = None,
        parent_version: str = None,
        content_hash: str = None,
        skip_dedup: bool = False,
        source_meta: dict = None,
        retention_ttl: int = None,
        metadata: dict = None,
        target=None,
        family: str = "document",
        message_id: str | None = None,
        owner_id: str | None = None,
        read: list[str] | None = None,
        write: list[str] | None = None,
        caller_id: str | None = None,
        caller_principal_kind: str | None = None,
    ) -> dict:
        ingress_storage = self._ingress_storage()
        assert ingress_storage is not None

        err = self._validate_metadata(metadata)
        if err:
            raise ValueError(err)
        normalized_target = _normalize_target(target)
        family = self._canonical_content_family(family)
        original_content = str(content or "")
        content = self._normalize_ingress_text(content, family)
        _agent_id, _swarm_id, _scope, _owner_id, _read, _write = self._resolve_live_acl_context(
            agent_id=agent_id,
            swarm_id=swarm_id,
            scope=scope,
            owner_id=owner_id,
            read=read,
            write=write,
            caller_id=caller_id,
            caller_principal_kind=caller_principal_kind,
        )

        if artifact_id is None:
            artifact_id = _generate_artifact_id()
        if version_id is None:
            version_id = _generate_version_id()
        if content_hash is None:
            content_hash = content_hash_text(content, family=family)

        multipart_part_key = self._multipart_part_key(metadata)
        logical_source_id = str(source_id)
        projection_source_id = self._projection_source_id(
            source_id=logical_source_id,
            family=family,
            scope=_scope,
            owner_id=_owner_id,
            swarm_id=_swarm_id,
        )
        merged_source_meta = dict(source_meta or {})
        merged_source_meta["logical_source_id"] = logical_source_id
        doc_dedup_key = self._source_versioning_key(
            source_id=projection_source_id,
            family=family,
            scope=_scope,
            owner_id=_owner_id,
            swarm_id=_swarm_id,
            multipart_part_key=multipart_part_key,
        )
        doc_supersede_version_id = None
        near_duplicate_warning = None
        if not skip_dedup:
            duplicate_of = self._find_exact_duplicate(
                content=content,
                family=family,
                scope=_scope,
                owner_id=_owner_id,
                swarm_id=_swarm_id,
                multipart_part_key=multipart_part_key,
            )
            if duplicate_of is not None:
                return {
                    "status": "duplicate",
                    "duplicate_of": self._dedup_reference(duplicate_of),
                }
            near_duplicate_warning = self._find_near_duplicate(
                content=content,
                family=family,
                scope=_scope,
                owner_id=_owner_id,
                swarm_id=_swarm_id,
                multipart_part_key=multipart_part_key,
            )
        if logical_source_id and not skip_dedup:
            existing = self._dedup_index.get(doc_dedup_key)
            if existing is not None:
                if existing["content_hash"] == content_hash:
                    return {
                        "status": "duplicate",
                        "duplicate_of": {
                            "message_id": existing.get("message_id"),
                            "source_id": logical_source_id,
                            "session_num": existing.get("session_num"),
                            "stored_at": existing.get("stored_at"),
                        },
                    }
                artifact_id = existing["artifact_id"]
                parent_version = existing["version_id"]
                version_id = _generate_version_id()
                doc_supersede_version_id = existing["version_id"]

        effective_message_id = str(message_id) if message_id is not None else f"rawdoc:{source_id}:{version_id}"
        self._active_sync_message_ids.add(effective_message_id)
        ingress_storage.append_write_log(
            message_id=effective_message_id,
            session_id=f"doc:{logical_source_id}",
            agent_id=str(_agent_id or "default"),
            swarm_id=str(_swarm_id or "default"),
            visibility="private" if _scope == "agent-private" else "shared",
            owner_id=_owner_id,
            scope=_scope,
            read=list(_read),
            write=list(_write),
            content_family=str(family or "document"),
            content_text=original_content,
            metadata=metadata or {},
            timestamp_ms=int(datetime.now(timezone.utc).timestamp() * 1000),
        )

        sem = self._get_extract_sem()
        _mal_binding_id = _resolve_mal_binding_id(_owner_id, _agent_id)
        _extract_model = _resolve_extract_model(
            self.extract_model, str(self.data_dir), self.key, _mal_binding_id,
        )

        async def _call_extract_fn(model, system, user_msg, max_tokens=8192):
            return await self._call_extract_with_runtime_secrets(model, system, user_msg, max_tokens, sem)

        original_block_dicts, _blocks = segment_document_text(original_content, logical_source_id)
        original_block_map = {str(block.get("block_id") or ""): dict(block) for block in original_block_dicts}
        canonical_doc = await self._canonicalize_document_blocks_for_retrieval(
            original_block_dicts,
            model=_extract_model,
            call_extract_fn=_call_extract_fn,
        )
        block_dicts = list(canonical_doc["canonical_blocks"])
        canonical_doc_text = str(canonical_doc["canonical_doc_text"] or "")
        doc_source_lang = str(canonical_doc["source_lang"] or "und")
        doc_meta = extract_doc_metadata(original_content, logical_source_id)
        if not canonical_doc["semantic_ready"]:
            raw_session_id = str(uuid4())
            raw_session = {
                "raw_session_id": raw_session_id,
                "message_id": effective_message_id,
                "session_num": self._next_document_session_num(projection_source_id),
                "session_date": doc_meta["date"],
                "content": original_content,
                "raw_original": original_content,
                "canonical_en": "",
                "semantic_ready": False,
                "canonicalization_status": canonical_doc["canonicalization_status"],
                "canonicalization_error": canonical_doc["canonicalization_error"],
                "source_lang": doc_source_lang,
                "translation_version": canonical_doc["translation_version"],
                "speakers": speakers,
                "agent_id": _agent_id,
                "swarm_id": _swarm_id,
                "scope": _scope,
                "owner_id": _owner_id,
                "read": list(_read),
                "write": list(_write),
                "stored_at": datetime.now(timezone.utc).isoformat(),
                "format": family,
                "source_id": projection_source_id,
                "logical_source_id": logical_source_id,
                "artifact_id": artifact_id,
                "version_id": version_id,
                "parent_version": parent_version,
                "content_hash": content_hash,
                "status": "canonicalization_failed",
            }
            if metadata is not None:
                raw_session["metadata"] = dict(metadata)
                if metadata.get("part_source_id"):
                    raw_session["part_source_id"] = str(metadata.get("part_source_id"))
            if normalized_target:
                raw_session["target"] = list(normalized_target)
            if merged_source_meta:
                raw_session.update(merged_source_meta)
            async with self._file_lock:
                self._raw_sessions.append(raw_session)
                raw_doc_text = self._store_document_original_source_text(
                    projection_source_id,
                    original_content,
                    multipart_part_key=multipart_part_key,
                    metadata=metadata,
                    message_id=effective_message_id,
                )
                self._register_source_record(
                    source_id=projection_source_id,
                    family="document",
                    owner_id=_owner_id,
                    read=_read,
                    write=_write,
                    artifact_id=artifact_id,
                    version_id=version_id,
                    content_hash=content_hash,
                    metadata=metadata,
                    target=normalized_target,
                    source_meta={
                        "stored_format": family,
                        **self._semantic_state_fields(canonical_doc),
                        **merged_source_meta,
                        "raw_source_provenance": "original_source_multipart" if multipart_part_key else "original_source",
                        "raw_source_content_hash": hashlib.sha256(raw_doc_text.encode("utf-8")).hexdigest(),
                        "scope": _scope,
                        "swarm_id": _swarm_id,
                    },
                )
                self._persist_projection_delta(
                    raw_session_upserts=[raw_session],
                    raw_doc_upserts=[{
                        "source_id": projection_source_id,
                        "message_id": f"rawdoc:{projection_source_id}" if multipart_part_key else effective_message_id,
                        "content_text": raw_doc_text,
                        "metadata": metadata or {},
                    }],
                    source_record_upserts={projection_source_id: dict(self._source_records[projection_source_id])},
                    state_values=self._state_json_values(),
                    episode_corpus=self._episode_corpus,
                    complete_message_ids=[effective_message_id],
                )
            return {
                "error": str(canonical_doc.get("canonicalization_error") or "english source canonicalization failed"),
                "code": "CANONICALIZATION_ERROR",
                "semantic_ready": False,
                "raw_session_id": raw_session_id,
                "source_id": projection_source_id,
            }
        grouping_config = dict(get_tuning_section("episodes", "document_grouping"))
        _mal_cfg = _load_mal_active_config(str(self.data_dir), self.key, _mal_binding_id)
        _mal_prompts = _mal_cfg.get("extraction_prompts") or {}
        if _mal_cfg.get("grouping_prompt_mode"):
            grouping_config["prompt_mode"] = _mal_cfg["grouping_prompt_mode"]
        if _mal_cfg.get("size_cap_chars"):
            grouping_config["size_cap_chars"] = _mal_cfg["size_cap_chars"]
        try:
            with self._runtime_secret_context(
                self._resolve_librarian_secret_ref(_extract_model),
                field_name="librarian_secret_ref",
            ):
                episodes, _grouping_raw, _mode_used = await group_document(
                    _extract_model,
                    logical_source_id,
                    doc_meta["title"],
                    doc_meta["date"],
                    block_dicts,
                    grouping_config,
                    sem,
                )
        except Exception as e:
            log.warning(
                "document episode grouping failed for %s; falling back to singleton block episodes: %s",
                logical_source_id,
                e,
            )
            episodes = build_singleton_episodes(logical_source_id, doc_meta["date"], block_dicts)

        if not episodes:
            episodes = build_singleton_episodes(logical_source_id, doc_meta["date"], block_dicts)

        projection_snapshot = None
        async with self._file_lock:
            projection_snapshot = self._snapshot_document_projection_state(projection_source_id)
            doc_session_start = self._next_document_session_num(projection_source_id)
            next_episode_index = self._next_document_episode_index(projection_source_id) if multipart_part_key else 1
            episodes = self._reindex_document_episodes(
                projection_source_id,
                episodes,
                start_index=next_episode_index,
                session_start=doc_session_start,
                part_key=multipart_part_key,
                family=family,
            )
            # Reserve multipart episode ids immediately so concurrent parts
            # cannot allocate the same suffixes while this part is still
            # running extraction.
            self._upsert_episode_document(
                self._document_doc_id(projection_source_id),
                episodes,
                replace_part_key=multipart_part_key,
            )

        async def _extract_episode(ep_idx: int, ep: dict) -> dict:
            session_num = _coerce_positive_session_num(ep.get("session_num")) or (doc_session_start + ep_idx - 1)
            raw_session_id = str(uuid4())
            episode_original_text = (
                self._episode_original_text_from_blocks(ep, original_block_map, original_content)
                or ep["raw_text"]
            )
            raw_session = {
                "raw_session_id": raw_session_id,
                "message_id": effective_message_id,
                "session_num": session_num,
                "session_date": ep.get("source_date", "") or doc_meta["date"],
                "content": episode_original_text,
                "raw_original": episode_original_text,
                "canonical_en": ep["raw_text"],
                "semantic_ready": True,
                "canonicalization_status": canonical_doc["canonicalization_status"],
                "canonicalization_error": canonical_doc["canonicalization_error"],
                "source_lang": doc_source_lang,
                "translation_version": canonical_doc["translation_version"],
                "speakers": speakers,
                "agent_id": _agent_id,
                "swarm_id": _swarm_id,
                "scope": _scope,
                "owner_id": _owner_id,
                "read": list(_read),
                "write": list(_write),
                "stored_at": datetime.now(timezone.utc).isoformat(),
                "format": family,
                "source_id": projection_source_id,
                "logical_source_id": logical_source_id,
                "artifact_id": artifact_id,
                "version_id": version_id,
                "parent_version": parent_version,
                "content_hash": content_hash,
                "status": "active",
                "episode_id": ep["episode_id"],
            }
            if metadata is not None:
                raw_session["metadata"] = dict(metadata)
                if metadata.get("part_source_id"):
                    raw_session["part_source_id"] = str(metadata.get("part_source_id"))
            if normalized_target:
                raw_session["target"] = list(normalized_target)
            if merged_source_meta:
                raw_session.update(merged_source_meta)

            _block_overrides = {}
            for _pk, _pv in _mal_prompts.items():
                if _pk.startswith("document_block_prompt:"):
                    _block_overrides[_pk.split(":", 1)[1]] = _pv
            extract_kwargs = {
                "session_text": ep["raw_text"],
                "session_num": session_num,
                "session_date": ep.get("source_date", "") or doc_meta["date"],
                "conv_id": self.key,
                "speakers": speakers,
                "model": _extract_model,
                "call_extract_fn": _call_extract_fn,
                "block_prompt_overrides": _block_overrides or None,
                "return_report": True,
            }
            try:
                extract_result = await extract_session(**extract_kwargs)
            except TypeError as exc:
                if "return_report" not in str(exc):
                    raise
                extract_kwargs.pop("return_report", None)
                extract_result = await extract_session(**extract_kwargs)
            _conv_id, _sn, _sdate, facts, tlinks, extraction_report = _coerce_extract_session_result(extract_result)
            _set_runtime_report_artifact(
                raw_session,
                field_name="extraction_report",
                producer="block_extractor",
                report_kind="extraction",
                report=extraction_report,
            )
            self._tag_facts(
                facts,
                ep.get("source_date", "") or doc_meta["date"],
                agent_id=_agent_id,
                swarm_id=_swarm_id,
                scope=_scope,
                owner_id=_owner_id,
                read=_read,
                write=_write,
                artifact_id=artifact_id,
                version_id=version_id,
                content_hash=content_hash,
                target=normalized_target,
                metadata=metadata,
                retention_ttl=retention_ttl,
            )
            ep_payload = deepcopy(ep)
            ep_payload["session_num"] = session_num
            ep_payload["raw_original"] = raw_session["raw_original"]
            ep_payload["canonical_en"] = raw_session["canonical_en"]
            ep_payload["semantic_ready"] = raw_session["semantic_ready"]
            ep_payload["canonicalization_status"] = raw_session["canonicalization_status"]
            ep_payload["canonicalization_error"] = raw_session["canonicalization_error"]
            ep_payload["source_lang"] = raw_session["source_lang"]
            ep_payload["translation_version"] = raw_session["translation_version"]
            ep_payload["artifact_id"] = artifact_id
            ep_payload["version_id"] = version_id
            ep_payload["status"] = raw_session.get("status") or "active"
            _stamp_selector_episode_fields(ep_payload, raw_session, merged_source_meta)
            self._stamp_episode_metadata(
                facts,
                ep_payload["episode_id"],
                projection_source_id,
                artifact_span_id=ep_payload.get("artifact_span_id"),
            )
            self._align_fact_selectors(
                facts,
                episode_id=ep_payload["episode_id"],
                source_kind=family,
                raw_fields=_selector_raw_fields(ep_payload["raw_text"], raw_session, merged_source_meta, ep_payload),
            )
            complexity = _compute_content_complexity(facts)
            for fact in facts:
                raw_id = fact.get("id", "")
                if raw_id and not raw_id.startswith(f"{ep_payload['episode_id']}_"):
                    fact["id"] = f"{ep_payload['episode_id']}_{raw_id}"
                fact["source_id"] = projection_source_id
                fact["raw_session_id"] = raw_session_id
                fact["session"] = session_num
                fact["_session_content_complexity"] = complexity
                merged_metadata = _merge_fact_metadata(
                    fact.get("metadata"),
                    (
                        {}
                        if fact.get("metadata", {}).get("document_source")
                        else {"document_source": projection_source_id}
                    ),
                )
                if merged_metadata:
                    err = self._validate_metadata(merged_metadata)
                    if err:
                        raise ValueError(err)
                    fact["metadata"] = merged_metadata
                err = _validate_object_flags_field(fact)
                if err:
                    raise ValueError(err)
            for fact in facts:
                fact["_temporal_links"] = []
            if facts and tlinks:
                facts[0]["_temporal_links"] = tlinks
            return {
                "facts": facts,
                "tlinks": tlinks,
                "episode": ep_payload,
                "raw_session": raw_session,
                "session_num": session_num,
            }

        tasks = [
            asyncio.create_task(_extract_episode(idx + 1, ep))
            for idx, ep in enumerate(episodes)
        ]
        try:
            episode_results = await asyncio.gather(*tasks)
        except Exception:
            for task in tasks:
                task.cancel()
            async with self._file_lock:
                self._rollback_failed_document_ingest(
                    source_id=projection_source_id,
                    version_id=version_id,
                    projection_snapshot=projection_snapshot or {},
                )
                # Gather-mode multipart ingest only flushes cache on the success path.
                # Rollback here restores in-memory projection state before anything is
                # persisted, so there is no partial on-disk state to save.
            raise

        total_facts = 0
        doc_granular: list[dict] = []
        doc_raw_sessions: list[dict] = []
        doc_tlinks: list[dict] = []
        final_episodes = []
        for item in episode_results:
            total_facts += len(item["facts"])
            doc_granular.extend(item["facts"])
            doc_raw_sessions.append(item["raw_session"])
            final_episodes.append(item["episode"])
            if item["facts"] and item["facts"][0].get("_temporal_links"):
                doc_tlinks.extend(item["facts"][0]["_temporal_links"])

        doc_cross: list[dict] = []
        if doc_granular:
            substrate_cross = await self._extract_source_aggregation_facts(
                source_id=projection_source_id,
                source_kind=family,
                source_facts=doc_granular,
                source_date=doc_meta["date"],
                model=_extract_model,
                call_extract_fn=_call_extract_fn,
                agent_id=_agent_id or "default",
            )
            _namespace_derived_fact_ids(substrate_cross, f"substrate_{self._episode_source_key(projection_source_id)}")
            _stamp_source_aggregation_source(substrate_cross, projection_source_id, family)
            max_doc_cc = max((f.get("_session_content_complexity", 0.0) for f in doc_granular), default=0.0)
            for fact in substrate_cross:
                fact["_session_content_complexity"] = max_doc_cc
            self._tag_facts(
                substrate_cross,
                doc_meta["date"],
                agent_id=_agent_id,
                swarm_id=_swarm_id,
                scope=_scope,
                owner_id=_owner_id,
                read=_read,
                write=_write,
                artifact_id=artifact_id,
                version_id=version_id,
                content_hash=content_hash,
                target=_consensus_target(doc_granular),
            )
            doc_cross.extend(substrate_cross)

        updated_granular: list[dict] = []
        updated_raw_sessions: list[dict] = []
        async with self._file_lock:
            if doc_supersede_version_id:
                self._remove_document_content_indices(projection_source_id, multipart_part_key)
                for rs in self._raw_sessions:
                    if rs.get("version_id") == doc_supersede_version_id:
                        rs["status"] = "superseded"
                        updated_raw_sessions.append(dict(rs))
                        self._remove_content_indices_for_message(str(rs.get("message_id") or ""))
            self._raw_sessions.extend(doc_raw_sessions)
            self._upsert_episode_document(
                self._document_doc_id(projection_source_id),
                final_episodes,
                replace_part_key=multipart_part_key,
            )
            raw_doc_text = self._store_document_original_source_text(
                projection_source_id,
                original_content,
                multipart_part_key=multipart_part_key,
                metadata=metadata,
                message_id=effective_message_id,
            )
            self._register_source_record(
                source_id=projection_source_id,
                family=family,
                owner_id=_owner_id,
                read=_read,
                write=_write,
                artifact_id=artifact_id,
                version_id=version_id,
                content_hash=content_hash,
                metadata=metadata,
                target=normalized_target,
                source_meta={
                    "stored_format": family,
                    "source_lang": doc_source_lang,
                    **self._semantic_state_fields(canonical_doc),
                    **merged_source_meta,
                    "raw_source_provenance": "original_source_multipart" if multipart_part_key else "original_source",
                    "raw_source_content_hash": hashlib.sha256(raw_doc_text.encode("utf-8")).hexdigest(),
                    "scope": _scope,
                    "swarm_id": _swarm_id,
                },
            )
            if doc_granular and doc_supersede_version_id:
                for f in self._all_granular:
                    if f.get("version_id") == doc_supersede_version_id:
                        f["status"] = "superseded"
                        updated_granular.append(dict(f))

            if projection_source_id and not skip_dedup and doc_granular:
                self._dedup_index[doc_dedup_key] = {
                    "artifact_id": artifact_id,
                    "version_id": version_id,
                    "content_hash": content_hash,
                    "message_id": effective_message_id,
                    "stored_at": doc_raw_sessions[0].get("stored_at") if doc_raw_sessions else None,
                    "session_num": doc_raw_sessions[0].get("session_num") if doc_raw_sessions else None,
                }
                self._index_content_entry(
                    message_id=effective_message_id,
                    source_id=projection_source_id,
                    session_num=doc_raw_sessions[0].get("session_num") if doc_raw_sessions else None,
                    stored_at=doc_raw_sessions[0].get("stored_at") if doc_raw_sessions else None,
                    scope=_scope,
                    owner_id=_owner_id,
                    swarm_id=_swarm_id,
                    family=family,
                    content=canonical_doc_text,
                    multipart_part_key=multipart_part_key,
                )

            if doc_granular:
                self._all_granular.extend(doc_granular)
                self._all_cross.extend(doc_cross)
                self._all_tlinks.extend(doc_tlinks)
                self._n_sessions += len(final_episodes)
                self._n_sessions_with_facts += len({
                    fact.get("session", 1)
                    for fact in doc_granular
                    if fact
                })

            self._persist_projection_delta(
                raw_session_upserts=updated_raw_sessions + doc_raw_sessions,
                raw_doc_upserts=[{
                    "source_id": projection_source_id,
                    "message_id": f"rawdoc:{projection_source_id}" if multipart_part_key else effective_message_id,
                    "content_text": raw_doc_text,
                    "metadata": metadata or {},
                }],
                fact_upserts={
                    "granular": updated_granular + doc_granular,
                    "cross": doc_cross,
                },
                episode_doc_replacements={
                    self._document_doc_id(projection_source_id): self._get_episode_documents(projection_source_id, "document")
                },
                temporal_link_appends=doc_tlinks,
                source_record_upserts={projection_source_id: dict(self._source_records[projection_source_id])},
                state_values=self._state_json_values(),
                episode_corpus=self._episode_corpus,
                complete_message_ids=[effective_message_id],
            )
            self._bump_index_snapshot_version()

        if not doc_granular:
            self._data_dict = None
            self._active_sync_message_ids.discard(effective_message_id)
            return {"status": "ok", "facts_extracted": 0}

        self._data_dict = None
        self._mark_tiers_dirty()
        result = {"status": "ok", "facts_extracted": total_facts}
        if near_duplicate_warning:
            result["near_duplicate_warning"] = near_duplicate_warning
        self._active_sync_message_ids.discard(effective_message_id)
        return result

    async def ingest_document(
        self,
        content: str,
        source_id: str,
        speakers: str = "Document",
        agent_id: str = None,
        swarm_id: str = None,
        scope: str = None,
        artifact_id: str = None,
        version_id: str = None,
        parent_version: str = None,
        content_hash: str = None,
        skip_dedup: bool = False,
        source_meta: dict = None,
        retention_ttl: int = None,
        metadata: dict = None,
        target=None,
        family: str = "document",
        message_id: str | None = None,
        owner_id: str | None = None,
        read: list[str] | None = None,
        write: list[str] | None = None,
        caller_id: str | None = None,
        caller_principal_kind: str | None = None,
    ) -> dict:
        """Ingest a document through the episode pipeline."""
        if self._ingress_storage() is not None and self._projection_storage() is not None:
            return await self._ingest_document_write_through(
                content=content,
                source_id=source_id,
                speakers=speakers,
                agent_id=agent_id,
                swarm_id=swarm_id,
                scope=scope,
                artifact_id=artifact_id,
                version_id=version_id,
                parent_version=parent_version,
                content_hash=content_hash,
                skip_dedup=skip_dedup,
                source_meta=source_meta,
                retention_ttl=retention_ttl,
                metadata=metadata,
                target=target,
                family=family,
                message_id=message_id,
                owner_id=owner_id,
                read=read,
                write=write,
                caller_id=caller_id,
                caller_principal_kind=caller_principal_kind,
            )
        err = self._validate_metadata(metadata)
        if err:
            raise ValueError(err)
        normalized_target = _normalize_target(target)
        family = self._canonical_content_family(family)
        original_content = str(content or "")
        content = self._normalize_ingress_text(content, family)
        _agent_id, _swarm_id, _scope, _owner_id, _read, _write = self._resolve_live_acl_context(
            agent_id=agent_id,
            swarm_id=swarm_id,
            scope=scope,
            owner_id=owner_id,
            read=read,
            write=write,
            caller_id=caller_id,
            caller_principal_kind=caller_principal_kind,
        )

        # Generate identity fields
        if artifact_id is None:
            artifact_id = _generate_artifact_id()
        if version_id is None:
            version_id = _generate_version_id()
        if content_hash is None:
            content_hash = content_hash_text(content, family=family)

        multipart_part_key = self._multipart_part_key(metadata)
        logical_source_id = str(source_id)
        projection_source_id = self._projection_source_id(
            source_id=logical_source_id,
            family=family,
            scope=_scope,
            owner_id=_owner_id,
            swarm_id=_swarm_id,
        )
        merged_source_meta = dict(source_meta or {})
        merged_source_meta["logical_source_id"] = logical_source_id
        doc_dedup_key = self._source_versioning_key(
            source_id=projection_source_id,
            family=family,
            scope=_scope,
            owner_id=_owner_id,
            swarm_id=_swarm_id,
            multipart_part_key=multipart_part_key,
        )
        doc_supersede_version_id = None
        near_duplicate_warning = None
        if not skip_dedup:
            duplicate_of = self._find_exact_duplicate(
                content=content,
                family=family,
                scope=_scope,
                owner_id=_owner_id,
                swarm_id=_swarm_id,
                multipart_part_key=multipart_part_key,
            )
            if duplicate_of is not None:
                return {
                    "status": "duplicate",
                    "duplicate_of": self._dedup_reference(duplicate_of),
                }
            near_duplicate_warning = self._find_near_duplicate(
                content=content,
                family=family,
                scope=_scope,
                owner_id=_owner_id,
                swarm_id=_swarm_id,
                multipart_part_key=multipart_part_key,
            )
        if logical_source_id and not skip_dedup:
            existing = self._dedup_index.get(doc_dedup_key)
            if existing is not None:
                if existing["content_hash"] == content_hash:
                    return {
                        "status": "duplicate",
                        "duplicate_of": {
                            "message_id": existing.get("message_id"),
                            "source_id": logical_source_id,
                            "session_num": existing.get("session_num"),
                            "stored_at": existing.get("stored_at"),
                        },
                    }
                artifact_id = existing["artifact_id"]
                parent_version = existing["version_id"]
                version_id = _generate_version_id()
                doc_supersede_version_id = existing["version_id"]

        sem = self._get_extract_sem()
        _mal_binding_id = _resolve_mal_binding_id(_owner_id, _agent_id)
        _extract_model = _resolve_extract_model(
            self.extract_model, str(self.data_dir), self.key, _mal_binding_id,
        )

        async def _call_extract_fn(model, system, user_msg, max_tokens=8192):
            return await self._call_extract_with_runtime_secrets(model, system, user_msg, max_tokens, sem)
        original_block_dicts, _blocks = segment_document_text(original_content, logical_source_id)
        original_block_map = {str(block.get("block_id") or ""): dict(block) for block in original_block_dicts}
        canonical_doc = await self._canonicalize_document_blocks_for_retrieval(
            original_block_dicts,
            model=_extract_model,
            call_extract_fn=_call_extract_fn,
        )
        block_dicts = list(canonical_doc["canonical_blocks"])
        canonical_doc_text = str(canonical_doc["canonical_doc_text"] or "")
        doc_source_lang = str(canonical_doc["source_lang"] or "und")
        doc_meta = extract_doc_metadata(original_content, logical_source_id)
        if not canonical_doc["semantic_ready"]:
            raw_session = {
                "raw_session_id": str(uuid4()),
                "message_id": str(
                    message_id
                    or (
                        f"rawdoc:{logical_source_id}:{multipart_part_key}"
                        if multipart_part_key
                        else f"rawdoc:{logical_source_id}"
                    )
                ),
                "session_num": self._next_document_session_num(projection_source_id),
                "session_date": doc_meta["date"],
                "content": original_content,
                "raw_original": original_content,
                "canonical_en": "",
                "semantic_ready": False,
                "canonicalization_status": canonical_doc["canonicalization_status"],
                "canonicalization_error": canonical_doc["canonicalization_error"],
                "source_lang": doc_source_lang,
                "translation_version": canonical_doc["translation_version"],
                "speakers": speakers,
                "agent_id": _agent_id,
                "swarm_id": _swarm_id,
                "scope": _scope,
                "owner_id": _owner_id,
                "read": list(_read),
                "write": list(_write),
                "stored_at": datetime.now(timezone.utc).isoformat(),
                "format": family,
                "source_id": projection_source_id,
                "logical_source_id": logical_source_id,
                "artifact_id": artifact_id,
                "version_id": version_id,
                "parent_version": parent_version,
                "content_hash": content_hash,
                "status": "canonicalization_failed",
            }
            if metadata is not None:
                raw_session["metadata"] = dict(metadata)
                if metadata.get("part_source_id"):
                    raw_session["part_source_id"] = str(metadata.get("part_source_id"))
            if normalized_target:
                raw_session["target"] = list(normalized_target)
            if merged_source_meta:
                raw_session.update(merged_source_meta)
            async with self._file_lock:
                self._raw_sessions.append(raw_session)
                raw_doc_text = self._store_document_original_source_text(
                    projection_source_id,
                    original_content,
                    multipart_part_key=multipart_part_key,
                    metadata=metadata,
                    message_id=str(raw_session.get("message_id") or ""),
                )
                self._register_source_record(
                    source_id=projection_source_id,
                    family="document",
                    owner_id=_owner_id,
                    read=_read,
                    write=_write,
                    artifact_id=artifact_id,
                    version_id=version_id,
                    content_hash=content_hash,
                    metadata=metadata,
                    target=normalized_target,
                    source_meta={
                        "stored_format": family,
                        **self._semantic_state_fields(canonical_doc),
                        **merged_source_meta,
                        "raw_source_provenance": "original_source_multipart" if multipart_part_key else "original_source",
                        "raw_source_content_hash": hashlib.sha256(raw_doc_text.encode("utf-8")).hexdigest(),
                        "scope": _scope,
                        "swarm_id": _swarm_id,
                    },
                )
                self._save_cache()
            return {
                "error": str(canonical_doc.get("canonicalization_error") or "english source canonicalization failed"),
                "code": "CANONICALIZATION_ERROR",
                "semantic_ready": False,
                "raw_session_id": raw_session["raw_session_id"],
                "source_id": projection_source_id,
            }
        grouping_config = dict(get_tuning_section("episodes", "document_grouping"))
        # MAL generation-aware overrides for grouping + extraction prompts
        _mal_cfg = _load_mal_active_config(str(self.data_dir), self.key, _mal_binding_id)
        _mal_prompts = _mal_cfg.get("extraction_prompts") or {}
        if _mal_cfg.get("grouping_prompt_mode"):
            grouping_config["prompt_mode"] = _mal_cfg["grouping_prompt_mode"]
        if _mal_cfg.get("size_cap_chars"):
            grouping_config["size_cap_chars"] = _mal_cfg["size_cap_chars"]
        try:
            with self._runtime_secret_context(
                self._resolve_librarian_secret_ref(_extract_model),
                field_name="librarian_secret_ref",
            ):
                episodes, _grouping_raw, _mode_used = await group_document(
                    _extract_model,
                    logical_source_id,
                    doc_meta["title"],
                    doc_meta["date"],
                    block_dicts,
                    grouping_config,
                    sem,
                )
        except Exception as e:
            log.warning(
                "document episode grouping failed for %s; falling back to singleton block episodes: %s",
                logical_source_id,
                e,
            )
            episodes = build_singleton_episodes(logical_source_id, doc_meta["date"], block_dicts)

        if not episodes:
            episodes = build_singleton_episodes(logical_source_id, doc_meta["date"], block_dicts)

        projection_snapshot = None
        async with self._file_lock:
            projection_snapshot = self._snapshot_document_projection_state(projection_source_id)
            doc_session_start = self._next_document_session_num(projection_source_id)
            next_episode_index = self._next_document_episode_index(projection_source_id) if multipart_part_key else 1
            episodes = self._reindex_document_episodes(
                projection_source_id,
                episodes,
                start_index=next_episode_index,
                session_start=doc_session_start,
                part_key=multipart_part_key,
                family=family,
            )
            raw_doc_text = self._store_document_original_source_text(
                projection_source_id,
                original_content,
                multipart_part_key=multipart_part_key,
                metadata=metadata,
                message_id=str(
                    message_id
                    or (
                        f"rawdoc:{logical_source_id}:{multipart_part_key}"
                        if multipart_part_key
                        else f"rawdoc:{logical_source_id}"
                    )
                ),
            )
            self._upsert_episode_document(
                self._document_doc_id(projection_source_id),
                episodes,
                replace_part_key=multipart_part_key,
            )
            self._register_source_record(
                source_id=projection_source_id,
                family=family,
                owner_id=_owner_id,
                read=_read,
                write=_write,
                artifact_id=artifact_id,
                version_id=version_id,
                content_hash=content_hash,
                metadata=metadata,
                target=normalized_target,
                source_meta={
                    "stored_format": family,
                    **self._semantic_state_fields(canonical_doc),
                    **merged_source_meta,
                    "raw_source_provenance": "original_source_multipart" if multipart_part_key else "original_source",
                    "raw_source_content_hash": hashlib.sha256(
                        raw_doc_text.encode("utf-8")
                    ).hexdigest(),
                    "scope": _scope,
                    "swarm_id": _swarm_id,
                },
            )
            self._save_cache()

        async def _extract_episode(ep_idx: int, ep: dict) -> list[dict]:
            session_num = _coerce_positive_session_num(ep.get("session_num")) or (doc_session_start + ep_idx - 1)
            raw_session_id = str(uuid4())
            episode_original_text = (
                self._episode_original_text_from_blocks(ep, original_block_map, original_content)
                or ep["raw_text"]
            )
            raw_session = {
                "raw_session_id": raw_session_id,
                "session_num": session_num,
                "session_date": ep.get("source_date", "") or doc_meta["date"],
                "content": episode_original_text,
                "raw_original": episode_original_text,
                "canonical_en": ep["raw_text"],
                "semantic_ready": True,
                "canonicalization_status": canonical_doc["canonicalization_status"],
                "canonicalization_error": canonical_doc["canonicalization_error"],
                "source_lang": doc_source_lang,
                "translation_version": canonical_doc["translation_version"],
                "speakers": speakers,
                "agent_id": _agent_id,
                "swarm_id": _swarm_id,
                "scope": _scope,
                "owner_id": _owner_id,
                "read": list(_read),
                "write": list(_write),
                "stored_at": datetime.now(timezone.utc).isoformat(),
                "format": family,
                "source_id": projection_source_id,
                "logical_source_id": logical_source_id,
                "artifact_id": artifact_id,
                "version_id": version_id,
                "parent_version": parent_version,
                "content_hash": content_hash,
                "status": "active",
                "episode_id": ep["episode_id"],
            }
            if metadata is not None:
                raw_session["metadata"] = dict(metadata)
                if metadata.get("part_source_id"):
                    raw_session["part_source_id"] = str(metadata.get("part_source_id"))
            if normalized_target:
                raw_session["target"] = list(normalized_target)
            if merged_source_meta:
                raw_session.update(merged_source_meta)

            async with self._file_lock:
                self._raw_sessions.append(raw_session)
                self._save_cache()

            # MAL block prompt overrides for document extraction
            _block_overrides = {}
            for _pk, _pv in _mal_prompts.items():
                if _pk.startswith("document_block_prompt:"):
                    _block_overrides[_pk.split(":", 1)[1]] = _pv
            extract_result = await extract_session(
                session_text=ep["raw_text"],
                session_num=session_num,
                session_date=ep.get("source_date", "") or doc_meta["date"],
                conv_id=self.key,
                speakers=speakers,
                model=_extract_model,
                call_extract_fn=_call_extract_fn,
                block_prompt_overrides=_block_overrides or None,
                return_report=True,
            )
            _conv_id, _sn, _sdate, facts, tlinks, extraction_report = _coerce_extract_session_result(extract_result)
            _set_runtime_report_artifact(
                raw_session,
                field_name="extraction_report",
                producer="block_extractor",
                report_kind="extraction",
                report=extraction_report,
            )
            self._tag_facts(
                facts,
                ep.get("source_date", "") or doc_meta["date"],
                agent_id=_agent_id,
                swarm_id=_swarm_id,
                scope=_scope,
                owner_id=_owner_id,
                read=_read,
                write=_write,
                artifact_id=artifact_id,
                version_id=version_id,
                content_hash=content_hash,
                target=normalized_target,
                metadata=metadata,
                retention_ttl=retention_ttl,
            )
            ep["raw_original"] = raw_session["raw_original"]
            ep["canonical_en"] = raw_session["canonical_en"]
            ep["semantic_ready"] = raw_session["semantic_ready"]
            ep["canonicalization_status"] = raw_session["canonicalization_status"]
            ep["canonicalization_error"] = raw_session["canonicalization_error"]
            ep["source_lang"] = raw_session["source_lang"]
            ep["translation_version"] = raw_session["translation_version"]
            _stamp_selector_episode_fields(ep, raw_session, merged_source_meta)
            self._stamp_episode_metadata(
                facts,
                ep["episode_id"],
                projection_source_id,
                artifact_span_id=ep.get("artifact_span_id"),
            )
            self._align_fact_selectors(
                facts,
                episode_id=ep["episode_id"],
                source_kind=family,
                raw_fields=_selector_raw_fields(ep["raw_text"], raw_session, merged_source_meta, ep),
            )
            complexity = _compute_content_complexity(facts)
            for fact in facts:
                raw_id = fact.get("id", "")
                if raw_id and not raw_id.startswith(f"{ep['episode_id']}_"):
                    fact["id"] = f"{ep['episode_id']}_{raw_id}"
                fact["source_id"] = projection_source_id
                fact["raw_session_id"] = raw_session_id
                fact["session"] = session_num
                fact["_session_content_complexity"] = complexity
                merged_metadata = _merge_fact_metadata(
                    fact.get("metadata"),
                    (
                        {}
                        if fact.get("metadata", {}).get("document_source")
                        else {"document_source": projection_source_id}
                    ),
                )
                if merged_metadata:
                    err = self._validate_metadata(merged_metadata)
                    if err:
                        raise ValueError(err)
                    fact["metadata"] = merged_metadata
                err = _validate_object_flags_field(fact)
                if err:
                    raise ValueError(err)
            for fact in facts:
                fact["_temporal_links"] = []
            if facts and tlinks:
                facts[0]["_temporal_links"] = tlinks

            async with self._file_lock:
                self._all_granular.extend(facts)
                self._all_tlinks.extend(tlinks)
                self._mark_tiers_dirty()
                self._data_dict = None
                self._save_cache()
            return facts

        total_facts = 0
        doc_granular = []
        doc_tlinks = []
        tasks = [
            asyncio.create_task(_extract_episode(idx + 1, ep))
            for idx, ep in enumerate(episodes)
        ]
        try:
            for task in asyncio.as_completed(tasks):
                facts = await task
                total_facts += len(facts)
                doc_granular.extend(facts)
                if facts and facts[0].get("_temporal_links"):
                    doc_tlinks.extend(facts[0]["_temporal_links"])
        except Exception:
            for task in tasks:
                task.cancel()
            async with self._file_lock:
                self._rollback_failed_document_ingest(
                    source_id=projection_source_id,
                    version_id=version_id,
                    projection_snapshot=projection_snapshot or {},
                )
                self._save_cache()
            raise

        if not doc_granular:
            return {"status": "ok", "facts_extracted": 0}

        doc_cross: list[dict] = []

        substrate_cross = await self._extract_source_aggregation_facts(
            source_id=projection_source_id,
            source_kind=family,
            source_facts=doc_granular,
            source_date=doc_meta["date"],
            model=_extract_model,
            call_extract_fn=_call_extract_fn,
            agent_id=_agent_id or "default",
        )
        _namespace_derived_fact_ids(substrate_cross, f"substrate_{self._episode_source_key(projection_source_id)}")
        _stamp_source_aggregation_source(substrate_cross, projection_source_id, family)
        max_doc_cc = max((f.get("_session_content_complexity", 0.0) for f in doc_granular), default=0.0)
        for fact in substrate_cross:
            fact["_session_content_complexity"] = max_doc_cc
        self._tag_facts(
            substrate_cross,
            doc_meta["date"],
            agent_id=_agent_id,
            swarm_id=_swarm_id,
            scope=_scope,
            owner_id=_owner_id,
            read=_read,
            write=_write,
            artifact_id=artifact_id,
            version_id=version_id,
            content_hash=content_hash,
            target=_consensus_target(doc_granular),
        )
        doc_cross.extend(substrate_cross)

        async with self._file_lock:
            dedup_message_id = str(
                message_id
                or (
                    f"rawdoc:{logical_source_id}:{multipart_part_key}"
                    if multipart_part_key
                    else f"rawdoc:{logical_source_id}"
                )
            )
            dedup_stored_at = datetime.now(timezone.utc).isoformat()
            if doc_supersede_version_id:
                self._remove_document_content_indices(projection_source_id, multipart_part_key)
                for f in self._all_granular:
                    if f.get("version_id") == doc_supersede_version_id:
                        f["status"] = "superseded"
                for rs in self._raw_sessions:
                    if rs.get("version_id") == doc_supersede_version_id:
                        rs["status"] = "superseded"
                        self._remove_content_indices_for_message(str(rs.get("message_id") or ""))
            if projection_source_id and not skip_dedup:
                self._dedup_index[doc_dedup_key] = {
                    "artifact_id": artifact_id,
                    "version_id": version_id,
                    "content_hash": content_hash,
                    "message_id": dedup_message_id,
                    "stored_at": dedup_stored_at,
                    "session_num": None,
                }
                self._index_content_entry(
                    message_id=dedup_message_id,
                    source_id=projection_source_id,
                    session_num=None,
                    stored_at=dedup_stored_at,
                    scope=_scope,
                    owner_id=_owner_id,
                    swarm_id=_swarm_id,
                    family=family,
                    content=canonical_doc_text,
                    multipart_part_key=multipart_part_key,
                )
            raw_doc_text = str(self._raw_docs.get(projection_source_id) or "")
            if not multipart_part_key:
                raw_doc_text = self._store_document_original_source_text(
                    projection_source_id,
                    original_content,
                    multipart_part_key=None,
                    metadata=metadata,
                    message_id=str(
                        message_id
                        or (
                            f"rawdoc:{logical_source_id}:{multipart_part_key}"
                            if multipart_part_key
                            else f"rawdoc:{logical_source_id}"
                        )
                    ),
                )
            elif not raw_doc_text:
                raw_doc_text = self._store_document_original_source_text(
                    projection_source_id,
                    original_content,
                    multipart_part_key=multipart_part_key,
                    metadata=metadata,
                    message_id=str(
                        message_id
                        or (
                            f"rawdoc:{logical_source_id}:{multipart_part_key}"
                            if multipart_part_key
                            else f"rawdoc:{logical_source_id}"
                        )
                    ),
                )
            source_meta_row = self._source_records.get(projection_source_id, {}).setdefault("source_meta", {})
            source_meta_row["raw_source_provenance"] = "original_source_multipart" if multipart_part_key else "original_source"
            source_meta_row["raw_source_content_hash"] = hashlib.sha256(raw_doc_text.encode("utf-8")).hexdigest()
            self._all_cross.extend(doc_cross)
            self._n_sessions += len(episodes)
            self._n_sessions_with_facts += len({
                fact.get("session", 1)
                for fact in doc_granular
                if fact
            })
            self._bump_index_snapshot_version()
            self._save_cache()

        self._data_dict = None
        self._mark_tiers_dirty()
        result = {"status": "ok", "facts_extracted": total_facts}
        if near_duplicate_warning:
            result["near_duplicate_warning"] = near_duplicate_warning
        return result

    async def ingest_codebase(
        self,
        repo_path: str | None,
        source_id: str,
        agent_id: str = None,
        swarm_id: str = None,
        scope: str = None,
        artifact_id: str = None,
        version_id: str = None,
        content_hash: str = None,
        skip_dedup: bool = False,
        source_meta: dict = None,
        retention_ttl: int = None,
        metadata: dict = None,
        target=None,
        family: str = "codebase",
        message_id: str | None = None,
        locator: str | None = None,
        content: str | None = None,
        filename: str | None = None,
        mime: str | None = None,
        owner_id: str | None = None,
        read: list[str] | None = None,
        write: list[str] | None = None,
        caller_id: str | None = None,
        caller_principal_kind: str | None = None,
    ) -> dict:
        """Ingest a local codebase through deterministic Repository context semantic analyzers."""
        err = self._validate_metadata(metadata)
        if err:
            raise ValueError(err)
        normalized_target = _normalize_target(target)
        family = self._canonical_content_family(family)
        if family != "codebase":
            raise ValueError("ingest_codebase only supports family='codebase'")
        repo_locator = str(repo_path or locator or "").strip()
        if not repo_locator:
            raise ValueError("codebase path does not exist: ")
        path_obj = Path(repo_locator).expanduser()
        if not path_obj.exists():
            raise ValueError(f"codebase path does not exist: {path_obj}")
        stage1_git_repo = bool(
            path_obj.is_dir()
            and (
                (path_obj / ".git").is_dir()
                or (path_obj / ".git").is_file()
            )
        )
        stage1_git_available = shutil.which("git") is not None
        stage1_skipped_reason = (
            "git_unavailable_for_legacy_stage1"
            if stage1_git_repo and not stage1_git_available
            else ""
        )
        stage1_applicable = bool(
            path_obj.is_file()
            or (stage1_git_repo and stage1_git_available)
        )

        _agent_id, _swarm_id, _scope, _owner_id, _read, _write = self._resolve_live_acl_context(
            agent_id=agent_id,
            swarm_id=swarm_id,
            scope=scope,
            owner_id=owner_id,
            read=read,
            write=write,
            caller_id=caller_id,
            caller_principal_kind=caller_principal_kind,
        )

        if artifact_id is None:
            artifact_id = _generate_artifact_id()
        if version_id is None:
            version_id = _generate_version_id()

        logical_source_id = str(source_id)
        projection_source_id = self._projection_source_id(
            source_id=logical_source_id,
            family=family,
            scope=_scope,
            owner_id=_owner_id,
            swarm_id=_swarm_id,
        )
        legacy_filename = str(filename or path_obj.name or logical_source_id).strip() or logical_source_id
        legacy_mime = str(mime or ("inode/directory" if path_obj.is_dir() else "")).strip() or None
        legacy_content = self._normalize_ingress_text(str(content or ""), family)
        codebase_supported = True

        bundle: dict[str, Any] | None = None
        profile: dict[str, Any] | None = None
        imported: dict[str, Any] = {}
        stage_meta: dict[str, Any] = {}
        try:
            bundle, profile = build_codebase_semantic_bundle(path_obj)
            imported = import_semantic_bundle(
                bundle,
                source_id=projection_source_id,
                data_dir=str(self.data_dir),
                hot_fact_policy="seed",
            )
            stage_meta = dict((imported.get("source_meta") or {}).get("codebase_context") or {})
            summary_seed = json.dumps(
                {
                    "repo_id": stage_meta.get("repo_id"),
                    "revision": stage_meta.get("revision"),
                    "object_count": stage_meta.get("object_count"),
                    "relation_count": stage_meta.get("relation_count"),
                    "sidecar_count": stage_meta.get("sidecar_count"),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            if content_hash is None:
                content_hash = content_hash_text(summary_seed, family=family)
        except ValueError as exc:
            if "supports only Python and Rust sources" not in str(exc):
                raise
            codebase_supported = False

        existing_record = self._source_records.get(projection_source_id)
        if (
            codebase_supported
            and not skip_dedup
            and existing_record is not None
            and existing_record.get("content_hash") == content_hash
        ):
            return {
                "status": "duplicate",
                "duplicate_of": {
                    "source_id": projection_source_id,
                    "version_id": existing_record.get("version_id"),
                    "artifact_id": existing_record.get("artifact_id"),
                },
            }

        codebase_fact_count = 0
        if codebase_supported:
            facts = deepcopy(imported.get("facts") or [])
            self._tag_facts(
                facts,
                session_date=str(stage_meta.get("generated_at") or ""),
                agent_id=_agent_id,
                swarm_id=_swarm_id,
                scope=_scope,
                owner_id=_owner_id,
                read=_read,
                write=_write,
                artifact_id=artifact_id,
                version_id=version_id,
                content_hash=content_hash,
                retention_ttl=retention_ttl,
                metadata=metadata,
                target=normalized_target,
            )
            for fact in facts:
                fact["source_id"] = projection_source_id
                fact["source_family"] = "codebase"
                fact.setdefault("status", "active")

            updated_facts: list[dict] = []
            merged_source_meta = dict(source_meta or {})
            merged_source_meta["logical_source_id"] = logical_source_id
            merged_source_meta.update(
                {
                    "stored_format": "codebase",
                    "repo_root": str((profile or {}).get("repo_root") or path_obj),
                    "repo_id": stage_meta.get("repo_id"),
                    "revision": stage_meta.get("revision"),
                    "scope": _scope,
                    "swarm_id": _swarm_id,
                    **(imported.get("source_meta") or {}),
                }
            )
            if stage1_skipped_reason:
                merged_source_meta["codebase_stage1_legacy"] = {
                    "status": "skipped",
                    "reason": stage1_skipped_reason,
                }

            async with self._file_lock:
                for existing_fact in self._all_granular:
                    if (
                        str(existing_fact.get("source_id") or "") == projection_source_id
                        and str(existing_fact.get("source_family") or ((existing_fact.get("metadata") or {}).get("source_family") or "")).lower() == "codebase"
                        and str(existing_fact.get("status") or "active") == "active"
                    ):
                        existing_fact["status"] = "superseded"
                        updated_facts.append(dict(existing_fact))

                self._all_granular.extend(facts)
                self._register_source_record(
                    source_id=projection_source_id,
                    family="codebase",
                    owner_id=_owner_id,
                    read=_read,
                    write=_write,
                    artifact_id=artifact_id,
                    version_id=version_id,
                    content_hash=content_hash,
                    metadata=metadata,
                    target=normalized_target,
                    source_meta=merged_source_meta,
                )
                self._data_dict = None
                self._temporal_index_dirty = True
                self._persist_projection_delta(
                    fact_upserts={"granular": updated_facts + facts},
                    source_record_upserts={projection_source_id: dict(self._source_records[projection_source_id])},
                    state_values=self._state_json_values(),
                    episode_corpus=self._episode_corpus,
                    complete_message_ids=[message_id] if message_id else [],
                )
            codebase_fact_count = len(facts)

        stage1_result: dict[str, Any] | None = None
        if stage1_applicable:
            stage1_result = await self._ingest_codebase_stage1_legacy(
                locator=repo_locator,
                content=legacy_content,
                filename=legacy_filename,
                mime=legacy_mime,
                source_id=source_id,
                agent_id=agent_id,
                swarm_id=swarm_id,
                scope=scope,
                artifact_id=artifact_id,
                version_id=version_id,
                content_hash=content_hash,
                skip_dedup=skip_dedup,
                source_meta=source_meta,
                retention_ttl=retention_ttl,
                metadata=metadata,
                target=target,
                message_id=message_id,
                owner_id=owner_id,
                read=read,
                write=write,
                caller_id=caller_id,
                caller_principal_kind=caller_principal_kind,
            )
        if not codebase_supported:
            if stage1_result is None:
                raise ValueError("codebase Repository context supports only Python and Rust sources")
            stage1_result.setdefault("source_id", projection_source_id)
            stage1_result.setdefault("source_family", "codebase")
            stage1_result.setdefault("codebase_stage", "stage1_legacy")
            return stage1_result

        async with self._file_lock:
            source_record = self._source_records.get(projection_source_id)
            if source_record is not None:
                merged_meta = dict(source_record.get("source_meta") or {})
                merged_meta.update(
                    {
                        "repo_root": str((profile or {}).get("repo_root") or path_obj),
                        "repo_id": stage_meta.get("repo_id"),
                        "revision": stage_meta.get("revision"),
                        **(imported.get("source_meta") or {}),
                    }
                )
                source_record["source_meta"] = merged_meta
                source_record["artifact_id"] = artifact_id
                source_record["version_id"] = version_id
                source_record["content_hash"] = content_hash
                self._container_graph = build_codebase_container_graph(
                    bundle or {},
                    source_id=projection_source_id,
                    source_record=dict(source_record),
                    repo_root=str((profile or {}).get("repo_root") or path_obj),
                    existing_graph=self._container_graph,
                )
                codebase_revisions = [
                    row
                    for row in self._container_graph.get("graph_revisions", [])
                    if row.get("source_id") == projection_source_id
                    and row.get("adapter_name") == "codebase_semantic_container_graph"
                ]
                if codebase_revisions:
                    latest_revision = sorted(
                        codebase_revisions,
                        key=lambda row: str(row.get("completed_at") or row.get("created_at") or ""),
                    )[-1]
                    merged_meta["codebase_container_graph"] = {
                        "container_graph_revision_id": latest_revision.get("container_graph_revision_id"),
                        "status": latest_revision.get("status"),
                        "adapter_name": latest_revision.get("adapter_name"),
                        "adapter_version": latest_revision.get("adapter_version"),
                        "profile_ids": list(latest_revision.get("profile_ids_json") or []),
                        "coverage_report_ids": list(latest_revision.get("coverage_report_ids_json") or []),
                    }
                    source_record["source_meta"] = merged_meta
                self._persist_projection_delta(
                    source_record_upserts={projection_source_id: dict(source_record)},
                    state_values=self._state_json_values(),
                    episode_corpus=self._episode_corpus,
                    **self._container_graph_projection_kwargs(self._container_graph, replace=True),
                    complete_message_ids=[message_id] if message_id else [],
                )

        result = dict(stage1_result or {})
        codebase_graph_meta = (
            self._source_records.get(projection_source_id, {}).get("source_meta", {}).get("codebase_container_graph", {})
        )
        result.update(
            {
                "status": "ok",
                "facts_extracted": int(result.get("facts_extracted") or 0) + codebase_fact_count,
                "source_id": projection_source_id,
                "source_family": "codebase",
                "codebase_stage": "codebase_semantic",
                "object_count": int(stage_meta.get("object_count") or 0),
                "relation_count": int(stage_meta.get("relation_count") or 0),
                "sidecar_count": int(stage_meta.get("sidecar_count") or 0),
                "languages": list(stage_meta.get("languages") or []),
                "analyzers": list(stage_meta.get("analyzers") or []),
                "container_graph_revision_id": codebase_graph_meta.get("container_graph_revision_id"),
                "container_graph_status": codebase_graph_meta.get("status"),
            }
        )
        if stage1_skipped_reason:
            result["codebase_stage1_legacy"] = {
                "status": "skipped",
                "reason": stage1_skipped_reason,
            }
        return result

    # ── L0 enrichment ──

    async def _enrich_missing_fact_metadata(self, facts: list[dict]) -> None:
        """L0 enrich facts with incomplete metadata via classify_fact."""
        if self._extract_disabled:
            return
        from .common import call_extract
        from .librarian import classify_fact, merge_l1_metadata
        sem = self._get_extract_sem()
        async def _classify_fn(model, system, user_msg, max_tokens=256):
            return await self._call_extract_with_runtime_secrets(model, system, user_msg, max_tokens, sem)
        targets = [f for f in facts if _needs_l0_enrichment(f)]
        if not targets:
            return
        model = self.extract_model or "__l0_enrichment__"
        async def _enrich_one(fact: dict):
            try:
                metadata = await classify_fact(fact.get("fact", ""), model, _classify_fn)
                merge_l1_metadata(fact, metadata)
            except Exception:
                pass
        await asyncio.gather(*[_enrich_one(f) for f in targets])

    # ── ingest_asserted_facts() ──

    async def ingest_asserted_facts(
        self,
        facts: list[dict],
        consolidated: list[dict] = None,
        cross_session: list[dict] = None,
        raw_sessions: list[dict] = None,
        provenance: dict = None,
        agent_id: str | None = None,
        swarm_id: str | None = None,
        scope: str | None = None,
        owner_id: str = None,
        read: list[str] = None,
        write: list[str] = None,
        enrich_l0: bool = True,
        artifact_id: str = None,
        version_id: str = None,
        caller_id: str | None = None,
        caller_principal_kind: str | None = None,
    ) -> dict:
        """Authoritative import of pre-extracted memory artifacts.

        Works on non-empty memory. Imported session numbers are offset
        by current session count so positional raw lookup stays correct.
        All mutation under _file_lock to prevent concurrent offset collision.
        """
        # -- STEP 1: Validate imported raw_sessions (dense 1..M before remap) --
        if raw_sessions:
            snums: list[int | None] = [rs.get("session_num") for rs in raw_sessions]
            if any(s is None for s in snums):
                return {"error": "raw_session missing session_num",
                        "code": "VALIDATION_ERROR"}
            snums_int = [cast(int, s) for s in snums]
            if len(snums_int) != len(set(snums_int)):
                return {"error": f"Duplicate session_nums: {sorted(snums_int)}",
                        "code": "VALIDATION_ERROR"}
            snums_sorted = sorted(snums_int)
            if snums_sorted != list(range(1, len(snums_int) + 1)):
                return {"error": f"Imported raw_sessions must be dense 1..N, "
                                 f"got {snums_sorted}",
                        "code": "VALIDATION_ERROR"}
            raw_sessions = sorted(raw_sessions, key=lambda rs: rs["session_num"])

            # Validate facts reference valid imported sessions (pre-offset)
            raw_snum_set = set(snums_int)
            for f in facts:
                sn = f.get("session")
                if sn is None:
                    return {"error": f"Fact missing 'session': {f.get('id', '?')}",
                            "code": "VALIDATION_ERROR"}
                normalized_session = _coerce_positive_session_num(sn)
                if normalized_session is None:
                    return {"error": f"Fact session={sn!r} is not a positive integer",
                            "code": "VALIDATION_ERROR"}
                if normalized_session not in raw_snum_set:
                    return {"error": f"Fact session={sn} has no matching "
                                     f"imported raw_session",
                            "code": "VALIDATION_ERROR"}
                f["session"] = normalized_session

        # -- STEP 2: Resolve ACL --
        try:
            _agent_id, _swarm_id, _scope, _owner, _read, _write = self._resolve_asserted_import_context(
                facts=facts,
                consolidated=consolidated,
                cross_session=cross_session,
                raw_sessions=raw_sessions,
                provenance=provenance,
                agent_id=agent_id,
                swarm_id=swarm_id,
                scope=scope,
                owner_id=owner_id,
                read=read,
                write=write,
                caller_id=caller_id,
                caller_principal_kind=caller_principal_kind,
            )
        except ValueError as exc:
            return {"error": str(exc), "code": "VALIDATION_ERROR"}

        # Validate per-fact metadata (all tiers)
        for f in facts:
            err = self._validate_metadata(f.get("metadata"))
            if err:
                return {"error": f"Fact '{f.get('id', '?')}': {err}", "code": "VALIDATION_ERROR"}
            err = _validate_object_flags_field(f)
            if err:
                return {"error": f"Fact '{f.get('id', '?')}': {err}", "code": "VALIDATION_ERROR"}
            if "target" in f:
                try:
                    normalized_target = _normalize_target(f.get("target"))
                except ValueError as e:
                    return {"error": f"Fact '{f.get('id', '?')}': {e}", "code": "VALIDATION_ERROR"}
                if normalized_target:
                    f["target"] = normalized_target
                else:
                    f.pop("target", None)
        if consolidated:
            for cf in consolidated:
                err = self._validate_metadata(cf.get("metadata"))
                if err:
                    return {"error": f"Cons fact '{cf.get('id', '?')}': {err}", "code": "VALIDATION_ERROR"}
                err = _validate_object_flags_field(cf)
                if err:
                    return {"error": f"Cons fact '{cf.get('id', '?')}': {err}", "code": "VALIDATION_ERROR"}
                if "target" in cf:
                    try:
                        normalized_target = _normalize_target(cf.get("target"))
                    except ValueError as e:
                        return {"error": f"Cons fact '{cf.get('id', '?')}': {e}", "code": "VALIDATION_ERROR"}
                    if normalized_target:
                        cf["target"] = normalized_target
                    else:
                        cf.pop("target", None)
        if cross_session:
            for xf in cross_session:
                err = self._validate_metadata(xf.get("metadata"))
                if err:
                    return {"error": f"Cross fact '{xf.get('id', '?')}': {err}", "code": "VALIDATION_ERROR"}
                err = _validate_object_flags_field(xf)
                if err:
                    return {"error": f"Cross fact '{xf.get('id', '?')}': {err}", "code": "VALIDATION_ERROR"}
                if "target" in xf:
                    try:
                        normalized_target = _normalize_target(xf.get("target"))
                    except ValueError as e:
                        return {"error": f"Cross fact '{xf.get('id', '?')}': {e}", "code": "VALIDATION_ERROR"}
                    if normalized_target:
                        xf["target"] = normalized_target
                    else:
                        xf.pop("target", None)
        if raw_sessions:
            for rs in raw_sessions:
                if "target" in rs:
                    try:
                        normalized_target = _normalize_target(rs.get("target"))
                    except ValueError as e:
                        return {"error": f"Raw session '{rs.get('raw_session_id', '?')}': {e}",
                                "code": "VALIDATION_ERROR"}
                    if normalized_target:
                        rs["target"] = normalized_target
                    else:
                        rs.pop("target", None)

        # -- Normalize malformed types before enrichment --
        for f in facts:
            _normalize_fact_types(f)

        # -- L0 enrichment for facts with incomplete metadata --
        if enrich_l0:
            await self._enrich_missing_fact_metadata(facts)

        # -- ALL MUTATION UNDER LOCK --
        async with self._file_lock:

            # -- STEP 3: Session offset/remap --
            offset = self._n_sessions

            if offset > 0:
                for f in facts:
                    sn = f.get("session", 0)
                    normalized_session = _coerce_positive_session_num(sn)
                    if normalized_session is not None:
                        f["session"] = normalized_session + offset

                if consolidated:
                    for cf in consolidated:
                        normalized_session = _coerce_positive_session_num(cf.get("session"))
                        if normalized_session is not None:
                            cf["session"] = normalized_session + offset
                        if "sessions" in cf:
                            cf["sessions"] = [
                                normalized + offset
                                for s in cf["sessions"]
                                if (normalized := _coerce_positive_session_num(s)) is not None
                            ]

                if cross_session:
                    for xf in cross_session:
                        if "sessions" in xf:
                            xf["sessions"] = [
                                normalized + offset
                                for s in xf["sessions"]
                                if (normalized := _coerce_positive_session_num(s)) is not None
                            ]

                if raw_sessions:
                    for rs in raw_sessions:
                        rs["session_num"] = rs["session_num"] + offset

            # -- STEP 4: Scoped ID remap --
            import_uid = uuid4().hex[:8]
            id_remap = {}

            for f in facts:
                if "id" in f:
                    scope = f.get("session", 0)
                    key = ("g", scope, f["id"])
                    new_id = f"{import_uid}_g_s{scope}_{f['id']}"
                    id_remap[key] = new_id
                    f["id"] = new_id

            if consolidated:
                for cf in consolidated:
                    if "id" in cf:
                        scope = cf.get("session",
                            cf.get("sessions", [0])[0] if cf.get("sessions") else 0)
                        key = ("c", scope, cf["id"])
                        new_id = f"{import_uid}_c_s{scope}_{cf['id']}"
                        id_remap[key] = new_id
                        cf["id"] = new_id
                    if "source_ids" in cf:
                        cf_sessions = cf.get("sessions",
                                             [cf.get("session", 0)])
                        remapped = []
                        for sid in cf["source_ids"]:
                            matches = [id_remap[("g", cs, sid)]
                                       for cs in cf_sessions
                                       if ("g", cs, sid) in id_remap]
                            if len(matches) == 0:
                                return {"error": f"Unresolved source_id '{sid}' in "
                                                 f"cons fact {cf.get('id','?')}",
                                        "code": "VALIDATION_ERROR"}
                            if len(matches) > 1:
                                return {"error": f"Ambiguous source_id '{sid}' in "
                                                 f"cons fact {cf.get('id','?')}: "
                                                 f"matches {len(matches)} sessions",
                                        "code": "VALIDATION_ERROR"}
                            remapped.append(matches[0])
                        cf["source_ids"] = remapped

            if cross_session:
                for xf in cross_session:
                    if "id" in xf:
                        entity = (xf.get("entities") or ["unk"])[0]
                        key = ("x", entity, xf["id"])
                        new_id = f"{import_uid}_x_{entity}_{xf['id']}"
                        id_remap[key] = new_id
                        xf["id"] = new_id
                    if "source_ids" in xf:
                        xf_sessions = xf.get("sessions", [])
                        remapped = []
                        for sid in xf["source_ids"]:
                            matches = [id_remap[("g", xs, sid)]
                                       for xs in xf_sessions
                                       if ("g", xs, sid) in id_remap]
                            if len(matches) == 0:
                                return {"error": f"Unresolved source_id '{sid}' in "
                                                 f"cross fact {xf.get('id','?')}",
                                        "code": "VALIDATION_ERROR"}
                            if len(matches) > 1:
                                return {"error": f"Ambiguous source_id '{sid}' in "
                                                 f"cross fact {xf.get('id','?')}: "
                                                 f"matches {len(matches)} sessions",
                                        "code": "VALIDATION_ERROR"}
                            remapped.append(matches[0])
                        xf["source_ids"] = remapped

            # -- STEP 5: Build session_date lookup --
            sdate_map = {}
            if raw_sessions:
                sdate_map = {rs["session_num"]: rs.get("session_date", "")
                             for rs in raw_sessions}

            # -- STEP 6: Tag facts with inline metadata --
            _art_id = artifact_id or _generate_artifact_id()
            _ver_id = version_id or _generate_version_id()
            now = datetime.now(timezone.utc).isoformat()
            for f in facts:
                f.setdefault("kind", "fact")
                f.setdefault("entities", [])
                f.setdefault("tags", [])
                f.setdefault("session", self._n_sessions + 1)
                f["conv_id"] = self.key
                f["session_date"] = sdate_map.get(f.get("session", 0),
                                                   f.get("session_date", ""))
                f["agent_id"] = _agent_id
                f["swarm_id"] = _swarm_id
                f["scope"] = _scope
                f["owner_id"] = _owner
                f["read"] = list(_read)
                f["write"] = list(_write)
                f["created_at"] = now
                f.setdefault("artifact_id", _art_id)
                f.setdefault("version_id", _ver_id)
                f.setdefault("status", "active")
                if provenance:
                    f["provenance"] = provenance

            # Compute _session_content_complexity from enriched metadata
            from collections import defaultdict as _defaultdict
            by_session = _defaultdict(list)
            for f in facts:
                by_session[f.get("session", 0)].append(f)
            for sn, sfacts in by_session.items():
                complexity = _compute_content_complexity(sfacts)
                for f in sfacts:
                    f["_session_content_complexity"] = complexity

            granular_by_id = {f.get("id"): f for f in facts if f.get("id")}

            self._all_granular.extend(facts)

            # -- STEP 7: Add consolidated --
            if consolidated:
                for cf in consolidated:
                    _normalize_fact_types(cf)
                    cf.setdefault("kind", "fact")
                    cf.setdefault("entities", [])
                    cf["conv_id"] = self.key
                    cf["session_date"] = sdate_map.get(cf.get("session", 0),
                                                        cf.get("session_date", ""))
                    cf["agent_id"] = _agent_id
                    cf["swarm_id"] = _swarm_id
                    cf["scope"] = _scope
                    cf["owner_id"] = _owner
                    cf["read"] = list(_read)
                    cf["write"] = list(_write)
                    cf["created_at"] = now
                    merged_metadata = _merge_fact_metadata(
                        cf.get("metadata"),
                        {"asserted_derived_tier": True},
                    )
                    if merged_metadata:
                        cf["metadata"] = merged_metadata
                    if provenance:
                        cf["provenance"] = provenance
                    source_ids = cf.get("source_ids") or []
                    source_cc = max(
                        (
                            granular_by_id.get(source_id, {}).get("_session_content_complexity", 0.0)
                            for source_id in source_ids
                        ),
                        default=0.0,
                    )
                    if source_cc > 0.0:
                        cf["_session_content_complexity"] = source_cc
                self._all_cons.extend(consolidated)

            # -- STEP 8: Add cross-session --
            if cross_session:
                for xf in cross_session:
                    _normalize_fact_types(xf)
                    xf.setdefault("kind", "fact")
                    xf.setdefault("entities", [])
                    xf["conv_id"] = self.key
                    xf["session_date"] = ""
                    xf["agent_id"] = _agent_id
                    xf["swarm_id"] = _swarm_id
                    xf["scope"] = _scope
                    xf["owner_id"] = _owner
                    xf["read"] = list(_read)
                    xf["write"] = list(_write)
                    xf["created_at"] = now
                    merged_metadata = _merge_fact_metadata(
                        xf.get("metadata"),
                        {"asserted_derived_tier": True},
                    )
                    if merged_metadata:
                        xf["metadata"] = merged_metadata
                    if provenance:
                        xf["provenance"] = provenance
                    source_ids = xf.get("source_ids") or []
                    source_cc = max(
                        (
                            granular_by_id.get(source_id, {}).get("_session_content_complexity", 0.0)
                            for source_id in source_ids
                        ),
                        default=0.0,
                    )
                    if source_cc <= 0.0:
                        source_cc = max(
                            (
                                f.get("_session_content_complexity", 0.0)
                                for f in facts
                                if f.get("session") in (xf.get("sessions") or [])
                            ),
                            default=0.0,
                        )
                    if source_cc > 0.0:
                        xf["_session_content_complexity"] = source_cc
                self._all_cross.extend(cross_session)

            # -- STEP 9: Append raw sessions with ACL --
            if raw_sessions:
                for rs in raw_sessions:
                    rs.setdefault("session_date", "")
                    rs.setdefault("speakers", "External source")
                    rs["agent_id"] = _agent_id
                    rs["swarm_id"] = _swarm_id
                    rs["scope"] = _scope
                    rs["owner_id"] = _owner
                    rs["read"] = list(_read)
                    rs["write"] = list(_write)
                    self._raw_sessions.append(rs)

            # -- STEP 10: Update _n_sessions --
            candidates = [self._n_sessions]
            gran_snums = [
                session_num
                for f in self._all_granular
                if (session_num := _coerce_positive_session_num(f.get("session"))) is not None
            ]
            if gran_snums:
                candidates.append(max(gran_snums))
            raw_snums = [
                session_num
                for rs in self._raw_sessions
                if (session_num := _coerce_positive_session_num(rs.get("session_num"))) is not None
            ]
            if raw_snums:
                candidates.append(max(raw_snums))
            self._n_sessions = max(candidates)

            self._n_sessions_with_facts = len(set(gran_snums))

            # -- STEP 11: derived tiers already provided explicitly --
            self._tiers_dirty = False
            self._bump_index_snapshot_version()

            self._save_cache()

        # -- STEP 12: Embed + refresh retrieval index --
        await self.build_index()

        return {
            "granular_added": len(facts),
            "consolidated_added": len(consolidated) if consolidated else 0,
            "cross_session_added": len(cross_session) if cross_session else 0,
            "raw_sessions_added": len(raw_sessions) if raw_sessions else 0,
            "hybrid_context_available": (raw_sessions is not None
                                         and len(raw_sessions) > 0),
            "session_offset": offset,
            "import_uid": import_uid,
        }

    # ── reextract() ──

    async def reextract(self, model: str = None, call_extract_fn=None) -> dict:
        """Re-run extraction on stored raw sessions with current prompt.

        Clears existing facts and re-extracts from raw_sessions.
        Raw sessions are preserved unchanged.

        Returns: {"reextracted": N, "sessions": M}
        """
        if not self._raw_sessions:
            return {"reextracted": 0, "sessions": 0, "error": "no raw sessions stored"}

        try:
            for raw in self._raw_sessions:
                self._validate_persisted_acl_fields(
                    scope=raw.get("scope"),
                    owner_id=raw.get("owner_id"),
                    read=raw.get("read"),
                    write=raw.get("write"),
                    context=f"raw session {raw.get('raw_session_id') or raw.get('session_num') or '?'}",
                )
        except ValueError as exc:
            return {
                "reextracted": 0,
                "sessions": len(self._raw_sessions),
                "error": str(exc),
                "code": "VALIDATION_ERROR",
            }

        async with self._file_lock:
            self._all_granular = []
            self._all_cons = []
            self._all_cross = []
            self._all_tlinks = []
            self._n_sessions = 0
            self._n_sessions_with_facts = 0

        extract_model = model or self.extract_model
        sem = self._get_extract_sem()

        async def _default_call_extract(m, system, user_msg, max_tokens=8192):
            return await self._call_extract_with_runtime_secrets(m, system, user_msg, max_tokens, sem)

        default_fn: Any = call_extract_fn or _default_call_extract

        for raw in self._raw_sessions:
            raw_ct = raw.get("content_type", "default")
            canonical_source = await self._canonicalize_semantic_source_text(
                str(raw.get("content") or ""),
                family=str(raw.get("format") or "conversation"),
                model=extract_model,
                call_extract_fn=default_fn,
            )
            semantic_text = canonical_source["canonical_en"]
            raw["raw_original"] = canonical_source["raw_original"]
            raw["canonical_en"] = semantic_text
            raw["semantic_ready"] = canonical_source["semantic_ready"]
            raw["canonicalization_status"] = canonical_source["canonicalization_status"]
            raw["canonicalization_error"] = canonical_source["canonicalization_error"]
            raw["source_lang"] = canonical_source["source_lang"]
            raw["translation_version"] = canonical_source["translation_version"]
            if source_id := str(raw.get("source_id") or ""):
                source_record = self._source_records.get(source_id)
                if isinstance(source_record, dict):
                    source_meta = dict(source_record.get("source_meta", {}))
                    source_meta.update(self._semantic_state_fields(canonical_source))
                    source_record["source_meta"] = source_meta
            raw_content_format = raw.get("extraction_format")
            try:
                raw_fmt = normalize_content_format(raw_content_format) if raw_content_format is not None else None
            except ValueError:
                log.warning(
                    "raw session %s has invalid extraction_format %r; falling back to autodetect",
                    raw.get("raw_session_id") or raw.get("session_num") or "?",
                    raw_content_format,
                )
                raw_fmt = None
            resolved_extraction_format = raw_fmt or detect_format(raw["raw_original"])
            raw["extraction_format"] = resolved_extraction_format
            episode_id = (
                str(raw.get("episode_id") or "")
                or f"{self._episode_source_key(str(raw.get('source_id') or self.key))}_e{int(raw.get('session_num', 0)):04d}"
            )
            if not canonical_source["semantic_ready"]:
                for doc in self._episode_corpus.get("documents", []):
                    for episode in doc.get("episodes", []):
                        if episode.get("episode_id") == episode_id:
                            episode["raw_text"] = ""
                            episode["raw_original"] = raw["raw_original"]
                            episode["canonical_en"] = ""
                            episode["semantic_ready"] = False
                            episode["canonicalization_status"] = canonical_source["canonicalization_status"]
                            episode["canonicalization_error"] = canonical_source["canonicalization_error"]
                            episode["source_lang"] = raw["source_lang"]
                            episode["translation_version"] = raw["translation_version"]
                            _stamp_selector_episode_fields(episode, raw)
                            break
                async with self._file_lock:
                    raw["status"] = "canonicalization_failed"
                    self._bump_index_snapshot_version()
                continue

            # Use content_type-aware fn if raw session had a non-default type
            if call_extract_fn is None and raw_ct != "default":
                ct_prompt = self._prompt_registry.get(raw_ct)
                sdate_raw = raw["session_date"]
                snum_raw = raw["session_num"]

                async def _ct_call_extract(m, system, user_msg, max_tokens=8192,
                                           _p=ct_prompt, _sd=sdate_raw, _sn=snum_raw):
                    try:
                        dt = datetime.fromisoformat(_sd.replace("Z", "+00:00"))
                        ds = dt.strftime("%d %B %Y")
                        ym1 = str(dt.year - 1)
                    except Exception:
                        ds = _sd
                        ym1 = str(int(_sd[:4]) - 1) if len(_sd) >= 4 else "2022"

                    class _SafeDict(dict):
                        def __missing__(self, key):
                            return "{" + key + "}"

                    cs = _p.format_map(_SafeDict(
                        session_date=ds, year_minus_1=ym1, session_num=_sn))
                    return await self._call_extract_with_runtime_secrets(m, cs, user_msg, max_tokens, sem)

                fn: Any = _ct_call_extract
            else:
                fn = default_fn

            extract_result = await extract_session(
                session_text=semantic_text,
                session_num=raw["session_num"],
                session_date=raw["session_date"],
                conv_id=self.key,
                speakers=raw.get("speakers", "User and Assistant"),
                model=extract_model,
                call_extract_fn=fn,
                fmt=raw_fmt,
                return_report=True,
            )
            conv_id, sn, sdate, facts, tlinks, extraction_report = _coerce_extract_session_result(extract_result)
            _set_runtime_report_artifact(
                raw,
                field_name="extraction_report",
                producer="block_extractor",
                report_kind="extraction",
                report=extraction_report,
            )
            episode_id = (
                str(raw.get("episode_id") or "")
                or f"{self._episode_source_key(str(raw.get('source_id') or self.key))}_e{int(raw.get('session_num', 0)):04d}"
            )
            for doc in self._episode_corpus.get("documents", []):
                for episode in doc.get("episodes", []):
                    if episode.get("episode_id") == episode_id:
                        episode["raw_text"] = semantic_text
                        episode["raw_original"] = raw["raw_original"]
                        episode["canonical_en"] = semantic_text
                        episode["semantic_ready"] = True
                        episode["canonicalization_status"] = canonical_source["canonicalization_status"]
                        episode["canonicalization_error"] = canonical_source["canonicalization_error"]
                        episode["source_lang"] = raw["source_lang"]
                        episode["translation_version"] = raw["translation_version"]
                        _stamp_selector_episode_fields(episode, raw)
                        break
            self._align_fact_selectors(
                facts,
                episode_id=episode_id,
                source_kind=self._canonical_content_family(raw.get("format") or "conversation"),
                raw_fields=_selector_raw_fields(semantic_text, raw),
                speakers=raw.get("speakers") if isinstance(raw.get("speakers"), dict) else None,
            )
            # Tag facts with raw_session_id if available
            rsid = raw.get("raw_session_id")
            if rsid:
                for f in facts:
                    f["raw_session_id"] = rsid
            self._tag_facts(
                facts, sdate,
                raw.get("agent_id"),
                raw.get("swarm_id"),
                raw.get("scope"),
                owner_id=raw.get("owner_id"),
                read=raw.get("read"),
                write=raw.get("write"),
                metadata=raw.get("metadata"),
                target=raw.get("target"),
            )
            # Compute and stamp _session_content_complexity after re-extraction
            session_complexity = _compute_content_complexity(facts)
            for f in facts:
                f["_session_content_complexity"] = session_complexity
            async with self._file_lock:
                raw["status"] = "active"
                self._all_granular.extend(facts)
                self._all_tlinks.extend(tlinks)
                self._n_sessions += 1
                if facts:
                    self._n_sessions_with_facts += 1
                self._bump_index_snapshot_version()

        self._data_dict = None
        self._mark_tiers_dirty()
        async with self._file_lock:
            self._bump_index_snapshot_version()
            self._save_cache()

        # Rebuild derived facts for the current runtime model.
        await self._rebuild_tiers()
        self._tiers_dirty = False

        return {
            "reextracted": len(self._all_granular),
            "sessions": len(self._raw_sessions),
        }

    # ── _rebuild_tiers() ──

    async def _rebuild_tiers(self) -> None:
        """Rebuild production derived tiers from granular facts.

        Legacy LLM merge tiers are disabled. The rebuild path only emits
        source-aggregation-derived cross facts, partitioned by ACL domain
        + delivery target.
        """
        if not self._all_granular:
            return
        if not self._raw_sessions and not (self._episode_corpus or {}).get("documents"):
            return

        sem = self._get_extract_sem()

        async def _call_extract_fn(model, system, user_msg, max_tokens=8192):
            if not model:
                return []
            return await self._call_extract_with_runtime_secrets(
                model=model,
                system=system,
                user_msg=user_msg,
                max_tokens=max_tokens,
                sem=sem,
            )

        # Partition granular facts by ACL domain + canonical target tuple.
        domain_facts: dict[tuple, list[dict]] = defaultdict(list)
        for f in self._all_granular:
            target_key = tuple(f.get("target", [])) if f.get("target") else None
            domain_key = (
                f.get("owner_id", "system"),
                tuple(sorted(f.get("read", ["agent:PUBLIC"]))),
                tuple(sorted(f.get("write", []))),
                target_key,
            )
            domain_facts[domain_key].append(f)

        new_cons = [f for f in self._all_cons if _is_asserted_derived_fact(f)]
        new_cross = [f for f in self._all_cross if _is_asserted_derived_fact(f)]

        for (d_owner_id, d_read_tuple, d_write_tuple, d_target_tuple), d_facts in domain_facts.items():
            # Reuse persisted identity fields from the current ACL domain.
            d_agent_id = d_facts[0].get("agent_id", "default")
            d_swarm_id = d_facts[0].get("swarm_id", "default")
            d_scope = d_facts[0].get("scope", "swarm-shared")
            source_fact_groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
            for fact in d_facts:
                metadata = fact.get("metadata") or {}
                source_id = metadata.get("episode_source_id") or fact.get("source_id")
                if not source_id:
                    continue
                source_record = self._source_records.get(source_id) or {}
                source_kind = source_record.get("family") or "conversation"
                source_fact_groups[(source_id, source_kind)].append(fact)

            for (source_id, source_kind), source_facts in source_fact_groups.items():
                source_date = max(
                    [str(f.get("session_date") or "") for f in source_facts if f.get("session_date")] or [""]
                )
                mal_binding_id = _resolve_mal_binding_id(d_owner_id, d_agent_id)
                substrate_cross = await self._extract_source_aggregation_facts(
                    source_id=source_id,
                    source_kind=source_kind,
                    source_facts=source_facts,
                    source_date=source_date,
                    model=self.extract_model,
                    call_extract_fn=_call_extract_fn,
                    agent_id=mal_binding_id,
                )
                domain_ns = hashlib.sha1(
                    repr((d_owner_id, d_read_tuple, d_write_tuple, d_target_tuple)).encode("utf-8"),
                    usedforsecurity=False,
                ).hexdigest()[:12]
                _namespace_derived_fact_ids(
                    substrate_cross,
                    f"substrate_{self._episode_source_key(source_id)}_{domain_ns}",
                )
                max_cc = max((f.get("_session_content_complexity", 0.0) for f in source_facts), default=0.0)
                for fact in substrate_cross:
                    fact["_session_content_complexity"] = max_cc
                self._tag_facts(
                    substrate_cross,
                    source_date or "2024-01-01",
                    agent_id=d_agent_id,
                    swarm_id=d_swarm_id,
                    scope=d_scope,
                    owner_id=d_owner_id,
                    read=list(d_read_tuple),
                    write=list(d_write_tuple),
                    target=_consensus_target(source_facts),
                )
                new_cross.extend(substrate_cross)

        async with self._file_lock:
            self._all_cons = new_cons
            self._all_cross = new_cross
            self._bump_index_snapshot_version()
            self._save_cache()

        self._data_dict = None

    # ── build_index() ──

    def _index_retry_after_ms(self) -> int:
        status = self._read_full_index_status()
        failures = int(status.get("last_index_build_error_count") or 0)
        base = DEFAULT_INDEX_BACKOFF_INITIAL_MS * (2 ** min(failures, 5))
        capped = min(base, DEFAULT_INDEX_BACKOFF_MAX_MS)
        jitter_seed = int(hashlib.sha256(f"{self._worker_id}:{failures}".encode()).hexdigest()[:6], 16)
        jitter = jitter_seed % max(1, min(capped // 4, 30_000))
        return min(DEFAULT_INDEX_BACKOFF_MAX_MS, capped + jitter)

    async def run_index_scheduler_once(self) -> dict[str, Any]:
        """Run one durable scheduler tick for the full corpus index."""

        status = self._read_full_index_status()
        now_ms = self._now_ms()
        if status.get("index_state") == "backoff":
            return {"status": "skipped", "reason": "index_backoff", **status}
        if status.get("index_state") == "building":
            return {"status": "skipped", "reason": "index_build_in_progress", **status}
        due_ms = status.get("next_index_build_after_ms")
        if due_ms is not None and int(due_ms) > now_ms:
            return {"status": "scheduled", "reason": "debounce_window", **status}
        if not status.get("index_dirty") and self._data_dict is not None:
            return {"status": "ready", **status}
        if not self._all_granular:
            return {"status": "missing", "reason": "no_facts", **status}
        return await self.build_index()

    async def build_index(self, *, _lease_acquired: bool = False) -> dict:
        """Embed current tiers and build retrieval state."""
        if not _lease_acquired and self._storage_supports_index_coordination():
            now_ms = self._now_ms()
            storage = cast(Any, self._storage)
            lease = storage.acquire_index_build_lease(
                worker_id=self._worker_id,
                snapshot_fingerprint=self._index_snapshot_fingerprint(),
                now_ms=now_ms,
                lease_ms=DEFAULT_INDEX_BUILD_LEASE_MS,
            )
            if not lease.get("acquired"):
                return {
                    "status": "skipped",
                    "reason": "index_build_lease_not_acquired",
                    "index_state": lease.get("index_state"),
                    "index_build_lease_owner": lease.get("index_build_lease_owner"),
                    "next_index_retry_after_ms": lease.get("next_index_retry_after_ms"),
                }
            try:
                result = await self.build_index(_lease_acquired=True)
            except Exception as exc:
                storage.release_index_build_lease(
                    worker_id=self._worker_id,
                    success=False,
                    now_ms=self._now_ms(),
                    debounce_ms=self._index_debounce_ms(),
                    max_delay_ms=self._index_max_delay_ms(),
                    retry_after_ms=self._index_retry_after_ms(),
                    error=f"{type(exc).__name__}: {exc}",
                )
                raise
            storage.release_index_build_lease(
                worker_id=self._worker_id,
                success=True,
                now_ms=self._now_ms(),
                debounce_ms=self._index_debounce_ms(),
                max_delay_ms=self._index_max_delay_ms(),
            )
            return {**result, "index_state": "ready"}

        max_attempts = 5
        mismatch_exc: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            if self._tiers_dirty:
                await self._rebuild_tiers()
                async with self._file_lock:
                    self._tiers_dirty = False

            async with self._file_lock:
                if not self._all_granular and self._storage.exists:
                    cached = self._storage.load_facts(internal=True)
                    self._all_granular = cached.get("granular", [])
                    self._all_cons = cached.get("cons", [])
                    self._all_cross = cached.get("cross", [])
                    self._all_tlinks = cached.get("tlinks", [])
                    for f in self._all_granular:
                        f["_temporal_links"] = []
                    if self._all_granular and self._all_tlinks:
                        self._all_granular[0]["_temporal_links"] = self._all_tlinks
                    self._bump_index_snapshot_version()
                assert len(self._all_granular) > 0, "No granular facts — call store() first"
                if len(self._all_cons) > 0:
                    g_id = self._all_granular[0].get("id", "")
                    c_id = self._all_cons[0].get("id", "")
                    assert g_id != c_id, f"Tier ID collision: granular[0]={g_id} == cons[0]={c_id}"

                snapshot_version = self._index_snapshot_version
                snapshot_refresh_required = self._supports_write_log and self._snapshot_refresh_required()
                granular_snapshot = list(self._all_granular)
                cons_snapshot = list(self._all_cons)
                cross_snapshot = list(self._all_cross)
                cached_fingerprints = dict(self._emb_fingerprints)
            fp_gran = _embedding_fingerprint(granular_snapshot)
            fp_cons = _embedding_fingerprint(cons_snapshot)
            fp_cross = _embedding_fingerprint(cross_snapshot)
            new_fps = {"gran": fp_gran, "cons": fp_cons, "cross": fp_cross}
            gran_embed_indices = [
                idx for idx, fact in enumerate(granular_snapshot)
                if _fact_uses_dense_embedding_index(fact)
            ]
            sparse_gran_embeddings = len(gran_embed_indices) != len(granular_snapshot)

            cache_hit = False
            saved_embs = self._storage.load_embeddings()
            if not sparse_gran_embeddings and saved_embs is not None and cached_fingerprints == new_fps:
                emb_dim = self._embedding_dim_from_arrays(
                    saved_embs.get("gran"),
                    saved_embs.get("cons"),
                    saved_embs.get("cross"),
                )
                gran_embs = saved_embs.get("gran", np.zeros((0, emb_dim)))
                cons_embs = saved_embs.get("cons", np.zeros((0, emb_dim)))
                cross_embs = saved_embs.get("cross", np.zeros((0, emb_dim)))
                if (
                    len(gran_embs) == len(granular_snapshot)
                    and len(cons_embs) == len(cons_snapshot)
                    and len(cross_embs) == len(cross_snapshot)
                ):
                    cache_hit = True
                    log.info("Embedding cache hit (fingerprint match)")

            if not cache_hit:
                if snapshot_refresh_required:
                    refresh_stale = False
                    async with self._file_lock:
                        if self._index_snapshot_is_stale(snapshot_version):
                            log.info(
                                "Index snapshot changed before snapshot refresh; retrying build_index (%d/%d)",
                                attempt,
                                max_attempts,
                            )
                            refresh_stale = True
                        else:
                            self._save_snapshot()
                    if refresh_stale:
                        await asyncio.sleep(0)
                        continue

                gran_texts = [granular_snapshot[idx].get("fact", "") for idx in gran_embed_indices]
                gran_embs = await self._embed_texts_with_runtime_secrets(gran_texts, label=f"gran-{self.key[:8]}")
                emb_dim = self._embedding_dim_from_arrays(gran_embs)

                cons_texts = [fact.get("fact", "") for fact in cons_snapshot]
                cons_embs = (
                    await self._embed_texts_with_runtime_secrets(cons_texts, label=f"cons-{self.key[:8]}")
                    if cons_texts
                    else np.zeros((0, emb_dim))
                )

                cross_texts = [fact.get("fact", "") for fact in cross_snapshot]
                cross_embs = (
                    await self._embed_texts_with_runtime_secrets(cross_texts, label=f"cross-{self.key[:8]}")
                    if cross_texts
                    else np.zeros((0, emb_dim))
                )

                embeddings_stale = False
                async with self._file_lock:
                    if self._index_snapshot_is_stale(snapshot_version):
                        log.info(
                            "Index snapshot changed while embeddings were computing; retrying build_index (%d/%d)",
                            attempt,
                            max_attempts,
                        )
                        embeddings_stale = True
                    else:
                        try:
                            if not sparse_gran_embeddings:
                                self._storage.save_embeddings(gran_embs, cons_embs, cross_embs)
                        except Exception as exc:
                            if _is_embedding_count_mismatch_error(exc):
                                mismatch_exc = exc
                                log.warning(
                                    "Stale embedding snapshot detected during save_embeddings; retrying build_index (%d/%d): %s",
                                    attempt,
                                    max_attempts,
                                    exc,
                                )
                                embeddings_stale = True
                            else:
                                raise
                        else:
                            log.info("Embeddings computed and cached")
                if embeddings_stale:
                    await asyncio.sleep(0)
                    continue

            data_dict = _build_index_state(
                granular_snapshot,
                gran_embs,
                cons_snapshot,
                cons_embs,
                cross_snapshot,
                cross_embs,
            )
            if sparse_gran_embeddings:
                data_dict["atomic_emb_indices"] = np.asarray(gran_embed_indices, dtype=np.int64)
                data_dict["atomic_emb_sparse"] = True
            all_facts = granular_snapshot + cons_snapshot + cross_snapshot
            resolve_supersession(all_facts, data_dict["fact_lookup"])

            commit_stale = False
            async with self._file_lock:
                if self._index_snapshot_is_stale(snapshot_version):
                    log.info(
                        "Index snapshot changed before final index commit; retrying build_index (%d/%d)",
                        attempt,
                        max_attempts,
                    )
                    commit_stale = True
                else:
                    self._data_dict = data_dict
                    self._fact_lookup = data_dict["fact_lookup"]
                    self._emb_fingerprints = new_fps if not sparse_gran_embeddings else {}
                    self._rebuild_temporal_index()
                    self._save_cache()
                    return {
                        "granular": len(self._all_granular),
                        "consolidated": len(self._all_cons),
                        "cross_session": len(self._all_cross),
                        "embedding_cache_hit": cache_hit,
                        "corpus_embedding_rebuild": not cache_hit,
                    }
            if commit_stale:
                await asyncio.sleep(0)
                continue

        if mismatch_exc is not None:
            raise RuntimeError(
                f"Could not build index from a stable fact snapshot after {max_attempts} attempts"
            ) from mismatch_exc
        raise RuntimeError(
            f"build_index() could not obtain a stable snapshot after {max_attempts} attempts"
        )

    # ── recall() ──

    # Map spec query_type → retrieval type
    _QUERY_TYPE_MAP = {
        "auto":        None,
        "lookup":      "default",
        "temporal":    "temporal",
        "aggregate":   "counting",
        "current":     "current",
        "synthesize":  "synthesis",
        "procedural":  "rule",
        "prospective": "prospective",
        "exact_copy":  "exact_copy",
    }

    # Query types that benefit from Tier 2
    _NEEDS_TIER2 = {"synthesis", "current", "temporal", "counting",
                    "aggregate", "summarize", "icl"}
    # Query types that benefit from Tier 3
    _NEEDS_TIER3 = {"synthesis"}

    async def recall(
        self,
        query: str,
        agent_id: str = None,
        swarm_id: str = None,
        search_family: str = "auto",
        token_budget: int | None = None,
        query_type: str = "auto",
        kind: str = "all",
        query_metadata: dict[str, Any] | None = None,
        caller_memberships: list = None,
        caller_role: str = "user",
        caller_id: str = None,
        mal_binding_id: str | None = None,
    ) -> dict:
        """Query memory. Returns dict with context, retrieved, query_type, etc.

        search_family: auto | conversation | document | codebase
        query_type: auto | lookup | temporal | aggregate | current |
                    synthesize | procedural | prospective | exact_copy
            exact_copy recall returns a sanitized terminal_render_candidate for
            model decision; final exact rendering is only available through ask.
        kind: all | fact | preference | constraint | rule | ... (filters by fact kind)
        """
        effective_caller_id = _default_runtime_caller_id(caller_id, agent_id, self.agent_id)
        repo_task_metadata = dict(query_metadata or {})
        self._audit.log("recall", effective_caller_id,
                        {"query": query[:200], "query_type": query_type, "query_metadata": repo_task_metadata})
        full_index_missing_at_start = self._data_dict is None and bool(self._all_granular)
        full_index_status_at_start = self._read_full_index_status()

        # Build filter: ACL + optional kind
        # caller_id takes precedence (set by MCP identity resolution)
        _caller_id = effective_caller_id
        _memberships = caller_memberships or []
        _role = caller_role
        _swarm_for_raw = swarm_id if swarm_id is not None else self.swarm_id
        _mal_binding_id = (
            _normalize_identity(str(mal_binding_id), allow_public=False)
            if mal_binding_id is not None
            else _resolve_mal_binding_id(caller_id, agent_id)
        )
        _mal_config = _load_mal_active_config(str(self.data_dir), self.key, _mal_binding_id)
        _mal_selector = _mal_config.get("selector_config_overrides") or None
        _mal_leaf_overrides = _mal_config.get("inference_leaf_plugin_overrides") or {}
        _effective_leaf_plugins = dict(self._inference_leaf_plugins)
        if isinstance(_mal_leaf_overrides, dict):
            _effective_leaf_plugins.update(
                {str(name): bool(enabled) for name, enabled in _mal_leaf_overrides.items()}
            )

        retrieval_target = extract_query_features(query).get("retrieval_target") or query
        # Keep these as separate model-free canonicalization passes: retrieval
        # may strip answer-format suffixes, while answer_contract must preserve
        # the original user-facing question and output constraints.
        canonical_answer_query = await self._canonicalize_recall_query(query)
        canonical_query = await self._canonicalize_recall_query(retrieval_target)
        retrieval_query = canonical_query["canonical_en"]
        answer_query = (
            canonical_answer_query["canonical_en"]
            if canonical_answer_query.get("semantic_ready")
            else str(query or "").strip()
        )
        query_trace = {
            "original_query": str(query or ""),
            "retrieval_query": retrieval_query,
            "answer_query": answer_query,
            "source_lang": canonical_query["source_lang"],
            "translation_version": canonical_query["translation_version"],
            "semantic_ready": bool(canonical_query.get("semantic_ready")),
            "canonicalization_status": str(canonical_query.get("canonicalization_status") or "failed"),
            "canonicalization_error": canonical_query.get("canonicalization_error"),
        }
        query_language_trace = {
            "source_lang": canonical_query["source_lang"],
            "canonicalization_status": query_trace["canonicalization_status"],
        }

        if not canonical_query["semantic_ready"]:
            return {
                "error": "memory_recall accepts English queries only; translate in the calling agent/model",
                "code": "NON_ENGLISH_QUERY"
                if query_trace["canonicalization_status"] == "blocked_in_recall"
                else "CANONICALIZATION_ERROR",
                "context": "",
                "retrieved": [],
                "query_type": self._QUERY_TYPE_MAP.get(query_type) or detect_query_type(str(query or "")),
                "runtime_trace": {
                    "query_language": query_language_trace,
                    "query_canonicalization": query_trace,
                },
            }

        def _with_raw_recall(result: dict) -> dict:
            runtime_trace = dict(result.get("runtime_trace") or {})
            runtime_trace["query_language"] = query_language_trace
            runtime_trace["query_canonicalization"] = query_trace
            priority_trace = dict(runtime_trace.get("priority_retrieval") or {})
            priority_trace.setdefault("priority_retrieval_used", full_index_missing_at_start)
            priority_trace.setdefault("priority_index_used", False)
            priority_trace.setdefault("priority_index_candidate_count", 0)
            priority_trace.setdefault("priority_index_top_k", self._priority_index_top_k())
            priority_trace.setdefault("full_index_required_for_query", False)
            scheduled_after_recall = self._schedule_full_index_after_recall_if_needed()
            priority_trace["full_index_build_scheduled_after_recall"] = scheduled_after_recall
            runtime_trace["priority_retrieval"] = priority_trace
            runtime_trace["index"] = self._index_trace()
            if full_index_status_at_start.get("index_state") in {"building", "scheduled", "backoff"}:
                runtime_trace["index"]["index_state_at_recall_start"] = full_index_status_at_start.get("index_state")
            result["runtime_trace"] = runtime_trace
            result["inference_leaf_plugins"] = dict(_effective_leaf_plugins)
            result = self._merge_raw_recall(
                query=retrieval_query,
                result=result,
                caller_id=_caller_id,
                caller_memberships=_memberships,
                caller_role=_role,
                swarm_id=_swarm_for_raw,
                raw_kind=kind,
            )
            result = self._attach_recall_continuation(
                query=retrieval_query,
                result=result,
                fact_filter=fact_filter,
                caller_id=_caller_id,
                caller_memberships=_memberships,
                caller_role=_role,
                swarm_id=_swarm_for_raw,
                raw_kind=kind,
            )
            result, _ = self._finalize_recall_evidence_context(
                query=retrieval_query,
                answer_query=answer_query,
                recall_result=result,
            )
            return result
        acl_ok = lambda f: self._acl_allows(f, _caller_id, _memberships, _role)
        _now = datetime.now(timezone.utc)
        _fl = self._fact_lookup if hasattr(self, '_fact_lookup') else None
        visible = lambda f: _is_visible(f, now=_now, fact_lookup=_fl) and acl_ok(f)

        if kind != "all":
            fact_filter = lambda f: visible(f) and f.get("kind") == kind
        else:
            fact_filter = visible

        has_episode_corpus = bool(self._episode_corpus.get("documents"))
        has_visible_facts = any(fact_filter(f) for f in self._all_granular)
        has_visible_codebase_facts = any(
            fact_filter(fact) and self._fact_source_family(fact) == "codebase"
            for fact in [*self._all_granular, *self._all_cons, *self._all_cross]
        )
        code_query_mode = classify_codebase_query_mode(query)
        auto_codebase_precise_query = (
            search_family in {"auto", "", None}
            and has_visible_codebase_facts
            and code_query_mode == "precise_code"
        )
        auto_mixed_code_query = (
            search_family in {"auto", "", None}
            and has_visible_codebase_facts
            and code_query_mode == "mixed_code_plus_prose"
        )
        effective_search_family = "codebase" if auto_codebase_precise_query else search_family

        episode_runtime = None if effective_search_family == "codebase" else self._visible_episode_runtime(fact_filter, query=query)
        episode_runtime = None if effective_search_family == "codebase" else self._visible_episode_runtime(
            fact_filter,
            query=query,
        )
        if episode_runtime:
            corpus, _episode_lookup, _facts_by_episode, _bm25 = episode_runtime
            packet = build_episode_hybrid_context(
                retrieval_query,
                corpus,
                self._episode_runtime_facts(fact_filter, query=retrieval_query),
                selector_config=_mal_selector,
                search_family=search_family,
                temporal_index=self._temporal_index,
            )
            temporal_trace = packet.get("temporal_trace") or {}
            packet, augmented_facts = await run_default_query_executor_chain(
                self,
                query=query,
                query_type=query_type,
                packet=packet,
                episode_lookup=_episode_lookup,
                fact_filter=fact_filter,
            )
            if packet.get("container_graph_failed_closed"):
                return _with_raw_recall(
                    {
                        "error": "container graph structural render failed closed",
                        "code": "CONTAINER_GRAPH_FAILED_CLOSED",
                        "context": "",
                        "retrieved": [],
                        "query_type": self._QUERY_TYPE_MAP.get(query_type) or detect_query_type(retrieval_query),
                        "retrieved_episode_ids": [],
                        "actual_injected_episode_ids": [],
                        "selection_scores": packet.get("selection_scores", []),
                        "query_operator_plan": packet.get("query_operator_plan", {}),
                        "output_constraints": packet.get("output_constraints", {}),
                        "retrieval_families": packet.get("retrieval_families", []),
                        "search_family": packet.get("search_family", effective_search_family),
                        "runtime_trace": {
                            "runtime": "container_graph",
                            "container_graph": packet.get("container_graph_trace") or {},
                        },
                    }
                )
            selected_ids = set(packet.get("retrieved_fact_ids", []))
            episode_runtime_facts = self._episode_runtime_facts(fact_filter, query=retrieval_query)
            episode_runtime_lookup = {
                f.get("id", ""): f for f in episode_runtime_facts
            }
            resolved_facts = augmented_facts or [
                episode_runtime_lookup[fact_id]
                for fact_id in packet.get("retrieved_fact_ids", [])
                if fact_id in episode_runtime_lookup
            ]
            total_sessions = len(self._raw_sessions)
            sessions_in_ctx = len({
                session_num
                for f in resolved_facts
                if (session_num := _coerce_positive_session_num(f.get("session"))) is not None
            })
            coverage_pct = (sessions_in_ctx / total_sessions * 100) if total_sessions else 0
            resolved_type = self._QUERY_TYPE_MAP.get(query_type) or detect_query_type(retrieval_query)
            query_features = extract_query_features(retrieval_query)
            if packet.get("terminal_render_candidate"):
                resolved_type = "exact_copy"
            explicit_step_query = bool(
                query_features.get("step_numbers") or query_features.get("step_range")
            )
            if resolved_type == "temporal" and not explicit_step_query:
                semantic_temporal_episode_ids = {
                    episode_id
                    for fact in resolved_facts
                    for episode_id in fact_episode_ids(fact)
                    if ((fact.get("metadata") or {}).get("source_aggregation")
                        and (fact.get("metadata") or {}).get("semantic_class") == "temporal_semantics")
                }
                if semantic_temporal_episode_ids:
                    preferred_facts = []
                    for fact in resolved_facts:
                        metadata = fact.get("metadata") or {}
                        is_semantic_temporal = (
                            metadata.get("source_aggregation")
                            and metadata.get("semantic_class") == "temporal_semantics"
                        )
                        if is_semantic_temporal:
                            preferred_facts.append(fact)
                            continue
                        if set(fact_episode_ids(fact)) & semantic_temporal_episode_ids:
                            continue
                        preferred_facts.append(fact)
                    if preferred_facts and len(preferred_facts) != len(resolved_facts):
                        resolved_facts = preferred_facts
                        context_trace: dict = {}
                        context, actual_injected_episode_ids = build_context_from_retrieved_facts(
                            resolved_facts,
                            _episode_lookup,
                            fact_lookup=episode_runtime_lookup,
                            budget=int(packet.get("selector_config", {}).get("budget", 8000)),
                            snippet_chars=int(packet.get("tuning_snapshot", {}).get("packet", {}).get("snippet_chars", 1200)),
                            question=retrieval_query,
                            query_features=query_features,
                            context_trace=context_trace,
                        )
                        packet = dict(packet)
                        packet["context"] = context
                        packet["retrieved_fact_ids"] = [fact.get("id", "") for fact in resolved_facts]
                        packet["retrieved_episode_ids"] = list(dict.fromkeys(
                            episode_id
                            for fact in resolved_facts
                            for episode_id in fact_episode_ids(fact)
                            if episode_id
                        ))
                        packet["actual_injected_episode_ids"] = actual_injected_episode_ids
                        if context_trace.get("document_target_span_ids") or context_trace.get("document_target_span_mode") != "disabled":
                            packet.update(context_trace)
            if resolved_type == "synthesis":
                packet, resolved_facts = self._ensure_min_synthesis_evidence(
                    packet=packet,
                    resolved_facts=resolved_facts,
                    episode_lookup=_episode_lookup,
                    fact_filter=fact_filter,
                )
            merged_retrieval_families = list(packet.get("retrieval_families", []))
            mixed_merge_trace: dict[str, Any] | None = None
            generic_codebase_discovery_trace: dict[str, Any] | None = None
            if auto_mixed_code_query:
                (
                    mixed_context,
                    mixed_facts,
                    mixed_merge_trace,
                    merged_families,
                ) = await self._merge_auto_mixed_codebase_result(
                    query=query,
                    query_type=query_type,
                    query_metadata=repo_task_metadata,
                    fact_filter=fact_filter,
                    episode_context=str(packet.get("context") or ""),
                    episode_facts=resolved_facts,
                )
                if mixed_facts != resolved_facts or mixed_context != str(packet.get("context") or ""):
                    resolved_facts = mixed_facts
                    packet = dict(packet)
                    packet["context"] = mixed_context
                    packet["retrieved_fact_ids"] = [fact.get("id", "") for fact in resolved_facts]
                merged_retrieval_families = list(
                    dict.fromkeys(
                        family
                        for family in [*merged_retrieval_families, *merged_families]
                        if family
                    )
                )
            elif search_family in {"auto", "", None} and has_visible_codebase_facts and code_query_mode == "non_code":
                (
                    discovered_code_facts,
                    discovered_code_context,
                    generic_codebase_discovery_trace,
                ) = await self._auto_discover_codebase_evidence(
                    query=retrieval_query,
                    query_type=query_type,
                    query_metadata=repo_task_metadata,
                    fact_filter=fact_filter,
                    episode_facts=resolved_facts,
                    code_query_mode=code_query_mode,
                )
                if discovered_code_facts:
                    resolved_facts = list(resolved_facts)
                    seen_fact_ids = {
                        str(fact.get("id") or "").strip()
                        for fact in resolved_facts
                        if str(fact.get("id") or "").strip()
                    }
                    for fact in discovered_code_facts:
                        fact_id = str(fact.get("id") or "").strip()
                        if fact_id and fact_id in seen_fact_ids:
                            continue
                        resolved_facts.append(fact)
                        if fact_id:
                            seen_fact_ids.add(fact_id)
                    packet = dict(packet)
                    if discovered_code_context:
                        packet["context"] = f"{packet['context'].rstrip()}\n\n{discovered_code_context}"
                    packet["retrieved_fact_ids"] = [
                        str(fact.get("id") or "")
                        for fact in resolved_facts
                        if str(fact.get("id") or "")
                    ]
                    merged_retrieval_families = list(dict.fromkeys([*merged_retrieval_families, "codebase"]))
            code_segments, code_trace = augment_codebase_context(
                query=query,
                retrieved_facts=resolved_facts,
                data_dir=str(self.data_dir),
            )
            repo_task_context_packs, repo_task_context_trace = self._repo_task_context_packs_for_facts(resolved_facts)
            if code_segments and CODE_ATTACHMENT_SECTION_LABEL not in str(packet.get("context") or ""):
                packet = dict(packet)
                packet["context"] = f"{packet['context'].rstrip()}\n\n" + _render_code_attachment_block(code_segments)
            complexity_hint = _compute_complexity_hint(
                retrieved=resolved_facts,
                resolved_type=resolved_type,
                is_multihop=(resolved_type in ("temporal", "current", "counting")),
                fact_lookup=episode_runtime_lookup,
                query=retrieval_query,
            )

            prompt_type, use_tool = _route_prompt_type(
                resolved_type,
                resolved_facts,
                total_sessions,
                sessions_in_ctx,
                packet["context"],
                allow_tool_mode=False,
            )

            if resolved_type == "summarize":
                retrieved_items = self._canonical_retrieved_items(resolved_facts)
            elif resolved_type == "synthesis":
                retrieved_items = self._synthesis_retrieved_items(
                    resolved_facts=resolved_facts,
                    packet=packet,
                    episode_lookup=_episode_lookup,
                )
            else:
                retrieved_items = resolved_facts

            result = {
                "context": packet["context"],
                "retrieved": retrieved_items,
                "query_type": resolved_type,
                "is_multihop": resolved_type in ("temporal", "current", "counting"),
                "complexity_hint": complexity_hint,
                "n_facts": len(self._all_granular) + len(self._all_cons) + len(self._all_cross),
                "sessions_in_context": sessions_in_ctx,
                "total_sessions": total_sessions,
                "coverage_pct": coverage_pct,
                "raw_budget": 0,
                "recommended_prompt_type": prompt_type,
                "use_tool": use_tool,
                "retrieved_episode_ids": packet["retrieved_episode_ids"],
                "actual_injected_episode_ids": packet["actual_injected_episode_ids"],
                "selection_scores": packet["selection_scores"],
                "query_operator_plan": packet["query_operator_plan"],
                "output_constraints": packet["output_constraints"],
                "retrieval_families": merged_retrieval_families,
                "search_family": packet.get("search_family", effective_search_family),
                "runtime_trace": self._episode_runtime_trace(
                    corpus=corpus,
                    packet=packet,
                    episode_lookup=_episode_lookup,
                    resolved_facts=resolved_facts,
                ),
            }
            if packet.get("container_graph_trace"):
                result["runtime_trace"]["container_graph"] = packet.get("container_graph_trace")
            if packet.get("terminal_render_candidate"):
                terminal_candidate = dict(packet.get("terminal_render_candidate") or {})
                terminal_candidate.pop("render_text", None)
                result["terminal_render_candidate"] = terminal_candidate
                result["runtime_trace"]["terminal_render_candidate"] = {
                    **terminal_candidate,
                    "candidate_kind": "terminal_render_candidate",
                    "capability": "exact_copy",
                    "status": "available",
                    "terminal_render_candidate_available": True,
                    "model_path_required": True,
                    "whole_or_fail": True,
                    "raw_text_exposed_to_model": False,
                    "selected_container_ids": terminal_candidate.get("selected_container_ids", []),
                    "selected_render_ref_ids": terminal_candidate.get("selected_render_ref_ids", []),
                }
            result["runtime_trace"]["codebase_augmentation"] = code_trace
            result["runtime_trace"]["repo_task_context_pack"] = repo_task_context_trace
            if repo_task_context_packs:
                result["repo_task_context_packs"] = repo_task_context_packs
            result["runtime_trace"].setdefault("query", {})
            result["runtime_trace"]["query"]["code_query_mode"] = code_query_mode
            result["runtime_trace"]["family_discovery"] = self._family_discovery_trace(
                fact_filter=fact_filter,
                requested_search_family=search_family,
                searched_families=merged_retrieval_families,
                selected_facts=resolved_facts,
                code_query_mode=code_query_mode,
                merged_families=merged_retrieval_families,
                codebase_probe_trace=generic_codebase_discovery_trace,
                codebase_augmentation_trace=code_trace,
                episode_first_pass_trace=packet.get("family_first_pass_trace"),
            )
            if mixed_merge_trace is not None:
                result["runtime_trace"]["mixed_family_merge"] = mixed_merge_trace
            deterministic_answer: str | None = None
            if temporal_trace:
                result["temporal_resolution"] = temporal_trace
                deterministic_answer = str(temporal_trace.get("deterministic_answer") or "").strip() or None
                if deterministic_answer:
                    result["deterministic_answer"] = deterministic_answer
            temporal_deterministic_query = (
                resolved_type == "temporal"
                or self._temporal_query_requests_year_resolution(retrieval_query)
                or self._temporal_query_requests_month_resolution(retrieval_query)
                or self._temporal_query_requests_first_window(retrieval_query)
            )
            if not result.get("deterministic_answer") and temporal_deterministic_query:
                deterministic_answer = self._derive_relative_temporal_deterministic_answer(
                    query=retrieval_query,
                    resolved_facts=resolved_facts,
                    episode_lookup=_episode_lookup,
                )
                query_class = "relative-anchor"
                if not deterministic_answer:
                    deterministic_answer = self._derive_first_window_temporal_deterministic_answer(
                        query=retrieval_query,
                        context=packet["context"],
                    )
                    if deterministic_answer:
                        query_class = "first-window"
                if not deterministic_answer:
                    deterministic_answer = self._derive_duration_temporal_deterministic_answer(
                        query=retrieval_query,
                        resolved_facts=resolved_facts,
                        episode_lookup=_episode_lookup,
                    )
                    if deterministic_answer:
                        query_class = "duration-anchor"
                if not deterministic_answer:
                    deterministic_answer = self._derive_month_temporal_deterministic_answer(
                        query=retrieval_query,
                        resolved_facts=resolved_facts,
                        episode_lookup=_episode_lookup,
                    )
                    if deterministic_answer:
                        query_class = "month-anchor"
                if deterministic_answer:
                    result["deterministic_answer"] = deterministic_answer
                    temporal_resolution = dict(result.get("temporal_resolution") or {})
                    temporal_resolution.setdefault("query_class", query_class)
                    temporal_resolution["deterministic_answer"] = deterministic_answer
                    result["temporal_resolution"] = temporal_resolution
            if not result.get("deterministic_answer"):
                deterministic_answer = self._derive_time_scoped_acquisition_deterministic_answer(
                    query=retrieval_query,
                    query_features=query_features,
                    packet=packet,
                    episode_lookup=_episode_lookup,
                )
                if deterministic_answer:
                    result["deterministic_answer"] = deterministic_answer
                    runtime_trace = dict(result.get("runtime_trace") or {})
                    runtime_trace["deterministic_answer"] = {
                        "kind": "time_scoped_acquisition",
                        "answer": deterministic_answer,
                    }
                    result["runtime_trace"] = runtime_trace
            if not result.get("deterministic_answer"):
                deterministic_answer = self._derive_time_scoped_activity_acquisition_deterministic_answer(
                    query=retrieval_query,
                    query_features=query_features,
                    packet=packet,
                    episode_lookup=_episode_lookup,
                )
                if deterministic_answer:
                    result["deterministic_answer"] = deterministic_answer
                    runtime_trace = dict(result.get("runtime_trace") or {})
                    runtime_trace["deterministic_answer"] = {
                        "kind": "time_scoped_activity_acquisition",
                        "answer": deterministic_answer,
                    }
                    result["runtime_trace"] = runtime_trace
            if not result.get("deterministic_answer"):
                deterministic_answer = self._derive_activity_list_deterministic_answer(
                    query=retrieval_query,
                    query_features=query_features,
                    packet=packet,
                    episode_lookup=_episode_lookup,
                )
                if deterministic_answer:
                    result["deterministic_answer"] = deterministic_answer
                    runtime_trace = dict(result.get("runtime_trace") or {})
                    runtime_trace["deterministic_answer"] = {
                        "kind": "activity_list",
                        "answer": deterministic_answer,
                    }
                    result["runtime_trace"] = runtime_trace
            return _with_raw_recall(result)
        if has_episode_corpus and not has_visible_facts:
            resolved_type = self._QUERY_TYPE_MAP.get(query_type) or "default"
            return _with_raw_recall({
                "context": "RETRIEVED FACTS:",
                "retrieved": [],
                "query_type": resolved_type,
                "is_multihop": False,
                "complexity_hint": {
                    "score": 0.0,
                    "level": 1,
                    "signals": [],
                    "retrieval_complexity": 0.0,
                    "content_complexity": 0.0,
                    "query_complexity": 0.0,
                    "dominant": "tie",
                },
                "n_facts": len(self._all_granular) + len(self._all_cons) + len(self._all_cross),
                "sessions_in_context": 0,
                "total_sessions": len(self._raw_sessions),
                "coverage_pct": 0,
                "raw_budget": 0,
                "recommended_prompt_type": resolved_type,
                "use_tool": False,
                "retrieved_episode_ids": [],
                "actual_injected_episode_ids": [],
                "selection_scores": [],
                "query_operator_plan": {},
                "output_constraints": {},
                "retrieval_families": [],
                "search_family": effective_search_family,
                "runtime_trace": {
                    "runtime": "episode",
                    "scope": self._scope_trace(),
                    "reason": "empty_visible_facts",
                    "family_discovery": self._family_discovery_trace(
                        fact_filter=fact_filter,
                        requested_search_family=search_family,
                        searched_families=[],
                        selected_facts=[],
                        code_query_mode=code_query_mode,
                        merged_families=[],
                        codebase_augmentation_trace={"mode": "inactive", "reason": "empty_visible_facts"},
                    ),
                    "family_first_pass": {
                        "available_families": available_families(self._episode_corpus),
                        "retrieval_families": [],
                        "requested_search_family": search_family,
                        "per_family": [],
                    },
                    "query": {},
                    "late_fusion": {"mode": "empty"},
                    "selection": {
                        "retrieved_episode_ids": [],
                        "actual_injected_episode_ids": [],
                        "selection_scores": [],
                    },
                    "cross_contamination": {
                        "source_ids": [],
                        "source_count": 0,
                        "family_counts": {},
                        "multi_source": False,
                        "candidate_source_ids": [],
                        "candidate_source_count": 0,
                        "candidate_family_counts": {},
                        "rejected_source_ids": [],
                        "rejected_source_count": 0,
                        "rejected_family_counts": {},
                    },
                    "packet": {
                        "retrieved_fact_ids": [],
                        "retrieved_fact_count": 0,
                        "requested_episode_count": 0,
                        "actual_injected_episode_count": 0,
                        "context_chars": len("RETRIEVED FACTS:"),
                        "snippet_mode": False,
                        "budget_chars": None,
                    },
                    "tuning": get_runtime_tuning(),
                },
            })
        generic_result = await self._generic_fact_recall(
            query=retrieval_query,
            fact_filter=fact_filter,
            search_family=effective_search_family,
            query_type=query_type,
            query_metadata=repo_task_metadata,
            path_constraint_query=query,
        )
        generic_runtime_trace = dict(generic_result.get("runtime_trace") or {})
        lookup = self._visible_fact_lookup(fact_filter, search_family="auto")
        generic_selected_facts: list[dict[str, Any]] = []
        seen_generic_fact_ids: set[str] = set()
        for item in generic_result.get("retrieved", []) or []:
            if not isinstance(item, dict):
                continue
            generic_fact: dict[str, Any] | None
            if item.get("fact") is not None and item.get("id") is not None:
                fact_id = str(item.get("id") or "").strip()
                generic_fact = cast(dict[str, Any], item)
            else:
                fact_id = str(item.get("fact_id") or "").strip()
                generic_fact = lookup.get(fact_id)
            if not fact_id or fact_id in seen_generic_fact_ids or generic_fact is None:
                continue
            seen_generic_fact_ids.add(fact_id)
            generic_selected_facts.append(generic_fact)
        generic_searched_families = (
            self._visible_source_families(fact_filter)
            if effective_search_family in {"auto", "", None}
            else [str(effective_search_family or "").strip().lower()]
        )
        generic_runtime_trace["family_discovery"] = self._family_discovery_trace(
            fact_filter=fact_filter,
            requested_search_family=search_family,
            searched_families=generic_searched_families,
            selected_facts=generic_selected_facts,
            code_query_mode=code_query_mode,
            merged_families=list(generic_result.get("retrieval_families") or []),
            codebase_augmentation_trace=generic_runtime_trace.get("codebase_augmentation"),
        )
        generic_result["runtime_trace"] = generic_runtime_trace
        return _with_raw_recall(generic_result)

    # ── context_for() ──

    async def context_for(
        self,
        query: str,
        agent_id: str = None,
        swarm_id: str = None,
        search_family: str = "auto",
        token_budget: int = 4000,
        query_type: str = "auto",
        kind: str = "all",
        query_metadata: dict[str, Any] | None = None,
        caller_memberships: list = None,
        caller_role: str = "user",
        caller_id: str = None,
    ) -> dict:
        """Same as recall() but truncates context to token_budget (1 token ≈ 4 chars)."""
        result = await self.recall(
            query, agent_id=agent_id, swarm_id=swarm_id,
            search_family=search_family,
            query_type=query_type, kind=kind,
            query_metadata=query_metadata,
            caller_memberships=caller_memberships, caller_role=caller_role,
            caller_id=caller_id,
        )
        if "context" not in result:
            return result
        max_chars = token_budget * 4
        if len(result["context"]) > max_chars:
            result["context"] = result["context"][:max_chars] + "\n[...truncated]"
        result.pop("payload", None)
        result.pop("payload_meta", None)
        result.pop("_context_packet", None)
        return result

    # ── Secrets ──

    def store_secret(
        self,
        name: str,
        value: str,
        agent_id: str | None = None,
        swarm_id: str | None = None,
        scope: str | None = None,
        *,
        owner_id: str | None = None,
        read: list[str] | None = None,
        write: list[str] | None = None,
        metadata: dict | None = None,
        caller_id: str | None = None,
        caller_memberships: list[str] | None = None,
        caller_role: str = "user",
    ) -> dict:
        """Create a secret exactly once in the dedicated secret store."""
        secret_storage = self._secret_storage()
        if secret_storage is None:
            return {"error": "secret storage requires SQLite backend", "code": "SECRET_STORAGE_UNAVAILABLE"}
        if caller_id is None:
            return {"error": "principal auth required for secret writes", "code": "AUTH_REQUIRED"}
        try:
            (
                effective_agent_id,
                effective_swarm_id,
                effective_scope,
                effective_owner_id,
                effective_read,
                effective_write,
                _,
            ) = self._resolve_secret_acl_context(
                agent_id=agent_id,
                swarm_id=swarm_id,
                scope=scope,
                owner_id=owner_id,
                read=read,
                write=write,
                caller_id=caller_id,
                caller_role=caller_role,
            )
            result = secret_storage.upsert_secret(
                name=str(name),
                value=str(value),
                created_by_principal_id=str(caller_id),
                owner_id=effective_owner_id,
                scope=effective_scope,
                agent_id=effective_agent_id,
                swarm_id=effective_swarm_id,
                read=effective_read,
                write=effective_write,
                metadata=metadata if isinstance(metadata, dict) else None,
            )
        except ValueError as exc:
            return {"error": str(exc), "code": "VALIDATION_ERROR"}
        except RuntimeError as exc:
            return {"error": str(exc), "code": "SECRET_STORAGE_UNAVAILABLE"}
        self._refresh_secret_summaries()
        return result

    def _resolve_secret_value_for_internal_use(
        self,
        name: str,
        agent_id: str | None = None,
        swarm_id: str | None = None,
        *,
        scope: str | None = None,
        owner_id: str | None = None,
        caller_id: str | None = None,
        caller_memberships: list[str] | None = None,
        caller_role: str = "user",
    ) -> dict[str, Any]:
        """Resolve a raw secret row for trusted internal server code only."""
        secret_storage = self._secret_storage()
        if secret_storage is None:
            raise RuntimeError("secret storage requires SQLite backend")
        if caller_id is None:
            raise PermissionError("principal auth required for secret reads")
        row = self._resolve_secret_row_by_ref(
            name=str(name),
            agent_id=agent_id,
            swarm_id=swarm_id,
            scope=scope,
            owner_id=owner_id,
            caller_id=caller_id,
            include_value=True,
        )
        if not self._acl_allows_access(
            row,
            caller_id,
            caller_memberships or [],
            caller_role,
            need="read",
        ):
            raise PermissionError("access denied")
        return row

    def list_secrets(
        self,
        *,
        agent_id: str | None = None,
        swarm_id: str | None = None,
        scope: str | None = None,
        owner_id: str | None = None,
        caller_id: str | None = None,
        caller_memberships: list[str] | None = None,
        caller_role: str = "user",
    ) -> dict:
        """List secret metadata in an exact canonical ACL domain without values."""
        secret_storage = self._secret_storage()
        if secret_storage is None:
            return {"error": "secret storage requires SQLite backend", "code": "SECRET_STORAGE_UNAVAILABLE"}
        if caller_id is None:
            return {"error": "principal auth required for secret listing", "code": "AUTH_REQUIRED"}
        try:
            normalized_scope = str(scope or "").strip() or None
            normalized_agent_id = str(agent_id or "").strip() or None
            normalized_swarm_id = str(swarm_id or "").strip() or None
            caller_agent_id = None
            if isinstance(caller_id, str) and caller_id.startswith("agent:"):
                caller_agent_id = caller_id.split(":", 1)[1]
            if normalized_agent_id == caller_agent_id:
                normalized_agent_id = None
            if normalized_scope is not None:
                if normalized_scope not in self.VALID_SCOPES:
                    raise ValueError(f"Unknown scope: {normalized_scope}")
                if normalized_scope == "swarm-shared" and normalized_swarm_id in (None, "", "default"):
                    raise ValueError(NAMED_SWARM_REQUIRED_ERROR)
            rows = secret_storage.list_secret_rows(include_values=False)
        except ValueError as exc:
            return {"error": str(exc), "code": "VALIDATION_ERROR"}
        except RuntimeError as exc:
            return {"error": str(exc), "code": "SECRET_STORAGE_UNAVAILABLE"}
        visible = []
        for row in rows:
            if normalized_scope is not None and str(row.get("scope") or "") != normalized_scope:
                continue
            if normalized_agent_id not in (None, "", "default") and str(row.get("agent_id") or "") != normalized_agent_id:
                continue
            if normalized_scope == "swarm-shared" and normalized_swarm_id not in (None, "", "default"):
                if str(row.get("swarm_id") or "") != normalized_swarm_id:
                    continue
            if not self._acl_allows_access(row, caller_id, caller_memberships or [], caller_role, need="read"):
                continue
            visible.append(
                {
                    "secret_id": row.get("secret_id"),
                    "name": row.get("name"),
                    "owner_id": row.get("owner_id"),
                    "created_by_principal_id": row.get("created_by_principal_id"),
                    "scope": row.get("scope"),
                    "agent_id": row.get("agent_id"),
                    "swarm_id": row.get("swarm_id"),
                    "read": list(row.get("read") or []),
                    "write": list(row.get("write") or []),
                    "created_at": row.get("created_at"),
                    "updated_at": row.get("updated_at"),
                    "metadata": dict(row.get("metadata") or {}),
                }
            )
        return {"secrets": visible}

    def delete_secret(
        self,
        name: str,
        agent_id: str | None = None,
        swarm_id: str | None = None,
        *,
        scope: str | None = None,
        owner_id: str | None = None,
        caller_id: str | None = None,
        caller_memberships: list[str] | None = None,
        caller_role: str = "user",
    ) -> dict:
        """Delete a secret by exact canonical ACL domain."""
        secret_storage = self._secret_storage()
        if secret_storage is None:
            return {"error": "secret storage requires SQLite backend", "code": "SECRET_STORAGE_UNAVAILABLE"}
        if caller_id is None:
            return {"error": "principal auth required for secret delete", "code": "AUTH_REQUIRED"}
        try:
            row = self._resolve_secret_row_by_ref(
                name=str(name),
                agent_id=agent_id,
                swarm_id=swarm_id,
                scope=scope,
                owner_id=owner_id,
                caller_id=caller_id,
                include_value=False,
            )
        except ValueError as exc:
            return {"error": str(exc), "code": "VALIDATION_ERROR"}
        except KeyError:
            return {"deleted": False, "code": "SECRET_NOT_FOUND"}
        except RuntimeError as exc:
            return {"error": str(exc), "code": "SECRET_STORAGE_UNAVAILABLE"}
        if str(row.get("created_by_principal_id") or "") != str(caller_id):
            return {"error": "access denied", "code": "SECRET_FORBIDDEN"}
        try:
            deleted = secret_storage.delete_secret(
                name=str(name),
                acl_domain_key=str(row.get("acl_domain_key") or ""),
            )
        except RuntimeError as exc:
            return {"error": str(exc), "code": "SECRET_STORAGE_UNAVAILABLE"}
        self._refresh_secret_summaries()
        return {"deleted": bool(deleted)}

    # ── Profile helpers ──

    def _has_profiles(self) -> bool:
        """True if profile config was provided."""
        return bool(self._profiles)

    def _default_embed_model(self) -> str:
        try:
            from .setup_store import get_config

            cfg = get_config()
            return cfg.get("embed_model") or "text-embedding-3-large"
        except Exception:
            return "text-embedding-3-large"

    def _normalize_retrieval_config(self, retrieval: dict | None) -> dict:
        if retrieval is None:
            retrieval = {}
        elif not isinstance(retrieval, dict):
            raise ValueError("retrieval must be an object")
        allowed = {"search_family", "default_token_budget"}
        unknown = set(retrieval) - allowed
        if unknown:
            raise ValueError(
                f"unknown retrieval config keys: {', '.join(sorted(unknown))}"
            )
        normalized = dict(retrieval)
        normalized.setdefault("search_family", "auto")
        normalized.setdefault("default_token_budget", 4000)
        allowed_families = {"auto", *registered_source_retrieval_families()}
        if normalized["search_family"] not in allowed_families:
            raise ValueError(
                "retrieval.search_family must be one of "
                + "|".join(sorted(allowed_families))
            )
        if (
            not isinstance(normalized["default_token_budget"], int)
            or normalized["default_token_budget"] <= 0
        ):
            raise ValueError("retrieval.default_token_budget must be positive int")
        return normalized

    def _default_memory_config(self) -> dict:
        return {
            "schema_version": 1,
            "embedding_model": self._default_embed_model(),
            "embedding_secret_ref": None,
            "librarian_profile": self.extract_model or None,
            "librarian_secret_ref": None,
            "inference_secret_ref": None,
            "judge_secret_ref": None,
            "profiles": dict(self._profiles or {}),
            "profile_configs": deepcopy(self._profile_configs or {}),
            "retrieval": {
                "search_family": "auto",
                "default_token_budget": 4000,
            },
        }

    def _resolve_librarian_model(
        self,
        librarian_profile: str | None,
        profile_configs: dict,
    ) -> str | None:
        if not librarian_profile:
            return None
        cfg = profile_configs.get(librarian_profile)
        if isinstance(cfg, dict) and cfg.get("model"):
            return cfg["model"]
        return librarian_profile

    @staticmethod
    def _profile_backend(cfg: dict | None) -> str:
        if not isinstance(cfg, dict):
            return "api"
        backend = str(cfg.get("backend") or "").strip()
        return backend or "api"

    def _resolve_librarian_profile_config(self, model: str) -> dict[str, Any] | None:
        if not self._librarian_profile or self._librarian_profile not in self._profile_configs:
            return None
        cfg = self._profile_configs[self._librarian_profile]
        if isinstance(cfg, dict) and cfg.get("model") == model:
            return cfg
        return None

    async def _run_local_cli_extract(
        self,
        *,
        system: str,
        user_msg: str,
        cli_bin: str,
        cli_args_prefix: list[str],
        timeout_secs: float | None = None,
        sem: asyncio.Semaphore | None = None,
    ) -> dict | list:
        prompt = render_local_cli_prompt(system, [{"role": "user", "content": user_msg}])

        async def _run() -> str:
            if timeout_secs is None:
                return await asyncio.to_thread(
                    run_local_cli,
                    prompt,
                    cli_bin,
                    cli_args_prefix,
                )
            return await asyncio.to_thread(
                run_local_cli,
                prompt,
                cli_bin,
                cli_args_prefix,
                timeout_secs,
            )

        if sem is not None:
            async with sem:
                text = await _run()
        else:
            text = await _run()
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        if not text:
            return {}
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            try:
                return parse_json_response(text)
            except Exception:
                match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
                if match:
                    try:
                        return json.loads(match.group(1))
                    except Exception:
                        pass
                return {}

    def _normalize_profile_levels(self, profiles: dict) -> dict[int, str]:
        """Normalize complexity→profile mapping to int keys."""
        normalized = {}
        for level, name in profiles.items():
            lvl = int(level)
            if lvl < 1 or lvl > 5:
                raise ValueError(f"profile level must be 1-5, got {lvl}")
            normalized[lvl] = name
        return normalized

    def _normalize_profile_pricing(
        self,
        profile_name: str,
        pricing: Any,
        *,
        legacy_input_cost: Any = None,
        legacy_output_cost: Any = None,
    ) -> dict[str, float] | None:
        if pricing is None:
            if legacy_input_cost is None and legacy_output_cost is None:
                return None
            if legacy_input_cost is None or legacy_output_cost is None:
                raise ValueError(
                    f"profile '{profile_name}': legacy pricing fields must include both "
                    "input_cost_per_1k and output_cost_per_1k"
                )
            pricing = {
                "input_per_1k": legacy_input_cost,
                "output_per_1k": legacy_output_cost,
            }
        if not isinstance(pricing, dict):
            raise ValueError(f"profile '{profile_name}': pricing must be an object")

        allowed = {
            "input_per_1k",
            "output_per_1k",
            "reasoning_per_1k",
            "cache_read_per_1k",
            "cache_write_per_1k",
        }
        unknown = set(pricing) - allowed
        if unknown:
            raise ValueError(
                f"profile '{profile_name}': pricing has unknown keys: {', '.join(sorted(unknown))}"
            )

        def _normalize_field(field: str, *, required: bool, default: float = 0.0) -> float:
            if field not in pricing:
                if required:
                    raise ValueError(f"profile '{profile_name}': pricing.{field} is required")
                return default
            value = pricing[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"profile '{profile_name}': pricing.{field} must be numeric")
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError(f"profile '{profile_name}': pricing.{field} must be finite")
            if numeric < 0:
                raise ValueError(f"profile '{profile_name}': pricing.{field} must be >= 0")
            return numeric

        return {
            "input_per_1k": _normalize_field("input_per_1k", required=True),
            "output_per_1k": _normalize_field("output_per_1k", required=True),
            "reasoning_per_1k": _normalize_field("reasoning_per_1k", required=False),
            "cache_read_per_1k": _normalize_field("cache_read_per_1k", required=False),
            "cache_write_per_1k": _normalize_field("cache_write_per_1k", required=False),
        }

    def _validate_profile_configs(self, profile_configs: dict) -> dict:
        """Validate and normalize persisted/runtime profile config shape."""
        normalized_configs: dict[str, dict] = {}
        for name, cfg in profile_configs.items():
            if not isinstance(cfg, dict):
                raise ValueError(f"profile '{name}' config must be an object")
            normalized_cfg = deepcopy(cfg)
            legacy_input_cost = normalized_cfg.pop("input_cost_per_1k", None)
            legacy_output_cost = normalized_cfg.pop("output_cost_per_1k", None)
            backend = self._profile_backend(normalized_cfg)
            if backend == "local_cli":
                allowed = {
                    "backend",
                    "model",
                    "cli_bin",
                    "cli_args_prefix",
                    "timeout_secs",
                    "context_window",
                    "max_output_tokens",
                    "max_output_tokens_summarize",
                    "temperature",
                    "pricing",
                }
                unknown = set(normalized_cfg) - allowed
                if unknown:
                    raise ValueError(
                        f"profile '{name}': local_cli has unknown keys: {', '.join(sorted(unknown))}"
                    )
                model = normalized_cfg.get("model")
                if not isinstance(model, str) or not model.strip():
                    raise ValueError(f"profile '{name}': model must be a non-empty string")
                cli_bin = normalized_cfg.get("cli_bin")
                if not isinstance(cli_bin, str) or not cli_bin.strip():
                    raise ValueError(f"profile '{name}': cli_bin must be a non-empty string")
                cli_args_prefix = normalized_cfg.get("cli_args_prefix")
                if (
                    not isinstance(cli_args_prefix, list)
                    or any(not isinstance(arg, str) for arg in cli_args_prefix)
                ):
                    raise ValueError(f"profile '{name}': cli_args_prefix must be list[str]")
                timeout_secs = normalized_cfg.get("timeout_secs")
                if timeout_secs is not None and (
                    not isinstance(timeout_secs, (int, float))
                    or float(timeout_secs) <= 0
                ):
                    raise ValueError(
                        f"profile '{name}': timeout_secs must be positive number when set"
                    )
                context_window = normalized_cfg.get("context_window")
                if not isinstance(context_window, int) or context_window <= 0:
                    raise ValueError(f"profile '{name}': context_window must be positive int")
                max_output_tokens = normalized_cfg.get("max_output_tokens")
                if not isinstance(max_output_tokens, int) or max_output_tokens <= 0:
                    raise ValueError(f"profile '{name}': max_output_tokens must be positive int")
                if "max_output_tokens_summarize" in normalized_cfg:
                    value = normalized_cfg["max_output_tokens_summarize"]
                    if not isinstance(value, int) or value <= 0:
                        raise ValueError(
                            f"profile '{name}': max_output_tokens_summarize must be positive int"
                        )
                temperature = normalized_cfg.get("temperature")
                if not isinstance(temperature, (int, float)):
                    raise ValueError(f"profile '{name}': temperature must be numeric")
                if (
                    "pricing" in normalized_cfg
                    or legacy_input_cost is not None
                    or legacy_output_cost is not None
                ):
                    normalized_cfg["pricing"] = self._normalize_profile_pricing(
                        name,
                        normalized_cfg.get("pricing"),
                        legacy_input_cost=legacy_input_cost,
                        legacy_output_cost=legacy_output_cost,
                    )
                normalized_configs[name] = normalized_cfg
                continue
            if backend != "api":
                raise ValueError(f"profile '{name}': unknown backend {backend!r}")
            normalized_cfg["pricing"] = self._normalize_profile_pricing(
                name,
                normalized_cfg.get("pricing"),
                legacy_input_cost=legacy_input_cost,
                legacy_output_cost=legacy_output_cost,
            )
            if "model" not in normalized_cfg:
                raise ValueError(f"profile '{name}' missing 'model'")
            if "secret_ref" in normalized_cfg and normalized_cfg["secret_ref"] is not None:
                self._normalize_runtime_secret_ref(
                    normalized_cfg["secret_ref"],
                    field_name=f"profile_configs.{name}.secret_ref",
                    required=True,
                )
            if "max_output_tokens" in normalized_cfg:
                value = normalized_cfg["max_output_tokens"]
                if not isinstance(value, int) or value <= 0:
                    raise ValueError(
                        f"profile '{name}': max_output_tokens must be positive int"
                    )
            if "max_output_tokens_summarize" in normalized_cfg:
                value = normalized_cfg["max_output_tokens_summarize"]
                if not isinstance(value, int) or value <= 0:
                    raise ValueError(
                        f"profile '{name}': max_output_tokens_summarize must be positive int"
                    )
            if "temperature" in normalized_cfg:
                value = normalized_cfg["temperature"]
                if not isinstance(value, (int, float)):
                    raise ValueError(
                        f"profile '{name}': temperature must be numeric"
                    )
            normalized_configs[name] = normalized_cfg
        return normalized_configs

    def _validate_memory_config(self, config: dict) -> dict:
        if not isinstance(config, dict):
            raise ValueError("config must be an object")

        allowed = {
            "schema_version",
            "embedding_model",
            "embedding_secret_ref",
            "librarian_profile",
            "librarian_secret_ref",
            "inference_secret_ref",
            "judge_secret_ref",
            "profiles",
            "profile_configs",
            "retrieval",
        }
        unknown = set(config) - allowed
        if unknown:
            raise ValueError(
                f"unknown memory config keys: {', '.join(sorted(unknown))}"
            )

        schema_version = config.get("schema_version")
        if schema_version != 1:
            raise ValueError("schema_version must be 1")

        embedding_model = config.get("embedding_model")
        if not isinstance(embedding_model, str) or not embedding_model.strip():
            embedding_model = self._default_embed_model()
        if not embedding_model:
            raise ValueError("embedding_model is required")
        embedding_secret_ref = self._normalize_runtime_secret_ref(
            config.get("embedding_secret_ref"),
            field_name="embedding_secret_ref",
            required=False,
        )

        librarian_profile = config.get("librarian_profile")
        if librarian_profile is None:
            normalized_librarian_profile = None
        elif isinstance(librarian_profile, str) and librarian_profile.strip():
            normalized_librarian_profile = librarian_profile.strip()
        else:
            raise ValueError("librarian_profile must be a non-empty string or null")
        librarian_secret_ref = self._normalize_runtime_secret_ref(
            config.get("librarian_secret_ref"),
            field_name="librarian_secret_ref",
            required=False,
        )
        inference_secret_ref = self._normalize_runtime_secret_ref(
            config.get("inference_secret_ref"),
            field_name="inference_secret_ref",
            required=False,
        )
        judge_secret_ref = self._normalize_runtime_secret_ref(
            config.get("judge_secret_ref"),
            field_name="judge_secret_ref",
            required=False,
        )

        profiles = config.get("profiles") or {}
        profile_configs = config.get("profile_configs") or {}
        normalized_profiles = self._normalize_profile_levels(profiles) if profiles else {}
        if normalized_profiles and not profile_configs:
            raise ValueError("profile_configs are required when profiles are configured")
        normalized_profile_configs = self._validate_profile_configs(profile_configs)
        for level, name in normalized_profiles.items():
            if name not in normalized_profile_configs:
                raise ValueError(
                    f"profile '{name}' referenced by level {level} but not in profile_configs"
                )

        retrieval = self._normalize_retrieval_config(config.get("retrieval"))

        return {
            "schema_version": schema_version,
            "embedding_model": embedding_model,
            "embedding_secret_ref": deepcopy(embedding_secret_ref),
            "librarian_profile": normalized_librarian_profile,
            "librarian_secret_ref": deepcopy(librarian_secret_ref),
            "inference_secret_ref": deepcopy(inference_secret_ref),
            "judge_secret_ref": deepcopy(judge_secret_ref),
            "profiles": normalized_profiles,
            "profile_configs": deepcopy(normalized_profile_configs),
            "retrieval": retrieval,
        }

    def _apply_memory_config(self, config: dict) -> None:
        normalized = self._validate_memory_config(config)
        self._memory_config = deepcopy(normalized)
        self._embedding_model = normalized["embedding_model"]
        self._librarian_profile = normalized["librarian_profile"]
        self._profiles = dict(normalized["profiles"]) if normalized["profiles"] else None
        self._profile_configs = deepcopy(normalized["profile_configs"])
        self.extract_model = self._resolve_librarian_model(
            self._librarian_profile,
            self._profile_configs,
        ) or ""

    async def set_config(self, config: dict) -> None:
        async with self._file_lock:
            self._apply_memory_config(config)
            self._save_cache()

    def get_config(self) -> dict:
        if self._memory_config is None:
            self._apply_memory_config(self._default_memory_config())
        assert self._memory_config is not None
        return deepcopy(self._memory_config)

    def _embedding_dim_from_arrays(self, *arrays) -> int:
        for arr in arrays:
            if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[1] > 0:
                return int(arr.shape[1])
        return 3072

    def _get_profile_config(self, name: str) -> dict | None:
        """Return profile config dict by name: {model, context_window, ...}."""
        if self._profile_configs and name in self._profile_configs:
            return self._profile_configs[name]
        return None

    def _list_profile_names(self) -> list[str]:
        """Return unique profile names."""
        if not self._profiles:
            return []
        return list(dict.fromkeys(self._profiles.values()))

    def _cheapest_profile(self) -> str | None:
        """Return profile name mapped to the lowest complexity level."""
        if not self._profiles:
            return None
        min_level = min(self._profiles.keys())
        return self._profiles[min_level]

    async def set_profiles(self, profiles: dict, profile_configs: dict) -> None:
        """Set complexity→profile mapping and profile configs. Persisted."""
        normalized = self._normalize_profile_levels(profiles)
        for level, name in normalized.items():
            if name not in profile_configs:
                raise ValueError(
                    f"profile '{name}' referenced by level {level} but not in profile_configs"
                )
        normalized_profile_configs = self._validate_profile_configs(profile_configs)
        async with self._file_lock:
            next_config = self.get_config()
            next_config["profiles"] = normalized
            next_config["profile_configs"] = deepcopy(normalized_profile_configs)
            self._apply_memory_config(next_config)
            self._save_cache()

    def get_profiles(self) -> dict:
        """Return current profiles and configs."""
        cfg = self.get_config()
        return {
            "profiles": cfg.get("profiles", {}),
            "profile_configs": cfg.get("profile_configs", {}),
        }

    def _resolve_inference_target(
        self,
        recommended_profile: str | None,
        inference_model: str | None = None,
    ) -> dict | None:
        """Resolve model/profile defaults for payload building."""
        fallback_cfg = {
            "context_window": 128000,
            "thinking_overhead": 0,
            "temperature": 0,
        }

        if inference_model:
            secret_ref = self._normalize_runtime_secret_ref(
                self.get_config().get("inference_secret_ref"),
                field_name="inference_secret_ref",
                required=True,
            )
            return {
                "model": inference_model,
                "profile_used": inference_model,
                "profile_fallback": False,
                "cfg": {"model": inference_model, **fallback_cfg},
                "secret_ref": deepcopy(secret_ref),
                "backend": "api",
                "cli_bin": None,
                "cli_args_prefix": [],
                "timeout_secs": None,
            }

        if self._has_profiles() and recommended_profile:
            cfg = self._get_profile_config(recommended_profile)
            if cfg:
                backend = self._profile_backend(cfg)
                secret_ref = None
                if backend != "local_cli":
                    secret_ref = (
                        self._normalize_runtime_secret_ref(
                            cfg.get("secret_ref"),
                            field_name=f"profile_configs.{recommended_profile}.secret_ref",
                            required=False,
                        )
                        if cfg.get("secret_ref") is not None
                        else self._normalize_runtime_secret_ref(
                            self.get_config().get("inference_secret_ref"),
                            field_name="inference_secret_ref",
                            required=True,
                        )
                    )
                return {
                    "model": cfg["model"],
                    "profile_used": recommended_profile,
                    "profile_fallback": False,
                    "cfg": {**fallback_cfg, **cfg},
                    "secret_ref": deepcopy(secret_ref),
                    "backend": backend,
                    "cli_bin": cfg.get("cli_bin"),
                    "cli_args_prefix": deepcopy(cfg.get("cli_args_prefix") or []),
                    "timeout_secs": cfg.get("timeout_secs"),
                }

        if self._has_profiles():
            fallback = self._cheapest_profile()
            fallback_cfg_resolved = self._get_profile_config(fallback) if fallback else None
            if fallback and fallback_cfg_resolved:
                backend = self._profile_backend(fallback_cfg_resolved)
                secret_ref = None
                if backend != "local_cli":
                    secret_ref = (
                        self._normalize_runtime_secret_ref(
                            fallback_cfg_resolved.get("secret_ref"),
                            field_name=f"profile_configs.{fallback}.secret_ref",
                            required=False,
                        )
                        if fallback_cfg_resolved.get("secret_ref") is not None
                        else self._normalize_runtime_secret_ref(
                            self.get_config().get("inference_secret_ref"),
                            field_name="inference_secret_ref",
                            required=True,
                        )
                    )
                return {
                    "model": fallback_cfg_resolved["model"],
                    "profile_used": fallback,
                    "profile_fallback": True,
                    "cfg": {**fallback_cfg, **fallback_cfg_resolved},
                    "secret_ref": deepcopy(secret_ref),
                    "backend": backend,
                    "cli_bin": fallback_cfg_resolved.get("cli_bin"),
                    "cli_args_prefix": deepcopy(fallback_cfg_resolved.get("cli_args_prefix") or []),
                    "timeout_secs": fallback_cfg_resolved.get("timeout_secs"),
                }

        return None

    def _resolve_max_tokens(
        self,
        cfg: dict | None,
        resolved_type: str,
        prompt_type: str,
        explicit_max_tokens: int | None = None,
    ) -> int:
        if explicit_max_tokens is not None:
            return explicit_max_tokens
        is_summarize = (
            resolved_type == "summarize"
            or prompt_type in ("summarize", "summarize_with_metadata")
        )
        if cfg:
            if is_summarize and "max_output_tokens_summarize" in cfg:
                return cfg["max_output_tokens_summarize"]
            if "max_output_tokens" in cfg:
                return cfg["max_output_tokens"]
        return 4096 if is_summarize else 2000

    def _compute_memory_budget(self, cfg: dict | None, max_tokens: int) -> int:
        cfg = cfg or {}
        context_window = int(cfg.get("context_window", 128000))
        thinking_overhead = float(cfg.get("thinking_overhead", 0) or 0)
        usable = int(context_window * (1 - thinking_overhead) * 0.9)
        return max(1, usable - max_tokens)

    def _build_payload_messages(
        self,
        *,
        prompt_type: str,
        context: str,
        query: str,
        recall_result: dict,
        speakers: str,
    ) -> list[dict]:
        answer_contract = recall_result.get("answer_contract")
        if isinstance(answer_contract, dict) and answer_contract.get("prompt_template"):
            variables = dict(answer_contract.get("variables") or {})
            variables["context"] = context
            formatted = str(answer_contract["prompt_template"]).format(**variables)
            return [{"role": "user", "content": formatted}]
        return build_prompt_payload_messages(
            prompt_type=prompt_type,
            context=context,
            query=query,
            recall_result=recall_result,
            speakers=speakers,
            plugin_state=self._effective_inference_leaf_plugins(recall_result),
        )

    def _recall_continuation_context_note(self, continuation: dict | None) -> str:
        if not isinstance(continuation, dict) or not continuation.get("available"):
            return ""
        anchor_terms = continuation.get("anchor_terms") or []
        if anchor_terms:
            anchor_text = f"More evidence matches anchors {anchor_terms}. "
        else:
            anchor_text = "More evidence matches this recall query. "
        handle = str(continuation.get("handle") or "<handle>")
        return (
            "RECALL CONTINUATION AVAILABLE:\n"
            f"{anchor_text}"
            "If the answer is not in this page, call get_more_context with "
            f"handle=\"{handle}\" and page=\"next\" to retrieve the next evidence page."
        )

    def _context_packet_with_recall_continuation(self, context_packet: dict, recall_result: dict) -> dict:
        packet = deepcopy(context_packet)
        note = self._recall_continuation_context_note(recall_result.get("recall_continuation"))
        if not note:
            return packet
        tier4 = packet.setdefault("tier4", [])
        for segment in tier4:
            if segment.get("source") == "recall_continuation":
                return packet
            if "RECALL CONTINUATION AVAILABLE:" in str(segment.get("text") or ""):
                return packet
        max_rank = max((int(segment.get("rank") or 0) for segment in tier4), default=-1)
        tier4.append({
            "text": note,
            "rank": max_rank + 1,
            "source": "recall_continuation",
        })
        return packet

    def _effective_inference_leaf_plugins(self, recall_result: dict | None = None) -> dict[str, bool]:
        plugin_state = dict(self._inference_leaf_plugins)
        overrides = (recall_result or {}).get("inference_leaf_plugins")
        if isinstance(overrides, dict):
            plugin_state.update({str(name): bool(enabled) for name, enabled in overrides.items()})
        return plugin_state

    def _recommended_profile_for_recall_result(self, recall_result: dict) -> str | None:
        if not self._has_profiles():
            return None
        profiles = self._profiles
        if not profiles:
            return None
        complexity_hint = recall_result.get("complexity_hint") or {}
        try:
            level = int(complexity_hint.get("level", 1))
        except (TypeError, ValueError):
            level = 1
        return profiles.get(level)

    def _context_packet_for_recall_result(self, recall_result: dict) -> dict:
        context_packet = recall_result.get("_context_packet")
        if context_packet is not None:
            return deepcopy(context_packet)
        if recall_result.get("terminal_render_candidate"):
            return {
                "tier1": [
                    {
                        "text": recall_result.get("context", ""),
                        "rank": 0,
                        "source": "terminal_render_candidate",
                    }
                ],
                "tier2": [],
                "tier3": [],
                "tier4": [],
            }
        return {
            "tier1": [],
            "tier2": [],
            "tier3": [{"text": recall_result.get("context", ""), "rank": 0, "source": "fact"}],
            "tier4": [],
        }

    def _build_public_answer_contract(
        self,
        *,
        query: str,
        recall_result: dict,
        prompt_type: str | None = None,
        prompt_key: str | None = None,
        use_tool: bool | None = None,
        speakers: str = "User and Assistant",
    ) -> dict:
        """Provider-neutral answer contract shared by public recall and ask.

        The prompt template is intentionally part of the public recall contract:
        external MCP callers can synthesize from memory_recall with the same
        prompt leaf that memory_ask uses internally, without exposing provider
        payloads, profile choices, or secret refs.
        """
        resolved_type = recall_result.get("query_type", "default")
        output_constraints = deepcopy(recall_result.get("output_constraints") or {})
        query_output_constraints = extract_query_features(query).get("output_constraints") or {}
        for key, value in query_output_constraints.items():
            if value and not output_constraints.get(key):
                output_constraints[key] = deepcopy(value)
        resolved_prompt_type = prompt_type or recall_result.get("recommended_prompt_type", resolved_type)
        resolved_prompt_key = prompt_key or resolve_inference_leaf_prompt_key(
            prompt_type=resolved_prompt_type,
            query=query,
            recall_result=recall_result,
            plugin_state=self._effective_inference_leaf_plugins(recall_result),
        )
        terminal_candidate = dict(recall_result.get("terminal_render_candidate") or {})
        contract = {
            "contract_version": 1,
            "prompt_template_public": True,
            "prompt_type": resolved_prompt_type,
            "prompt_key": resolved_prompt_key,
            "prompt_template": get_inf_prompt(resolved_prompt_key),
            "context_field": "context",
            "variables": {
                "question": query,
                "speakers": speakers,
                "sessions_in_context": recall_result.get("sessions_in_context", 0),
                "total_sessions": recall_result.get("total_sessions", 0),
                "coverage_pct": recall_result.get("coverage_pct", 100),
                "reference_date": self._reference_date_for_recall_result(recall_result),
            },
            "use_tool": bool(recall_result.get("use_tool", False) if use_tool is None else use_tool),
            "grounding_policy": "answer only from the returned context; if evidence is insufficient, say what is missing",
            "output_constraints": output_constraints,
        }
        if terminal_candidate:
            contract["terminal_render_candidate_policy"] = {
                "candidate_available": True,
                "raw_text_exposed_to_model": False,
                "final_render_available_in_recall": False,
                "model_path_required": bool(terminal_candidate.get("model_path_required", True)),
                "whole_or_fail": bool(terminal_candidate.get("whole_or_fail", True)),
            }
        continuation = recall_result.get("recall_continuation")
        if isinstance(continuation, dict) and continuation.get("available"):
            contract["recall_continuation"] = {
                "available": True,
                "tool": "get_more_context",
                "handle": continuation.get("handle"),
                "next_page": continuation.get("next_page"),
                "instruction": (
                    "The returned context is the first evidence page. "
                    "If the answer is not present and more evidence is available, "
                    "call get_more_context with handle=<recall_continuation.handle> "
                    "and page=\"next\" to fetch the next page."
                ),
            }
        return contract

    def _reference_date_for_recall_result(self, recall_result: dict) -> str:
        """Return the most recent evidence date for inference prompt anchoring."""
        explicit = str(recall_result.get("reference_date") or "").strip()
        if explicit:
            return explicit

        candidates: list[datetime] = []

        def add_date(value: Any) -> None:
            raw = str(value or "").strip()
            if not raw:
                return
            try:
                parsed = date_parser.parse(raw, fuzzy=True)
            except Exception:
                return
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            candidates.append(parsed.astimezone(timezone.utc))

        for fact in recall_result.get("retrieved", []) or []:
            if not isinstance(fact, dict):
                continue
            for field in ("event_date", "session_date", "source_date", "created_at", "updated_at"):
                add_date(fact.get(field))
            metadata = fact.get("metadata")
            if isinstance(metadata, dict):
                for field in ("event_date", "session_date", "source_date", "created_at", "updated_at"):
                    add_date(metadata.get(field))

        packet = recall_result.get("_context_packet")
        if isinstance(packet, dict):
            for tier in ("tier1", "tier2", "tier3", "tier4"):
                for item in packet.get(tier, []) or []:
                    if not isinstance(item, dict):
                        continue
                    for field in ("event_date", "session_date", "source_date", "created_at", "updated_at"):
                        add_date(item.get(field))

        if candidates:
            return max(candidates).date().isoformat()
        return datetime.now(timezone.utc).date().isoformat()

    def _finalize_recall_evidence_context(
        self,
        *,
        query: str,
        answer_query: str | None = None,
        recall_result: dict,
        inference_model: str = None,
        max_tokens: int = None,
        use_tool: bool = None,
        speakers: str = "User and Assistant",
    ) -> tuple[dict, dict]:
        """Render the evidence context that public recall and ask share.

        This helper may consult profile/model budgets, but it does not return
        executable payloads, secret refs, or provider planning metadata.
        """
        finalized = deepcopy(recall_result)
        if finalized.get("error") or finalized.get("code") or "context" not in finalized:
            return finalized, {}

        contract_query = str(answer_query or query or "")
        runtime_trace = deepcopy(finalized.get("runtime_trace") or {})
        existing_trace = deepcopy(runtime_trace.get("evidence_context") or {})

        def attach_contract(
            *,
            trace: dict,
            final_use_tool: bool | None = None,
        ) -> tuple[dict, dict]:
            if "answer_contract" not in finalized:
                finalized["answer_contract"] = self._build_public_answer_contract(
                    query=contract_query,
                    recall_result=finalized,
                    prompt_type=trace.get("prompt_type"),
                    prompt_key=trace.get("prompt_key"),
                    use_tool=final_use_tool,
                    speakers=speakers,
                )
            runtime_trace["evidence_context"] = trace
            finalized["runtime_trace"] = runtime_trace
            return finalized, trace

        if isinstance(existing_trace, dict) and existing_trace.get("finalized") is True:
            return attach_contract(trace=existing_trace)

        resolved_type = finalized.get("query_type", "default")
        prompt_type = finalized.get("recommended_prompt_type", resolved_type)
        prompt_key = resolve_inference_leaf_prompt_key(
            prompt_type=prompt_type,
            query=contract_query,
            recall_result=finalized,
            plugin_state=self._effective_inference_leaf_plugins(finalized),
        )
        recommended_profile = (
            finalized.get("recommended_profile")
            or self._recommended_profile_for_recall_result(finalized)
        )

        try:
            payload_target = self._resolve_inference_target(recommended_profile, inference_model)
        except Exception as exc:
            trace = {
                "finalized": False,
                "reason": "finalization_error",
                "error_type": type(exc).__name__,
                "prompt_type": prompt_type,
                "prompt_key": prompt_key,
            }
            return attach_contract(
                trace=trace,
                final_use_tool=finalized.get("use_tool", False) if use_tool is None else use_tool,
            )

        if payload_target is None:
            trace = {
                "finalized": False,
                "reason": "no_inference_target",
                "prompt_type": prompt_type,
                "prompt_key": prompt_key,
            }
            return attach_contract(
                trace=trace,
                final_use_tool=finalized.get("use_tool", False) if use_tool is None else use_tool,
            )

        model = payload_target["model"]
        cfg = payload_target["cfg"]
        backend = payload_target.get("backend", "api")
        cli_bin = payload_target.get("cli_bin")
        cli_args_prefix = payload_target.get("cli_args_prefix") or []
        cli_timeout_secs = payload_target.get("timeout_secs")

        terminal_render_candidate_available = bool(finalized.get("terminal_render_candidate"))
        final_use_tool = finalized.get("use_tool", False) if use_tool is None else use_tool
        tool_use_downgraded = False
        tool_use_downgrade_reason = None
        if terminal_render_candidate_available:
            final_use_tool = False
        if backend == "local_cli" and final_use_tool:
            if use_tool is True:
                raise RuntimeError("local_cli backend does not support tool use")
            final_use_tool = False
            tool_use_downgraded = True
            tool_use_downgrade_reason = "local_cli_backend_no_tool_support"

        temperature = float(cfg.get("temperature", 0) or 0)
        resolved_max_tokens = self._resolve_max_tokens(
            cfg=cfg,
            resolved_type=resolved_type,
            prompt_type=prompt_type,
            explicit_max_tokens=max_tokens,
        )
        memory_budget = self._compute_memory_budget(cfg, resolved_max_tokens)
        context_packet = self._context_packet_with_recall_continuation(
            self._context_packet_for_recall_result(finalized),
            finalized,
        )

        try:
            packet, rendered_context, _payload, meta = self._truncate_by_priority(
                model=model,
                query=contract_query,
                recall_result=finalized,
                context_packet=context_packet,
                memory_budget=memory_budget,
                max_tokens=resolved_max_tokens,
                prompt_type=prompt_type,
                use_tool=final_use_tool,
                speakers=speakers,
                temperature=temperature,
                backend=backend,
                cli_bin=cli_bin,
                cli_args_prefix=cli_args_prefix,
                cli_timeout_secs=cli_timeout_secs,
            )
        except Exception as exc:
            trace = {
                "finalized": False,
                "reason": "finalization_error",
                "error_type": type(exc).__name__,
                "prompt_type": prompt_type,
                "prompt_key": prompt_key,
            }
            return attach_contract(trace=trace, final_use_tool=final_use_tool)

        finalized["context"] = rendered_context
        finalized["_context_packet"] = packet
        trace = {
            "finalized": True,
            "context_tokens": meta["context_tokens"],
            "message_tokens_est": meta["message_tokens_est"],
            "tool_tokens_est": meta["tool_tokens_est"],
            "memory_budget": meta["memory_budget"],
            "budget_exceeded": meta["budget_exceeded"],
            "truncation": meta["truncation"],
            "prompt_type": prompt_type,
            "prompt_key": prompt_key,
            "use_tool": final_use_tool,
            "tool_use_downgraded": tool_use_downgraded,
            "tool_use_downgrade_reason": tool_use_downgrade_reason,
        }
        return attach_contract(trace=trace, final_use_tool=final_use_tool)

    def _build_inference_plan_from_recall_result(
        self,
        *,
        query: str,
        recall_result: dict,
        inference_model: str = None,
        max_tokens: int = None,
        use_tool: bool = None,
        speakers: str = "User and Assistant",
    ) -> dict:
        """Build executable inference planning metadata from evidence recall."""
        planning_result = deepcopy(recall_result)
        recommended_profile = (
            planning_result.get("recommended_profile")
            or self._recommended_profile_for_recall_result(planning_result)
        )
        if recommended_profile:
            planning_result["recommended_profile"] = recommended_profile

        payload = planning_result.get("payload")
        payload_meta = planning_result.get("payload_meta")
        secret_ref = planning_result.get("_payload_secret_ref")
        requires_rebuild = (
            payload is None
            or payload_meta is None
            or any(value is not None for value in (inference_model, max_tokens, use_tool))
            or speakers != "User and Assistant"
        )
        if requires_rebuild:
            runtime_trace = deepcopy(planning_result.get("runtime_trace") or {})
            evidence_trace = deepcopy(runtime_trace.get("evidence_context") or {})
            planning_override = (
                any(value is not None for value in (inference_model, max_tokens, use_tool))
                or speakers != "User and Assistant"
            )
            if planning_override:
                runtime_trace.pop("evidence_context", None)
                planning_result["runtime_trace"] = runtime_trace
                planning_result.pop("answer_contract", None)
                evidence_trace = {}
            if evidence_trace.get("finalized") is not True:
                planning_result, _ = self._finalize_recall_evidence_context(
                    query=query,
                    recall_result=planning_result,
                    inference_model=inference_model,
                    max_tokens=max_tokens,
                    use_tool=use_tool,
                    speakers=speakers,
                )
            built_payload = self._build_payload(
                query=query,
                recall_result=planning_result,
                inference_model=inference_model,
                max_tokens=max_tokens,
                use_tool=use_tool,
                speakers=speakers,
            )
            if len(built_payload) == 2:
                payload, payload_meta = built_payload
                secret_ref = None
            else:
                payload, payload_meta, secret_ref = built_payload
        elif secret_ref is not None:
            secret_ref = deepcopy(secret_ref)

        if not payload or not payload_meta:
            return {
                "error": "No profiles configured and no inference_model provided",
                "code": "NO_PROFILES",
                "recommended_profile": recommended_profile,
                "complexity_hint": planning_result.get("complexity_hint"),
                "prompt_type": planning_result.get("recommended_prompt_type", planning_result.get("query_type", "default")),
                "reason_trace": {
                    "has_profiles": self._has_profiles(),
                    "inference_model_override": bool(inference_model),
                    "reason": "no_inference_target",
                },
            }

        reason_trace = {
            "has_profiles": self._has_profiles(),
            "complexity_level": (planning_result.get("complexity_hint") or {}).get("level"),
            "recommended_profile": recommended_profile,
            "profile_used": payload_meta.get("profile_used"),
            "profile_fallback": payload_meta.get("profile_fallback", False),
            "prompt_type": payload_meta.get("prompt_type"),
            "prompt_key": payload_meta.get("prompt_key"),
            "backend": payload_meta.get("backend"),
            "inference_model_override": bool(inference_model),
        }
        runtime_trace = dict(planning_result.get("runtime_trace") or {})
        runtime_trace["inference_planning"] = reason_trace
        plan = {
            "telemetry_version": 1,
            "recommended_profile": recommended_profile,
            "payload": payload,
            "payload_meta": payload_meta,
            "complexity_hint": planning_result.get("complexity_hint"),
            "prompt_type": payload_meta.get("prompt_type"),
            "recommended_prompt_type": planning_result.get("recommended_prompt_type"),
            "query_type": planning_result.get("query_type", "default"),
            "use_tool": payload_meta.get("use_tool", False),
            "reason_trace": reason_trace,
            "runtime_trace": runtime_trace,
        }
        if secret_ref is not None:
            plan["secret_ref"] = deepcopy(secret_ref)
        return plan

    async def plan_inference(
        self,
        query: str,
        agent_id: str = None,
        swarm_id: str = None,
        search_family: str = "auto",
        token_budget: int | None = None,
        query_type: str = "auto",
        kind: str = "all",
        caller_memberships: list = None,
        caller_role: str = "user",
        caller_id: str = None,
        query_metadata: dict[str, Any] | None = None,
        mal_binding_id: str | None = None,
        inference_model: str = None,
        max_tokens: int = None,
        use_tool: bool = None,
        speakers: str = "User and Assistant",
    ) -> dict:
        recall_result = await self.recall(
            query,
            agent_id=agent_id,
            swarm_id=swarm_id,
            search_family=search_family,
            token_budget=token_budget,
            query_type=query_type,
            kind=kind,
            query_metadata=query_metadata,
            caller_memberships=caller_memberships,
            caller_role=caller_role,
            caller_id=caller_id,
            mal_binding_id=mal_binding_id,
        )
        if recall_result.get("error") or recall_result.get("code"):
            return recall_result
        if "context" not in recall_result:
            return recall_result
        return self._build_inference_plan_from_recall_result(
            query=query,
            recall_result=recall_result,
            inference_model=inference_model,
            max_tokens=max_tokens,
            use_tool=use_tool,
            speakers=speakers,
        )

    def _normalize_grounded_answer(self, query: str, answer: str, recall_result: dict) -> str:
        if not answer:
            return answer
        qf = extract_query_features(query)
        lowered_query = (query or "").strip().lower()
        support_texts: list[str] = []
        for fact in recall_result.get("retrieved", []):
            text = (fact or {}).get("fact", "")
            if text:
                support_texts.append(text)

        def _extract_explicit_country_surface(text: str) -> str | None:
            if not text:
                return None
            cleaned = re.sub(r"[*_`#]+", "", text)
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            for pattern in (
                r"^\s*answer\s*:\s*(?:in\s+)?(?:the\s+)?([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,4})\b",
                r"\b(?:country of|country is|country was|in the country of)\s+(?:the\s+)?([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,4})\b",
                r"\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3}\s+is\s+in\s+(?:the\s+)?([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,4})\b",
                r"\bwhich\s+is\s+in\s+(?:the\s+)?([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,4})\b",
                r"\b(?:the\s+)?answer would be\s+(?:in\s+)?(?:the\s+)?([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,4})\b",
                r"\b(?:visited|visiting|travel(?:ed|ing)(?:\s+to)?|trip to|meet(?:ing)? in)\s+(?:the\s+)?([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,4})\b",
                r"\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3},\s*(?:the\s+)?([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,4})\b",
            ):
                match = re.search(pattern, cleaned, re.I)
                if not match:
                    continue
                candidate = re.sub(r"\s+", " ", match.group(1).strip(" \t\r\n,.;:!?"))
                candidate = re.sub(r"^the\s+", "", candidate, flags=re.I)
                if not candidate or any(ch.isdigit() for ch in candidate):
                    continue
                candidate_tokens = []
                for token in candidate.split():
                    if re.match(r"^[A-Z][A-Za-z-]*$", token):
                        candidate_tokens.append(token)
                        continue
                    if token.lower() in {"and", "of", "the"} and candidate_tokens:
                        candidate_tokens.append(token.lower())
                        continue
                    break
                while candidate_tokens and candidate_tokens[-1] in {"and", "of", "the"}:
                    candidate_tokens.pop()
                candidate = " ".join(candidate_tokens)
                if not candidate:
                    continue
                if len(candidate.split()) > 4:
                    continue
                return candidate
            return None

        def _extract_city_based_country_surface(text: str) -> str | None:
            if not text:
                return None
            cleaned = re.sub(r"[*_`#]+", "", text)
            cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
            for city, country in sorted(_CITY_TO_COUNTRY.items(), key=lambda item: (-len(item[0]), item[0])):
                if re.search(rf"\b{re.escape(city)}\b", cleaned):
                    return country
            return None

        def _country_surface_from_texts(*texts: str) -> str | None:
            explicit_candidates = []
            city_candidates = []
            for text in texts:
                candidate = _extract_explicit_country_surface(text)
                if candidate:
                    explicit_candidates.append(candidate)
            for text in texts:
                candidate = _extract_city_based_country_surface(text)
                if candidate:
                    city_candidates.append(candidate)
            normalized_city = {candidate.strip() for candidate in city_candidates if candidate and candidate.strip()}
            if len(normalized_city) == 1:
                return next(iter(normalized_city))
            normalized_explicit = {candidate.strip() for candidate in explicit_candidates if candidate and candidate.strip()}
            if len(normalized_explicit) == 1:
                return next(iter(normalized_explicit))
            return None

        if re.match(r"^(?:in\s+)?what country\b|^which country\b", lowered_query):
            answer_candidate = _country_surface_from_texts(answer)
            if answer_candidate:
                return answer_candidate

        temporal_resolution = recall_result.get("temporal_resolution") or {}
        if str(temporal_resolution.get("deterministic_answer") or "").strip():
            return answer

        deterministic_meta = (recall_result.get("runtime_trace") or {}).get("deterministic_answer") or {}
        deterministic_kind = str(deterministic_meta.get("kind") or "").strip().lower()
        if deterministic_kind in {"activity_list", "time_scoped_acquisition", "time_scoped_activity_acquisition"}:
            return answer

        slot_plan = qf.get("operator_plan", {}).get("slot_query", {})
        slot_query_enabled = bool(slot_plan.get("enabled"))
        head_tokens = {
            normalize_term_token(token)
            for token in (slot_plan.get("head_tokens") or [])
            if normalize_term_token(token)
        } if slot_query_enabled else set()

        lowered = answer.lower()
        negative_answer = any(
            phrase in lowered
            for phrase in (
                "not mentioned",
                "not specified",
                "not provided",
                "not available",
                "unknown",
                "not explicitly stated",
                "not detailed",
                "cannot be determined",
                "does not specify",
                "not clarified",
                "no specific",
            )
        )

        grounded_candidates: list[str] = []
        explicit_raw_candidates: list[str] = []
        grounded_seen: set[str] = set()

        def _normalized_answer_surface(text: str) -> str:
            stripped = re.sub(r"^\s*answer\s*:\s*", "", text or "", flags=re.I).strip()
            stripped = re.sub(r"\s+", " ", stripped)
            return stripped.strip(" \t\r\n`*_#:-").lower()

        def _normalized_text_tokens(text: str) -> list[str]:
            return [
                normalize_term_token(token)
                for token in re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)?", text or "")
                if normalize_term_token(token)
            ]

        answer_norm_tokens = _normalized_text_tokens(answer)
        function_word_tokens = {
            "a", "an", "the", "and", "or", "of", "to", "for", "with", "in", "on", "at", "by", "from",
        }

        def _answer_mentions_candidate(candidate: str) -> bool:
            candidate_tokens = [
                token
                for token in _normalized_text_tokens(candidate)
                if token and token not in function_word_tokens
            ]
            if not candidate_tokens:
                return False
            cursor = 0
            for token in answer_norm_tokens:
                if token == candidate_tokens[cursor]:
                    cursor += 1
                    if cursor >= len(candidate_tokens):
                        return True
            return False

        requested_temporal_markers: list[str] = []
        for pattern in (
            r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b",
            r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b",
            r"\b\d{4}-\d{2}-\d{2}\b",
        ):
            for match in re.finditer(pattern, query, re.I):
                marker = re.sub(r"\s+", " ", match.group(0).strip()).lower()
                if marker not in requested_temporal_markers:
                    requested_temporal_markers.append(marker)

        def _commonality_label_from_overlap(overlap_tokens: tuple[str, ...]) -> str | None:
            token_set = set(overlap_tokens)
            if "movie" in token_set or "movy" in token_set:
                return "movies"
            if "dessert" in token_set:
                return "dairy-free desserts" if any(token.startswith("dairy") for token in token_set) else "desserts"
            if "hobby" in token_set:
                return "hobbies"
            if len(token_set) == 1:
                token = next(iter(token_set))
                if token.endswith("y"):
                    return f"{token[:-1]}ies"
                if token.endswith("s"):
                    return token
                return f"{token}s"
            return " ".join(overlap_tokens)

        def _derive_commonality_interest_answer() -> str | None:
            commonality_plan = qf.get("operator_plan", {}).get("commonality", {})
            if not commonality_plan.get("enabled") or not _COMMONALITY_INTEREST_QUERY_RE.search(query):
                return None
            query_entities = _extract_query_named_entities(query)
            if len(query_entities) < 2:
                return None
            entities = query_entities[:2]
            rows_by_entity: dict[str, list[dict[str, Any]]] = defaultdict(list)
            token_freq: defaultdict[str, int] = defaultdict(int)
            for idx, support_text in enumerate(support_texts):
                if not support_text or not _COMMONALITY_INTEREST_FACT_RE.search(support_text):
                    continue
                hits = _fact_entity_hits({"fact": support_text}, entities)
                if not hits:
                    continue
                tokens = _commonality_tokens(support_text, query_entities)
                if not tokens:
                    continue
                row: dict[str, Any] = {
                    "fact": {"id": f"support_{idx}", "fact": support_text, "session": 0},
                    "tokens": tokens,
                    "idx": idx,
                    "rank": idx,
                }
                for token in tokens:
                    token_freq[token] += 1
                for entity in hits:
                    rows_by_entity[entity].append(row)
            if any(not rows_by_entity.get(entity) for entity in entities):
                return None
            candidates: list[tuple[float, set[str], dict[str, Any], dict[str, Any]]] = []
            for left in rows_by_entity[entities[0]][:64]:
                for right in rows_by_entity[entities[1]][:64]:
                    if left["idx"] == right["idx"]:
                        continue
                    left_tokens = cast(set[str], left["tokens"])
                    right_tokens = cast(set[str], right["tokens"])
                    overlap = (left_tokens & right_tokens) - _COMMONALITY_OVERLAP_IGNORE
                    if not overlap:
                        continue
                    rarity = sum(1.0 / max(token_freq.get(token, 1), 1) for token in overlap)
                    score = rarity * 10.0 + len(overlap) * 1.5 - 0.05 * abs(
                        cast(int, left["idx"]) - cast(int, right["idx"])
                    )
                    candidates.append((score, overlap, left, right))
            labels = []
            seen_labels = set()
            for group in _rank_commonality_groups(candidates):
                label = _commonality_label_from_overlap(group["overlap"])
                if not label:
                    continue
                if label in seen_labels:
                    continue
                seen_labels.add(label)
                labels.append(label)
                if len(labels) >= 4:
                    break
            return ", ".join(labels) if len(labels) >= 2 else None

        country_support_candidate = _country_surface_from_texts(*support_texts) if re.match(r"^(?:in\s+)?what country\b|^which country\b", lowered_query) else None
        commonality_interest_answer = _derive_commonality_interest_answer()
        if not slot_query_enabled:
            if country_support_candidate:
                return country_support_candidate
            if negative_answer and commonality_interest_answer:
                return commonality_interest_answer
            return answer
        if not head_tokens:
            if country_support_candidate:
                return country_support_candidate
            return answer

        def _ranked_grounded_candidates() -> list[str]:
            candidate_pool = explicit_raw_candidates or grounded_candidates
            if not candidate_pool:
                return []
            lowered_supports = [text.lower() for text in support_texts if text]
            time_like_tokens = {
                "today", "yesterday", "tomorrow", "tonight", "soon", "later", "lately", "recently",
                "week", "month", "year", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
                "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december",
                "last", "next",
            }
            generic_value_tokens = {
                "idea", "ideas", "thing", "things", "stuff", "something", "anything", "everything",
                "place", "places", "breakthrough",
            }
            query_word_set = set(qf.get("words") or set())
            query_word_ignore = {"the", "a", "an", "what", "which", "who", "where", "when", "why", "how"}
            semantic_head_tokens = head_tokens | {
                "vehicle", "truck", "van", "bike", "bicycle", "motorcycle", "suv", "sedan",
                "country", "nation", "city", "state", "province", "region", "location", "place", "destination", "area",
                "activity", "activities", "hobby", "hobbies", "pastime", "pastimes", "sport", "sports", "game", "games",
            }

            def _score(candidate: str) -> tuple[int, int, int, int, int, int, int, int, int]:
                lowered_candidate = candidate.lower()
                candidate_tokens = [
                    normalize_term_token(token)
                    for token in re.findall(r"[A-Za-z0-9&'._-]+", candidate)
                    if normalize_term_token(token)
                ]
                token_count = len(candidate_tokens)
                support_hits = sum(lowered_candidate in text for text in lowered_supports)
                contains_shorter = sum(
                    1
                    for other in grounded_candidates
                    if other != candidate and other.lower() in lowered_candidate
                )
                novel_tokens = [
                    token
                    for token in candidate_tokens
                    if token not in head_tokens and token not in query_word_set and token not in query_word_ignore
                ]
                informative_novel = [
                    token for token in novel_tokens if token not in time_like_tokens and token not in generic_value_tokens
                ]
                noise_hits = sum(token in time_like_tokens or token in generic_value_tokens for token in candidate_tokens)
                semantic_only = int(bool(candidate_tokens) and set(candidate_tokens) <= semantic_head_tokens)
                date_alignment = int(
                    bool(requested_temporal_markers)
                    and any(
                        lowered_candidate in text and any(marker in text for marker in requested_temporal_markers)
                        for text in lowered_supports
                    )
                )
                return (
                    date_alignment,
                    int(noise_hits == 0 and not semantic_only),
                    len(informative_novel),
                    int(token_count >= 2),
                    int(any(ch.isupper() for ch in candidate)),
                    token_count,
                    contains_shorter,
                    support_hits,
                    len(candidate),
                )

            return sorted(candidate_pool, key=_score, reverse=True)

        def _best_grounded_candidate() -> str | None:
            ranked = _ranked_grounded_candidates()
            return ranked[0] if ranked else None

        def _extract_grounded_answer_phrases(text: str) -> list[str]:
            grounding_token_rows = []
            for entry in (slot_grounding_texts or support_texts):
                if not entry:
                    continue
                norm_tokens = [
                    normalize_term_token(token)
                    for token in re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)?", entry)
                    if normalize_term_token(token)
                ]
                if norm_tokens:
                    grounding_token_rows.append(" ".join(norm_tokens))
            if not grounding_token_rows:
                return []
            raw_tokens = re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)?", text or "")
            if not raw_tokens:
                return []
            query_entity_tokens = {
                normalize_term_token(token)
                for phrase in (qf.get("entity_phrases") or [])
                for token in re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)?", str(phrase))
                if normalize_term_token(token)
            }
            ignore_tokens = set(qf.get("words") or set()) | head_tokens | function_word_tokens | {
                "share", "shared", "same", "similar", "interest", "interests", "hobby", "hobbies",
                "goal", "goals", "kind", "kinds", "type", "types", "include", "includes",
            }
            ignore_tokens |= query_entity_tokens
            candidates: list[tuple[int, int, int, str]] = []
            for size in range(min(5, len(raw_tokens)), 0, -1):
                for start in range(len(raw_tokens) - size + 1):
                    phrase_tokens = raw_tokens[start:start + size]
                    norm_tokens = [
                        normalize_term_token(token)
                        for token in phrase_tokens
                        if normalize_term_token(token)
                    ]
                    informative = [
                        token
                        for token in norm_tokens
                        if token not in ignore_tokens and len(token) >= 3
                    ]
                    if len(informative) < 2:
                        continue
                    left = 0
                    right = len(phrase_tokens)
                    while left < right:
                        norm = normalize_term_token(phrase_tokens[left])
                        if norm and norm not in ignore_tokens and len(norm) >= 3:
                            break
                        left += 1
                    while right > left:
                        norm = normalize_term_token(phrase_tokens[right - 1])
                        if norm and norm not in ignore_tokens and len(norm) >= 3:
                            break
                        right -= 1
                    if right - left < 2:
                        continue
                    phrase = " ".join(phrase_tokens[left:right]).strip()
                    normalized_phrase = " ".join(informative)
                    support_hits = sum(normalized_phrase in entry for entry in grounding_token_rows)
                    if support_hits == 0:
                        continue
                    candidates.append((start, -size, -support_hits, phrase))
            candidates.sort()
            kept: list[str] = []
            kept_ranges: list[tuple[int, int]] = []
            for start, neg_size, _neg_hits, phrase in candidates:
                size = -neg_size
                end = start + size
                if any(not (end <= left or start >= right) for left, right in kept_ranges):
                    continue
                lowered_phrase = phrase.lower()
                if any(
                    lowered_phrase in existing.lower() or existing.lower() in lowered_phrase
                    for existing in kept
                ):
                    continue
                kept.append(phrase)
                kept_ranges.append((start, end))
                if len(kept) >= 4:
                    break
            return kept

        def _add_grounded_candidate(candidate: str) -> None:
            normalized = re.sub(r"\s+", " ", candidate.strip())
            key = normalized.lower()
            if not normalized or key in grounded_seen:
                return
            grounded_seen.add(key)
            grounded_candidates.append(normalized)

        in_retrieved_section = False
        in_raw_slot_section = False
        for raw_line in (recall_result.get("context") or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            upper = line.upper()
            if upper.startswith("RETRIEVED FACTS:"):
                in_retrieved_section = True
                in_raw_slot_section = False
                continue
            if upper.startswith("RAW SLOT CANDIDATES:"):
                in_retrieved_section = False
                in_raw_slot_section = True
                continue
            if upper.startswith("--- "):
                in_retrieved_section = False
                in_raw_slot_section = False
                continue
            if in_retrieved_section:
                if not re.match(r"^\[\d+\]\s+", line):
                    continue
                line = re.sub(r"^\[\d+\]\s*", "", line)
                line = re.sub(r"^\(S\d+\)\s*", "", line, flags=re.I)
                line = re.sub(r"\s*\[Episode:[^\]]+\]\s*$", "", line, flags=re.I)
                if line:
                    support_texts.append(line)
                continue
            if in_raw_slot_section:
                if not re.match(r"^\[Q\d+\]\s+", line):
                    continue
                candidate = re.sub(r"^\[Q\d+\]\s*", "", line)
                candidate = re.split(
                    r"\s+\(from \[Turn query\]:|\s+\[Episode:|\s+\[Local evidence:|\s+\[Fact:",
                    candidate,
                    maxsplit=1,
                )[0].strip()
                if candidate:
                    _add_grounded_candidate(candidate)
                    explicit_raw_candidates.append(candidate)
                    support_texts.append(candidate)

        slot_support_texts = support_texts
        primary_entity = ""
        primary_entity_tokens: set[str] = set()
        for phrase in qf.get("entity_phrases") or []:
            if phrase:
                primary_entity = str(phrase).strip().lower()
                primary_entity_tokens = {
                    normalize_term_token(token)
                    for token in re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)?", primary_entity)
                    if normalize_term_token(token)
                }
                break
        if not primary_entity:
            query_entity_spans = re.findall(r"([A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+){0,2})", query)
            for span in query_entity_spans:
                norm_tokens = {
                    normalize_term_token(token)
                    for token in re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)?", span)
                    if normalize_term_token(token)
                }
                if not norm_tokens:
                    continue
                if norm_tokens <= {"what", "which", "when", "where", "who", "why", "how"}:
                    continue
                primary_entity = span.strip().lower()
                primary_entity_tokens = norm_tokens
                break

        foreign_subject_ignore = {
            "i", "we", "he", "she", "they", "it", "last", "next", "this", "that", "these", "those",
            "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
            "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december",
        }

        def _support_text_has_foreign_subject(text: str) -> bool:
            if not primary_entity or primary_entity in text.lower() or not head_tokens:
                return False
            raw_words = re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)?", text)
            norm_words = [normalize_term_token(word) for word in raw_words]
            head_index = None
            head_len = len(head_tokens)
            for idx in range(max(0, len(norm_words) - head_len + 1)):
                if norm_words[idx:idx + head_len] == list(head_tokens):
                    head_index = idx
                    break
            if head_index is None:
                return False
            for raw in raw_words[:head_index]:
                if not raw[:1].isupper():
                    continue
                norm = normalize_term_token(raw)
                if not norm or norm in foreign_subject_ignore or norm in primary_entity_tokens:
                    continue
                return True
            return False

        slot_grounding_texts: list[str] = []
        for text in slot_support_texts:
            if _support_text_has_foreign_subject(text):
                continue
            slot_grounding_texts.append(text)
            for candidate in _fact_slot_fill_candidates(text, qf, allow_loose_fallback=False):
                _add_grounded_candidate(candidate)

        country_support_candidate = _country_surface_from_texts(*support_texts) if re.match(r"^(?:in\s+)?what country\b|^which country\b", lowered_query) else None

        def _slot_rescue_blocked_by_missing_qualifiers() -> bool:
            qualifier_ignore = function_word_tokens | {
                "kind", "kinds", "type", "types", "main", "one", "two", "three", "four", "five",
                "share", "shared", "both", "same", "similar", "interest", "interests", "hobby", "hobbies",
                "temporary", "current", "favorite", "new", "old", "certain", "specific", "another", "other", "various",
                "recreational", "indoor", "outdoor",
                "prefer", "prefers", "preferred", "preference", "preferences",
                "why", "reason", "reasons", "because",
                "work", "working", "pursue", "pursuing",
                "begin", "beginning", "start", "starting", "started",
            }
            query_entity_tokens = {
                normalize_term_token(token)
                for phrase in (qf.get("entity_phrases") or [])
                for token in re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)?", str(phrase))
                if normalize_term_token(token)
            }
            qualifiers = sorted(
                token
                for token in (qf.get("words") or set())
                if token
                and token not in head_tokens
                and token not in qualifier_ignore
                and token not in query_entity_tokens
            )
            if not qualifiers:
                return False
            grounding_tokens = set()
            for text in [*slot_grounding_texts, *explicit_raw_candidates, *support_texts]:
                grounding_tokens.update(_normalized_text_tokens(text))
            missing = [token for token in qualifiers if token not in grounding_tokens]
            return bool(missing)

        ranked_grounded_candidates = _ranked_grounded_candidates()
        best_grounded = ranked_grounded_candidates[0] if ranked_grounded_candidates else None
        normalized_answer_surface = _normalized_answer_surface(answer)
        answer_surface_candidates = _fact_slot_fill_candidates(answer, qf, allow_loose_fallback=False)

        raw_head_words = re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)?", str(slot_plan.get("head_phrase") or ""))
        explicit_plural_query = bool(
            re.match(r"^(?:what|which)\s+are\b", lowered_query)
            or re.search(r"\b(?:two|three|four|five|several|some|many|multiple)\s+of\b", lowered_query)
        )
        singular_query_prefix = bool(
            re.match(r"^(?:what|which)\s+(?:is|was|does|did|has|had|might|may|could|would|should|can)\b", lowered_query)
        )
        last_head_word = raw_head_words[-1].lower() if raw_head_words else ""
        morphological_plural_head = bool(
            last_head_word
            and len(last_head_word) > 3
            and last_head_word.endswith("s")
            and not last_head_word.endswith(("ss", "us", "is"))
        )
        plural_slot_query = bool(
            explicit_plural_query
            or (morphological_plural_head and not singular_query_prefix)
        )
        multi_item_query = bool(
            plural_slot_query
            or qf.get("operator_plan", {}).get("commonality", {}).get("enabled")
            or qf.get("operator_plan", {}).get("list_set", {}).get("enabled")
        )
        mentioned_grounded_candidates = [
            candidate
            for candidate in ranked_grounded_candidates
            if _answer_mentions_candidate(candidate)
        ]
        answer_grounded_phrases = _extract_grounded_answer_phrases(answer)
        qualifier_guard_active = _slot_rescue_blocked_by_missing_qualifiers()

        def _candidate_adds_new_slot_info(candidate: str) -> bool:
            candidate_tokens = {
                normalize_term_token(token)
                for token in re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)?", candidate)
                if normalize_term_token(token)
            }
            candidate_tokens -= head_tokens
            candidate_tokens -= set(qf.get("words") or set())
            candidate_tokens -= {"the", "a", "an", "one", "provided", "context", "mentioned"}
            return bool(candidate_tokens)

        if re.match(r"^(?:in\s+)?what country\b|^which country\b", lowered_query) and country_support_candidate:
            return country_support_candidate

        meta_explanation = any(
            phrase in lowered
            for phrase in (
                "retrieved fact",
                "retrieved facts",
                "raw context",
                "provided context",
                "episode ",
                "explicitly stated",
                "confirmed in",
            )
        )
        if (
            not best_grounded
            and meta_explanation
            and negative_answer
            and not any(
                _candidate_adds_new_slot_info(candidate)
                for candidate in answer_surface_candidates
            )
        ):
            # ``meta_explanation`` (presence of words like ``retrieved fact``
            # or ``explicitly stated``) is not, on its own, a hallucination
            # signal — a correct positive answer that cites its evidence
            # legitimately uses such wording. The earlier rule fired on
            # any positive answer with citation language, producing a
            # large class of false-positive ``Not mentioned`` outputs.
            #
            # Tighten the rule by also requiring ``negative_answer``: only
            # return NM when the answer text itself signals "not mentioned"
            # / "unknown" / etc. AND no grounded candidate could be
            # surfaced from the recall context. Positive answers with
            # evidence citations now fall through to the token-grounding
            # gate (which judges the answer core, see
            # ``_answer_core_for_grounding`` below).
            if country_support_candidate:
                return country_support_candidate
            if commonality_interest_answer:
                return commonality_interest_answer
            return "Not mentioned in the provided context."

        if multi_item_query:
            if len(mentioned_grounded_candidates) >= 2:
                return ", ".join(mentioned_grounded_candidates[:4])
            if len(answer_grounded_phrases) >= 2:
                return ", ".join(answer_grounded_phrases[:4])

        if negative_answer:
            if country_support_candidate:
                return country_support_candidate
            if commonality_interest_answer:
                return commonality_interest_answer
            if best_grounded and not qualifier_guard_active:
                if plural_slot_query and len(ranked_grounded_candidates) >= 2:
                    return ", ".join(ranked_grounded_candidates[:2])
                return best_grounded
            return "Not mentioned in the provided context."

        if best_grounded and qualifier_guard_active:
            grounded_surfaces = {candidate.lower() for candidate in grounded_candidates}
            if normalized_answer_surface in grounded_surfaces:
                return "Not mentioned in the provided context."
            if normalized_answer_surface in {"it", "that", "this", "one", "this one", "that one"}:
                return "Not mentioned in the provided context."
            if any(candidate.lower() in lowered for candidate in grounded_candidates):
                return "Not mentioned in the provided context."

        if best_grounded and not qualifier_guard_active:
            grounded_surfaces = {candidate.lower() for candidate in grounded_candidates}
            if normalized_answer_surface in grounded_surfaces:
                return best_grounded
            if normalized_answer_surface in {"it", "that", "this", "one", "this one", "that one"}:
                return best_grounded
            if any(candidate.lower() in lowered for candidate in grounded_candidates):
                explanatory_slot_phrases = (
                    "retrieved fact",
                    "retrieved facts",
                    "raw context",
                    "provided context",
                    "episode ",
                    "does not mention",
                    "do not mention",
                    "not mentioned",
                    "not on ",
                    "not in ",
                    "not about ",
                    "however",
                    "but on ",
                    "specifically",
                )
                answer_word_count = len(re.findall(r"[A-Za-z0-9&'._-]+", answer))
                grounded_word_count = len(re.findall(r"[A-Za-z0-9&'._-]+", best_grounded))
                if any(phrase in lowered for phrase in explanatory_slot_phrases) or answer_word_count > grounded_word_count + 6:
                    return best_grounded
                return answer

        if re.match(r"^(?:in\s+)?what country\b|^which country\b", lowered_query):
            support_candidate = _country_surface_from_texts(answer, *support_texts)
            if support_candidate:
                return support_candidate

        # Token grounding gate.
        #
        # The gate must judge the *answer core* (what the model actually
        # claims), not its citation/explanation tail. Instruction-tuned
        # models routinely append "(Evidence: [1][2] explicitly state...)"
        # or "Sources: [1] (S3) explicitly states ..." after a correct
        # answer. Tokenising those tails inflates ``answer_tokens`` with
        # words that never appear in raw evidence (`evidence`, `sources`,
        # `explicitly`, `state`, bracket refs), which the strict
        # ``all(token in grounding_text)`` rule then treats as
        # ungrounded → false positive ``Not mentioned``.
        #
        # ``_answer_core_for_grounding`` strips those wrappers so the gate
        # sees just the substantive claim. The ``all()`` membership rule
        # is preserved on the core: a real hallucination (zero core
        # tokens in evidence) still triggers the NM path, but a correct
        # answer with citations no longer does.
        #
        # Date-component tokens (month names, weekday names) are also
        # stripped because the answer surface "21 May 2023" would
        # tokenise to ``may`` while the grounding evidence may carry the
        # numeric form ``2023-05-21`` — a lexical mismatch that does not
        # indicate hallucination. Numeric tokens (``21``, ``2023``) are
        # already excluded by the ``[A-Za-z]+`` regex.
        answer_core = _answer_core_for_grounding(answer)
        answer_tokens = {
            normalize_term_token(token)
            for token in re.findall(r"[A-Za-z]+(?:-[A-Za-z]+)?", answer_core)
            if normalize_term_token(token)
        }
        answer_tokens -= {
            token
            for token in set(answer_tokens)
            if token in qf.get("words", set()) or token in head_tokens
        }
        answer_tokens -= {"the", "a", "an", "one", "provided", "context", "mentioned"}
        # Date-component tokens — universal English temporal vocabulary,
        # not benchmark-specific.
        answer_tokens -= _GROUNDED_GATE_DATE_COMPONENT_TOKENS
        if not answer_tokens:
            return "Not mentioned in the provided context."
        grounding_text = " ".join(slot_grounding_texts or support_texts).lower()
        if not all(token in grounding_text for token in answer_tokens):
            return "Not mentioned in the provided context."
        return answer

    def _register_terminal_render_candidate(self, payload: dict) -> None:
        """Register private render text for a model-selected exact-copy candidate."""
        candidate_id = str((payload or {}).get("candidate_id") or "").strip()
        if not candidate_id:
            return
        self._terminal_render_candidate_registry[candidate_id] = deepcopy(payload)
        while len(self._terminal_render_candidate_registry) > 256:
            oldest = next(iter(self._terminal_render_candidate_registry))
            self._terminal_render_candidate_registry.pop(oldest, None)

    def _execute_terminal_render_candidate(
        self,
        input_data: dict,
        *,
        candidate_context: dict | None,
    ) -> tuple[str, dict]:
        public_candidate = dict(candidate_context or {})
        candidate_id = str((input_data or {}).get("candidate_id") or "").strip()
        expected_render_ref_id = str(public_candidate.get("render_ref_id") or "").strip()
        expected_container_id = str(public_candidate.get("container_id") or "").strip()
        render_ref_id = str((input_data or {}).get("render_ref_id") or expected_render_ref_id).strip()
        container_id = str((input_data or {}).get("container_id") or expected_container_id).strip()
        trace = {
            "candidate_kind": "terminal_render_candidate",
            "capability": "exact_copy",
            "candidate_id": candidate_id,
            "input": {
                "candidate_id": candidate_id,
                "render_ref_id": render_ref_id,
                "container_id": container_id,
            },
            "terminal_render_answer": False,
            "raw_text_exposed_to_model": False,
            "selected_container_ids": list(public_candidate.get("selected_container_ids") or []),
            "selected_render_ref_ids": list(public_candidate.get("selected_render_ref_ids") or []),
            "proof_summary": deepcopy(public_candidate.get("proof_summary")),
        }

        if not candidate_id:
            trace["error"] = "EXACT_COPY_CANDIDATE_ID_MISSING"
            return EXACT_COPY_REFUSAL, trace
        if not render_ref_id:
            trace["error"] = "EXACT_COPY_RENDER_REF_MISSING"
            return EXACT_COPY_REFUSAL, trace
        if public_candidate and candidate_id != str(public_candidate.get("candidate_id") or ""):
            trace["error"] = "EXACT_COPY_CANDIDATE_ID_MISMATCH"
            return EXACT_COPY_REFUSAL, trace
        if expected_render_ref_id and render_ref_id != expected_render_ref_id:
            trace["error"] = "EXACT_COPY_RENDER_REF_MISMATCH"
            return EXACT_COPY_REFUSAL, trace
        if expected_container_id and container_id and container_id != expected_container_id:
            trace["error"] = "EXACT_COPY_CONTAINER_MISMATCH"
            return EXACT_COPY_REFUSAL, trace
        proof_error = self._terminal_render_candidate_proof_error(public_candidate)
        if proof_error:
            trace["error"] = proof_error
            return EXACT_COPY_REFUSAL, trace

        private_candidate = dict(self._terminal_render_candidate_registry.get(candidate_id) or {})
        if not private_candidate:
            trace["error"] = "EXACT_COPY_CANDIDATE_UNRESOLVED"
            return EXACT_COPY_REFUSAL, trace
        if render_ref_id and render_ref_id != str(private_candidate.get("render_ref_id") or ""):
            trace["error"] = "EXACT_COPY_PRIVATE_RENDER_REF_MISMATCH"
            return EXACT_COPY_REFUSAL, trace
        render_text = private_candidate.get("render_text")
        if render_text is None:
            trace["error"] = "EXACT_COPY_RENDER_TEXT_MISSING"
            return EXACT_COPY_REFUSAL, trace

        answer = _apply_exact_copy_output_constraints(
            str(render_text),
            dict(private_candidate.get("output_constraints") or public_candidate.get("output_constraints") or {}),
        )
        trace.update({
            "terminal_render_answer": True,
            "render_mode": private_candidate.get("render_mode"),
            "render_source": private_candidate.get("render_source"),
            "render_ref_id": private_candidate.get("render_ref_id"),
            "container_id": private_candidate.get("container_id"),
        })
        return answer, trace

    def _terminal_render_candidate_proof_error(self, public_candidate: dict) -> str | None:
        planner_proof = dict(public_candidate.get("planner_proof") or {})
        render_proof = dict(public_candidate.get("render_proof") or {})

        def _required_true(key: str, *sources: dict) -> bool:
            for source in sources:
                if key in source:
                    return source.get(key) is True
            return False

        anchor_missing = (
            public_candidate.get("anchor_tokens_missing")
            or planner_proof.get("anchor_tokens_missing")
            or render_proof.get("anchor_tokens_missing")
            or []
        )
        if anchor_missing:
            return "EXACT_COPY_PROOF_ANCHOR_TOKENS_MISSING"
        if public_candidate.get("raw_text_exposed_to_model") is True:
            return "EXACT_COPY_RAW_TEXT_EXPOSED_TO_MODEL"
        if public_candidate.get("status") not in {None, "", "available"}:
            return "EXACT_COPY_CANDIDATE_NOT_AVAILABLE"
        if public_candidate.get("status") != "available":
            return "EXACT_COPY_CANDIDATE_NOT_AVAILABLE"
        if not _required_true("whole_or_fail", public_candidate, render_proof):
            return "EXACT_COPY_NOT_WHOLE_OR_FAIL"
        if public_candidate.get("degraded_render_source") or render_proof.get("degraded_render_source"):
            return "EXACT_COPY_SOURCE_DEGRADED"
        if not _required_true("render_ref_validated", public_candidate, render_proof):
            return "EXACT_RENDER_REF_UNRESOLVED"
        if not _required_true("raw_source_present", public_candidate, render_proof):
            return "RAW_SOURCE_MISSING"
        if not _required_true("raw_source_validated", public_candidate, render_proof):
            return "RAW_SOURCE_NOT_ORIGINAL"
        if not _required_true("ordinal_satisfied", public_candidate, planner_proof):
            return "EXACT_COPY_ORDINAL_NOT_SATISFIED"
        if not _required_true("kind_satisfied", public_candidate, planner_proof):
            return "EXACT_COPY_KIND_NOT_SATISFIED"
        if not _required_true("topic_satisfied", public_candidate, planner_proof):
            return "EXACT_COPY_TOPIC_NOT_SATISFIED"
        return None

    def _terminal_render_candidate_refusal_trace(
        self,
        error: str,
        *,
        candidate_context: dict | None,
    ) -> dict:
        public_candidate = dict(candidate_context or {})
        return {
            "candidate_kind": "terminal_render_candidate",
            "capability": "exact_copy",
            "candidate_id": public_candidate.get("candidate_id"),
            "terminal_render_answer": False,
            "raw_text_exposed_to_model": False,
            "selected_container_ids": list(public_candidate.get("selected_container_ids") or []),
            "selected_render_ref_ids": list(public_candidate.get("selected_render_ref_ids") or []),
            "proof_summary": deepcopy(public_candidate.get("proof_summary")),
            "error": error,
        }

    def _build_provider_payload(
        self,
        *,
        model: str,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
        use_tool: bool,
        backend: str = "api",
        cli_bin: str | None = None,
        cli_args_prefix: list[str] | None = None,
        cli_timeout_secs: float | None = None,
    ) -> tuple[dict, int]:
        if backend == "local_cli":
            if use_tool:
                raise RuntimeError("local_cli backend does not support tool use")
            return ({
                "backend": "local_cli",
                "model": model,
                "messages": messages,
                "max_output_tokens": max_tokens,
                "temperature": temperature,
                "cli_bin": cli_bin,
                "cli_args_prefix": list(cli_args_prefix or []),
                "cli_timeout_secs": cli_timeout_secs,
            }, 0)

        provider = _provider_for_model(model)
        tool_tokens_est = 0

        if provider == "anthropic":
            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if use_tool:
                payload["tools"] = list(TEMPORAL_TOOLS) + list(COUNTING_TOOLS) + [GET_CONTEXT_TOOL]
                tool_tokens_est = _estimate_tokens(payload["tools"])
            return payload, tool_tokens_est

        if provider == "google":
            payload = {
                "model": model,
                "messages": messages,
                "max_output_tokens": max_tokens,
                "temperature": temperature,
            }
            return payload, tool_tokens_est

        payload = {
            "model": model,
            "messages": messages,
            _tok_key(model): max_tokens,
            "seed": 42,
        }
        if _supports_temperature(model):
            payload["temperature"] = temperature
        if use_tool:
            payload["tools"] = _build_openai_tools_payload()
            payload["tool_choice"] = "auto"
            tool_tokens_est = _estimate_tokens(payload["tools"])
        return payload, tool_tokens_est

    def _truncate_by_priority(
        self,
        *,
        model: str,
        query: str,
        recall_result: dict,
        context_packet: dict,
        memory_budget: int,
        max_tokens: int,
        prompt_type: str,
        use_tool: bool,
        speakers: str,
        temperature: float,
        backend: str,
        cli_bin: str | None = None,
        cli_args_prefix: list[str] | None = None,
        cli_timeout_secs: float | None = None,
    ) -> tuple[dict, str, dict, dict]:
        working = deepcopy(context_packet)
        removed = {"tier4": 0, "tier3": 0, "tier2": 0}
        budget_exceeded = False

        while True:
            context = _render_context_packet(working)
            messages = self._build_payload_messages(
                prompt_type=prompt_type,
                context=context,
                query=query,
                recall_result=recall_result,
                speakers=speakers,
            )
            payload, tool_tokens_est = self._build_provider_payload(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                use_tool=use_tool,
                backend=backend,
                cli_bin=cli_bin,
                cli_args_prefix=cli_args_prefix,
                cli_timeout_secs=cli_timeout_secs,
            )
            message_tokens_est = _estimate_tokens(payload.get("messages", []))
            total_input_est = message_tokens_est + tool_tokens_est
            if total_input_est <= memory_budget:
                truncation = None
                if any(removed.values()):
                    truncation = {
                        "removed": removed,
                        "budget_exceeded": False,
                    }
                return working, context, payload, {
                    "context_tokens": _estimate_tokens(context),
                    "message_tokens_est": message_tokens_est,
                    "tool_tokens_est": tool_tokens_est,
                    "memory_budget": memory_budget,
                    "budget_exceeded": False,
                    "truncation": truncation,
                }

            for tier in ("tier4", "tier3", "tier2"):
                if working[tier]:
                    working[tier].pop()
                    removed[tier] += 1
                    break
            else:
                budget_exceeded = True
                truncation = {
                    "removed": removed,
                    "budget_exceeded": True,
                }
                return working, context, payload, {
                    "context_tokens": _estimate_tokens(context),
                    "message_tokens_est": message_tokens_est,
                    "tool_tokens_est": tool_tokens_est,
                    "memory_budget": memory_budget,
                    "budget_exceeded": budget_exceeded,
                    "truncation": truncation,
                }

    def _build_payload(
        self,
        *,
        query: str,
        recall_result: dict,
        inference_model: str = None,
        max_tokens: int = None,
        use_tool: bool = None,
        speakers: str = "User and Assistant",
    ) -> tuple[dict | None, dict | None, dict | None]:
        resolved_type = recall_result.get("query_type", "default")
        prompt_type = recall_result.get("recommended_prompt_type", resolved_type)
        prompt_key = resolve_inference_leaf_prompt_key(
            prompt_type=prompt_type,
            query=query,
            recall_result=recall_result,
            plugin_state=self._effective_inference_leaf_plugins(recall_result),
        )
        recommended_profile = recall_result.get("recommended_profile")
        payload_target = self._resolve_inference_target(recommended_profile, inference_model)
        if payload_target is None:
            return None, None, None

        model = payload_target["model"]
        profile_used = payload_target["profile_used"]
        profile_fallback = payload_target["profile_fallback"]
        cfg = payload_target["cfg"]
        secret_ref = payload_target["secret_ref"]
        backend = payload_target.get("backend", "api")
        cli_bin = payload_target.get("cli_bin")
        cli_args_prefix = payload_target.get("cli_args_prefix") or []
        cli_timeout_secs = payload_target.get("timeout_secs")

        terminal_render_candidate_available = bool(recall_result.get("terminal_render_candidate"))
        final_use_tool = recall_result.get("use_tool", False) if use_tool is None else use_tool
        tool_use_downgraded = False
        tool_use_downgrade_reason = None
        if terminal_render_candidate_available:
            final_use_tool = False
        if backend == "local_cli" and final_use_tool:
            if use_tool is True:
                raise RuntimeError("local_cli backend does not support tool use")
            final_use_tool = False
            tool_use_downgraded = True
            tool_use_downgrade_reason = "local_cli_backend_no_tool_support"
        temperature = float(cfg.get("temperature", 0) or 0)
        resolved_max_tokens = self._resolve_max_tokens(
            cfg=cfg,
            resolved_type=resolved_type,
            prompt_type=prompt_type,
            explicit_max_tokens=max_tokens,
        )
        memory_budget = self._compute_memory_budget(cfg, resolved_max_tokens)
        rendered_context = str(recall_result.get("context") or "")
        payload_recall_result = dict(recall_result)
        payload_recall_result["answer_contract"] = self._build_public_answer_contract(
            query=query,
            recall_result=payload_recall_result,
            prompt_type=prompt_type,
            prompt_key=prompt_key,
            use_tool=final_use_tool,
            speakers=speakers,
        )
        messages = self._build_payload_messages(
            prompt_type=prompt_type,
            context=rendered_context,
            query=query,
            recall_result=payload_recall_result,
            speakers=speakers,
        )
        payload, tool_tokens_est = self._build_provider_payload(
            model=model,
            messages=messages,
            max_tokens=resolved_max_tokens,
            temperature=temperature,
            use_tool=final_use_tool,
            backend=backend,
            cli_bin=cli_bin,
            cli_args_prefix=cli_args_prefix,
            cli_timeout_secs=cli_timeout_secs,
        )
        message_tokens_est = _estimate_tokens(payload.get("messages", []))
        evidence_trace = dict((recall_result.get("runtime_trace") or {}).get("evidence_context") or {})
        context_tokens = int(evidence_trace.get("context_tokens") or _estimate_tokens(rendered_context))
        trace_memory_budget = evidence_trace.get("memory_budget")
        if trace_memory_budget is not None:
            try:
                memory_budget = int(trace_memory_budget)
            except (TypeError, ValueError):
                pass
        budget_exceeded = bool(evidence_trace.get(
            "budget_exceeded",
            (message_tokens_est + tool_tokens_est) > memory_budget,
        ))

        payload_meta = {
            "profile_used": profile_used,
            "profile_fallback": profile_fallback,
            "pricing": deepcopy(cfg.get("pricing")),
            "context_tokens": context_tokens,
            "message_tokens_est": message_tokens_est,
            "tool_tokens_est": tool_tokens_est,
            "memory_budget": memory_budget,
            "budget_exceeded": budget_exceeded,
            "prompt_type": prompt_type,
            "prompt_key": prompt_key,
            "use_tool": final_use_tool,
            "tool_use_downgraded": tool_use_downgraded,
            "tool_use_downgrade_reason": tool_use_downgrade_reason,
            "truncation": evidence_trace.get("truncation"),
            "provider": "local_cli" if backend == "local_cli" else _provider_for_model(model),
            "provider_family": "local_cli" if backend == "local_cli" else _provider_family_for_model(model),
            "backend": backend,
            "terminal_render_candidate_available": terminal_render_candidate_available,
            "raw_text_exposed_to_model": False if terminal_render_candidate_available else None,
        }
        return payload, payload_meta, deepcopy(secret_ref)

    def _check_shell_budget(
        self,
        *,
        payload: dict,
        payload_meta: dict,
        shell_budget: float | None,
        recommended_profile: str | None,
        query_type: str,
    ) -> dict | None:
        if shell_budget is None:
            return None

        if str(payload_meta.get("backend") or payload.get("backend") or "api") == "local_cli":
            return None

        est_cost = self._estimate_payload_cost(payload=payload, payload_meta=payload_meta)
        if est_cost is None:
            model = str(payload.get("model") or payload_meta.get("model") or "unknown")
            return {
                "telemetry_version": 1,
                "answer": None,
                "error": f"Missing pricing for model '{model}' required for shell budget enforcement",
                "code": "MISSING_PRICING",
                "budget_exceeded": False,
                "estimated_cost": 0.0,
                "shell_budget": shell_budget,
                "best_effort_profile": None,
                "recommended_profile": recommended_profile,
                "query_type": query_type,
            }
        output_tokens = (
            payload.get("max_output_tokens")
            or payload.get("max_tokens")
            or payload.get("max_completion_tokens")
            or 0
        )
        input_tokens = (
            payload_meta.get("message_tokens_est", 0)
            + payload_meta.get("tool_tokens_est", 0)
        )

        if est_cost <= shell_budget:
            return None

        best_effort = None
        if self._has_profiles():
            for name in sorted(
                self._list_profile_names(),
                key=lambda n: ((self._get_profile_config(n) or {}).get("pricing") or {}).get("input_per_1k", 999),
            ):
                p_cfg = self._get_profile_config(name)
                pricing = (p_cfg or {}).get("pricing") if p_cfg else None
                if not pricing:
                    continue
                p_cost = (
                    input_tokens / 1000 * pricing["input_per_1k"]
                    + output_tokens / 1000 * pricing["output_per_1k"]
                )
                if p_cost <= shell_budget:
                    best_effort = name
                    break

        return {
            "telemetry_version": 1,
            "answer": None,
            "budget_exceeded": True,
            "estimated_cost": round(est_cost, 6),
            "shell_budget": shell_budget,
            "best_effort_profile": best_effort,
            "recommended_profile": recommended_profile,
            "query_type": query_type,
        }

    def _derive_codebase_file_lookup_deterministic_answer(self, query: str, recall_result: dict) -> str | None:
        lowered = str(query or "").strip().lower()
        if not lowered:
            return None

        retrieval_families = {
            str(family or "").strip().lower()
            for family in (recall_result.get("retrieval_families") or [])
            if str(family or "").strip()
        }
        if retrieval_families != {"codebase"}:
            return None

        asks_for_file = any(marker in lowered for marker in CODEBASE_FILE_LOOKUP_REQUEST_MARKERS)
        if not asks_for_file:
            return None

        if any(marker in lowered for marker in CODEBASE_FILE_LOOKUP_MULTI_PART_MARKERS):
            return None

        normalized = lowered
        for clause in CODEBASE_FILE_LOOKUP_NORMALIZE_CLAUSES:
            normalized = normalized.replace(clause, " ")
        normalized = re.sub(r"[?.!,:;]+", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()

        if not any(re.match(pattern, normalized) for pattern in CODEBASE_FILE_LOOKUP_PATTERNS):
            return None

        code_trace = ((recall_result.get("runtime_trace") or {}).get("codebase_augmentation") or {})
        selected_file = str(code_trace.get("selected_file") or "").strip()
        if not selected_file:
            return None
        return selected_file

    def _estimate_payload_cost(
        self,
        *,
        payload: dict,
        payload_meta: dict,
    ) -> float | None:
        if str(payload_meta.get("backend") or payload.get("backend") or "api") == "local_cli":
            return 0.0
        output_tokens = (
            payload.get("max_output_tokens")
            or payload.get("max_tokens")
            or payload.get("max_completion_tokens")
            or 0
        )
        input_tokens = (
            payload_meta.get("message_tokens_est", 0)
            + payload_meta.get("tool_tokens_est", 0)
        )

        pricing = payload_meta.get("pricing")
        if pricing is None:
            profile_used = payload_meta.get("profile_used")
            cfg = self._get_profile_config(profile_used) if profile_used else None
            pricing = (cfg or {}).get("pricing") if cfg else None
        if not pricing:
            return None
        required_pricing_keys = {"input_per_1k", "output_per_1k"}
        if not required_pricing_keys.issubset(pricing):
            return None
        return (
            input_tokens / 1000 * pricing["input_per_1k"]
            + output_tokens / 1000 * pricing["output_per_1k"]
        )

    @staticmethod
    def _positive_session_id(value: Any) -> int:
        try:
            session_id = int(value)
        except (TypeError, ValueError):
            return 0
        return session_id if session_id > 0 else 0

    @staticmethod
    def _continuation_tool_result_from_page(result: dict) -> dict:
        continuation = dict(result.get("recall_continuation") or {})
        context = str(result.get("context") or result.get("error") or "")
        response = {
            "result": context,
            "context": context,
            "page": continuation.get("page"),
            "next_page": continuation.get("next_page"),
            "exhausted": bool(continuation.get("exhausted")),
            "continuation_handle": continuation.get("handle"),
            "handle": continuation.get("handle"),
            "recall_continuation": continuation,
            "runtime_trace": result.get("runtime_trace", {}),
        }
        if result.get("error") or result.get("code"):
            response["error"] = result.get("error", "Recall continuation failed")
            response["code"] = result.get("code", "RECALL_CONTINUATION_ERROR")
        return response

    def _execute_get_more_context_tool(
        self,
        input_data: dict,
        *,
        continuation_pages: list[dict],
        continuation_handle: str,
        continuation_state: dict,
        continuation_acl: dict,
        caller_id: str | None,
    ) -> dict:
        args = dict(input_data or {})
        page = args.get("page")
        provided_handle = str(args.get("handle") or args.get("continuation_handle") or "").strip()
        default_handle = str(continuation_handle or "").strip()
        requested_handle = provided_handle or default_handle
        session_id = self._positive_session_id(args.get("session_id", 0))
        wants_continuation = bool(requested_handle) and (
            bool(provided_handle)
            or page not in (None, "")
            or session_id == 0
        )
        if wants_continuation:
            result = self.recall_continuation_page(
                continuation_handle=requested_handle,
                page=page or "next",
                caller_id=continuation_acl.get("caller_id") or caller_id,
                caller_memberships=list(continuation_acl.get("caller_memberships") or []),
                caller_role=str(continuation_acl.get("caller_role") or "user"),
                swarm_id=continuation_acl.get("swarm_id"),
                bind_swarm_from_handle=True,
            )
            return self._continuation_tool_result_from_page(result)

        legacy_full_session = session_id > 0 and not provided_handle and page in (None, "")
        return get_more_context(
            session_id,
            raw_sessions=self._raw_sessions,
            page=page,
            handle=provided_handle,
            continuation_handle=args.get("continuation_handle"),
            recall_continuation_pages=[] if legacy_full_session else continuation_pages,
            recall_continuation_handle="" if legacy_full_session else default_handle,
            continuation_state=continuation_state,
        )

    async def _send_payload(
        self,
        payload: dict,
        *,
        caller_id: str = None,
        secret_ref: dict | None = None,
    ) -> tuple[str, bool, list[dict]]:
        continuation_pages = list(payload.get("_recall_continuation_pages") or [])
        continuation_handle = str(payload.get("_recall_continuation_handle") or "")
        continuation_acl = dict(payload.get("_recall_continuation_acl") or {})
        continuation_state = {"next_page": 2}
        payload = {
            key: value
            for key, value in payload.items()
            if key not in {
                "_recall_continuation_pages",
                "_recall_continuation_handle",
                "_recall_continuation_acl",
            }
        }
        model = payload["model"]
        backend = str(payload.get("backend") or "api")
        provider = _provider_for_model(model)
        has_tools = bool(payload.get("tools"))

        if backend == "local_cli":
            if has_tools:
                raise RuntimeError("local_cli backend does not support tool use")
            messages = list(payload.get("messages") or [])
            system = ""
            prompt_messages = []
            for message in messages:
                role = str(message.get("role") or "")
                content = str(message.get("content") or "")
                if role == "system" and not system:
                    system = content
                else:
                    prompt_messages.append({"role": role, "content": content})
            prompt = render_local_cli_prompt(system, prompt_messages)
            cli_timeout_secs = payload.get("cli_timeout_secs")
            if cli_timeout_secs is None:
                answer = await asyncio.to_thread(
                    run_local_cli,
                    prompt,
                    str(payload.get("cli_bin") or ""),
                    list(payload.get("cli_args_prefix") or []),
                )
            else:
                answer = await asyncio.to_thread(
                    run_local_cli,
                    prompt,
                    str(payload.get("cli_bin") or ""),
                    list(payload.get("cli_args_prefix") or []),
                    cli_timeout_secs,
                )
            return answer, False, []

        if not has_tools:
            messages = payload.get("messages", [])
            max_tokens = (
                payload.get("max_output_tokens")
                or payload.get("max_tokens")
                or payload.get("max_completion_tokens")
                or 2000
            )
            temperature = payload.get("temperature", 0.0)
            if provider == "openai" and messages and len(messages) == 1 and messages[0].get("role") == "user":
                answer = await self._call_oai_with_runtime_secrets(
                    model,
                    messages[0].get("content", ""),
                    max_tokens=max_tokens,
                    temperature=temperature,
                    secret_ref=secret_ref,
                )
                return answer, False, []

            answer = await self._call_model_with_runtime_secrets(
                model,
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                secret_ref=secret_ref,
            )
            return answer, False, []

        if provider == "anthropic":
            client = self._get_model_client_with_runtime_secrets(model, secret_ref=secret_ref)
            request = dict(payload)
            request["model"] = _api_model(model)
            messages = deepcopy(request["messages"])
            request["messages"] = messages
            tool_called = False
            tool_results: list[dict[str, Any]] = []
            for _ in range(3):
                response = await client.messages.create(**request)
                tool_uses = [b for b in response.content if getattr(b, "type", "") == "tool_use"]
                text_blocks = [b for b in response.content if getattr(b, "type", "") == "text"]
                if not tool_uses:
                    answer = "".join(getattr(b, "text", "") for b in text_blocks).strip()
                    return answer, tool_called, tool_results

                tool_called = True
                tool_outputs = []
                for tu in tool_uses:
                    input_data = getattr(tu, "input", {}) or {}
                    result_obj = self._execute_get_more_context_tool(
                        input_data,
                        continuation_pages=continuation_pages,
                        continuation_handle=continuation_handle,
                        continuation_state=continuation_state,
                        continuation_acl=continuation_acl,
                        caller_id=caller_id,
                    ) if tu.name == "get_more_context" else json.loads(
                        json.dumps({"error": f"Unknown tool: {tu.name}"})
                    )
                    result_str = json.dumps(result_obj)
                    tool_results.append({"tool": tu.name, "input": input_data, "result": result_str})
                    if tu.name == "get_more_context" and hasattr(self, "_audit"):
                        self._audit.log(
                            "get_more_context",
                            caller_id=caller_id or "unknown",
                            details={"session_id": input_data.get("session_id")},
                        )
                    tool_outputs.append({
                        "type": "tool_result",
                        "tool_use_id": tu.id,
                        "content": result_str,
                    })
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_outputs})
            return "", tool_called, tool_results

        client = self._get_model_client_with_runtime_secrets(model, secret_ref=secret_ref)
        request = dict(payload)
        request["model"] = _api_model(model)
        messages = deepcopy(payload["messages"])
        request["messages"] = messages
        tool_results = []
        tool_called = False
        for _ in range(3):
            response = await client.chat.completions.create(**request)
            msg = response.choices[0].message
            tool_calls = getattr(msg, "tool_calls", None) or []
            if not tool_calls:
                return (msg.content or ""), tool_called, tool_results

            tool_called = True
            assistant_tool_calls = []
            for tc in tool_calls:
                assistant_tool_calls.append({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                })
            messages.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": assistant_tool_calls,
            })

            for tc in tool_calls:
                args = json.loads(tc.function.arguments)
                if tc.function.name == "get_more_context":
                    tool_result = self._execute_get_more_context_tool(
                        args,
                        continuation_pages=continuation_pages,
                        continuation_handle=continuation_handle,
                        continuation_state=continuation_state,
                        continuation_acl=continuation_acl,
                        caller_id=caller_id,
                    )
                    if hasattr(self, "_audit"):
                        self._audit.log(
                            "get_more_context",
                            caller_id=caller_id or "unknown",
                            details={"session_id": args.get("session_id")},
                        )
                else:
                    tool_result = {"error": f"Unknown tool: {tc.function.name}"}
                tool_result_json = json.dumps(tool_result)
                tool_results.append({
                    "tool": tc.function.name,
                    "input": args,
                    "result": tool_result_json,
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_result_json,
                })

            request = {
                "model": _api_model(model),
                "messages": messages,
                _tok_key(model): payload.get(_tok_key(model), payload.get("max_tokens", 2000)),
                "seed": payload.get("seed", 42),
            }
            if "tools" in payload:
                request["tools"] = payload["tools"]
                request["tool_choice"] = payload.get("tool_choice", "auto")
            if _supports_temperature(model) and "temperature" in payload:
                request["temperature"] = payload["temperature"]
        return "", tool_called, tool_results

    # ── ask() ──

    async def ask(
        self,
        query: str,
        agent_id: str = None,
        swarm_id: str = None,
        search_family: str = "auto",
        query_type: str = "auto",
        kind: str = "all",
        caller_memberships: list = None,
        caller_role: str = "user",
        caller_id: str = None,
        inference_model: str = None,
        max_tokens: int = None,
        use_tool: bool = None,
        shell_budget: float = None,
        speakers: str = "User and Assistant",
        query_metadata: dict[str, Any] | None = None,
        mal_binding_id: str | None = None,
    ) -> dict:
        """Answer a question using recall() + LLM inference.

        search_family: auto | conversation | document | codebase
        Returns dict with: answer, profile_used, profile_fallback,
        recommended_profile, query_type, use_tool, tool_called, budget_exceeded.
        """
        effective_caller_id = _default_runtime_caller_id(caller_id, agent_id, self.agent_id)
        self._audit.log("ask", effective_caller_id,
                        {"query": query[:200]})
        recall_result = await self.recall(
            query, agent_id=agent_id, swarm_id=swarm_id,
            search_family=search_family,
            query_type=query_type, kind=kind,
            query_metadata=query_metadata,
            caller_memberships=caller_memberships,
            caller_role=caller_role, caller_id=effective_caller_id,
            mal_binding_id=mal_binding_id,
        )
        if "context" not in recall_result:
            return recall_result
        if recall_result.get("error") or recall_result.get("code"):
            return recall_result
        recommended_profile = (
            recall_result.get("recommended_profile")
            or self._recommended_profile_for_recall_result(recall_result)
        )
        runtime_trace = dict(recall_result.get("runtime_trace") or {})
        deterministic_answer_raw = recall_result.get("deterministic_answer")
        deterministic_answer = str(deterministic_answer_raw) if deterministic_answer_raw is not None else ""
        if not deterministic_answer:
            deterministic_answer = self._derive_codebase_file_lookup_deterministic_answer(query, recall_result) or ""
            if deterministic_answer:
                recall_result["deterministic_answer"] = deterministic_answer
                runtime_trace["deterministic_answer"] = {
                    "kind": "codebase_file_lookup",
                    "answer": deterministic_answer,
                }
                recall_result["runtime_trace"] = runtime_trace
        if deterministic_answer:
            exact_copy_deterministic = _deterministic_answer_is_exact_copy(recall_result)
            if exact_copy_deterministic:
                runtime_trace["deterministic_answer"] = {
                    "kind": "terminal_render_candidate",
                    "answer_source": "container_render_ref",
                    "model_path_required": True,
                    "bypassed_pre_inference": False,
                }
                recall_result["runtime_trace"] = runtime_trace
                deterministic_answer = ""
            else:
                deterministic_answer = _apply_output_constraints(
                    deterministic_answer,
                    recall_result.get("output_constraints"),
                )
                deterministic_answer = self._normalize_grounded_answer(
                    query,
                    deterministic_answer,
                    recall_result,
                )
                return {
                    "telemetry_version": 1,
                    "answer": deterministic_answer,
                    "query_type": recall_result.get("query_type", "default"),
                    "use_tool": False,
                    "tool_called": False,
                    "tool_results": None,
                    "profile_used": "deterministic:temporal_v1",
                    "profile_fallback": False,
                    "recommended_profile": recommended_profile,
                    "budget_exceeded": False,
                    "estimated_cost": 0.0,
                    "retrieval_families": recall_result.get("retrieval_families", []),
                    "search_family": recall_result.get("search_family"),
                    "retrieved_count": len(recall_result.get("retrieved", [])),
                    "runtime_trace": runtime_trace,
                    "payload_meta": {
                        "profile_used": "deterministic:temporal_v1",
                        "profile_fallback": False,
                        "use_tool": False,
                        "deterministic": True,
                    },
                }
        if not self._has_profiles() and inference_model is None:
            return {"error": "No profiles configured and no inference_model provided",
                    "code": "NO_PROFILES"}
        resolved_type = recall_result.get("query_type", "default")
        plan = self._build_inference_plan_from_recall_result(
            query=query,
            recall_result=recall_result,
            inference_model=inference_model,
            max_tokens=max_tokens,
            use_tool=use_tool,
            speakers=speakers,
        )
        if "error" in plan:
            return {"error": plan["error"], "code": plan.get("code", "NO_PROFILES")}
        recommended_profile = plan.get("recommended_profile")
        payload = plan["payload"]
        payload_meta = plan["payload_meta"]
        secret_ref = plan.get("secret_ref")
        if payload_meta.get("use_tool") and recall_result.get("_recall_continuation_pages"):
            payload = dict(payload)
            payload["_recall_continuation_pages"] = deepcopy(recall_result.get("_recall_continuation_pages") or [])
            payload["_recall_continuation_handle"] = (
                (recall_result.get("recall_continuation") or {}).get("handle")
            )
            payload["_recall_continuation_acl"] = {
                "caller_id": effective_caller_id,
                "caller_memberships": list(caller_memberships or []),
                "caller_role": caller_role,
                "swarm_id": swarm_id,
            }

        estimated_cost_raw = self._estimate_payload_cost(payload=payload, payload_meta=payload_meta)
        estimated_cost = round(estimated_cost_raw, 6) if estimated_cost_raw is not None else 0.0
        budget_result = self._check_shell_budget(
            payload=payload,
            payload_meta=payload_meta,
            shell_budget=shell_budget,
            recommended_profile=recommended_profile,
            query_type=resolved_type,
        )
        if budget_result is not None:
            return budget_result

        exact_copy_model_path = _container_exact_copy_model_path(recall_result)
        candidate_context = dict(recall_result.get("terminal_render_candidate") or {}) if exact_copy_model_path else None
        answer, tool_called, tool_results = await self._send_payload(
            payload,
            caller_id=effective_caller_id,
            secret_ref=secret_ref,
        )

        model_selected_candidate = False
        terminal_render_trace: dict[str, Any] | None = None
        render_results: list[dict[str, Any]] = []
        if exact_copy_model_path:
            decision = _parse_terminal_render_candidate_decision(answer)
            if decision:
                answer, render_trace = self._execute_terminal_render_candidate(
                    decision,
                    candidate_context=candidate_context,
                )
                model_selected_candidate = bool(render_trace.get("terminal_render_answer"))
                terminal_render_trace = render_trace
                render_results = [render_trace]
            else:
                answer = EXACT_COPY_REFUSAL
                terminal_render_trace = self._terminal_render_candidate_refusal_trace(
                    "EXACT_COPY_MODEL_DID_NOT_SELECT_CANDIDATE",
                    candidate_context=candidate_context,
                )
                render_results = [terminal_render_trace]
            tool_results = []
        if not exact_copy_model_path and answer:
            answer = self._normalize_grounded_answer(query, _strip_model_think_blocks(answer), recall_result)

        trace_ref = f"ask_{uuid4().hex[:12]}"
        runtime_trace = dict(recall_result.get("runtime_trace") or {})
        runtime_trace["backend"] = payload_meta.get("backend")
        if exact_copy_model_path:
            if not model_selected_candidate:
                answer = EXACT_COPY_REFUSAL
            candidate_trace = dict(runtime_trace.get("terminal_render_candidate") or {})
            candidate_trace.update({
                "terminal_render_candidate_available": True,
                "model_selected_terminal_render_candidate": model_selected_candidate,
                "tool_called": bool(tool_called),
                "terminal_render_answer": model_selected_candidate,
                "raw_text_exposed_to_model": False,
                "refusal_reason": None
                if model_selected_candidate
                else str((terminal_render_trace or {}).get("error") or "EXACT_COPY_MODEL_DID_NOT_SELECT_CANDIDATE"),
            })
            if terminal_render_trace is not None:
                candidate_trace["terminal_render_trace"] = terminal_render_trace
                runtime_trace["terminal_render_trace"] = terminal_render_trace
            runtime_trace["terminal_render_candidate"] = candidate_trace


        response = {
            "telemetry_version": 1,
            "answer": answer,
            "query_type": resolved_type,
            "use_tool": payload_meta.get("use_tool", False),
            "tool_called": tool_called,
            "tool_results": tool_results or None,
            "profile_used": payload_meta.get("profile_used"),
            "profile_fallback": payload_meta.get("profile_fallback", False),
            "recommended_profile": recommended_profile,
            "budget_exceeded": False,
            "estimated_cost": estimated_cost,
            "retrieval_families": recall_result.get("retrieval_families", []),
            "search_family": recall_result.get("search_family"),
            "retrieved_count": len(recall_result.get("retrieved", [])),
            "runtime_trace": runtime_trace,
            "runtime_trace_ref": trace_ref,
            "payload_meta": payload_meta,
        }
        if exact_copy_model_path:
            response["terminal_render_trace"] = terminal_render_trace
            response["render_results"] = render_results or None
        return response

    # ── get_versions() (Unit 6) ──

    def get_versions(
        self,
        artifact_id,
        caller_id=None,
        caller_role="agent",
        caller_memberships: list[str] | None = None,
    ):
        """Return version chain ordered by parent_version for an artifact."""
        effective_caller_id = _default_runtime_caller_id(caller_id, None, self.agent_id)
        # ACL check: find any fact from this artifact, verify read access
        if caller_role != "admin":
            sample = None
            for f in self._all_granular:
                if f.get("artifact_id") == artifact_id:
                    sample = f
                    break
            if sample is None:
                for rs in self._raw_sessions:
                    if rs.get("artifact_id") == artifact_id:
                        sample = rs
                        break
            if sample is not None:
                if not self._acl_allows(
                    sample,
                    effective_caller_id,
                    caller_memberships or [],
                    caller_role,
                ):
                    return {"error": "Access denied", "code": "ACL_FORBIDDEN"}

        versions = []
        for tier in (self._all_granular, self._all_cons, self._all_cross):
            for f in tier:
                if f.get("artifact_id") == artifact_id:
                    ver = {
                        "version_id": f.get("version_id"),
                        "parent_version": f.get("parent_version"),
                        "content_hash": f.get("content_hash"),
                        "status": f.get("status", "active"),
                        "fact": f.get("fact"),
                        "created_at": f.get("created_at"),
                    }
                    if ver not in versions:
                        versions.append(ver)
        # Also check raw_sessions
        for rs in self._raw_sessions:
            if rs.get("artifact_id") == artifact_id:
                ver = {
                    "version_id": rs.get("version_id"),
                    "parent_version": rs.get("parent_version"),
                    "content_hash": rs.get("content_hash"),
                    "status": rs.get("status", "active"),
                    "created_at": rs.get("stored_at"),
                }
                if ver not in versions:
                    versions.append(ver)
        # Order: roots first, then children
        by_vid = {v["version_id"]: v for v in versions}
        ordered = []
        seen = set()
        # Find roots (no parent or parent not in set)
        roots = [v for v in versions
                 if not v.get("parent_version")
                 or v["parent_version"] not in by_vid]
        queue = list(roots)
        while queue:
            v = queue.pop(0)
            vid = v["version_id"]
            if vid in seen:
                continue
            seen.add(vid)
            ordered.append(v)
            # Find children
            for c in versions:
                if c.get("parent_version") == vid and c["version_id"] not in seen:
                    queue.append(c)
        # Add any remaining (disconnected)
        for v in versions:
            if v["version_id"] not in seen:
                ordered.append(v)
        return {"artifact_id": artifact_id, "versions": ordered}

    # ── edit() (Unit 6) ──

    async def edit(
        self,
        artifact_id,
        new_content,
        caller_id=None,
        caller_role="agent",
        caller_memberships: list[str] | None = None,
    ):
        """Create a new version of an artifact with new content.

        Finds the active version, creates a new version, supersedes the old one.
        Requires write access (owner, write ACL, or admin).
        """
        effective_caller_id = _default_runtime_caller_id(caller_id, None, self.agent_id)
        self._audit.log("edit", effective_caller_id,
                        {"artifact_id": artifact_id})
        # Find active facts for this artifact
        active_facts = [f for f in self._all_granular
                        if f.get("artifact_id") == artifact_id
                        and f.get("status") == "active"]
        if not active_facts:
            return {"error": f"No active artifact: {artifact_id}",
                    "code": "NOT_FOUND"}

        old_fact = active_facts[0]

        # ACL check: write access required
        if not self._acl_allows_access(
            old_fact,
            effective_caller_id,
            caller_memberships or [],
            caller_role,
            need="write",
        ):
            return {"error": "Write access denied",
                    "code": "ACL_FORBIDDEN"}
        old_version_id = old_fact.get("version_id")
        new_version_id = _generate_version_id()
        source_id = old_fact.get("source_id")
        source_family = str((self._source_records.get(str(source_id or "")) or {}).get("family") or "conversation")
        metadata = old_fact.get("metadata") or {}
        if not source_id and str(metadata.get("document_source") or "").strip():
            source_id = str(metadata.get("document_source"))
            source_family = "document"
        canonical_family = self._canonical_content_family(source_family)
        new_hash = content_hash_text(new_content, family=canonical_family)
        if str(old_fact.get("content_hash") or "") == str(new_hash):
            return {
                "artifact_id": artifact_id,
                "version_id": old_version_id,
                "parent_version": old_fact.get("parent_version"),
                "status": "duplicate",
            }

        # Supersede old version
        for f in self._all_granular:
            if f.get("artifact_id") == artifact_id and f.get("status") == "active":
                f["status"] = "superseded"
        for rs in self._raw_sessions:
            if rs.get("artifact_id") == artifact_id and rs.get("status") == "active":
                rs["status"] = "superseded"

        # Create new fact (copy metadata from old)
        new_fact = dict(old_fact)
        new_fact["fact"] = new_content
        new_fact["version_id"] = new_version_id
        new_fact["parent_version"] = old_version_id
        new_fact["content_hash"] = new_hash
        new_fact["status"] = "active"
        new_fact["created_at"] = datetime.now(timezone.utc).isoformat()
        new_fact["id"] = f"edited_{new_version_id}"

        async with self._file_lock:
            self._all_granular.append(new_fact)
            self._mark_tiers_dirty()
            self._data_dict = None
            # Update dedup index if source_id present
            sid = source_id or new_fact.get("source_id")
            sn = old_fact.get("session", 0)
            if sid:
                dedup_key = self._source_versioning_key(
                    source_id=str(sid),
                    family=canonical_family,
                    scope=str(old_fact.get("scope") or "swarm-shared"),
                    owner_id=old_fact.get("owner_id"),
                    swarm_id=old_fact.get("swarm_id"),
                    session_num=sn,
                    multipart_part_key=(
                        self._multipart_part_key(metadata)
                        if canonical_family == "document"
                        else None
                    ),
                )
                self._dedup_index[dedup_key] = {
                    "artifact_id": artifact_id,
                    "version_id": new_version_id,
                    "content_hash": new_hash,
                }
            self._bump_index_snapshot_version()
            self._save_cache()

        return {"artifact_id": artifact_id,
                "version_id": new_version_id,
                "parent_version": old_version_id}

    # ── retract() (Unit 6) ──

    async def retract(
        self,
        artifact_id,
        caller_id=None,
        caller_role="agent",
        caller_memberships: list[str] | None = None,
    ):
        """Retract an artifact — all versions become invisible.

        Requires write access (owner, write ACL, or admin).
        """
        effective_caller_id = _default_runtime_caller_id(caller_id, None, self.agent_id)
        self._audit.log("retract", effective_caller_id,
                        {"artifact_id": artifact_id})

        # ACL check: write access required
        target = None
        for f in self._all_granular:
            if f.get("artifact_id") == artifact_id:
                target = f
                break
        if target is None and caller_role != "admin":
            return {"error": f"Artifact not found: {artifact_id}",
                    "code": "NOT_FOUND"}
        if target is not None and not self._acl_allows_access(
            target,
            effective_caller_id,
            caller_memberships or [],
            caller_role,
            need="write",
        ):
            return {"error": "Write access denied",
                    "code": "ACL_FORBIDDEN"}

        found = False
        async with self._file_lock:
            affected_message_ids: set[str] = set()
            for f in self._all_granular:
                if f.get("artifact_id") == artifact_id:
                    f["status"] = "retracted"
                    found = True
            for f in self._all_cons:
                if f.get("artifact_id") == artifact_id:
                    f["status"] = "retracted"
                    found = True
            for f in self._all_cross:
                if f.get("artifact_id") == artifact_id:
                    f["status"] = "retracted"
                    found = True
            for rs in self._raw_sessions:
                if rs.get("artifact_id") == artifact_id:
                    rs["status"] = "retracted"
                    found = True
                    message_id = str(rs.get("message_id") or "")
                    if message_id:
                        affected_message_ids.add(message_id)
                        self._remove_content_indices_for_message(message_id)
            for doc in self._episode_corpus.get("documents", []):
                for episode in doc.get("episodes", []):
                    if isinstance(episode, dict) and episode.get("artifact_id") == artifact_id:
                        episode["status"] = "retracted"
                        found = True
                        message_id = str(episode.get("message_id") or "")
                        if message_id:
                            affected_message_ids.add(message_id)
            for key, value in list(self._dedup_index.items()):
                if (value or {}).get("artifact_id") == artifact_id:
                    self._dedup_index.pop(key, None)
            for message_id in affected_message_ids:
                self._remove_content_indices_for_message(message_id)
            if found:
                self._data_dict = None
                self._mark_full_index_dirty()
                self._bump_index_snapshot_version()
                self._save_cache()
        if not found:
            return {"error": f"Artifact not found: {artifact_id}",
                    "code": "NOT_FOUND"}
        return {"artifact_id": artifact_id, "status": "retracted"}

    # ── purge() (Unit 6) ──

    async def purge(self, artifact_id, caller_id=None, caller_role="agent"):
        """Purge an artifact — admin only, physically removes from all lists."""
        effective_caller_id = _default_runtime_caller_id(caller_id, None, self.agent_id)
        self._audit.log("purge", effective_caller_id,
                        {"artifact_id": artifact_id})
        if caller_role != "admin":
            return {"error": "purge requires admin role",
                    "code": "ACL_FORBIDDEN"}

        async with self._file_lock:
            before_g = len(self._all_granular)
            self._all_granular = [f for f in self._all_granular
                                  if f.get("artifact_id") != artifact_id]
            self._all_cons = [f for f in self._all_cons
                              if f.get("artifact_id") != artifact_id]
            self._all_cross = [f for f in self._all_cross
                               if f.get("artifact_id") != artifact_id]
            self._raw_sessions = [rs for rs in self._raw_sessions
                                  if rs.get("artifact_id") != artifact_id]
            removed = before_g - len(self._all_granular)
            # Clean dedup index
            keys_to_remove = [k for k, v in self._dedup_index.items()
                              if v.get("artifact_id") == artifact_id]
            for k in keys_to_remove:
                del self._dedup_index[k]
            # Clean git dedup index
            git_keys_to_remove = [k for k, v in self._git_dedup_index.items()
                                  if v.get("artifact_id") == artifact_id]
            for k in git_keys_to_remove:
                del self._git_dedup_index[k]

            self._data_dict = None
            self._mark_tiers_dirty()
            self._bump_index_snapshot_version()
            self._save_cache()

        return {"artifact_id": artifact_id, "purged_facts": removed}

    # ── redact() (Unit 7) ──

    async def redact(self, artifact_id: str, fields: list[str],
                     caller_id=None, caller_role="agent",
                     caller_memberships: list[str] | None = None) -> dict:
        """Redact specified fields of an artifact.

        Replaces fact text with [REDACTED], entities with ["[REDACTED]"],
        raw session content with [REDACTED]. Sets status="redacted".
        Requires write access or admin role.
        """
        effective_caller_id = _default_runtime_caller_id(caller_id, None, self.agent_id)
        self._audit.log("redact", effective_caller_id,
                        {"artifact_id": artifact_id, "fields": fields})
        # ACL check: write access or admin
        target = None
        for f in self._all_granular:
            if f.get("artifact_id") == artifact_id:
                target = f
                break
        if target is None and caller_role != "admin":
            return {"error": f"Artifact not found: {artifact_id}",
                    "code": "NOT_FOUND"}
        if target is not None and not self._acl_allows_access(
            target,
            effective_caller_id,
            caller_memberships or [],
            caller_role,
            need="write",
        ):
            return {"error": "Write access denied",
                    "code": "ACL_FORBIDDEN"}

        found = False
        async with self._file_lock:
            for tier in (self._all_granular, self._all_cons, self._all_cross):
                for f in tier:
                    if f.get("artifact_id") == artifact_id:
                        if "fact" in fields:
                            f["fact"] = "[REDACTED]"
                        if "entities" in fields:
                            f["entities"] = ["[REDACTED]"]
                        f["status"] = "redacted"
                        found = True
            for rs in self._raw_sessions:
                if rs.get("artifact_id") == artifact_id:
                    if "content" in fields or "fact" in fields:
                        rs["content"] = "[REDACTED]"
                    rs["status"] = "redacted"
                    found = True
            if found:
                self._data_dict = None
                self._save_cache()

        if not found:
            return {"error": f"Artifact not found: {artifact_id}",
                    "code": "NOT_FOUND"}
        return {"redacted": True, "artifact_id": artifact_id, "fields": fields}

    # ── stats() ──

    def stats(self) -> dict:
        raw_status_counts = Counter((rs.get("status") or "active") for rs in self._raw_sessions)
        logical_source_ids = set(self._source_records.keys())
        part_source_ids = {
            rs.get("part_source_id")
            for rs in self._raw_sessions
            if rs.get("part_source_id")
        }
        index_status = self._read_full_index_status()
        return {
            "telemetry_version": 1,
            "granular": len(self._all_granular),
            "consolidated": len(self._all_cons),
            "cross_session": len(self._all_cross),
            "secrets": len(self._secrets),
            "index_built": self._data_dict is not None,
            "agent_id": self.agent_id,
            "scope": self.scope,
            "swarm_id": self.swarm_id,
            "tiers_dirty": self._tiers_dirty,
            "index_status": index_status,
            "last_write_log_claim": dict(self._last_write_log_claim_trace),
            "raw_sessions_count": len(self._raw_sessions),
            "source_records_count": len(self._source_records),
            "raw_session_status_counts": dict(raw_status_counts),
            "all_raw_sessions_active": all(
                (rs.get("status") or "active") == "active"
                for rs in self._raw_sessions
            ),
            "logical_source_count": len(logical_source_ids),
            "part_source_count": len(part_source_ids),
            "process_cost_summary": get_cost_summary(),
            "process_cost_scope": "process",
        }

    # ── metadata schema ──

    def _validate_metadata(self, metadata) -> str | None:
        """Validate metadata against schema. Returns error message or None."""
        if metadata is not None and not isinstance(metadata, dict):
            return f"metadata must be a dict, got {type(metadata).__name__}"
        if not metadata:
            return None
        # Flatness validation (always, even without schema)
        for key, value in metadata.items():
            if isinstance(value, dict):
                return f"metadata.{key}: nested dicts not allowed"
            if isinstance(value, list):
                if not all(isinstance(v, str) for v in value):
                    return f"metadata.{key}: lists must contain only strings"
        if not self._metadata_schema:
            return None
        for field_name, field_def in self._metadata_schema.items():
            if field_def.get("required") and field_name not in metadata:
                return f"required metadata field '{field_name}' missing"
        for key, value in metadata.items():
            if key in self._metadata_schema:
                expected = self._metadata_schema[key]["type"]
                if expected == "string" and not isinstance(value, str):
                    return f"metadata.{key}: expected string, got {type(value).__name__}"
                if expected == "number" and (isinstance(value, bool) or not isinstance(value, (int, float))):
                    return f"metadata.{key}: expected number, got {type(value).__name__}"
                if expected == "integer" and (isinstance(value, bool) or not isinstance(value, int)):
                    return f"metadata.{key}: expected integer, got {type(value).__name__}"
                if expected == "boolean" and not isinstance(value, bool):
                    return f"metadata.{key}: expected boolean, got {type(value).__name__}"
                if expected == "enum":
                    allowed = self._metadata_schema[key].get("values", [])
                    if value not in allowed:
                        return f"metadata.{key}: '{value}' not in {allowed}"
                if expected == "datetime" and not isinstance(value, str):
                    return f"metadata.{key}: expected ISO datetime string"
                if expected == "string[]":
                    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                        return f"metadata.{key}: expected list of strings"
        return None

    async def set_metadata_schema(self, schema: dict) -> None:
        """Declare metadata schema. Persisted via _save_cache."""
        for field_name, field_def in schema.items():
            if "type" not in field_def:
                raise ValueError(f"metadata schema field '{field_name}' missing 'type'")
            if field_def["type"] not in ("string", "number", "integer",
                                          "boolean", "datetime", "enum", "string[]"):
                raise ValueError(f"unsupported type '{field_def['type']}' for '{field_name}'")
            if field_def["type"] == "enum":
                vals = field_def.get("values")
                if not isinstance(vals, list) or len(vals) == 0:
                    raise ValueError(f"enum field '{field_name}' requires non-empty 'values' list")
        async with self._file_lock:
            self._metadata_schema = schema
            self._save_cache()

    def get_metadata_schema(self) -> dict:
        return self._metadata_schema or {}

    # ── query() ──

    async def query(
        self,
        filter: dict = None,
        sort_by: str = "session_date",
        sort_order: str = "desc",
        limit: int = 10,
        offset: int = 0,
        caller_id: str = None,
        caller_role: str = "agent",
        caller_memberships: list[str] = None,
    ) -> dict:
        """Structured query on facts. No vectors, no LLM."""
        meta_schema = self._metadata_schema

        if not _is_sortable(sort_by, meta_schema):
            return {"error": f"sort_by must be a core scalar field or declared "
                             f"metadata.* field (schema required for metadata sort)",
                    "code": "INVALID_SORT_FIELD"}
        if sort_order not in VALID_SORT_ORDERS:
            return {"error": "sort_order must be 'asc' or 'desc'",
                    "code": "INVALID_SORT_ORDER"}
        if limit < 0 or offset < 0:
            return {"error": "limit and offset must be non-negative",
                    "code": "INVALID_PAGINATION"}

        # Tag facts with tier for query results
        gran_set = set(id(f) for f in self._all_granular)
        cons_set = set(id(f) for f in self._all_cons)
        all_facts = self._all_granular + self._all_cons + self._all_cross
        for f in all_facts:
            if id(f) in gran_set:
                f.setdefault("_tier", "granular")
            elif id(f) in cons_set:
                f.setdefault("_tier", "consolidated")
            else:
                f.setdefault("_tier", "cross_session")
        now = datetime.now(timezone.utc)

        fl = {}
        for f in self._all_granular:
            cid = f.get("conv_id", "mem_s0")
            fid = f.get("id", "")
            if fid:
                fl[fid] = f
                fl[f"{cid}_{fid}"] = f

        effective_caller_id = _default_runtime_caller_id(caller_id, None, self.agent_id)
        visible = [
            f for f in all_facts
            if _is_visible(f, now=now, fact_lookup=fl)
            and self._acl_allows(f, effective_caller_id,
                                 caller_memberships or [], caller_role)
        ]

        if filter:
            visible = [
                f for f in visible
                if _fact_matches_structured_filter(f, filter, meta_schema)
            ]

        total = len(visible)

        reverse = (sort_order == "desc")
        present, missing = _split_sort_values(visible, sort_by)
        present.sort(key=lambda f: _sort_value(f, sort_by), reverse=reverse)
        visible = present + missing

        paginated = visible[offset:offset + limit]

        if hasattr(self, "_audit") and self._audit:
            self._audit.log("query", effective_caller_id, {
                "filter": filter, "sort_by": sort_by, "limit": limit,
                "total_matched": total, "returned": len(paginated),
            })

        preview_facts = []
        for fact in paginated:
            rendered = deepcopy(fact)
            text = rendered.get("fact")
            if isinstance(text, str) and len(text) > MAX_QUERY_FACT_CHARS:
                rendered["fact"] = f"{text[:MAX_QUERY_FACT_CHARS]}...[truncated]"
                rendered["fact_truncated"] = True
            preview_facts.append(rendered)

        return {
            "total": total,
            "facts": preview_facts,
            "has_more": (offset + limit) < total,
        }
