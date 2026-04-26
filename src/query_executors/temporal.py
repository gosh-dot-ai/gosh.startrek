# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

from __future__ import annotations

from typing import Any

import numpy as np

from ..episode_features import extract_query_features
from ..episode_packet import build_context_from_retrieved_facts
from ..retrieval import source_local_fact_sweep
from ..runtime_contracts import QueryExecutorV1
from ..tuning import get_tuning_section


async def repair_temporal_grounding_packet(
    server,
    *,
    query: str,
    packet: dict,
    episode_lookup: dict[str, dict],
    fact_filter,
) -> tuple[dict, list[dict] | None]:
    query_features = packet.get("query_features") or extract_query_features(query)
    if not query_features.get("operator_plan", {}).get("temporal_grounding", {}).get("enabled", False):
        return packet, None

    requested_dates = server._temporal_query_surface_dates(query)
    if not requested_dates:
        return packet, None
    context_lower = str(packet.get("context") or "").lower()
    if any(date_text in context_lower for date_text in requested_dates):
        return packet, None

    selected_episode_ids = list(dict.fromkeys(
        list(packet.get("retrieved_episode_ids") or [])
        + list(packet.get("actual_injected_episode_ids") or [])
        + list(packet.get("fact_episode_ids") or [])
    ))
    selected_source_ids = {
        str((episode_lookup.get(ep_id) or {}).get("source_id") or "").strip()
        for ep_id in selected_episode_ids
    }
    selected_source_ids.discard("")
    if not selected_source_ids:
        return packet, None

    atomic_embs = (server._data_dict or {}).get("atomic_embs")
    if not isinstance(atomic_embs, np.ndarray) or len(atomic_embs) != len(server._all_granular):
        return packet, None

    requested_families = set(packet.get("retrieval_families") or [])
    requested_search_family = packet.get("search_family", "auto")
    candidate_facts: list[dict] = []
    candidate_embeddings: list[np.ndarray] = []
    fact_lookup: dict[str, dict] = {}
    emb_width = int(atomic_embs.shape[1]) if len(atomic_embs.shape) > 1 else 1
    emb_dtype = atomic_embs.dtype
    for idx, fact in enumerate(server._all_granular):
        if not fact_filter(fact):
            continue
        metadata = fact.get("metadata") or {}
        episode_id = str(metadata.get("episode_id") or "").strip()
        if not episode_id or episode_id not in episode_lookup:
            continue
        source_id = str(fact.get("source_id") or metadata.get("episode_source_id") or "").strip()
        if source_id not in selected_source_ids:
            continue
        family = (
            server._source_records.get(source_id, {}).get("family")
            or (episode_lookup.get(episode_id) or {}).get("source_type", "")
        )
        if requested_search_family not in ("auto", "", None) and family and family != requested_search_family:
            continue
        if requested_families and family and family not in requested_families:
            continue
        candidate_facts.append(fact)
        candidate_embeddings.append(atomic_embs[idx])
        fact_lookup[str(fact.get("id") or "")] = fact

    pseudo_fact_lookup: dict[str, dict] = {}
    zero_vec = np.zeros((emb_width,), dtype=emb_dtype)
    for ep_id, episode in episode_lookup.items():
        source_id = str((episode or {}).get("source_id") or "").strip()
        if source_id not in selected_source_ids:
            continue
        for pseudo in server._temporal_grounding_pseudo_facts(ep_id, episode):
            pseudo_fact_lookup[str(pseudo.get("id") or "")] = pseudo
            candidate_facts.append(pseudo)
            candidate_embeddings.append(zero_vec.copy())
    if not candidate_facts:
        return packet, None

    retrieval_target = query_features.get("retrieval_target") or query
    search_queries = [retrieval_target]
    stripped_query = server._strip_temporal_surface(retrieval_target)
    if stripped_query and stripped_query not in search_queries:
        search_queries.append(stripped_query)

    rescue_cfg = get_tuning_section("retrieval").get("semantic_rescue", {})
    ranked_rows: list[dict] = []
    seen_fact_ids: set[str] = set()
    traces: list[dict] = []
    for search_query in search_queries:
        query_embedding = await server._embed_query_with_runtime_secrets(search_query)
        sweep = source_local_fact_sweep(
            search_query,
            candidate_facts,
            np.asarray(candidate_embeddings),
            query_embedding=query_embedding,
            top_k=max(12, int(rescue_cfg.get("top_k", 8))),
            bm25_pool=max(36, int(rescue_cfg.get("bm25_pool", 24))),
            vector_pool=max(24, int(rescue_cfg.get("vector_pool", 24))),
            entity_pool=max(12, int(rescue_cfg.get("entity_pool", 12))),
            rrf_k=int(rescue_cfg.get("rrf_k", 60)),
        )
        traces.append({"query": search_query, "trace": sweep.get("trace", {})})
        for row in sweep.get("retrieved", []):
            fact = row.get("fact") or {}
            fact_id = str(fact.get("id") or "").strip()
            if not fact_id or fact_id in seen_fact_ids:
                continue
            seen_fact_ids.add(fact_id)
            ranked_rows.append(row)
    if not ranked_rows:
        return packet, None

    retrieved_facts: list[dict[str, Any]] = [
        fact
        for row in ranked_rows
        for fact in [row.get("fact")]
        if isinstance(fact, dict)
    ]
    retrieved_episode_ids: list[str] = []
    seen_episode_ids: set[str] = set()
    selection_scores: list[dict] = []
    for row in ranked_rows:
        fact = row.get("fact") or {}
        episode_id = str((fact.get("metadata") or {}).get("episode_id") or "").strip()
        if not episode_id or episode_id in seen_episode_ids:
            continue
        seen_episode_ids.add(episode_id)
        retrieved_episode_ids.append(episode_id)
        selection_scores.append({"episode_id": episode_id, "score": float(row.get("score", 0.0))})

    merged_fact_lookup = dict(fact_lookup)
    merged_fact_lookup.update(pseudo_fact_lookup)
    context, actual_injected_episode_ids = build_context_from_retrieved_facts(
        retrieved_facts,
        episode_lookup,
        fact_lookup=merged_fact_lookup,
        budget=int(packet.get("selector_config", {}).get("budget", 8000)),
        snippet_chars=int(packet.get("tuning_snapshot", {}).get("packet", {}).get("snippet_chars", 1200)),
        question=query,
        query_features=query_features,
    )

    packet = dict(packet)
    packet["context"] = context
    packet["retrieved_fact_ids"] = [str(fact.get("id") or "") for fact in retrieved_facts]
    packet["retrieved_episode_ids"] = retrieved_episode_ids
    packet["actual_injected_episode_ids"] = actual_injected_episode_ids
    packet["fact_episode_ids"] = retrieved_episode_ids
    packet["selection_scores"] = selection_scores
    packet["source_local_fact_sweep_trace"] = {
        "mode": "temporal_grounding_repair",
        "requested_search_family": requested_search_family,
        "retrieval_families": sorted(requested_families),
        "queries": traces,
        "selected_source_ids": sorted(selected_source_ids),
    }
    family_first_pass_trace = dict(packet.get("family_first_pass_trace") or {})
    family_first_pass_trace["temporal_grounding_repair"] = {
        "candidate_count": len(candidate_facts),
        "selected_fact_count": len(retrieved_facts),
        "selected_source_ids": sorted(selected_source_ids),
    }
    packet["family_first_pass_trace"] = family_first_pass_trace
    return packet, retrieved_facts


class TemporalExecutor(QueryExecutorV1):
    name = "temporal"
    priority = 500

    def supports(self, query_type: str, search_family: str | None, packet: dict) -> bool:
        return True

    def should_skip(
        self,
        *,
        packet: dict,
        query: str,
        query_type: str,
        episode_lookup: dict[str, dict],
        augmented_facts,
    ) -> bool:
        return False

    async def augment(self, server, *, packet: dict, query: str, query_type: str, episode_lookup: dict[str, dict], fact_filter):
        return await repair_temporal_grounding_packet(
            server,
            query=query,
            packet=packet,
            episode_lookup=episode_lookup,
            fact_filter=fact_filter,
        )

    def should_halt(
        self,
        *,
        packet: dict,
        query: str,
        query_type: str,
        episode_lookup: dict[str, dict],
        augmented_facts,
    ) -> bool:
        temporal_trace = packet.get("temporal_trace") or {}
        temporal_matched = bool(
            temporal_trace.get("query_class") in {"ordinal", "calendar-answer"}
            and temporal_trace.get("matched")
            and not temporal_trace.get("fallback")
        )
        return temporal_matched or augmented_facts is not None
