# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

from __future__ import annotations

import numpy as np

from ..coverage_recovery import (
    classify_coverage_query,
    compute_coverage_stats,
    merge_coverage_recovery_facts,
    needs_coverage_recovery,
)
from ..episode_features import extract_query_features
from ..episode_packet import build_context_from_retrieved_facts, fact_episode_ids
from ..retrieval import source_local_fact_sweep
from ..runtime_contracts import QueryExecutorV1


async def recover_multi_item_coverage_packet(
    server,
    *,
    query: str,
    packet: dict,
    episode_lookup: dict[str, dict],
    fact_filter,
) -> tuple[dict, list[dict] | None]:
    coverage_query_type = classify_coverage_query(query)
    if coverage_query_type == "none":
        return packet, None
    if (packet.get("temporal_trace") or {}).get("query_class") in {"ordinal", "calendar-answer", "calendar-seeking"}:
        return packet, None

    selected_source_ids = {
        (episode_lookup.get(ep_id) or {}).get("source_id", "")
        for ep_id in packet.get("retrieved_episode_ids", [])
    }
    selected_source_ids.discard("")
    selected_source_families = {
        (episode_lookup.get(ep_id) or {}).get("source_type", "")
        for ep_id in packet.get("retrieved_episode_ids", [])
    }
    selected_source_families.discard("")
    if not selected_source_ids:
        return packet, None

    current_fact_lookup: dict[str, dict] = {}
    for fact in server._all_granular + server._all_cross:
        fact_id = str(fact.get("id") or "").strip()
        if fact_id:
            current_fact_lookup[fact_id] = fact
    current_facts = [
        current_fact_lookup[fact_id]
        for fact_id in packet.get("retrieved_fact_ids", [])
        if fact_id in current_fact_lookup
    ]
    pre_stats = compute_coverage_stats(
        coverage_query_type,
        current_facts,
        selected_source_count=len(selected_source_ids),
    )
    if not needs_coverage_recovery(coverage_query_type, pre_stats):
        return packet, None

    requested_families = set(packet.get("retrieval_families") or [])
    requested_search_family = packet.get("search_family", "auto")
    query_features = extract_query_features(query)
    retrieval_target = query_features.get("retrieval_target") or query

    candidate_facts: list[dict] = []
    candidate_embeddings: list[np.ndarray] = []
    fact_lookup: dict[str, dict] = {}
    for facts, embeddings in (
        (server._all_granular, (server._data_dict or {}).get("atomic_embs")),
        (server._all_cross, (server._data_dict or {}).get("cross_embs")),
    ):
        if not isinstance(embeddings, np.ndarray) or len(embeddings) != len(facts):
            continue
        for idx, fact in enumerate(facts):
            if not fact_filter(fact):
                continue
            metadata = fact.get("metadata") or {}
            source_id = fact.get("source_id") or metadata.get("episode_source_id", "")
            if source_id not in selected_source_ids:
                continue
            episode_id = metadata.get("episode_id", "")
            family = (
                server._source_records.get(source_id, {}).get("family")
                or (episode_lookup.get(episode_id) or {}).get("source_type", "")
            )
            if requested_search_family not in ("auto", "", None) and family and family != requested_search_family:
                continue
            if requested_families and family and family not in requested_families:
                continue
            candidate_facts.append(fact)
            candidate_embeddings.append(embeddings[idx])
            fact_id = str(fact.get("id") or "").strip()
            if fact_id:
                fact_lookup[fact_id] = fact
    if not candidate_facts:
        return packet, None

    query_embedding = await server._embed_query_with_runtime_secrets(retrieval_target)
    base_target = max(
        len(current_facts) + 8,
        int(packet.get("selector_config", {}).get("supporting_facts_total", 12)),
        12,
    )
    sweep = source_local_fact_sweep(
        retrieval_target,
        candidate_facts,
        np.asarray(candidate_embeddings),
        query_embedding=query_embedding,
        top_k=base_target,
        bm25_pool=max(base_target * 2, 24),
        vector_pool=max(base_target * 2, 24),
        entity_pool=max(base_target, 12),
        rrf_k=60,
    )
    recovered_facts = [row.get("fact", {}) for row in sweep.get("retrieved", []) if row.get("fact")]
    if not recovered_facts:
        return packet, None

    merged_facts = merge_coverage_recovery_facts(
        coverage_query_type,
        current_facts,
        recovered_facts,
        max_facts=base_target,
    )
    post_stats = compute_coverage_stats(
        coverage_query_type,
        merged_facts,
        selected_source_count=len(selected_source_ids),
    )
    current_ids = [fact.get("id", "") for fact in current_facts]
    merged_ids = [fact.get("id", "") for fact in merged_facts]
    if merged_ids == current_ids:
        return packet, None
    if (
        post_stats.get("distinct_episodes", 0) <= pre_stats.get("distinct_episodes", 0)
        and post_stats.get("distinct_entities", 0) <= pre_stats.get("distinct_entities", 0)
        and post_stats.get("distinct_support_spans", 0) <= pre_stats.get("distinct_support_spans", 0)
    ):
        return packet, None

    context, actual_injected_episode_ids = build_context_from_retrieved_facts(
        merged_facts,
        episode_lookup,
        fact_lookup=fact_lookup,
        budget=int(packet.get("selector_config", {}).get("budget", 8000)),
        snippet_chars=int(packet.get("tuning_snapshot", {}).get("packet", {}).get("snippet_chars", 1200)),
        question=query,
        query_features=packet.get("query_features") or query_features,
    )
    merged_episode_ids = list(
        dict.fromkeys(
            episode_id
            for fact in merged_facts
            for episode_id in fact_episode_ids(fact)
            if episode_id
        )
    )
    packet = dict(packet)
    packet["context"] = context
    packet["retrieved_fact_ids"] = merged_ids
    packet["fact_episode_ids"] = merged_episode_ids
    packet["actual_injected_episode_ids"] = actual_injected_episode_ids
    packet["coverage_recovery_trace"] = {
        "query_type": coverage_query_type,
        "selected_source_ids": sorted(selected_source_ids),
        "selected_source_families": sorted(selected_source_families),
        "pre_stats": pre_stats,
        "post_stats": post_stats,
        "candidate_count": len(candidate_facts),
        "selected_fact_count": len(merged_facts),
        "sweep": sweep.get("trace", {}),
    }
    family_first_pass_trace = dict(packet.get("family_first_pass_trace") or {})
    family_first_pass_trace["coverage_recovery"] = {
        "query_type": coverage_query_type,
        "pre_stats": pre_stats,
        "post_stats": post_stats,
        "candidate_count": len(candidate_facts),
    }
    packet["family_first_pass_trace"] = family_first_pass_trace
    return packet, merged_facts


class CoverageRecoveryExecutor(QueryExecutorV1):
    name = "coverage_recovery"
    priority = 200

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
        return await recover_multi_item_coverage_packet(
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
        return False
