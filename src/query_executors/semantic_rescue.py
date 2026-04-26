# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

from __future__ import annotations

import numpy as np

from ..episode_features import extract_query_features
from ..episode_packet import build_context_from_retrieved_facts
from ..retrieval import source_local_fact_sweep
from ..runtime_contracts import QueryExecutorV1
from ..tuning import get_tuning_section


async def rescue_episode_packet_with_semantic_fact_sweep(
    server,
    *,
    query: str,
    packet: dict,
    episode_lookup: dict[str, dict],
    fact_filter,
) -> tuple[dict, list[dict] | None]:
    if packet.get("retrieved_fact_ids") or packet.get("actual_injected_episode_ids"):
        return packet, None

    atomic_embs = (server._data_dict or {}).get("atomic_embs")
    if not isinstance(atomic_embs, np.ndarray) or len(atomic_embs) != len(server._all_granular):
        return packet, None

    query_features = extract_query_features(query)
    retrieval_target = query_features.get("retrieval_target") or query
    requested_families = set(packet.get("retrieval_families") or [])
    requested_search_family = packet.get("search_family", "auto")

    candidate_facts = []
    candidate_indices = []
    for idx, fact in enumerate(server._all_granular):
        if not fact_filter(fact):
            continue
        metadata = fact.get("metadata") or {}
        episode_id = metadata.get("episode_id", "")
        if not episode_id or episode_id not in episode_lookup:
            continue
        source_id = fact.get("source_id") or metadata.get("episode_source_id", "")
        family = (
            server._source_records.get(source_id, {}).get("family")
            or (episode_lookup.get(episode_id) or {}).get("source_type", "")
        )
        if requested_search_family not in ("auto", "", None) and family and family != requested_search_family:
            continue
        if requested_families and family and family not in requested_families:
            continue
        candidate_facts.append(fact)
        candidate_indices.append(idx)
    if not candidate_facts:
        return packet, None

    rescue_cfg = get_tuning_section("retrieval").get("semantic_rescue", {})
    query_embedding = await server._embed_query_with_runtime_secrets(retrieval_target)
    sweep = source_local_fact_sweep(
        retrieval_target,
        candidate_facts,
        atomic_embs[candidate_indices],
        query_embedding=query_embedding,
        top_k=int(rescue_cfg.get("top_k", 8)),
        bm25_pool=int(rescue_cfg.get("bm25_pool", 24)),
        vector_pool=int(rescue_cfg.get("vector_pool", 24)),
        entity_pool=int(rescue_cfg.get("entity_pool", 12)),
        rrf_k=int(rescue_cfg.get("rrf_k", 60)),
    )
    ranked_rows = sweep.get("retrieved", [])
    retrieved_facts = [row.get("fact", {}) for row in ranked_rows if row.get("fact")]
    if not retrieved_facts:
        return packet, None

    selection_scores = []
    retrieved_episode_ids = []
    seen_episode_ids = set()
    for row in ranked_rows:
        fact = row.get("fact") or {}
        episode_id = (fact.get("metadata") or {}).get("episode_id", "")
        if not episode_id or episode_id in seen_episode_ids:
            continue
        seen_episode_ids.add(episode_id)
        retrieved_episode_ids.append(episode_id)
        selection_scores.append({"episode_id": episode_id, "score": float(row.get("score", 0.0))})

    context, actual_injected_episode_ids = build_context_from_retrieved_facts(
        retrieved_facts,
        episode_lookup,
        fact_lookup={fact.get("id", ""): fact for fact in server._all_granular},
        budget=int(packet.get("selector_config", {}).get("budget", 8000)),
        snippet_chars=int(packet.get("tuning_snapshot", {}).get("packet", {}).get("snippet_chars", 1200)),
        question=query,
        query_features=query_features,
    )

    packet = dict(packet)
    packet["context"] = context
    packet["retrieved_fact_ids"] = [fact.get("id", "") for fact in retrieved_facts]
    packet["retrieved_episode_ids"] = retrieved_episode_ids
    packet["actual_injected_episode_ids"] = actual_injected_episode_ids
    packet["fact_episode_ids"] = retrieved_episode_ids
    packet["selection_scores"] = selection_scores
    packet["source_local_fact_sweep_trace"] = {
        **(sweep.get("trace", {}) or {}),
        "mode": "episode_semantic_rescue",
        "requested_search_family": requested_search_family,
        "retrieval_families": sorted(requested_families),
    }
    family_first_pass_trace = dict(packet.get("family_first_pass_trace") or {})
    family_first_pass_trace["semantic_rescue"] = {
        "candidate_count": len(candidate_facts),
        "selected_fact_count": len(retrieved_facts),
    }
    packet["family_first_pass_trace"] = family_first_pass_trace
    return packet, retrieved_facts


class SemanticRescueExecutor(QueryExecutorV1):
    name = "semantic_rescue"
    priority = 100

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
        return augmented_facts is not None

    async def augment(self, server, *, packet: dict, query: str, query_type: str, episode_lookup: dict[str, dict], fact_filter):
        return await rescue_episode_packet_with_semantic_fact_sweep(
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
