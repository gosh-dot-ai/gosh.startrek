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
from ..episode_packet import build_context_from_retrieved_facts, fact_episode_ids
from ..retrieval import source_local_fact_sweep
from ..runtime_contracts import QueryExecutorV1
from ..tuning import get_runtime_tuning


async def augment_conversation_structural_packet(
    server,
    *,
    query: str,
    packet: dict,
    episode_lookup: dict[str, dict],
    fact_filter,
) -> tuple[dict, list[dict] | None]:
    operator_plan = packet.get("query_operator_plan", {})
    if not any(
        operator_plan.get(name, {}).get("enabled", False)
        for name in ("commonality", "compare_diff", "list_set", "slot_query", "compositional")
    ):
        return packet, None

    selected_source_ids = {
        (episode_lookup.get(ep_id) or {}).get("source_id", "")
        for ep_id in packet.get("retrieved_episode_ids", [])
    }
    selected_source_ids.discard("")
    if not selected_source_ids:
        return packet, None

    selected_source_families = {
        (episode_lookup.get(ep_id) or {}).get("source_type", "")
        for ep_id in packet.get("retrieved_episode_ids", [])
    }
    selected_source_families.discard("")
    if selected_source_families != {"conversation"}:
        return packet, None

    tuning = get_runtime_tuning()
    operator_tuning = tuning["operators"]
    query_features = extract_query_features(query)
    operator_plan = query_features.get("operator_plan") or {}
    retrieval_target = query_features.get("retrieval_target") or query

    atomic_embs = (server._data_dict or {}).get("atomic_embs")
    cross_embs = (server._data_dict or {}).get("cross_embs")

    candidate_facts = []
    candidate_embeddings = []
    fact_lookup: dict[str, dict] = {}
    for facts, embeddings in ((server._all_granular, atomic_embs), (server._all_cross, cross_embs)):
        if not isinstance(embeddings, np.ndarray) or len(embeddings) != len(facts):
            continue
        for idx, fact in enumerate(facts):
            if not fact_filter(fact):
                continue
            source_id = fact.get("source_id") or (fact.get("metadata") or {}).get("episode_source_id", "")
            if source_id not in selected_source_ids:
                continue
            candidate_facts.append(fact)
            candidate_embeddings.append(embeddings[idx])
            fact_lookup[fact.get("id", "")] = fact
    if not candidate_facts:
        return packet, None

    query_embedding = await server._embed_query_with_runtime_secrets(retrieval_target)
    top_k = int(operator_tuning.get("conversation_structural_fact_sweep_top_k", 12))
    bm25_pool = int(operator_tuning.get("conversation_structural_fact_sweep_bm25_pool", 24))
    vector_pool = int(operator_tuning.get("conversation_structural_fact_sweep_vector_pool", 24))
    entity_pool = int(operator_tuning.get("conversation_structural_fact_sweep_entity_pool", 12))
    if operator_plan.get("compositional", {}).get("enabled", False):
        top_k = max(top_k, 24)
        bm25_pool = max(bm25_pool, 64)
        vector_pool = max(vector_pool, 64)
        entity_pool = max(entity_pool, 24)
    sweep = source_local_fact_sweep(
        retrieval_target,
        candidate_facts,
        np.asarray(candidate_embeddings),
        query_embedding=query_embedding,
        top_k=top_k,
        bm25_pool=bm25_pool,
        vector_pool=vector_pool,
        entity_pool=entity_pool,
        rrf_k=int(operator_tuning.get("conversation_structural_fact_sweep_rrf_k", 60)),
    )
    retrieved_facts = [row["fact"] for row in sweep.get("retrieved", [])]
    if not retrieved_facts:
        return packet, None

    if operator_plan.get("commonality", {}).get("enabled", False):
        from ..memory import _augment_commonality_facts

        commonality_extras = _augment_commonality_facts(query, retrieved_facts, candidate_facts, limit=6)
        if commonality_extras:
            existing_ids = {str(fact.get("id") or "") for fact in retrieved_facts}
            for fact in commonality_extras:
                fact_id = str(fact.get("id") or "")
                if fact_id and fact_id in existing_ids:
                    continue
                retrieved_facts.append(fact)
                if fact_id:
                    existing_ids.add(fact_id)

    context, actual_injected_episode_ids = build_context_from_retrieved_facts(
        retrieved_facts,
        episode_lookup,
        fact_lookup=fact_lookup,
        budget=int(packet.get("selector_config", {}).get("budget", 8000)),
        snippet_chars=int(packet.get("tuning_snapshot", {}).get("packet", {}).get("snippet_chars", 1200)),
        question=query,
        query_features=query_features,
    )

    packet = dict(packet)
    packet["context"] = context
    packet["actual_injected_episode_ids"] = actual_injected_episode_ids
    packet["retrieved_fact_ids"] = [fact.get("id", "") for fact in retrieved_facts]
    packet["fact_episode_ids"] = list(
        dict.fromkeys(
            episode_id
            for fact in retrieved_facts
            for episode_id in fact_episode_ids(fact)
            if episode_id
        )
    )
    packet["source_local_fact_sweep_trace"] = sweep.get("trace", {})
    return packet, retrieved_facts


class ConversationStructuralExecutor(QueryExecutorV1):
    name = "conversation_structural"
    priority = 400

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
        return await augment_conversation_structural_packet(
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
