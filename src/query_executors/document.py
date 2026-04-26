# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

from __future__ import annotations

from collections import Counter

from ..episode_features import extract_query_features
from ..episode_packet import (
    _fact_content_tokens,
    _pseudo_facts_from_episode,
    _select_bounded_chain_seed_facts,
    build_bounded_chain_candidate_bundle,
    build_context_from_retrieved_facts,
    fact_episode_ids,
)
from ..runtime_contracts import QueryExecutorV1
from ..tuning import get_runtime_tuning


async def augment_document_structural_packet(
    server,
    *,
    query: str,
    packet: dict,
    episode_lookup: dict[str, dict],
    fact_filter,
) -> tuple[dict, list[dict] | None]:
    operator_plan = packet.get("query_operator_plan", {})
    if not operator_plan.get("bounded_chain", {}).get("enabled", False):
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
    if selected_source_families != {"document"}:
        return packet, None

    tuning = get_runtime_tuning()
    operator_tuning = tuning["operators"]
    query_features = extract_query_features(query)
    retrieval_target = query_features.get("retrieval_target") or query
    structural_qf = extract_query_features(retrieval_target)

    candidate_facts = []
    fact_lookup: dict[str, dict] = {}
    pseudo_fact_lookup: dict[str, dict] = {}
    for facts in (server._all_granular, server._all_cross):
        for fact in facts:
            if not fact_filter(fact):
                continue
            source_id = fact.get("source_id") or (fact.get("metadata") or {}).get("episode_source_id", "")
            if source_id not in selected_source_ids:
                continue
            candidate_facts.append(fact)
            fact_lookup[fact.get("id", "")] = fact
    for ep_id in packet.get("retrieved_episode_ids", []):
        episode = episode_lookup.get(ep_id)
        if not episode or episode.get("source_id", "") not in selected_source_ids:
            continue
        for pseudo in _pseudo_facts_from_episode(ep_id, episode, qf=structural_qf):
            pseudo_fact_lookup[pseudo.get("id", "")] = pseudo
            candidate_facts.append(pseudo)
    if not candidate_facts:
        return packet, None

    seed_fact_count = max(0, int(operator_tuning.get("document_structural_seed_fact_count", 1)))
    query_specificity_bonus = float(
        packet.get("tuning_snapshot", {}).get("packet", {}).get("query_specificity_bonus", 0.0)
    )
    seed_facts = _select_bounded_chain_seed_facts(
        candidate_facts,
        structural_qf,
        token_freq=Counter(
            token
            for fact in candidate_facts
            for token in set(_fact_content_tokens(fact.get("fact", ""), structural_qf))
        ),
        query_specificity_bonus=query_specificity_bonus,
        seed_count=seed_fact_count,
    )
    seed_fact_ids = [fact.get("id", "") for fact in seed_facts if fact.get("id", "")]
    if not seed_facts:
        for fact_id in packet.get("retrieved_fact_ids", []):
            candidate_fact = fact_lookup.get(fact_id) or pseudo_fact_lookup.get(fact_id)
            if not isinstance(candidate_fact, dict):
                continue
            if not (candidate_fact.get("fact") or "").strip():
                continue
            seed_fact_ids.append(fact_id)
            seed_facts.append(candidate_fact)
            if len(seed_facts) >= seed_fact_count:
                break
    if not seed_facts:
        return packet, None

    bundle = build_bounded_chain_candidate_bundle(
        retrieval_target,
        seed_facts,
        candidate_facts,
        max_candidates=int(operator_tuning.get("document_structural_candidate_bundle_top_k", 18)),
        query_specificity_bonus=query_specificity_bonus,
    )
    retrieved_facts = bundle.get("facts", [])
    if not retrieved_facts:
        return packet, None

    context_trace: dict = {}
    context, actual_injected_episode_ids = build_context_from_retrieved_facts(
        retrieved_facts,
        episode_lookup,
        fact_lookup=fact_lookup,
        budget=int(packet.get("selector_config", {}).get("budget", 8000)),
        snippet_chars=int(packet.get("tuning_snapshot", {}).get("packet", {}).get("snippet_chars", 1200)),
        question=query,
        query_features=query_features,
        allow_multi_episode_snippets=False,
        context_trace=context_trace,
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
    packet["source_local_fact_sweep_trace"] = {
        **(bundle.get("trace", {}) or {}),
        "family": "document",
        "seed_fact_ids": seed_fact_ids,
    }
    if context_trace.get("document_target_span_ids") or context_trace.get("document_target_span_mode") != "disabled":
        packet.update(context_trace)
    return packet, retrieved_facts


class DocumentStructuralExecutor(QueryExecutorV1):
    name = "document_structural"
    priority = 300

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
        return await augment_document_structural_packet(
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
