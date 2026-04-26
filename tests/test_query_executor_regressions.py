# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

from __future__ import annotations

import asyncio

import numpy as np

from src.memory import MemoryServer


def _episode(episode_id: str, source_id: str, raw_text: str, *, family: str = "conversation") -> dict:
    return {
        "episode_id": episode_id,
        "source_id": source_id,
        "source_type": family,
        "source_date": "2024-06-01",
        "topic_key": "session_1",
        "state_label": "session",
        "currentness": "unknown",
        "raw_text": raw_text,
        "provenance": {"raw_span": [0, len(raw_text)]},
    }


async def _fake_embed_query(_text, **_kwargs):
    return np.array([1.0, 0.0], dtype=np.float32)


def test_semantic_fact_sweep_rescue_executor_preserves_existing_recall_behavior(tmp_path, monkeypatch):
    server = MemoryServer(str(tmp_path), "semantic_rescue_guard")
    episode_id = "conv_guard_e01"
    source_id = "conv_guard"
    server._episode_corpus = {
        "documents": [{
            "doc_id": f"conversation:{source_id}",
            "episodes": [_episode(episode_id, source_id, "Alice keeps the launch checklist in Notion.")],
        }],
    }
    server._all_granular = [
        {
            "id": "fact_relevant",
            "fact": "Alice keeps the launch checklist in Notion.",
            "kind": "fact",
            "entities": ["Alice", "Notion"],
            "source_id": source_id,
            "session": 1,
            "metadata": {"episode_id": episode_id, "episode_source_id": source_id},
        },
        {
            "id": "fact_irrelevant",
            "fact": "Alice likes herbal tea.",
            "kind": "fact",
            "entities": ["Alice"],
            "source_id": source_id,
            "session": 1,
            "metadata": {"episode_id": episode_id, "episode_source_id": source_id},
        },
    ]
    server._data_dict = {
        "atomic_embs": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    }
    monkeypatch.setattr("src.memory.embed_query", _fake_embed_query)

    packet = {
        "retrieved_fact_ids": [],
        "actual_injected_episode_ids": [],
        "retrieved_episode_ids": [],
        "retrieval_families": ["conversation"],
        "search_family": "conversation",
        "selector_config": {"budget": 4000},
        "tuning_snapshot": {"packet": {"snippet_chars": 600}},
    }
    episode_lookup = {episode_id: server._episode_corpus["documents"][0]["episodes"][0]}

    augmented, retrieved = asyncio.run(
        server._rescue_episode_packet_with_semantic_fact_sweep(
            query="Where does Alice keep the launch checklist?",
            packet=packet,
            episode_lookup=episode_lookup,
            fact_filter=lambda _fact: True,
        )
    )

    assert retrieved is not None
    assert [fact["id"] for fact in retrieved][:1] == ["fact_relevant"]
    assert "fact_relevant" in augmented["retrieved_fact_ids"]
    assert augmented["actual_injected_episode_ids"] == [episode_id]
    assert "launch checklist in Notion" in augmented["context"]


def test_semantic_fact_sweep_rescue_executor_is_noop_when_packet_is_sufficient(tmp_path):
    server = MemoryServer(str(tmp_path), "semantic_rescue_noop")
    episode_id = "conv_noop_e01"
    source_id = "conv_noop"
    packet = {
        "retrieved_fact_ids": ["fact_existing"],
        "actual_injected_episode_ids": [episode_id],
        "retrieval_families": ["conversation"],
        "search_family": "conversation",
    }
    episode_lookup = {episode_id: _episode(episode_id, source_id, "Existing context already present.")}

    augmented, retrieved = asyncio.run(
        server._rescue_episode_packet_with_semantic_fact_sweep(
            query="Where is the context?",
            packet=packet,
            episode_lookup=episode_lookup,
            fact_filter=lambda _fact: True,
        )
    )

    assert augmented == packet
    assert retrieved is None


def test_semantic_fact_sweep_rescue_executor_expands_when_packet_is_underfilled(tmp_path, monkeypatch):
    server = MemoryServer(str(tmp_path), "semantic_rescue_expand")
    episode_id = "doc_expand_e01"
    source_id = "DOC-EXPAND"
    server._episode_corpus = {
        "documents": [{
            "doc_id": f"document:{source_id}",
            "episodes": [_episode(
                episode_id,
                source_id,
                "Permit T-60 was granted on 2026-02-12 with an annual fee of 900 TKT.",
                family="document",
            )],
        }],
    }
    server._all_granular = [
        {
            "id": "fact_permit",
            "fact": "Permit T-60 was granted on 2026-02-12 with an annual fee of 900 TKT.",
            "kind": "fact",
            "entities": ["Permit T-60"],
            "source_id": source_id,
            "session": 1,
            "metadata": {"episode_id": episode_id, "episode_source_id": source_id},
        }
    ]
    server._data_dict = {
        "atomic_embs": np.array([[1.0, 0.0]], dtype=np.float32),
    }
    monkeypatch.setattr("src.memory.embed_query", _fake_embed_query)

    packet = {
        "retrieved_fact_ids": [],
        "actual_injected_episode_ids": [],
        "retrieved_episode_ids": [],
        "retrieval_families": ["document"],
        "search_family": "document",
        "selector_config": {"budget": 4000},
        "tuning_snapshot": {"packet": {"snippet_chars": 600}},
    }
    episode_lookup = {episode_id: server._episode_corpus["documents"][0]["episodes"][0]}

    augmented, retrieved = asyncio.run(
        server._rescue_episode_packet_with_semantic_fact_sweep(
            query="When was permit T-60 granted and what is the annual fee?",
            packet=packet,
            episode_lookup=episode_lookup,
            fact_filter=lambda _fact: True,
        )
    )

    assert retrieved is not None
    assert augmented["retrieved_fact_ids"] == ["fact_permit"]
    assert augmented["actual_injected_episode_ids"] == [episode_id]
    assert "annual fee of 900 TKT" in augmented["context"]
