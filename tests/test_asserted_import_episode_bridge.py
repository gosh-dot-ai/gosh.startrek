# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

from __future__ import annotations

import numpy as np
import pytest

from src.episode_packet import fact_episode_ids
from src.memory import MemoryServer

DIM = 3072


def _patch_embed(monkeypatch):
    async def mock_embed(texts, **_kw):
        return np.zeros((len(texts), DIM), dtype=np.float32)

    async def mock_eq(_text, **_kw):
        return np.zeros(DIM, dtype=np.float32)

    monkeypatch.setattr("src.memory.embed_texts", mock_embed)
    monkeypatch.setattr("src.memory.embed_query", mock_eq)
    monkeypatch.setattr("src.memory.resolve_supersession", lambda _facts, _legacy: None)


def _facts_for_session(session_num: int, *, count: int = 2, prefix: str = "f") -> list[dict]:
    return [
        {
            "id": f"{prefix}{session_num:02d}_{idx}",
            "fact": f"Neutral fact {idx} from session {session_num}",
            "kind": "event",
            "entities": ["Alice"],
            "tags": [],
            "session": session_num,
        }
        for idx in range(count)
    ]


def _raw_session(
    session_num: int,
    *,
    session_date: str,
    content: str,
    episode_id: str | None = None,
    source_id: str | None = None,
) -> dict:
    row: dict = {
        "session_num": session_num,
        "session_date": session_date,
        "content": content,
        "speakers": "User and Assistant",
    }
    if episode_id is not None:
        row["episode_id"] = episode_id
    if source_id is not None:
        row["source_id"] = source_id
    return row


@pytest.mark.asyncio
async def test_asserted_import_creates_visible_episode_runtime(tmp_path, monkeypatch):
    _patch_embed(monkeypatch)
    server = MemoryServer(data_dir=str(tmp_path), key="bridge_visible")

    result = await server.ingest_asserted_facts(
        facts=_facts_for_session(1),
        raw_sessions=[_raw_session(1, session_date="2024-05-15", content="Alice talked.")],
        scope="agent-private",
    )

    assert result["raw_sessions_added"] == 1
    runtime = server._visible_episode_runtime(lambda _fact: True)
    assert runtime is not None
    corpus, episode_lookup, facts_by_episode, _bm25 = runtime
    assert corpus["documents"]
    assert episode_lookup
    assert facts_by_episode


@pytest.mark.asyncio
async def test_asserted_import_facts_carry_episode_metadata(tmp_path, monkeypatch):
    _patch_embed(monkeypatch)
    server = MemoryServer(data_dir=str(tmp_path), key="bridge_meta")

    await server.ingest_asserted_facts(
        facts=_facts_for_session(1, count=3),
        raw_sessions=[_raw_session(1, session_date="2024-05-15", content="Alice talked.")],
        scope="agent-private",
    )

    for fact in server._all_granular:
        assert fact_episode_ids(fact)
        assert fact["metadata"]["episode_source_id"] == "bridge_meta"


@pytest.mark.asyncio
async def test_asserted_import_temporal_span_has_bridgeable_episode_id(tmp_path, monkeypatch):
    _patch_embed(monkeypatch)
    server = MemoryServer(data_dir=str(tmp_path), key="bridge_tindex")
    facts = [
        {
            "id": "f01_0",
            "fact": "Alice signed a neutral agreement on May 15, 2024.",
            "kind": "event",
            "entities": ["Alice"],
            "tags": [],
            "session": 1,
            "event_date": "2024-05-15",
        }
    ]

    await server.ingest_asserted_facts(
        facts=facts,
        raw_sessions=[
            _raw_session(
                1,
                session_date="2024-05-15",
                content="Alice signed a neutral agreement on May 15, 2024.",
            )
        ],
        scope="agent-private",
    )
    server._rebuild_temporal_index()

    known_episode_ids = {
        ep.get("episode_id")
        for doc in server._episode_corpus.get("documents", [])
        for ep in doc.get("episodes", [])
    }
    spans = server._build_temporal_text_spans()
    bridgeable = [
        span
        for span in spans
        if str((span.get("payload") or {}).get("episode_id") or "").strip()
    ]
    assert bridgeable
    assert all((span.get("payload") or {})["episode_id"] in known_episode_ids for span in bridgeable)


@pytest.mark.asyncio
async def test_asserted_import_preserves_acl_status_and_existing_episode_id(tmp_path, monkeypatch):
    _patch_embed(monkeypatch)
    server = MemoryServer(data_dir=str(tmp_path), key="bridge_acl")
    custom_id = "custom_source_e0001"

    await server.ingest_asserted_facts(
        facts=_facts_for_session(1),
        raw_sessions=[
            _raw_session(
                1,
                session_date="2024-05-15",
                content="Alice talked.",
                episode_id=custom_id,
            )
        ],
        scope="agent-private",
        agent_id="agent-1",
        swarm_id="swarm-1",
        owner_id="owner-1",
        read=["reader-a"],
        write=["writer-a"],
    )

    episode = server._episode_corpus["documents"][0]["episodes"][0]
    assert episode["episode_id"] == custom_id
    assert episode["agent_id"] == "agent-1"
    assert episode["swarm_id"] == "swarm-1"
    assert episode["scope"] == "agent-private"
    assert episode["owner_id"] == "owner-1"
    assert episode["read"] == ["reader-a"]
    assert episode["write"] == ["writer-a"]
    assert episode["status"] == "active"
    assert server._raw_sessions[0]["episode_id"] == custom_id
    assert all(fact["metadata"]["episode_id"] == custom_id for fact in server._all_granular)


@pytest.mark.asyncio
async def test_asserted_import_does_not_fabricate_event_date_from_session_date(tmp_path, monkeypatch):
    _patch_embed(monkeypatch)
    server = MemoryServer(data_dir=str(tmp_path), key="bridge_no_fabrication")
    facts = _facts_for_session(1, count=2)

    await server.ingest_asserted_facts(
        facts=facts,
        raw_sessions=[_raw_session(1, session_date="2024-05-15", content="Alice talked.")],
        scope="agent-private",
    )

    for fact in server._all_granular:
        assert fact.get("event_date") in (None, "")


@pytest.mark.asyncio
async def test_asserted_import_groups_by_resolved_source_id(tmp_path, monkeypatch):
    _patch_embed(monkeypatch)
    server = MemoryServer(data_dir=str(tmp_path), key="bridge_multi")

    await server.ingest_asserted_facts(
        facts=_facts_for_session(1, prefix="a") + _facts_for_session(2, prefix="b"),
        raw_sessions=[
            _raw_session(1, session_date="2024-05-15", content="From source A.", source_id="conv_a"),
            _raw_session(2, session_date="2024-05-16", content="From source B.", source_id="conv_b"),
        ],
        scope="agent-private",
    )

    docs = {doc["doc_id"]: doc for doc in server._episode_corpus["documents"]}
    assert "conversation:conv_a" in docs
    assert "conversation:conv_b" in docs
    for doc_id, doc in docs.items():
        expected_source = doc_id.split(":", 1)[1]
        for episode in doc["episodes"]:
            assert episode["source_id"] == expected_source


@pytest.mark.asyncio
async def test_asserted_import_same_source_preserves_prior_episodes(tmp_path, monkeypatch):
    _patch_embed(monkeypatch)
    server = MemoryServer(data_dir=str(tmp_path), key="bridge_repeat")

    await server.ingest_asserted_facts(
        facts=_facts_for_session(1, prefix="first"),
        raw_sessions=[_raw_session(1, session_date="2024-05-15", content="First neutral session.")],
        scope="agent-private",
    )
    first_episode_id = server._raw_sessions[0]["episode_id"]

    await server.ingest_asserted_facts(
        facts=_facts_for_session(1, prefix="second"),
        raw_sessions=[_raw_session(1, session_date="2024-05-16", content="Second neutral session.")],
        scope="agent-private",
    )
    second_episode_id = server._raw_sessions[1]["episode_id"]

    assert first_episode_id != second_episode_id
    doc = next(
        doc
        for doc in server._episode_corpus["documents"]
        if doc["doc_id"] == "conversation:bridge_repeat"
    )
    episode_ids = {episode["episode_id"] for episode in doc["episodes"]}
    assert first_episode_id in episode_ids
    assert second_episode_id in episode_ids

    runtime = server._visible_episode_runtime(lambda _fact: True)
    assert runtime is not None
    _corpus, episode_lookup, facts_by_episode, _bm25 = runtime
    assert first_episode_id in episode_lookup
    assert second_episode_id in episode_lookup
    assert first_episode_id in facts_by_episode
    assert second_episode_id in facts_by_episode
    fact_episode_refs = {
        fact["metadata"]["episode_id"]
        for fact in server._all_granular
    }
    assert {first_episode_id, second_episode_id}.issubset(fact_episode_refs)
