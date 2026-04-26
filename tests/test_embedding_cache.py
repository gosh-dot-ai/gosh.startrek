# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

import asyncio
import json

import numpy as np
import pytest

from src.memory import MemoryServer, _embedding_fingerprint
from tests._memory_llm_mocks import patch_memory_llm_runtime

DIM = 3072


async def _store(ms: MemoryServer, *args, **kwargs):
    kwargs.setdefault("scope", "agent-private")
    return await ms.store(*args, **kwargs)


# ── Helpers ──

def _make_facts(texts):
    return [{"id": f"f{i}", "fact": t, "kind": "event", "entities": [],
             "tags": [], "session": 1, "scope": "swarm-shared",
             "agent_id": "default", "swarm_id": "default"} for i, t in enumerate(texts)]


def _patch_extraction(monkeypatch):
    async def mock_extract(**kwargs):
        sn = kwargs.get("session_num", 1)
        return ("conv", sn, "2024-06-01",
                _make_facts([f"fact {sn}-{i}" for i in range(3)]),
                [])

    async def mock_consolidate(**kwargs):
        return ("conv", 1, "2024-06-01",
                [{"id": "c0", "fact": "Consolidated fact", "kind": "summary",
                  "entities": [], "tags": []}])

    async def mock_cross(**kwargs):
        return ("conv", "entity", [{"id": "x0", "fact": "Cross fact",
                                     "kind": "profile", "entities": [], "tags": []}])

    monkeypatch.setattr("src.memory.extract_session", mock_extract)
    monkeypatch.setattr("src.memory.resolve_supersession", lambda f, l: None)
    patch_memory_llm_runtime(monkeypatch)


def _patch_embed_with_counter(monkeypatch):
    """Patch embed_texts with a call counter."""
    counter = {"calls": 0}

    async def mock_embed_texts(texts, **kwargs):
        counter["calls"] += 1
        return np.random.randn(len(texts), DIM).astype(np.float32)

    async def mock_embed_query(text, **kwargs):
        return np.random.randn(DIM).astype(np.float32)

    monkeypatch.setattr("src.memory.embed_texts", mock_embed_texts)
    monkeypatch.setattr("src.memory.embed_query", mock_embed_query)
    return counter


# ── Tests ──

def test_fingerprint_deterministic():
    """Same facts → same fingerprint."""
    facts = _make_facts(["alpha", "beta", "gamma"])
    assert _embedding_fingerprint(facts) == _embedding_fingerprint(facts)


def test_fingerprint_changes_on_text_change():
    """Different text (same count) → different fingerprint."""
    a = _make_facts(["alpha", "beta", "gamma"])
    b = _make_facts(["alpha", "beta", "CHANGED"])
    assert _embedding_fingerprint(a) != _embedding_fingerprint(b)


def test_fingerprint_changes_on_count_change():
    """Different count → different fingerprint."""
    a = _make_facts(["alpha", "beta"])
    b = _make_facts(["alpha", "beta", "gamma"])
    assert _embedding_fingerprint(a) != _embedding_fingerprint(b)


def test_build_index_caches_embeddings(tmp_path, monkeypatch):
    """Second build_index() reuses cached embeddings (no embed_texts call)."""
    _patch_extraction(monkeypatch)
    counter = _patch_embed_with_counter(monkeypatch)

    ms = MemoryServer(str(tmp_path), "conv1")
    asyncio.run(_store(ms, "Hello", session_num=1, session_date="2024-06-01"))
    asyncio.run(ms.build_index())
    calls_after_first = counter["calls"]
    assert calls_after_first > 0

    # Second build_index — should hit cache
    asyncio.run(ms.build_index())
    assert counter["calls"] == calls_after_first, "embed_texts called again — cache miss"


def test_build_index_reembeds_on_text_change(tmp_path, monkeypatch):
    """Changed fact text (same count) → re-embed (fingerprint mismatch)."""
    _patch_extraction(monkeypatch)
    counter = _patch_embed_with_counter(monkeypatch)

    ms = MemoryServer(str(tmp_path), "conv1")
    asyncio.run(_store(ms, "Hello", session_num=1, session_date="2024-06-01"))
    asyncio.run(ms.build_index())
    calls_first = counter["calls"]

    # Mutate a fact text (same count)
    ms._all_granular[0]["fact"] = "TOTALLY DIFFERENT TEXT"
    asyncio.run(ms.build_index())
    assert counter["calls"] > calls_first, "embed_texts NOT called after text change"


def test_build_index_reembeds_on_count_change(tmp_path, monkeypatch):
    """Added fact → re-embed (count mismatch)."""
    _patch_extraction(monkeypatch)
    counter = _patch_embed_with_counter(monkeypatch)

    ms = MemoryServer(str(tmp_path), "conv1")
    asyncio.run(_store(ms, "Hello", session_num=1, session_date="2024-06-01"))
    asyncio.run(ms.build_index())
    calls_first = counter["calls"]

    # Add a fact
    ms._all_granular.append({"id": "fnew", "fact": "New fact", "kind": "event",
                             "entities": [], "tags": [], "session": 2,
                             "scope": "swarm-shared"})
    asyncio.run(ms.build_index())
    assert counter["calls"] > calls_first, "embed_texts NOT called after count change"


def test_cache_survives_reload(tmp_path, monkeypatch):
    """New MemoryServer instance loads cached embeddings by fingerprint."""
    _patch_extraction(monkeypatch)
    counter = _patch_embed_with_counter(monkeypatch)

    ms = MemoryServer(str(tmp_path), "conv1")
    asyncio.run(_store(ms, "Hello", session_num=1, session_date="2024-06-01"))
    asyncio.run(ms.build_index())
    calls_first = counter["calls"]

    # New instance — should load from disk, no embed calls
    ms2 = MemoryServer(str(tmp_path), "conv1")
    asyncio.run(ms2.build_index())
    assert counter["calls"] == calls_first, "embed_texts called on reload — fingerprint cache broken"
