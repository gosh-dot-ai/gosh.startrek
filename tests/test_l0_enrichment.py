# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

import asyncio
import copy

import numpy as np
import pytest

from src.memory import MemoryServer, _needs_l0_enrichment

DIM = 3072


# ── Patches ──

def _patch_embed(monkeypatch):
    async def mock_embed(texts, **kw):
        return np.random.randn(len(texts), DIM).astype(np.float32)
    async def mock_eq(text, **kw):
        return np.random.randn(DIM).astype(np.float32)
    monkeypatch.setattr("src.memory.embed_texts", mock_embed)
    monkeypatch.setattr("src.memory.embed_query", mock_eq)
    monkeypatch.setattr("src.memory.resolve_supersession", lambda f, l: None)


def _make_raw(n=1):
    return [
        {"session_num": i, "session_date": f"2024-0{i}-01",
         "content": f"Raw session {i}", "speakers": "User and Assistant"}
        for i in range(1, n + 1)
    ]


# ── Tests ──


@pytest.mark.asyncio
async def test_complete_metadata_no_llm(tmp_path, monkeypatch):
    """Facts with kind+entities+tags -> classify_fact NOT called (count=0)."""
    _patch_embed(monkeypatch)

    call_count = 0

    async def mock_classify(text, model, call_fn):
        nonlocal call_count
        call_count += 1
        return {"kind": "action_item", "entities": ["x"], "tags": ["y"]}

    monkeypatch.setattr("src.librarian.classify_fact", mock_classify)

    server = MemoryServer(data_dir=str(tmp_path), key="complete_meta")
    facts = [
        {"id": "f_01", "fact": "Complete fact", "kind": "event",
         "entities": ["Alice"], "tags": ["test"], "session": 1},
    ]
    raw = _make_raw(1)
    result = await server.ingest_asserted_facts(
        facts=facts, raw_sessions=raw, scope="agent-private")

    assert "error" not in result
    assert call_count == 0


@pytest.mark.asyncio
async def test_missing_kind_enriched(tmp_path, monkeypatch):
    """Fact without kind -> classify_fact called, kind populated."""
    _patch_embed(monkeypatch)

    call_count = 0

    async def mock_classify(text, model, call_fn):
        nonlocal call_count
        call_count += 1
        return {"kind": "action_item", "entities": ["feature"], "tags": ["dev"]}

    monkeypatch.setattr("src.librarian.classify_fact", mock_classify)

    server = MemoryServer(data_dir=str(tmp_path), key="missing_kind")
    facts = [
        {"id": "f_01", "fact": "Needs enrichment", "session": 1},
    ]
    raw = _make_raw(1)
    result = await server.ingest_asserted_facts(
        facts=facts, raw_sessions=raw, scope="agent-private")

    assert "error" not in result
    assert call_count == 1
    # After enrichment + setdefault, kind should be populated
    f = server._all_granular[0]
    assert f["kind"] == "action_item"
    assert f["entities"] == ["feature"]
    assert f["tags"] == ["dev"]


@pytest.mark.asyncio
async def test_partial_metadata_merge(tmp_path, monkeypatch):
    """Fact with kind but no entities/tags -> kind preserved, entities/tags enriched."""
    _patch_embed(monkeypatch)

    async def mock_classify(text, model, call_fn):
        return {"kind": "decision", "entities": ["Bob"], "tags": ["meeting"]}

    monkeypatch.setattr("src.librarian.classify_fact", mock_classify)

    server = MemoryServer(data_dir=str(tmp_path), key="partial_meta")
    facts = [
        {"id": "f_01", "fact": "Has kind only", "kind": "preference", "session": 1},
    ]
    raw = _make_raw(1)
    result = await server.ingest_asserted_facts(
        facts=facts, raw_sessions=raw, scope="agent-private")

    assert "error" not in result
    f = server._all_granular[0]
    # kind="preference" is not "fact" (default), so merge_l1_metadata preserves it
    assert f["kind"] == "preference"
    # entities/tags should be enriched from classify_fact
    assert f["entities"] == ["Bob"]
    assert f["tags"] == ["meeting"]


@pytest.mark.asyncio
async def test_authoritative_tiers_untouched(tmp_path, monkeypatch):
    """Cons/cross without kind -> classify_fact NOT called (count=0).
    Only granular facts are enriched."""
    _patch_embed(monkeypatch)

    call_count = 0

    async def mock_classify(text, model, call_fn):
        nonlocal call_count
        call_count += 1
        return {"kind": "action_item", "entities": ["x"], "tags": ["y"]}

    monkeypatch.setattr("src.librarian.classify_fact", mock_classify)

    server = MemoryServer(data_dir=str(tmp_path), key="auth_tiers")
    # Granular facts are complete -> no enrichment
    facts = [
        {"id": "f_01", "fact": "Complete fact", "kind": "event",
         "entities": ["Alice"], "tags": ["test"], "session": 1},
        {"id": "f_02", "fact": "Another fact", "kind": "fact",
         "entities": ["Bob"], "tags": ["test"], "session": 1},
    ]
    # Cons/cross have no kind — but enrichment only targets granular facts list
    cons = [
        {"id": "c_01", "fact": "Consolidated without kind",
         "session": 1, "source_ids": ["f_01"]}
    ]
    cross = [
        {"id": "x_01", "fact": "Cross without kind",
         "entities": ["Alice"], "sessions": [1],
         "source_ids": ["f_02"]}
    ]
    raw = _make_raw(1)
    result = await server.ingest_asserted_facts(
        facts=facts, consolidated=cons, cross_session=cross, raw_sessions=raw,
        scope="agent-private")

    assert "error" not in result
    # classify_fact should NOT have been called because granular facts are complete
    assert call_count == 0


@pytest.mark.asyncio
async def test_no_model_defaults(tmp_path, monkeypatch):
    """Server with extract_model=None -> import works, defaults applied."""
    _patch_embed(monkeypatch)

    call_count = 0

    async def mock_classify(text, model, call_fn):
        nonlocal call_count
        call_count += 1
        return {"kind": "action_item", "entities": ["x"], "tags": ["y"]}

    monkeypatch.setattr("src.librarian.classify_fact", mock_classify)

    server = MemoryServer(data_dir=str(tmp_path), key="no_model",
                          extract_model=None)
    facts = [
        {"id": "f_01", "fact": "No model fact", "session": 1},
    ]
    raw = _make_raw(1)
    result = await server.ingest_asserted_facts(
        facts=facts, raw_sessions=raw, scope="agent-private")

    assert "error" not in result
    # No model -> enrichment skipped, classify_fact not called
    assert call_count == 0
    # Defaults should be applied by setdefault
    f = server._all_granular[0]
    assert f["kind"] == "fact"  # setdefault
    assert f["entities"] == []  # setdefault
    assert f["tags"] == []      # setdefault


@pytest.mark.asyncio
async def test_complexity_after_enrich(tmp_path, monkeypatch):
    """Enriched fact has _session_content_complexity."""
    _patch_embed(monkeypatch)

    async def mock_classify(text, model, call_fn):
        return {"kind": "action_item", "entities": ["feature"], "tags": ["dev"]}

    monkeypatch.setattr("src.librarian.classify_fact", mock_classify)

    server = MemoryServer(data_dir=str(tmp_path), key="complexity_test")
    facts = [
        {"id": "f_01", "fact": "Enrichable fact", "session": 1},
    ]
    raw = _make_raw(1)
    result = await server.ingest_asserted_facts(
        facts=facts, raw_sessions=raw, scope="agent-private")

    assert "error" not in result
    f = server._all_granular[0]
    assert "_session_content_complexity" in f
    assert isinstance(f["_session_content_complexity"], float)


@pytest.mark.asyncio
async def test_enrich_disabled(tmp_path, monkeypatch):
    """enrich_l0=False -> classify_fact NOT called."""
    _patch_embed(monkeypatch)

    call_count = 0

    async def mock_classify(text, model, call_fn):
        nonlocal call_count
        call_count += 1
        return {"kind": "action_item", "entities": ["x"], "tags": ["y"]}

    monkeypatch.setattr("src.librarian.classify_fact", mock_classify)

    server = MemoryServer(data_dir=str(tmp_path), key="enrich_disabled")
    facts = [
        {"id": "f_01", "fact": "Should not be enriched", "session": 1},
    ]
    raw = _make_raw(1)
    result = await server.ingest_asserted_facts(
        facts=facts, raw_sessions=raw, scope="agent-private", enrich_l0=False)

    assert "error" not in result
    assert call_count == 0
    # Defaults still applied by setdefault
    f = server._all_granular[0]
    assert f["kind"] == "fact"


def test_malformed_types_normalized(tmp_path, monkeypatch):
    """Malformed types (entities=string, tags=string) are fixed before storage."""
    _patch_embed(monkeypatch)
    server = MemoryServer(str(tmp_path), "malformed", extract_model=None)
    asyncio.run(server.ingest_asserted_facts(
        facts=[
            {"id": "f1", "fact": "Test", "session": 1,
             "kind": "event", "entities": "Alice", "tags": "x"},
            {"id": "f2", "fact": "Test2", "session": 1,
             "kind": "", "entities": 123, "tags": None},
        ],
        scope="agent-private",
        enrich_l0=False,
    ))
    f1 = next(f for f in server._all_granular if f["id"].endswith("_f1"))
    f2 = next(f for f in server._all_granular if f["id"].endswith("_f2"))

    # String entities → wrapped in list
    assert isinstance(f1["entities"], list)
    assert f1["entities"] == ["Alice"]
    # String tags → wrapped in list
    assert isinstance(f1["tags"], list)
    assert f1["tags"] == ["x"]

    # Empty kind → removed, setdefault fills "fact"
    assert f2["kind"] == "fact"
    # Non-list entities/tags → normalized
    assert isinstance(f2["entities"], list)
    assert isinstance(f2["tags"], list)


def test_malformed_types_normalized_cons_cross(tmp_path, monkeypatch):
    """Malformed types in cons/cross are also normalized."""
    _patch_embed(monkeypatch)
    server = MemoryServer(str(tmp_path), "malformed2", extract_model=None)
    asyncio.run(server.ingest_asserted_facts(
        facts=[{"id": "f1", "fact": "G", "session": 1,
                "kind": "fact", "entities": [], "tags": []}],
        consolidated=[{"id": "c1", "fact": "C", "session": 1,
                        "entities": "Bob", "tags": "y"}],
        cross_session=[{"id": "x1", "fact": "X",
                         "entities": "Carol", "tags": 42}],
        scope="agent-private",
        enrich_l0=False,
    ))
    cf = server._all_cons[0]
    assert isinstance(cf["entities"], list)
    assert cf["entities"] == ["Bob"]
    assert isinstance(cf["tags"], list)
    assert cf["tags"] == ["y"]

    xf = server._all_cross[0]
    assert isinstance(xf["entities"], list)
    assert xf["entities"] == ["Carol"]
    assert isinstance(xf["tags"], list)
