# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

import asyncio

import pytest

from src.memory import MemoryServer, _derive_acl_from_scope, _normalize_identity

# ── Identity normalization ──

def test_normalize_valid():
    assert _normalize_identity("user:mitja") == "user:mitja"
    assert _normalize_identity("agent:default") == "agent:default"
    assert _normalize_identity("swarm:alpha") == "swarm:alpha"
    assert _normalize_identity("system") == "system"
    assert _normalize_identity("anonymous") == "anonymous"


def test_normalize_no_prefix_raises():
    with pytest.raises(ValueError):
        _normalize_identity("mitja")


def test_normalize_bad_prefix_raises():
    with pytest.raises(ValueError):
        _normalize_identity("admin:root")


def test_normalize_public_as_owner_raises():
    """agent:PUBLIC is valid in ACL lists but NOT as owner_id."""
    with pytest.raises(ValueError):
        _normalize_identity("agent:PUBLIC", allow_public=False)


def test_normalize_public_in_acl():
    """agent:PUBLIC is allowed in ACL context."""
    assert _normalize_identity("agent:PUBLIC", allow_public=True) == "agent:PUBLIC"


# ── ACL derivation from explicit ingress scope ──

def test_derive_agent_private():
    acl = _derive_acl_from_scope("agent-private", "alice", "default")
    assert acl["owner_id"] == "agent:alice"
    assert acl["read"] == []
    assert acl["write"] == []


def test_derive_swarm_shared_requires_named_swarm():
    with pytest.raises(ValueError):
        _derive_acl_from_scope("swarm-shared", "default", "default")


def test_derive_swarm_shared_named():
    acl = _derive_acl_from_scope("swarm-shared", "bob", "alpha")
    assert acl["owner_id"] == "agent:bob"
    assert acl["read"] == ["swarm:alpha"]
    assert acl["write"] == ["swarm:alpha"]


def test_derive_system_wide():
    acl = _derive_acl_from_scope("system-wide", "x", "y")
    assert acl["owner_id"] == "system"
    assert acl["read"] == ["agent:PUBLIC"]
    assert acl["write"] == ["agent:PUBLIC"]


def test_derive_no_scope_rejected():
    with pytest.raises(ValueError):
        _derive_acl_from_scope(None, "x", "y")


# ── ACL on sessions and facts ──

def _patch_all(monkeypatch, **overrides):
    async def mock_extract(**kwargs):
        sn = kwargs.get("session_num", 1)
        return ("conv", sn, "2024-06-01", [
            {"id": f"f{sn}_0", "fact": f"Fact {sn}", "kind": "event",
             "entities": [], "tags": [], "session": sn}], [])

    async def mock_consolidate(**kwargs):
        return ("conv", 1, "2024-06-01", [
            {"id": "c0", "fact": "Cons", "kind": "summary", "entities": [], "tags": []}])

    async def mock_cross(**kwargs):
        return ("conv", "e", [
            {"id": "x0", "fact": "Cross", "kind": "profile", "entities": [], "tags": []}])

    async def mock_embed(texts, **kw):
        import numpy as np
        return np.random.randn(len(texts), 3072).astype(np.float32)

    async def mock_embed_q(text, **kw):
        import numpy as np
        return np.random.randn(3072).astype(np.float32)

    monkeypatch.setattr("src.memory.extract_session", mock_extract)
    monkeypatch.setattr("src.memory.resolve_supersession", lambda f, l: None)
    monkeypatch.setattr("src.memory.embed_texts", mock_embed)
    monkeypatch.setattr("src.memory.embed_query", mock_embed_q)


def test_store_with_acl(tmp_path, monkeypatch):
    """Explicit scope on store deterministically stamps owner-only ACL."""
    _patch_all(monkeypatch)
    ms = MemoryServer(str(tmp_path), "conv1")
    result = asyncio.run(ms.store(
        "Hello",
        session_num=1,
        session_date="2024-06-01",
        agent_id="mitja",
        scope="agent-private",
    ))
    assert result["facts_extracted"] == 1

    rs = ms._raw_sessions[0]
    assert rs["owner_id"] == "agent:mitja"
    assert rs["read"] == []
    assert rs["write"] == []

    fact = ms._all_granular[0]
    assert fact["owner_id"] == "agent:mitja"
    assert fact["read"] == []


def test_store_missing_scope_rejected(tmp_path, monkeypatch):
    """Live direct store fails closed when scope is omitted."""
    _patch_all(monkeypatch)
    ms = MemoryServer(str(tmp_path), "conv2")
    with pytest.raises(ValueError):
        asyncio.run(ms.store("Hello", session_num=1, session_date="2024-06-01"))
    assert not ms._raw_sessions
    assert not ms._all_granular


def test_store_agent_private_derives_owner(tmp_path, monkeypatch):
    """agent-private store derives owner/read/write from scope."""
    _patch_all(monkeypatch)
    ms = MemoryServer(str(tmp_path), "conv3", agent_id="alice")
    asyncio.run(
        ms.store(
            "Hello",
            session_num=1,
            session_date="2024-06-01",
            scope="agent-private",
        )
    )

    fact = ms._all_granular[0]
    assert fact["owner_id"] == "agent:alice"
    assert fact["read"] == []
    assert fact["write"] == []
    assert ms._acl_allows(fact, "agent:alice", [], "user")
    assert not ms._acl_allows(fact, "agent:bob", [], "user")


def test_store_named_swarm_derives_swarm_acl(tmp_path, monkeypatch):
    """swarm-shared + named swarm derives swarm-scoped ACL."""
    _patch_all(monkeypatch)
    ms = MemoryServer(str(tmp_path), "conv3b", agent_id="alice", swarm_id="alpha")
    asyncio.run(ms.store(
        "Hello",
        session_num=1,
        session_date="2024-06-01",
        scope="swarm-shared",
    ))

    fact = ms._all_granular[0]
    assert fact["owner_id"] == "agent:alice"
    assert fact["read"] == ["swarm:alpha"]
    assert fact["write"] == ["swarm:alpha"]
