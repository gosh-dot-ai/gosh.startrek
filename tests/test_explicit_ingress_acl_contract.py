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
import pytest

import src.mcp_server as mcp_mod
from src.ingest import ingest_input
from tests._auth_helpers import bootstrap_harness
from tests._mcp_auth import auth_token_for_agent, auth_token_for_owner, install_test_verified_auth
from src.mcp_server import (
    memory_import as _memory_import,
    memory_ingest as _memory_ingest,
    memory_ingest_asserted_facts as _memory_ingest_asserted_facts,
    memory_ingest_document as _memory_ingest_document,
    memory_list,
    memory_store as _memory_store,
    memory_write as _memory_write,
)
from src.memory import MemoryServer

DIM = 3072


def _patch_embeddings(monkeypatch):
    async def _embed_texts(texts, **kwargs):
        return np.zeros((len(texts), DIM), dtype=np.float32)

    async def _embed_query(text, **kwargs):
        return np.zeros(DIM, dtype=np.float32)

    monkeypatch.setattr("src.memory.embed_texts", _embed_texts)
    monkeypatch.setattr("src.memory.embed_query", _embed_query)
    monkeypatch.setattr("src.memory.resolve_supersession", lambda facts, lookup: None)


def _patch_extract(monkeypatch, mode: str):
    async def _extract_session(**kwargs):
        sn = kwargs.get("session_num", 1)
        if mode == "error":
            raise RuntimeError("extract boom")
        if mode == "zero":
            return ("conv", sn, kwargs.get("session_date", "2024-06-01"), [], [])
        return (
            "conv",
            sn,
            kwargs.get("session_date", "2024-06-01"),
            [{
                "id": f"f{sn}",
                "fact": f"Fact {sn}",
                "kind": "fact",
                "entities": [],
                "tags": [],
                "session": sn,
            }],
            [],
        )

    async def _session_merge_stub(**kwargs):
        return ("conv", kwargs.get("sn", 1), "2024-06-01", [])

    async def _cross_merge_stub(**kwargs):
        return ("conv", kwargs.get("ename", "entity"), [])

    async def _source_aggregation(*args, **kwargs):
        return []

    async def _group_document(*args, **kwargs):
        source_id = args[1]
        source_date = args[3]
        blocks = args[4]
        raw_text = "\n".join(str(block.get("text") or "") for block in blocks) or "document body"
        return (
            [{
                "episode_id": f"{source_id}_e0001",
                "source_id": source_id,
                "source_date": source_date or "2024-06-01",
                "raw_text": raw_text,
                "topic_key": "doc",
                "state_label": "doc",
            }],
            "",
            "patched",
        )

    monkeypatch.setattr("src.memory.extract_session", _extract_session)
    monkeypatch.setattr("src.memory.group_document", _group_document)
    monkeypatch.setattr(MemoryServer, "_extract_source_aggregation_facts", _source_aggregation)


@pytest.fixture(autouse=True)
def _reset_state(tmp_path, monkeypatch):
    mcp_mod.data_dir = str(tmp_path)
    mcp_mod.registry.clear()
    mcp_mod.courier_registry.clear()
    mcp_mod.connections.clear()
    mcp_mod.sub_to_conn.clear()
    mcp_mod._active_connections.clear()
    install_test_verified_auth(monkeypatch)
    _patch_embeddings(monkeypatch)
    _patch_extract(monkeypatch, "success")
    yield tmp_path
    mcp_mod.registry.clear()


async def memory_store(*args, **kwargs):
    if "token" not in kwargs:
        kwargs["token"] = auth_token_for_agent(kwargs.get("agent_id"))
    return await _memory_store(*args, **kwargs)


async def memory_write(*args, **kwargs):
    if "token" not in kwargs:
        kwargs["token"] = auth_token_for_agent(kwargs.get("agent_id"))
    return await _memory_write(*args, **kwargs)


async def memory_ingest_document(*args, **kwargs):
    if "token" not in kwargs:
        kwargs["token"] = auth_token_for_agent(kwargs.get("agent_id"))
    return await _memory_ingest_document(*args, **kwargs)


async def memory_ingest(*args, **kwargs):
    if "token" not in kwargs:
        kwargs["token"] = auth_token_for_agent(kwargs.get("agent_id"))
    return await _memory_ingest(*args, **kwargs)


async def memory_ingest_asserted_facts(*args, **kwargs):
    if "token" not in kwargs:
        kwargs["token"] = auth_token_for_agent(kwargs.get("agent_id"))
    return await _memory_ingest_asserted_facts(*args, **kwargs)


async def memory_import(*args, **kwargs):
    if "auth_token" not in kwargs:
        kwargs["auth_token"] = auth_token_for_agent(kwargs.get("agent_id"))
    return await _memory_import(*args, **kwargs)


@pytest.mark.asyncio
async def test_live_mcp_ingress_requires_explicit_scope(monkeypatch):
    _patch_extract(monkeypatch, "success")

    results = [
        await memory_store(
            key="store_missing",
            content="hello",
            session_num=1,
            session_date="2024-06-01",
            agent_id="alice",
        ),
        await memory_write(
            key="write_missing",
            message_id="m1",
            session_id="s1",
            content="hello",
            content_family="chat",
            timestamp_ms=1,
            agent_id="alice",
        ),
        await memory_ingest_document(
            key="doc_missing",
            content="doc",
            source_id="DOC-1",
            agent_id="alice",
        ),
        await memory_ingest(
            key="ingest_missing",
            text="User: hi\nAssistant: ok",
            session_num=1,
            session_date="2024-06-01",
            agent_id="alice",
        ),
        await memory_ingest_asserted_facts(
            key="asserted_missing",
            facts=[{
                "id": "f1",
                "fact": "asserted",
                "kind": "fact",
                "entities": [],
                "tags": [],
                "session": 1,
            }],
            agent_id="alice",
        ),
        await memory_import(
            key="import_missing",
            source_format="text",
            content="import me",
            agent_id="alice",
        ),
    ]

    for result in results:
        assert result["code"] == "VALIDATION_ERROR"
        assert "scope" in result["error"].lower()


@pytest.mark.asyncio
async def test_direct_python_ingress_requires_explicit_scope(monkeypatch, tmp_path):
    _patch_extract(monkeypatch, "success")
    server = MemoryServer(str(tmp_path), "direct_missing")

    with pytest.raises(ValueError):
        await server.store("hello", 1, "2024-06-01")
    with pytest.raises(ValueError):
        await server.write(
            message_id="m1",
            session_id="s1",
            content="hello",
            content_family="chat",
            timestamp_ms=1,
        )
    with pytest.raises(ValueError):
        await server.ingest_document("doc", source_id="DOC-1")
    asserted = await server.ingest_asserted_facts(
        facts=[{
            "id": "f1",
            "fact": "asserted",
            "kind": "fact",
            "entities": [],
            "tags": [],
            "session": 1,
        }],
    )
    assert asserted["code"] == "VALIDATION_ERROR"
    assert "scope" in asserted["error"].lower()
    with pytest.raises(ValueError):
        await ingest_input(
            server,
            text="User: hi\nAssistant: ok",
            session_num=1,
            session_date="2024-06-01",
        )


@pytest.mark.asyncio
async def test_live_mcp_ingress_requires_authenticated_caller(monkeypatch):
    _patch_extract(monkeypatch, "success")

    results = [
        await _memory_store(
            key="auth_store",
            content="hello",
            session_num=1,
            session_date="2024-06-01",
            agent_id="alice",
            scope="agent-private",
        ),
        await _memory_write(
            key="auth_write",
            message_id="m1",
            session_id="s1",
            content="hello",
            content_family="chat",
            timestamp_ms=1,
            agent_id="alice",
            scope="agent-private",
        ),
        await _memory_ingest_document(
            key="auth_doc",
            content="doc",
            source_id="DOC-1",
            agent_id="alice",
            scope="agent-private",
        ),
        await _memory_ingest(
            key="auth_ingest",
            text="User: hi\nAssistant: ok",
            session_num=1,
            session_date="2024-06-01",
            agent_id="alice",
            scope="agent-private",
        ),
        await _memory_ingest_asserted_facts(
            key="auth_asserted",
            facts=[{
                "id": "f1",
                "fact": "asserted",
                "kind": "fact",
                "entities": [],
                "tags": [],
                "session": 1,
            }],
            agent_id="alice",
            scope="agent-private",
        ),
        await _memory_import(
            key="auth_import",
            source_format="text",
            content="import me",
            agent_id="alice",
            scope="agent-private",
        ),
    ]

    for result in results:
        assert result["code"] == "AUTH_REQUIRED"


@pytest.mark.asyncio
async def test_direct_python_explicit_owner_acl_is_allowed_without_authenticated_caller(monkeypatch, tmp_path):
    _patch_extract(monkeypatch, "success")
    server = MemoryServer(str(tmp_path), "direct_owner_acl")

    result = await server.ingest_asserted_facts(
        facts=[{
            "id": "f1",
            "fact": "asserted",
            "kind": "fact",
            "entities": [],
            "tags": [],
            "session": 1,
        }],
        raw_sessions=[{
            "session_num": 1,
            "session_date": "2024-06-01",
            "content": "asserted",
        }],
        scope="agent-private",
        owner_id="user:alice",
        read=[],
        write=[],
        enrich_l0=False,
    )

    assert "error" not in result
    assert server._raw_sessions[0]["owner_id"] == "user:alice"
    assert server._all_granular[0]["owner_id"] == "user:alice"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["zero", "error"])
async def test_system_wide_visibility_is_independent_of_extraction_outcome(monkeypatch, mode):
    _patch_extract(monkeypatch, mode)
    result = await memory_store(
        key=f"public_{mode}",
        content="public payload",
        session_num=1,
        session_date="2024-06-01",
        agent_id="owner",
        scope="system-wide",
    )
    assert "error" not in result

    server = mcp_mod.registry[f"public_{mode}"]
    assert server._instance_config["_derived_read"] == ["agent:PUBLIC"]
    outsider = await memory_list(key=f"public_{mode}", agent_id="outsider")
    assert outsider.get("code") not in ("FORBIDDEN", "ACL_FORBIDDEN")


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["success", "zero"])
async def test_agent_private_never_widens_even_when_extraction_changes(monkeypatch, mode):
    _patch_extract(monkeypatch, mode)
    result = await memory_store(
        key=f"private_{mode}",
        content="private payload",
        session_num=1,
        session_date="2024-06-01",
        agent_id="owner",
        scope="agent-private",
    )
    assert "error" not in result

    server = mcp_mod.registry[f"private_{mode}"]
    assert server._instance_config["_derived_read"] == []
    assert server._instance_config["_derived_write"] == []
    outsider = await memory_list(
        key=f"private_{mode}",
        agent_id="outsider",
        token=auth_token_for_agent("outsider"),
    )
    assert outsider.get("code") in ("FORBIDDEN", "ACL_FORBIDDEN")


@pytest.mark.asyncio
async def test_swarm_shared_requires_named_swarm_and_never_maps_default_to_public(monkeypatch):
    _patch_extract(monkeypatch, "success")
    harness = bootstrap_harness(monkeypatch, mcp_mod.data_dir, patch_extraction=False)
    alice = harness.issue("agent:alice", kind="agent")
    bob = harness.issue("agent:bob", kind="agent")
    mallory = harness.issue("agent:mallory", kind="agent")
    harness.create_swarm("alpha", "agent:alice")
    harness.grant(harness.admin_token, swarm_id="alpha", principal_id="agent:bob", role="member")

    invalid = await memory_store(
        key="swarm_invalid",
        content="shared payload",
        session_num=1,
        session_date="2024-06-01",
        agent_id="alice",
        scope="swarm-shared",
        swarm_id="default",
        token=alice,
    )
    assert invalid["code"] == "VALIDATION_ERROR"

    ok = await _memory_store(
        key="swarm_named",
        content="shared payload",
        session_num=1,
        session_date="2024-06-01",
        agent_id="alice",
        scope="swarm-shared",
        swarm_id="alpha",
        token=alice,
    )
    assert "error" not in ok
    server = mcp_mod.registry["swarm_named"]
    assert server._instance_config["_derived_read"] == ["swarm:alpha"]
    assert server._instance_config["_derived_write"] == ["swarm:alpha"]
    assert (
        await memory_list(
            key="swarm_named",
            agent_id="bob",
            swarm_id="alpha",
            token=bob,
        )
    ).get("code") not in ("FORBIDDEN", "ACL_FORBIDDEN")
    assert (
        await memory_list(
            key="swarm_named",
            agent_id="mallory",
            swarm_id="beta",
            token=mallory,
        )
    ).get("code") in ("FORBIDDEN", "ACL_FORBIDDEN")


@pytest.mark.asyncio
async def test_canonical_acl_inherits_across_root_and_derived_objects(monkeypatch, tmp_path):
    _patch_extract(monkeypatch, "success")
    harness = bootstrap_harness(monkeypatch, mcp_mod.data_dir, patch_extraction=False)
    alice = harness.issue("agent:alice", kind="agent")
    harness.create_swarm("alpha", "agent:alice")
    server = MemoryServer(str(tmp_path), "inherit")

    store_result = await server.store(
        "hello",
        session_num=1,
        session_date="2024-06-01",
        agent_id="alice",
        swarm_id="alpha",
        scope="swarm-shared",
    )
    assert store_result["facts_extracted"] == 1
    raw = server._raw_sessions[0]
    fact = server._all_granular[0]
    source = server._source_records["inherit"]
    for obj in (raw, fact, source):
        assert obj["scope"] == "swarm-shared"
        assert obj["agent_id"] == "alice"
        assert obj["swarm_id"] == "alpha"
        assert obj["owner_id"] == "agent:alice"
        assert obj["read"] == ["swarm:alpha"]
        assert obj["write"] == ["swarm:alpha"]

    await memory_write(
        key="inherit_write",
        message_id="m1",
        session_id="s1",
        content="queued",
        content_family="chat",
        timestamp_ms=1,
        agent_id="alice",
        swarm_id="alpha",
        scope="swarm-shared",
        token=alice,
    )
    queued = mcp_mod.registry["inherit_write"]
    entry = queued._storage.list_write_log_entries(states=["pending"], order="asc")[0]
    assert entry["scope"] == "swarm-shared"
    assert entry["owner_id"] == "agent:alice"
    assert entry["read"] == ["swarm:alpha"]
    assert entry["write"] == ["swarm:alpha"]
    processed = await queued.process_write_log_once()
    assert processed == 1
    assert queued._raw_sessions[0]["read"] == ["swarm:alpha"]
    assert queued._all_granular[0]["read"] == ["swarm:alpha"]

    doc_result = await server.ingest_document(
        "# Title\n\nBody",
        source_id="DOC-1",
        agent_id="alice",
        scope="system-wide",
    )
    assert doc_result["facts_extracted"] >= 0
    doc_source = server._source_records["DOC-1"]
    assert doc_source["scope"] == "system-wide"
    assert doc_source["owner_id"] == "system"
    assert doc_source["read"] == ["agent:PUBLIC"]
    assert doc_source["write"] == ["agent:PUBLIC"]
