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

from src.identity import content_hash_text
from src.memory import MemoryServer
from tests._auth_helpers import bootstrap_harness

DIM = 3072


@pytest.fixture(autouse=True)
def _patch_llm(monkeypatch):
    async def mock_extract(**kwargs):
        sn = kwargs.get("session_num", 1)
        text = kwargs.get("session_text", "")
        facts = [
            {"id": f"f{sn}_{i}", "fact": f"{text[:40]} (fact {i})", "kind": "event",
             "entities": [], "tags": [], "session": sn}
            for i in range(3)
        ]
        return ("conv", sn, "2024-06-01", facts, [])

    async def mock_consolidate(**kwargs):
        return ("conv", 1, "2024-06-01", [])

    async def mock_cross(**kwargs):
        return ("conv", "e", [])

    monkeypatch.setattr("src.memory.extract_session", mock_extract)
    async def _aembed_texts(texts, **kw):
        return np.random.randn(len(texts), DIM).astype(np.float32)
    async def _aembed_query(text, **kw):
        return np.random.randn(DIM).astype(np.float32)
    monkeypatch.setattr("src.memory.embed_texts", _aembed_texts)
    monkeypatch.setattr("src.memory.embed_query", _aembed_query)
    monkeypatch.setattr("src.memory.resolve_supersession", lambda f, l: None)


@pytest.mark.asyncio
async def test_raw_session_stored_on_store(tmp_path):
    """store() must persist raw content before extraction."""
    server = MemoryServer(data_dir=str(tmp_path), key="raw_test")
    content = "User: I spent 70 hours playing Assassin's Creed Odyssey."
    result = await server.store(content, 1, "2024-01-01", scope="system-wide")
    assert len(server._raw_sessions) == 1
    assert server._raw_sessions[0]["content"] == content
    assert server._raw_sessions[0]["format"] == "conversation"
    assert server._raw_sessions[0]["extraction_format"] == "CONVERSATION"
    assert server._raw_sessions[0]["session_num"] == 1
    assert server._raw_sessions[0]["status"] == "active"
    assert result["extraction_format"] == "CONVERSATION"


@pytest.mark.asyncio
async def test_raw_session_stored_with_per_call_identity(tmp_path):
    """Raw session must record per-call agent_id/swarm_id/scope, not instance defaults."""
    server = MemoryServer(data_dir=str(tmp_path), key="raw_identity",
                          agent_id="default_agent", swarm_id="default_swarm")
    content = "User: I moved to Seattle last month."
    await server.store(content, 1, "2024-01-01",
                       agent_id="agent_x", swarm_id="sw1", scope="agent-private")
    assert server._raw_sessions[0]["agent_id"] == "agent_x"
    assert server._raw_sessions[0]["swarm_id"] == "sw1"
    assert server._raw_sessions[0]["scope"] == "agent-private"


@pytest.mark.asyncio
async def test_raw_sessions_survive_cache_roundtrip(tmp_path):
    """raw_sessions must survive save -> reload cycle."""
    server = MemoryServer(data_dir=str(tmp_path), key="raw_persist")
    content = "User: I have 38 pre-1920 American coins in my collection."
    await server.store(content, 1, "2024-01-01", scope="system-wide")

    server2 = MemoryServer(data_dir=str(tmp_path), key="raw_persist")
    assert len(server2._raw_sessions) == 1
    assert server2._raw_sessions[0]["content"] == content


@pytest.mark.asyncio
async def test_zero_fact_store_still_persists_raw_session_to_disk(tmp_path, monkeypatch):
    async def _zero_fact_extract(**kwargs):
        return ("conv", kwargs.get("session_num", 1), "2024-01-01", [], [])

    monkeypatch.setattr("src.memory.extract_session", _zero_fact_extract)

    server = MemoryServer(data_dir=str(tmp_path), key="raw_zero_fact")
    content = "User: giant multipart session that currently extracts no facts."

    result = await server.store(content, 1, "2024-01-01", scope="system-wide")

    assert result["facts_extracted"] == 0
    assert len(server._raw_sessions) == 1
    assert server._raw_sessions[0]["content"] == content
    assert server._raw_sessions[0]["status"] == "active"

    server2 = MemoryServer(data_dir=str(tmp_path), key="raw_zero_fact")
    assert len(server2._raw_sessions) == 1
    assert server2._raw_sessions[0]["content"] == content
    assert server2._raw_sessions[0]["status"] == "active"


@pytest.mark.asyncio
async def test_reextract_zero_fact_session_stays_active(tmp_path, monkeypatch):
    async def _zero_fact_extract(**kwargs):
        return ("conv", kwargs.get("session_num", 1), "2024-01-01", [], [])

    server = MemoryServer(data_dir=str(tmp_path), key="reextract_zero_fact_active")
    await server.store("User: I adopted a cat.", 1, "2024-01-01", scope="system-wide")

    monkeypatch.setattr("src.memory.extract_session", _zero_fact_extract)

    result = await server.reextract()

    assert result["sessions"] == 1
    assert result["reextracted"] == 0
    assert server._raw_sessions[0]["status"] == "active"


@pytest.mark.asyncio
async def test_reextract_preserves_explicit_extraction_format(tmp_path, monkeypatch):
    seen: list[str | None] = []

    async def _capture_extract(**kwargs):
        seen.append(kwargs.get("fmt"))
        sn = kwargs.get("session_num", 1)
        return (
            "conv",
            sn,
            kwargs.get("session_date", "2024-01-01"),
            [{
                "id": f"f{sn}",
                "fact": "trace fact",
                "kind": "event",
                "entities": [],
                "tags": [],
                "session": sn,
            }],
            [],
        )

    monkeypatch.setattr("src.memory.extract_session", _capture_extract)

    server = MemoryServer(data_dir=str(tmp_path), key="reextract_format")
    content = "$ pytest\nEXECUTION RESULT\n2 failed, 18 passed"
    stored = await server.store(
        content,
        1,
        "2024-01-01",
        scope="system-wide",
        content_format="CODE_TRACE",
    )

    assert stored["extraction_format"] == "CODE_TRACE"
    assert server._raw_sessions[0]["extraction_format"] == "CODE_TRACE"
    assert seen == ["CODE_TRACE"]

    seen.clear()
    result = await server.reextract()

    assert result["reextracted"] >= 1
    assert seen == ["CODE_TRACE"]
    assert server._raw_sessions[0]["extraction_format"] == "CODE_TRACE"

    server2 = MemoryServer(data_dir=str(tmp_path), key="reextract_format")
    assert server2._raw_sessions[0]["extraction_format"] == "CODE_TRACE"


@pytest.mark.asyncio
async def test_raw_session_stored_on_ingest_document(tmp_path):
    """ingest_document() must persist raw chunks."""
    server = MemoryServer(data_dir=str(tmp_path), key="raw_doc")
    content = "This is a short document about water infrastructure."
    await server.ingest_document(content, source_id="DOC-001", scope="system-wide")
    assert len(server._raw_sessions) >= 1
    assert server._raw_sessions[0]["format"] == "document"
    assert server._raw_sessions[0]["source_id"] == "DOC-001"


@pytest.mark.asyncio
async def test_document_raw_sessions_remain_distinct_across_different_sources(tmp_path):
    server = MemoryServer(data_dir=str(tmp_path), key="raw_doc_distinct")

    await server.ingest_document("Document one.", source_id="DOC-001", scope="system-wide")
    await server.ingest_document("Document two.", source_id="DOC-002", scope="system-wide")

    assert [(rs["source_id"], rs["session_num"]) for rs in server._raw_sessions[:2]] == [
        ("DOC-001", 1),
        ("DOC-002", 2),
    ]

    reloaded = MemoryServer(data_dir=str(tmp_path), key="raw_doc_distinct")
    assert [(rs["source_id"], rs["session_num"]) for rs in reloaded._raw_sessions[:2]] == [
        ("DOC-001", 1),
        ("DOC-002", 2),
    ]


@pytest.mark.asyncio
async def test_reextract_replaces_facts_preserves_raw(tmp_path):
    """reextract() must clear facts and re-extract; raw sessions unchanged."""
    server = MemoryServer(data_dir=str(tmp_path), key="reextract_test")
    content = "User: I have 38 pre-1920 American coins in my collection."
    await server.store(content, 1, "2024-01-01", scope="system-wide")
    original_raw = server._raw_sessions[0]["content"]

    result = await server.reextract()
    assert result["sessions"] == 1
    assert "reextracted" in result
    # Raw sessions unchanged
    assert len(server._raw_sessions) == 1
    assert server._raw_sessions[0]["content"] == original_raw


@pytest.mark.asyncio
async def test_reextract_restores_pending_raw_session_to_active(tmp_path):
    server = MemoryServer(data_dir=str(tmp_path), key="reextract_status")
    await server.store("User: I adopted a cat.", 1, "2024-01-01", scope="system-wide")
    server._raw_sessions[0]["status"] = "pending_reextract"

    result = await server.reextract()

    assert result["sessions"] == 1
    assert server._raw_sessions[0]["status"] == "active"


@pytest.mark.asyncio
async def test_reextract_preserves_private_fact_acl(tmp_path):
    server = MemoryServer(data_dir=str(tmp_path), key="reextract_acl")
    await server.store(
        "User: private medical note.",
        1,
        "2024-01-01",
        agent_id="alice",
        scope="agent-private",
    )

    before = server._all_granular[0]
    assert before["owner_id"] == "agent:alice"
    assert before["read"] == []
    assert before["write"] == []

    result = await server.reextract()

    assert result["sessions"] == 1
    assert server._all_granular
    for fact in server._all_granular:
        assert fact["owner_id"] == "agent:alice"
        assert fact["read"] == []
        assert fact["write"] == []


@pytest.mark.asyncio
async def test_reextract_fails_closed_for_legacy_raw_without_acl(tmp_path):
    server = MemoryServer(data_dir=str(tmp_path), key="reextract_missing_acl")
    server.scope = "swarm-shared"
    server._raw_sessions = [{
        "raw_session_id": "rs1",
        "session_num": 1,
        "session_date": "2024-01-01",
        "content": "legacy raw session",
        "speakers": "User and Assistant",
    }]

    result = await server.reextract()

    assert result["code"] == "VALIDATION_ERROR"
    assert "missing persisted ACL fields" in result["error"]
    assert server._all_granular == []


@pytest.mark.asyncio
async def test_zero_fact_retry_does_not_poison_dedup_or_supersede_old_version(tmp_path, monkeypatch):
    server = MemoryServer(data_dir=str(tmp_path), key="dedup_zero_fact")
    original = "User: I researched adoption agencies."
    updated = "User: I researched adoption agencies and family law."

    first = await server.store(original, 1, "2024-01-01", source_id="SRC-1", scope="system-wide")
    assert first["facts_extracted"] == 3
    dedup_key = server._source_versioning_key(
        source_id="SRC-1",
        family="conversation",
        scope="system-wide",
        owner_id="system",
        swarm_id="default",
        session_num=1,
    )
    original_version = server._dedup_index[dedup_key]["version_id"]
    original_hash = server._dedup_index[dedup_key]["content_hash"]

    async def _zero_fact_extract(**kwargs):
        return ("conv", kwargs.get("session_num", 1), "2024-01-01", [], [])

    monkeypatch.setattr("src.memory.extract_session", _zero_fact_extract)

    second = await server.store(updated, 1, "2024-01-02", source_id="SRC-1", scope="system-wide")
    assert second["facts_extracted"] == 0
    assert server._dedup_index[dedup_key]["version_id"] == original_version
    assert server._dedup_index[dedup_key]["content_hash"] == original_hash
    active_original = [
        rs for rs in server._raw_sessions
        if rs.get("version_id") == original_version and rs.get("status") == "active"
    ]
    assert active_original

    server2 = MemoryServer(data_dir=str(tmp_path), key="dedup_zero_fact")
    assert server2._dedup_index[dedup_key]["version_id"] == original_version
    assert server2._dedup_index[dedup_key]["content_hash"] == content_hash_text(original, family="conversation")

    async def _success_extract(**kwargs):
        sn = kwargs.get("session_num", 1)
        text = kwargs.get("session_text", "")
        facts = [
            {"id": f"f{sn}_{i}", "fact": f"{text[:40]} (fact {i})", "kind": "event",
             "entities": [], "tags": [], "session": sn}
            for i in range(3)
        ]
        return ("conv", sn, "2024-06-01", facts, [])

    monkeypatch.setattr("src.memory.extract_session", _success_extract)

    third = await server2.store(updated, 1, "2024-01-02", source_id="SRC-1", scope="system-wide")
    assert third["facts_extracted"] == 3
    assert server2._dedup_index[dedup_key]["content_hash"] == content_hash_text(updated, family="conversation")


@pytest.mark.asyncio
async def test_reextract_empty_returns_error(tmp_path):
    """reextract() on empty server returns error dict, not exception."""
    server = MemoryServer(data_dir=str(tmp_path), key="reextract_empty")
    result = await server.reextract()
    assert "error" in result


@pytest.mark.asyncio
async def test_mcp_memory_reextract_no_sessions(tmp_path):
    """MCP memory_reextract returns NO_RAW_SESSIONS code when empty."""
    import src.mcp_server as mcp_mod
    from _pytest.monkeypatch import MonkeyPatch
    monkeypatch = MonkeyPatch()
    auth = bootstrap_harness(monkeypatch, tmp_path)
    mcp_mod.data_dir = str(tmp_path)
    mcp_mod.registry.clear()
    mcp_mod.courier_registry.clear()
    mcp_mod.connections.clear()
    mcp_mod.sub_to_conn.clear()
    result = await mcp_mod.memory_reextract(key="empty_key", token=auth.admin_token)
    assert result.get("code") == "NO_RAW_SESSIONS"
    monkeypatch.undo()


@pytest.mark.asyncio
async def test_old_cache_without_raw_sessions_loads_cleanly(tmp_path):
    """Cache without raw_sessions field must load without error (backward compat)."""
    cache = {
        "granular": [], "cons": [], "cross": [], "tlinks": [],
        "n_sessions": 0, "n_sessions_with_facts": 0
        # intentionally no raw_sessions
    }
    (tmp_path / "compat_test.json").write_text(json.dumps(cache))
    server = MemoryServer(data_dir=str(tmp_path), key="compat_test")
    assert server._raw_sessions == []
