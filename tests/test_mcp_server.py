# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import threading
import time
from typing import Any

from httpx import ASGITransport, AsyncClient
import numpy as np
import pytest
from starlette.testclient import TestClient

import src.mcp_server as mod
from src.local_cli_backend import LocalCliTimeoutError
from src.mcp_server import (
    _get_memory,
    courier_subscribe,
    courier_unsubscribe,
    get_more_context,
    mcp,
    memory_ask,
    memory_build_index,
    memory_flush,
    memory_get,
    memory_get_schema,
    memory_ingest,
    memory_ingest_document,
    memory_list,
    memory_migrate_jsonnpz,
    memory_plan_inference,
    memory_query,
    memory_recall,
    memory_reextract,
    memory_write,
    memory_write_status,
    memory_stats,
    memory_store,
    sse_cleanup,
    sse_endpoint,
)
from tests._auth_helpers import bootstrap_harness

DIM = 3072
_AUTH = None
_TOKEN_CACHE: dict[str, str] = {}


def test_memory_recall_docstring_describes_iterative_codebase_recall():
    doc = memory_recall.__doc__ or ""
    assert "iterative memory tool" in doc
    assert "call memory_recall again" in doc
    assert 'search_family="codebase"' in doc
    assert "exact files, symbols, config entries" in doc
    assert "previous patch attempts" in doc
    assert "verification state" in doc
    assert "Do not generate code changes from absent evidence" in doc
    assert "lookup: exact file/symbol/config/test lookup" in doc
    assert "Do not generate code changes from absent evidence" in doc
    assert "lookup: exact file/symbol/config/test lookup" in doc


def test_startup_log_lines_redact_server_token():
    token = "live-token-that-must-not-leak"
    lines = mod.startup_log_lines(
        title="gosh.memory MCP Server",
        listening="http://127.0.0.1:8765",
        data_dir="/data",
        embeddings="openai / text-embedding-3-large",
        tls="off",
        token=token,
        token_path="/root/.gosh-memory/token",
    )
    rendered = "\n".join(lines)

    assert token not in rendered
    assert "Token fingerprint: sha256:" in rendered
    assert "Token saved to: /root/.gosh-memory/token" in rendered
    assert "%(asctime)s" in mod.STARTUP_LOG_FORMAT


def test_token_fingerprint_is_stable_and_non_secret():
    token = "another-live-token"
    first = mod.token_fingerprint(token)
    second = mod.token_fingerprint(token)

    assert first == second
    assert first.startswith("sha256:")
    assert token not in first


# ── Patches ──

def _patch_extraction(monkeypatch):
    async def mock_extract_session(**kwargs):
        sn = kwargs.get("session_num", 1)
        facts = [
            {"id": f"f{i}", "fact": f"Fact {i}", "kind": "event",
             "entities": ["Alice"], "tags": ["test"], "session": sn}
            for i in range(3)
        ]
        return ("conv", sn, "2024-06-01", facts, [])

    async def mock_session_merge_stub(**kwargs):
        return ("conv", 1, "2024-06-01", [
            {"id": "c0", "fact": "Consolidated", "kind": "summary",
             "entities": ["Alice"], "tags": []}
        ])

    async def mock_cross_merge_stub(**kwargs):
        return ("conv", "alice", [
            {"id": "x0", "fact": "Cross-session", "kind": "profile",
             "entities": ["Alice"], "tags": []}
        ])

    async def mock_call_extract(*args, **kwargs):
        return {}

    async def mock_call_oai(*args, **kwargs):
        return "test answer"

    async def mock_call_model(*args, **kwargs):
        return "test answer"

    monkeypatch.setattr("src.memory.extract_session", mock_extract_session)
    monkeypatch.setattr("src.memory.call_extract", mock_call_extract)
    monkeypatch.setattr("src.common.call_extract", mock_call_extract)
    monkeypatch.setattr("src.memory.call_oai", mock_call_oai)
    monkeypatch.setattr("src.common.call_oai", mock_call_oai)
    monkeypatch.setattr("src.memory._call_model", mock_call_model)
    monkeypatch.setattr("src.common._call_model", mock_call_model)


def _patch_embeddings(monkeypatch):
    async def mock_embed_texts(texts, **kw):
        return np.random.randn(len(texts), DIM).astype(np.float32)

    async def mock_embed_query(text, **kw):
        return np.random.randn(DIM).astype(np.float32)

    monkeypatch.setattr("src.memory.embed_texts", mock_embed_texts)
    monkeypatch.setattr("src.memory.embed_query", mock_embed_query)


def _patch_resolve_supersession(monkeypatch):
    monkeypatch.setattr("src.memory.resolve_supersession", lambda f, l: None)


def _patch_all(monkeypatch):
    _patch_extraction(monkeypatch)
    _patch_embeddings(monkeypatch)
    _patch_resolve_supersession(monkeypatch)


def _patch_extract_capture_format(monkeypatch):
    seen: dict[str, Any] = {}

    async def mock_extract_session(**kwargs):
        seen["fmt"] = kwargs.get("fmt")
        sn = kwargs.get("session_num", 1)
        facts = [{
            "id": "f0",
            "fact": "captured",
            "kind": "event",
            "entities": ["Alice"],
            "tags": ["test"],
            "session": sn,
        }]
        return ("conv", sn, kwargs.get("session_date", "2024-06-01"), facts, [])

    monkeypatch.setattr("src.memory.extract_session", mock_extract_session)
    return seen


@pytest.fixture(autouse=True)
def reset_state(tmp_path, monkeypatch):
    """Reset module state before each test."""
    global _AUTH
    mod.data_dir = str(tmp_path)
    monkeypatch.setenv("GOSH_MEMORY_ALLOW_PLAINTEXT_SECRETS", "1")
    mod.registry.clear()
    mod.courier_registry.clear()
    mod.connections.clear()
    mod.sub_to_conn.clear()
    mod._active_connections.clear()
    _patch_all(monkeypatch)
    _AUTH = bootstrap_harness(monkeypatch, tmp_path)
    _TOKEN_CACHE.clear()
    yield
    # Stop couriers — set flag directly (event loop is closed after asyncio.run)
    for c in mod.courier_registry.values():
        c._running = False


def _principal_token(name: str = "test-admin", kind: str = "agent") -> str:
    if name == "admin":
        return _AUTH.admin_token
    principal_id = name if ":" in name else f"{kind}:{name}"
    token = _TOKEN_CACHE.get(principal_id)
    if token is None:
        token = _AUTH.issue(principal_id, kind=kind)
        _TOKEN_CACHE[principal_id] = token
    return token


def _ensure_swarm(swarm_id: str, owner_name: str, members: list[str] | None = None) -> None:
    owner_principal = owner_name if ":" in owner_name else f"agent:{owner_name}"
    _principal_token(owner_principal)
    try:
        _AUTH.create_swarm(swarm_id, owner_principal)
    except Exception:
        pass
    for member in members or []:
        principal_id = member if ":" in member else f"agent:{member}"
        _principal_token(principal_id)
        try:
            _AUTH.grant(_AUTH.admin_token, swarm_id=swarm_id, principal_id=principal_id, role="member")
        except Exception:
            pass


# ── Tests ──

def test_list_tools_returns_all():
    result = asyncio.run(mcp.list_tools())
    names = {t.name for t in result}
    expected = {
        "memory_store", "memory_write", "memory_write_status", "memory_recall", "memory_plan_inference",
        "get_more_context",
        "memory_ingest_document", "memory_ingest",
        "memory_ingest_asserted_facts",
        "memory_build_index", "memory_flush", "memory_migrate_jsonnpz", "memory_stats",
        "memory_reextract", "memory_list", "memory_get",
        "memory_admin_backfill_original_raw_sources",
        "courier_subscribe", "courier_unsubscribe",
        "memory_store_secret", "memory_list_secrets", "memory_delete_secret",
        "memory_import", "memory_import_history",
        "memory_list_prompts", "memory_get_prompt", "memory_set_prompt",
        "memory_set_config", "memory_get_config",
        "memory_set_profiles", "memory_get_profiles",
        "auth_bootstrap_admin", "principal_create", "principal_get",
        "principal_disable", "auth_token_issue", "auth_token_revoke",
        "auth_token_list", "swarm_create", "swarm_get", "swarm_list",
        "membership_grant", "membership_revoke", "membership_register", "membership_unregister", "membership_list",
        "memory_ask",
        "memory_edit", "memory_retract", "memory_purge", "memory_get_versions",
        "memory_redact",
        "memory_query", "memory_set_schema", "memory_get_schema",
        "memory_mal_configure", "memory_mal_feedback", "memory_mal_trigger",
        "memory_mal_status", "memory_mal_list_feedback", "memory_mal_get_artifact",
        "memory_mal_rollback",
    }
    assert names == expected


def test_memory_ingest_text_routes_and_tags(tmp_path):
    mod.data_dir = str(tmp_path)
    mod.registry.clear()
    writer_token = _principal_token("ingest-writer")
    result = asyncio.run(memory_ingest(
        key="ingest_text",
        text="User: hello\nAssistant: hi",
        session_num=1,
        session_date="2024-06-01",
        scope="agent-private",
        token=writer_token,
    ))
    assert result["source_family"] == "conversation"
    server = mod.registry["ingest_text"]
    assert server._all_granular
    assert server._episode_corpus["documents"]


def test_memory_migrate_jsonnpz_requires_admin(tmp_path):
    mod.data_dir = str(tmp_path)
    result = asyncio.run(memory_migrate_jsonnpz(key="legacy_key", token=_principal_token("alice")))
    assert result["code"] == "FORBIDDEN"


def test_memory_get_schema_missing_instance_returns_not_found(tmp_path):
    mod.data_dir = str(tmp_path)
    mod.registry.clear()
    result = asyncio.run(memory_get_schema(key="fresh_schema", token=_principal_token("owner")))
    assert result["code"] == "NOT_FOUND"


def test_memory_flush_maps_to_build_index(monkeypatch):
    calls = []

    async def _fake_build_index(**kwargs):
        calls.append(kwargs)
        return {"granular": 1, "consolidated": 2, "cross_session": 3}

    monkeypatch.setattr("src.mcp_server.memory_build_index", _fake_build_index)

    result = asyncio.run(memory_flush(key="flush_key", agent_id="alice", token="tok"))

    assert result["granular"] == 1
    assert result["consolidated"] == 2
    assert result["cross_session"] == 3
    assert result["rebuilt"] is True
    assert result["total_consolidated"] == 2
    assert result["total_cross_session"] == 3
    assert calls == [{"key": "flush_key", "agent_id": "alice", "token": "tok", "agent_key": None}]


def test_memory_ingest_step_trace_preserves_document_family_even_with_session_metadata(tmp_path):
    mod.data_dir = str(tmp_path)
    mod.registry.clear()
    writer_token = _principal_token("step-trace-writer")
    result = asyncio.run(memory_ingest(
        key="ingest_step_trace",
        text="[Step 0] Action: left\nObservation: moved left",
        session_num=1,
        session_date="2024-06-01",
        speakers="Game",
        scope="agent-private",
        token=writer_token,
    ))
    assert result["source_family"] == "document"
    assert "step_trace_text" in result["detection_evidence"]["signals"]
    assert "conversation_fields_present" in result["detection_evidence"]["signals"]
    server = mod.registry["ingest_step_trace"]
    assert server._all_granular
    assert server._source_records[result["source_id"]]["family"] == "document"
    assert server._episode_corpus["documents"]


def test_registry_creates_on_demand():
    assert len(mod.registry) == 0
    writer_token = _principal_token("registry-writer")
    asyncio.run(memory_store(
        key="test_key", content="Hello", session_num=1,
        session_date="2024-06-01",
        scope="agent-private",
        token=writer_token,
    ))
    assert "test_key" in mod.registry


def test_registry_reuses_instance():
    writer_token = _principal_token("reuse-writer")
    asyncio.run(memory_store(
        key="reuse", content="First", session_num=1,
        session_date="2024-06-01",
        scope="agent-private",
        token=writer_token,
    ))
    server1 = mod.registry["reuse"]
    asyncio.run(memory_store(
        key="reuse", content="Second", session_num=2,
        session_date="2024-06-02",
        scope="agent-private",
        token=writer_token,
    ))
    server2 = mod.registry["reuse"]
    assert server1 is server2


def test_memory_store_autodetects_conversation_text(monkeypatch):
    seen = _patch_extract_capture_format(monkeypatch)
    writer_token = _principal_token("conv-writer")

    result = asyncio.run(memory_store(
        key="fmt_conv",
        content="User: My favorite database is PostgreSQL.\nAssistant: Noted.",
        session_num=1,
        session_date="2024-06-01",
        scope="agent-private",
        token=writer_token,
    ))

    assert result["status"] == "ok"
    assert result["extraction_format"] == "CONVERSATION"
    assert seen["fmt"] is None
    assert _get_memory("fmt_conv")._raw_sessions[0]["extraction_format"] == "CONVERSATION"


def test_memory_store_autodetects_agent_trace(monkeypatch):
    seen = _patch_extract_capture_format(monkeypatch)
    writer_token = _principal_token("trace-writer")

    result = asyncio.run(memory_store(
        key="fmt_trace",
        content="[Step 1]\nAction: click button\nObservation: dialog opened",
        session_num=1,
        session_date="2024-06-01",
        scope="agent-private",
        token=writer_token,
    ))

    assert result["status"] == "ok"
    assert result["extraction_format"] == "AGENT_TRACE"
    assert seen["fmt"] is None
    assert _get_memory("fmt_trace")._raw_sessions[0]["extraction_format"] == "AGENT_TRACE"


def test_memory_store_autodetects_web_dom(monkeypatch):
    seen = _patch_extract_capture_format(monkeypatch)
    writer_token = _principal_token("webdom-writer")

    result = asyncio.run(memory_store(
        key="fmt_webdom",
        content='RootWebArea "Dashboard"\n  button "Save"\n  focused: true',
        session_num=1,
        session_date="2024-06-01",
        scope="agent-private",
        token=writer_token,
    ))

    assert result["status"] == "ok"
    assert result["extraction_format"] == "WEB_DOM"
    assert seen["fmt"] is None
    assert _get_memory("fmt_webdom")._raw_sessions[0]["extraction_format"] == "WEB_DOM"


def test_memory_store_autodetects_code_trace(monkeypatch):
    seen = _patch_extract_capture_format(monkeypatch)
    writer_token = _principal_token("code-writer")

    result = asyncio.run(memory_store(
        key="fmt_code",
        content="$ pytest\nEXECUTION RESULT\n2 failed, 18 passed",
        session_num=1,
        session_date="2024-06-01",
        scope="agent-private",
        token=writer_token,
    ))

    assert result["status"] == "ok"
    assert result["extraction_format"] == "CODE_TRACE"
    assert seen["fmt"] is None
    assert _get_memory("fmt_code")._raw_sessions[0]["extraction_format"] == "CODE_TRACE"


def test_memory_store_explicit_content_format_beats_autodetect(monkeypatch):
    seen = _patch_extract_capture_format(monkeypatch)
    writer_token = _principal_token("override-writer")

    result = asyncio.run(memory_store(
        key="fmt_override",
        content="[Step 1]\nAction: click button\nObservation: dialog opened",
        session_num=1,
        session_date="2024-06-01",
        scope="agent-private",
        content_format="conversation",
        token=writer_token,
    ))

    assert result["status"] == "ok"
    assert result["extraction_format"] == "CONVERSATION"
    assert seen["fmt"] == "CONVERSATION"
    assert _get_memory("fmt_override")._raw_sessions[0]["extraction_format"] == "CONVERSATION"


def test_memory_store_rejects_invalid_content_format():
    writer_token = _principal_token("badfmt-writer")

    result = asyncio.run(memory_store(
        key="fmt_invalid",
        content="hello",
        session_num=1,
        session_date="2024-06-01",
        scope="agent-private",
        content_format="BAD_FORMAT",
        token=writer_token,
    ))

    assert result["code"] == "VALIDATION_ERROR"
    assert "content_format" in result["error"]

@pytest.mark.asyncio
async def test_admin_reload_uses_internal_sqlite_view_for_acl_isolated_projection_state(tmp_path):
    mod.data_dir = str(tmp_path)
    server = _get_memory("admin_reload_internal")

    await server.store(
        "CONTENT_A",
        1,
        "2024-06-01",
        source_id="SRC",
        agent_id="agent-a",
        swarm_id="sw1",
        scope="agent-private",
    )
    await server.store(
        "CONTENT_B",
        1,
        "2024-06-02",
        source_id="SRC",
        agent_id="agent-b",
        swarm_id="sw1",
        scope="agent-private",
    )

    app = mod.create_app(app_data_dir=str(tmp_path))
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/admin/reload",
            json={"key": "admin_reload_internal"},
            headers={
                "x-server-token": mod.SERVER_TOKEN,
                "authorization": f"Bearer {_principal_token('admin')}",
            },
        )

    assert response.status_code == 200
    persisted = server._storage.load_facts(internal=True)
    assert [
        (row.get("source_id"), row.get("logical_source_id"))
        for row in server._raw_sessions
    ] == [
        (row.get("source_id"), row.get("logical_source_id"))
        for row in persisted.get("raw_sessions", [])
    ]
    assert any("@@" in str(row.get("source_id") or "") for row in server._raw_sessions)


@pytest.mark.asyncio
async def test_admin_reload_restores_persisted_profiles_and_embedding_fingerprints(tmp_path):
    mod.data_dir = str(tmp_path)
    server = _get_memory("admin_reload_config")

    await server.set_profiles(
        {1: "fast"},
        {"fast": {"model": "m-fast", "pricing": {"input_per_1k": 0.15, "output_per_1k": 0.60}}},
    )
    await server.store(
        "hello world",
        1,
        "2024-06-01",
        source_id="SRC",
        agent_id="agent-a",
        scope="agent-private",
    )
    await server.build_index()

    peer = mod.MemoryServer(str(tmp_path), "admin_reload_config")
    await peer.set_profiles(
        {1: "deep"},
        {"deep": {"model": "m-deep", "pricing": {"input_per_1k": 2.0, "output_per_1k": 8.0}}},
    )
    persisted = peer._storage.load_facts(internal=True)
    assert persisted["_emb_fingerprints"]

    server._emb_fingerprints = {}
    assert server.get_profiles()["profiles"] == {1: "fast"}

    app = mod.create_app(app_data_dir=str(tmp_path))
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/admin/reload",
            json={"key": "admin_reload_config"},
            headers={
                "x-server-token": mod.SERVER_TOKEN,
                "authorization": f"Bearer {_principal_token('admin')}",
            },
        )

    assert response.status_code == 200
    assert server.get_profiles()["profiles"] == {1: "deep"}
    assert server._profile_configs == {
        "deep": {
            "model": "m-deep",
            "pricing": {
                "input_per_1k": 2.0,
                "output_per_1k": 8.0,
                "reasoning_per_1k": 0.0,
                "cache_read_per_1k": 0.0,
                "cache_write_per_1k": 0.0,
            },
        }
    }
    assert server._emb_fingerprints == persisted["_emb_fingerprints"]


@pytest.mark.asyncio
async def test_admin_reload_requires_persisted_admin_principal(tmp_path):
    mod.data_dir = str(tmp_path)
    _get_memory("admin_reload_requires_auth")

    app = mod.create_app(app_data_dir=str(tmp_path))
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/admin/reload",
            json={"key": "admin_reload_requires_auth"},
            headers={"x-server-token": mod.SERVER_TOKEN},
        )

    assert response.status_code == 401
    assert response.json()["code"] == "AUTH_REQUIRED"


def test_memory_store_system_wide_uses_canonical_system_owner():
    asyncio.run(memory_store(
        key="mcp_system_owner",
        content="public note",
        session_num=1,
        session_date="2024-06-01",
        scope="system-wide",
        token=_principal_token("alice"),
        agent_id="alice",
    ))
    server = mod.registry["mcp_system_owner"]
    fact = server._all_granular[-1]
    assert fact["scope"] == "system-wide"
    assert fact["owner_id"] == "system"
    assert fact["read"] == ["agent:PUBLIC"]
    assert fact["write"] == ["agent:PUBLIC"]


def test_create_app_localhost_bind_keeps_host_header_protection(tmp_path):
    app = mod.create_app(app_data_dir=str(tmp_path), bind_host="127.0.0.1")
    with TestClient(app) as client:
        response = client.post(
            "/mcp",
            headers={
                "host": "memory:8765",
                "content-type": "application/json",
                "accept": "application/json, text/event-stream",
                "x-server-token": mod.SERVER_TOKEN,
            },
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {"name": "test", "version": "0.1"},
                },
            },
        )

    assert response.status_code == 421
    assert response.text == "Invalid Host header"


def test_create_app_non_local_bind_accepts_container_host_header(tmp_path):
    app = mod.create_app(app_data_dir=str(tmp_path), bind_host="0.0.0.0")
    with TestClient(app) as client:
        response = client.post(
            "/mcp",
            headers={
                "host": "memory:8765",
                "content-type": "application/json",
                "accept": "application/json, text/event-stream",
                "x-server-token": mod.SERVER_TOKEN,
            },
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {"name": "test", "version": "0.1"},
                },
            },
        )

    assert response.status_code == 200
    assert response.text != "Invalid Host header"


def test_memory_tools_reject_empty_key():
    result = asyncio.run(memory_recall(key="", query="test"))
    assert result["code"] == "VALIDATION_ERROR"
    assert "non-empty" in result["error"]


def test_memory_ask_does_not_override_memory_profiles_with_server_default(monkeypatch):
    monkeypatch.setattr(mod.cfg, "inference_model", "anthropic/claude-sonnet-4-6")
    server = _get_memory("ask_profile_default")
    captured = {}

    async def _fake_ask(**kwargs):
        captured.update(kwargs)
        return {"answer": "ok", "profile_used": "fast"}

    monkeypatch.setattr(server, "_has_profiles", lambda: True)
    monkeypatch.setattr(server, "ask", _fake_ask)

    result = asyncio.run(memory_ask(key="ask_profile_default", query="test", token=_principal_token("admin")))

    assert result["answer"] == "ok"
    assert captured["inference_model"] is None


def test_memory_query_returns_truncated_fact_preview():
    server = _get_memory("query_preview")
    long_text = "A" * 1500
    server._all_granular = [{
        "id": "fact-1",
        "fact": long_text,
        "kind": "task_result",
        "session": 1,
        "conv_id": "query_preview",
        "owner_id": "system",
        "read": ["agent:PUBLIC"],
        "write": ["agent:PUBLIC"],
        "created_at": "2024-01-01T00:00:00+00:00",
        "metadata": {"task_id": "task-1"},
    }]

    result = asyncio.run(memory_query(key="query_preview", filter={"metadata.task_id": "task-1"}, token=_principal_token("admin")))

    assert result["total"] == 1
    assert result["facts"][0]["fact_truncated"] is True
    assert len(result["facts"][0]["fact"]) < len(long_text)


def test_per_call_tagging_and_restore():
    asyncio.run(memory_store(
        key="tag_test", content="Private info", session_num=1,
        session_date="2024-06-01",
        agent_id="agent_x", scope="agent-private", swarm_id="swarm_1",
        token=_principal_token("agent_x"),
    ))
    server = mod.registry["tag_test"]
    # Facts should be tagged with per-call values
    last_fact = server._all_granular[-1]
    assert last_fact["agent_id"] == "agent_x"
    assert last_fact["scope"] == "agent-private"
    assert last_fact["swarm_id"] == "swarm_1"
    # Server defaults restored
    assert server.agent_id == "default"
    assert server.scope == "swarm-shared"
    assert server.swarm_id == "default"

def test_memory_build_index_blocking():
    writer_token = _principal_token("index-writer")
    asyncio.run(memory_store(
        key="idx_test", content="Hello", session_num=1,
        session_date="2024-06-01",
        scope="agent-private",
        token=writer_token,
    ))
    result = asyncio.run(memory_build_index(key="idx_test", token=_principal_token("admin")))
    assert "granular" in result
    assert "consolidated" in result
    assert "cross_session" in result
    assert result["granular"] >= 3


def test_memory_build_index_no_facts_error():
    # Create empty server
    _get_memory("empty_key")
    result = asyncio.run(memory_build_index(key="empty_key", token=_principal_token("admin")))
    assert result["code"] == "NO_FACTS"
    assert "error" in result


def test_memory_recall_exposes_actual_injected_episode_ids():
    writer_token = _principal_token("recall-eps-writer")
    asyncio.run(memory_store(
        key="mcp_recall_eps",
        content="User: Alice has 3 apples.\nAssistant: noted",
        session_num=1,
        session_date="2024-06-01",
        scope="agent-private",
        token=writer_token,
    ))
    asyncio.run(memory_build_index(key="mcp_recall_eps", token=_principal_token("admin")))

    result = asyncio.run(memory_recall(key="mcp_recall_eps", query="How many apples does Alice have?", token=_principal_token("admin")))

    assert "actual_injected_episode_ids" in result
    assert result["actual_injected_episode_ids"]
    assert result["actual_injected_episode_ids"] == result["runtime_trace"]["selection"]["actual_injected_episode_ids"]
    assert result["actual_injected_episode_ids"] == ["mcp_recall_eps_e0001"]


def test_memory_recall_exposes_episode_selection_trace():
    writer_token = _principal_token("recall-trace-writer")
    asyncio.run(memory_store(
        key="mcp_recall_trace",
        content="User: Alice has 3 apples and 2 pears.\nAssistant: noted",
        session_num=1,
        session_date="2024-06-01",
        scope="agent-private",
        token=writer_token,
    ))
    asyncio.run(memory_build_index(key="mcp_recall_trace", token=_principal_token("admin")))

    result = asyncio.run(memory_recall(key="mcp_recall_trace", query="How many apples does Alice have?", token=_principal_token("admin")))

    assert result["telemetry_version"] == 1
    assert "retrieved_episode_ids" in result
    assert "selection_scores" in result
    assert result["retrieved_episode_ids"] == result["runtime_trace"]["selection"]["retrieved_episode_ids"]
    assert result["selection_scores"] == result["runtime_trace"]["selection"]["selection_scores"]


def test_memory_stats_exposes_validity_and_cost_summary():
    writer_token = _principal_token("stats-writer")
    asyncio.run(memory_store(
        key="mcp_stats",
        content="User: Alice has 3 apples.\nAssistant: noted",
        session_num=1,
        session_date="2024-06-01",
        scope="agent-private",
        token=writer_token,
    ))
    asyncio.run(memory_build_index(key="mcp_stats", token=_principal_token("admin")))

    result = asyncio.run(memory_stats(key="mcp_stats", token=_principal_token("admin")))

    assert result["telemetry_version"] == 1
    assert result["raw_sessions_count"] == 1
    assert result["source_records_count"] == 1
    assert result["all_raw_sessions_active"] is True
    assert result["logical_source_count"] == 1
    assert result["part_source_count"] == 0
    assert result["raw_session_status_counts"]["active"] == 1
    assert result["process_cost_scope"] == "process"
    assert "process_cost_summary" in result
    assert set(result["process_cost_summary"].keys()) == {
        "input_tokens",
        "output_tokens",
        "embed_tokens",
        "cost_usd",
        "calls",
    }


def test_courier_subscribe_returns_sub_id():
    async def _test():
        # C2: register connection first via SSE so it's in _active_connections
        mod._active_connections["test_conn"] = "test"
        mod.connections["test_conn"] = asyncio.Queue()
        result = await courier_subscribe(
            key="sub_test",
            filter={"kind": "event"},
            connection_id="test_conn",
            token=_principal_token("admin"),
        )
        return result

    result = asyncio.run(_test())
    sub_id = result["sub_id"]
    assert sub_id.startswith("sub_")
    assert mod.sub_to_conn[sub_id] == "test_conn"


def test_courier_subscribe_rejects_unknown_connection():
    """C2: courier_subscribe must reject unknown connection_ids."""
    async def _test():
        result = await courier_subscribe(
            key="sub_test",
            filter={},
            connection_id="hijacked_conn",
            token=_principal_token("admin"),
        )
        return result

    result = asyncio.run(_test())
    assert result["code"] == "INVALID_CONNECTION"


def test_courier_unsubscribe_idempotent():
    result = asyncio.run(courier_unsubscribe(sub_id="sub_nonexistent"))
    assert result == {"status": "ok"}


def test_sse_sends_connected_event():
    async def _test():
        response = await sse_endpoint(None)
        assert len(mod.connections) == 1
        conn_id = list(mod.connections.keys())[0]
        queue = mod.connections[conn_id]
        event = queue.get_nowait()
        assert event["type"] == "connected"
        assert event["connection_id"] == conn_id
        assert len(conn_id) > 0

    asyncio.run(_test())


def test_sse_cleanup_removes_subscriptions():
    async def _test():
        # Set up connection
        await sse_endpoint(None)
        conn_id = list(mod.connections.keys())[0]

        # Manually register subscription
        sub_id = "sub_cleanup_test"
        mod.sub_to_conn[sub_id] = conn_id

        assert conn_id in mod.connections
        assert sub_id in mod.sub_to_conn

        # Trigger cleanup
        await sse_cleanup(conn_id)

        assert conn_id not in mod.connections
        assert sub_id not in mod.sub_to_conn

    asyncio.run(_test())


def test_unknown_tool_returns_error():
    async def _test():
        try:
            result = await mcp.call_tool("nonexistent.tool", {})
            # If it returns instead of raising, check for error
            if isinstance(result, dict):
                return result
            text = result[0].text if result else ""
            return {"text": text}
        except Exception as e:
            return {"error": str(e)}

    result = asyncio.run(_test())
    assert "error" in result or "Unknown tool" in str(result)


def test_memory_recall_returns_token_estimate():
    """memory_recall must include token_estimate in response."""
    writer_token = _principal_token("tok-est-writer")
    asyncio.run(memory_store(
        key="tok_est", content="Alice met Bob on Monday.",
        session_num=1, session_date="2024-06-01",
        scope="agent-private",
        token=writer_token,
    ))
    result = asyncio.run(memory_recall(
        key="tok_est", query="What happened?",
        token=_principal_token("admin"),
    ))
    assert "token_estimate" in result
    assert isinstance(result["token_estimate"], int)
    assert result["token_estimate"] == len(result["context"]) // 4


def test_memory_plan_inference_returns_planning_package_and_is_acl_gated():
    owner_token = _principal_token("planner-owner")
    asyncio.run(memory_store(
        key="plan_acl",
        content="Alice chose SQLite for storage.",
        session_num=1,
        session_date="2024-06-01",
        scope="agent-private",
        token=owner_token,
    ))
    server = mod.registry["plan_acl"]
    profiles = {1: "fast", 2: "fast", 3: "fast", 4: "fast", 5: "fast"}
    profile_configs = {
        "fast": {
            "model": "openai/gpt-4o-mini",
            "pricing": {
                "input_per_1k": 0.0,
                "output_per_1k": 0.0,
                "reasoning_per_1k": 0.0,
                "cache_read_per_1k": 0.0,
                "cache_write_per_1k": 0.0,
            },
            "secret_ref": {"name": "planner-secret", "scope": "system-wide"},
        }
    }
    asyncio.run(server.set_profiles(profiles, profile_configs))

    plan = asyncio.run(memory_plan_inference(
        key="plan_acl",
        query="What storage did Alice choose?",
        token=owner_token,
    ))

    assert plan["recommended_profile"] == "fast"
    assert "payload" in plan
    assert plan["payload_meta"]["profile_used"] == "fast"
    assert plan["secret_ref"] == {"name": "planner-secret", "scope": "system-wide"}
    assert "value" not in plan["secret_ref"]

    denied = asyncio.run(memory_plan_inference(
        key="plan_acl",
        query="What storage did Alice choose?",
        token=_principal_token("planner-outsider"),
    ))
    assert denied["code"] == "FORBIDDEN"


def test_memory_plan_inference_uses_derived_read_acl_for_public_memory():
    owner_token = _principal_token("planner-public-owner")
    reader_token = _principal_token("planner-public-reader")
    asyncio.run(memory_store(
        key="plan_public_acl",
        content="Alice published the public storage decision.",
        session_num=1,
        session_date="2024-06-01",
        scope="system-wide",
        token=owner_token,
    ))
    server = mod.registry["plan_public_acl"]
    profiles = {1: "fast", 2: "fast", 3: "fast", 4: "fast", 5: "fast"}
    profile_configs = {
        "fast": {
            "model": "openai/gpt-4o-mini",
            "pricing": {
                "input_per_1k": 0.0,
                "output_per_1k": 0.0,
                "reasoning_per_1k": 0.0,
                "cache_read_per_1k": 0.0,
                "cache_write_per_1k": 0.0,
            },
            "secret_ref": {"name": "planner-secret", "scope": "system-wide"},
        }
    }
    asyncio.run(server.set_profiles(profiles, profile_configs))

    recall = asyncio.run(memory_recall(
        key="plan_public_acl",
        query="What did Alice publish?",
        token=reader_token,
    ))
    plan = asyncio.run(memory_plan_inference(
        key="plan_public_acl",
        query="What did Alice publish?",
        token=reader_token,
    ))

    assert "context" in recall
    assert "answer_contract" in recall
    assert recall["answer_contract"]["prompt_key"]
    assert "{context}" in recall["answer_contract"]["prompt_template"]
    assert "payload" not in recall["answer_contract"]
    assert "payload_meta" not in recall["answer_contract"]
    assert "secret_ref" not in recall["answer_contract"]
    assert plan["payload_meta"]["profile_used"] == "fast"


def test_memory_recall_non_english_query_returns_error_response():
    writer_token = _principal_token("recall-lang-writer")
    asyncio.run(memory_store(
        key="recall_lang",
        content="Alice chose SQLite for storage.",
        session_num=1,
        session_date="2024-06-01",
        scope="agent-private",
        token=writer_token,
    ))

    result = asyncio.run(memory_recall(
        key="recall_lang",
        query="Что Алиса выбрала для хранения?",
        token=writer_token,
    ))

    assert result["code"] == "NON_ENGLISH_QUERY"
    assert result["error"] == "memory_recall accepts English queries only; translate in the calling agent/model"
    assert result["runtime_trace"]["query_language"]["source_lang"] == "non_en"
    assert "context" not in result
    assert "payload" not in result
    assert "payload_meta" not in result


def test_memory_recall_omits_inference_planning_hints_from_mcp_response(monkeypatch):
    writer_token = _principal_token("recall-hints-writer")
    asyncio.run(memory_store(
        key="recall_hints",
        content="Alice chose SQLite for storage.",
        session_num=1,
        session_date="2024-06-01",
        scope="agent-private",
        token=writer_token,
    ))
    server = mod.registry["recall_hints"]

    async def fake_recall(**_kwargs):
        return {
            "context": "RETRIEVED FACTS:\n- Alice chose SQLite for storage.",
            "retrieved": [],
            "query_type": "lookup",
            "runtime_trace": {"evidence_context": {"finalized": True}},
            "recommended_profile": "fast",
            "payload": {"model": "openai/gpt-4o-mini"},
            "payload_meta": {"profile_used": "fast"},
            "_payload_secret_ref": {"name": "hidden-secret", "scope": "system-wide"},
            "secret_ref": {"name": "hidden-secret", "scope": "system-wide"},
            "recall_continuation": {
                "available": True,
                "handle": "opaque",
                "next_page": 2,
                "page_size": 5,
                "candidate_count": 6,
                "returned_count": 5,
                "exhausted": False,
                "anchor_terms": ["storage"],
                "tool": "get_more_context",
                "tool_usage": "call get_more_context with page=\"next\" or without session_id to fetch the next evidence page",
            },
        }

    monkeypatch.setattr(server, "recall", fake_recall)

    result = asyncio.run(memory_recall(
        key="recall_hints",
        query="What storage did Alice choose?",
        token=writer_token,
    ))

    assert "context" in result
    assert "recommended_prompt_type" not in result
    assert "use_tool" not in result
    assert "payload" not in result
    assert "payload_meta" not in result
    assert "_payload_secret_ref" not in result
    assert "secret_ref" not in result
    assert result["recall_continuation"]["available"] is True
    assert result["recall_continuation"]["handle"] == "opaque"
    assert result["recall_continuation"]["tool"] == "get_more_context"
    assert result["recall_continuation"]["mcp_tool"] == "get_more_context"
    assert "get_more_context" in result["recall_continuation"]["tool_usage"]


def test_memory_recall_public_continuation_pages_through_get_more_context(monkeypatch):
    writer_token = _principal_token("recall-continuation-writer")
    asyncio.run(memory_store(
        key="recall_public_continuation",
        content="Project Alpha workflow checkpoint.",
        session_num=1,
        session_date="2024-06-01",
        scope="agent-private",
        token=writer_token,
    ))
    server = mod.registry["recall_public_continuation"]

    async def fake_recall(**_kwargs):
        return {
            "context": (
                "RETRIEVED FACTS:\n- Project Alpha workflow checkpoint.\n\n"
                "RECALL CONTINUATION AVAILABLE:\n"
                "More evidence matches anchors ['project', 'alpha']. "
                "If the answer is not in this page, call get_more_context with page=\"next\" "
                "or without session_id to retrieve the next evidence page."
            ),
            "retrieved": [],
            "query_type": "lookup",
            "runtime_trace": {"evidence_context": {"finalized": True}},
            "_recall_continuation_pages": [
                {
                    "page": 2,
                    "next_page": None,
                    "exhausted": True,
                    "context": "RECALL CONTINUATION PAGE 2:\nRETRIEVED FACTS:\n- Project Alpha ships Friday.",
                    "returned_count": 1,
                }
            ],
            "recall_continuation": {
                "available": True,
                "handle": "opaque-public",
                "next_page": 2,
                "page_size": 5,
                "candidate_count": 6,
                "returned_count": 5,
                "exhausted": False,
                "anchor_terms": ["project", "alpha"],
                "tool": "get_more_context",
                "tool_usage": "call get_more_context with page=\"next\" or without session_id to fetch the next evidence page",
            },
            "answer_contract": {
                "prompt_template": "Context:\n{context}\nIf missing, call get_more_context with page=\"next\" or without session_id to fetch the next evidence page.",
                "variables": {},
                "recall_continuation": {
                    "available": True,
                    "tool": "get_more_context",
                    "handle": "opaque-public",
                    "next_page": 2,
                    "instruction": "call get_more_context with page=\"next\" or no session_id to fetch the next page.",
                },
            },
        }

    monkeypatch.setattr(server, "recall", fake_recall)

    first = asyncio.run(memory_recall(
        key="recall_public_continuation",
        query="When does Project Alpha ship?",
        token=writer_token,
    ))

    assert first["recall_continuation"]["tool"] == "get_more_context"
    assert first["recall_continuation"]["mcp_tool"] == "get_more_context"
    assert 'handle="opaque-public"' in first["context"]
    assert "get_more_context" in first["context"]
    assert first["answer_contract"]["recall_continuation"]["tool"] == "get_more_context"
    assert "get_more_context" in first["answer_contract"]["recall_continuation"]["instruction"]
    assert "get_more_context" in first["answer_contract"]["prompt_template"]

    second = asyncio.run(get_more_context(
        handle="opaque-public",
        page="next",
        token=writer_token,
    ))

    assert "Project Alpha ships Friday" in second["context"]
    assert second["query_type"] == "continuation"
    assert second["recall_continuation"]["handle"] == "opaque-public"
    assert second["recall_continuation"]["exhausted"] is True
    assert "payload" not in second
    assert "payload_meta" not in second


def test_get_more_context_uses_handle_bound_swarm_when_omitted(monkeypatch):
    writer_token = _principal_token("recall-continuation-team-writer")
    asyncio.run(memory_store(
        key="recall_team_continuation",
        content="Project Alpha team workflow checkpoint.",
        session_num=1,
        session_date="2024-06-01",
        scope="agent-private",
        swarm_id="team-gosh",
        token=writer_token,
    ))
    server = mod.registry["recall_team_continuation"]

    async def fake_recall(**kwargs):
        assert kwargs["swarm_id"] == "team-gosh"
        return {
            "context": "RETRIEVED FACTS:\n- Project Alpha team workflow checkpoint.",
            "retrieved": [],
            "query_type": "lookup",
            "runtime_trace": {"evidence_context": {"finalized": True}},
            "_recall_continuation_pages": [
                {
                    "page": 2,
                    "next_page": None,
                    "exhausted": True,
                    "context": "RECALL CONTINUATION PAGE 2:\nRETRIEVED FACTS:\n- Team Alpha ships Friday.",
                    "returned_count": 1,
                }
            ],
            "recall_continuation": {
                "available": True,
                "handle": "opaque-team-gosh",
                "next_page": 2,
                "page_size": 5,
                "candidate_count": 6,
                "returned_count": 5,
                "exhausted": False,
                "anchor_terms": ["project", "alpha"],
                "tool": "get_more_context",
                "tool_usage": (
                    "call get_more_context with handle=<handle> and page=\"next\" "
                    "to fetch the next evidence page"
                ),
            },
        }

    monkeypatch.setattr(server, "recall", fake_recall)

    first = asyncio.run(memory_recall(
        key="recall_team_continuation",
        query="When does Project Alpha ship?",
        swarm_id="team-gosh",
        token=writer_token,
    ))

    assert first["recall_continuation"]["handle"] == "opaque-team-gosh"

    second = asyncio.run(get_more_context(
        handle="opaque-team-gosh",
        page="next",
        token=writer_token,
    ))

    assert second.get("code") is None
    assert "Team Alpha ships Friday" in second["context"]
    trace = second["runtime_trace"]["recall_continuation_trace"]
    assert trace["handle_bound_swarm_id"] is True


def test_get_more_context_invalid_handle_returns_structured_error():
    result = asyncio.run(get_more_context(
        handle="missing-continuation-handle",
        page="next",
        token=_principal_token("missing-continuation-reader"),
    ))

    assert result["code"] == "RECALL_CONTINUATION_NOT_FOUND"
    assert result["recall_continuation"]["available"] is False
    assert result["recall_continuation"]["exhausted"] is True


def test_memory_write_exposes_raw_recall_and_status(tmp_path):
    mod.data_dir = str(tmp_path)
    mod.registry.clear()
    writer_token = _principal_token("raw-writer")
    result = asyncio.run(memory_write(
        key="write_raw",
        message_id="msg-1",
        session_id="sess-1",
        content="Fresh write about mango orchards",
        content_family="chat",
        timestamp_ms=1712000000000,
        scope="agent-private",
        token=writer_token,
    ))
    assert result["message_id"] == "msg-1"
    assert result["extraction_state"] == "pending"

    status = asyncio.run(memory_write_status(key="write_raw", message_id="msg-1", token=writer_token))
    assert status["extraction_state"] == "pending"

    recall = asyncio.run(memory_recall(key="write_raw", query="mango orchards", token=writer_token))
    assert "RECENT RAW WRITES:" in recall["context"]
    assert "Fresh write about mango orchards" in recall["context"]
    assert recall["raw_recall_count"] == 1
    assert "mango orchards" in recall["context"].lower()
    assert "answer_contract" in recall
    for field in ("payload", "payload_meta", "_payload_secret_ref", "secret_ref", "recommended_profile"):
        assert field not in recall


def test_memory_write_worker_promotes_chat_entry_into_extracted_memory(tmp_path):
    mod.data_dir = str(tmp_path)
    mod.registry.clear()
    writer_token = _principal_token("worker-writer")
    asyncio.run(memory_write(
        key="write_worker",
        message_id="msg-2",
        session_id="sess-2",
        content="User: Alice planted tulips.\nAssistant: noted",
        content_family="chat",
        timestamp_ms=1712000001000,
        scope="agent-private",
        token=writer_token,
    ))
    server = mod.registry["write_worker"]

    processed = asyncio.run(server.process_write_log_once())
    assert processed == 1

    status = asyncio.run(memory_write_status(key="write_worker", message_id="msg-2", token=writer_token))
    assert status["extraction_state"] == "complete"
    assert server._raw_sessions
    assert any(rs.get("message_id") == "msg-2" for rs in server._raw_sessions)
    assert server._all_granular


def test_memory_write_rejects_unknown_content_family(tmp_path):
    mod.data_dir = str(tmp_path)
    mod.registry.clear()
    writer_token = _principal_token("bad-family-writer")
    result = asyncio.run(memory_write(
        key="write_bad_family",
        message_id="msg-bad",
        session_id="sess-bad",
        content="hello",
        content_family="weird",
        timestamp_ms=1712000002000,
        scope="agent-private",
        token=writer_token,
    ))
    assert result["code"] == "VALIDATION_ERROR"
    assert "Unsupported content_family" in result["error"]


def test_memory_write_enforces_strict_ingress_metadata_contract(tmp_path):
    mod.data_dir = str(tmp_path)
    mod.registry.clear()
    writer_token = _principal_token("metadata-writer")

    valid = asyncio.run(memory_write(
        key="write_metadata_valid",
        message_id="msg-valid",
        session_id="sess-valid",
        content="hello",
        content_family="chat",
        timestamp_ms=1712000002500,
        scope="agent-private",
        metadata={"role": "user", "tags": ["alpha", "beta"]},
        token=writer_token,
    ))
    invalid_bool = asyncio.run(memory_write(
        key="write_metadata_valid",
        message_id="msg-bool",
        session_id="sess-bool",
        content="hello",
        content_family="chat",
        timestamp_ms=1712000002501,
        scope="agent-private",
        metadata={"approved": True},
        token=writer_token,
    ))
    invalid_nested = asyncio.run(memory_write(
        key="write_metadata_valid",
        message_id="msg-nested",
        session_id="sess-nested",
        content="hello",
        content_family="chat",
        timestamp_ms=1712000002502,
        scope="agent-private",
        metadata={"extra": {"nested": "value"}},
        token=writer_token,
    ))
    valid_controls = asyncio.run(memory_write(
        key="write_metadata_valid",
        message_id="msg-controls",
        session_id="sess-controls",
        content="hello",
        content_family="chat",
        timestamp_ms=1712000002503,
        scope="agent-private",
        metadata={"part_idx": 1, "turn_number": 2, "role": "user"},
        token=writer_token,
    ))

    assert valid["inserted"] is True
    assert valid_controls["inserted"] is True
    assert invalid_bool["code"] == "VALIDATION_ERROR"
    assert "expected string or list of strings" in invalid_bool["error"]
    assert invalid_nested["code"] == "VALIDATION_ERROR"
    assert "expected string or list of strings" in invalid_nested["error"]
def test_memory_store_local_cli_timeout_returns_explicit_error(tmp_path, monkeypatch):
    mod.data_dir = str(tmp_path)
    mod.registry.clear()
    from src.librarian import extract_session as real_extract_session

    def _fake_run_local_cli(prompt, cli_bin, cli_args_prefix):
        raise LocalCliTimeoutError("local_cli subprocess timed out (timeout_secs=0.05)")

    monkeypatch.setattr("src.memory.extract_session", real_extract_session)
    monkeypatch.setattr("src.memory.run_local_cli", _fake_run_local_cli)
    server = _get_memory("store_local_cli_timeout")
    asyncio.run(server.set_config({
        "schema_version": 1,
        "embedding_model": "text-embedding-3-small",
        "librarian_profile": "fast",
        "profiles": {1: "fast"},
        "profile_configs": {
            "fast": {
                "backend": "local_cli",
                "model": "local/my-cli",
                "cli_bin": "/abs/path/to/my-cli",
                "cli_args_prefix": ["run"],
                "context_window": 200000,
                "max_output_tokens": 4096,
                "temperature": 0,
            }
        },
        "retrieval": {"search_family": "auto", "default_token_budget": 4000},
    }))

    writer_token = _principal_token("store-timeout-writer")
    result = asyncio.run(memory_store(
        key="store_local_cli_timeout",
        content="Alice: The backup key is behind the picture frame.\nBob: Carol needs it tomorrow morning.",
        session_num=1,
        session_date="2024-06-01",
        scope="agent-private",
        token=writer_token,
    ))

    assert result["code"] == "LOCAL_CLI_TIMEOUT"
    assert result["status"] == "extraction_failed"
    assert result["facts_extracted"] == 0
    assert server._raw_sessions[0]["status"] == "extraction_failed"


def test_memory_write_and_status_support_concurrent_calls(tmp_path):
    mod.data_dir = str(tmp_path)
    mod.registry.clear()
    server = _get_memory("write_concurrent")
    mod._ensure_instance_config(server, "agent:agent-a")
    agent_token = _principal_token("agent-a")
    total = 6

    def _write(i: int) -> dict:
        return asyncio.run(memory_write(
            key="write_concurrent",
            message_id=f"msg-{i}",
            session_id=f"sess-{i}",
            content=f"parallel kiwi write {i}",
            content_family="chat",
            timestamp_ms=1712000003000 + i,
            agent_id="agent-a",
            swarm_id="sw1",
            scope="agent-private",
            token=agent_token,
        ))

    with ThreadPoolExecutor(max_workers=6) as pool:
        write_results = list(pool.map(_write, range(total)))

    assert all(result["inserted"] is True for result in write_results)

    def _status(i: int) -> dict:
        return asyncio.run(memory_write_status(
            key="write_concurrent",
            message_id=f"msg-{i}",
            agent_id="agent-a",
            swarm_id="sw1",
            token=agent_token,
        ))

    with ThreadPoolExecutor(max_workers=6) as pool:
        statuses = list(pool.map(_status, range(total)))

    assert all(status["extraction_state"] == "pending" for status in statuses)
    recall = asyncio.run(memory_recall(
        key="write_concurrent",
        query="parallel kiwi",
        agent_id="agent-a",
        swarm_id="sw1",
        token=agent_token,
    ))
    assert recall["raw_recall_count"] == total
    assert "parallel kiwi write 0" in recall["context"].lower()


@pytest.mark.asyncio
async def test_memory_write_raw_recall_respects_acl_after_concurrent_writes(tmp_path):
    mod.data_dir = str(tmp_path)
    mod.registry.clear()
    server = _get_memory("write_acl")
    mod._ensure_instance_config(server, "agent:agent-a")
    mod._expand_instance_acl_for_scope(server, scope="swarm-shared", swarm_id="sw1")
    _ensure_swarm("sw1", "agent-a", members=["agent-b", "agent-c"])
    token_a = _principal_token("agent-a")
    token_b = _principal_token("agent-b")
    token_c = _principal_token("agent-c")

    writes = [
        {
            "message_id": "shared-1",
            "session_id": "sess-shared",
            "content": "sharedapple orchard note",
            "agent_id": "agent-a",
            "scope": "swarm-shared",
        },
        {
            "message_id": "private-a",
            "session_id": "sess-private-a",
            "content": "alphaapple orchard note",
            "agent_id": "agent-a",
            "scope": "agent-private",
        },
        {
            "message_id": "private-b",
            "session_id": "sess-private-b",
            "content": "betaapple orchard note",
            "agent_id": "agent-b",
            "scope": "agent-private",
        },
    ]

    async def _write(payload: dict) -> dict:
        return await memory_write(
            key="write_acl",
            message_id=payload["message_id"],
            session_id=payload["session_id"],
            content=payload["content"],
            content_family="chat",
            timestamp_ms=1712000004000,
            agent_id=payload["agent_id"],
            swarm_id="sw1",
            scope=payload["scope"],
            token=token_a if payload["agent_id"] == "agent-a" else token_b,
        )

    results = await asyncio.gather(*(_write(payload) for payload in writes))

    assert all(result["inserted"] is True for result in results)

    recall_a = await memory_recall(key="write_acl", query="orchard", agent_id="agent-a", swarm_id="sw1", token=token_a)
    recall_b = await memory_recall(key="write_acl", query="orchard", agent_id="agent-b", swarm_id="sw1", token=token_b)
    recall_c = await memory_recall(key="write_acl", query="orchard", agent_id="agent-c", swarm_id="sw1", token=token_c)

    assert recall_a["raw_recall_count"] == 2
    assert "sharedapple" in recall_a["context"]
    assert "alphaapple" in recall_a["context"]
    assert "betaapple" not in recall_a["context"]

    assert recall_b["raw_recall_count"] == 2
    assert "sharedapple" in recall_b["context"]
    assert "betaapple" in recall_b["context"]
    assert "alphaapple" not in recall_b["context"]

    assert recall_c["raw_recall_count"] == 1
    assert "sharedapple" in recall_c["context"]
    assert "alphaapple" not in recall_c["context"]
    assert "betaapple" not in recall_c["context"]


def test_memory_write_status_hides_private_write_state_from_other_agents(tmp_path):
    mod.data_dir = str(tmp_path)
    mod.registry.clear()
    _ensure_swarm("sw1", "agent-a", members=["agent-b"])
    token_a = _principal_token("agent-a")
    token_b = _principal_token("agent-b")

    shared = asyncio.run(memory_write(
        key="write_status_acl",
        message_id="shared-1",
        session_id="sess-shared",
        content="shared note",
        content_family="chat",
        timestamp_ms=1712000004500,
        agent_id="agent-a",
        swarm_id="sw1",
        scope="swarm-shared",
        token=token_a,
    ))
    private = asyncio.run(memory_write(
        key="write_status_acl",
        message_id="private-1",
        session_id="sess-private",
        content="private note",
        content_family="chat",
        timestamp_ms=1712000004501,
        agent_id="agent-a",
        swarm_id="sw1",
        scope="agent-private",
        token=token_a,
    ))

    assert shared["inserted"] is True
    assert private["inserted"] is True

    owner_status = asyncio.run(memory_write_status(
        key="write_status_acl",
        message_id="private-1",
        agent_id="agent-a",
        swarm_id="sw1",
        token=token_a,
    ))
    other_status = asyncio.run(memory_write_status(
        key="write_status_acl",
        message_id="private-1",
        agent_id="agent-b",
        swarm_id="sw1",
        token=token_b,
    ))

    assert owner_status["extraction_state"] == "pending"
    assert other_status == {"error": "Write private-1 not found", "code": "NOT_FOUND"}


def test_ensure_instance_config_is_atomic_under_concurrency(tmp_path, monkeypatch):
    mod.data_dir = str(tmp_path)
    mod.registry.clear()
    server = _get_memory("instance_config_race")
    barrier = threading.Barrier(2)
    original = mod._ensure_instance_config

    def _wrapped(server_obj, owner_id):
        barrier.wait(timeout=1)
        return original(server_obj, owner_id)

    monkeypatch.setattr(mod, "_ensure_instance_config", _wrapped)

    with ThreadPoolExecutor(max_workers=2) as pool:
        created = list(pool.map(lambda owner: mod._ensure_instance_config(server, owner), ["agent:a", "agent:b"]))

    assert sum(bool(item) for item in created) == 1
    assert server._instance_config["owner_id"] in {"agent:a", "agent:b"}


def test_memory_write_worker_processes_batch_after_concurrent_writes(tmp_path, monkeypatch):
    _patch_all(monkeypatch)
    mod.data_dir = str(tmp_path)
    mod.registry.clear()
    total = 5
    _ensure_swarm("sw1", "agent-a")
    agent_token = _principal_token("agent-a")

    def _write(i: int) -> dict:
        return asyncio.run(memory_write(
            key="write_batch",
            message_id=f"batch-{i}",
            session_id=f"sess-{i}",
            content=f"User: note {i}\nAssistant: ok",
            content_family="chat",
            timestamp_ms=1712000005000 + i,
            agent_id="agent-a",
            swarm_id="sw1",
            scope="agent-private",
            token=agent_token,
        ))

    with ThreadPoolExecutor(max_workers=5) as pool:
        results = list(pool.map(_write, range(total)))

    assert all(result["inserted"] is True for result in results)
    server = mod.registry["write_batch"]

    processed = asyncio.run(server.process_write_log_once(batch_size=total))
    assert processed == total
    assert len(server._raw_sessions) == total
    assert len(server._all_granular) == total * 3
    assert not server._storage.list_write_log_entries(states=["pending", "in_progress", "failed"], order="asc")

    for i in range(total):
        status = asyncio.run(memory_write_status(key="write_batch", message_id=f"batch-{i}", agent_id="agent-a", swarm_id="sw1", token=agent_token))
        assert status["extraction_state"] == "complete"



def test_memory_write_worker_soak_hundreds_of_entries(tmp_path, monkeypatch):
    _patch_all(monkeypatch)
    mod.data_dir = str(tmp_path)
    mod.registry.clear()
    total = 100
    _ensure_swarm("sw1", "agent-a")
    agent_token = _principal_token("agent-a")

    def _write(i: int) -> dict:
        return asyncio.run(memory_write(
            key="write_soak",
            message_id=f"soak-{i}",
            session_id=f"sess-{i}",
            content=f"User: soak note {i}\nAssistant: ok",
            content_family="chat",
            timestamp_ms=1712000006000 + i,
            agent_id="agent-a",
            swarm_id="sw1",
            scope="agent-private",
            token=agent_token,
        ))

    with ThreadPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(_write, range(total)))

    assert all(result["inserted"] is True for result in results)
    server = mod.registry["write_soak"]

    processed = 0
    rounds = 0
    while processed < total and rounds < 20:
        processed += asyncio.run(server.process_write_log_once(batch_size=16))
        rounds += 1

    assert processed == total
    assert len(server._raw_sessions) == total
    assert len(server._all_granular) == total * 3
    assert not server._storage.list_write_log_entries(states=["pending", "in_progress", "failed"], order="asc")


def test_memory_write_failed_entries_remain_recallable_under_concurrent_failures(tmp_path, monkeypatch):
    mod.data_dir = str(tmp_path)
    mod.registry.clear()
    total = 6
    _ensure_swarm("sw1", "agent-a")
    agent_token = _principal_token("agent-a")

    def _write(i: int) -> dict:
        return asyncio.run(memory_write(
            key="write_fail",
            message_id=f"fail-{i}",
            session_id=f"sess-{i}",
            content=f"failure papaya note {i}",
            content_family="chat",
            timestamp_ms=1712000007000 + i,
            agent_id="agent-a",
            swarm_id="sw1",
            scope="agent-private",
            token=agent_token,
        ))

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(_write, range(total)))

    assert all(result["inserted"] is True for result in results)
    server = mod.registry["write_fail"]

    async def _boom(_entry):
        raise RuntimeError("boom")

    monkeypatch.setattr(server, "_extract_write_log_entry", _boom)
    monkeypatch.setattr(server, "_should_retry_write_entry", lambda entry, now_ms: True)

    for _ in range(3):
        asyncio.run(server.process_write_log_once(batch_size=total))

    for i in range(total):
        status = asyncio.run(memory_write_status(key="write_fail", message_id=f"fail-{i}", agent_id="agent-a", swarm_id="sw1", token=agent_token))
        assert status["extraction_state"] == "failed"
        assert status["extraction_attempts"] == 3

    recall = asyncio.run(memory_recall(key="write_fail", query="papaya", agent_id="agent-a", swarm_id="sw1", token=agent_token))
    assert recall["raw_recall_count"] == total
    assert "failure papaya note 0" in recall["context"].lower()


def test_memory_write_receipt_latency_under_parallel_load(tmp_path):
    mod.data_dir = str(tmp_path)
    mod.registry.clear()
    server = _get_memory("write_latency_parallel")
    mod._ensure_instance_config(server, "agent:agent-a")
    total = 20
    agent_token = _principal_token("agent-a")

    def _timed_write(i: int) -> float:
        start = time.perf_counter()
        result = asyncio.run(memory_write(
            key="write_latency_parallel",
            message_id=f"lat-{i}",
            session_id=f"sess-{i}",
            content=f"parallel latency note {i}",
            content_family="chat",
            timestamp_ms=1712000008000 + i,
            agent_id="agent-a",
            swarm_id="sw1",
            scope="agent-private",
            token=agent_token,
        ))
        assert result["inserted"] is True
        return time.perf_counter() - start

    with ThreadPoolExecutor(max_workers=8) as pool:
        latencies = sorted(pool.map(_timed_write, range(total)))

    p95 = latencies[int(total * 0.95) - 1]
    assert p95 < 0.1

    start = time.perf_counter()
    recall = asyncio.run(memory_recall(
        key="write_latency_parallel",
        query="parallel latency",
        agent_id="agent-a",
        swarm_id="sw1",
        token=agent_token,
    ))
    recall_elapsed = time.perf_counter() - start
    assert recall_elapsed < 0.2
    assert recall["raw_recall_count"] >= 1
    assert "parallel latency note" in recall["context"].lower()
