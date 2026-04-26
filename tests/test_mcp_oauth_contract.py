# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

import asyncio

import pytest

import src.mcp_server as mcp_mod
from src.mcp_server import (
    auth_bootstrap_admin,
    auth_token_issue,
    auth_token_list,
    auth_token_revoke,
    memory_ingest_asserted_facts,
    memory_import,
    memory_import_history,
    memory_ingest,
    memory_ingest_document,
    memory_list,
    memory_recall,
    memory_store,
    memory_store_secret,
    memory_write,
    memory_write_status,
    principal_create,
    swarm_create,
)
from tests._auth_helpers import bootstrap_harness, configure_bootstrap_env
from tests._memory_embed_mocks import patch_memory_embeddings


@pytest.fixture(autouse=True)
def _reset_registry():
    mcp_mod.registry.clear()
    yield
    mcp_mod.registry.clear()


def test_missing_token_rejected_on_protected_tool(tmp_path, monkeypatch):
    patch_memory_embeddings(monkeypatch)
    bootstrap_harness(monkeypatch, tmp_path, patch_extraction=True)
    result = asyncio.run(memory_store(
        key="auth-required",
        content="hello",
        session_num=1,
        session_date="2026-04-06",
    ))
    assert result["code"] == "AUTH_REQUIRED"


def _asserted_payload() -> tuple[list[dict], list[dict]]:
    facts = [{
        "id": "f1",
        "fact": "Imported fact",
        "kind": "event",
        "entities": [],
        "tags": [],
        "session": 1,
    }]
    raw_sessions = [{
        "raw_session_id": "rs1",
        "session_num": 1,
        "session_date": "2026-04-07",
        "content": "Imported session",
    }]
    return facts, raw_sessions


async def _invoke_live_tool(
    tool_name: str,
    *,
    token: str,
    agent_id: str | None,
):
    if tool_name == "store":
        return await memory_store(
            key="agent-write-store",
            content="hello",
            session_num=1,
            session_date="2026-04-06",
            token=token,
            agent_id=agent_id,
            scope="agent-private",
        )
    if tool_name == "write":
        return await memory_write(
            key="agent-write-raw",
            message_id="msg-1",
            session_id="sess-1",
            content="hello",
            content_family="chat",
            timestamp_ms=1712450000000,
            token=token,
            agent_id=agent_id,
            scope="agent-private",
        )
    if tool_name == "document":
        return await memory_ingest_document(
            key="agent-write-doc",
            content="Document body",
            source_id="DOC-1",
            token=token,
            agent_id=agent_id,
            scope="agent-private",
        )
    if tool_name == "ingest":
        return await memory_ingest(
            key="agent-write-ingest",
            text="User: hi\nAssistant: ok",
            session_num=1,
            session_date="2026-04-06",
            token=token,
            agent_id=agent_id,
            scope="agent-private",
        )
    if tool_name == "asserted":
        facts, raw_sessions = _asserted_payload()
        return await memory_ingest_asserted_facts(
            key="agent-write-asserted",
            facts=facts,
            raw_sessions=raw_sessions,
            token=token,
            agent_id=agent_id,
            scope="agent-private",
            enrich_l0=False,
        )
    if tool_name == "import":
        return await memory_import(
            key="agent-write-import",
            source_format="text",
            content="hello import",
            auth_token=token,
            agent_id=agent_id,
            scope="agent-private",
        )
    if tool_name == "import_history":
        payload = '[{"created_at":"2026-04-08T00:00:00Z","messages":[{"role":"user","content":"hello import history"}]}]'
        return await memory_import_history(
            key="agent-write-import-history",
            source_format="conversation_json",
            content=payload,
            auth_token=token,
            agent_id=agent_id,
            scope="agent-private",
        )
    raise AssertionError(f"unknown tool_name {tool_name}")


@pytest.mark.asyncio
async def test_env_bootstrap_admin_provisions_persisted_admin_token(tmp_path, monkeypatch):
    bootstrap_token = configure_bootstrap_env(monkeypatch, tmp_path, patch_extraction=True)
    result = await auth_bootstrap_admin(
        principal_id="service:site-admin",
        kind="service",
        token=bootstrap_token,
    )
    assert result["status"] == "ok"
    assert result["principal_id"] == "service:site-admin"
    assert result["token_kind"] == "admin"


@pytest.mark.asyncio
async def test_persisted_admin_token_can_provision_principal_and_swarm(tmp_path, monkeypatch):
    harness = bootstrap_harness(monkeypatch, tmp_path, patch_extraction=True)
    created = await principal_create(
        principal_id="agent:alice",
        kind="agent",
        token=harness.admin_token,
    )
    assert created["status"] == "ok"

    issued = await auth_token_issue(
        principal_id="agent:alice",
        token_kind="agent",
        token=harness.admin_token,
    )
    assert issued["status"] == "ok"
    assert issued["principal_id"] == "agent:alice"

    swarm = await swarm_create(
        swarm_id="alpha",
        owner_principal_id="agent:alice",
        token=harness.admin_token,
    )
    assert swarm["status"] == "ok"
    assert swarm["swarm"]["swarm_id"] == "alpha"


@pytest.mark.asyncio
async def test_memory_import_keeps_source_token_separate_from_auth_token(tmp_path, monkeypatch):
    patch_memory_embeddings(monkeypatch)
    harness = bootstrap_harness(monkeypatch, tmp_path, patch_extraction=True)
    alice_token = harness.issue("agent:alice", kind="agent")

    missing_auth = await memory_import(
        key="import-auth",
        source_format="text",
        content="hello import",
        token="source-token-only",
    )
    assert missing_auth["code"] == "AUTH_REQUIRED"

    ok = await memory_import(
        key="import-auth",
        source_format="text",
        content="hello import",
        auth_token=alice_token,
        token="source-token-only",
        agent_id="alice",
        scope="agent-private",
    )
    assert ok["sessions_processed"] == 1


@pytest.mark.asyncio
async def test_revoked_persisted_token_fails_closed(tmp_path, monkeypatch):
    patch_memory_embeddings(monkeypatch)
    harness = bootstrap_harness(monkeypatch, tmp_path, patch_extraction=True)
    alice_token = harness.issue("agent:alice", kind="agent")

    listed = await auth_token_list(principal_id="agent:alice", token=alice_token)
    token_id = listed["tokens"][0]["token_id"]
    revoked = await auth_token_revoke(token_id=token_id, token=alice_token)
    assert revoked["status"] == "ok"

    result = await memory_store(
        key="revoked",
        content="hello",
        session_num=1,
        session_date="2026-04-06",
        token=alice_token,
        agent_id="alice",
    )
    assert result["code"] == "AUTH_REVOKED"


@pytest.mark.asyncio
async def test_agent_principal_live_write_derives_agent_id_when_omitted(tmp_path, monkeypatch):
    patch_memory_embeddings(monkeypatch)
    harness = bootstrap_harness(monkeypatch, tmp_path, patch_extraction=True)
    petya = harness.issue("agent:petya", kind="agent")

    stored = await memory_store(
        key="derive-agent-store",
        content="hello",
        session_num=1,
        session_date="2026-04-06",
        token=petya,
        scope="agent-private",
    )
    written = await memory_write(
        key="derive-agent-write",
        message_id="msg-derive",
        session_id="sess-derive",
        content="hello raw",
        content_family="chat",
        timestamp_ms=1712450000000,
        token=petya,
        scope="agent-private",
    )

    assert stored["status"] == "ok"
    assert written["inserted"] is True
    assert mcp_mod.registry["derive-agent-store"]._all_granular[0]["agent_id"] == "petya"
    persisted_entry = mcp_mod.registry["derive-agent-write"].write_status("msg-derive")
    assert persisted_entry["agent_id"] == "petya"


@pytest.mark.asyncio
async def test_agent_principal_live_write_accepts_matching_agent_id(tmp_path, monkeypatch):
    patch_memory_embeddings(monkeypatch)
    harness = bootstrap_harness(monkeypatch, tmp_path, patch_extraction=True)
    petya = harness.issue("agent:petya", kind="agent")

    result = await memory_store(
        key="matching-agent-store",
        content="hello",
        session_num=1,
        session_date="2026-04-06",
        token=petya,
        agent_id="petya",
        scope="agent-private",
    )

    assert result["status"] == "ok"
    assert mcp_mod.registry["matching-agent-store"]._all_granular[0]["agent_id"] == "petya"


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", ["store", "write", "document", "ingest", "asserted"])
@pytest.mark.parametrize("bad_agent_id", ["vasya", "default"])
async def test_live_content_writes_reject_mismatched_or_default_agent_id(tmp_path, monkeypatch, tool_name, bad_agent_id):
    patch_memory_embeddings(monkeypatch)
    harness = bootstrap_harness(monkeypatch, tmp_path, patch_extraction=True)
    petya = harness.issue("agent:petya", kind="agent")

    result = await _invoke_live_tool(tool_name, token=petya, agent_id=bad_agent_id)

    assert result["code"] == "VALIDATION_ERROR"
    assert "agent_id" in result["error"]


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", ["store", "write", "document", "ingest", "asserted", "import", "import_history"])
@pytest.mark.parametrize("bad_agent_id", ["vasya", "default"])
async def test_import_and_live_content_writes_reject_mismatched_or_default_agent_id(tmp_path, monkeypatch, tool_name, bad_agent_id):
    patch_memory_embeddings(monkeypatch)
    harness = bootstrap_harness(monkeypatch, tmp_path, patch_extraction=True)
    petya = harness.issue("agent:petya", kind="agent")

    result = await _invoke_live_tool(tool_name, token=petya, agent_id=bad_agent_id)

    assert result["code"] == "VALIDATION_ERROR"
    assert "agent_id" in result["error"]


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", ["store", "write", "document", "ingest", "asserted", "import", "import_history"])
@pytest.mark.parametrize("principal_id,kind", [("user:mitja", "user"), ("service:ci", "service")])
async def test_non_agent_principals_cannot_write_live_content_or_import(tmp_path, monkeypatch, tool_name, principal_id, kind):
    patch_memory_embeddings(monkeypatch)
    harness = bootstrap_harness(monkeypatch, tmp_path, patch_extraction=True)
    token = harness.issue(principal_id, kind=kind)

    result = await _invoke_live_tool(tool_name, token=token, agent_id=None)

    assert result["code"] == "FORBIDDEN"
    assert "agent principal" in result["error"]


@pytest.mark.asyncio
async def test_import_derives_agent_id_from_agent_principal_when_omitted(tmp_path, monkeypatch):
    patch_memory_embeddings(monkeypatch)
    harness = bootstrap_harness(monkeypatch, tmp_path, patch_extraction=True)
    petya = harness.issue("agent:petya", kind="agent")

    imported = await memory_import(
        key="derive-agent-import",
        source_format="text",
        content="import me",
        auth_token=petya,
        scope="agent-private",
    )
    imported_history = await memory_import_history(
        key="derive-agent-import-history",
        source_format="conversation_json",
        content='[{"created_at":"2026-04-08T00:00:00Z","messages":[{"role":"user","content":"hello from history"}]}]',
        auth_token=petya,
        scope="agent-private",
    )

    assert imported["sessions_processed"] == 1
    assert imported_history["sessions_processed"] == 1
    assert mcp_mod.registry["derive-agent-import"]._raw_sessions[0]["agent_id"] == "petya"
    assert mcp_mod.registry["derive-agent-import-history"]._raw_sessions[0]["agent_id"] == "petya"


@pytest.mark.asyncio
async def test_read_and_status_paths_do_not_use_agent_id_as_auth_source(tmp_path, monkeypatch):
    patch_memory_embeddings(monkeypatch)
    harness = bootstrap_harness(monkeypatch, tmp_path, patch_extraction=True)
    alice = harness.issue("agent:alice", kind="agent")
    bob = harness.issue("agent:bob", kind="agent")

    stored = await memory_store(
        key="agent-id-spoof",
        content="Alice private note",
        session_num=1,
        session_date="2026-04-06",
        token=alice,
        scope="agent-private",
    )
    written = await memory_write(
        key="agent-id-spoof-write",
        message_id="msg-spoof",
        session_id="sess-spoof",
        content="Alice raw write",
        content_family="chat",
        timestamp_ms=1712450000001,
        token=alice,
        scope="agent-private",
    )

    assert stored["status"] == "ok"
    assert written["inserted"] is True

    owner_list = await memory_list(key="agent-id-spoof", token=alice, agent_id="mallory")
    owner_status = await memory_write_status(
        key="agent-id-spoof-write",
        message_id="msg-spoof",
        token=alice,
        agent_id="mallory",
    )
    owner_recall = await memory_recall(key="agent-id-spoof", query="private", token=alice, agent_id="mallory")
    spoofed = await memory_list(key="agent-id-spoof", token=bob, agent_id="alice")

    assert owner_list["total"] >= 1
    assert owner_status["extraction_state"] == "pending"
    assert owner_recall["retrieved_count"] >= 1
    assert spoofed["code"] in ("FORBIDDEN", "ACL_FORBIDDEN")


@pytest.mark.asyncio
async def test_live_ingress_tools_require_explicit_scope(tmp_path, monkeypatch):
    patch_memory_embeddings(monkeypatch)
    harness = bootstrap_harness(monkeypatch, tmp_path, patch_extraction=True)
    alice = harness.issue("agent:alice", kind="agent")

    store_result = await memory_store(
        key="scope-required-store",
        content="hello",
        session_num=1,
        session_date="2026-04-06",
        token=alice,
        agent_id="alice",
    )
    write_result = await memory_write(
        key="scope-required-write",
        message_id="msg-1",
        session_id="sess-1",
        content="hello",
        content_family="chat",
        timestamp_ms=1712450000000,
        token=alice,
        agent_id="alice",
    )
    document_result = await memory_ingest_document(
        key="scope-required-doc",
        content="Document body",
        source_id="DOC-1",
        token=alice,
        agent_id="alice",
    )
    secret_result = await memory_store_secret(
        key="scope-required-secret",
        name="DB_PASS",
        value="hunter2",
        token=alice,
        agent_id="alice",
    )

    assert store_result["code"] == "VALIDATION_ERROR"
    assert write_result["code"] == "VALIDATION_ERROR"
    assert document_result["code"] == "VALIDATION_ERROR"
    assert secret_result["code"] == "VALIDATION_ERROR"


@pytest.mark.asyncio
async def test_asserted_import_tool_requires_explicit_scope_or_payload_scope(tmp_path, monkeypatch):
    patch_memory_embeddings(monkeypatch)
    harness = bootstrap_harness(monkeypatch, tmp_path, patch_extraction=True)
    alice = harness.issue("agent:alice", kind="agent")
    facts = [{
        "id": "f1",
        "fact": "Imported fact",
        "kind": "event",
        "entities": [],
        "tags": [],
        "session": 1,
    }]
    raw_sessions = [{
        "raw_session_id": "rs1",
        "session_num": 1,
        "session_date": "2026-04-07",
        "content": "Imported session",
    }]

    missing_scope = await memory_ingest_asserted_facts(
        key="asserted-scope-required",
        facts=[dict(facts[0])],
        raw_sessions=[dict(raw_sessions[0])],
        token=alice,
        agent_id="alice",
    )
    imported = await memory_ingest_asserted_facts(
        key="asserted-scope-required",
        facts=[dict(facts[0])],
        raw_sessions=[dict(raw_sessions[0])],
        token=alice,
        agent_id="alice",
        scope="agent-private",
    )

    assert missing_scope["code"] == "VALIDATION_ERROR"
    assert imported["granular_added"] == 1
    server = mcp_mod.registry["asserted-scope-required"]
    fact = server._all_granular[0]
    assert fact["scope"] == "agent-private"
    assert fact["owner_id"] == "agent:alice"
    assert fact["read"] == []
    assert fact["write"] == []


@pytest.mark.asyncio
async def test_agent_task_artifacts_require_explicit_scope(tmp_path, monkeypatch):
    patch_memory_embeddings(monkeypatch)
    harness = bootstrap_harness(monkeypatch, tmp_path, patch_extraction=True)
    alice = harness.issue("agent:alice", kind="agent")

    result = await memory_ingest_asserted_facts(
        key="agent-task-artifacts",
        facts=[
            {
                "id": "task_result_task-1",
                "fact": "READY",
                "kind": "task_result",
                "session": 1,
                "entities": [],
                "tags": ["agent_result", "task:task-1"],
                "metadata": {
                    "task_id": "task-1",
                    "task_fact_id": "task_fact_1",
                    "status": "done",
                },
            },
            {
                "id": "task_session_task-1",
                "fact": "Agent alice completed task-1.",
                "kind": "task_session",
                "session": 1,
                "entities": ["alice", "task-1"],
                "tags": ["agent_session", "task:task-1"],
                "metadata": {
                    "task_id": "task-1",
                    "task_fact_id": "task_fact_1",
                    "status": "done",
                },
            },
        ],
        token=alice,
        agent_id="alice",
    )

    assert result["code"] == "VALIDATION_ERROR"
    assert "scope" in result["error"].lower()
