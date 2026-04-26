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
from src.mcp_server import memory_import, memory_ingest_document, memory_list, memory_store
from tests._auth_helpers import bootstrap_harness, reset_authority_state
from tests._memory_embed_mocks import patch_memory_embeddings


@pytest.fixture(autouse=True)
def _reset_registry():
    mcp_mod.registry.clear()
    yield
    mcp_mod.registry.clear()


@pytest.fixture
def auth(tmp_path, monkeypatch):
    patch_memory_embeddings(monkeypatch)
    return bootstrap_harness(monkeypatch, tmp_path, patch_extraction=True)


@pytest.mark.asyncio
async def test_first_write_instance_creation_via_ingest_document_uses_authenticated_owner(auth):
    alice = auth.issue("agent:alice", kind="agent")
    result = await memory_ingest_document(
        key="docs",
        content="Document body",
        source_id="DOC1",
        agent_id="alice",
        scope="agent-private",
        token=alice,
    )
    assert result.get("status") in (None, "ok") or "facts_extracted" in result
    server = mcp_mod.registry["docs"]
    assert server._instance_config["owner_id"] == "agent:alice"


@pytest.mark.asyncio
async def test_memory_import_uses_principal_auth_and_creates_instance(auth):
    alice = auth.issue("agent:alice", kind="agent")
    result = await memory_import(
        key="imports",
        source_format="text",
        content="hello import",
        agent_id="alice",
        scope="agent-private",
        auth_token=alice,
    )
    assert result["sessions_processed"] == 1
    assert mcp_mod.registry["imports"]._instance_config["owner_id"] == "agent:alice"


@pytest.mark.asyncio
async def test_membership_persists_across_reopen_for_acl(auth, tmp_path, monkeypatch):
    alice = auth.issue("agent:alice", kind="agent")
    bob = auth.issue("agent:bob", kind="agent")
    auth.create_swarm("alpha", "agent:alice")
    auth.grant(auth.admin_token, swarm_id="alpha", principal_id="agent:bob", role="member")

    await memory_import(
        key="reopen",
        source_format="text",
        content="shared import",
        agent_id="alice",
        swarm_id="alpha",
        scope="swarm-shared",
        auth_token=alice,
    )
    assert (await memory_list(key="reopen", agent_id="bob", swarm_id="alpha", token=bob))["total"] >= 1

    mcp_mod.registry.clear()
    reset_authority_state()
    mcp_mod.data_dir = str(tmp_path)
    monkeypatch.setenv("GOSH_MEMORY_ADMIN_TOKEN", auth.bootstrap_token)
    mcp_mod.ADMIN_TOKEN = auth.bootstrap_token

    reopened = await memory_list(key="reopen", agent_id="bob", swarm_id="alpha", token=bob)
    assert reopened["total"] >= 1


@pytest.mark.asyncio
async def test_system_wide_store_widens_instance_read_even_when_extraction_emits_zero_facts(auth, monkeypatch):
    alice = auth.issue("agent:alice", kind="agent")
    bob = auth.issue("agent:bob", kind="agent")

    async def _zero_fact_store(self, *args, **kwargs):
        return {"status": "ok", "facts_extracted": 0}

    monkeypatch.setattr("src.memory.MemoryServer.store", _zero_fact_store)

    stored = await memory_store(
        key="public-zero-fact",
        content="public note",
        session_num=1,
        session_date="2026-04-07",
        agent_id="alice",
        scope="system-wide",
        token=alice,
    )
    listed = await memory_list(
        key="public-zero-fact",
        agent_id="bob",
        token=bob,
    )

    assert stored["status"] == "ok"
    assert listed["total"] == 0


@pytest.mark.asyncio
async def test_agent_private_write_cannot_append_into_foreign_key(auth):
    alice = auth.issue("agent:alice", kind="agent")
    bob = auth.issue("agent:bob", kind="agent")

    created = await memory_store(
        key="foreign-private",
        content="alice private",
        session_num=1,
        session_date="2026-04-07",
        agent_id="alice",
        scope="agent-private",
        token=alice,
    )
    denied = await memory_store(
        key="foreign-private",
        content="bob private",
        session_num=2,
        session_date="2026-04-07",
        agent_id="bob",
        scope="agent-private",
        token=bob,
    )

    assert created["status"] == "ok"
    assert denied["code"] == "FORBIDDEN"
    server = mcp_mod.registry["foreign-private"]
    assert len(server._raw_sessions) == 1
    assert server._raw_sessions[0]["owner_id"] == "agent:alice"
