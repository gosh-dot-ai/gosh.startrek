# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

import asyncio

import src.mcp_server as mcp_mod
from src.mcp_server import memory_list, memory_store

from tests._auth_helpers import bootstrap_harness
from tests._memory_embed_mocks import patch_memory_embeddings


def test_agent_id_no_longer_grants_identity(tmp_path, monkeypatch):
    bootstrap_harness(monkeypatch, tmp_path, patch_extraction=True)
    ctx = mcp_mod._resolve_identity(agent_id="charlie")
    assert ctx.owner_id == "anonymous"
    assert ctx.auth_error_code == "AUTH_REQUIRED"


def test_agent_key_is_not_identity_source(tmp_path, monkeypatch):
    bootstrap_harness(monkeypatch, tmp_path, patch_extraction=True)
    ctx = mcp_mod._resolve_identity(agent_key="user:alice")
    assert ctx.owner_id == "anonymous"
    assert ctx.auth_error_code == "AUTH_REQUIRED"


def test_protected_tool_rejects_missing_or_invalid_token(tmp_path, monkeypatch):
    patch_memory_embeddings(monkeypatch)
    bootstrap_harness(monkeypatch, tmp_path, patch_extraction=True)

    missing = asyncio.run(memory_store(
        key="protected",
        content="hello",
        session_num=1,
        session_date="2026-04-06",
    ))
    invalid = asyncio.run(memory_store(
        key="protected",
        content="hello",
        session_num=1,
        session_date="2026-04-06",
        token="bad-token",
    ))
    assert missing["code"] == "AUTH_REQUIRED"
    assert invalid["code"] == "INVALID_TOKEN"


def test_persisted_principal_token_grants_owned_access(tmp_path, monkeypatch):
    patch_memory_embeddings(monkeypatch)
    harness = bootstrap_harness(monkeypatch, tmp_path, patch_extraction=True)
    alice_token = harness.issue("agent:alice", kind="agent")
    bob_token = harness.issue("agent:bob", kind="agent")

    result = asyncio.run(memory_store(
        key="owned",
        content="Alice note",
        session_num=1,
        session_date="2026-04-06",
        agent_id="alice",
        scope="agent-private",
        token=alice_token,
    ))
    assert result["facts_extracted"] >= 0

    ok = asyncio.run(memory_list(key="owned", agent_id="alice", token=alice_token))
    blocked = asyncio.run(memory_list(key="owned", agent_id="bob", token=bob_token))
    assert ok["total"] >= 1
    assert blocked["code"] in ("FORBIDDEN", "ACL_FORBIDDEN")
