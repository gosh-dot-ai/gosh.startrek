# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

from datetime import datetime, timezone

import pytest

import src.mcp_server as mcp_mod
from tests._auth_helpers import bootstrap_harness


@pytest.fixture()
def auth(tmp_path, monkeypatch):
    mcp_mod.data_dir = str(tmp_path)
    mcp_mod.registry.clear()
    mcp_mod.courier_registry.clear()
    mcp_mod.connections.clear()
    mcp_mod.sub_to_conn.clear()
    return bootstrap_harness(monkeypatch, tmp_path, patch_extraction=True)


def _seed(tmp_path, auth, key="get_test"):
    server = mcp_mod._get_memory(key)
    now = datetime.now(timezone.utc).isoformat()
    token_a = auth.issue("agent:a", kind="agent")
    token_b = auth.issue("agent:b", kind="agent")
    token_agent_x = auth.issue("agent:agent_x", kind="agent")
    auth.create_swarm("sw1", "agent:a")
    auth.grant(auth.admin_token, swarm_id="sw1", principal_id="agent:b")
    server._all_granular = [
        {"id": "fact_001", "fact": "shared fact", "kind": "fact", "created_at": now,
         "agent_id": "a", "swarm_id": "sw1", "scope": "swarm-shared",
         "owner_id": "agent:a", "read": ["swarm:sw1"], "write": ["swarm:sw1"]},
        {"id": "fact_002", "fact": "private fact", "kind": "fact", "created_at": now,
         "agent_id": "agent_x", "swarm_id": "sw1", "scope": "agent-private",
         "owner_id": "agent:agent_x", "read": [], "write": []},
    ]
    server._save_cache()
    return server, {"a": token_a, "b": token_b, "agent_x": token_agent_x}


@pytest.mark.asyncio
async def test_memory_get_returns_fact(tmp_path, auth):
    _, tokens = _seed(tmp_path, auth)
    result = await mcp_mod.memory_get(key="get_test", fact_id="fact_001",
                                       agent_id="b", swarm_id="sw1", token=tokens["b"])
    assert "fact" in result
    assert result["fact"]["id"] == "fact_001"


@pytest.mark.asyncio
async def test_memory_get_not_found(tmp_path, auth):
    _, tokens = _seed(tmp_path, auth)
    result = await mcp_mod.memory_get(key="get_test", fact_id="nonexistent",
                                       agent_id="a", swarm_id="sw1", token=tokens["a"])
    assert result.get("code") == "NOT_FOUND"


@pytest.mark.asyncio
async def test_memory_get_scope_forbidden(tmp_path, auth):
    """Agent B cannot fetch agent_x's private fact by ID."""
    _, tokens = _seed(tmp_path, auth)
    result = await mcp_mod.memory_get(key="get_test", fact_id="fact_002",
                                       agent_id="b", swarm_id="sw1", token=tokens["b"])
    assert result.get("code") == "ACL_FORBIDDEN"


@pytest.mark.asyncio
async def test_memory_get_owner_can_fetch_private(tmp_path, auth):
    """Owner can fetch their own private fact."""
    _, tokens = _seed(tmp_path, auth)
    result = await mcp_mod.memory_get(key="get_test", fact_id="fact_002",
                                       agent_id="agent_x", swarm_id="sw1", token=tokens["agent_x"])
    assert "fact" in result
    assert result["fact"]["fact"] == "private fact"


@pytest.mark.asyncio
async def test_memory_get_searches_all_tiers(tmp_path, auth):
    """memory_get finds facts in cons and cross tiers, not only granular."""
    server = mcp_mod._get_memory("tiers_get")
    token_a = auth.issue("agent:a", kind="agent")
    auth.create_swarm("sw1", "agent:a")
    now = datetime.now(timezone.utc).isoformat()
    server._all_cons = [{"id": "cons_001", "fact": "consolidated", "kind": "fact",
                          "created_at": now, "agent_id": "a", "swarm_id": "sw1",
                          "scope": "swarm-shared",
                          "owner_id": "agent:a", "read": ["swarm:sw1"], "write": ["swarm:sw1"]}]
    server._save_cache()

    result = await mcp_mod.memory_get(key="tiers_get", fact_id="cons_001",
                                       agent_id="a", swarm_id="sw1", token=token_a)
    assert "fact" in result, "memory_get did not find cons-tier fact"
