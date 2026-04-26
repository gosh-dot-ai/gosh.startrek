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


def _seed_server(tmp_path, auth, key="list_test"):
    server = mcp_mod._get_memory(key)
    now = datetime.now(timezone.utc).isoformat()
    orch_token = auth.issue("agent:orch", kind="agent")
    agent_x_token = auth.issue("agent:agent_x", kind="agent")
    auth.issue("agent:analyst", kind="agent")
    auth.create_swarm("sw1", "agent:orch")
    server._all_granular = [
        {"fact": "task A", "kind": "task", "created_at": now,
         "agent_id": "orch", "swarm_id": "sw1", "scope": "swarm-shared",
         "owner_id": "agent:orch", "read": ["swarm:sw1"], "write": ["swarm:sw1"]},
        {"fact": "analysis B", "kind": "analysis", "created_at": now,
         "agent_id": "analyst", "swarm_id": "sw1", "scope": "swarm-shared",
         "owner_id": "agent:analyst", "read": ["swarm:sw1"], "write": ["swarm:sw1"]},
        {"fact": "private C", "kind": "fact", "created_at": now,
         "agent_id": "agent_x", "swarm_id": "sw1", "scope": "agent-private",
         "owner_id": "agent:agent_x", "read": [], "write": []},
    ]
    server._save_cache()
    return server, {"orch": orch_token, "agent_x": agent_x_token}


@pytest.mark.asyncio
async def test_memory_list_returns_all_visible(tmp_path, auth):
    _, tokens = _seed_server(tmp_path, auth)
    result = await mcp_mod.memory_list(
        key="list_test", agent_id="orch", swarm_id="sw1", token=tokens["orch"]
    )
    assert result["total"] == 2  # agent-private not visible to orch
    kinds = {f["kind"] for f in result["facts"]}
    assert "task" in kinds
    assert "analysis" in kinds


@pytest.mark.asyncio
async def test_memory_list_filter_by_kind(tmp_path, auth):
    _, tokens = _seed_server(tmp_path, auth)
    result = await mcp_mod.memory_list(
        key="list_test", agent_id="orch", swarm_id="sw1", kind="task", token=tokens["orch"]
    )
    assert result["total"] == 1
    assert result["facts"][0]["kind"] == "task"


@pytest.mark.asyncio
async def test_memory_list_agent_private_visible_to_owner(tmp_path, auth):
    _, tokens = _seed_server(tmp_path, auth)
    result = await mcp_mod.memory_list(
        key="list_test", agent_id="agent_x", swarm_id="sw1", token=tokens["agent_x"]
    )
    facts_by_kind = {f["kind"] for f in result["facts"]}
    assert "fact" in facts_by_kind


@pytest.mark.asyncio
async def test_memory_list_pagination(tmp_path, auth):
    server = mcp_mod._get_memory("page_test")
    token_a = auth.issue("agent:a", kind="agent")
    auth.create_swarm("sw1", "agent:a")
    now = datetime.now(timezone.utc).isoformat()
    server._all_granular = [
        {"fact": f"fact {i}", "kind": "fact", "created_at": now,
         "agent_id": "a", "swarm_id": "sw1", "scope": "swarm-shared",
         "owner_id": "agent:a", "read": ["swarm:sw1"], "write": ["swarm:sw1"]}
        for i in range(10)
    ]
    server._save_cache()

    page1 = await mcp_mod.memory_list(key="page_test", agent_id="a", swarm_id="sw1",
                                       limit=3, offset=0, token=token_a)
    page2 = await mcp_mod.memory_list(key="page_test", agent_id="a", swarm_id="sw1",
                                       limit=3, offset=3, token=token_a)

    assert len(page1["facts"]) == 3
    assert len(page2["facts"]) == 3
    assert page1["total"] == 10
    assert {f["fact"] for f in page1["facts"]}.isdisjoint(
        {f["fact"] for f in page2["facts"]})


@pytest.mark.asyncio
async def test_memory_list_cross_swarm_isolation(tmp_path, auth):
    """Agent from sw2 cannot see sw1 swarm-shared facts."""
    server = mcp_mod._get_memory("iso_test")
    auth.issue("agent:a", kind="agent")
    token_b = auth.issue("agent:b", kind="agent")
    auth.create_swarm("sw1", "agent:a")
    now = datetime.now(timezone.utc).isoformat()
    server._all_granular = [
        {"fact": "sw1 secret", "kind": "fact", "created_at": now,
         "agent_id": "a", "swarm_id": "sw1", "scope": "swarm-shared",
         "owner_id": "agent:a", "read": ["swarm:sw1"], "write": ["swarm:sw1"]},
    ]
    server._save_cache()

    result = await mcp_mod.memory_list(key="iso_test", agent_id="b", swarm_id="sw2", token=token_b)
    assert result["total"] == 0
