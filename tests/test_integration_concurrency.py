# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

import asyncio

import numpy as np
import pytest

import src.mcp_server as mcp_mod
from tests._auth_helpers import bootstrap_harness

DIM = 3072


@pytest.fixture(autouse=True)
def _patch_and_reset(tmp_path, monkeypatch):
    mcp_mod.data_dir = str(tmp_path)
    mcp_mod.registry.clear()
    mcp_mod.courier_registry.clear()
    mcp_mod.connections.clear()
    mcp_mod.sub_to_conn.clear()

    async def mock_extract(**kwargs):
        text = kwargs.get("session_text", "")
        sn = kwargs.get("session_num", 1)
        facts = [
            {"id": f"f{sn}_{i}", "fact": text, "kind": "event",
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

    yield
    for c in mcp_mod.courier_registry.values():
        c._running = False


@pytest.fixture()
def auth(tmp_path, monkeypatch):
    return bootstrap_harness(monkeypatch, tmp_path, patch_extraction=False)


@pytest.mark.asyncio
async def test_concurrent_mcp_store_no_cross_contamination(tmp_path, auth):
    """50 concurrent MCP memory_store calls across 5 agents — zero cross-contamination."""
    agents = [f"agent_{i}" for i in range(5)]
    tokens = {agent: auth.issue(f"agent:{agent}", kind="agent") for agent in agents}
    auth.create_swarm("sw_concurrent", f"agent:{agents[0]}")
    for agent in agents[1:]:
        auth.grant(auth.admin_token, swarm_id="sw_concurrent", principal_id=f"agent:{agent}")

    async def store_for_agent(agent_id, session_num):
        return await mcp_mod.memory_store(
            key="concurrency_test",
            content=f"Fact from {agent_id} session {session_num}.",
            session_num=session_num,
            session_date="2024-01-01",
            agent_id=agent_id,
            swarm_id="sw_concurrent",
            scope="swarm-shared",
            token=tokens[agent_id],
        )

    tasks = [store_for_agent(agents[i % 5], i + 1) for i in range(50)]
    results = await asyncio.gather(*tasks)
    assert all(result.get("status") == "ok" for result in results)

    server = mcp_mod._get_memory("concurrency_test")
    for fact in server._all_granular:
        aid = fact.get("agent_id")
        assert aid in agents, f"Fact has unknown agent_id: {aid}"
        if "Fact from" in fact.get("fact", ""):
            for other_agent in agents:
                if other_agent in fact.get("fact", "") and other_agent != aid:
                    pytest.fail(
                        f"Cross-contamination: fact from {other_agent} "
                        f"tagged as {aid}: {fact}"
                    )


@pytest.mark.asyncio
async def test_memory_list_consistent_after_concurrent_writes(tmp_path, auth):
    """memory_list returns consistent results after concurrent stores."""
    agents = [f"agent_{i}" for i in range(3)]
    tokens = {agent: auth.issue(f"agent:{agent}", kind="agent") for agent in agents}
    auth.create_swarm("sw1", "agent:agent_0")
    for agent in agents[1:]:
        auth.grant(auth.admin_token, swarm_id="sw1", principal_id=f"agent:{agent}")

    async def write(i):
        agent = f"agent_{i % 3}"
        await mcp_mod.memory_store(
            key="list_concurrent",
            content=f"Item {i} stored by {agent}.",
            session_num=i,
            session_date="2024-01-01",
            agent_id=agent,
            swarm_id="sw1",
            scope="swarm-shared",
            token=tokens[agent],
        )

    await asyncio.gather(*[write(i) for i in range(30)])

    result = await mcp_mod.memory_list(
        key="list_concurrent", agent_id="agent_0", swarm_id="sw1", token=tokens["agent_0"]
    )
    assert result["total"] > 0
    for f in result["facts"]:
        assert "agent_id" in f
        assert "scope" in f
