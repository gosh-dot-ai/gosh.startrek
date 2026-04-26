# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

import asyncio
from contextlib import suppress

import pytest

from src import mcp_server
from src.courier import Courier
from src.memory import MemoryServer
from tests._auth_helpers import bootstrap_harness, reset_authority_state
from tests._memory_embed_mocks import patch_memory_embeddings


@pytest.fixture
def auth(tmp_path, monkeypatch):
    patch_memory_embeddings(monkeypatch)
    mcp_server.registry.clear()
    mcp_server.courier_registry.clear()
    mcp_server.connections.clear()
    mcp_server.sub_to_conn.clear()
    mcp_server._active_connections.clear()
    return bootstrap_harness(monkeypatch, tmp_path, patch_extraction=True)


@pytest.fixture(autouse=True)
def _reset_runtime():
    yield
    async def _stop_all():
        for courier in list(mcp_server.courier_registry.values()):
            with suppress(Exception):
                await courier.stop()
    with suppress(Exception):
        asyncio.run(_stop_all())
    mcp_server.registry.clear()
    mcp_server.courier_registry.clear()
    mcp_server.connections.clear()
    mcp_server.sub_to_conn.clear()
    mcp_server._active_connections.clear()
    reset_authority_state()


def test_memory_list_and_get_respect_admin_bypass(auth, tmp_path):
    ms = MemoryServer(str(tmp_path), "admin_list")
    asyncio.run(ms.store("Secret", session_num=1, session_date="2024-06-01", owner_id="agent:alice", read=[], write=[]))
    mcp_server.registry["admin_list"] = ms
    fact_id = ms._all_granular[0]["id"]

    result = asyncio.run(mcp_server.memory_list(key="admin_list", token=auth.admin_token))
    got = asyncio.run(mcp_server.memory_get(key="admin_list", fact_id=fact_id, token=auth.admin_token))

    assert result["total"] > 0
    assert "fact" in got


def test_courier_subscribe_uses_persisted_memberships(auth, tmp_path):
    alice = auth.issue("agent:alice", kind="agent")
    bob = auth.issue("agent:bob", kind="agent")
    auth.create_swarm("alpha", "agent:alice")
    auth.grant(auth.admin_token, swarm_id="alpha", principal_id="agent:bob", role="member")

    ms = MemoryServer(str(tmp_path), "courier")
    asyncio.run(ms.store(
        "Swarm data",
        session_num=1,
        session_date="2024-06-01",
        owner_id="agent:alice",
        read=["swarm:alpha"],
        write=["swarm:alpha"],
    ))

    courier = Courier(ms)
    delivered = []

    async def _test():
        async def cb(fact):
            delivered.append(fact)

        resolved = mcp_server._get_authority().resolve_token(bob)
        await courier.subscribe(
            filter={},
            callback=cb,
            deliver_existing=True,
            owner_id=resolved.principal_id,
            memberships=resolved.memberships,
            caller_role=resolved.caller_role,
        )

    asyncio.run(_test())
    assert delivered


@pytest.mark.asyncio
async def test_courier_subscription_stops_after_membership_revoke(auth):
    alice = auth.issue("agent:alice", kind="agent")
    bob = auth.issue("agent:bob", kind="agent")
    auth.create_swarm("alpha", "agent:alice")
    auth.grant(auth.admin_token, swarm_id="alpha", principal_id="agent:bob", role="member")

    mcp_server._active_connections["conn-membership"] = "test"
    mcp_server.connections["conn-membership"] = asyncio.Queue()
    subscribed = await mcp_server.courier_subscribe(
        key="courier-membership",
        connection_id="conn-membership",
        agent_id="bob",
        swarm_id="alpha",
        token=bob,
    )
    assert subscribed["sub_id"].startswith("sub_")

    await mcp_server.membership_revoke(
        swarm_id="alpha",
        principal_id="agent:bob",
        token=auth.admin_token,
    )
    await mcp_server.memory_store(
        key="courier-membership",
        content="shared fact after revoke",
        session_num=1,
        session_date="2026-04-07",
        agent_id="alice",
        swarm_id="alpha",
        scope="swarm-shared",
        token=alice,
    )
    await mcp_server.courier_registry["courier-membership"]._poll()
    assert mcp_server.connections["conn-membership"].empty()


@pytest.mark.asyncio
async def test_courier_subscription_stops_after_token_revoke(auth):
    alice = auth.issue("agent:alice", kind="agent")
    bob = auth.issue("agent:bob", kind="agent")

    mcp_server._active_connections["conn-token"] = "test"
    mcp_server.connections["conn-token"] = asyncio.Queue()
    subscribed = await mcp_server.courier_subscribe(
        key="courier-token",
        connection_id="conn-token",
        agent_id="bob",
        token=bob,
    )
    assert subscribed["sub_id"].startswith("sub_")

    listed = await mcp_server.auth_token_list(principal_id="agent:bob", token=bob)
    token_id = listed["tokens"][0]["token_id"]
    revoked = await mcp_server.auth_token_revoke(token_id=token_id, token=bob)
    assert revoked["status"] == "ok"

    stored = await mcp_server.memory_store(
        key="courier-token",
        content="public fact after token revoke",
        session_num=1,
        session_date="2026-04-07",
        agent_id="alice",
        scope="system-wide",
        token=alice,
    )
    await mcp_server.courier_registry["courier-token"]._poll()
    denied = await mcp_server.memory_list(key="courier-token", agent_id="bob", token=bob)

    assert stored["status"] == "ok"
    assert denied["code"] == "AUTH_REVOKED"
    assert mcp_server.connections["conn-token"].empty()
    assert subscribed["sub_id"] not in mcp_server.courier_registry["courier-token"]._subscriptions


def test_direct_live_write_apis_require_explicit_scope(tmp_path):
    server = MemoryServer(str(tmp_path), "direct-scope")

    with pytest.raises(ValueError, match="scope must be provided explicitly"):
        asyncio.run(server.store("hello", 1, "2026-04-07", scope=None))
    with pytest.raises(ValueError, match="scope must be provided explicitly"):
        asyncio.run(server.write(
            message_id="msg-1",
            session_id="sess-1",
            content="hello",
            content_family="chat",
            timestamp_ms=1712450000000,
            scope=None,
        ))
    with pytest.raises(ValueError, match="scope must be provided explicitly"):
        asyncio.run(server.ingest_document("Document body", source_id="DOC-1", scope=None))


def test_direct_ingest_asserted_facts_fails_closed_without_acl_context(tmp_path):
    server = MemoryServer(str(tmp_path), "asserted-direct")

    result = asyncio.run(server.ingest_asserted_facts(
        facts=[{
            "id": "f1",
            "fact": "Imported fact",
            "kind": "event",
            "entities": [],
            "tags": [],
            "session": 1,
        }],
        raw_sessions=[{
            "raw_session_id": "rs1",
            "session_num": 1,
            "session_date": "2026-04-07",
            "content": "Imported session",
        }],
    ))

    assert result["code"] == "VALIDATION_ERROR"
