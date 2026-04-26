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
from src.config import MemoryConfig
from src.mcp_server import (
    memory_ask,
    memory_delete_secret,
    memory_edit,
    memory_ingest,
    memory_ingest_asserted_facts,
    memory_ingest_document,
    memory_list,
    memory_list_secrets,
    memory_query,
    memory_recall,
    memory_retract,
    memory_set_config,
    memory_set_profiles,
    memory_set_prompt,
    memory_set_schema,
    memory_store,
    memory_store_secret,
)
from tests._auth_helpers import bootstrap_harness
from tests._memory_embed_mocks import patch_memory_embeddings


@pytest.fixture(autouse=True)
def _reset_registry():
    mcp_mod.registry.clear()
    yield
    mcp_mod.registry.clear()


@pytest.fixture
def auth(tmp_path, monkeypatch):
    monkeypatch.setenv("GOSH_MEMORY_ALLOW_PLAINTEXT_SECRETS", "1")
    monkeypatch.setattr(
        mcp_mod,
        "cfg",
        MemoryConfig(extraction_model="", inference_model="", judge_model="", embed_model=""),
    )
    patch_memory_embeddings(monkeypatch)
    return bootstrap_harness(monkeypatch, tmp_path, patch_extraction=True)


async def _shared_seed(auth):
    alice = auth.issue("agent:alice", kind="agent")
    bob = auth.issue("agent:bob", kind="agent")
    mallory = auth.issue("agent:mallory", kind="agent")
    auth.create_swarm("alpha", "agent:alice")
    auth.grant(auth.admin_token, swarm_id="alpha", principal_id="agent:bob", role="member")
    result = await memory_store(
        key="shared",
        content="Shared alpha fact",
        session_num=1,
        session_date="2026-04-06",
        agent_id="alice",
        swarm_id="alpha",
        scope="swarm-shared",
        token=alice,
    )
    assert result["facts_extracted"] >= 0
    artifact_id = mcp_mod.registry["shared"]._all_granular[0]["artifact_id"]
    return alice, bob, mallory, artifact_id


async def _init_instance(
    *,
    key: str,
    token: str,
    agent_id: str = "default",
    swarm_id: str = "default",
    owner_id: str | None = None,
) -> dict:
    from httpx import ASGITransport, AsyncClient

    body = {"key": key}
    if owner_id is not None:
        body["owner_id"] = owner_id
    app = mcp_mod.create_app(app_data_dir=mcp_mod.data_dir)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/admin/memory/init",
            json=body,
            headers={
                "X-GOSH-MEMORY-TOKEN": mcp_mod.SERVER_TOKEN,
                "Authorization": f"Bearer {token}",
            },
        )
    return response.json()


@pytest.mark.asyncio
async def test_swarm_shared_resource_uses_persisted_membership(auth):
    alice, bob, mallory, _artifact_id = await _shared_seed(auth)

    listed = await memory_list(key="shared", agent_id="bob", swarm_id="alpha", token=bob)
    recalled = await memory_recall(key="shared", query="alpha", agent_id="bob", swarm_id="alpha", token=bob)
    queried = await memory_query(key="shared", agent_id="bob", swarm_id="alpha", token=bob)
    asked = await memory_ask(key="shared", query="alpha", agent_id="bob", swarm_id="alpha", token=bob)

    assert listed["total"] >= 1
    assert recalled["retrieved_count"] >= 1
    assert queried["total"] >= 1
    assert "answer" in asked or asked.get("code") == "NO_PROFILES"

    blocked = await memory_list(key="shared", agent_id="mallory", swarm_id="alpha", token=mallory)
    assert blocked["code"] in ("FORBIDDEN", "ACL_FORBIDDEN")

    await mcp_mod.membership_revoke(swarm_id="alpha", principal_id="agent:bob", token=auth.admin_token)
    revoked = await memory_recall(key="shared", query="alpha", agent_id="bob", swarm_id="alpha", token=bob)
    assert revoked["code"] in ("FORBIDDEN", "ACL_FORBIDDEN")


@pytest.mark.asyncio
async def test_swarm_id_param_alone_gives_no_access(auth):
    alice, _bob, mallory, _artifact_id = await _shared_seed(auth)
    blocked = await memory_list(
        key="shared",
        agent_id="mallory",
        swarm_id="alpha",
        token=mallory,
    )
    assert blocked["code"] in ("FORBIDDEN", "ACL_FORBIDDEN")

    owner_ok = await memory_list(key="shared", agent_id="alice", swarm_id="alpha", token=alice)
    assert owner_ok["total"] >= 1


@pytest.mark.asyncio
async def test_edit_and_retract_use_same_membership_resolution(auth):
    _alice, bob, _mallory, artifact_id = await _shared_seed(auth)
    edited = await memory_edit(
        key="shared",
        artifact_id=artifact_id,
        new_content="Bob updates the shared alpha fact",
        agent_id="bob",
        swarm_id="alpha",
        token=bob,
    )
    assert edited.get("code") not in ("FORBIDDEN", "ACL_FORBIDDEN")

    retracted = await memory_retract(
        key="shared",
        artifact_id=artifact_id,
        agent_id="bob",
        swarm_id="alpha",
        token=bob,
    )
    assert retracted["status"] == "retracted"


@pytest.mark.asyncio
async def test_secret_tools_follow_canonical_row_acl(auth):
    alice = auth.issue("agent:alice", kind="agent")
    bob = auth.issue("agent:bob", kind="agent")
    mallory = auth.issue("agent:mallory", kind="agent")
    auth.create_swarm("alpha", "agent:alice")
    auth.grant(auth.admin_token, swarm_id="alpha", principal_id="agent:bob", role="member")
    created = await _init_instance(
        key="secret-key",
        token=alice,
        agent_id="alice",
    )

    stored = await memory_store_secret(
        key="secret-key",
        name="DB_PASS",
        value="hunter2",
        agent_id="alice",
        swarm_id="alpha",
        scope="swarm-shared",
        token=alice,
    )
    assert created["status"] == "ok"
    assert stored["stored"] is True
    assert "value" not in stored

    member = await memory_list_secrets(
        key="secret-key",
        agent_id="bob",
        swarm_id="alpha",
        scope="swarm-shared",
        token=bob,
    )
    assert [secret["name"] for secret in member["secrets"]] == ["DB_PASS"]
    assert "value" not in member["secrets"][0]
    assert member["secrets"][0]["created_by_principal_id"] == "agent:alice"
    assert "hunter2" not in str(member)

    outsider = await memory_list_secrets(
        key="secret-key",
        agent_id="mallory",
        swarm_id="alpha",
        scope="swarm-shared",
        token=mallory,
    )
    assert outsider == {"secrets": []}

    missing = await memory_list_secrets(
        key="secret-key",
        agent_id="bob",
        swarm_id="alpha",
        scope="swarm-shared",
    )
    assert missing["code"] == "AUTH_REQUIRED"


@pytest.mark.asyncio
async def test_secret_swarm_access_uses_canonical_acl_not_raw_swarm_param(auth):
    alice = auth.issue("agent:alice", kind="agent")
    bob = auth.issue("agent:bob", kind="agent")
    mallory = auth.issue("agent:mallory", kind="agent")
    auth.create_swarm("alpha", "agent:alice")
    auth.grant(auth.admin_token, swarm_id="alpha", principal_id="agent:bob", role="member")

    public_seed = await memory_store(
        key="secret-bypass",
        content="public anchor",
        session_num=1,
        session_date="2026-04-07",
        agent_id="alice",
        scope="system-wide",
        token=alice,
    )
    stored = await memory_store_secret(
        key="secret-bypass",
        name="SWARM_ONLY",
        value="alpha-secret",
        agent_id="alice",
        swarm_id="alpha",
        scope="swarm-shared",
        token=alice,
    )
    member = await memory_list_secrets(
        key="secret-bypass",
        agent_id="bob",
        swarm_id="alpha",
        scope="swarm-shared",
        token=bob,
    )
    outsider = await memory_list_secrets(
        key="secret-bypass",
        agent_id="mallory",
        swarm_id="alpha",
        scope="swarm-shared",
        token=mallory,
    )

    assert public_seed["status"] == "ok"
    assert stored["stored"] is True
    assert [secret["name"] for secret in member["secrets"]] == ["SWARM_ONLY"]
    assert outsider == {"secrets": []}


@pytest.mark.asyncio
async def test_secret_store_does_not_widen_instance_write_acl_for_ordinary_data(auth):
    alice = auth.issue("agent:alice", kind="agent")
    bob = auth.issue("agent:bob", kind="agent")
    auth.create_swarm("alpha", "agent:alice")
    auth.grant(auth.admin_token, swarm_id="alpha", principal_id="agent:bob", role="member")
    created = await _init_instance(
        key="secret-no-instance-widen",
        token=alice,
        agent_id="alice",
    )

    stored = await memory_store_secret(
        key="secret-no-instance-widen",
        name="SWARM_ONLY",
        value="alpha-secret",
        agent_id="alice",
        swarm_id="alpha",
        scope="swarm-shared",
        token=alice,
    )
    secret_list = await memory_list_secrets(
        key="secret-no-instance-widen",
        agent_id="bob",
        swarm_id="alpha",
        scope="swarm-shared",
        token=bob,
    )
    ordinary_write = await memory_store(
        key="secret-no-instance-widen",
        content="Bob should not gain unrelated private write access from secret delegation",
        session_num=1,
        session_date="2026-04-07",
        agent_id="bob",
        scope="agent-private",
        token=bob,
    )

    assert created["status"] == "ok"
    assert stored["stored"] is True
    assert [secret["name"] for secret in secret_list["secrets"]] == ["SWARM_ONLY"]
    assert "swarm:alpha" not in mcp_mod.registry["secret-no-instance-widen"]._instance_config.get("_derived_write", [])
    assert ordinary_write["code"] in ("FORBIDDEN", "ACL_FORBIDDEN")


@pytest.mark.asyncio
async def test_secret_store_can_delegate_agent_private_secret_to_other_agent(auth):
    alice = auth.issue("agent:alice", kind="agent")
    petya = auth.issue("agent:petya", kind="agent")
    created = await _init_instance(
        key="secret-spoof",
        token=alice,
        agent_id="alice",
    )

    delegated = await memory_store_secret(
        key="secret-spoof",
        name="PETYA_KEY",
        value="v1",
        agent_id="petya",
        scope="agent-private",
        token=alice,
    )
    owner_list = await memory_list_secrets(
        key="secret-spoof",
        agent_id="petya",
        scope="agent-private",
        token=petya,
    )
    creator_list = await memory_list_secrets(
        key="secret-spoof",
        agent_id="alice",
        scope="agent-private",
        token=alice,
    )

    assert created["status"] == "ok"
    assert delegated["stored"] is True
    assert [secret["name"] for secret in owner_list["secrets"]] == ["PETYA_KEY"]
    assert owner_list["secrets"][0]["owner_id"] == "agent:petya"
    assert owner_list["secrets"][0]["created_by_principal_id"] == "agent:alice"
    assert creator_list == {"secrets": []}


@pytest.mark.asyncio
async def test_admin_can_store_secret_with_the_same_canonical_acl_contract(auth):
    alice = auth.issue("agent:alice", kind="agent")
    bob = auth.issue("agent:bob", kind="agent")
    mallory = auth.issue("agent:mallory", kind="agent")
    auth.create_swarm("alpha", "agent:alice")
    auth.grant(auth.admin_token, swarm_id="alpha", principal_id="agent:bob", role="member")
    created = await _init_instance(
        key="admin-secret-grants",
        token=alice,
        agent_id="alice",
    )

    delegated_private = await memory_store_secret(
        key="admin-secret-grants",
        name="BOB_ONLY",
        value="bob-secret",
        agent_id="bob",
        scope="agent-private",
        token=auth.admin_token,
    )
    delegated_swarm = await memory_store_secret(
        key="admin-secret-grants",
        name="ALPHA_SHARED",
        value="alpha-secret",
        agent_id="alice",
        swarm_id="alpha",
        scope="swarm-shared",
        token=auth.admin_token,
    )
    bob_private = await memory_list_secrets(
        key="admin-secret-grants",
        agent_id="bob",
        scope="agent-private",
        token=bob,
    )
    alice_private = await memory_list_secrets(
        key="admin-secret-grants",
        agent_id="alice",
        scope="agent-private",
        token=alice,
    )
    bob_swarm = await memory_list_secrets(
        key="admin-secret-grants",
        agent_id="bob",
        swarm_id="alpha",
        scope="swarm-shared",
        token=bob,
    )
    mallory_swarm = await memory_list_secrets(
        key="admin-secret-grants",
        agent_id="mallory",
        swarm_id="alpha",
        scope="swarm-shared",
        token=mallory,
    )
    admin_private = await memory_list_secrets(
        key="admin-secret-grants",
        scope="agent-private",
        token=auth.admin_token,
    )
    admin_swarm = await memory_list_secrets(
        key="admin-secret-grants",
        swarm_id="alpha",
        scope="swarm-shared",
        token=auth.admin_token,
    )

    assert created["status"] == "ok"
    assert delegated_private["stored"] is True
    assert delegated_swarm["stored"] is True
    assert [secret["name"] for secret in bob_private["secrets"]] == ["BOB_ONLY"]
    assert bob_private["secrets"][0]["owner_id"] == "agent:bob"
    assert bob_private["secrets"][0]["read"] == []
    assert bob_private["secrets"][0]["write"] == []
    assert alice_private == {"secrets": []}
    assert [secret["name"] for secret in bob_swarm["secrets"]] == ["ALPHA_SHARED"]
    assert bob_swarm["secrets"][0]["owner_id"] == "agent:alice"
    assert bob_swarm["secrets"][0]["read"] == ["swarm:alpha"]
    assert bob_swarm["secrets"][0]["write"] == ["swarm:alpha"]
    assert mallory_swarm == {"secrets": []}
    assert {secret["name"] for secret in admin_private["secrets"]} == {"BOB_ONLY"}
    assert {secret["name"] for secret in admin_swarm["secrets"]} == {"ALPHA_SHARED"}


@pytest.mark.asyncio
async def test_system_wide_secret_uses_public_canonical_acl(auth):
    alice = auth.issue("agent:alice", kind="agent")
    bob = auth.issue("agent:bob", kind="agent")
    auth.create_swarm("alpha", "agent:alice")
    auth.grant(auth.admin_token, swarm_id="alpha", principal_id="agent:bob", role="member")
    created = await _init_instance(
        key="system-secret",
        token=alice,
        agent_id="alice",
    )

    stored = await memory_store_secret(
        key="system-secret",
        name="SYSTEM_KEY",
        value="v1",
        agent_id="alice",
        swarm_id="alpha",
        scope="system-wide",
        token=alice,
    )
    assert created["status"] == "ok"
    public_list = await memory_list_secrets(
        key="system-secret",
        agent_id="bob",
        swarm_id="alpha",
        scope="system-wide",
        token=bob,
    )
    owner_list = await memory_list_secrets(
        key="system-secret",
        agent_id="alice",
        swarm_id="alpha",
        scope="system-wide",
        token=alice,
    )

    assert stored["stored"] is True
    public_names = {secret["name"] for secret in public_list["secrets"]}
    assert "SYSTEM_KEY" in public_names
    system_secret = next(secret for secret in public_list["secrets"] if secret["name"] == "SYSTEM_KEY")
    assert system_secret["owner_id"] == "system"
    assert system_secret["read"] == ["agent:PUBLIC"]
    assert system_secret["write"] == ["agent:PUBLIC"]
    assert owner_list["secrets"] and "value" not in owner_list["secrets"][0]


@pytest.mark.asyncio
async def test_secret_delete_is_creator_only_even_for_member_admin_and_bootstrap(auth):
    alice = auth.issue("agent:alice", kind="agent")
    bob = auth.issue("agent:bob", kind="agent")
    auth.create_swarm("alpha", "agent:alice")
    auth.grant(auth.admin_token, swarm_id="alpha", principal_id="agent:bob", role="member")
    created = await _init_instance(
        key="secret-delete",
        token=alice,
        agent_id="alice",
    )

    stored = await memory_store_secret(
        key="secret-delete",
        name="DB_PASS",
        value="hunter2",
        agent_id="alice",
        swarm_id="alpha",
        scope="swarm-shared",
        token=alice,
    )
    assert created["status"] == "ok"
    member_delete = await memory_delete_secret(
        key="secret-delete",
        name="DB_PASS",
        agent_id="bob",
        swarm_id="alpha",
        scope="swarm-shared",
        token=bob,
    )
    admin_delete = await memory_delete_secret(
        key="secret-delete",
        name="DB_PASS",
        agent_id="alice",
        swarm_id="alpha",
        scope="swarm-shared",
        token=auth.admin_token,
    )
    bootstrap_delete = await memory_delete_secret(
        key="secret-delete",
        name="DB_PASS",
        agent_id="alice",
        swarm_id="alpha",
        scope="swarm-shared",
        token=auth.bootstrap_token,
    )
    creator_delete = await memory_delete_secret(
        key="secret-delete",
        name="DB_PASS",
        agent_id="alice",
        swarm_id="alpha",
        scope="swarm-shared",
        token=alice,
    )

    assert stored["stored"] is True
    assert member_delete["code"] == "SECRET_FORBIDDEN"
    assert admin_delete["code"] == "SECRET_FORBIDDEN"
    assert bootstrap_delete["code"] in ("AUTH_REQUIRED", "FORBIDDEN", "ACL_FORBIDDEN")
    assert creator_delete["deleted"] is True


@pytest.mark.asyncio
async def test_secret_creator_can_delete_after_swarm_membership_revoke(auth):
    alice = auth.issue("agent:alice", kind="agent")
    auth.create_swarm("alpha", "agent:alice")
    created = await _init_instance(
        key="secret-delete-revoked",
        token=alice,
        agent_id="alice",
    )

    stored = await memory_store_secret(
        key="secret-delete-revoked",
        name="DB_PASS",
        value="hunter2",
        agent_id="alice",
        swarm_id="alpha",
        scope="swarm-shared",
        token=alice,
    )
    assert created["status"] == "ok"
    revoked = await mcp_mod.membership_revoke(
        swarm_id="alpha",
        principal_id="agent:alice",
        token=auth.admin_token,
    )
    deleted = await memory_delete_secret(
        key="secret-delete-revoked",
        name="DB_PASS",
        agent_id="alice",
        swarm_id="alpha",
        scope="swarm-shared",
        token=alice,
    )

    assert stored["stored"] is True
    assert revoked["status"] == "ok"
    assert revoked["membership"]["status"] == "revoked"
    assert deleted["deleted"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("label", "call"),
    [
        (
            "memory_set_config",
            lambda token: memory_set_config(
                key="fresh-setup-config",
                config={"schema_version": 1},
                agent_id="admin",
                token=token,
            ),
        ),
        (
            "memory_set_profiles",
            lambda token: memory_set_profiles(
                key="fresh-setup-profiles",
                profiles={1: "fast"},
                profile_configs={
                    "fast": {
                        "model": "openai/gpt-4o-mini",
                        "pricing": {
                            "input_per_1k": 0.0,
                            "output_per_1k": 0.0,
                            "reasoning_per_1k": 0.0,
                            "cache_read_per_1k": 0.0,
                            "cache_write_per_1k": 0.0,
                        },
                    }
                },
                agent_id="admin",
                token=token,
            ),
        ),
        (
            "memory_set_schema",
            lambda token: memory_set_schema(
                key="fresh-setup-schema",
                schema={"priority": {"type": "string"}},
                agent_id="admin",
                token=token,
            ),
        ),
        (
            "memory_set_prompt",
            lambda token: memory_set_prompt(
                key="fresh-setup-prompt",
                content_type="legal",
                prompt="Extract legal clauses.",
                agent_id="admin",
                token=token,
            ),
        ),
    ],
)
async def test_setup_tools_return_not_found_on_fresh_key(auth, label, call):
    result = await call(auth.admin_token)

    assert result["code"] == "NOT_FOUND", label
    key = {
        "memory_set_config": "fresh-setup-config",
        "memory_set_profiles": "fresh-setup-profiles",
        "memory_set_schema": "fresh-setup-schema",
        "memory_set_prompt": "fresh-setup-prompt",
    }[label]
    assert mcp_mod.registry[key]._instance_config is None


@pytest.mark.asyncio
async def test_memory_init_on_fresh_key_creates_instance_with_caller_owner(auth):
    alice = auth.issue("agent:alice", kind="agent")

    result = await _init_instance(
        key="init-fresh",
        token=alice,
        agent_id="alice",
    )

    assert result["status"] == "ok"
    assert result["owner_id"] == "agent:alice"
    assert mcp_mod.registry["init-fresh"]._instance_config["owner_id"] == "agent:alice"


@pytest.mark.asyncio
async def test_memory_init_repeat_returns_already_exists(auth):
    alice = auth.issue("agent:alice", kind="agent")

    first = await _init_instance(
        key="init-repeat",
        token=alice,
        agent_id="alice",
    )
    second = await _init_instance(
        key="init-repeat",
        token=alice,
        agent_id="alice",
    )

    assert first["status"] == "ok"
    assert second["code"] == "ALREADY_EXISTS"


@pytest.mark.asyncio
async def test_memory_init_explicit_owner_allowed_only_for_admin(auth):
    alice = auth.issue("agent:alice", kind="agent")
    auth.issue("agent:bob", kind="agent")
    auth.create_swarm("alpha", "agent:alice")

    denied = await _init_instance(
        key="init-owner-denied",
        token=alice,
        agent_id="alice",
        owner_id="agent:bob",
    )
    allowed = await _init_instance(
        key="init-owner-admin",
        token=auth.admin_token,
        owner_id="agent:bob",
    )
    bad_system = await _init_instance(
        key="init-owner-system",
        token=auth.admin_token,
        owner_id="system",
    )
    bad_anonymous = await _init_instance(
        key="init-owner-anonymous",
        token=auth.admin_token,
        owner_id="anonymous",
    )
    bad_swarm = await _init_instance(
        key="init-owner-swarm",
        token=auth.admin_token,
        owner_id="swarm:alpha",
    )
    missing_principal = await _init_instance(
        key="init-owner-missing",
        token=auth.admin_token,
        owner_id="agent:missing",
    )

    assert denied["code"] == "FORBIDDEN"
    assert mcp_mod.registry["init-owner-denied"]._instance_config is None
    assert allowed["status"] == "ok"
    assert mcp_mod.registry["init-owner-admin"]._instance_config["owner_id"] == "agent:bob"
    assert bad_system["code"] == "VALIDATION_ERROR"
    assert bad_anonymous["code"] == "VALIDATION_ERROR"
    assert bad_swarm["code"] == "VALIDATION_ERROR"
    assert missing_principal["code"] == "NOT_FOUND"
    assert mcp_mod.registry["init-owner-system"]._instance_config is None
    assert mcp_mod.registry["init-owner-anonymous"]._instance_config is None
    assert mcp_mod.registry["init-owner-swarm"]._instance_config is None
    assert mcp_mod.registry["init-owner-missing"]._instance_config is None


@pytest.mark.asyncio
async def test_setup_tools_work_after_memory_init(auth):
    owner = auth.issue("agent:owner", kind="agent")

    created = await _init_instance(
        key="setup-ready",
        token=owner,
        agent_id="owner",
    )
    config = await memory_set_config(
        key="setup-ready",
        config={"schema_version": 1},
        agent_id="owner",
        token=owner,
    )
    profiles = await memory_set_profiles(
        key="setup-ready",
        profiles={1: "fast"},
        profile_configs={
            "fast": {
                "model": "openai/gpt-4o-mini",
                "pricing": {
                    "input_per_1k": 0.0,
                    "output_per_1k": 0.0,
                    "reasoning_per_1k": 0.0,
                    "cache_read_per_1k": 0.0,
                    "cache_write_per_1k": 0.0,
                },
            }
        },
        agent_id="owner",
        token=owner,
    )
    schema = await memory_set_schema(
        key="setup-ready",
        schema={"priority": {"type": "string"}},
        agent_id="owner",
        token=owner,
    )
    prompt = await memory_set_prompt(
        key="setup-ready",
        content_type="legal",
        prompt="Extract legal clauses.",
        agent_id="owner",
        token=owner,
    )

    assert created["status"] == "ok"
    assert config["status"] == "ok"
    assert profiles["status"] == "ok"
    assert schema["status"] == "ok"
    assert prompt["stored"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("key", "call"),
    [
        (
            "create-via-store",
            lambda token: memory_store(
                key="create-via-store",
                content="hello",
                session_num=1,
                session_date="2026-04-09",
                agent_id="alice",
                scope="agent-private",
                token=token,
            ),
        ),
        (
            "create-via-document",
            lambda token: memory_ingest_document(
                key="create-via-document",
                content="Document body",
                source_id="DOC-1",
                agent_id="alice",
                scope="agent-private",
                token=token,
            ),
        ),
        (
            "create-via-ingest",
            lambda token: memory_ingest(
                key="create-via-ingest",
                text="User: hi\nAssistant: hello",
                session_num=1,
                session_date="2026-04-09",
                agent_id="alice",
                scope="agent-private",
                token=token,
            ),
        ),
        (
            "create-via-asserted",
            lambda token: memory_ingest_asserted_facts(
                key="create-via-asserted",
                facts=[
                    {
                        "id": "seed-f1",
                        "fact": "seed fact",
                        "kind": "event",
                        "entities": [],
                        "tags": [],
                        "session": 1,
                    }
                ],
                raw_sessions=[
                    {
                        "raw_session_id": "raw-seed",
                        "session_num": 1,
                        "session_date": "2026-04-09",
                        "content": "seed content",
                    }
                ],
                agent_id="alice",
                scope="agent-private",
                token=token,
            ),
        ),
    ],
)
async def test_data_write_paths_still_create_instance_on_first_write(auth, key, call):
    alice = auth.issue("agent:alice", kind="agent")

    result = await call(alice)

    assert result.get("code") not in ("NOT_FOUND", "FORBIDDEN", "ACL_FORBIDDEN", "VALIDATION_ERROR")
    assert mcp_mod.registry[key]._instance_config["owner_id"] == "agent:alice"


@pytest.mark.asyncio
async def test_config_first_admin_then_first_store_by_agent_does_not_capture_namespace(auth):
    alice = auth.issue("agent:alice", kind="agent")

    config_attempt = await memory_set_config(
        key="config-first-no-capture",
        config={"schema_version": 1},
        agent_id="admin",
        token=auth.admin_token,
    )
    created = await memory_store(
        key="config-first-no-capture",
        content="hello",
        session_num=1,
        session_date="2026-04-09",
        agent_id="alice",
        scope="agent-private",
        token=alice,
    )

    assert config_attempt["code"] == "NOT_FOUND"
    assert created["status"] == "ok"
    assert mcp_mod.registry["config-first-no-capture"]._instance_config["owner_id"] == "agent:alice"


@pytest.mark.asyncio
async def test_store_first_by_agent_then_config_after_admin_preserves_owner(auth):
    alice = auth.issue("agent:alice", kind="agent")

    created = await memory_store(
        key="store-first-then-config",
        content="hello",
        session_num=1,
        session_date="2026-04-09",
        agent_id="alice",
        scope="agent-private",
        token=alice,
    )
    configured = await memory_set_config(
        key="store-first-then-config",
        config={"schema_version": 1},
        agent_id="admin",
        token=auth.admin_token,
    )

    assert created["status"] == "ok"
    assert configured["status"] == "ok"
    assert mcp_mod.registry["store-first-then-config"]._instance_config["owner_id"] == "agent:alice"


@pytest.mark.asyncio
async def test_store_secret_returns_not_found_on_fresh_key(auth):
    alice = auth.issue("agent:alice", kind="agent")

    result = await memory_store_secret(
        key="fresh-secret-setup",
        name="api-key",
        value="super-secret",
        agent_id="alice",
        scope="agent-private",
        token=alice,
    )

    assert result["code"] == "NOT_FOUND"
    assert mcp_mod.registry["fresh-secret-setup"]._instance_config is None


@pytest.mark.asyncio
async def test_memory_init_then_store_secret_succeeds(auth):
    alice = auth.issue("agent:alice", kind="agent")

    created = await _init_instance(
        key="init-then-secret",
        token=alice,
        agent_id="alice",
    )
    stored = await memory_store_secret(
        key="init-then-secret",
        name="api-key",
        value="super-secret",
        agent_id="alice",
        scope="agent-private",
        token=alice,
    )

    assert created["status"] == "ok"
    assert stored["stored"] is True
    assert mcp_mod.registry["init-then-secret"]._instance_config["owner_id"] == "agent:alice"


@pytest.mark.asyncio
async def test_admin_memory_init_then_store_secret_does_not_capture_owner(auth):
    auth.issue("agent:alice", kind="agent")
    created = await _init_instance(
        key="admin-secret-setup",
        token=auth.admin_token,
        owner_id="agent:alice",
    )
    stored = await memory_store_secret(
        key="admin-secret-setup",
        name="api-key",
        value="super-secret",
        agent_id="admin",
        scope="agent-private",
        token=auth.admin_token,
    )

    assert created["status"] == "ok"
    assert stored["stored"] is True
    assert mcp_mod.registry["admin-secret-setup"]._instance_config["owner_id"] == "agent:alice"


@pytest.mark.asyncio
async def test_schema_and_prompt_updates_work_after_first_data_write(auth):
    alice = auth.issue("agent:alice", kind="agent")

    created = await memory_store(
        key="store-first-then-schema-prompt",
        content="hello",
        session_num=1,
        session_date="2026-04-09",
        agent_id="alice",
        scope="agent-private",
        token=alice,
    )
    schema = await memory_set_schema(
        key="store-first-then-schema-prompt",
        schema={"priority": {"type": "string"}},
        agent_id="admin",
        token=auth.admin_token,
    )
    prompt = await memory_set_prompt(
        key="store-first-then-schema-prompt",
        content_type="legal",
        prompt="Extract legal clauses.",
        agent_id="admin",
        token=auth.admin_token,
    )

    assert created["status"] == "ok"
    assert schema["status"] == "ok"
    assert prompt["stored"] is True
    assert mcp_mod.registry["store-first-then-schema-prompt"]._instance_config["owner_id"] == "agent:alice"
