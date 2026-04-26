# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

from __future__ import annotations

import json

import pytest

import src.mcp_server as mcp_mod
from src.membership import AuthorityService
from src.mcp_server import (
    auth_bootstrap_admin,
    auth_token_issue,
    auth_token_revoke,
    memory_store,
    membership_grant,
    membership_revoke,
    principal_create,
    principal_disable,
    swarm_create,
)
from src.storage import SQLiteAuthorityStorage
from tests._auth_helpers import bootstrap_harness, configure_bootstrap_env
from tests._memory_embed_mocks import patch_memory_embeddings


@pytest.fixture(autouse=True)
def _reset_registry():
    mcp_mod.registry.clear()
    yield
    mcp_mod.registry.clear()


async def _issue_tool_token(
    *,
    admin_token: str,
    principal_id: str,
    kind: str,
    token_kind: str,
    expires_at: str | None = None,
) -> dict:
    created = await principal_create(
        principal_id=principal_id,
        kind=kind,
        token=admin_token,
    )
    assert created.get("status") == "ok" or created.get("code") == "ALREADY_EXISTS"
    issued = await auth_token_issue(
        principal_id=principal_id,
        token_kind=token_kind,
        expires_at=expires_at,
        token=admin_token,
    )
    assert issued["status"] == "ok"
    return issued


@pytest.mark.asyncio
async def test_bootstrap_is_sealed_after_first_success_same_or_different_principal(tmp_path, monkeypatch):
    bootstrap_token = configure_bootstrap_env(monkeypatch, tmp_path, patch_extraction=True)

    first = await auth_bootstrap_admin(
        principal_id="service:first-admin",
        kind="service",
        token=bootstrap_token,
    )
    second_same = await auth_bootstrap_admin(
        principal_id="service:first-admin",
        kind="service",
        token=bootstrap_token,
    )
    second_other = await auth_bootstrap_admin(
        principal_id="service:other-admin",
        kind="service",
        token=bootstrap_token,
    )

    assert first["status"] == "ok"
    assert second_same["code"] == "BOOTSTRAP_ALREADY_USED"
    assert second_other["code"] == "BOOTSTRAP_ALREADY_USED"
    assert "service:first-admin" in second_same["error"]
    assert "service:first-admin" in second_other["error"]
    assert bootstrap_token not in json.dumps(second_same)
    assert bootstrap_token not in json.dumps(second_other)


def test_bootstrap_state_persists_across_reopen(tmp_path):
    storage = SQLiteAuthorityStorage(str(tmp_path))
    service = AuthorityService(storage)
    issued = service.bootstrap_admin(principal_id="service:first-admin", kind="service")
    assert issued["token_kind"] == "admin"

    reopened = SQLiteAuthorityStorage(str(tmp_path))
    state = reopened.bootstrap_state_get()

    assert state is not None
    assert state["bootstrapped_principal_id"] == "service:first-admin"
    assert state["bootstrapped_at"]
    assert state["bootstrapped_token_id"]


@pytest.mark.asyncio
async def test_persisted_admin_token_can_manage_but_cannot_bootstrap_or_issue_bootstrap(tmp_path, monkeypatch):
    harness = bootstrap_harness(monkeypatch, tmp_path, patch_extraction=True)

    created = await principal_create(
        principal_id="agent:alice",
        kind="agent",
        token=harness.admin_token,
    )
    swarm = await swarm_create(
        swarm_id="alpha",
        owner_principal_id="agent:alice",
        token=harness.admin_token,
    )
    member_token = await _issue_tool_token(
        admin_token=harness.admin_token,
        principal_id="agent:bob",
        kind="agent",
        token_kind="agent",
    )
    granted = await membership_grant(
        swarm_id="alpha",
        principal_id="agent:bob",
        role="member",
        token=harness.admin_token,
    )
    revoked = await membership_revoke(
        swarm_id="alpha",
        principal_id="agent:bob",
        token=harness.admin_token,
    )
    bootstrap_issue = await auth_token_issue(
        principal_id="agent:alice",
        token_kind="bootstrap",
        token=harness.admin_token,
    )
    bootstrap_call = await auth_bootstrap_admin(
        principal_id="service:another-admin",
        kind="service",
        token=harness.admin_token,
    )

    assert created["status"] == "ok"
    assert swarm["status"] == "ok"
    assert member_token["token_kind"] == "agent"
    assert granted["status"] == "ok"
    assert revoked["status"] == "ok"
    assert bootstrap_issue["code"] == "FORBIDDEN"
    assert bootstrap_call["code"] == "FORBIDDEN"


@pytest.mark.asyncio
async def test_revoked_expired_and_disabled_admin_tokens_fail_closed(tmp_path, monkeypatch):
    harness = bootstrap_harness(monkeypatch, tmp_path, patch_extraction=True)
    await principal_create(principal_id="service:ops", kind="service", token=harness.admin_token)

    revoked = await auth_token_issue(
        principal_id="service:ops",
        token_kind="admin",
        token=harness.admin_token,
    )
    await auth_token_revoke(token_id=revoked["token_id"], token=harness.admin_token)
    revoked_result = await principal_create(
        principal_id="agent:revoked-check",
        kind="agent",
        token=revoked["token"],
    )

    expired = await auth_token_issue(
        principal_id="service:ops",
        token_kind="admin",
        expires_at="2000-01-01T00:00:00+00:00",
        token=harness.admin_token,
    )
    expired_result = await principal_create(
        principal_id="agent:expired-check",
        kind="agent",
        token=expired["token"],
    )

    active = await auth_token_issue(
        principal_id="service:ops",
        token_kind="admin",
        token=harness.admin_token,
    )
    await principal_disable(principal_id="service:ops", token=harness.admin_token)
    disabled_result = await principal_create(
        principal_id="agent:disabled-check",
        kind="agent",
        token=active["token"],
    )

    assert revoked_result["code"] == "AUTH_REVOKED"
    assert expired_result["code"] == "AUTH_EXPIRED"
    assert disabled_result["code"] == "AUTH_DISABLED"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("principal_id", "kind", "token_kind"),
    [
        ("agent:alice", "agent", "agent"),
        ("user:mitja", "user", "user"),
        ("agent:joiner", "agent", "join"),
    ],
)
async def test_non_admin_token_classes_cannot_call_admin_surface_or_bootstrap(
    tmp_path,
    monkeypatch,
    principal_id,
    kind,
    token_kind,
):
    harness = bootstrap_harness(monkeypatch, tmp_path, patch_extraction=True)
    issued = await _issue_tool_token(
        admin_token=harness.admin_token,
        principal_id=principal_id,
        kind=kind,
        token_kind=token_kind,
    )

    created = await principal_create(
        principal_id="agent:forbidden",
        kind="agent",
        token=issued["token"],
    )
    swarm = await swarm_create(
        swarm_id="forbidden-swarm",
        owner_principal_id=principal_id,
        token=issued["token"],
    )
    bootstrap = await auth_bootstrap_admin(
        principal_id="service:forbidden-bootstrap",
        kind="service",
        token=issued["token"],
    )

    assert created["code"] == "FORBIDDEN"
    assert swarm["code"] == "FORBIDDEN"
    assert bootstrap["code"] == "FORBIDDEN"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("principal_id", "kind", "token_kind"),
    [
        ("agent:alice", "agent", "agent"),
        ("user:mitja", "user", "user"),
        ("agent:joiner", "agent", "join"),
    ],
)
async def test_revoked_expired_and_disabled_non_admin_tokens_fail_closed_on_memory_tools(
    tmp_path,
    monkeypatch,
    principal_id,
    kind,
    token_kind,
):
    patch_memory_embeddings(monkeypatch)
    harness = bootstrap_harness(monkeypatch, tmp_path, patch_extraction=True)

    revoked = await _issue_tool_token(
        admin_token=harness.admin_token,
        principal_id=principal_id,
        kind=kind,
        token_kind=token_kind,
    )
    await auth_token_revoke(token_id=revoked["token_id"], token=harness.admin_token)
    revoked_result = await memory_store(
        key=f"revoked-{token_kind}",
        content="hello",
        session_num=1,
        session_date="2026-04-06",
        token=revoked["token"],
        agent_id=str(principal_id).split(":", 1)[1],
        scope="agent-private",
    )

    expired = await _issue_tool_token(
        admin_token=harness.admin_token,
        principal_id=principal_id,
        kind=kind,
        token_kind=token_kind,
        expires_at="2000-01-01T00:00:00+00:00",
    )
    expired_result = await memory_store(
        key=f"expired-{token_kind}",
        content="hello",
        session_num=1,
        session_date="2026-04-06",
        token=expired["token"],
        agent_id=str(principal_id).split(":", 1)[1],
        scope="agent-private",
    )

    active = await _issue_tool_token(
        admin_token=harness.admin_token,
        principal_id=principal_id,
        kind=kind,
        token_kind=token_kind,
    )
    await principal_disable(principal_id=principal_id, token=harness.admin_token)
    disabled_result = await memory_store(
        key=f"disabled-{token_kind}",
        content="hello",
        session_num=1,
        session_date="2026-04-06",
        token=active["token"],
        agent_id=str(principal_id).split(":", 1)[1],
        scope="agent-private",
    )

    assert revoked_result["code"] == "AUTH_REVOKED"
    assert expired_result["code"] == "AUTH_EXPIRED"
    assert disabled_result["code"] == "AUTH_DISABLED"


@pytest.mark.asyncio
async def test_agent_id_and_swarm_id_do_not_authenticate_caller_and_no_empty_token_fallback_exists(tmp_path, monkeypatch):
    patch_memory_embeddings(monkeypatch)
    bootstrap_harness(monkeypatch, tmp_path, patch_extraction=True)

    by_agent = await memory_store(
        key="shortcut-agent",
        content="hello",
        session_num=1,
        session_date="2026-04-06",
        token="",
        agent_id="alice",
        scope="agent-private",
    )
    by_swarm = await memory_store(
        key="shortcut-swarm",
        content="hello",
        session_num=1,
        session_date="2026-04-06",
        token="",
        agent_id="alice",
        swarm_id="alpha",
        scope="swarm-shared",
    )

    assert by_agent["code"] == "AUTH_REQUIRED"
    assert by_swarm["code"] == "AUTH_REQUIRED"
    assert not hasattr(mcp_mod, "_verified_auth_resolver")


@pytest.mark.asyncio
async def test_raw_token_values_do_not_leak_in_error_payloads(tmp_path, monkeypatch):
    patch_memory_embeddings(monkeypatch)
    bootstrap_harness(monkeypatch, tmp_path, patch_extraction=True)
    invalid_token = "gm_admin_LEAKME_123456789"

    result = await memory_store(
        key="no-token-leak",
        content="hello",
        session_num=1,
        session_date="2026-04-06",
        token=invalid_token,
        agent_id="alice",
        scope="agent-private",
    )

    assert result["code"] == "INVALID_TOKEN"
    payload = json.dumps(result)
    assert invalid_token not in payload
