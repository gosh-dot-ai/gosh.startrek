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
from src.mcp_server import auth_bootstrap_admin, auth_token_issue, principal_create, swarm_create
from src.memory import MemoryServer

from tests._auth_helpers import bootstrap_harness, configure_bootstrap_env


def _make_fact(fid, owner_id="agent:alice", read=None, **extra):
    fact = {
        "id": fid,
        "fact": f"Fact {fid}",
        "kind": "event",
        "entities": [],
        "tags": [],
        "session": 1,
        "scope": "agent-private",
        "agent_id": "alice",
        "swarm_id": "default",
        "conv_id": "test",
        "owner_id": owner_id,
        "read": read if read is not None else [],
        "write": [],
        "created_at": "2024-01-01T00:00:00+00:00",
    }
    fact.update(extra)
    return fact


def test_admin_role_bypasses_fact_acl(tmp_path):
    ms = MemoryServer(str(tmp_path), "admin1")
    fact = _make_fact("f1", owner_id="agent:alice", read=[])
    assert ms._acl_allows(fact, "agent:bob", caller_role="user") is False
    assert ms._acl_allows(fact, "agent:bob", caller_role="admin") is True


@pytest.mark.asyncio
async def test_env_bootstrap_token_can_create_persisted_admin(tmp_path, monkeypatch):
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
async def test_persisted_admin_token_can_provision_principal(tmp_path, monkeypatch):
    harness = bootstrap_harness(monkeypatch, tmp_path, patch_extraction=True)
    result = await principal_create(
        principal_id="agent:alice",
        kind="agent",
        token=harness.admin_token,
    )
    assert result["status"] == "ok"
    assert result["principal"]["principal_id"] == "agent:alice"


@pytest.mark.asyncio
async def test_env_bootstrap_token_is_bootstrap_only_not_regular_admin_surface(tmp_path, monkeypatch):
    harness = bootstrap_harness(monkeypatch, tmp_path, patch_extraction=True)

    created = await principal_create(
        principal_id="agent:alice",
        kind="agent",
        token=harness.bootstrap_token,
    )
    issued = await auth_token_issue(
        principal_id=harness.admin_principal_id,
        token_kind="admin",
        token=harness.bootstrap_token,
    )
    swarm = await swarm_create(
        swarm_id="alpha",
        owner_principal_id=harness.admin_principal_id,
        token=harness.bootstrap_token,
    )

    assert created["code"] == "FORBIDDEN"
    assert issued["code"] == "FORBIDDEN"
    assert swarm["code"] == "FORBIDDEN"


@pytest.mark.asyncio
async def test_persisted_bootstrap_token_cannot_call_admin_surface(tmp_path, monkeypatch):
    harness = bootstrap_harness(monkeypatch, tmp_path, patch_extraction=True)

    bootstrap_issued = await auth_token_issue(
        principal_id=harness.admin_principal_id,
        token_kind="bootstrap",
        token=harness.admin_token,
    )

    assert bootstrap_issued["code"] == "FORBIDDEN"


@pytest.mark.asyncio
async def test_persisted_admin_token_cannot_call_bootstrap_again(tmp_path, monkeypatch):
    harness = bootstrap_harness(monkeypatch, tmp_path, patch_extraction=True)

    result = await auth_bootstrap_admin(
        principal_id="service:site-admin",
        kind="service",
        token=harness.admin_token,
    )

    assert result["code"] == "FORBIDDEN"


def test_unknown_token_is_not_admin(tmp_path, monkeypatch):
    bootstrap_harness(monkeypatch, tmp_path, patch_extraction=True)
    ctx = mcp_mod._resolve_identity(token="not-admin")
    assert ctx.caller_role == "user"
    assert ctx.authenticated is False
