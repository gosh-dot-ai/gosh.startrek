# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

import pytest

import src.mcp_server as mcp_mod
from src.mcp_server import ConnectionContext, _resolve_identity

from tests._auth_helpers import bootstrap_harness


def test_connection_context_defaults():
    ctx = ConnectionContext()
    assert ctx.owner_id == "system"
    assert ctx.agent_id == "default"
    assert ctx.caller_role == "user"
    assert ctx.authenticated is False


def test_missing_token_fails_closed(tmp_path, monkeypatch):
    bootstrap_harness(monkeypatch, tmp_path, patch_extraction=True)
    ctx = _resolve_identity(agent_id="alice", swarm_id="alpha")
    assert ctx.owner_id == "anonymous"
    assert ctx.authenticated is False
    assert ctx.auth_error_code == "AUTH_REQUIRED"
    assert ctx.swarm_id == "alpha"


def test_bootstrap_admin_token_resolves_system_admin(tmp_path, monkeypatch):
    harness = bootstrap_harness(monkeypatch, tmp_path, patch_extraction=True)
    ctx = _resolve_identity(token=harness.bootstrap_token)
    assert ctx.owner_id == "system"
    assert ctx.caller_role == "admin"
    assert ctx.authenticated is True
    assert ctx.token_kind == "bootstrap"


def test_persisted_principal_token_resolves_memberships(tmp_path, monkeypatch):
    harness = bootstrap_harness(monkeypatch, tmp_path, patch_extraction=True)
    alice_token = harness.issue("agent:alice", kind="agent")
    harness.create_swarm("alpha", "agent:alice")

    ctx = _resolve_identity(token=alice_token, swarm_id="beta")
    assert ctx.owner_id == "agent:alice"
    assert ctx.authenticated is True
    assert "swarm:alpha" in ctx.memberships
    assert "swarm:beta" not in ctx.memberships


def test_invalid_token_does_not_fall_back_to_agent_key(tmp_path, monkeypatch):
    bootstrap_harness(monkeypatch, tmp_path, patch_extraction=True)
    ctx = _resolve_identity(token="bad-token", agent_key="agent:evil", agent_id="evil")
    assert ctx.owner_id == "anonymous"
    assert ctx.authenticated is False
    assert ctx.auth_error_code == "INVALID_TOKEN"


@pytest.mark.asyncio
async def test_authorization_bearer_header_resolves_principal_for_tool_call(tmp_path, monkeypatch):
    harness = bootstrap_harness(monkeypatch, tmp_path, patch_extraction=True)
    alice_token = harness.issue("agent:alice", kind="agent")
    token_ctx = mcp_mod._request_principal_token.set(alice_token)
    try:
        ctx = _resolve_identity()
    finally:
        mcp_mod._request_principal_token.reset(token_ctx)

    assert ctx.owner_id == "agent:alice"
    assert ctx.authenticated is True
    assert ctx.auth_source == "principal_token"
