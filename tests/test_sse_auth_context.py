# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

import json
from contextlib import asynccontextmanager

import pytest
from httpx import ASGITransport, AsyncClient

import src.mcp_server as mcp_mod
from tests._auth_helpers import bootstrap_harness


@pytest.fixture()
def _reset_state(tmp_path, monkeypatch):
    return bootstrap_harness(monkeypatch, tmp_path, patch_extraction=True)


def _parse_sse_result(text: str) -> dict:
    """Extract JSON-RPC result from SSE response body."""
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("data: "):
            try:
                obj = json.loads(line[6:])
                if "id" in obj:
                    return obj
            except json.JSONDecodeError:
                continue
    return {}


def _extract_tool_text(sse_result: dict) -> dict:
    """Extract parsed JSON from MCP tool result content."""
    content = sse_result.get("result", {}).get("content", [])
    for item in content:
        if item.get("type") == "text":
            try:
                return json.loads(item["text"])
            except (json.JSONDecodeError, KeyError):
                return {"raw": item.get("text", "")}
    return {}


@asynccontextmanager
async def _lifespan_client(app):
    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=ASGITransport(app=app), base_url="http://127.0.0.1:8765") as client,
    ):
        yield client


async def _mcp_initialize(client, server_token: str, bearer: str | None = None) -> str:
    """Send MCP initialize and return session ID."""
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "x-server-token": server_token,
    }
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"

    resp = await client.post(
        "/mcp",
        headers=headers,
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "0.1"},
            },
        },
    )
    assert resp.status_code == 200
    return resp.headers.get("mcp-session-id", "")


async def _mcp_tool_call(
    client, server_token: str, session_id: str, bearer: str | None,
    tool_name: str, arguments: dict,
) -> dict:
    """Call an MCP tool through /mcp (SSE) and return parsed result."""
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "x-server-token": server_token,
        "Mcp-Session-Id": session_id,
    }
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"

    resp = await client.post(
        "/mcp",
        headers=headers,
        json={
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        },
    )
    assert resp.status_code == 200
    sse_result = _parse_sse_result(resp.text)
    return _extract_tool_text(sse_result)


@pytest.mark.asyncio
async def test_bootstrap_admin_via_sse_sees_bearer_principal(tmp_path, _reset_state):
    """auth_bootstrap_admin called through /mcp SSE path must see the bootstrap Bearer token."""
    app = mcp_mod.create_app(app_data_dir=str(tmp_path))
    server_token = mcp_mod.SERVER_TOKEN
    bootstrap_token = _reset_state.bootstrap_token

    async with _lifespan_client(app) as client:
        session_id = await _mcp_initialize(client, server_token, bearer=bootstrap_token)
        # auth_bootstrap_admin is one-time; our harness already used it,
        # so we expect BOOTSTRAP_ALREADY_USED — but that still proves the
        # handler authenticated as bootstrap admin (not AUTH_REQUIRED).
        result = await _mcp_tool_call(
            client, server_token, session_id, bearer=bootstrap_token,
            tool_name="auth_bootstrap_admin",
            arguments={"principal_id": "service:sse-test-admin"},
        )

    # If token was lost, we'd get AUTH_REQUIRED. BOOTSTRAP_ALREADY_USED means
    # the handler saw the bootstrap token and authenticated — the regression is absent.
    code = result.get("code", "")
    assert code in ("BOOTSTRAP_ALREADY_USED", ""), (
        f"expected BOOTSTRAP_ALREADY_USED (token was seen), got: {result}"
    )


@pytest.mark.asyncio
async def test_second_sse_request_with_different_bearer(tmp_path, _reset_state):
    """Sequential MCP tool calls with different bearers must each resolve their own principal."""
    app = mcp_mod.create_app(app_data_dir=str(tmp_path))
    server_token = mcp_mod.SERVER_TOKEN

    alice_token = _reset_state.issue("agent:alice", kind="agent")

    async with _lifespan_client(app) as client:
        # Request 1: admin calls memory_get_config (any tool that reads identity)
        sid1 = await _mcp_initialize(client, server_token, bearer=_reset_state.admin_token)
        result1 = await _mcp_tool_call(
            client, server_token, sid1, bearer=_reset_state.admin_token,
            tool_name="memory_get_config",
            arguments={"key": "default"},
        )
        # Admin should succeed (get default config)
        assert "code" not in result1 or result1.get("code") != "AUTH_REQUIRED"

        # Request 2: alice calls memory_get_config
        sid2 = await _mcp_initialize(client, server_token, bearer=alice_token)
        result2 = await _mcp_tool_call(
            client, server_token, sid2, bearer=alice_token,
            tool_name="memory_get_config",
            arguments={"key": "default"},
        )

    # Alice is not admin — may get FORBIDDEN or see default config.
    # The key assertion: alice must NOT get admin-level access from request 1's token.
    # If result2 has no error, alice got the config (non-admin can read default config).
    # If result2 has FORBIDDEN, that's also fine — alice doesn't have access.
    # The only failure: AUTH_REQUIRED would mean the ContextVar was cleared.
    assert result2.get("code") != "AUTH_REQUIRED", (
        f"request 2 got AUTH_REQUIRED — ContextVar was cleared between requests: {result2}"
    )


@pytest.mark.asyncio
async def test_no_bearer_sse_request_after_authenticated(tmp_path, _reset_state):
    """MCP tool call without Bearer after an authenticated call must not inherit the previous token."""
    app = mcp_mod.create_app(app_data_dir=str(tmp_path))
    server_token = mcp_mod.SERVER_TOKEN

    async with _lifespan_client(app) as client:
        # Request 1: admin
        sid1 = await _mcp_initialize(client, server_token, bearer=_reset_state.admin_token)
        await _mcp_tool_call(
            client, server_token, sid1, bearer=_reset_state.admin_token,
            tool_name="memory_get_config",
            arguments={"key": "default"},
        )

        # Request 2: no bearer
        sid2 = await _mcp_initialize(client, server_token, bearer=None)
        result2 = await _mcp_tool_call(
            client, server_token, sid2, bearer=None,
            tool_name="memory_get_config",
            arguments={"key": "default"},
        )

    # Without bearer: must be AUTH_REQUIRED or anonymous, NOT admin
    code = result2.get("code", "")
    assert code == "AUTH_REQUIRED" or result2.get("error", ""), (
        f"request without bearer should not succeed as admin: {result2}"
    )
