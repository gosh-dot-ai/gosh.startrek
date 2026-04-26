# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

from __future__ import annotations

import hashlib
import ipaddress
import logging
from datetime import datetime, timezone

from httpx import ASGITransport, AsyncClient
import pytest

import src.gosh_secrets as gosh_secrets
import src.mcp_server as mcp_mod
from src.mcp_server import mcp
from src.storage import SQLiteAuthorityStorage
from tests._auth_helpers import bootstrap_harness


@pytest.fixture(autouse=True)
def _reset_state(tmp_path, monkeypatch):
    monkeypatch.setenv("GOSH_MEMORY_ALLOW_PLAINTEXT_SECRETS", "1")
    mcp_mod.data_dir = str(tmp_path)
    mcp_mod.registry.clear()
    mcp_mod.courier_registry.clear()
    mcp_mod.connections.clear()
    mcp_mod.sub_to_conn.clear()
    mcp_mod._active_connections.clear()
    harness = bootstrap_harness(monkeypatch, tmp_path)
    yield harness
    for courier in mcp_mod.courier_registry.values():
        courier._running = False


def _headers(token: str | None, *, use_new_header: bool = True) -> dict[str, str]:
    headers = {}
    if use_new_header:
        headers["X-GOSH-MEMORY-TOKEN"] = mcp_mod.SERVER_TOKEN
    else:
        headers["x-server-token"] = mcp_mod.SERVER_TOKEN
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    return headers


async def _http_bearer(client: AsyncClient, *, token: str | None, body: dict, use_new_header: bool = True):
    return await _http_action(
        client,
        path="/api/v1/gosh-secrets/http/bearer",
        token=token,
        body=body,
        use_new_header=use_new_header,
    )


async def _http_header_value(client: AsyncClient, *, token: str | None, body: dict, use_new_header: bool = True):
    return await _http_action(
        client,
        path="/api/v1/gosh-secrets/http/header-value",
        token=token,
        body=body,
        use_new_header=use_new_header,
    )


async def _http_action(
    client: AsyncClient,
    *,
    path: str,
    token: str | None,
    body: dict,
    use_new_header: bool = True,
):
    return await client.post(
        path,
        json=body,
        headers=_headers(token, use_new_header=use_new_header),
    )


def _allow_public_host(monkeypatch):
    monkeypatch.setattr(
        gosh_secrets,
        "_resolve_host_addresses",
        lambda hostname, port: {ipaddress.ip_address("93.184.216.34")},
    )


@pytest.mark.asyncio
async def test_secret_http_path_is_not_exposed_as_mcp_tool():
    names = {tool.name for tool in await mcp.list_tools()}
    assert "memory_get_secret" not in names
    assert "memory_rotate_secret" not in names
    assert "memory_resolve_secret" not in names
    assert "gosh_secrets_http_bearer" not in names
    assert "gosh_secrets_http_header_value" not in names


@pytest.mark.asyncio
async def test_plaintext_secret_resolve_endpoint_removed(tmp_path, _reset_state):
    app = mcp_mod.create_app(app_data_dir=str(tmp_path))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/v1/secrets/resolve",
            json={"key": "x", "name": "y", "scope": "agent-private", "purpose": "removed"},
            headers=_headers("bogus"),
        )

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_authorized_principal_can_execute_http_bearer_action_without_leaking_secret(
    tmp_path,
    _reset_state,
    monkeypatch,
    caplog,
):
    harness = _reset_state
    _allow_public_host(monkeypatch)
    alice_token = harness.issue("agent:alice", kind="agent")
    server = mcp_mod._get_memory("http_bearer_own")
    known_secret = "top-secret-bearer"
    stored = server.store_secret(
        "API_KEY",
        known_secret,
        agent_id="alice",
        scope="agent-private",
        caller_id="agent:alice",
    )
    captured: dict[str, object] = {}

    async def _fake_request(*, method, url, headers, body):
        captured["method"] = method
        captured["url"] = url
        captured["headers"] = dict(headers)
        captured["body"] = body
        return 201, {"Content-Type": "application/json", "Server": "hidden"}, b'{"ok":true}'

    monkeypatch.setattr(gosh_secrets, "_perform_bearer_http_request", _fake_request)

    app = mcp_mod.create_app(app_data_dir=str(tmp_path))
    with caplog.at_level(logging.INFO, logger="gosh.secrets.audit"):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await _http_bearer(
                client,
                token=alice_token,
                body={
                    "key": "http_bearer_own",
                    "name": "API_KEY",
                    "scope": "agent-private",
                    "purpose": "call-upstream",
                    "method": "POST",
                    "url": "https://api.example.com/token",
                    "headers": {"X-Test": "1"},
                    "body": {"hello": "world"},
                },
            )

    assert stored["stored"] is True
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.json() == {
        "status": 201,
        "headers": {"Content-Type": "application/json"},
        "body": '{"ok":true}',
    }
    assert captured["method"] == "POST"
    assert captured["url"] == "https://api.example.com/token"
    assert captured["headers"]["Authorization"] == f"Bearer {known_secret}"
    assert captured["headers"]["X-Test"] == "1"
    assert captured["headers"]["Content-Type"] == "application/json"
    assert known_secret not in response.text
    assert known_secret not in caplog.text
    assert "call-upstream" in caplog.text


@pytest.mark.asyncio
async def test_authorized_principal_can_execute_http_header_value_action_without_leaking_secret(
    tmp_path,
    _reset_state,
    monkeypatch,
    caplog,
):
    harness = _reset_state
    _allow_public_host(monkeypatch)
    alice_token = harness.issue("agent:alice", kind="agent")
    server = mcp_mod._get_memory("http_header_own")
    known_secret = "top-secret-header"
    stored = server.store_secret(
        "X_API_KEY",
        known_secret,
        agent_id="alice",
        scope="agent-private",
        caller_id="agent:alice",
    )
    captured: dict[str, object] = {}

    async def _fake_request(*, method, url, headers, body):
        captured["method"] = method
        captured["url"] = url
        captured["headers"] = dict(headers)
        captured["body"] = body
        return 202, {"Content-Type": "application/json", "Server": "hidden"}, b'{"ok":true}'

    monkeypatch.setattr(gosh_secrets, "_perform_header_value_http_request", _fake_request)

    app = mcp_mod.create_app(app_data_dir=str(tmp_path))
    with caplog.at_level(logging.INFO, logger="gosh.secrets.audit"):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await _http_header_value(
                client,
                token=alice_token,
                body={
                    "key": "http_header_own",
                    "name": "X_API_KEY",
                    "scope": "agent-private",
                    "purpose": "call-upstream-header",
                    "method": "POST",
                    "url": "https://api.example.com/token",
                    "header_name": "x-api-key",
                    "headers": {"X-Test": "1"},
                    "body": {"hello": "world"},
                },
            )

    assert stored["stored"] is True
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.json() == {
        "status": 202,
        "headers": {"Content-Type": "application/json"},
        "body": '{"ok":true}',
    }
    assert captured["method"] == "POST"
    assert captured["url"] == "https://api.example.com/token"
    assert captured["headers"]["x-api-key"] == known_secret
    assert captured["headers"]["X-Test"] == "1"
    assert captured["headers"]["Content-Type"] == "application/json"
    assert known_secret not in response.text
    assert known_secret not in caplog.text
    assert "call-upstream-header" in caplog.text


@pytest.mark.asyncio
async def test_http_header_value_rejects_caller_override_of_secret_header(tmp_path, _reset_state, monkeypatch):
    harness = _reset_state
    _allow_public_host(monkeypatch)
    alice_token = harness.issue("agent:alice", kind="agent")
    server = mcp_mod._get_memory("http_header_override")
    secret_value = "no-override-secret"
    server.store_secret(
        "X_API_KEY",
        secret_value,
        agent_id="alice",
        scope="agent-private",
        caller_id="agent:alice",
    )

    called = False

    async def _should_not_run(*, method, url, headers, body):
        nonlocal called
        called = True
        return 200, {"Content-Type": "text/plain"}, b"unexpected"

    monkeypatch.setattr(gosh_secrets, "_perform_header_value_http_request", _should_not_run)

    app = mcp_mod.create_app(app_data_dir=str(tmp_path))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await _http_header_value(
            client,
            token=alice_token,
            body={
                "key": "http_header_override",
                "name": "X_API_KEY",
                "scope": "agent-private",
                "purpose": "override-must-fail",
                "method": "GET",
                "url": "https://api.example.com/x",
                "header_name": "x-api-key",
                "headers": {"x-api-key": "evil"},
            },
        )

    assert response.status_code == 400
    assert response.json()["code"] == "VALIDATION_ERROR"
    assert secret_value not in response.text
    assert called is False


@pytest.mark.asyncio
async def test_swarm_member_can_execute_http_bearer_until_membership_revoke(tmp_path, _reset_state, monkeypatch):
    harness = _reset_state
    _allow_public_host(monkeypatch)
    alice_token = harness.issue("agent:alice", kind="agent")
    bob_token = harness.issue("agent:bob", kind="agent")
    harness.create_swarm("alpha", "agent:alice")
    harness.grant(harness.admin_token, swarm_id="alpha", principal_id="agent:bob", role="member")

    server = mcp_mod._get_memory("http_bearer_swarm")
    server.store_secret(
        "TEAM_API_KEY",
        "shared-secret",
        agent_id="alice",
        swarm_id="alpha",
        scope="swarm-shared",
        caller_id="agent:alice",
    )

    async def _fake_request(*, method, url, headers, body):
        return 200, {"Content-Type": "text/plain"}, b"upstream-ok"

    monkeypatch.setattr(gosh_secrets, "_perform_bearer_http_request", _fake_request)

    app = mcp_mod.create_app(app_data_dir=str(tmp_path))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        allowed = await _http_bearer(
            client,
            token=bob_token,
            body={
                "key": "http_bearer_swarm",
                "name": "TEAM_API_KEY",
                "scope": "swarm-shared",
                "swarm_id": "alpha",
                "agent_id": "alice",
                "purpose": "swarm-allowed",
                "method": "GET",
                "url": "https://api.example.com/x",
            },
        )
        revoked = await mcp_mod.membership_revoke(
            swarm_id="alpha",
            principal_id="agent:bob",
            token=harness.admin_token,
        )
        denied = await _http_bearer(
            client,
            token=bob_token,
            body={
                "key": "http_bearer_swarm",
                "name": "TEAM_API_KEY",
                "scope": "swarm-shared",
                "swarm_id": "alpha",
                "agent_id": "alice",
                "purpose": "swarm-after-revoke",
                "method": "GET",
                "url": "https://api.example.com/x",
            },
        )

    assert allowed.status_code == 200
    assert allowed.json()["body"] == "upstream-ok"
    assert revoked["status"] == "ok"
    assert denied.status_code == 403
    assert denied.json()["code"] == "SECRET_FORBIDDEN"


@pytest.mark.asyncio
async def test_swarm_member_can_execute_http_header_value_until_membership_revoke(tmp_path, _reset_state, monkeypatch):
    harness = _reset_state
    _allow_public_host(monkeypatch)
    alice_token = harness.issue("agent:alice", kind="agent")
    bob_token = harness.issue("agent:bob", kind="agent")
    harness.create_swarm("alpha", "agent:alice")
    harness.grant(harness.admin_token, swarm_id="alpha", principal_id="agent:bob", role="member")

    server = mcp_mod._get_memory("http_header_swarm")
    server.store_secret(
        "TEAM_API_KEY",
        "shared-secret",
        agent_id="alice",
        swarm_id="alpha",
        scope="swarm-shared",
        caller_id="agent:alice",
    )

    async def _fake_request(*, method, url, headers, body):
        return 200, {"Content-Type": "text/plain"}, b"upstream-ok"

    monkeypatch.setattr(gosh_secrets, "_perform_header_value_http_request", _fake_request)

    app = mcp_mod.create_app(app_data_dir=str(tmp_path))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        allowed = await _http_header_value(
            client,
            token=bob_token,
            body={
                "key": "http_header_swarm",
                "name": "TEAM_API_KEY",
                "scope": "swarm-shared",
                "swarm_id": "alpha",
                "agent_id": "alice",
                "purpose": "swarm-header-allowed",
                "method": "GET",
                "url": "https://api.example.com/x",
                "header_name": "x-api-key",
            },
        )
        revoked = await mcp_mod.membership_revoke(
            swarm_id="alpha",
            principal_id="agent:bob",
            token=harness.admin_token,
        )
        denied = await _http_header_value(
            client,
            token=bob_token,
            body={
                "key": "http_header_swarm",
                "name": "TEAM_API_KEY",
                "scope": "swarm-shared",
                "swarm_id": "alpha",
                "agent_id": "alice",
                "purpose": "swarm-header-after-revoke",
                "method": "GET",
                "url": "https://api.example.com/x",
                "header_name": "x-api-key",
            },
        )

    assert allowed.status_code == 200
    assert allowed.json()["body"] == "upstream-ok"
    assert revoked["status"] == "ok"
    assert denied.status_code == 403
    assert denied.json()["code"] == "SECRET_FORBIDDEN"


@pytest.mark.asyncio
async def test_bootstrap_env_token_cannot_call_http_bearer(tmp_path, _reset_state, monkeypatch):
    harness = _reset_state
    _allow_public_host(monkeypatch)
    server = mcp_mod._get_memory("http_bearer_bootstrap")
    server.store_secret(
        "BOOTSTRAP_BLOCKED",
        "bootstrap-secret",
        agent_id="alice",
        scope="agent-private",
        caller_id="agent:alice",
    )

    app = mcp_mod.create_app(app_data_dir=str(tmp_path))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await _http_bearer(
            client,
            token=harness.bootstrap_token,
            body={
                "key": "http_bearer_bootstrap",
                "name": "BOOTSTRAP_BLOCKED",
                "scope": "agent-private",
                "agent_id": "alice",
                "purpose": "should-fail",
                "method": "GET",
                "url": "https://api.example.com/x",
            },
        )

    assert response.status_code == 403
    assert response.json()["code"] == "FORBIDDEN"
    assert "bootstrap-secret" not in response.text


@pytest.mark.asyncio
async def test_bootstrap_env_token_cannot_call_http_header_value(tmp_path, _reset_state, monkeypatch):
    harness = _reset_state
    _allow_public_host(monkeypatch)
    server = mcp_mod._get_memory("http_header_bootstrap")
    server.store_secret(
        "BOOTSTRAP_BLOCKED",
        "bootstrap-secret",
        agent_id="alice",
        scope="agent-private",
        caller_id="agent:alice",
    )

    app = mcp_mod.create_app(app_data_dir=str(tmp_path))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await _http_header_value(
            client,
            token=harness.bootstrap_token,
            body={
                "key": "http_header_bootstrap",
                "name": "BOOTSTRAP_BLOCKED",
                "scope": "agent-private",
                "agent_id": "alice",
                "purpose": "should-fail",
                "method": "GET",
                "url": "https://api.example.com/x",
                "header_name": "x-api-key",
            },
        )

    assert response.status_code == 403
    assert response.json()["code"] == "FORBIDDEN"
    assert "bootstrap-secret" not in response.text


@pytest.mark.asyncio
async def test_persisted_bootstrap_token_cannot_call_http_bearer(tmp_path, _reset_state, monkeypatch):
    harness = _reset_state
    _allow_public_host(monkeypatch)
    storage = SQLiteAuthorityStorage(str(tmp_path))
    harness.issue("agent:alice", kind="agent")
    legacy_token = "gm_bootstrap_legacy_secret_token"
    storage.token_insert(
        token_id="tok_legacy_bootstrap_secret",
        principal_id="agent:alice",
        token_hash=hashlib.sha256(legacy_token.encode("utf-8")).digest(),
        token_kind="bootstrap",
        description="legacy bootstrap token",
        issued_at=datetime.now(timezone.utc).isoformat(),
        issued_by="system",
        expires_at=None,
        metadata={},
    )
    server = mcp_mod._get_memory("http_bearer_legacy_bootstrap")
    server.store_secret(
        "LEGACY_BOOTSTRAP_BLOCKED",
        "legacy-bootstrap-secret",
        agent_id="alice",
        scope="agent-private",
        caller_id="agent:alice",
    )

    app = mcp_mod.create_app(app_data_dir=str(tmp_path))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await _http_bearer(
            client,
            token=legacy_token,
            body={
                "key": "http_bearer_legacy_bootstrap",
                "name": "LEGACY_BOOTSTRAP_BLOCKED",
                "scope": "agent-private",
                "agent_id": "alice",
                "purpose": "legacy-bootstrap-must-fail",
                "method": "GET",
                "url": "https://api.example.com/x",
            },
        )

    assert response.status_code == 403
    assert response.json()["code"] == "FORBIDDEN"
    assert "legacy-bootstrap-secret" not in response.text


@pytest.mark.asyncio
async def test_persisted_bootstrap_token_cannot_call_http_header_value(tmp_path, _reset_state, monkeypatch):
    harness = _reset_state
    _allow_public_host(monkeypatch)
    storage = SQLiteAuthorityStorage(str(tmp_path))
    harness.issue("agent:alice", kind="agent")
    legacy_token = "gm_bootstrap_legacy_secret_token_header"
    storage.token_insert(
        token_id="tok_legacy_bootstrap_secret_header",
        principal_id="agent:alice",
        token_hash=hashlib.sha256(legacy_token.encode("utf-8")).digest(),
        token_kind="bootstrap",
        description="legacy bootstrap token",
        issued_at=datetime.now(timezone.utc).isoformat(),
        issued_by="system",
        expires_at=None,
        metadata={},
    )
    server = mcp_mod._get_memory("http_header_legacy_bootstrap")
    server.store_secret(
        "LEGACY_BOOTSTRAP_BLOCKED",
        "legacy-bootstrap-secret",
        agent_id="alice",
        scope="agent-private",
        caller_id="agent:alice",
    )

    app = mcp_mod.create_app(app_data_dir=str(tmp_path))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await _http_header_value(
            client,
            token=legacy_token,
            body={
                "key": "http_header_legacy_bootstrap",
                "name": "LEGACY_BOOTSTRAP_BLOCKED",
                "scope": "agent-private",
                "agent_id": "alice",
                "purpose": "legacy-bootstrap-must-fail",
                "method": "GET",
                "url": "https://api.example.com/x",
                "header_name": "x-api-key",
            },
        )

    assert response.status_code == 403
    assert response.json()["code"] == "FORBIDDEN"
    assert "legacy-bootstrap-secret" not in response.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("setup_mode", "expected_code"),
    [
        ("revoked", "AUTH_REVOKED"),
        ("expired", "AUTH_EXPIRED"),
        ("disabled", "AUTH_DISABLED"),
    ],
)
async def test_invalidated_tokens_fail_closed_for_http_bearer(tmp_path, _reset_state, monkeypatch, setup_mode, expected_code):
    harness = _reset_state
    _allow_public_host(monkeypatch)
    principal_id = "agent:alice"
    agent_token = harness.issue(principal_id, kind="agent")
    server = mcp_mod._get_memory(f"http_bearer_{setup_mode}")
    server.store_secret(
        "TOKEN_STATE_SECRET",
        "never-leak-this-value",
        agent_id="alice",
        scope="agent-private",
        caller_id=principal_id,
    )

    if setup_mode == "revoked":
        token_row = mcp_mod._get_authority()._storage.resolve_token_hash(hashlib.sha256(agent_token.encode("utf-8")).digest())
        assert token_row is not None
        mcp_mod._get_authority()._storage.token_revoke(
            token_row["token_id"],
            revoked_at=datetime.now(timezone.utc).isoformat(),
            revoked_by=harness.admin_principal_id,
        )
    elif setup_mode == "expired":
        expired = mcp_mod._get_authority().issue_token(
            actor=mcp_mod._get_authority().resolve_token(harness.admin_token),
            principal_id=principal_id,
            token_kind="agent",
            expires_at="2000-01-01T00:00:00+00:00",
        )
        agent_token = expired["token"]
    else:
        mcp_mod._get_authority()._storage.principal_set_status(principal_id, "disabled")

    app = mcp_mod.create_app(app_data_dir=str(tmp_path))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await _http_bearer(
            client,
            token=agent_token,
            body={
                "key": f"http_bearer_{setup_mode}",
                "name": "TOKEN_STATE_SECRET",
                "scope": "agent-private",
                "purpose": f"{setup_mode}-must-fail",
                "method": "GET",
                "url": "https://api.example.com/x",
            },
        )

    expected_status = 401 if expected_code.startswith("AUTH_") else 403
    assert response.status_code == expected_status
    assert response.json()["code"] == expected_code
    assert "never-leak-this-value" not in response.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("setup_mode", "expected_code"),
    [
        ("revoked", "AUTH_REVOKED"),
        ("expired", "AUTH_EXPIRED"),
        ("disabled", "AUTH_DISABLED"),
    ],
)
async def test_invalidated_tokens_fail_closed_for_http_header_value(
    tmp_path,
    _reset_state,
    monkeypatch,
    setup_mode,
    expected_code,
):
    harness = _reset_state
    _allow_public_host(monkeypatch)
    principal_id = "agent:alice"
    agent_token = harness.issue(principal_id, kind="agent")
    server = mcp_mod._get_memory(f"http_header_{setup_mode}")
    server.store_secret(
        "TOKEN_STATE_SECRET",
        "never-leak-this-value",
        agent_id="alice",
        scope="agent-private",
        caller_id=principal_id,
    )

    if setup_mode == "revoked":
        token_row = mcp_mod._get_authority()._storage.resolve_token_hash(hashlib.sha256(agent_token.encode("utf-8")).digest())
        assert token_row is not None
        mcp_mod._get_authority()._storage.token_revoke(
            token_row["token_id"],
            revoked_at=datetime.now(timezone.utc).isoformat(),
            revoked_by=harness.admin_principal_id,
        )
    elif setup_mode == "expired":
        expired = mcp_mod._get_authority().issue_token(
            actor=mcp_mod._get_authority().resolve_token(harness.admin_token),
            principal_id=principal_id,
            token_kind="agent",
            expires_at="2000-01-01T00:00:00+00:00",
        )
        agent_token = expired["token"]
    else:
        mcp_mod._get_authority()._storage.principal_set_status(principal_id, "disabled")

    app = mcp_mod.create_app(app_data_dir=str(tmp_path))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await _http_header_value(
            client,
            token=agent_token,
            body={
                "key": f"http_header_{setup_mode}",
                "name": "TOKEN_STATE_SECRET",
                "scope": "agent-private",
                "purpose": f"{setup_mode}-must-fail",
                "method": "GET",
                "url": "https://api.example.com/x",
                "header_name": "x-api-key",
            },
        )

    expected_status = 401 if expected_code.startswith("AUTH_") else 403
    assert response.status_code == expected_status
    assert response.json()["code"] == expected_code
    assert "never-leak-this-value" not in response.text


@pytest.mark.asyncio
async def test_agent_id_and_swarm_id_do_not_authenticate_http_bearer(tmp_path, _reset_state, monkeypatch):
    _ = _reset_state
    _allow_public_host(monkeypatch)
    app = mcp_mod.create_app(app_data_dir=str(tmp_path))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await _http_bearer(
            client,
            token=None,
            body={
                "key": "http_bearer_missing_auth",
                "name": "ANY",
                "scope": "swarm-shared",
                "agent_id": "alice",
                "swarm_id": "alpha",
                "purpose": "missing-auth",
                "method": "GET",
                "url": "https://api.example.com/x",
            },
        )

    assert response.status_code == 401
    assert response.json()["code"] == "AUTH_REQUIRED"


@pytest.mark.asyncio
async def test_agent_id_and_swarm_id_do_not_authenticate_http_header_value(tmp_path, _reset_state, monkeypatch):
    _ = _reset_state
    _allow_public_host(monkeypatch)
    app = mcp_mod.create_app(app_data_dir=str(tmp_path))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await _http_header_value(
            client,
            token=None,
            body={
                "key": "http_header_missing_auth",
                "name": "ANY",
                "scope": "swarm-shared",
                "agent_id": "alice",
                "swarm_id": "alpha",
                "purpose": "missing-auth",
                "method": "GET",
                "url": "https://api.example.com/x",
                "header_name": "x-api-key",
            },
        )

    assert response.status_code == 401
    assert response.json()["code"] == "AUTH_REQUIRED"


@pytest.mark.asyncio
async def test_http_bearer_requires_exact_canonical_lookup_without_fallback(tmp_path, _reset_state, monkeypatch):
    harness = _reset_state
    _allow_public_host(monkeypatch)
    alice_token = harness.issue("agent:alice", kind="agent")
    server = mcp_mod._get_memory("http_bearer_exact")
    server.store_secret(
        "EXACT_ONLY",
        "system-secret",
        agent_id="alice",
        scope="agent-private",
        caller_id="agent:alice",
    )

    app = mcp_mod.create_app(app_data_dir=str(tmp_path))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await _http_bearer(
            client,
            token=alice_token,
            body={
                "key": "http_bearer_exact",
                "name": "EXACT_ONLY",
                "scope": "system-wide",
                "purpose": "wrong-domain-should-not-fallback",
                "method": "GET",
                "url": "https://api.example.com/x",
            },
        )

    assert response.status_code == 404
    assert response.json()["code"] == "SECRET_NOT_FOUND"


@pytest.mark.asyncio
async def test_http_header_value_requires_exact_canonical_lookup_without_fallback(tmp_path, _reset_state, monkeypatch):
    harness = _reset_state
    _allow_public_host(monkeypatch)
    alice_token = harness.issue("agent:alice", kind="agent")
    server = mcp_mod._get_memory("http_header_exact")
    server.store_secret(
        "EXACT_ONLY",
        "system-secret",
        agent_id="alice",
        scope="agent-private",
        caller_id="agent:alice",
    )

    app = mcp_mod.create_app(app_data_dir=str(tmp_path))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await _http_header_value(
            client,
            token=alice_token,
            body={
                "key": "http_header_exact",
                "name": "EXACT_ONLY",
                "scope": "system-wide",
                "purpose": "wrong-domain-should-not-fallback",
                "method": "GET",
                "url": "https://api.example.com/x",
                "header_name": "x-api-key",
            },
        )

    assert response.status_code == 404
    assert response.json()["code"] == "SECRET_NOT_FOUND"


@pytest.mark.asyncio
async def test_http_bearer_blocks_upstream_secret_reflection_and_never_logs_secret(
    tmp_path,
    _reset_state,
    monkeypatch,
    caplog,
):
    harness = _reset_state
    _allow_public_host(monkeypatch)
    alice_token = harness.issue("agent:alice", kind="agent")
    server = mcp_mod._get_memory("http_bearer_no_leak")
    known_secret = "dont-log-this-secret"
    server.store_secret(
        "PRIVATE_KEY",
        known_secret,
        agent_id="alice",
        scope="agent-private",
        caller_id="agent:alice",
    )

    async def _reflect_secret(*, method, url, headers, body):
        echoed = headers["Authorization"].encode("utf-8")
        return 200, {"Content-Type": "text/plain"}, echoed

    monkeypatch.setattr(gosh_secrets, "_perform_bearer_http_request", _reflect_secret)

    app = mcp_mod.create_app(app_data_dir=str(tmp_path))
    with caplog.at_level(logging.INFO, logger="gosh.secrets.audit"):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await _http_bearer(
                client,
                token=alice_token,
                body={
                    "key": "http_bearer_no_leak",
                    "name": "PRIVATE_KEY",
                    "scope": "agent-private",
                    "purpose": "no-secret-reflection",
                    "method": "GET",
                    "url": "https://api.example.com/x",
                },
            )

    assert response.status_code == 502
    assert response.json()["code"] == "SECRET_LEAK_BLOCKED"
    assert known_secret not in response.text
    assert known_secret not in caplog.text


@pytest.mark.asyncio
async def test_http_header_value_blocks_upstream_secret_reflection_and_never_logs_secret(
    tmp_path,
    _reset_state,
    monkeypatch,
    caplog,
):
    harness = _reset_state
    _allow_public_host(monkeypatch)
    alice_token = harness.issue("agent:alice", kind="agent")
    server = mcp_mod._get_memory("http_header_no_leak")
    known_secret = "dont-log-this-secret-header"
    server.store_secret(
        "PRIVATE_KEY",
        known_secret,
        agent_id="alice",
        scope="agent-private",
        caller_id="agent:alice",
    )

    async def _reflect_secret(*, method, url, headers, body):
        echoed = headers["x-api-key"].encode("utf-8")
        return 200, {"Content-Type": "text/plain"}, echoed

    monkeypatch.setattr(gosh_secrets, "_perform_header_value_http_request", _reflect_secret)

    app = mcp_mod.create_app(app_data_dir=str(tmp_path))
    with caplog.at_level(logging.INFO, logger="gosh.secrets.audit"):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await _http_header_value(
                client,
                token=alice_token,
                body={
                    "key": "http_header_no_leak",
                    "name": "PRIVATE_KEY",
                    "scope": "agent-private",
                    "purpose": "no-secret-reflection-header",
                    "method": "GET",
                    "url": "https://api.example.com/x",
                    "header_name": "x-api-key",
                },
            )

    assert response.status_code == 502
    assert response.json()["code"] == "SECRET_LEAK_BLOCKED"
    assert known_secret not in response.text
    assert known_secret not in caplog.text


@pytest.mark.asyncio
async def test_admin_has_no_special_secret_use_bypass(tmp_path, _reset_state, monkeypatch):
    harness = _reset_state
    _allow_public_host(monkeypatch)
    server = mcp_mod._get_memory("http_bearer_admin_denied")
    server.store_secret(
        "ADMIN_BLOCKED",
        "private-value",
        agent_id="alice",
        scope="agent-private",
        caller_id="agent:alice",
    )

    app = mcp_mod.create_app(app_data_dir=str(tmp_path))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await _http_bearer(
            client,
            token=harness.admin_token,
            body={
                "key": "http_bearer_admin_denied",
                "name": "ADMIN_BLOCKED",
                "scope": "agent-private",
                "agent_id": "alice",
                "purpose": "admin-is-not-a-secret-bypass",
                "method": "GET",
                "url": "https://api.example.com/x",
            },
        )

    assert response.status_code == 403
    assert response.json()["code"] == "SECRET_FORBIDDEN"


@pytest.mark.asyncio
async def test_admin_has_no_special_secret_use_bypass_for_http_header_value(tmp_path, _reset_state, monkeypatch):
    harness = _reset_state
    _allow_public_host(monkeypatch)
    server = mcp_mod._get_memory("http_header_admin_denied")
    server.store_secret(
        "ADMIN_BLOCKED",
        "private-value",
        agent_id="alice",
        scope="agent-private",
        caller_id="agent:alice",
    )

    app = mcp_mod.create_app(app_data_dir=str(tmp_path))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await _http_header_value(
            client,
            token=harness.admin_token,
            body={
                "key": "http_header_admin_denied",
                "name": "ADMIN_BLOCKED",
                "scope": "agent-private",
                "agent_id": "alice",
                "purpose": "admin-is-not-a-secret-bypass",
                "method": "GET",
                "url": "https://api.example.com/x",
                "header_name": "x-api-key",
            },
        )

    assert response.status_code == 403
    assert response.json()["code"] == "SECRET_FORBIDDEN"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("url", "expected_code"),
    [
        ("file:///etc/passwd", "VALIDATION_ERROR"),
        ("http://127.0.0.1:8000/secret", "SSRF_FORBIDDEN"),
        ("http://10.0.0.9:9000/secret", "SSRF_FORBIDDEN"),
    ],
)
async def test_http_bearer_ssrf_guards_block_bad_scheme_and_private_targets(
    tmp_path,
    _reset_state,
    monkeypatch,
    url,
    expected_code,
):
    harness = _reset_state
    alice_token = harness.issue("agent:alice", kind="agent")
    server = mcp_mod._get_memory("http_bearer_ssrf")
    server.store_secret(
        "SSRF_KEY",
        "ssrf-secret",
        agent_id="alice",
        scope="agent-private",
        caller_id="agent:alice",
    )

    called = False

    async def _should_not_run(*, method, url, headers, body):
        nonlocal called
        called = True
        return 200, {"Content-Type": "text/plain"}, b"unexpected"

    monkeypatch.setattr(gosh_secrets, "_perform_bearer_http_request", _should_not_run)

    app = mcp_mod.create_app(app_data_dir=str(tmp_path))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await _http_bearer(
            client,
            token=alice_token,
            body={
                "key": "http_bearer_ssrf",
                "name": "SSRF_KEY",
                "scope": "agent-private",
                "purpose": "ssrf-guard",
                "method": "GET",
                "url": url,
            },
        )

    status = 400 if expected_code == "VALIDATION_ERROR" else 403
    assert response.status_code == status
    assert response.json()["code"] == expected_code
    assert called is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("url", "expected_code"),
    [
        ("file:///etc/passwd", "VALIDATION_ERROR"),
        ("http://127.0.0.1:8000/secret", "SSRF_FORBIDDEN"),
        ("http://10.0.0.9:9000/secret", "SSRF_FORBIDDEN"),
    ],
)
async def test_http_header_value_ssrf_guards_block_bad_scheme_and_private_targets(
    tmp_path,
    _reset_state,
    monkeypatch,
    url,
    expected_code,
):
    harness = _reset_state
    alice_token = harness.issue("agent:alice", kind="agent")
    server = mcp_mod._get_memory("http_header_ssrf")
    server.store_secret(
        "SSRF_KEY",
        "ssrf-secret",
        agent_id="alice",
        scope="agent-private",
        caller_id="agent:alice",
    )

    called = False

    async def _should_not_run(*, method, url, headers, body):
        nonlocal called
        called = True
        return 200, {"Content-Type": "text/plain"}, b"unexpected"

    monkeypatch.setattr(gosh_secrets, "_perform_header_value_http_request", _should_not_run)

    app = mcp_mod.create_app(app_data_dir=str(tmp_path))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await _http_header_value(
            client,
            token=alice_token,
            body={
                "key": "http_header_ssrf",
                "name": "SSRF_KEY",
                "scope": "agent-private",
                "purpose": "ssrf-guard",
                "method": "GET",
                "url": url,
                "header_name": "x-api-key",
            },
        )

    status = 400 if expected_code == "VALIDATION_ERROR" else 403
    assert response.status_code == status
    assert response.json()["code"] == expected_code
    assert called is False
