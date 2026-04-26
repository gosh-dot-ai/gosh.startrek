# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

from __future__ import annotations

import base64
import hashlib
import logging
from datetime import datetime, timezone

from httpx import ASGITransport, AsyncClient
import pytest

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


def _headers(token: str | None, *, include_perimeter: bool = True) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if include_perimeter:
        headers["X-GOSH-MEMORY-TOKEN"] = mcp_mod.SERVER_TOKEN
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    return headers


async def _resolve_agent_secrets(
    client: AsyncClient,
    *,
    token: str | None,
    body: dict,
    include_perimeter: bool = True,
):
    return await client.post(
        "/api/v1/agent/secrets/resolve",
        json=body,
        headers=_headers(token, include_perimeter=include_perimeter),
    )


async def _register_agent_public_key(
    client: AsyncClient,
    *,
    token: str | None,
    public_key: str,
    principal_id: str | None = None,
    algorithm: str = "x25519",
    key_id: str | None = None,
    include_perimeter: bool = True,
):
    body: dict[str, str] = {"public_key": public_key, "algorithm": algorithm}
    if principal_id is not None:
        body["principal_id"] = principal_id
    if key_id is not None:
        body["key_id"] = key_id
    return await client.post(
        "/api/v1/agent/public-key/register",
        json=body,
        headers=_headers(token, include_perimeter=include_perimeter),
    )


def _generate_keypair() -> tuple[object, str]:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import x25519

    private_key = x25519.X25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return private_key, base64.b64encode(public_key).decode("ascii")


def _fingerprint(public_key_b64: str) -> str:
    raw = base64.b64decode(public_key_b64.encode("ascii"))
    return f"sha256:{hashlib.sha256(raw).hexdigest()}"


def _decrypt_ciphertext(private_key: object, ciphertext_b64: str) -> str:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF

    envelope = base64.b64decode(ciphertext_b64.encode("ascii"))
    assert envelope[:4] == b"GMS1"
    ephemeral_public = envelope[4:36]
    nonce = envelope[36:48]
    ciphertext = envelope[48:]

    shared_key = private_key.exchange(
        __import__(
            "cryptography.hazmat.primitives.asymmetric.x25519",
            fromlist=["X25519PublicKey"],
        ).X25519PublicKey.from_public_bytes(ephemeral_public)
    )
    aes_key = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=b"gosh.memory/agent-secrets/v1",
    ).derive(shared_key)
    plaintext = AESGCM(aes_key).decrypt(
        nonce,
        ciphertext,
        b"gosh.memory/agent-secrets/v1",
    )
    return plaintext.decode("utf-8")


@pytest.mark.asyncio
async def test_agent_secret_resolve_is_not_exposed_as_mcp_tool():
    names = {tool.name for tool in await mcp.list_tools()}
    assert "memory_get_secret" not in names
    assert "memory_rotate_secret" not in names
    assert "agent_secrets_resolve" not in names


@pytest.mark.asyncio
async def test_agent_secret_resolve_requires_bearer_and_perimeter(tmp_path, _reset_state):
    app = mcp_mod.create_app(app_data_dir=str(tmp_path))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        missing_bearer = await _resolve_agent_secrets(
            client,
            token=None,
            body={"key": "default", "refs": [{"name": "anthropic", "scope": "system-wide"}]},
        )
        missing_perimeter = await _resolve_agent_secrets(
            client,
            token=_reset_state.admin_token,
            body={"key": "default", "refs": [{"name": "anthropic", "scope": "system-wide"}]},
            include_perimeter=False,
        )

    assert missing_bearer.status_code == 401
    assert missing_bearer.json()["code"] == "AUTH_REQUIRED"
    assert missing_perimeter.status_code == 401


@pytest.mark.asyncio
async def test_agent_with_acl_access_gets_only_encrypted_blob_and_correct_key_can_decrypt(
    tmp_path,
    _reset_state,
    caplog,
):
    harness = _reset_state
    alice_token = harness.issue("agent:alice", kind="agent")
    private_key, public_key_b64 = _generate_keypair()
    wrong_private_key, _ = _generate_keypair()

    server = mcp_mod._get_memory("agent-secret-own")
    secret_value = "anthropic-top-secret"
    stored = server.store_secret(
        "anthropic",
        secret_value,
        agent_id="alice",
        scope="agent-private",
        caller_id="agent:alice",
    )

    app = mcp_mod.create_app(app_data_dir=str(tmp_path))
    with caplog.at_level(logging.INFO, logger="gosh.secrets.audit"):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            registered = await _register_agent_public_key(
                client,
                token=alice_token,
                public_key=public_key_b64,
            )
            response = await _resolve_agent_secrets(
                client,
                token=alice_token,
                body={
                    "key": "agent-secret-own",
                    "refs": [{"name": "anthropic", "scope": "agent-private"}],
                },
            )

    assert stored["stored"] is True
    assert registered.status_code == 200
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    payload = response.json()
    assert "value" not in response.text
    assert "value_encoding" not in response.text
    assert secret_value not in response.text
    assert secret_value not in caplog.text
    assert payload["secrets"][0]["algorithm"] == "x25519-hkdf-sha256-aes256gcm-v1"
    assert payload["secrets"][0]["key_id"] == _fingerprint(public_key_b64)
    assert _decrypt_ciphertext(private_key, payload["secrets"][0]["ciphertext"]) == secret_value
    with pytest.raises(Exception):
        _decrypt_ciphertext(wrong_private_key, payload["secrets"][0]["ciphertext"])


@pytest.mark.asyncio
async def test_agent_selector_does_not_authenticate_caller_for_secret_delivery(tmp_path, _reset_state):
    harness = _reset_state
    harness.issue("agent:alice", kind="agent")
    bob_token = harness.issue("agent:bob", kind="agent")
    _private_key, public_key_b64 = _generate_keypair()

    server = mcp_mod._get_memory("agent-secret-deny")
    server.store_secret(
        "anthropic",
        "never-leak",
        agent_id="alice",
        scope="agent-private",
        caller_id="agent:alice",
    )

    app = mcp_mod.create_app(app_data_dir=str(tmp_path))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        registered = await _register_agent_public_key(
            client,
            token=bob_token,
            public_key=public_key_b64,
        )
        response = await _resolve_agent_secrets(
            client,
            token=bob_token,
            body={
                "key": "agent-secret-deny",
                "refs": [{"name": "anthropic", "scope": "agent-private", "agent_id": "alice"}],
            },
        )

    assert registered.status_code == 200
    assert response.status_code == 403
    assert response.json()["code"] == "SECRET_FORBIDDEN"
    assert "never-leak" not in response.text


@pytest.mark.asyncio
async def test_swarm_member_can_resolve_until_membership_is_revoked(tmp_path, _reset_state):
    harness = _reset_state
    alice_token = harness.issue("agent:alice", kind="agent")
    bob_token = harness.issue("agent:bob", kind="agent")
    bob_private, bob_public = _generate_keypair()
    harness.create_swarm("alpha", "agent:alice")
    harness.grant(harness.admin_token, swarm_id="alpha", principal_id="agent:bob", role="member")

    server = mcp_mod._get_memory("agent-secret-swarm")
    secret_value = "shared-cluster-secret"
    server.store_secret(
        "groq",
        secret_value,
        agent_id="alice",
        swarm_id="alpha",
        scope="swarm-shared",
        caller_id="agent:alice",
    )

    app = mcp_mod.create_app(app_data_dir=str(tmp_path))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        registered = await _register_agent_public_key(
            client,
            token=bob_token,
            public_key=bob_public,
        )
        before = await _resolve_agent_secrets(
            client,
            token=bob_token,
            body={
                "key": "agent-secret-swarm",
                "refs": [{"name": "groq", "scope": "swarm-shared", "swarm_id": "alpha"}],
            },
        )
        revoked = await mcp_mod.membership_revoke(
            swarm_id="alpha",
            principal_id="agent:bob",
            token=harness.admin_token,
        )
        after = await _resolve_agent_secrets(
            client,
            token=bob_token,
            body={
                "key": "agent-secret-swarm",
                "refs": [{"name": "groq", "scope": "swarm-shared", "swarm_id": "alpha"}],
            },
        )

    assert registered.status_code == 200
    assert before.status_code == 200
    assert _decrypt_ciphertext(bob_private, before.json()["secrets"][0]["ciphertext"]) == secret_value
    assert revoked["status"] == "ok"
    assert after.status_code == 403
    assert after.json()["code"] == "SECRET_FORBIDDEN"


@pytest.mark.asyncio
async def test_bootstrap_and_persisted_bootstrap_tokens_cannot_call_agent_secret_resolve(tmp_path, _reset_state):
    harness = _reset_state
    harness.issue("agent:alice", kind="agent")
    storage = SQLiteAuthorityStorage(str(tmp_path))
    legacy_token = "gm_bootstrap_legacy_agent_secret"
    storage.token_insert(
        token_id="tok_legacy_agent_secret",
        principal_id="agent:alice",
        token_hash=hashlib.sha256(legacy_token.encode("utf-8")).digest(),
        token_kind="bootstrap",
        description="legacy bootstrap token",
        issued_at=datetime.now(timezone.utc).isoformat(),
        issued_by="system",
        expires_at=None,
        metadata={},
    )

    app = mcp_mod.create_app(app_data_dir=str(tmp_path))
    body = {"key": "default", "refs": [{"name": "anthropic", "scope": "system-wide"}]}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        env_bootstrap = await _resolve_agent_secrets(client, token=harness.bootstrap_token, body=body)
        persisted_bootstrap = await _resolve_agent_secrets(client, token=legacy_token, body=body)

    assert env_bootstrap.status_code == 403
    assert env_bootstrap.json()["code"] == "FORBIDDEN"
    assert persisted_bootstrap.status_code == 403
    assert persisted_bootstrap.json()["code"] == "FORBIDDEN"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("setup_mode", "expected_code"),
    [
        ("revoked", "AUTH_REVOKED"),
        ("expired", "AUTH_EXPIRED"),
        ("disabled", "AUTH_DISABLED"),
    ],
)
async def test_invalidated_tokens_fail_closed_for_agent_secret_resolve(
    tmp_path,
    _reset_state,
    setup_mode,
    expected_code,
):
    harness = _reset_state
    principal_id = "agent:alice"
    agent_token = harness.issue(principal_id, kind="agent")
    _private_key, public_key_b64 = _generate_keypair()

    server = mcp_mod._get_memory(f"agent-secret-{setup_mode}")
    server.store_secret(
        "anthropic",
        "never-leak-this-value",
        agent_id="alice",
        scope="agent-private",
        caller_id=principal_id,
    )

    app = mcp_mod.create_app(app_data_dir=str(tmp_path))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        registered = await _register_agent_public_key(
            client,
            token=agent_token,
            public_key=public_key_b64,
        )

    assert registered.status_code == 200

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

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await _resolve_agent_secrets(
            client,
            token=agent_token,
            body={
                "key": f"agent-secret-{setup_mode}",
                "refs": [{"name": "anthropic", "scope": "agent-private"}],
            },
        )

    assert response.status_code == 401
    assert response.json()["code"] == expected_code
    assert "never-leak-this-value" not in response.text


@pytest.mark.asyncio
async def test_exact_canonical_lookup_is_preserved_for_agent_secret_resolve(tmp_path, _reset_state):
    harness = _reset_state
    bob_token = harness.issue("agent:bob", kind="agent")
    _private_key, public_key_b64 = _generate_keypair()
    harness.issue("agent:alice", kind="agent")
    harness.create_swarm("alpha", "agent:alice")
    harness.grant(harness.admin_token, swarm_id="alpha", principal_id="agent:bob", role="member")

    server = mcp_mod._get_memory("agent-secret-exact")
    server.store_secret(
        "groq",
        "alpha-secret",
        agent_id="alice",
        swarm_id="alpha",
        scope="swarm-shared",
        caller_id="agent:alice",
    )

    app = mcp_mod.create_app(app_data_dir=str(tmp_path))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        registered = await _register_agent_public_key(
            client,
            token=bob_token,
            public_key=public_key_b64,
        )
        wrong_scope = await _resolve_agent_secrets(
            client,
            token=bob_token,
            body={"key": "agent-secret-exact", "refs": [{"name": "groq", "scope": "system-wide"}]},
        )
        missing_swarm = await _resolve_agent_secrets(
            client,
            token=bob_token,
            body={"key": "agent-secret-exact", "refs": [{"name": "groq", "scope": "swarm-shared"}]},
        )

    assert registered.status_code == 200
    assert wrong_scope.status_code == 404
    assert wrong_scope.json()["code"] == "SECRET_NOT_FOUND"
    assert missing_swarm.status_code == 400
    assert missing_swarm.json()["code"] == "VALIDATION_ERROR"


@pytest.mark.asyncio
async def test_agent_public_key_register_endpoint_and_invalid_material_fail_closed(tmp_path, _reset_state):
    harness = _reset_state
    agent_token = harness.issue("agent:alice", kind="agent")
    user_token = harness.issue("user:mitja", kind="user")
    server = mcp_mod._get_memory("agent-secret-public-key")
    server.store_secret(
        "anthropic",
        "never-leak",
        agent_id="alice",
        scope="agent-private",
        caller_id="agent:alice",
    )

    app = mcp_mod.create_app(app_data_dir=str(tmp_path))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        missing = await _resolve_agent_secrets(
            client,
            token=agent_token,
            body={"key": "agent-secret-public-key", "refs": [{"name": "anthropic", "scope": "agent-private"}]},
        )
        invalid_registration = await _register_agent_public_key(
            client,
            token=agent_token,
            public_key="not-valid-base64",
        )
        self_register = await _register_agent_public_key(
            client,
            token=agent_token,
            public_key=_generate_keypair()[1],
        )
        admin_override_key = _generate_keypair()[1]
        admin_register = await _register_agent_public_key(
            client,
            token=harness.admin_token,
            principal_id="agent:alice",
            public_key=admin_override_key,
        )
        forbidden_override = await _register_agent_public_key(
            client,
            token=agent_token,
            principal_id="agent:bob",
            public_key=_generate_keypair()[1],
        )

    assert missing.status_code == 409
    assert missing.json()["code"] == "MISSING_AGENT_PUBLIC_KEY"
    assert invalid_registration.status_code == 400
    assert invalid_registration.json()["code"] == "VALIDATION_ERROR"
    assert self_register.status_code == 200
    assert self_register.json()["principal_id"] == "agent:alice"
    assert admin_register.status_code == 200
    assert admin_register.json()["principal_id"] == "agent:alice"
    assert admin_register.json()["key_id"] == _fingerprint(admin_override_key)
    assert forbidden_override.status_code == 403
    assert forbidden_override.json()["code"] == "FORBIDDEN"

    authority = mcp_mod._get_authority()
    principal = authority.get_principal(
        actor=authority.resolve_token(harness.admin_token),
        principal_id="agent:alice",
    )
    principal_metadata = dict(principal.get("metadata") or {})
    principal_metadata["agent_public_key"] = {
        "algorithm": "x25519",
        "public_key": "not-valid-base64",
        "key_id": "broken",
    }
    authority._storage.principal_set_metadata("agent:alice", principal_metadata)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        invalid = await _resolve_agent_secrets(
            client,
            token=agent_token,
            body={"key": "agent-secret-public-key", "refs": [{"name": "anthropic", "scope": "agent-private"}]},
        )
        non_agent = await _resolve_agent_secrets(
            client,
            token=user_token,
            body={"key": "agent-secret-public-key", "refs": [{"name": "anthropic", "scope": "agent-private"}]},
        )

    assert invalid.status_code == 500
    assert invalid.json()["code"] == "INVALID_AGENT_PUBLIC_KEY"
    assert non_agent.status_code == 403
    assert non_agent.json()["code"] == "AGENT_PRINCIPAL_REQUIRED"
