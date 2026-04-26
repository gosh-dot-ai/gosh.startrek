# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

import json

import numpy as np
import pytest

from src.memory import MemoryServer

DIM = 3072


@pytest.fixture(autouse=True)
def _patch_embed(monkeypatch):
    async def _aembed_texts(texts, **kw):
        return np.random.randn(len(texts), DIM).astype(np.float32)

    async def _aembed_query(text, **kw):
        return np.random.randn(DIM).astype(np.float32)

    monkeypatch.setattr("src.memory.embed_texts", _aembed_texts)
    monkeypatch.setattr("src.memory.embed_query", _aembed_query)
    monkeypatch.setattr("src.memory.resolve_supersession", lambda f, l: None)


@pytest.fixture(autouse=True)
def _allow_plaintext_secrets(monkeypatch):
    monkeypatch.setenv("GOSH_MEMORY_ALLOW_PLAINTEXT_SECRETS", "1")


def _secret_row(
    server: MemoryServer,
    *,
    name: str,
    scope: str,
    owner_id: str,
    swarm_id: str | None = None,
) -> dict:
    storage = server._secret_storage()
    assert storage is not None
    for row in storage.list_secret_rows(include_values=True):
        if str(row.get("name") or "") != name:
            continue
        if str(row.get("scope") or "") != scope:
            continue
        if str(row.get("owner_id") or "") != owner_id:
            continue
        if str(row.get("swarm_id") or "") != str(swarm_id or ""):
            continue
        return row
    raise AssertionError(f"secret row not found: name={name!r} scope={scope!r} owner_id={owner_id!r}")


def test_secret_create_returns_metadata_only_and_persists_creator(tmp_path):
    server = MemoryServer(data_dir=str(tmp_path), key="sec_create")
    result = server.store_secret(
        "API_KEY",
        "sk-123",
        agent_id="alice",
        scope="agent-private",
        caller_id="agent:alice",
    )
    row = _secret_row(server, name="API_KEY", scope="agent-private", owner_id="agent:alice")

    assert result["stored"] is True
    assert "value" not in result
    assert "sk-123" not in json.dumps(result)
    assert row["value"] == "sk-123"
    assert row["created_by_principal_id"] == "agent:alice"
    assert not hasattr(server, "get_secret")
    assert not hasattr(server, "rotate_secret")


def test_secret_create_is_create_once_and_duplicate_preserves_original_row(tmp_path):
    server = MemoryServer(data_dir=str(tmp_path), key="sec_duplicate")
    first = server.store_secret(
        "KEY",
        "v1",
        agent_id="alice",
        scope="agent-private",
        caller_id="agent:alice",
        metadata={"version": 1},
    )
    duplicate = server.store_secret(
        "KEY",
        "v2",
        agent_id="alice",
        scope="agent-private",
        caller_id="agent:alice",
        metadata={"version": 2},
    )
    row = _secret_row(server, name="KEY", scope="agent-private", owner_id="agent:alice")

    assert first["stored"] is True
    assert duplicate == {
        "stored": False,
        "code": "SECRET_ALREADY_EXISTS",
        "secret_id": first["secret_id"],
        "acl_domain_key": first["acl_domain_key"],
    }
    assert "v2" not in json.dumps(duplicate)
    assert row["value"] == "v1"
    assert row["created_by_principal_id"] == "agent:alice"
    assert row["metadata"] == {"version": 1}


def test_agent_private_secret_write_can_delegate_to_other_agent_domain(tmp_path):
    server = MemoryServer(data_dir=str(tmp_path), key="sec_spoof")
    first = server.store_secret(
        "PETYA_KEY",
        "v1",
        agent_id="petya",
        scope="agent-private",
        caller_id="agent:petya",
    )
    delegated = server.store_secret(
        "PETYA_KEY",
        "evil",
        agent_id="petya",
        scope="agent-private",
        caller_id="agent:alice",
    )
    row = _secret_row(server, name="PETYA_KEY", scope="agent-private", owner_id="agent:petya")

    assert first["stored"] is True
    assert delegated["stored"] is False
    assert delegated["code"] == "SECRET_ALREADY_EXISTS"
    assert row["value"] == "v1"
    assert row["created_by_principal_id"] == "agent:petya"


def test_creator_only_delete_overrides_acl_write_and_admin(tmp_path):
    server = MemoryServer(data_dir=str(tmp_path), key="sec_delete")
    stored = server.store_secret(
        "DB_PASS",
        "hunter2",
        agent_id="alice",
        swarm_id="sw1",
        scope="swarm-shared",
        caller_id="agent:alice",
    )
    member_delete = server.delete_secret(
        "DB_PASS",
        swarm_id="sw1",
        scope="swarm-shared",
        caller_id="agent:bob",
        caller_memberships=["swarm:sw1"],
    )
    admin_delete = server.delete_secret(
        "DB_PASS",
        swarm_id="sw1",
        scope="swarm-shared",
        caller_id="service:root",
        caller_role="admin",
    )
    creator_delete = server.delete_secret(
        "DB_PASS",
        swarm_id="sw1",
        scope="swarm-shared",
        caller_id="agent:alice",
        caller_memberships=["swarm:sw1"],
    )
    after = server.list_secrets(
        swarm_id="sw1",
        scope="swarm-shared",
        caller_id="agent:alice",
        caller_memberships=["swarm:sw1"],
    )

    assert stored["stored"] is True
    assert member_delete["code"] == "SECRET_FORBIDDEN"
    assert admin_delete["code"] == "SECRET_FORBIDDEN"
    assert creator_delete["deleted"] is True
    assert after == {"secrets": []}


def test_creator_only_delete_does_not_require_current_swarm_membership(tmp_path):
    server = MemoryServer(data_dir=str(tmp_path), key="sec_delete_revoked")
    stored = server.store_secret(
        "DB_PASS",
        "hunter2",
        agent_id="alice",
        swarm_id="sw1",
        scope="swarm-shared",
        caller_id="agent:alice",
    )
    creator_delete = server.delete_secret(
        "DB_PASS",
        swarm_id="sw1",
        scope="swarm-shared",
        caller_id="agent:alice",
        caller_memberships=[],
    )

    assert stored["stored"] is True
    assert creator_delete["deleted"] is True


def test_delete_missing_secret_returns_not_found(tmp_path):
    server = MemoryServer(data_dir=str(tmp_path), key="sec_missing_delete")
    result = server.delete_secret(
        "MISSING",
        scope="system-wide",
        caller_id="agent:alice",
    )
    assert result == {"deleted": False, "code": "SECRET_NOT_FOUND"}


def test_system_wide_secret_uses_public_canonical_acl(tmp_path):
    server = MemoryServer(data_dir=str(tmp_path), key="sec_system_private")
    stored = server.store_secret(
        "GLOBAL_KEY",
        "value",
        agent_id="alice",
        scope="system-wide",
        caller_id="agent:alice",
    )
    owner_list = server.list_secrets(
        scope="system-wide",
        caller_id="agent:alice",
    )
    outsider_list = server.list_secrets(
        scope="system-wide",
        caller_id="agent:bob",
    )

    assert stored["stored"] is True
    assert [secret["name"] for secret in owner_list["secrets"]] == ["GLOBAL_KEY"]
    assert [secret["name"] for secret in outsider_list["secrets"]] == ["GLOBAL_KEY"]
    assert outsider_list["secrets"][0]["owner_id"] == "system"
    assert outsider_list["secrets"][0]["read"] == ["agent:PUBLIC"]
    assert outsider_list["secrets"][0]["write"] == ["agent:PUBLIC"]


@pytest.mark.parametrize(
    ("name", "scope", "agent_id", "swarm_id", "expected_owner_id", "expected_read", "expected_write"),
    [
        ("PRIVATE_KEY", "agent-private", "alice", None, "agent:alice", [], []),
        ("SWARM_KEY", "swarm-shared", "alice", "alpha", "agent:alice", ["swarm:alpha"], ["swarm:alpha"]),
        ("SYSTEM_KEY", "system-wide", "alice", None, "system", ["agent:PUBLIC"], ["agent:PUBLIC"]),
    ],
)
def test_secret_scope_shorthand_expands_to_canonical_acl_fields(
    tmp_path,
    name,
    scope,
    agent_id,
    swarm_id,
    expected_owner_id,
    expected_read,
    expected_write,
):
    server = MemoryServer(data_dir=str(tmp_path), key=f"sec_acl_{name.lower()}")
    stored = server.store_secret(
        name,
        f"value-for-{name.lower()}",
        agent_id=agent_id,
        swarm_id=swarm_id,
        scope=scope,
        caller_id="agent:alice",
    )
    row = _secret_row(
        server,
        name=name,
        scope=scope,
        owner_id=expected_owner_id,
        swarm_id=swarm_id if scope == "swarm-shared" else None,
    )

    assert stored["stored"] is True
    assert row["owner_id"] == expected_owner_id
    assert row["read"] == expected_read
    assert row["write"] == expected_write
    assert row["created_by_principal_id"] == "agent:alice"


def test_list_secrets_returns_metadata_only_and_creator_id(tmp_path):
    server = MemoryServer(data_dir=str(tmp_path), key="sec_list")
    known_secret = "dont-leak-me"
    server.store_secret(
        "KEY",
        known_secret,
        agent_id="alice",
        scope="agent-private",
        caller_id="agent:alice",
        metadata={"provider": "test"},
    )
    listed = server.list_secrets(
        scope="agent-private",
        caller_id="agent:alice",
    )
    payload = json.dumps(listed)

    assert listed["secrets"] == [
        {
            "secret_id": listed["secrets"][0]["secret_id"],
            "name": "KEY",
            "owner_id": "agent:alice",
            "created_by_principal_id": "agent:alice",
            "scope": "agent-private",
            "agent_id": "alice",
            "swarm_id": None,
            "read": [],
            "write": [],
            "created_at": listed["secrets"][0]["created_at"],
            "updated_at": listed["secrets"][0]["updated_at"],
            "metadata": {"provider": "test"},
        }
    ]
    assert "value" not in listed["secrets"][0]
    assert "preview" not in payload
    assert known_secret not in payload


def test_secrets_persist_roundtrip_with_creator_metadata(tmp_path):
    server = MemoryServer(data_dir=str(tmp_path), key="sec_roundtrip")
    server.store_secret(
        "RELOAD_KEY",
        "xyz",
        agent_id="alice",
        scope="agent-private",
        caller_id="agent:alice",
    )
    server2 = MemoryServer(data_dir=str(tmp_path), key="sec_roundtrip")
    listed = server2.list_secrets(
        scope="agent-private",
        caller_id="agent:alice",
    )

    assert listed["secrets"][0]["name"] == "RELOAD_KEY"
    assert listed["secrets"][0]["created_by_principal_id"] == "agent:alice"


@pytest.mark.asyncio
async def test_secrets_not_in_recall_or_snapshot_surfaces(tmp_path):
    server = MemoryServer(data_dir=str(tmp_path), key="sec_recall")
    server.store_secret(
        "SECRET_KEY",
        "leak-me",
        agent_id="a",
        scope="agent-private",
        caller_id="agent:a",
    )
    server._all_granular = [
        {
            "fact": "User lives in Seattle.",
            "kind": "fact",
            "id": "af_001",
            "conv_id": "sec_recall",
            "agent_id": "a",
            "swarm_id": "sw",
            "scope": "swarm-shared",
            "created_at": "2024-01-01T00:00:00+00:00",
            "owner_id": "agent:a",
            "read": [],
            "write": [],
        }
    ]
    server._all_cons = []
    server._all_cross = []
    await server.build_index()
    result = await server.recall("SECRET_KEY API key credentials")
    payload = server._storage.load_facts(internal=True)

    assert "leak-me" not in result["context"]
    assert "secrets" not in payload


def test_stats_includes_secrets_count(tmp_path):
    server = MemoryServer(data_dir=str(tmp_path), key="sec_stats")
    server.store_secret("K1", "v1", agent_id="a", scope="agent-private", caller_id="agent:a")
    server.store_secret("K2", "v2", agent_id="a", scope="agent-private", caller_id="agent:a")
    stats = server.stats()
    assert stats["secrets"] == 2


def test_old_cache_without_secrets_loads_cleanly(tmp_path):
    cache = {
        "granular": [],
        "cons": [],
        "cross": [],
        "tlinks": [],
        "raw_sessions": [],
        "n_sessions": 0,
        "n_sessions_with_facts": 0,
    }
    (tmp_path / "compat.json").write_text(json.dumps(cache))
    server = MemoryServer(data_dir=str(tmp_path), key="compat")
    assert server._secrets == []


def test_secret_store_requires_explicit_scope_and_auth(tmp_path):
    server = MemoryServer(data_dir=str(tmp_path), key="sec_scope")
    assert server.store_secret("KEY", "val", caller_id="agent:a")["code"] == "VALIDATION_ERROR"
    assert server.list_secrets(scope="system-wide")["code"] == "AUTH_REQUIRED"


def test_secret_plaintext_storage_fails_closed_without_override(tmp_path, monkeypatch):
    monkeypatch.delenv("GOSH_MEMORY_ALLOW_PLAINTEXT_SECRETS", raising=False)
    server = MemoryServer(data_dir=str(tmp_path), key="sec_plaintext_forbidden")
    result = server.store_secret(
        "API_KEY",
        "sk-123",
        agent_id="alice",
        scope="agent-private",
        caller_id="agent:alice",
    )
    assert result["code"] == "SECRET_STORAGE_UNAVAILABLE"
