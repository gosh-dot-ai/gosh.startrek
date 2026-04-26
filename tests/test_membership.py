# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

import hashlib
import time
from datetime import datetime, timedelta, timezone

import pytest

from src.membership import AuthorityError, AuthorityService
from src.storage import SQLiteAuthorityStorage


def _service(tmp_path) -> AuthorityService:
    return AuthorityService(SQLiteAuthorityStorage(str(tmp_path)))


def test_bootstrap_admin_and_resolve_token(tmp_path):
    service = _service(tmp_path)
    issued = service.bootstrap_admin(principal_id="service:admin", kind="service")
    resolved = service.resolve_token(issued["token"])
    assert resolved.principal_id == "service:admin"
    assert resolved.caller_role == "admin"


def test_bootstrap_admin_is_one_time_and_persists_state(tmp_path):
    storage = SQLiteAuthorityStorage(str(tmp_path))
    service = AuthorityService(storage)

    first = service.bootstrap_admin(principal_id="service:first-admin", kind="service")
    state = storage.bootstrap_state_get()

    assert first["token_kind"] == "admin"
    assert state is not None
    assert state["bootstrapped_principal_id"] == "service:first-admin"

    with pytest.raises(AuthorityError) as same_exc:
        service.bootstrap_admin(principal_id="service:first-admin", kind="service")
    with pytest.raises(AuthorityError) as other_exc:
        service.bootstrap_admin(principal_id="service:other-admin", kind="service")

    assert same_exc.value.code == "BOOTSTRAP_ALREADY_USED"
    assert other_exc.value.code == "BOOTSTRAP_ALREADY_USED"

    reopened = _service(tmp_path)
    with pytest.raises(AuthorityError) as reopen_exc:
        reopened.bootstrap_admin(principal_id="service:reopen-admin", kind="service")
    assert reopen_exc.value.code == "BOOTSTRAP_ALREADY_USED"


@pytest.mark.parametrize(
    ("principal_status", "expires_at", "revoked_at"),
    [
        ("active", None, None),
        ("active", None, "2026-04-08T00:00:00+00:00"),
        ("active", "2000-01-01T00:00:00+00:00", None),
        ("disabled", None, None),
    ],
)
def test_legacy_admin_history_seals_bootstrap_on_upgrade(tmp_path, principal_status, expires_at, revoked_at):
    storage = SQLiteAuthorityStorage(str(tmp_path))
    service = AuthorityService(storage)
    issued_at = datetime.now(timezone.utc).isoformat()
    storage.principal_upsert(
        principal_id="service:existing-admin",
        kind="service",
        display_name="Existing Admin",
        status=principal_status,
        created_at=issued_at,
        created_by="system",
        metadata={},
    )
    storage.token_insert(
        token_id="tok_existing_admin",
        principal_id="service:existing-admin",
        token_hash=hashlib.sha256(b"gm_admin_existing").digest(),
        token_kind="admin",
        description="legacy admin token",
        issued_at=issued_at,
        issued_by="system",
        expires_at=expires_at,
        metadata={},
    )
    if revoked_at is not None:
        storage.token_revoke(
            "tok_existing_admin",
            revoked_at=revoked_at,
            revoked_by="system",
        )

    assert storage.bootstrap_state_get() is None
    assert storage.admin_bootstrap_history_exists() is True

    with pytest.raises(AuthorityError) as exc:
        service.bootstrap_admin(principal_id="service:new-admin", kind="service")

    assert exc.value.code == "BOOTSTRAP_ALREADY_USED"
    state = storage.bootstrap_state_get()
    assert state is not None
    assert state["bootstrapped_principal_id"] == "service:existing-admin"
    assert state["bootstrapped_token_id"] == "tok_existing_admin"
    assert state["bootstrapped_at"] == issued_at


def test_swarm_membership_persists_across_reopen(tmp_path):
    service = _service(tmp_path)
    issued = service.bootstrap_admin(principal_id="service:admin", kind="service")
    admin = service.resolve_token(issued["token"])
    service.create_principal(actor=admin, principal_id="agent:alice", kind="agent")
    service.create_principal(actor=admin, principal_id="agent:bob", kind="agent")
    service.create_swarm(actor=admin, swarm_id="alpha", owner_principal_id="agent:alice")
    service.grant_membership(actor=admin, swarm_id="alpha", principal_id="agent:bob", role="member")

    reopened = _service(tmp_path)
    bob_token = reopened.issue_token(actor=admin, principal_id="agent:bob", token_kind="agent")
    bob = reopened.resolve_token(bob_token["token"])
    assert "swarm:alpha" in bob.memberships


def test_owner_and_manager_membership_rules(tmp_path):
    service = _service(tmp_path)
    issued = service.bootstrap_admin(principal_id="service:admin", kind="service")
    admin = service.resolve_token(issued["token"])
    for principal in ("agent:owner", "agent:manager", "agent:member", "agent:other"):
        service.create_principal(actor=admin, principal_id=principal, kind="agent")
    service.create_swarm(actor=admin, swarm_id="alpha", owner_principal_id="agent:owner")
    owner_token = service.issue_token(actor=admin, principal_id="agent:owner", token_kind="agent")
    owner = service.resolve_token(owner_token["token"])
    service.grant_membership(actor=owner, swarm_id="alpha", principal_id="agent:manager", role="manager")
    service.grant_membership(actor=owner, swarm_id="alpha", principal_id="agent:member", role="member")

    manager_token = service.issue_token(actor=admin, principal_id="agent:manager", token_kind="agent")
    manager = service.resolve_token(manager_token["token"])
    service.grant_membership(actor=manager, swarm_id="alpha", principal_id="agent:other", role="member")

    member_token = service.issue_token(actor=admin, principal_id="agent:member", token_kind="agent")
    member = service.resolve_token(member_token["token"])
    with pytest.raises(AuthorityError) as exc:
        service.grant_membership(actor=member, swarm_id="alpha", principal_id="agent:other", role="member")
    assert exc.value.code == "FORBIDDEN"


def test_revoked_and_expired_tokens_fail_closed(tmp_path):
    service = _service(tmp_path)
    issued = service.bootstrap_admin(principal_id="service:admin", kind="service")
    admin = service.resolve_token(issued["token"])
    service.create_principal(actor=admin, principal_id="agent:alice", kind="agent")

    revoked = service.issue_token(actor=admin, principal_id="agent:alice", token_kind="agent")
    service.revoke_token(actor=admin, token_id=revoked["token_id"])
    with pytest.raises(AuthorityError) as rev_exc:
        service.resolve_token(revoked["token"])
    assert rev_exc.value.code == "AUTH_REVOKED"

    expired = service.issue_token(
        actor=admin,
        principal_id="agent:alice",
        token_kind="agent",
        expires_at="2000-01-01T00:00:00+00:00",
    )
    with pytest.raises(AuthorityError) as exp_exc:
        service.resolve_token(expired["token"])
    assert exp_exc.value.code == "AUTH_EXPIRED"


def test_expired_membership_can_be_regranted(tmp_path):
    service = _service(tmp_path)
    issued = service.bootstrap_admin(principal_id="service:admin", kind="service")
    admin = service.resolve_token(issued["token"])
    service.create_principal(actor=admin, principal_id="agent:alice", kind="agent")
    service.create_principal(actor=admin, principal_id="agent:bob", kind="agent")
    service.create_swarm(actor=admin, swarm_id="alpha", owner_principal_id="agent:alice")

    expires_at = (datetime.now(timezone.utc) + timedelta(milliseconds=50)).isoformat()
    first = service.grant_membership(
        actor=admin,
        swarm_id="alpha",
        principal_id="agent:bob",
        role="member",
        expires_at=expires_at,
    )
    time.sleep(0.1)
    second = service.grant_membership(
        actor=admin,
        swarm_id="alpha",
        principal_id="agent:bob",
        role="member",
    )

    assert first["membership_id"] != second["membership_id"]
    assert "swarm:alpha" in service.memberships_for("agent:bob")


def test_persisted_admin_cannot_issue_or_use_bootstrap_tokens(tmp_path):
    storage = SQLiteAuthorityStorage(str(tmp_path))
    service = AuthorityService(storage)
    issued = service.bootstrap_admin(principal_id="service:admin", kind="service")
    admin = service.resolve_token(issued["token"])
    service.create_principal(actor=admin, principal_id="agent:alice", kind="agent")

    with pytest.raises(AuthorityError) as issue_exc:
        service.issue_token(actor=admin, principal_id="agent:alice", token_kind="bootstrap")
    assert issue_exc.value.code == "FORBIDDEN"

    legacy_token = "gm_bootstrap_legacy_token"
    storage.token_insert(
        token_id="tok_legacy_bootstrap",
        principal_id="agent:alice",
        token_hash=hashlib.sha256(legacy_token.encode("utf-8")).digest(),
        token_kind="bootstrap",
        description="legacy bootstrap token",
        issued_at=datetime.now(timezone.utc).isoformat(),
        issued_by="system",
        expires_at=None,
        metadata={},
    )
    with pytest.raises(AuthorityError) as resolve_exc:
        service.resolve_token(legacy_token)
    assert resolve_exc.value.code == "FORBIDDEN"
