#!/usr/bin/env python3
# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from .storage import SQLiteAuthorityStorage

_PRINCIPAL_KINDS = {"user", "agent", "service"}
_TOKEN_KINDS = {"bootstrap", "admin", "user", "agent", "join"}
_PRINCIPAL_STATUSES = {"active", "disabled"}
_SWARM_STATUSES = {"active", "disabled"}
_MEMBERSHIP_ROLES = {"owner", "manager", "member"}
_MEMBERSHIP_STATUSES = {"active", "revoked"}
_PRINCIPAL_ID_RE = re.compile(r"^(user|agent|service):[A-Za-z0-9._:@/-]+$")
_SWARM_ID_RE = re.compile(r"^[A-Za-z0-9._:-]+$")
_AGENT_PUBLIC_KEY_METADATA_FIELD = "agent_public_key"
_AGENT_PUBLIC_KEY_ALGORITHM = "x25519"


class AuthorityError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass
class ResolvedPrincipal:
    principal_id: str
    principal_kind: str
    token_id: str
    token_kind: str
    memberships: list[str]
    caller_role: str
    display_name: str | None = None


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_optional_datetime(value: str | None) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None


def _token_hash(token: str) -> bytes:
    return hashlib.sha256(token.encode("utf-8")).digest()


def _validate_principal_id(principal_id: str, kind: str | None = None) -> str:
    principal_id = str(principal_id or "").strip()
    if not _PRINCIPAL_ID_RE.match(principal_id):
        raise AuthorityError(
            "VALIDATION_ERROR",
            "principal_id must be canonical user:/agent:/service: identifier",
        )
    if kind is not None and not principal_id.startswith(f"{kind}:"):
        raise AuthorityError(
            "VALIDATION_ERROR",
            f"principal_id must use {kind}: prefix",
        )
    return principal_id


def _validate_swarm_id(swarm_id: str) -> str:
    swarm_id = str(swarm_id or "").strip()
    if not swarm_id or not _SWARM_ID_RE.match(swarm_id):
        raise AuthorityError(
            "VALIDATION_ERROR",
            "swarm_id must be non-empty and contain only letters, digits, ., _, :, -",
        )
    return swarm_id


def _normalize_agent_public_key_binding(
    *,
    public_key: str,
    algorithm: str = _AGENT_PUBLIC_KEY_ALGORITHM,
    key_id: str | None = None,
) -> dict[str, str]:
    algorithm_text = str(algorithm or _AGENT_PUBLIC_KEY_ALGORITHM).strip().lower()
    if algorithm_text != _AGENT_PUBLIC_KEY_ALGORITHM:
        raise AuthorityError(
            "VALIDATION_ERROR",
            f"agent public key algorithm must be {_AGENT_PUBLIC_KEY_ALGORITHM}",
        )
    public_key_text = str(public_key or "").strip()
    if not public_key_text:
        raise AuthorityError(
            "VALIDATION_ERROR",
            "public_key must be a base64-encoded raw x25519 public key",
        )
    try:
        public_key_bytes = base64.b64decode(public_key_text.encode("ascii"), validate=True)
    except (ValueError, binascii.Error) as exc:
        raise AuthorityError(
            "VALIDATION_ERROR",
            "public_key must be a base64-encoded raw x25519 public key",
        ) from exc
    if len(public_key_bytes) != 32:
        raise AuthorityError(
            "VALIDATION_ERROR",
            "public_key must be a 32-byte raw x25519 public key",
        )
    normalized_key_id = str(key_id or "").strip() or f"sha256:{hashlib.sha256(public_key_bytes).hexdigest()}"
    return {
        "algorithm": algorithm_text,
        "public_key": public_key_text,
        "key_id": normalized_key_id,
    }


class AuthorityService:
    """Small explicit built-in auth + swarm authority layer."""

    def __init__(self, storage: SQLiteAuthorityStorage):
        self._storage = storage

    @property
    def path(self):
        return self._storage.path

    @staticmethod
    def _validate_principal_kind(kind: str) -> str:
        kind = str(kind or "").strip()
        if kind not in _PRINCIPAL_KINDS:
            raise AuthorityError("VALIDATION_ERROR", f"principal kind must be one of {sorted(_PRINCIPAL_KINDS)}")
        return kind

    @staticmethod
    def _validate_token_kind(token_kind: str) -> str:
        token_kind = str(token_kind or "").strip()
        if token_kind not in _TOKEN_KINDS:
            raise AuthorityError("VALIDATION_ERROR", f"token_kind must be one of {sorted(_TOKEN_KINDS)}")
        return token_kind

    @staticmethod
    def _validate_membership_role(role: str) -> str:
        role = str(role or "").strip()
        if role not in _MEMBERSHIP_ROLES:
            raise AuthorityError("VALIDATION_ERROR", f"role must be one of {sorted(_MEMBERSHIP_ROLES)}")
        return role

    @staticmethod
    def _ensure_future_or_none(expires_at: str | None) -> str | None:
        if expires_at in (None, ""):
            return None
        dt = _parse_optional_datetime(expires_at)
        if dt is None:
            raise AuthorityError("VALIDATION_ERROR", "expires_at must be an ISO datetime")
        return dt.astimezone(timezone.utc).isoformat()

    def _require_admin(self, actor: ResolvedPrincipal | None) -> None:
        if actor is None or actor.caller_role != "admin":
            raise AuthorityError("FORBIDDEN", "admin principal required")

    def _principal_or_error(self, principal_id: str) -> dict:
        principal = self._storage.principal_get(principal_id)
        if principal is None:
            raise AuthorityError("NOT_FOUND", f"principal {principal_id} not found")
        return principal

    def _swarm_or_error(self, swarm_id: str) -> dict:
        swarm = self._storage.swarm_get(swarm_id)
        if swarm is None:
            raise AuthorityError("NOT_FOUND", f"swarm {swarm_id} not found")
        return swarm

    def _membership_active(self, *, swarm_id: str, principal_id: str) -> dict | None:
        membership = self._storage.membership_active_row(swarm_id=swarm_id, principal_id=principal_id)
        if membership is None:
            return None
        expires_at = _parse_optional_datetime(membership.get("expires_at"))
        if expires_at is not None and expires_at <= datetime.now(timezone.utc):
            self._storage.membership_revoke(
                membership_id=str(membership["membership_id"]),
                revoked_at=_utcnow_iso(),
                revoked_by="system:expiry",
            )
            return None
        swarm = self._storage.swarm_get(swarm_id)
        if not swarm or swarm.get("status") != "active":
            return None
        return membership

    def _principal_membership_role(self, *, swarm_id: str, principal_id: str) -> str | None:
        membership = self._membership_active(swarm_id=swarm_id, principal_id=principal_id)
        if membership is None:
            return None
        return str(membership.get("role") or "")

    def memberships_for(self, principal_id: str) -> list[str]:
        groups: list[str] = []
        for membership in self._storage.membership_list(principal_id=principal_id, include_revoked=False):
            swarm_id = str(membership.get("swarm_id") or "")
            if not swarm_id:
                continue
            if self._membership_active(swarm_id=swarm_id, principal_id=principal_id) is None:
                continue
            groups.append(f"swarm:{swarm_id}")
        return list(dict.fromkeys(groups))

    def resolve_token(self, token: str) -> ResolvedPrincipal:
        token = str(token or "").strip()
        if not token:
            raise AuthorityError("AUTH_REQUIRED", "principal token required")
        row = self._storage.resolve_token_hash(_token_hash(token))
        if row is None:
            raise AuthorityError("INVALID_TOKEN", "invalid principal token")
        if str(row.get("principal_status") or "") != "active":
            raise AuthorityError("AUTH_DISABLED", "principal is disabled")
        if row.get("revoked_at"):
            raise AuthorityError("AUTH_REVOKED", "principal token is revoked")
        expires_at = _parse_optional_datetime(row.get("expires_at"))
        if expires_at is not None and expires_at <= datetime.now(timezone.utc):
            raise AuthorityError("AUTH_EXPIRED", "principal token is expired")
        principal_id = str(row.get("principal_id") or "")
        token_kind = str(row.get("token_kind") or "")
        if token_kind == "bootstrap":  # noqa: S105 - token kind enum value, not a credential
            raise AuthorityError("FORBIDDEN", "persisted bootstrap tokens are reserved for env bootstrap flow")
        return ResolvedPrincipal(
            principal_id=principal_id,
            principal_kind=str(row.get("principal_kind") or ""),
            token_id=str(row.get("token_id") or ""),
            token_kind=token_kind,
            memberships=self.memberships_for(principal_id),
            caller_role="admin" if token_kind == "admin" else "user",  # noqa: S105 - role derives from token kind enum
            display_name=row.get("display_name"),
        )

    def bootstrap_admin(
        self,
        *,
        principal_id: str,
        kind: str,
        display_name: str | None = None,
        description: str | None = None,
        expires_at: str | None = None,
        metadata: dict | None = None,
    ) -> dict:
        kind = self._validate_principal_kind(kind)
        principal_id = _validate_principal_id(principal_id, kind)
        expires_at = self._ensure_future_or_none(expires_at)
        token_value = f"gm_admin_{secrets.token_urlsafe(32)}"
        token_id = f"tok_{uuid4().hex}"
        issued_at = _utcnow_iso()
        try:
            result = self._storage.bootstrap_admin_once(
                principal_id=principal_id,
                kind=kind,
                display_name=display_name,
                description=description or "bootstrap admin token",
                expires_at=expires_at,
                metadata=metadata or {"bootstrapped": True},
                token_id=token_id,
                token_hash=_token_hash(token_value),
                issued_at=issued_at,
                issued_by="system",
            )
        except RuntimeError as exc:
            state = self._storage.bootstrap_state_get()
            message = "bootstrap already sealed for this authority store"
            if state and state.get("bootstrapped_principal_id"):
                message = (
                    f"bootstrap already sealed for this authority store by "
                    f"{state['bootstrapped_principal_id']}"
                )
            raise AuthorityError("BOOTSTRAP_ALREADY_USED", message) from exc
        except PermissionError as exc:
            raise AuthorityError("FORBIDDEN", str(exc)) from exc
        token_record = result.get("token") or {}
        principal = result.get("principal") or {}
        return {
            "token": token_value,
            "principal_id": principal.get("principal_id", principal_id),
            "token_id": token_record.get("token_id", token_id),
            "token_kind": token_record.get("token_kind", "admin"),
            "description": token_record.get("description"),
            "issued_at": token_record.get("issued_at", issued_at),
            "issued_by": token_record.get("issued_by", "system"),
            "expires_at": token_record.get("expires_at", expires_at),
            "revoked_at": token_record.get("revoked_at"),
            "revoked_by": token_record.get("revoked_by"),
            "last_used_at": token_record.get("last_used_at"),
            "metadata": token_record.get("metadata", metadata or {"bootstrapped": True}),
        }

    def create_principal(
        self,
        *,
        actor: ResolvedPrincipal,
        principal_id: str,
        kind: str,
        display_name: str | None = None,
        metadata: dict | None = None,
    ) -> dict:
        self._require_admin(actor)
        kind = self._validate_principal_kind(kind)
        principal_id = _validate_principal_id(principal_id, kind)
        if self._storage.principal_get(principal_id) is not None:
            raise AuthorityError("ALREADY_EXISTS", f"principal {principal_id} already exists")
        return self._storage.principal_upsert(
            principal_id=principal_id,
            kind=kind,
            display_name=display_name,
            status="active",
            created_at=_utcnow_iso(),
            created_by=actor.principal_id,
            metadata=metadata or {},
        )

    def get_principal(self, *, actor: ResolvedPrincipal, principal_id: str | None = None) -> dict:
        target_principal_id = actor.principal_id if principal_id in (None, "") else _validate_principal_id(str(principal_id))
        if actor.caller_role != "admin" and target_principal_id != actor.principal_id:
            raise AuthorityError("FORBIDDEN", "cannot read another principal")
        return self._principal_or_error(target_principal_id)

    def disable_principal(self, *, actor: ResolvedPrincipal, principal_id: str) -> dict:
        self._require_admin(actor)
        target = self._principal_or_error(_validate_principal_id(principal_id))
        if target.get("status") == "disabled":
            return target
        return self._storage.principal_set_status(principal_id, "disabled") or target

    def get_agent_public_key_binding(self, *, principal_id: str) -> dict | None:
        principal = self._principal_or_error(_validate_principal_id(principal_id, "agent"))
        metadata = principal.get("metadata") if isinstance(principal.get("metadata"), dict) else {}
        binding = metadata.get(_AGENT_PUBLIC_KEY_METADATA_FIELD) if isinstance(metadata, dict) else None
        if not isinstance(binding, dict):
            return None
        public_key = str(binding.get("public_key") or "").strip()
        if not public_key:
            return None
        return {
            "principal_id": principal["principal_id"],
            "algorithm": str(binding.get("algorithm") or "x25519").strip() or "x25519",
            "public_key": public_key,
            "key_id": str(binding.get("key_id") or "").strip() or None,
        }

    def set_agent_public_key_binding(
        self,
        *,
        actor: ResolvedPrincipal,
        principal_id: str,
        public_key: str,
        algorithm: str = "x25519",
        key_id: str | None = None,
    ) -> dict:
        target_principal_id = _validate_principal_id(principal_id, "agent")
        if actor.caller_role != "admin" and target_principal_id != actor.principal_id:
            raise AuthorityError("FORBIDDEN", "cannot set another principal's public key")
        principal = self._principal_or_error(target_principal_id)
        binding = _normalize_agent_public_key_binding(
            public_key=public_key,
            algorithm=algorithm,
            key_id=key_id,
        )
        metadata = dict(principal.get("metadata") or {})
        metadata[_AGENT_PUBLIC_KEY_METADATA_FIELD] = binding
        return self._storage.principal_set_metadata(target_principal_id, metadata) or principal

    def issue_token(
        self,
        *,
        actor: ResolvedPrincipal | None,
        principal_id: str,
        token_kind: str,
        description: str | None = None,
        expires_at: str | None = None,
        metadata: dict | None = None,
        issued_by: str | None = None,
        allow_without_actor: bool = False,
    ) -> dict:
        if not allow_without_actor:
            self._require_admin(actor)
        principal = self._principal_or_error(_validate_principal_id(principal_id))
        if principal.get("status") != "active":
            raise AuthorityError("FORBIDDEN", "cannot issue token for disabled principal")
        token_kind = self._validate_token_kind(token_kind)
        if token_kind == "bootstrap":  # noqa: S105 - token kind enum value, not a credential
            raise AuthorityError("FORBIDDEN", "bootstrap tokens are reserved for env bootstrap flow")
        expires_at = self._ensure_future_or_none(expires_at)
        token_value = f"gm_{token_kind}_{secrets.token_urlsafe(32)}"
        record = self._storage.token_insert(
            token_id=f"tok_{uuid4().hex}",
            principal_id=principal_id,
            token_hash=_token_hash(token_value),
            token_kind=token_kind,
            description=description,
            issued_at=_utcnow_iso(),
            issued_by=issued_by if issued_by is not None else (actor.principal_id if actor is not None else "system"),
            expires_at=expires_at,
            metadata=metadata or {},
        )
        return {"token": token_value, **record}

    def revoke_token(self, *, actor: ResolvedPrincipal, token_id: str) -> dict:
        target = self._storage.token_get(token_id)
        if target is None:
            raise AuthorityError("NOT_FOUND", f"token {token_id} not found")
        if actor.caller_role != "admin" and str(target.get("principal_id") or "") != actor.principal_id:
            raise AuthorityError("FORBIDDEN", "cannot revoke another principal's token")
        return self._storage.token_revoke(
            token_id,
            revoked_at=_utcnow_iso(),
            revoked_by=actor.principal_id,
        ) or target

    def list_tokens(self, *, actor: ResolvedPrincipal, principal_id: str | None = None) -> dict:
        target_principal_id = principal_id
        if actor.caller_role != "admin":
            if principal_id not in (None, "", actor.principal_id):
                raise AuthorityError("FORBIDDEN", "cannot list another principal's tokens")
            target_principal_id = actor.principal_id
        elif target_principal_id not in (None, ""):
            target_principal_id = _validate_principal_id(str(target_principal_id))
        tokens = self._storage.token_list(principal_id=target_principal_id if target_principal_id not in (None, "") else None)
        return {"principal_id": target_principal_id, "tokens": tokens}

    def create_swarm(
        self,
        *,
        actor: ResolvedPrincipal,
        swarm_id: str,
        owner_principal_id: str,
        display_name: str | None = None,
        metadata: dict | None = None,
    ) -> dict:
        self._require_admin(actor)
        swarm_id = _validate_swarm_id(swarm_id)
        owner_principal_id = _validate_principal_id(owner_principal_id)
        if self._storage.swarm_get(swarm_id) is not None:
            raise AuthorityError("ALREADY_EXISTS", f"swarm {swarm_id} already exists")
        owner = self._principal_or_error(owner_principal_id)
        if owner.get("status") != "active":
            raise AuthorityError("FORBIDDEN", "swarm owner principal must be active")
        swarm = self._storage.swarm_insert(
            swarm_id=swarm_id,
            display_name=display_name,
            owner_principal_id=owner_principal_id,
            status="active",
            created_at=_utcnow_iso(),
            created_by=actor.principal_id,
            metadata=metadata or {},
        )
        self._storage.membership_insert(
            membership_id=f"m_{uuid4().hex}",
            swarm_id=swarm_id,
            principal_id=owner_principal_id,
            role="owner",
            status="active",
            granted_at=_utcnow_iso(),
            granted_by=actor.principal_id,
            expires_at=None,
            metadata={"auto_created": True},
        )
        return swarm

    def get_swarm(self, *, actor: ResolvedPrincipal, swarm_id: str) -> dict:
        swarm_id = _validate_swarm_id(swarm_id)
        swarm = self._swarm_or_error(swarm_id)
        if actor.caller_role == "admin":
            return swarm
        if actor.principal_id == swarm.get("owner_principal_id"):
            return swarm
        if self._principal_membership_role(swarm_id=swarm_id, principal_id=actor.principal_id):
            return swarm
        raise AuthorityError("FORBIDDEN", "swarm access denied")

    def list_swarms(self, *, actor: ResolvedPrincipal) -> dict:
        swarms = self._storage.swarm_list()
        if actor.caller_role == "admin":
            return {"swarms": swarms}
        visible = []
        for swarm in swarms:
            sid = str(swarm.get("swarm_id") or "")
            if actor.principal_id == swarm.get("owner_principal_id"):
                visible.append(swarm)
                continue
            if sid and self._principal_membership_role(swarm_id=sid, principal_id=actor.principal_id):
                visible.append(swarm)
        return {"swarms": visible}

    def _assert_can_manage_membership(
        self,
        *,
        actor: ResolvedPrincipal,
        swarm_id: str,
        requested_role: str,
    ) -> None:
        requested_role = self._validate_membership_role(requested_role)
        if actor.caller_role == "admin":
            return
        actor_role = self._principal_membership_role(swarm_id=swarm_id, principal_id=actor.principal_id)
        if actor_role == "owner" and requested_role in {"manager", "member"}:
            return
        if actor_role == "manager" and requested_role == "member":
            return
        raise AuthorityError("FORBIDDEN", "membership mutation not allowed")

    def grant_membership(
        self,
        *,
        actor: ResolvedPrincipal,
        swarm_id: str,
        principal_id: str,
        role: str,
        expires_at: str | None = None,
        metadata: dict | None = None,
    ) -> dict:
        swarm_id = _validate_swarm_id(swarm_id)
        principal_id = _validate_principal_id(principal_id)
        role = self._validate_membership_role(role)
        expires_at = self._ensure_future_or_none(expires_at)
        self._swarm_or_error(swarm_id)
        target_principal = self._principal_or_error(principal_id)
        if target_principal.get("status") != "active":
            raise AuthorityError("FORBIDDEN", "target principal must be active")
        self._assert_can_manage_membership(actor=actor, swarm_id=swarm_id, requested_role=role)
        existing = self._membership_active(swarm_id=swarm_id, principal_id=principal_id)
        if existing is not None:
            if existing.get("role") == role and str(existing.get("expires_at") or "") == str(expires_at or ""):
                return existing
            self._storage.membership_revoke(
                membership_id=str(existing["membership_id"]),
                revoked_at=_utcnow_iso(),
                revoked_by=actor.principal_id,
            )
        return self._storage.membership_insert(
            membership_id=f"m_{uuid4().hex}",
            swarm_id=swarm_id,
            principal_id=principal_id,
            role=role,
            status="active",
            granted_at=_utcnow_iso(),
            granted_by=actor.principal_id,
            expires_at=expires_at,
            metadata=metadata or {},
        )

    def revoke_membership(
        self,
        *,
        actor: ResolvedPrincipal,
        swarm_id: str,
        principal_id: str,
    ) -> dict:
        swarm_id = _validate_swarm_id(swarm_id)
        principal_id = _validate_principal_id(principal_id)
        existing = self._membership_active(swarm_id=swarm_id, principal_id=principal_id)
        if existing is None:
            raise AuthorityError("NOT_FOUND", f"active membership not found for {principal_id} in {swarm_id}")
        self._assert_can_manage_membership(
            actor=actor,
            swarm_id=swarm_id,
            requested_role=str(existing.get("role") or "member"),
        )
        return self._storage.membership_revoke(
            membership_id=str(existing["membership_id"]),
            revoked_at=_utcnow_iso(),
            revoked_by=actor.principal_id,
        ) or existing

    def list_memberships(
        self,
        *,
        actor: ResolvedPrincipal,
        swarm_id: str | None = None,
        principal_id: str | None = None,
        include_revoked: bool = False,
    ) -> dict:
        if swarm_id not in (None, ""):
            swarm_id = _validate_swarm_id(str(swarm_id))
        if principal_id not in (None, ""):
            principal_id = _validate_principal_id(str(principal_id))
        if actor.caller_role != "admin":
            if swarm_id not in (None, ""):
                actor_role = self._principal_membership_role(
                    swarm_id=str(swarm_id),
                    principal_id=actor.principal_id,
                )
                if actor_role not in {"owner", "manager"} and actor.principal_id != principal_id:
                    raise AuthorityError("FORBIDDEN", "membership list denied")
            elif principal_id not in (None, "", actor.principal_id):
                raise AuthorityError("FORBIDDEN", "cannot list another principal's memberships")
            else:
                principal_id = actor.principal_id
        memberships = self._storage.membership_list(
            swarm_id=swarm_id if swarm_id not in (None, "") else None,
            principal_id=principal_id if principal_id not in (None, "") else None,
            include_revoked=include_revoked,
        )
        filtered = []
        for membership in memberships:
            if include_revoked or self._membership_active(
                swarm_id=str(membership.get("swarm_id") or ""),
                principal_id=str(membership.get("principal_id") or ""),
            ) is not None:
                filtered.append(membership)
        return {"memberships": filtered}
