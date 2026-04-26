# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import src.mcp_server as mcp_mod
from src.membership import ResolvedPrincipal

_AUTH_HARNESSES: dict[str, AuthorityHarness] = {}


def _default_bootstrap_token() -> str:
    return str(os.environ.get("GOSH_TEST_BOOTSTRAP_ADMIN_TOKEN", "bootstrap-admin"))


def _default_admin_principal_id() -> str:
    return str(os.environ.get("GOSH_TEST_ADMIN_PRINCIPAL", "service:bootstrap-admin"))


def _patch_auth_extraction(monkeypatch) -> None:
    async def mock_extract_session(**kwargs):
        sn = int(kwargs.get("session_num", 1) or 1)
        session_date = str(kwargs.get("session_date") or "2026-04-06")
        session_text = str(
            kwargs.get("session_text")
            or kwargs.get("content")
            or kwargs.get("text")
            or ""
        ).strip()
        fact_text = session_text or f"fact for session {sn}"
        fact = {
            "id": f"s{sn}_f1",
            "fact": fact_text,
            "kind": "event",
            "entities": [],
            "tags": ["auth"],
            "session": sn,
        }
        return ("auth-conv", sn, session_date, [fact], [])

    async def mock_session_merge_stub(**kwargs):
        return ("auth-conv", 1, "2026-04-06", [])

    async def mock_cross_merge_stub(**kwargs):
        return ("auth-conv", "auth", [])

    async def mock_call_extract(*args, **kwargs):
        return {}

    async def mock_call_oai(*args, **kwargs):
        return "test answer"

    async def mock_call_model(*args, **kwargs):
        return "test answer"

    monkeypatch.setattr("src.memory.extract_session", mock_extract_session)
    monkeypatch.setattr("src.memory.call_extract", mock_call_extract)
    monkeypatch.setattr("src.common.call_extract", mock_call_extract)
    monkeypatch.setattr("src.memory.call_oai", mock_call_oai)
    monkeypatch.setattr("src.common.call_oai", mock_call_oai)
    monkeypatch.setattr("src.memory._call_model", mock_call_model)
    monkeypatch.setattr("src.common._call_model", mock_call_model)


@dataclass
class AuthorityHarness:
    bootstrap_token: str
    admin_token: str
    admin_principal_id: str

    def resolve(self, token: str) -> ResolvedPrincipal:
        return mcp_mod._get_authority().resolve_token(token)

    def issue(
        self,
        principal_id: str,
        *,
        kind: str | None = None,
        token_kind: str | None = None,
        display_name: str | None = None,
    ) -> str:
        if kind is None:
            kind = str(principal_id).split(":", 1)[0]
        authority = mcp_mod._get_authority()
        admin_actor = authority.resolve_token(self.admin_token)
        if authority.get_principal(actor=admin_actor, principal_id=self.admin_principal_id):
            pass
        try:
            authority.get_principal(actor=admin_actor, principal_id=principal_id)
        except Exception:
            authority.create_principal(
                actor=admin_actor,
                principal_id=principal_id,
                kind=kind,
                display_name=display_name,
            )
        issued = authority.issue_token(
            actor=admin_actor,
            principal_id=principal_id,
            token_kind=token_kind or ("agent" if kind == "agent" else "user"),
        )
        return issued["token"]

    def create_swarm(self, swarm_id: str, owner_principal_id: str) -> dict:
        authority = mcp_mod._get_authority()
        admin_actor = authority.resolve_token(self.admin_token)
        return authority.create_swarm(
            actor=admin_actor,
            swarm_id=swarm_id,
            owner_principal_id=owner_principal_id,
        )

    def grant(self, actor_token: str, *, swarm_id: str, principal_id: str, role: str = "member") -> dict:
        authority = mcp_mod._get_authority()
        actor = authority.resolve_token(actor_token)
        return authority.grant_membership(
            actor=actor,
            swarm_id=swarm_id,
            principal_id=principal_id,
            role=role,
        )

    def register_agent_public_key(
        self,
        principal_id: str,
        *,
        public_key: str,
        algorithm: str = "x25519",
        key_id: str | None = None,
    ) -> dict:
        authority = mcp_mod._get_authority()
        actor = authority.resolve_token(self.admin_token)
        return authority.set_agent_public_key_binding(
            actor=actor,
            principal_id=principal_id,
            public_key=public_key,
            algorithm=algorithm,
            key_id=key_id,
        )


def reset_authority_state() -> None:
    service = getattr(mcp_mod, "_authority_service", None)
    if service is not None:
        try:
            service._storage.close()
        except Exception:
            pass
    mcp_mod._authority_service = None
    mcp_mod._authority_data_dir = None
    _AUTH_HARNESSES.clear()
    import sys

    mcp_auth = sys.modules.get("tests._mcp_auth")
    if mcp_auth is not None and hasattr(mcp_auth, "_HARNESSES"):
        mcp_auth._HARNESSES.clear()


def configure_bootstrap_env(
    monkeypatch,
    data_dir,
    *,
    bootstrap_token: str | None = None,
    patch_extraction: bool = False,
) -> str:
    if bootstrap_token is None:
        bootstrap_token = _default_bootstrap_token()
    mcp_mod.data_dir = str(data_dir)
    monkeypatch.setenv("GOSH_MEMORY_ALLOW_PLAINTEXT_SECRETS", "1")
    monkeypatch.setenv("GOSH_MEMORY_ADMIN_TOKEN", bootstrap_token)
    mcp_mod.ADMIN_TOKEN = bootstrap_token
    if patch_extraction:
        _patch_auth_extraction(monkeypatch)
    reset_authority_state()
    return bootstrap_token


def bootstrap_harness(
    monkeypatch,
    data_dir,
    *,
    bootstrap_token: str | None = None,
    patch_extraction: bool = False,
) -> AuthorityHarness:
    if bootstrap_token is None:
        bootstrap_token = _default_bootstrap_token()
    resolved_dir = str(Path(data_dir).resolve())
    configure_bootstrap_env(
        monkeypatch,
        resolved_dir,
        bootstrap_token=bootstrap_token,
        patch_extraction=patch_extraction,
    )
    cached = _AUTH_HARNESSES.get(resolved_dir)
    if cached is not None:
        return cached
    admin_principal_id = _default_admin_principal_id()
    issued = mcp_mod._get_authority().bootstrap_admin(
        principal_id=admin_principal_id,
        kind="service",
        display_name="Bootstrap Admin",
    )
    harness = AuthorityHarness(
        bootstrap_token=bootstrap_token,
        admin_token=issued["token"],
        admin_principal_id=admin_principal_id,
    )
    _AUTH_HARNESSES[resolved_dir] = harness
    return harness
