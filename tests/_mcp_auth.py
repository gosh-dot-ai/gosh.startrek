# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

from __future__ import annotations

import os
from pathlib import Path

import src.mcp_server as mcp_mod

from tests._auth_helpers import AuthorityHarness, _AUTH_HARNESSES, configure_bootstrap_env

_HARNESSES: dict[str, AuthorityHarness] = {}


def install_test_verified_auth(monkeypatch) -> None:
    """Configure env bootstrap state; actual harness is bootstrapped lazily per data_dir."""
    bootstrap_token = str(
        getattr(mcp_mod, "ADMIN_TOKEN", "")
        or os.environ.get("GOSH_TEST_BOOTSTRAP_ADMIN_TOKEN", "bootstrap-admin")
    )
    configure_bootstrap_env(monkeypatch, mcp_mod.data_dir, bootstrap_token=bootstrap_token)


def _current_harness() -> AuthorityHarness:
    key = str(Path(mcp_mod.data_dir).resolve())
    harness = _HARNESSES.get(key) or _AUTH_HARNESSES.get(key)
    if harness is None:
        admin_principal_id = str(
            os.environ.get("GOSH_TEST_ADMIN_PRINCIPAL", "service:bootstrap-admin")
        )
        issued = mcp_mod._get_authority().bootstrap_admin(
            principal_id=admin_principal_id,
            kind="service",
            display_name="Bootstrap Admin",
        )
        harness = AuthorityHarness(
            bootstrap_token=str(
                getattr(mcp_mod, "ADMIN_TOKEN", "")
                or os.environ.get("GOSH_TEST_BOOTSTRAP_ADMIN_TOKEN", "bootstrap-admin")
            ),
            admin_token=issued["token"],
            admin_principal_id=admin_principal_id,
        )
    _HARNESSES[key] = harness
    return harness


def auth_token_for_agent(agent_id: str | None = None) -> str:
    raw = str(agent_id or "default").strip()
    if not raw or raw == "default":
        return ""
    if raw.startswith("agent:"):
        return _current_harness().issue(raw, kind="agent")
    if raw.startswith("user:"):
        return _current_harness().issue(raw, kind="user")
    if raw.startswith("service:"):
        return _current_harness().issue(raw, kind="service")
    if raw.startswith("swarm:"):
        raise RuntimeError("swarm identifiers are ACL groups, not principal tokens")
    return _current_harness().issue(f"agent:{raw}", kind="agent")


def auth_token_for_owner(owner_id: str) -> str:
    raw = str(owner_id or "").strip()
    if raw == "system":
        return _current_harness().admin_token
    if raw.startswith("agent:"):
        return _current_harness().issue(raw, kind="agent")
    if raw.startswith("user:"):
        return _current_harness().issue(raw, kind="user")
    if raw.startswith("service:"):
        return _current_harness().issue(raw, kind="service")
    return _current_harness().issue(f"agent:{raw}", kind="agent")
