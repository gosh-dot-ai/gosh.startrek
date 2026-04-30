#!/usr/bin/env python3
# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

import argparse
import asyncio
import contextvars
import hashlib
import json
import logging
import os
import secrets
import threading
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from src.config import MemoryConfig
from src.courier import Courier
from src.gosh_secrets import (
    GoshSecretsError,
    http_bearer_with_secret,
    http_header_value_with_secret,
    resolve_secrets_for_agent,
)
from src.membership import AuthorityError, AuthorityService, ResolvedPrincipal
from src.memory import MemoryServer, _is_visible, _normalize_identity
from src.storage import SQLiteAuthorityStorage, migrate_jsonnpz_to_sqlite

log = logging.getLogger(__name__)
STARTUP_LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"


def token_fingerprint(token: str) -> str:
    digest = hashlib.sha256(str(token or "").encode("utf-8")).hexdigest()
    return f"sha256:{digest[:12]}"


def configure_startup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format=STARTUP_LOG_FORMAT,
    )


def startup_log_lines(
    *,
    title: str,
    listening: str,
    data_dir: str | None = None,
    embeddings: str | None = None,
    tls: str | None = None,
    token: str,
    token_path: str | Path,
    summary: str | None = None,
) -> list[str]:
    header = title if summary is None else f"{title} — {summary}"
    lines = [
        header,
        f"Listening: {listening}",
    ]
    if data_dir is not None:
        lines.append(f"Data dir: {data_dir}")
    if embeddings is not None:
        lines.append(f"Embeddings: {embeddings}")
    if tls is not None:
        lines.append(f"TLS: {tls}")
    lines.extend([
        "POST /mcp — MCP tool calls",
        "GET /mcp/sse — Courier SSE stream",
        f"Token fingerprint: {token_fingerprint(token)}",
        f"Token saved to: {token_path}",
    ])
    return lines


def log_startup_lines(lines: list[str]) -> None:
    configure_startup_logging()
    startup_log = logging.getLogger("gosh.memory.startup")
    for line in lines:
        startup_log.info("%s", line)


def _safe_tool(fn):
    """Wrap MCP tool handler with structured error handling + logging."""
    import functools

    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        try:
            return await fn(*args, **kwargs)
        except ValueError as e:
            return {"error": str(e), "code": "VALIDATION_ERROR", "tool": fn.__name__}
        except Exception as e:
            tool_name = fn.__name__
            log.error("MCP tool %s failed: %s", tool_name, e, exc_info=True)
            return {"error": str(e), "code": "INTERNAL_ERROR", "tool": tool_name}

    return wrapper


def _sanitize_terminal_render_candidate(candidate: Any) -> Any:
    if isinstance(candidate, dict):
        return {
            key: _sanitize_terminal_render_candidate(value)
            for key, value in candidate.items()
            if key != "render_text"
        }
    if isinstance(candidate, list):
        return [_sanitize_terminal_render_candidate(value) for value in candidate]
    return candidate


# ── Identity context ──

@dataclass
class ConnectionContext:
    """Resolved caller identity for an MCP tool call."""
    owner_id: str = "system"
    agent_id: str = "default"
    swarm_id: str = "default"
    caller_role: str = "user"  # "user" | "admin"
    memberships: list[str] = field(default_factory=list)
    authenticated: bool = False
    auth_source: str = "none"
    auth_error: str | None = None
    auth_error_code: str | None = None
    principal_kind: str | None = None
    token_id: str | None = None
    token_kind: str | None = None

ADMIN_TOKEN = os.environ.get("GOSH_MEMORY_ADMIN_TOKEN", "")
_authority_lock = threading.Lock()
_authority_service: AuthorityService | None = None
_authority_data_dir: str | None = None


def _get_authority() -> AuthorityService:
    """Return the server-wide persisted authority service for the current data_dir."""
    global _authority_service, _authority_data_dir
    current_dir = str(Path(data_dir).resolve())
    with _authority_lock:
        if _authority_service is not None and _authority_data_dir == current_dir:
            return _authority_service
        enc_key_hex = os.environ.get("GOSH_MEMORY_ENCRYPTION_KEY")
        enc_key = bytes.fromhex(enc_key_hex) if enc_key_hex else None
        storage = SQLiteAuthorityStorage(current_dir, encryption_key=enc_key)
        _authority_service = AuthorityService(storage)
        _authority_data_dir = current_dir
        return _authority_service


def _resolve_identity(
    agent_id: str | None = "default",
    swarm_id: str = "default",
    token: str = None,
    agent_key: str = None,
) -> ConnectionContext:
    """Resolve caller principal from bootstrap env token or persisted principal token.

    `agent_id` and `swarm_id` remain data/scope selectors only. They do not grant
    caller identity or memberships.
    """
    effective_agent_id = str(agent_id or "default").strip() or "default"
    ctx = ConnectionContext(swarm_id=swarm_id, agent_id=effective_agent_id)
    token = str(token or "").strip() or _request_principal_token.get()

    if token and ADMIN_TOKEN and token == ADMIN_TOKEN:
        ctx.owner_id = "system"
        ctx.caller_role = "admin"
        ctx.authenticated = True
        ctx.auth_source = "bootstrap_admin_token"
        ctx.principal_kind = "service"
        ctx.token_kind = "bootstrap"  # noqa: S105 - token kind label, not a credential
        return ctx

    if not token:
        ctx.owner_id = "anonymous"
        ctx.auth_error = "principal token required"
        ctx.auth_error_code = "AUTH_REQUIRED"
        return ctx

    if agent_key:
        log.debug("Ignoring unverified agent_key identity input for principal auth")

    try:
        resolved = _get_authority().resolve_token(token)
    except AuthorityError as exc:
        ctx.owner_id = "anonymous"
        ctx.auth_error = str(exc)
        ctx.auth_error_code = exc.code
        return ctx
    ctx.owner_id = resolved.principal_id
    ctx.caller_role = resolved.caller_role
    ctx.memberships = list(resolved.memberships)
    ctx.authenticated = True
    ctx.auth_source = "principal_token"
    ctx.principal_kind = resolved.principal_kind
    ctx.token_id = resolved.token_id
    ctx.token_kind = resolved.token_kind
    return ctx


# ── Token authentication ──

SERVER_TOKEN = os.environ.get("GOSH_MEMORY_TOKEN", secrets.token_urlsafe(32))

# ── Module-level state ──

mcp = FastMCP(name="gosh-memory", streamable_http_path="/mcp")

registry: dict[str, MemoryServer] = {}
courier_registry: dict[str, Courier] = {}
connections: dict[str, asyncio.Queue] = {}
sub_to_conn: dict[str, str] = {}
_registry_lock = threading.Lock()
_instance_config_lock = threading.Lock()
# C2: track active SSE connections for hijack prevention
_active_connections: dict[str, str] = {}  # connection_id -> remote address or session id
# NOTE: .reset() is deliberately NOT called after requests. SSE/streaming tool
# handlers run after the middleware returns; resetting would clear the token
# before they read it. Each new request overwrites via .set().
# Only read this var from request-scoped code paths (tool handlers, admin endpoints).
_request_principal_token: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_principal_token",
    default=None,
)

data_dir: str = "./data"
cfg: MemoryConfig = MemoryConfig()
_write_log_worker_task: asyncio.Task | None = None

VALID_SCOPES = {"agent-private", "swarm-shared", "system-wide"}
_LOCALHOST_BIND_HOSTS = {"127.0.0.1", "localhost", "::1"}


def _transport_security_for_bind_host(bind_host: str | None) -> TransportSecuritySettings | None:
    host = str(bind_host or "127.0.0.1").strip().lower()
    if host in _LOCALHOST_BIND_HOSTS:
        return TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=["127.0.0.1:*", "localhost:*", "[::1]:*"],
            allowed_origins=["http://127.0.0.1:*", "http://localhost:*", "http://[::1]:*"],
        )
    return None


def _configure_mcp_bind_host(bind_host: str | None) -> None:
    """Retune FastMCP transport security for the actual server bind host."""
    mcp.settings.host = str(bind_host or "127.0.0.1")
    mcp.settings.transport_security = _transport_security_for_bind_host(bind_host)
    # The session manager captures transport security settings on first app construction.
    mcp._session_manager = None


def _json_field_text(payload: dict[str, Any], field: str, *, required: bool) -> str | None:
    value = payload.get(field)
    text = str(value or "").strip()
    if not text:
        if required:
            raise ValueError(f"{field} must be provided explicitly")
        return None
    return text


def _get_memory(key: str) -> MemoryServer:
    if not isinstance(key, str) or not key.strip():
        raise ValueError("key must be non-empty")
    server = registry.get(key)
    if server is not None:
        return server
    with _registry_lock:
        server = registry.get(key)
        if server is not None:
            return server
        server = MemoryServer(
            data_dir=data_dir,
            key=key,
            extract_model=cfg.extraction_model,
        )
        registry[key] = server
        return server


def _ensure_instance_config(server: MemoryServer, owner_id: str) -> bool:
    """Create instance config on first MCP write if not exists. Returns True if created."""
    with _instance_config_lock:
        if server._instance_config is not None:
            return False
        server._instance_config = {
            "owner_id": owner_id,
            "read": [],
            "_derived_read": [],
            "_derived_write": [],
            "write": [],
        }
        return True


def _missing_instance_error(server: MemoryServer) -> dict | None:
    if getattr(server, "_instance_config", None) is None:
        return {"error": "No memory instance exists for this key", "code": "NOT_FOUND"}
    return None


def _canonical_init_owner_id(owner_id: str, *, ctx: ConnectionContext) -> str:
    try:
        normalized = _normalize_identity(str(owner_id), allow_public=False)
    except ValueError as exc:
        raise ValueError("owner_id must be canonical") from exc
    if not normalized.startswith(("user:", "agent:", "service:")):
        raise ValueError("owner_id must be an existing principal_id")
    _get_authority().get_principal(actor=_ctx_actor(ctx), principal_id=normalized)
    return normalized


def _require_authenticated_principal(
    ctx: ConnectionContext,
    *,
    allow_bootstrap: bool = False,
) -> dict | None:
    """Fail closed unless the caller was resolved from a verified principal token."""
    if ctx.authenticated:
        if ctx.auth_source == "bootstrap_admin_token" and not allow_bootstrap:
            return {
                "error": "bootstrap admin token may only be used for auth_bootstrap_admin",
                "code": "FORBIDDEN",
            }
        return None
    return {
        "error": ctx.auth_error or "principal token required",
        "code": ctx.auth_error_code or "AUTH_REQUIRED",
    }


def _require_admin(ctx: ConnectionContext, *, allow_bootstrap: bool = False) -> dict | None:
    auth_error = _require_authenticated_principal(ctx, allow_bootstrap=allow_bootstrap)
    if auth_error:
        return auth_error
    if ctx.caller_role != "admin":
        return {"error": "admin principal required", "code": "FORBIDDEN"}
    return None


def _require_bootstrap_admin(ctx: ConnectionContext) -> dict | None:
    auth_error = _require_authenticated_principal(ctx, allow_bootstrap=True)
    if auth_error:
        return auth_error
    if ctx.auth_source != "bootstrap_admin_token" or ctx.caller_role != "admin":
        return {"error": "bootstrap admin token required", "code": "FORBIDDEN"}
    return None


def _ctx_memberships(ctx: ConnectionContext) -> list[str]:
    return list(dict.fromkeys(ctx.memberships or []))


def _resolve_live_content_agent_id(
    server: MemoryServer,
    ctx: ConnectionContext,
    requested_agent_id: str | None,
) -> tuple[str | None, dict | None]:
    """Resolve trusted producer agent_id for protected live content writes."""
    try:
        return (
            server._resolve_live_writer_agent_id(
                requested_agent_id=requested_agent_id,
                caller_id=ctx.owner_id,
                caller_principal_kind=ctx.principal_kind,
            ),
            None,
        )
    except PermissionError as exc:
        return None, {"error": str(exc), "code": "FORBIDDEN"}
    except ValueError as exc:
        return None, {"error": str(exc), "code": "VALIDATION_ERROR"}


def _resolve_mal_binding_id(ctx: ConnectionContext, requested_agent_id: str | None) -> str:
    """Select the canonical MAL binding for the current caller.

    Non-admin callers are always bound to their verified principal identity.
    Admin callers may explicitly target another binding via agent_id; absent an
    explicit selector, they operate on the system binding.
    """
    if ctx.caller_role != "admin":
        return _normalize_identity(str(ctx.owner_id), allow_public=False)
    raw = str(requested_agent_id or "").strip()
    if not raw or raw == "default":
        return "system"
    if raw.startswith(("agent:", "user:", "service:")) or raw in {"system", "anonymous"}:
        return _normalize_identity(raw, allow_public=False)
    return _normalize_identity(f"agent:{raw}", allow_public=False)


def _ctx_actor(ctx: ConnectionContext) -> ResolvedPrincipal:
    auth_error = _require_authenticated_principal(ctx)
    if auth_error:
        raise AuthorityError(
            auth_error.get("code", "AUTH_REQUIRED"),
            auth_error.get("error", "principal token required"),
        )
    if ctx.auth_source == "bootstrap_admin_token":
        raise AuthorityError(
            "FORBIDDEN",
            "bootstrap admin token may only be used for auth_bootstrap_admin",
        )
    return ResolvedPrincipal(
        principal_id=ctx.owner_id,
        principal_kind=ctx.principal_kind or "user",
        token_id=ctx.token_id or "",
        token_kind=ctx.token_kind or "user",
        memberships=_ctx_memberships(ctx),
        caller_role=ctx.caller_role,
    )


def _authority_error(exc: AuthorityError) -> dict:
    return {"error": str(exc), "code": exc.code}


def _require_scope_membership(
    ctx: ConnectionContext,
    *,
    scope: str,
    swarm_id: str,
) -> dict | None:
    if scope == "swarm-shared" and swarm_id and swarm_id != "default" and ctx.caller_role != "admin":
        grant = f"swarm:{swarm_id}"
        if grant not in _ctx_memberships(ctx):
            return {
                "error": f"active membership in {grant} required",
                "code": "ACL_FORBIDDEN",
            }
    return None


def _resource_acl(ctx: ConnectionContext, *, scope: str, swarm_id: str) -> tuple[str, list[str], list[str]]:
    read, write = _acl_defaults(ctx.owner_id, scope, swarm_id)
    if scope == "system-wide":
        return "system", read, write
    return ctx.owner_id, read, write


def _expand_instance_read_for_scope(
    server: MemoryServer,
    *,
    scope: str,
    swarm_id: str,
    owner_id: str | None = None,
) -> bool:
    """Mirror shared/public fact visibility at the instance read gate.

    Instance ACL stays owner-only by default. When the owner writes shared/public
    content, widen only the read gate so collaborators can reach the visible
    facts without implicitly granting write access to the whole memory key.
    """
    cfg = getattr(server, "_instance_config", None)
    if not cfg:
        return False

    if scope == "system-wide":
        grant = "agent:PUBLIC"
    elif scope == "swarm-shared" and swarm_id and swarm_id != "default":
        grant = f"swarm:{swarm_id}"
    else:
        return False

    read = list(cfg.get("_derived_read", []))
    if grant in read:
        return False
    read.append(grant)
    cfg["_derived_read"] = read
    return True


def _expand_instance_write_for_scope(
    server: MemoryServer,
    *,
    scope: str,
    swarm_id: str,
    owner_id: str | None = None,
) -> bool:
    """Enable future writes for the scope principals already allowed by content ACL."""
    cfg = getattr(server, "_instance_config", None)
    if not cfg:
        return False

    if scope == "swarm-shared" and swarm_id and swarm_id != "default":
        grant = f"swarm:{swarm_id}"
    else:
        return False

    write = list(cfg.get("_derived_write", []))
    if grant in write:
        return False
    write.append(grant)
    cfg["_derived_write"] = write
    return True


def _expand_instance_acl_for_scope(
    server: MemoryServer,
    *,
    scope: str,
    swarm_id: str,
    owner_id: str | None = None,
) -> bool:
    """Apply all derived instance ACL grants implied by a stored scope."""
    changed = _expand_instance_read_for_scope(
        server,
        scope=scope,
        swarm_id=swarm_id,
        owner_id=owner_id,
    )
    if _expand_instance_write_for_scope(
        server,
        scope=scope,
        swarm_id=swarm_id,
        owner_id=owner_id,
    ):
        changed = True
    return changed


def _content_write_instance_denied(
    server: MemoryServer,
    ctx: ConnectionContext,
    *,
    scope: str,
    swarm_id: str,
) -> dict | None:
    """Apply instance ACL to content writes.

    The instance gate remains authoritative for private writes. The only
    narrow exception is named swarm-shared content, where an active member may
    append shared rows before the instance gate is widened to that swarm.
    """
    denied = _check_instance_acl(server, ctx.owner_id, "write", ctx.caller_role, ctx.memberships)
    if not denied:
        return None
    if (
        scope == "swarm-shared"
        and swarm_id
        and swarm_id != "default"
        and (ctx.caller_role == "admin" or f"swarm:{swarm_id}" in _ctx_memberships(ctx))
    ):
        return None
    return denied


def _validate_explicit_scope(scope: str | None) -> dict | None:
    if scope is None or str(scope).strip() == "":
        return {
            "error": "scope must be provided explicitly",
            "code": "VALIDATION_ERROR",
        }
    return None


def _should_widen_instance_acl(
    result: dict | None,
    *,
    scope: str,
    swarm_id: str,
    write_kind: str,
) -> bool:
    """Apply derived instance ACL only for writes that establish visible shared state.

    Rules:
    - named swarm-shared writes always widen so collaborators can reach the key
      even when extraction yields zero facts
    - system-wide writes always widen immediately; public key visibility must
      not depend on extraction output
    - document ingests widen on success because the raw document itself is the
      shared payload, not just extracted facts
    """
    if not isinstance(result, dict):
        return False
    if result.get("status") == "duplicate":
        return False
    if scope == "swarm-shared" and swarm_id and swarm_id != "default":
        return True
    if scope == "system-wide":
        return True
    if write_kind == "document" and (
        scope == "system-wide"
        or (scope == "swarm-shared" and swarm_id and swarm_id != "default")
    ):
        return True
    try:
        return int(result.get("facts_extracted") or 0) > 0
    except Exception:
        return False


def _check_instance_acl(server: MemoryServer, owner_id: str, need: str,
                         caller_role: str = "user",
                         memberships: list[str] = None,
                         include_derived: bool = False) -> dict | None:
    """Check instance-level ACL. Returns error dict if denied, None if allowed.

    Uses the same ACL model as per-fact _acl_allows():
    admin → owner → system-sees-system → agent:PUBLIC → direct grant → membership → deny.
    need: "read" or "write".
    """
    cfg = server._instance_config
    if cfg is None:
        return None  # No instance config yet — ACL not active
    if caller_role == "admin":
        return None
    if cfg["owner_id"] == owner_id:
        return None
    # system caller sees system-owned instances
    if owner_id == "system" and cfg["owner_id"] == "system":
        return None
    granted = list(cfg.get(need, []))
    if include_derived and need == "read":
        granted.extend(cfg.get("_derived_read", []))
    if need == "write":
        granted.extend(cfg.get("_derived_write", []))
    # Public grant
    if "agent:PUBLIC" in granted:
        return None
    # Direct grant
    if owner_id in granted:
        return None
    # Membership grant (persisted authority memberships only)
    all_memberships = set(memberships or [])
    for m in all_memberships:
        if m in granted:
            return None
    return {"error": "Access denied by instance ACL", "code": "FORBIDDEN"}


async def _run_write_log_workers(poll_interval: float = 0.5, batch_size: int = 8) -> None:
    while True:
        try:
            for server in list(registry.values()):
                try:
                    await server.process_write_log_once(batch_size=batch_size)
                except Exception:
                    log.exception("write-log worker failed for key=%s", getattr(server, "key", "?"))
            await asyncio.sleep(poll_interval)
        except asyncio.CancelledError:
            raise



def _get_courier(key: str) -> Courier:
    if key not in courier_registry:
        mem = _get_memory(key)
        courier = Courier(mem)
        courier_registry[key] = courier
        asyncio.create_task(courier.run(poll_interval=1.0))
    return courier_registry[key]


# ── MCP Tools ──

async def _memory_init_instance(
    key: str,
    *,
    ctx: ConnectionContext,
    owner_id: str | None = None,
) -> dict:
    """Create an empty instance without writing any memory content."""
    server = _get_memory(key)
    auth_error = _require_authenticated_principal(ctx)
    if auth_error:
        return auth_error
    if getattr(server, "_instance_config", None) is not None:
        return {"error": "Memory instance already exists for this key", "code": "ALREADY_EXISTS"}
    resolved_owner_id = ctx.owner_id
    if owner_id is not None:
        admin_error = _require_admin(ctx)
        if admin_error:
            return admin_error
        if not str(owner_id).strip():
            return {"error": "owner_id must be non-empty", "code": "VALIDATION_ERROR"}
        try:
            resolved_owner_id = _canonical_init_owner_id(owner_id, ctx=ctx)
        except ValueError as exc:
            return {"error": str(exc), "code": "VALIDATION_ERROR"}
        except AuthorityError as exc:
            return _authority_error(exc)
    created = _ensure_instance_config(server, resolved_owner_id)
    if not created:
        return {"error": "Memory instance already exists for this key", "code": "ALREADY_EXISTS"}
    async with server._file_lock:
        server._save_cache()
    return {"status": "ok", "created": True, "owner_id": resolved_owner_id}


@mcp.tool(name="memory_store")
@_safe_tool
async def memory_store(
    key: str,
    content: str,
    session_num: int,
    session_date: str,
    speakers: str = "User and Assistant",
    agent_id: str | None = None,
    swarm_id: str = "default",
    scope: str | None = None,
    upsert_by_key: str = None,
    content_type: str = "default",
    content_format: str | None = None,
    librarian_prompt: str = None,
    source_id: str = None,
    retention_ttl: int = None,
    metadata: dict = None,
    target: str | list[str] | None = None,
    token: str = None,
    agent_key: str = None,
) -> dict:
    """Store a conversation turn. Extracts atomic facts and persists to disk.

    upsert_by_key: if set, replaces existing session with same key (agent-private only).
    content_type: prompt registry key (default, financial, technical, personal, regulatory, agent_trace).
    content_format: optional extraction format override (e.g. CONVERSATION, AGENT_TRACE, WEB_DOM, CODE_TRACE).
    librarian_prompt: inline extraction prompt override (agent-private only).
    source_id: optional identifier for dedup (same source_id + session_num = dedup key).
    retention_ttl: seconds until facts expire (None = never).
    metadata: optional dict of metadata to attach to extracted facts.
    target: optional delivery target(s). Normalized to top-level list[str].
    """
    server = _get_memory(key)
    ctx = _resolve_identity(agent_id=agent_id, swarm_id=swarm_id, token=token, agent_key=agent_key)
    auth_error = _require_authenticated_principal(ctx)
    if auth_error:
        return auth_error
    effective_agent_id, live_agent_error = _resolve_live_content_agent_id(server, ctx, agent_id)
    if live_agent_error:
        return live_agent_error
    scope_error = _validate_explicit_scope(scope)
    if scope_error:
        return scope_error
    if scope not in VALID_SCOPES:
        return {"error": f"Unknown scope: {scope}", "code": "INVALID_SCOPE"}
    scope_error = _require_scope_membership(ctx, scope=scope, swarm_id=swarm_id)
    if scope_error:
        return scope_error
    created = _ensure_instance_config(server, ctx.owner_id)
    denied = _content_write_instance_denied(server, ctx, scope=scope, swarm_id=swarm_id)
    if denied:
        return denied
    owner_id, read_acl, write_acl = _resource_acl(ctx, scope=scope, swarm_id=swarm_id)
    if created:
        async with server._file_lock:
            server._save_cache()
    result = await server.store(content, session_num, session_date, speakers,
                                agent_id=effective_agent_id, swarm_id=swarm_id, scope=scope,
                                upsert_by_key=upsert_by_key,
                                content_type=content_type,
                                content_format=content_format,
                                librarian_prompt=librarian_prompt,
                                owner_id=owner_id,
                                read=read_acl,
                                write=write_acl,
                                source_id=source_id,
                                retention_ttl=retention_ttl,
                                metadata=metadata,
                                target=target,
                                caller_id=ctx.owner_id,
                                caller_principal_kind=ctx.principal_kind)
    if _should_widen_instance_acl(result, scope=scope, swarm_id=swarm_id, write_kind="store") and _expand_instance_acl_for_scope(server, scope=scope, swarm_id=swarm_id, owner_id=ctx.owner_id):
        async with server._file_lock:
            server._save_cache()
    return result


@mcp.tool(name="memory_write")
@_safe_tool
async def memory_write(
    key: str,
    message_id: str,
    session_id: str,
    content: str,
    content_family: str,
    timestamp_ms: int,
    agent_id: str | None = None,
    swarm_id: str = "default",
    scope: str | None = None,
    metadata: dict = None,
    token: str = None,
    agent_key: str = None,
) -> dict:
    """Append a raw write-log entry without blocking on extraction."""
    server = _get_memory(key)
    ctx = _resolve_identity(agent_id=agent_id, swarm_id=swarm_id, token=token, agent_key=agent_key)
    auth_error = _require_authenticated_principal(ctx)
    if auth_error:
        return auth_error
    effective_agent_id, live_agent_error = _resolve_live_content_agent_id(server, ctx, agent_id)
    if live_agent_error:
        return live_agent_error
    scope_error = _validate_explicit_scope(scope)
    if scope_error:
        return scope_error
    if scope not in VALID_SCOPES:
        return {"error": f"Unknown scope: {scope}", "code": "INVALID_SCOPE"}
    scope_error = _require_scope_membership(ctx, scope=scope, swarm_id=swarm_id)
    if scope_error:
        return scope_error
    created = _ensure_instance_config(server, ctx.owner_id)
    denied = _content_write_instance_denied(server, ctx, scope=scope, swarm_id=swarm_id)
    if denied:
        return denied
    owner_id, read_acl, write_acl = _resource_acl(ctx, scope=scope, swarm_id=swarm_id)
    if created:
        async with server._file_lock:
            server._save_cache()
    try:
        result = await server.write(
            message_id=message_id,
            session_id=session_id,
            content=content,
            content_family=content_family,
            timestamp_ms=timestamp_ms,
            agent_id=effective_agent_id,
            swarm_id=swarm_id,
            scope=scope,
            owner_id=owner_id,
            read=read_acl,
            write=write_acl,
            metadata=metadata,
            caller_id=ctx.owner_id,
            caller_principal_kind=ctx.principal_kind,
        )
    except ValueError as e:
        return {"error": str(e), "code": "VALIDATION_ERROR"}
    if _expand_instance_acl_for_scope(server, scope=scope, swarm_id=swarm_id, owner_id=ctx.owner_id):
        async with server._file_lock:
            server._save_cache()
    return result


@mcp.tool(name="memory_write_status")
@_safe_tool
async def memory_write_status(
    key: str,
    message_id: str,
    agent_id: str = "default",
    swarm_id: str = "default",
    token: str = None,
    agent_key: str = None,
) -> dict:
    """Check extraction state for a write-log entry."""
    server = _get_memory(key)
    ictx = _resolve_identity(agent_id=agent_id, swarm_id=swarm_id, token=token, agent_key=agent_key)
    auth_error = _require_authenticated_principal(ictx)
    if auth_error:
        return auth_error
    denied = _check_instance_acl(server, ictx.owner_id, "read", ictx.caller_role, ictx.memberships, include_derived=True)
    if denied:
        return denied
    status = server.write_status(message_id)
    if status is None:
        return {"error": f"Write {message_id} not found", "code": "NOT_FOUND"}
    _memberships = _ctx_memberships(ictx)
    if not server._raw_entry_acl_allows(status, ictx.owner_id, _memberships, ictx.caller_role):
        return {"error": f"Write {message_id} not found", "code": "NOT_FOUND"}
    return {
        "message_id": status.get("message_id"),
        "extraction_state": status.get("extraction_state"),
        "extraction_attempts": status.get("extraction_attempts"),
        "last_extraction_attempt_ms": status.get("last_extraction_attempt_ms"),
        "duplicate_of": (status.get("metadata") or {}).get("duplicate_of"),
        "near_duplicate_of": (status.get("metadata") or {}).get("near_duplicate_of"),
    }



def _public_recall_continuation(continuation: dict | None) -> dict | None:
    if not isinstance(continuation, dict):
        return None
    public = deepcopy(continuation)
    if public.get("available"):
        public["tool"] = "get_more_context"
        public["mcp_tool"] = "get_more_context"
        public["tool_usage"] = (
            "call get_more_context with handle=<handle> and page=\"next\" "
            "to fetch the next evidence page"
        )
    return public


def _public_recall_continuation_instruction(handle: str | None = None) -> str:
    handle_text = "<handle>"
    if handle:
        handle_text = str(handle)
    return (
        "call get_more_context with handle="
        f"\"{handle_text}\" and page=\"next\" to fetch the next evidence page"
    )


def _public_recall_context(context: str, continuation: dict | None) -> str:
    if not isinstance(continuation, dict) or not continuation.get("available"):
        return str(context or "")
    handle = str(continuation.get("handle") or "")
    text = str(context or "")
    replacements = {
        "call get_more_context with page=\"next\" or without session_id to retrieve the next evidence page":
            _public_recall_continuation_instruction(handle).replace("fetch", "retrieve"),
        "call get_more_context with page=\"next\" or without session_id to fetch the next evidence page":
            _public_recall_continuation_instruction(handle),
        "call get_more_context with page=\"next\" or no session_id to fetch the next page":
            _public_recall_continuation_instruction(handle).replace("evidence page", "page"),
        "call memory_recall with continuation_handle=<handle> and page=\"next\" to fetch the next evidence page":
            _public_recall_continuation_instruction(handle),
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _public_answer_contract(contract: dict | None) -> dict | None:
    if not isinstance(contract, dict):
        return None
    public = deepcopy(contract)
    continuation = public.get("recall_continuation")
    if isinstance(continuation, dict) and continuation.get("available"):
        handle = str(continuation.get("handle") or "")
        continuation["tool"] = "get_more_context"
        continuation["mcp_tool"] = "get_more_context"
        continuation["instruction"] = (
            "The returned context is the first evidence page. If the answer is not present "
            "and more evidence is available, "
            f"{_public_recall_continuation_instruction(handle)}."
        )
        public["recall_continuation"] = continuation
        prompt_template = public.get("prompt_template")
        if isinstance(prompt_template, str):
            public["prompt_template"] = _public_recall_context(prompt_template, continuation)
    return public


@mcp.tool(name="memory_recall")
@_safe_tool
async def memory_recall(
    key: str,
    query: str = "",
    agent_id: str = "default",
    swarm_id: str = "default",
    search_family: str = "auto",
    token_budget: int = 4000,
    query_type: str = "auto",
    kind: str = "all",
    query_metadata: dict | None = None,
    continuation_handle: str | None = None,
    page: int | str | None = None,
    token: str = None,
    agent_key: str = None,
) -> dict:
    """Use memory_recall as an iterative memory tool.

    You may call memory_recall multiple times before producing a final answer.
    If retrieved context is missing, omitted, truncated, ambiguous, stale, or
    insufficient to ground the requested answer/edit, call memory_recall again
    with a narrower query instead of guessing.

    For repository/code tasks, use search_family="codebase". Ask focused
    follow-up recall queries for exact files, symbols, config entries, related
    tests, workflows/commands, dependency context, previous patch attempts,
    test failures, or verification state as needed.

    Queries must be English. If the user/task prompt is not English, translate
    it in the calling agent/model before invoking memory_recall. A non-English
    query fails closed with NON_ENGLISH_QUERY and no retrieval or inference
    planning is performed.

    Successful responses may include answer_contract. Its prompt_template is
    intentionally public: it is the stable synthesis contract that lets MCP
    callers answer from recall evidence with the same prompt leaf memory_ask
    uses internally. Provider payloads, profile choices, and secret refs remain
    outside memory_recall.

    If a response includes recall_continuation.available=true, fetch later
    evidence pages by calling get_more_context with
    handle=<recall_continuation.handle> and page="next". The handle is
    family-agnostic: callers do not pass conversation/document/codebase
    internals. The older memory_recall continuation_handle path remains
    accepted for compatibility, but get_more_context is the preferred public
    paging contract.

    Do not generate code changes from absent evidence. If repeated focused
    recall calls cannot provide the required evidence, report the precise
    missing evidence.

    search_family:
        auto | conversation | document | codebase
        For repository file edits, use search_family="codebase".
    query_type:
        auto | lookup | temporal | aggregate | current | synthesize |
        procedural | prospective | exact_copy
        For codebase tasks:
        - lookup: exact file/symbol/config/test lookup
        - current: latest operation state or current repo-work state
        - aggregate: grouped/listed evidence
        - synthesize: broader codebase context
        - exact_copy: exact source/span rendering when available
        exact_copy returns a sanitized terminal_render_candidate for model
        decision. Final exact rendering is only performed by memory_ask; recall
        intentionally exposes no raw render text and no exact-copy render tool.
    kind: all | fact | preference | decision | constraint | rule | ...
    token: OAuth bearer token for identity resolution
    agent_key: API key for identity resolution
    """
    server = _get_memory(key)
    ictx = _resolve_identity(agent_id=agent_id, swarm_id=swarm_id, token=token, agent_key=agent_key)
    auth_error = _require_authenticated_principal(ictx)
    if auth_error:
        return auth_error
    denied = _check_instance_acl(
        server,
        ictx.owner_id,
        "read",
        ictx.caller_role,
        ictx.memberships,
        include_derived=True,
    )
    if denied:
        return denied
    _memberships = _ctx_memberships(ictx)
    mal_binding_id = _resolve_mal_binding_id(ictx, agent_id)
    if continuation_handle:
        result = server.recall_continuation_page(
            continuation_handle=continuation_handle,
            page=page or "next",
            caller_id=ictx.owner_id,
            caller_memberships=_memberships,
            caller_role=ictx.caller_role,
            swarm_id=swarm_id,
            bind_swarm_from_handle=True,
        )
        if result.get("error") or result.get("code"):
            return {
                "error": result.get("error", "Recall continuation failed"),
                "code": result.get("code", "RECALL_CONTINUATION_ERROR"),
                "recall_continuation": _public_recall_continuation(result.get("recall_continuation")) or {},
            }
        continuation = _public_recall_continuation(result.get("recall_continuation")) or {}
        context = _public_recall_context(str(result.get("context") or ""), continuation)
        return {
            "telemetry_version": 1,
            "context": context,
            "retrieved_count": 0,
            "query_type": result.get("query_type", "continuation"),
            "token_estimate": len(context) // 4,
            "sessions_in_context": 0,
            "total_sessions": 0,
            "coverage_pct": 0,
            "recall_continuation": continuation,
            "runtime_trace": result.get("runtime_trace", {}),
        }
    if not str(query or "").strip():
        return {"error": "query is required unless continuation_handle is provided", "code": "MISSING_QUERY"}
    try:
        result = await server.recall(
            query=query,
            agent_id=agent_id,
            swarm_id=swarm_id,
            search_family=search_family,
            token_budget=token_budget,
            query_type=query_type,
            kind=kind,
            query_metadata=query_metadata,
            caller_memberships=_memberships,
            caller_role=ictx.caller_role,
            caller_id=ictx.owner_id,
            mal_binding_id=mal_binding_id,
        )
    except Exception as e:
        import traceback
        log.error("memory_recall error:\n%s", traceback.format_exc())
        return {"error": str(e), "code": "RECALL_ERROR"}
    server._remember_recall_continuation(
        result,
        caller_id=ictx.owner_id,
        caller_memberships=_memberships,
        caller_role=ictx.caller_role,
        swarm_id=swarm_id,
    )
    for inference_field in ("recommended_profile", "payload", "payload_meta", "_payload_secret_ref", "secret_ref"):
        result.pop(inference_field, None)
    if result.get("error") or result.get("code") or "context" not in result:
        resp = {
            "error": result.get("error", "Recall failed"),
            "code": result.get("code", "RECALL_ERROR"),
            "query_type": result.get("query_type", "default"),
        }
        if "runtime_trace" in result:
            resp["runtime_trace"] = result["runtime_trace"]
        return resp
    result_context = _public_recall_context(
        str(result.get("context") or ""),
        result.get("recall_continuation"),
    )
    max_chars = token_budget * 4
    if len(result_context) > max_chars:
        result_context = result_context[:max_chars] + "\n[...truncated]"
    default_hint = {
        "score": 0.0, "level": 1, "signals": [],
        "retrieval_complexity": 0.0, "content_complexity": 0.0, "dominant": "tie",
    }
    resp = {
        "telemetry_version": 1,
        "context": result_context,
        "retrieved_count": len(result.get("retrieved", [])),
        "query_type": result.get("query_type", "default"),
        "token_estimate": len(result_context) // 4,
        "complexity_hint": result.get("complexity_hint", default_hint),
        "sessions_in_context": result.get("sessions_in_context", 0),
        "total_sessions": result.get("total_sessions", 0),
        "coverage_pct": result.get("coverage_pct", 0),
        "raw_budget": result.get("raw_budget", 5000),
    }
    if "retrieval_families" in result:
        resp["retrieval_families"] = result["retrieval_families"]
    if "search_family" in result:
        resp["search_family"] = result["search_family"]
    if "answer_contract" in result:
        resp["answer_contract"] = _public_answer_contract(result["answer_contract"])
    if "terminal_render_candidate" in result:
        resp["terminal_render_candidate"] = _sanitize_terminal_render_candidate(
            deepcopy(result["terminal_render_candidate"])
        )
    if "actual_injected_episode_ids" in result:
        resp["actual_injected_episode_ids"] = result["actual_injected_episode_ids"]
    if "retrieved_episode_ids" in result:
        resp["retrieved_episode_ids"] = result["retrieved_episode_ids"]
    if "selection_scores" in result:
        resp["selection_scores"] = result["selection_scores"]
    if "runtime_trace" in result:
        resp["runtime_trace"] = result["runtime_trace"]
    if "repo_task_context_packs" in result:
        resp["repo_task_context_packs"] = result["repo_task_context_packs"]
    if "raw_recall_count" in result:
        resp["raw_recall_count"] = result["raw_recall_count"]
    if "recall_continuation" in result:
        resp["recall_continuation"] = _public_recall_continuation(result["recall_continuation"])
    return resp


def _find_memory_for_recall_continuation(handle: str, key: str | None = None) -> MemoryServer | None:
    normalized_handle = str(handle or "").strip()
    if not normalized_handle:
        return None
    if key and str(key).strip():
        return _get_memory(str(key).strip())
    with _registry_lock:
        servers = list(registry.values())
    for server in servers:
        continuations = getattr(server, "_recall_continuations", {})
        if isinstance(continuations, dict) and normalized_handle in continuations:
            return server
    return None


@mcp.tool(name="get_more_context")
@_safe_tool
async def get_more_context(
    handle: str = "",
    page: int | str | None = "next",
    key: str = "",
    agent_id: str = "default",
    swarm_id: str | None = None,
    continuation_handle: str | None = None,
    token: str = None,
    agent_key: str = None,
) -> dict:
    """Fetch the next family-agnostic recall continuation page.

    Preferred call shape:
        get_more_context(handle=<recall_continuation.handle>, page="next")

    The handle is opaque: callers do not pass conversation, document, codebase,
    source, file, cursor internals, or the first recall swarm_id. The legacy
    memory_recall continuation_handle path remains accepted for compatibility.
    """
    resolved_handle = str(handle or continuation_handle or "").strip()
    if not resolved_handle:
        return {"error": "handle is required", "code": "MISSING_HANDLE"}

    identity_swarm_id = swarm_id or "default"
    ictx = _resolve_identity(agent_id=agent_id, swarm_id=identity_swarm_id, token=token, agent_key=agent_key)
    auth_error = _require_authenticated_principal(ictx)
    if auth_error:
        return auth_error

    server = _find_memory_for_recall_continuation(resolved_handle, key=key)
    if server is None:
        return {
            "error": "Recall continuation not found",
            "code": "RECALL_CONTINUATION_NOT_FOUND",
            "recall_continuation": {
                "available": False,
                "handle": resolved_handle,
                "exhausted": True,
            },
        }

    denied = _check_instance_acl(
        server,
        ictx.owner_id,
        "read",
        ictx.caller_role,
        ictx.memberships,
        include_derived=True,
    )
    if denied:
        return denied

    result = server.recall_continuation_page(
        continuation_handle=resolved_handle,
        page=page or "next",
        caller_id=ictx.owner_id,
        caller_memberships=_ctx_memberships(ictx),
        caller_role=ictx.caller_role,
        swarm_id=swarm_id,
        bind_swarm_from_handle=True,
    )
    if result.get("error") or result.get("code"):
        return {
            "error": result.get("error", "Recall continuation failed"),
            "code": result.get("code", "RECALL_CONTINUATION_ERROR"),
            "recall_continuation": _public_recall_continuation(result.get("recall_continuation")) or {},
            "runtime_trace": result.get("runtime_trace", {}),
        }
    continuation = _public_recall_continuation(result.get("recall_continuation")) or {}
    context = _public_recall_context(str(result.get("context") or ""), continuation)
    return {
        "telemetry_version": 1,
        "context": context,
        "entries_rendered": (
            (result.get("runtime_trace") or {})
            .get("recall_continuation_trace", {})
            .get("entries_rendered", 0)
        ),
        "query_type": result.get("query_type", "continuation"),
        "token_estimate": len(context) // 4,
        "recall_continuation": continuation,
        "runtime_trace": result.get("runtime_trace", {}),
    }


@mcp.tool(name="memory_plan_inference")
@_safe_tool
async def memory_plan_inference(
    key: str,
    query: str,
    agent_id: str = "default",
    swarm_id: str = "default",
    search_family: str = "auto",
    token_budget: int = 4000,
    query_type: str = "auto",
    kind: str = "all",
    query_metadata: dict | None = None,
    inference_model: str = None,
    max_tokens: int = None,
    use_tool: bool = None,
    speakers: str = "User and Assistant",
    token: str = None,
    agent_key: str = None,
) -> dict:
    """Build executable inference planning metadata separately from evidence recall.

    Requires authenticated instance read access like profile/config reads. The
    returned secret_ref is an opaque runtime secret reference, never a secret value.
    """
    server = _get_memory(key)
    missing = _missing_instance_error(server)
    if missing:
        return missing
    ictx = _resolve_identity(agent_id=agent_id, swarm_id=swarm_id, token=token, agent_key=agent_key)
    auth_error = _require_authenticated_principal(ictx)
    if auth_error:
        return auth_error
    denied = _check_instance_acl(
        server,
        ictx.owner_id,
        "read",
        ictx.caller_role,
        ictx.memberships,
        include_derived=True,
    )
    if denied:
        return denied
    _memberships = _ctx_memberships(ictx)
    mal_binding_id = _resolve_mal_binding_id(ictx, agent_id)
    try:
        return await server.plan_inference(
            query=query,
            agent_id=agent_id,
            swarm_id=swarm_id,
            search_family=search_family,
            token_budget=token_budget,
            query_type=query_type,
            kind=kind,
            query_metadata=query_metadata,
            caller_memberships=_memberships,
            caller_role=ictx.caller_role,
            caller_id=ictx.owner_id,
            mal_binding_id=mal_binding_id,
            inference_model=inference_model,
            max_tokens=max_tokens,
            use_tool=use_tool,
            speakers=speakers,
        )
    except Exception as e:
        import traceback
        log.error("memory_plan_inference error:\n%s", traceback.format_exc())
        return {"error": str(e), "code": "PLAN_INFERENCE_ERROR"}


@mcp.tool(name="memory_ask")
@_safe_tool
async def memory_ask(
    key: str,
    query: str,
    agent_id: str = "default",
    swarm_id: str = "default",
    search_family: str = "auto",
    query_type: str = "auto",
    kind: str = "all",
    inference_model: str = None,
    max_tokens: int = None,
    use_tool: bool = None,
    shell_budget: float = None,
    speakers: str = "User and Assistant",
    query_metadata: dict | None = None,
    token: str = None,
    agent_key: str = None,
) -> dict:
    """Ask a question and get an answer using memory + LLM inference.

    search_family: auto | conversation | document | codebase
    Calls recall() internally, selects model via profiles, runs inference.
    Returns answer + metadata (profile_used, tool_called, budget_exceeded).
    """
    server = _get_memory(key)
    ictx = _resolve_identity(agent_id=agent_id, swarm_id=swarm_id, token=token, agent_key=agent_key)
    auth_error = _require_authenticated_principal(ictx)
    if auth_error:
        return auth_error
    denied = _check_instance_acl(
        server,
        ictx.owner_id,
        "read",
        ictx.caller_role,
        ictx.memberships,
        include_derived=True,
    )
    if denied:
        return denied
    _memberships = _ctx_memberships(ictx)
    mal_binding_id = _resolve_mal_binding_id(ictx, agent_id)
    return await server.ask(
        query=query,
        agent_id=agent_id,
        swarm_id=swarm_id,
        search_family=search_family,
        query_type=query_type,
        kind=kind,
        query_metadata=query_metadata,
        caller_memberships=_memberships,
        caller_role=ictx.caller_role,
        caller_id=ictx.owner_id,
        mal_binding_id=mal_binding_id,
        inference_model=inference_model or (cfg.inference_model if not server._has_profiles() else None),
        max_tokens=max_tokens,
        use_tool=use_tool,
        shell_budget=shell_budget,
        speakers=speakers,
    )


@mcp.tool(name="memory_set_profiles")
@_safe_tool
async def memory_set_profiles(
    key: str,
    profiles: dict,
    profile_configs: dict,
    agent_id: str = "default",
    swarm_id: str = "default",
    token: str = None,
    agent_key: str = None,
) -> dict:
    """Set inference profiles. Requires instance write on an existing instance."""
    server = _get_memory(key)
    missing = _missing_instance_error(server)
    if missing:
        return missing
    ictx = _resolve_identity(
        agent_id=agent_id, swarm_id=swarm_id,
        token=token, agent_key=agent_key,
    )
    auth_error = _require_authenticated_principal(ictx)
    if auth_error:
        return auth_error
    denied = _check_instance_acl(
        server, ictx.owner_id, "write", ictx.caller_role, ictx.memberships,
    )
    if denied:
        return denied
    await server.set_profiles(profiles, profile_configs)
    return {"status": "ok", "profiles": len(profiles), "configs": len(profile_configs)}


@mcp.tool(name="memory_set_config")
@_safe_tool
async def memory_set_config(
    key: str,
    config: dict,
    agent_id: str = "default",
    swarm_id: str = "default",
    token: str = None,
    agent_key: str = None,
) -> dict:
    """Set canonical memory-owned runtime config. Requires instance write on an existing instance."""
    server = _get_memory(key)
    missing = _missing_instance_error(server)
    if missing:
        return missing
    ictx = _resolve_identity(
        agent_id=agent_id, swarm_id=swarm_id,
        token=token, agent_key=agent_key,
    )
    auth_error = _require_authenticated_principal(ictx)
    if auth_error:
        return auth_error
    denied = _check_instance_acl(
        server, ictx.owner_id, "write", ictx.caller_role, ictx.memberships,
    )
    if denied:
        return denied
    await server.set_config(config)
    return {"status": "ok", "schema_version": config.get("schema_version")}


@mcp.tool(name="memory_get_config")
@_safe_tool
async def memory_get_config(
    key: str,
    agent_id: str = "default",
    swarm_id: str = "default",
    token: str = None,
    agent_key: str = None,
) -> dict:
    """Get canonical memory-owned runtime config. Requires instance read; not creation-capable."""
    server = _get_memory(key)
    missing = _missing_instance_error(server)
    if missing:
        return missing
    ictx = _resolve_identity(
        agent_id=agent_id, swarm_id=swarm_id,
        token=token, agent_key=agent_key,
    )
    auth_error = _require_authenticated_principal(ictx)
    if auth_error:
        return auth_error
    denied = _check_instance_acl(server, ictx.owner_id, "read", ictx.caller_role, ictx.memberships)
    if denied:
        return denied
    return server.get_config()


@mcp.tool(name="memory_get_profiles")
@_safe_tool
async def memory_get_profiles(
    key: str,
    agent_id: str = "default",
    swarm_id: str = "default",
    token: str = None,
    agent_key: str = None,
) -> dict:
    """Get inference profiles. Requires instance read; not creation-capable."""
    server = _get_memory(key)
    missing = _missing_instance_error(server)
    if missing:
        return missing
    ictx = _resolve_identity(
        agent_id=agent_id, swarm_id=swarm_id,
        token=token, agent_key=agent_key,
    )
    auth_error = _require_authenticated_principal(ictx)
    if auth_error:
        return auth_error
    denied = _check_instance_acl(server, ictx.owner_id, "read", ictx.caller_role, ictx.memberships)
    if denied:
        return denied
    return server.get_profiles()


@mcp.tool(name="memory_ingest_document")
@_safe_tool
async def memory_ingest_document(
    key: str,
    content: str,
    source_id: str,
    agent_id: str | None = None,
    swarm_id: str = "default",
    scope: str | None = None,
    retention_ttl: int = None,
    metadata: dict = None,
    target: str | list[str] | None = None,
    token: str = None,
    agent_key: str = None,
) -> dict:
    """Ingest a document. Extracts facts across all 3 tiers."""
    server = _get_memory(key)
    ctx = _resolve_identity(agent_id=agent_id, swarm_id=swarm_id, token=token, agent_key=agent_key)
    auth_error = _require_authenticated_principal(ctx)
    if auth_error:
        return auth_error
    effective_agent_id, live_agent_error = _resolve_live_content_agent_id(server, ctx, agent_id)
    if live_agent_error:
        return live_agent_error
    scope_error = _validate_explicit_scope(scope)
    if scope_error:
        return scope_error
    if scope not in VALID_SCOPES:
        return {"error": f"Unknown scope: {scope}", "code": "INVALID_SCOPE"}
    scope_error = _require_scope_membership(ctx, scope=scope, swarm_id=swarm_id)
    if scope_error:
        return scope_error
    created = _ensure_instance_config(server, ctx.owner_id)
    denied = _content_write_instance_denied(server, ctx, scope=scope, swarm_id=swarm_id)
    if denied:
        return denied
    owner_id, read_acl, write_acl = _resource_acl(ctx, scope=scope, swarm_id=swarm_id)
    if created:
        async with server._file_lock:
            server._save_cache()
    try:
        result = await server.ingest_document(
            content,
            source_id,
            agent_id=effective_agent_id,
            swarm_id=swarm_id,
            scope=scope,
            retention_ttl=retention_ttl,
            metadata=metadata,
            target=target,
            owner_id=owner_id,
            read=read_acl,
            write=write_acl,
            caller_id=ctx.owner_id,
            caller_principal_kind=ctx.principal_kind,
        )
    except ValueError as e:
        return {"error": str(e), "code": "VALIDATION_ERROR"}
    if not isinstance(result, dict):
        result = {"status": "ok", "facts_extracted": int(result or 0)}
    if _should_widen_instance_acl(result, scope=scope, swarm_id=swarm_id, write_kind="document") and _expand_instance_acl_for_scope(server, scope=scope, swarm_id=swarm_id, owner_id=ctx.owner_id):
        async with server._file_lock:
            server._save_cache()
    return result


@mcp.tool(name="memory_admin_backfill_original_raw_sources")
@_safe_tool
async def memory_admin_backfill_original_raw_sources(
    key: str,
    sources: list[dict],
    dry_run: bool = False,
    strict: bool = True,
    agent_id: str = "default",
    swarm_id: str = "default",
    token: str = None,
    agent_key: str = None,
) -> dict:
    """Admin-only source-level original raw backfill; never reruns extraction."""
    ctx = _resolve_identity(agent_id=agent_id, swarm_id=swarm_id, token=token, agent_key=agent_key)
    admin_error = _require_admin(ctx)
    if admin_error:
        return admin_error
    server = _get_memory(key)
    missing = _missing_instance_error(server)
    if missing:
        return missing
    if not isinstance(sources, list):
        return {"error": "sources must be a list", "code": "VALIDATION_ERROR"}
    result = server.backfill_original_raw_source_entries(
        sources,
        dry_run=bool(dry_run),
        strict=bool(strict),
        manifest_label="memory_admin_api",
    )
    return {
        "status": "ok",
        "backfill_path": "memory_admin_api",
        "no_new_extraction": True,
        "expected_answers_read": False,
        **result,
    }


@mcp.tool(name="memory_ingest")
@_safe_tool
async def memory_ingest(
    key: str,
    text: str = None,
    path: str = None,
    url: str = None,
    source_id: str = None,
    session_num: int = None,
    session_date: str = None,
    speakers: str = "User and Assistant",
    agent_id: str | None = None,
    swarm_id: str = "default",
    scope: str | None = None,
    retention_ttl: int = None,
    metadata: dict = None,
    target: str | list[str] | None = None,
    token: str = None,
    agent_key: str = None,
) -> dict:
    """Unified transport-only ingest. Exactly one of text/path/url."""
    server = _get_memory(key)
    ctx = _resolve_identity(agent_id=agent_id, swarm_id=swarm_id, token=token, agent_key=agent_key)
    auth_error = _require_authenticated_principal(ctx)
    if auth_error:
        return auth_error
    effective_agent_id, live_agent_error = _resolve_live_content_agent_id(server, ctx, agent_id)
    if live_agent_error:
        return live_agent_error
    scope_error = _validate_explicit_scope(scope)
    if scope_error:
        return scope_error
    if scope not in VALID_SCOPES:
        return {"error": f"Unknown scope: {scope}", "code": "INVALID_SCOPE"}
    scope_error = _require_scope_membership(ctx, scope=scope, swarm_id=swarm_id)
    if scope_error:
        return scope_error
    created = _ensure_instance_config(server, ctx.owner_id)
    denied = _content_write_instance_denied(server, ctx, scope=scope, swarm_id=swarm_id)
    if denied:
        return denied
    owner_id, read_acl, write_acl = _resource_acl(ctx, scope=scope, swarm_id=swarm_id)
    if created:
        async with server._file_lock:
            server._save_cache()

    from src.ingest import ingest_input

    try:
        result = await ingest_input(
            server,
            text=text,
            path=path,
            url=url,
            source_id=source_id,
            session_num=session_num,
            session_date=session_date,
            speakers=speakers,
            agent_id=effective_agent_id,
            swarm_id=swarm_id,
            scope=scope,
            retention_ttl=retention_ttl,
            metadata=metadata,
            target=target,
            owner_id=owner_id,
            read=read_acl,
            write=write_acl,
            caller_id=ctx.owner_id,
            caller_principal_kind=ctx.principal_kind,
        )
        if _should_widen_instance_acl(result, scope=scope, swarm_id=swarm_id, write_kind="store") and _expand_instance_acl_for_scope(server, scope=scope, swarm_id=swarm_id, owner_id=ctx.owner_id):
            async with server._file_lock:
                server._save_cache()
        return result
    except ValueError as e:
        return {"error": str(e), "code": "VALIDATION_ERROR"}


def _acl_defaults(owner_id: str, scope: str, swarm_id: str) -> tuple[list[str], list[str]]:
    """Derive default resource ACL from verified owner + explicit scope selector."""
    if scope == "swarm-shared" and swarm_id and swarm_id != "default":
        grant = f"swarm:{swarm_id}"
        return [grant], [grant]
    if scope == "system-wide":
        return ["agent:PUBLIC"], ["agent:PUBLIC"]
    return [], []


@mcp.tool(name="memory_ingest_asserted_facts")
@_safe_tool
async def memory_ingest_asserted_facts(
    key: str,
    facts: list[dict],
    consolidated: list[dict] = None,
    cross_session: list[dict] = None,
    raw_sessions: list[dict] = None,
    provenance: dict = None,
    agent_id: str | None = None,
    swarm_id: str = "default",
    scope: str | None = None,
    token: str = None,
    agent_key: str = None,
    enrich_l0: bool = True,
) -> dict:
    """Ingest pre-extracted facts. ACL derived from caller identity."""
    server = _get_memory(key)
    ctx = _resolve_identity(
        agent_id=agent_id, swarm_id=swarm_id,
        token=token, agent_key=agent_key)
    auth_error = _require_authenticated_principal(ctx)
    if auth_error:
        return auth_error
    effective_agent_id, live_agent_error = _resolve_live_content_agent_id(server, ctx, agent_id)
    if live_agent_error:
        return live_agent_error
    inferred_scope = None
    inferred_swarm = None
    for fact_group in (facts or [], consolidated or [], cross_session or []):
        for fact in fact_group:
            fact_scope = str(fact.get("scope") or "").strip()
            fact_swarm = str(fact.get("swarm_id") or "").strip()
            if fact_scope and inferred_scope is None:
                inferred_scope = fact_scope
            if fact_swarm and inferred_swarm is None:
                inferred_swarm = fact_swarm
            if fact_scope and inferred_scope not in (None, fact_scope):
                return {"error": "mixed asserted fact scopes require explicit split", "code": "VALIDATION_ERROR"}
            if fact_swarm and inferred_swarm not in (None, fact_swarm):
                return {"error": "mixed asserted fact swarm_ids require explicit split", "code": "VALIDATION_ERROR"}
    if provenance:
        prov_scope = str(provenance.get("scope") or "").strip()
        prov_swarm = str(provenance.get("swarm_id") or "").strip()
        if prov_scope:
            if inferred_scope not in (None, prov_scope):
                return {"error": "provenance.scope conflicts with fact scope", "code": "VALIDATION_ERROR"}
            inferred_scope = prov_scope
        if prov_swarm:
            if inferred_swarm not in (None, prov_swarm):
                return {"error": "provenance.swarm_id conflicts with fact swarm_id", "code": "VALIDATION_ERROR"}
            inferred_swarm = prov_swarm
    if scope and inferred_scope not in (None, scope):
        return {"error": "explicit scope conflicts with asserted fact scope", "code": "VALIDATION_ERROR"}
    if swarm_id and inferred_swarm not in (None, "", swarm_id):
        return {"error": "explicit swarm_id conflicts with asserted fact swarm_id", "code": "VALIDATION_ERROR"}
    effective_scope = scope or inferred_scope
    scope_error = _validate_explicit_scope(effective_scope)
    if scope_error:
        return scope_error
    if effective_scope not in VALID_SCOPES:
        return {"error": f"Unknown scope: {effective_scope}", "code": "INVALID_SCOPE"}
    effective_swarm = inferred_swarm or swarm_id
    created = _ensure_instance_config(server, ctx.owner_id)
    denied = _content_write_instance_denied(server, ctx, scope=effective_scope, swarm_id=effective_swarm)
    if denied:
        return denied
    if created:
        async with server._file_lock:
            server._save_cache()
    scope_error = _require_scope_membership(ctx, scope=effective_scope, swarm_id=effective_swarm)
    if scope_error:
        return scope_error
    _owner_id, _read, _write = _resource_acl(ctx, scope=effective_scope, swarm_id=effective_swarm)
    result = await server.ingest_asserted_facts(
        facts=facts, consolidated=consolidated,
        cross_session=cross_session, raw_sessions=raw_sessions,
        provenance=provenance,
        agent_id=effective_agent_id,
        swarm_id=effective_swarm,
        scope=effective_scope,
        owner_id=_owner_id,
        read=_read, write=_write,
        enrich_l0=enrich_l0,
        caller_id=ctx.owner_id,
        caller_principal_kind=ctx.principal_kind)
    changed = False
    for fact_group in (facts, consolidated or [], cross_session or []):
        for fact in fact_group:
            fact_scope = fact.get("scope") or effective_scope
            fact_swarm_id = fact.get("swarm_id") or effective_swarm
            if _expand_instance_acl_for_scope(server, scope=fact_scope, swarm_id=fact_swarm_id, owner_id=ctx.owner_id):
                changed = True
    if changed:
        async with server._file_lock:
            server._save_cache()
    return result


@mcp.tool(name="memory_build_index")
@_safe_tool
async def memory_build_index(
    key: str,
    agent_id: str = "default",
    token: str = None,
    agent_key: str = None,
) -> dict:
    """Build embedding index for all three tiers. Blocking."""
    server = _get_memory(key)
    ctx = _resolve_identity(agent_id=agent_id, token=token, agent_key=agent_key)
    auth_error = _require_authenticated_principal(ctx)
    if auth_error:
        return auth_error
    denied = _check_instance_acl(server, ctx.owner_id, "write", ctx.caller_role, ctx.memberships)
    if denied:
        return denied
    try:
        result = await server.build_index()
        # Add embedding config to result
        try:
            from .setup_store import get_config
            _cfg = get_config()
            result["embed_provider"] = _cfg.get("embed_provider", "openai")
            result["embed_model"] = _cfg.get("embed_model", "text-embedding-3-large")
        except Exception:
            result["embed_provider"] = "openai"
            result["embed_model"] = "text-embedding-3-large"
        return result
    except AssertionError as e:
        msg = str(e)
        code = "NO_FACTS" if "No granular facts" in msg else "INDEX_NOT_BUILT"
        return {"error": msg, "code": code}


@mcp.tool(name="memory_flush")
@_safe_tool
async def memory_flush(
    key: str,
    agent_id: str = "default",
    token: str = None,
    agent_key: str = None,
) -> dict:
    """Refresh persisted retrieval state using the current runtime path."""
    result = await memory_build_index(
        key=key,
        agent_id=agent_id,
        token=token,
        agent_key=agent_key,
    )
    if isinstance(result, dict) and "error" not in result:
        result = dict(result)
        result.setdefault("rebuilt", True)
        result.setdefault("total_consolidated", int(result.get("consolidated", 0) or 0))
        result.setdefault("total_cross_session", int(result.get("cross_session", 0) or 0))
    return result


@mcp.tool(name="memory_migrate_jsonnpz")
@_safe_tool
async def memory_migrate_jsonnpz(
    key: str,
    token: str = None,
) -> dict:
    """Explicitly migrate one legacy JSON/NPZ key into the persisted runtime backend."""
    ctx = _resolve_identity(token=token)
    admin_error = _require_admin(ctx)
    if admin_error:
        return admin_error
    result = migrate_jsonnpz_to_sqlite(data_dir, key)
    return {"status": "ok", **result}


@mcp.tool(name="memory_stats")
@_safe_tool
async def memory_stats(
    key: str,
    agent_id: str = "default",
    token: str = None,
    agent_key: str = None,
) -> dict:
    """Return memory stats for a key."""
    server = _get_memory(key)
    ctx = _resolve_identity(agent_id=agent_id, token=token, agent_key=agent_key)
    auth_error = _require_authenticated_principal(ctx)
    if auth_error:
        return auth_error
    denied = _check_instance_acl(server, ctx.owner_id, "read", ctx.caller_role, ctx.memberships, include_derived=True)
    if denied:
        return denied
    stats = server.stats()
    # Add embedding config to stats
    try:
        from .setup_store import get_config
        _cfg = get_config()
        stats["embed_provider"] = _cfg.get("embed_provider", "openai")
        stats["embed_model"] = _cfg.get("embed_model", "text-embedding-3-large")
    except Exception:
        stats["embed_provider"] = "openai"
        stats["embed_model"] = "text-embedding-3-large"
    return stats


@mcp.tool(name="memory_reextract")
@_safe_tool
async def memory_reextract(
    key: str,
    model: str = None,
    agent_id: str = "default",
    token: str = None,
    agent_key: str = None,
) -> dict:
    """Re-run Librarian extraction on stored raw sessions.

    Use when extraction prompt has been improved.
    Preserves raw sessions, replaces extracted facts.
    """
    server = _get_memory(key)
    ctx = _resolve_identity(agent_id=agent_id, token=token, agent_key=agent_key)
    auth_error = _require_authenticated_principal(ctx)
    if auth_error:
        return auth_error
    denied = _check_instance_acl(server, ctx.owner_id, "write", ctx.caller_role, ctx.memberships)
    if denied:
        return denied
    result = await server.reextract(model=model)
    if "error" in result:
        return {"error": result["error"], "code": "NO_RAW_SESSIONS"}
    return result


@mcp.tool(name="memory_list")
@_safe_tool
async def memory_list(
    key: str,
    agent_id: str = "default",
    swarm_id: str = "default",
    kind: str = None,
    limit: int = None,
    offset: int = 0,
    token: str = None,
    agent_key: str = None,
) -> dict:
    """List facts in memory, filtered by ACL and optional kind. Supports pagination."""
    server = _get_memory(key)
    ictx = _resolve_identity(agent_id=agent_id, swarm_id=swarm_id, token=token, agent_key=agent_key)
    auth_error = _require_authenticated_principal(ictx)
    if auth_error:
        return auth_error
    denied = _check_instance_acl(server, ictx.owner_id, "read", ictx.caller_role, ictx.memberships, include_derived=True)
    if denied:
        return denied
    server._audit.log("list", ictx.owner_id, {"kind": kind})
    all_facts = server._all_granular + server._all_cons + server._all_cross
    _memberships = _ctx_memberships(ictx)
    from datetime import datetime as _dt
    from datetime import timezone as _tz
    _now = _dt.now(_tz.utc)
    _fl = server._fact_lookup if hasattr(server, '_fact_lookup') else None
    visible = [f for f in all_facts
               if _is_visible(f, now=_now, fact_lookup=_fl) and server._acl_allows(f, ictx.owner_id, _memberships, ictx.caller_role)]

    if kind:
        visible = [f for f in visible if f.get("kind") == kind]

    total = len(visible)

    if offset:
        visible = visible[offset:]
    if limit is not None:
        visible = visible[:limit]

    return {"total": total, "facts": visible}


@mcp.tool(name="memory_get")
@_safe_tool
async def memory_get(
    key: str,
    fact_id: str,
    agent_id: str = "default",
    swarm_id: str = "default",
    token: str = None,
    agent_key: str = None,
) -> dict:
    """Get a specific fact by ID. Searches all three tiers."""
    server = _get_memory(key)
    ictx = _resolve_identity(agent_id=agent_id, swarm_id=swarm_id, token=token, agent_key=agent_key)
    auth_error = _require_authenticated_principal(ictx)
    if auth_error:
        return auth_error
    denied = _check_instance_acl(server, ictx.owner_id, "read", ictx.caller_role, ictx.memberships, include_derived=True)
    if denied:
        return denied
    server._audit.log("get", ictx.owner_id, {"fact_id": fact_id})
    all_facts = server._all_granular + server._all_cons + server._all_cross

    _memberships = _ctx_memberships(ictx)
    from datetime import datetime as _dt
    from datetime import timezone as _tz
    _now = _dt.now(_tz.utc)
    _fl = server._fact_lookup if hasattr(server, '_fact_lookup') else None
    for f in all_facts:
        if f.get("id") == fact_id:
            if not _is_visible(f, now=_now, fact_lookup=_fl):
                return {"code": "NOT_FOUND", "error": f"Fact {fact_id} not found"}
            if server._acl_allows(f, ictx.owner_id, _memberships, ictx.caller_role):
                return {"fact": f}
            else:
                return {"code": "ACL_FORBIDDEN", "error": "Access denied by ACL"}

    return {"code": "NOT_FOUND", "error": f"Fact {fact_id} not found"}


@mcp.tool(name="memory_edit")
@_safe_tool
async def memory_edit(
    key: str,
    artifact_id: str,
    new_content: str,
    agent_id: str = "default",
    swarm_id: str = "default",
    token: str = None,
    agent_key: str = None,
) -> dict:
    """Edit an artifact: create a new version with new content, supersede old."""
    server = _get_memory(key)
    ictx = _resolve_identity(agent_id=agent_id, swarm_id=swarm_id, token=token, agent_key=agent_key)
    auth_error = _require_authenticated_principal(ictx)
    if auth_error:
        return auth_error
    denied = _check_instance_acl(server, ictx.owner_id, "write", ictx.caller_role, ictx.memberships)
    if denied:
        return denied
    return await server.edit(artifact_id, new_content,
                             caller_id=ictx.owner_id, caller_role=ictx.caller_role,
                             caller_memberships=_ctx_memberships(ictx))


@mcp.tool(name="memory_retract")
@_safe_tool
async def memory_retract(
    key: str,
    artifact_id: str,
    agent_id: str = "default",
    swarm_id: str = "default",
    token: str = None,
    agent_key: str = None,
) -> dict:
    """Retract an artifact — makes all versions invisible."""
    server = _get_memory(key)
    ictx = _resolve_identity(agent_id=agent_id, swarm_id=swarm_id, token=token, agent_key=agent_key)
    auth_error = _require_authenticated_principal(ictx)
    if auth_error:
        return auth_error
    denied = _check_instance_acl(server, ictx.owner_id, "write", ictx.caller_role, ictx.memberships)
    if denied:
        return denied
    return await server.retract(artifact_id,
                                caller_id=ictx.owner_id, caller_role=ictx.caller_role,
                                caller_memberships=_ctx_memberships(ictx))


@mcp.tool(name="memory_query")
@_safe_tool
async def memory_query(
    key: str,
    filter: dict = None,
    sort_by: str = "session_date",
    sort_order: str = "desc",
    limit: int = 10,
    offset: int = 0,
    agent_id: str = "default",
    swarm_id: str = "default",
    token: str = None,
    agent_key: str = None,
) -> dict:
    """Structured query on facts. No vectors, no LLM.

    Filter by any field from fact schema or metadata.* fields:
    - Scalar fields (kind, owner_id, session, scope, ...): exact match
    - List fields (entities, tags, read, write, ...): contains match
    - metadata.* fields: exact match or range operators
    - Range: {"metadata.price": {"gte": 180, "lt": 200}}
    """
    server = _get_memory(key)
    ictx = _resolve_identity(
        agent_id=agent_id, swarm_id=swarm_id,
        token=token, agent_key=agent_key)
    auth_error = _require_authenticated_principal(ictx)
    if auth_error:
        return auth_error
    denied = _check_instance_acl(server, ictx.owner_id, "read", ictx.caller_role, ictx.memberships, include_derived=True)
    if denied:
        return denied
    _memberships = _ctx_memberships(ictx)
    return await server.query(
        filter=filter,
        sort_by=sort_by,
        sort_order=sort_order,
        limit=limit,
        offset=offset,
        caller_id=ictx.owner_id,
        caller_role=ictx.caller_role,
        caller_memberships=_memberships,
    )


@mcp.tool(name="memory_set_schema")
@_safe_tool
async def memory_set_schema(
    key: str, schema: dict,
    agent_id: str = "default", swarm_id: str = "default",
    token: str = None, agent_key: str = None,
) -> dict:
    """Declare metadata schema. Requires instance write access on an existing instance."""
    server = _get_memory(key)
    missing = _missing_instance_error(server)
    if missing:
        return missing
    ictx = _resolve_identity(
        agent_id=agent_id, swarm_id=swarm_id,
        token=token, agent_key=agent_key)
    auth_error = _require_authenticated_principal(ictx)
    if auth_error:
        return auth_error
    denied = _check_instance_acl(server, ictx.owner_id, "write", ictx.caller_role, ictx.memberships)
    if denied:
        return denied
    await server.set_metadata_schema(schema)
    return {"status": "ok", "fields": len(schema)}


@mcp.tool(name="memory_get_schema")
@_safe_tool
async def memory_get_schema(
    key: str,
    agent_id: str = "default", swarm_id: str = "default",
    token: str = None, agent_key: str = None,
) -> dict:
    """Get current metadata schema. Requires instance read access."""
    server = _get_memory(key)
    missing = _missing_instance_error(server)
    if missing:
        return missing
    ictx = _resolve_identity(
        agent_id=agent_id, swarm_id=swarm_id,
        token=token, agent_key=agent_key)
    auth_error = _require_authenticated_principal(ictx)
    if auth_error:
        return auth_error
    denied = _check_instance_acl(server, ictx.owner_id, "read", ictx.caller_role, ictx.memberships)
    if denied:
        return denied
    return {"schema": server.get_metadata_schema()}


@mcp.tool(name="memory_purge")
@_safe_tool
async def memory_purge(
    key: str,
    artifact_id: str,
    agent_id: str = "default",
    swarm_id: str = "default",
    token: str = None,
    agent_key: str = None,
) -> dict:
    """Purge an artifact — requires instance write access."""
    server = _get_memory(key)
    ictx = _resolve_identity(agent_id=agent_id, swarm_id=swarm_id, token=token, agent_key=agent_key)
    auth_error = _require_authenticated_principal(ictx)
    if auth_error:
        return auth_error
    denied = _check_instance_acl(server, ictx.owner_id, "write", ictx.caller_role, ictx.memberships)
    if denied:
        return denied
    return await server.purge(artifact_id,
                              caller_id=ictx.owner_id, caller_role=ictx.caller_role)


@mcp.tool(name="memory_redact")
@_safe_tool
async def memory_redact(
    key: str,
    artifact_id: str,
    fields: list[str] = None,
    agent_id: str = "default",
    swarm_id: str = "default",
    token: str = None,
    agent_key: str = None,
) -> dict:
    """Redact fields of an artifact — requires instance write access."""
    server = _get_memory(key)
    ictx = _resolve_identity(agent_id=agent_id, swarm_id=swarm_id, token=token, agent_key=agent_key)
    auth_error = _require_authenticated_principal(ictx)
    if auth_error:
        return auth_error
    denied = _check_instance_acl(server, ictx.owner_id, "write", ictx.caller_role, ictx.memberships)
    if denied:
        return denied
    return await server.redact(artifact_id, fields or ["fact", "entities", "content"],
                               caller_id=ictx.owner_id, caller_role=ictx.caller_role,
                               caller_memberships=_ctx_memberships(ictx))


@mcp.tool(name="memory_get_versions")
@_safe_tool
async def memory_get_versions(
    key: str,
    artifact_id: str,
    agent_id: str = "default",
    swarm_id: str = "default",
    token: str = None,
    agent_key: str = None,
) -> dict:
    """Get version chain for an artifact."""
    server = _get_memory(key)
    ictx = _resolve_identity(agent_id=agent_id, swarm_id=swarm_id, token=token, agent_key=agent_key)
    auth_error = _require_authenticated_principal(ictx)
    if auth_error:
        return auth_error
    denied = _check_instance_acl(server, ictx.owner_id, "read", ictx.caller_role, ictx.memberships)
    if denied:
        return denied
    return server.get_versions(artifact_id,
                               caller_id=ictx.owner_id, caller_role=ictx.caller_role,
                               caller_memberships=_ctx_memberships(ictx))


@mcp.tool(name="courier_subscribe")
@_safe_tool
async def courier_subscribe(
    key: str,
    connection_id: str = "",
    deliver_existing: bool = False,
    filter: dict = None,
    agent_id: str = "default",
    swarm_id: str = "default",
    token: str = None,
    agent_key: str = None,
) -> dict:
    """Subscribe to new facts matching filter. Push via SSE stream."""
    # C2: verify connection_id is a known active SSE connection
    if connection_id and connection_id not in _active_connections:
        return {"error": "Unknown connection_id — connect via /mcp/sse first",
                "code": "INVALID_CONNECTION"}

    courier = _get_courier(key)

    # Pre-generate sub_id and wire routing BEFORE subscribe()
    # so deliver_existing can route events through the SSE queue
    from uuid import uuid4 as _uuid4
    pre_sub_id = f"sub_{_uuid4().hex[:8]}"
    sub_to_conn[pre_sub_id] = connection_id

    async def _push(fact: dict):
        cid = sub_to_conn.get(pre_sub_id)
        if not cid or cid not in connections:
            return
        live_ctx = _resolve_identity(token=token)
        if not live_ctx.authenticated:
            await courier.unsubscribe(pre_sub_id)
            sub_to_conn.pop(pre_sub_id, None)
            return
        denied = _check_instance_acl(
            server,
            live_ctx.owner_id,
            "read",
            live_ctx.caller_role,
            live_ctx.memberships,
            include_derived=True,
        )
        if denied:
            return
        live_memberships = _ctx_memberships(live_ctx)
        if not server._acl_allows(fact, live_ctx.owner_id, live_memberships, live_ctx.caller_role):
            return
        await connections[cid].put({
            "type": "artifact",
            "sub_id": pre_sub_id,
            "payload": fact,
        })

    # Resolve subscriber identity for ACL
    ictx = _resolve_identity(agent_id=agent_id, swarm_id=swarm_id, token=token, agent_key=agent_key)
    auth_error = _require_authenticated_principal(ictx)
    if auth_error:
        return auth_error
    server = _get_memory(key)
    denied = _check_instance_acl(server, ictx.owner_id, "read", ictx.caller_role, ictx.memberships, include_derived=True)
    if denied:
        return denied
    _memberships = _ctx_memberships(ictx)

    sub_id = await courier.subscribe(
        filter=filter or {},
        callback=_push,
        deliver_existing=deliver_existing,
        owner_id=ictx.owner_id,
        memberships=_memberships,
        pre_sub_id=pre_sub_id,
        caller_role=ictx.caller_role,
    )
    return {"sub_id": sub_id}


@mcp.tool(name="courier_unsubscribe")
@_safe_tool
async def courier_unsubscribe(sub_id: str) -> dict:
    """Unsubscribe from Courier push. Idempotent."""
    for courier in courier_registry.values():
        await courier.unsubscribe(sub_id)
    sub_to_conn.pop(sub_id, None)
    return {"status": "ok"}


@mcp.tool(name="auth_bootstrap_admin")
@_safe_tool
async def auth_bootstrap_admin(
    principal_id: str,
    kind: str = "service",
    display_name: str | None = None,
    description: str | None = None,
    expires_at: str | None = None,
    metadata: dict | None = None,
    token: str = None,
) -> dict:
    """Bootstrap the one-time first persisted admin principal using the env bootstrap token."""
    ctx = _resolve_identity(token=token)
    bootstrap_error = _require_bootstrap_admin(ctx)
    if bootstrap_error:
        return bootstrap_error
    try:
        result = _get_authority().bootstrap_admin(
            principal_id=principal_id,
            kind=kind,
            display_name=display_name,
            description=description,
            expires_at=expires_at,
            metadata=metadata,
        )
    except AuthorityError as exc:
        return _authority_error(exc)
    return {"status": "ok", **result}


@mcp.tool(name="principal_create")
@_safe_tool
async def principal_create(
    principal_id: str,
    kind: str,
    display_name: str | None = None,
    metadata: dict | None = None,
    token: str = None,
) -> dict:
    ctx = _resolve_identity(token=token)
    try:
        principal = _get_authority().create_principal(
            actor=_ctx_actor(ctx),
            principal_id=principal_id,
            kind=kind,
            display_name=display_name,
            metadata=metadata,
        )
    except AuthorityError as exc:
        return _authority_error(exc)
    return {"status": "ok", "principal": principal}


@mcp.tool(name="principal_get")
@_safe_tool
async def principal_get(
    principal_id: str | None = None,
    token: str = None,
) -> dict:
    ctx = _resolve_identity(token=token)
    try:
        principal = _get_authority().get_principal(
            actor=_ctx_actor(ctx),
            principal_id=principal_id,
        )
    except AuthorityError as exc:
        return _authority_error(exc)
    return {"principal": principal}


@mcp.tool(name="principal_disable")
@_safe_tool
async def principal_disable(principal_id: str, token: str = None) -> dict:
    ctx = _resolve_identity(token=token)
    try:
        principal = _get_authority().disable_principal(
            actor=_ctx_actor(ctx),
            principal_id=principal_id,
        )
    except AuthorityError as exc:
        return _authority_error(exc)
    return {"status": "ok", "principal": principal}


@mcp.tool(name="auth_token_issue")
@_safe_tool
async def auth_token_issue(
    principal_id: str,
    token_kind: str = "user",  # noqa: S107 - token kind enum default, not a credential
    description: str | None = None,
    expires_at: str | None = None,
    metadata: dict | None = None,
    token: str = None,
) -> dict:
    ctx = _resolve_identity(token=token)
    try:
        issued = _get_authority().issue_token(
            actor=_ctx_actor(ctx),
            principal_id=principal_id,
            token_kind=token_kind,
            description=description,
            expires_at=expires_at,
            metadata=metadata,
        )
    except AuthorityError as exc:
        return _authority_error(exc)
    return {"status": "ok", **issued}


@mcp.tool(name="auth_token_revoke")
@_safe_tool
async def auth_token_revoke(token_id: str, token: str = None) -> dict:
    ctx = _resolve_identity(token=token)
    try:
        revoked = _get_authority().revoke_token(
            actor=_ctx_actor(ctx),
            token_id=token_id,
        )
    except AuthorityError as exc:
        return _authority_error(exc)
    return {"status": "ok", "token": revoked}


@mcp.tool(name="auth_token_list")
@_safe_tool
async def auth_token_list(principal_id: str | None = None, token: str = None) -> dict:
    ctx = _resolve_identity(token=token)
    try:
        return _get_authority().list_tokens(
            actor=_ctx_actor(ctx),
            principal_id=principal_id,
        )
    except AuthorityError as exc:
        return _authority_error(exc)


@mcp.tool(name="swarm_create")
@_safe_tool
async def swarm_create(
    swarm_id: str,
    owner_principal_id: str,
    display_name: str | None = None,
    metadata: dict | None = None,
    token: str = None,
) -> dict:
    ctx = _resolve_identity(token=token)
    try:
        swarm = _get_authority().create_swarm(
            actor=_ctx_actor(ctx),
            swarm_id=swarm_id,
            owner_principal_id=owner_principal_id,
            display_name=display_name,
            metadata=metadata,
        )
    except AuthorityError as exc:
        return _authority_error(exc)
    return {"status": "ok", "swarm": swarm}


@mcp.tool(name="swarm_get")
@_safe_tool
async def swarm_get(swarm_id: str, token: str = None) -> dict:
    ctx = _resolve_identity(token=token)
    try:
        swarm = _get_authority().get_swarm(
            actor=_ctx_actor(ctx),
            swarm_id=swarm_id,
        )
    except AuthorityError as exc:
        return _authority_error(exc)
    return {"swarm": swarm}


@mcp.tool(name="swarm_list")
@_safe_tool
async def swarm_list(token: str = None) -> dict:
    ctx = _resolve_identity(token=token)
    try:
        return _get_authority().list_swarms(actor=_ctx_actor(ctx))
    except AuthorityError as exc:
        return _authority_error(exc)


@mcp.tool(name="membership_grant")
@_safe_tool
async def membership_grant(
    swarm_id: str,
    principal_id: str,
    role: str = "member",
    expires_at: str | None = None,
    metadata: dict | None = None,
    token: str = None,
) -> dict:
    ctx = _resolve_identity(token=token)
    try:
        membership = _get_authority().grant_membership(
            actor=_ctx_actor(ctx),
            swarm_id=swarm_id,
            principal_id=principal_id,
            role=role,
            expires_at=expires_at,
            metadata=metadata,
        )
    except AuthorityError as exc:
        return _authority_error(exc)
    return {"status": "ok", "membership": membership}


@mcp.tool(name="membership_register")
@_safe_tool
async def membership_register(
    identity: str,
    group: str,
    key: str = "default",
    token: str = None,
) -> dict:
    del key
    if not str(group).startswith("swarm:"):
        return {"error": "membership_register only accepts swarm:* groups", "code": "VALIDATION_ERROR"}
    return await membership_grant(
        swarm_id=str(group).split(":", 1)[1],
        principal_id=identity,
        token=token,
    )


@mcp.tool(name="membership_revoke")
@_safe_tool
async def membership_revoke(
    swarm_id: str,
    principal_id: str,
    token: str = None,
) -> dict:
    ctx = _resolve_identity(token=token)
    try:
        membership = _get_authority().revoke_membership(
            actor=_ctx_actor(ctx),
            swarm_id=swarm_id,
            principal_id=principal_id,
        )
    except AuthorityError as exc:
        return _authority_error(exc)
    return {"status": "ok", "membership": membership}


@mcp.tool(name="membership_unregister")
@_safe_tool
async def membership_unregister(
    identity: str,
    group: str,
    key: str = "default",
    token: str = None,
) -> dict:
    del key
    if not str(group).startswith("swarm:"):
        return {"error": "membership_unregister only accepts swarm:* groups", "code": "VALIDATION_ERROR"}
    return await membership_revoke(
        swarm_id=str(group).split(":", 1)[1],
        principal_id=identity,
        token=token,
    )


@mcp.tool(name="membership_list")
@_safe_tool
async def membership_list(
    swarm_id: str | None = None,
    principal_id: str | None = None,
    include_revoked: bool = False,
    token: str = None,
) -> dict:
    ctx = _resolve_identity(token=token)
    try:
        return _get_authority().list_memberships(
            actor=_ctx_actor(ctx),
            swarm_id=swarm_id,
            principal_id=principal_id,
            include_revoked=include_revoked,
        )
    except AuthorityError as exc:
        return _authority_error(exc)


@mcp.tool(name="memory_store_secret")
@_safe_tool
async def memory_store_secret(
    key: str,
    name: str,
    value: str,
    agent_id: str = "default",
    swarm_id: str = "default",
    scope: str | None = None,
    token: str = None,
    agent_key: str = None,
) -> dict:
    """Create a secret exactly once in the dedicated secret store."""
    server = _get_memory(key)
    ctx = _resolve_identity(agent_id=agent_id, swarm_id=swarm_id, token=token, agent_key=agent_key)
    auth_error = _require_authenticated_principal(ctx)
    if auth_error:
        return auth_error
    scope_error = _validate_explicit_scope(scope)
    if scope_error:
        return scope_error
    if scope not in VALID_SCOPES:
        return {"error": f"Unknown scope: {scope}", "code": "INVALID_SCOPE"}
    missing = _missing_instance_error(server)
    if missing:
        return missing
    denied = _check_instance_acl(server, ctx.owner_id, "write", ctx.caller_role, ctx.memberships)
    if denied:
        return denied
    result = server.store_secret(
        name,
        value,
        agent_id=agent_id,
        swarm_id=swarm_id,
        scope=scope,
        caller_id=ctx.owner_id,
        caller_memberships=_ctx_memberships(ctx),
        caller_role=ctx.caller_role,
    )
    return result

@mcp.tool(name="memory_list_secrets")
@_safe_tool
async def memory_list_secrets(
    key: str,
    agent_id: str = "default",
    swarm_id: str = "default",
    scope: str | None = None,
    token: str = None,
    agent_key: str = None,
) -> dict:
    """List visible secret metadata using canonical row ACL filters."""
    server = _get_memory(key)
    ctx = _resolve_identity(agent_id=agent_id, swarm_id=swarm_id, token=token, agent_key=agent_key)
    auth_error = _require_authenticated_principal(ctx)
    if auth_error:
        return auth_error
    scope_error = _validate_explicit_scope(scope)
    if scope_error:
        return scope_error
    if scope not in VALID_SCOPES:
        return {"error": f"Unknown scope: {scope}", "code": "INVALID_SCOPE"}
    return server.list_secrets(
        agent_id=agent_id,
        swarm_id=swarm_id,
        scope=scope,
        caller_id=ctx.owner_id,
        caller_memberships=_ctx_memberships(ctx),
        caller_role=ctx.caller_role,
    )


@mcp.tool(name="memory_delete_secret")
@_safe_tool
async def memory_delete_secret(
    key: str,
    name: str,
    agent_id: str = "default",
    swarm_id: str = "default",
    scope: str | None = None,
    token: str = None,
    agent_key: str = None,
) -> dict:
    """Delete a secret by exact name in an explicit canonical ACL domain."""
    server = _get_memory(key)
    ctx = _resolve_identity(agent_id=agent_id, swarm_id=swarm_id, token=token, agent_key=agent_key)
    auth_error = _require_authenticated_principal(ctx)
    if auth_error:
        return auth_error
    scope_error = _validate_explicit_scope(scope)
    if scope_error:
        return scope_error
    if scope not in VALID_SCOPES:
        return {"error": f"Unknown scope: {scope}", "code": "INVALID_SCOPE"}
    return server.delete_secret(
        name,
        agent_id=agent_id,
        swarm_id=swarm_id,
        scope=scope,
        caller_id=ctx.owner_id,
        caller_memberships=_ctx_memberships(ctx),
        caller_role=ctx.caller_role,
    )


@mcp.tool(name="memory_import")
@_safe_tool
async def memory_import(
    key: str,
    source_format: str,
    content: str = None,
    path: str = None,
    source_uri: str = None,
    token: str = None,
    options: str = None,
    content_type: str = "default",
    agent_id: str | None = None,
    swarm_id: str = "default",
    scope: str | None = None,
    agent_key: str = None,
    auth_token: str = None,
) -> dict:
    """Import data into memory. Same capabilities as the CLI.

    token: source/repo auth for git clone only.
    auth_token: verified caller principal token.
    """
    ALL_FORMATS = {"conversation_json", "text", "directory", "git"}

    server = _get_memory(key)
    ctx = _resolve_identity(
        agent_id=agent_id,
        swarm_id=swarm_id,
        token=auth_token,
        agent_key=agent_key,
    )
    auth_error = _require_authenticated_principal(ctx)
    if auth_error:
        return auth_error
    effective_agent_id, live_agent_error = _resolve_live_content_agent_id(server, ctx, agent_id)
    if live_agent_error:
        return live_agent_error
    scope_error = _validate_explicit_scope(scope)
    if scope_error:
        return scope_error
    if scope not in VALID_SCOPES:
        return {"error": f"Unknown scope: {scope}", "code": "INVALID_SCOPE"}
    scope_error = _require_scope_membership(ctx, scope=scope, swarm_id=swarm_id)
    if scope_error:
        return scope_error
    created = _ensure_instance_config(server, ctx.owner_id)
    denied = _content_write_instance_denied(server, ctx, scope=scope, swarm_id=swarm_id)
    if denied:
        return denied
    if created:
        async with server._file_lock:
            server._save_cache()
    owner_id, read_acl, write_acl = _resource_acl(ctx, scope=scope, swarm_id=swarm_id)

    if source_format not in ALL_FORMATS:
        return {"error": f"Unknown format: {source_format!r}. Supported: {sorted(ALL_FORMATS)}",
                "code": "UNKNOWN_FORMAT"}

    try:
        opts = json.loads(options) if options else {}
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON in options: {e}", "code": "INVALID_OPTIONS"}

    if source_format == "git":
        if not source_uri:
            return {"error": "source_uri required for git format", "code": "MISSING_PARAM"}
        from .git_importer import import_git
        if token:
            opts["token"] = token
        opts.setdefault("content_type", content_type)
        try:
            sessions = import_git(source_uri, opts)
        except Exception as e:
            return {"error": str(e), "code": "GIT_ERROR"}
    elif source_format == "directory":
        if not path:
            return {"error": "path required for directory format", "code": "MISSING_PARAM"}
        from pathlib import Path as P
        dir_path = P(path)
        if not dir_path.exists():
            return {"error": f"Path not found: {path}", "code": "PATH_NOT_FOUND"}
        if not dir_path.is_dir():
            return {"error": f"Not a directory: {path}", "code": "NOT_A_DIRECTORY"}
        importable_suffixes = {
            ".txt", ".md", ".json", ".py", ".rst", ".yaml", ".yml",
            ".csv", ".xml", ".html", ".log", ".cfg", ".ini", ".toml",
            ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".c",
            ".cpp", ".h", ".hpp", ".rb", ".sh", ".sql", ".proto",
        }
        parts = []
        for f in sorted(dir_path.rglob("*")):
            if f.is_file() and f.suffix in importable_suffixes:
                try:
                    rel = f.relative_to(dir_path).as_posix()
                except ValueError:
                    rel = f.name
                parts.append(f"---FILE: {rel}---")
                parts.append(f.read_text(encoding="utf-8", errors="replace"))
        from .importers import parse_history
        try:
            sessions = parse_history("directory", "\n".join(parts))
        except Exception as e:
            return {"error": str(e), "code": "PARSE_ERROR"}
    else:
        if not content:
            return {"error": "content required for this format", "code": "MISSING_PARAM"}
        from .importers import parse_history
        try:
            sessions = parse_history(source_format, content)
        except Exception as e:
            return {"error": str(e), "code": "PARSE_ERROR"}

    if not sessions:
        return {"error": "No sessions parsed from input", "code": "EMPTY_INPUT"}

    total_facts = 0
    errors = []
    skipped = 0
    for session in sessions:
        if source_format == "git" and session.get("source_id") and session.get("artifact_path"):
            dedup_key = (session["source_id"], session["artifact_path"])
            existing = server._git_dedup_index.get(dedup_key)
            if existing and existing.get("blob_sha") == session.get("blob_sha"):
                skipped += 1
                continue
        try:
            from .identity import _generate_artifact_id, _generate_version_id, content_hash_text
            store_kwargs = {
                "content": session["content"],
                "session_num": session["session_num"],
                "session_date": session["session_date"],
                "speakers": session.get("speakers", "User and Assistant"),
                "agent_id": effective_agent_id,
                "swarm_id": swarm_id,
                "scope": scope,
                "owner_id": owner_id,
                "read": read_acl,
                "write": write_acl,
                "content_type": session.get("content_type", content_type),
                "caller_id": ctx.owner_id,
                "caller_principal_kind": ctx.principal_kind,
            }
            if source_format == "git" and session.get("artifact_path"):
                dedup_key = (session.get("source_id", ""), session["artifact_path"])
                existing = server._git_dedup_index.get(dedup_key)
                if existing:
                    art_id = existing.get("artifact_id", _generate_artifact_id())
                    ver_id = _generate_version_id()
                    parent_ver = existing.get("version_id")
                else:
                    art_id = _generate_artifact_id()
                    ver_id = _generate_version_id()
                    parent_ver = None
                from .identity import content_hash_git
                blob = session.get("blob_sha")
                ch = content_hash_git(blob) if blob else content_hash_text(session["content"])
                store_kwargs["source_id"] = session.get("source_id")
                store_kwargs["artifact_id"] = art_id
                store_kwargs["version_id"] = ver_id
                store_kwargs["parent_version"] = parent_ver
                store_kwargs["content_hash"] = ch
                store_kwargs["skip_dedup"] = True
                store_kwargs["source_meta"] = {
                    "artifact_path": session["artifact_path"],
                    "blob_sha": session.get("blob_sha"),
                    "storage_mode": session.get("storage_mode", "inline"),
                }
            result = await server.store(**store_kwargs)
            total_facts += result.get("facts_extracted", 0)
            if source_format == "git" and session.get("source_id") and session.get("artifact_path"):
                dedup_key = (session["source_id"], session["artifact_path"])
                server._git_dedup_index[dedup_key] = {
                    "blob_sha": session.get("blob_sha"),
                    "artifact_id": store_kwargs.get("artifact_id", result.get("artifact_id", "")),
                    "version_id": store_kwargs.get("version_id", ""),
                }
                server._save_cache()
        except Exception as e:
            errors.append({"session": session["session_num"], "error": str(e)})

    sessions_processed = len(sessions) - len(errors) - skipped
    resp: dict[str, Any] = {
        "sessions_processed": sessions_processed,
        "total_sessions": len(sessions),
        "facts_extracted": total_facts,
        "errors": errors,
    }
    if skipped:
        resp["skipped_unchanged"] = skipped
    if sessions_processed > 0 and _expand_instance_acl_for_scope(server, scope=scope, swarm_id=swarm_id, owner_id=ctx.owner_id):
        async with server._file_lock:
            server._save_cache()
    return resp


# Backward compat alias
@mcp.tool(name="memory_import_history")
@_safe_tool
async def memory_import_history(
    key: str,
    source_format: str,
    content: str,
    agent_id: str | None = None,
    swarm_id: str = "default",
    scope: str | None = None,
    auth_token: str = None,
    agent_key: str = None,
) -> dict:
    return await memory_import(
        key=key,
        source_format=source_format,
        content=content,
        agent_id=agent_id,
        swarm_id=swarm_id,
        scope=scope,
        auth_token=auth_token,
        agent_key=agent_key,
    )


@mcp.tool(name="memory_list_prompts")
@_safe_tool
async def memory_list_prompts(
    key: str,
    agent_id: str = "default",
    token: str = None,
    agent_key: str = None,
) -> dict:
    server = _get_memory(key)
    ctx = _resolve_identity(agent_id=agent_id, token=token, agent_key=agent_key)
    auth_error = _require_authenticated_principal(ctx)
    if auth_error:
        return auth_error
    denied = _check_instance_acl(server, ctx.owner_id, "read", ctx.caller_role, ctx.memberships)
    if denied:
        return denied
    return {"prompts": server._prompt_registry.list()}


@mcp.tool(name="memory_get_prompt")
@_safe_tool
async def memory_get_prompt(
    key: str,
    content_type: str,
    agent_id: str = "default",
    token: str = None,
    agent_key: str = None,
) -> dict:
    server = _get_memory(key)
    ctx = _resolve_identity(agent_id=agent_id, token=token, agent_key=agent_key)
    auth_error = _require_authenticated_principal(ctx)
    if auth_error:
        return auth_error
    denied = _check_instance_acl(server, ctx.owner_id, "read", ctx.caller_role, ctx.memberships)
    if denied:
        return denied
    if not server._prompt_registry.exists(content_type):
        return {"error": f"prompt not found: {content_type!r}", "code": "PROMPT_NOT_FOUND"}
    prompt = server._prompt_registry.get(content_type)
    custom_path = server._prompt_registry._custom_path(content_type)
    source = "custom" if custom_path.exists() else "builtin"
    return {"content_type": content_type, "prompt": prompt, "source": source}


@mcp.tool(name="memory_set_prompt")
@_safe_tool
async def memory_set_prompt(
    key: str,
    content_type: str,
    prompt: str,
    agent_id: str = "default",
    token: str = None,
    agent_key: str = None,
) -> dict:
    if not content_type or not content_type.replace("_", "").isalnum():
        return {"error": "content_type must be alphanumeric with underscores", "code": "INVALID_CONTENT_TYPE"}
    server = _get_memory(key)
    missing = _missing_instance_error(server)
    if missing:
        return missing
    ctx = _resolve_identity(agent_id=agent_id, token=token, agent_key=agent_key)
    auth_error = _require_authenticated_principal(ctx)
    if auth_error:
        return auth_error
    denied = _check_instance_acl(server, ctx.owner_id, "write", ctx.caller_role, ctx.memberships)
    if denied:
        return denied
    server._prompt_registry.set(content_type, prompt)
    return {"stored": True, "content_type": content_type}

# ── MAL tools ──

_mal_stores: dict[tuple[str, str], dict] = {}


def _get_mal_stores(data_dir_path: str, server=None) -> dict:
    key = server.key if server else ""
    cache_key = (data_dir_path, key)
    if cache_key not in _mal_stores:
        from src.mal.artifact_store import ArtifactStore
        from src.mal.control_store import ControlStore
        from src.mal.feedback_store import FeedbackStore
        from src.mal.scheduler import Scheduler
        control = ControlStore(data_dir_path)
        feedback = FeedbackStore(data_dir_path, control)
        artifacts = ArtifactStore(data_dir_path)
        _mal_stores[cache_key] = {
            "control": control,
            "feedback": feedback,
            "artifacts": artifacts,
            "scheduler": Scheduler(data_dir_path, control, feedback,
                                   artifacts=artifacts, server=server),
        }
    elif server is not None:
        _mal_stores[cache_key]["scheduler"]._server = server
    return _mal_stores[cache_key]


@mcp.tool(name="memory_mal_configure")
@_safe_tool
async def memory_mal_configure(
    key: str,
    agent_id: str = "default",
    enabled: bool = False,
    auto_collect_feedback: bool = False,
    auto_trigger: bool = False,
    min_signals: int = None,
    token: str = None,
    agent_key: str = None,
) -> dict:
    """Enable/disable MAL and set control flags for a binding. Requires write ACL.

    min_signals: minimum independent failure signals before MAL accepts
    any pipeline change (default 10). Lower = more aggressive adaptation.
    """
    server = _get_memory(key)
    ictx = _resolve_identity(agent_id=agent_id, swarm_id="default", token=token, agent_key=agent_key)
    auth_error = _require_authenticated_principal(ictx)
    if auth_error:
        return auth_error
    denied = _check_instance_acl(server, ictx.owner_id, "write", ictx.caller_role, ictx.memberships)
    if denied:
        return denied
    stores = _get_mal_stores(str(server.data_dir), server=server)
    binding_id = _resolve_mal_binding_id(ictx, agent_id)
    fields: dict[str, Any] = {
        "enabled": enabled,
        "auto_collect_feedback": auto_collect_feedback,
        "auto_trigger": auto_trigger,
    }
    if min_signals is not None:
        fields["min_signals"] = max(2, int(min_signals))
    stores["control"].set(key, binding_id, **fields)
    return {
        "status": "ok",
        "key": key,
        "agent_id": binding_id,
        "requested_agent_id": agent_id,
        "config": stores["control"].get(key, binding_id),
    }


@mcp.tool(name="memory_mal_feedback")
@_safe_tool
async def memory_mal_feedback(
    key: str,
    verdict: str,
    query: str,
    agent_id: str = "default",
    signal_source: str = "user",
    runtime_trace_ref: str = None,
    runtime_trace: dict = None,
    response_excerpt: str = None,
    corrected_answer: str = None,
    retry_chain_id: str = None,
    source_ids_hint: list[str] = None,
    token: str = None,
    agent_key: str = None,
) -> dict:
    """Submit a single feedback event to the MAL queue. Requires write ACL.

    runtime_trace_ref: stable ID from ask() result for linking
    runtime_trace: full trace payload from ask() for diagnosis (required for trigger)
    """
    server = _get_memory(key)
    ictx = _resolve_identity(agent_id=agent_id, swarm_id="default", token=token, agent_key=agent_key)
    auth_error = _require_authenticated_principal(ictx)
    if auth_error:
        return auth_error
    denied = _check_instance_acl(server, ictx.owner_id, "write", ictx.caller_role, ictx.memberships)
    if denied:
        return denied
    stores = _get_mal_stores(str(server.data_dir), server=server)
    binding_id = _resolve_mal_binding_id(ictx, agent_id)
    event = {
        "signal_source": signal_source,
        "verdict": verdict,
        "query": query,
        "runtime_trace_ref": runtime_trace_ref,
        "runtime_trace": runtime_trace,
        "response_excerpt": response_excerpt,
        "corrected_answer": corrected_answer,
        "retry_chain_id": retry_chain_id,
        "source_ids_hint": source_ids_hint,
    }
    event_id = stores["feedback"].submit(key, binding_id, event)
    return {"status": "ok", "feedback_event_id": event_id, "agent_id": binding_id}


@mcp.tool(name="memory_mal_trigger")
@_safe_tool
async def memory_mal_trigger(
    key: str,
    agent_id: str = "default",
    feedback_event_ids: list[str] = None,
    estimate_only: bool = False,
    force: bool = False,
    token: str = None,
    agent_key: str = None,
) -> dict:
    """Trigger a MAL adaptation run. Requires write ACL."""
    server = _get_memory(key)
    ictx = _resolve_identity(agent_id=agent_id, swarm_id="default", token=token, agent_key=agent_key)
    auth_error = _require_authenticated_principal(ictx)
    if auth_error:
        return auth_error
    denied = _check_instance_acl(server, ictx.owner_id, "write", ictx.caller_role, ictx.memberships)
    if denied:
        return denied
    stores = _get_mal_stores(str(server.data_dir), server=server)
    binding_id = _resolve_mal_binding_id(ictx, agent_id)
    return stores["scheduler"].trigger(
        key, agent_id=binding_id,
        feedback_event_ids=feedback_event_ids,
        estimate_only=estimate_only,
        force=force,
    )


@mcp.tool(name="memory_mal_status")
@_safe_tool
async def memory_mal_status(
    key: str,
    agent_id: str = "default",
    token: str = None,
    agent_key: str = None,
) -> dict:
    """Report MAL binding state, queued feedback count, and convergence. Requires read ACL."""
    server = _get_memory(key)
    ictx = _resolve_identity(agent_id=agent_id, swarm_id="default", token=token, agent_key=agent_key)
    auth_error = _require_authenticated_principal(ictx)
    if auth_error:
        return auth_error
    denied = _check_instance_acl(server, ictx.owner_id, "read", ictx.caller_role, ictx.memberships)
    if denied:
        return denied
    stores = _get_mal_stores(str(server.data_dir), server=server)
    binding_id = _resolve_mal_binding_id(ictx, agent_id)
    control = stores["control"].get(key, binding_id)
    queued = stores["feedback"].list_queued(key, binding_id)
    convergence = stores["scheduler"].get_convergence_state(key, binding_id)
    latest = stores["artifacts"].get_latest(key, binding_id)
    return {
        "key": key,
        "agent_id": binding_id,
        "requested_agent_id": agent_id,
        "enabled": control.get("enabled", False),
        "auto_collect_feedback": control.get("auto_collect_feedback", False),
        "auto_trigger": control.get("auto_trigger", False),
        "queued_feedback_count": len(queued),
        "convergence_state": convergence.get("convergence_state", "active"),
        "rejected_streak": convergence.get("rejected_streak", 0),
        "latest_artifact_id": latest["artifact_id"] if latest else None,
        "latest_artifact_version": latest["version"] if latest else None,
    }


@mcp.tool(name="memory_mal_list_feedback")
@_safe_tool
async def memory_mal_list_feedback(
    key: str,
    agent_id: str = "default",
    status_filter: str = "queued",
    token: str = None,
    agent_key: str = None,
) -> dict:
    """List feedback events in the MAL queue. Requires read ACL."""
    server = _get_memory(key)
    ictx = _resolve_identity(agent_id=agent_id, swarm_id="default", token=token, agent_key=agent_key)
    auth_error = _require_authenticated_principal(ictx)
    if auth_error:
        return auth_error
    denied = _check_instance_acl(server, ictx.owner_id, "read", ictx.caller_role, ictx.memberships)
    if denied:
        return denied
    stores = _get_mal_stores(str(server.data_dir), server=server)
    binding_id = _resolve_mal_binding_id(ictx, agent_id)
    if status_filter == "queued":
        events = stores["feedback"].list_queued(key, binding_id)
    elif status_filter == "eligible":
        events = stores["feedback"].list_trigger_eligible(key, binding_id)
    else:
        events = stores["feedback"]._all_events(key, binding_id)
    return {"events": events, "count": len(events), "agent_id": binding_id}


@mcp.tool(name="memory_mal_get_artifact")
@_safe_tool
async def memory_mal_get_artifact(
    key: str,
    agent_id: str = "default",
    artifact_id: str = None,
    token: str = None,
    agent_key: str = None,
) -> dict:
    """Get a MAL artifact by id, or the latest if no id given. Requires read ACL."""
    server = _get_memory(key)
    ictx = _resolve_identity(agent_id=agent_id, swarm_id="default", token=token, agent_key=agent_key)
    auth_error = _require_authenticated_principal(ictx)
    if auth_error:
        return auth_error
    denied = _check_instance_acl(server, ictx.owner_id, "read", ictx.caller_role, ictx.memberships)
    if denied:
        return denied
    stores = _get_mal_stores(str(server.data_dir), server=server)
    binding_id = _resolve_mal_binding_id(ictx, agent_id)
    if artifact_id:
        artifact = stores["artifacts"].get(key, binding_id, artifact_id)
    else:
        artifact = stores["artifacts"].get_latest(key, binding_id)
    if artifact is None:
        return {"error": "No artifact found", "code": "NOT_FOUND"}
    return {"artifact": artifact, "agent_id": binding_id}


@mcp.tool(name="memory_mal_rollback")
@_safe_tool
async def memory_mal_rollback(
    key: str,
    agent_id: str = "default",
    to_artifact_id: str = None,
    confirm: bool = False,
    token: str = None,
    agent_key: str = None,
) -> dict:
    """Rollback MAL binding to a prior artifact state. Two-step protocol.

    confirm=False: preview rollback plan (no side effects)
    confirm=True: execute rollback
    """
    server = _get_memory(key)
    ictx = _resolve_identity(agent_id=agent_id, swarm_id="default", token=token, agent_key=agent_key)
    auth_error = _require_authenticated_principal(ictx)
    if auth_error:
        return auth_error
    denied = _check_instance_acl(server, ictx.owner_id, "write", ictx.caller_role, ictx.memberships)
    if denied:
        return denied
    stores = _get_mal_stores(str(server.data_dir), server=server)
    binding_id = _resolve_mal_binding_id(ictx, agent_id)
    from src.mal.apply import ApplyEngine, current_gen_dir, plan_rollback

    latest = stores["artifacts"].get_latest(key, binding_id)
    if latest is None:
        return {"error": "No artifact to rollback from", "code": "NO_ARTIFACT"}

    if to_artifact_id:
        target = stores["artifacts"].get(key, binding_id, to_artifact_id)
        if target is None:
            return {"error": f"Target artifact {to_artifact_id} not found", "code": "NOT_FOUND"}
        target_state = target["materialized_state"]
    else:
        # Rollback to zero state (defaults)
        target_state = {
            "selector_config_overrides": {},
            "grouping_prompt_mode": "strict_small",
            "size_cap_chars": 12000,
            "extraction_prompts": {},
            "inference_leaf_plugin_overrides": {},
        }

    current_state = latest["materialized_state"]
    rollback_plan = plan_rollback(current_state, target_state)

    if not confirm:
        return {
            "target_artifact_id": to_artifact_id,
            "rollback_plan": rollback_plan,
            "requires_confirmation": True,
        }

    # Execute rollback
    data_dir = str(server.data_dir)
    engine = ApplyEngine(data_dir)
    gen_dir = current_gen_dir(data_dir, key, binding_id)
    current_gen_num = int(gen_dir.name.replace("gen_", "")) if gen_dir.name.startswith("gen_") else 0

    result = engine.apply_generation(
        key=key, agent_id=binding_id,
        materialized_state=target_state,
        previous_gen=current_gen_num,
    )

    return {
        "agent_id": binding_id,
        "target_artifact_id": to_artifact_id,
        "rollback_plan": rollback_plan,
        "apply_status": result.get("final_status"),
        "confirmed": True,
    }


# ── SSE endpoint ──

async def sse_cleanup(conn_id: str):
    """Remove SSE connection and all its subscriptions."""
    for sid, cid in list(sub_to_conn.items()):
        if cid == conn_id:
            for courier in courier_registry.values():
                await courier.unsubscribe(sid)
            sub_to_conn.pop(sid, None)
    connections.pop(conn_id, None)
    _active_connections.pop(conn_id, None)


async def sse_endpoint(request):
    """SSE stream endpoint. Sends connected event, then pushes Courier artifacts."""
    from starlette.responses import StreamingResponse

    conn_id = str(uuid4())
    queue: asyncio.Queue = asyncio.Queue()
    connections[conn_id] = queue
    # C2: register this connection as active (for hijack prevention)
    remote = ""
    if request is not None:
        client = getattr(request, "client", None)
        remote = f"{client.host}:{client.port}" if client else "unknown"
    _active_connections[conn_id] = remote

    await queue.put({"type": "connected", "connection_id": conn_id})

    async def event_generator():
        try:
            while True:
                event = await queue.get()
                yield f"data: {json.dumps(event)}\n\n"
        except asyncio.CancelledError:
            await sse_cleanup(conn_id)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── App factory ──

def create_app(app_data_dir="./data", app_cfg=None, bind_host: str = "127.0.0.1"):
    """Create combined Starlette app with MCP + SSE.

    The MCP SDK's session manager requires lifespan initialization
    (task group setup). We forward it from the outer Starlette app.
    """
    global data_dir, cfg
    data_dir = app_data_dir
    if app_cfg:
        cfg = app_cfg
    _configure_mcp_bind_host(bind_host)

    # H2 fix: persist embed_model to config so embed_batch() picks it up
    if app_cfg and app_cfg.embed_model:
        try:
            from .setup_store import load_config, save_config
            file_cfg = load_config()
            if file_cfg.get("embed_model") != app_cfg.embed_model:
                file_cfg["embed_model"] = app_cfg.embed_model
                save_config(file_cfg)
        except Exception:
            pass  # best-effort — don't block server startup

    import contextlib

    from starlette.applications import Starlette
    from starlette.middleware import Middleware
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import JSONResponse
    from starlette.routing import Mount, Route

    def _json_no_store(payload: dict[str, Any], *, status_code: int = 200) -> JSONResponse:
        return JSONResponse(
            payload,
            status_code=status_code,
            headers={"Cache-Control": "no-store"},
        )

    # C1: Token authentication middleware
    class TokenAuthMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            # /health is public (monitoring probes need it)
            if request.url.path == "/health":
                return await call_next(request)
            principal_token = None
            auth_header = request.headers.get("authorization", "")
            if auth_header.lower().startswith("bearer "):
                principal_token = auth_header[7:].strip() or None
            token = request.headers.get("x-gosh-memory-token") or request.headers.get("x-server-token", "")
            if token != SERVER_TOKEN:
                return JSONResponse({"error": "Unauthorized"}, status_code=401)
            # Set principal token AFTER perimeter-token check passes.
            # MCP streamable HTTP handlers may read this after middleware
            # returns, so /mcp requests must not reset here. Plain REST
            # endpoints complete inside call_next and must reset to avoid
            # leaking a bearer into direct in-process tool calls.
            token_handle = _request_principal_token.set(principal_token)
            response = await call_next(request)
            if not request.url.path.startswith("/mcp"):
                _request_principal_token.reset(token_handle)
            return response

    mcp_app = mcp.streamable_http_app()

    async def health(request):
        return JSONResponse({"status": "ok"})

    # C3: Admin reload endpoint — lets CLI import notify running server
    async def admin_reload(request):
        ctx = _resolve_identity(token=_request_principal_token.get())
        auth_error = _require_admin(ctx)
        if auth_error:
            status_code = 401 if auth_error.get("code", "").startswith("AUTH_") else 403
            return JSONResponse(auth_error, status_code=status_code)
        body = await request.json()
        key = body.get("key")
        if key and key in registry:
            server = registry[key]
            if server._storage.exists:
                server._reload_runtime_from_storage()
        return JSONResponse({"status": "reloaded"})

    async def admin_memory_init(request):
        content_type = str(request.headers.get("content-type") or "").lower()
        if not content_type.startswith("application/json"):
            return _json_no_store(
                {"error": "Content-Type must be application/json", "code": "VALIDATION_ERROR"},
                status_code=400,
            )
        try:
            body = await request.json()
        except Exception:
            return _json_no_store(
                {"error": "request body must be valid JSON", "code": "VALIDATION_ERROR"},
                status_code=400,
            )
        if not isinstance(body, dict):
            return _json_no_store(
                {"error": "request body must be a JSON object", "code": "VALIDATION_ERROR"},
                status_code=400,
            )

        ctx = _resolve_identity(token=_request_principal_token.get())
        key = _json_field_text(body, "key", required=True)
        owner_id = _json_field_text(body, "owner_id", required=False)
        result = await _memory_init_instance(key, ctx=ctx, owner_id=owner_id)
        code = str(result.get("code") or "")
        if not code:
            return _json_no_store(result, status_code=200)
        if code == "VALIDATION_ERROR":
            return _json_no_store(result, status_code=400)
        if code == "ALREADY_EXISTS":
            return _json_no_store(result, status_code=409)
        if code.startswith("AUTH_"):
            return _json_no_store(result, status_code=401)
        return _json_no_store(result, status_code=403)

    def _parse_agent_public_key_register_body(body: dict[str, Any]) -> dict[str, str | None]:
        public_key = _json_field_text(body, "public_key", required=True)
        algorithm = _json_field_text(body, "algorithm", required=False)
        key_id = _json_field_text(body, "key_id", required=False)
        principal_id = _json_field_text(body, "principal_id", required=False)
        return {
            "public_key": public_key or "",
            "algorithm": algorithm,
            "key_id": key_id,
            "principal_id": principal_id,
        }

    async def agent_public_key_register(request):
        content_type = str(request.headers.get("content-type") or "").lower()
        if not content_type.startswith("application/json"):
            return _json_no_store(
                {"error": "Content-Type must be application/json", "code": "VALIDATION_ERROR"},
                status_code=400,
            )
        try:
            body = await request.json()
        except Exception:
            return _json_no_store(
                {"error": "request body must be valid JSON", "code": "VALIDATION_ERROR"},
                status_code=400,
            )
        if not isinstance(body, dict):
            return _json_no_store(
                {"error": "request body must be a JSON object", "code": "VALIDATION_ERROR"},
                status_code=400,
            )

        try:
            parsed = _parse_agent_public_key_register_body(body)
        except ValueError as exc:
            return _json_no_store(
                {"error": str(exc), "code": "VALIDATION_ERROR"},
                status_code=400,
            )

        ctx = _resolve_identity(token=_request_principal_token.get())
        try:
            actor = _ctx_actor(ctx)
        except AuthorityError as exc:
            status_code = 401 if exc.code.startswith("AUTH_") else 403
            return _json_no_store(_authority_error(exc), status_code=status_code)

        target_principal_id = parsed["principal_id"] or actor.principal_id
        if not parsed["principal_id"] and str(actor.principal_kind or "") != "agent":
            return _json_no_store(
                {"error": "agent principal required", "code": "AGENT_PRINCIPAL_REQUIRED"},
                status_code=403,
            )

        try:
            binding = _get_authority().set_agent_public_key_binding(
                actor=actor,
                principal_id=target_principal_id,
                public_key=parsed["public_key"] or "",
                algorithm=parsed["algorithm"] or "x25519",
                key_id=parsed["key_id"],
            )
        except AuthorityError as exc:
            status_code = 401 if exc.code.startswith("AUTH_") else 403
            if exc.code == "VALIDATION_ERROR":
                status_code = 400
            elif exc.code == "NOT_FOUND":
                status_code = 404
            return _json_no_store(_authority_error(exc), status_code=status_code)

        metadata = binding.get("metadata") if isinstance(binding.get("metadata"), dict) else {}
        key_binding = metadata.get("agent_public_key") if isinstance(metadata, dict) else {}
        return _json_no_store(
            {
                "status": "ok",
                "principal_id": target_principal_id,
                "algorithm": str(key_binding.get("algorithm") or "x25519"),
                "key_id": str(key_binding.get("key_id") or ""),
            },
            status_code=200,
        )

    def _parse_gosh_secrets_http_body(body: dict, *, require_header_name: bool = False) -> dict[str, str | None]:
        key = _json_field_text(body, "key", required=True)
        name = _json_field_text(body, "name", required=True)
        scope = _json_field_text(body, "scope", required=True)
        purpose = _json_field_text(body, "purpose", required=True)
        method = _json_field_text(body, "method", required=True)
        url = _json_field_text(body, "url", required=True)
        agent_id = _json_field_text(body, "agent_id", required=False)
        swarm_id = _json_field_text(body, "swarm_id", required=False)
        header_name = _json_field_text(body, "header_name", required=require_header_name)
        if scope == "swarm-shared" and not swarm_id:
            raise ValueError("swarm_id must be provided explicitly for swarm-shared scope")
        return {
            "key": key,
            "name": name,
            "scope": scope,
            "purpose": purpose,
            "method": method,
            "url": url,
            "agent_id": agent_id,
            "swarm_id": swarm_id,
            "header_name": header_name,
        }

    def _parse_agent_secret_resolve_body(body: dict[str, Any]) -> dict[str, Any]:
        key = _json_field_text(body, "key", required=True)
        refs = body.get("refs")
        if not isinstance(refs, list) or not refs:
            raise ValueError("refs must be a non-empty array")
        parsed_refs: list[dict[str, str | None]] = []
        for idx, entry in enumerate(refs):
            if not isinstance(entry, dict):
                raise ValueError(f"refs[{idx}] must be an object")
            name = _json_field_text(entry, "name", required=True)
            scope = _json_field_text(entry, "scope", required=True)
            agent_id = _json_field_text(entry, "agent_id", required=False)
            swarm_id = _json_field_text(entry, "swarm_id", required=False)
            if scope == "swarm-shared" and not swarm_id:
                raise ValueError(f"refs[{idx}].swarm_id must be provided explicitly for swarm-shared scope")
            parsed_refs.append(
                {
                    "name": name,
                    "scope": scope,
                    "agent_id": agent_id,
                    "swarm_id": swarm_id,
                }
            )
        return {"key": key, "refs": parsed_refs}

    async def gosh_secrets_http_bearer(request):
        content_type = str(request.headers.get("content-type") or "").lower()
        if not content_type.startswith("application/json"):
            return _json_no_store(
                {"error": "Content-Type must be application/json", "code": "VALIDATION_ERROR"},
                status_code=400,
            )
        try:
            body = await request.json()
        except Exception:
            return _json_no_store(
                {"error": "request body must be valid JSON", "code": "VALIDATION_ERROR"},
                status_code=400,
            )
        if not isinstance(body, dict):
            return _json_no_store(
                {"error": "request body must be a JSON object", "code": "VALIDATION_ERROR"},
                status_code=400,
            )

        try:
            parsed = _parse_gosh_secrets_http_body(body)
        except ValueError as exc:
            return _json_no_store(
                {"error": str(exc), "code": "VALIDATION_ERROR"},
                status_code=400,
            )

        ctx = _resolve_identity(token=_request_principal_token.get())
        try:
            actor = _ctx_actor(ctx)
        except AuthorityError as exc:
            status_code = 401 if exc.code.startswith("AUTH_") else 403
            return _json_no_store(_authority_error(exc), status_code=status_code)

        try:
            server = _get_memory(parsed["key"])
            payload = await http_bearer_with_secret(
                server,
                actor=actor,
                name=parsed["name"],
                scope=parsed["scope"],
                purpose=parsed["purpose"],
                method=parsed["method"],
                url=parsed["url"],
                agent_id=parsed["agent_id"],
                swarm_id=parsed["swarm_id"],
                headers=body.get("headers"),
                body=body.get("body"),
            )
            return _json_no_store(payload, status_code=200)
        except ValueError as exc:
            return _json_no_store(
                {"error": str(exc), "code": "VALIDATION_ERROR"},
                status_code=400,
            )
        except GoshSecretsError as exc:
            return _json_no_store(
                {"error": str(exc), "code": exc.code},
                status_code=exc.status_code,
            )

    async def gosh_secrets_http_header_value(request):
        content_type = str(request.headers.get("content-type") or "").lower()
        if not content_type.startswith("application/json"):
            return _json_no_store(
                {"error": "Content-Type must be application/json", "code": "VALIDATION_ERROR"},
                status_code=400,
            )
        try:
            body = await request.json()
        except Exception:
            return _json_no_store(
                {"error": "request body must be valid JSON", "code": "VALIDATION_ERROR"},
                status_code=400,
            )
        if not isinstance(body, dict):
            return _json_no_store(
                {"error": "request body must be a JSON object", "code": "VALIDATION_ERROR"},
                status_code=400,
            )

        try:
            parsed = _parse_gosh_secrets_http_body(body, require_header_name=True)
        except ValueError as exc:
            return _json_no_store(
                {"error": str(exc), "code": "VALIDATION_ERROR"},
                status_code=400,
            )

        ctx = _resolve_identity(token=_request_principal_token.get())
        try:
            actor = _ctx_actor(ctx)
        except AuthorityError as exc:
            status_code = 401 if exc.code.startswith("AUTH_") else 403
            return _json_no_store(_authority_error(exc), status_code=status_code)

        try:
            server = _get_memory(parsed["key"])
            payload = await http_header_value_with_secret(
                server,
                actor=actor,
                name=parsed["name"],
                scope=parsed["scope"],
                purpose=parsed["purpose"],
                method=parsed["method"],
                url=parsed["url"],
                header_name=parsed["header_name"] or "",
                agent_id=parsed["agent_id"],
                swarm_id=parsed["swarm_id"],
                headers=body.get("headers"),
                body=body.get("body"),
            )
            return _json_no_store(payload, status_code=200)
        except ValueError as exc:
            return _json_no_store(
                {"error": str(exc), "code": "VALIDATION_ERROR"},
                status_code=400,
            )
        except GoshSecretsError as exc:
            return _json_no_store(
                {"error": str(exc), "code": exc.code},
                status_code=exc.status_code,
            )

    async def agent_secrets_resolve(request):
        content_type = str(request.headers.get("content-type") or "").lower()
        if not content_type.startswith("application/json"):
            return _json_no_store(
                {"error": "Content-Type must be application/json", "code": "VALIDATION_ERROR"},
                status_code=400,
            )
        try:
            body = await request.json()
        except Exception:
            return _json_no_store(
                {"error": "request body must be valid JSON", "code": "VALIDATION_ERROR"},
                status_code=400,
            )
        if not isinstance(body, dict):
            return _json_no_store(
                {"error": "request body must be a JSON object", "code": "VALIDATION_ERROR"},
                status_code=400,
            )

        try:
            parsed = _parse_agent_secret_resolve_body(body)
        except ValueError as exc:
            return _json_no_store(
                {"error": str(exc), "code": "VALIDATION_ERROR"},
                status_code=400,
            )

        ctx = _resolve_identity(token=_request_principal_token.get())
        try:
            actor = _ctx_actor(ctx)
        except AuthorityError as exc:
            status_code = 401 if exc.code.startswith("AUTH_") else 403
            return _json_no_store(_authority_error(exc), status_code=status_code)

        try:
            server = _get_memory(parsed["key"])
            payload = resolve_secrets_for_agent(
                server,
                authority=_get_authority(),
                actor=actor,
                key=parsed["key"],
                refs=parsed["refs"],
            )
            return _json_no_store(payload, status_code=200)
        except GoshSecretsError as exc:
            return _json_no_store(
                {"error": str(exc), "code": exc.code},
                status_code=exc.status_code,
            )

    @contextlib.asynccontextmanager
    async def lifespan(app):
        global _write_log_worker_task
        async with mcp.session_manager.run():
            _write_log_worker_task = asyncio.create_task(_run_write_log_workers())
            try:
                yield
            finally:
                if _write_log_worker_task is not None:
                    _write_log_worker_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await _write_log_worker_task
                    _write_log_worker_task = None

    return Starlette(
        routes=[
            Route("/health", health, methods=["GET"]),
            Route("/admin/reload", admin_reload, methods=["POST"]),
            Route("/api/v1/admin/memory/init", admin_memory_init, methods=["POST"]),
            Route("/api/v1/agent/public-key/register", agent_public_key_register, methods=["POST"]),
            Route("/api/v1/agent/secrets/resolve", agent_secrets_resolve, methods=["POST"]),
            Route("/api/v1/gosh-secrets/http/bearer", gosh_secrets_http_bearer, methods=["POST"]),
            Route("/api/v1/gosh-secrets/http/header-value", gosh_secrets_http_header_value, methods=["POST"]),
            Route("/mcp/sse", endpoint=sse_endpoint),
            Mount("/", app=mcp_app),
        ],
        lifespan=lifespan,
        middleware=[Middleware(TokenAuthMiddleware)],
    )


# ── CLI ──

def parse_args():
    p = argparse.ArgumentParser(description="GOSH Memory MCP Server")
    p.add_argument("--data-dir", type=str, default="./data")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--extraction-model", type=str, default=None)
    p.add_argument("--inference-model", type=str, default=None)
    p.add_argument("--judge-model", type=str, default=None)
    p.add_argument("--embed-model", type=str, default=None)
    p.add_argument("--server-token", type=str, default=None,
                   help="Server auth token (if not set, auto-generated)")
    return p.parse_args()


def _save_token():
    """Save server token to ~/.gosh-memory/token on startup."""
    token_path = Path.home() / ".gosh-memory" / "token"
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(SERVER_TOKEN)
    token_path.chmod(0o600)
    return token_path


if __name__ == "__main__":
    import uvicorn

    args = parse_args()
    if args.server_token:
        SERVER_TOKEN = args.server_token
    app_cfg = MemoryConfig.from_args(args)
    app = create_app(app_data_dir=args.data_dir, app_cfg=app_cfg, bind_host=args.host)

    token_path = _save_token()

    # Resolve embedding config for startup display
    try:
        from .setup_store import get_config as _get_cfg
        _scfg = _get_cfg()
        _embed_prov = _scfg.get("embed_provider", "openai")
        _embed_mod = _scfg.get("embed_model", "text-embedding-3-large")
    except Exception:
        _embed_prov, _embed_mod = "openai", "text-embedding-3-large"

    log_startup_lines(startup_log_lines(
        title="GOSH Memory MCP Server",
        summary=app_cfg.summary(),
        listening=f"http://{args.host}:{args.port}",
        embeddings=f"{_embed_prov} / {_embed_mod}",
        token=SERVER_TOKEN,
        token_path=token_path,
    ))

    uvicorn.run(app, host=args.host, port=args.port)
