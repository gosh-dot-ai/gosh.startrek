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
import ipaddress
import json
import logging
import os
import re
import socket
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

import httpx

from src.membership import AuthorityService, ResolvedPrincipal
from src.memory import MemoryServer

log = logging.getLogger("gosh.secrets.audit")
IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address
_AGENT_SECRET_ALGORITHM = "x25519-hkdf-sha256-aes256gcm-v1"  # noqa: S105
_AGENT_PUBLIC_KEY_ALGORITHM = "x25519"
_AGENT_SECRET_INFO = b"gosh.memory/agent-secrets/v1"
_AGENT_SECRET_MAGIC = b"GMS1"

_SAFE_RESPONSE_HEADERS = {
    "cache-control",
    "content-length",
    "content-type",
    "etag",
    "last-modified",
}
_DISALLOWED_REQUEST_HEADERS = {
    "authorization",
    "content-length",
    "connection",
    "host",
    "proxy-authorization",
    "transfer-encoding",
}
_ALLOWED_HTTP_METHODS = {
    "DELETE",
    "GET",
    "HEAD",
    "OPTIONS",
    "PATCH",
    "POST",
    "PUT",
}
_HEADER_NAME_RE = re.compile(r"^[A-Za-z0-9-]+$")


class GoshSecretsError(Exception):
    """Public-safe error for the trusted gosh.secrets HTTP layer."""

    def __init__(self, code: str, message: str, *, status_code: int):
        super().__init__(message)
        self.code = code
        self.status_code = status_code


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _audit_secret_action(
    *,
    principal_id: str,
    secret_id: str | None,
    name: str,
    scope: str,
    agent_id: str | None,
    swarm_id: str | None,
    purpose: str,
    allowed: bool,
    action: str,
    method: str,
    url: str,
    header_name: str | None = None,
) -> None:
    event = {
        "timestamp": _utcnow_iso(),
        "principal_id": principal_id,
        "secret_id": secret_id,
        "name": name,
        "scope": scope,
        "agent_id": agent_id,
        "swarm_id": swarm_id,
        "purpose": purpose,
        "allowed": allowed,
        "action": action,
        "method": method,
        "url": url,
    }
    if header_name:
        event["header_name"] = header_name
    log.info("gosh.secrets.action %s", json.dumps(event, sort_keys=True))


def _audit_agent_secret_delivery(
    *,
    principal_id: str,
    secret_id: str | None,
    key: str,
    name: str,
    scope: str,
    agent_id: str | None,
    swarm_id: str | None,
    allowed: bool,
    code: str | None = None,
    key_id: str | None = None,
) -> None:
    event = {
        "timestamp": _utcnow_iso(),
        "principal_id": principal_id,
        "secret_id": secret_id,
        "key": key,
        "name": name,
        "scope": scope,
        "agent_id": agent_id,
        "swarm_id": swarm_id,
        "allowed": allowed,
        "action": "agent_secret_resolve",
    }
    if code:
        event["code"] = code
    if key_id:
        event["key_id"] = key_id
    log.info("gosh.secrets.agent %s", json.dumps(event, sort_keys=True))


def _resolve_secret_row_for_trusted_use(
    server: MemoryServer,
    *,
    actor: ResolvedPrincipal,
    name: str,
    scope: str,
    purpose: str,
    action: str,
    method: str,
    url: str,
    agent_id: str | None = None,
    swarm_id: str | None = None,
    header_name: str | None = None,
) -> dict[str, Any]:
    try:
        row = server._resolve_secret_value_for_internal_use(
            name=name,
            agent_id=agent_id,
            swarm_id=swarm_id,
            scope=scope,
            caller_id=actor.principal_id,
            caller_memberships=actor.memberships,
            caller_role="user",
        )
    except ValueError as exc:
        _audit_secret_action(
            principal_id=actor.principal_id,
            secret_id=None,
            name=name,
            scope=scope,
            agent_id=agent_id,
            swarm_id=swarm_id,
            purpose=purpose,
            allowed=False,
            action=action,
            method=method,
            url=url,
            header_name=header_name,
        )
        raise GoshSecretsError("VALIDATION_ERROR", str(exc), status_code=400) from exc
    except KeyError as exc:
        _audit_secret_action(
            principal_id=actor.principal_id,
            secret_id=None,
            name=name,
            scope=scope,
            agent_id=agent_id,
            swarm_id=swarm_id,
            purpose=purpose,
            allowed=False,
            action=action,
            method=method,
            url=url,
            header_name=header_name,
        )
        raise GoshSecretsError("SECRET_NOT_FOUND", "secret not found", status_code=404) from exc
    except PermissionError as exc:
        _audit_secret_action(
            principal_id=actor.principal_id,
            secret_id=None,
            name=name,
            scope=scope,
            agent_id=agent_id,
            swarm_id=swarm_id,
            purpose=purpose,
            allowed=False,
            action=action,
            method=method,
            url=url,
            header_name=header_name,
        )
        raise GoshSecretsError("SECRET_FORBIDDEN", "access denied", status_code=403) from exc
    _audit_secret_action(
        principal_id=actor.principal_id,
        secret_id=str(row.get("secret_id") or ""),
        name=name,
        scope=scope,
        agent_id=agent_id,
        swarm_id=swarm_id,
        purpose=purpose,
        allowed=True,
        action=action,
        method=method,
        url=url,
        header_name=header_name,
    )
    return row


def _resolve_secret_row_for_agent_delivery(
    server: MemoryServer,
    *,
    actor: ResolvedPrincipal,
    key: str,
    name: str,
    scope: str,
    agent_id: str | None = None,
    swarm_id: str | None = None,
) -> dict[str, Any]:
    try:
        row = server._resolve_secret_value_for_internal_use(
            name=name,
            agent_id=agent_id,
            swarm_id=swarm_id,
            scope=scope,
            caller_id=actor.principal_id,
            caller_memberships=actor.memberships,
            caller_role="user",
        )
    except ValueError as exc:
        _audit_agent_secret_delivery(
            principal_id=actor.principal_id,
            secret_id=None,
            key=key,
            name=name,
            scope=scope,
            agent_id=agent_id,
            swarm_id=swarm_id,
            allowed=False,
            code="VALIDATION_ERROR",
        )
        raise GoshSecretsError("VALIDATION_ERROR", str(exc), status_code=400) from exc
    except KeyError as exc:
        _audit_agent_secret_delivery(
            principal_id=actor.principal_id,
            secret_id=None,
            key=key,
            name=name,
            scope=scope,
            agent_id=agent_id,
            swarm_id=swarm_id,
            allowed=False,
            code="SECRET_NOT_FOUND",
        )
        raise GoshSecretsError("SECRET_NOT_FOUND", "secret not found", status_code=404) from exc
    except PermissionError as exc:
        _audit_agent_secret_delivery(
            principal_id=actor.principal_id,
            secret_id=None,
            key=key,
            name=name,
            scope=scope,
            agent_id=agent_id,
            swarm_id=swarm_id,
            allowed=False,
            code="SECRET_FORBIDDEN",
        )
        raise GoshSecretsError("SECRET_FORBIDDEN", "access denied", status_code=403) from exc

    return row


def _fingerprint_public_key(public_key_bytes: bytes) -> str:
    return f"sha256:{hashlib.sha256(public_key_bytes).hexdigest()}"


def _load_agent_public_key_binding(
    authority: AuthorityService,
    *,
    actor: ResolvedPrincipal,
) -> tuple[Any, str]:
    if str(actor.principal_kind or "") != "agent":
        raise GoshSecretsError(
            "AGENT_PRINCIPAL_REQUIRED",
            "agent principal required",
            status_code=403,
        )
    binding = authority.get_agent_public_key_binding(principal_id=actor.principal_id)
    if binding is None:
        raise GoshSecretsError(
            "MISSING_AGENT_PUBLIC_KEY",
            "agent public key is not registered",
            status_code=409,
        )
    algorithm = str(binding.get("algorithm") or _AGENT_PUBLIC_KEY_ALGORITHM).strip().lower()
    if algorithm != _AGENT_PUBLIC_KEY_ALGORITHM:
        raise GoshSecretsError(
            "INVALID_AGENT_PUBLIC_KEY",
            "stored agent public key is invalid",
            status_code=500,
        )
    public_key_b64 = str(binding.get("public_key") or "").strip()
    if not public_key_b64:
        raise GoshSecretsError(
            "MISSING_AGENT_PUBLIC_KEY",
            "agent public key is not registered",
            status_code=409,
        )
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import x25519
    except ImportError as exc:
        raise GoshSecretsError(
            "CRYPTO_UNAVAILABLE",
            "secret delivery crypto is unavailable",
            status_code=503,
        ) from exc
    try:
        public_key_bytes = base64.b64decode(public_key_b64.encode("ascii"), validate=True)
        if len(public_key_bytes) != 32:
            raise ValueError("bad x25519 public key length")
        public_key = x25519.X25519PublicKey.from_public_bytes(public_key_bytes)
        # Force one serialization round-trip so invalid objects fail here.
        public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
    except (ValueError, binascii.Error) as exc:
        raise GoshSecretsError(
            "INVALID_AGENT_PUBLIC_KEY",
            "stored agent public key is invalid",
            status_code=500,
        ) from exc
    key_id = str(binding.get("key_id") or "").strip() or _fingerprint_public_key(public_key_bytes)
    return public_key, key_id


def _encrypt_secret_for_agent(secret_value: str, public_key: Any) -> str:
    try:
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import x25519
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    except ImportError as exc:
        raise GoshSecretsError(
            "CRYPTO_UNAVAILABLE",
            "secret delivery crypto is unavailable",
            status_code=503,
        ) from exc
    ephemeral_private = x25519.X25519PrivateKey.generate()
    shared_key = ephemeral_private.exchange(public_key)
    aes_key = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=_AGENT_SECRET_INFO,
    ).derive(shared_key)
    nonce = os.urandom(12)
    ciphertext = AESGCM(aes_key).encrypt(
        nonce,
        str(secret_value).encode("utf-8"),
        _AGENT_SECRET_INFO,
    )
    ephemeral_public_bytes = ephemeral_private.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    envelope = _AGENT_SECRET_MAGIC + ephemeral_public_bytes + nonce + ciphertext
    return base64.b64encode(envelope).decode("ascii")


def _resolve_host_addresses(hostname: str, port: int) -> set[IPAddress]:
    try:
        infos = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise GoshSecretsError("UPSTREAM_DNS_ERROR", "unable to resolve upstream host", status_code=502) from exc
    resolved: set[IPAddress] = set()
    for family, _, _, _, sockaddr in infos:
        if family in (socket.AF_INET, socket.AF_INET6):
            resolved.add(ipaddress.ip_address(sockaddr[0]))
    if not resolved:
        raise GoshSecretsError("UPSTREAM_DNS_ERROR", "unable to resolve upstream host", status_code=502)
    return resolved


def _ensure_public_http_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise GoshSecretsError("VALIDATION_ERROR", "url scheme must be http or https", status_code=400)
    if not parsed.hostname:
        raise GoshSecretsError("VALIDATION_ERROR", "url must include a hostname", status_code=400)
    if parsed.username or parsed.password:
        raise GoshSecretsError("VALIDATION_ERROR", "url must not include userinfo", status_code=400)
    if parsed.scheme == "http" and parsed.port is None:
        port = 80
    elif parsed.scheme == "https" and parsed.port is None:
        port = 443
    else:
        port = int(parsed.port or 0)
    for addr in _resolve_host_addresses(parsed.hostname, port):
        if (
            addr.is_loopback
            or addr.is_private
            or addr.is_link_local
            or addr.is_multicast
            or addr.is_reserved
            or addr.is_unspecified
        ):
            raise GoshSecretsError("SSRF_FORBIDDEN", "upstream host is not allowed", status_code=403)


def _sanitize_request_headers(headers: Any) -> dict[str, str]:
    if headers is None:
        return {}
    if not isinstance(headers, dict):
        raise GoshSecretsError("VALIDATION_ERROR", "headers must be an object", status_code=400)
    sanitized: dict[str, str] = {}
    for raw_name, raw_value in headers.items():
        name = str(raw_name or "").strip()
        if not name:
            raise GoshSecretsError("VALIDATION_ERROR", "header names must be non-empty", status_code=400)
        lower = name.lower()
        if lower in _DISALLOWED_REQUEST_HEADERS:
            raise GoshSecretsError("VALIDATION_ERROR", f"header {name} is not allowed", status_code=400)
        if not isinstance(raw_value, (str, int, float, bool)):
            raise GoshSecretsError("VALIDATION_ERROR", f"header {name} must be a scalar string value", status_code=400)
        sanitized[name] = str(raw_value)
    return sanitized


def _encode_request_body(body: Any, headers: dict[str, str]) -> bytes | None:
    if body is None:
        return None
    if isinstance(body, (dict, list)):
        if not any(h.lower() == "content-type" for h in headers):
            headers["Content-Type"] = "application/json"
        return json.dumps(body).encode("utf-8")
    if isinstance(body, str):
        return body.encode("utf-8")
    raise GoshSecretsError("VALIDATION_ERROR", "body must be string, object, array, or null", status_code=400)


async def _perform_bearer_http_request(
    *,
    method: str,
    url: str,
    headers: dict[str, str],
    body: bytes | None,
) -> tuple[int, dict[str, str], bytes]:
    return await _perform_http_request(
        method=method,
        url=url,
        headers=headers,
        body=body,
    )


async def _perform_header_value_http_request(
    *,
    method: str,
    url: str,
    headers: dict[str, str],
    body: bytes | None,
) -> tuple[int, dict[str, str], bytes]:
    return await _perform_http_request(
        method=method,
        url=url,
        headers=headers,
        body=body,
    )


async def _perform_http_request(
    *,
    method: str,
    url: str,
    headers: dict[str, str],
    body: bytes | None,
) -> tuple[int, dict[str, str], bytes]:
    try:
        async with httpx.AsyncClient(
            follow_redirects=False,
            timeout=20.0,
            trust_env=False,
        ) as client:
            response = await client.request(
                method=method,
                url=url,
                headers=headers,
                content=body,
            )
    except httpx.HTTPError as exc:
        raise GoshSecretsError("UPSTREAM_REQUEST_FAILED", "upstream request failed", status_code=502) from exc
    return response.status_code, dict(response.headers), bytes(response.content)


def _filter_response_headers(headers: dict[str, str]) -> dict[str, str]:
    filtered: dict[str, str] = {}
    for name, value in headers.items():
        if str(name).lower() in _SAFE_RESPONSE_HEADERS:
            filtered[name] = value
    return filtered


def _body_to_text(body: bytes) -> str:
    return body.decode("utf-8", errors="replace")


def _contains_secret_material(secret_value: str, headers: dict[str, str], body: bytes) -> bool:
    secret_bytes = secret_value.encode("utf-8")
    if secret_bytes and secret_bytes in body:
        return True
    for value in headers.values():
        if secret_value and secret_value in str(value):
            return True
    return False


def _normalize_http_action_request(method: str, url: str) -> tuple[str, str]:
    normalized_method = str(method or "").strip().upper()
    if normalized_method not in _ALLOWED_HTTP_METHODS:
        raise GoshSecretsError(
            "VALIDATION_ERROR",
            f"method must be one of {sorted(_ALLOWED_HTTP_METHODS)}",
            status_code=400,
        )
    normalized_url = str(url or "").strip()
    if not normalized_url:
        raise GoshSecretsError("VALIDATION_ERROR", "url must be provided explicitly", status_code=400)
    _ensure_public_http_url(normalized_url)
    return normalized_method, normalized_url


def _normalize_secret_header_name(header_name: Any) -> str:
    normalized = str(header_name or "").strip()
    if not normalized:
        raise GoshSecretsError("VALIDATION_ERROR", "header_name must be provided explicitly", status_code=400)
    if not _HEADER_NAME_RE.fullmatch(normalized):
        raise GoshSecretsError("VALIDATION_ERROR", "header_name must be a simple HTTP header token", status_code=400)
    if normalized.lower() in _DISALLOWED_REQUEST_HEADERS:
        raise GoshSecretsError("VALIDATION_ERROR", f"header {normalized} is not allowed", status_code=400)
    return normalized


def _prepare_secret_http_request(
    server: MemoryServer,
    *,
    actor: ResolvedPrincipal,
    name: str,
    scope: str,
    purpose: str,
    action: str,
    method: str,
    url: str,
    agent_id: str | None,
    swarm_id: str | None,
    headers: Any,
    body: Any,
    header_name: str | None = None,
) -> tuple[dict[str, Any], str, str, dict[str, str], bytes | None]:
    normalized_method, normalized_url = _normalize_http_action_request(method, url)
    row = _resolve_secret_row_for_trusted_use(
        server,
        actor=actor,
        name=name,
        scope=scope,
        purpose=purpose,
        action=action,
        method=normalized_method,
        url=normalized_url,
        agent_id=agent_id,
        swarm_id=swarm_id,
        header_name=header_name,
    )
    request_headers = _sanitize_request_headers(headers)
    request_body = _encode_request_body(body, request_headers)
    return row, normalized_method, normalized_url, request_headers, request_body


async def http_bearer_with_secret(
    server: MemoryServer,
    *,
    actor: ResolvedPrincipal,
    name: str,
    scope: str,
    purpose: str,
    method: str,
    url: str,
    agent_id: str | None = None,
    swarm_id: str | None = None,
    headers: Any = None,
    body: Any = None,
) -> dict[str, Any]:
    """Execute one trusted HTTP request with `Authorization: Bearer <secret>`.

    The secret value is resolved inside gosh.memory and used only inside the
    server process. It is never returned to the caller.
    """
    try:
        row, normalized_method, normalized_url, request_headers, request_body = _prepare_secret_http_request(
            server,
            actor=actor,
            name=name,
            scope=scope,
            purpose=purpose,
            action="http_bearer",
            method=method,
            url=url,
            agent_id=agent_id,
            swarm_id=swarm_id,
            headers=headers,
            body=body,
        )
    except RuntimeError as exc:
        raise GoshSecretsError(
            "SECRET_STORAGE_UNAVAILABLE",
            str(exc),
            status_code=503,
        ) from exc

    request_headers["Authorization"] = f"Bearer {row['value']}"
    status, response_headers, response_body = await _perform_bearer_http_request(
        method=normalized_method,
        url=normalized_url,
        headers=request_headers,
        body=request_body,
    )
    filtered_headers = _filter_response_headers(response_headers)
    if _contains_secret_material(str(row["value"]), filtered_headers, response_body):
        raise GoshSecretsError(
            "SECRET_LEAK_BLOCKED",
            "upstream response contained protected secret material",
            status_code=502,
        )
    return {
        "status": status,
        "headers": filtered_headers,
        "body": _body_to_text(response_body),
    }


async def http_header_value_with_secret(
    server: MemoryServer,
    *,
    actor: ResolvedPrincipal,
    name: str,
    scope: str,
    purpose: str,
    method: str,
    url: str,
    header_name: str,
    agent_id: str | None = None,
    swarm_id: str | None = None,
    headers: Any = None,
    body: Any = None,
) -> dict[str, Any]:
    """Execute one trusted HTTP request with the secret injected into one header."""
    normalized_header_name = _normalize_secret_header_name(header_name)
    try:
        row, normalized_method, normalized_url, request_headers, request_body = _prepare_secret_http_request(
            server,
            actor=actor,
            name=name,
            scope=scope,
            purpose=purpose,
            action="http_header_value",
            method=method,
            url=url,
            agent_id=agent_id,
            swarm_id=swarm_id,
            headers=headers,
            body=body,
            header_name=normalized_header_name,
        )
    except RuntimeError as exc:
        raise GoshSecretsError(
            "SECRET_STORAGE_UNAVAILABLE",
            str(exc),
            status_code=503,
        ) from exc
    if any(existing.lower() == normalized_header_name.lower() for existing in request_headers):
        raise GoshSecretsError(
            "VALIDATION_ERROR",
            f"header {normalized_header_name} must not be supplied explicitly",
            status_code=400,
        )
    request_headers[normalized_header_name] = str(row["value"])
    status, response_headers, response_body = await _perform_header_value_http_request(
        method=normalized_method,
        url=normalized_url,
        headers=request_headers,
        body=request_body,
    )
    filtered_headers = _filter_response_headers(response_headers)
    if _contains_secret_material(str(row["value"]), filtered_headers, response_body):
        raise GoshSecretsError(
            "SECRET_LEAK_BLOCKED",
            "upstream response contained protected secret material",
            status_code=502,
        )
    return {
        "status": status,
        "headers": filtered_headers,
        "body": _body_to_text(response_body),
    }


def resolve_secrets_for_agent(
    server: MemoryServer,
    *,
    authority: AuthorityService,
    actor: ResolvedPrincipal,
    key: str,
    refs: list[dict[str, str | None]],
) -> dict[str, Any]:
    """Resolve secrets for one authenticated agent and return only sealed ciphertext."""
    public_key, key_id = _load_agent_public_key_binding(authority, actor=actor)
    secrets: list[dict[str, Any]] = []
    for ref in refs:
        name = str(ref.get("name") or "")
        scope = str(ref.get("scope") or "")
        agent_id = ref.get("agent_id")
        swarm_id = ref.get("swarm_id")
        row = _resolve_secret_row_for_agent_delivery(
            server,
            actor=actor,
            key=key,
            name=name,
            scope=scope,
            agent_id=agent_id,
            swarm_id=swarm_id,
        )
        ciphertext = _encrypt_secret_for_agent(str(row.get("value") or ""), public_key)
        _audit_agent_secret_delivery(
            principal_id=actor.principal_id,
            secret_id=str(row.get("secret_id") or ""),
            key=key,
            name=name,
            scope=scope,
            agent_id=agent_id,
            swarm_id=swarm_id,
            allowed=True,
            key_id=key_id,
        )
        item: dict[str, Any] = {
            "name": name,
            "scope": scope,
            "algorithm": _AGENT_SECRET_ALGORITHM,
            "key_id": key_id,
            "ciphertext": ciphertext,
        }
        if agent_id:
            item["agent_id"] = agent_id
        if swarm_id:
            item["swarm_id"] = swarm_id
        secrets.append(item)
    return {"secrets": secrets}
