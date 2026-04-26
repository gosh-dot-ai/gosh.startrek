#!/usr/bin/env python3
# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

from __future__ import annotations

from pathlib import Path

from .memory import LIVE_SCOPE_REQUIRED_ERROR
from .normalizer import normalize_text
from .source_adapters.registry import get_source_adapter
from .source_detect import detect_source_family
from .source_loader import load_source


async def ingest_input(
    server,
    text: str | None = None,
    path: str | None = None,
    url: str | None = None,
    metadata: dict | None = None,
    retention_ttl: int | None = None,
    target=None,
    agent_id: str | None = None,
    swarm_id: str | None = None,
    scope: str | None = None,
    source_id: str | None = None,
    owner_id: str | None = None,
    read: list[str] | None = None,
    write: list[str] | None = None,
    caller_id: str | None = None,
    caller_principal_kind: str | None = None,
    **kwargs,
) -> dict:
    """Unified ingest: load -> detect -> route."""
    if scope is None or str(scope).strip() == "":
        raise ValueError(LIVE_SCOPE_REQUIRED_ERROR)
    loaded = await load_source(text=text, path=path, url=url)
    family, evidence = detect_source_family(
        loaded.raw_text,
        filename=loaded.filename,
        mime=loaded.mime,
        is_directory=loaded.is_directory,
        is_repo=loaded.is_repo,
    )
    signals = list(evidence.get("signals", []))
    has_any_conversation_fields = any(
        kwargs.get(name) is not None
        for name in ("session_num", "session_date", "speakers")
    )
    if has_any_conversation_fields:
        if "conversation_fields_present" not in signals:
            signals.append("conversation_fields_present")
        evidence = {**evidence, "signals": signals}

    normalized_text = normalize_text(loaded.raw_text, family=family)

    adapter = get_source_adapter(family)
    if adapter is None or not adapter.ingestable:
        if family == "media":
            raise ValueError(
                f"Source family '{family}' is not yet supported. "
                "Only conversation, document, and codebase are implemented."
            )
        raise ValueError(
            f"Source family '{family}' is not registered for ingest"
        )
    if family == "conversation" and kwargs.get("session_num") is None:
        evidence.setdefault("signals", []).append("auto_session_num")
    result = await adapter.ingest(
        server,
        loaded=loaded,
        normalized_text=normalized_text,
        metadata=metadata,
        retention_ttl=retention_ttl,
        target=target,
        agent_id=agent_id,
        swarm_id=swarm_id,
        scope=scope,
        source_id=source_id,
        owner_id=owner_id,
        read=read,
        write=write,
        caller_id=caller_id,
        caller_principal_kind=caller_principal_kind,
        **kwargs,
    )

    result["source_family"] = family
    result["detection_evidence"] = evidence
    result["transport"] = loaded.transport
    result["locator"] = loaded.locator
    return result
