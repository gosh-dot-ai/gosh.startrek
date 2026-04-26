# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

from __future__ import annotations

import re
from typing import Any

_CONVERSATION_QUERY_RE = re.compile(
    r"\b("
    r"did i|did we|what did .* say|who said|who told|when did .* mention|"
    r"remember|conversation|chat|we talked|we discussed|told me|mentioned to me"
    r")\b",
    re.I,
)


class ConversationAdapter:
    family = "conversation"
    priority = 100
    ingestable = True
    retrievable = True

    def supports_query(self, query: str) -> bool:
        return bool(_CONVERSATION_QUERY_RE.search(query or ""))

    async def ingest(
        self,
        server,
        *,
        loaded,
        normalized_text: str,
        metadata: dict | None,
        retention_ttl: int | None,
        target: Any,
        agent_id: str | None,
        swarm_id: str | None,
        scope: str | None,
        source_id: str | None,
        owner_id: str | None,
        read: list[str] | None,
        write: list[str] | None,
        caller_id: str | None,
        caller_principal_kind: str | None,
        **kwargs: Any,
    ) -> dict:
        session_num = kwargs.get("session_num")
        if session_num is None:
            session_nums = [
                rs.get("session_num", 0)
                for rs in getattr(server, "_raw_sessions", [])
                if isinstance(rs, dict) and isinstance(rs.get("session_num"), int)
            ]
            session_num = (max(session_nums) if session_nums else 0) + 1
        return await server.store(
            content=normalized_text,
            session_num=session_num,
            session_date=kwargs.get("session_date") or "",
            speakers=kwargs.get("speakers") or "User and Assistant",
            agent_id=agent_id,
            swarm_id=swarm_id,
            scope=scope,
            metadata=metadata,
            retention_ttl=retention_ttl,
            target=target,
            source_id=source_id or loaded.filename,
            owner_id=owner_id,
            read=read,
            write=write,
            caller_id=caller_id,
            caller_principal_kind=caller_principal_kind,
        )
