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

_CODEBASE_QUERY_RE = re.compile(
    r"\b("
    r"code|codebase|repo|repository|worktree|remote|branch|tag|commit|merge|sha|"
    r"diff|patch|hunk|file|directory|blob|tree|pull request|merge request|pr|mr|"
    r"review|comment|issue|release|artifact|check|ci|workflow"
    r")\b",
    re.I,
)


class CodebaseAdapter:
    family = "codebase"
    priority = 80
    ingestable = True
    retrievable = True

    def supports_query(self, query: str) -> bool:
        return bool(_CODEBASE_QUERY_RE.search(query or ""))

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
        sid = source_id or loaded.filename or "codebase"
        result = await server.ingest_codebase(
            repo_path=loaded.locator,
            locator=loaded.locator,
            content=normalized_text,
            filename=loaded.filename,
            mime=loaded.mime,
            source_id=sid,
            agent_id=agent_id,
            swarm_id=swarm_id,
            scope=scope,
            metadata=metadata,
            retention_ttl=retention_ttl,
            target=target,
            owner_id=owner_id,
            read=read,
            write=write,
            caller_id=caller_id,
            caller_principal_kind=caller_principal_kind,
            source_meta={
                "ingest_transport": loaded.transport,
                "ingest_locator": loaded.locator,
                "ingest_mime": loaded.mime,
                "is_directory": loaded.is_directory,
                "is_repo": loaded.is_repo,
                "fetch_metadata": dict(loaded.fetch_metadata or {}),
            },
        )
        if not isinstance(result, dict):
            result = {"status": "ok", "facts_extracted": int(result or 0)}
        result["source_id"] = sid
        return result
