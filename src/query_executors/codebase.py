# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

from __future__ import annotations

from ..codebase_query import augment_codebase_structural_packet
from ..runtime_contracts import QueryExecutorV1


class CodebaseQueryExecutor(QueryExecutorV1):
    name = "codebase_structural"
    priority = 350

    def supports(self, query_type: str, search_family: str | None, packet: dict) -> bool:
        if str(search_family or "").strip().lower() == "codebase":
            return True
        retrieval_families = [str(family or "").strip().lower() for family in (packet.get("retrieval_families") or [])]
        return "codebase" in retrieval_families

    def should_skip(
        self,
        *,
        packet: dict,
        query: str,
        query_type: str,
        episode_lookup: dict[str, dict],
        augmented_facts,
    ) -> bool:
        return False

    async def augment(self, server, *, packet: dict, query: str, query_type: str, episode_lookup: dict[str, dict], fact_filter):
        return await augment_codebase_structural_packet(
            server,
            query=query,
            packet=packet,
            episode_lookup=episode_lookup,
            fact_filter=fact_filter,
        )

    def should_halt(
        self,
        *,
        packet: dict,
        query: str,
        query_type: str,
        episode_lookup: dict[str, dict],
        augmented_facts,
    ) -> bool:
        return False
