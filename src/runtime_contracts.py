# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .memory import MemoryServer
    from .source_loader import LoadedSource


FactFilter = Callable[[dict], bool]

PROTECTED_OVERLAY_STATE_KEYS = frozenset({
    "_all_granular",
    "_episode_corpus",
    "_raw_sessions",
})


@runtime_checkable
class SourceAdapterV1(Protocol):
    family: str
    priority: int
    ingestable: bool
    retrievable: bool

    def supports_query(self, query: str) -> bool: ...

    async def ingest(
        self,
        server: MemoryServer,
        *,
        loaded: LoadedSource,
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
    ) -> dict: ...


@runtime_checkable
class OverlayPluginV1(Protocol):
    name: str
    priority: int

    def supports(self, *, family: str, metadata: dict | None = None) -> bool: ...

    def apply(
        self,
        *,
        family: str,
        metadata: dict | None,
        base_facts: Sequence[dict],
        extracted_facts: Sequence[dict],
    ) -> dict[str, list[dict]]: ...


@runtime_checkable
class QueryExecutorV1(Protocol):
    name: str
    priority: int

    def supports(
        self,
        query_type: str,
        search_family: str | None,
        packet: dict,
    ) -> bool: ...

    def should_skip(
        self,
        *,
        packet: dict,
        query: str,
        query_type: str,
        episode_lookup: dict[str, dict],
        augmented_facts: Sequence[dict] | None,
    ) -> bool: ...

    async def augment(
        self,
        server: MemoryServer,
        *,
        packet: dict,
        query: str,
        query_type: str,
        episode_lookup: dict[str, dict],
        fact_filter: FactFilter,
    ) -> tuple[dict, list[dict] | None]: ...

    def should_halt(
        self,
        *,
        packet: dict,
        query: str,
        query_type: str,
        episode_lookup: dict[str, dict],
        augmented_facts: Sequence[dict] | None,
    ) -> bool: ...


@runtime_checkable
class PromptHookV1(Protocol):
    name: str
    priority: int

    def applies(self, *, prompt_type: str, operator_plan: dict, recall_result: dict) -> bool: ...

    def resolve_prompt_key(
        self,
        *,
        prompt_type: str,
        operator_plan: dict,
        recall_result: dict,
        plugin_state: Mapping[str, bool] | None = None,
    ) -> str | None: ...


def validate_overlay_delta_only(delta: Mapping[str, Sequence[dict]]) -> dict[str, list[dict]]:
    keys = set(delta)
    for protected in PROTECTED_OVERLAY_STATE_KEYS:
        if protected in keys:
            raise ValueError(f"overlay plugins may not write protected runtime state: {protected}")
    if keys - {"facts", "temporal_links"}:
        unexpected = ", ".join(sorted(keys - {"facts", "temporal_links"}))
        raise ValueError(f"overlay plugins may only emit facts/temporal_links deltas, got {unexpected}")
    normalized = {
        "facts": list(delta.get("facts") or []),
        "temporal_links": list(delta.get("temporal_links") or []),
    }
    return normalized


def assert_overlay_post_extraction_only(*, base_facts: Sequence[dict], extracted_facts: Sequence[dict]) -> None:
    base_ids = [
        str(fact.get("id") or f"idx:{idx}")
        for idx, fact in enumerate(base_facts)
    ]
    extracted_ids = {
        str(fact.get("id") or f"idx:{idx}")
        for idx, fact in enumerate(extracted_facts)
    }
    missing = [fact_id for fact_id in base_ids if fact_id not in extracted_ids]
    if missing:
        raise ValueError("overlay plugins may not rewrite or erase extracted facts")
