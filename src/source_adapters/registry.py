# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

from __future__ import annotations

from ..runtime_contracts import SourceAdapterV1
from .codebase import CodebaseAdapter
from .conversation import ConversationAdapter
from .document import DocumentAdapter

_ADAPTERS: tuple[SourceAdapterV1, ...] = tuple(
    sorted(
        (ConversationAdapter(), DocumentAdapter(), CodebaseAdapter()),
        key=lambda adapter: adapter.priority,
        reverse=True,
    )
)
_ADAPTERS_BY_FAMILY = {adapter.family: adapter for adapter in _ADAPTERS}


def get_source_adapters() -> list[SourceAdapterV1]:
    return list(_ADAPTERS)


def get_source_adapter(family: str) -> SourceAdapterV1 | None:
    return _ADAPTERS_BY_FAMILY.get(str(family or "").strip().lower())


def registered_source_families() -> list[str]:
    return [adapter.family for adapter in _ADAPTERS if adapter.ingestable]


def registered_source_retrieval_families() -> list[str]:
    return [adapter.family for adapter in _ADAPTERS if adapter.retrievable]


def route_query_to_registered_families(
    query: str,
    available: list[str],
    explicit_family: str | None = None,
) -> list[str]:
    allowed = [
        family
        for family in available
        if family in registered_source_retrieval_families()
    ]
    if explicit_family and explicit_family != "auto":
        return [family for family in allowed if family == explicit_family]
    if len(allowed) <= 1:
        return allowed
    matched = [
        family
        for family in allowed
        if (adapter := get_source_adapter(family)) is not None and adapter.supports_query(query)
    ]
    if matched:
        return matched
    return []
