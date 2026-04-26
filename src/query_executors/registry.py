# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

from __future__ import annotations

from collections.abc import Iterable

from ..runtime_contracts import QueryExecutorV1
from .codebase import CodebaseQueryExecutor
from .container_graph import ContainerGraphExecutor
from .conversation import ConversationStructuralExecutor
from .coverage import CoverageRecoveryExecutor
from .document import DocumentStructuralExecutor
from .semantic_rescue import SemanticRescueExecutor
from .temporal import TemporalExecutor

_BUILTIN_QUERY_EXECUTORS = (
    TemporalExecutor(),
    ContainerGraphExecutor(),
    ConversationStructuralExecutor(),
    CodebaseQueryExecutor(),
    DocumentStructuralExecutor(),
    CoverageRecoveryExecutor(),
    SemanticRescueExecutor(),
)
_REGISTERED_QUERY_EXECUTORS: dict[str, QueryExecutorV1] = {}


def register_query_executor(executor: QueryExecutorV1) -> QueryExecutorV1:
    _REGISTERED_QUERY_EXECUTORS[str(executor.name)] = executor
    return executor


def register_query_executors(executors: Iterable[QueryExecutorV1]) -> None:
    for executor in executors:
        register_query_executor(executor)


register_query_executors(_BUILTIN_QUERY_EXECUTORS)


def get_default_query_executors() -> list[QueryExecutorV1]:
    return sorted(
        _REGISTERED_QUERY_EXECUTORS.values(),
        key=lambda executor: executor.priority,
        reverse=True,
    )


def _executor_should_skip(
    executor,
    *,
    packet: dict,
    query: str,
    query_type: str,
    episode_lookup: dict[str, dict],
    augmented_facts: list[dict] | None,
) -> bool:
    should_skip = getattr(executor, "should_skip", None)
    if callable(should_skip):
        return bool(
            should_skip(
                packet=packet,
                query=query,
                query_type=query_type,
                episode_lookup=episode_lookup,
                augmented_facts=augmented_facts,
            )
        )
    return False


def _executor_should_halt(
    executor,
    *,
    packet: dict,
    query: str,
    query_type: str,
    episode_lookup: dict[str, dict],
    augmented_facts: list[dict] | None,
) -> bool:
    should_halt = getattr(executor, "should_halt", None)
    if callable(should_halt):
        return bool(
            should_halt(
                packet=packet,
                query=query,
                query_type=query_type,
                episode_lookup=episode_lookup,
                augmented_facts=augmented_facts,
            )
        )
    return False


async def run_default_query_executor_chain(
    server,
    *,
    query: str,
    query_type: str,
    packet: dict,
    episode_lookup: dict[str, dict],
    fact_filter,
) -> tuple[dict, list[dict] | None]:
    augmented_facts: list[dict] | None = None
    for executor in get_default_query_executors():
        if _executor_should_skip(
            executor,
            packet=packet,
            query=query,
            query_type=query_type,
            episode_lookup=episode_lookup,
            augmented_facts=augmented_facts,
        ):
            continue
        if not executor.supports(query_type, packet.get("search_family"), packet):
            continue
        packet, next_facts = await executor.augment(
            server,
            packet=packet,
            query=query,
            query_type=query_type,
            episode_lookup=episode_lookup,
            fact_filter=fact_filter,
        )
        if next_facts is not None:
            augmented_facts = next_facts
        if _executor_should_halt(
            executor,
            packet=packet,
            query=query,
            query_type=query_type,
            episode_lookup=episode_lookup,
            augmented_facts=augmented_facts,
        ):
            break
    return packet, augmented_facts
