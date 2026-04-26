# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

from __future__ import annotations

from collections.abc import Mapping

from ..episode_features import extract_query_features
from ..inference import get_inf_prompt
from .mapping import resolve_inference_prompt_key


def resolve_prompt_key(
    *,
    prompt_type: str,
    query: str,
    recall_result: dict,
    plugin_state: Mapping[str, bool] | None = None,
) -> str:
    query_features = extract_query_features(query)
    operator_plan = recall_result.get("query_operator_plan") or query_features.get("operator_plan") or {}
    state = dict(plugin_state or {})
    state.update(recall_result.get("inference_leaf_plugins") or {})
    return resolve_inference_prompt_key(
        prompt_type,
        operator_plan,
        plugin_state=state,
        query=query,
        recall_result=recall_result,
    )


def build_payload_messages(
    *,
    prompt_type: str,
    context: str,
    query: str,
    recall_result: dict,
    speakers: str,
    plugin_state: Mapping[str, bool] | None = None,
) -> list[dict]:
    prompt_key = resolve_prompt_key(
        prompt_type=prompt_type,
        query=query,
        recall_result=recall_result,
        plugin_state=plugin_state,
    )
    prompt = get_inf_prompt(prompt_key)
    formatted = prompt.format(
        context=context,
        question=query,
        speakers=speakers,
        sessions_in_context=recall_result.get("sessions_in_context", 0),
        total_sessions=recall_result.get("total_sessions", 0),
        coverage_pct=recall_result.get("coverage_pct", 100),
        reference_date="2023-01-01",
    )
    return [{"role": "user", "content": formatted}]
