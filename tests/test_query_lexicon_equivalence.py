# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

from __future__ import annotations

from typing import Any

import pytest

from src.codebase_semantic_runtime import classify_codebase_query_mode, query_requests_precise_code
from src.prompt_routing.mapping import resolve_inference_prompt_key
from src.memory import MemoryServer, _render_code_attachment_block


QUERY_LEXICON_CASES: list[dict[str, Any]] = [
    {
        "name": "exact_code_lookup",
        "query": "Show the exact code for issue signature",
        "retrieval_families": ["codebase"],
        "expected_precise_code": True,
        "expected_code_query_mode": "precise_code",
        "expected_prompt_key": "code_slot",
        "expected_file_lookup": None,
    },
    {
        "name": "qualified_name_lookup",
        "query": "What is the exact Python qualified name of the function?",
        "retrieval_families": ["codebase"],
        "expected_precise_code": True,
        "expected_code_query_mode": "precise_code",
        "expected_prompt_key": "code_slot",
        "expected_file_lookup": None,
    },
    {
        "name": "show_source_lookup",
        "query": "Show source for cinder_signature",
        "retrieval_families": ["codebase"],
        "expected_precise_code": True,
        "expected_code_query_mode": "precise_code",
        "expected_prompt_key": "code_slot",
        "expected_file_lookup": None,
    },
    {
        "name": "signature_lookup",
        "query": "What is the signature of cinder_signature?",
        "retrieval_families": ["codebase"],
        "expected_precise_code": True,
        "expected_code_query_mode": "precise_code",
        "expected_prompt_key": "code_slot",
        "expected_file_lookup": None,
    },
    {
        "name": "parameter_operator_priority",
        "query": "What parameters does this API take?",
        "retrieval_families": ["codebase"],
        "operator_plan": {
            "list_set": {"enabled": True},
            "ordinal": {"enabled": False},
            "commonality": {"enabled": False},
            "compare_diff": {"enabled": False},
            "bounded_chain": {"enabled": False},
        },
        "expected_precise_code": True,
        "expected_code_query_mode": "precise_code",
        "expected_prompt_key": "list_set",
        "expected_file_lookup": None,
    },
    {
        "name": "field_marker_without_operator_plan",
        "query": "Which field stores the owner team?",
        "retrieval_families": ["codebase"],
        "expected_precise_code": True,
        "expected_code_query_mode": "mixed_code_plus_prose",
        "expected_prompt_key": "codebase",
        "expected_file_lookup": None,
    },
    {
        "name": "body_lookup",
        "query": "Show me the implementation body",
        "retrieval_families": ["codebase"],
        "expected_precise_code": True,
        "expected_code_query_mode": "precise_code",
        "expected_prompt_key": "codebase",
        "expected_file_lookup": None,
    },
    {
        "name": "ast_lookup",
        "query": "Show the AST for this callable",
        "retrieval_families": ["codebase"],
        "expected_precise_code": True,
        "expected_code_query_mode": "precise_code",
        "expected_prompt_key": "code_slot",
        "expected_file_lookup": None,
    },
    {
        "name": "line_marker_positive",
        "query": "Show line 42 for cinder_signature",
        "retrieval_families": ["codebase"],
        "expected_precise_code": True,
        "expected_code_query_mode": "precise_code",
        "expected_prompt_key": "code_slot",
        "expected_file_lookup": None,
    },
    {
        "name": "chain_query_calls_this",
        "query": "What calls this?",
        "retrieval_families": ["codebase"],
        "expected_precise_code": False,
        "expected_code_query_mode": "non_code",
        "expected_prompt_key": "code_chain",
        "expected_file_lookup": None,
    },
    {
        "name": "chain_query_dependency_path",
        "query": "Trace the dependency path for this callable",
        "retrieval_families": ["codebase"],
        "expected_precise_code": False,
        "expected_code_query_mode": "non_code",
        "expected_prompt_key": "code_chain",
        "expected_file_lookup": None,
    },
    {
        "name": "risk_review_query",
        "query": "What breaks if we change this?",
        "retrieval_families": ["codebase"],
        "expected_precise_code": False,
        "expected_code_query_mode": "non_code",
        "expected_prompt_key": "risk_review",
        "expected_file_lookup": None,
    },
    {
        "name": "blast_radius_query",
        "query": "What is the blast radius of this change?",
        "retrieval_families": ["codebase"],
        "expected_precise_code": True,
        "expected_code_query_mode": "precise_code",
        "expected_prompt_key": "risk_review",
        "expected_file_lookup": None,
    },
    {
        "name": "mixed_chat_doc_code",
        "query": "Using all available memory sources, find the incident codename mentioned in chat, the owner team named in the document, and the exact Python qualified name.",
        "retrieval_families": ["conversation", "document", "codebase"],
        "runtime_code_query_mode": "mixed_code_plus_prose",
        "codebase_mode": "whole_file",
        "expected_precise_code": True,
        "expected_code_query_mode": "mixed_code_plus_prose",
        "expected_prompt_key": "codebase_mixed",
        "expected_file_lookup": None,
    },
    {
        "name": "mixed_file_subquestion_three_lines",
        "query": "Answer in exactly three labelled lines: chat_codename=<...> document_owner=<...> code_file=<which file defines the codename function>",
        "retrieval_families": ["conversation", "document", "codebase"],
        "runtime_code_query_mode": "mixed_code_plus_prose",
        "codebase_mode": "whole_file",
        "selected_file": "pkg/service.py",
        "expected_precise_code": False,
        "expected_code_query_mode": "mixed_code_plus_prose",
        "expected_prompt_key": "codebase_mixed",
        "expected_file_lookup": None,
    },
    {
        "name": "mixed_owner_team_blocks_shortcut",
        "query": "Which file defines cinder_signature and which owner team runs it?",
        "retrieval_families": ["conversation", "document", "codebase"],
        "runtime_code_query_mode": "mixed_code_plus_prose",
        "codebase_mode": "whole_file",
        "selected_file": "pkg/service.py",
        "expected_precise_code": True,
        "expected_code_query_mode": "mixed_code_plus_prose",
        "expected_prompt_key": "codebase_mixed",
        "expected_file_lookup": None,
    },
    {
        "name": "pure_file_lookup_answer_only",
        "query": "Which file defines cinder_signature? Answer with only the file path.",
        "retrieval_families": ["codebase"],
        "runtime_code_query_mode": "precise_code",
        "codebase_mode": "whole_file",
        "selected_file": "pkg/service.py",
        "expected_precise_code": True,
        "expected_code_query_mode": "precise_code",
        "expected_prompt_key": "code_slot",
        "expected_file_lookup": "pkg/service.py",
    },
    {
        "name": "what_file_lookup",
        "query": "What file defines cinder_signature?",
        "retrieval_families": ["codebase"],
        "runtime_code_query_mode": "precise_code",
        "codebase_mode": "whole_file",
        "selected_file": "pkg/service.py",
        "expected_precise_code": True,
        "expected_code_query_mode": "precise_code",
        "expected_prompt_key": "code_slot",
        "expected_file_lookup": "pkg/service.py",
    },
    {
        "name": "which_file_is_in_lookup",
        "query": "Which file is cinder_signature in?",
        "retrieval_families": ["codebase"],
        "runtime_code_query_mode": "precise_code",
        "codebase_mode": "whole_file",
        "selected_file": "pkg/service.py",
        "expected_precise_code": True,
        "expected_code_query_mode": "precise_code",
        "expected_prompt_key": "code_slot",
        "expected_file_lookup": None,
    },
    {
        "name": "which_file_with_punctuation",
        "query": "Which file defines cinder_signature, exactly?",
        "retrieval_families": ["codebase"],
        "runtime_code_query_mode": "precise_code",
        "codebase_mode": "whole_file",
        "selected_file": "pkg/service.py",
        "expected_precise_code": True,
        "expected_code_query_mode": "precise_code",
        "expected_prompt_key": "code_slot",
        "expected_file_lookup": "pkg/service.py",
    },
    {
        "name": "uppercase_file_lookup",
        "query": "WHICH FILE DEFINES CINDER_SIGNATURE?",
        "retrieval_families": ["codebase"],
        "runtime_code_query_mode": "precise_code",
        "codebase_mode": "whole_file",
        "selected_file": "pkg/service.py",
        "expected_precise_code": True,
        "expected_code_query_mode": "precise_code",
        "expected_prompt_key": "code_slot",
        "expected_file_lookup": "pkg/service.py",
    },
    {
        "name": "file_defines_without_prefix",
        "query": "file defines cinder_signature",
        "retrieval_families": ["codebase"],
        "runtime_code_query_mode": "precise_code",
        "codebase_mode": "whole_file",
        "selected_file": "pkg/service.py",
        "expected_precise_code": True,
        "expected_code_query_mode": "precise_code",
        "expected_prompt_key": "code_slot",
        "expected_file_lookup": None,
    },
    {
        "name": "newline_only_blocks_shortcut",
        "query": "Which file defines cinder_signature?\nReturn only the file path.",
        "retrieval_families": ["codebase"],
        "runtime_code_query_mode": "precise_code",
        "codebase_mode": "whole_file",
        "selected_file": "pkg/service.py",
        "expected_precise_code": True,
        "expected_code_query_mode": "precise_code",
        "expected_prompt_key": "code_slot",
        "expected_file_lookup": None,
    },
    {
        "name": "newline_labelled_query_blocks_shortcut",
        "query": "chat_codename=<...>\ndocument_owner=<...>\ncode_file=<which file defines cinder_signature>",
        "retrieval_families": ["conversation", "document", "codebase"],
        "runtime_code_query_mode": "mixed_code_plus_prose",
        "codebase_mode": "whole_file",
        "selected_file": "pkg/service.py",
        "expected_precise_code": True,
        "expected_code_query_mode": "mixed_code_plus_prose",
        "expected_prompt_key": "codebase_mixed",
        "expected_file_lookup": None,
    },
    {
        "name": "no_hit_requested_codebase_lane",
        "query": "What is the exact Python qualified name of the function?",
        "search_family": "codebase",
        "retrieval_families": [],
        "runtime_code_query_mode": "precise_code",
        "expected_precise_code": True,
        "expected_code_query_mode": "precise_code",
        "expected_prompt_key": "lookup",
        "expected_file_lookup": None,
    },
    {
        "name": "explicit_conversation_family_blocks_code_leafs",
        "query": "Show the exact code for cinder_signature",
        "search_family": "conversation",
        "retrieval_families": ["conversation", "codebase"],
        "runtime_code_query_mode": "precise_code",
        "codebase_mode": "whole_file",
        "expected_precise_code": True,
        "expected_code_query_mode": "precise_code",
        "expected_prompt_key": "lookup",
        "expected_file_lookup": None,
    },
    {
        "name": "explicit_document_family_blocks_code_leafs",
        "query": "Show the exact code for cinder_signature",
        "search_family": "document",
        "retrieval_families": ["document", "codebase"],
        "runtime_code_query_mode": "precise_code",
        "codebase_mode": "whole_file",
        "expected_precise_code": True,
        "expected_code_query_mode": "precise_code",
        "expected_prompt_key": "lookup",
        "expected_file_lookup": None,
    },
    {
        "name": "generic_codebase_recall",
        "query": "Summarize this module",
        "retrieval_families": ["codebase"],
        "codebase_mode": "whole_file",
        "expected_precise_code": False,
        "expected_code_query_mode": "non_code",
        "expected_prompt_key": "codebase",
        "expected_file_lookup": None,
    },
    {
        "name": "codebase_mode_without_retrieved_family_still_codebase",
        "query": "Summarize this module",
        "runtime_code_query_mode": "non_code",
        "codebase_mode": "whole_file",
        "expected_precise_code": False,
        "expected_code_query_mode": "non_code",
        "expected_prompt_key": "codebase",
        "expected_file_lookup": None,
    },
    {
        "name": "mixed_mode_without_code_markers_stays_lookup",
        "query": "Use chat and document to answer who owns CINDER-42.",
        "retrieval_families": ["conversation", "document", "codebase"],
        "runtime_code_query_mode": "non_code",
        "codebase_mode": "whole_file",
        "expected_precise_code": False,
        "expected_code_query_mode": "non_code",
        "expected_prompt_key": "lookup",
        "expected_file_lookup": None,
    },
    {
        "name": "slot_query_non_code",
        "query": "Which teammate owns the March runbook?",
        "operator_plan": {
            "slot_query": {"enabled": True},
            "list_set": {"enabled": False},
            "ordinal": {"enabled": False},
            "commonality": {"enabled": False},
            "compare_diff": {"enabled": False},
            "bounded_chain": {"enabled": False},
            "local_anchor": {"enabled": False},
            "temporal_grounding": {"enabled": False},
        },
        "expected_precise_code": False,
        "expected_code_query_mode": "non_code",
        "expected_prompt_key": "slot_query",
        "expected_file_lookup": None,
    },
    {
        "name": "lineage_is_not_line_marker",
        "query": "Give the lineage of CINDER-42",
        "expected_precise_code": False,
        "expected_code_query_mode": "non_code",
        "expected_prompt_key": "lookup",
        "expected_file_lookup": None,
    },
    {
        "name": "headline_is_not_lines_marker",
        "query": "Give the headline summary",
        "expected_precise_code": True,
        "expected_code_query_mode": "precise_code",
        "expected_prompt_key": "lookup",
        "expected_file_lookup": None,
    },
]


def _build_recall_result(case: dict[str, Any]) -> dict[str, Any]:
    runtime_trace: dict[str, Any] = {}
    query_trace: dict[str, Any] = {}
    if case.get("runtime_code_query_mode") is not None:
        query_trace["code_query_mode"] = case["runtime_code_query_mode"]
    if query_trace:
        runtime_trace["query"] = query_trace
    codebase_aug: dict[str, Any] = {}
    if case.get("codebase_mode") is not None:
        codebase_aug["mode"] = case["codebase_mode"]
    if case.get("selected_file") is not None:
        codebase_aug["selected_file"] = case["selected_file"]
    if codebase_aug:
        runtime_trace["codebase_augmentation"] = codebase_aug
    return {
        "search_family": case.get("search_family"),
        "retrieval_families": list(case.get("retrieval_families") or []),
        "runtime_trace": runtime_trace,
    }


def _deterministic_file_lookup(case: dict[str, Any]) -> str | None:
    server = object.__new__(MemoryServer)
    return MemoryServer._derive_codebase_file_lookup_deterministic_answer(
        server,
        case["query"],
        _build_recall_result(case),
    )


@pytest.mark.parametrize("case", QUERY_LEXICON_CASES, ids=[case["name"] for case in QUERY_LEXICON_CASES])
def test_query_requests_precise_code_equivalence(case: dict[str, Any]) -> None:
    assert query_requests_precise_code(case["query"]) is case["expected_precise_code"]


@pytest.mark.parametrize("case", QUERY_LEXICON_CASES, ids=[case["name"] for case in QUERY_LEXICON_CASES])
def test_classify_codebase_query_mode_equivalence(case: dict[str, Any]) -> None:
    assert classify_codebase_query_mode(case["query"]) == case["expected_code_query_mode"]


@pytest.mark.parametrize("case", QUERY_LEXICON_CASES, ids=[case["name"] for case in QUERY_LEXICON_CASES])
def test_codebase_file_lookup_shortcut_equivalence(case: dict[str, Any]) -> None:
    assert _deterministic_file_lookup(case) == case["expected_file_lookup"]


@pytest.mark.parametrize("case", QUERY_LEXICON_CASES, ids=[case["name"] for case in QUERY_LEXICON_CASES])
def test_inference_prompt_key_equivalence(case: dict[str, Any]) -> None:
    assert resolve_inference_prompt_key(
        case.get("prompt_type", "lookup"),
        case.get("operator_plan") or {},
        query=case["query"],
        recall_result=_build_recall_result(case),
    ) == case["expected_prompt_key"]


def test_code_attachment_label_remains_exact() -> None:
    rendered = _render_code_attachment_block([
        {
            "text": 'FILE: pkg/service.py\ndef cinder_signature() -> str:\n    return "sig:CINDER-42"',
            "rank": 0,
            "file_path": "pkg/service.py",
        }
    ])
    assert rendered == (
        '--- SOURCE FILES ---\n'
        'FILE: pkg/service.py\ndef cinder_signature() -> str:\n    return "sig:CINDER-42"'
    )
