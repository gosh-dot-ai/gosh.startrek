# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path

import pytest

from tests._live_provider_stack import LiveProviderStack


pytestmark = pytest.mark.e2e

DEFAULT_LIVE_INFERENCE_MODEL = "qwen/qwen3-32b"
MIXED_QUERY = (
    "Using all available memory sources, find the incident codename mentioned in chat, "
    "the owner team named in the document, and the exact Python qualified name of the "
    "cinder_signature function that returns the codename in code."
)
ASK_QUERY = (
    "Using all available memory sources, answer in exactly three labelled lines:\n"
    "chat_codename=<...>\n"
    "document_owner=<...>\n"
    "code_symbol=<...>\n"
    "Use the incident codename from chat, the owner team from the document, and the exact "
    "Python qualified name of the cinder_signature function that returns the codename in code."
)


def _build_live_codebase_repo(repo_root: Path) -> None:
    (repo_root / "pkg").mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=repo_root, check=True, capture_output=True, text=True)
    (repo_root / "pyproject.toml").write_text("[project]\nname='live_repo'\n", encoding="utf-8")
    (repo_root / "pkg" / "service.py").write_text(
        textwrap.dedent(
            """\
            def cinder_signature() -> str:
                return "sig:CINDER-42"
            """
        ),
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "pyproject.toml", "pkg/service.py"], cwd=repo_root, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "-c", "user.name=GOSH Test", "-c", "user.email=test@gosh.ai", "commit", "-m", "initial"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )


def test_codebase_mixed_retrieval_live(
    live_provider_stack: LiveProviderStack,
    tmp_path: Path,
) -> None:
    agent_id = "alice"
    key = live_provider_stack.unique_key("codebase_mixed_retrieval")
    repo_root = tmp_path / "cinder_live_repo"
    _build_live_codebase_repo(repo_root)

    live_provider_stack.set_config(key, agent_id=agent_id)

    store_result = live_provider_stack.agent_mcp(
        agent_id,
        "memory_store",
        {
            "key": key,
            "agent_id": agent_id,
            "scope": "agent-private",
            "source_id": "CHAT",
            "session_num": 1,
            "session_date": "2026-04-12",
            "content": (
                "User: The incident codename is CINDER-42. "
                "Assistant: Noted, I recorded CINDER-42 as the incident codename."
            ),
        },
    )
    assert store_result["facts_extracted"] > 0, store_result

    ingest_doc_result = live_provider_stack.agent_mcp(
        agent_id,
        "memory_ingest_document",
        {
            "key": key,
            "agent_id": agent_id,
            "scope": "agent-private",
            "source_id": "DOC",
            "content": "Runbook: the owner team for CINDER-42 is Platform Reliability.",
        },
    )
    assert ingest_doc_result["status"] == "ok", ingest_doc_result

    ingest_code_result = live_provider_stack.agent_mcp(
        agent_id,
        "memory_ingest",
        {
            "key": key,
            "agent_id": agent_id,
            "scope": "agent-private",
            "path": str(repo_root),
            "source_id": "CODE",
        },
    )
    assert ingest_code_result["status"] == "ok", ingest_code_result
    assert ingest_code_result["source_family"] == "codebase"

    build_result = live_provider_stack.agent_mcp(
        agent_id,
        "memory_build_index",
        {"key": key, "agent_id": agent_id},
    )
    assert build_result["granular"] > 0, build_result

    recall_result = live_provider_stack.agent_mcp(
        agent_id,
        "memory_recall",
        {
            "key": key,
            "agent_id": agent_id,
            "query": MIXED_QUERY,
        },
    )
    context = str(recall_result.get("context") or "")
    runtime_trace = dict(recall_result.get("runtime_trace") or {})
    mixed_trace = dict(runtime_trace.get("mixed_family_merge") or {})
    assert "CINDER-42" in context
    assert "Platform Reliability" in context
    assert "--- REPOSITORY CONTEXT PACK ---" in context
    assert "[File:" in context
    assert mixed_trace.get("mode") == "auto", recall_result
    assert {"episode", "codebase"} <= set(mixed_trace.get("lanes") or []), recall_result
    assert "codebase" in set(mixed_trace.get("merged_families") or []), recall_result
    assert "codebase" in set(recall_result.get("retrieval_families") or []), recall_result

    ask_result = live_provider_stack.agent_mcp(
        agent_id,
        "memory_ask",
        {
            "key": key,
            "agent_id": agent_id,
            "query": ASK_QUERY,
            "inference_model": DEFAULT_LIVE_INFERENCE_MODEL,
        },
    )
    answer = str(ask_result.get("answer") or "")
    answer_lower = answer.lower()
    assert "cinder-42" in answer_lower, ask_result
    assert "platform reliability" in answer_lower, ask_result
    assert "codebase" in set(ask_result.get("retrieval_families") or []), ask_result
    ask_trace = dict(ask_result.get("runtime_trace") or {})
    ask_mixed_trace = dict(ask_trace.get("mixed_family_merge") or {})
    assert ask_mixed_trace.get("mode") == "auto", ask_result
