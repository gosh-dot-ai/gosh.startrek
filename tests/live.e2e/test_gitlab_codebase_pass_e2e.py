# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tests._live_provider_stack import LiveProviderStack


pytestmark = pytest.mark.e2e
DEFAULT_LIVE_INFERENCE_MODEL = "qwen/qwen3-32b"
PUBLIC_GITLAB_REPO = "https://gitlab.com/gitlab-projects-templates/python/python-basic.git"


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip()


def _clone_public_gitlab_repo(tmp_path: Path) -> Path:
    repo_dir = tmp_path / "gitlab-python-basic"
    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "20",
            "--single-branch",
            PUBLIC_GITLAB_REPO,
            str(repo_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return repo_dir


def test_public_gitlab_codebase_pass_e2e(
    live_provider_stack: LiveProviderStack,
    tmp_path: Path,
) -> None:
    repo_root = _clone_public_gitlab_repo(tmp_path)
    agent_id = "alice"
    key = live_provider_stack.unique_key("gitlab_codebase_pass")

    live_provider_stack.set_config(key, agent_id=agent_id)
    ingest_result = live_provider_stack.agent_mcp(
        agent_id,
        "memory_ingest",
        {
            "key": key,
            "agent_id": agent_id,
            "scope": "agent-private",
            "path": str(repo_root),
            "source_id": "public_gitlab_repo",
        },
    )
    assert ingest_result["status"] == "ok", ingest_result
    assert ingest_result["source_family"] == "codebase"
    assert ingest_result["codebase_stage"] == "codebase_semantic"
    assert ingest_result["facts_extracted"] > 0
    assert ingest_result["object_count"] > 0
    assert ingest_result["relation_count"] > 0
    assert ingest_result["sidecar_count"] > 0
    assert "python" in ingest_result["languages"]
    assert "python_ast" in ingest_result["analyzers"]

    build_result = live_provider_stack.agent_mcp(
        agent_id,
        "memory_build_index",
        {"key": key, "agent_id": agent_id},
    )
    assert build_result["granular"] > 0, build_result

    recall_code = live_provider_stack.agent_mcp(
        agent_id,
        "memory_recall",
        {
            "key": key,
            "agent_id": agent_id,
            "query": "Show the exact code for _merge_conf_dicts.",
            "search_family": "codebase",
        },
    )
    runtime_trace = recall_code["runtime_trace"]
    trace = runtime_trace.get("codebase_context") or runtime_trace.get("codebase_augmentation") or {}
    assert recall_code["search_family"] == "codebase"
    assert recall_code["retrieval_families"] == ["codebase"]
    assert (
        "--- SOURCE FILES ---" in recall_code["context"]
        or "--- REPOSITORY CONTEXT PACK ---" in recall_code["context"]
    )
    assert "_merge_conf_dicts" in recall_code["context"]
    assert "skeleton/src/tools/config.py" in recall_code["context"]
    if trace.get("mode") in {"whole_file", "windowed_file"}:
        assert trace["selected_file"] == "skeleton/src/tools/config.py"
        assert trace["highlight_spans"], recall_code
    else:
        assert trace.get("mode") == "active", recall_code
        selected = trace.get("selected") or []
        assert any(item.get("path") == "skeleton/src/tools/config.py" for item in selected), recall_code

    ask_code = live_provider_stack.agent_mcp(
        agent_id,
        "memory_ask",
        {
            "key": key,
            "agent_id": agent_id,
            "query": "Which file defines _merge_conf_dicts? Answer with only the file path.",
            "search_family": "codebase",
            "inference_model": DEFAULT_LIVE_INFERENCE_MODEL,
        },
    )
    answer = str(ask_code.get("answer") or "").lower()
    assert "skeleton/src/tools/config.py" in answer
