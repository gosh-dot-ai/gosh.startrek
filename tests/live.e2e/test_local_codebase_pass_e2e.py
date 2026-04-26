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


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip()


def _build_local_fixture_repo(repo_root: Path) -> None:
    (repo_root / "src").mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=repo_root, check=True, capture_output=True, text=True)
    (repo_root / "pyproject.toml").write_text("[project]\nname='local-fixture'\n", encoding="utf-8")
    (repo_root / "src" / "runner.py").write_text(
        textwrap.dedent(
            """\
            def build_codebase_semantic_bundle(path: str) -> dict:
                return {"path": path, "status": "ok"}
            """
        ),
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "pyproject.toml", "src/runner.py"], cwd=repo_root, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "-c", "user.name=GOSH Test", "-c", "user.email=test@gosh.ai", "commit", "-m", "initial"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )


def test_local_gosh_memory_codebase_pass_e2e(live_provider_stack: LiveProviderStack, tmp_path: Path) -> None:
    repo_root = tmp_path / "local-fixture"
    _build_local_fixture_repo(repo_root)
    agent_id = "alice"
    key = live_provider_stack.unique_key("local_codebase_pass")

    live_provider_stack.set_config(key, agent_id=agent_id)
    ingest_result = live_provider_stack.agent_mcp(
        agent_id,
        "memory_ingest",
        {
            "key": key,
            "agent_id": agent_id,
            "scope": "agent-private",
            "path": str(repo_root),
            "source_id": "local_gosh_memory_repo",
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
            "query": "Show the exact code for build_codebase_semantic_bundle in src/runner.py.",
            "search_family": "codebase",
        },
    )
    runtime_trace = recall_code["runtime_trace"]
    trace = runtime_trace.get("codebase_context") or runtime_trace.get("codebase_augmentation") or {}
    assert recall_code["retrieval_families"] == ["codebase"]
    assert (
        "--- SOURCE FILES ---" in recall_code["context"]
        or "--- REPOSITORY CONTEXT PACK ---" in recall_code["context"]
    )
    assert "build_codebase_semantic_bundle" in recall_code["context"]
    assert "src/runner.py" in recall_code["context"]
    if trace.get("mode") in {"whole_file", "windowed_file"}:
        assert trace["selected_file"] == "src/runner.py"
        assert trace["highlight_spans"], recall_code
    else:
        assert trace.get("mode") == "active", recall_code
        selected = trace.get("selected") or []
        assert any(item.get("path") == "src/runner.py" for item in selected), recall_code

    ask_code = live_provider_stack.agent_mcp(
        agent_id,
        "memory_ask",
        {
            "key": key,
            "agent_id": agent_id,
            "query": "Which file defines build_codebase_semantic_bundle? Answer with only the file path.",
            "search_family": "codebase",
            "inference_model": DEFAULT_LIVE_INFERENCE_MODEL,
        },
    )
    answer = str(ask_code.get("answer") or "").lower()
    assert "src/runner.py" in answer
