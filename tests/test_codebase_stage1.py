# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import requests

from src.codebase_hosting import fetch_hosting_stage1_snapshot
from src.codebase_ontology import hydrate_codebase_object_row
from src.episodes import build_episode_lookup
from src.ingest import ingest_input
from src.memory import MemoryServer
from src.query_executors.registry import get_default_query_executors, run_default_query_executor_chain
from src.source_adapters.codebase import CodebaseAdapter
from src.source_loader import LoadedSource


DIM = 32


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip()


def _token_vec(text: str, *, dim: int = DIM) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    for token in re.findall(r"[a-z0-9_./:-]+", (text or "").lower()):
        slot = hashlib.md5(token.encode("utf-8")).digest()[0] % dim
        vec[slot] += 1.0
    if not np.any(vec):
        vec[0] = 1.0
    return vec


def _patch_codebase_runtime(monkeypatch) -> None:
    async def mock_embed_texts(texts, **kwargs):
        return np.asarray([_token_vec(text) for text in texts], dtype=np.float32)

    async def mock_embed_query(text, **kwargs):
        return _token_vec(text)

    monkeypatch.setattr("src.memory.embed_texts", mock_embed_texts)
    monkeypatch.setattr("src.memory.embed_query", mock_embed_query)
    monkeypatch.setattr("src.memory.resolve_supersession", lambda facts, lookup: None)
    monkeypatch.setattr(MemoryServer, "_resolve_embedding_secret_ref", lambda self, **kwargs: None)


def _load_codebase_graph(server: MemoryServer, source_id: str) -> dict:
    source_record = server._source_records.get(source_id) or {}
    source_meta = dict(source_record.get("source_meta") or {})
    graph_ref = str(source_meta.get("codebase_graph_ref") or "")
    assert graph_ref
    return json.loads((server.data_dir / graph_ref).read_text(encoding="utf-8"))


def _load_codebase_relation_rows(server: MemoryServer, source_id: str) -> list[dict]:
    source_record = server._source_records.get(source_id) or {}
    source_meta = dict(source_record.get("source_meta") or {})
    relation_ref = str(source_meta.get("codebase_relation_ref") or "")
    assert relation_ref
    return [
        json.loads(line)
        for line in (server.data_dir / relation_ref).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_codebase_object_rows(server: MemoryServer, source_id: str) -> list[dict]:
    source_record = server._source_records.get(source_id) or {}
    source_meta = dict(source_record.get("source_meta") or {})
    object_ref = str(source_meta.get("codebase_object_ref") or "")
    assert object_ref
    return [
        hydrate_codebase_object_row(json.loads(line))
        for line in (server.data_dir / object_ref).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _all_codebase_object_rows(server: MemoryServer, source_id: str) -> list[dict]:
    rows = [
        fact
        for fact in server._all_granular
        if fact.get("kind") == "codebase_object"
        and str(fact.get("source_id") or "") == source_id
    ]
    rows.extend(_load_codebase_object_rows(server, source_id))
    return rows


def _attach_remote(repo: Path, remote_url: str) -> str:
    _git(repo, "remote", "add", "origin", remote_url)
    return remote_url


def _attach_github_remote(repo: Path, owner: str = "acme", name: str = "mini_repo") -> str:
    return _attach_remote(repo, f"https://github.com/{owner}/{name}.git")


def _attach_gitlab_remote(repo: Path, owner: str = "acme", name: str = "mini_repo") -> str:
    return _attach_remote(repo, f"https://gitlab.com/{owner}/{name}.git")


def _make_git_repo(
    tmp_path: Path,
    *,
    with_remote: bool = False,
    remote_provider: str = "github",
) -> tuple[Path, dict[str, str]]:
    repo = tmp_path / "mini_repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "tester@example.com")
    _git(repo, "config", "user.name", "Tester")

    (repo / "README.md").write_text("# Mini Repo\n\nStage 1 codebase ingest fixture.\n", encoding="utf-8")
    src_dir = repo / "src"
    src_dir.mkdir()
    (src_dir / "billing.py").write_text(
        "def parse_invoice_total(amount: float) -> float:\n"
        "    return amount * 1.07\n",
        encoding="utf-8",
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "Initial billing parser")
    _git(repo, "branch", "-M", "main")
    first_sha = _git(repo, "rev-parse", "HEAD")

    _git(repo, "checkout", "-qb", "feature/invoice")
    (src_dir / "billing.py").write_text(
        "def parse_invoice_total(amount: float) -> float:\n"
        "    taxed = amount * 1.07\n"
        "    return round(taxed, 2)\n",
        encoding="utf-8",
    )
    docs_dir = repo / "docs"
    docs_dir.mkdir()
    (docs_dir / "guide.md").write_text("Billing parser notes.\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "Update billing parser")
    feature_sha = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-q", "main")
    _git(repo, "merge", "--no-ff", "-qm", "Merge feature/invoice", "feature/invoice")
    head_sha = _git(repo, "rev-parse", "HEAD")

    if with_remote and remote_provider == "gitlab":
        remote_url = _attach_gitlab_remote(repo)
    else:
        remote_url = _attach_github_remote(repo) if with_remote else ""
    return repo, {
        "first_sha": first_sha,
        "feature_sha": feature_sha,
        "head_sha": head_sha,
        "head_short": head_sha[:12],
        "feature_short": feature_sha[:12],
        "branch": _git(repo, "rev-parse", "--abbrev-ref", "HEAD"),
        "remote_url": remote_url,
        "remote_provider": remote_provider if with_remote else "",
    }


def _make_history_repo(tmp_path: Path, *, commit_count: int) -> tuple[Path, dict[str, str]]:
    repo = tmp_path / "history_repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "tester@example.com")
    _git(repo, "config", "user.name", "Tester")

    log_file = repo / "history.txt"
    log_file.write_text("0\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "Commit 0")
    _git(repo, "branch", "-M", "main")
    commit_shas = [_git(repo, "rev-parse", "HEAD")]

    for idx in range(1, commit_count):
        log_file.write_text(log_file.read_text(encoding="utf-8") + f"{idx}\n", encoding="utf-8")
        _git(repo, "add", "history.txt")
        _git(repo, "commit", "-qm", f"Commit {idx}")
        commit_shas.append(_git(repo, "rev-parse", "HEAD"))

    return repo, {
        "head_sha": commit_shas[-1],
        "oldest_sha": commit_shas[0],
        "commit_count": str(commit_count),
    }


def _make_wide_diff_repo(tmp_path: Path, *, file_count: int) -> tuple[Path, dict[str, str]]:
    repo = tmp_path / "wide_repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "tester@example.com")
    _git(repo, "config", "user.name", "Tester")

    (repo / "README.md").write_text("# Wide Repo\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-qm", "Initial repo")
    _git(repo, "branch", "-M", "main")

    bulk_dir = repo / "bulk"
    bulk_dir.mkdir()
    target_path = ""
    for idx in range(file_count):
        rel_path = f"bulk/file_{idx:03d}.py"
        (repo / rel_path).write_text(f"VALUE_{idx} = {idx}\n", encoding="utf-8")
        target_path = rel_path
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", f"Add {file_count} generated files")
    return repo, {
        "head_sha": _git(repo, "rev-parse", "HEAD"),
        "target_path": target_path,
        "file_count": str(file_count),
    }


def _fake_hosting_snapshot(info: dict[str, str]) -> dict:
    return {
        "status": "ok",
        "provider": "github",
        "repo_full_name": "acme/mini_repo",
        "repo_meta": {
            "default_branch": "main",
            "html_url": "https://github.com/acme/mini_repo",
        },
        "pull_requests": [
            {
                "number": 17,
                "title": "Billing rounding fix",
                "body": "Rounds invoice totals and adds docs.",
                "state": "open",
                "created_at": "2024-01-03T10:00:00Z",
                "updated_at": "2024-01-04T09:00:00Z",
                "merged_at": "",
                "author": "reviewer",
                "base_branch": "main",
                "head_branch": "feature/invoice",
                "head_sha": info["feature_sha"],
                "labels": ["billing", "bug"],
                "assignees": ["alice"],
                "files": [
                    {"path": "src/billing.py", "status": "modified", "previous_filename": ""},
                    {"path": "docs/guide.md", "status": "added", "previous_filename": ""},
                ],
                "commits": [info["feature_sha"]],
                "reviews": [
                    {
                        "id": "501",
                        "state": "APPROVED",
                        "author": "reviewer",
                        "body": "Looks good overall.",
                        "submitted_at": "2024-01-04T08:55:00Z",
                        "commit_sha": info["feature_sha"],
                    }
                ],
                "review_comments": [
                    {
                        "id": "9001",
                        "review_id": "501",
                        "author": "reviewer",
                        "body": "Please add rounding tests.",
                        "created_at": "2024-01-04T08:56:00Z",
                        "path": "src/billing.py",
                        "commit_sha": info["feature_sha"],
                    }
                ],
                "check_runs": [
                    {
                        "id": "ci-test",
                        "name": "ci/test",
                        "status": "completed",
                        "conclusion": "success",
                        "started_at": "2024-01-04T08:00:00Z",
                        "completed_at": "2024-01-04T08:10:00Z",
                        "head_sha": info["feature_sha"],
                    }
                ],
                "statuses": [
                    {
                        "id": "lint",
                        "name": "lint",
                        "status": "success",
                        "conclusion": "Lint passed",
                        "created_at": "2024-01-04T08:05:00Z",
                        "updated_at": "2024-01-04T08:06:00Z",
                        "head_sha": info["feature_sha"],
                    }
                ],
            }
        ],
        "issues": [
            {
                "number": 9,
                "title": "Billing parser rounds incorrectly",
                "body": "Investigate rounding.",
                "state": "open",
                "created_at": "2024-01-02T12:00:00Z",
                "updated_at": "2024-01-03T12:00:00Z",
                "closed_at": "",
                "author": "maintainer",
                "labels": ["bug"],
                "assignees": ["alice"],
                "comments": [
                    {
                        "id": "issue-c1",
                        "author": "maintainer",
                        "body": "This looks related to PR 17.",
                        "created_at": "2024-01-03T12:05:00Z",
                        "updated_at": "2024-01-03T12:05:00Z",
                    }
                ],
            }
        ],
        "commit_comments": [
            {
                "id": "commit-c1",
                "author": "maintainer",
                "body": "Please keep the rounding behavior explicit.",
                "created_at": "2024-01-03T09:30:00Z",
                "updated_at": "2024-01-03T09:30:00Z",
                "commit_sha": info["feature_sha"],
                "path": "src/billing.py",
            }
        ],
        "releases": [
            {
                "id": "release-1",
                "tag_name": "v1.0.0",
                "name": "Version 1.0.0",
                "body": "First staged release.",
                "state": "published",
                "created_at": "2024-01-01T10:00:00Z",
                "published_at": "2024-01-01T12:00:00Z",
                "author": "maintainer",
                "assets": [
                    {
                        "id": "asset-1",
                        "name": "mini_repo.tar.gz",
                        "size": 1024,
                        "content_type": "application/gzip",
                        "created_at": "2024-01-01T12:00:00Z",
                        "updated_at": "2024-01-01T12:05:00Z",
                        "download_url": "https://github.com/acme/mini_repo/releases/download/v1.0.0/mini_repo.tar.gz",
                    }
                ],
            }
        ],
        "workflow_artifacts": [
            {
                "id": "artifact-coverage",
                "name": "coverage-report",
                "size_in_bytes": 2048,
                "created_at": "2024-01-04T08:15:00Z",
                "updated_at": "2024-01-04T08:20:00Z",
                "expired": False,
                "workflow_run_id": "run-17",
            }
        ],
    }


def _fake_gitlab_snapshot(info: dict[str, str]) -> dict:
    return {
        "status": "ok",
        "provider": "gitlab",
        "repo_full_name": "acme/mini_repo",
        "repo_meta": {
            "default_branch": "main",
            "html_url": "https://gitlab.com/acme/mini_repo",
        },
        "pull_requests": [],
        "merge_requests": [
            {
                "request_kind": "merge_request",
                "number": 23,
                "title": "Billing rounding fix",
                "body": "Rounds invoice totals and adds docs.",
                "state": "opened",
                "created_at": "2024-01-03T10:00:00Z",
                "updated_at": "2024-01-04T09:00:00Z",
                "merged_at": "",
                "author": "reviewer",
                "base_branch": "main",
                "head_branch": "feature/invoice",
                "head_sha": info["feature_sha"],
                "labels": ["billing", "bug"],
                "assignees": ["alice"],
                "files": [
                    {"path": "src/billing.py", "status": "modified", "previous_filename": ""},
                    {"path": "docs/guide.md", "status": "added", "previous_filename": ""},
                ],
                "commits": [info["feature_sha"]],
                "reviews": [
                    {
                        "id": "reviewer:alice",
                        "state": "requested",
                        "author": "alice",
                        "body": "GitLab reviewer assignment",
                        "submitted_at": "2024-01-04T08:55:00Z",
                        "commit_sha": info["feature_sha"],
                    }
                ],
                "review_comments": [
                    {
                        "id": "mr-note-1",
                        "review_id": "discussion-1",
                        "author": "reviewer",
                        "body": "Please add rounding tests.",
                        "created_at": "2024-01-04T08:56:00Z",
                        "path": "src/billing.py",
                        "commit_sha": info["feature_sha"],
                    }
                ],
                "check_runs": [
                    {
                        "id": "pipeline-77",
                        "name": "main",
                        "status": "success",
                        "conclusion": "success",
                        "started_at": "2024-01-04T08:00:00Z",
                        "completed_at": "2024-01-04T08:10:00Z",
                        "head_sha": info["feature_sha"],
                    }
                ],
                "statuses": [],
            }
        ],
        "issues": [
            {
                "number": 9,
                "title": "Billing parser rounds incorrectly",
                "body": "Investigate rounding.",
                "state": "opened",
                "created_at": "2024-01-02T12:00:00Z",
                "updated_at": "2024-01-03T12:00:00Z",
                "closed_at": "",
                "author": "maintainer",
                "labels": ["bug"],
                "assignees": ["alice"],
                "comments": [
                    {
                        "id": "issue-note-1",
                        "author": "maintainer",
                        "body": "This looks related to MR 23.",
                        "created_at": "2024-01-03T12:05:00Z",
                        "updated_at": "2024-01-03T12:05:00Z",
                    }
                ],
            }
        ],
        "commit_comments": [
            {
                "id": "commit-c1",
                "author": "maintainer",
                "body": "Please keep the rounding behavior explicit.",
                "created_at": "2024-01-03T09:30:00Z",
                "updated_at": "2024-01-03T09:30:00Z",
                "commit_sha": info["feature_sha"],
                "path": "src/billing.py",
            }
        ],
        "releases": [
            {
                "id": "v1.0.0",
                "tag_name": "v1.0.0",
                "name": "Version 1.0.0",
                "body": "First staged release.",
                "state": "published",
                "created_at": "2024-01-01T10:00:00Z",
                "published_at": "2024-01-01T12:00:00Z",
                "author": "maintainer",
                "assets": [
                    {
                        "id": "source-1",
                        "name": "mini_repo.tar.gz",
                        "size": 1024,
                        "content_type": "package",
                        "created_at": "2024-01-01T12:00:00Z",
                        "updated_at": "2024-01-01T12:05:00Z",
                        "download_url": "https://gitlab.com/acme/mini_repo/-/releases/v1.0.0/downloads/mini_repo.tar.gz",
                    }
                ],
            }
        ],
        "workflow_artifacts": [],
        "feature_statuses": {"review_comments": "ok", "issue_comments": "ok", "check_runs": "ok"},
    }


def _patch_hosting_snapshot(monkeypatch, snapshot: dict) -> None:
    monkeypatch.setattr(
        "src.codebase_hosting.fetch_hosting_stage1_snapshot",
        lambda **kwargs: deepcopy(snapshot),
    )


async def _run_codebase_executor(server: MemoryServer, query: str) -> tuple[dict, list[dict] | None]:
    episode_lookup = build_episode_lookup(server._episode_corpus)
    packet = {
        "search_family": "codebase",
        "retrieval_families": ["codebase"],
        "retrieved_episode_ids": list(episode_lookup),
        "selector_config": {"budget": 8000, "supporting_facts_total": 12},
        "tuning_snapshot": {"packet": {"snippet_chars": 1200}},
        "query_operator_plan": {},
        "retrieved_fact_ids": [],
        "actual_injected_episode_ids": [],
        "selection_scores": [],
    }
    return await run_default_query_executor_chain(
        server,
        query=query,
        query_type="lookup",
        packet=packet,
        episode_lookup=episode_lookup,
        fact_filter=lambda _fact: True,
    )


@pytest.mark.asyncio
async def test_codebase_adapter_does_not_call_ingest_document() -> None:
    calls: dict[str, object] = {}

    class FakeServer:
        async def ingest_document(self, **kwargs):  # pragma: no cover - should never run
            raise AssertionError("codebase adapter must not call ingest_document")

        async def ingest_codebase(self, **kwargs):
            calls.update(kwargs)
            return {"status": "ok", "facts_extracted": 7}

    loaded = LoadedSource(
        raw_text="src/billing.py\nREADME.md\n",
        transport="path",
        locator="/tmp/repo",
        filename="repo",
        mime="inode/directory",
        is_directory=True,
        is_repo=True,
        fetch_metadata={"entry_count": 2},
    )

    result = await CodebaseAdapter().ingest(
        FakeServer(),
        loaded=loaded,
        normalized_text=loaded.raw_text,
        metadata={"bench": "stage1"},
        retention_ttl=60,
        target=["agent:test"],
        agent_id="tester",
        swarm_id="default",
        scope="agent-private",
        source_id="mini_repo",
        owner_id="agent:test",
        read=[],
        write=[],
        caller_id="agent:test",
        caller_principal_kind="agent",
    )

    assert result["facts_extracted"] == 7
    assert calls["source_id"] == "mini_repo"
    assert calls["locator"] == "/tmp/repo"
    assert calls["filename"] == "repo"
    assert calls["mime"] == "inode/directory"


@pytest.mark.asyncio
async def test_codebase_ingest_creates_codebase_source_records_and_units(tmp_path, monkeypatch):
    _patch_codebase_runtime(monkeypatch)
    repo, info = _make_git_repo(tmp_path, with_remote=True)
    _patch_hosting_snapshot(monkeypatch, _fake_hosting_snapshot(info))
    server = MemoryServer(str(tmp_path), "codebase_stage1_runtime")

    result = await ingest_input(
        server,
        path=str(repo),
        scope="agent-private",
        source_id="mini_repo",
    )
    await server.build_index()
    graph = _load_codebase_graph(server, "mini_repo")
    object_rows = _all_codebase_object_rows(server, "mini_repo")

    object_types = {
        str(((fact.get("metadata") or {}).get("codebase") or {}).get("object_type") or "")
        for fact in object_rows
    }
    relation_types = {
        str(((fact.get("metadata") or {}).get("codebase") or {}).get("relation_type") or "")
        for fact in server._all_granular
        if fact.get("kind") == "codebase_relation"
    }
    relation_types.update(
        str(row.get("relation_type") or "").strip()
        for row in _load_codebase_relation_rows(server, "mini_repo")
        if str(row.get("relation_type") or "").strip()
    )

    assert result["source_family"] == "codebase"
    assert result["objects_extracted"] > 0
    assert result["hosting"]["status"] == "ok"
    assert result["head_commit"] == info["head_sha"]
    assert result["codebase_graph_ref"]
    assert server._source_records["mini_repo"]["family"] == "codebase"
    assert server._episode_corpus["documents"]
    assert all(
        episode.get("source_type") == "codebase"
        for doc in server._episode_corpus["documents"]
        for episode in doc.get("episodes", [])
    )
    assert all(session.get("format") == "codebase" for session in server._raw_sessions)
    assert {"repository", "worktree", "branch", "commit", "merge", "file", "directory", "diff", "hunk"} <= object_types
    assert {
        "pull_request",
        "review",
        "review_comment",
        "issue",
        "issue_comment",
        "commit_comment",
        "check_run",
        "release",
        "artifact",
        "label",
        "assignee",
    } <= object_types
    assert {
        "repo_has_branch",
        "branch_points_to_commit",
        "commit_modifies_file",
        "diff_contains_hunk",
        "pr_includes_commit",
        "review_comment_belongs_to_pr",
        "check_run_belongs_to_pr",
        "issue_comment_belongs_to_issue",
        "commit_comment_belongs_to_commit",
        "artifact_belongs_to_release",
    } <= relation_types
    assert graph["hosting"]["status"] == "ok"
    assert graph["object_counts"]["pull_request"] >= 1
    assert any(pr.get("number") == 17 for pr in graph["hosting"]["pull_requests"])


def test_fetch_hosting_stage1_snapshot_dispatches_gitlab_provider(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    def fake_gitlab_snapshot(repo_full_name: str, *, host: str) -> dict:
        calls.append((repo_full_name, host))
        return {"status": "ok", "provider": "gitlab", "repo_full_name": repo_full_name}

    monkeypatch.setattr("src.codebase_hosting._fetch_gitlab_snapshot", fake_gitlab_snapshot)

    result = fetch_hosting_stage1_snapshot(
        provider="gitlab",
        repo_owner="gitlab-org",
        repo_name="gitlab-test",
        host="gitlab.com",
    )

    assert result["status"] == "ok"
    assert calls == [("gitlab-org/gitlab-test", "gitlab.com")]


def test_fetch_hosting_stage1_snapshot_keeps_partial_gitlab_snapshot_when_optional_endpoint_is_rate_limited(monkeypatch) -> None:
    class _FakeResponse:
        def __init__(self, status_code: int, payload: object, *, headers: dict[str, str] | None = None):
            self.status_code = status_code
            self._payload = payload
            self.headers = dict(headers or {})

        def json(self):
            return self._payload

        def raise_for_status(self):
            if self.status_code >= 400:
                raise requests.HTTPError(f"status {self.status_code}", response=self)

    class _FakeGitLabSession:
        def get(self, url: str, params=None, timeout=60):
            del timeout
            endpoint = url.split("/api/v4", 1)[1]
            page = int((params or {}).get("page", 1))
            if endpoint == "/projects/acme%2Fmini_repo":
                return _FakeResponse(200, {"default_branch": "main", "web_url": "https://gitlab.com/acme/mini_repo"})
            if endpoint == "/projects/acme%2Fmini_repo/merge_requests":
                return _FakeResponse(
                    200,
                    [
                        {
                            "iid": 23,
                            "title": "Stage 1 MR",
                            "description": "Public merge request",
                            "state": "opened",
                            "created_at": "2024-02-01T10:00:00Z",
                            "updated_at": "2024-02-02T10:00:00Z",
                            "merged_at": "",
                            "author": {"username": "alice"},
                            "target_branch": "main",
                            "source_branch": "feature/codebase",
                            "diff_refs": {"head_sha": "abc123"},
                            "labels": ["stage1"],
                            "assignees": [{"username": "bob"}],
                            "reviewers": [{"username": "carol"}],
                        }
                    ] if page == 1 else [],
                )
            if endpoint == "/projects/acme%2Fmini_repo/issues":
                return _FakeResponse(200, [])
            if endpoint == "/projects/acme%2Fmini_repo/releases":
                return _FakeResponse(
                    200,
                    [
                        {
                            "tag_name": "v1.0.0",
                            "name": "v1.0.0",
                            "description": "First release",
                            "created_at": "2024-02-03T10:00:00Z",
                            "released_at": "2024-02-03T11:00:00Z",
                            "author": {"username": "release-bot"},
                            "assets": {"sources": [], "links": []},
                        }
                    ] if page == 1 else [],
                )
            if endpoint == "/projects/acme%2Fmini_repo/merge_requests/23":
                return _FakeResponse(
                    200,
                    {
                        "iid": 23,
                        "title": "Stage 1 MR",
                        "description": "Public merge request",
                        "state": "opened",
                        "created_at": "2024-02-01T10:00:00Z",
                        "updated_at": "2024-02-02T10:00:00Z",
                        "merged_at": "",
                        "author": {"username": "alice"},
                        "target_branch": "main",
                        "source_branch": "feature/codebase",
                        "diff_refs": {"head_sha": "abc123"},
                        "labels": ["stage1"],
                        "assignees": [{"username": "bob"}],
                        "reviewers": [{"username": "carol"}],
                    },
                )
            if endpoint == "/projects/acme%2Fmini_repo/merge_requests/23/changes":
                return _FakeResponse(200, {"changes": [{"new_path": "src/codebase.py", "old_path": "src/codebase.py"}]})
            if endpoint == "/projects/acme%2Fmini_repo/merge_requests/23/commits":
                return _FakeResponse(200, [{"id": "abc123"}] if page == 1 else [])
            if endpoint == "/projects/acme%2Fmini_repo/merge_requests/23/notes":
                return _FakeResponse(200, [])
            if endpoint == "/projects/acme%2Fmini_repo/merge_requests/23/pipelines":
                return _FakeResponse(200, [])
            if endpoint == "/projects/acme%2Fmini_repo/repository/commits/abc123/statuses":
                return _FakeResponse(200, [])
            if endpoint == "/projects/acme%2Fmini_repo/repository/commits/abc123/comments":
                return _FakeResponse(429, {"message": "Too Many Requests"})
            raise AssertionError((endpoint, params))

    monkeypatch.setattr("src.codebase_hosting._gitlab_token", lambda: "")
    monkeypatch.setattr("src.codebase_hosting._gitlab_session", lambda _token: _FakeGitLabSession())

    result = fetch_hosting_stage1_snapshot(
        provider="gitlab",
        repo_owner="acme",
        repo_name="mini_repo",
        host="gitlab.com",
    )

    assert result["status"] == "ok"
    assert result["provider"] == "gitlab"
    assert result["merge_requests"]
    assert result["releases"]
    assert result["feature_statuses"]["commit_comments"] == "unavailable_environment"
    assert result["commit_comments"] == []


def test_fetch_hosting_stage1_snapshot_reuses_local_commit_files_for_github(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    monkeypatch.setattr("src.codebase_hosting._github_token", lambda: "test-token")
    monkeypatch.setattr("src.codebase_hosting._github_session", lambda _token: object())

    def fake_get(_session, endpoint: str, *, params=None):
        calls.append(("get", endpoint))
        if endpoint == "/repos/acme/mini_repo":
            return {"default_branch": "main", "html_url": "https://github.com/acme/mini_repo"}
        if endpoint == "/repos/acme/mini_repo/commits/abc123/status":
            return {"statuses": []}
        raise AssertionError(endpoint)

    def fake_paginated(_session, endpoint: str, *, params=None, key=None):
        calls.append(("paginated", endpoint))
        if endpoint == "/repos/acme/mini_repo/pulls":
            return [
                {
                    "number": 17,
                    "title": "Add billing fix",
                    "body": "Uses local commit graph for files.",
                    "state": "open",
                    "created_at": "2024-01-01T10:00:00Z",
                    "updated_at": "2024-01-01T11:00:00Z",
                    "merged_at": "",
                    "user": {"login": "alice"},
                    "base": {"ref": "main"},
                    "head": {"ref": "feature/billing", "sha": "abc123"},
                    "labels": [],
                    "assignees": [],
                }
            ]
        if endpoint == "/repos/acme/mini_repo/issues":
            return []
        if endpoint == "/repos/acme/mini_repo/comments":
            return []
        if endpoint == "/repos/acme/mini_repo/releases":
            return []
        if endpoint == "/repos/acme/mini_repo/actions/artifacts":
            return []
        if endpoint == "/repos/acme/mini_repo/pulls/17/commits":
            return [{"sha": "abc123"}]
        if endpoint == "/repos/acme/mini_repo/pulls/17/files":
            raise AssertionError("github files endpoint should be skipped when local commit files are available")
        if endpoint == "/repos/acme/mini_repo/pulls/17/reviews":
            return []
        if endpoint == "/repos/acme/mini_repo/pulls/17/comments":
            return []
        if endpoint == "/repos/acme/mini_repo/commits/abc123/check-runs":
            return []
        raise AssertionError(endpoint)

    monkeypatch.setattr("src.codebase_hosting._github_get", fake_get)
    monkeypatch.setattr("src.codebase_hosting._github_paginated", fake_paginated)

    result = fetch_hosting_stage1_snapshot(
        provider="github",
        repo_owner="acme",
        repo_name="mini_repo",
        known_commit_files={
            "abc123": [
                {
                    "path": "src/billing.py",
                    "status": "modified",
                    "previous_filename": "",
                }
            ]
        },
    )

    assert result["status"] == "ok"
    assert result["pull_requests"][0]["files"] == [
        {
            "path": "src/billing.py",
            "status": "modified",
            "previous_filename": "",
        }
    ]
    assert ("paginated", "/repos/acme/mini_repo/pulls/17/files") not in calls


@pytest.mark.asyncio
async def test_codebase_ingest_creates_gitlab_hosting_units_via_builder(tmp_path, monkeypatch):
    _patch_codebase_runtime(monkeypatch)
    repo, info = _make_git_repo(tmp_path, with_remote=True, remote_provider="gitlab")
    _patch_hosting_snapshot(monkeypatch, _fake_gitlab_snapshot(info))
    server = MemoryServer(str(tmp_path), "codebase_gitlab_runtime")

    result = await ingest_input(
        server,
        path=str(repo),
        scope="agent-private",
        source_id="mini_repo_gitlab",
    )
    await server.build_index()
    graph = _load_codebase_graph(server, "mini_repo_gitlab")
    object_rows = _all_codebase_object_rows(server, "mini_repo_gitlab")

    object_types = {
        str(((fact.get("metadata") or {}).get("codebase") or {}).get("object_type") or "")
        for fact in object_rows
    }
    relation_types = {
        str(((fact.get("metadata") or {}).get("codebase") or {}).get("relation_type") or "")
        for fact in server._all_granular
        if fact.get("kind") == "codebase_relation"
    }

    assert result["hosting"]["status"] == "ok"
    assert result["hosting"]["provider"] == "gitlab"
    assert {"merge_request", "review", "review_comment", "issue", "check_run", "release", "artifact"} <= object_types
    assert {
        "merge_request_belongs_to_repo",
        "mr_includes_commit",
        "review_belongs_to_merge_request",
        "review_comment_belongs_to_merge_request",
        "check_run_belongs_to_merge_request",
    } <= relation_types
    assert graph["hosting"]["provider"] == "gitlab"
    assert graph["object_counts"]["merge_request"] >= 1
    assert any(mr.get("number") == 23 for mr in graph["hosting"]["merge_requests"])


@pytest.mark.asyncio
async def test_codebase_retrieval_family_routes_to_codebase_executor(tmp_path, monkeypatch):
    _patch_codebase_runtime(monkeypatch)
    repo, _info = _make_git_repo(tmp_path)
    server = MemoryServer(str(tmp_path), "codebase_route_runtime")

    await ingest_input(
        server,
        path=str(repo),
        scope="agent-private",
        source_id="mini_repo",
    )
    await server.build_index()
    recall_result = await server.recall(
        "Which commit modified src/billing.py?",
        search_family="codebase",
    )

    assert recall_result["search_family"] == "codebase"
    assert recall_result["retrieval_families"] == ["codebase"]
    assert any(executor.name == "codebase_structural" for executor in get_default_query_executors())


@pytest.mark.asyncio
async def test_codebase_executor_retrieves_commit_for_file(tmp_path, monkeypatch):
    _patch_codebase_runtime(monkeypatch)
    repo, info = _make_git_repo(tmp_path)
    server = MemoryServer(str(tmp_path), "codebase_commit_runtime")

    await ingest_input(
        server,
        path=str(repo),
        scope="agent-private",
        source_id="mini_repo",
    )
    await server.build_index()
    recall_result = await server.recall(
        "Which commit modifies file src/billing.py?",
        search_family="codebase",
    )

    assert recall_result["retrieval_families"] == ["codebase"]
    assert recall_result["retrieved"]
    assert "src/billing.py" in recall_result["context"]
    assert info["head_short"] in recall_result["context"] or info["head_sha"] in recall_result["context"]
    assert any("src/billing.py" in fact.get("fact", "") and "modifies" in fact.get("fact", "") for fact in recall_result["retrieved"])


@pytest.mark.asyncio
async def test_codebase_executor_links_commit_to_branch_and_pr(tmp_path, monkeypatch):
    _patch_codebase_runtime(monkeypatch)
    repo, info = _make_git_repo(tmp_path, with_remote=True)
    _patch_hosting_snapshot(monkeypatch, _fake_hosting_snapshot(info))
    server = MemoryServer(str(tmp_path), "codebase_branch_pr_runtime")

    await ingest_input(
        server,
        path=str(repo),
        scope="agent-private",
        source_id="mini_repo",
    )
    await server.build_index()

    packet, retrieved = await _run_codebase_executor(
        server,
        f"Which PR includes commit {info['feature_sha']} and which branch does it target?",
    )

    assert "pull request 17" in packet["context"].lower()
    assert "branch main" in packet["context"].lower()
    assert any("includes commit" in fact.get("fact", "") for fact in (retrieved or []))
    assert any("targets base branch main" in fact.get("fact", "").lower() for fact in (retrieved or []))


@pytest.mark.asyncio
async def test_codebase_executor_surfaces_review_comments_and_checks_for_pr(tmp_path, monkeypatch):
    _patch_codebase_runtime(monkeypatch)
    repo, info = _make_git_repo(tmp_path, with_remote=True)
    _patch_hosting_snapshot(monkeypatch, _fake_hosting_snapshot(info))
    server = MemoryServer(str(tmp_path), "codebase_reviews_runtime")

    await ingest_input(
        server,
        path=str(repo),
        scope="agent-private",
        source_id="mini_repo",
    )
    await server.build_index()

    packet, _retrieved = await _run_codebase_executor(
        server,
        "What review comments and checks are attached to PR 17?",
    )

    context_lower = packet["context"].lower()
    assert "please add rounding tests" in context_lower
    assert "ci/test" in context_lower
    assert "success" in context_lower


@pytest.mark.asyncio
async def test_codebase_path_uses_no_document_chunking_or_grouping(tmp_path, monkeypatch):
    _patch_codebase_runtime(monkeypatch)
    repo, _info = _make_git_repo(tmp_path)
    server = MemoryServer(str(tmp_path), "codebase_no_document_runtime")

    async def fail_ingest_document(*args, **kwargs):  # pragma: no cover - should never run
        raise AssertionError("codebase path must not route through ingest_document")

    monkeypatch.setattr(MemoryServer, "ingest_document", fail_ingest_document)
    monkeypatch.setattr("src.memory.segment_document_text", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("segment_document_text must not run")))
    monkeypatch.setattr("src.memory.group_document", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("group_document must not run")))

    result = await ingest_input(
        server,
        path=str(repo),
        scope="agent-private",
        source_id="mini_repo",
    )

    assert result["source_family"] == "codebase"


@pytest.mark.asyncio
async def test_codebase_ingest_persists_full_commit_graph_without_recent_commit_cap(tmp_path, monkeypatch):
    _patch_codebase_runtime(monkeypatch)
    repo, info = _make_history_repo(tmp_path, commit_count=30)
    server = MemoryServer(str(tmp_path), "codebase_history_runtime")

    result = await ingest_input(
        server,
        path=str(repo),
        scope="agent-private",
        source_id="history_repo",
    )
    graph = _load_codebase_graph(server, "history_repo")

    assert int(info["commit_count"]) == 30
    assert result["object_counts"]["commit"] == 30
    assert len(graph["commits"]) == 30
    assert graph["object_counts"]["commit"] == 30
    assert graph["commits"][-1]["sha"] == info["oldest_sha"]


@pytest.mark.asyncio
async def test_codebase_ingest_materializes_hunks_for_large_commit_without_cutoff(tmp_path, monkeypatch):
    _patch_codebase_runtime(monkeypatch)
    repo, info = _make_wide_diff_repo(tmp_path, file_count=140)
    server = MemoryServer(str(tmp_path), "codebase_large_hunks_runtime")

    result = await ingest_input(
        server,
        path=str(repo),
        scope="agent-private",
        source_id="wide_repo",
    )
    graph = _load_codebase_graph(server, "wide_repo")
    object_rows = _load_codebase_object_rows(server, "wide_repo")
    object_fact_counts = {
        object_type: sum(
            1
            for fact in object_rows
            if str(((fact.get("metadata") or {}).get("codebase") or {}).get("object_type") or "") == object_type
        )
        for object_type in ("diff", "hunk")
    }
    head_commit = next(commit for commit in graph["commits"] if commit["sha"] == info["head_sha"])
    diff_rows = [
        fact
        for fact in object_rows
        if str(((fact.get("metadata") or {}).get("codebase") or {}).get("object_type") or "") == "diff"
        and str(((fact.get("metadata") or {}).get("codebase") or {}).get("commit_sha") or "") == info["head_sha"]
    ]
    hunk_rows = [
        fact
        for fact in object_rows
        if str(((fact.get("metadata") or {}).get("codebase") or {}).get("object_type") or "") == "hunk"
        and str(((fact.get("metadata") or {}).get("codebase") or {}).get("commit_sha") or "") == info["head_sha"]
    ]
    target_diff = next(
        fact
        for fact in diff_rows
        if str(((fact.get("metadata") or {}).get("codebase") or {}).get("path") or "") == info["target_path"]
    )

    assert head_commit["changed_file_count"] == int(info["file_count"])
    assert len(diff_rows) == int(info["file_count"])
    assert len(hunk_rows) == int(info["file_count"])
    assert str(((target_diff.get("metadata") or {}).get("codebase") or {}).get("hydration_state") or "") == "hydrated"
    assert object_fact_counts["diff"] == result["object_counts"]["diff"] == graph["object_counts"]["diff"]
    assert object_fact_counts["hunk"] == result["object_counts"]["hunk"] == graph["object_counts"]["hunk"]
    assert object_fact_counts["diff"] > 128
    assert object_fact_counts["hunk"] > 128


@pytest.mark.asyncio
async def test_codebase_summary_counts_match_persisted_diff_units_without_global_cap(tmp_path, monkeypatch):
    _patch_codebase_runtime(monkeypatch)
    repo, info = _make_wide_diff_repo(tmp_path, file_count=520)
    server = MemoryServer(str(tmp_path), "codebase_wide_runtime")

    result = await ingest_input(
        server,
        path=str(repo),
        scope="agent-private",
        source_id="wide_repo_counts",
    )
    await server.build_index()
    graph = _load_codebase_graph(server, "wide_repo_counts")
    object_rows = _load_codebase_object_rows(server, "wide_repo_counts")
    diff_fact_count = sum(
        1
        for fact in object_rows
        if str(((fact.get("metadata") or {}).get("codebase") or {}).get("object_type") or "") == "diff"
    )

    recall_result = await server.recall(
        f"Which commit modified file {info['target_path']}?",
        search_family="codebase",
    )

    assert diff_fact_count == result["object_counts"]["diff"] == graph["object_counts"]["diff"]
    assert diff_fact_count > 512
    assert info["target_path"] in recall_result["context"]
    assert info["head_sha"][:12] in recall_result["context"] or info["head_sha"] in recall_result["context"]
    assert any(info["target_path"] in fact.get("fact", "") for fact in recall_result["retrieved"])
