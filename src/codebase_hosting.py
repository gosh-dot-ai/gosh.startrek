# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

from __future__ import annotations

import os
import subprocess
from collections import Counter
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from urllib.parse import quote

import requests

from .codebase_ontology import (
    build_codebase_anchor,
    build_codebase_object_unit,
    build_codebase_relation_fact,
)

_GITHUB_API_ROOT = "https://api.github.com"
_MAX_HOSTING_DETAIL_WORKERS = 24
_GITLAB_OPTIONAL_UNAVAILABLE_STATUSES = {401, 403, 404, 408, 429, 500, 502, 503, 504}


def _detail_worker_count(item_count: int) -> int:
    if item_count <= 0:
        return 1
    cpu = max(1, os.cpu_count() or 1)
    ceiling = min(_MAX_HOSTING_DETAIL_WORKERS, max(4, cpu * 2))
    return max(1, min(ceiling, item_count))


def _dedupe_strings(values: Iterable[str] | None) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values or []:
        item = str(value or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _entity_name(entity: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = str((entity or {}).get(key) or "").strip()
        if value:
            return value
    return ""


def _is_open_state(value: str) -> bool:
    return str(value or "").strip().lower() in {"open", "opened"}


def _coalesce_hosting_files_from_known_commits(
    commit_shas: list[str],
    known_commit_files: dict[str, list[dict[str, str]]] | None,
) -> list[dict[str, str]] | None:
    if not commit_shas or not known_commit_files:
        return None
    merged: dict[str, dict[str, str]] = {}
    for commit_sha in commit_shas:
        file_rows = known_commit_files.get(str(commit_sha or "").strip())
        if file_rows is None:
            return None
        for file_row in file_rows:
            path = str(file_row.get("path") or "").strip()
            if not path:
                continue
            status = str(file_row.get("status") or "modified").strip() or "modified"
            previous_filename = str(file_row.get("previous_filename") or "").strip()
            current = merged.get(path)
            if current is None:
                merged[path] = {
                    "filename": path,
                    "status": status,
                    "previous_filename": previous_filename,
                }
                continue
            if status == "renamed":
                current["status"] = "renamed"
                current["previous_filename"] = previous_filename or current.get("previous_filename", "")
            elif current.get("status") != "renamed" and status in {"added", "deleted"}:
                current["status"] = status
    return [merged[path] for path in sorted(merged)]


def _github_token() -> str:
    for env_name in ("GITHUB_TOKEN", "GH_TOKEN", "GITHUB_API_TOKEN"):
        value = str(os.environ.get(env_name) or "").strip()
        if value:
            return value
    proc = subprocess.run(
        ["gh", "auth", "token"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode == 0:
        return str(proc.stdout or "").strip()
    return ""


def _gitlab_token() -> str:
    for env_name in ("GITLAB_TOKEN", "GLAB_TOKEN", "GITLAB_API_TOKEN"):
        value = str(os.environ.get(env_name) or "").strip()
        if value:
            return value
    try:
        proc = subprocess.run(
            ["glab", "auth", "token"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return ""
    if proc.returncode == 0:
        return str(proc.stdout or "").strip()
    return ""


def _github_session(token: str) -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "gosh-memory-codebase-pass/1.0",
        }
    )
    return session


def _github_get(session: requests.Session, endpoint: str, *, params: dict[str, Any] | None = None) -> Any:
    response = session.get(f"{_GITHUB_API_ROOT}{endpoint}", params=params, timeout=60)
    response.raise_for_status()
    return response.json()


def _github_paginated(
    session: requests.Session,
    endpoint: str,
    *,
    params: dict[str, Any] | None = None,
    key: str | None = None,
) -> list[dict[str, Any]]:
    url = f"{_GITHUB_API_ROOT}{endpoint}"
    query = dict(params or {})
    results: list[dict[str, Any]] = []
    while url:
        response = session.get(url, params=query or None, timeout=60)
        response.raise_for_status()
        payload = response.json()
        page_items: list[dict[str, Any]] = []
        if key is None:
            if isinstance(payload, list):
                page_items = [item for item in payload if isinstance(item, dict)]
        else:
            raw_items = payload.get(key) if isinstance(payload, dict) else []
            if isinstance(raw_items, list):
                page_items = [item for item in raw_items if isinstance(item, dict)]
        results.extend(page_items)
        query = {}
        url = response.links.get("next", {}).get("url") or ""
    return results


def _gitlab_api_root(host: str) -> str:
    host_name = str(host or "").strip() or "gitlab.com"
    scheme = "https://" if not host_name.startswith(("http://", "https://")) else ""
    return f"{scheme}{host_name.rstrip('/')}/api/v4"


def _gitlab_session(token: str) -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": "gosh-memory-codebase-pass/1.0"})
    if token:
        session.headers["PRIVATE-TOKEN"] = token
    return session


def _gitlab_get(
    session: requests.Session,
    api_root: str,
    endpoint: str,
    *,
    params: dict[str, Any] | None = None,
) -> Any:
    response = session.get(f"{api_root}{endpoint}", params=params, timeout=60)
    response.raise_for_status()
    return response.json()


def _gitlab_paginated(
    session: requests.Session,
    api_root: str,
    endpoint: str,
    *,
    params: dict[str, Any] | None = None,
    optional: bool = False,
) -> tuple[list[dict[str, Any]], str]:
    query = dict(params or {})
    query.setdefault("per_page", 100)
    results: list[dict[str, Any]] = []
    page = 1
    while True:
        page_query = dict(query)
        page_query["page"] = page
        try:
            response = session.get(f"{api_root}{endpoint}", params=page_query, timeout=60)
        except requests.RequestException:
            if optional:
                return [], "unavailable_environment"
            raise
        if optional and response.status_code in _GITLAB_OPTIONAL_UNAVAILABLE_STATUSES:
            return [], "unavailable_environment"
        response.raise_for_status()
        try:
            payload = response.json()
        except ValueError:
            if optional:
                return [], "unavailable_environment"
            raise
        page_items = payload if isinstance(payload, list) else []
        results.extend(item for item in page_items if isinstance(item, dict))
        next_page = str(response.headers.get("X-Next-Page") or "").strip()
        if not next_page:
            return results, "ok"
        try:
            page = int(next_page)
        except ValueError:
            if optional:
                return results, "unavailable_environment"
            raise


def _merge_feature_status(existing: str | None, incoming: str | None) -> str:
    current = str(existing or "").strip() or "ok"
    new = str(incoming or "").strip() or "ok"
    if current != "ok":
        return current
    return new


def _compact_pull_request(pr: dict[str, Any], *, files: list[dict[str, Any]], commits: list[dict[str, Any]], reviews: list[dict[str, Any]], review_comments: list[dict[str, Any]], check_runs: list[dict[str, Any]], statuses: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "number": int(pr.get("number") or 0),
        "title": str(pr.get("title") or "").strip(),
        "body": str(pr.get("body") or "").strip(),
        "state": str(pr.get("state") or "").strip(),
        "created_at": str(pr.get("created_at") or "").strip(),
        "updated_at": str(pr.get("updated_at") or "").strip(),
        "merged_at": str(pr.get("merged_at") or "").strip(),
        "author": str((pr.get("user") or {}).get("login") or "").strip(),
        "base_branch": str((pr.get("base") or {}).get("ref") or "").strip(),
        "head_branch": str((pr.get("head") or {}).get("ref") or "").strip(),
        "head_sha": str((pr.get("head") or {}).get("sha") or "").strip(),
        "labels": _dedupe_strings([str((label or {}).get("name") or "").strip() for label in (pr.get("labels") or []) if isinstance(label, dict)]),
        "assignees": _dedupe_strings([str((assignee or {}).get("login") or "").strip() for assignee in (pr.get("assignees") or []) if isinstance(assignee, dict)]),
        "files": [
            {
                "path": str(item.get("filename") or "").strip(),
                "status": str(item.get("status") or "").strip(),
                "previous_filename": str(item.get("previous_filename") or "").strip(),
            }
            for item in files
            if str(item.get("filename") or "").strip()
        ],
        "commits": [
            str((item or {}).get("sha") or "").strip()
            for item in commits
            if str((item or {}).get("sha") or "").strip()
        ],
        "reviews": [
            {
                "id": str(item.get("id") or "").strip(),
                "state": str(item.get("state") or "").strip(),
                "author": str((item.get("user") or {}).get("login") or "").strip(),
                "body": str(item.get("body") or "").strip(),
                "submitted_at": str(item.get("submitted_at") or "").strip(),
                "commit_sha": str(item.get("commit_id") or "").strip(),
            }
            for item in reviews
            if str(item.get("id") or "").strip()
        ],
        "review_comments": [
            {
                "id": str(item.get("id") or "").strip(),
                "review_id": str(item.get("pull_request_review_id") or "").strip(),
                "author": str((item.get("user") or {}).get("login") or "").strip(),
                "body": str(item.get("body") or "").strip(),
                "created_at": str(item.get("created_at") or "").strip(),
                "path": str(item.get("path") or "").strip(),
                "commit_sha": str(item.get("commit_id") or "").strip(),
            }
            for item in review_comments
            if str(item.get("id") or "").strip()
        ],
        "check_runs": [
            {
                "id": str(item.get("id") or "").strip(),
                "name": str(item.get("name") or "").strip(),
                "status": str(item.get("status") or "").strip(),
                "conclusion": str(item.get("conclusion") or "").strip(),
                "started_at": str(item.get("started_at") or "").strip(),
                "completed_at": str(item.get("completed_at") or "").strip(),
                "head_sha": str(item.get("head_sha") or "").strip(),
            }
            for item in check_runs
            if str(item.get("id") or "").strip()
        ],
        "statuses": [
            {
                "id": str(item.get("id") or "").strip() or f"status:{str(item.get('context') or '').strip()}:{str(item.get('sha') or '').strip()}",
                "name": str(item.get("context") or "").strip(),
                "status": str(item.get("state") or "").strip(),
                "conclusion": str(item.get("description") or "").strip(),
                "created_at": str(item.get("created_at") or "").strip(),
                "updated_at": str(item.get("updated_at") or "").strip(),
                "head_sha": str(item.get("sha") or "").strip(),
            }
            for item in statuses
            if str(item.get("context") or "").strip()
        ],
    }


def _compact_issue(issue: dict[str, Any], *, comments: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "number": int(issue.get("number") or 0),
        "title": str(issue.get("title") or "").strip(),
        "body": str(issue.get("body") or "").strip(),
        "state": str(issue.get("state") or "").strip(),
        "created_at": str(issue.get("created_at") or "").strip(),
        "updated_at": str(issue.get("updated_at") or "").strip(),
        "closed_at": str(issue.get("closed_at") or "").strip(),
        "author": str((issue.get("user") or {}).get("login") or "").strip(),
        "labels": _dedupe_strings([str((label or {}).get("name") or "").strip() for label in (issue.get("labels") or []) if isinstance(label, dict)]),
        "assignees": _dedupe_strings([str((assignee or {}).get("login") or "").strip() for assignee in (issue.get("assignees") or []) if isinstance(assignee, dict)]),
        "comments": [
            {
                "id": str(item.get("id") or "").strip(),
                "author": str((item.get("user") or {}).get("login") or "").strip(),
                "body": str(item.get("body") or "").strip(),
                "created_at": str(item.get("created_at") or "").strip(),
                "updated_at": str(item.get("updated_at") or "").strip(),
            }
            for item in comments
            if str(item.get("id") or "").strip()
        ],
    }


def _fetch_github_snapshot(
    repo_full_name: str,
    *,
    known_commit_files: dict[str, list[dict[str, str]]] | None = None,
) -> dict[str, Any]:
    token = _github_token()
    if not token:
        return {
            "status": "unavailable_environment",
            "provider": "github",
            "repo_full_name": repo_full_name,
            "reason": "No GitHub token or gh auth token available",
        }
    try:
        session = _github_session(token)
        repo_meta = _github_get(session, f"/repos/{repo_full_name}")
        with ThreadPoolExecutor(max_workers=4) as pool:
            pulls_future = pool.submit(
                _github_paginated,
                _github_session(token),
                f"/repos/{repo_full_name}/pulls",
                params={"state": "all", "per_page": 100},
            )
            issues_future = pool.submit(
                _github_paginated,
                _github_session(token),
                f"/repos/{repo_full_name}/issues",
                params={"state": "all", "per_page": 100},
            )
            commit_comments_future = pool.submit(
                _github_paginated,
                _github_session(token),
                f"/repos/{repo_full_name}/comments",
                params={"per_page": 100},
            )
            releases_future = pool.submit(
                _github_paginated,
                _github_session(token),
                f"/repos/{repo_full_name}/releases",
                params={"per_page": 100},
            )
            workflow_artifacts_future = pool.submit(
                _github_paginated,
                _github_session(token),
                f"/repos/{repo_full_name}/actions/artifacts",
                params={"per_page": 100},
                key="artifacts",
            )
            pulls = pulls_future.result()

        def _fetch_pull_request(pr: dict[str, Any]) -> dict[str, Any]:
            number = int(pr.get("number") or 0)
            task_session = _github_session(token)
            commits = _github_paginated(task_session, f"/repos/{repo_full_name}/pulls/{number}/commits", params={"per_page": 100})
            commit_shas = [
                str((item or {}).get("sha") or "").strip()
                for item in commits
                if str((item or {}).get("sha") or "").strip()
            ]
            files = _coalesce_hosting_files_from_known_commits(commit_shas, known_commit_files)
            if files is None:
                files = _github_paginated(
                    task_session,
                    f"/repos/{repo_full_name}/pulls/{number}/files",
                    params={"per_page": 100},
                )
            reviews = _github_paginated(task_session, f"/repos/{repo_full_name}/pulls/{number}/reviews", params={"per_page": 100})
            review_comments = _github_paginated(task_session, f"/repos/{repo_full_name}/pulls/{number}/comments", params={"per_page": 100})
            head_sha = str((pr.get("head") or {}).get("sha") or "").strip()
            check_runs = _github_paginated(
                task_session,
                f"/repos/{repo_full_name}/commits/{head_sha}/check-runs",
                params={"per_page": 100},
                key="check_runs",
            ) if head_sha else []
            combined_status = _github_get(task_session, f"/repos/{repo_full_name}/commits/{head_sha}/status") if head_sha else {}
            return _compact_pull_request(
                pr,
                files=files,
                commits=commits,
                reviews=reviews,
                review_comments=review_comments,
                check_runs=check_runs,
                statuses=list(combined_status.get("statuses") or []),
            )
        with ThreadPoolExecutor(max_workers=_detail_worker_count(len(pulls))) as pool:
            compact_pulls = list(pool.map(_fetch_pull_request, pulls))

        issue_rows = issues_future.result()
        pure_issues = [issue for issue in issue_rows if not isinstance(issue.get("pull_request"), dict)]

        def _fetch_issue(issue: dict[str, Any]) -> dict[str, Any]:
            number = int(issue.get("number") or 0)
            task_session = _github_session(token)
            comments = _github_paginated(task_session, f"/repos/{repo_full_name}/issues/{number}/comments", params={"per_page": 100})
            return _compact_issue(issue, comments=comments)
        with ThreadPoolExecutor(max_workers=_detail_worker_count(len(pure_issues))) as pool:
            compact_issues = list(pool.map(_fetch_issue, pure_issues))

        commit_comments = commit_comments_future.result()
        releases = releases_future.result()
        workflow_artifacts = workflow_artifacts_future.result()
        return {
            "status": "ok",
            "provider": "github",
            "repo_full_name": repo_full_name,
            "repo_meta": {
                "default_branch": str(repo_meta.get("default_branch") or "").strip(),
                "html_url": str(repo_meta.get("html_url") or "").strip(),
            },
            "pull_requests": compact_pulls,
            "issues": compact_issues,
            "commit_comments": [
                {
                    "id": str(item.get("id") or "").strip(),
                    "author": str((item.get("user") or {}).get("login") or "").strip(),
                    "body": str(item.get("body") or "").strip(),
                    "created_at": str(item.get("created_at") or "").strip(),
                    "updated_at": str(item.get("updated_at") or "").strip(),
                    "commit_sha": str(item.get("commit_id") or "").strip(),
                    "path": str(item.get("path") or "").strip(),
                }
                for item in commit_comments
                if str(item.get("id") or "").strip()
            ],
            "releases": [
                {
                    "id": str(item.get("id") or "").strip(),
                    "tag_name": str(item.get("tag_name") or "").strip(),
                    "name": str(item.get("name") or "").strip(),
                    "body": str(item.get("body") or "").strip(),
                    "state": str((item.get("draft") and "draft") or (item.get("prerelease") and "prerelease") or "published"),
                    "created_at": str(item.get("created_at") or "").strip(),
                    "published_at": str(item.get("published_at") or "").strip(),
                    "author": str((item.get("author") or {}).get("login") or "").strip(),
                    "assets": [
                        {
                            "id": str(asset.get("id") or "").strip(),
                            "name": str(asset.get("name") or "").strip(),
                            "size": int(asset.get("size") or 0),
                            "content_type": str(asset.get("content_type") or "").strip(),
                            "created_at": str(asset.get("created_at") or "").strip(),
                            "updated_at": str(asset.get("updated_at") or "").strip(),
                            "download_url": str(asset.get("browser_download_url") or "").strip(),
                        }
                        for asset in (item.get("assets") or [])
                        if isinstance(asset, dict) and str(asset.get("id") or "").strip()
                    ],
                }
                for item in releases
                if str(item.get("id") or "").strip()
            ],
            "workflow_artifacts": [
                {
                    "id": str(item.get("id") or "").strip(),
                    "name": str(item.get("name") or "").strip(),
                    "size_in_bytes": int(item.get("size_in_bytes") or 0),
                    "created_at": str(item.get("created_at") or "").strip(),
                    "updated_at": str(item.get("updated_at") or "").strip(),
                    "expired": bool(item.get("expired")),
                    "workflow_run_id": str((item.get("workflow_run") or {}).get("id") or "").strip(),
                }
                for item in workflow_artifacts
                if str(item.get("id") or "").strip()
            ],
        }
    except Exception as exc:
        return {
            "status": "unavailable_environment",
            "provider": "github",
            "repo_full_name": repo_full_name,
            "reason": str(exc),
        }


def _gitlab_change_status(change: dict[str, Any]) -> str:
    if bool(change.get("deleted_file")):
        return "deleted"
    if bool(change.get("renamed_file")):
        return "renamed"
    if bool(change.get("new_file")):
        return "added"
    return "modified"


def _compact_gitlab_merge_request(
    mr: dict[str, Any],
    *,
    changes: list[dict[str, Any]],
    commits: list[dict[str, Any]],
    notes: list[dict[str, Any]],
    pipelines: list[dict[str, Any]],
    commit_statuses: list[dict[str, Any]],
) -> dict[str, Any]:
    reviewers = [
        {
            "id": f"{int(mr.get('iid') or 0)}:{_entity_name(item, 'username', 'name')}",
            "state": "requested",
            "author": _entity_name(item, "username", "name"),
            "body": "GitLab reviewer assignment",
            "submitted_at": str(mr.get("updated_at") or mr.get("created_at") or "").strip(),
            "commit_sha": str((mr.get("diff_refs") or {}).get("head_sha") or mr.get("sha") or "").strip(),
        }
        for item in (mr.get("reviewers") or [])
        if _entity_name(item, "username", "name")
    ]
    review_comments = [
        {
            "id": str(item.get("id") or "").strip(),
            "review_id": str(item.get("discussion_id") or "").strip(),
            "author": _entity_name(item.get("author") or {}, "username", "name"),
            "body": str(item.get("body") or "").strip(),
            "created_at": str(item.get("created_at") or "").strip(),
            "path": str(item.get("path") or "").strip(),
            "commit_sha": str(item.get("commit_sha") or "").strip(),
        }
        for item in notes
        if str(item.get("id") or "").strip() and str(item.get("body") or "").strip()
    ]
    return {
        "request_kind": "merge_request",
        "number": int(mr.get("iid") or 0),
        "title": str(mr.get("title") or "").strip(),
        "body": str(mr.get("description") or "").strip(),
        "state": str(mr.get("state") or "").strip(),
        "created_at": str(mr.get("created_at") or "").strip(),
        "updated_at": str(mr.get("updated_at") or "").strip(),
        "merged_at": str(mr.get("merged_at") or "").strip(),
        "author": _entity_name(mr.get("author") or {}, "username", "name"),
        "base_branch": str(mr.get("target_branch") or "").strip(),
        "head_branch": str(mr.get("source_branch") or "").strip(),
        "head_sha": str((mr.get("diff_refs") or {}).get("head_sha") or mr.get("sha") or "").strip(),
        "labels": _dedupe_strings(str(label or "").strip() for label in (mr.get("labels") or [])),
        "assignees": _dedupe_strings(
            _entity_name(item, "username", "name")
            for item in (mr.get("assignees") or [])
            if isinstance(item, dict)
        ),
        "files": [
            {
                "path": str(item.get("new_path") or item.get("old_path") or "").strip(),
                "status": _gitlab_change_status(item),
                "previous_filename": str(item.get("old_path") or "").strip()
                if bool(item.get("renamed_file"))
                else "",
            }
            for item in changes
            if str(item.get("new_path") or item.get("old_path") or "").strip()
        ],
        "commits": [
            str(item.get("id") or "").strip()
            for item in commits
            if str(item.get("id") or "").strip()
        ],
        "reviews": reviewers,
        "review_comments": review_comments,
        "check_runs": [
            {
                "id": str(item.get("id") or "").strip(),
                "name": str(item.get("ref") or item.get("source") or "pipeline").strip(),
                "status": str(item.get("status") or "").strip(),
                "conclusion": str(item.get("status") or "").strip(),
                "started_at": str(item.get("created_at") or "").strip(),
                "completed_at": str(item.get("updated_at") or "").strip(),
                "head_sha": str(item.get("sha") or "").strip(),
            }
            for item in pipelines
            if str(item.get("id") or "").strip()
        ],
        "statuses": [
            {
                "id": str(item.get("id") or "").strip()
                or f"status:{str(item.get('name') or '').strip()}:{str(item.get('sha') or '').strip()}",
                "name": str(item.get("name") or item.get("status") or "").strip(),
                "status": str(item.get("status") or "").strip(),
                "conclusion": str((item.get("allow_failure") and "allow_failure") or item.get("status") or "").strip(),
                "created_at": str(item.get("created_at") or "").strip(),
                "updated_at": str(item.get("finished_at") or item.get("created_at") or "").strip(),
                "head_sha": str(item.get("sha") or "").strip(),
            }
            for item in commit_statuses
            if str(item.get("name") or item.get("status") or "").strip()
        ],
    }


def _compact_gitlab_issue(issue: dict[str, Any], *, comments: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "number": int(issue.get("iid") or 0),
        "title": str(issue.get("title") or "").strip(),
        "body": str(issue.get("description") or "").strip(),
        "state": str(issue.get("state") or "").strip(),
        "created_at": str(issue.get("created_at") or "").strip(),
        "updated_at": str(issue.get("updated_at") or "").strip(),
        "closed_at": str(issue.get("closed_at") or "").strip(),
        "author": _entity_name(issue.get("author") or {}, "username", "name"),
        "labels": _dedupe_strings(str(label or "").strip() for label in (issue.get("labels") or [])),
        "assignees": _dedupe_strings(
            _entity_name(item, "username", "name")
            for item in (issue.get("assignees") or [])
            if isinstance(item, dict)
        ),
        "comments": [
            {
                "id": str(item.get("id") or "").strip(),
                "author": _entity_name(item.get("author") or {}, "username", "name"),
                "body": str(item.get("body") or "").strip(),
                "created_at": str(item.get("created_at") or "").strip(),
                "updated_at": str(item.get("updated_at") or "").strip(),
            }
            for item in comments
            if str(item.get("id") or "").strip()
        ],
    }


def _gitlab_release_assets(release: dict[str, Any]) -> list[dict[str, Any]]:
    assets = dict(release.get("assets") or {})
    rows: list[dict[str, Any]] = []
    for idx, source in enumerate(assets.get("sources") or [], start=1):
        url = str(source.get("url") or "").strip()
        if not url:
            continue
        rows.append(
            {
                "id": f"{release.get('tag_name')}:source:{idx}",
                "name": str(source.get("format") or f"source-{idx}").strip(),
                "size": 0,
                "content_type": "application/octet-stream",
                "created_at": str(release.get("created_at") or "").strip(),
                "updated_at": str(release.get("released_at") or release.get("created_at") or "").strip(),
                "download_url": url,
            }
        )
    for idx, link in enumerate(assets.get("links") or [], start=1):
        url = str(link.get("url") or "").strip() or str(link.get("direct_asset_url") or "").strip()
        if not url:
            continue
        rows.append(
            {
                "id": str(link.get("id") or f"{release.get('tag_name')}:link:{idx}").strip(),
                "name": str(link.get("name") or f"link-{idx}").strip(),
                "size": 0,
                "content_type": str(link.get("link_type") or "link").strip(),
                "created_at": str(release.get("created_at") or "").strip(),
                "updated_at": str(release.get("released_at") or release.get("created_at") or "").strip(),
                "download_url": url,
            }
        )
    return rows


def _fetch_gitlab_snapshot(repo_full_name: str, *, host: str) -> dict[str, Any]:
    api_root = _gitlab_api_root(host)
    token = _gitlab_token()
    session = _gitlab_session(token)
    project_id = quote(repo_full_name, safe="")
    feature_statuses: dict[str, str] = {}
    try:
        repo_meta = _gitlab_get(session, api_root, f"/projects/{project_id}")
        with ThreadPoolExecutor(max_workers=3) as pool:
            merge_requests_future = pool.submit(
                _gitlab_paginated,
                _gitlab_session(token),
                api_root,
                f"/projects/{project_id}/merge_requests",
                params={"state": "all"},
            )
            issues_future = pool.submit(
                _gitlab_paginated,
                _gitlab_session(token),
                api_root,
                f"/projects/{project_id}/issues",
                params={"state": "all"},
            )
            releases_future = pool.submit(
                _gitlab_paginated,
                _gitlab_session(token),
                api_root,
                f"/projects/{project_id}/releases",
            )
            merge_requests, _ = merge_requests_future.result()

        def _fetch_merge_request(mr: dict[str, Any]) -> dict[str, Any]:
            task_session = _gitlab_session(token)
            iid = int(mr.get("iid") or 0)
            detail = _gitlab_get(task_session, api_root, f"/projects/{project_id}/merge_requests/{iid}")
            changes_payload = _gitlab_get(task_session, api_root, f"/projects/{project_id}/merge_requests/{iid}/changes")
            commits, _ = _gitlab_paginated(
                task_session,
                api_root,
                f"/projects/{project_id}/merge_requests/{iid}/commits",
            )
            notes, note_status = _gitlab_paginated(
                task_session,
                api_root,
                f"/projects/{project_id}/merge_requests/{iid}/notes",
                optional=True,
            )
            feature_statuses["review_comments"] = _merge_feature_status(
                feature_statuses.get("review_comments"),
                note_status,
            )
            pipelines, pipeline_status = _gitlab_paginated(
                task_session,
                api_root,
                f"/projects/{project_id}/merge_requests/{iid}/pipelines",
                optional=True,
            )
            feature_statuses["check_runs"] = _merge_feature_status(
                feature_statuses.get("check_runs"),
                pipeline_status,
            )
            head_sha = str((detail.get("diff_refs") or {}).get("head_sha") or detail.get("sha") or "").strip()
            commit_statuses, status_state = _gitlab_paginated(
                task_session,
                api_root,
                f"/projects/{project_id}/repository/commits/{head_sha}/statuses",
                optional=True,
            ) if head_sha else ([], "ok")
            feature_statuses["check_statuses"] = _merge_feature_status(
                feature_statuses.get("check_statuses"),
                status_state,
            )
            return _compact_gitlab_merge_request(
                detail,
                changes=list(changes_payload.get("changes") or []),
                commits=commits,
                notes=notes,
                pipelines=pipelines,
                commit_statuses=commit_statuses,
            )

        with ThreadPoolExecutor(max_workers=_detail_worker_count(len(merge_requests))) as pool:
            compact_merge_requests = list(pool.map(_fetch_merge_request, merge_requests))

        issue_rows, _ = issues_future.result()

        def _fetch_issue(issue: dict[str, Any]) -> dict[str, Any]:
            iid = int(issue.get("iid") or 0)
            task_session = _gitlab_session(token)
            comments, comment_status = _gitlab_paginated(
                task_session,
                api_root,
                f"/projects/{project_id}/issues/{iid}/notes",
                optional=True,
            )
            feature_statuses["issue_comments"] = _merge_feature_status(
                feature_statuses.get("issue_comments"),
                comment_status,
            )
            return _compact_gitlab_issue(issue, comments=comments)

        with ThreadPoolExecutor(max_workers=_detail_worker_count(len(issue_rows))) as pool:
            compact_issues = list(pool.map(_fetch_issue, issue_rows))

        commit_comments: list[dict[str, Any]] = []
        seen_commit_shas: set[str] = set()
        for mr in compact_merge_requests:
            for sha in mr.get("commits", []):
                if sha in seen_commit_shas:
                    continue
                seen_commit_shas.add(sha)
                comments, comment_status = _gitlab_paginated(
                    session,
                    api_root,
                    f"/projects/{project_id}/repository/commits/{sha}/comments",
                    optional=True,
                )
                feature_statuses["commit_comments"] = _merge_feature_status(
                    feature_statuses.get("commit_comments"),
                    comment_status,
                )
                for item in comments:
                    comment_id = str(item.get("note") or "").strip()
                    created_at = str(item.get("created_at") or "").strip()
                    body = str(item.get("note") or "").strip()
                    if not body:
                        continue
                    commit_comments.append(
                        {
                            "id": str(item.get("id") or f"{sha}:{created_at}:{len(commit_comments)+1}").strip(),
                            "author": _entity_name(item.get("author") or {}, "username", "name"),
                            "body": body,
                            "created_at": created_at,
                            "updated_at": created_at,
                            "commit_sha": sha,
                            "path": str(item.get("path") or "").strip(),
                        }
                    )

        releases, _ = releases_future.result()
        return {
            "status": "ok",
            "provider": "gitlab",
            "repo_full_name": repo_full_name,
            "repo_meta": {
                "default_branch": str(repo_meta.get("default_branch") or "").strip(),
                "html_url": str(repo_meta.get("web_url") or "").strip(),
            },
            "merge_requests": compact_merge_requests,
            "pull_requests": [],
            "issues": compact_issues,
            "commit_comments": commit_comments,
            "releases": [
                {
                    "id": str(item.get("tag_name") or item.get("name") or "").strip(),
                    "tag_name": str(item.get("tag_name") or "").strip(),
                    "name": str(item.get("name") or "").strip(),
                    "body": str(item.get("description") or "").strip(),
                    "state": str((item.get("upcoming_release") and "upcoming") or "published"),
                    "created_at": str(item.get("created_at") or "").strip(),
                    "published_at": str(item.get("released_at") or "").strip(),
                    "author": _entity_name(item.get("author") or {}, "username", "name"),
                    "assets": _gitlab_release_assets(item),
                }
                for item in releases
                if str(item.get("tag_name") or item.get("name") or "").strip()
            ],
            "workflow_artifacts": [],
            "feature_statuses": feature_statuses,
        }
    except Exception as exc:
        return {
            "status": "unavailable_environment",
            "provider": "gitlab",
            "repo_full_name": repo_full_name,
            "reason": str(exc),
        }


def fetch_hosting_stage1_snapshot(
    *,
    provider: str,
    repo_owner: str,
    repo_name: str,
    host: str = "",
    known_commit_files: dict[str, list[dict[str, str]]] | None = None,
) -> dict[str, Any]:
    provider_name = str(provider or "").strip().lower()
    repo_full_name = "/".join(part for part in (str(repo_owner or "").strip(), str(repo_name or "").strip()) if part)
    if not repo_full_name:
        return {
            "status": "unavailable_environment",
            "provider": provider_name or "unknown",
            "repo_full_name": "",
            "reason": "Missing repo owner/name for hosting resolver",
        }
    if provider_name == "github":
        return _fetch_github_snapshot(repo_full_name, known_commit_files=known_commit_files)
    if provider_name == "gitlab":
        return _fetch_gitlab_snapshot(repo_full_name, host=host or "gitlab.com")
    return {
        "status": "unsupported_provider",
        "provider": provider_name or "unknown",
        "repo_full_name": repo_full_name,
        "reason": f"Hosting resolver for provider '{provider_name or 'unknown'}' is not implemented",
    }


def build_hosting_stage1_units(
    *,
    repo_id: str,
    repo_display: str,
    repo_owner: str,
    repo_name: str,
    provider: str,
    host: str,
    known_branches: set[str],
    known_commits: set[str],
    known_files: set[str],
    known_commit_files: dict[str, list[dict[str, str]]] | None = None,
) -> dict[str, Any]:
    snapshot = fetch_hosting_stage1_snapshot(
        provider=provider,
        repo_owner=repo_owner,
        repo_name=repo_name,
        host=host,
        known_commit_files=known_commit_files,
    )
    status = str(snapshot.get("status") or "").strip() or "unavailable_environment"
    units: list[dict[str, Any]] = []
    object_counts: Counter = Counter()
    relation_count = 0

    if status != "ok":
        return {
            "units": units,
            "summary": {
                "status": status,
                "provider": str(snapshot.get("provider") or provider or "").strip(),
                "repo_full_name": str(snapshot.get("repo_full_name") or "").strip(),
                "reason": str(snapshot.get("reason") or "").strip(),
                "object_counts": {},
                "relation_count": 0,
                "feature_statuses": dict(snapshot.get("feature_statuses") or {}),
            },
            "graph": {
                "status": status,
                "provider": str(snapshot.get("provider") or provider or "").strip(),
                "repo_full_name": str(snapshot.get("repo_full_name") or "").strip(),
                "reason": str(snapshot.get("reason") or "").strip(),
                "pull_requests": [],
                "merge_requests": [],
                "issues": [],
                "commit_comments": [],
                "releases": [],
                "workflow_artifacts": [],
                "feature_statuses": dict(snapshot.get("feature_statuses") or {}),
            },
        }

    created_object_ids: set[str] = set()

    def _append_unit(unit: dict[str, Any]) -> None:
        nonlocal relation_count
        object_id = str(unit.get("unit_key") or "").strip()
        if not object_id or object_id in created_object_ids:
            return
        created_object_ids.add(object_id)
        units.append(unit)
        object_counts[str(unit.get("object_type") or "").strip() or "unknown"] += 1
        relation_count += max(0, len(unit.get("facts") or []) - 1)

    def _ensure_branch_stub(branch_name: str) -> None:
        branch = str(branch_name or "").strip()
        if not branch or branch in known_branches or f"branch:{branch}" in created_object_ids:
            return
        _append_unit(
            build_codebase_object_unit(
                object_type="branch",
                object_id=f"branch:{branch}",
                raw_text="\n".join(
                    [
                        f"Branch name: {branch}",
                        f"Repository: {repo_display}",
                        "Origin: hosting metadata",
                    ]
                ),
                fact_text=f"Branch {branch} is referenced by hosting metadata for repository {repo_display}.",
                anchor=build_codebase_anchor("branch", branch_name=branch),
                currentness="unknown",
                topic_key=branch,
                state_label="branch",
                entities=[branch, repo_display],
                tags=["branch", "hosting"],
                extra_metadata={"branch_name": branch, "hosting_stub": True},
                relation_facts=[
                    build_codebase_relation_fact(
                        relation_type="repo_has_branch",
                        from_id=repo_id,
                        to_id=f"branch:{branch}",
                        fact_text=f"Repository {repo_display} has branch {branch}.",
                        anchor=build_codebase_anchor("branch", branch_name=branch),
                        entities=[repo_display, branch],
                    )
                ],
            )
        )

    def _ensure_commit_stub(commit_sha: str) -> None:
        sha = str(commit_sha or "").strip()
        if not sha or sha in known_commits or f"commit:{sha}" in created_object_ids:
            return
        _append_unit(
            build_codebase_object_unit(
                object_type="commit",
                object_id=f"commit:{sha}",
                raw_text="\n".join(
                    [
                        f"Commit SHA: {sha}",
                        f"Repository: {repo_display}",
                        "Origin: hosting metadata",
                    ]
                ),
                fact_text=f"Commit {sha} is referenced by hosting metadata for repository {repo_display}.",
                anchor=build_codebase_anchor("commit", commit_sha=sha),
                currentness="unknown",
                topic_key=sha[:12],
                state_label="commit",
                entities=[sha, repo_display],
                tags=["commit", "hosting"],
                extra_metadata={"commit_sha": sha, "hosting_stub": True},
            )
        )

    def _ensure_file_stub(file_path: str) -> None:
        path = str(file_path or "").strip()
        if not path or path in known_files or f"file:{path}" in created_object_ids:
            return
        directory_path = str(os.path.dirname(path) or ".").replace("\\", "/")
        _append_unit(
            build_codebase_object_unit(
                object_type="file",
                object_id=f"file:{path}",
                raw_text="\n".join(
                    [
                        f"File: {path}",
                        f"Directory: {directory_path}",
                        "Origin: hosting metadata",
                    ]
                ),
                fact_text=f"File {path} is referenced by hosting metadata for repository {repo_display}.",
                anchor=build_codebase_anchor("file", path=path),
                currentness="unknown",
                topic_key=path.replace("/", "_"),
                state_label="file",
                entities=[path, directory_path],
                tags=["file", "hosting"],
                extra_metadata={"path": path, "directory_path": directory_path, "hosting_stub": True},
            )
        )

    label_units: set[str] = set()
    assignee_units: set[str] = set()

    def _ensure_label(label_name: str) -> str:
        label = str(label_name or "").strip()
        object_id = f"label:{label}"
        if label and object_id not in label_units:
            label_units.add(object_id)
            _append_unit(
                build_codebase_object_unit(
                    object_type="label",
                    object_id=object_id,
                    raw_text=f"Label: {label}\nRepository: {repo_display}",
                    fact_text=f"Label {label} exists in repository {repo_display}.",
                    anchor=build_codebase_anchor("label", label_name=label),
                    currentness="current",
                    topic_key=label,
                    state_label="label",
                    entities=[label, repo_display],
                    tags=["label", "hosting"],
                    extra_metadata={"label_name": label},
                )
            )
        return object_id

    def _ensure_assignee(login: str) -> str:
        assignee = str(login or "").strip()
        object_id = f"assignee:{assignee}"
        if assignee and object_id not in assignee_units:
            assignee_units.add(object_id)
            _append_unit(
                build_codebase_object_unit(
                    object_type="assignee",
                    object_id=object_id,
                    raw_text=f"Assignee: {assignee}\nRepository: {repo_display}",
                    fact_text=f"Assignee {assignee} is referenced in repository {repo_display}.",
                    anchor=build_codebase_anchor("assignee", assignee_identity=assignee),
                    currentness="current",
                    topic_key=assignee,
                    state_label="assignee",
                    entities=[assignee, repo_display],
                    tags=["assignee", "hosting"],
                    extra_metadata={"assignee_identity": assignee},
                )
            )
        return object_id

    for request in [*list(snapshot.get("pull_requests") or []), *list(snapshot.get("merge_requests") or [])]:
        number = int(request.get("number") or 0)
        request_kind = str(request.get("request_kind") or "pull_request").strip() or "pull_request"
        request_label = "Merge request" if request_kind == "merge_request" else "Pull request"
        request_short = "mr" if request_kind == "merge_request" else "pr"
        number_key = "mr_number" if request_kind == "merge_request" else "pr_number"
        repo_relation_type = f"{request_kind}_belongs_to_repo"
        targets_relation_type = f"{request_short}_targets_branch"
        comes_from_relation_type = f"{request_short}_comes_from_branch"
        includes_commit_relation_type = f"{request_short}_includes_commit"
        modifies_file_relation_type = f"{request_short}_modifies_file"
        label_relation_type = f"{request_kind}_has_label"
        assignee_relation_type = f"{request_kind}_has_assignee"
        review_parent_relation_type = (
            "review_belongs_to_merge_request" if request_kind == "merge_request" else "review_belongs_to_pr"
        )
        review_comment_parent_relation_type = (
            "review_comment_belongs_to_merge_request"
            if request_kind == "merge_request"
            else "review_comment_belongs_to_pr"
        )
        check_parent_relation_type = (
            "check_run_belongs_to_merge_request"
            if request_kind == "merge_request"
            else "check_run_belongs_to_pr"
        )
        request_id = f"{request_kind}:{number}"
        base_branch = str(request.get("base_branch") or "").strip()
        head_branch = str(request.get("head_branch") or "").strip()
        head_sha = str(request.get("head_sha") or "").strip()
        anchor_kwargs = {
            number_key: number,
            "base_branch": base_branch,
            "head_branch": head_branch,
            "commit_sha": head_sha,
            "created_at": request.get("created_at"),
            "updated_at": request.get("updated_at"),
            "merged_at": request.get("merged_at"),
        }
        _ensure_branch_stub(base_branch)
        _ensure_branch_stub(head_branch)
        if head_sha:
            _ensure_commit_stub(head_sha)
        relation_facts = [
            build_codebase_relation_fact(
                relation_type=repo_relation_type,
                from_id=request_id,
                to_id=repo_id,
                fact_text=f"{request_label} {number} belongs to repository {repo_display}.",
                anchor=build_codebase_anchor(request_kind, **{number_key: number}),
                entities=[str(number), repo_display],
            )
        ]
        if base_branch:
            relation_facts.append(
                build_codebase_relation_fact(
                    relation_type=targets_relation_type,
                    from_id=request_id,
                    to_id=f"branch:{base_branch}",
                    fact_text=f"{request_label} {number} targets base branch {base_branch}.",
                    anchor=build_codebase_anchor(request_kind, **{number_key: number, "branch_name": base_branch}),
                    entities=[str(number), base_branch],
                )
            )
        if head_branch:
            relation_facts.append(
                build_codebase_relation_fact(
                    relation_type=comes_from_relation_type,
                    from_id=request_id,
                    to_id=f"branch:{head_branch}",
                    fact_text=f"{request_label} {number} comes from head branch {head_branch}.",
                    anchor=build_codebase_anchor(request_kind, **{number_key: number, "branch_name": head_branch}),
                    entities=[str(number), head_branch],
                )
            )
        for commit_sha in request.get("commits", []):
            _ensure_commit_stub(commit_sha)
            relation_facts.append(
                build_codebase_relation_fact(
                    relation_type=includes_commit_relation_type,
                    from_id=request_id,
                    to_id=f"commit:{commit_sha}",
                    fact_text=f"{request_label} {number} includes commit {commit_sha}.",
                    anchor=build_codebase_anchor(request_kind, **{number_key: number, "commit_sha": commit_sha}),
                    entities=[str(number), commit_sha],
                )
            )
        for file_row in request.get("files", []):
            file_path = str(file_row.get("path") or "").strip()
            if not file_path:
                continue
            _ensure_file_stub(file_path)
            relation_facts.append(
                build_codebase_relation_fact(
                    relation_type=modifies_file_relation_type,
                    from_id=request_id,
                    to_id=f"file:{file_path}",
                    fact_text=f"{request_label} {number} modifies file {file_path}.",
                    anchor=build_codebase_anchor("file", path=file_path, **{number_key: number}),
                    entities=[str(number), file_path],
                )
            )
        for label_name in request.get("labels", []):
            label_id = _ensure_label(label_name)
            if label_id:
                relation_facts.append(
                    build_codebase_relation_fact(
                        relation_type=label_relation_type,
                        from_id=request_id,
                        to_id=label_id,
                        fact_text=f"{request_label} {number} has label {label_name}.",
                        anchor=build_codebase_anchor("label", label_name=label_name, **{number_key: number}),
                        entities=[str(number), label_name],
                    )
                )
        for assignee in request.get("assignees", []):
            assignee_id = _ensure_assignee(assignee)
            if assignee_id:
                relation_facts.append(
                    build_codebase_relation_fact(
                        relation_type=assignee_relation_type,
                        from_id=request_id,
                        to_id=assignee_id,
                        fact_text=f"{request_label} {number} is assigned to {assignee}.",
                        anchor=build_codebase_anchor("assignee", assignee_identity=assignee, **{number_key: number}),
                        entities=[str(number), assignee],
                    )
                )

        _append_unit(
            build_codebase_object_unit(
                object_type=request_kind,
                object_id=request_id,
                raw_text="\n".join(
                    [
                        f"{request_label} number: {number}",
                        f"Title: {request.get('title') or '(none)'}",
                        f"State: {request.get('state') or '(unknown)'}",
                        f"Base branch: {base_branch or '(none)'}",
                        f"Head branch: {head_branch or '(none)'}",
                        f"Head SHA: {head_sha or '(none)'}",
                        f"Created at: {request.get('created_at') or '(unknown)'}",
                        f"Updated at: {request.get('updated_at') or '(unknown)'}",
                        f"Merged at: {request.get('merged_at') or '(not merged)'}",
                        f"Author: {request.get('author') or '(unknown)'}",
                        f"Labels: {', '.join(request.get('labels') or []) or '(none)'}",
                        f"Assignees: {', '.join(request.get('assignees') or []) or '(none)'}",
                    ]
                ),
                fact_text=f"{request_label} {number} in repository {repo_display} is titled '{request.get('title') or '(none)'}' and is in state {request.get('state') or 'unknown'}.",
                anchor=build_codebase_anchor(request_kind, **anchor_kwargs),
                source_date=str(request.get("updated_at") or request.get("created_at") or ""),
                currentness="current" if _is_open_state(str(request.get("state") or "")) else "historical",
                topic_key=f"{request_short}_{number}",
                state_label=request_kind,
                entities=[str(number), base_branch, head_branch, head_sha, *(request.get("labels") or [])[:4]],
                tags=[request_kind, "hosting", str(request.get("state") or "").lower()],
                extra_metadata={
                    number_key: number,
                    f"{request_short}_title": request.get("title"),
                    f"{request_short}_body": request.get("body"),
                    f"{request_short}_state": request.get("state"),
                    "base_branch": base_branch,
                    "head_branch": head_branch,
                    "commit_sha": head_sha,
                    "author": request.get("author"),
                },
                relation_facts=relation_facts,
            )
        )

        for review in request.get("reviews", []):
            review_id = str(review.get("id") or "").strip()
            if not review_id:
                continue
            review_anchor_kwargs = {
                number_key: number,
                "review_state": review.get("state"),
                "commit_sha": review.get("commit_sha"),
                "created_at": review.get("submitted_at"),
            }
            _append_unit(
                build_codebase_object_unit(
                    object_type="review",
                    object_id=f"review:{review_id}",
                    raw_text="\n".join(
                        [
                            f"Review ID: {review_id}",
                            f"{request_label} number: {number}",
                            f"State: {review.get('state') or '(unknown)'}",
                            f"Author: {review.get('author') or '(unknown)'}",
                            f"Submitted at: {review.get('submitted_at') or '(unknown)'}",
                            f"Commit SHA: {review.get('commit_sha') or '(none)'}",
                            f"Body: {review.get('body') or '(none)'}",
                        ]
                    ),
                    fact_text=f"Review {review_id} belongs to {request_label.lower()} {number} and is in state {review.get('state') or 'unknown'}.",
                    anchor=build_codebase_anchor("review", **review_anchor_kwargs),
                    source_date=str(review.get("submitted_at") or ""),
                    currentness="historical",
                    topic_key=f"review_{review_id}",
                    state_label="review",
                    entities=[review_id, str(number), review.get("author") or ""],
                    tags=["review", "hosting", str(review.get("state") or "").lower()],
                    extra_metadata={
                        "review_state": review.get("state"),
                        "author": review.get("author"),
                        number_key: number,
                        "commit_sha": review.get("commit_sha"),
                    },
                    relation_facts=[
                        build_codebase_relation_fact(
                            relation_type=review_parent_relation_type,
                            from_id=f"review:{review_id}",
                            to_id=request_id,
                            fact_text=f"Review {review_id} belongs to {request_label.lower()} {number}.",
                            anchor=build_codebase_anchor("review", **{number_key: number}),
                            entities=[review_id, str(number)],
                        )
                    ],
                )
            )

        for comment in request.get("review_comments", []):
            comment_id = str(comment.get("id") or "").strip()
            if not comment_id:
                continue
            review_id = str(comment.get("review_id") or "").strip()
            file_path = str(comment.get("path") or "").strip()
            commit_sha = str(comment.get("commit_sha") or "").strip()
            if file_path:
                _ensure_file_stub(file_path)
            if commit_sha:
                _ensure_commit_stub(commit_sha)
            comment_anchor_kwargs = {
                number_key: number,
                "review_id": review_id,
                "comment_timestamp": comment.get("created_at"),
                "path": file_path,
                "commit_sha": commit_sha,
            }
            relation_facts = [
                build_codebase_relation_fact(
                    relation_type=review_comment_parent_relation_type,
                    from_id=f"review_comment:{comment_id}",
                    to_id=request_id,
                    fact_text=f"Review comment {comment_id} belongs to {request_label.lower()} {number}.",
                    anchor=build_codebase_anchor("review_comment", **{number_key: number}),
                    entities=[comment_id, str(number)],
                )
            ]
            if review_id:
                relation_facts.append(
                    build_codebase_relation_fact(
                        relation_type="review_comment_belongs_to_review",
                        from_id=f"review_comment:{comment_id}",
                        to_id=f"review:{review_id}",
                        fact_text=f"Review comment {comment_id} belongs to review {review_id}.",
                        anchor=build_codebase_anchor("review_comment", **{number_key: number, "review_id": review_id}),
                        entities=[comment_id, review_id],
                    )
                )
            _append_unit(
                build_codebase_object_unit(
                    object_type="review_comment",
                    object_id=f"review_comment:{comment_id}",
                    raw_text="\n".join(
                        [
                            f"Review comment ID: {comment_id}",
                            f"{request_label} number: {number}",
                            f"Review ID: {review_id or '(none)'}",
                            f"Author: {comment.get('author') or '(unknown)'}",
                            f"Created at: {comment.get('created_at') or '(unknown)'}",
                            f"File path: {file_path or '(none)'}",
                            f"Commit SHA: {commit_sha or '(none)'}",
                            f"Body: {comment.get('body') or '(none)'}",
                        ]
                    ),
                    fact_text=f"Review comment {comment_id} on {request_label.lower()} {number} says: {comment.get('body') or '(none)'}.",
                    anchor=build_codebase_anchor("review_comment", **comment_anchor_kwargs),
                    source_date=str(comment.get("created_at") or ""),
                    currentness="historical",
                    topic_key=f"review_comment_{comment_id}",
                    state_label="review_comment",
                    entities=[comment_id, str(number), comment.get("author") or "", file_path],
                    tags=["review_comment", "hosting"],
                    extra_metadata={
                        number_key: number,
                        "review_id": review_id,
                        "author": comment.get("author"),
                        "path": file_path,
                        "commit_sha": commit_sha,
                    },
                    relation_facts=relation_facts,
                )
            )

        for check in [*(request.get("check_runs") or []), *(request.get("statuses") or [])]:
            check_id = str(check.get("id") or "").strip()
            if not check_id:
                continue
            head_sha_value = str(check.get("head_sha") or head_sha or "").strip()
            if head_sha_value:
                _ensure_commit_stub(head_sha_value)
            relation_facts = [
                build_codebase_relation_fact(
                    relation_type=check_parent_relation_type,
                    from_id=f"check_run:{check_id}",
                    to_id=request_id,
                    fact_text=f"Check run {check.get('name') or check_id} belongs to {request_label.lower()} {number}.",
                    anchor=build_codebase_anchor("check_run", **{number_key: number, "check_name": check.get("name")}),
                    entities=[check_id, str(number), check.get("name") or ""],
                )
            ]
            if head_sha_value:
                relation_facts.append(
                    build_codebase_relation_fact(
                        relation_type="check_run_belongs_to_commit",
                        from_id=f"check_run:{check_id}",
                        to_id=f"commit:{head_sha_value}",
                        fact_text=f"Check run {check.get('name') or check_id} belongs to commit {head_sha_value}.",
                        anchor=build_codebase_anchor("check_run", commit_sha=head_sha_value, check_name=check.get("name")),
                        entities=[check_id, head_sha_value, check.get("name") or ""],
                    )
                )
            _append_unit(
                build_codebase_object_unit(
                    object_type="check_run",
                    object_id=f"check_run:{check_id}",
                    raw_text="\n".join(
                        [
                            f"Check run ID: {check_id}",
                            f"{request_label} number: {number}",
                            f"Name: {check.get('name') or '(none)'}",
                            f"State: {check.get('status') or '(unknown)'}",
                            f"Conclusion: {check.get('conclusion') or '(none)'}",
                            f"Head SHA: {head_sha_value or '(none)'}",
                        ]
                    ),
                    fact_text=f"Check run {check.get('name') or check_id} on {request_label.lower()} {number} is {check.get('status') or 'unknown'} with conclusion {check.get('conclusion') or '(none)'}.",
                    anchor=build_codebase_anchor(
                        "check_run",
                        **{
                            number_key: number,
                            "commit_sha": head_sha_value,
                            "check_name": check.get("name"),
                            "check_state": check.get("status"),
                            "check_conclusion": check.get("conclusion"),
                        },
                    ),
                    source_date=str(check.get("completed_at") or check.get("updated_at") or check.get("created_at") or ""),
                    currentness="current" if _is_open_state(str(request.get("state") or "")) else "historical",
                    topic_key=f"check_{check_id}",
                    state_label="check_run",
                    entities=[check_id, str(number), check.get("name") or "", head_sha_value],
                    tags=["check_run", "hosting", str(check.get("status") or "").lower()],
                    extra_metadata={
                        number_key: number,
                        "commit_sha": head_sha_value,
                        "check_name": check.get("name"),
                        "check_state": check.get("status"),
                        "check_conclusion": check.get("conclusion"),
                    },
                    relation_facts=relation_facts,
                )
            )

    for issue in snapshot.get("issues", []):
        number = int(issue.get("number") or 0)
        issue_id = f"issue:{number}"
        relation_facts = [
            build_codebase_relation_fact(
                relation_type="issue_belongs_to_repo",
                from_id=issue_id,
                to_id=repo_id,
                fact_text=f"Issue {number} belongs to repository {repo_display}.",
                anchor=build_codebase_anchor("issue", issue_number=number),
                entities=[str(number), repo_display],
            )
        ]
        for label_name in issue.get("labels", []):
            label_id = _ensure_label(label_name)
            if label_id:
                relation_facts.append(
                    build_codebase_relation_fact(
                        relation_type="issue_has_label",
                        from_id=issue_id,
                        to_id=label_id,
                        fact_text=f"Issue {number} has label {label_name}.",
                        anchor=build_codebase_anchor("label", label_name=label_name, issue_number=number),
                        entities=[str(number), label_name],
                    )
                )
        for assignee in issue.get("assignees", []):
            assignee_id = _ensure_assignee(assignee)
            if assignee_id:
                relation_facts.append(
                    build_codebase_relation_fact(
                        relation_type="issue_has_assignee",
                        from_id=issue_id,
                        to_id=assignee_id,
                        fact_text=f"Issue {number} is assigned to {assignee}.",
                        anchor=build_codebase_anchor("assignee", assignee_identity=assignee, issue_number=number),
                        entities=[str(number), assignee],
                    )
                )
        _append_unit(
            build_codebase_object_unit(
                object_type="issue",
                object_id=issue_id,
                raw_text="\n".join(
                    [
                        f"Issue number: {number}",
                        f"Title: {issue.get('title') or '(none)'}",
                        f"State: {issue.get('state') or '(unknown)'}",
                        f"Author: {issue.get('author') or '(unknown)'}",
                        f"Created at: {issue.get('created_at') or '(unknown)'}",
                        f"Updated at: {issue.get('updated_at') or '(unknown)'}",
                        f"Closed at: {issue.get('closed_at') or '(not closed)'}",
                        f"Labels: {', '.join(issue.get('labels') or []) or '(none)'}",
                        f"Assignees: {', '.join(issue.get('assignees') or []) or '(none)'}",
                    ]
                ),
                fact_text=f"Issue {number} in repository {repo_display} is titled '{issue.get('title') or '(none)'}' and is in state {issue.get('state') or 'unknown'}.",
                anchor=build_codebase_anchor(
                    "issue",
                    issue_number=number,
                    created_at=issue.get("created_at"),
                    updated_at=issue.get("updated_at"),
                ),
                source_date=str(issue.get("updated_at") or issue.get("created_at") or ""),
                currentness="current" if _is_open_state(str(issue.get("state") or "")) else "historical",
                topic_key=f"issue_{number}",
                state_label="issue",
                entities=[str(number), issue.get("author") or "", *(issue.get("labels") or [])[:4]],
                tags=["issue", "hosting", str(issue.get("state") or "").lower()],
                extra_metadata={
                    "issue_number": number,
                    "issue_title": issue.get("title"),
                    "issue_body": issue.get("body"),
                    "issue_state": issue.get("state"),
                    "author": issue.get("author"),
                },
                relation_facts=relation_facts,
            )
        )
        for comment in issue.get("comments", []):
            comment_id = str(comment.get("id") or "").strip()
            if not comment_id:
                continue
            _append_unit(
                build_codebase_object_unit(
                    object_type="issue_comment",
                    object_id=f"issue_comment:{comment_id}",
                    raw_text="\n".join(
                        [
                            f"Issue comment ID: {comment_id}",
                            f"Issue number: {number}",
                            f"Author: {comment.get('author') or '(unknown)'}",
                            f"Created at: {comment.get('created_at') or '(unknown)'}",
                            f"Body: {comment.get('body') or '(none)'}",
                        ]
                    ),
                    fact_text=f"Issue comment {comment_id} belongs to issue {number} and says: {comment.get('body') or '(none)'}.",
                    anchor=build_codebase_anchor(
                        "issue_comment",
                        issue_number=number,
                        comment_timestamp=comment.get("created_at"),
                    ),
                    source_date=str(comment.get("created_at") or ""),
                    currentness="historical",
                    topic_key=f"issue_comment_{comment_id}",
                    state_label="issue_comment",
                    entities=[comment_id, str(number), comment.get("author") or ""],
                    tags=["issue_comment", "hosting"],
                    extra_metadata={
                        "issue_number": number,
                        "author": comment.get("author"),
                    },
                    relation_facts=[
                        build_codebase_relation_fact(
                            relation_type="issue_comment_belongs_to_issue",
                            from_id=f"issue_comment:{comment_id}",
                            to_id=issue_id,
                            fact_text=f"Issue comment {comment_id} belongs to issue {number}.",
                            anchor=build_codebase_anchor("issue_comment", issue_number=number),
                            entities=[comment_id, str(number)],
                        )
                    ],
                )
            )

    for comment in snapshot.get("commit_comments", []):
        comment_id = str(comment.get("id") or "").strip()
        commit_sha = str(comment.get("commit_sha") or "").strip()
        if not comment_id or not commit_sha:
            continue
        _ensure_commit_stub(commit_sha)
        _append_unit(
            build_codebase_object_unit(
                object_type="commit_comment",
                object_id=f"commit_comment:{comment_id}",
                raw_text="\n".join(
                    [
                        f"Commit comment ID: {comment_id}",
                        f"Commit SHA: {commit_sha}",
                        f"Author: {comment.get('author') or '(unknown)'}",
                        f"Created at: {comment.get('created_at') or '(unknown)'}",
                        f"File path: {comment.get('path') or '(none)'}",
                        f"Body: {comment.get('body') or '(none)'}",
                    ]
                ),
                fact_text=f"Commit comment {comment_id} belongs to commit {commit_sha} and says: {comment.get('body') or '(none)'}.",
                anchor=build_codebase_anchor(
                    "commit_comment",
                    commit_sha=commit_sha,
                    comment_timestamp=comment.get("created_at"),
                    path=comment.get("path"),
                ),
                source_date=str(comment.get("created_at") or ""),
                currentness="historical",
                topic_key=f"commit_comment_{comment_id}",
                state_label="commit_comment",
                entities=[comment_id, commit_sha, comment.get("author") or "", comment.get("path") or ""],
                tags=["commit_comment", "hosting"],
                extra_metadata={
                    "commit_sha": commit_sha,
                    "author": comment.get("author"),
                    "path": comment.get("path"),
                },
                relation_facts=[
                    build_codebase_relation_fact(
                        relation_type="commit_comment_belongs_to_commit",
                        from_id=f"commit_comment:{comment_id}",
                        to_id=f"commit:{commit_sha}",
                        fact_text=f"Commit comment {comment_id} belongs to commit {commit_sha}.",
                        anchor=build_codebase_anchor("commit_comment", commit_sha=commit_sha),
                        entities=[comment_id, commit_sha],
                    )
                ],
            )
        )

    for release in snapshot.get("releases", []):
        release_id = str(release.get("id") or "").strip()
        if not release_id:
            continue
        release_object_id = f"release:{release_id}"
        _append_unit(
            build_codebase_object_unit(
                object_type="release",
                object_id=release_object_id,
                raw_text="\n".join(
                    [
                        f"Release ID: {release_id}",
                        f"Tag: {release.get('tag_name') or '(none)'}",
                        f"Name: {release.get('name') or '(none)'}",
                        f"State: {release.get('state') or '(unknown)'}",
                        f"Author: {release.get('author') or '(unknown)'}",
                        f"Created at: {release.get('created_at') or '(unknown)'}",
                        f"Published at: {release.get('published_at') or '(unknown)'}",
                    ]
                ),
                fact_text=f"Release {release.get('tag_name') or release_id} belongs to repository {repo_display}.",
                anchor=build_codebase_anchor(
                    "release",
                    tag_name=release.get("tag_name"),
                    created_at=release.get("created_at"),
                    updated_at=release.get("published_at"),
                ),
                source_date=str(release.get("published_at") or release.get("created_at") or ""),
                currentness="historical",
                topic_key=f"release_{release.get('tag_name') or release_id}",
                state_label="release",
                entities=[release.get("tag_name") or "", release.get("name") or "", repo_display],
                tags=["release", "hosting"],
                extra_metadata={
                    "tag_name": release.get("tag_name"),
                    "name": release.get("name"),
                    "state": release.get("state"),
                    "author": release.get("author"),
                },
                relation_facts=[
                    build_codebase_relation_fact(
                        relation_type="release_belongs_to_repo",
                        from_id=release_object_id,
                        to_id=repo_id,
                        fact_text=f"Release {release.get('tag_name') or release_id} belongs to repository {repo_display}.",
                        anchor=build_codebase_anchor("release", tag_name=release.get("tag_name")),
                        entities=[release.get("tag_name") or release_id, repo_display],
                    )
                ],
            )
        )
        for asset in release.get("assets", []):
            asset_id = str(asset.get("id") or "").strip()
            if not asset_id:
                continue
            _append_unit(
                build_codebase_object_unit(
                    object_type="artifact",
                    object_id=f"artifact:{asset_id}",
                    raw_text="\n".join(
                        [
                            f"Artifact ID: {asset_id}",
                            f"Release tag: {release.get('tag_name') or '(none)'}",
                            f"Name: {asset.get('name') or '(none)'}",
                            f"Content type: {asset.get('content_type') or '(none)'}",
                            f"Size bytes: {asset.get('size') or 0}",
                            f"Created at: {asset.get('created_at') or '(unknown)'}",
                            f"Updated at: {asset.get('updated_at') or '(unknown)'}",
                        ]
                    ),
                    fact_text=f"Artifact {asset.get('name') or asset_id} belongs to release {release.get('tag_name') or release_id}.",
                    anchor=build_codebase_anchor(
                        "artifact",
                        tag_name=release.get("tag_name"),
                        created_at=asset.get("created_at"),
                        updated_at=asset.get("updated_at"),
                    ),
                    source_date=str(asset.get("updated_at") or asset.get("created_at") or ""),
                    currentness="historical",
                    topic_key=f"artifact_{asset_id}",
                    state_label="artifact",
                    entities=[asset.get("name") or "", release.get("tag_name") or "", repo_display],
                    tags=["artifact", "hosting"],
                    extra_metadata={
                        "artifact_name": asset.get("name"),
                        "content_type": asset.get("content_type"),
                        "download_url": asset.get("download_url"),
                    },
                    relation_facts=[
                        build_codebase_relation_fact(
                            relation_type="artifact_belongs_to_release",
                            from_id=f"artifact:{asset_id}",
                            to_id=release_object_id,
                            fact_text=f"Artifact {asset.get('name') or asset_id} belongs to release {release.get('tag_name') or release_id}.",
                            anchor=build_codebase_anchor("artifact", tag_name=release.get("tag_name")),
                            entities=[asset.get("name") or asset_id, release.get("tag_name") or release_id],
                        )
                    ],
                )
            )

    for artifact in snapshot.get("workflow_artifacts", []):
        artifact_id = str(artifact.get("id") or "").strip()
        if not artifact_id or f"artifact:{artifact_id}" in created_object_ids:
            continue
        _append_unit(
            build_codebase_object_unit(
                object_type="artifact",
                object_id=f"artifact:{artifact_id}",
                raw_text="\n".join(
                    [
                        f"Artifact ID: {artifact_id}",
                        f"Name: {artifact.get('name') or '(none)'}",
                        f"Workflow run ID: {artifact.get('workflow_run_id') or '(none)'}",
                        f"Expired: {artifact.get('expired')}",
                        f"Created at: {artifact.get('created_at') or '(unknown)'}",
                        f"Updated at: {artifact.get('updated_at') or '(unknown)'}",
                    ]
                ),
                fact_text=f"Workflow artifact {artifact.get('name') or artifact_id} is available for repository {repo_display}.",
                anchor=build_codebase_anchor(
                    "artifact",
                    created_at=artifact.get("created_at"),
                    updated_at=artifact.get("updated_at"),
                ),
                source_date=str(artifact.get("updated_at") or artifact.get("created_at") or ""),
                currentness="historical",
                topic_key=f"artifact_{artifact_id}",
                state_label="artifact",
                entities=[artifact.get("name") or "", artifact.get("workflow_run_id") or "", repo_display],
                tags=["artifact", "hosting", "workflow"],
                extra_metadata={
                    "artifact_name": artifact.get("name"),
                    "workflow_run_id": artifact.get("workflow_run_id"),
                    "expired": artifact.get("expired"),
                },
            )
        )

    return {
        "units": units,
        "summary": {
            "status": "ok",
            "provider": str(snapshot.get("provider") or provider or "").strip(),
            "repo_full_name": str(snapshot.get("repo_full_name") or "").strip(),
            "reason": "",
            "object_counts": dict(object_counts),
            "relation_count": relation_count,
            "feature_statuses": dict(snapshot.get("feature_statuses") or {}),
        },
        "graph": snapshot,
    }
