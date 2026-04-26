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
import os
import re
import subprocess
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .codebase_hosting import build_hosting_stage1_units
from .codebase_ontology import (
    build_codebase_anchor,
    build_codebase_object_fact,
    build_codebase_object_row,
    build_codebase_object_unit,
    build_codebase_relation_fact,
)

_DIFF_MARKER_RE = re.compile(r"^(diff --git|@@ |--- |\+\+\+ )", re.M)
_PATCH_HEADER_RE = re.compile(r"^diff --git a/(.+?) b/(.+)$")
_HUNK_RE = re.compile(
    r"^@@ -(?P<old_start>\d+)(?:,(?P<old_count>\d+))? "
    r"\+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@(?P<label>.*)$"
)
_REMOTE_URL_RE = re.compile(
    r"^(?:https?://|ssh://git@|git@)(?P<host>[^/:]+)[:/](?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?$"
)
_MAX_COMMIT_DETAIL_WORKERS = 16
_COMMIT_META_SENTINEL = "__GOSH_CODEBASE_META_END__"
_MAX_PATCH_STREAM_BYTES = 8_000_000


def _commit_detail_worker_count(commit_count: int) -> int:
    cpu = max(1, os.cpu_count() or 1)
    ceiling = min(_MAX_COMMIT_DETAIL_WORKERS, max(4, cpu * 2))
    if commit_count <= 0:
        return 1
    return max(1, min(ceiling, commit_count))


def _run_git(repo_root: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or f"git {' '.join(args)} failed")
    return proc.stdout


def _maybe_git(repo_root: Path, *args: str) -> str:
    try:
        return _run_git(repo_root, *args)
    except Exception:
        return ""


def _dedupe_strings(values: list[str] | tuple[str, ...] | None) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values or []:
        item = str(value or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()


def _sha256_bytes(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


class _UnitSpool:
    def __init__(self) -> None:
        fd, path = tempfile.mkstemp(prefix="gosh_codebase_units_", suffix=".jsonl")
        self._path = Path(path)
        self._fh = self._path.open("w", encoding="utf-8")

    @property
    def path(self) -> str:
        return str(self._path)

    def append(self, unit: dict[str, Any]) -> None:
        self._fh.write(json.dumps(unit, ensure_ascii=False, separators=(",", ":")))
        self._fh.write("\n")

    def extend(self, units: list[dict[str, Any]]) -> None:
        for unit in units:
            self.append(unit)

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()


class _DigestAccumulator:
    def __init__(self) -> None:
        self._sha = hashlib.sha256()

    def update(self, unit: dict[str, Any]) -> None:
        payload = {
            "unit_key": unit.get("unit_key"),
            "object_type": unit.get("object_type"),
            "raw_hash": _sha256_text(str((unit.get("episode") or {}).get("raw_text") or "")),
            "fact_count": len(unit.get("facts") or []),
        }
        self._sha.update(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )

    def hexdigest(self) -> str:
        return "sha256:" + self._sha.hexdigest()


def _build_relation_batch_unit(
    *,
    batch_id: str,
    parent_object_id: str,
    source_date: str,
    summary_text: str,
    relation_facts: list[dict[str, Any]],
) -> dict[str, Any]:
    rendered = str(summary_text or "").strip()
    return {
        "unit_key": batch_id,
        "object_type": "relation_batch",
        "episode": {
            "topic_key": batch_id,
            "state_label": "relation_batch",
            "source_date": str(source_date or ""),
            "currentness": "historical",
            "raw_text": rendered,
            "provenance": {
                "source_section_path": batch_id,
                "raw_span": [0, len(rendered)],
            },
            "metadata": {
                "codebase": {
                    "object_type": "relation_batch",
                    "object_id": batch_id,
                    "parent_object_id": parent_object_id,
                }
            },
        },
        "facts": relation_facts,
    }


def _commit_graph_row(row: dict[str, Any]) -> dict[str, Any]:
    modified_paths = [
        str(diff.get("path") or diff.get("rename_to") or diff.get("rename_from") or "").strip()
        for diff in (row.get("diffs") or [])
        if str(diff.get("path") or diff.get("rename_to") or diff.get("rename_from") or "").strip()
    ]
    return {
        "sha": row["sha"],
        "parents": list(row.get("parents") or []),
        "author_name": row.get("author_name") or "",
        "author_email": row.get("author_email") or "",
        "authored_at": row.get("authored_at") or "",
        "committer_name": row.get("committer_name") or "",
        "committer_email": row.get("committer_email") or "",
        "committed_at": row.get("committed_at") or "",
        "subject": row.get("subject") or "",
        "body": row.get("body") or "",
        "branches": list(row.get("branches") or []),
        "changed_file_count": int(row.get("changed_file_count") or len(modified_paths)),
        "modified_paths_sample": modified_paths[:32],
        "modified_paths_truncated": len(modified_paths) > 32,
    }


def _build_object_batch_unit(
    *,
    batch_id: str,
    batch_type: str,
    source_date: str,
    summary_lines: list[str],
    facts: list[dict[str, Any]],
    currentness: str = "historical",
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rendered = "\n".join(line for line in summary_lines if str(line or "").strip())
    return {
        "unit_key": batch_id,
        "object_type": batch_type,
        "episode": {
            "topic_key": batch_id,
            "state_label": batch_type,
            "source_date": str(source_date or ""),
            "currentness": currentness,
            "raw_text": rendered,
            "provenance": {
                "source_section_path": batch_id,
                "raw_span": [0, len(rendered)],
            },
            "metadata": {
                "codebase": {
                    "object_type": batch_type,
                    "object_id": batch_id,
                    **(extra_metadata or {}),
                }
            },
        },
        "facts": facts,
    }


def _parse_remote_url(url: str) -> dict[str, str]:
    candidate = str(url or "").strip()
    match = _REMOTE_URL_RE.match(candidate)
    if not match:
        return {}
    repo_name = match.group("repo")
    if repo_name.endswith(".git"):
        repo_name = repo_name[:-4]
    host = match.group("host")
    provider = "github" if "github" in host else "gitlab" if "gitlab" in host else host
    return {
        "host": host,
        "provider": provider,
        "owner": match.group("owner"),
        "repo": repo_name,
    }


def _looks_like_diff(text: str, filename: str | None = None) -> bool:
    suffix = Path(filename or "").suffix.lower()
    return suffix in {".diff", ".patch"} or bool(_DIFF_MARKER_RE.search(str(text or "")))


def _parse_patch_lines(lines: Any, *, diff_scope: str) -> list[dict[str, Any]]:
    diffs: list[dict[str, Any]] = []
    current_diff: dict[str, Any] | None = None
    current_hunk: dict[str, Any] | None = None

    def _flush_hunk() -> None:
        nonlocal current_hunk
        if current_diff is None or current_hunk is None:
            current_hunk = None
            return
        current_diff.setdefault("hunks", []).append(current_hunk)
        current_hunk = None

    def _flush_diff() -> None:
        nonlocal current_diff
        _flush_hunk()
        if current_diff is not None:
            diffs.append(current_diff)
        current_diff = None

    for raw_line in lines:
        header_match = _PATCH_HEADER_RE.match(raw_line)
        if header_match:
            _flush_diff()
            old_path, new_path = header_match.groups()
            path = new_path if new_path != "/dev/null" else old_path
            current_diff = {
                "diff_id": f"{diff_scope}:{path}",
                "path": path,
                "old_path": old_path,
                "new_path": new_path,
                "status": "modified",
                "hunks": [],
            }
            continue
        if current_diff is None:
            continue
        if raw_line.startswith("new file mode "):
            current_diff["status"] = "added"
            continue
        if raw_line.startswith("deleted file mode "):
            current_diff["status"] = "deleted"
            continue
        if raw_line.startswith("rename from "):
            current_diff["status"] = "renamed"
            current_diff["rename_from"] = raw_line.split(" ", 2)[2]
            continue
        if raw_line.startswith("rename to "):
            current_diff["rename_to"] = raw_line.split(" ", 2)[2]
            continue
        if raw_line.startswith("index "):
            current_diff["index"] = raw_line.split(" ", 1)[1].strip()
            continue
        hunk_match = _HUNK_RE.match(raw_line)
        if hunk_match:
            _flush_hunk()
            current_hunk = {
                "header": raw_line.strip(),
                "label": str(hunk_match.group("label") or "").strip(),
                "old_start": int(hunk_match.group("old_start")),
                "old_count": int(hunk_match.group("old_count") or 1),
                "new_start": int(hunk_match.group("new_start")),
                "new_count": int(hunk_match.group("new_count") or 1),
                "line_count": 0,
            }
            continue
        if current_hunk is not None and not raw_line.startswith(("--- ", "+++ ")):
            current_hunk["line_count"] = int(current_hunk.get("line_count") or 0) + 1

    _flush_diff()
    return diffs


def _parse_patch(patch_text: str, *, diff_scope: str) -> list[dict[str, Any]]:
    return _parse_patch_lines(str(patch_text or "").splitlines(), diff_scope=diff_scope)


def _iter_git_stdout(repo_root: Path, *args: str):
    proc = subprocess.Popen(
        ["git", "-C", str(repo_root), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            yield line.rstrip("\n")
    finally:
        proc.stdout.close()
    stderr = proc.stderr.read() if proc.stderr is not None else ""
    if proc.stderr is not None:
        proc.stderr.close()
    code = proc.wait()
    if code != 0:
        raise RuntimeError(stderr.strip() or f"git {' '.join(args)} failed")


def _parse_patch_stream(
    repo_root: Path,
    *args: str,
    diff_scope: str,
    max_bytes: int = _MAX_PATCH_STREAM_BYTES,
) -> tuple[list[dict[str, Any]], bool]:
    proc = subprocess.Popen(
        ["git", "-C", str(repo_root), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert proc.stdout is not None
    diffs: list[dict[str, Any]] = []
    current_diff: dict[str, Any] | None = None
    current_hunk: dict[str, Any] | None = None
    truncated = False
    bytes_seen = 0

    def _flush_hunk() -> None:
        nonlocal current_hunk
        if current_diff is None or current_hunk is None:
            current_hunk = None
            return
        current_diff.setdefault("hunks", []).append(current_hunk)
        current_hunk = None

    def _flush_diff() -> None:
        nonlocal current_diff
        _flush_hunk()
        if current_diff is not None:
            diffs.append(current_diff)
        current_diff = None

    try:
        for raw_line in proc.stdout:
            bytes_seen += len(raw_line.encode("utf-8", "ignore"))
            if max_bytes > 0 and bytes_seen > max_bytes:
                truncated = True
                proc.kill()
                break
            line = raw_line.rstrip("\n")
            header_match = _PATCH_HEADER_RE.match(line)
            if header_match:
                _flush_diff()
                old_path, new_path = header_match.groups()
                path = new_path if new_path != "/dev/null" else old_path
                current_diff = {
                    "diff_id": f"{diff_scope}:{path}",
                    "path": path,
                    "old_path": old_path,
                    "new_path": new_path,
                    "status": "modified",
                    "hunks": [],
                }
                continue
            if current_diff is None:
                continue
            if line.startswith("new file mode "):
                current_diff["status"] = "added"
                continue
            if line.startswith("deleted file mode "):
                current_diff["status"] = "deleted"
                continue
            if line.startswith("rename from "):
                current_diff["status"] = "renamed"
                current_diff["rename_from"] = line.split(" ", 2)[2]
                continue
            if line.startswith("rename to "):
                current_diff["rename_to"] = line.split(" ", 2)[2]
                continue
            if line.startswith("index "):
                current_diff["index"] = line.split(" ", 1)[1].strip()
                continue
            hunk_match = _HUNK_RE.match(line)
            if hunk_match:
                _flush_hunk()
                current_hunk = {
                    "header": line.strip(),
                    "label": str(hunk_match.group("label") or "").strip(),
                    "old_start": int(hunk_match.group("old_start")),
                    "old_count": int(hunk_match.group("old_count") or 1),
                    "new_start": int(hunk_match.group("new_start")),
                    "new_count": int(hunk_match.group("new_count") or 1),
                    "line_count": 0,
                }
                continue
            if current_hunk is not None and not line.startswith(("--- ", "+++ ")):
                current_hunk["line_count"] = int(current_hunk.get("line_count") or 0) + 1
    finally:
        proc.stdout.close()
    stderr = proc.stderr.read() if proc.stderr is not None else ""
    if proc.stderr is not None:
        proc.stderr.close()
    code = proc.wait()
    _flush_diff()
    if truncated:
        return diffs, True
    if code != 0:
        raise RuntimeError(stderr.strip() or f"git {' '.join(args)} failed")
    return diffs, False


def _chunked(values: list[str], size: int) -> list[list[str]]:
    return [values[idx : idx + size] for idx in range(0, len(values), size)]


def _parse_name_status_rows(text: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for raw_line in str(text or "").splitlines():
        if not raw_line.strip():
            continue
        parts = raw_line.split("\t")
        if len(parts) < 2:
            continue
        status_token = parts[0].strip()
        status_letter = status_token[:1].upper()
        if status_letter in {"R", "C"} and len(parts) >= 3:
            old_path = parts[1].strip()
            new_path = parts[2].strip()
            rows.append(
                {
                    "status": "renamed" if status_letter == "R" else "copied",
                    "old_path": old_path,
                    "new_path": new_path,
                    "path": new_path or old_path,
                }
            )
            continue
        path = parts[1].strip()
        status = {
            "A": "added",
            "D": "deleted",
            "M": "modified",
            "T": "type_changed",
            "U": "unmerged",
        }.get(status_letter, "modified")
        rows.append(
            {
                "status": status,
                "old_path": path,
                "new_path": path,
                "path": path,
            }
        )
    return rows


def _file_blob_map(repo_root: Path) -> dict[str, str]:
    tracked: dict[str, str] = {}
    raw = _maybe_git(repo_root, "ls-files", "-s", "-z")
    if not raw:
        return tracked
    for item in raw.split("\0"):
        if not item.strip():
            continue
        meta, _, path = item.partition("\t")
        parts = meta.split()
        if len(parts) >= 2 and path:
            tracked[path] = parts[1]
    return tracked


def _repo_file_paths(repo_root: Path) -> list[str]:
    raw = _maybe_git(repo_root, "ls-files", "-z", "--cached", "--others", "--exclude-standard")
    paths = [item for item in raw.split("\0") if item]
    return sorted(set(paths))


def _build_commit_branch_map(repo_root: Path, branch_rows: list[dict[str, str]]) -> dict[str, list[str]]:
    commit_to_branches: dict[str, list[str]] = defaultdict(list)
    for branch in branch_rows:
        branch_name = str(branch.get("name") or "").strip()
        if not branch_name:
            continue
        for line in _maybe_git(repo_root, "rev-list", branch_name).splitlines():
            commit_sha = line.strip()
            if not commit_sha:
                continue
            commit_to_branches.setdefault(commit_sha, []).append(branch_name)
    return {sha: _dedupe_strings(names) for sha, names in commit_to_branches.items()}


def _commit_details(
    repo_root: Path,
    commit_sha: str,
    *,
    containing_branches: list[str],
) -> dict[str, Any]:
    meta_blob = _run_git(
        repo_root,
        "show",
        "--format=%H%x00%P%x00%an%x00%ae%x00%aI%x00%cn%x00%ce%x00%cI%x00%s%x00%b%x00"
        + _COMMIT_META_SENTINEL,
        "--name-status",
        "--find-renames",
        "--no-ext-diff",
        "--no-color",
        commit_sha,
    )
    meta_text, _, changed_blob = meta_blob.partition(_COMMIT_META_SENTINEL)
    meta = meta_text.split("\0")
    while len(meta) < 10:
        meta.append("")
    changed_files = _parse_name_status_rows(
        changed_blob
    )
    diffs_by_path = {
        str(row.get("path") or ""): dict(row)
        for row in changed_files
        if str(row.get("path") or "").strip()
    }
    if changed_files:
        diff_scope = f"commit:{commit_sha}"
        try:
            parsed_diffs, truncated = _parse_patch_stream(
                repo_root,
                "diff-tree",
                "--root",
                "-r",
                "-p",
                "--unified=0",
                "--no-ext-diff",
                "--no-color",
                "--no-renames",
                commit_sha,
                diff_scope=diff_scope,
            )
        except Exception:
            parsed_diffs = []
            truncated = False
        for diff in parsed_diffs:
            path = str(diff.get("path") or "")
            if path and path in diffs_by_path:
                merged = dict(diffs_by_path[path])
                merged.update(diff)
                merged["hydration_state"] = "deferred" if truncated else "hydrated"
                merged["patch_ref"] = f"git:show:{commit_sha}:{path}"
                diffs_by_path[path] = merged
    for path, diff in list(diffs_by_path.items()):
        if "hydration_state" not in diff:
            diff["hydration_state"] = "deferred"
            diff["patch_ref"] = f"git:show:{commit_sha}:{path}"
    diffs = [diffs_by_path[path] for path in sorted(diffs_by_path)]
    return {
        "sha": meta[0].strip(),
        "parents": [parent for parent in meta[1].split() if parent],
        "author_name": meta[2].strip(),
        "author_email": meta[3].strip(),
        "authored_at": meta[4].strip(),
        "committer_name": meta[5].strip(),
        "committer_email": meta[6].strip(),
        "committed_at": meta[7].strip(),
        "subject": meta[8].strip(),
        "body": meta[9].strip(),
        "diffs": diffs,
        "changed_file_count": len(changed_files),
        "branches": _dedupe_strings(containing_branches),
    }


def _tree_parent(path: str) -> str | None:
    if not path or path == ".":
        return None
    parent = str(Path(path).parent).replace("\\", "/")
    return "." if parent in {"", "."} else parent


def _directory_rows(file_paths: list[str]) -> list[dict[str, Any]]:
    all_dirs = {"."}
    for file_path in file_paths:
        cursor = _tree_parent(file_path) or "."
        while cursor:
            all_dirs.add(cursor)
            parent = _tree_parent(cursor)
            if parent is None:
                break
            cursor = parent
    return [
        {
            "path": directory,
            "parent": _tree_parent(directory) or "",
        }
        for directory in sorted(all_dirs)
    ]


def _hosting_commit_file_rows(commit_row: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for diff in commit_row.get("diffs", []):
        path = str(diff.get("path") or diff.get("rename_to") or diff.get("rename_from") or "").strip()
        if not path:
            continue
        rows.append(
            {
                "path": path,
                "status": str(diff.get("status") or "modified").strip() or "modified",
                "previous_filename": str(diff.get("rename_from") or "").strip(),
            }
        )
    return rows


def _add_repo_tree_units(
    *,
    repo_root: Path,
    repo_name: str,
    repo_id: str,
    tracked_blob_map: dict[str, str],
    file_paths: list[str],
    unit_sink: Any,
    object_counts: Counter,
) -> None:
    direct_dirs: dict[str, set[str]] = defaultdict(set)
    direct_files: dict[str, set[str]] = defaultdict(set)
    all_dirs = {"."}
    for file_path in file_paths:
        parent = _tree_parent(file_path) or "."
        direct_files[parent].add(file_path)
        cursor = parent
        while cursor:
            all_dirs.add(cursor)
            parent_cursor = _tree_parent(cursor)
            if parent_cursor is None:
                break
            direct_dirs[parent_cursor].add(cursor)
            cursor = parent_cursor

    for directory in sorted(all_dirs):
        dir_id = f"directory:{directory}"
        direct_child_dirs = sorted(direct_dirs.get(directory, set()))
        direct_child_files = sorted(direct_files.get(directory, set()))
        parent_dir = _tree_parent(directory)
        relation_facts: list[dict[str, Any]] = []
        for child_dir in direct_child_dirs:
            relation_facts.append(
                build_codebase_relation_fact(
                    relation_type="directory_contains_directory",
                    from_id=dir_id,
                    to_id=f"directory:{child_dir}",
                    fact_text=f"Directory {directory} contains directory {child_dir}.",
                    anchor=build_codebase_anchor("directory", path=child_dir),
                    entities=[directory, child_dir],
                )
            )
        for child_file in direct_child_files:
            relation_facts.append(
                build_codebase_relation_fact(
                    relation_type="directory_contains_file",
                    from_id=dir_id,
                    to_id=f"file:{child_file}",
                    fact_text=f"Directory {directory} contains file {child_file}.",
                    anchor=build_codebase_anchor("file", path=child_file),
                    entities=[directory, child_file],
                )
            )
        lines = [
            f"Directory: {directory}",
            f"Repository: {repo_name}",
            f"Direct child directories: {', '.join(direct_child_dirs) if direct_child_dirs else '(none)'}",
            f"Direct child files: {', '.join(direct_child_files[:12]) if direct_child_files else '(none)'}",
        ]
        unit_sink.append(
            build_codebase_object_unit(
                object_type="directory",
                object_id=dir_id,
                raw_text="\n".join(lines),
                fact_text=(
                    f"Directory {directory} exists in repository {repo_name} with "
                    f"{len(direct_child_dirs)} child directories and {len(direct_child_files)} files."
                ),
                anchor=build_codebase_anchor("directory", path=directory, local_path=str(repo_root / directory) if directory != "." else str(repo_root)),
                currentness="current",
                topic_key=directory.replace("/", "_") or "root",
                state_label="directory",
                entities=[directory, repo_name],
                tags=["directory", "tree"],
                extra_metadata={"path": directory, "repo_name": repo_name},
                relation_facts=relation_facts,
            )
        )
        object_counts["directory"] += 1
        if parent_dir and directory != ".":
            continue

    for file_path in file_paths:
        file_abspath = repo_root / file_path
        file_parent = _tree_parent(file_path) or "."
        blob_sha = tracked_blob_map.get(file_path, "")
        try:
            raw = file_abspath.read_bytes()
        except Exception:
            raw = b""
        lines = [
            f"File: {file_path}",
            f"Directory: {file_parent}",
            f"Blob SHA: {blob_sha or '(untracked)'}",
            f"Content hash: {_sha256_bytes(raw) if raw else '(unavailable)'}",
            f"Size bytes: {len(raw)}",
        ]
        unit_sink.append(
            build_codebase_object_unit(
                object_type="file",
                object_id=f"file:{file_path}",
                raw_text="\n".join(lines),
                fact_text=(
                    f"File {file_path} exists under directory {file_parent}"
                    + (f" with blob sha {blob_sha}." if blob_sha else ".")
                ),
                anchor=build_codebase_anchor(
                    "file",
                    path=file_path,
                    local_path=str(file_abspath),
                    blob_sha=blob_sha,
                    content_hash=_sha256_bytes(raw) if raw else "",
                ),
                currentness="current",
                topic_key=file_path.replace("/", "_"),
                state_label="file",
                entities=[file_path, file_parent],
                tags=["file", "tree"],
                extra_metadata={"path": file_path, "directory_path": file_parent, "blob_sha": blob_sha},
            )
        )
        object_counts["file"] += 1


def _add_commit_units(
    *,
    repo_name: str,
    branch_names: set[str],
    commit_rows: list[dict[str, Any]],
    unit_sink: Any,
    object_sink: Any,
    relation_sink: Any,
    object_counts: Counter,
) -> None:
    for row in commit_rows:
        commit_sha = row["sha"]
        commit_type = "merge" if len(row["parents"]) > 1 else "commit"
        commit_id = f"commit:{commit_sha}"
        modified_paths = [diff["path"] for diff in row["diffs"]]
        relation_facts: list[dict[str, Any]] = []
        for parent_sha in row["parents"]:
            relation_facts.append(
                build_codebase_relation_fact(
                    relation_type="commit_parent_of_commit",
                    from_id=f"commit:{parent_sha}",
                    to_id=commit_id,
                    fact_text=f"{parent_sha} parent_of {commit_sha}",
                    anchor=build_codebase_anchor("commit", commit_sha=commit_sha, parent_sha=parent_sha),
                    entities=[parent_sha, commit_sha],
                )
            )
        for branch_name in row["branches"]:
            relation_type = "branch_contains_commit"
            relation_facts.append(
                build_codebase_relation_fact(
                    relation_type=relation_type,
                    from_id=f"branch:{branch_name}",
                    to_id=commit_id,
                    fact_text=f"{branch_name} contains {commit_sha}",
                    anchor=build_codebase_anchor("branch", branch_name=branch_name, commit_sha=commit_sha),
                    entities=[branch_name, commit_sha],
                )
            )
        for file_path in modified_paths:
            relation_sink.append(
                {
                    "relation_type": "commit_modifies_file",
                    "from_id": commit_id,
                    "to_id": f"file:{file_path}",
                    "fact": f"{commit_sha} modifies {file_path}",
                    "anchor": build_codebase_anchor("file", path=file_path, commit_sha=commit_sha),
                    "entities": [commit_sha, file_path],
                }
            )
            relation_sink.append(
                {
                    "relation_type": "commit_contains_diff",
                    "from_id": commit_id,
                    "to_id": f"diff:commit:{commit_sha}:{file_path}",
                    "fact": f"{commit_sha} diff {file_path}",
                    "anchor": build_codebase_anchor("diff", commit_sha=commit_sha, path=file_path),
                    "entities": [commit_sha, file_path],
                }
            )
        commit_lines = [
            f"Commit SHA: {commit_sha}",
            f"Commit type: {commit_type}",
            f"Parents: {', '.join(row['parents']) if row['parents'] else '(root)'}",
            f"Author: {row['author_name']} <{row['author_email']}>",
            f"Committer: {row['committer_name']} <{row['committer_email']}>",
            f"Authored at: {row['authored_at']}",
            f"Committed at: {row['committed_at']}",
            f"Subject: {row['subject'] or '(none)'}",
            f"Containing branches: {', '.join(row['branches']) if row['branches'] else '(unknown)'}",
            f"Changed file count: {row['changed_file_count']}",
            f"Modified files: {', '.join(modified_paths[:8]) if modified_paths else '(none)'}",
        ]
        if len(modified_paths) > 8:
            commit_lines.append(
                f"Commit-file relations: {row['changed_file_count']} changed files are persisted; raw text preview shows the first 8."
            )
        if row["body"]:
            commit_lines.append(f"Body: {row['body']}")
        unit_sink.append(
            build_codebase_object_unit(
                object_type=commit_type,
                object_id=commit_id,
                raw_text="\n".join(commit_lines),
                fact_text=(
                    f"Commit {commit_sha} in repository {repo_name} was authored by {row['author_name']} "
                    f"and committed by {row['committer_name']}. Subject: {row['subject'] or '(none)'}."
                ),
                anchor=build_codebase_anchor(
                    "commit",
                    commit_sha=commit_sha,
                    parent_shas=row["parents"],
                    authored_at=row["authored_at"],
                    committed_at=row["committed_at"],
                ),
                source_date=row["committed_at"],
                currentness="current" if any(branch in branch_names for branch in row["branches"]) else "historical",
                topic_key=commit_sha[:12],
                state_label=commit_type,
                entities=[commit_sha, row["author_name"], row["committer_name"], *modified_paths[:6]],
                tags=[commit_type, "git"],
                extra_metadata={
                    "commit_sha": commit_sha,
                    "parent_shas": row["parents"],
                    "author": row["author_name"],
                    "committer": row["committer_name"],
                    "subject": row["subject"],
                    "branches": row["branches"],
                    "changed_file_count": row["changed_file_count"],
                    "modified_files_sample": modified_paths[:16],
                    "modified_files_truncated": len(modified_paths) > 16,
                },
                relation_facts=relation_facts,
            )
        )
        object_counts[commit_type] += 1

        for diff in row["diffs"]:
            diff_path = diff.get("path") or diff.get("rename_to") or diff.get("rename_from") or ""
            diff_id = f"diff:commit:{commit_sha}:{diff_path}"
            emitted_hunk = False
            for hunk_idx, hunk in enumerate(diff.get("hunks", []), start=1):
                emitted_hunk = True
                hunk_id = f"hunk:commit:{commit_sha}:{diff_path}:{hunk_idx}"
                relation_sink.append(
                    {
                        "relation_type": "diff_contains_hunk",
                        "from_id": diff_id,
                        "to_id": hunk_id,
                        "fact": f"{commit_sha} {diff_path} {hunk.get('header', '')}",
                        "anchor": build_codebase_anchor(
                            "hunk",
                            commit_sha=commit_sha,
                            path=diff_path,
                            hunk_header=hunk.get("header"),
                            old_range=[hunk.get("old_start"), hunk.get("old_count")],
                            new_range=[hunk.get("new_start"), hunk.get("new_count")],
                        ),
                        "entities": [commit_sha, diff_path, hunk.get("header", "")],
                    }
                )
                hunk_lines = [
                    f"Hunk header: {hunk.get('header', '')}",
                    f"Commit SHA: {commit_sha}",
                    f"File path: {diff_path}",
                    f"Old range: {hunk.get('old_start')}:{hunk.get('old_count')}",
                    f"New range: {hunk.get('new_start')}:{hunk.get('new_count')}",
                    f"Changed line count: {hunk.get('line_count', 0)}",
                ]
                object_sink.append(
                    build_codebase_object_row(
                        object_type="hunk",
                        object_id=hunk_id,
                        fact_text=(
                            f"Hunk {hunk.get('header', '')} belongs to commit {commit_sha} and file {diff_path}."
                        ),
                        anchor=build_codebase_anchor(
                            "hunk",
                            commit_sha=commit_sha,
                            path=diff_path,
                            hunk_header=hunk.get("header"),
                            old_range=[hunk.get("old_start"), hunk.get("old_count")],
                            new_range=[hunk.get("new_start"), hunk.get("new_count")],
                        ),
                        entities=[commit_sha, diff_path, hunk.get("header", "")],
                        tags=["hunk", diff.get("status", "modified"), "git"],
                        extra_metadata={
                            "commit_sha": commit_sha,
                            "path": diff_path,
                            "hunk_header": hunk.get("header"),
                            "old_range": [hunk.get("old_start"), hunk.get("old_count")],
                            "new_range": [hunk.get("new_start"), hunk.get("new_count")],
                            "status": diff.get("status", "modified"),
                            "hydration_state": diff.get("hydration_state", "hydrated"),
                            "patch_ref": diff.get("patch_ref", ""),
                        },
                    )
                )
                object_counts["hunk"] += 1
            if not emitted_hunk and str(diff.get("hydration_state") or "deferred") == "deferred":
                hunk_id = f"hunk:commit:{commit_sha}:{diff_path}:deferred"
                relation_sink.append(
                    {
                        "relation_type": "diff_contains_hunk",
                        "from_id": diff_id,
                        "to_id": hunk_id,
                        "fact": f"{commit_sha} {diff_path} deferred patch hydration",
                        "anchor": build_codebase_anchor(
                            "hunk",
                            commit_sha=commit_sha,
                            path=diff_path,
                            patch_ref=diff.get("patch_ref", ""),
                            hydration_state="deferred",
                        ),
                        "entities": [commit_sha, diff_path],
                    }
                )
                object_sink.append(
                    build_codebase_object_row(
                        object_type="hunk",
                        object_id=hunk_id,
                        fact_text=(
                            f"Hunk details for commit {commit_sha} and file {diff_path} are deferred to {diff.get('patch_ref', '(none)')}."
                        ),
                        anchor=build_codebase_anchor(
                            "hunk",
                            commit_sha=commit_sha,
                            path=diff_path,
                            patch_ref=diff.get("patch_ref", ""),
                            hydration_state="deferred",
                        ),
                        entities=[commit_sha, diff_path],
                        tags=["hunk", diff.get("status", "modified"), "git", "deferred"],
                        extra_metadata={
                            "commit_sha": commit_sha,
                            "path": diff_path,
                            "status": diff.get("status", "modified"),
                            "hydration_state": "deferred",
                            "patch_ref": diff.get("patch_ref", ""),
                        },
                    )
                )
                object_counts["hunk"] += 1
            object_sink.append(
                build_codebase_object_row(
                    object_type="diff",
                    object_id=diff_id,
                    fact_text=f"Diff for commit {commit_sha} changes file {diff_path} with {len(diff.get('hunks', []))} hunks.",
                    anchor=build_codebase_anchor(
                        "diff",
                        commit_sha=commit_sha,
                        path=diff_path,
                        status=diff.get("status", "modified"),
                    ),
                    entities=[commit_sha, diff_path],
                    tags=["diff", diff.get("status", "modified"), "git"],
                    extra_metadata={
                        "commit_sha": commit_sha,
                        "path": diff_path,
                        "status": diff.get("status", "modified"),
                        "rename_from": diff.get("rename_from", ""),
                        "rename_to": diff.get("rename_to", ""),
                        "hydration_state": diff.get("hydration_state", "deferred"),
                        "patch_ref": diff.get("patch_ref", ""),
                        "hunk_count": len(diff.get("hunks", [])),
                    },
                )
            )
            object_counts["diff"] += 1


def _build_repo_bundle(repo_root: Path, logical_source_id: str) -> dict[str, Any]:
    repo_root = Path(_run_git(repo_root, "rev-parse", "--show-toplevel").strip()).resolve()
    head_sha = _maybe_git(repo_root, "rev-parse", "HEAD").strip()
    head_branch = _maybe_git(repo_root, "rev-parse", "--abbrev-ref", "HEAD").strip() or "HEAD"
    upstream_branch = _maybe_git(repo_root, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}").strip()
    remote_lines = _maybe_git(repo_root, "remote", "-v").splitlines()
    remote_rows: list[dict[str, str]] = []
    seen_remote_names: set[str] = set()
    for line in remote_lines:
        parts = line.split()
        if len(parts) < 3 or parts[2] != "(fetch)":
            continue
        name = parts[0].strip()
        if not name or name in seen_remote_names:
            continue
        seen_remote_names.add(name)
        url = parts[1].strip()
        parsed = _parse_remote_url(url)
        remote_rows.append({"name": name, "url": url, **parsed})

    preferred_remote = remote_rows[0] if remote_rows else {}
    repo_name = preferred_remote.get("repo") or repo_root.name
    repo_owner = preferred_remote.get("owner") or ""
    provider = preferred_remote.get("provider") or "local"
    remote_url = preferred_remote.get("url") or ""
    repo_display = f"{repo_owner}/{repo_name}" if repo_owner else repo_name
    repo_id = f"repository:{repo_display}"

    branch_rows: list[dict[str, str]] = []
    branch_output = _maybe_git(
        repo_root,
        "for-each-ref",
        "refs/heads",
        "refs/remotes",
        "--format=%(refname:short)\t%(objectname)\t%(upstream:short)",
    )
    for line in branch_output.splitlines():
        name, _, rest = line.partition("\t")
        if not rest:
            continue
        commit_sha, _, upstream = rest.partition("\t")
        branch_rows.append(
            {
                "name": name.strip(),
                "commit_sha": commit_sha.strip(),
                "upstream": upstream.strip(),
            }
        )
    branch_names = {row["name"] for row in branch_rows if row.get("name")}

    tag_rows: list[dict[str, str]] = []
    tag_output = _maybe_git(
        repo_root,
        "for-each-ref",
        "--sort=-creatordate",
        "refs/tags",
        "--format=%(refname:short)\t%(objectname)\t%(creatordate:iso-strict)\t%(subject)",
    )
    for line in tag_output.splitlines():
        name, _, rest = line.partition("\t")
        if not rest:
            continue
        object_sha, _, tail = rest.partition("\t")
        created_at, _, subject = tail.partition("\t")
        tag_rows.append(
            {
                "name": name.strip(),
                "commit_sha": object_sha.strip(),
                "created_at": created_at.strip(),
                "subject": subject.strip(),
            }
        )

    commit_shas = [
        line.strip()
        for line in _maybe_git(repo_root, "rev-list", "--all", "--date-order").splitlines()
        if line.strip()
    ]
    tracked_blob_map = _file_blob_map(repo_root)
    file_paths = _repo_file_paths(repo_root)
    directory_rows = _directory_rows(file_paths)
    tip_branch_map: dict[str, list[str]] = defaultdict(list)
    for branch in branch_rows:
        branch_name = str(branch.get("name") or "").strip()
        commit_sha = str(branch.get("commit_sha") or "").strip()
        if branch_name and commit_sha:
            tip_branch_map[commit_sha].append(branch_name)
    graph_commits: list[dict[str, Any]] = []

    unit_spool = _UnitSpool()
    object_spool = _UnitSpool()
    relation_spool = _UnitSpool()
    object_counts: Counter = Counter()
    digest = _DigestAccumulator()

    def _append_unit(unit: dict[str, Any]) -> None:
        unit_spool.append(unit)
        digest.update(unit)

    class _AppendSink:
        def append(self, unit: dict[str, Any]) -> None:
            _append_unit(unit)

    unit_sink = _AppendSink()

    repo_relation_facts: list[dict[str, Any]] = []
    repo_relation_facts.append(
        build_codebase_relation_fact(
            relation_type="repo_contains_directory",
            from_id=repo_id,
            to_id="directory:.",
            fact_text=f"Repository {repo_display} contains directory .",
            anchor=build_codebase_anchor("directory", path="."),
            entities=[repo_display, "."],
        )
    )
    if remote_rows:
        for remote in remote_rows:
            repo_relation_facts.append(
                build_codebase_relation_fact(
                    relation_type="repo_has_remote",
                    from_id=repo_id,
                    to_id=f"remote:{remote['name']}",
                    fact_text=f"Repository {repo_display} has remote {remote['name']} at {remote['url']}.",
                    anchor=build_codebase_anchor("remote", remote_name=remote["name"], remote_url=remote["url"]),
                    entities=[repo_display, remote["name"], remote["url"]],
                )
            )
    repo_relation_facts.append(
        build_codebase_relation_fact(
            relation_type="repo_has_worktree",
            from_id=repo_id,
            to_id=f"worktree:{repo_root}",
            fact_text=f"Repository {repo_display} has worktree {repo_root}.",
            anchor=build_codebase_anchor("worktree", local_path=str(repo_root)),
            entities=[repo_display, str(repo_root)],
        )
    )
    for branch in branch_rows:
        repo_relation_facts.append(
            build_codebase_relation_fact(
                relation_type="repo_has_branch",
                from_id=repo_id,
                to_id=f"branch:{branch['name']}",
                fact_text=f"Repository {repo_display} has branch {branch['name']}.",
                anchor=build_codebase_anchor("branch", branch_name=branch["name"], commit_sha=branch["commit_sha"]),
                entities=[repo_display, branch["name"], branch["commit_sha"]],
            )
        )
    for tag in tag_rows:
        repo_relation_facts.append(
            build_codebase_relation_fact(
                relation_type="repo_has_tag",
                from_id=repo_id,
                to_id=f"tag:{tag['name']}",
                fact_text=f"Repository {repo_display} has tag {tag['name']}.",
                anchor=build_codebase_anchor("tag", tag_name=tag["name"], commit_sha=tag["commit_sha"]),
                entities=[repo_display, tag["name"], tag["commit_sha"]],
            )
        )

    repo_lines = [
        f"Repository: {repo_display}",
        f"Local path: {repo_root}",
        f"Provider: {provider}",
        f"Remote URL: {remote_url or '(none)'}",
        f"HEAD branch: {head_branch}",
        f"HEAD commit: {head_sha}",
        f"Tracked upstream: {upstream_branch or '(none)'}",
    ]
    _append_unit(
        build_codebase_object_unit(
            object_type="repository",
            object_id=repo_id,
            raw_text="\n".join(repo_lines),
            fact_text=(
                f"Repository {repo_display} is available at {repo_root}"
                + (f" with remote {remote_url}." if remote_url else ".")
            ),
            anchor=build_codebase_anchor(
                "repository",
                repo_name=repo_name,
                repo_owner=repo_owner,
                provider=provider,
                local_path=str(repo_root),
                remote_url=remote_url,
            ),
            currentness="current",
            topic_key=repo_name,
            state_label="repository",
            entities=[repo_display, repo_name, repo_owner, head_branch, head_sha],
            tags=["repository", provider],
            extra_metadata={
                "repo_name": repo_name,
                "repo_owner": repo_owner,
                "provider": provider,
                "local_path": str(repo_root),
                "remote_url": remote_url,
                "head_branch": head_branch,
                "head_commit": head_sha,
            },
            relation_facts=repo_relation_facts,
        )
    )
    object_counts["repository"] += 1

    worktree_relation_facts = []
    if head_branch and head_branch != "HEAD":
        worktree_relation_facts.append(
            build_codebase_relation_fact(
                relation_type="worktree_tracks_branch",
                from_id=f"worktree:{repo_root}",
                to_id=f"branch:{head_branch}",
                fact_text=f"Worktree {repo_root} tracks branch {head_branch}.",
                anchor=build_codebase_anchor("worktree", local_path=str(repo_root), branch_name=head_branch),
                entities=[str(repo_root), head_branch],
            )
        )
    _append_unit(
        build_codebase_object_unit(
            object_type="worktree",
            object_id=f"worktree:{repo_root}",
            raw_text="\n".join(
                [
                    f"Worktree path: {repo_root}",
                    f"Repository: {repo_display}",
                    f"Tracked branch: {head_branch}",
                    f"Tracked upstream: {upstream_branch or '(none)'}",
                    f"HEAD commit: {head_sha}",
                ]
            ),
            fact_text=(
                f"Worktree {repo_root} belongs to repository {repo_display}, tracks branch {head_branch}, "
                f"and is at HEAD commit {head_sha}."
            ),
            anchor=build_codebase_anchor(
                "worktree",
                local_path=str(repo_root),
                branch_name=head_branch,
                upstream_branch=upstream_branch,
                commit_sha=head_sha,
            ),
            currentness="current",
            topic_key=head_branch or "worktree",
            state_label="worktree",
            entities=[str(repo_root), repo_display, head_branch, head_sha],
            tags=["worktree"],
            extra_metadata={
                "local_path": str(repo_root),
                "branch_name": head_branch,
                "upstream_branch": upstream_branch,
                "commit_sha": head_sha,
            },
            relation_facts=worktree_relation_facts,
        )
    )
    object_counts["worktree"] += 1

    for remote in remote_rows:
        _append_unit(
            build_codebase_object_unit(
                object_type="remote",
                object_id=f"remote:{remote['name']}",
                raw_text="\n".join(
                    [
                        f"Remote name: {remote['name']}",
                        f"Remote URL: {remote['url']}",
                        f"Provider: {remote.get('provider') or 'unknown'}",
                        f"Host: {remote.get('host') or 'unknown'}",
                    ]
                ),
                fact_text=f"Remote {remote['name']} points to {remote['url']}.",
                anchor=build_codebase_anchor(
                    "remote",
                    remote_name=remote["name"],
                    remote_url=remote["url"],
                    provider=remote.get("provider"),
                    host=remote.get("host"),
                ),
                currentness="current",
                topic_key=remote["name"],
                state_label="remote",
                entities=[remote["name"], remote["url"], remote.get("host", "")],
                tags=["remote", remote.get("provider", "local")],
                extra_metadata=remote,
            )
        )
        object_counts["remote"] += 1

    for branch in branch_rows:
        relation_facts = [
            build_codebase_relation_fact(
                relation_type="branch_points_to_commit",
                from_id=f"branch:{branch['name']}",
                to_id=f"commit:{branch['commit_sha']}",
                fact_text=f"Branch {branch['name']} points to commit {branch['commit_sha']}.",
                anchor=build_codebase_anchor("branch", branch_name=branch["name"], commit_sha=branch["commit_sha"]),
                entities=[branch["name"], branch["commit_sha"]],
            )
        ]
        _append_unit(
            build_codebase_object_unit(
                object_type="branch",
                object_id=f"branch:{branch['name']}",
                raw_text="\n".join(
                    [
                        f"Branch name: {branch['name']}",
                        f"Commit SHA: {branch['commit_sha']}",
                        f"Upstream: {branch['upstream'] or '(none)'}",
                    ]
                ),
                fact_text=(
                    f"Branch {branch['name']} currently points to commit {branch['commit_sha']}"
                    + (f" and tracks upstream {branch['upstream']}." if branch["upstream"] else ".")
                ),
                anchor=build_codebase_anchor(
                    "branch",
                    branch_name=branch["name"],
                    commit_sha=branch["commit_sha"],
                    upstream_branch=branch["upstream"],
                ),
                currentness="current" if branch["name"] == head_branch else "historical",
                topic_key=branch["name"],
                state_label="branch",
                entities=[branch["name"], branch["commit_sha"], branch["upstream"]],
                tags=["branch"],
                extra_metadata={
                    "branch_name": branch["name"],
                    "commit_sha": branch["commit_sha"],
                    "upstream_branch": branch["upstream"],
                },
                relation_facts=relation_facts,
            )
        )
        object_counts["branch"] += 1

    for tag in tag_rows:
        _append_unit(
            build_codebase_object_unit(
                object_type="tag",
                object_id=f"tag:{tag['name']}",
                raw_text="\n".join(
                    [
                        f"Tag name: {tag['name']}",
                        f"Commit SHA: {tag['commit_sha']}",
                        f"Created at: {tag['created_at'] or '(unknown)'}",
                        f"Subject: {tag['subject'] or '(none)'}",
                    ]
                ),
                fact_text=f"Tag {tag['name']} points to commit {tag['commit_sha']}.",
                anchor=build_codebase_anchor(
                    "tag",
                    tag_name=tag["name"],
                    commit_sha=tag["commit_sha"],
                    created_at=tag["created_at"],
                ),
                source_date=tag["created_at"],
                currentness="historical",
                topic_key=tag["name"],
                state_label="tag",
                entities=[tag["name"], tag["commit_sha"]],
                tags=["tag"],
                extra_metadata=tag,
            )
        )
        object_counts["tag"] += 1

    _add_repo_tree_units(
        repo_root=repo_root,
        repo_name=repo_display,
        repo_id=repo_id,
        tracked_blob_map=tracked_blob_map,
        file_paths=file_paths,
        unit_sink=unit_sink,
        object_counts=object_counts,
    )
    def _build_commit_row(commit_sha: str) -> dict[str, Any]:
        return _commit_details(
            repo_root,
            commit_sha,
            containing_branches=tip_branch_map.get(commit_sha, []),
        )

    commit_rows: list[dict[str, Any]] = []
    hosting_executor: ThreadPoolExecutor | None = None
    hosting_future: Future[dict[str, Any]] | None = None
    try:
        with ThreadPoolExecutor(max_workers=_commit_detail_worker_count(len(commit_shas))) as pool:
            for row in pool.map(_build_commit_row, commit_shas):
                commit_rows.append(row)
                graph_commits.append(_commit_graph_row(row))
        known_commit_files = {
            str(row.get("sha") or "").strip(): _hosting_commit_file_rows(row)
            for row in commit_rows
            if str(row.get("sha") or "").strip()
        }
        historical_file_paths = {
            str(file_row.get("path") or "").strip()
            for rows in known_commit_files.values()
            for file_row in rows
            if str(file_row.get("path") or "").strip()
        }
        known_hosting_files = set(file_paths) | historical_file_paths
        if provider != "local":
            hosting_executor = ThreadPoolExecutor(max_workers=1)
            hosting_future = hosting_executor.submit(
                build_hosting_stage1_units,
                repo_id=repo_id,
                repo_display=repo_display,
                repo_owner=repo_owner,
                repo_name=repo_name,
                provider=provider,
                host=str(preferred_remote.get("host") or ""),
                known_branches=set(branch_names),
                known_commits=set(commit_shas),
                known_files=known_hosting_files,
                known_commit_files=known_commit_files,
            )
        for row in commit_rows:
            _add_commit_units(
                repo_name=repo_display,
                branch_names=branch_names,
                commit_rows=[row],
                unit_sink=unit_sink,
                object_sink=object_spool,
                relation_sink=relation_spool,
                object_counts=object_counts,
            )

        if hosting_future is not None:
            hosting_bundle = hosting_future.result()
        else:
            hosting_bundle = build_hosting_stage1_units(
                repo_id=repo_id,
                repo_display=repo_display,
                repo_owner=repo_owner,
                repo_name=repo_name,
                provider=provider,
                host=str(preferred_remote.get("host") or ""),
                known_branches=set(branch_names),
                known_commits=set(commit_shas),
                known_files=known_hosting_files,
                known_commit_files=known_commit_files,
            )
    finally:
        if hosting_executor is not None:
            hosting_executor.shutdown(wait=True)

    for unit in list(hosting_bundle.get("units") or []):
        _append_unit(unit)
    object_counts.update(Counter((hosting_bundle.get("summary") or {}).get("object_counts") or {}))
    persisted_object_counts = Counter(object_counts)
    content_hash = digest.hexdigest()
    unit_spool.close()
    object_spool.close()
    relation_spool.close()
    return {
        "unit_spool_ref": unit_spool.path,
        "object_spool_ref": object_spool.path,
        "relation_spool_ref": relation_spool.path,
        "content_hash": content_hash,
        "summary": {
            "repo_name": repo_display,
            "repo_root": str(repo_root),
            "head_commit": head_sha,
            "head_branch": head_branch,
            "object_counts": dict(persisted_object_counts),
            "commit_count": len(commit_shas),
            "file_count": len(file_paths),
            "hosting": dict(hosting_bundle.get("summary") or {}),
            "relation_store": {"status": "ok"},
        },
        "source_meta": {
            "codebase_stage": "stage1",
            "repo_root": str(repo_root),
            "repo_name": repo_display,
            "repo_owner": repo_owner,
            "provider": provider,
            "remote_url": remote_url,
            "head_commit": head_sha,
            "head_branch": head_branch,
            "upstream_branch": upstream_branch,
            "hosting": dict(hosting_bundle.get("summary") or {}),
        },
        "graph": {
            "repo": {
                "repo_id": repo_id,
                "repo_name": repo_name,
                "repo_owner": repo_owner,
                "repo_display": repo_display,
                "repo_root": str(repo_root),
                "provider": provider,
                "remote_url": remote_url,
                "head_commit": head_sha,
                "head_branch": head_branch,
                "upstream_branch": upstream_branch,
            },
            "branches": branch_rows,
            "tags": tag_rows,
            "directories": directory_rows,
            "files": [
                {
                    "path": path,
                    "directory_path": _tree_parent(path) or ".",
                    "blob_sha": tracked_blob_map.get(path, ""),
                }
                for path in file_paths
            ],
            "commits": graph_commits,
            "hosting": dict(hosting_bundle.get("graph") or {}),
            "object_counts": dict(persisted_object_counts),
        },
    }


def _build_file_bundle(path: Path, logical_source_id: str, content: str | None = None) -> dict[str, Any]:
    path = path.resolve()
    raw_text = content if content is not None else path.read_text(encoding="utf-8", errors="replace")
    parent = path.parent.name or "."
    unit = build_codebase_object_unit(
        object_type="file",
        object_id=f"file:{path.name}",
        raw_text="\n".join(
            [
                f"File: {path.name}",
                f"Local path: {path}",
                f"Directory: {path.parent}",
                f"Content hash: {_sha256_text(raw_text)}",
            ]
        ),
        fact_text=f"File {path.name} exists at {path}.",
        anchor=build_codebase_anchor("file", path=str(path), local_path=str(path), content_hash=_sha256_text(raw_text)),
        currentness="current",
        topic_key=path.name,
        state_label="file",
        entities=[path.name, str(path.parent), parent],
        tags=["file", "codebase"],
        extra_metadata={"path": str(path), "directory_path": str(path.parent)},
    )
    return {
        "units": [unit],
        "content_hash": _sha256_text(raw_text),
        "summary": {
            "repo_name": logical_source_id,
            "repo_root": str(path.parent),
            "object_counts": {"file": 1},
            "commit_count": 0,
            "file_count": 1,
            "hosting": {"status": "not_applicable", "object_counts": {}, "relation_count": 0},
        },
        "source_meta": {
            "codebase_stage": "stage1",
            "repo_root": str(path.parent),
            "repo_name": logical_source_id,
            "head_commit": "",
            "head_branch": "",
        },
        "graph": {
            "repo": {
                "repo_id": f"repository:{logical_source_id}",
                "repo_name": logical_source_id,
                "repo_owner": "",
                "repo_display": logical_source_id,
                "repo_root": str(path.parent),
                "provider": "local",
                "remote_url": "",
                "head_commit": "",
                "head_branch": "",
                "upstream_branch": "",
            },
            "branches": [],
            "tags": [],
            "directories": [{"path": ".", "parent": ""}],
            "files": [
                {
                    "path": str(path),
                    "directory_path": str(path.parent),
                    "blob_sha": "",
                }
            ],
            "commits": [],
            "hosting": {"status": "not_applicable", "object_counts": {}, "relation_count": 0},
            "object_counts": {"file": 1},
        },
    }


def _build_inline_diff_bundle(text: str, logical_source_id: str, filename: str | None) -> dict[str, Any]:
    diff_scope = f"inline:{logical_source_id}"
    parsed_diffs = _parse_patch(text, diff_scope=diff_scope)
    units: list[dict[str, Any]] = []
    object_counts: Counter = Counter()
    for diff in parsed_diffs:
        diff_path = diff.get("path") or filename or "inline.patch"
        diff_id = f"diff:inline:{diff_path}"
        relation_facts = []
        for hunk_idx, hunk in enumerate(diff.get("hunks", []), start=1):
            hunk_id = f"hunk:inline:{diff_path}:{hunk_idx}"
            relation_facts.append(
                build_codebase_relation_fact(
                    relation_type="diff_contains_hunk",
                    from_id=diff_id,
                    to_id=hunk_id,
                    fact_text=f"Inline diff {diff_id} contains hunk {hunk.get('header', '')}.",
                    anchor=build_codebase_anchor(
                        "hunk",
                        path=diff_path,
                        hunk_header=hunk.get("header"),
                        old_range=[hunk.get("old_start"), hunk.get("old_count")],
                        new_range=[hunk.get("new_start"), hunk.get("new_count")],
                    ),
                    entities=[diff_path, hunk.get("header", "")],
                )
            )
            units.append(
                build_codebase_object_unit(
                    object_type="hunk",
                    object_id=hunk_id,
                    raw_text="\n".join([hunk.get("header", ""), *(hunk.get("lines") or [])]),
                    fact_text=f"Hunk {hunk.get('header', '')} belongs to inline diff {diff_id}.",
                    anchor=build_codebase_anchor(
                        "hunk",
                        path=diff_path,
                        hunk_header=hunk.get("header"),
                        old_range=[hunk.get("old_start"), hunk.get("old_count")],
                        new_range=[hunk.get("new_start"), hunk.get("new_count")],
                    ),
                    currentness="current",
                    topic_key=f"{diff_path}_{hunk_idx}",
                    state_label="hunk",
                    entities=[diff_path, hunk.get("header", "")],
                    tags=["hunk", "diff"],
                    extra_metadata={"path": diff_path, "hunk_header": hunk.get("header")},
                )
            )
            object_counts["hunk"] += 1
        units.append(
            build_codebase_object_unit(
                object_type="diff",
                object_id=diff_id,
                raw_text="\n".join(
                    [
                        f"Inline diff file path: {diff_path}",
                        f"Status: {diff.get('status', 'modified')}",
                        f"Hunk count: {len(diff.get('hunks', []))}",
                    ]
                ),
                fact_text=f"Inline diff changes file {diff_path} with {len(diff.get('hunks', []))} hunks.",
                anchor=build_codebase_anchor("diff", path=diff_path, status=diff.get("status", "modified")),
                currentness="current",
                topic_key=diff_path.replace("/", "_"),
                state_label="diff",
                entities=[diff_path],
                tags=["diff"],
                extra_metadata={"path": diff_path, "status": diff.get("status", "modified")},
                relation_facts=relation_facts,
            )
        )
        object_counts["diff"] += 1
    digest = _sha256_text(text)
    return {
        "units": units,
        "content_hash": digest,
        "summary": {
            "repo_name": logical_source_id,
            "repo_root": "",
            "object_counts": dict(object_counts),
            "commit_count": 0,
            "file_count": len(parsed_diffs),
            "hosting": {"status": "not_applicable", "object_counts": {}, "relation_count": 0},
        },
        "source_meta": {
            "codebase_stage": "stage1",
            "repo_root": "",
            "repo_name": logical_source_id,
            "head_commit": "",
            "head_branch": "",
        },
        "graph": {
            "repo": {
                "repo_id": f"repository:{logical_source_id}",
                "repo_name": logical_source_id,
                "repo_owner": "",
                "repo_display": logical_source_id,
                "repo_root": "",
                "provider": "inline",
                "remote_url": "",
                "head_commit": "",
                "head_branch": "",
                "upstream_branch": "",
            },
            "branches": [],
            "tags": [],
            "directories": [],
            "files": [
                {
                    "path": str(diff.get("path") or filename or "inline.patch"),
                    "directory_path": _tree_parent(str(diff.get("path") or filename or "inline.patch")) or ".",
                    "blob_sha": "",
                }
                for diff in parsed_diffs
            ],
            "commits": [],
            "hosting": {"status": "not_applicable", "object_counts": {}, "relation_count": 0},
            "object_counts": dict(object_counts),
        },
    }


def build_codebase_stage1_bundle(
    *,
    locator: str | None,
    content: str | None,
    filename: str | None,
    mime: str | None,
    logical_source_id: str,
) -> dict[str, Any]:
    del mime  # reserved for future provider-specific resolvers
    path = Path(str(locator or "")).expanduser() if locator else None
    if path and path.exists() and path.is_dir():
        return _build_repo_bundle(path, logical_source_id)
    if path and path.exists() and path.is_file():
        if _looks_like_diff(content or "", path.name):
            return _build_inline_diff_bundle(content or path.read_text(encoding="utf-8", errors="replace"), logical_source_id, path.name)
        return _build_file_bundle(path, logical_source_id, content=content)
    if _looks_like_diff(content or "", filename):
        return _build_inline_diff_bundle(content or "", logical_source_id, filename)
    return {
        "units": [
            build_codebase_object_unit(
                object_type="file",
                object_id=f"file:{filename or logical_source_id}",
                raw_text="\n".join(
                    [
                        f"Inline codebase object: {filename or logical_source_id}",
                        f"Content hash: {_sha256_text(content or '')}",
                    ]
                ),
                fact_text=f"Inline codebase object {filename or logical_source_id} is available for retrieval.",
                anchor=build_codebase_anchor("inline", filename=filename or logical_source_id, content_hash=_sha256_text(content or "")),
                currentness="current",
                topic_key=filename or logical_source_id,
                state_label="file",
                entities=[filename or logical_source_id],
                tags=["file", "inline"],
                extra_metadata={"path": filename or logical_source_id},
            )
        ],
        "content_hash": _sha256_text(content or ""),
        "summary": {
            "repo_name": logical_source_id,
            "repo_root": "",
            "object_counts": {"file": 1},
            "commit_count": 0,
            "file_count": 1,
            "hosting": {"status": "not_applicable", "object_counts": {}, "relation_count": 0},
        },
        "source_meta": {
            "codebase_stage": "stage1",
            "repo_root": "",
            "repo_name": logical_source_id,
            "head_commit": "",
            "head_branch": "",
        },
        "graph": {
            "repo": {
                "repo_id": f"repository:{logical_source_id}",
                "repo_name": logical_source_id,
                "repo_owner": "",
                "repo_display": logical_source_id,
                "repo_root": "",
                "provider": "inline",
                "remote_url": "",
                "head_commit": "",
                "head_branch": "",
                "upstream_branch": "",
            },
            "branches": [],
            "tags": [],
            "directories": [],
            "files": [{"path": filename or logical_source_id, "directory_path": ".", "blob_sha": ""}],
            "commits": [],
            "hosting": {"status": "not_applicable", "object_counts": {}, "relation_count": 0},
            "object_counts": {"file": 1},
        },
    }
