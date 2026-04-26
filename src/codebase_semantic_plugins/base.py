#!/usr/bin/env python3
# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ..codebase_semantic_bundle import stable_semantic_id


@dataclass(frozen=True)
class CodebaseSemanticPluginProfile:
    input_path: Path
    repo_root: Path
    repo_id: str
    revision: str
    files: list[Path]
    supported_extensions: list[str]
    skipped_files: list[str]


class CodebaseSemanticPlugin(Protocol):
    plugin_name: str
    supported_extensions: frozenset[str]

    def supports_file(self, path: Path) -> bool: ...

    def build_bundle(
        self,
        *,
        repo_root: Path,
        repo_id: str,
        revision: str,
        files: list[Path],
    ) -> dict[str, Any]: ...


def git_output(repo_root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except Exception:
        return None
    output = result.stdout.strip()
    return output or None


def repo_revision(repo_root: Path, files: list[Path]) -> str:
    git_head = git_output(repo_root, "rev-parse", "HEAD")
    if git_head:
        return git_head
    digest = hashlib.sha256()
    for file_path in sorted(files):
        digest.update(str(file_path).encode("utf-8"))
        digest.update(file_path.read_bytes())
    return digest.hexdigest()


def repo_remote(repo_root: Path) -> str | None:
    return git_output(repo_root, "config", "--get", "remote.origin.url")


def repo_root_for_path(path: Path) -> Path:
    if path.is_dir():
        root = git_output(path, "rev-parse", "--show-toplevel")
        return Path(root) if root else path
    parent = path.parent
    root = git_output(parent, "rev-parse", "--show-toplevel")
    return Path(root) if root else parent


def repo_id(repo_root: Path) -> str:
    remote = repo_remote(repo_root)
    if remote:
        return stable_semantic_id("repo", remote)
    return stable_semantic_id("repo", str(repo_root.resolve()))


def relative_file_path(repo_root: Path, file_path: Path) -> str:
    try:
        return str(file_path.resolve().relative_to(repo_root.resolve()))
    except Exception:
        return file_path.name


def span(start_line: int, end_line: int, start_col: int | None = None, end_col: int | None = None) -> dict[str, int | None]:
    return {
        "start_line": start_line,
        "end_line": end_line,
        "start_col": start_col,
        "end_col": end_col,
    }


def snippet_for_span(source: str, region: dict[str, int | None]) -> str:
    lines = source.splitlines()
    start_line = region.get("start_line") or 1
    end_line = region.get("end_line") or start_line
    start = max(1, int(start_line)) - 1
    end = min(len(lines), int(end_line))
    return "\n".join(lines[start:end]).strip()
