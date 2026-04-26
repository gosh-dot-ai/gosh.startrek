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
import sys
from functools import cache
from pathlib import Path


ARCHIVE_REPO_URL = "https://github.com/Futurizt/multibench.archive.git"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _candidate_roots() -> list[Path]:
    repo_root = _repo_root()
    env_root = os.environ.get("GOSH_MULTIBENCH_ARCHIVE_ROOT")
    candidates = [
        Path(env_root) if env_root else None,
        repo_root.parent / "multibench.archive",
        repo_root.parent.parent / "multibench.archive",
        Path("/tmp/multibench.archive"),
        Path("/media/futurizt/BIG/gosh/gosh.ai/multibench.archive"),
        Path("/media/futurizt/Store/Git/multibench.archive"),
    ]
    return [path for path in candidates if path is not None]


def _looks_like_archive_repo(path: Path) -> bool:
    return (path / "multibench").is_dir() and (path / "validation").is_dir()


def _clone_archive_repo(target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", ARCHIVE_REPO_URL, str(target)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return target


def _discover_archive_repo_root() -> Path | None:
    for candidate in _candidate_roots():
        if _looks_like_archive_repo(candidate):
            os.environ.setdefault("GOSH_MULTIBENCH_ARCHIVE_ROOT", str(candidate))
            return candidate
    return None


@cache
def archive_repo_root(*, auto_clone: bool = True) -> Path:
    discovered = _discover_archive_repo_root()
    if discovered is not None:
        return discovered
    default_target = Path(os.environ.get("GOSH_MULTIBENCH_ARCHIVE_CACHE", "/tmp/multibench.archive"))
    if not _looks_like_archive_repo(default_target):
        if not auto_clone:
            raise FileNotFoundError(
                "Archive repo is not available locally and auto-clone is disabled. "
                "Set GOSH_MULTIBENCH_ARCHIVE_ROOT or enable cloning for archive-dependent tests."
            )
        _clone_archive_repo(default_target)
    os.environ.setdefault("GOSH_MULTIBENCH_ARCHIVE_ROOT", str(default_target))
    return default_target


def ensure_archive_repo_on_sys_path(*, auto_clone: bool = True) -> Path | None:
    try:
        root = archive_repo_root(auto_clone=auto_clone)
    except FileNotFoundError:
        return None
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def archive_repo_path(relative_path: str | os.PathLike[str]) -> Path:
    path = archive_repo_root() / Path(relative_path)
    if not path.exists():
        raise FileNotFoundError(f"Archive repo path does not exist: {path}")
    return path
