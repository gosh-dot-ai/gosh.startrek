#!/usr/bin/env python3
# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..codebase_semantic_bundle import merge_semantic_bundles
from .base import CodebaseSemanticPluginProfile, relative_file_path, repo_id, repo_revision, repo_root_for_path
from .registry import get_codebase_semantic_plugins

_CODEBASE_SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "node_modules",
    ".venv",
    "venv",
    "dist",
    "build",
    "target",
}

_CODEBASE_SOURCE_LIKE_EXTENSIONS = {
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".go",
    ".h",
    ".hpp",
    ".java",
    ".js",
    ".jsx",
    ".mjs",
    ".py",
    ".rs",
    ".sh",
    ".ts",
    ".tsx",
    ".vue",
    ".yaml",
    ".yml",
}


def _is_supported_repo_file(repo_root: Path, file_path: Path) -> bool:
    try:
        rel_parts = file_path.relative_to(repo_root).parts
    except Exception:
        rel_parts = file_path.parts
    return file_path.is_file() and not any(part in _CODEBASE_SKIP_DIRS for part in rel_parts)


def _plugin_supports_file(plugin: Any, file_path: Path) -> bool:
    supports_file = getattr(plugin, "supports_file", None)
    if callable(supports_file):
        try:
            return bool(supports_file(file_path))
        except Exception:
            return False
    return file_path.suffix.lower() in getattr(plugin, "supported_extensions", frozenset())


def discover_supported_codebase_files(path: str | Path) -> dict[str, Any]:
    input_path = Path(path).expanduser().resolve()
    repo_root = repo_root_for_path(input_path)
    plugins = get_codebase_semantic_plugins()
    supported_extensions = sorted({suffix for plugin in plugins for suffix in plugin.supported_extensions})
    if input_path.is_file():
        candidate_files = [input_path]
    else:
        candidate_files = [
            file_path
            for file_path in sorted(repo_root.rglob("*"))
            if _is_supported_repo_file(repo_root, file_path)
        ]
    supported_by_canonical_path: dict[str, Path] = {}
    unsupported_files: list[str] = []
    for file_path in candidate_files:
        canonical_path = relative_file_path(repo_root, file_path)
        if any(_plugin_supports_file(plugin, file_path) for plugin in plugins):
            supported_by_canonical_path.setdefault(canonical_path, file_path)
        elif file_path.suffix.lower() in _CODEBASE_SOURCE_LIKE_EXTENSIONS:
            unsupported_files.append(canonical_path)
    files = [supported_by_canonical_path[key] for key in sorted(supported_by_canonical_path)]
    unsupported_files = list(dict.fromkeys(unsupported_files))
    revision = repo_revision(repo_root, files)
    profile = CodebaseSemanticPluginProfile(
        input_path=input_path,
        repo_root=repo_root,
        repo_id=repo_id(repo_root),
        revision=revision,
        files=files,
        supported_extensions=supported_extensions,
        skipped_files=unsupported_files,
    )
    return {
        "input_path": profile.input_path,
        "repo_root": profile.repo_root,
        "repo_id": profile.repo_id,
        "revision": profile.revision,
        "files": profile.files,
        "supported_extensions": profile.supported_extensions,
        "skipped_files": profile.skipped_files,
    }


def _normalize_skipped_files(existing: Any, extra: list[str]) -> list[str]:
    values = []
    if isinstance(existing, list):
        for item in existing:
            if isinstance(item, str) and item.strip():
                values.append(item.strip())
    values.extend(extra)
    return values


def _unsupported_capability_gaps(skipped_files: list[str]) -> list[dict[str, Any]]:
    capability_files: dict[str, list[str]] = {}
    for rel_path in skipped_files:
        suffix = Path(rel_path).suffix.lower()
        if suffix in {".js", ".jsx", ".mjs", ".ts", ".tsx"}:
            capability = "js_ts_code_graph"
        elif suffix == ".vue":
            capability = "vue_code_graph"
        elif suffix == ".go":
            capability = "go_code_graph"
        elif suffix == ".java":
            capability = "java_code_graph"
        elif suffix in {".c", ".cc", ".cpp", ".cxx", ".h", ".hpp"}:
            capability = "c_cpp_code_graph"
        elif suffix == ".sh":
            capability = "shell_code_graph"
        else:
            continue
        capability_files.setdefault(capability, []).append(rel_path)
    return [
        {
            "capability": capability,
            "status": "explicitly_absent",
            "files": sorted(dict.fromkeys(paths)),
            "reason": "no_enabled_semantic_plugin_for_source_family",
        }
        for capability, paths in sorted(capability_files.items())
    ]


def build_codebase_semantic_bundle(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    profile = discover_supported_codebase_files(path)
    repo_root = Path(profile["repo_root"])
    repo_id_value = str(profile["repo_id"])
    revision = str(profile["revision"])
    files: list[Path] = list(profile["files"])

    bundles: list[dict[str, Any]] = []
    skipped_files: list[str] = list(profile.get("skipped_files") or [])
    plugin_failure_notes: list[str] = [
        f"unsupported semantic plugin file: {path}"
        for path in skipped_files
    ]
    for plugin in get_codebase_semantic_plugins():
        plugin_files = sorted({file_path for file_path in files if _plugin_supports_file(plugin, file_path)})
        if not plugin_files:
            continue
        try:
            bundles.append(
                plugin.build_bundle(
                    repo_root=repo_root,
                    repo_id=repo_id_value,
                    revision=revision,
                    files=plugin_files,
                )
            )
        except Exception as exc:
            failed_paths = [relative_file_path(repo_root, file_path) for file_path in plugin_files]
            skipped_files.extend(failed_paths)
            plugin_failure_notes.append(
                f"plugin {plugin.plugin_name} failed for {', '.join(failed_paths[:8])}: {exc}"
            )

    if not bundles:
        if plugin_failure_notes:
            raise ValueError("; ".join(plugin_failure_notes))
        raise ValueError("codebase Repository context supports only Python and Rust sources")

    merged = merge_semantic_bundles(bundles)
    gap_report = dict(merged.get("gap_report") or {})
    notes = list(gap_report.get("notes") or [])
    notes.extend(plugin_failure_notes)
    gap_report["notes"] = notes
    gap_report["skipped_files"] = list(
        dict.fromkeys(_normalize_skipped_files(gap_report.get("skipped_files"), skipped_files))
    )
    unsupported_capability_gaps = _unsupported_capability_gaps(gap_report["skipped_files"])
    missing_capabilities = list(gap_report.get("missing_capabilities") or [])
    missing_capabilities.extend(unsupported_capability_gaps)
    gap_report["missing_capabilities"] = missing_capabilities
    for gap in unsupported_capability_gaps:
        display_files = ", ".join(gap.get("files") or [])
        notes.append(f"unsupported capability {gap['capability']} explicitly_absent for files: {display_files}")
    gap_report["notes"] = list(dict.fromkeys(notes))
    merged["gap_report"] = gap_report
    profile["skipped_files"] = list(dict.fromkeys(_normalize_skipped_files(profile.get("skipped_files"), skipped_files)))
    return merged, profile
