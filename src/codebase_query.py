# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

from __future__ import annotations

import json
import re
import subprocess
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from .codebase_ontology import (
    build_codebase_anchor,
    build_codebase_object_unit,
    build_codebase_relation_fact,
    codebase_metadata_from_fact,
    codebase_object_id_from_fact,
    codebase_relation_endpoints,
    hydrate_codebase_object_row,
)
from .container_graph import normalize_container_graph
from .episode_features import extract_query_features
from .episode_packet import build_context_from_retrieved_facts, fact_episode_ids
from .retrieval import source_local_fact_sweep

_PATH_RE = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_./-]+")
_SHA_RE = re.compile(r"\b[0-9a-f]{7,40}\b", re.I)
_BRANCH_RE = re.compile(r"\bbranch(?:es)?\s+([A-Za-z0-9_./-]+)\b", re.I)
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_./-]+")
_FILE_TOKEN_RE = re.compile(r"\b[A-Za-z0-9_.-]+\.(?:py|pyi|js|jsx|ts|tsx|vue|go|rs|java|c|cc|cpp|h|hpp|json|toml|ya?ml|xml|ini|cfg|conf|gradle|mod|sum|lock)\b")
_SOURCE_WINDOW_SEPARATOR = (
    "\n[source-window gap: non-contiguous source omitted; metadata only]\n"
)
_PATCH_CONTEXT_WHOLE_FILE_MAX_LINES = 5000
_PATCH_CONTEXT_WHOLE_FILE_MAX_CHARS = 200000
_PATCH_CONTEXT_TOTAL_BUDGET = 300000
_PATCH_CONTEXT_POLICY = {
    "mode": "whole_file_first",
    "whole_file_max_lines": _PATCH_CONTEXT_WHOLE_FILE_MAX_LINES,
    "whole_file_max_chars": _PATCH_CONTEXT_WHOLE_FILE_MAX_CHARS,
    "total_patch_context_budget": _PATCH_CONTEXT_TOTAL_BUDGET,
    "required_paths_fail_closed": True,
}
_STANDALONE_REPO_FILE_NAMES = {
    "Cargo.toml",
    "Cargo.lock",
    "Dockerfile",
    "Gemfile",
    "Gemfile.lock",
    "Makefile",
    "Pipfile",
    "Pipfile.lock",
    "go.mod",
    "go.sum",
    "package.json",
    "package-lock.json",
    "pnpm-lock.yaml",
    "poetry.lock",
    "pyproject.toml",
    "requirements.txt",
    "setup.cfg",
    "setup.py",
    "tox.ini",
    "tsconfig.json",
    "yarn.lock",
}
_PATH_STRIP_CHARS = "\"'`.,:;()[]{}<>"
_PATCH_VERB_RE = re.compile(r"\b(?:patch|unified\s+diff|return\s+(?:a\s+)?(?:unified\s+)?diff)\b", re.I)
_QUERY_STOPWORDS = {
    "what",
    "which",
    "where",
    "when",
    "does",
    "did",
    "that",
    "with",
    "from",
    "into",
    "this",
    "these",
    "those",
    "there",
    "their",
    "have",
    "has",
    "will",
    "would",
    "could",
    "should",
    "only",
    "answer",
    "base",
    "bench",
    "commit",
    "context",
    "description",
    "diff",
    "expected",
    "fixing",
    "hidden",
    "issue",
    "minimal",
    "patch",
    "problem",
    "produce",
    "repository",
    "requirements",
    "return",
    "short",
    "statement",
    "swe",
    "tests",
    "e.g",
}


def _repo_task_contract(query_metadata: dict[str, Any] | None) -> dict[str, Any]:
    metadata = dict(query_metadata or {})
    repo_scope_raw = metadata.get("repo_scope")
    repo_scope: dict[str, Any] = repo_scope_raw if isinstance(repo_scope_raw, dict) else {}
    task_mode = str(metadata.get("task_mode") or "").strip()
    output_artifact = str(metadata.get("output_artifact") or "").strip()
    work_item_kind = str(metadata.get("work_item_kind") or "").strip()
    required_render_mode = str(metadata.get("required_render_mode") or "").strip()
    active = bool(
        work_item_kind.startswith("repo:")
        or task_mode
        or output_artifact
        or required_render_mode
        or repo_scope
    )
    return {
        "active": active,
        "work_item_kind": work_item_kind or "repo:work_item",
        "task_mode": task_mode or ("patch_generation" if output_artifact == "unified_diff" else ""),
        "answer_contract": str(metadata.get("answer_contract") or "").strip(),
        "output_artifact": output_artifact,
        "required_render_mode": required_render_mode,
        "repo_scope": {
            "repo_id": str(repo_scope.get("repo_id") or repo_scope.get("repo") or "").strip(),
            "commit_id": str(repo_scope.get("commit_id") or repo_scope.get("commit") or "").strip(),
        },
        "source_id": str(metadata.get("source_id") or "").strip(),
        "work_item_id": str(metadata.get("work_item_id") or "").strip(),
        "activation_policy": "codebase_family_plus_repo_work_item_contract",
    }


def _is_patch_generation_contract(repo_task_contract: dict[str, Any], query: str) -> bool:
    task_mode = str(repo_task_contract.get("task_mode") or "").strip()
    output_artifact = str(repo_task_contract.get("output_artifact") or "").strip()
    render_mode = str(repo_task_contract.get("required_render_mode") or "").strip()
    if task_mode == "patch_generation" or output_artifact == "unified_diff":
        return True
    if render_mode in {"patch_safe_source_context", "patch_safe_source_windows"}:
        return True
    return bool(_PATCH_VERB_RE.search(str(query or "")))


def _looks_like_repo_file_path(value: str) -> bool:
    normalized = _normalize_path(value)
    parts = [part for part in normalized.split("/") if part]
    if not parts:
        return False
    name = parts[-1]
    if name.lower() in {entry.lower() for entry in _STANDALONE_REPO_FILE_NAMES}:
        return True
    return bool(_FILE_TOKEN_RE.fullmatch(name))


def _extract_path_tokens(text: str) -> list[str]:
    raw = str(text or "")
    candidates = [*_PATH_RE.findall(raw), *_FILE_TOKEN_RE.findall(raw)]
    for name in _STANDALONE_REPO_FILE_NAMES:
        if re.search(rf"(?<![A-Za-z0-9_.-]){re.escape(name)}(?![A-Za-z0-9_.-])", raw):
            candidates.append(name)
    out: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        value = _normalize_path(candidate).strip(_PATH_STRIP_CHARS)
        if value.startswith(("a/", "b/")) and value.count("/") >= 2:
            value = value[2:]
        if not value or "://" in value or value.startswith(("../", "./")):
            continue
        if value in {".", ".."}:
            continue
        if not _looks_like_repo_file_path(value):
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def _path_constraint_role(path: str, source: str) -> str:
    lowered = _normalize_path(path).lower()
    parts = [part for part in lowered.split("/") if part]
    name = parts[-1] if parts else lowered
    if (
        source == "selected_test_files"
        or name.startswith("test_")
        or name.endswith(("_test.py", "_test.go", "_test.rs"))
        or re.search(r"(?:^|[._-])(test|spec)\.(js|jsx|mjs|ts|tsx)$", name)
        or any(part in {"test", "tests", "spec", "specs"} for part in parts)
    ):
        return "test_source"
    if "/.github/workflows/" in f"/{lowered}" or any(part == ".github" for part in parts):
        return "workflow_source"
    standalone_names = {entry.lower() for entry in _STANDALONE_REPO_FILE_NAMES}
    if name in standalone_names or lowered.endswith((".yml", ".yaml", ".json", ".toml", ".ini", ".cfg", ".conf", ".xml", ".gradle")):
        return "config_source"
    if lowered.endswith((".md", ".rst", ".txt", ".adoc", ".asciidoc")):
        return "supporting_source"
    if source == "setup_command":
        return "supporting_source"
    return "editable_source"


def _constraint_required(source: str, line: str) -> bool:
    if source in {"requirement", "interface", "selected_test_files", "setup_command"}:
        return True
    lowered = str(line or "").lower()
    if "example" in lowered or "for example" in lowered or "e.g." in lowered:
        return False
    return source in {"query", "metadata"}


def _add_path_constraint(
    constraints: dict[str, dict[str, Any]],
    *,
    path: str,
    source: str,
    reason: str,
    required: bool,
) -> None:
    norm = _normalize_path(path)
    key = norm.lower()
    if not norm or not key:
        return
    role = _path_constraint_role(norm, source)
    current = constraints.get(key)
    if current is None:
        constraints[key] = {
            "path": norm,
            "source": source,
            "sources": [source],
            "required": bool(required),
            "reason": reason,
            "reasons": [reason],
            "role": role,
        }
        return
    if source not in current["sources"]:
        current["sources"].append(source)
    if reason not in current["reasons"]:
        current["reasons"].append(reason)
    current["required"] = bool(current.get("required") or required)
    if current.get("role") != "editable_source" and role == "editable_source":
        current["role"] = role
    current["source"] = current["sources"][0]
    current["reason"] = current["reasons"][0]


def _query_section_source(line: str, current: str) -> str:
    lowered = line.strip().lower()
    if lowered.startswith("requirements"):
        return "requirement"
    if lowered.startswith("interface"):
        return "interface"
    if lowered.startswith("selected test files") or lowered.startswith("selected tests"):
        return "selected_test_files"
    if lowered.startswith("environment setup command") or lowered.startswith("setup command"):
        return "setup_command"
    if lowered.startswith("problem statement") or lowered.startswith("description"):
        return "query"
    return current


def _path_constraint_metadata_key_is_identity(key_path: tuple[str, ...]) -> bool:
    key_blob = ".".join(key_path).lower()
    identity_keys = {
        "repo",
        "repo_id",
        "repository",
        "source_id",
        "work_item_id",
        "instance_id",
        "commit",
        "commit_id",
        "base_commit",
    }
    return any(part in identity_keys for part in key_path) or key_blob.endswith("repo_scope.repo_id")


def _query_line_is_repo_identity(line: str) -> bool:
    lowered = line.strip().lower()
    return lowered.startswith(("repository:", "repo:", "base commit:", "language:"))


def _metadata_strings(value: Any, *, key_path: tuple[str, ...] = ()) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    if _path_constraint_metadata_key_is_identity(key_path):
        return rows
    if isinstance(value, dict):
        for key, child in value.items():
            rows.extend(_metadata_strings(child, key_path=(*key_path, str(key))))
    elif isinstance(value, (list, tuple, set)):
        for child in value:
            rows.extend(_metadata_strings(child, key_path=key_path))
    elif isinstance(value, str):
        key_blob = ".".join(key_path).lower()
        if "selected" in key_blob and "test" in key_blob:
            source = "selected_test_files"
        elif "setup" in key_blob or "command" in key_blob:
            source = "setup_command"
        elif "interface" in key_blob:
            source = "interface"
        elif "requirement" in key_blob:
            source = "requirement"
        else:
            source = "metadata"
        rows.append((source, value))
    return rows


def extract_repo_work_item_path_constraints(
    query: str,
    query_metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Extract generic repo-relative path constraints from a repository work item."""

    constraints: dict[str, dict[str, Any]] = {}
    current_source = "query"
    for line in str(query or "").splitlines():
        current_source = _query_section_source(line, current_source)
        if _query_line_is_repo_identity(line):
            continue
        for path in _extract_path_tokens(line):
            _add_path_constraint(
                constraints,
                path=path,
                source=current_source,
                reason=f"explicit_path_in_{current_source}",
                required=_constraint_required(current_source, line),
            )
    for source, text_value in _metadata_strings(query_metadata or {}):
        for path in _extract_path_tokens(text_value):
            _add_path_constraint(
                constraints,
                path=path,
                source=source,
                reason=f"explicit_path_in_{source}",
                required=_constraint_required(source, text_value),
            )
    return sorted(
        constraints.values(),
        key=lambda row: (
            0 if row.get("role") in {"editable_source", "config_source"} else 1,
            str(row.get("path") or ""),
        ),
    )


def _normalize_path(value: str) -> str:
    return str(value or "").strip().replace("\\", "/")


def _path_matches(query_path: str, candidate_path: str) -> bool:
    lhs = _normalize_path(query_path).lower()
    rhs = _normalize_path(candidate_path).lower()
    if not lhs or not rhs:
        return False
    return lhs == rhs or rhs.endswith(f"/{lhs}") or lhs.endswith(f"/{rhs}")


def _sha_matches(query_sha: str, candidate_sha: str) -> bool:
    lhs = str(query_sha or "").strip().lower()
    rhs = str(candidate_sha or "").strip().lower()
    if not lhs or not rhs:
        return False
    return rhs.startswith(lhs) or lhs.startswith(rhs)


def _query_exact_terms(query: str) -> set[str]:
    terms = {
        term.strip()
        for term in _PATH_RE.findall(str(query or ""))
    }
    terms.update(match.group(0).lower() for match in _SHA_RE.finditer(str(query or "")))
    return {term for term in terms if term}


def _query_priority_terms(query: str) -> set[str]:
    terms = set(_query_exact_terms(query))
    for token in _WORD_RE.findall(str(query or "").lower()):
        if len(token) < 4 or token in _QUERY_STOPWORDS:
            continue
        terms.add(token)
    return terms


def _query_graph_signals(query: str) -> dict[str, list[str]]:
    lowered = str(query or "").lower()
    branch_names: list[str] = []
    for match in _BRANCH_RE.finditer(lowered):
        branch_name = str(match.group(1) or "").strip()
        if branch_name and branch_name not in branch_names:
            branch_names.append(branch_name)
    return {
        "paths": sorted({_normalize_path(term) for term in _PATH_RE.findall(str(query or "")) if term.strip()}),
        "shas": sorted({match.group(0).lower() for match in _SHA_RE.finditer(str(query or ""))}),
        "branches": branch_names,
    }


def _query_needs_sidecar_scan(query: str) -> bool:
    lowered = str(query or "").lower()
    signals = _query_graph_signals(query)
    if signals["paths"] or signals["shas"]:
        return True
    return any(token in lowered for token in (" diff", "diff ", " hunk", "hunk ", " patch", "patch "))


def _relation_priority(fact: dict[str, Any], query_terms: set[str]) -> tuple[int, int, str]:
    codebase = codebase_metadata_from_fact(fact)
    relation_type = str(codebase.get("relation_type") or "")
    fact_text = str(fact.get("fact") or "")
    hit = 0
    lowered = fact_text.lower()
    for term in query_terms:
        if term.lower() in lowered:
            hit += 1
    return (hit, len(fact_text), relation_type)


def _candidate_exact_support_facts(
    query: str,
    candidate_facts: list[dict[str, Any]],
    *,
    max_facts: int = 12,
) -> list[dict[str, Any]]:
    query_terms = _query_priority_terms(query)
    if not query_terms:
        return []

    scored: list[tuple[tuple[int, int, int], dict[str, Any]]] = []
    for fact in candidate_facts:
        fact_text = str(fact.get("fact") or "").lower()
        hit_count = sum(1 for term in query_terms if term.lower() in fact_text)
        if hit_count <= 0:
            continue
        kind_bonus = 2 if fact.get("kind") == "codebase_relation" else 1
        scored.append(((hit_count, kind_bonus, len(fact_text)), fact))
    scored.sort(key=lambda item: item[0], reverse=True)

    selected: list[dict[str, Any]] = []
    seen_fact_ids: set[str] = set()
    for _score, fact in scored:
        fact_id = str(fact.get("id") or "").strip()
        if fact_id and fact_id in seen_fact_ids:
            continue
        if fact_id:
            seen_fact_ids.add(fact_id)
        selected.append(fact)
        if len(selected) >= max_facts:
            break
    return selected


def _expand_seed_neighbors(
    *,
    query: str,
    seed_facts: list[dict[str, Any]],
    candidate_facts: list[dict[str, Any]],
    max_extra_facts: int = 12,
    max_relations_per_object: int = 4,
) -> list[dict[str, Any]]:
    object_facts, relations_by_object, _fact_lookup = _build_graph_indices(candidate_facts)
    query_terms = _query_priority_terms(query)
    extras: list[dict[str, Any]] = []
    seen_ids = {
        str(fact.get("id") or "").strip()
        for fact in seed_facts
        if str(fact.get("id") or "").strip()
    }
    queue: list[str] = []
    for fact in seed_facts:
        object_id = codebase_object_id_from_fact(fact)
        if object_id:
            queue.append(object_id)
        from_id, to_id = codebase_relation_endpoints(fact)
        if from_id:
            queue.append(from_id)
        if to_id:
            queue.append(to_id)

    visited_objects: set[str] = set()
    while queue and len(extras) < max_extra_facts:
        object_id = queue.pop(0)
        if not object_id or object_id in visited_objects:
            continue
        visited_objects.add(object_id)

        object_fact = object_facts.get(object_id)
        if object_fact is not None:
            fact_id = str(object_fact.get("id") or "").strip()
            if fact_id and fact_id not in seen_ids:
                seen_ids.add(fact_id)
                extras.append(object_fact)
                if len(extras) >= max_extra_facts:
                    break

        adjacent = sorted(
            relations_by_object.get(object_id, []),
            key=lambda fact: _relation_priority(fact, query_terms),
            reverse=True,
        )
        for relation in adjacent[:max_relations_per_object]:
            fact_id = str(relation.get("id") or "").strip()
            if fact_id and fact_id in seen_ids:
                continue
            if fact_id:
                seen_ids.add(fact_id)
            extras.append(relation)
            if len(extras) >= max_extra_facts:
                break
            from_id, to_id = codebase_relation_endpoints(relation)
            for neighbor in (from_id, to_id):
                if neighbor and neighbor not in visited_objects and neighbor not in queue:
                    queue.append(neighbor)
    return extras


def _build_graph_indices(candidate_facts: list[dict[str, Any]]) -> tuple[dict[str, dict], dict[str, list[dict]], dict[str, dict]]:
    object_facts: dict[str, dict] = {}
    relations_by_object: dict[str, list[dict]] = defaultdict(list)
    fact_lookup: dict[str, dict] = {}
    for fact in candidate_facts:
        fact_id = str(fact.get("id") or "").strip()
        if fact_id:
            fact_lookup[fact_id] = fact
        object_id = codebase_object_id_from_fact(fact)
        if fact.get("kind") == "codebase_object" and object_id and object_id not in object_facts:
            object_facts[object_id] = fact
        from_id, to_id = codebase_relation_endpoints(fact)
        if fact.get("kind") == "codebase_relation":
            if from_id:
                relations_by_object[from_id].append(fact)
            if to_id and to_id != from_id:
                relations_by_object[to_id].append(fact)
    return object_facts, relations_by_object, fact_lookup


def _transient_fact_rows(source_id: str, unit: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fact in unit.get("facts") or []:
        row = deepcopy(fact)
        raw_id = str(row.get("id") or "").strip()
        row["id"] = f"graph:{source_id}:{raw_id}"
        row["source_id"] = source_id
        rows.append(row)
    return rows


def _load_codebase_graph(server, source_id: str) -> dict[str, Any] | None:
    cache = getattr(server, "_codebase_graph_cache", None)
    if not isinstance(cache, dict):
        cache = {}
        server._codebase_graph_cache = cache
    if source_id in cache:
        return cache[source_id]
    record = (getattr(server, "_source_records", {}) or {}).get(source_id) or {}
    source_meta = dict(record.get("source_meta") or {})
    graph_ref = str(source_meta.get("codebase_graph_ref") or "").strip()
    if not graph_ref:
        cache[source_id] = None
        return None
    graph_path = Path(server.data_dir) / graph_ref
    try:
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
    except Exception:
        graph = None
    cache[source_id] = graph
    return graph


def _iter_codebase_relation_rows(server, source_id: str):
    record = (getattr(server, "_source_records", {}) or {}).get(source_id) or {}
    source_meta = dict(record.get("source_meta") or {})
    relation_ref = str(source_meta.get("codebase_relation_ref") or "").strip()
    if not relation_ref:
        return
    relation_path = Path(server.data_dir) / relation_ref
    try:
        with relation_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if isinstance(row, dict):
                    hydrated = hydrate_codebase_object_row(row)
                    if hydrated:
                        yield hydrated
    except Exception:
        return


def _iter_codebase_object_rows(server, source_id: str):
    record = (getattr(server, "_source_records", {}) or {}).get(source_id) or {}
    source_meta = dict(record.get("source_meta") or {})
    object_ref = str(source_meta.get("codebase_object_ref") or "").strip()
    if not object_ref:
        return
    object_path = Path(server.data_dir) / object_ref
    try:
        with object_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if isinstance(row, dict):
                    yield row
    except Exception:
        return


def _build_object_support_facts(
    *,
    source_id: str,
    server,
    query: str,
    max_facts: int = 48,
) -> list[dict[str, Any]]:
    signals = _query_graph_signals(query)
    query_terms = _query_priority_terms(query)
    if not any(signals.values()) and not query_terms:
        return []

    scored_rows: list[tuple[tuple[int, int, int], dict[str, Any]]] = []
    for row in _iter_codebase_object_rows(server, source_id) or ():
        metadata = codebase_metadata_from_fact(row)
        fact_text = str(row.get("fact") or "").strip()
        anchor = dict(metadata.get("anchor") or {})
        anchor_path = _normalize_path(str(anchor.get("path") or ""))
        anchor_commit = str(anchor.get("commit_sha") or "").strip().lower()
        score = 0
        if signals["paths"] and anchor_path:
            if any(_path_matches(path, anchor_path) for path in signals["paths"]):
                score += 4
        if signals["shas"] and anchor_commit:
            if any(_sha_matches(query_sha, anchor_commit) for query_sha in signals["shas"]):
                score += 4
        lowered = fact_text.lower()
        term_hits = sum(1 for term in query_terms if term.lower() in lowered)
        if term_hits <= 0 and score <= 0:
            continue
        scored_rows.append(((score, term_hits, len(fact_text)), row))

    scored_rows.sort(key=lambda item: item[0], reverse=True)
    facts: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for _score, row in scored_rows[:max_facts]:
        fact = deepcopy(row)
        fact["id"] = f"graph:{source_id}:{fact.get('id') or ''}"
        fact["source_id"] = source_id
        fact_id = str(fact.get("id") or "").strip()
        if not fact_id or fact_id in seen_ids:
            continue
        seen_ids.add(fact_id)
        facts.append(fact)
    return facts


def _build_relation_support_facts(
    *,
    source_id: str,
    server,
    query: str,
    max_facts: int = 48,
) -> list[dict[str, Any]]:
    signals = _query_graph_signals(query)
    query_terms = _query_priority_terms(query)
    if not any(signals.values()) and not query_terms:
        return []

    scored_rows: list[tuple[tuple[int, int, int], dict[str, Any]]] = []
    for row in _iter_codebase_relation_rows(server, source_id) or ():
        fact_text = str(row.get("fact") or "").strip()
        anchor = dict(row.get("anchor") or {})
        anchor_path = _normalize_path(str(anchor.get("path") or ""))
        anchor_branch = str(anchor.get("branch_name") or "").strip().lower()
        anchor_commit = str(anchor.get("commit_sha") or "").strip().lower()
        score = 0
        if signals["paths"] and anchor_path:
            if any(_path_matches(path, anchor_path) for path in signals["paths"]):
                score += 4
        if signals["shas"]:
            if any(_sha_matches(query_sha, anchor_commit) for query_sha in signals["shas"]):
                score += 4
            if any(
                _sha_matches(query_sha, token)
                for query_sha in signals["shas"]
                for token in (
                    str(row.get("from_id") or "").strip().lower(),
                    str(row.get("to_id") or "").strip().lower(),
                )
            ):
                score += 2
        if signals["branches"] and anchor_branch:
            if any(branch == anchor_branch for branch in signals["branches"]):
                score += 4
        lowered = fact_text.lower()
        term_hits = sum(1 for term in query_terms if term.lower() in lowered)
        if term_hits <= 0 and score <= 0:
            continue
        scored_rows.append(((score, term_hits, len(fact_text)), row))

    scored_rows.sort(key=lambda item: item[0], reverse=True)
    facts: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for _score, row in scored_rows[:max_facts]:
        fact = build_codebase_relation_fact(
            relation_type=str(row.get("relation_type") or "").strip(),
            from_id=str(row.get("from_id") or "").strip(),
            to_id=str(row.get("to_id") or "").strip(),
            fact_text=str(row.get("fact") or "").strip(),
            anchor=dict(row.get("anchor") or {}),
            entities=list(row.get("entities") or []),
        )
        fact["id"] = f"graph:{source_id}:{fact['id']}"
        fact["source_id"] = source_id
        fact_id = str(fact.get("id") or "").strip()
        if fact_id in seen_ids:
            continue
        seen_ids.add(fact_id)
        facts.append(fact)
    return facts


def _build_graph_commit_support_facts(
    *,
    source_id: str,
    graph: dict[str, Any],
    query: str,
    max_facts: int = 36,
) -> list[dict[str, Any]]:
    signals = _query_graph_signals(query)
    if not any(signals.values()):
        return []

    repo = dict(graph.get("repo") or {})
    repo_name = str(repo.get("repo_display") or repo.get("repo_name") or source_id)
    repo_root = str(repo.get("repo_root") or "").strip()
    branch_rows = {
        str(row.get("name") or "").strip(): dict(row)
        for row in (graph.get("branches") or [])
        if str(row.get("name") or "").strip()
    }
    file_rows = {
        _normalize_path(str(row.get("path") or "")): dict(row)
        for row in (graph.get("files") or [])
        if str(row.get("path") or "").strip()
    }

    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    def _append_unit(unit: dict[str, Any]) -> None:
        if len(rows) >= max_facts:
            return
        for fact in _transient_fact_rows(source_id, unit):
            fact_id = str(fact.get("id") or "").strip()
            if fact_id and fact_id in seen_ids:
                continue
            if fact_id:
                seen_ids.add(fact_id)
            rows.append(fact)
            if len(rows) >= max_facts:
                break

    def _resolve_containing_branches(commit_sha: str) -> list[str]:
        if not repo_root or not commit_sha:
            return []
        try:
            proc = subprocess.run(
                ["git", "-C", repo_root, "branch", "--all", "--contains", commit_sha, "--format=%(refname:short)"],
                check=False,
                capture_output=True,
                text=True,
            )
        except Exception:
            return []
        if proc.returncode != 0:
            return []
        names: list[str] = []
        for line in proc.stdout.splitlines():
            name = str(line or "").strip()
            if not name:
                continue
            if name.startswith("remotes/"):
                name = name[len("remotes/") :]
            if name not in names:
                names.append(name)
        return names

    for commit in (graph.get("commits") or []):
        commit_sha = str(commit.get("sha") or "").strip()
        if not commit_sha:
            continue
        diffs = list(commit.get("diffs") or [])
        modified_paths_sample = [
            _normalize_path(str(path or ""))
            for path in (commit.get("modified_paths_sample") or [])
            if _normalize_path(str(path or ""))
        ]
        matched_diffs = [
            diff
            for diff in diffs
            if any(_path_matches(path, str(diff.get("path") or "")) for path in signals["paths"])
        ]
        if not matched_diffs and modified_paths_sample:
            matched_diffs = [
                {"path": path, "status": "modified", "hydration_state": "deferred", "hunks": []}
                for path in modified_paths_sample
                if any(_path_matches(query_path, path) for query_path in signals["paths"])
            ]
        branch_hits = [
            branch_name
            for branch_name in (commit.get("branches") or [])
            if branch_name in signals["branches"]
        ]
        sha_hit = any(_sha_matches(query_sha, commit_sha) for query_sha in signals["shas"])
        resolved_branches = list(commit.get("branches") or [])
        if sha_hit and repo_root:
            lazy_branches = _resolve_containing_branches(commit_sha)
            if lazy_branches:
                resolved_branches = lazy_branches
                branch_hits = [
                    branch_name
                    for branch_name in resolved_branches
                    if branch_name in signals["branches"]
                ]
        if not (matched_diffs or branch_hits or sha_hit):
            continue

        commit_relations: list[dict[str, Any]] = []
        for branch_name in list(dict.fromkeys(resolved_branches))[:12]:
            commit_relations.append(
                build_codebase_relation_fact(
                    relation_type="branch_contains_commit",
                    from_id=f"branch:{branch_name}",
                    to_id=f"commit:{commit_sha}",
                    fact_text=f"{branch_name} contains {commit_sha}",
                    anchor=build_codebase_anchor("branch", branch_name=branch_name, commit_sha=commit_sha),
                    entities=[branch_name, commit_sha],
                )
            )
        for diff in matched_diffs[:8]:
            diff_path = _normalize_path(str(diff.get("path") or ""))
            if not diff_path:
                continue
            commit_relations.append(
                build_codebase_relation_fact(
                    relation_type="commit_modifies_file",
                    from_id=f"commit:{commit_sha}",
                    to_id=f"file:{diff_path}",
                    fact_text=f"{commit_sha} modifies {diff_path}",
                    anchor=build_codebase_anchor("file", path=diff_path, commit_sha=commit_sha),
                    entities=[commit_sha, diff_path],
                )
            )
            commit_relations.append(
                build_codebase_relation_fact(
                    relation_type="commit_contains_diff",
                    from_id=f"commit:{commit_sha}",
                    to_id=f"diff:commit:{commit_sha}:{diff_path}",
                    fact_text=f"{commit_sha} diff {diff_path}",
                    anchor=build_codebase_anchor("diff", commit_sha=commit_sha, path=diff_path),
                    entities=[commit_sha, diff_path],
                )
            )

        commit_lines = [
            f"Commit SHA: {commit_sha}",
            f"Repository: {repo_name}",
            f"Parents: {', '.join(commit.get('parents') or []) or '(root)'}",
            f"Author: {commit.get('author_name') or '(unknown)'} <{commit.get('author_email') or '(unknown)'}>",
            f"Committer: {commit.get('committer_name') or '(unknown)'} <{commit.get('committer_email') or '(unknown)'}>",
            f"Authored at: {commit.get('authored_at') or '(unknown)'}",
            f"Committed at: {commit.get('committed_at') or '(unknown)'}",
            f"Subject: {commit.get('subject') or '(none)'}",
            f"Containing branches: {', '.join(resolved_branches) or '(unknown)'}",
            f"Changed file count: {commit.get('changed_file_count') or len(diffs)}",
        ]
        if matched_diffs:
            commit_lines.append(
                "Matched diff paths: " + ", ".join(_normalize_path(str(diff.get("path") or "")) for diff in matched_diffs[:12])
            )
        _append_unit(
            build_codebase_object_unit(
                object_type="merge" if len(commit.get("parents") or []) > 1 else "commit",
                object_id=f"commit:{commit_sha}",
                raw_text="\n".join(commit_lines),
                fact_text=(
                    f"Commit {commit_sha} in repository {repo_name} was authored by {commit.get('author_name') or '(unknown)'} "
                    f"and committed by {commit.get('committer_name') or '(unknown)'}. Subject: {commit.get('subject') or '(none)'}."
                ),
                anchor=build_codebase_anchor(
                    "commit",
                    commit_sha=commit_sha,
                    parent_shas=commit.get("parents") or [],
                    authored_at=commit.get("authored_at"),
                    committed_at=commit.get("committed_at"),
                ),
                source_date=str(commit.get("committed_at") or ""),
                currentness="current" if resolved_branches else "historical",
                topic_key=commit_sha[:12],
                state_label="commit",
                entities=[commit_sha, *resolved_branches[:6]],
                tags=["commit", "git", "graph_support"],
                extra_metadata={
                    "commit_sha": commit_sha,
                    "parent_shas": list(commit.get("parents") or []),
                    "author": commit.get("author_name"),
                    "committer": commit.get("committer_name"),
                    "subject": commit.get("subject"),
                    "branches": list(resolved_branches),
                    "changed_file_count": commit.get("changed_file_count") or len(diffs) or len(modified_paths_sample),
                    "modified_files_sample": modified_paths_sample[:16],
                    "modified_files_truncated": bool(commit.get("modified_paths_truncated")),
                    "graph_support": True,
                },
                relation_facts=commit_relations,
            )
        )

        for branch_name in list(dict.fromkeys(resolved_branches))[:8]:
            branch_row = branch_rows.get(branch_name, {})
            _append_unit(
                build_codebase_object_unit(
                    object_type="branch",
                    object_id=f"branch:{branch_name}",
                    raw_text="\n".join(
                        [
                            f"Branch name: {branch_name}",
                            f"Commit SHA: {branch_row.get('commit_sha') or commit_sha}",
                            f"Upstream: {branch_row.get('upstream') or '(none)'}",
                        ]
                    ),
                    fact_text=(
                        f"Branch {branch_name} currently points to commit {branch_row.get('commit_sha') or commit_sha}"
                        + (f" and tracks upstream {branch_row.get('upstream')}." if branch_row.get("upstream") else ".")
                    ),
                    anchor=build_codebase_anchor(
                        "branch",
                        branch_name=branch_name,
                        commit_sha=branch_row.get("commit_sha") or commit_sha,
                        upstream_branch=branch_row.get("upstream"),
                    ),
                    currentness="current" if branch_name == str(repo.get("head_branch") or "") else "historical",
                    topic_key=branch_name,
                    state_label="branch",
                    entities=[branch_name, branch_row.get("commit_sha") or commit_sha],
                    tags=["branch", "graph_support"],
                    extra_metadata={
                        "branch_name": branch_name,
                        "commit_sha": branch_row.get("commit_sha") or commit_sha,
                        "upstream_branch": branch_row.get("upstream") or "",
                        "graph_support": True,
                    },
                    relation_facts=[
                        build_codebase_relation_fact(
                            relation_type="branch_points_to_commit",
                            from_id=f"branch:{branch_name}",
                            to_id=f"commit:{branch_row.get('commit_sha') or commit_sha}",
                            fact_text=f"{branch_name} -> {branch_row.get('commit_sha') or commit_sha}",
                            anchor=build_codebase_anchor(
                                "branch",
                                branch_name=branch_name,
                                commit_sha=branch_row.get("commit_sha") or commit_sha,
                            ),
                            entities=[branch_name, branch_row.get("commit_sha") or commit_sha],
                        )
                    ],
                )
            )

        for diff in matched_diffs[:8]:
            diff_path = _normalize_path(str(diff.get("path") or ""))
            file_row = file_rows.get(diff_path, {})
            if diff_path:
                _append_unit(
                    build_codebase_object_unit(
                        object_type="file",
                        object_id=f"file:{diff_path}",
                        raw_text="\n".join(
                            [
                                f"File: {diff_path}",
                                f"Directory: {file_row.get('directory_path') or str(Path(diff_path).parent) or '.'}",
                                f"Blob SHA: {file_row.get('blob_sha') or '(unknown)'}",
                            ]
                        ),
                        fact_text=f"File {diff_path} exists in repository {repo_name}.",
                        anchor=build_codebase_anchor("file", path=diff_path, blob_sha=file_row.get("blob_sha") or ""),
                        currentness="current",
                        topic_key=diff_path.replace("/", "_"),
                        state_label="file",
                        entities=[diff_path, file_row.get("directory_path") or "."],
                        tags=["file", "graph_support"],
                        extra_metadata={
                            "path": diff_path,
                            "directory_path": file_row.get("directory_path") or str(Path(diff_path).parent) or ".",
                            "blob_sha": file_row.get("blob_sha") or "",
                            "graph_support": True,
                        },
                    )
                )

            diff_relations: list[dict[str, Any]] = []
            for hunk_idx, hunk in enumerate(list(diff.get("hunks") or [])[:4], start=1):
                hunk_header = str(hunk.get("header") or "").strip()
                hunk_id = f"hunk:commit:{commit_sha}:{diff_path}:{hunk_idx}"
                diff_relations.append(
                    build_codebase_relation_fact(
                        relation_type="diff_contains_hunk",
                        from_id=f"diff:commit:{commit_sha}:{diff_path}",
                        to_id=hunk_id,
                        fact_text=f"{commit_sha} {diff_path} {hunk_header}",
                        anchor=build_codebase_anchor(
                            "hunk",
                            commit_sha=commit_sha,
                            path=diff_path,
                            hunk_header=hunk_header,
                            old_range=[hunk.get("old_start"), hunk.get("old_count")],
                            new_range=[hunk.get("new_start"), hunk.get("new_count")],
                        ),
                        entities=[commit_sha, diff_path, hunk_header],
                    )
                )
                _append_unit(
                    build_codebase_object_unit(
                        object_type="hunk",
                        object_id=hunk_id,
                        raw_text="\n".join(
                            [
                                f"Hunk header: {hunk_header}",
                                f"Commit SHA: {commit_sha}",
                                f"File path: {diff_path}",
                                f"Old range: {hunk.get('old_start')}:{hunk.get('old_count')}",
                                f"New range: {hunk.get('new_start')}:{hunk.get('new_count')}",
                            ]
                        ),
                        fact_text=f"Hunk {hunk_header} belongs to commit {commit_sha} and file {diff_path}.",
                        anchor=build_codebase_anchor(
                            "hunk",
                            commit_sha=commit_sha,
                            path=diff_path,
                            hunk_header=hunk_header,
                            old_range=[hunk.get("old_start"), hunk.get("old_count")],
                            new_range=[hunk.get("new_start"), hunk.get("new_count")],
                        ),
                        source_date=str(commit.get("committed_at") or ""),
                        currentness="historical",
                        topic_key=f"{commit_sha[:10]}_{hunk_idx}",
                        state_label="hunk",
                        entities=[commit_sha, diff_path, hunk_header],
                        tags=["hunk", "graph_support"],
                        extra_metadata={
                            "commit_sha": commit_sha,
                            "path": diff_path,
                            "hunk_header": hunk_header,
                            "graph_support": True,
                        },
                    )
                )

            _append_unit(
                build_codebase_object_unit(
                    object_type="diff",
                    object_id=f"diff:commit:{commit_sha}:{diff_path}",
                    raw_text="\n".join(
                        [
                            f"Diff object: diff:commit:{commit_sha}:{diff_path}",
                            f"Commit SHA: {commit_sha}",
                            f"File path: {diff_path}",
                            f"Status: {diff.get('status', 'modified')}",
                            f"Hydration state: {diff.get('hydration_state', 'deferred')}",
                            f"Hunk count: {len(diff.get('hunks', []))}",
                            f"Patch ref: {diff.get('patch_ref') or '(none)'}",
                        ]
                    ),
                    fact_text=f"Diff for commit {commit_sha} changes file {diff_path} with {len(diff.get('hunks', []))} hunks.",
                    anchor=build_codebase_anchor(
                        "diff",
                        commit_sha=commit_sha,
                        path=diff_path,
                        status=diff.get("status", "modified"),
                    ),
                    source_date=str(commit.get("committed_at") or ""),
                    currentness="historical",
                    topic_key=f"{commit_sha[:10]}_{diff_path.replace('/', '_')}",
                    state_label="diff",
                    entities=[commit_sha, diff_path],
                    tags=["diff", "graph_support", diff.get("status", "modified")],
                    extra_metadata={
                        "commit_sha": commit_sha,
                        "path": diff_path,
                        "status": diff.get("status", "modified"),
                        "hydration_state": diff.get("hydration_state", "deferred"),
                        "patch_ref": diff.get("patch_ref", ""),
                        "graph_support": True,
                    },
                    relation_facts=diff_relations,
                )
            )

        if len(rows) >= max_facts:
            break

    return rows[:max_facts]


def expand_codebase_evidence_bundle(
    *,
    query: str,
    seed_facts: list[dict[str, Any]],
    candidate_facts: list[dict[str, Any]],
    max_facts: int = 18,
    max_relations_per_object: int = 8,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    object_facts, relations_by_object, _fact_lookup = _build_graph_indices(candidate_facts)
    query_terms = _query_priority_terms(query)
    retrieved: list[dict[str, Any]] = []
    seen_fact_ids: set[str] = set()
    queue: list[str] = []
    seed_object_ids: list[str] = []

    def _add_fact(fact: dict[str, Any]) -> None:
        fact_id = str(fact.get("id") or "").strip()
        if fact_id and fact_id in seen_fact_ids:
            return
        if fact_id:
            seen_fact_ids.add(fact_id)
        retrieved.append(fact)

    for fact in seed_facts:
        _add_fact(fact)
        object_id = codebase_object_id_from_fact(fact)
        if object_id and object_id not in seed_object_ids:
            seed_object_ids.append(object_id)
        from_id, to_id = codebase_relation_endpoints(fact)
        for endpoint in (from_id, to_id):
            if endpoint and endpoint not in seed_object_ids:
                seed_object_ids.append(endpoint)

    queue.extend(seed_object_ids)
    visited_objects: set[str] = set()

    while queue and len(retrieved) < max_facts:
        object_id = queue.pop(0)
        if not object_id or object_id in visited_objects:
            continue
        visited_objects.add(object_id)
        object_fact = object_facts.get(object_id)
        if object_fact is not None:
            _add_fact(object_fact)
            if len(retrieved) >= max_facts:
                break
        adjacent = sorted(
            relations_by_object.get(object_id, []),
            key=lambda fact: _relation_priority(fact, query_terms),
            reverse=True,
        )
        for relation in adjacent[:max_relations_per_object]:
            _add_fact(relation)
            if len(retrieved) >= max_facts:
                break
            from_id, to_id = codebase_relation_endpoints(relation)
            for neighbor in (from_id, to_id):
                if neighbor and neighbor not in visited_objects and neighbor not in queue:
                    queue.append(neighbor)

    if len(retrieved) < max_facts and query_terms:
        extras = [
            fact
            for fact in candidate_facts
            if str(fact.get("id") or "").strip() not in seen_fact_ids
            and any(term.lower() in str(fact.get("fact") or "").lower() for term in query_terms)
        ]
        for fact in extras:
            _add_fact(fact)
            if len(retrieved) >= max_facts:
                break

    trace = {
        "seed_object_ids": seed_object_ids,
        "visited_objects": sorted(visited_objects),
        "query_terms": sorted(query_terms),
        "retrieved_fact_ids": [str(fact.get("id") or "") for fact in retrieved],
    }
    return retrieved, trace


def _source_record_codebase_scope_values(record: dict[str, Any]) -> tuple[str, str]:
    source_meta_raw = record.get("source_meta")
    source_meta: dict[str, Any] = source_meta_raw if isinstance(source_meta_raw, dict) else {}
    metadata_raw = record.get("metadata")
    metadata: dict[str, Any] = metadata_raw if isinstance(metadata_raw, dict) else {}
    codebase_raw = source_meta.get("codebase_context")
    codebase: dict[str, Any] = codebase_raw if isinstance(codebase_raw, dict) else {}
    repo_id = str(
        codebase.get("repo_id")
        or source_meta.get("repo_id")
        or metadata.get("repo_id")
        or ""
    ).strip()
    revision = str(
        codebase.get("revision")
        or source_meta.get("revision")
        or metadata.get("revision")
        or metadata.get("commit_id")
        or metadata.get("base_commit")
        or ""
    ).strip()
    return repo_id, revision


def _source_record_matches_repo_scope(record: dict[str, Any], repo_scope: dict[str, Any] | None) -> bool:
    if not isinstance(repo_scope, dict) or not repo_scope:
        return True
    wanted_repo = str(repo_scope.get("repo_id") or repo_scope.get("repo") or "").strip()
    wanted_commit = str(repo_scope.get("commit_id") or repo_scope.get("commit") or "").strip()
    if not wanted_repo and not wanted_commit:
        return True
    repo_id, revision = _source_record_codebase_scope_values(record)
    if wanted_repo and repo_id and repo_id != wanted_repo:
        return False
    if wanted_commit and revision and revision != wanted_commit:
        return False
    return True


def _codebase_source_ids(
    server,
    *,
    source_ids: set[str] | None = None,
    repo_scope: dict[str, Any] | None = None,
) -> set[str]:
    records = getattr(server, "_source_records", {}) or {}
    codebase_ids = {
        source_id
        for source_id, record in records.items()
        if (record or {}).get("family") == "codebase"
    }
    scoped_ids = {
        source_id
        for source_id in codebase_ids
        if _source_record_matches_repo_scope(records.get(source_id) or {}, repo_scope)
    }
    # Apply repo scope only when it resolves to at least one codebase source. Older
    # caches and synthetic tests may omit portable repo_id/commit metadata; source
    # selection must not silently disable Repository context in that case.
    candidate_ids = scoped_ids or codebase_ids
    if source_ids:
        filtered = {source_id for source_id in source_ids if source_id in candidate_ids}
        if filtered:
            return filtered
        # Repository work items often carry a work-item/source id that is distinct
        # from the codebase projection source id. Fall back to scoped codebase
        # sources instead of dropping exact-source planning.
        return candidate_ids
    return candidate_ids


def _codebase_active_revision_ids(server, source_ids: set[str]) -> set[str]:
    graph = normalize_container_graph(getattr(server, "_container_graph", None))
    return {
        str(row.get("container_graph_revision_id") or "")
        for row in graph.get("graph_revisions", [])
        if str(row.get("source_id") or "") in source_ids
        and str(row.get("family") or "") == "codebase"
        and str(row.get("adapter_name") or "") == "codebase_semantic_container_graph"
        and str(row.get("status") or "") == "active"
    }


def _codebase_query_terms(query: str) -> dict[str, set[str]]:
    raw_query = str(query or "")
    lowered = raw_query.lower()
    contextual_stopwords = set(_QUERY_STOPWORDS)
    repo_refs: set[str] = set()
    for line in raw_query.splitlines():
        if line.lower().startswith("repository:"):
            repo_ref = line.split(":", 1)[1].strip().lower()
            if repo_ref:
                repo_refs.add(repo_ref)
            for part in re.split(r"[^a-z0-9_]+", repo_ref):
                if len(part) >= 3:
                    contextual_stopwords.add(part)
    def _repo_path_like(term: str) -> bool:
        value = _normalize_path(term).lower()
        if "/" not in value:
            return False
        if _FILE_TOKEN_RE.search(value):
            return True
        if value.startswith(("./", "/")):
            return True
        return len([part for part in value.split("/") if part]) >= 3

    paths = {
        _normalize_path(term).lower()
        for term in _PATH_RE.findall(raw_query)
        if term.strip()
        and _normalize_path(term).lower() not in repo_refs
        and _repo_path_like(term)
    }
    file_tokens = {term.lower() for term in _FILE_TOKEN_RE.findall(raw_query)}
    symbols = {
        match.group(1).lower()
        for pattern in (
            r"\bclass\s+([A-Za-z_][A-Za-z0-9_]*)\b",
            r"\bfunction\s+named\s+([A-Za-z_][A-Za-z0-9_]*)\b",
            r"\bName:\s*([A-Za-z_][A-Za-z0-9_.]*)\b",
        )
        for match in re.finditer(pattern, raw_query)
    }
    symbols.update(match.group(0).lower() for match in re.finditer(r"\b[A-Z][A-Za-z0-9_]{2,}\b", raw_query))
    for path in paths:
        name = Path(path).name.lower()
        if name:
            file_tokens.add(name)
    tokens: set[str] = set()
    exact_terms: set[str] = set(paths | file_tokens)
    for token in _WORD_RE.findall(lowered):
        token = token.strip(".,:;()[]{}<>\"'")
        if len(token) < 3 or token in contextual_stopwords:
            continue
        tokens.add(token)
        if any(sep in token for sep in (".", "_", "/", "-")):
            if "/" not in token or _repo_path_like(token):
                exact_terms.add(token)
            if "." in token:
                tail = token.rsplit(".", 1)[-1]
                if len(tail) >= 3 and tail not in contextual_stopwords:
                    symbols.add(tail)
            for part in re.split(r"[./_-]+", token):
                if len(part) >= 3 and part not in contextual_stopwords:
                    tokens.add(part)
    return {
        "tokens": tokens,
        "exact_terms": exact_terms,
        "paths": paths,
        "file_tokens": file_tokens,
        "symbols": symbols,
    }


def _codebase_render_ref_map(graph: dict[str, Any], revision_ids: set[str]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("render_ref_id") or ""): row
        for row in graph.get("render_refs", [])
        if str(row.get("container_graph_revision_id") or "") in revision_ids
        and str(row.get("render_ref_id") or "")
    }


def _codebase_search_text(container: dict[str, Any], render_ref: dict[str, Any] | None) -> tuple[str, str]:
    traits = dict(container.get("traits_json") or {})
    render_json = dict((render_ref or {}).get("ref_json") or {})
    metadata_text = " ".join(
        str(value or "")
        for value in (
            container.get("kind_fq"),
            traits.get("name"),
            traits.get("qualified_name"),
            traits.get("path"),
            json.dumps(traits.get("payload") or {}, ensure_ascii=False, sort_keys=True),
        )
    )
    render_text = str(render_json.get("text") or "")
    return metadata_text.lower(), render_text.lower()


def _codebase_container_score(
    container: dict[str, Any],
    render_ref: dict[str, Any] | None,
    query_terms: dict[str, set[str]],
) -> tuple[int, list[str]]:
    traits = dict(container.get("traits_json") or {})
    path = _normalize_path(str(traits.get("path") or "")).lower()
    name = str(traits.get("name") or "").lower()
    qualified_name = str(traits.get("qualified_name") or "").lower()
    kind_fq = str(container.get("kind_fq") or "")
    metadata_text, render_text = _codebase_search_text(container, render_ref)
    score = 0
    reasons: list[str] = []

    for query_path in query_terms["paths"]:
        if _path_matches(query_path, path):
            score += 40
            reasons.append(f"path:{query_path}")
    for file_token in query_terms["file_tokens"]:
        if file_token and Path(path).name.lower() == file_token:
            score += 30
            reasons.append(f"filename:{file_token}")
    for term in query_terms["exact_terms"]:
        if not term:
            continue
        if term in path:
            score += 18
            reasons.append(f"path_term:{term}")
        elif term in name or term in qualified_name:
            score += 14
            reasons.append(f"name_term:{term}")
        elif term in render_text:
            score += 10
            reasons.append(f"source_term:{term}")
        elif term in metadata_text:
            score += 6
            reasons.append(f"metadata_term:{term}")
    for symbol in query_terms.get("symbols", set()):
        if not symbol:
            continue
        if symbol == name or qualified_name.endswith(f".{symbol}"):
            score += 24
            reasons.append(f"symbol:{symbol}")
        elif symbol in name or symbol in qualified_name:
            score += 10
            reasons.append(f"symbol_term:{symbol}")

    token_hits = 0
    for token in query_terms["tokens"]:
        if token == name or qualified_name.endswith(f".{token}"):
            score += 22
            reasons.append(f"name_token:{token}")
        elif token in path or token in name or token in qualified_name:
            token_hits += 2
        elif token in render_text:
            token_hits += 1
    if token_hits:
        score += min(token_hits, 30)
        reasons.append(f"token_hits:{token_hits}")

    if kind_fq in {"code:file", "code:config", "code:test_case", "code:class", "code:function", "code:method"}:
        score += 3
    if "test" in query_terms["tokens"] and (kind_fq == "code:test_case" or "/test" in path or path.startswith("test")):
        score += 5
        reasons.append("test_scope")
    if not render_ref:
        score -= 20
        reasons.append("missing_render_ref")
    return score, reasons


def _codebase_best_file_render_refs(
    graph: dict[str, Any],
    revision_ids: set[str],
    render_by_id: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for container in graph.get("containers", []):
        if str(container.get("container_graph_revision_id") or "") not in revision_ids:
            continue
        if str(container.get("kind_fq") or "") not in {"code:file", "code:config", "code:dependency_manifest", "code:lockfile"}:
            continue
        traits = dict(container.get("traits_json") or {})
        path = _normalize_path(str(traits.get("path") or ""))
        render_ref = render_by_id.get(str(container.get("primary_render_ref_id") or ""))
        if render_ref is None:
            continue
        render_json = dict(render_ref.get("ref_json") or {})
        text = str(render_json.get("text") or "")
        if not path or not text:
            continue
        existing = best.get(path)
        if existing is None or len(str((existing.get("ref_json") or {}).get("text") or "")) < len(text):
            best[path] = render_ref
    return best


_CODEBASE_TYPED_SPAN_KINDS = {
    "code:symbol",
    "code:class",
    "code:function",
    "code:method",
    "code:type",
    "code:interface",
    "code:test_case",
    "code:fixture",
    "code:mock",
    "code:config",
}


def _codebase_container_span(container: dict[str, Any]) -> dict[str, Any] | None:
    render_ref_json = container.get("render_ref_json") or {}
    if isinstance(render_ref_json, dict) and isinstance(render_ref_json.get("span"), dict):
        return dict(render_ref_json["span"])
    for ref in container.get("span_refs_json") or []:
        if isinstance(ref, dict) and isinstance(ref.get("span"), dict):
            return dict(ref["span"])
    return None


def _codebase_container_pinned_window(
    container: dict[str, Any],
    *,
    base_line: int,
    line_count: int,
    padding: int = 2,
) -> tuple[int, int] | None:
    kind_fq = str(container.get("kind_fq") or "")
    if kind_fq not in _CODEBASE_TYPED_SPAN_KINDS:
        return None
    if kind_fq == "code:config":
        traits = dict(container.get("traits_json") or {})
        payload = dict(traits.get("payload") or {})
        if payload.get("config_object_kind") != "option":
            return None
    span = _codebase_container_span(container)
    if not span:
        return None
    try:
        start_line = int(span.get("start_line") or 0)
        end_line = int(span.get("end_line") or start_line)
    except (TypeError, ValueError):
        return None
    if start_line <= 0 or end_line < start_line:
        return None
    start = max(0, start_line - base_line - padding)
    end = min(line_count, end_line - base_line + padding + 1)
    if end <= start:
        return None
    return start, end


def _codebase_merge_windows(windows: list[tuple[int, int]], *, max_gap: int = 6) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(windows):
        if end <= start:
            continue
        if merged and start <= merged[-1][1] + max_gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _codebase_file_items_by_path(
    graph: dict[str, Any],
    revision_ids: set[str],
    render_by_id: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    file_kinds = {"code:file", "code:config", "code:dependency_manifest", "code:lockfile", "code:workflow"}
    items: dict[str, dict[str, Any]] = {}
    for container in graph.get("containers", []):
        if str(container.get("container_graph_revision_id") or "") not in revision_ids:
            continue
        if str(container.get("kind_fq") or "") not in file_kinds:
            continue
        traits = dict(container.get("traits_json") or {})
        path = _normalize_path(str(traits.get("path") or ""))
        if not path:
            continue
        render_ref = render_by_id.get(str(container.get("primary_render_ref_id") or ""))
        render_text = str(((render_ref or {}).get("ref_json") or {}).get("text") or "")
        existing = items.get(path)
        existing_text = str((((existing or {}).get("render_ref") or {}).get("ref_json") or {}).get("text") or "")
        if existing is None or len(render_text) > len(existing_text):
            items[path] = {
                "score": 0,
                "container": container,
                "render_ref": render_ref,
                "reasons": [],
                "supporting_container_ids": [container.get("container_id")],
                "supporting_kinds": [container.get("kind_fq")],
                "supporting": [],
            }
    return items


def _codebase_find_file_item(path: str, file_items: dict[str, dict[str, Any]]) -> tuple[str | None, dict[str, Any] | None]:
    normalized = _normalize_path(path).lower()
    if not normalized:
        return None, None
    exact = [candidate_path for candidate_path in file_items if candidate_path.lower() == normalized]
    if exact:
        selected = exact[0]
        return selected, file_items[selected]
    suffix = [candidate_path for candidate_path in file_items if _path_matches(normalized, candidate_path)]
    if len(suffix) == 1:
        selected = suffix[0]
        return selected, file_items[selected]
    basename = Path(normalized).name
    if basename and "/" not in normalized:
        basename_matches = [candidate_path for candidate_path in file_items if Path(candidate_path).name.lower() == basename]
        if len(basename_matches) == 1:
            selected = basename_matches[0]
            return selected, file_items[selected]
    return None, None


def _codebase_resolve_path_constraints(
    constraints: list[dict[str, Any]],
    file_items: dict[str, dict[str, Any]],
) -> tuple[list[tuple[str, dict[str, Any], dict[str, Any]]], list[dict[str, Any]], list[dict[str, Any]]]:
    selected: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    resolution: list[dict[str, Any]] = []
    omitted_required: list[dict[str, Any]] = []
    for constraint in constraints:
        requested_path = str(constraint.get("path") or "").strip()
        resolved_path, item = _codebase_find_file_item(requested_path, file_items)
        if not item or not resolved_path:
            row: dict[str, Any] = {
                "path": requested_path,
                "status": "failed_closed" if constraint.get("required") else "not_applicable",
                "code": "REQUIRED_PATH_NOT_FOUND" if constraint.get("required") else "PATH_NOT_FOUND",
                "reason": "explicit_path_constraint_not_found",
                "constraint": constraint,
            }
            resolution.append(row)
            if constraint.get("required"):
                omitted_required.append(row)
            continue
        render_ref = item.get("render_ref")
        render_json = dict((render_ref or {}).get("ref_json") or {})
        if not render_ref:
            row = {
                "path": requested_path,
                "resolved_path": resolved_path,
                "status": "failed_closed",
                "code": "REQUIRED_PATH_NO_RENDER_REF",
                "reason": "explicit_path_constraint_has_no_render_ref",
                "container_id": (item.get("container") or {}).get("container_id"),
                "constraint": constraint,
            }
            resolution.append(row)
            if constraint.get("required"):
                omitted_required.append(row)
            continue
        if render_json.get("render_source") != "repo_file_blob_span" or render_json.get("text_exact") is not True:
            row = {
                "path": requested_path,
                "resolved_path": resolved_path,
                "status": "failed_closed",
                "code": "REQUIRED_PATH_RENDER_NOT_EXACT",
                "reason": "explicit_path_constraint_render_not_exact_repo_blob",
                "container_id": (item.get("container") or {}).get("container_id"),
                "render_ref_id": render_ref.get("render_ref_id"),
                "render_source": render_json.get("render_source"),
                "text_exact": render_json.get("text_exact"),
                "constraint": constraint,
            }
            resolution.append(row)
            if constraint.get("required"):
                omitted_required.append(row)
            continue
        row = {
            "path": requested_path,
            "resolved_path": resolved_path,
            "status": "selected",
            "container_id": (item.get("container") or {}).get("container_id"),
            "render_ref_id": render_ref.get("render_ref_id"),
            "render_mode": "exact_copy",
            "render_source": render_json.get("render_source"),
            "whole_file": None,
            "line_ranges": [],
            "reason": "explicit_path_constraint",
            "constraint": constraint,
        }
        resolution.append(row)
        selected.append((resolved_path, item, row))
    return selected, resolution, omitted_required


def _codebase_config_source_required_by_patch_query(
    item: dict[str, Any],
    query_terms: dict[str, set[str]],
) -> bool:
    """Return true when a config file is required by typed config evidence.

    This is a generic repo-work rule: if a patch task asks about configuration
    behavior and the Repository context graph has structured config option containers in a
    file that match the query, that file is required patch context. The core
    still does not parse config syntax here; it only consumes plugin-emitted
    semantic config containers.
    """

    container = item.get("container") or {}
    if str(container.get("kind_fq") or "") != "code:config":
        return False
    entries = _codebase_semantic_config_option_entries(
        item.get("supporting") or [],
        query_terms,
        max_entries=1,
    )
    return bool(entries)


def _codebase_promote_required_config_constraints(
    *,
    candidate_by_path: dict[str, dict[str, Any]],
    query_terms: dict[str, set[str]],
    explicit_path_constraints: list[dict[str, Any]],
    path_constraint_resolution: list[dict[str, Any]],
    omitted_required_files: list[dict[str, Any]],
) -> None:
    """Promote typed config-source matches into required path constraints."""

    existing_paths = {
        _normalize_path(str(row.get("path") or "")).lower()
        for row in explicit_path_constraints
        if str(row.get("path") or "").strip()
    }
    for path, item in sorted(candidate_by_path.items()):
        normalized_path = _normalize_path(path)
        if not normalized_path or normalized_path.lower() in existing_paths:
            continue
        if not _codebase_config_source_required_by_patch_query(item, query_terms):
            continue
        constraint = {
            "path": normalized_path,
            "source": "query",
            "sources": ["query"],
            "required": True,
            "reason": "config_semantics_required_by_patch_query",
            "reasons": ["config_semantics_required_by_patch_query"],
            "role": "config_source",
            "derived": True,
        }
        render_ref = item.get("render_ref")
        render_json = dict((render_ref or {}).get("ref_json") or {})
        if not render_ref:
            row = {
                "path": normalized_path,
                "resolved_path": normalized_path,
                "status": "failed_closed",
                "code": "REQUIRED_PATH_NO_RENDER_REF",
                "reason": "required_config_source_has_no_render_ref",
                "container_id": (item.get("container") or {}).get("container_id"),
                "constraint": constraint,
            }
            omitted_required_files.append(row)
        elif render_json.get("render_source") != "repo_file_blob_span" or render_json.get("text_exact") is not True:
            row = {
                "path": normalized_path,
                "resolved_path": normalized_path,
                "status": "failed_closed",
                "code": "REQUIRED_PATH_RENDER_NOT_EXACT",
                "reason": "required_config_source_render_not_exact_repo_blob",
                "container_id": (item.get("container") or {}).get("container_id"),
                "render_ref_id": render_ref.get("render_ref_id"),
                "render_source": render_json.get("render_source"),
                "text_exact": render_json.get("text_exact"),
                "constraint": constraint,
            }
            omitted_required_files.append(row)
        else:
            row = {
                "path": normalized_path,
                "resolved_path": normalized_path,
                "status": "selected",
                "container_id": (item.get("container") or {}).get("container_id"),
                "render_ref_id": render_ref.get("render_ref_id"),
                "render_mode": "exact_copy",
                "render_source": render_json.get("render_source"),
                "whole_file": None,
                "line_ranges": [],
                "reason": "config_semantics_required_by_patch_query",
                "constraint": constraint,
            }
            item["path_constraint"] = constraint
            item["path_constraint_resolution"] = row
            item["reasons"] = list(dict.fromkeys(["config_semantics_required_by_patch_query", *list(item.get("reasons") or [])]))[:12]
            item["score"] = max(int(item.get("score") or 0), 900_000)
        explicit_path_constraints.append(constraint)
        path_constraint_resolution.append(row)
        existing_paths.add(normalized_path.lower())


def _language_for_source_block(path: str) -> str:
    suffix = Path(path).suffix.lower().lstrip(".")
    language_by_suffix = {
        "py": "python",
        "pyi": "python",
        "js": "javascript",
        "jsx": "javascript",
        "ts": "typescript",
        "tsx": "typescript",
        "yml": "yaml",
        "yaml": "yaml",
        "toml": "toml",
        "json": "json",
        "rs": "rust",
        "go": "go",
        "java": "java",
        "sh": "bash",
        "bash": "bash",
    }
    return language_by_suffix.get(suffix, "text")


def _codebase_raw_line_windows(
    *,
    text: str,
    query_terms: dict[str, set[str]],
    base_line: int = 1,
    radius: int = 24,
    max_windows: int = 8,
    max_chars: int = 60000,
) -> tuple[str, list[dict[str, Any]]]:
    lines = text.splitlines()
    if not lines:
        return "", []
    _rendered, windows = _codebase_line_windows(
        text=text,
        query_terms=query_terms,
        base_line=base_line,
        radius=radius,
        max_windows=max_windows,
        max_chars=max_chars,
        include_line_numbers=False,
    )
    clean_parts: list[str] = []
    for window in windows:
        start_line = int(window.get("start_line") or base_line)
        end_line = int(window.get("end_line") or start_line)
        start_idx = max(0, start_line - base_line)
        end_idx = min(len(lines), end_line - base_line + 1)
        if end_idx <= start_idx:
            continue
        clean_parts.append("\n".join(lines[start_idx:end_idx]))
    return "\n\n".join(clean_parts).strip("\n")[:max_chars].rstrip(), windows


def _codebase_patch_role_limits(role: str) -> tuple[bool, int, int, int]:
    """Return whole-file policy and window budgets for patch source roles."""

    normalized = str(role or "").strip()
    if normalized in {"editable_source", "config_source"}:
        return True, _PATCH_CONTEXT_WHOLE_FILE_MAX_CHARS, 60000, 12000
    if normalized == "test_source":
        return True, 20000, 9000, 3000
    if normalized in {"workflow_source", "supporting_source"}:
        return True, 16000, 8000, 3000
    return True, 12000, 8000, 3000


def _codebase_render_patch_source(
    *,
    render_json: dict[str, Any],
    query_terms: dict[str, set[str]],
    remaining_budget: int,
    role: str,
) -> tuple[str, list[dict[str, Any]], bool, str | None]:
    text = str(render_json.get("text") or "")
    if not text:
        return "", [], False, "REQUIRED_PATH_NO_RENDER_REF"
    lines = text.splitlines()
    base_line = int((render_json.get("span") or {}).get("start_line") or 1)
    allow_whole_file, whole_file_max_chars, window_max_chars, min_window_chars = _codebase_patch_role_limits(role)
    whole_file_fits = (
        allow_whole_file
        and len(lines) <= _PATCH_CONTEXT_WHOLE_FILE_MAX_LINES
        and len(text) <= min(_PATCH_CONTEXT_WHOLE_FILE_MAX_CHARS, whole_file_max_chars)
        and len(text) <= remaining_budget
    )
    if whole_file_fits:
        return text, [{"start_line": base_line, "end_line": base_line + len(lines) - 1}], True, None
    if remaining_budget < min(min_window_chars, max(1, len(text))):
        return "", [], False, "REQUIRED_PATH_EXCEEDS_CONTEXT_BUDGET"
    rendered, windows = _codebase_raw_line_windows(
        text=text,
        query_terms=query_terms,
        base_line=base_line,
        max_chars=min(max(remaining_budget, 0), window_max_chars),
    )
    if not rendered:
        return "", [], False, "REQUIRED_PATCH_CONTEXT_INCOMPLETE"
    return rendered, windows, False, None


_PATCH_ROLE_PRIORITY = {
    "editable_source": 0,
    "config_source": 0,
    "test_source": 1,
    "workflow_source": 2,
    "supporting_source": 2,
}


def _codebase_patch_candidate_role(item: dict[str, Any]) -> str:
    return str((item.get("path_constraint") or {}).get("role") or "semantic_code_candidate")


def _codebase_patch_candidate_sort_key(path: str, item: dict[str, Any]) -> tuple[int, int, str]:
    role = _codebase_patch_candidate_role(item)
    explicit_order = item.get("explicit_selection_order")
    try:
        order = int(explicit_order if explicit_order is not None else 10_000)
    except (TypeError, ValueError):
        order = 10_000
    return (_PATCH_ROLE_PRIORITY.get(role, 3), order, path)


def _codebase_typed_span_priority(container: dict[str, Any]) -> int:
    kind_fq = str(container.get("kind_fq") or "")
    traits = dict(container.get("traits_json") or {})
    payload = dict(traits.get("payload") or {})
    if kind_fq == "code:config" and payload.get("config_object_kind") == "option":
        return 35
    if kind_fq in {"code:class", "code:type", "code:interface"}:
        return 40
    if kind_fq in {"code:function", "code:method", "code:test_case", "code:fixture", "code:mock"}:
        return 30
    if kind_fq == "code:symbol":
        return 10
    return 0


def _codebase_typed_span_anchor_priority(container: dict[str, Any], query_terms: dict[str, set[str]]) -> int:
    kind_fq = str(container.get("kind_fq") or "")
    traits = dict(container.get("traits_json") or {})
    payload = dict(traits.get("payload") or {})
    qualified_name = str(traits.get("qualified_name") or traits.get("name") or "").lower()
    if kind_fq == "code:config" and payload.get("config_object_kind") == "option":
        qualified_name = " ".join(
            str(value or "").lower()
            for value in (
                payload.get("option"),
                payload.get("type"),
                payload.get("default"),
                json.dumps(payload.get("references") or [], ensure_ascii=False),
            )
        )
        query_names = set(query_terms.get("tokens") or set()) | set(query_terms.get("symbols") or set())
        if any(len(token) >= 3 and token in qualified_name for token in query_names):
            return 45
        return 0
    simple_name = qualified_name.rsplit(".", 1)[-1]
    if not simple_name:
        return 0
    query_names = set(query_terms.get("tokens") or set()) | set(query_terms.get("symbols") or set())
    if simple_name not in query_names:
        return 0
    if kind_fq in {"code:class", "code:type", "code:interface"}:
        return 80
    if kind_fq in {"code:function", "code:method", "code:test_case"}:
        return 50
    if kind_fq == "code:symbol":
        return 20
    return 10


def _codebase_window_is_covered(window: tuple[int, int], windows: list[tuple[int, int]]) -> bool:
    start, end = window
    return any(existing_start <= start and end <= existing_end for existing_start, existing_end in windows)


def _codebase_ordered_uncovered_windows(
    windows: list[tuple[int, int]],
    existing: list[tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
    ordered: list[tuple[int, int]] = list(existing or [])
    for window in windows:
        start, end = window
        if end <= start or _codebase_window_is_covered(window, ordered):
            continue
        ordered.append(window)
    return ordered


def _codebase_line_windows(
    *,
    text: str,
    query_terms: dict[str, set[str]],
    base_line: int = 1,
    radius: int = 22,
    max_windows: int = 4,
    max_chars: int = 9000,
    prefer_config_sections: bool = False,
    pinned_windows: list[tuple[int, int]] | None = None,
    include_line_numbers: bool = True,
) -> tuple[str, list[dict[str, Any]]]:
    lines = text.splitlines()
    if not lines:
        return "", []
    exact_terms = {
        term
        for term in query_terms["exact_terms"]
        if len(term) >= 4 and (any(sep in term for sep in (".", "_", "/", "-")) or term in query_terms["file_tokens"])
    }
    search_tokens = {term for term in query_terms["tokens"] if len(term) >= 5}
    symbols = {term for term in query_terms.get("symbols", set()) if len(term) >= 3}
    scored_hits: list[tuple[int, int]] = []
    for idx, line in enumerate(lines):
        lowered = line.lower()
        line_score = 0
        exact_hit = False
        for term in exact_terms:
            if term and term in lowered:
                line_score += 6
                exact_hit = True
        for term in search_tokens:
            if term and term in lowered:
                line_score += 2
        for symbol in symbols:
            if symbol and symbol in lowered:
                line_score += 4
        if line_score and re.search(r"\b(?:class|def|function|describe|it|test)\b", lowered):
            line_score += 8
        if symbols and re.search(r"\b(?:class|def)\s+([A-Za-z_][A-Za-z0-9_]*)", line):
            declared = re.search(r"\b(?:class|def)\s+([A-Za-z_][A-Za-z0-9_]*)", line)
            if declared and declared.group(1).lower() in symbols:
                line_score += 24
        if exact_hit and re.search(r"\b(?:class|def)\b", lowered):
            line_score += 8
        if line_score:
            scored_hits.append((line_score, idx))
    if not scored_hits:
        scored_hits = [(1, 0)]
    scored_hits.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    selected_hits: list[int] = []
    for _score, idx in scored_hits:
        if any(abs(idx - existing) <= radius for existing in selected_hits):
            continue
        selected_hits.append(idx)
        if len(selected_hits) >= max_windows:
            break
    if prefer_config_sections and pinned_windows:
        # Config source windows are selected from semantic option containers.
        # Do not add loose lexical hits from unrelated config branches.
        selected_hits = []
    selected_hits.sort()
    windows: list[tuple[int, int]] = []
    if pinned_windows:
        # Keep semantic span order from the planner. Sorting by file position can bury the
        # work-item's primary symbol below less relevant earlier windows in large files.
        windows = _codebase_ordered_uncovered_windows(list(pinned_windows))
    supplemental_windows: list[tuple[int, int]] = []
    for hit in selected_hits:
        start = max(0, hit - radius)
        end = min(len(lines), hit + radius + 1)
        supplemental_windows = _codebase_merge_windows([*supplemental_windows, (start, end)])
        if len(windows) + len(supplemental_windows) >= max_windows:
            break
    for window in supplemental_windows:
        if len(windows) >= max_windows:
            break
        windows = _codebase_ordered_uncovered_windows([window], windows)
    if not windows:
        windows = _codebase_merge_windows(supplemental_windows)
    rendered_parts: list[str] = []
    trace_windows: list[dict[str, Any]] = []
    used_chars = 0
    for start, end in windows:
        if include_line_numbers:
            part_lines = [
                f"{line_no:>5}: {line}"
                for line_no, line in enumerate(lines[start:end], start=base_line + start)
            ]
            part = "\n".join(part_lines)
        else:
            start_line = base_line + start
            end_line = base_line + end - 1
            part = (
                f"[source lines {start_line}-{end_line}; metadata only, not repository code]\n"
                + "\n".join(lines[start:end])
            )
        if used_chars + len(part) > max_chars:
            remaining = max_chars - used_chars
            if remaining <= 200:
                break
            part = part[:remaining].rstrip()
        rendered_parts.append(part)
        used_chars += len(part)
        trace_windows.append({"start_line": base_line + start, "end_line": base_line + end - 1})
        if used_chars >= max_chars:
            break
    return _SOURCE_WINDOW_SEPARATOR.join(rendered_parts), trace_windows


def _codebase_config_query_prefixes(query_terms: dict[str, set[str]]) -> set[str]:
    prefixes: set[str] = set()
    ignored_parts = {"config", "val", "settings", "setting", "option", "options"}
    for term in query_terms.get("exact_terms", set()):
        raw_term = str(term or "").lower()
        if "." not in raw_term:
            continue
        parts = [
            part
            for part in re.split(r"[./]+", raw_term)
            if len(part) >= 2 and part not in ignored_parts
        ]
        if len(parts) >= 2:
            prefixes.add(parts[0])
    return prefixes


def _codebase_requested_config_tokens(query_terms: dict[str, set[str]]) -> list[str]:
    requested: set[str] = set()
    for values in (query_terms.get("tokens", set()), query_terms.get("symbols", set()), query_terms.get("exact_terms", set())):
        for raw in values:
            for token in re.findall(r"[A-Za-z][A-Za-z0-9_]*_[A-Za-z0-9_]+", str(raw or "")):
                requested.add(token.lower())
    return sorted(requested)


def _codebase_query_requests_default_literal_substitution(query_terms: dict[str, set[str]]) -> bool:
    requested = set(_codebase_requested_config_tokens(query_terms))
    if any(token.startswith("default_") for token in requested):
        return True
    tokens = set(query_terms.get("tokens") or set()) | set(query_terms.get("symbols") or set())
    return "default" in tokens and any(token in tokens for token in {"size", "family", "font", "value", "values"})


def _codebase_semantic_config_option_entries(
    supporting: list[dict[str, Any]],
    query_terms: dict[str, set[str]],
    *,
    max_entries: int = 512,
) -> list[dict[str, Any]]:
    """Return structured config candidates from Repository context semantic containers."""

    prefixes = _codebase_config_query_prefixes(query_terms)
    requested_tokens = _codebase_requested_config_tokens(query_terms)
    requests_default_literal_substitution = _codebase_query_requests_default_literal_substitution(query_terms)
    query_tokens = set(query_terms.get("tokens") or set()) | set(query_terms.get("symbols") or set())
    exact_terms = set(query_terms.get("exact_terms") or set())

    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for support in supporting:
        container = dict(support.get("container") or {})
        traits = dict(container.get("traits_json") or {})
        payload = dict(traits.get("payload") or {})
        if str(container.get("kind_fq") or "") != "code:config":
            continue
        if payload.get("config_object_kind") != "option":
            continue
        option = str(payload.get("option") or traits.get("name") or "").strip()
        if not option or option in seen:
            continue
        seen.add(option)
        option_lower = option.lower()
        option_prefix = str(payload.get("option_prefix") or option_lower.split(".", 1)[0]).lower()
        if prefixes and option_prefix not in prefixes and option_lower not in exact_terms:
            continue
        type_name = str(payload.get("type") or "")
        type_name_lower = type_name.lower()
        default_value = str(payload.get("default") or "")
        searchable = json.dumps(payload, ensure_ascii=False, sort_keys=True).lower()
        references = sorted(
            token
            for token in requested_tokens
            if token in searchable
        )
        default_tokens = list(payload.get("default_tokens") or [])
        leading_literal_token = str(payload.get("leading_literal_token") or "")
        leading_literal_kind = str((default_tokens[0] or {}).get("kind") or "") if default_tokens else ""
        render_ref = dict(support.get("render_ref") or {})
        render_json = dict(render_ref.get("ref_json") or {})
        span = _codebase_container_span(container) or dict(render_json.get("span") or {})
        path = _normalize_path(str(traits.get("path") or render_json.get("path") or payload.get("path") or ""))
        has_leading_default_literal = bool(
            leading_literal_token
            and default_tokens
            and (
                leading_literal_kind in {"size_literal", "number_literal"}
                or (leading_literal_kind == "literal" and bool(re.search(r"[A-Za-z0-9]", leading_literal_token)))
            )
        )
        exact_option = option_lower in exact_terms or any(term and term in option_lower for term in exact_terms)
        requested_reference = any(
            token in references
            and (
                token in query_tokens
                or token.replace("_", ".") in exact_terms
                or any(token in term for term in exact_terms)
            )
            for token in references
        )
        type_anchor = any(
            len(str(token or "")) >= 3
            and (
                str(token).lower() in type_name_lower
                or str(token).lower().rstrip("s") in type_name_lower
            )
            for token in query_tokens
        )
        option_anchor = any(
            len(str(token or "")) >= 3
            and (
                str(token).lower() in option_lower
                or str(token).lower().rstrip("s") in option_lower
            )
            for token in query_tokens
        )
        type_or_option_anchor = bool(type_anchor or option_anchor)
        default_literal_candidate = bool(
            prefixes
            and requests_default_literal_substitution
            and has_leading_default_literal
            and (type_anchor or (not type_name_lower and option_anchor))
        )
        if prefixes and not (
            exact_option
            or requested_reference
            or (type_or_option_anchor and references)
            or default_literal_candidate
        ):
            continue
        score = 0
        reasons: list[str] = []
        if exact_option:
            score += 40
            reasons.append("option_exact_term")
        if option_prefix in prefixes:
            score += 24
            reasons.append(f"option_prefix:{option_prefix}")
        if requested_reference:
            score += 28
            reasons.append("requested_reference")
        if default_literal_candidate:
            score += 18
            reasons.append("default_literal_candidate")
        for token in query_tokens:
            token = str(token or "").lower()
            if len(token) < 3:
                continue
            token_singular = token[:-1] if token.endswith("s") else token
            if token in option_lower or token_singular in option_lower:
                score += 10
                reasons.append(f"option_token:{token}")
            elif token in str(type_name).lower() or token_singular in str(type_name).lower():
                score += 8
                reasons.append(f"type_token:{token}")
            elif option_prefix in prefixes and (token in str(default_value).lower() or token in searchable):
                score += 3
        if option_prefix in prefixes and any(token in searchable for token in query_tokens if len(str(token)) >= 4):
            score += 8
            reasons.append("prefix_block_token")
        if score <= 0:
            continue
        entries.append(
            {
                "path": path,
                "option": option,
                "config_object_kind": str(payload.get("config_object_kind") or "option"),
                "option_prefix": option_prefix,
                "type": type_name,
                "default": default_value,
                "start_line": span.get("start_line"),
                "end_line": span.get("end_line"),
                "source_span": {
                    "path": path,
                    "start_line": span.get("start_line"),
                    "end_line": span.get("end_line"),
                },
                "render_ref_id": render_ref.get("render_ref_id") or container.get("primary_render_ref_id"),
                "references": references,
                "requested_tokens": requested_tokens,
                "default_tokens": default_tokens,
                "leading_literal_token": leading_literal_token,
                "container_id": container.get("container_id"),
                "proof_source_fields": list(payload.get("proof_source_fields") or []),
                "score": score,
                "trace": {"reasons": list(dict.fromkeys(reasons))[:8]},
            }
        )
    entries.sort(key=lambda row: (int(row.get("score") or 0), str(row.get("option") or "")), reverse=True)
    return entries[:max_entries]


def _codebase_config_entry_reason_set(entry: dict[str, Any]) -> set[str]:
    return {
        str(reason or "")
        for reason in ((entry.get("trace") or {}).get("reasons") or [])
        if str(reason or "")
    }


def _codebase_config_entry_has_default_literal(entry: dict[str, Any]) -> bool:
    leading = str(entry.get("leading_literal_token") or "")
    if not leading:
        return False
    tokens = list(entry.get("default_tokens") or [])
    if not tokens:
        return False
    kind = str((tokens[0] or {}).get("kind") or "")
    return kind in {"size_literal", "number_literal"} or (kind == "literal" and bool(re.search(r"[A-Za-z0-9]", leading)))


def _codebase_config_transform_type_families(entries: list[dict[str, Any]]) -> set[str]:
    families: set[str] = set()
    for entry in entries:
        type_name = str(entry.get("type") or "").strip().lower()
        if not type_name:
            continue
        reasons = _codebase_config_entry_reason_set(entry)
        if "default_literal_candidate" in reasons or (
            entry.get("references") and _codebase_config_entry_has_default_literal(entry)
        ):
            families.add(type_name)
    return families


def _codebase_config_applicability_checklist(
    entries: list[dict[str, Any]],
    query_terms: dict[str, set[str]],
) -> list[dict[str, Any]]:
    """Build mandatory config/default transformation coverage rows."""

    if not entries:
        return []
    prefixes = _codebase_config_query_prefixes(query_terms)
    requested_tokens = _codebase_requested_config_tokens(query_terms)
    type_families = _codebase_config_transform_type_families(entries)
    rows: list[dict[str, Any]] = []
    for entry in entries:
        reasons = _codebase_config_entry_reason_set(entry)
        option = str(entry.get("option") or "")
        option_prefix = str(entry.get("option_prefix") or option.lower().split(".", 1)[0]).lower()
        type_name = str(entry.get("type") or "").strip()
        type_lower = type_name.lower()
        has_default_literal = _codebase_config_entry_has_default_literal(entry)
        same_section = bool(option_prefix and (not prefixes or option_prefix in prefixes))
        same_type_family = bool(type_lower and type_lower in type_families)
        default_literal_candidate = "default_literal_candidate" in reasons
        applicable = bool(default_literal_candidate or (has_default_literal and same_section and same_type_family))
        selection_reasons: list[str] = []
        rejection_reasons: list[str] = []
        if applicable:
            selection_reasons.append("default_literal_candidate")
            if requested_tokens:
                selection_reasons.append("matches_requested_token_family")
            if same_section:
                selection_reasons.append("same_config_section")
            if same_type_family:
                selection_reasons.append("same_option_type_family")
            if entry.get("references"):
                selection_reasons.append("references_existing_default_token")
        else:
            if not has_default_literal:
                rejection_reasons.append("no_transformable_default_literal")
            if not same_section:
                rejection_reasons.append("different_config_section")
            if type_name and type_families and not same_type_family:
                rejection_reasons.append("different_option_type_family")
            if "option_exact_term" in reasons or "requested_reference" in reasons:
                rejection_reasons.append("requested_anchor_not_transform_target")
            if not rejection_reasons:
                rejection_reasons.append("not_applicable_to_config_default_transformation")
        rows.append(
            {
                "path": entry.get("path"),
                "config_key": option,
                "config_object_kind": entry.get("config_object_kind") or "option",
                "current_default": entry.get("default"),
                "current_type": entry.get("type"),
                "source_span": entry.get("source_span")
                or {
                    "path": entry.get("path"),
                    "start_line": entry.get("start_line"),
                    "end_line": entry.get("end_line"),
                },
                "render_ref_id": entry.get("render_ref_id"),
                "applicable": applicable,
                "required": applicable,
                "selection_reasons": list(dict.fromkeys(selection_reasons)),
                "rejection_reasons": list(dict.fromkeys(rejection_reasons)),
                "coverage_status": "selected" if applicable else "rejected",
                "trace": {
                    "source": "plugin_emitted_config_option",
                    "planner": "codebase_graph_query_context",
                    "raw_entry_reasons": sorted(reasons),
                },
            }
        )
    rows.sort(
        key=lambda row: (
            0 if row.get("applicable") else 1,
            str(row.get("path") or ""),
            int((row.get("source_span") or {}).get("start_line") or 0),
            str(row.get("config_key") or ""),
        )
    )
    return rows


def _codebase_requested_config_keys_from_terms(
    query_terms: dict[str, set[str]],
    checklist: list[dict[str, Any]],
) -> list[str]:
    requested = _codebase_requested_config_tokens(query_terms)
    if not requested:
        return []
    prefixes = _codebase_config_query_prefixes(query_terms)
    if not prefixes:
        prefixes = {
            str(row.get("config_key") or "").split(".", 1)[0].lower()
            for row in checklist
            if str(row.get("config_key") or "")
        }
    keys: list[str] = []
    for token in requested:
        if "." in token:
            keys.append(token)
            continue
        if prefixes:
            keys.extend(f"{prefix}.{token}" for prefix in sorted(prefixes) if prefix)
        else:
            keys.append(token)
    return sorted(dict.fromkeys(keys))


def _line_span_for_term(text: str, term: str) -> dict[str, Any]:
    lowered = text.lower()
    needle = str(term or "").lower()
    if not needle:
        return {}
    for idx, line in enumerate(lowered.splitlines(), start=1):
        if needle in line:
            return {"start_line": idx, "end_line": idx}
    return {}


def _codebase_config_type_validation_evidence(
    selected_items: list[dict[str, Any]],
    checklist: list[dict[str, Any]],
    query_terms: dict[str, set[str]],
) -> list[dict[str, Any]]:
    """Return structured type/validation evidence for config transformation work."""

    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()

    def add_row(row: dict[str, Any]) -> None:
        key = (
            str(row.get("path") or ""),
            str(row.get("symbol_or_config_key") or ""),
            str(row.get("evidence_kind") or ""),
            str((row.get("source_span") or {}).get("start_line") or ""),
        )
        if key in seen:
            return
        seen.add(key)
        rows.append(row)

    applicable_rows = [row for row in checklist if row.get("applicable")]
    requested_keys = _codebase_requested_config_keys_from_terms(query_terms, checklist)
    types = sorted(
        {
            str(row.get("current_type") or "").strip()
            for row in applicable_rows
            if str(row.get("current_type") or "").strip()
        }
    )
    for row in checklist:
        span = dict(row.get("source_span") or {})
        config_key = str(row.get("config_key") or "")
        if row.get("current_type"):
            add_row(
                {
                    "path": row.get("path"),
                    "symbol_or_config_key": config_key,
                    "evidence_kind": "schema_type",
                    "source_span": span,
                    "render_ref_id": row.get("render_ref_id"),
                    "summary": f"{config_key} declares type {row.get('current_type')}",
                    "applicability": "direct" if row.get("applicable") else "sibling",
                }
            )
        if row.get("current_default"):
            add_row(
                {
                    "path": row.get("path"),
                    "symbol_or_config_key": config_key,
                    "evidence_kind": "default_value",
                    "source_span": span,
                    "render_ref_id": row.get("render_ref_id"),
                    "summary": f"{config_key} current default is {row.get('current_default')}",
                    "applicability": "direct" if row.get("applicable") else "sibling",
                }
            )

    for requested_key in requested_keys:
        if any(str(row.get("config_key") or "").lower() == requested_key.lower() for row in checklist):
            continue
        sibling_types = ", ".join(types) or "unknown"
        sibling_keys = [
            str(row.get("config_key") or "")
            for row in applicable_rows
            if str(row.get("config_key") or "")
        ][:12]
        if sibling_keys:
            add_row(
                {
                    "path": None,
                    "symbol_or_config_key": requested_key,
                    "evidence_kind": "sibling_option_type",
                    "source_span": {},
                    "render_ref_id": None,
                    "summary": (
                        f"{requested_key} is requested but absent; applicable sibling options "
                        f"use type family {sibling_types}: {', '.join(sibling_keys)}"
                    ),
                    "applicability": "sibling",
                }
            )
        else:
            add_row(
                {
                    "path": None,
                    "symbol_or_config_key": requested_key,
                    "evidence_kind": "validation_gap",
                    "source_span": {},
                    "render_ref_id": None,
                    "summary": f"No sibling type evidence found for requested config key {requested_key}",
                    "applicability": "unknown",
                    "coverage_status": "failed_closed",
                    "failure_code": "missing_config_validation_evidence",
                }
            )

    source_items = [
        item
        for item in selected_items
        if str(item.get("kind_fq") or "") not in {"code:config", "code:dependency_manifest", "code:lockfile"}
    ]
    for type_name in types:
        if not type_name:
            continue
        class_re = re.compile(rf"\bclass\s+{re.escape(type_name)}\b")
        for item in source_items:
            text = str(item.get("text") or "")
            if not text:
                continue
            if class_re.search(text):
                span = _line_span_for_term(text, f"class {type_name}")
                add_row(
                    {
                        "path": item.get("path"),
                        "symbol_or_config_key": type_name,
                        "evidence_kind": "parser",
                        "source_span": {"path": item.get("path"), **span},
                        "render_ref_id": item.get("render_ref_id"),
                        "summary": f"Source contains parser/type class {type_name}",
                        "applicability": "direct",
                    }
                )
            if "validationerror" in text.lower() or "_basic_py_validation" in text:
                span = _line_span_for_term(text, "ValidationError") or _line_span_for_term(text, "_basic_py_validation")
                add_row(
                    {
                        "path": item.get("path"),
                        "symbol_or_config_key": type_name,
                        "evidence_kind": "validator",
                        "source_span": {"path": item.get("path"), **span},
                        "render_ref_id": item.get("render_ref_id"),
                        "summary": f"Source includes validation logic near {type_name}",
                        "applicability": "nearby",
                    }
                )

    requested_terms = set(_codebase_requested_config_tokens(query_terms)) | {
        str(row.get("config_key") or "").rsplit(".", 1)[-1].lower()
        for row in applicable_rows
    }
    for item in selected_items:
        path = str(item.get("path") or "")
        if "test" not in path.lower():
            continue
        text = str(item.get("text") or "")
        lowered = text.lower()
        matched = sorted(term for term in requested_terms if term and term in lowered)
        if matched:
            span = _line_span_for_term(text, matched[0])
            add_row(
                {
                    "path": path,
                    "symbol_or_config_key": ",".join(matched[:8]),
                    "evidence_kind": "test_assertion",
                    "source_span": {"path": path, **span},
                    "render_ref_id": item.get("render_ref_id"),
                    "summary": f"Test source mentions config/type terms: {', '.join(matched[:8])}",
                    "applicability": "test",
                }
            )

    if requested_keys and not any(row.get("evidence_kind") in {"schema_type", "parser", "validator", "sibling_option_type"} for row in rows):
        add_row(
            {
                "path": None,
                "symbol_or_config_key": ",".join(requested_keys),
                "evidence_kind": "validation_gap",
                "source_span": {},
                "render_ref_id": None,
                "summary": "No config validation/type evidence found for requested config change",
                "applicability": "unknown",
                "coverage_status": "failed_closed",
                "failure_code": "missing_config_validation_evidence",
            }
        )
    return rows


def _codebase_render_config_applicability_checklist(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    lines = [
        "--- CONFIG APPLICABILITY CHECKLIST ---",
        "Treat this structured checklist as mandatory coverage evidence for config/default transformations.",
        "Every applicable=true required=true row must be handled by a patch or explicitly reported as blocked by missing evidence.",
        "config_key | path | default | type | applicable | required | coverage_status | selection_reasons | rejection_reasons | render_ref | lines",
    ]
    for row in rows:
        span = dict(row.get("source_span") or {})
        selection = ",".join(row.get("selection_reasons") or []) or "-"
        rejection = ",".join(row.get("rejection_reasons") or []) or "-"
        lines.append(
            f"{row.get('config_key')} | {row.get('path') or '-'} | {row.get('current_default') or '-'} | "
            f"{row.get('current_type') or '-'} | {row.get('applicable')} | {row.get('required')} | "
            f"{row.get('coverage_status')} | {selection} | {rejection} | "
            f"{row.get('render_ref_id') or '-'} | L{span.get('start_line')}-L{span.get('end_line')}"
        )
    return "\n".join(lines)


def _codebase_render_config_type_validation_evidence(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    lines = [
        "--- CONFIG TYPE / VALIDATION EVIDENCE ---",
        "Use this evidence when adding or changing config options. If required evidence is missing, call memory_recall with a focused codebase query before patching.",
        "symbol_or_config_key | evidence_kind | path | applicability | summary | render_ref | lines",
    ]
    for row in rows:
        span = dict(row.get("source_span") or {})
        lines.append(
            f"{row.get('symbol_or_config_key') or '-'} | {row.get('evidence_kind')} | "
            f"{row.get('path') or '-'} | {row.get('applicability') or '-'} | "
            f"{row.get('summary') or '-'} | {row.get('render_ref_id') or '-'} | "
            f"L{span.get('start_line')}-L{span.get('end_line')}"
        )
    return "\n".join(lines)


def _codebase_render_semantic_config_option_entries(entries: list[dict[str, Any]]) -> str:
    if not entries:
        return ""
    lines = [
        "Structured config option candidates from Repository context semantic bundle:",
        "Treat this table as an applicability checklist for config-default substitution work items.",
        "reason=default_literal_candidate means the option has a leading literal default under the requested config prefix/type even if it does not yet reference the old token.",
        "option | type | default | default_tokens | leading_literal | requested_refs | selection_reasons | lines",
    ]
    for entry in entries:
        references = ", ".join(entry.get("references") or []) or "-"
        default_tokens = " ".join(
            f"{row.get('token')}:{row.get('kind')}"
            for row in entry.get("default_tokens") or []
        ) or "-"
        reasons = ",".join((entry.get("trace") or {}).get("reasons") or []) or "-"
        lines.append(
            f"{entry.get('option')} | {entry.get('type') or '-'} | "
            f"{entry.get('default') or '-'} | {default_tokens} | "
            f"{entry.get('leading_literal_token') or '-'} | {references} | {reasons} | "
            f"L{entry.get('start_line')}-L{entry.get('end_line')}"
        )
    return "\n".join(lines)


def _codebase_semantic_config_option_windows(
    entries: list[dict[str, Any]],
    *,
    base_line: int = 1,
    line_count: int,
    max_windows: int = 12,
    padding: int = 0,
) -> list[tuple[int, int]]:
    """Return ordered source windows for structured config candidates.

    Candidate tables are useful for reasoning, but patch generation still needs
    the exact raw source around each key. These windows are derived from the
    same exact render_ref-backed source that produced the candidate entries.
    """
    windows: list[tuple[int, int]] = []
    for entry in entries:
        try:
            start_line = int(entry.get("start_line") or 0)
            end_line = int(entry.get("end_line") or 0)
        except (TypeError, ValueError):
            continue
        if start_line <= 0 or end_line < start_line:
            continue
        start = max(0, start_line - base_line - padding)
        end = min(line_count, end_line - base_line + 1 + padding)
        if end <= start:
            continue
        windows = _codebase_ordered_uncovered_windows([(start, end)], windows)
        if len(windows) >= max_windows:
            break
    return windows


def _repo_task_query_context_pack(
    *,
    query: str,
    selected_items: list[dict[str, Any]],
    rejected_candidates: list[dict[str, Any]],
    source_ids: set[str],
    revision_ids: set[str],
    repo_task_contract: dict[str, Any] | None = None,
    explicit_path_constraints: list[dict[str, Any]] | None = None,
    path_constraint_resolution: list[dict[str, Any]] | None = None,
    patch_context_policy: dict[str, Any] | None = None,
    selected_whole_files: list[str] | None = None,
    selected_windowed_files: list[str] | None = None,
    omitted_required_files: list[dict[str, Any]] | None = None,
    operation_state: list[dict[str, Any]] | None = None,
    config_applicability_checklist: list[dict[str, Any]] | None = None,
    config_type_validation_evidence: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    semantic_candidates = []
    editable_source_files = []
    test_source_files = []
    supporting_source_files = []
    commands = []
    workflows = []
    tests = []
    operation_state_rows = list(operation_state or [])
    for item in selected_items:
        entry = {
            "path": item.get("path"),
            "kind_fq": item.get("kind_fq"),
            "container_id": item.get("container_id"),
            "render_ref_id": item.get("render_ref_id"),
            "render_source": item.get("render_source"),
            "text_exact": item.get("text_exact"),
            "score": item.get("score"),
            "reasons": list(item.get("reasons") or []),
            "windows": list(item.get("windows") or []),
            "typed_span_windows": list(item.get("typed_span_windows") or []),
            "config_option_entries": list(item.get("config_option_entries") or []),
            "config_applicability_checklist": list(item.get("config_applicability_checklist") or []),
            "path_constraint": item.get("path_constraint"),
            "whole_file": item.get("whole_file"),
            "source_block_mode": item.get("source_block_mode"),
            "trace": {
                "reason": item.get("selection_reason") or "query_selected_from_full_typed_operator_domain",
                "source": "codebase_graph_query_context",
            },
        }
        role = str((item.get("path_constraint") or {}).get("role") or item.get("role") or "")
        if role == "editable_source":
            editable_source_files.append(entry)
        elif role == "test_source":
            test_source_files.append(entry)
        elif role:
            supporting_source_files.append(entry)
        kind_fq = str(item.get("kind_fq") or "")
        if kind_fq == "code:command":
            commands.append(entry)
        elif kind_fq == "code:workflow_job":
            workflows.append(entry)
        elif kind_fq == "code:test_case":
            tests.append(entry)
        else:
            semantic_candidates.append(entry)
    omitted_required = [
        row for row in (path_constraint_resolution or [])
        if row.get("status") == "failed_closed" and (row.get("constraint") or {}).get("required")
    ]
    return {
        "artifact_kind": "context_pack",
        "status": "transient",
        "context_pack_kind": "repo_task_context_pack",
        "context_pack_generation": "query_specific",
        "work_item": {
            "query": query,
            "source_ids": sorted(source_ids),
            "active_revision_ids": sorted(revision_ids),
            "repo_task_contract": repo_task_contract or {},
            "trace": {"reason": "memory_recall_query_work_item"},
        },
        "repo_scope": dict((repo_task_contract or {}).get("repo_scope") or {}),
        "task_contract": {
            "work_item_kind": (repo_task_contract or {}).get("work_item_kind") or "repo:work_item",
            "task_mode": (repo_task_contract or {}).get("task_mode"),
            "output_artifact": (repo_task_contract or {}).get("output_artifact"),
            "required_render_mode": (repo_task_contract or {}).get("required_render_mode"),
        },
        "sections": {
            "requirements": [{"text": query, "trace": {"reason": "caller_query"}}],
            "mentioned_code": [],
            "explicit_path_constraints": list(explicit_path_constraints or []),
            "editable_source_files": editable_source_files,
            "test_source_files": test_source_files,
            "supporting_source_files": supporting_source_files,
            "semantic_code_candidates": semantic_candidates,
            "config_applicability_checklist": list(config_applicability_checklist or []),
            "config_type_validation_evidence": list(config_type_validation_evidence or []),
            "dependency_neighborhood": [],
            "tests": tests,
            "commands": commands,
            "workflows": workflows,
            "failures": [],
            "docs": [],
            "conversation": [],
            "history": [],
            "operation_state": operation_state_rows,
            "patch_attempts": [row for row in operation_state_rows if row.get("kind_fq") == "operation:patch_attempt"],
            "verification": [row for row in operation_state_rows if row.get("kind_fq") == "operation:verification_result"],
            "rejected_candidates": rejected_candidates,
        },
        "patch_context_policy": dict(patch_context_policy or {}),
        "path_constraint_resolution": {
            str(row.get("path") or row.get("resolved_path") or idx): row
            for idx, row in enumerate(path_constraint_resolution or [])
        },
        "selected_whole_files": list(selected_whole_files or []),
        "selected_windowed_files": list(selected_windowed_files or []),
        "omitted_required_files": list(omitted_required_files or omitted_required),
        "coverage_report_refs": [],
        "risk_report": {"status": "not_generated" if not operation_state_rows else "has_operation_state", "reason": "no_patch_attempt" if not operation_state_rows else "operation_state_available"},
        "completeness_report": {
            "status": "query_pack",
            "silent_absence": False,
            "selected_candidate_count": len(selected_items),
            "rejected_candidate_count": len(rejected_candidates),
            "explicit_path_constraint_count": len(explicit_path_constraints or []),
            "omitted_required_file_count": len(omitted_required_files or omitted_required),
            "operation_state_count": len(operation_state_rows),
            "config_applicability_checklist_count": len(config_applicability_checklist or []),
            "config_type_validation_evidence_count": len(config_type_validation_evidence or []),
        },
        "trace": {
            "planner": "codebase_graph_query_context",
            "context_pack_kind": "repo_task_context_pack",
            "operator_domain_policy": "full_typed_container_domain_in_scope",
            "seed_domain_policy": "facts_and_query_are_seed_only",
            "artifact_persistence": "transient_memory_recall_response",
            "explicit_path_constraint_count": len(explicit_path_constraints or []),
            "path_constraint_resolution_count": len(path_constraint_resolution or []),
            "operation_state_count": len(operation_state_rows),
            "config_applicability_checklist_count": len(config_applicability_checklist or []),
            "config_type_validation_evidence_count": len(config_type_validation_evidence or []),
        },
    }


def _codebase_operation_state(graph: dict[str, Any], revision_ids: set[str]) -> list[dict[str, Any]]:
    artifact_by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for artifact in graph.get("artifacts", []) or []:
        if str(artifact.get("container_graph_revision_id") or "") not in revision_ids:
            continue
        subject_id = str(artifact.get("subject_id") or "")
        if subject_id:
            artifact_by_subject[subject_id].append(artifact)

    rows: list[dict[str, Any]] = []
    for container in graph.get("containers", []) or []:
        if str(container.get("container_graph_revision_id") or "") not in revision_ids:
            continue
        kind_fq = str(container.get("kind_fq") or "")
        if not kind_fq.startswith("operation:"):
            continue
        traits = dict(container.get("traits_json") or {})
        payload = dict(traits.get("payload") or {})
        container_id = str(container.get("container_id") or "")
        artifacts = [
            {
                "artifact_id": artifact.get("artifact_id"),
                "artifact_kind": artifact.get("artifact_kind"),
                "payload": artifact.get("payload_json"),
            }
            for artifact in artifact_by_subject.get(container_id, [])[:8]
        ]
        rows.append(
            {
                "container_id": container_id,
                "kind_fq": kind_fq,
                "status": payload.get("status") or container.get("status"),
                "created_at": payload.get("created_at") or container.get("created_at"),
                "payload": payload,
                "artifacts": artifacts,
                "trace": {"reason": "persisted_repo_operation_state", "source": "container_graph"},
            }
        )
    rows.sort(key=lambda row: str(row.get("created_at") or ""), reverse=True)
    return rows[:24]


def build_codebase_context(
    server,
    *,
    query: str,
    source_ids: set[str] | None = None,
    query_metadata: dict[str, Any] | None = None,
    path_constraint_query: str | None = None,
    max_files: int = 8,
    budget: int = 64000,
) -> tuple[str | None, dict[str, Any]]:
    repo_task_contract = _repo_task_contract(query_metadata)
    patch_generation = _is_patch_generation_contract(repo_task_contract, query)
    if patch_generation:
        budget = max(int(budget), _PATCH_CONTEXT_TOTAL_BUDGET)
    selected_source_ids = _codebase_source_ids(
        server,
        source_ids=source_ids,
        repo_scope=repo_task_contract.get("repo_scope") if isinstance(repo_task_contract, dict) else None,
    )
    if not selected_source_ids:
        return None, {"mode": "inactive", "reason": "no_codebase_source_ids"}
    graph = normalize_container_graph(getattr(server, "_container_graph", None))
    revision_ids = _codebase_active_revision_ids(server, selected_source_ids)
    if not revision_ids:
        return None, {
            "mode": "failed_closed",
            "reason": "no_active_codebase_container_graph_revision",
            "source_ids": sorted(selected_source_ids),
        }

    operation_state = _codebase_operation_state(graph, revision_ids)
    render_by_id = _codebase_render_ref_map(graph, revision_ids)
    query_terms = _codebase_query_terms(query)
    if not (query_terms["tokens"] or query_terms["exact_terms"]):
        return None, {"mode": "inactive", "reason": "no_query_anchors", "source_ids": sorted(selected_source_ids)}

    constraint_query = str(path_constraint_query if path_constraint_query is not None else query)
    explicit_path_constraints = extract_repo_work_item_path_constraints(constraint_query, query_metadata)
    patch_context_policy = deepcopy(_PATCH_CONTEXT_POLICY) if patch_generation else {}
    query_mentions_config = bool(
        {"config", "configuration", "setting", "settings", "schema", "option", "options"}
        & query_terms["tokens"]
    )

    scored: list[tuple[int, dict[str, Any], dict[str, Any] | None, list[str]]] = []
    rejected: list[dict[str, Any]] = []
    for container in graph.get("containers", []):
        if str(container.get("container_graph_revision_id") or "") not in revision_ids:
            continue
        kind_fq = str(container.get("kind_fq") or "")
        if not (
            kind_fq.startswith("code:")
            or kind_fq.startswith("operation:")
            or kind_fq in {"repo:work_item", "repo:issue", "repo:bug_report", "repo:refactor_request"}
        ):
            continue
        render_ref = render_by_id.get(str(container.get("primary_render_ref_id") or ""))
        score, reasons = _codebase_container_score(container, render_ref, query_terms)
        if score <= 0:
            if len(rejected) < 48:
                rejected.append(
                    {
                        "container_id": container.get("container_id"),
                        "kind_fq": kind_fq,
                        "trace": {"reason": "score_below_threshold", "score": score},
                    }
                )
            continue
        scored.append((score, container, render_ref, reasons))
    if not scored and not explicit_path_constraints:
        return None, {
            "mode": "failed_closed",
            "reason": "no_codebase_candidates_matched_query",
            "source_ids": sorted(selected_source_ids),
            "active_revision_ids": sorted(revision_ids),
            "query_terms": {key: sorted(value) for key, value in query_terms.items()},
        }
    scored.sort(key=lambda item: (item[0], str((item[1].get("traits_json") or {}).get("path") or "")), reverse=True)

    file_render_refs = _codebase_best_file_render_refs(graph, revision_ids, render_by_id)
    file_items = _codebase_file_items_by_path(graph, revision_ids, render_by_id)
    candidate_by_path: dict[str, dict[str, Any]] = {}
    for score, container, render_ref, reasons in scored:
        traits = dict(container.get("traits_json") or {})
        path = _normalize_path(str(traits.get("path") or ""))
        if not path:
            continue
        support = {
            "score": score,
            "container": container,
            "render_ref": render_ref,
            "reasons": list(reasons),
        }
        if path not in candidate_by_path:
            candidate_by_path[path] = {
                "score": score,
                "container": container,
                "render_ref": file_render_refs.get(path) or render_ref,
                "reasons": list(reasons),
                "supporting_container_ids": [container.get("container_id")],
                "supporting_kinds": [container.get("kind_fq")],
                "supporting": [support],
            }
        else:
            current = candidate_by_path[path]
            current["score"] = max(int(current["score"]), score)
            current["reasons"] = list(dict.fromkeys([*current["reasons"], *reasons]))[:12]
            container_id = container.get("container_id")
            if container_id not in current["supporting_container_ids"]:
                current["supporting_container_ids"].append(container_id)
                current["supporting_kinds"].append(container.get("kind_fq"))
                current["supporting"].append(support)

    explicit_selected: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    path_constraint_resolution: list[dict[str, Any]] = []
    omitted_required_files: list[dict[str, Any]] = []
    if explicit_path_constraints:
        explicit_selected, path_constraint_resolution, omitted_required_files = _codebase_resolve_path_constraints(
            explicit_path_constraints,
            file_items,
        )
        for order, (path, explicit_item, resolution_row) in enumerate(explicit_selected):
            score = 1_000_000 - order
            explicit_current = candidate_by_path.get(path)
            if explicit_current is None:
                candidate_by_path[path] = {
                    "score": score,
                    "container": explicit_item.get("container"),
                    "render_ref": explicit_item.get("render_ref"),
                    "reasons": ["explicit_path_constraint"],
                    "supporting_container_ids": list(explicit_item.get("supporting_container_ids") or []),
                    "supporting_kinds": list(explicit_item.get("supporting_kinds") or []),
                    "supporting": list(explicit_item.get("supporting") or []),
                }
                explicit_current = candidate_by_path[path]
            else:
                explicit_current["score"] = max(int(explicit_current.get("score") or 0), score)
                explicit_current["render_ref"] = explicit_item.get("render_ref") or explicit_current.get("render_ref")
                explicit_current["container"] = explicit_item.get("container") or explicit_current.get("container")
                explicit_current["reasons"] = list(dict.fromkeys(["explicit_path_constraint", *list(explicit_current.get("reasons") or [])]))[:12]
            explicit_current["path_constraint"] = dict(resolution_row.get("constraint") or {})
            explicit_current["path_constraint_resolution"] = resolution_row
            explicit_current["explicit_selection_order"] = order

    if patch_generation and query_mentions_config:
        _codebase_promote_required_config_constraints(
            candidate_by_path=candidate_by_path,
            query_terms=query_terms,
            explicit_path_constraints=explicit_path_constraints,
            path_constraint_resolution=path_constraint_resolution,
            omitted_required_files=omitted_required_files,
        )

    ranked_paths = sorted(candidate_by_path.items(), key=lambda pair: (int(pair[1]["score"]), pair[0]), reverse=True)
    if patch_generation:
        explicit_paths = [
            path
            for path, item in candidate_by_path.items()
            if item.get("path_constraint")
        ]
        selected_path_items = [
            (path, candidate_by_path[path])
            for path in sorted(
                explicit_paths,
                key=lambda candidate_path: _codebase_patch_candidate_sort_key(
                    candidate_path,
                    candidate_by_path[candidate_path],
                ),
            )
        ]
        selected_paths = {path for path, _item in selected_path_items}
        # Required explicit path constraints are non-evictable, but they must
        # not consume the entire budget for directly related semantic/code/config
        # neighbors. The character budget still gates final rendering.
        optional_limit = len(selected_path_items) + max(0, max_files)
        for path, item in ranked_paths:
            if path in selected_paths:
                continue
            if len(selected_path_items) >= optional_limit:
                rejected.append(
                    {
                        "path": path,
                        "container_id": (item.get("container") or {}).get("container_id"),
                        "kind_fq": (item.get("container") or {}).get("kind_fq"),
                        "trace": {"reason": "optional_candidate_evicted_by_explicit_path_constraints"},
                    }
                )
                continue
            selected_path_items.append((path, item))
            selected_paths.add(path)
    elif explicit_selected:
        selected_path_items = [
            (path, candidate_by_path[path])
            for path, _explicit_item, _resolution in sorted(
                explicit_selected,
                key=lambda row: _codebase_patch_candidate_sort_key(row[0], candidate_by_path[row[0]]),
            )
            if path in candidate_by_path
        ]
        selected_paths = {path for path, _item in selected_path_items}
        optional_limit = len(selected_path_items) + max(0, max_files)
        for path, item in ranked_paths:
            if path in selected_paths:
                continue
            if len(selected_path_items) >= optional_limit:
                rejected.append(
                    {
                        "path": path,
                        "container_id": (item.get("container") or {}).get("container_id"),
                        "kind_fq": (item.get("container") or {}).get("kind_fq"),
                        "trace": {"reason": "optional_candidate_evicted_by_explicit_path_constraints"},
                    }
                )
                continue
            selected_path_items.append((path, item))
            selected_paths.add(path)
    else:
        selected_path_items = ranked_paths[:max_files]
        selected_paths = {path for path, _item in selected_path_items}
        if query_mentions_config:
            config_candidates = [
                (path, item)
                for path, item in ranked_paths
                if str((item.get("container") or {}).get("kind_fq") or "") in {"code:config", "code:dependency_manifest", "code:lockfile"}
                and int(item.get("score") or 0) > 0
            ]
            for path, item in config_candidates[:1]:
                if path in selected_paths:
                    continue
                if len(selected_path_items) >= max_files:
                    selected_path_items = selected_path_items[:-1]
                    selected_paths = {selected_path for selected_path, _ in selected_path_items}
                selected_path_items.append((path, item))
                selected_paths.add(path)
        selected_path_items = sorted(
            selected_path_items,
            key=lambda pair: (
                1
                if query_mentions_config
                and str((pair[1].get("container") or {}).get("kind_fq") or "") in {"code:config", "code:dependency_manifest", "code:lockfile"}
                else 0,
                int(pair[1]["score"]),
                pair[0],
            ),
            reverse=True,
        )

    selected_items: list[dict[str, Any]] = []
    selected_whole_files: list[str] = []
    selected_windowed_files: list[str] = []
    remaining_source_budget = _PATCH_CONTEXT_TOTAL_BUDGET if patch_generation else budget
    for path, item in selected_path_items:
        render_ref = item.get("render_ref")
        render_json = dict((render_ref or {}).get("ref_json") or {})
        text = str(render_json.get("text") or "")
        if not text:
            if patch_generation and (item.get("path_constraint") or {}).get("required"):
                row = dict(item.get("path_constraint_resolution") or {})
                row.update({"path": path, "status": "failed_closed", "code": "REQUIRED_PATH_NO_RENDER_REF"})
                omitted_required_files.append(row)
            continue
        base_line = int((render_json.get("span") or {}).get("start_line") or 1)
        selected_kind = str((item.get("container") or {}).get("kind_fq") or "")
        prefer_config_sections = selected_kind in {"code:config", "code:dependency_manifest", "code:lockfile"}
        text_line_count = len(text.splitlines())
        typed_span_windows: list[dict[str, Any]] = []
        pinned_windows: list[tuple[int, int]] = []
        config_option_entries = (
            _codebase_semantic_config_option_entries(
                item.get("supporting") or [],
                query_terms,
                max_entries=512,
            )
            if prefer_config_sections
            else []
        )
        config_applicability_rows = (
            _codebase_config_applicability_checklist(config_option_entries, query_terms)
            if config_option_entries
            else []
        )
        config_option_container_ids = {
            str(entry.get("container_id") or "")
            for entry in config_option_entries
            if str(entry.get("container_id") or "")
        }
        supporting_candidates = item.get("supporting") or []
        if prefer_config_sections and config_option_container_ids:
            supporting_candidates = []
        supporting = sorted(
            supporting_candidates,
            key=lambda entry: (
                _codebase_typed_span_anchor_priority(entry.get("container") or {}, query_terms),
                int(entry.get("score") or 0),
                _codebase_typed_span_priority(entry.get("container") or {}),
                str((entry.get("container") or {}).get("container_id") or ""),
            ),
            reverse=True,
        )
        for entry in supporting:
            container = entry.get("container") or {}
            if (
                prefer_config_sections
                and config_option_container_ids
                and str(container.get("container_id") or "") not in config_option_container_ids
            ):
                continue
            kind_fq = str(container.get("kind_fq") or "")
            if kind_fq == "code:symbol" and len(pinned_windows) >= 4:
                continue
            window = _codebase_container_pinned_window(
                container,
                base_line=base_line,
                line_count=text_line_count,
                padding=2,
            )
            if window is None or _codebase_window_is_covered(window, pinned_windows):
                continue
            span = _codebase_container_span(container) or {}
            traits = dict(container.get("traits_json") or {})
            pinned_windows.append(window)
            typed_span_windows.append(
                {
                    "container_id": container.get("container_id"),
                    "kind_fq": container.get("kind_fq"),
                    "name": traits.get("name"),
                    "qualified_name": traits.get("qualified_name"),
                    "start_line": span.get("start_line"),
                    "end_line": span.get("end_line"),
                    "score": int(entry.get("score") or 0),
                    "reasons": list(entry.get("reasons") or [])[:8],
                    "render_policy": "full_typed_container_span",
                }
            )
            if len(pinned_windows) >= 10:
                break
        if config_option_entries:
            config_windows = _codebase_semantic_config_option_windows(
                config_option_entries,
                base_line=base_line,
                line_count=text_line_count,
                max_windows=12,
                padding=0,
            )
            pinned_windows = _codebase_ordered_uncovered_windows(config_windows, pinned_windows)
        source_role = _codebase_patch_candidate_role(item)
        explicit_path_context = bool(item.get("path_constraint"))
        if patch_generation or explicit_path_context:
            rendered, windows, whole_file, render_error = _codebase_render_patch_source(
                render_json=render_json,
                query_terms=query_terms,
                remaining_budget=max(0, remaining_source_budget),
                role=source_role,
            )
            if render_error:
                row = dict(item.get("path_constraint_resolution") or {})
                row.update(
                    {
                        "path": path,
                        "status": "failed_closed" if (item.get("path_constraint") or {}).get("required") else "not_applicable",
                        "code": render_error,
                        "reason": "explicit_path_constraint_render_failed" if item.get("path_constraint") else "optional_candidate_render_failed",
                        "container_id": (item.get("container") or {}).get("container_id"),
                        "render_ref_id": (render_ref or {}).get("render_ref_id"),
                    }
                )
                if (item.get("path_constraint") or {}).get("required"):
                    omitted_required_files.append(row)
                else:
                    rejected.append(row)
                continue
            remaining_source_budget -= len(rendered)
            selected_resolution_row = item.get("path_constraint_resolution")
            if isinstance(selected_resolution_row, dict):
                selected_resolution_row.update(
                    {
                        "status": "selected" if whole_file else "windowed",
                        "whole_file": whole_file,
                        "line_ranges": windows,
                        "render_mode": "exact_copy",
                        "render_source": render_json.get("render_source"),
                        "reason": "explicit_path_constraint",
                    }
                )
            if whole_file:
                selected_whole_files.append(path)
            else:
                selected_windowed_files.append(path)
        else:
            rendered, windows = _codebase_line_windows(
                text=text,
                query_terms=query_terms,
                base_line=base_line,
                radius=14,
                max_windows=14 if pinned_windows else 6,
                max_chars=26000 if prefer_config_sections else 22000,
                prefer_config_sections=prefer_config_sections,
                pinned_windows=pinned_windows,
                include_line_numbers=False,
            )
            whole_file = False
        if not rendered:
            continue
        selected_items.append(
            {
                "path": path,
                "score": int(item["score"]),
                "kind_fq": (item["container"] or {}).get("kind_fq"),
                "container_id": (item["container"] or {}).get("container_id"),
                "render_ref_id": (render_ref or {}).get("render_ref_id"),
                "render_source": render_json.get("render_source"),
                "raw_source_present": render_json.get("raw_source_present"),
                "text_exact": render_json.get("text_exact"),
                "byte_exact": render_json.get("byte_exact"),
                "reasons": item["reasons"][:12],
                "windows": windows,
                "typed_span_windows": typed_span_windows,
                "config_option_entries": config_option_entries,
                "config_applicability_checklist": config_applicability_rows,
                "path_constraint": item.get("path_constraint"),
                "path_constraint_resolution": item.get("path_constraint_resolution"),
                "whole_file": whole_file,
                "source_block_mode": "whole_file" if whole_file else ("contiguous_windows" if (patch_generation or explicit_path_context) else "ranked_windows"),
                "language": _language_for_source_block(path),
                "role": source_role if (patch_generation or explicit_path_context) else (item.get("path_constraint") or {}).get("role"),
                "text": rendered,
            }
        )

    config_applicability_checklist: list[dict[str, Any]] = []
    seen_config_checklist: set[tuple[str, str]] = set()
    for item in selected_items:
        for row in item.get("config_applicability_checklist") or []:
            key = (str(row.get("path") or ""), str(row.get("config_key") or ""))
            if key in seen_config_checklist:
                continue
            seen_config_checklist.add(key)
            config_applicability_checklist.append(row)
    config_type_validation_evidence = (
        _codebase_config_type_validation_evidence(selected_items, config_applicability_checklist, query_terms)
        if config_applicability_checklist
        else []
    )

    if patch_generation and omitted_required_files:
        failure_context = "Not enough grounded context.\n\n--- PATCH CONTEXT FAILURE ---\n" + "\n".join(
            f"{row.get('code')}: {row.get('path') or row.get('resolved_path')}"
            for row in omitted_required_files
        )
        return failure_context, {
            "mode": "failed_closed",
            "reason": "REQUIRED_PATCH_CONTEXT_INCOMPLETE",
            "source_ids": sorted(selected_source_ids),
            "active_revision_ids": sorted(revision_ids),
            "repo_task_contract": repo_task_contract,
            "output_artifact": repo_task_contract.get("output_artifact"),
            "task_mode": repo_task_contract.get("task_mode"),
            "required_render_mode": repo_task_contract.get("required_render_mode"),
            "explicit_path_constraints": explicit_path_constraints,
            "path_constraint_resolution": path_constraint_resolution,
            "selected_whole_files": selected_whole_files,
            "selected_windowed_files": selected_windowed_files,
            "omitted_required_files": omitted_required_files,
            "patch_context_policy": patch_context_policy,
            "source_text_exposed_to_model": False,
            "gold_or_expected_exposed": False,
            "query_context_pack": _repo_task_query_context_pack(
                query=query,
                selected_items=selected_items,
                rejected_candidates=rejected,
                source_ids=selected_source_ids,
                revision_ids=revision_ids,
                repo_task_contract=repo_task_contract,
                explicit_path_constraints=explicit_path_constraints,
                path_constraint_resolution=path_constraint_resolution,
                patch_context_policy=patch_context_policy,
                selected_whole_files=selected_whole_files,
                selected_windowed_files=selected_windowed_files,
                omitted_required_files=omitted_required_files,
                operation_state=operation_state,
                config_applicability_checklist=config_applicability_checklist,
                config_type_validation_evidence=config_type_validation_evidence,
            ),
        }

    if not selected_items:
        return None, {
            "mode": "failed_closed",
            "reason": "matched_candidates_without_renderable_source",
            "source_ids": sorted(selected_source_ids),
            "active_revision_ids": sorted(revision_ids),
            "repo_task_contract": repo_task_contract,
            "explicit_path_constraints": explicit_path_constraints,
            "path_constraint_resolution": path_constraint_resolution,
            "patch_context_policy": patch_context_policy,
            "operation_state": operation_state,
        }

    lines = [
        "--- REPOSITORY CONTEXT PACK ---",
        "Context pack kind: repo_task_context_pack",
        "Planner: codebase_graph_query_context",
        "Seed policy: vector/BM25/facts identify scope only; selected files come from full typed container graph.",
        "Render policy: source blocks below are rendered from container_render_refs with original repo file/blob spans.",
        "Patch policy: explicit repository paths are hard constraints; patch-generation source is whole-file-first exact source.",
        "Selected/rejected candidates are traced in runtime_trace.codebase_context.",
    ]
    if repo_task_contract.get("active"):
        lines.extend([
            "",
            "--- REPOSITORY WORK ITEM CONTRACT ---",
            f"Work item kind: {repo_task_contract.get('work_item_kind')}",
            f"Task mode: {repo_task_contract.get('task_mode')}",
            f"Output artifact: {repo_task_contract.get('output_artifact')}",
            f"Answer contract: {repo_task_contract.get('answer_contract')}",
            f"Required render mode: {repo_task_contract.get('required_render_mode')}",
            f"Repo scope: {repo_task_contract.get('repo_scope')}",
        ])
    if patch_generation:
        lines.extend(["", "--- PATCH CONTEXT POLICY ---", json.dumps(patch_context_policy, sort_keys=True)])
    if explicit_path_constraints:
        lines.extend(["", "--- EXPLICIT PATH CONSTRAINTS ---"])
        for constraint in explicit_path_constraints:
            lines.append(
                f"{constraint.get('path')} role={constraint.get('role')} "
                f"required={constraint.get('required')} source={constraint.get('source')} "
                f"reason={constraint.get('reason')}"
            )
        lines.extend(["", "--- PATH CONSTRAINT RESOLUTION ---", "path_constraint_resolution:"])
        for row in path_constraint_resolution:
            constraint = dict(row.get("constraint") or {})
            lines.append(
                f"{row.get('path') or row.get('resolved_path')} status={row.get('status')} "
                f"role={constraint.get('role')} required={constraint.get('required')} "
                f"render_source={row.get('render_source')} whole_file={row.get('whole_file')} "
                f"failure_code={row.get('code')} reason={row.get('reason')}"
            )
        lines.extend([
            "",
            "selected_whole_files: " + json.dumps(selected_whole_files, sort_keys=True),
            "selected_windowed_files: " + json.dumps(selected_windowed_files, sort_keys=True),
            "omitted_required_files: " + json.dumps(omitted_required_files, sort_keys=True),
        ])
    config_checklist_text = _codebase_render_config_applicability_checklist(config_applicability_checklist)
    if config_checklist_text:
        lines.extend(["", config_checklist_text])
    config_validation_text = _codebase_render_config_type_validation_evidence(config_type_validation_evidence)
    if config_validation_text:
        lines.extend(["", config_validation_text])
    if operation_state:
        lines.extend(["", "--- CURRENT OPERATION STATE ---"])
        for row in operation_state[:8]:
            payload = dict(row.get("payload") or {})
            lines.append(
                f"{row.get('kind_fq')} status={row.get('status')} "
                f"id={payload.get('patch_attempt_id') or payload.get('test_run_id') or payload.get('verification_result_id') or row.get('container_id')} "
                f"created_at={row.get('created_at')}"
            )
    used = sum(len(line) + 1 for line in lines)
    for item in selected_items:
        windows = list(item.get("windows") or [])
        line_label = ",".join(
            f"{row.get('start_line')}-{row.get('end_line')}"
            for row in windows
            if row.get("start_line") is not None and row.get("end_line") is not None
        ) or "unknown"
        if patch_generation or item.get("path_constraint"):
            role = item.get("role") or "semantic_code_candidate"
            whole_file_label = "true" if item.get("whole_file") else "false"
            header = (
                f"\n[File: {item['path']}] [render_source={item['render_source']}] "
                f"[whole_file={whole_file_label}] [lines={line_label}] [role={role}]"
            )
            resolution = dict(item.get("path_constraint_resolution") or {})
            proof = (
                f"\nSource proof: render_ref_id={item.get('render_ref_id')} "
                f"text_exact={item.get('text_exact')} byte_exact={item.get('byte_exact')} "
                f"resolution_status={resolution.get('status')} "
                f"resolution_reason={resolution.get('reason')} "
                f"selection={item.get('source_block_mode')} reasons={', '.join(item['reasons'][:6])}"
            )
            config_entries = _codebase_render_semantic_config_option_entries(item.get("config_option_entries") or [])
            config_block = f"\n{config_entries}" if config_entries else ""
            language = item.get("language") or "text"
            block = f"{header}{proof}{config_block}\n```{language}\n{item['text']}\n```"
        else:
            header = (
                f"\n[File: {item['path']}] score={item['score']} kind={item['kind_fq']} "
                f"render_source={item['render_source']} text_exact={item['text_exact']} "
                f"reasons={', '.join(item['reasons'][:6])}"
            )
            config_entries = _codebase_render_semantic_config_option_entries(item.get("config_option_entries") or [])
            config_block = f"\n{config_entries}" if config_entries else ""
            block = f"{header}{config_block}\n```text\n{item['text']}\n```"
        required_item = bool((item.get("path_constraint") or {}).get("required"))
        if used + len(block) > budget and not required_item:
            rejected.append(
                {
                    "path": item.get("path"),
                    "container_id": item.get("container_id"),
                    "kind_fq": item.get("kind_fq"),
                    "trace": {"reason": "optional_candidate_evicted_by_context_budget"},
                }
            )
            continue
        lines.append(block)
        used += len(block)

    trace = {
        "mode": "active",
        "planner": "codebase_graph_query_context",
        "source_ids": sorted(selected_source_ids),
        "active_revision_ids": sorted(revision_ids),
        "operator_domain_policy": "full_typed_container_domain_in_scope",
        "seed_domain_policy": "facts_and_query_are_seed_only",
        "render_policy": "container_render_refs_original_repo_blob_spans",
        "repo_task_contract": repo_task_contract,
        "output_artifact": repo_task_contract.get("output_artifact"),
        "task_mode": repo_task_contract.get("task_mode"),
        "required_render_mode": repo_task_contract.get("required_render_mode"),
        "path_constraint_source": "raw_work_item_query" if constraint_query != str(query or "") else "query",
        "explicit_path_constraints": explicit_path_constraints,
        "path_constraint_resolution": path_constraint_resolution,
        "selected_whole_files": selected_whole_files,
        "selected_windowed_files": selected_windowed_files,
        "omitted_required_files": omitted_required_files,
        "patch_context_policy": patch_context_policy,
        "operation_state": operation_state,
        "config_applicability_checklist": config_applicability_checklist,
        "config_type_validation_evidence": config_type_validation_evidence,
        "source_text_exposed_to_model": bool(selected_items),
        "gold_or_expected_exposed": False,
        "query_terms": {key: sorted(value) for key, value in query_terms.items()},
        "candidate_count": len(scored),
        "selected": [
            {key: value for key, value in item.items() if key != "text"}
            for item in selected_items
        ],
        "rejected_candidates": rejected,
        "query_context_pack": _repo_task_query_context_pack(
            query=query,
            selected_items=selected_items,
            rejected_candidates=rejected,
            source_ids=selected_source_ids,
            revision_ids=revision_ids,
            repo_task_contract=repo_task_contract,
            explicit_path_constraints=explicit_path_constraints,
            path_constraint_resolution=path_constraint_resolution,
            patch_context_policy=patch_context_policy,
            selected_whole_files=selected_whole_files,
            selected_windowed_files=selected_windowed_files,
            omitted_required_files=omitted_required_files,
            operation_state=operation_state,
            config_applicability_checklist=config_applicability_checklist,
            config_type_validation_evidence=config_type_validation_evidence,
        ),
    }
    return "\n".join(lines), trace


async def augment_codebase_structural_packet(
    server,
    *,
    query: str,
    packet: dict,
    episode_lookup: dict[str, dict],
    fact_filter,
) -> tuple[dict, list[dict] | None]:
    selected_source_ids = {
        (episode_lookup.get(ep_id) or {}).get("source_id", "")
        for ep_id in packet.get("retrieved_episode_ids", [])
    }
    selected_source_ids.discard("")
    if not selected_source_ids:
        selected_source_ids = {
            source_id
            for source_id, record in getattr(server, "_source_records", {}).items()
            if (record or {}).get("family") == "codebase"
        }
    if not selected_source_ids:
        return packet, None

    atomic_embs = (getattr(server, "_data_dict", None) or {}).get("atomic_embs")
    atomic_emb_indices = (getattr(server, "_data_dict", None) or {}).get("atomic_emb_indices")
    if not isinstance(atomic_embs, np.ndarray):
        return packet, None

    candidate_facts: list[dict[str, Any]] = []
    vector_candidate_facts: list[dict[str, Any]] = []
    candidate_embeddings: list[np.ndarray] = []
    fact_lookup: dict[str, dict[str, Any]] = {}
    sparse_index_map: dict[int, int] = {}
    if atomic_emb_indices is None:
        if len(atomic_embs) != len(getattr(server, "_all_granular", [])):
            return packet, None
    else:
        try:
            sparse_index_map = {
                int(fact_idx): emb_idx
                for emb_idx, fact_idx in enumerate(list(atomic_emb_indices))
                if isinstance(fact_idx, (int, np.integer))
            }
        except Exception:
            sparse_index_map = {}
    for idx, fact in enumerate(getattr(server, "_all_granular", [])):
        if not fact_filter(fact):
            continue
        source_id = str(fact.get("source_id") or (fact.get("metadata") or {}).get("episode_source_id") or "").strip()
        if source_id not in selected_source_ids:
            continue
        if fact.get("kind") not in {"codebase_object", "codebase_relation"}:
            continue
        candidate_facts.append(fact)
        emb_idx = sparse_index_map.get(idx, idx if atomic_emb_indices is None else None)
        if emb_idx is not None and 0 <= emb_idx < len(atomic_embs):
            vector_candidate_facts.append(fact)
            candidate_embeddings.append(atomic_embs[emb_idx])
        fact_id = str(fact.get("id") or "").strip()
        if fact_id:
            fact_lookup[fact_id] = fact

    graph_support_facts: list[dict[str, Any]] = []
    needs_sidecar_scan = _query_needs_sidecar_scan(query)
    for source_id in sorted(selected_source_ids):
        graph = _load_codebase_graph(server, source_id)
        if not graph:
            graph = {}
        support_facts = [
            *_build_graph_commit_support_facts(
                source_id=source_id,
                graph=graph,
                query=query,
                max_facts=max(24, int(packet.get("selector_config", {}).get("supporting_facts_total", 12)) * 2),
            ),
        ]
        if needs_sidecar_scan:
            support_facts.extend(
                _build_object_support_facts(
                    source_id=source_id,
                    server=server,
                    query=query,
                    max_facts=max(32, int(packet.get("selector_config", {}).get("supporting_facts_total", 12)) * 2),
                )
            )
            support_facts.extend(
                _build_relation_support_facts(
                    source_id=source_id,
                    server=server,
                    query=query,
                    max_facts=max(32, int(packet.get("selector_config", {}).get("supporting_facts_total", 12)) * 3),
                )
            )
        for fact in support_facts:
            fact_id = str(fact.get("id") or "").strip()
            if fact_id in fact_lookup:
                continue
            fact_lookup[fact_id] = fact
            graph_support_facts.append(fact)

    if not candidate_facts and not graph_support_facts:
        return packet, None

    query_features = extract_query_features(query)
    retrieval_target = query_features.get("retrieval_target") or query
    seed_facts: list[dict[str, Any]] = list(graph_support_facts)
    seed_fact_ids = {
        str(fact.get("id") or "").strip()
        for fact in seed_facts
        if str(fact.get("id") or "").strip()
    }
    for fact in _candidate_exact_support_facts(
        query,
        candidate_facts,
        max_facts=max(8, int(packet.get("selector_config", {}).get("supporting_facts_total", 12))),
    ):
        fact_id = str(fact.get("id") or "").strip()
        if fact_id and fact_id in seed_fact_ids:
            continue
        if fact_id:
            seed_fact_ids.add(fact_id)
        seed_facts.append(fact)
    for fact in _expand_seed_neighbors(
        query=query,
        seed_facts=list(seed_facts),
        candidate_facts=candidate_facts,
        max_extra_facts=max(16, int(packet.get("selector_config", {}).get("supporting_facts_total", 12))),
    ):
        fact_id = str(fact.get("id") or "").strip()
        if fact_id and fact_id in seed_fact_ids:
            continue
        if fact_id:
            seed_fact_ids.add(fact_id)
        seed_facts.append(fact)

    if vector_candidate_facts and candidate_embeddings:
        query_embedding = await server._embed_query_with_runtime_secrets(retrieval_target)
        sweep = source_local_fact_sweep(
            retrieval_target,
            vector_candidate_facts,
            np.asarray(candidate_embeddings),
            query_embedding=query_embedding,
            top_k=14,
            bm25_pool=36,
            vector_pool=36,
            entity_pool=18,
            rrf_k=60,
        )
        for row in sweep.get("retrieved", []):
            fact = row["fact"]
            fact_id = str(fact.get("id") or "").strip()
            if fact_id and fact_id in seed_fact_ids:
                continue
            if fact_id:
                seed_fact_ids.add(fact_id)
            seed_facts.append(fact)
        sweep_trace = sweep.get("trace") or {}
    else:
        sweep_trace = {}
    if not seed_facts:
        return packet, None

    all_candidate_facts: list[dict[str, Any]] = []
    seen_candidate_ids: set[str] = set()
    for fact in [*candidate_facts, *graph_support_facts]:
        fact_id = str(fact.get("id") or "").strip()
        if fact_id and fact_id in seen_candidate_ids:
            continue
        if fact_id:
            seen_candidate_ids.add(fact_id)
        all_candidate_facts.append(fact)

    retrieved_facts, graph_trace = expand_codebase_evidence_bundle(
        query=query,
        seed_facts=seed_facts,
        candidate_facts=all_candidate_facts,
        max_facts=max(18, int(packet.get("selector_config", {}).get("supporting_facts_total", 12))),
    )
    if not retrieved_facts:
        return packet, None

    context, actual_injected_episode_ids = build_context_from_retrieved_facts(
        retrieved_facts,
        episode_lookup,
        fact_lookup=fact_lookup,
        budget=int(packet.get("selector_config", {}).get("budget", 8000)),
        snippet_chars=int(packet.get("tuning_snapshot", {}).get("packet", {}).get("snippet_chars", 1200)),
        question=query,
        query_features=query_features,
    )

    packet = dict(packet)
    packet["context"] = context
    packet["actual_injected_episode_ids"] = actual_injected_episode_ids
    packet["retrieved_fact_ids"] = [str(fact.get("id") or "") for fact in retrieved_facts]
    packet["fact_episode_ids"] = list(
        dict.fromkeys(
            episode_id
            for fact in retrieved_facts
            for episode_id in fact_episode_ids(fact)
            if episode_id
        )
    )
    packet["source_local_fact_sweep_trace"] = {
        **sweep_trace,
        "family": "codebase",
        "graph": graph_trace,
        "graph_support_fact_ids": [str(fact.get("id") or "") for fact in graph_support_facts],
    }
    return packet, retrieved_facts
