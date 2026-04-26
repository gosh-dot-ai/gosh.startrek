#!/usr/bin/env python3
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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..codebase_semantic_bundle import stable_semantic_id
from .base import relative_file_path
from .base import snippet_for_span as _snippet_for_span
from .base import span as _make_span


class _GoCollector:
    def __init__(self, *, repo_root: Path, repo_id: str, revision: str, file_path: Path):
        self.repo_root = repo_root
        self.repo_id = repo_id
        self.revision = revision
        self.file_path = file_path
        self.rel_path = relative_file_path(repo_root, file_path)
        self.language = "go"
        self.plugin_id = "go_static"
        self.plugin_version = "1"
        self.source = file_path.read_text(encoding="utf-8")
        self.lines = self.source.splitlines()
        self.objects: list[dict[str, Any]] = []
        self.relations: list[dict[str, Any]] = []
        self.sidecars: list[dict[str, Any]] = []
        self.relation_ids: set[str] = set()
        self.declarations_by_name: dict[str, str] = {}
        self.package_name = self._package_name()
        self.module_name = self.rel_path[:-3].replace("/", ".") if self.rel_path.endswith(".go") else self.rel_path.replace("/", ".")
        self.module_id = self._add_object(
            object_type="module",
            name=self.module_name,
            region=_make_span(1, max(1, len(self.lines))),
            payload={
                "name": self.module_name,
                "qualified_name": self.module_name,
                "package": self.package_name,
                "file_name": self.file_path.name,
                "language_family": "go",
            },
            add_sidecar=True,
        )

    def _package_name(self) -> str:
        for line in self.lines[:40]:
            match = re.match(r"\s*package\s+([A-Za-z_][A-Za-z0-9_]*)", line)
            if match:
                return match.group(1)
        return "main"

    def _object_id(self, object_type: str, name: str, region: dict[str, int | None]) -> str:
        return stable_semantic_id("obj", self.repo_id, self.revision, self.rel_path, object_type, name, region["start_line"], region["end_line"])

    def _relation_id(self, relation_type: str, from_id: str, to_id: str, region: dict[str, int | None]) -> str:
        return stable_semantic_id("rel", self.repo_id, self.revision, self.rel_path, relation_type, from_id, to_id, region["start_line"], region["end_line"])

    def _add_sidecar(self, node_id: str, node_kind: str, region: dict[str, int | None], payload: dict[str, Any]) -> None:
        sidecar_payload = {
            "fragment_id": stable_semantic_id("fragment", node_id),
            "root_node_id": node_id,
            "root_kind": node_kind,
            "file_path": self.rel_path,
            "span": region,
            "code": _snippet_for_span(self.source, region),
            **payload,
        }
        sidecar_id = stable_semantic_id("sidecar", self.repo_id, self.revision, self.rel_path, node_id, node_kind)
        self.sidecars.append(
            {
                "sidecar_id": sidecar_id,
                "sidecar_kind": "semantic_snapshot",
                "format_family": "static_code",
                "format_name": "go_static_fragment_v1",
                "format_version": "1",
                "encoding": "utf-8",
                "compression": "none",
                "repo_id": self.repo_id,
                "revision": self.revision,
                "file_path": self.rel_path,
                "span": region,
                "node_id": node_id,
                "storage_ref": f"inline:{sidecar_id}",
                "content_hash": stable_semantic_id("payload", sidecar_id, json.dumps(sidecar_payload, sort_keys=True, ensure_ascii=False)),
                "byte_size": len(json.dumps(sidecar_payload, ensure_ascii=False)),
                "producer": self.plugin_id,
                "metadata": {"language": self.language, "node_kind": node_kind},
                "payload": sidecar_payload,
            }
        )

    def _add_object(self, *, object_type: str, name: str, region: dict[str, int | None], payload: dict[str, Any], add_sidecar: bool = False) -> str:
        qualified_name = str(payload.get("qualified_name") or f"{self.package_name}.{name}")
        object_id = self._object_id(object_type, qualified_name, region)
        row_payload: dict[str, Any] = {"name": name, "qualified_name": qualified_name, **payload}
        row = {
            "id": object_id,
            "object_type": object_type,
            "repo_id": self.repo_id,
            "revision": self.revision,
            "file_path": self.rel_path,
            "span": region,
            "language": self.language,
            "analyzer_id": self.plugin_id,
            "analyzer_version": self.plugin_version,
            "derivation_type": "observed",
            "payload": row_payload,
        }
        self.objects.append(row)
        if object_type not in {"module", "import"}:
            self.declarations_by_name.setdefault(name, object_id)
        if add_sidecar:
            self._add_sidecar(object_id, object_type, region, row_payload)
        return object_id

    def _add_relation(self, relation_type: str, from_id: str, to_id: str, region: dict[str, int | None], payload: dict[str, Any] | None = None) -> None:
        rel_id = self._relation_id(relation_type, from_id, to_id, region)
        if rel_id in self.relation_ids:
            return
        self.relation_ids.add(rel_id)
        self.relations.append(
            {
                "id": rel_id,
                "relation_type": relation_type,
                "from_id": from_id,
                "to_id": to_id,
                "repo_id": self.repo_id,
                "revision": self.revision,
                "file_path": self.rel_path,
                "span": region,
                "language": self.language,
                "analyzer_id": self.plugin_id,
                "analyzer_version": self.plugin_version,
                "derivation_type": "resolved",
                "payload": dict(payload or {}),
            }
        )

    def _add_declared(self, object_type: str, name: str, line_no: int, payload: dict[str, Any] | None = None) -> str:
        region = _make_span(line_no, line_no)
        object_id = self._add_object(
            object_type=object_type,
            name=name,
            region=region,
            payload={"package": self.package_name, "language_family": "go", **(payload or {})},
            add_sidecar=True,
        )
        self._add_relation("declares", self.module_id, object_id, region, {"from_type": "module", "to_type": object_type})
        return object_id

    def _collect_imports(self) -> None:
        in_block = False
        for idx, line in enumerate(self.lines, start=1):
            stripped = line.strip()
            if stripped.startswith("import ("):
                in_block = True
                continue
            if in_block and stripped == ")":
                in_block = False
                continue
            candidates: list[str] = []
            if in_block:
                candidates.extend(re.findall(r'"([^"]+)"', stripped))
            else:
                match = re.match(r'\s*import\s+(?:[A-Za-z_][A-Za-z0-9_]*\s+|\.\s+|_\s+)?"([^"]+)"', line)
                if match:
                    candidates.append(match.group(1))
            for import_path in candidates:
                region = _make_span(idx, idx)
                import_id = self._add_object(
                    object_type="import",
                    name=import_path,
                    region=region,
                    payload={"import_path": import_path, "package": self.package_name, "language_family": "go"},
                )
                self._add_relation("imports", self.module_id, import_id, region, {"import_path": import_path})

    def collect(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        self._collect_imports()
        for idx, line in enumerate(self.lines, start=1):
            stripped = line.strip()
            type_match = re.match(r"type\s+([A-Za-z_][A-Za-z0-9_]*)\s+(struct|interface)\b", stripped)
            if type_match:
                object_type = "class" if type_match.group(2) == "struct" else "interface"
                self._add_declared(object_type, type_match.group(1), idx, {"go_type_kind": type_match.group(2)})
                continue
            alias_match = re.match(r"type\s+([A-Za-z_][A-Za-z0-9_]*)\b", stripped)
            if alias_match:
                self._add_declared("type", alias_match.group(1), idx, {"go_type_kind": "alias"})
                continue
            method_match = re.match(r"func\s+\((?P<receiver>[^)]+)\)\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\(", stripped)
            if method_match:
                receiver = method_match.group("receiver").strip()
                name = method_match.group("name")
                self._add_declared("method", name, idx, {"receiver": receiver})
                continue
            func_match = re.match(r"func\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", stripped)
            if func_match:
                name = func_match.group(1)
                object_type = "test_case" if self.file_path.name.endswith("_test.go") and name.startswith("Test") else "callable"
                object_id = self._add_declared(object_type, name, idx, {"test_case_name": name} if object_type == "test_case" else {})
                if object_type == "test_case" and name.startswith("Test"):
                    target_name = name[4:]
                    target_id = self.declarations_by_name.get(target_name)
                    if target_id:
                        self._add_relation("test_covers", object_id, target_id, _make_span(idx, idx), {"reason": "go_test_name_matches_target"})
        return self.objects, self.relations, self.sidecars


def _go_bundle(repo_root: Path, repo_id: str, revision: str, files: list[Path]) -> dict[str, Any]:
    objects: list[dict[str, Any]] = []
    relations: list[dict[str, Any]] = []
    sidecars: list[dict[str, Any]] = []
    notes: list[str] = []
    for file_path in sorted(files):
        try:
            collector = _GoCollector(repo_root=repo_root, repo_id=repo_id, revision=revision, file_path=file_path)
            file_objects, file_relations, file_sidecars = collector.collect()
        except UnicodeDecodeError:
            notes.append(f"go file skipped due to non-utf8 content: {relative_file_path(repo_root, file_path)}")
            continue
        objects.extend(file_objects)
        relations.extend(file_relations)
        sidecars.extend(file_sidecars)
    return {
        "objects": objects,
        "relations": relations,
        "provenance": {
            "repo_id": repo_id,
            "revision": revision,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_root": str(repo_root),
        },
        "capability_report": {
            "supported_languages": ["go"],
            "supported_capabilities": [
                "go_code_graph",
                "packages",
                "imports",
                "types",
                "interfaces",
                "methods",
                "functions",
                "tests",
            ],
            "analyzer_protocol_version": "codebase-v1",
            "schema_version": "1",
        },
        "gap_report": {"skipped_files": [], "notes": notes},
        "sidecars": sidecars,
    }


@dataclass(frozen=True)
class GoSemanticPlugin:
    plugin_name: str = "go_static"
    supported_extensions: frozenset[str] = frozenset({".go"})

    def supports_file(self, path: Path) -> bool:
        return path.suffix.lower() in self.supported_extensions

    def build_bundle(self, *, repo_root: Path, repo_id: str, revision: str, files: list[Path]) -> dict[str, Any]:
        return _go_bundle(repo_root, repo_id, revision, files)


GO_SEMANTIC_PLUGIN = GoSemanticPlugin()
SEMANTIC_PLUGIN = GO_SEMANTIC_PLUGIN
