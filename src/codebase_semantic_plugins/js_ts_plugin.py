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

_JS_TS_SUFFIXES = frozenset({".js", ".jsx", ".mjs", ".ts", ".tsx"})


def _language_for_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".ts", ".tsx"}:
        return "typescript"
    return "javascript"


def _module_name(rel_path: str) -> str:
    stem = re.sub(r"\.[^.]+$", "", rel_path)
    return stem.replace("/", ".")


class _JsTsCollector:
    def __init__(self, *, repo_root: Path, repo_id: str, revision: str, file_path: Path):
        self.repo_root = repo_root
        self.repo_id = repo_id
        self.revision = revision
        self.file_path = file_path
        self.rel_path = relative_file_path(repo_root, file_path)
        self.language = _language_for_path(file_path)
        self.plugin_id = "js_ts_static"
        self.plugin_version = "1"
        self.source = file_path.read_text(encoding="utf-8")
        self.lines = self.source.splitlines()
        self.objects: list[dict[str, Any]] = []
        self.relations: list[dict[str, Any]] = []
        self.sidecars: list[dict[str, Any]] = []
        self.relation_ids: set[str] = set()
        self.declarations_by_name: dict[str, str] = {}
        self.module_name = _module_name(self.rel_path)
        self.module_id = self._add_object(
            object_type="module",
            name=self.module_name,
            region=_make_span(1, max(1, len(self.lines))),
            payload={
                "name": self.module_name,
                "qualified_name": self.module_name,
                "file_name": self.file_path.name,
                "language_family": "js_ts",
            },
            add_sidecar=True,
        )

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
                "format_name": "js_ts_static_fragment_v1",
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

    def _add_object(
        self,
        *,
        object_type: str,
        name: str,
        region: dict[str, int | None],
        payload: dict[str, Any],
        add_sidecar: bool = False,
    ) -> str:
        qualified_name = str(payload.get("qualified_name") or f"{self.module_name}.{name}")
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
        if object_type not in {"module", "import", "export"}:
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

    def _add_declared_object(self, object_type: str, name: str, line_no: int, exported: bool, payload: dict[str, Any] | None = None) -> str:
        region = _make_span(line_no, line_no)
        symbol_role = "symbol"
        if re.match(r"^use[A-Z]", name):
            symbol_role = "hook"
        elif object_type in {"callable", "class"} and name[:1].isupper() and self.file_path.suffix.lower() in {".jsx", ".tsx"}:
            symbol_role = "component"
        object_id = self._add_object(
            object_type=object_type,
            name=name,
            region=region,
            payload={
                "exported": exported,
                "symbol_role": symbol_role,
                "language_family": "js_ts",
                **(payload or {}),
            },
            add_sidecar=True,
        )
        self._add_relation("declares", self.module_id, object_id, region, {"from_type": "module", "to_type": object_type})
        if exported:
            self._add_export(name, line_no, declaration_id=object_id)
        return object_id

    def _add_import(self, import_path: str, line_no: int, imported_name: str = "") -> None:
        region = _make_span(line_no, line_no)
        name = imported_name or import_path
        import_id = self._add_object(
            object_type="import",
            name=name,
            region=region,
            payload={"import_path": import_path, "imported_name": imported_name, "language_family": "js_ts"},
        )
        self._add_relation("imports", self.module_id, import_id, region, {"import_path": import_path, "symbol": imported_name})

    def _add_export(self, exported_name: str, line_no: int, declaration_id: str = "") -> None:
        region = _make_span(line_no, line_no)
        export_id = self._add_object(
            object_type="export",
            name=exported_name,
            region=region,
            payload={"exported_name": exported_name, "declaration_id": declaration_id, "language_family": "js_ts"},
        )
        self._add_relation("exports", self.module_id, export_id, region, {"exported_name": exported_name, "declaration_id": declaration_id})

    def collect(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        for idx, line in enumerate(self.lines, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("//"):
                continue
            for match in re.finditer(r"\bimport\s+(?:type\s+)?(?:(?P<names>[^'\"]+?)\s+from\s+)?[\"'](?P<path>[^\"']+)[\"']", line):
                self._add_import(match.group("path"), idx, (match.group("names") or "").strip())
            for match in re.finditer(r"\brequire\(\s*[\"'](?P<path>[^\"']+)[\"']\s*\)", line):
                self._add_import(match.group("path"), idx)

            export_prefix = bool(re.match(r"^export\b", stripped))
            patterns = [
                ("class", r"^(?:export\s+default\s+|export\s+)?class\s+(?P<name>[A-Za-z_$][\w$]*)"),
                ("interface", r"^(?:export\s+)?interface\s+(?P<name>[A-Za-z_$][\w$]*)"),
                ("type", r"^(?:export\s+)?(?:type|enum)\s+(?P<name>[A-Za-z_$][\w$]*)"),
                ("callable", r"^(?:export\s+)?(?:async\s+)?function\s+(?P<name>[A-Za-z_$][\w$]*)"),
                ("callable", r"^(?:export\s+)?(?:const|let|var)\s+(?P<name>[A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*=>"),
            ]
            for object_type, pattern in patterns:
                decl_match = re.match(pattern, stripped)
                if decl_match:
                    self._add_declared_object(object_type, decl_match.group("name"), idx, export_prefix)
                    break

            named_export = re.match(r"^export\s*\{(?P<names>[^}]+)\}", stripped)
            if named_export:
                for raw_name in named_export.group("names").split(","):
                    name = raw_name.strip().split(" as ", 1)[-1].strip()
                    if name:
                        self._add_export(name, idx, declaration_id=self.declarations_by_name.get(name, ""))

            test_match = re.search(r"\b(?:it|test)\s*\(\s*[\"'](?P<name>[^\"']+)[\"']", line)
            if test_match:
                region = _make_span(idx, idx)
                test_name = test_match.group("name")
                test_id = self._add_object(
                    object_type="test_case",
                    name=test_name,
                    region=region,
                    payload={"test_case_name": test_name, "framework_hint": "js_test", "language_family": "js_ts"},
                    add_sidecar=True,
                )
                self._add_relation("declares", self.module_id, test_id, region, {"from_type": "module", "to_type": "test_case"})
        return self.objects, self.relations, self.sidecars


def _js_ts_bundle(repo_root: Path, repo_id: str, revision: str, files: list[Path]) -> dict[str, Any]:
    objects: list[dict[str, Any]] = []
    relations: list[dict[str, Any]] = []
    sidecars: list[dict[str, Any]] = []
    notes: list[str] = []
    for file_path in sorted(files):
        try:
            collector = _JsTsCollector(repo_root=repo_root, repo_id=repo_id, revision=revision, file_path=file_path)
            file_objects, file_relations, file_sidecars = collector.collect()
        except UnicodeDecodeError:
            notes.append(f"js/ts file skipped due to non-utf8 content: {relative_file_path(repo_root, file_path)}")
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
            "supported_languages": ["javascript", "typescript", "jsx", "tsx"],
            "supported_capabilities": [
                "js_ts_code_graph",
                "imports",
                "exports",
                "classes",
                "functions",
                "types",
                "components_hooks",
                "tests",
            ],
            "analyzer_protocol_version": "codebase-v1",
            "schema_version": "1",
        },
        "gap_report": {"skipped_files": [], "notes": notes},
        "sidecars": sidecars,
    }


@dataclass(frozen=True)
class JsTsSemanticPlugin:
    plugin_name: str = "js_ts_static"
    supported_extensions: frozenset[str] = _JS_TS_SUFFIXES

    def supports_file(self, path: Path) -> bool:
        return path.suffix.lower() in self.supported_extensions

    def build_bundle(self, *, repo_root: Path, repo_id: str, revision: str, files: list[Path]) -> dict[str, Any]:
        return _js_ts_bundle(repo_root, repo_id, revision, files)


JS_TS_SEMANTIC_PLUGIN = JsTsSemanticPlugin()
SEMANTIC_PLUGIN = JS_TS_SEMANTIC_PLUGIN
