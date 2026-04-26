#!/usr/bin/env python3
# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

from __future__ import annotations

import ast
import json
import sys
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..codebase_semantic_bundle import stable_semantic_id
from .base import relative_file_path
from .base import snippet_for_span as _snippet_for_span
from .base import span as _make_span


@dataclass
class PythonExtractionState:
    objects: list[dict[str, Any]]
    relations: list[dict[str, Any]]
    sidecars: list[dict[str, Any]]
    by_simple_name: dict[str, list[str]]
    by_qualified_name: dict[str, str]
    callable_contexts: list[dict[str, Any]]
    notes: list[str]


class _PythonCollector(ast.NodeVisitor):
    def __init__(self, *, repo_root: Path, repo_id: str, revision: str, file_path: Path):
        self.repo_root = repo_root
        self.repo_id = repo_id
        self.revision = revision
        self.file_path = file_path
        self.rel_path = relative_file_path(repo_root, file_path)
        self.language = "python"
        self.plugin_id = "python_ast"
        self.plugin_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        self.source = file_path.read_text(encoding="utf-8")
        self.lines = self.source.splitlines()
        self.module_name = self.rel_path[:-3].replace("/", ".") if self.rel_path.endswith(".py") else self.rel_path.replace("/", ".")
        self.objects: list[dict[str, Any]] = []
        self.relations: list[dict[str, Any]] = []
        self.relation_ids: set[str] = set()
        self.sidecars: list[dict[str, Any]] = []
        self.by_simple_name: dict[str, list[str]] = defaultdict(list)
        self.by_qualified_name: dict[str, str] = {}
        self.callable_contexts: list[dict[str, Any]] = []
        self.notes: list[str] = []
        self.name_stack: list[str] = []
        self.owner_stack: list[str] = []
        self.module_id = self._add_object(
            object_type="module",
            name=self.module_name,
            span=_make_span(1, max(1, len(self.lines))),
            payload={
                "name": self.module_name,
                "qualified_name": self.module_name,
                "file_name": self.file_path.name,
            },
            node=ast.parse(self.source),
            add_sidecar=True,
        )

    def _qualified_name(self, *parts: str) -> str:
        suffix = ".".join(part for part in parts if part)
        if not suffix:
            return self.module_name
        return f"{self.module_name}.{suffix}"

    def _object_id(self, object_type: str, name: str, span: dict[str, int | None]) -> str:
        return stable_semantic_id(
            "obj",
            self.repo_id,
            self.revision,
            self.rel_path,
            object_type,
            name,
            span["start_line"],
            span["end_line"],
        )

    def _relation_id(self, relation_type: str, from_id: str, to_id: str, span: dict[str, int | None]) -> str:
        return stable_semantic_id(
            "rel",
            self.repo_id,
            self.revision,
            self.rel_path,
            relation_type,
            from_id,
            to_id,
            span["start_line"],
            span["end_line"],
        )

    def _add_sidecar(self, node_id: str, node_kind: str, node: ast.AST, payload: dict[str, Any]) -> dict[str, Any]:
        sidecar_id = stable_semantic_id("sidecar", self.repo_id, self.revision, self.rel_path, node_id, node_kind)
        content_hash = stable_semantic_id("payload", sidecar_id, json.dumps(payload, sort_keys=True, ensure_ascii=False))
        sidecar = {
            "sidecar_id": sidecar_id,
            "sidecar_kind": "ast",
            "format_family": "native_ast",
            "format_name": "python_ast_fragment_v1",
            "format_version": "1",
            "encoding": "utf-8",
            "compression": "none",
            "repo_id": self.repo_id,
            "revision": self.revision,
            "file_path": self.rel_path,
            "span": payload["span"],
            "node_id": node_id,
            "storage_ref": f"inline:{sidecar_id}",
            "content_hash": content_hash,
            "byte_size": len(json.dumps(payload, ensure_ascii=False)),
            "producer": self.plugin_id,
            "metadata": {"language": self.language, "node_kind": node_kind},
            "payload": payload,
        }
        self.sidecars.append(sidecar)
        return sidecar

    def _register_name(self, object_id: str, name: str, qualified_name: str) -> None:
        self.by_simple_name[name].append(object_id)
        self.by_qualified_name[qualified_name] = object_id

    def _add_object(
        self,
        *,
        object_type: str,
        name: str,
        span: dict[str, int | None],
        payload: dict[str, Any],
        node: ast.AST | None = None,
        add_sidecar: bool = False,
    ) -> str:
        qualified_name = str(payload.get("qualified_name") or name)
        object_id = self._object_id(object_type, qualified_name, span)
        object_payload: dict[str, Any] = deepcopy(payload)
        object_row: dict[str, Any] = {
            "id": object_id,
            "object_type": object_type,
            "repo_id": self.repo_id,
            "revision": self.revision,
            "file_path": self.rel_path,
            "span": span,
            "language": self.language,
            "analyzer_id": self.plugin_id,
            "analyzer_version": self.plugin_version,
            "derivation_type": "observed",
            "payload": object_payload,
        }
        self.objects.append(object_row)
        self._register_name(object_id, name, qualified_name)
        if add_sidecar and node is not None:
            snippet = _snippet_for_span(self.source, span)
            sidecar = self._add_sidecar(
                object_id,
                node.__class__.__name__,
                node,
                {
                    "fragment_id": stable_semantic_id("fragment", object_id),
                    "root_node_id": object_id,
                    "root_kind": node.__class__.__name__,
                    "file_path": self.rel_path,
                    "span": span,
                    "parent_node_id": None,
                    "child_node_ids": [],
                    "payload_ref": None,
                    "code": snippet,
                    "ast_dump": ast.dump(node, include_attributes=True),
                },
            )
            payload_dict = object_row.get("payload")
            if isinstance(payload_dict, dict):
                payload_dict["sidecar_id"] = sidecar["sidecar_id"]
        return object_id

    def _add_relation(self, relation_type: str, from_id: str, to_id: str, span: dict[str, int | None], payload: dict[str, Any]) -> None:
        relation_id = self._relation_id(relation_type, from_id, to_id, span)
        if relation_id in self.relation_ids:
            return
        self.relation_ids.add(relation_id)
        self.relations.append(
            {
                "id": relation_id,
                "relation_type": relation_type,
                "from_id": from_id,
                "to_id": to_id,
                "repo_id": self.repo_id,
                "revision": self.revision,
                "file_path": self.rel_path,
                "span": span,
                "language": self.language,
                "analyzer_id": self.plugin_id,
                "analyzer_version": self.plugin_version,
                "derivation_type": "resolved",
                "payload": deepcopy(payload),
            }
        )

    def visit_Import(self, node: ast.Import) -> Any:
        for alias in node.names:
            import_name = alias.asname or alias.name
            span = _make_span(node.lineno, getattr(node, "end_lineno", node.lineno))
            object_id = self._add_object(
                object_type="import",
                name=import_name,
                span=span,
                payload={
                    "name": import_name,
                    "qualified_name": import_name,
                    "import_path": alias.name,
                    "alias": alias.asname,
                },
            )
            self._add_relation(
                "declares",
                self.module_id,
                object_id,
                span,
                {"from_type": "module", "to_type": "import"},
            )
            self._add_relation(
                "imports",
                self.module_id,
                object_id,
                span,
                {"import_path": alias.name},
            )
        return self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:
        module_name = node.module or ""
        for alias in node.names:
            import_name = alias.asname or alias.name
            span = _make_span(node.lineno, getattr(node, "end_lineno", node.lineno))
            object_id = self._add_object(
                object_type="import",
                name=import_name,
                span=span,
                payload={
                    "name": import_name,
                    "qualified_name": import_name,
                    "import_path": f"{module_name}.{alias.name}" if module_name else alias.name,
                    "alias": alias.asname,
                },
            )
            self._add_relation(
                "declares",
                self.module_id,
                object_id,
                span,
                {"from_type": "module", "to_type": "import"},
            )
            self._add_relation(
                "imports",
                self.module_id,
                object_id,
                span,
                {"import_path": module_name, "symbol": alias.name},
            )
        return self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        qualified_name = self._qualified_name(*self.name_stack, node.name)
        span = _make_span(node.lineno, getattr(node, "end_lineno", node.lineno))
        class_id = self._add_object(
            object_type="class",
            name=node.name,
            span=span,
            payload={
                "name": node.name,
                "qualified_name": qualified_name,
                "bases": [ast.unparse(base) if hasattr(ast, "unparse") else getattr(base, "id", "") for base in node.bases],
                "docstring": ast.get_docstring(node),
            },
            node=node,
            add_sidecar=True,
        )
        owner_id = self.owner_stack[-1] if self.owner_stack else self.module_id
        self._add_relation(
            "declares",
            owner_id,
            class_id,
            span,
            {"from_type": "module" if owner_id == self.module_id else "class", "to_type": "class"},
        )
        self.name_stack.append(node.name)
        self.owner_stack.append(class_id)
        self.generic_visit(node)
        self.owner_stack.pop()
        self.name_stack.pop()
        return None

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        return self._visit_callable(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        return self._visit_callable(node)

    def _visit_callable(self, node: ast.AST) -> Any:
        assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        qualified_name = self._qualified_name(*self.name_stack, node.name)
        span = _make_span(node.lineno, getattr(node, "end_lineno", node.lineno))
        is_test_case = node.name.startswith("test_") or self.file_path.name.startswith("test_")
        object_type = "test_case" if is_test_case else "callable"
        signature = ast.get_source_segment(self.source, node)
        if not signature:
            signature = node.name
        signature_line = signature.splitlines()[0].strip()
        object_id = self._add_object(
            object_type=object_type,
            name=node.name,
            span=span,
            payload={
                "name": node.name,
                "qualified_name": qualified_name,
                "signature": signature_line,
                "docstring": ast.get_docstring(node),
                "async": isinstance(node, ast.AsyncFunctionDef),
            },
            node=node,
            add_sidecar=True,
        )
        owner_id = self.owner_stack[-1] if self.owner_stack else self.module_id
        self._add_relation(
            "declares",
            owner_id,
            object_id,
            span,
            {"from_type": "module" if owner_id == self.module_id else "class", "to_type": object_type},
        )

        for index, arg in enumerate(node.args.args):
            param_span = _make_span(
                getattr(arg, "lineno", node.lineno),
                getattr(arg, "end_lineno", getattr(arg, "lineno", node.lineno)),
            )
            param_id = self._add_object(
                object_type="parameter",
                name=arg.arg,
                span=param_span,
                payload={
                    "name": arg.arg,
                    "qualified_name": f"{qualified_name}.{arg.arg}",
                    "position": index,
                    "callable_id": object_id,
                },
            )
            self._add_relation(
                "declares",
                object_id,
                param_id,
                param_span,
                {"from_type": object_type, "to_type": "parameter"},
            )

        self.callable_contexts.append(
            {
                "callable_id": object_id,
                "callable_name": node.name,
                "qualified_name": qualified_name,
                "object_type": object_type,
                "node": node,
            }
        )
        self.name_stack.append(node.name)
        self.owner_stack.append(object_id)
        self.generic_visit(node)
        self.owner_stack.pop()
        self.name_stack.pop()
        return None


def _callee_name_from_ast(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parts = []
        cursor: ast.AST | None = node
        while isinstance(cursor, ast.Attribute):
            parts.append(cursor.attr)
            cursor = cursor.value
        if isinstance(cursor, ast.Name):
            parts.append(cursor.id)
        parts.reverse()
        return ".".join(parts)
    return None


def _python_plugin_bundle(repo_root: Path, repo_id: str, revision: str, files: list[Path]) -> dict[str, Any]:
    state = PythonExtractionState(objects=[], relations=[], sidecars=[], by_simple_name=defaultdict(list), by_qualified_name={}, callable_contexts=[], notes=[])
    collectors: list[_PythonCollector] = []
    for file_path in files:
        collector = _PythonCollector(repo_root=repo_root, repo_id=repo_id, revision=revision, file_path=file_path)
        collector.visit(ast.parse(collector.source))
        collectors.append(collector)
        state.objects.extend(collector.objects)
        state.relations.extend(collector.relations)
        state.sidecars.extend(collector.sidecars)
        state.notes.extend(collector.notes)
        for name, ids in collector.by_simple_name.items():
            state.by_simple_name[name].extend(ids)
        state.by_qualified_name.update(collector.by_qualified_name)
        state.callable_contexts.extend(collector.callable_contexts)

    object_lookup = {row["id"]: row for row in state.objects}
    seen_relation_ids = {row["id"] for row in state.relations}
    for collector in collectors:
        for context in collector.callable_contexts:
            node = context["node"]
            caller_id = context["callable_id"]
            object_type = context["object_type"]
            local_test_targets: set[str] = set()
            for child in ast.walk(node):
                if not isinstance(child, ast.Call):
                    continue
                callee_name = _callee_name_from_ast(child.func)
                if not callee_name:
                    continue
                span = _make_span(
                    child.lineno,
                    getattr(child, "end_lineno", child.lineno),
                    getattr(child, "col_offset", None),
                    getattr(child, "end_col_offset", None),
                )
                callsite_id = stable_semantic_id(
                    "obj",
                    repo_id,
                    revision,
                    collector.rel_path,
                    "callsite",
                    context["qualified_name"],
                    callee_name,
                    child.lineno,
                    getattr(child, "col_offset", 0),
                )
                if callsite_id not in object_lookup:
                    callsite_row = {
                        "id": callsite_id,
                        "object_type": "callsite",
                        "repo_id": repo_id,
                        "revision": revision,
                        "file_path": collector.rel_path,
                        "span": span,
                        "language": "python",
                        "analyzer_id": collector.plugin_id,
                        "analyzer_version": collector.plugin_version,
                        "derivation_type": "observed",
                        "payload": {
                            "name": callee_name,
                            "qualified_name": f"callsite::{callee_name}@{collector.rel_path}:{child.lineno}",
                            "callee_name": callee_name,
                            "caller_id": caller_id,
                            "caller_qualified_name": context["qualified_name"],
                        },
                    }
                    state.objects.append(callsite_row)
                    object_lookup[callsite_id] = callsite_row
                    relation_id = stable_semantic_id("rel", repo_id, revision, collector.rel_path, "declares", caller_id, callsite_id, child.lineno)
                    if relation_id not in seen_relation_ids:
                        state.relations.append(
                            {
                                "id": relation_id,
                                "relation_type": "declares",
                                "from_id": caller_id,
                                "to_id": callsite_id,
                                "repo_id": repo_id,
                                "revision": revision,
                                "file_path": collector.rel_path,
                                "span": span,
                                "language": "python",
                                "analyzer_id": collector.plugin_id,
                                "analyzer_version": collector.plugin_version,
                                "derivation_type": "observed",
                                "payload": {"from_type": object_type, "to_type": "callsite"},
                            }
                        )
                        seen_relation_ids.add(relation_id)
                target_ids = state.by_qualified_name.get(callee_name)
                if target_ids:
                    resolved_ids = [target_ids]
                else:
                    resolved_ids = state.by_simple_name.get(callee_name.split(".")[-1], [])
                for target_id in resolved_ids:
                    relation = {
                        "id": stable_semantic_id("rel", repo_id, revision, collector.rel_path, "calls", caller_id, target_id, child.lineno),
                        "relation_type": "calls",
                        "from_id": caller_id,
                        "to_id": target_id,
                        "repo_id": repo_id,
                        "revision": revision,
                        "file_path": collector.rel_path,
                        "span": span,
                        "language": "python",
                        "analyzer_id": collector.plugin_id,
                        "analyzer_version": collector.plugin_version,
                        "derivation_type": "resolved",
                        "payload": {"callsite_id": callsite_id, "callee_name": callee_name},
                    }
                    if relation["id"] not in seen_relation_ids:
                        state.relations.append(relation)
                        seen_relation_ids.add(relation["id"])
                    if object_type == "test_case":
                        local_test_targets.add(target_id)
            if object_type == "test_case":
                for target_id in sorted(local_test_targets):
                    relation = {
                        "id": stable_semantic_id("rel", repo_id, revision, collector.rel_path, "test_covers", caller_id, target_id),
                        "relation_type": "test_covers",
                        "from_id": caller_id,
                        "to_id": target_id,
                        "repo_id": repo_id,
                        "revision": revision,
                        "file_path": collector.rel_path,
                        "span": _make_span(node.lineno, getattr(node, "end_lineno", node.lineno)),
                        "language": "python",
                        "analyzer_id": collector.plugin_id,
                        "analyzer_version": collector.plugin_version,
                        "derivation_type": "approximate",
                        "payload": {"reason": "test_case_calls_target"},
                    }
                    if relation["id"] not in seen_relation_ids:
                        state.relations.append(relation)
                        seen_relation_ids.add(relation["id"])

    return {
        "objects": state.objects,
        "relations": state.relations,
        "provenance": {
            "repo_id": repo_id,
            "revision": revision,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_root": str(repo_root),
        },
        "capability_report": {
            "supported_languages": ["python"],
            "supported_capabilities": ["module_graph", "structural_semantics", "call_graph", "tests"],
            "analyzer_protocol_version": "codebase-v1",
            "schema_version": "1",
        },
        "gap_report": {
            "skipped_files": [],
            "notes": state.notes,
        },
        "sidecars": state.sidecars,
    }

@dataclass(frozen=True)
class PythonSemanticPlugin:
    plugin_name: str = "python_ast"
    supported_extensions: frozenset[str] = frozenset({".py"})

    def build_bundle(
        self,
        *,
        repo_root: Path,
        repo_id: str,
        revision: str,
        files: list[Path],
    ) -> dict[str, Any]:
        return _python_plugin_bundle(repo_root, repo_id, revision, files)


PYTHON_SEMANTIC_PLUGIN = PythonSemanticPlugin()
SEMANTIC_PLUGIN = PYTHON_SEMANTIC_PLUGIN
