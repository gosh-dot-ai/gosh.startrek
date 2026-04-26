# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

from __future__ import annotations

import json
import shutil
import subprocess
import textwrap
from types import ModuleType, SimpleNamespace
from pathlib import Path

import pytest

from src.codebase_semantic_bundle import normalize_semantic_bundle
from src.codebase_semantic_importer import import_semantic_bundle
from src.codebase_semantic_plugins import build_codebase_semantic_bundle
from src.codebase_semantic_plugins import registry as plugin_registry
from src.codebase_semantic_plugins.base import relative_file_path
from src.codebase_semantic_plugins.registry import get_codebase_semantic_plugins
from src.codebase_semantic_plugins.runner import discover_supported_codebase_files


def _valid_bundle() -> dict:
    return {
        "objects": [
            {
                "id": "obj_module",
                "object_type": "module",
                "repo_id": "repo_1",
                "revision": "rev_1",
                "file_path": "pkg/service.py",
                "span": {"start_line": 1, "end_line": 10, "start_col": 0, "end_col": None},
                "language": "python",
                "analyzer_id": "python_ast",
                "analyzer_version": "3.10",
                "derivation_type": "observed",
                "payload": {"name": "pkg.service", "qualified_name": "pkg.service"},
            },
            {
                "id": "obj_callable",
                "object_type": "callable",
                "repo_id": "repo_1",
                "revision": "rev_1",
                "file_path": "pkg/service.py",
                "span": {"start_line": 3, "end_line": 4, "start_col": 0, "end_col": None},
                "language": "python",
                "analyzer_id": "python_ast",
                "analyzer_version": "3.10",
                "derivation_type": "observed",
                "payload": {
                    "name": "issue",
                    "qualified_name": "pkg.service.issue",
                    "signature": "def issue(name: str) -> str",
                },
            },
        ],
        "relations": [
            {
                "id": "rel_declares",
                "relation_type": "declares",
                "from_id": "obj_module",
                "to_id": "obj_callable",
                "repo_id": "repo_1",
                "revision": "rev_1",
                "file_path": "pkg/service.py",
                "span": {"start_line": 3, "end_line": 4, "start_col": 0, "end_col": None},
                "language": "python",
                "analyzer_id": "python_ast",
                "analyzer_version": "3.10",
                "derivation_type": "observed",
                "payload": {"from_type": "module", "to_type": "callable"},
            }
        ],
        "provenance": {
            "repo_id": "repo_1",
            "revision": "rev_1",
            "generated_at": "2026-04-12T10:00:00Z",
            "source_root": "/tmp/repo",
        },
        "capability_report": {
            "supported_languages": ["python"],
            "supported_capabilities": ["module_graph"],
        },
        "gap_report": {"skipped_files": [], "notes": []},
        "sidecars": [
            {
                "sidecar_id": "sc_callable",
                "sidecar_kind": "ast",
                "format_family": "native_ast",
                "format_name": "python_ast_fragment_v1",
                "format_version": "1",
                "encoding": "utf-8",
                "compression": "none",
                "repo_id": "repo_1",
                "revision": "rev_1",
                "file_path": "pkg/service.py",
                "span": {"start_line": 3, "end_line": 4, "start_col": 0, "end_col": None},
                "node_id": "obj_callable",
                "storage_ref": "inline:sc_callable",
                "content_hash": "pending_hash",
                "byte_size": 12,
                "producer": "python_ast",
                "metadata": {"language": "python"},
                "payload": {
                    "fragment_id": "frag_1",
                    "root_node_id": "obj_callable",
                    "root_kind": "FunctionDef",
                    "file_path": "pkg/service.py",
                    "span": {"start_line": 3, "end_line": 4, "start_col": 0, "end_col": None},
                    "parent_node_id": "obj_module",
                    "child_node_ids": [],
                    "payload_ref": None,
                    "code": "def issue(name: str) -> str:\n    return name",
                },
            }
        ],
    }


def _create_python_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "python_repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "tests").mkdir(parents=True)
    (repo / "pkg" / "service.py").write_text(
        textwrap.dedent(
            """\
            import pathlib

            class Permit:
                def __init__(self, name: str):
                    self.name = name

            def audit(name: str) -> str:
                return name.upper()

            def issue(name: str) -> Permit:
                audit(name)
                return Permit(name)
            """
        ),
        encoding="utf-8",
    )
    (repo / "tests" / "test_service.py").write_text(
        textwrap.dedent(
            """\
            from pkg.service import issue

            def test_issue_creates_permit():
                permit = issue("x")
                assert permit.name == "x"
            """
        ),
        encoding="utf-8",
    )
    return repo


def _create_rust_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "rust_repo"
    (repo / "src").mkdir(parents=True)
    (repo / "Cargo.toml").write_text(
        textwrap.dedent(
            """\
            [package]
            name = "rust_probe"
            version = "0.1.0"
            edition = "2021"
            """
        ),
        encoding="utf-8",
    )
    (repo / "src" / "lib.rs").write_text(
        textwrap.dedent(
            """\
            pub struct Permit {
                pub id: String,
            }

            pub fn issue(name: &str) -> Permit {
                Permit { id: name.to_string() }
            }
            """
        ),
        encoding="utf-8",
    )
    return repo


def _create_js_ts_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "js_ts_repo"
    (repo / "src").mkdir(parents=True)
    (repo / "tests").mkdir(parents=True)
    (repo / "package.json").write_text(
        json.dumps({"scripts": {"test": "vitest run", "build": "tsc --noEmit"}}, indent=2),
        encoding="utf-8",
    )
    (repo / "src" / "Widget.tsx").write_text(
        textwrap.dedent(
            """\
            import React from 'react';
            import { client } from './api';

            export interface WidgetProps { name: string }
            export type WidgetState = { enabled: boolean };

            export function useWidget(name: string): WidgetState {
                return { enabled: Boolean(name) };
            }

            export class WidgetController {
                start() { return client.start(); }
            }

            export const WidgetView = (props: WidgetProps) => <div>{props.name}</div>;
            export { WidgetController as Controller };
            """
        ),
        encoding="utf-8",
    )
    (repo / "tests" / "widget.test.ts").write_text(
        textwrap.dedent(
            """\
            import { WidgetView } from '../src/Widget';

            test('renders widget', () => {
                expect(WidgetView).toBeDefined();
            });
            """
        ),
        encoding="utf-8",
    )
    return repo


def _create_go_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "go_repo"
    repo.mkdir()
    (repo / "go.mod").write_text("module example.com/demo\n\ngo 1.22\n", encoding="utf-8")
    (repo / "service.go").write_text(
        textwrap.dedent(
            """\
            package demo

            import (
                "context"
                "fmt"
            )

            type Store interface { Save(context.Context, string) error }
            type Server struct { store Store }

            func NewServer(store Store) *Server { return &Server{store: store} }
            func (s *Server) Save(ctx context.Context, value string) error {
                return s.store.Save(ctx, fmt.Sprint(value))
            }
            """
        ),
        encoding="utf-8",
    )
    (repo / "service_test.go").write_text(
        textwrap.dedent(
            """\
            package demo

            import "testing"

            func TestNewServer(t *testing.T) {
                if NewServer(nil) == nil { t.Fatal("nil") }
            }
            """
        ),
        encoding="utf-8",
    )
    return repo


def test_normalize_semantic_bundle_accepts_valid_bundle():
    normalized = normalize_semantic_bundle(_valid_bundle())

    assert len(normalized["objects"]) == 2
    assert len(normalized["relations"]) == 1
    assert len(normalized["sidecars"]) == 1
    assert normalized["objects"][1]["payload"]["qualified_name"] == "pkg.service.issue"
    assert normalized["sidecars"][0]["node_id"] == "obj_callable"


def test_plugin_registry_exposes_separate_language_plugins(tmp_path):
    repo = _create_python_repo(tmp_path)

    plugins = get_codebase_semantic_plugins()
    profile = discover_supported_codebase_files(repo)

    plugin_names = {plugin.plugin_name for plugin in plugins}
    assert {"python_ast", "rustdoc", "ecosystem_semantics", "js_ts_static", "go_static"} <= plugin_names
    supported_suffixes = {suffix for plugin in plugins for suffix in plugin.supported_extensions}
    assert {".py", ".rs", ".go", ".js", ".ts", ".tsx", ".json", ".toml", ".yaml", ".yml"} <= supported_suffixes
    assert {".py", ".rs", ".go", ".js", ".ts", ".tsx", ".json", ".toml", ".yaml", ".yml"} <= set(profile["supported_extensions"])


def test_plugin_registry_discovers_semantic_plugin_exports(monkeypatch):
    fake_plugin = SimpleNamespace(plugin_name="fake_lang", supported_extensions=frozenset({".fake"}))
    fake_module = ModuleType("src.codebase_semantic_plugins.fake_plugin")
    fake_module.SEMANTIC_PLUGIN = fake_plugin

    monkeypatch.setattr(
        plugin_registry,
        "iter_modules",
        lambda _paths: [
            SimpleNamespace(name="base"),
            SimpleNamespace(name="fake_plugin"),
        ],
    )
    monkeypatch.setattr(
        plugin_registry,
        "import_module",
        lambda name: fake_module if name.endswith(".fake_plugin") else ModuleType(name),
    )

    plugins = plugin_registry.get_codebase_semantic_plugins()

    assert plugins == [fake_plugin]


def test_normalize_semantic_bundle_rejects_malformed_sidecar():
    bundle = _valid_bundle()
    bundle["sidecars"][0]["format_name"] = ""

    with pytest.raises(ValueError, match="format_name"):
        normalize_semantic_bundle(bundle)


def test_import_semantic_bundle_persists_sidecars_and_builds_hot_facts(tmp_path):
    imported = import_semantic_bundle(_valid_bundle(), source_id="repo_source", data_dir=str(tmp_path))

    assert len(imported["facts"]) == 3
    assert imported["facts"][0]["source_family"] == "codebase"
    assert imported["facts"][0]["metadata"]["codebase"]["stage"] == "codebase_semantic"
    sidecars = imported["source_meta"]["codebase_context"]["semantic_sidecars"]
    assert len(sidecars) == 1
    storage_ref = sidecars[0]["storage_ref"]
    assert (tmp_path / storage_ref).exists()
    payload = json.loads((tmp_path / storage_ref).read_text(encoding="utf-8"))
    assert payload["root_node_id"] == "obj_callable"


def test_import_semantic_bundle_seed_policy_keeps_hot_scope_facts_only(tmp_path):
    imported = import_semantic_bundle(
        _valid_bundle(),
        source_id="repo_source",
        data_dir=str(tmp_path),
        hot_fact_policy="seed",
    )

    assert {fact["semantic_type"] for fact in imported["facts"]} == {"module", "callable"}
    assert all(fact["semantic_kind"] == "object" for fact in imported["facts"])
    assert all(fact["metadata"]["codebase"]["skip_embedding"] is False for fact in imported["facts"])


def test_python_plugin_builds_real_bundle_from_python_ast(tmp_path):
    repo = _create_python_repo(tmp_path)

    bundle, profile = build_codebase_semantic_bundle(repo)

    assert {path.name for path in profile["files"]} == {"service.py", "test_service.py"}
    object_types = {row["object_type"] for row in bundle["objects"]}
    relation_types = {row["relation_type"] for row in bundle["relations"]}
    qualified_names = {row["payload"].get("qualified_name") for row in bundle["objects"]}
    assert {"module", "class", "callable", "parameter", "import", "callsite", "test_case"} <= object_types
    assert {"declares", "imports", "calls", "test_covers"} <= relation_types
    assert "pkg.service.issue" in qualified_names
    assert "test_issue_creates_permit" in " ".join(str(name) for name in qualified_names)


def test_js_ts_plugin_builds_import_export_symbol_test_and_command_semantics(tmp_path):
    repo = _create_js_ts_repo(tmp_path)

    bundle, profile = build_codebase_semantic_bundle(repo)

    rel_paths = {relative_file_path(Path(profile["repo_root"]), path) for path in profile["files"]}
    assert {"src/Widget.tsx", "tests/widget.test.ts", "package.json"} <= rel_paths
    object_types = {row["object_type"] for row in bundle["objects"]}
    relation_types = {row["relation_type"] for row in bundle["relations"]}
    qualified_names = {str(row["payload"].get("qualified_name") or row["payload"].get("name") or "") for row in bundle["objects"]}
    roles = {row["payload"].get("symbol_role") for row in bundle["objects"]}
    capabilities = set(bundle["capability_report"].get("supported_capabilities") or [])

    assert {"module", "import", "export", "interface", "type", "class", "callable", "test_case", "command"} <= object_types
    assert {"declares", "imports", "exports", "command_targets"} <= relation_types
    assert any(name.endswith("WidgetView") for name in qualified_names)
    assert "component" in roles
    assert "hook" in roles
    assert "js_ts_code_graph" in capabilities


def test_go_plugin_builds_package_import_type_function_method_test_and_command_semantics(tmp_path):
    repo = _create_go_repo(tmp_path)

    bundle, profile = build_codebase_semantic_bundle(repo)

    rel_paths = {relative_file_path(Path(profile["repo_root"]), path) for path in profile["files"]}
    assert {"service.go", "service_test.go", "go.mod"} <= rel_paths
    object_types = {row["object_type"] for row in bundle["objects"]}
    relation_types = {row["relation_type"] for row in bundle["relations"]}
    names = {str(row["payload"].get("name") or "") for row in bundle["objects"]}
    capabilities = set(bundle["capability_report"].get("supported_capabilities") or [])

    assert {"module", "import", "interface", "class", "callable", "method", "test_case", "dependency_manifest", "command"} <= object_types
    assert {"declares", "imports", "command_targets"} <= relation_types
    assert {"Store", "Server", "NewServer", "Save", "TestNewServer"} <= names
    assert "go_code_graph" in capabilities


def test_python_plugin_deduplicates_repeated_relation_ids(tmp_path):
    repo = tmp_path / "duplicate_rel_repo"
    repo.mkdir()
    (repo / "module.py").write_text("from os import path, path\n", encoding="utf-8")

    bundle, profile = build_codebase_semantic_bundle(repo)

    assert {path.name for path in profile["files"]} == {"module.py"}
    relation_ids = [row["id"] for row in bundle["relations"]]
    assert len(relation_ids) == len(set(relation_ids))


def test_codebase_discovery_deduplicates_symlinked_canonical_paths(tmp_path):
    repo = tmp_path / "symlink_repo"
    (repo / "pkg").mkdir(parents=True)
    target = repo / "pkg" / "release.py"
    target.write_text("from os import path\n", encoding="utf-8")
    link = repo / "release_alias.py"
    try:
        link.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"symlink unavailable: {exc}")

    profile = discover_supported_codebase_files(repo)
    canonical_paths = [relative_file_path(Path(profile["repo_root"]), file_path) for file_path in profile["files"]]

    assert canonical_paths == ["pkg/release.py"]
    bundle, _ = build_codebase_semantic_bundle(repo)
    relation_ids = [row["id"] for row in bundle["relations"]]
    assert len(relation_ids) == len(set(relation_ids))


@pytest.mark.skipif(shutil.which("cargo") is None or shutil.which("rustdoc") is None, reason="Rust toolchain unavailable")
def test_rust_plugin_builds_real_bundle_from_rustdoc(tmp_path):
    repo = _create_rust_repo(tmp_path)

    bundle, profile = build_codebase_semantic_bundle(repo)

    assert {"lib.rs", "Cargo.toml"} <= {path.name for path in profile["files"]}
    object_types = {row["object_type"] for row in bundle["objects"]}
    relation_types = {row["relation_type"] for row in bundle["relations"]}
    qualified_names = {row["payload"].get("qualified_name") for row in bundle["objects"]}
    assert {"module", "class", "field", "callable"} <= object_types
    assert {"dependency_manifest", "command"} <= object_types
    assert {"declares"} <= relation_types
    assert {"command_targets"} <= relation_types
    assert "rust_probe::Permit" in qualified_names
    assert "rust_probe::issue" in qualified_names
