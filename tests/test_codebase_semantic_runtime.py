# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

from __future__ import annotations

import shutil
import textwrap
import json
import re
from pathlib import Path

import numpy as np
import pytest

from src.codebase_semantic_plugins.python_plugin import PYTHON_SEMANTIC_PLUGIN
import src.codebase_query as codebase_query
from src.codebase_container_graph import build_codebase_container_graph
from src.codebase_query import (
    build_codebase_context,
    _codebase_config_applicability_checklist,
    _codebase_config_type_validation_evidence,
    _codebase_line_windows,
    _codebase_query_terms,
    _codebase_semantic_config_option_entries,
    _codebase_semantic_config_option_windows,
    extract_repo_work_item_path_constraints,
)
from src.codebase_semantic_plugins.runner import build_codebase_semantic_bundle
from src.codebase_semantic_sidecars import CodebaseSemanticSidecarStore
from src.memory import MemoryServer


def _semantic_vec(text: str, *, dim: int = 24) -> np.ndarray:
    lowered = text.lower()
    keys = [
        "permit",
        "issue",
        "audit",
        "test",
        "call",
        "signature",
        "code",
        "field",
        "class",
        "bundle",
        "python",
        "window",
        "cinder",
        "file",
    ]
    vec = np.zeros(dim, dtype=np.float32)
    for idx, key in enumerate(keys):
        if key in lowered:
            vec[idx] = 1.0
    if vec.sum() == 0.0:
        vec[-1] = 1.0
    return vec


def test_codebase_runtime_source_has_no_benchmark_repo_literals():
    src_root = Path(__file__).resolve().parents[1] / "src"
    forbidden = {
        "qutebrowser",
        "swe-pro",
        "swe_bench",
        "mrcr-",
        "2needle",
        "4needle",
        "8needle",
        "EXPECTED_ANSWERS_READ",
    }
    offenders = []
    for path in sorted(src_root.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for literal in forbidden:
            if literal in text:
                offenders.append(f"{path.relative_to(src_root)}:{literal}")
    assert offenders == []


def test_codebase_runtime_has_no_legacy_phase_names():
    root = Path(__file__).resolve().parents[1]
    forbidden = ["Sta" + "ge2", "sta" + "ge2", "STA" + "GE2"]
    checked_suffixes = {".py", ".md", ".toml", ".yaml", ".yml", ".json"}
    hits = []
    for base in (root / "src", root / "tests"):
        for path in base.rglob("*"):
            if not path.is_file() or path.suffix not in checked_suffixes:
                continue
            content = path.read_text(errors="ignore")
            for token in forbidden:
                if token in content:
                    hits.append(str(path.relative_to(root)))
                    break
    assert hits == []


async def _mock_embed_texts(texts, **kwargs):
    return np.stack([_semantic_vec(text) for text in texts]).astype(np.float32)


async def _mock_embed_query(text, **kwargs):
    return _semantic_vec(text).astype(np.float32)


def _create_python_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "python_repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "tests").mkdir(parents=True)
    (repo / "pkg" / "service.py").write_text(
        textwrap.dedent(
            """\
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


def _create_crowded_symbol_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "crowded_repo"
    target_dir = repo / "src" / "codebase_semantic_plugins"
    target_dir.mkdir(parents=True)
    (repo / "pyproject.toml").write_text("[project]\nname='crowded_repo'\n", encoding="utf-8")
    lines = [
        "def helper(value: str) -> str:\n",
        "    return value.upper()\n\n",
        "def build_codebase_semantic_bundle(path: str) -> str:\n",
        "    helper(path)\n",
        "    helper(path + '-a')\n",
        "    helper(path + '-b')\n",
        "    return f'bundle:{path}'\n\n",
    ]
    for idx in range(20):
        lines.extend(
            [
                f"def consumer_{idx:02d}() -> str:\n",
                f"    return build_codebase_semantic_bundle('case-{idx:02d}')\n\n",
            ]
        )
    (target_dir / "runner.py").write_text("".join(lines), encoding="utf-8")
    return repo


def _create_cross_file_symbol_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "cross_file_symbol_repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "callers").mkdir(parents=True)
    (repo / "pyproject.toml").write_text("[project]\nname='cross_file_symbol_repo'\n", encoding="utf-8")
    (repo / "pkg" / "runner.py").write_text(
        textwrap.dedent(
            """\
            def helper(value: str) -> str:
                return value.upper()

            def build_codebase_semantic_bundle(path: str) -> str:
                helper(path)
                helper(path + "-a")
                helper(path + "-b")
                return f"bundle:{path}"
            """
        ),
        encoding="utf-8",
    )
    for idx in range(20):
        (repo / "callers" / f"consumer_{idx:02d}.py").write_text(
            textwrap.dedent(
                f"""\
                from pkg.runner import build_codebase_semantic_bundle

                def consume_{idx:02d}() -> str:
                    return build_codebase_semantic_bundle("case-{idx:02d}")
                """
            ),
            encoding="utf-8",
        )
    return repo


def _create_large_python_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "large_python_repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "pyproject.toml").write_text("[project]\nname='large_python_repo'\n", encoding="utf-8")
    lines = [f"# filler {idx}\n" for idx in range(1, 1851)]
    lines.extend(
        [
            "\n",
            "def cinder_signature() -> str:\n",
            "    return \"sig:CINDER-42\"\n",
            "\n",
        ]
    )
    lines.extend(f"# tail filler {idx}\n" for idx in range(1851, 2206))
    (repo / "pkg" / "huge_service.py").write_text("".join(lines), encoding="utf-8")
    return repo


def _create_font_symbol_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "font_symbol_repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "pyproject.toml").write_text("[project]\nname='font_symbol_repo'\n", encoding="utf-8")
    font_body = [
        "class Font:\n",
        "    def __init__(self, value: str):\n",
        "        self.value = value\n",
        "        self.default_family = None\n",
        "\n",
        "    def set_default_family(self, default_family):\n",
        "        family_candidates = list(default_family)\n",
    ]
    for idx in range(60):
        font_body.append(f"        marker_{idx} = {idx}\n")
        if idx == 30:
            font_body.append("        typed_span_only_bridge = marker_30\n")
    font_body.extend(
        [
            "        resolved = family_candidates[0]\n",
            "        self.default_family = resolved\n",
            "        return resolved\n",
            "\n",
            "class QtFont(Font):\n",
            "    def to_py(self):\n",
            "        return f\"font:{self.default_family}:{self.value}\"\n",
        ]
    )
    (repo / "pkg" / "fonts.py").write_text("".join(font_body), encoding="utf-8")
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


def _create_multi_manifest_rust_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "multi_rust_repo"
    for crate_name in ("good_crate", "broken_crate"):
        crate_root = repo / "crates" / crate_name
        (crate_root / "src").mkdir(parents=True)
        (crate_root / "Cargo.toml").write_text(
            textwrap.dedent(
                f"""\
                [package]
                name = "{crate_name}"
                version = "0.1.0"
                edition = "2021"
                """
            ),
            encoding="utf-8",
        )
        (crate_root / "src" / "lib.rs").write_text(
            textwrap.dedent(
                f"""\
                pub fn issue() -> &'static str {{
                    "{crate_name}"
                }}
                """
            ),
            encoding="utf-8",
        )
    return repo


def _make_server(tmp_path: Path, key: str = "codebase_runtime") -> MemoryServer:
    server = MemoryServer(str(tmp_path / "data"), key)
    server._embed_texts_with_runtime_secrets = _mock_embed_texts
    server._embed_query_with_runtime_secrets = _mock_embed_query
    return server


def _active_codebase_facts(server: MemoryServer, source_id: str) -> list[dict]:
    return [
        fact
        for fact in server._all_granular
        if fact.get("source_id") == source_id
        and str(fact.get("source_family") or "").lower() == "codebase"
        and str(fact.get("status") or "active") == "active"
    ]


def _superseded_codebase_facts(server: MemoryServer, source_id: str) -> list[dict]:
    return [
        fact
        for fact in server._all_granular
        if fact.get("source_id") == source_id
        and str(fact.get("source_family") or "").lower() == "codebase"
        and str(fact.get("status") or "") == "superseded"
    ]


@pytest.mark.asyncio
async def test_ingest_codebase_persists_codebase_source_meta_and_reload(tmp_path):
    repo = _create_python_repo(tmp_path)
    server = _make_server(tmp_path)

    result = await server.ingest_codebase(str(repo), source_id="repo", scope="agent-private")

    assert result["status"] == "ok"
    assert result["source_family"] == "codebase"
    assert server._source_records["repo"]["family"] == "codebase"
    stage_meta = server._source_records["repo"]["source_meta"]["codebase_context"]
    assert stage_meta["object_count"] > 0
    assert stage_meta["relation_count"] > 0
    assert stage_meta["sidecar_count"] > 0
    reloaded = MemoryServer(str(tmp_path / "data"), "codebase_runtime")
    reloaded_stage_meta = reloaded._source_records["repo"]["source_meta"]["codebase_context"]
    assert reloaded_stage_meta["object_count"] == stage_meta["object_count"]
    assert reloaded_stage_meta["semantic_sidecars"]


@pytest.mark.asyncio
async def test_ingest_codebase_skips_legacy_stage1_when_git_binary_unavailable(tmp_path, monkeypatch):
    repo = _create_python_repo(tmp_path)
    (repo / ".git").write_text("gitdir: /unavailable/git/worktree\n", encoding="utf-8")
    server = _make_server(tmp_path, key="codebase_no_git_runtime")
    original_which = shutil.which
    monkeypatch.setattr("src.memory.shutil.which", lambda name: None if name == "git" else original_which(name))

    def _fail_stage1(**_kwargs):
        raise AssertionError("legacy git Stage 1 must not run when git is unavailable")

    monkeypatch.setattr("src.memory.build_codebase_stage1_bundle", _fail_stage1)

    result = await server.ingest_codebase(str(repo), source_id="repo", scope="agent-private")

    assert result["status"] == "ok"
    assert result["codebase_stage"] == "codebase_semantic"
    assert result["codebase_stage1_legacy"] == {
        "status": "skipped",
        "reason": "git_unavailable_for_legacy_stage1",
    }
    source_meta = server._source_records["repo"]["source_meta"]
    assert source_meta["codebase_stage1_legacy"]["status"] == "skipped"
    assert source_meta["codebase_context"]["object_count"] > 0


@pytest.mark.asyncio
async def test_ingest_codebase_materializes_work_item_metadata_into_context_pack(tmp_path):
    repo = _create_python_repo(tmp_path)
    server = _make_server(tmp_path, key="codebase_work_item_runtime")

    await server.ingest_codebase(
        str(repo),
        source_id="repo",
        scope="agent-private",
        metadata={
            "work_item_work_item_id": "issue-123",
            "work_item_work_item_kind": "repo:bug_report",
            "work_item_problem_statement": "Audit events are missing permit names.",
            "work_item_requirements": "Patch the service and preserve tests.",
            "work_item_selected_test_files_to_run": "tests/test_service.py",
        },
    )

    work_item = next(row for row in server._container_graph["containers"] if row["kind_fq"] == "repo:work_item")
    assert work_item["traits_json"]["work_item"]["work_item_id"] == "issue-123"
    assert work_item["traits_json"]["work_item_kind"] == "repo:bug_report"

    context_pack = next(
        row["payload_json"]
        for row in server._container_graph["artifacts"]
        if row["artifact_kind"] == "context_pack"
    )
    requirements = context_pack["sections"]["requirements"]
    assert any(item["kind"] == "problem_statement" for item in requirements)
    assert any("tests/test_service.py" in item["text"] for item in requirements)
    assert context_pack["work_item"]["work_item_kind"] == "repo:bug_report"


@pytest.mark.asyncio
async def test_ingest_codebase_materializes_codebase_container_graph_contracts_and_artifacts(tmp_path):
    repo = _create_python_repo(tmp_path)
    server = _make_server(tmp_path, key="codebase_container_graph_runtime")

    result = await server.ingest_codebase(str(repo), source_id="repo", scope="agent-private")

    assert result["container_graph_status"] == "active"
    graph = server._container_graph
    revision = next(
        row
        for row in graph["graph_revisions"]
        if row["source_id"] == "repo" and row["adapter_name"] == "codebase_semantic_container_graph"
    )
    assert revision["status"] == "active"
    assert revision["family"] == "codebase"
    kind_fqs = {row["kind_fq"] for row in graph["containers"] if row["container_graph_revision_id"] == revision["container_graph_revision_id"]}
    assert {"repo:repository", "repo:work_item", "code:file", "code:function", "code:class", "code:test_case"} <= kind_fqs
    ref_types = {row["ref_type"] for row in graph["refs"] if row["container_graph_revision_id"] == revision["container_graph_revision_id"]}
    assert {"file_line_range", "file_char_range", "lsp_symbol", "ast_node", "test_case"} <= ref_types
    lookup_keys = {(row["lookup_ns"], row["lookup_key"]) for row in graph["ref_lookup"]}
    assert ("file_line_range", "repo_id") in lookup_keys
    assert ("file_line_range", "path") in lookup_keys
    assert ("lsp_symbol", "fully_qualified_name") in lookup_keys
    assert ("ast_node", "ast_node_id") in lookup_keys
    render_refs = [
        row
        for row in graph["render_refs"]
        if row["container_graph_revision_id"] == revision["container_graph_revision_id"]
    ]
    assert render_refs
    assert {row["ref_json"]["render_source"] for row in render_refs} == {"repo_file_blob_span"}
    assert all(row["ref_json"]["text_exact"] is True for row in render_refs)
    contract_kinds = {row["contract_kind"] for row in graph["contracts"]}
    assert {"family_profile_contract", "relation_capability_contract", "render_contract", "coverage_policy_contract"} <= contract_kinds
    artifacts = [
        row
        for row in graph["artifacts"]
        if row["container_graph_revision_id"] == revision["container_graph_revision_id"]
    ]
    artifact_kinds = {row["artifact_kind"] for row in artifacts}
    assert {"coverage_report", "validator_report", "context_pack"} <= artifact_kinds
    context_pack = next(row for row in artifacts if row["artifact_kind"] == "context_pack")
    assert context_pack["payload_json"]["context_pack_kind"] == "repo_task_context_pack"
    assert context_pack["payload_json"]["sections"]["semantic_code_candidates"]
    assert context_pack["payload_json"]["sections"]["tests"]
    assert context_pack["payload_json"]["trace"]["operator_domain_policy"] == "full_typed_container_domain_in_scope"
    reloaded = MemoryServer(str(tmp_path / "data"), "codebase_container_graph_runtime")
    reloaded_revision = next(
        row
        for row in reloaded._container_graph["graph_revisions"]
        if row["container_graph_revision_id"] == revision["container_graph_revision_id"]
    )
    assert reloaded_revision["status"] == "active"
    assert any(row["artifact_kind"] == "context_pack" for row in reloaded._container_graph["artifacts"])


@pytest.mark.asyncio
async def test_codebase_exact_render_uses_original_file_not_sidecar_snippet(tmp_path):
    repo = tmp_path / "render_repo"
    (repo / "pkg").mkdir(parents=True)
    source_text = "def exact_space():\n    return 'kept'    \n"
    (repo / "pkg" / "service.py").write_text(source_text, encoding="utf-8")
    server = _make_server(tmp_path, key="codebase_render_runtime")

    await server.ingest_codebase(str(repo), source_id="repo", scope="agent-private")

    render_refs = [
        row
        for row in server._container_graph["render_refs"]
        if row["ref_json"].get("path") == "pkg/service.py"
        and row["ref_json"].get("span", {}).get("start_line") == 1
        and row["ref_json"].get("span", {}).get("end_line") == 2
    ]
    assert render_refs
    rendered_texts = {row["ref_json"]["text"] for row in render_refs}
    assert source_text.rstrip("\n") in rendered_texts
    assert all(row["ref_json"]["raw_source_provenance"] == "repo_worktree_original_file" for row in render_refs)
    assert all(row["ref_json"]["text"].endswith("    ") for row in render_refs)


def test_codebase_graph_revision_fails_closed_when_required_render_missing(tmp_path):
    repo = _create_python_repo(tmp_path)
    bundle, _profile = build_codebase_semantic_bundle(repo)
    source_record = {
        "source_id": "repo",
        "family": "codebase",
        "source_meta": {"logical_source_id": "repo"},
        "read": [],
        "write": [],
    }

    graph = build_codebase_container_graph(
        bundle,
        source_id="repo",
        source_record=source_record,
        repo_root=tmp_path / "missing_repo_root",
    )

    revision = next(row for row in graph["graph_revisions"] if row["adapter_name"] == "codebase_semantic_container_graph")
    assert revision["status"] == "failed_closed"
    validator = next(row for row in graph["artifacts"] if row["artifact_kind"] == "validator_report")
    assert validator["payload_json"]["ok"] is False
    assert any(error["code"] == "MISSING_EXACT_RENDER_REFS" for error in validator["payload_json"]["errors"])


@pytest.mark.asyncio
async def test_ingest_codebase_reports_unsupported_language_gap_without_silent_success(tmp_path):
    repo = _create_python_repo(tmp_path)
    (repo / "src").mkdir(exist_ok=True)
    (repo / "src" / "Widget.vue").write_text("<template><div /></template>\n", encoding="utf-8")
    server = _make_server(tmp_path, key="codebase_gap_runtime")

    await server.ingest_codebase(str(repo), source_id="repo", scope="agent-private")

    gap_report = server._source_records["repo"]["source_meta"]["codebase_context"]["gap_report"]
    assert "src/Widget.vue" in gap_report["skipped_files"]
    assert any("unsupported semantic plugin file: src/Widget.vue" in note for note in gap_report["notes"])
    coverage = next(
        row["payload_json"]
        for row in server._container_graph["artifacts"]
        if row["artifact_kind"] == "coverage_report"
    )
    assert coverage["gap_coverage"]["unsupported_file_count"] >= 1
    assert coverage["gap_coverage"]["status"] == "explicitly_reported"
    assert coverage["silent_absence_count"] == 0


@pytest.mark.asyncio
async def test_ingest_codebase_missing_path_fails_cleanly(tmp_path):
    server = _make_server(tmp_path, key="missing_codebase_runtime")
    missing = tmp_path / "missing_repo"

    with pytest.raises(ValueError, match="codebase path does not exist"):
        await server.ingest_codebase(str(missing), source_id="repo", scope="agent-private")

    assert server._all_granular == []
    assert server._source_records == {}
    assert not (tmp_path / "data" / "codebase_semantic_sidecars").exists()


@pytest.mark.asyncio
async def test_ingest_codebase_unsupported_repo_fails_without_partial_persistence(tmp_path):
    repo = tmp_path / "unsupported_repo"
    repo.mkdir()
    (repo / "README.txt").write_text("notes only\n", encoding="utf-8")
    server = _make_server(tmp_path, key="unsupported_codebase_runtime")

    with pytest.raises(ValueError, match="supports only Python and Rust sources"):
        await server.ingest_codebase(str(repo), source_id="repo", scope="agent-private")

    assert server._all_granular == []
    assert server._source_records == {}
    assert not (tmp_path / "data" / "codebase_semantic_sidecars").exists()


@pytest.mark.asyncio
async def test_ingest_codebase_duplicate_repo_returns_duplicate_without_new_active_facts(tmp_path):
    repo = _create_python_repo(tmp_path)
    server = _make_server(tmp_path, key="duplicate_codebase_runtime")

    first = await server.ingest_codebase(str(repo), source_id="repo", scope="agent-private")
    active_before = list(_active_codebase_facts(server, "repo"))
    version_before = server._source_records["repo"]["version_id"]

    second = await server.ingest_codebase(str(repo), source_id="repo", scope="agent-private")

    assert first["status"] == "ok"
    assert second["status"] == "duplicate"
    assert second["duplicate_of"]["source_id"] == "repo"
    assert len(_active_codebase_facts(server, "repo")) == len(active_before)
    assert server._source_records["repo"]["version_id"] == version_before


@pytest.mark.asyncio
async def test_ingest_codebase_changed_repo_supersedes_old_active_facts(tmp_path):
    repo = _create_python_repo(tmp_path)
    server = _make_server(tmp_path, key="supersede_codebase_runtime")

    first = await server.ingest_codebase(str(repo), source_id="repo", scope="agent-private")
    version_before = server._source_records["repo"]["version_id"]

    service_path = repo / "pkg" / "service.py"
    service_path.write_text(
        service_path.read_text(encoding="utf-8")
        + "\n\ndef revoke(name: str) -> str:\n    return f'revoked:{name}'\n",
        encoding="utf-8",
    )

    second = await server.ingest_codebase(str(repo), source_id="repo", scope="agent-private")

    assert first["status"] == "ok"
    assert second["status"] == "ok"
    assert server._source_records["repo"]["version_id"] != version_before
    assert _superseded_codebase_facts(server, "repo")
    active_after = _active_codebase_facts(server, "repo")
    assert len(active_after) == second["facts_extracted"]
    assert any("revoke" in fact.get("fact", "") for fact in active_after)


@pytest.mark.asyncio
async def test_recall_codebase_hot_query_does_not_attach_source_files(tmp_path, monkeypatch):
    repo = _create_python_repo(tmp_path)
    server = _make_server(tmp_path)
    await server.ingest_codebase(str(repo), source_id="repo", scope="agent-private")

    hydrate_calls = []
    original = CodebaseSemanticSidecarStore.hydrate_sidecar

    def _recording_hydrate(self, sidecar_ref):
        hydrate_calls.append(sidecar_ref["sidecar_id"])
        return original(self, sidecar_ref)

    monkeypatch.setattr(CodebaseSemanticSidecarStore, "hydrate_sidecar", _recording_hydrate)

    result = await server.recall("Which callable calls audit?", search_family="codebase")

    assert "audit" in result["context"]
    assert "pkg.service.issue" in result["context"]
    assert "--- SOURCE FILES ---" not in result["context"]
    assert hydrate_calls == []
    assert result["runtime_trace"]["codebase_augmentation"]["mode"] == "hot_only"


@pytest.mark.asyncio
async def test_recall_codebase_local_cli_profile_downgrades_tool_payload(tmp_path):
    repo = _create_python_repo(tmp_path)
    server = _make_server(tmp_path, key="codebase_local_cli_recall")
    await server.ingest_codebase(str(repo), source_id="repo", scope="agent-private")
    await server.set_profiles(
        {1: "local", 5: "local"},
        {
            "local": {
                "backend": "local_cli",
                "max_output_tokens": 2000,
                "context_window": 128000,
            }
        },
    )

    result = await server.recall(
        "Write a summary of all the codebase facts about audit and issue.",
        search_family="codebase",
    )
    plan = await server.plan_inference(
        "Write a summary of all the codebase facts about audit and issue.",
        search_family="codebase",
    )

    assert result["query_type"] == "summarize"
    assert "payload" not in result
    assert "payload_meta" not in result
    assert plan["payload"]["backend"] == "local_cli"
    assert "tools" not in plan["payload"]
    assert plan["payload_meta"]["backend"] == "local_cli"
    assert plan["payload_meta"]["use_tool"] is False
    assert plan["payload_meta"]["tool_use_downgraded"] is True
    assert plan["payload_meta"]["tool_use_downgrade_reason"] == "local_cli_backend_no_tool_support"
    assert "audit" in result["context"]


@pytest.mark.asyncio
async def test_recall_codebase_exposes_repo_task_context_pack_artifact(tmp_path):
    repo = _create_python_repo(tmp_path)
    server = _make_server(tmp_path, key="codebase_context_pack_runtime")
    await server.ingest_codebase(str(repo), source_id="repo", scope="agent-private")

    result = await server.recall(
        "Return a unified diff that updates the callable which calls audit.",
        search_family="codebase",
        query_type="auto",
        query_metadata={
            "work_item_kind": "repo:work_item",
            "task_mode": "patch_generation",
            "output_artifact": "unified_diff",
            "required_render_mode": "patch_safe_source_windows",
            "repo_scope": {"repo_id": "org/repo", "commit_id": "abc1234"},
            "source_id": "repo",
        },
    )

    trace = result["runtime_trace"]["repo_task_context_pack"]
    assert trace["mode"] == "active"
    assert trace["context_pack_kind"] == "repo_task_context_pack"
    assert trace["operator_domain_policy"] == "full_typed_container_domain_in_scope"
    packs = result["repo_task_context_packs"]
    assert len(packs) >= 1
    pack = next(row for row in packs if row.get("context_pack_generation") == "query_specific")
    assert pack["context_pack_kind"] == "repo_task_context_pack"
    assert pack["work_item"]["repo_task_contract"]["task_mode"] == "patch_generation"
    assert pack["work_item"]["repo_task_contract"]["output_artifact"] == "unified_diff"
    assert pack["work_item"]["repo_task_contract"]["required_render_mode"] == "patch_safe_source_windows"
    assert result["runtime_trace"]["codebase_context"]["repo_task_contract"]["activation_policy"] == "codebase_family_plus_repo_work_item_contract"
    assert pack["sections"]["semantic_code_candidates"]
    assert pack["sections"]["requirements"]
    assert all("trace" in item for item in pack["sections"]["semantic_code_candidates"])


def test_repo_work_item_path_constraint_extractor_roles():
    constraints = extract_repo_work_item_path_constraints(
        "Repository: org/repo\nFix pkg/service.py. Selected test files: tests/test_service.py. Use pyproject.toml.",
        {"repo_scope": {"repo_id": "org/repo"}, "setup_command": "pytest tests/test_service.py"},
    )

    by_path = {row["path"]: row for row in constraints}
    assert by_path["pkg/service.py"]["role"] == "editable_source"
    assert by_path["pkg/service.py"]["required"] is True
    assert by_path["tests/test_service.py"]["role"] == "test_source"
    assert by_path["pyproject.toml"]["role"] == "config_source"
    assert "org/repo" not in by_path


@pytest.mark.asyncio
async def test_repo_patch_context_uses_explicit_paths_as_whole_file_constraints(tmp_path):
    repo = _create_python_repo(tmp_path)
    server = _make_server(tmp_path, key="codebase_patch_context_constraints")
    await server.ingest_codebase(str(repo), source_id="repo", scope="agent-private")

    result = await server.recall(
        "Return a unified diff that updates pkg/service.py and keeps tests/test_service.py passing.",
        search_family="codebase",
        query_type="auto",
        query_metadata={
            "work_item_kind": "repo:work_item",
            "task_mode": "patch_generation",
            "output_artifact": "unified_diff",
            "required_render_mode": "patch_safe_source_context",
            "selected_test_files": ["tests/test_service.py"],
            "source_id": "repo",
        },
    )

    context = result["context"]
    trace = result["runtime_trace"]["codebase_context"]
    by_path = {row["path"]: row for row in trace["explicit_path_constraints"]}
    assert by_path["pkg/service.py"]["role"] == "editable_source"
    assert by_path["tests/test_service.py"]["role"] == "test_source"
    assert "pkg/service.py" in trace["selected_whole_files"]
    assert "tests/test_service.py" in trace["selected_whole_files"]
    assert "[File: pkg/service.py] [render_source=repo_file_blob_span] [whole_file=true]" in context
    assert "[File: tests/test_service.py] [render_source=repo_file_blob_span] [whole_file=true]" in context
    assert "def issue(name: str) -> Permit:" in context
    assert "def test_issue_creates_permit():" in context
    assert "[source lines" not in context
    assert "[source-window gap" not in context
    assert not re.search(r"^\s*\d+:", context, re.M)

    pack = next(row for row in result["repo_task_context_packs"] if row.get("context_pack_generation") == "query_specific")
    assert pack["patch_context_policy"]["mode"] == "whole_file_first"
    assert pack["sections"]["explicit_path_constraints"]
    assert pack["sections"]["editable_source_files"][0]["path"] == "pkg/service.py"
    assert any(row["path"] == "tests/test_service.py" for row in pack["sections"]["test_source_files"])


@pytest.mark.asyncio
async def test_repo_patch_context_extracts_paths_from_raw_work_item_not_canonical_query(tmp_path):
    repo = _create_python_repo(tmp_path)
    server = _make_server(tmp_path, key="codebase_patch_context_raw_work_item")
    await server.ingest_codebase(str(repo), source_id="repo", scope="agent-private")

    query = """\
You are fixing a repository work item. Return only a unified diff patch.

Repository: org/repo

Base commit: abc1234

Requirements:

- The file pkg/service.py should update the issue implementation.

Interface:

Location: pkg/service.py | function issue.

Selected test files:

["tests/test_service.py"]
"""

    result = await server.recall(
        query,
        search_family="codebase",
        query_type="auto",
        query_metadata={
            "work_item_kind": "repo:work_item",
            "task_mode": "patch_generation",
            "output_artifact": "unified_diff",
            "required_render_mode": "patch_safe_source_context",
            "repo_scope": {"repo_id": "org/repo", "commit_id": "abc1234"},
            "source_id": "repo",
        },
    )

    trace = result["runtime_trace"]["codebase_context"]
    assert trace["path_constraint_source"] == "raw_work_item_query"
    by_path = {row["path"]: row for row in trace["explicit_path_constraints"]}
    assert "org/repo" not in by_path
    assert by_path["pkg/service.py"]["required"] is True
    assert "requirement" in by_path["pkg/service.py"]["sources"]
    assert "interface" in by_path["pkg/service.py"]["sources"]
    assert by_path["tests/test_service.py"]["required"] is True
    assert by_path["tests/test_service.py"]["source"] == "selected_test_files"


@pytest.mark.asyncio
async def test_repo_patch_context_uses_codebase_scope_when_metadata_source_id_is_work_item(tmp_path):
    repo = _create_python_repo(tmp_path)
    server = _make_server(tmp_path, key="codebase_patch_context_work_item_source")
    await server.ingest_codebase(str(repo), source_id="repo", scope="agent-private")

    result = await server.recall(
        "Return a unified diff that updates pkg/service.py and keeps tests/test_service.py passing.",
        search_family="codebase",
        query_type="auto",
        query_metadata={
            "work_item_kind": "repo:work_item",
            "task_mode": "patch_generation",
            "output_artifact": "unified_diff",
            "required_render_mode": "patch_safe_source_context",
            "selected_test_files": ["tests/test_service.py"],
            # Work-item ids are valid task metadata but are not necessarily the
            # codebase projection source id. Repository context must fall back to the
            # active codebase source in scope rather than silently disappearing.
            "source_id": "work_item_123",
            "work_item_id": "work_item_123",
        },
    )

    context = result["context"]
    trace = result["runtime_trace"]["codebase_context"]
    assert trace["mode"] == "active"
    assert trace["source_ids"] == ["repo"]
    assert "pkg/service.py" in trace["selected_whole_files"]
    assert "tests/test_service.py" in trace["selected_whole_files"]
    assert "[File: pkg/service.py] [render_source=repo_file_blob_span] [whole_file=true]" in context
    assert "[File: tests/test_service.py] [render_source=repo_file_blob_span] [whole_file=true]" in context


@pytest.mark.asyncio
async def test_repo_patch_context_uses_container_graph_when_visible_facts_empty(tmp_path):
    repo = _create_python_repo(tmp_path)
    server = _make_server(tmp_path, key="codebase_patch_context_graph_only")
    await server.ingest_codebase(str(repo), source_id="repo", scope="agent-private")

    # Some cache-only replay stores have an active Repository context container graph but
    # no visible fact rows. Repository patch context must still come from the
    # typed graph/render refs instead of disappearing behind empty fact recall.
    server._all_granular = []
    server._all_cons = []
    server._all_cross = []

    result = await server.recall(
        "Return a unified diff that updates pkg/service.py and keeps tests/test_service.py passing.",
        search_family="codebase",
        query_type="auto",
        query_metadata={
            "work_item_kind": "repo:work_item",
            "task_mode": "patch_generation",
            "output_artifact": "unified_diff",
            "required_render_mode": "patch_safe_source_context",
            "selected_test_files": ["tests/test_service.py"],
            "source_id": "repo",
        },
    )

    context = result["context"]
    trace = result["runtime_trace"]["codebase_context"]
    assert result["runtime_trace"]["reason"] == "codebase_context_without_visible_facts"
    assert trace["mode"] == "active"
    assert trace["operator_domain_policy"] == "full_typed_container_domain_in_scope"
    assert "pkg/service.py" in trace["selected_whole_files"]
    assert "tests/test_service.py" in trace["selected_whole_files"]
    assert "--- REPOSITORY CONTEXT PACK ---" in context
    assert "[File: pkg/service.py] [render_source=repo_file_blob_span] [whole_file=true]" in context
    assert "[File: tests/test_service.py] [render_source=repo_file_blob_span] [whole_file=true]" in context
    assert result["repo_task_context_packs"][0]["context_pack_kind"] == "repo_task_context_pack"


@pytest.mark.asyncio
async def test_repo_patch_context_keeps_semantic_config_neighbor_after_required_paths(tmp_path):
    repo = _create_python_repo(tmp_path)
    config_dir = repo / "config"
    config_dir.mkdir()
    (config_dir / "settings.yml").write_text(
        "fonts:\n  default_family: default_family\n  default_size: 10pt\n",
        encoding="utf-8",
    )
    server = _make_server(tmp_path, key="codebase_patch_context_optional_config")
    await server.ingest_codebase(str(repo), source_id="repo", scope="agent-private")

    ctx, trace = build_codebase_context(
        server,
        query="Return a unified diff that updates pkg/service.py for the default_size setting.",
        source_ids={"repo"},
        query_metadata={
            "work_item_kind": "repo:work_item",
            "task_mode": "patch_generation",
            "output_artifact": "unified_diff",
            "required_render_mode": "patch_safe_source_context",
            "selected_test_files": ["tests/test_service.py"],
            "source_id": "repo",
        },
        path_constraint_query=(
            "Return a unified diff that updates pkg/service.py for the default_size setting. "
            "Selected test files: tests/test_service.py"
        ),
        max_files=1,
    )

    assert ctx is not None
    assert trace["mode"] == "active"
    assert "pkg/service.py" in trace["selected_whole_files"]
    assert "tests/test_service.py" in trace["selected_whole_files"]
    assert "config/settings.yml" in trace["selected_whole_files"]
    assert "[File: config/settings.yml] [render_source=repo_file_blob_span] [whole_file=true]" in ctx
    assert "default_size: 10pt" in ctx


@pytest.mark.asyncio
async def test_repo_patch_context_promotes_typed_config_source_to_required_constraint(tmp_path):
    repo = _create_python_repo(tmp_path)
    config_dir = repo / "config"
    config_dir.mkdir()
    (config_dir / "settings.yml").write_text(
        "\n".join(
            [
                "fonts.default_family:",
                "  default: []",
                "  type:",
                "    name: ListOrValue",
                "    valtype: Font",
                "fonts.statusbar:",
                "  default: 10pt default_family",
                "  type: Font",
                "fonts.prompts:",
                "  default: 10pt sans-serif",
                "  type: Font",
            ]
        ),
        encoding="utf-8",
    )
    server = _make_server(tmp_path, key="codebase_patch_context_required_config")
    await server.ingest_codebase(str(repo), source_id="repo", scope="agent-private")

    ctx, trace = build_codebase_context(
        server,
        query="Return a unified diff that updates pkg/service.py and adds fonts.default_size config behavior.",
        source_ids={"repo"},
        query_metadata={
            "work_item_kind": "repo:work_item",
            "task_mode": "patch_generation",
            "output_artifact": "unified_diff",
            "required_render_mode": "patch_safe_source_context",
            "selected_test_files": ["tests/test_service.py"],
            "source_id": "repo",
        },
        path_constraint_query=(
            "Return a unified diff that updates pkg/service.py and adds fonts.default_size config behavior. "
            "Selected test files: tests/test_service.py"
        ),
        max_files=1,
    )

    assert ctx is not None
    by_path = {row["path"]: row for row in trace["explicit_path_constraints"]}
    assert by_path["config/settings.yml"]["role"] == "config_source"
    assert by_path["config/settings.yml"]["required"] is True
    assert by_path["config/settings.yml"]["derived"] is True
    assert by_path["config/settings.yml"]["reason"] == "config_semantics_required_by_patch_query"
    assert "config/settings.yml" in trace["selected_whole_files"]
    selected_paths = [row["path"] for row in trace["selected"]]
    assert selected_paths.index("config/settings.yml") < selected_paths.index("tests/test_service.py")
    config_item = next(row for row in trace["selected"] if row["path"] == "config/settings.yml")
    assert config_item["role"] == "config_source"
    assert "config_semantics_required_by_patch_query" in config_item["reasons"]
    pack = trace["query_context_pack"]
    assert any(row["path"] == "config/settings.yml" for row in pack["sections"]["supporting_source_files"])
    assert "path_constraint_resolution:" in ctx
    assert "selected_whole_files:" in ctx


@pytest.mark.asyncio
async def test_repo_patch_context_budget_failure_is_typed(tmp_path, monkeypatch):
    repo = _create_python_repo(tmp_path)
    monkeypatch.setattr(codebase_query, "_PATCH_CONTEXT_TOTAL_BUDGET", 64)
    server = _make_server(tmp_path, key="codebase_patch_context_budget_failure")
    await server.ingest_codebase(str(repo), source_id="repo", scope="agent-private")

    ctx, trace = build_codebase_context(
        server,
        query="Return a unified diff that updates pkg/service.py.",
        source_ids={"repo"},
        query_metadata={
            "work_item_kind": "repo:work_item",
            "task_mode": "patch_generation",
            "output_artifact": "unified_diff",
            "required_render_mode": "patch_safe_source_context",
            "source_id": "repo",
        },
    )

    assert ctx.startswith("Not enough grounded context.")
    assert trace["mode"] == "failed_closed"
    assert trace["reason"] == "REQUIRED_PATCH_CONTEXT_INCOMPLETE"
    assert any(row.get("code") == "REQUIRED_PATH_EXCEEDS_CONTEXT_BUDGET" for row in trace["omitted_required_files"])


@pytest.mark.asyncio
async def test_repo_operation_patch_attempt_apply_test_and_verification_are_persisted(tmp_path):
    repo = _create_python_repo(tmp_path)
    server = _make_server(tmp_path, key="codebase_operation_memory")
    await server.ingest_codebase(str(repo), source_id="repo", scope="agent-private")

    diff_text = """diff --git a/pkg/service.py b/pkg/service.py
--- a/pkg/service.py
+++ b/pkg/service.py
@@ -6,5 +6,5 @@ def audit(name: str) -> str:
 def issue(name: str) -> Permit:
     audit(name)
-    return Permit(name)
+    return Permit(name.strip())
"""
    patch = server.record_repo_patch_attempt(
        diff_text=diff_text,
        source_id="repo",
        source_context_refs=["render_ref_1"],
        model="diagnostic-model",
        profile="repo_task",
    )
    assert patch["status"] == "ok"
    assert patch["payload"]["touched_files"] == ["pkg/service.py"]
    assert patch["payload"]["hunks"]

    apply_result = server.record_repo_patch_apply_result(
        patch_attempt_id=patch["patch_attempt_id"],
        exit_code=1,
        stderr="error: patch failed: pkg/service.py:6",
        failed_files=["pkg/service.py"],
        status="failed",
    )
    assert apply_result["status"] == "ok"
    assert apply_result["payload"]["status"] == "failed"

    test_run = server.record_repo_test_run(
        patch_attempt_id=patch["patch_attempt_id"],
        commands=["pytest tests/test_service.py"],
        selected_tests=["tests/test_service.py"],
        exit_code=1,
        stderr="AssertionError",
        status="failed",
    )
    assert test_run["status"] == "ok"

    verification = server.record_repo_verification_result(
        patch_attempt_id=patch["patch_attempt_id"],
        status="refuted",
        evidence_refs=[apply_result["container_id"], test_run["container_id"]],
        remaining_gaps=[{"code": "PATCH_APPLY_FAILED"}],
    )
    assert verification["status"] == "ok"

    graph = server._container_graph
    kinds = {row["kind_fq"] for row in graph["containers"]}
    assert "operation:patch_attempt" in kinds
    assert "operation:patch_apply_result" in kinds
    assert "operation:test_run" in kinds
    assert "operation:verification_result" in kinds
    relation_kinds = {row["relation_kind"] for row in graph["relations"]}
    assert "has_patch_attempt" in relation_kinds
    assert "patch_touches" in relation_kinds
    assert "verified_by" in relation_kinds
    assert "verification_refutes" in relation_kinds
    artifact_kinds = {row["artifact_kind"] for row in graph["artifacts"]}
    assert "operation_patch_attempt" in artifact_kinds
    assert "operation_patch_apply_result" in artifact_kinds
    assert "operation_test_run" in artifact_kinds
    assert "operation_verification_result" in artifact_kinds

    result = await server.recall(
        "Return a unified diff that updates pkg/service.py and keeps tests/test_service.py passing.",
        search_family="codebase",
        query_type="auto",
        query_metadata={
            "work_item_kind": "repo:work_item",
            "task_mode": "patch_generation",
            "output_artifact": "unified_diff",
            "required_render_mode": "patch_safe_source_context",
            "selected_test_files": ["tests/test_service.py"],
            "source_id": "repo",
        },
    )

    operation_state = result["runtime_trace"]["codebase_context"]["operation_state"]
    assert any(row["kind_fq"] == "operation:patch_attempt" for row in operation_state)
    assert any(row["kind_fq"] == "operation:patch_apply_result" for row in operation_state)
    assert any(row["kind_fq"] == "operation:test_run" for row in operation_state)
    assert any(row["kind_fq"] == "operation:verification_result" for row in operation_state)
    pack = result["runtime_trace"]["codebase_context"]["query_context_pack"]
    assert pack["sections"]["operation_state"]
    assert pack["sections"]["patch_attempts"]
    assert "--- CURRENT OPERATION STATE ---" in result["context"]


@pytest.mark.asyncio
async def test_non_patch_codebase_query_does_not_force_patch_context(tmp_path):
    repo = _create_python_repo(tmp_path)
    server = _make_server(tmp_path, key="codebase_non_patch_context")
    await server.ingest_codebase(str(repo), source_id="repo", scope="agent-private")

    result = await server.recall("Which callable calls audit?", search_family="codebase")

    trace = result["runtime_trace"]["codebase_context"]
    assert trace.get("patch_context_policy") == {}
    assert trace.get("explicit_path_constraints") == []
    assert "--- EXPLICIT PATH CONSTRAINTS ---" not in result["context"]



@pytest.mark.asyncio
async def test_ingest_codebase_materializes_generic_config_render_refs(tmp_path):
    repo = _create_python_repo(tmp_path)
    (repo / "config").mkdir()
    (repo / "config" / "settings.yml").write_text(
        "ui:\n  default_family: default_family\n  default_size: 10pt\n",
        encoding="utf-8",
    )
    server = _make_server(tmp_path, key="codebase_generic_config_runtime")

    await server.ingest_codebase(str(repo), source_id="repo", scope="agent-private")

    graph = server._container_graph
    config_container = next(
        row
        for row in graph["containers"]
        if row["kind_fq"] == "code:config"
        and (row.get("traits_json") or {}).get("path") == "config/settings.yml"
    )
    render_ref = next(
        row
        for row in graph["render_refs"]
        if row["render_ref_id"] == config_container["primary_render_ref_id"]
    )
    assert render_ref["ref_json"]["render_source"] == "repo_file_blob_span"
    assert render_ref["ref_json"]["text_exact"] is True
    assert "default_size: 10pt" in render_ref["ref_json"]["text"]


@pytest.mark.asyncio
async def test_recall_codebase_context_lifts_query_paths_from_full_container_graph(tmp_path):
    repo = _create_python_repo(tmp_path)
    (repo / "config").mkdir()
    (repo / "config" / "settings.yml").write_text(
        "ui:\n  default_family: default_family\n  default_size: 10pt\n",
        encoding="utf-8",
    )
    server = _make_server(tmp_path, key="codebase_context_query_context_runtime")
    await server.ingest_codebase(str(repo), source_id="repo", scope="agent-private")

    result = await server.recall(
        "Update pkg/service.py and config/settings.yml so issue uses the default_size setting.",
        search_family="codebase",
    )

    trace = result["runtime_trace"]["codebase_context"]
    assert trace["mode"] == "active"
    assert trace["operator_domain_policy"] == "full_typed_container_domain_in_scope"
    assert trace["seed_domain_policy"] == "facts_and_query_are_seed_only"
    assert "--- REPOSITORY CONTEXT PACK ---" in result["context"]
    assert "[File: pkg/service.py]" in result["context"]
    assert "def issue(name: str) -> Permit:" in result["context"]
    assert "[File: config/settings.yml]" in result["context"]
    assert "default_size: 10pt" in result["context"]
    assert trace["selected"]
    assert trace["rejected_candidates"]


@pytest.mark.asyncio
async def test_recall_codebase_exact_renders_explicit_paths_without_patch_contract(tmp_path):
    repo = _create_python_repo(tmp_path)
    (repo / "package.json").write_text(
        json.dumps({"scripts": {"test": "pytest tests/test_service.py"}}),
        encoding="utf-8",
    )
    server = _make_server(tmp_path, key="codebase_context_explicit_path_recall")
    await server.ingest_codebase(str(repo), source_id="repo", scope="agent-private")

    result = await server.recall(
        "Show exact source for tests/test_service.py and package.json before I edit issue handling.",
        search_family="codebase",
        query_type="lookup",
    )

    context = result["context"]
    trace = result["runtime_trace"]["codebase_context"]
    by_path = {row["path"]: row for row in trace["explicit_path_constraints"]}

    assert by_path["tests/test_service.py"]["role"] == "test_source"
    assert by_path["package.json"]["role"] == "config_source"
    assert "tests/test_service.py" in trace["selected_whole_files"]
    assert "package.json" in trace["selected_whole_files"]
    assert "[File: tests/test_service.py] [render_source=repo_file_blob_span] [whole_file=true]" in context
    assert "[File: package.json] [render_source=repo_file_blob_span] [whole_file=true]" in context
    assert "def test_issue_creates_permit():" in context
    assert '"test": "pytest tests/test_service.py"' in context
    test_block = context.split("[File: tests/test_service.py]", 1)[1].split("\n[File:", 1)[0]
    package_block = context.split("[File: package.json]", 1)[1].split("\n[File:", 1)[0]
    for source_block in (test_block, package_block):
        assert "[source lines" not in source_block
        assert "[source-window gap" not in source_block
        assert not re.search(r"^\s*\d+:", source_block, re.M)

    resolution_by_path = {row["path"]: row for row in trace["path_constraint_resolution"]}
    assert resolution_by_path["tests/test_service.py"]["render_source"] == "repo_file_blob_span"
    assert resolution_by_path["tests/test_service.py"]["render_mode"] == "exact_copy"
    assert resolution_by_path["tests/test_service.py"]["status"] == "selected"
    pack = next(row for row in result["repo_task_context_packs"] if row.get("context_pack_generation") == "query_specific")
    assert any(row["path"] == "tests/test_service.py" for row in pack["sections"]["test_source_files"])
    assert any(row["path"] == "package.json" for row in pack["sections"]["supporting_source_files"])


@pytest.mark.asyncio
async def test_recall_codebase_context_survives_profile_payload_truncation(tmp_path):
    repo = _create_python_repo(tmp_path)
    (repo / "config").mkdir()
    (repo / "config" / "settings.yml").write_text(
        "ui:\n  default_family: default_family\n  default_size: 10pt\n",
        encoding="utf-8",
    )
    server = _make_server(tmp_path, key="codebase_context_query_context_payload")
    await server.set_profiles(
        {1: "fast"},
        {
            "fast": {
                "backend": "local_cli",
                "context_window": 128000,
                "max_output_tokens": 1024,
            }
        },
    )
    await server.ingest_codebase(str(repo), source_id="repo", scope="agent-private")

    result = await server.recall(
        "Update pkg/service.py and config/settings.yml so issue uses the default_size setting.",
        search_family="codebase",
    )
    plan = await server.plan_inference(
        "Update pkg/service.py and config/settings.yml so issue uses the default_size setting.",
        search_family="codebase",
    )

    assert "payload_meta" not in result
    assert plan["payload_meta"]["backend"] == "local_cli"
    assert "--- REPOSITORY CONTEXT PACK ---" in result["context"]
    assert "[File: config/settings.yml]" in result["context"]
    assert "default_size: 10pt" in result["context"]
    payload_text = "\n".join(str(message.get("content") or "") for message in plan["payload"]["messages"])
    assert "--- REPOSITORY CONTEXT PACK ---" in payload_text
    assert "[File: config/settings.yml]" in payload_text
    assert "default_size: 10pt" in payload_text


@pytest.mark.asyncio
async def test_repo_patch_context_uses_strong_profile_and_is_not_tier4_evicted(tmp_path):
    repo = _create_python_repo(tmp_path)
    server = _make_server(tmp_path, key="codebase_patch_context_profile_priority")
    await server.set_profiles(
        {1: "fast", 5: "strong"},
        {
            "fast": {
                "backend": "local_cli",
                "context_window": 600,
                "max_output_tokens": 128,
            },
            "strong": {
                "backend": "local_cli",
                "context_window": 128000,
                "max_output_tokens": 1024,
            },
        },
    )
    await server.ingest_codebase(str(repo), source_id="repo", scope="agent-private")

    query = "Return a unified diff that updates pkg/service.py and keeps tests/test_service.py passing."
    query_metadata = {
        "work_item_kind": "repo:work_item",
        "task_mode": "patch_generation",
        "output_artifact": "unified_diff",
        "required_render_mode": "patch_safe_source_context",
        "selected_test_files": ["tests/test_service.py"],
        "source_id": "repo",
    }
    result = await server.recall(
        query,
        search_family="codebase",
        query_type="auto",
        query_metadata=query_metadata,
    )
    plan = await server.plan_inference(
        query,
        search_family="codebase",
        query_type="auto",
        query_metadata=query_metadata,
    )

    assert "recommended_profile" not in result
    assert plan["recommended_profile"] == "strong"
    assert plan["payload_meta"]["profile_used"] == "strong"
    assert plan["payload_meta"]["budget_exceeded"] is False
    assert result["context"].startswith("--- REPOSITORY CONTEXT PACK ---")
    assert "RETRIEVED FACTS:" not in result["context"]
    payload_text = "\n".join(str(message.get("content") or "") for message in plan["payload"]["messages"])
    assert "--- REPOSITORY CONTEXT PACK ---" in payload_text
    assert "[File: pkg/service.py] [render_source=repo_file_blob_span] [whole_file=true]" in payload_text
    assert "[File: tests/test_service.py] [render_source=repo_file_blob_span] [whole_file=true]" in payload_text


@pytest.mark.asyncio
async def test_recall_codebase_context_falls_back_when_query_embedding_unavailable(tmp_path):
    repo = _create_python_repo(tmp_path)
    (repo / "config").mkdir()
    (repo / "config" / "settings.yml").write_text(
        "ui:\n  default_family: default_family\n  default_size: 10pt\n",
        encoding="utf-8",
    )
    server = _make_server(tmp_path, key="codebase_context_query_context_no_embedding")
    await server.ingest_codebase(str(repo), source_id="repo", scope="agent-private")

    async def _raise_query_embedding(_text, **_kwargs):
        raise RuntimeError("embedding unavailable")

    server._embed_query_with_runtime_secrets = _raise_query_embedding
    result = await server.recall(
        "Update pkg/service.py and config/settings.yml so issue uses the default_size setting.",
        search_family="codebase",
    )

    trace = result["runtime_trace"]
    assert trace["embedding"]["mode"] == "lexical_seed_fallback"
    assert trace["embedding"]["vector_seed_disabled"] is True
    assert trace["codebase_context"]["mode"] == "active"
    assert "--- REPOSITORY CONTEXT PACK ---" in result["context"]
    assert "[File: config/settings.yml]" in result["context"]
    assert "default_size: 10pt" in result["context"]


@pytest.mark.asyncio
async def test_recall_codebase_uses_semantic_config_option_candidates(tmp_path):
    repo = _create_python_repo(tmp_path)
    config_dir = repo / "config"
    config_dir.mkdir()
    generated_font_rows = []
    for idx in range(40):
        generated_font_rows.extend(
            [
                f"fonts.generated_{idx}:",
                "  default: 10pt default_family",
                "  type: Font",
            ]
        )
    (config_dir / "settings.yml").write_text(
        "\n".join(
            [
                "## unrelated",
                *[f"unrelated.option_{idx}:" for idx in range(40)],
                "## fonts",
                "fonts.default_family:",
                "  default: []",
                *generated_font_rows,
                "fonts.statusbar:",
                "  default: 10pt default_family",
                "  type: Font",
                "## colors",
                "colors.statusbar.normal.bg:",
            ]
        ),
        encoding="utf-8",
    )
    server = _make_server(tmp_path, key="codebase_semantic_config_candidates")
    await server.ingest_codebase(str(repo), source_id="repo", scope="agent-private")

    result = await server.recall(
        "Add fonts.default_size and update fonts.default_family consumers.",
        search_family="codebase",
    )

    trace = result["runtime_trace"]["codebase_context"]
    selected_config = next(
        item
        for item in trace["selected"]
        if item["path"] == "config/settings.yml"
    )
    assert selected_config["config_option_entries"]
    assert "Structured config option candidates from Repository context semantic bundle:" in result["context"]
    assert "fonts.statusbar:" in result["context"]
    assert "colors.statusbar.normal.bg:" not in result["context"]
    query_packs = [
        pack
        for pack in result["repo_task_context_packs"]
        if pack.get("context_pack_generation") == "query_specific"
    ]
    assert query_packs
    assert query_packs[0]["sections"]["semantic_code_candidates"]
    checklist = query_packs[0]["sections"]["config_applicability_checklist"]
    assert checklist
    assert any(row["config_key"] == "fonts.statusbar" and row["required"] for row in checklist)
    assert "--- CONFIG APPLICABILITY CHECKLIST ---" in result["context"]


def _semantic_config_supporting_entries(text: str) -> list[dict]:
    # Exercise the production ecosystem plugin extraction through the public runner
    # in integration tests; this helper only creates query-runtime container inputs.
    lines = text.splitlines()
    key_rows = [
        (idx + 1, line.split(":", 1)[0])
        for idx, line in enumerate(lines)
        if re.match(r"^[A-Za-z0-9_.-]+:", line)
    ]
    supporting = []
    for index, (start, option) in enumerate(key_rows):
        end = key_rows[index + 1][0] - 1 if index + 1 < len(key_rows) else len(lines)
        block = "\n".join(lines[start - 1 : end])
        default_match = re.search(r"(?m)^\s*default:\s*(.*)$", block)
        type_match = re.search(r"(?m)^\s*type:\s*([A-Za-z0-9_.-]+)", block)
        default_value = default_match.group(1).strip() if default_match else ""
        default_tokens = [
            {"token": token, "kind": "size_literal" if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?(?:pt|px)", token) else "symbol"}
            for token in default_value.split()
        ]
        supporting.append(
            {
                "container": {
                    "container_id": f"cfg-{index}",
                    "kind_fq": "code:config",
                    "primary_render_ref_id": f"render-cfg-{index}",
                    "render_ref_json": {
                        "path": "config/settings.yml",
                        "span": {"start_line": start, "end_line": end},
                    },
                    "traits_json": {
                        "name": option,
                        "qualified_name": f"config/settings.yml:{option}",
                        "path": "config/settings.yml",
                        "payload": {
                            "config_object_kind": "option",
                            "option": option,
                            "option_prefix": option.lower().split(".", 1)[0],
                            "type": type_match.group(1) if type_match else "",
                            "default": default_value,
                            "references": [row["token"] for row in default_tokens if row["kind"] == "symbol"],
                            "default_tokens": default_tokens,
                            "leading_literal_token": default_tokens[0]["token"] if default_tokens else "",
                        },
                    },
                },
                "score": 1,
                "render_ref": {
                    "render_ref_id": f"render-cfg-{index}",
                    "ref_json": {
                        "path": "config/settings.yml",
                        "span": {"start_line": start, "end_line": end},
                        "render_source": "repo_file_blob_span",
                        "text_exact": True,
                    },
                },
                "reasons": ["test_semantic_option"],
            }
        )
    return supporting



def test_codebase_config_applicability_checklist_marks_matching_defaults_required():
    text = "\n".join(
        [
            "## ui",
            "ui.default_family:",
            "  default: []",
            "  type:",
            "    name: ListOrValue",
            "    valtype: Font",
            "ui.completion.entry:",
            "  default: 10pt default_family",
            "  type: Font",
            "ui.prompts:",
            "  default: 10pt sans-serif",
            "  type: Font",
            "ui.tabs:",
            "  default: 10pt default_family",
            "  type: QtFont",
            "ui.web.size.default:",
            "  default: 16",
            "  type: Int",
        ]
    )
    query_terms = _codebase_query_terms(
        "Add ui.default_size and replace default size literals with default_size token for all UI font settings and ui.default_family consumers."
    )
    entries = _codebase_semantic_config_option_entries(_semantic_config_supporting_entries(text), query_terms)
    checklist = _codebase_config_applicability_checklist(entries, query_terms)
    by_key = {row["config_key"]: row for row in checklist}

    assert by_key["ui.completion.entry"]["applicable"] is True
    assert by_key["ui.completion.entry"]["required"] is True
    assert "default_literal_candidate" in by_key["ui.completion.entry"]["selection_reasons"]
    assert by_key["ui.prompts"]["current_default"] == "10pt sans-serif"
    assert by_key["ui.prompts"]["applicable"] is True
    assert by_key["ui.prompts"]["required"] is True
    assert "same_option_type_family" in by_key["ui.prompts"]["selection_reasons"]
    assert by_key["ui.default_family"]["coverage_status"] == "rejected"
    assert "no_transformable_default_literal" in by_key["ui.default_family"]["rejection_reasons"]
    matching_defaults = {
        entry["option"]
        for entry in entries
        if "default_literal_candidate" in (entry.get("trace") or {}).get("reasons", [])
    }
    checklist_keys = {row["config_key"] for row in checklist}
    assert matching_defaults <= checklist_keys


def test_codebase_config_type_validation_evidence_exposes_siblings_parsers_tests_and_gaps():
    config_text = "\n".join(
        [
            "ui.default_family:",
            "  default: []",
            "  type: ListOrValue",
            "ui.entry:",
            "  default: 10pt default_family",
            "  type: Font",
            "ui.prompts:",
            "  default: 10pt sans-serif",
            "  type: Font",
        ]
    )
    query_terms = _codebase_query_terms(
        "Add ui.default_size and update all UI Font defaults to use default_size default_family."
    )
    entries = _codebase_semantic_config_option_entries(_semantic_config_supporting_entries(config_text), query_terms)
    checklist = _codebase_config_applicability_checklist(entries, query_terms)
    selected_items = [
        {
            "path": "config/settings.yml",
            "kind_fq": "code:config",
            "render_ref_id": "render-config-file",
            "text": config_text,
        },
        {
            "path": "pkg/config_types.py",
            "kind_fq": "code:file",
            "render_ref_id": "render-types",
            "text": "class Font:\n    def to_py(self, value):\n        self._basic_py_validation(value, str)\n        raise ValidationError(value, 'bad')\n",
        },
        {
            "path": "tests/test_config_types.py",
            "kind_fq": "code:file",
            "render_ref_id": "render-tests",
            "text": "def test_default_size_validation():\n    assert parse('default_size default_family')\n",
        },
    ]

    evidence = _codebase_config_type_validation_evidence(selected_items, checklist, query_terms)
    kinds = {row["evidence_kind"] for row in evidence}
    assert {"schema_type", "default_value", "sibling_option_type", "parser", "validator", "test_assertion"} <= kinds
    assert any(row["symbol_or_config_key"] == "ui.default_size" and row["evidence_kind"] == "sibling_option_type" for row in evidence)

    gap = _codebase_config_type_validation_evidence([], [], _codebase_query_terms("Add app.default_size config option."))
    assert gap
    assert gap[0]["coverage_status"] == "failed_closed"
    assert gap[0]["failure_code"] == "missing_config_validation_evidence"


def test_codebase_config_option_entries_are_structured_from_semantic_containers():
    text = "\n".join(
        [
            "## fonts",
            "fonts.default_family:",
            "  default: []",
            "  type:",
            "    name: ListOrValue",
            "    valtype: Font",
            "fonts.completion.entry:",
            "  default: 10pt default_family",
            "  type: Font",
            "fonts.tabs:",
            "  default: bold 10pt default_family",
            "  type: QtFont",
            "fonts.prompts:",
            "  default: 10pt sans-serif",
            "  type: Font",
            "fonts.web.size.default:",
            "  default: 16",
            "  type: Int",
            "colors.statusbar.normal.bg:",
            "  default: black",
            "  type: Color",
        ]
    )

    entries = _codebase_semantic_config_option_entries(
        _semantic_config_supporting_entries(text),
        _codebase_query_terms("Add fonts.default_size and update fonts.default_family consumers."),
    )

    by_option = {entry["option"]: entry for entry in entries}
    assert by_option["fonts.completion.entry"]["type"] == "Font"
    assert by_option["fonts.completion.entry"]["default"] == "10pt default_family"
    assert by_option["fonts.completion.entry"]["references"] == ["default_family"]
    assert by_option["fonts.completion.entry"]["leading_literal_token"] == "10pt"
    assert by_option["fonts.completion.entry"]["default_tokens"] == [
        {"token": "10pt", "kind": "size_literal"},
        {"token": "default_family", "kind": "symbol"},
    ]
    assert "default_size" in by_option["fonts.completion.entry"]["requested_tokens"]
    assert by_option["fonts.tabs"]["type"] == "QtFont"
    assert by_option["fonts.prompts"]["default"] == "10pt sans-serif"
    assert by_option["fonts.prompts"]["references"] == []
    assert by_option["fonts.prompts"]["leading_literal_token"] == "10pt"
    assert "default_literal_candidate" in by_option["fonts.prompts"]["trace"]["reasons"]
    assert "fonts.web.size.default" not in by_option
    assert "colors.statusbar.normal.bg" not in by_option


def test_codebase_semantic_config_option_windows_pin_exact_source_sections():
    text = "\n".join(
        [
            "## unrelated",
            *[f"unrelated.option_{idx}:" for idx in range(30)],
            "## fonts",
            "fonts.default_family:",
            "  default: []",
            "  type:",
            "    name: ListOrValue",
            "    valtype: Font",
            "fonts.completion.entry:",
            "  default: 10pt default_family",
            "  type: Font",
            "fonts.completion.category:",
            "  default: bold 10pt default_family",
            "  type: Font",
            "fonts.tabs:",
            "  default: 10pt default_family",
            "  type: QtFont",
            "fonts.prompts:",
            "  default: 10pt sans-serif",
            "  type: Font",
            "fonts.web.size.default:",
            "  default: 16",
            "  type: Int",
            "## colors",
            "colors.statusbar.normal.bg:",
            "  default: black",
        ]
    )
    query_terms = _codebase_query_terms("Add fonts.default_size and update default_family font options.")
    entries = _codebase_semantic_config_option_entries(_semantic_config_supporting_entries(text), query_terms)

    windows = _codebase_semantic_config_option_windows(entries, line_count=len(text.splitlines()))
    rendered, trace_windows = _codebase_line_windows(
        text=text,
        query_terms=query_terms,
        prefer_config_sections=True,
        pinned_windows=windows,
        include_line_numbers=False,
        max_windows=8,
        max_chars=4000,
    )

    assert trace_windows[0]["start_line"] <= text.splitlines().index("fonts.completion.entry:") + 1
    assert "fonts.completion.entry:" in rendered
    assert "fonts.completion.category:" in rendered
    assert "fonts.tabs:" in rendered
    assert "fonts.prompts:" in rendered
    assert "fonts.web.size.default:" not in rendered
    assert "\n...\n" not in rendered
    assert "[source-window gap:" in rendered
    assert "unrelated.option_0:" not in rendered


def test_codebase_line_windows_render_pinned_semantic_spans_before_supplemental_order():
    lines = [f"line {idx}" for idx in range(1, 121)]
    lines[79] = "class TargetSymbol:"
    lines[9] = "class EarlyNoise:"

    rendered, windows = _codebase_line_windows(
        text="\n".join(lines),
        query_terms=_codebase_query_terms("Fix TargetSymbol behavior"),
        pinned_windows=[(78, 83), (8, 12)],
        include_line_numbers=False,
    )

    assert windows[0]["start_line"] == 79
    assert rendered.find("class TargetSymbol:") < rendered.find("class EarlyNoise:")


@pytest.mark.asyncio
async def test_recall_codebase_context_renders_full_typed_symbol_spans(tmp_path):
    repo = _create_font_symbol_repo(tmp_path)
    server = _make_server(tmp_path, key="codebase_context_typed_span_context")
    await server.ingest_codebase(str(repo), source_id="repo", scope="agent-private")

    result = await server.recall(
        "Fix QtFont default family behavior in pkg/fonts.py using Font.set_default_family.",
        search_family="codebase",
    )

    assert "[File: pkg/fonts.py]" in result["context"]
    assert "def set_default_family(self, default_family):" in result["context"]
    assert "typed_span_only_bridge = marker_30" in result["context"]
    assert "resolved = family_candidates[0]" in result["context"]
    assert "class QtFont(Font):" in result["context"]
    codebase_context = result["context"].split("--- REPOSITORY CONTEXT PACK ---", 1)[1]
    assert "[source lines " in codebase_context
    assert "--- source lines " not in codebase_context
    assert not re.search(r"(?m)^\s*\d+:\s", codebase_context)
    trace = result["runtime_trace"]["codebase_context"]
    item = next(row for row in trace["selected"] if row["path"] == "pkg/fonts.py")
    typed_spans = item["typed_span_windows"]
    assert typed_spans
    assert any(
        row["kind_fq"] in {"code:class", "code:method"}
        and str(row.get("qualified_name") or "").endswith(("Font", "Font.set_default_family"))
        and row["render_policy"] == "full_typed_container_span"
        for row in typed_spans
    )


@pytest.mark.asyncio
async def test_recall_codebase_js_ts_repo_surfaces_symbols_tests_and_commands(tmp_path):
    repo = tmp_path / "js_ts_runtime_repo"
    (repo / "src").mkdir(parents=True)
    (repo / "tests").mkdir(parents=True)
    (repo / "package.json").write_text(
        json.dumps({"scripts": {"test": "vitest run", "typecheck": "tsc --noEmit"}}),
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
            """
        ),
        encoding="utf-8",
    )
    (repo / "tests" / "widget.test.ts").write_text(
        "import { WidgetView } from '../src/Widget';\n"
        "test('renders widget', () => { expect(WidgetView).toBeDefined(); });\n",
        encoding="utf-8",
    )
    server = _make_server(tmp_path, key="codebase_js_ts_runtime")

    await server.ingest_codebase(str(repo), source_id="repo", scope="agent-private")
    result = await server.recall(
        "Return a unified diff that updates src/Widget.tsx and keeps tests/widget.test.ts passing.",
        search_family="codebase",
        query_type="auto",
        query_metadata={
            "work_item_kind": "repo:work_item",
            "task_mode": "patch_generation",
            "output_artifact": "unified_diff",
            "required_render_mode": "patch_safe_source_context",
            "selected_test_files": ["tests/widget.test.ts"],
            "source_id": "repo",
        },
    )

    context = result["context"]
    assert "[File: src/Widget.tsx] [render_source=repo_file_blob_span] [whole_file=true]" in context
    assert "[File: tests/widget.test.ts] [render_source=repo_file_blob_span] [whole_file=true]" in context
    assert "export class WidgetController" in context
    assert "export const WidgetView" in context
    assert "test('renders widget'" in context
    kind_fqs = {row["kind_fq"] for row in server._container_graph["containers"]}
    assert {"code:import", "code:export", "code:interface", "code:type", "code:class", "code:function", "code:test_case", "code:command"} <= kind_fqs
    pack = next(row for row in result["repo_task_context_packs"] if row.get("context_pack_generation") == "query_specific")
    assert pack["sections"]["commands"]
    assert any(row["path"] == "tests/widget.test.ts" for row in pack["sections"]["test_source_files"])


@pytest.mark.asyncio
async def test_recall_codebase_go_repo_surfaces_types_methods_tests_and_commands(tmp_path):
    repo = tmp_path / "go_runtime_repo"
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
        "package demo\n\nimport \"testing\"\n\n"
        "func TestNewServer(t *testing.T) { if NewServer(nil) == nil { t.Fatal(\"nil\") } }\n",
        encoding="utf-8",
    )
    server = _make_server(tmp_path, key="codebase_go_runtime")

    await server.ingest_codebase(str(repo), source_id="repo", scope="agent-private")
    result = await server.recall(
        "Return a unified diff that updates service.go and keeps service_test.go passing.",
        search_family="codebase",
        query_type="auto",
        query_metadata={
            "work_item_kind": "repo:work_item",
            "task_mode": "patch_generation",
            "output_artifact": "unified_diff",
            "required_render_mode": "patch_safe_source_context",
            "selected_test_files": ["service_test.go"],
            "source_id": "repo",
        },
    )

    context = result["context"]
    assert "[File: service.go] [render_source=repo_file_blob_span] [whole_file=true]" in context
    assert "[File: service_test.go] [render_source=repo_file_blob_span] [whole_file=true]" in context
    assert "type Store interface" in context
    assert "func (s *Server) Save" in context
    assert "func TestNewServer" in context
    kind_fqs = {row["kind_fq"] for row in server._container_graph["containers"]}
    assert {"code:import", "code:interface", "code:class", "code:function", "code:method", "code:test_case", "code:command"} <= kind_fqs
    pack = next(row for row in result["repo_task_context_packs"] if row.get("context_pack_generation") == "query_specific")
    assert pack["sections"]["commands"]
    assert any(row["path"] == "service_test.go" for row in pack["sections"]["test_source_files"])


@pytest.mark.asyncio
async def test_ingest_codebase_materializes_ecosystem_commands_workflows_and_manifests(tmp_path):
    repo = _create_python_repo(tmp_path)
    (repo / "package.json").write_text(
        json.dumps({"scripts": {"test": "pytest", "lint": "ruff check ."}}),
        encoding="utf-8",
    )
    (repo / "Makefile").write_text("test:\n\tpytest\n", encoding="utf-8")
    (repo / "requirements.txt").write_text("pytest\n", encoding="utf-8")
    (repo / "go.mod").write_text("module example.com/app\n\ngo 1.22\n", encoding="utf-8")
    workflow_dir = repo / ".github" / "workflows"
    workflow_dir.mkdir(parents=True)
    (workflow_dir / "ci.yml").write_text(
        textwrap.dedent(
            """\
            name: CI
            jobs:
              test:
                runs-on: ubuntu-latest
                steps:
                  - run: pytest
            """
        ),
        encoding="utf-8",
    )
    server = _make_server(tmp_path, key="codebase_ecosystem_runtime")

    await server.ingest_codebase(str(repo), source_id="repo", scope="agent-private")

    graph = server._container_graph
    kind_fqs = {row["kind_fq"] for row in graph["containers"]}
    assert {"code:dependency_manifest", "code:command", "code:workflow", "code:workflow_job"} <= kind_fqs
    manifest_kinds = {
        (row.get("traits_json") or {}).get("payload", {}).get("manifest_kind")
        for row in graph["containers"]
        if row["kind_fq"] == "code:dependency_manifest"
    }
    assert {
        "npm_package_manifest",
        "make_manifest",
        "python_requirements_manifest",
        "go_module_manifest",
    } <= manifest_kinds
    relation_capabilities = {
        (row.get("traits_json") or {}).get("relation_capability")
        for row in graph["relations"]
    }
    assert {"command_targets", "workflow_runs"} <= relation_capabilities
    pack = next(
        row["payload_json"]
        for row in graph["artifacts"]
        if row["artifact_kind"] == "context_pack"
        and (row.get("payload_json") or {}).get("context_pack_kind") == "repo_task_context_pack"
    )
    assert pack["sections"]["commands"]
    assert pack["sections"]["workflows"]
    capability_contract = next(row for row in graph["contracts"] if row["contract_kind"] == "relation_capability_contract")
    assert "workflow_runs" in capability_contract["payload_json"]["relation_capabilities"]
    assert capability_contract["payload_json"]["mapped_relations"]["job_targets"] == "workflow_runs"


@pytest.mark.asyncio
async def test_ingest_codebase_materializes_generic_resource_config_files(tmp_path):
    repo = _create_python_repo(tmp_path)
    (repo / "Dockerfile").write_text("FROM python:3.11\n", encoding="utf-8")
    (repo / "docker-compose.yml").write_text("services:\n  app:\n    build: .\n", encoding="utf-8")
    (repo / "migrations").mkdir()
    (repo / "migrations" / "001_create_items.sql").write_text("CREATE TABLE items(id integer);\n", encoding="utf-8")
    (repo / "templates").mkdir()
    (repo / "templates" / "item.html").write_text("<h1>{{ item.name }}</h1>\n", encoding="utf-8")
    (repo / "locales").mkdir()
    (repo / "locales" / "en.properties").write_text("item.title=Item\n", encoding="utf-8")
    (repo / "openapi.yaml").write_text("openapi: 3.0.0\ninfo:\n  title: Demo\n", encoding="utf-8")
    server = _make_server(tmp_path, key="codebase_resource_config_runtime")

    await server.ingest_codebase(str(repo), source_id="repo", scope="agent-private")

    config_paths = {
        (row.get("traits_json") or {}).get("path")
        for row in server._container_graph["containers"]
        if row["kind_fq"] == "code:config"
    }
    assert {
        "Dockerfile",
        "docker-compose.yml",
        "migrations/001_create_items.sql",
        "templates/item.html",
        "locales/en.properties",
        "openapi.yaml",
    } <= config_paths


@pytest.mark.asyncio
async def test_ingest_codebase_materializes_js_ts_go_and_gap_reports_unsupported_vue(tmp_path):
    repo = _create_python_repo(tmp_path)
    (repo / "package.json").write_text(
        json.dumps({"scripts": {"test": "vitest"}}),
        encoding="utf-8",
    )
    (repo / "go.mod").write_text("module example.com/app\n\ngo 1.22\n", encoding="utf-8")
    (repo / "web").mkdir()
    (repo / "web" / "Widget.tsx").write_text("export function Widget() { return null }\n", encoding="utf-8")
    (repo / "web" / "Legacy.vue").write_text("<template><div /></template>\n", encoding="utf-8")
    (repo / "cmd").mkdir()
    (repo / "cmd" / "server.go").write_text("package main\nfunc main() {}\n", encoding="utf-8")
    server = _make_server(tmp_path, key="codebase_js_go_gap_runtime")

    await server.ingest_codebase(str(repo), source_id="repo", scope="agent-private")

    stage_meta = server._source_records["repo"]["source_meta"]["codebase_context"]
    gap_report = stage_meta["gap_report"]
    notes = gap_report["notes"]
    skipped_files = gap_report["skipped_files"]
    missing_capabilities = {
        row["capability"]: row for row in gap_report["missing_capabilities"]
    }
    assert "web/Widget.tsx" not in skipped_files
    assert "cmd/server.go" not in skipped_files
    assert "web/Legacy.vue" in skipped_files
    assert any("unsupported semantic plugin file: web/Legacy.vue" in note for note in notes)
    assert missing_capabilities["vue_code_graph"]["status"] == "explicitly_absent"
    assert "web/Legacy.vue" in missing_capabilities["vue_code_graph"]["files"]
    kind_fqs = {row["kind_fq"] for row in server._container_graph["containers"]}
    assert {"code:function", "code:command"} <= kind_fqs
    commands = [
        row
        for row in server._container_graph["containers"]
        if row["kind_fq"] == "code:command"
    ]
    command_names = {
        (row.get("traits_json") or {}).get("name")
        or ((row.get("traits_json") or {}).get("payload") or {}).get("name")
        for row in commands
    }
    assert any(str(name).endswith("npm:test") for name in command_names)
    assert any(str(name).endswith("go test") for name in command_names)


@pytest.mark.asyncio
async def test_recall_codebase_precise_query_hydrates_whole_file(tmp_path, monkeypatch):
    repo = _create_python_repo(tmp_path)
    server = _make_server(tmp_path)
    await server.ingest_codebase(str(repo), source_id="repo", scope="agent-private")

    hydrate_calls = []
    original = CodebaseSemanticSidecarStore.hydrate_sidecar

    def _recording_hydrate(self, sidecar_ref):
        hydrate_calls.append(sidecar_ref["sidecar_id"])
        return original(self, sidecar_ref)

    monkeypatch.setattr(CodebaseSemanticSidecarStore, "hydrate_sidecar", _recording_hydrate)

    result = await server.recall("Show the exact code for issue signature", search_family="codebase")

    trace = result["runtime_trace"]["codebase_augmentation"]
    assert "--- SOURCE FILES ---" in result["context"]
    assert "[File: pkg/service.py]" in result["context"]
    assert "class Permit:" in result["context"]
    assert "def issue(name: str) -> Permit:" in result["context"]
    assert hydrate_calls
    assert trace["mode"] == "whole_file"
    assert trace["selected_file"] == "pkg/service.py"
    assert trace["highlight_spans"]


@pytest.mark.asyncio
async def test_recall_codebase_precise_query_large_file_uses_windowed_file(tmp_path):
    repo = _create_large_python_repo(tmp_path)
    server = _make_server(tmp_path, key="windowed_codebase_runtime")
    await server.ingest_codebase(str(repo), source_id="repo", scope="agent-private")

    result = await server.recall("Show the exact code for cinder_signature", search_family="codebase")

    trace = result["runtime_trace"]["codebase_augmentation"]
    assert "--- SOURCE FILES ---" in result["context"]
    assert "[File: pkg/huge_service.py]" in result["context"]
    assert "[Window:" in result["context"]
    assert "def cinder_signature() -> str:" in result["context"]
    assert "[Window: L1-L2209]" not in result["context"]
    assert "   1: # filler 1" not in result["context"]
    assert trace["mode"] == "windowed_file"
    assert trace["selected_file"] == "pkg/huge_service.py"
    assert trace["truncation_mode"] == "large_file_window"
    assert trace["line_count"] > 2000


@pytest.mark.asyncio
async def test_recall_codebase_precise_query_degrades_cleanly_when_file_sidecar_missing(tmp_path):
    repo = _create_python_repo(tmp_path)
    server = _make_server(tmp_path, key="missing_sidecar_runtime")
    await server.ingest_codebase(str(repo), source_id="repo", scope="agent-private")

    for fact in _active_codebase_facts(server, "repo"):
        file_sidecar_ref = fact.get("file_sidecar_ref")
        if not isinstance(file_sidecar_ref, dict):
            continue
        sidecar_path = Path(server.data_dir) / str(file_sidecar_ref["storage_ref"])
        if sidecar_path.exists():
            sidecar_path.unlink()

    result = await server.recall("Show the exact code for issue signature", search_family="codebase")

    trace = result["runtime_trace"]["codebase_augmentation"]
    assert "--- SOURCE FILES ---" not in result["context"]
    assert "pkg.service.issue" in result["context"]
    assert trace["mode"] == "hot_only"
    assert trace["reason"] == "file_hydration_failed"
    assert trace["failed_sidecars"]


@pytest.mark.asyncio
async def test_recall_codebase_precise_query_degrades_cleanly_when_file_sidecar_corrupted(tmp_path):
    repo = _create_python_repo(tmp_path)
    server = _make_server(tmp_path, key="corrupt_sidecar_runtime")
    await server.ingest_codebase(str(repo), source_id="repo", scope="agent-private")

    for fact in _active_codebase_facts(server, "repo"):
        file_sidecar_ref = fact.get("file_sidecar_ref")
        if not isinstance(file_sidecar_ref, dict):
            continue
        sidecar_path = Path(server.data_dir) / str(file_sidecar_ref["storage_ref"])
        sidecar_path.write_text("{not-json", encoding="utf-8")

    result = await server.recall("Show the exact code for issue signature", search_family="codebase")

    trace = result["runtime_trace"]["codebase_augmentation"]
    assert "--- SOURCE FILES ---" not in result["context"]
    assert trace["mode"] == "hot_only"
    assert trace["reason"] == "file_hydration_failed"


@pytest.mark.asyncio
async def test_recall_codebase_precise_query_prefers_defining_file_over_crowded_symbol_noise(tmp_path):
    repo = _create_crowded_symbol_repo(tmp_path)
    server = _make_server(tmp_path, key="crowded_symbol_runtime")
    await server.ingest_codebase(str(repo), source_id="repo", scope="agent-private")

    crowded_matches = [
        fact
        for fact in _active_codebase_facts(server, "repo")
        if any("build_codebase_semantic_bundle" in entity for entity in (fact.get("entities") or []))
    ]
    assert len(crowded_matches) > 10

    result = await server.recall(
        "Show the exact code for build_codebase_semantic_bundle in src/codebase_semantic_plugins/runner.py.",
        search_family="codebase",
    )

    trace = result["runtime_trace"]["codebase_augmentation"]
    assert "--- SOURCE FILES ---" in result["context"]
    assert "[File: src/codebase_semantic_plugins/runner.py]" in result["context"]
    assert "def build_codebase_semantic_bundle(path: str) -> str:" in result["context"]
    assert trace["mode"] == "whole_file"
    assert trace["selected_file"] == "src/codebase_semantic_plugins/runner.py"
    assert trace["highlight_spans"]


@pytest.mark.asyncio
async def test_recall_codebase_precise_query_without_path_prefers_defining_file_over_importers(tmp_path):
    repo = _create_cross_file_symbol_repo(tmp_path)
    server = _make_server(tmp_path, key="cross_file_symbol_runtime")
    await server.ingest_codebase(str(repo), source_id="repo", scope="agent-private")

    result = await server.recall(
        "Show the exact code for build_codebase_semantic_bundle",
        search_family="codebase",
    )

    trace = result["runtime_trace"]["codebase_augmentation"]
    assert "--- SOURCE FILES ---" in result["context"]
    assert "[File: pkg/runner.py]" in result["context"]
    assert "def build_codebase_semantic_bundle(path: str) -> str:" in result["context"]
    source_files_section = result["context"].split("--- SOURCE FILES ---", 1)[1]
    assert "[File: callers/consumer_00.py]" not in source_files_section
    assert trace["mode"] == "whole_file"
    assert trace["selected_file"] == "pkg/runner.py"
    assert trace["highlight_spans"]


@pytest.mark.asyncio
async def test_ingest_codebase_isolates_plugin_failure_and_keeps_python_results(tmp_path, monkeypatch):
    repo = _create_python_repo(tmp_path)
    (repo / "Cargo.toml").write_text(
        "[package]\nname = \"broken_rust_probe\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
        encoding="utf-8",
    )
    (repo / "src").mkdir(exist_ok=True)
    (repo / "src" / "lib.rs").write_text("pub fn broken() {}\n", encoding="utf-8")

    class BrokenRustPlugin:
        plugin_name = "broken_rust"
        supported_extensions = frozenset({".rs"})

        def build_bundle(self, *, repo_root: Path, repo_id: str, revision: str, files: list[Path]) -> dict:
            raise RuntimeError("rust toolchain unavailable")

    monkeypatch.setattr(
        "src.codebase_semantic_plugins.runner.get_codebase_semantic_plugins",
        lambda: [PYTHON_SEMANTIC_PLUGIN, BrokenRustPlugin()],
    )

    server = _make_server(tmp_path, key="plugin_failure_runtime")
    result = await server.ingest_codebase(str(repo), source_id="repo", scope="agent-private")

    assert result["status"] == "ok"
    assert "python" in result["languages"]
    stage_meta = server._source_records["repo"]["source_meta"]["codebase_context"]
    notes = stage_meta["gap_report"]["notes"]
    skipped_files = stage_meta["gap_report"]["skipped_files"]
    assert any("broken_rust" in note for note in notes)
    assert any(path.endswith("src/lib.rs") for path in skipped_files)
    assert any("pkg.service.issue" in fact.get("fact", "") for fact in _active_codebase_facts(server, "repo"))


@pytest.mark.asyncio
async def test_ingest_codebase_rust_per_manifest_degrade_keeps_good_crate(tmp_path, monkeypatch):
    repo = _create_multi_manifest_rust_repo(tmp_path)

    def _fake_rust_bundle(manifest_path: Path, repo_id: str, revision: str) -> dict:
        crate_name = manifest_path.parent.name
        rel_file = manifest_path.parent.relative_to(repo).as_posix() + "/src/lib.rs"
        if crate_name == "broken_crate":
            raise RuntimeError("rustdoc failed")
        return {
            "objects": [
                {
                    "id": f"obj_module_{crate_name}",
                    "object_type": "module",
                    "repo_id": repo_id,
                    "revision": revision,
                    "file_path": rel_file,
                    "span": {"start_line": 1, "end_line": 3, "start_col": 0, "end_col": None},
                    "language": "rust",
                    "analyzer_id": "rustdoc",
                    "analyzer_version": "test",
                    "derivation_type": "observed",
                    "payload": {"name": crate_name, "qualified_name": crate_name},
                },
                {
                    "id": f"obj_callable_{crate_name}",
                    "object_type": "callable",
                    "repo_id": repo_id,
                    "revision": revision,
                    "file_path": rel_file,
                    "span": {"start_line": 1, "end_line": 3, "start_col": 0, "end_col": None},
                    "language": "rust",
                    "analyzer_id": "rustdoc",
                    "analyzer_version": "test",
                    "derivation_type": "observed",
                    "payload": {"name": "issue", "qualified_name": f"{crate_name}::issue", "signature": "pub fn issue() -> &'static str"},
                },
            ],
            "relations": [
                {
                    "id": f"rel_declares_{crate_name}",
                    "relation_type": "declares",
                    "from_id": f"obj_module_{crate_name}",
                    "to_id": f"obj_callable_{crate_name}",
                    "repo_id": repo_id,
                    "revision": revision,
                    "file_path": rel_file,
                    "span": {"start_line": 1, "end_line": 3, "start_col": 0, "end_col": None},
                    "language": "rust",
                    "analyzer_id": "rustdoc",
                    "analyzer_version": "test",
                    "derivation_type": "observed",
                    "payload": {"from_type": "module", "to_type": "callable"},
                }
            ],
            "provenance": {
                "repo_id": repo_id,
                "revision": revision,
                "generated_at": "2026-04-13T10:00:00Z",
                "source_root": str(repo),
            },
            "capability_report": {
                "supported_languages": ["rust"],
                "supported_capabilities": ["module_graph"],
                "analyzer_protocol_version": "codebase-v1",
                "schema_version": "1",
            },
            "gap_report": {"skipped_files": [], "notes": []},
            "sidecars": [
                {
                    "sidecar_id": f"sc_module_{crate_name}",
                    "sidecar_kind": "semantic_snapshot",
                    "format_family": "rustdoc",
                    "format_name": "rustdoc_fragment_v1",
                    "format_version": "1",
                    "encoding": "utf-8",
                    "compression": "none",
                    "repo_id": repo_id,
                    "revision": revision,
                    "file_path": rel_file,
                    "span": {"start_line": 1, "end_line": 3, "start_col": 0, "end_col": None},
                    "node_id": f"obj_module_{crate_name}",
                    "storage_ref": f"inline:sc_module_{crate_name}",
                    "content_hash": "pending",
                    "byte_size": 16,
                    "producer": "rustdoc",
                    "metadata": {"language": "rust", "node_kind": "module"},
                    "payload": {
                        "fragment_id": f"frag_module_{crate_name}",
                        "root_node_id": f"obj_module_{crate_name}",
                        "root_kind": "rust_source_file",
                        "file_path": rel_file,
                        "span": {"start_line": 1, "end_line": 3, "start_col": 0, "end_col": None},
                        "parent_node_id": None,
                        "child_node_ids": [],
                        "payload_ref": None,
                        "code": f"pub fn issue() -> &'static str {{\n    \"{crate_name}\"\n}}",
                    },
                }
            ],
        }

    monkeypatch.setattr(
        "src.codebase_semantic_plugins.rust_plugin._rustdoc_bundle_for_manifest",
        _fake_rust_bundle,
    )

    server = _make_server(tmp_path, key="rust_manifest_degrade_runtime")
    result = await server.ingest_codebase(str(repo), source_id="repo", scope="agent-private")

    assert result["status"] == "ok"
    assert "rust" in result["languages"]
    stage_meta = server._source_records["repo"]["source_meta"]["codebase_context"]
    notes = stage_meta["gap_report"]["notes"]
    skipped_files = stage_meta["gap_report"]["skipped_files"]
    assert any("crates/broken_crate/Cargo.toml" in note for note in notes)
    assert any(path.endswith("crates/broken_crate/src/lib.rs") for path in skipped_files)
    facts = _active_codebase_facts(server, "repo")
    assert any("good_crate::issue" in fact.get("fact", "") for fact in facts)
    assert not any("broken_crate::issue" in fact.get("fact", "") for fact in facts)


@pytest.mark.asyncio
@pytest.mark.skipif(shutil.which("cargo") is None or shutil.which("rustdoc") is None, reason="Rust toolchain unavailable")
async def test_ingest_codebase_supports_real_rust_plugin(tmp_path):
    repo = _create_rust_repo(tmp_path)
    server = _make_server(tmp_path, key="codebase_rust_runtime")

    result = await server.ingest_codebase(str(repo), source_id="rust_repo", scope="agent-private")

    assert result["status"] == "ok"
    assert "rust" in result["languages"]
    facts = [fact for fact in server._all_granular if fact.get("source_id") == "rust_repo"]
    assert any("rust_probe::Permit" in fact.get("fact", "") for fact in facts)
