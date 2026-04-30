# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.episode_extraction import build_singleton_episodes
from src.ingest import ingest_input
from src.memory import MemoryServer
from src.codebase_semantic_sidecars import CodebaseSemanticSidecarStore
from tests._memory_llm_mocks import patch_memory_llm_runtime


def _semantic_vec(text: str, *, dim: int = 24) -> np.ndarray:
    lowered = text.lower()
    keys = [
        "sam",
        "planning",
        "conversation",
        "report",
        "checklist",
        "document",
        "permit",
        "issue",
        "audit",
        "signature",
        "code",
        "exact",
        "cinder-42",
        "platform",
        "reliability",
        "owner",
        "team",
        "codename",
        "qualified",
        "python",
        "function",
        "leather",
        "jacket",
        "sales",
        "statistics",
        "auth",
        "payment",
    ]
    vec = np.zeros(dim, dtype=np.float32)
    for idx, key in enumerate(keys):
        if key in lowered:
            vec[idx] = 1.0
    if vec.sum() == 0.0:
        vec[-1] = 1.0
    return vec


async def _mock_embed_texts(texts, **kwargs):
    return np.stack([_semantic_vec(text) for text in texts]).astype(np.float32)


async def _mock_embed_query(text, **kwargs):
    return _semantic_vec(text).astype(np.float32)


def _make_server(tmp_path: Path, key: str = "codebase_e2e") -> MemoryServer:
    server = MemoryServer(str(tmp_path / "data"), key, extract_model="groq/qwen30b")
    server._embed_texts_with_runtime_secrets = _mock_embed_texts
    server._embed_query_with_runtime_secrets = _mock_embed_query
    server._profile_configs = {
        "qwen": {
            "model": "groq/qwen30b",
            "embed_model": "chatgpt",
        }
    }
    return server


def _create_codebase_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "code_repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "pyproject.toml").write_text("[project]\nname='code_repo'\n", encoding="utf-8")
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
    return repo


def _create_cinder_codebase_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "cinder_code_repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "pyproject.toml").write_text("[project]\nname='cinder_code_repo'\n", encoding="utf-8")
    (repo / "pkg" / "service.py").write_text(
        textwrap.dedent(
            """\
            def cinder_signature() -> str:
                return "sig:CINDER-42"
            """
        ),
        encoding="utf-8",
    )
    return repo


def _create_leather_codebase_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "leather_code_repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "pyproject.toml").write_text("[project]\nname='leather-code-repo'\n", encoding="utf-8")
    (repo / "pkg" / "leather_jacket_sales.py").write_text(
        textwrap.dedent(
            """\
            LEATHER_JACKET_SALES_STATISTICS = {"reviews": 128, "sales": 42}

            def leather(jacket):
                return LEATHER_JACKET_SALES_STATISTICS
            """
        ),
        encoding="utf-8",
    )
    return repo


def _create_unrelated_codebase_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "unrelated_code_repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "pyproject.toml").write_text("[project]\nname='unrelated-code-repo'\n", encoding="utf-8")
    (repo / "pkg" / "payments.py").write_text(
        textwrap.dedent(
            """\
            def authorize_payment(user_id: str, cents: int) -> bool:
                return bool(user_id and cents > 0)

            def rotate_auth_token(token: str) -> str:
                return token[::-1]
            """
        ),
        encoding="utf-8",
    )
    return repo


def _patch_public_runtime(monkeypatch):
    async def mock_extract_session(**kwargs):
        text = str(kwargs.get("session_text") or "")
        session_num = kwargs.get("session_num", 1)
        lowered = text.lower()
        if "sam discussed issue planning" in lowered:
            fact_text = "Sam discussed issue planning in conversation."
            entities = ["Sam", "issue planning"]
        elif "incident codename is cinder-42" in lowered:
            fact_text = "The chat says the incident codename is CINDER-42."
            entities = ["CINDER-42", "incident codename"]
        elif "permit checklist is required for issue triage" in lowered:
            fact_text = "The document says the permit checklist is required for issue triage."
            entities = ["permit checklist", "issue triage"]
        elif "owner team for cinder-42 is platform reliability" in lowered:
            fact_text = "The document says the owner team for CINDER-42 is Platform Reliability."
            entities = ["CINDER-42", "Platform Reliability", "owner team"]
        elif "leather jacket reviews" in lowered:
            fact_text = "The conversation says Leather Jacket reviews are rising."
            entities = ["Leather Jacket", "reviews"]
        elif "leather jacket product brief" in lowered:
            fact_text = "The document describes Leather Jacket product demand."
            entities = ["Leather Jacket", "product demand"]
        else:
            fact_text = text.splitlines()[0].strip() if text.strip() else "fallback fact"
            entities = ["generic"]
        facts = [
            {
                "id": "f0",
                "fact": fact_text,
                "kind": "fact",
                "entities": entities,
                "tags": ["test"],
                "session": session_num,
            }
        ]
        return ("conv", session_num, kwargs.get("session_date", "2024-06-01"), facts, [])

    async def mock_group_document(model, source_id, title, date, block_dicts, grouping_config, sem):
        return build_singleton_episodes(source_id, date, block_dicts), {"mode": "mock"}, "mock_singleton"

    monkeypatch.setattr("src.memory.extract_session", mock_extract_session)
    monkeypatch.setattr("src.memory.resolve_supersession", lambda facts, lookup: None)
    monkeypatch.setattr("src.memory.group_document", mock_group_document)
    patch_memory_llm_runtime(monkeypatch)


def _patch_inference_capture(monkeypatch, *, answer: str) -> dict[str, str]:
    seen: dict[str, str] = {}

    async def mock_call_oai(
        model,
        prompt,
        max_tokens=300,
        json_mode=False,
        temperature=0,
        semaphore=None,
    ):
        del model, max_tokens, json_mode, temperature, semaphore
        seen["prompt"] = str(prompt)
        return answer

    async def mock_call_model(
        model,
        messages,
        max_tokens=300,
        temperature=0.0,
        json_mode=False,
    ):
        del model, max_tokens, temperature, json_mode
        seen["prompt"] = "\n".join(str(message.get("content") or "") for message in messages if isinstance(message, dict))
        return answer

    monkeypatch.setattr("src.memory.call_oai", mock_call_oai)
    monkeypatch.setattr("src.common.call_oai", mock_call_oai)
    monkeypatch.setattr("src.memory._call_model", mock_call_model)
    monkeypatch.setattr("src.common._call_model", mock_call_model)
    return seen


async def _build_mixed_corpus(server: MemoryServer, tmp_path: Path, monkeypatch) -> Path:
    _patch_public_runtime(monkeypatch)
    repo = _create_codebase_repo(tmp_path)
    await server.store(
        "User: Sam discussed issue planning in conversation.\nAssistant: noted",
        session_num=1,
        session_date="2024-06-01",
        source_id="CHAT",
        scope="agent-private",
    )
    await server.ingest_document(
        "# Issue Report\n\nThe permit checklist is required for issue triage.",
        source_id="DOC",
        scope="agent-private",
    )
    await server.ingest_codebase(str(repo), source_id="CODE", scope="agent-private")
    return repo


async def _build_cinder_mixed_corpus(server: MemoryServer, tmp_path: Path, monkeypatch) -> Path:
    _patch_public_runtime(monkeypatch)
    repo = _create_cinder_codebase_repo(tmp_path)
    await server.store(
        "User: The incident codename is CINDER-42. Assistant: Noted.",
        session_num=1,
        session_date="2024-06-01",
        source_id="CHAT",
        scope="agent-private",
    )
    await server.ingest_document(
        "# Runbook\n\nRunbook: the owner team for CINDER-42 is Platform Reliability.",
        source_id="DOC",
        scope="agent-private",
    )
    await server.ingest_codebase(str(repo), source_id="CODE", scope="agent-private")
    return repo


async def _build_leather_multifamily_corpus(
    server: MemoryServer,
    tmp_path: Path,
    monkeypatch,
    *,
    codebase_related: bool = True,
) -> Path:
    _patch_public_runtime(monkeypatch)
    repo = _create_leather_codebase_repo(tmp_path) if codebase_related else _create_unrelated_codebase_repo(tmp_path)
    await server.store(
        "User: Leather Jacket reviews are rising. Assistant: Noted.",
        session_num=1,
        session_date="2024-06-01",
        source_id="CHAT",
        scope="agent-private",
    )
    await server.ingest_document(
        "# Leather Jacket Product Brief\n\nLeather Jacket product brief: demand is up.",
        source_id="DOC",
        scope="agent-private",
    )
    await server.ingest_codebase(str(repo), source_id="CODE", scope="agent-private")
    return repo


@pytest.mark.asyncio
async def test_ingest_input_codebase_manifest_only_repo_uses_ecosystem_semantics(tmp_path):
    repo = tmp_path / "manifest_repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname='manifest-only'\n", encoding="utf-8")
    server = _make_server(tmp_path, key="manifest_ingest_input")

    result = await ingest_input(server, path=str(repo), scope="agent-private")

    assert result["status"] == "ok"
    assert result["source_family"] == "codebase"
    assert result["codebase_stage"] == "codebase_semantic"
    assert result["container_graph_status"] == "active"
    assert any(row["kind_fq"] == "code:dependency_manifest" for row in server._container_graph["containers"])


@pytest.mark.asyncio
async def test_codebase_ingest_does_not_create_conversation_or_document_artifacts(tmp_path):
    server = _make_server(tmp_path, key="codebase_only_isolation")
    repo = _create_codebase_repo(tmp_path)

    result = await server.ingest_codebase(str(repo), source_id="CODE", scope="agent-private")

    assert result["status"] == "ok"
    assert server._raw_sessions == []
    assert not server._episode_corpus.get("documents")
    assert set(server._source_records) == {"CODE"}
    assert server._source_records["CODE"]["family"] == "codebase"
    assert server._source_records["CODE"]["source_meta"]["codebase_context"]["object_count"] > 0
    assert all(str(fact.get("source_family") or "").lower() == "codebase" for fact in server._all_granular)


@pytest.mark.asyncio
async def test_conversation_and_document_ingest_do_not_create_codebase_artifacts(tmp_path, monkeypatch):
    server = _make_server(tmp_path, key="non_codebase_isolation")
    _patch_public_runtime(monkeypatch)

    await server.store(
        "User: Sam discussed issue planning in conversation.\nAssistant: noted",
        session_num=1,
        session_date="2024-06-01",
        source_id="CHAT",
        scope="agent-private",
    )
    await server.ingest_document(
        "# Issue Report\n\nThe permit checklist is required for issue triage.",
        source_id="DOC",
        scope="agent-private",
    )

    assert server._source_records["CHAT"]["family"] == "conversation"
    assert server._source_records["DOC"]["family"] == "document"
    assert "codebase_context" not in server._source_records["CHAT"].get("source_meta", {})
    assert "codebase_context" not in server._source_records["DOC"].get("source_meta", {})
    assert all("sidecar_ref" not in fact for fact in server._all_granular)
    assert not (Path(server.data_dir) / "codebase_semantic_sidecars").exists()


@pytest.mark.asyncio
async def test_code_query_with_family_hint_stays_in_codebase_lane(tmp_path, monkeypatch):
    server = _make_server(tmp_path, key="codebase_lane")
    await _build_mixed_corpus(server, tmp_path, monkeypatch)

    result = await server.recall("Show the exact code for issue signature", search_family="codebase")

    assert "pkg.service.issue" in result["context"]
    assert "--- SOURCE FILES ---" in result["context"]
    assert "[File: pkg/service.py]" in result["context"]
    assert "class Permit:" in result["context"]
    assert "(callable)" in result["context"] or "(declares)" in result["context"]
    assert "Sam discussed issue planning in conversation." not in result["context"]
    assert "The document says the permit checklist is required for issue triage." not in result["context"]
    assert result["runtime_trace"]["codebase_augmentation"]["mode"] == "whole_file"
    assert result["runtime_trace"]["family_discovery"]["source_hydration"]["hydrated"] is True
    assert result["runtime_trace"]["family_discovery"]["source_hydration"]["hydrated"] is True


@pytest.mark.asyncio
async def test_conversation_query_with_family_hint_stays_out_of_codebase_lane(tmp_path, monkeypatch):
    server = _make_server(tmp_path, key="conversation_lane")
    await _build_mixed_corpus(server, tmp_path, monkeypatch)

    result = await server.recall("What did Sam discuss about issue planning?", search_family="conversation")

    assert "Sam discussed issue planning in conversation." in result["context"]
    assert "--- SOURCE FILES ---" not in result["context"]
    assert "[File: pkg/service.py]" not in result["context"]
    assert "(S1)" in result["context"]
    assert result["runtime_trace"]["codebase_augmentation"]["mode"] == "inactive"
    assert result["runtime_trace"]["family_discovery"]["per_family"]["codebase"]["searched"] is False


@pytest.mark.asyncio
async def test_document_query_with_family_hint_stays_out_of_codebase_lane(tmp_path, monkeypatch):
    server = _make_server(tmp_path, key="document_lane")
    await _build_mixed_corpus(server, tmp_path, monkeypatch)

    result = await server.recall("What does the report say about the permit checklist?", search_family="document")

    assert "The document says the permit checklist is required for issue triage." in result["context"]
    assert "--- SOURCE FILES ---" not in result["context"]
    assert "[File: pkg/service.py]" not in result["context"]
    assert result["runtime_trace"]["codebase_augmentation"]["mode"] == "inactive"
    assert result["runtime_trace"]["family_discovery"]["per_family"]["codebase"]["searched"] is False


@pytest.mark.asyncio
async def test_generic_mixed_query_does_not_hydrate_code_when_only_conversation_fact_matches(tmp_path, monkeypatch):
    server = _make_server(tmp_path, key="generic_no_code_hydration")
    await _build_mixed_corpus(server, tmp_path, monkeypatch)

    result = await server.recall("What did Sam discuss about issue planning?")

    assert "Sam discussed issue planning in conversation." in result["context"]
    assert "--- SOURCE FILES ---" not in result["context"]
    assert result["runtime_trace"]["codebase_augmentation"]["mode"] == "inactive"


@pytest.mark.asyncio
async def test_generic_auto_short_query_discovers_codebase_without_code_markers(tmp_path, monkeypatch):
    server = _make_server(tmp_path, key="generic_auto_leather_codebase")
    await _build_leather_multifamily_corpus(server, tmp_path, monkeypatch)

    result = await server.recall("Leather Jacket")

    assert "The conversation says Leather Jacket reviews are rising." in result["context"]
    assert "The document describes Leather Jacket product demand." in result["context"]
    assert "--- CODEBASE FACTS ---" in result["context"]
    assert "def leather(jacket):" in result["context"]
    assert "--- SOURCE FILES ---" not in result["context"]
    trace = result["runtime_trace"]["family_discovery"]
    assert {"conversation", "document", "codebase"} <= set(trace["available_families"])
    assert "codebase" in trace["searched_families"]
    assert trace["per_family"]["codebase"]["mode"] == "codebase_cheap_probe"
    assert trace["per_family"]["codebase"]["candidate_count"] > 0
    assert trace["per_family"]["codebase"]["selected_count"] > 0
    assert trace["per_family"]["codebase"]["score_summary"]["threshold"]["min_overlap"] == 2
    assert trace["source_hydration"]["hydrated"] is False


@pytest.mark.asyncio
async def test_codebase_hot_recall_continuation_returns_source_window_without_first_page_hydration(tmp_path, monkeypatch):
    server = _make_server(tmp_path, key="codebase_continuation_source_window")
    repo = _create_leather_codebase_repo(tmp_path)
    await server.ingest_codebase(str(repo), source_id="CODE", scope="agent-private")

    hydrate_calls: list[str] = []
    original = CodebaseSemanticSidecarStore.hydrate_sidecar

    def _recording_hydrate(self, sidecar_ref):
        hydrate_calls.append(str(sidecar_ref.get("sidecar_id") or ""))
        return original(self, sidecar_ref)

    monkeypatch.setattr(CodebaseSemanticSidecarStore, "hydrate_sidecar", _recording_hydrate)

    result = await server.recall("Leather Jacket sales statistics", search_family="codebase")

    assert "LEATHER_JACKET_SALES_STATISTICS" in result["context"]
    assert "--- SOURCE FILES ---" not in result["context"]
    assert hydrate_calls == []
    continuation = result["recall_continuation"]
    assert continuation["available"] is True
    assert continuation["typed_entry_counts"]["codebase_source"] > 0
    assert result["runtime_trace"]["recall_continuation_trace"]["families_with_continuation"] == ["codebase"]
    for continuation_page in result["_recall_continuation_pages"]:
        assert "context" not in continuation_page
        for entry in continuation_page["typed_entries"]:
            assert "raw" not in entry
            assert "content" not in entry
            sidecar_ref = entry.get("file_sidecar_ref") or {}
            assert "code" not in sidecar_ref
            assert "text" not in sidecar_ref

    page = server.recall_continuation_page(
        continuation_handle=continuation["handle"],
        page="next",
        caller_id="system",
        caller_memberships=[],
        caller_role="user",
        swarm_id="default",
    )

    assert "CODEBASE SOURCE WINDOWS:" in page["context"]
    assert "[Mode: source_window]" in page["context"]
    assert "LEATHER_JACKET_SALES_STATISTICS" in page["context"]
    assert "--- SOURCE FILES ---" not in page["context"]
    assert hydrate_calls
    page_trace = page["runtime_trace"]["recall_continuation_trace"]
    assert page_trace["typed_entry_counts"]["codebase_source"] > 0
    assert page_trace["source_hydration"]["hydrated"] is True


@pytest.mark.asyncio
async def test_memory_ask_tool_continuation_hydrates_codebase_source_windows(tmp_path, monkeypatch):
    server = _make_server(tmp_path, key="codebase_ask_continuation_source_window")
    server._profiles = {1: "qwen"}
    server._profile_configs["qwen"].update({
        "context_window": 128000,
        "max_output_tokens": 512,
        "secret_ref": {"name": "test-runtime-secret", "scope": "system-wide"},
        "pricing": {"input_per_1k": 0.0, "output_per_1k": 0.0},
    })
    repo = _create_leather_codebase_repo(tmp_path)
    await server.ingest_codebase(str(repo), source_id="CODE", scope="agent-private")

    hydrate_calls: list[str] = []
    original = CodebaseSemanticSidecarStore.hydrate_sidecar

    def _recording_hydrate(self, sidecar_ref):
        hydrate_calls.append(str(sidecar_ref.get("sidecar_id") or ""))
        return original(self, sidecar_ref)

    monkeypatch.setattr(CodebaseSemanticSidecarStore, "hydrate_sidecar", _recording_hydrate)

    class _FakeCompletions:
        def __init__(self):
            self.calls = 0

        async def create(self, **request):
            self.calls += 1
            if self.calls == 1:
                tool_call = SimpleNamespace(
                    id="call_more_context",
                    function=SimpleNamespace(
                        name="get_more_context",
                        arguments=json.dumps({"page": "next"}),
                    ),
                )
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[tool_call]))]
                )

            tool_messages = [
                str(message.get("content") or "")
                for message in request.get("messages", [])
                if isinstance(message, dict) and message.get("role") == "tool"
            ]
            assert tool_messages
            tool_payload = "\n".join(tool_messages)
            assert "CODEBASE SOURCE WINDOWS:" in tool_payload
            assert "[Mode: source_window]" in tool_payload
            assert "LEATHER_JACKET_SALES_STATISTICS" in tool_payload
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content="LEATHER_JACKET_SALES_STATISTICS",
                            tool_calls=[],
                        )
                    )
                ]
            )

    fake_completions = _FakeCompletions()
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=fake_completions))
    monkeypatch.setattr(server, "_get_model_client_with_runtime_secrets", lambda *_args, **_kwargs: fake_client)

    result = await server.ask(
        "Leather Jacket sales statistics",
        search_family="codebase",
        use_tool=True,
    )

    assert result["tool_called"] is True
    assert fake_completions.calls == 2
    assert hydrate_calls
    tool_result = json.loads(result["tool_results"][0]["result"])
    assert "CODEBASE SOURCE WINDOWS:" in tool_result["result"]
    assert "LEATHER_JACKET_SALES_STATISTICS" in tool_result["result"]
    assert tool_result["runtime_trace"]["recall_continuation_trace"]["source_hydration"]["hydrated"] is True


def test_codebase_continuation_falls_back_to_fact_refs_when_source_window_unavailable(tmp_path):
    server = MemoryServer(str(tmp_path / "data"), "codebase_continuation_fact_fallback")
    fact = {
        "id": "code-fact-no-source",
        "fact": "Codebase config declares Leather Jacket sale statistics in generated metadata.",
        "kind": "fact",
        "source_id": "CODE",
        "source_family": "codebase",
        "scope": "agent-private",
        "owner_id": "system",
        "agent_id": "default",
        "swarm_id": "default",
    }
    server._source_records["CODE"] = {
        "source_id": "CODE",
        "family": "codebase",
        "scope": "agent-private",
        "owner_id": "system",
        "agent_id": "default",
        "swarm_id": "default",
        "read": [],
        "write": [],
    }
    server._all_granular = [fact]

    result = server._attach_recall_continuation(
        query="Leather Jacket statistics",
        result={
            "context": "RETRIEVED FACTS:\n- Codebase config declares Leather Jacket sale statistics.",
            "retrieved": [fact],
            "query_type": "lookup",
            "search_family": "codebase",
            "retrieval_families": ["codebase"],
            "runtime_trace": {},
        },
        fact_filter=lambda row: server._acl_allows(row, "system", [], "user"),
        caller_id="system",
        caller_memberships=[],
        caller_role="user",
        swarm_id="default",
        raw_kind="all",
    )

    assert result["recall_continuation"]["available"] is True
    assert result["recall_continuation"]["typed_entry_counts"] == {"codebase_fact": 1}
    assert "no_codebase_sidecar_ref" in result["runtime_trace"]["recall_continuation_trace"]["skipped_reasons"]

    page = server.recall_continuation_page(
        continuation_handle=result["recall_continuation"]["handle"],
        page="next",
        caller_id="system",
        caller_memberships=[],
        caller_role="user",
        swarm_id="default",
    )

    assert "CODEBASE FACTS:" in page["context"]
    assert "Leather Jacket sale statistics" in page["context"]
    assert "source_window_reason=no_codebase_sidecar_ref" in page["context"]
    assert page["runtime_trace"]["recall_continuation_trace"]["source_hydration"]["hydrated"] is False


@pytest.mark.asyncio
async def test_generic_auto_mixed_recall_uses_one_family_agnostic_continuation_handle(tmp_path, monkeypatch):
    server = _make_server(tmp_path, key="generic_auto_single_continuation_handle")
    await _build_leather_multifamily_corpus(server, tmp_path, monkeypatch)

    result = await server.recall("Leather Jacket")

    assert "The conversation says Leather Jacket reviews are rising." in result["context"]
    assert "The document describes Leather Jacket product demand." in result["context"]
    assert "--- CODEBASE FACTS ---" in result["context"]
    continuation = result["recall_continuation"]
    assert continuation["available"] is True
    assert continuation["tool"] == "get_more_context"
    assert continuation["typed_entry_counts"]["codebase_source"] > 0

    page = server.recall_continuation_page(
        continuation_handle=continuation["handle"],
        page="next",
        caller_id="system",
        caller_memberships=[],
        caller_role="user",
        swarm_id="default",
    )

    assert "CODEBASE SOURCE WINDOWS:" in page["context"]
    assert "The conversation says Leather Jacket reviews are rising." not in page["context"]
    assert page["runtime_trace"]["recall_continuation_trace"]["handle_version"] == 2
    assert page["runtime_trace"]["recall_continuation_trace"]["has_more"] in {True, False}


@pytest.mark.asyncio
async def test_generic_auto_unrelated_codebase_does_not_pollute_context(tmp_path, monkeypatch):
    server = _make_server(tmp_path, key="generic_auto_leather_unrelated_codebase")
    await _build_leather_multifamily_corpus(server, tmp_path, monkeypatch, codebase_related=False)

    result = await server.recall("Leather Jacket")

    assert "The conversation says Leather Jacket reviews are rising." in result["context"]
    assert "The document describes Leather Jacket product demand." in result["context"]
    assert "--- CODEBASE FACTS ---" not in result["context"]
    assert "authorize_payment" not in result["context"]
    assert "--- SOURCE FILES ---" not in result["context"]
    trace = result["runtime_trace"]["family_discovery"]
    assert trace["per_family"]["codebase"]["searched"] is True
    assert trace["per_family"]["codebase"]["selected_count"] == 0
    assert trace["per_family"]["codebase"]["skipped_reason"] in {
        "no_probe_candidates",
        "filtered_below_overlap_threshold",
    }
    assert trace["source_hydration"]["hydrated"] is False


@pytest.mark.asyncio
async def test_generic_auto_does_not_full_scan_codebase_lookup_when_probe_misses(tmp_path, monkeypatch):
    server = _make_server(tmp_path, key="generic_auto_no_hidden_full_scan")
    await _build_leather_multifamily_corpus(server, tmp_path, monkeypatch)
    original_generic_fact_recall = server._generic_fact_recall

    async def mock_generic_fact_recall(**kwargs):
        if kwargs.get("search_family") == "codebase" and kwargs.get("codebase_probe_mode") is True:
            return {
                "context": "RETRIEVED FACTS:",
                "retrieved": [],
                "search_family": "codebase",
                "retrieval_families": [],
                "runtime_trace": {
                    "runtime": "fact",
                    "reason": "forced_probe_miss_for_regression",
                },
            }
        return await original_generic_fact_recall(**kwargs)

    monkeypatch.setattr(server, "_generic_fact_recall", mock_generic_fact_recall)

    result = await server.recall("Leather Jacket")

    assert "The conversation says Leather Jacket reviews are rising." in result["context"]
    assert "The document describes Leather Jacket product demand." in result["context"]
    assert "--- CODEBASE FACTS ---" not in result["context"]
    assert "def leather(jacket):" not in result["context"]
    trace = result["runtime_trace"]["family_discovery"]
    assert trace["per_family"]["codebase"]["searched"] is True
    assert trace["per_family"]["codebase"]["visible_fact_count"] > 0
    assert trace["per_family"]["codebase"]["candidate_count"] == 0
    assert trace["per_family"]["codebase"]["selected_count"] == 0
    assert trace["per_family"]["codebase"]["skipped_reason"] == "no_probe_candidates"


@pytest.mark.asyncio
async def test_generic_precise_code_query_auto_routes_into_codebase_lane(tmp_path, monkeypatch):
    server = _make_server(tmp_path, key="generic_code_hydration")
    await _build_mixed_corpus(server, tmp_path, monkeypatch)

    result = await server.recall("Show the exact code for issue signature")

    assert "pkg.service.issue" in result["context"]
    assert "--- SOURCE FILES ---" in result["context"]
    assert "[File: pkg/service.py]" in result["context"]
    assert "class Permit:" in result["context"]
    assert "Sam discussed issue planning in conversation." not in result["context"]
    assert "The document says the permit checklist is required for issue triage." not in result["context"]
    assert result["runtime_trace"]["codebase_augmentation"]["mode"] == "whole_file"


@pytest.mark.asyncio
async def test_auto_mixed_query_merges_conversation_document_and_codebase(tmp_path, monkeypatch):
    server = _make_server(tmp_path, key="auto_mixed_merge")
    await _build_cinder_mixed_corpus(server, tmp_path, monkeypatch)

    result = await server.recall(
        "Using all available memory sources, find the incident codename mentioned in chat, "
        "the owner team named in the document, and the exact Python qualified name of the "
        "function that returns the codename in code."
    )

    assert "CINDER-42" in result["context"]
    assert "Platform Reliability" in result["context"]
    assert "pkg.service.cinder_signature" in result["context"]
    assert "--- SOURCE FILES ---" in result["context"]
    assert "[File: pkg/service.py]" in result["context"]
    assert "def cinder_signature() -> str:" in result["context"]
    assert result["runtime_trace"]["codebase_augmentation"]["mode"] == "whole_file"
    assert result["runtime_trace"]["family_discovery"]["source_hydration"]["hydrated"] is True
    assert result["runtime_trace"]["mixed_family_merge"]["mode"] == "auto"
    assert set(result["runtime_trace"]["mixed_family_merge"]["lanes"]) == {"episode", "codebase"}
    assert {"conversation", "document", "codebase"} <= set(result["runtime_trace"]["mixed_family_merge"]["merged_families"])
    assert {"conversation", "document", "codebase"} <= set(result["retrieval_families"])


@pytest.mark.asyncio
async def test_auto_mixed_query_does_not_hydrate_without_codebase_facts(tmp_path, monkeypatch):
    server = _make_server(tmp_path, key="auto_mixed_no_code")
    _patch_public_runtime(monkeypatch)
    await server.store(
        "User: The incident codename is CINDER-42. Assistant: Noted.",
        session_num=1,
        session_date="2024-06-01",
        source_id="CHAT",
        scope="agent-private",
    )
    await server.ingest_document(
        "# Runbook\n\nRunbook: the owner team for CINDER-42 is Platform Reliability.",
        source_id="DOC",
        scope="agent-private",
    )

    result = await server.recall(
        "Using all available memory sources, find the incident codename mentioned in chat, "
        "the owner team named in the document, and the exact Python qualified name of the "
        "function that returns the codename in code."
    )

    assert "CINDER-42" in result["context"]
    assert "Platform Reliability" in result["context"]
    assert "--- SOURCE FILES ---" not in result["context"]
    assert result["runtime_trace"]["codebase_augmentation"]["mode"] == "inactive"
    assert "mixed_family_merge" not in result["runtime_trace"]


@pytest.mark.asyncio
async def test_explicit_codebase_query_stays_code_only_in_cinder_mixed_corpus(tmp_path, monkeypatch):
    server = _make_server(tmp_path, key="explicit_code_only")
    await _build_cinder_mixed_corpus(server, tmp_path, monkeypatch)

    result = await server.recall(
        "Show the exact Python qualified name and code for the function returning the codename.",
        search_family="codebase",
    )

    assert "pkg.service.cinder_signature" in result["context"]
    assert "--- SOURCE FILES ---" in result["context"]
    assert "Platform Reliability" not in result["context"]
    assert "The chat says the incident codename is CINDER-42." not in result["context"]
    assert result["search_family"] == "codebase"
    trace = result["runtime_trace"]["family_discovery"]
    assert trace["per_family"]["codebase"]["searched"] is True
    assert trace["per_family"]["conversation"]["searched"] is False
    assert trace["per_family"]["document"]["searched"] is False


@pytest.mark.asyncio
async def test_mixed_query_does_not_regress_narrow_precise_code_auto_behavior(tmp_path, monkeypatch):
    server = _make_server(tmp_path, key="mixed_vs_narrow_precise")
    await _build_cinder_mixed_corpus(server, tmp_path, monkeypatch)

    result = await server.recall("Show the exact code for cinder_signature")

    assert "pkg.service.cinder_signature" in result["context"]
    assert "--- SOURCE FILES ---" in result["context"]
    assert "Platform Reliability" not in result["context"]
    assert "The chat says the incident codename is CINDER-42." not in result["context"]
    assert result["runtime_trace"]["codebase_augmentation"]["mode"] == "whole_file"
    assert result["runtime_trace"]["family_discovery"]["source_hydration"]["hydrated"] is True


@pytest.mark.asyncio
async def test_mixed_recall_and_ask_use_codebase_mixed_prompt(tmp_path, monkeypatch):
    server = _make_server(tmp_path, key="mixed_prompt_path")
    await _build_cinder_mixed_corpus(server, tmp_path, monkeypatch)
    seen = _patch_inference_capture(
        monkeypatch,
        answer=(
            "chat_codename=CINDER-42\n"
            "document_owner=Platform Reliability\n"
            "code_symbol=pkg.service.cinder_signature"
        ),
    )

    query = (
        "Using all available memory sources, find the incident codename mentioned in chat, "
        "the owner team named in the document, and the exact Python qualified name of the "
        "function that returns the codename in code."
    )
    recall_result = await server.recall(query)
    ask_result = await server.ask(query, inference_model="groq/qwen30b")

    assert {"conversation", "document", "codebase"} <= set(recall_result["retrieval_families"])
    assert {"conversation", "document", "codebase"} <= set(ask_result["retrieval_families"])
    prompt = seen["prompt"]
    assert "requires BOTH prose memory evidence and codebase evidence" in prompt
    assert "CINDER-42" in prompt
    assert "Platform Reliability" in prompt
    assert "pkg.service.cinder_signature" in prompt
    assert "chat_codename=CINDER-42" in ask_result["answer"]
    assert "document_owner=Platform Reliability" in ask_result["answer"]
    assert "code_symbol=pkg.service.cinder_signature" in ask_result["answer"]


@pytest.mark.asyncio
async def test_pure_code_recall_and_ask_use_code_slot_prompt(tmp_path, monkeypatch):
    server = _make_server(tmp_path, key="pure_code_prompt_path")
    await _build_cinder_mixed_corpus(server, tmp_path, monkeypatch)
    seen = _patch_inference_capture(
        monkeypatch,
        answer="pkg.service.cinder_signature",
    )

    query = "What is the exact Python qualified name of the function that returns the codename?"
    recall_result = await server.recall(query, search_family="codebase")
    ask_result = await server.ask(query, search_family="codebase", inference_model="groq/qwen30b")

    assert set(recall_result["retrieval_families"]) == {"codebase"}
    assert set(ask_result["retrieval_families"]) == {"codebase"}
    prompt = seen["prompt"]
    assert "exact code object lookup question" in prompt
    assert "requires BOTH prose memory evidence and codebase evidence" not in prompt
    assert "pkg.service.cinder_signature" in ask_result["answer"]


@pytest.mark.asyncio
async def test_codebase_file_lookup_ask_uses_deterministic_selected_file(tmp_path, monkeypatch):
    server = _make_server(tmp_path, key="codebase_file_lookup_deterministic")
    await _build_cinder_mixed_corpus(server, tmp_path, monkeypatch)

    async def _fail_if_called(*args, **kwargs):
        raise AssertionError("LLM inference should not run for deterministic codebase file lookup")

    monkeypatch.setattr(server, "_call_model_with_runtime_secrets", _fail_if_called)
    monkeypatch.setattr(server, "_call_oai_with_runtime_secrets", _fail_if_called)

    query = "Which file defines cinder_signature? Answer with only the file path."
    ask_result = await server.ask(query, search_family="codebase", inference_model="groq/qwen30b")

    assert ask_result["answer"] == "pkg/service.py"
    assert ask_result["runtime_trace"]["deterministic_answer"]["kind"] == "codebase_file_lookup"
    assert set(ask_result["retrieval_families"]) == {"codebase"}


@pytest.mark.asyncio
async def test_mixed_query_with_file_subquestion_does_not_trigger_deterministic_file_lookup(tmp_path, monkeypatch):
    server = _make_server(tmp_path, key="mixed_file_lookup_not_deterministic")
    await _build_cinder_mixed_corpus(server, tmp_path, monkeypatch)
    seen = _patch_inference_capture(
        monkeypatch,
        answer=(
            "chat_codename=CINDER-42\n"
            "document_owner=Platform Reliability\n"
            "code_file=pkg/service.py"
        ),
    )

    query = (
        "Using all available memory sources, answer in exactly three labelled lines:\n"
        "chat_codename=<...>\n"
        "document_owner=<...>\n"
        "code_file=<which file defines the codename function>"
    )
    ask_result = await server.ask(query, inference_model="groq/qwen30b")

    assert ask_result["answer"] == (
        "chat_codename=CINDER-42\n"
        "document_owner=Platform Reliability\n"
        "code_file=pkg/service.py"
    )
    assert "chat_codename=<...>" in seen["prompt"]
    assert "document_owner=<...>" in seen["prompt"]
    assert "code_file=<which file defines the codename function>" in seen["prompt"]
    assert ask_result.get("runtime_trace", {}).get("deterministic_answer", {}).get("kind") != "codebase_file_lookup"
    assert ask_result["payload_meta"].get("deterministic") is not True
    assert {"conversation", "document", "codebase"} <= set(ask_result["retrieval_families"])
