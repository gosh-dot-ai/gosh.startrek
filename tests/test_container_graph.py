# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path

import pytest

from src.container_graph import (
    build_document_container_graph,
    empty_container_graph,
    graph_indexes,
    lift_episode_ids_to_container_ids,
    lift_episode_refs_to_container_ids,
    plan_document_structural_exact_copy,
    stable_hash,
    validate_container_exact_copy_render_refs,
)
from src.episode_features import extract_query_features
from src.memory import EXACT_COPY_REFUSAL, MemoryServer
from src.query_executors.container_graph import ContainerGraphExecutor
from src.storage import SQLiteStorageBackend


REPO_ROOT = Path(__file__).resolve().parents[1]


def _raw_artifact(idx: int, topic: str = "necks") -> str:
    return (
        f"[Artifact {idx:04d}]\n"
        "Instruction:\n"
        f"write a social media post about {topic}\n\n"
        "Response:\n"
        f"artifact {idx} response about {topic}\n"
        f"unique-line-{idx}\n"
    )


def _sample_document(n: int = 8) -> tuple[dict[str, str], dict, dict[str, dict]]:
    source_id = "DOC"
    raw_doc = "\n".join(_raw_artifact(idx) for idx in range(1, n + 1))
    episodes = [
        {
            "episode_id": f"DOC_e{idx:02d}",
            "source_id": source_id,
            "source_type": "document",
            "raw_text": _raw_artifact(idx),
            "raw_original": _raw_artifact(idx),
            "topic_key": "structural_context",
            "state_label": "structural_context",
            "currentness": "unknown",
        }
        for idx in range(1, n + 1)
    ]
    episode_corpus = {"documents": [{"doc_id": "document:DOC", "episodes": episodes}]}
    source_records = {
        source_id: {
            "family": "document",
            "version_id": "v1",
            "content_hash": "hash-doc",
            "read": ["agent:PUBLIC"],
            "write": ["agent:PUBLIC"],
        }
    }
    return {source_id: raw_doc}, episode_corpus, source_records


def _sample_document_from_artifacts(artifacts: list[tuple[str, str]]) -> tuple[dict[str, str], dict, dict[str, dict]]:
    source_id = "DOC"
    raw_parts = []
    episodes = []
    for idx, (instruction, response) in enumerate(artifacts, start=1):
        raw = (
            f"[Artifact {idx:04d}]\n"
            "Instruction:\n"
            f"{instruction}\n\n"
            "Response:\n"
            f"{response}\n"
        )
        raw_parts.append(raw)
        episodes.append(
            {
                "episode_id": f"DOC_e{idx:02d}",
                "source_id": source_id,
                "source_type": "document",
                "raw_text": raw,
                "raw_original": raw,
                "topic_key": "structural_context",
                "state_label": "structural_context",
                "currentness": "unknown",
            }
        )
    episode_corpus = {"documents": [{"doc_id": "document:DOC", "episodes": episodes}]}
    source_records = {
        source_id: {
            "family": "document",
            "version_id": "v1",
            "content_hash": "hash-doc",
            "read": ["agent:PUBLIC"],
            "write": ["agent:PUBLIC"],
        }
    }
    return {source_id: "\n".join(raw_parts)}, episode_corpus, source_records


def _terminal_render_candidate(**overrides) -> dict:
    """Build an audit-clean exact-copy candidate for ask/render path tests."""
    candidate = {
        "candidate_id": "candidate",
        "capability": "exact_copy",
        "status": "available",
        "render_ref_id": "rr1",
        "container_id": "c1",
        "selected_container_ids": ["c1"],
        "selected_render_ref_ids": ["rr1"],
        "raw_text_exposed_to_model": False,
        "render_ref_validated": True,
        "raw_source_present": True,
        "raw_source_validated": True,
        "whole_or_fail": True,
        "degraded_render_source": None,
        "ordinal_satisfied": True,
        "kind_satisfied": True,
        "topic_satisfied": True,
        "planner_proof": {
            "ordinal_satisfied": True,
            "kind_satisfied": True,
            "topic_satisfied": True,
            "anchor_tokens_missing": [],
        },
        "render_proof": {
            "render_ref_validated": True,
            "raw_source_present": True,
            "raw_source_validated": True,
            "whole_or_fail": True,
            "degraded_render_source": None,
        },
        "output_constraints": {},
    }
    for key, value in overrides.items():
        if key in {"planner_proof", "render_proof"} and isinstance(value, dict):
            candidate[key] = {**candidate.get(key, {}), **value}
        else:
            candidate[key] = value
    return candidate


def _exact_copy_profiles(model: str = "gpt-4o-mini") -> tuple[dict[int, str], dict[str, dict]]:
    return {
        1: "fast",
    }, {
        "fast": {
            "backend": "api",
            "model": model,
            "max_output_tokens": 2000,
            "context_window": 128000,
            "temperature": 0,
        }
    }


def _set_fake_send_payload(ms: MemoryServer, answer: str, assert_payload=None) -> None:
    async def _fake_send_payload(payload, **_kwargs):
        if assert_payload is not None:
            assert_payload(payload)
        return answer, False, []

    ms._send_payload = _fake_send_payload  # type: ignore[method-assign]


def test_container_stable_hash_is_deterministic():
    assert stable_hash("container", {"b": 2, "a": 1}) == stable_hash("container", {"a": 1, "b": 2})


def test_exact_copy_mcp_surface_has_no_dedicated_tool():
    source = (REPO_ROOT / "src" / "mcp_server.py").read_text(encoding="utf-8")
    tool_names = set()
    for line in source.splitlines():
        line = line.strip()
        if line.startswith("@mcp.tool(name="):
            tool_names.add(line.split("name=", 1)[1].split(")", 1)[0].strip("\"'"))

    assert "memory_recall" in tool_names
    assert "memory_ask" in tool_names
    assert not any(
        forbidden in tool_name
        for tool_name in tool_names
        for forbidden in ("container_graph_exact_copy", "exact_copy_tool", "terminal_render_tool")
    )


def test_exact_copy_runtime_source_has_no_mrcr_specific_logic():
    forbidden_literals = ("mrcr-", "2needle", "4needle", "8needle")
    for path in (REPO_ROOT / "src").rglob("*"):
        if path.suffix not in {".py", ".md"}:
            continue
        text = path.read_text(encoding="utf-8")
        lowered = text.lower()
        for literal in forbidden_literals:
            assert literal.lower() not in lowered, f"{literal} leaked into {path.relative_to(REPO_ROOT)}"
        assert "EXPECTED_ANSWERS_READ" not in text, f"EXPECTED_ANSWERS_READ leaked into {path.relative_to(REPO_ROOT)}"

    exact_copy_runtime_files = [
        REPO_ROOT / "src" / "container_graph.py",
        REPO_ROOT / "src" / "query_executors" / "container_graph.py",
        REPO_ROOT / "src" / "prompts" / "inference" / "container_exact_copy.md",
    ]
    for path in exact_copy_runtime_files:
        text = path.read_text(encoding="utf-8").lower()
        assert "expected_answers_read" not in text
        assert "ground_truth" not in text
        assert "scorer" not in text


def test_document_container_graph_builds_artifact_refs_and_source_order():
    raw_docs, episode_corpus, source_records = _sample_document(3)

    graph = build_document_container_graph(
        raw_docs=raw_docs,
        episode_corpus=episode_corpus,
        source_records=source_records,
    )

    artifacts = sorted(
        [row for row in graph["containers"] if row["kind_fq"] == "document:artifact"],
        key=lambda row: row["order_key_json"]["segments"],
    )
    assert len(artifacts) == 3
    assert [row["order_key_json"]["segments"] for row in artifacts] == [[1], [2], [3]]
    indexes = graph_indexes(graph)
    assert lift_episode_ids_to_container_ids(graph, ["DOC_e02"]) == [artifacts[1]["container_id"]]
    artifact_lookup_values = {
        row["value_text"]
        for row in graph["ref_lookup"]
        if row["lookup_ns"] == "document" and row["lookup_key"] == "artifact_id"
    }
    assert {"0001", "0002", "0003"} <= artifact_lookup_values
    render_text, _render_ref = (
        indexes["render_refs"][artifacts[1]["primary_render_ref_id"]]["ref_json"]["text"],
        indexes["render_refs"][artifacts[1]["primary_render_ref_id"]],
    )
    assert render_text == "artifact 2 response about necks\nunique-line-2\n"


def test_sqlite_container_graph_snapshot_roundtrip(tmp_path):
    raw_docs, episode_corpus, source_records = _sample_document(2)
    graph = build_document_container_graph(
        raw_docs=raw_docs,
        episode_corpus=episode_corpus,
        source_records=source_records,
    )
    storage = SQLiteStorageBackend(str(tmp_path), "container_roundtrip")
    storage.save_facts(
        {
            "granular": [],
            "cons": [],
            "cross": [],
            "tlinks": [],
            "raw_sessions": [],
            "raw_docs": raw_docs,
            "episode_corpus": episode_corpus,
            "source_records": source_records,
            "container_graph": graph,
        }
    )

    loaded = storage.load_facts(internal=True)
    assert loaded["container_graph"]["containers"] == graph["containers"]
    assert loaded["container_graph"]["refs"] == graph["refs"]
    assert loaded["container_graph"]["render_refs"] == graph["render_refs"]
    with sqlite3.connect(storage.path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM containers").fetchone()[0] == len(graph["containers"])
        assert conn.execute("SELECT COUNT(*) FROM container_refs").fetchone()[0] == len(graph["refs"])


def _projection_kwargs(graph: dict, *, replace: bool) -> dict:
    return {
        "replace_container_graph": replace,
        "container_graph_revision_upserts": graph["graph_revisions"],
        "container_upserts": graph["containers"],
        "container_relation_upserts": graph["relations"],
        "container_anchor_upserts": graph["anchors"],
        "container_evidence_upserts": graph["evidence"],
        "container_ref_upserts": graph["refs"],
        "container_ref_lookup_upserts": graph["ref_lookup"],
        "container_ref_range_upserts": graph["ref_ranges"],
        "container_render_ref_upserts": graph["render_refs"],
        "container_contract_upserts": graph["contracts"],
        "container_artifact_upserts": graph["artifacts"],
        "container_state_upserts": graph["state"],
    }


def test_sqlite_container_graph_incremental_replace_roundtrip(tmp_path):
    raw_docs, episode_corpus, source_records = _sample_document(2)
    graph_one = build_document_container_graph(
        raw_docs=raw_docs,
        episode_corpus=episode_corpus,
        source_records=source_records,
    )
    raw_docs, episode_corpus, source_records = _sample_document(3)
    graph_two = build_document_container_graph(
        raw_docs=raw_docs,
        episode_corpus=episode_corpus,
        source_records=source_records,
    )
    storage = SQLiteStorageBackend(str(tmp_path), "container_incremental")

    storage.persist_projection_delta(**_projection_kwargs(graph_one, replace=True))
    storage.persist_projection_delta(**_projection_kwargs(graph_two, replace=True))

    loaded = storage.load_facts(internal=True)["container_graph"]
    assert loaded["containers"] == graph_two["containers"]
    assert loaded["refs"] == graph_two["refs"]
    assert loaded["render_refs"] == graph_two["render_refs"]
    with sqlite3.connect(storage.path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM containers").fetchone()[0] == len(graph_two["containers"])


def test_document_graph_rebuild_preserves_non_document_rows():
    existing = empty_container_graph()
    existing["graph_revisions"].append(
        {
            "container_graph_revision_id": "cgr_conversation",
            "adapter_name": "conversation_adapter",
            "status": "active",
        }
    )
    existing["containers"].append(
        {
            "container_id": "ctr_conversation",
            "container_graph_revision_id": "cgr_conversation",
            "adapter_name": "conversation_adapter",
            "kind_fq": "conversation:turn",
            "status": "active",
        }
    )
    existing["render_refs"].append(
        {
            "render_ref_id": "render_conversation",
            "container_id": "ctr_conversation",
            "container_graph_revision_id": "cgr_conversation",
            "ref_json": {"text": "conversation"},
            "status": "active",
        }
    )
    existing["contracts"].append(
        {
            "contract_id": "contract_global_operator",
            "contract_kind": "operator_contract",
            "subject_kind": "global",
            "payload_json": {"operator": "nth"},
            "status": "active",
        }
    )
    existing["artifacts"].append(
        {
            "artifact_id": "artifact_context_pack",
            "artifact_kind": "context_pack",
            "container_graph_revision_ids_json": ["cgr_conversation"],
            "families_json": ["conversation", "codebase"],
            "payload_json": {"rows": []},
            "status": "active",
        }
    )
    raw_docs, episode_corpus, source_records = _sample_document(1)

    graph = build_document_container_graph(
        raw_docs=raw_docs,
        episode_corpus=episode_corpus,
        source_records=source_records,
        existing_graph=existing,
    )

    assert any(row.get("container_id") == "ctr_conversation" for row in graph["containers"])
    assert any(row.get("render_ref_id") == "render_conversation" for row in graph["render_refs"])
    assert any(row.get("contract_id") == "contract_global_operator" for row in graph["contracts"])
    assert any(row.get("artifact_id") == "artifact_context_pack" for row in graph["artifacts"])


def test_structural_nth_uses_full_operator_domain_not_seed_top_k():
    raw_docs, episode_corpus, source_records = _sample_document(8)
    graph = build_document_container_graph(
        raw_docs=raw_docs,
        episode_corpus=episode_corpus,
        source_records=source_records,
    )
    query = (
        "Prepend CANARY to the 5th (1 indexed) social media post about necks. "
        "Do not include any other text in your response."
    )

    plan = plan_document_structural_exact_copy(
        graph=graph,
        query=query,
        query_features=extract_query_features(query),
        seed_episode_ids=["DOC_e01"],
        fallback_source_ids=["DOC"],
    )

    assert plan is not None
    assert plan["status"] == "rendered"
    trace = plan["trace"]
    assert len(trace["operator_domain_container_ids"]) == 8
    assert trace["selected_episode_ids"] == ["DOC_e05"]
    assert trace["query_plan"]["operators"][0]["order_scope_id"] == trace["order_scope_id"]
    assert plan["render_text"] == "artifact 5 response about necks\nunique-line-5\n"


def test_structural_exact_copy_trace_declares_document_first_planner_scope():
    raw_docs, episode_corpus, source_records = _sample_document(3)
    graph = build_document_container_graph(
        raw_docs=raw_docs,
        episode_corpus=episode_corpus,
        source_records=source_records,
    )
    query = (
        "Prepend CANARY to the 2nd (1 indexed) social media post about necks. "
        "Do not include any other text in your response."
    )

    plan = plan_document_structural_exact_copy(
        graph=graph,
        query=query,
        query_features=extract_query_features(query),
        seed_episode_ids=["DOC_e01"],
        fallback_source_ids=["DOC"],
    )

    assert plan is not None
    assert plan["status"] == "rendered"
    trace = plan["trace"]
    assert trace["planner_implementation"] == "document_artifact_planner"
    assert trace["planner_implementation_scope"] == "document_artifact_first_implementation"
    assert trace["candidate_family"] == "document"
    assert trace["candidate_kind_fq"] == "document:artifact"
    assert trace["query_plan"]["intent"]["family_any"] == ["document"]
    assert trace["selected_index_in_matching_domain"] == 2
    assert trace["surface_anchor_tokens_requested"] == ["social", "media", "post", "necks"]
    assert trace["surface_anchor_tokens_matched"] == ["social", "media", "post", "necks"]
    assert "media" in trace["surface_anchor_tokens_matched"]
    assert "medium" not in trace["surface_anchor_tokens_matched"]
    assert set(trace["proof_source_fields"]) == {
        "traits_json.instruction_text",
        "render_ref_json.instruction_text",
        "render_ref_json.artifact_id",
    }


def test_structural_anchor_matches_about_topic_without_double_stemming():
    raw_docs, episode_corpus, source_records = _sample_document_from_artifacts(
        [
            ("write a song about trainings", "first training song"),
            ("write a song about cats", "cat song"),
            ("write a song about trainings", "second training song"),
            ("write a song about trainings", "third training song"),
        ]
    )
    graph = build_document_container_graph(
        raw_docs=raw_docs,
        episode_corpus=episode_corpus,
        source_records=source_records,
    )
    query = (
        "Prepend CANARY to the 2nd (1 indexed) song about trainings. "
        "Do not include any other text in your response."
    )

    plan = plan_document_structural_exact_copy(
        graph=graph,
        query=query,
        query_features=extract_query_features(query),
        seed_episode_ids=["DOC_e01"],
        fallback_source_ids=["DOC"],
    )

    assert plan is not None
    assert plan["status"] == "rendered"
    assert plan["trace"]["selected_episode_ids"] == ["DOC_e03"]
    assert plan["render_text"] == "second training song\n"


def test_structural_anchor_keeps_about_topic_that_looks_like_query_word():
    raw_docs, episode_corpus, source_records = _sample_document_from_artifacts(
        [
            ("write a song about family", "family song"),
            ("write a song about the who", "the who song"),
            ("write a song about weather", "weather song"),
        ]
    )
    graph = build_document_container_graph(
        raw_docs=raw_docs,
        episode_corpus=episode_corpus,
        source_records=source_records,
    )
    query = (
        "Prepend CANARY to the 1st (1 indexed) song about the who. "
        "Do not include any other text in your response."
    )

    plan = plan_document_structural_exact_copy(
        graph=graph,
        query=query,
        query_features=extract_query_features(query),
        seed_episode_ids=["DOC_e01"],
        fallback_source_ids=["DOC"],
    )

    assert plan is not None
    assert plan["status"] == "rendered"
    assert plan["trace"]["selected_episode_ids"] == ["DOC_e02"]
    assert plan["render_text"] == "the who song\n"


def test_structural_plan_does_not_use_global_single_scope_without_document_seed():
    raw_docs, episode_corpus, source_records = _sample_document(2)
    graph = build_document_container_graph(
        raw_docs=raw_docs,
        episode_corpus=episode_corpus,
        source_records=source_records,
    )
    query = (
        "Prepend CANARY to the 1st (1 indexed) social media post about necks. "
        "Do not include any other text in your response."
    )

    plan = plan_document_structural_exact_copy(
        graph=graph,
        query=query,
        query_features=extract_query_features(query),
        seed_episode_ids=[],
    )

    assert plan is not None
    assert plan["status"] == "failed_closed"
    assert plan["reason"] == "no_scope_domain"


def test_structural_scope_is_source_revision_not_bare_source_id():
    raw_docs, episode_corpus, source_records = _sample_document(2)
    graph_one = build_document_container_graph(
        raw_docs=raw_docs,
        episode_corpus=episode_corpus,
        source_records={"DOC": {**source_records["DOC"], "version_id": "rev-a"}},
    )
    raw_docs, episode_corpus, source_records = _sample_document(3)
    graph_two = build_document_container_graph(
        raw_docs=raw_docs,
        episode_corpus=episode_corpus,
        source_records={"DOC": {**source_records["DOC"], "version_id": "rev-b"}},
    )
    combined = {key: graph_one[key] + graph_two[key] for key in graph_one}
    query = (
        "Prepend CANARY to the 2nd (1 indexed) social media post about necks. "
        "Do not include any other text in your response."
    )

    plan = plan_document_structural_exact_copy(
        graph=combined,
        query=query,
        query_features=extract_query_features(query),
        seed_episode_ids=[],
        fallback_source_ids=["DOC"],
    )

    assert plan is not None
    assert plan["status"] == "failed_closed"
    assert plan["reason"] == "ambiguous_order_scope"


def test_structural_plan_accepts_explicit_order_scope_id():
    raw_docs, episode_corpus, source_records = _sample_document(2)
    graph = build_document_container_graph(
        raw_docs=raw_docs,
        episode_corpus=episode_corpus,
        source_records=source_records,
    )
    query = (
        "Prepend CANARY to the 2nd (1 indexed) social media post about necks. "
        "Do not include any other text in your response."
    )

    plan = plan_document_structural_exact_copy(
        graph=graph,
        query=query,
        query_features=extract_query_features(query),
        seed_episode_ids=[],
        explicit_order_scope_ids=["DOC:v1"],
    )

    assert plan is not None
    assert plan["status"] == "rendered"
    assert plan["trace"]["order_scope_id"] == "DOC:v1"
    assert plan["render_text"] == "artifact 2 response about necks\nunique-line-2\n"


def test_episode_lift_uses_doc_id_when_episode_ids_collide():
    raw_docs = {
        "DOC1": _raw_artifact(1, "alpha"),
        "DOC2": _raw_artifact(1, "beta"),
    }
    episode_corpus = {
        "documents": [
            {
                "doc_id": "document:DOC1",
                "episodes": [
                    {
                        "episode_id": "shared_e01",
                        "source_id": "DOC1",
                        "source_type": "document",
                        "raw_original": raw_docs["DOC1"],
                    }
                ],
            },
            {
                "doc_id": "document:DOC2",
                "episodes": [
                    {
                        "episode_id": "shared_e01",
                        "source_id": "DOC2",
                        "source_type": "document",
                        "raw_original": raw_docs["DOC2"],
                    }
                ],
            },
        ]
    }
    graph = build_document_container_graph(
        raw_docs=raw_docs,
        episode_corpus=episode_corpus,
        source_records={
            "DOC1": {"family": "document", "version_id": "v1"},
            "DOC2": {"family": "document", "version_id": "v1"},
        },
    )

    lifted = lift_episode_refs_to_container_ids(
        graph,
        [{"doc_id": "document:DOC2", "episode_id": "shared_e01"}],
    )

    assert len(lifted) == 1
    indexes = graph_indexes(graph)
    assert indexes["containers"][lifted[0]]["source_id"] == "DOC2"


def test_render_ref_preserves_response_edge_formatting():
    raw_doc = (
        "[Artifact 0001]\n"
        "Instruction:\n"
        "write a social media post about formatting\n\n"
        "Response:\n"
        "  leading spaces stay\n"
        "body\n"
        "\n"
    )
    episode_corpus = {
        "documents": [
            {
                "doc_id": "document:DOC",
                "episodes": [
                    {
                        "episode_id": "DOC_e01",
                        "source_id": "DOC",
                        "source_type": "document",
                        "raw_text": raw_doc,
                        "raw_original": raw_doc,
                    }
                ],
            }
        ]
    }
    graph = build_document_container_graph(
        raw_docs={"DOC": raw_doc},
        episode_corpus=episode_corpus,
        source_records={"DOC": {"family": "document", "version_id": "v1"}},
    )
    query = (
        "Prepend CANARY to the 1st (1 indexed) social media post about formatting. "
        "Do not include any other text in your response."
    )

    plan = plan_document_structural_exact_copy(
        graph=graph,
        query=query,
        query_features=extract_query_features(query),
        seed_episode_ids=["DOC_e01"],
        fallback_source_ids=["DOC"],
    )

    assert plan is not None
    assert plan["status"] == "rendered"
    assert plan["render_text"] == "  leading spaces stay\nbody\n\n"


def test_raw_document_markers_are_authoritative_over_bad_episode_spans():
    source_id = "DOC"
    raw_doc = "\n".join(_raw_artifact(idx, "pieces") for idx in range(1, 4))
    episode_corpus = {
        "documents": [
            {
                "doc_id": "document:DOC",
                "episodes": [
                    {
                        "episode_id": "DOC_e99",
                        "source_id": source_id,
                        "source_type": "document",
                        "raw_original": "cached episode text with stale span metadata",
                        "artifact_span_id": "DOC::artifact::9999",
                    }
                ],
            }
        ]
    }
    graph = build_document_container_graph(
        raw_docs={source_id: raw_doc},
        episode_corpus=episode_corpus,
        source_records={source_id: {"family": "document", "version_id": "v1"}},
    )
    query = (
        "Prepend CANARY to the 2nd (1 indexed) social media post about pieces. "
        "Do not include any other text in your response."
    )

    plan = plan_document_structural_exact_copy(
        graph=graph,
        query=query,
        query_features=extract_query_features(query),
        seed_episode_ids=["DOC_e99"],
        fallback_source_ids=["DOC"],
    )

    assert plan is not None
    assert plan["status"] == "rendered"
    assert plan["trace"]["selected_artifact_span_ids"] == ["DOC::artifact::0002"]
    assert plan["render_text"] == "artifact 2 response about pieces\nunique-line-2\n"


@pytest.mark.asyncio
async def test_container_executor_restores_original_output_constraints():
    raw_docs, episode_corpus, source_records = _sample_document(2)
    graph = build_document_container_graph(
        raw_docs=raw_docs,
        episode_corpus=episode_corpus,
        source_records=source_records,
    )

    class Server:
        def __init__(self):
            self.private_candidates = []

        def _ensure_container_graph(self):
            return graph

        def _register_terminal_render_candidate(self, payload):
            self.private_candidates.append(payload)

    server = Server()
    query = (
        "Prepend CANARY to the 2nd (1 indexed) social media post about necks. "
        "Do not include any other text in your response."
    )
    executor = ContainerGraphExecutor()
    packet, _ = await executor.augment(
        server,
        packet={
            "retrieved_episode_ids": ["DOC_e01"],
            "search_family": "document",
            "output_constraints": {},
            "query_operator_plan": {},
        },
        query=query,
        query_type="auto",
        episode_lookup={"DOC_e01": {"source_id": "DOC"}},
        fact_filter=lambda _fact: True,
    )

    assert "deterministic_answer" not in packet
    assert "render_text" not in packet["terminal_render_candidate"]
    assert server.private_candidates[0]["render_text"] == "artifact 2 response about necks\nunique-line-2\n"
    assert "artifact 2 response about necks" not in packet["context"]
    assert "TERMINAL RENDER CANDIDATE" in packet["context"]
    assert packet["terminal_render_candidate"]["candidate_id"]
    assert packet["terminal_render_candidate"]["raw_text_exposed_to_model"] is False
    assert packet["terminal_render_candidate"]["query_type"] == "exact_copy"
    assert packet["terminal_render_candidate"]["operator_kind"] == "nth"
    assert packet["terminal_render_candidate"]["requested_index"] == 2
    assert packet["terminal_render_candidate"]["target_kind"] == "social media post"
    assert packet["terminal_render_candidate"]["target_topic"] == "necks"
    assert packet["terminal_render_candidate"]["selected_index_in_matching_domain"] == 2
    assert packet["terminal_render_candidate"]["matching_domain_count"] == 2
    assert packet["terminal_render_candidate"]["ordinal_satisfied"] is True
    assert packet["terminal_render_candidate"]["kind_satisfied"] is True
    assert packet["terminal_render_candidate"]["topic_satisfied"] is True
    assert packet["terminal_render_candidate"]["surface_anchor_tokens_requested"] == ["social", "media", "post", "necks"]
    assert packet["terminal_render_candidate"]["surface_anchor_tokens_matched"] == ["social", "media", "post", "necks"]
    assert packet["terminal_render_candidate"]["normalized_anchor_tokens_requested"]
    assert packet["terminal_render_candidate"]["normalized_anchor_tokens_matched"]
    assert packet["terminal_render_candidate"]["anchor_tokens_missing"] == []
    assert packet["terminal_render_candidate"]["proof_source_fields"] == [
        "traits_json.instruction_text",
        "render_ref_json.instruction_text",
        "render_ref_json.artifact_id",
    ]
    assert packet["terminal_render_candidate"]["render_ref_validated"] is True
    assert packet["terminal_render_candidate"]["planner_implementation"] == "document_artifact_planner"
    assert "Requested selector:" in packet["context"]
    assert "Planner proof:" in packet["context"]
    assert "Render proof:" in packet["context"]
    assert "selected_index_in_matching_domain: 2" in packet["context"]
    assert "planner_implementation_scope: document_artifact_first_implementation" in packet["context"]
    assert "surface_anchor_tokens_requested: ['social', 'media', 'post', 'necks']" in packet["context"]
    assert "proof_source_fields: ['traits_json.instruction_text', 'render_ref_json.instruction_text', 'render_ref_json.artifact_id']" in packet["context"]
    assert packet["context"].index("selected_index_in_matching_domain") < packet["context"].index("Debug identifiers")
    assert packet["output_constraints"]["prepend_prefix"] == "CANARY"
    assert packet["output_constraints"]["return_only"] is True


@pytest.mark.asyncio
async def test_container_executor_skips_non_document_auto_ordinal():
    executor = ContainerGraphExecutor()
    query = (
        "Prepend CANARY to the 1st (1 indexed) conversation event. "
        "Do not include any other text in your response."
    )
    packet = {
        "search_family": "auto",
        "retrieval_families": ["document", "conversation"],
        "retrieved_episode_ids": ["CONV_e01"],
    }

    assert executor.should_skip(
        packet=packet,
        query=query,
        query_type="auto",
        episode_lookup={"CONV_e01": {"source_type": "conversation", "source_id": "CONV"}},
        augmented_facts=[],
    ) is True


@pytest.mark.asyncio
async def test_container_executor_does_not_intercept_codebase_exact_copy_packet():
    executor = ContainerGraphExecutor()
    query = (
        "Prepend CANARY to the 1st (1 indexed) code snippet about parsing. "
        "Do not include any other text in your response."
    )
    packet = {
        "search_family": "codebase",
        "retrieval_families": ["codebase"],
        "retrieved_episode_ids": ["CODE_e01"],
    }

    assert executor.supports("exact_copy", "codebase", packet) is False
    assert executor.should_skip(
        packet=packet,
        query=query,
        query_type="exact_copy",
        episode_lookup={"CODE_e01": {"source_type": "codebase", "source_id": "CODE"}},
        augmented_facts=[],
    ) is True


@pytest.mark.asyncio
async def test_container_executor_fails_closed_without_terminal_candidate_when_anchor_missing():
    raw_docs, episode_corpus, source_records = _sample_document(2)
    graph = build_document_container_graph(
        raw_docs=raw_docs,
        episode_corpus=episode_corpus,
        source_records=source_records,
    )

    class Server:
        def __init__(self):
            self.private_candidates = []

        def _ensure_container_graph(self):
            return graph

        def _register_terminal_render_candidate(self, payload):
            self.private_candidates.append(payload)

    query = (
        "Prepend CANARY to the 1st (1 indexed) social media post about volcanoes. "
        "Do not include any other text in your response."
    )
    packet, _ = await ContainerGraphExecutor().augment(
        Server(),
        packet={
            "retrieved_episode_ids": ["DOC_e01"],
            "search_family": "document",
            "output_constraints": {},
            "query_operator_plan": {},
        },
        query=query,
        query_type="exact_copy",
        episode_lookup={"DOC_e01": {"source_type": "document", "source_id": "DOC"}},
        fact_filter=lambda _fact: True,
    )

    assert packet["container_graph_status"] == "failed_closed"
    assert packet["container_graph_trace"]["fallback_reason"] == "empty_anchor_filtered_domain"
    assert "terminal_render_candidate" not in packet
    assert packet["context"] == ""


@pytest.mark.asyncio
async def test_container_executor_uses_order_scope_ids_as_scopes_not_sources():
    raw_docs, episode_corpus, source_records = _sample_document(2)
    graph = build_document_container_graph(
        raw_docs=raw_docs,
        episode_corpus=episode_corpus,
        source_records=source_records,
    )

    class Server:
        def _ensure_container_graph(self):
            return graph

    executor = ContainerGraphExecutor()
    query = (
        "Prepend CANARY to the 2nd (1 indexed) social media post about necks. "
        "Do not include any other text in your response."
    )
    packet, _ = await executor.augment(
        Server(),
        packet={
            "search_family": "auto",
            "document_order_scope_ids": ["DOC:v1"],
            "retrieved_episode_ids": [],
        },
        query=query,
        query_type="auto",
        episode_lookup={},
        fact_filter=lambda _fact: True,
    )

    assert packet["container_graph_status"] == "rendered"
    assert packet["container_graph_trace"]["order_scope_id"] == "DOC:v1"
    assert "deterministic_answer" not in packet
    assert "render_text" not in packet["terminal_render_candidate"]
    assert packet["terminal_render_candidate"]["render_ref_id"]


def test_exact_copy_decision_returns_terminal_exact_answer(tmp_path):
    profiles, profile_configs = _exact_copy_profiles()
    ms = MemoryServer(
        str(tmp_path),
        "container_exact_copy_ask",
        extract_model=None,
        profiles=profiles,
        profile_configs=profile_configs,
    )
    candidate_id = "candidate-edge"
    public_candidate = _terminal_render_candidate(
        candidate_id=candidate_id,
        output_constraints={"prepend_prefix": "CANARY"},
    )
    ms._register_terminal_render_candidate({
        **public_candidate,
        "render_mode": "exact_copy",
        "render_source": "raw_doc_marker_span",
        "render_text": "  leading spaces stay\nbody\n\n",
    })

    async def _fake_recall(*_args, **_kwargs):
        return {
            "context": (
                "--- TERMINAL RENDER CANDIDATE ---\n"
                "Capability: exact_copy\n"
                f"Candidate id: {candidate_id}\n"
                "Selected render refs: ['rr1']\n"
                "Raw exact text is hidden from the model.\n"
                '{"decision":"use_candidate","candidate_id":"candidate-edge"}'
            ),
            "terminal_render_candidate": public_candidate,
            "output_constraints": {"prepend_prefix": "CANARY"},
            "query_type": "exact_copy",
            "retrieved": [],
            "retrieval_families": ["document"],
            "search_family": "document",
            "recommended_profile": "fast",
            "runtime_trace": {"container_graph": {"render_mode": "exact_copy", "exact_copy_validated": True}},
        }

    def _assert_payload(prompt_payload):
        serialized_payload = json.dumps(prompt_payload, ensure_ascii=False)
        assert "TERMINAL RENDER CANDIDATE" in serialized_payload
        assert "leading spaces stay" not in serialized_payload
        assert prompt_payload["model"] == "gpt-4o-mini"

    ms.recall = _fake_recall  # type: ignore[method-assign]
    _set_fake_send_payload(
        ms,
        '{"decision":"use_candidate","candidate_id":"candidate-edge"}',
        assert_payload=_assert_payload,
    )

    result = asyncio.run(ms.ask("copy the first artifact exactly", use_tool=True))

    assert result["answer"] == "CANARY  leading spaces stay\nbody\n\n"
    assert result["tool_called"] is False
    assert result["tool_results"] is None
    assert result["terminal_render_trace"]["terminal_render_answer"] is True
    assert result["terminal_render_trace"]["raw_text_exposed_to_model"] is False
    assert result["render_results"][0]["terminal_render_answer"] is True
    assert result["profile_used"] == "fast"
    assert result["payload_meta"]["profile_used"] == "fast"
    assert result["payload_meta"]["prompt_type"] == "exact_copy"
    assert result["payload_meta"]["prompt_key"] == "container_exact_copy"
    assert result["payload_meta"]["raw_text_exposed_to_model"] is False
    assert result["runtime_trace"]["terminal_render_candidate"]["terminal_render_answer"] is True
    assert result["runtime_trace"]["container_graph"]["render_mode"] == "exact_copy"


def test_exact_copy_ask_calls_model_before_internal_terminal_render_and_hides_raw(tmp_path):
    profiles, profile_configs = _exact_copy_profiles()
    ms = MemoryServer(
        str(tmp_path),
        "container_exact_copy_model_gate",
        extract_model=None,
        profiles=profiles,
        profile_configs=profile_configs,
    )
    raw_phrase = "UNIQUE_RAW_PHRASE_SHOULD_ONLY_APPEAR_AFTER_MODEL_DECISION"
    candidate = _terminal_render_candidate(
        candidate_id="candidate-gated",
        render_ref_id="rr-gated",
        container_id="c-gated",
        selected_container_ids=["c-gated"],
        selected_render_ref_ids=["rr-gated"],
    )
    ms._register_terminal_render_candidate({**candidate, "render_text": f"{raw_phrase}\n"})
    calls: list[str] = []

    async def _fake_recall(*_args, **_kwargs):
        return {
            "context": (
                "--- TERMINAL RENDER CANDIDATE ---\n"
                "Requested selector:\n"
                "  requested_index: 1\n"
                "Planner proof:\n"
                "  selected_index_in_matching_domain: 1\n"
                "Raw exact text is hidden from the model."
            ),
            "terminal_render_candidate": candidate,
            "output_constraints": {},
            "query_type": "exact_copy",
            "retrieved": [],
            "retrieval_families": ["document"],
            "search_family": "document",
            "recommended_profile": "fast",
            "runtime_trace": {"terminal_render_candidate": {"candidate_id": "candidate-gated"}},
        }

    async def _fake_send_payload(payload, **_kwargs):
        calls.append("model")
        serialized_payload = json.dumps(payload, ensure_ascii=False)
        assert raw_phrase not in serialized_payload
        assert "TERMINAL RENDER CANDIDATE" in serialized_payload
        return '{"decision":"use_candidate","candidate_id":"candidate-gated"}', False, []

    original_execute = ms._execute_terminal_render_candidate

    def _wrapped_execute(input_data, *, candidate_context):
        calls.append("render")
        assert calls == ["model", "render"]
        return original_execute(input_data, candidate_context=candidate_context)

    ms.recall = _fake_recall  # type: ignore[method-assign]
    ms._send_payload = _fake_send_payload  # type: ignore[method-assign]
    ms._execute_terminal_render_candidate = _wrapped_execute  # type: ignore[method-assign]

    result = asyncio.run(ms.ask("copy the first artifact exactly"))

    assert calls == ["model", "render"]
    assert result["answer"] == f"{raw_phrase}\n"
    assert result["payload_meta"]["raw_text_exposed_to_model"] is False
    assert result["payload_meta"]["use_tool"] is False
    assert result["terminal_render_trace"]["raw_text_exposed_to_model"] is False
    assert result["tool_called"] is False
    assert result["tool_results"] is None
    assert result["render_results"][0]["terminal_render_answer"] is True


def test_exact_copy_refusal_does_not_prefix(tmp_path):
    profiles, profile_configs = _exact_copy_profiles()
    ms = MemoryServer(
        str(tmp_path),
        "container_exact_copy_refusal",
        extract_model=None,
        profiles=profiles,
        profile_configs=profile_configs,
    )
    public_candidate = _terminal_render_candidate(
        candidate_id="candidate-refusal",
        output_constraints={"prepend_prefix": "CANARY"},
    )
    ms._register_terminal_render_candidate({**public_candidate, "render_text": "selected raw body\n"})

    async def _fake_recall(*_args, **_kwargs):
        return {
            "context": (
                "--- TERMINAL RENDER CANDIDATE ---\n"
                "Capability: exact_copy\n"
                "Candidate id: candidate-refusal\n"
                "Raw exact text is hidden from the model."
            ),
            "terminal_render_candidate": public_candidate,
            "output_constraints": {"prepend_prefix": "CANARY"},
            "query_type": "exact_copy",
            "retrieved": [],
            "retrieval_families": ["document"],
            "search_family": "document",
            "recommended_profile": "fast",
            "runtime_trace": {"container_graph": {"render_mode": "exact_copy", "exact_copy_validated": True}},
        }

    ms.recall = _fake_recall  # type: ignore[method-assign]
    _set_fake_send_payload(
        ms,
        "The model tried to answer without selecting the terminal render candidate.",
    )

    result = asyncio.run(ms.ask("copy the first artifact exactly"))

    assert result["answer"] == EXACT_COPY_REFUSAL
    assert not result["answer"].startswith("CANARY")
    assert result["tool_called"] is False
    assert result["tool_results"] is None
    assert result["terminal_render_trace"]["terminal_render_answer"] is False
    assert result["terminal_render_trace"]["error"] == "EXACT_COPY_MODEL_DID_NOT_SELECT_CANDIDATE"
    assert result["payload_meta"]["prompt_key"] == "container_exact_copy"


@pytest.mark.parametrize(
    ("field", "section", "expected_error"),
    [
        ("whole_or_fail", "render_proof", "EXACT_COPY_NOT_WHOLE_OR_FAIL"),
        ("render_ref_validated", "render_proof", "EXACT_RENDER_REF_UNRESOLVED"),
        ("raw_source_present", "render_proof", "RAW_SOURCE_MISSING"),
        ("raw_source_validated", "render_proof", "RAW_SOURCE_NOT_ORIGINAL"),
        ("ordinal_satisfied", "planner_proof", "EXACT_COPY_ORDINAL_NOT_SATISFIED"),
        ("kind_satisfied", "planner_proof", "EXACT_COPY_KIND_NOT_SATISFIED"),
        ("topic_satisfied", "planner_proof", "EXACT_COPY_TOPIC_NOT_SATISFIED"),
    ],
)
def test_exact_copy_proof_requires_explicit_true_for_required_fields(tmp_path, field, section, expected_error):
    ms = MemoryServer(str(tmp_path), "container_exact_copy_strict_proof", extract_model=None)
    candidate = _terminal_render_candidate()
    candidate.pop(field, None)
    candidate[section].pop(field, None)

    assert ms._terminal_render_candidate_proof_error(candidate) == expected_error


@pytest.mark.parametrize(
    ("model_answer", "expected_error"),
    [
        ('{"decision":"use_candidate","candidate_id":"wrong-candidate"}', "EXACT_COPY_CANDIDATE_ID_MISMATCH"),
        (
            '{"decision":"use_candidate","candidate_id":"candidate-refusal","render_ref_id":"wrong-render"}',
            "EXACT_COPY_RENDER_REF_MISMATCH",
        ),
        (
            '{"decision":"use_candidate","candidate_id":"candidate-refusal","container_id":"wrong-container"}',
            "EXACT_COPY_CONTAINER_MISMATCH",
        ),
        ('{"decision":"use_candidate"}', "EXACT_COPY_MODEL_DID_NOT_SELECT_CANDIDATE"),
        ("selected raw body\n", "EXACT_COPY_MODEL_DID_NOT_SELECT_CANDIDATE"),
    ],
)
def test_exact_copy_rejects_wrong_handle_or_copied_text(tmp_path, model_answer, expected_error):
    profiles, profile_configs = _exact_copy_profiles()
    ms = MemoryServer(
        str(tmp_path),
        "container_exact_copy_bad_handle",
        extract_model=None,
        profiles=profiles,
        profile_configs=profile_configs,
    )
    public_candidate = _terminal_render_candidate(candidate_id="candidate-refusal")
    ms._register_terminal_render_candidate({**public_candidate, "render_text": "selected raw body\n"})

    async def _fake_recall(*_args, **_kwargs):
        return {
            "context": "--- TERMINAL RENDER CANDIDATE ---\nRaw exact text is hidden from the model.",
            "terminal_render_candidate": public_candidate,
            "output_constraints": {},
            "query_type": "exact_copy",
            "retrieved": [],
            "retrieval_families": ["document"],
            "search_family": "document",
            "recommended_profile": "fast",
            "runtime_trace": {"terminal_render_candidate": {"candidate_id": "candidate-refusal"}},
        }

    def _assert_payload(prompt_payload):
        assert "selected raw body" not in json.dumps(prompt_payload, ensure_ascii=False)

    ms.recall = _fake_recall  # type: ignore[method-assign]
    _set_fake_send_payload(ms, model_answer, assert_payload=_assert_payload)

    result = asyncio.run(ms.ask("copy the first artifact exactly"))

    assert result["answer"] == EXACT_COPY_REFUSAL
    assert result["tool_results"] is None
    assert result["terminal_render_trace"]["error"] == expected_error


def test_exact_copy_rejects_incomplete_decision(tmp_path):
    profiles, profile_configs = _exact_copy_profiles()
    ms = MemoryServer(
        str(tmp_path),
        "container_exact_copy_bad_decision",
        extract_model=None,
        profiles=profiles,
        profile_configs=profile_configs,
    )
    public_candidate = _terminal_render_candidate(
        candidate_id="candidate-refusal",
        output_constraints={"prepend_prefix": "CANARY"},
    )
    ms._register_terminal_render_candidate({**public_candidate, "render_text": "selected raw body\n"})

    async def _fake_recall(*_args, **_kwargs):
        return {
            "context": "Raw exact text is hidden from the model.",
            "terminal_render_candidate": public_candidate,
            "output_constraints": {"prepend_prefix": "CANARY"},
            "query_type": "exact_copy",
            "retrieved": [],
            "retrieval_families": ["document"],
            "search_family": "document",
            "recommended_profile": "fast",
            "runtime_trace": {"container_graph": {"render_mode": "exact_copy", "exact_copy_validated": True}},
        }

    ms.recall = _fake_recall  # type: ignore[method-assign]
    _set_fake_send_payload(ms, '{"candidate_id":"candidate-refusal"}')

    result = asyncio.run(ms.ask("copy the first artifact exactly"))

    assert result["answer"] == EXACT_COPY_REFUSAL
    assert result["tool_called"] is False
    assert result["tool_results"] is None
    assert result["terminal_render_trace"]["error"] == "EXACT_COPY_MODEL_DID_NOT_SELECT_CANDIDATE"


def test_exact_copy_preserves_literal_think_tags(tmp_path):
    profiles, profile_configs = _exact_copy_profiles()
    ms = MemoryServer(
        str(tmp_path),
        "container_exact_copy_prefix_restore",
        extract_model=None,
        profiles=profiles,
        profile_configs=profile_configs,
    )

    raw_body = "  <think>literal source tag</think>\nbody\n\n"
    public_candidate = _terminal_render_candidate(
        candidate_id="candidate-think",
        output_constraints={"prepend_prefix": "CANARY"},
    )
    ms._register_terminal_render_candidate({**public_candidate, "render_text": raw_body})

    async def _fake_recall(*_args, **_kwargs):
        return {
            "context": (
                "--- TERMINAL RENDER CANDIDATE ---\n"
                "Capability: exact_copy\n"
                "Candidate id: candidate-think\n"
                "Raw exact text is hidden from the model."
            ),
            "terminal_render_candidate": public_candidate,
            "output_constraints": {"prepend_prefix": "CANARY"},
            "query_type": "exact_copy",
            "retrieved": [],
            "retrieval_families": ["document"],
            "search_family": "document",
            "recommended_profile": "fast",
            "runtime_trace": {"container_graph": {"render_mode": "exact_copy", "exact_copy_validated": True}},
        }

    ms.recall = _fake_recall  # type: ignore[method-assign]
    _set_fake_send_payload(ms, '{"decision":"use_candidate","candidate_id":"candidate-think"}')

    result = asyncio.run(ms.ask("copy the first artifact exactly"))

    assert result["answer"] == f"CANARY{raw_body}"
    assert "<think>literal source tag</think>" in result["answer"]
    assert result["payload_meta"]["prompt_key"] == "container_exact_copy"


@pytest.mark.parametrize("family", ["codebase", "conversation"])
def test_exact_copy_candidate_ask_path_is_family_generic(tmp_path, family):
    profiles, profile_configs = _exact_copy_profiles()
    ms = MemoryServer(
        str(tmp_path),
        f"exact_copy_{family}",
        extract_model=None,
        profiles=profiles,
        profile_configs=profile_configs,
    )
    candidate = _terminal_render_candidate(
        candidate_id=f"{family}-candidate",
        family=family,
        render_ref_id=f"{family}-render-ref",
        container_id=f"{family}-container",
        selected_container_ids=[f"{family}-container"],
        selected_render_ref_ids=[f"{family}-render-ref"],
    )
    ms._register_terminal_render_candidate({**candidate, "render_text": f"{family} exact body\n"})

    async def _fake_recall(*_args, **_kwargs):
        return {
            "context": (
                "--- TERMINAL RENDER CANDIDATE ---\n"
                "Capability: exact_copy\n"
                f"Family: {family}\n"
                f"Candidate id: {family}-candidate\n"
                "Raw exact text is hidden from the model."
            ),
            "terminal_render_candidate": candidate,
            "output_constraints": {},
            "query_type": "exact_copy",
            "retrieved": [],
            "retrieval_families": [family],
            "search_family": family,
            "recommended_profile": "fast",
            "runtime_trace": {"terminal_render_candidate": {"family": family}},
        }

    def _assert_payload(prompt_payload):
        assert f"{family} exact body" not in json.dumps(prompt_payload, ensure_ascii=False)

    ms.recall = _fake_recall  # type: ignore[method-assign]
    _set_fake_send_payload(
        ms,
        f'{{"decision":"use_candidate","candidate_id":"{family}-candidate"}}',
        assert_payload=_assert_payload,
    )

    result = asyncio.run(ms.ask(f"copy the {family} unit exactly"))

    assert result["answer"] == f"{family} exact body\n"
    assert result["tool_called"] is False
    assert result["tool_results"] is None
    assert result["terminal_render_trace"]["terminal_render_answer"] is True
    assert result["runtime_trace"]["terminal_render_candidate"]["terminal_render_answer"] is True
    assert result["retrieval_families"] == [family]


def test_provider_payload_never_registers_container_exact_copy_tool(tmp_path):
    ms = MemoryServer(str(tmp_path), "container_exact_copy_payload", extract_model=None)

    payload, _tool_tokens = ms._build_provider_payload(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": "choose exact-copy candidate"}],
        max_tokens=128,
        temperature=0,
        use_tool=True,
    )

    tool_names = [tool["function"]["name"] for tool in payload["tools"]]
    assert tool_names == ["get_more_context"]


@pytest.mark.parametrize(
    ("model", "expected_provider"),
    [
        ("gpt-4.1-mini", "openai"),
        ("anthropic/claude-opus-4-6", "anthropic"),
        ("google/gemini-2.5-flash", "google"),
    ],
)
def test_exact_copy_payload_does_not_create_provider_tool_payload(tmp_path, model, expected_provider):
    profiles = {1: "fast"}
    profile_configs = {
        "fast": {
            "backend": "api",
            "model": model,
            "max_output_tokens": 2000,
            "context_window": 128000,
            "temperature": 0,
        }
    }
    ms = MemoryServer(
        str(tmp_path),
        f"exact_copy_payload_{expected_provider}",
        extract_model=None,
        profiles=profiles,
        profile_configs=profile_configs,
    )
    payload, payload_meta, _secret_ref = ms._build_payload(
        query="copy exactly",
        recall_result={
            "context": "--- TERMINAL RENDER CANDIDATE ---\nRaw exact text is hidden from the model.",
            "terminal_render_candidate": {
                "candidate_id": "candidate-provider",
                "raw_text_exposed_to_model": False,
            },
            "query_type": "exact_copy",
            "recommended_prompt_type": "exact_copy",
            "recommended_profile": "fast",
            "output_constraints": {},
        },
        use_tool=True,
    )

    assert payload is not None
    assert payload_meta is not None
    assert "tools" not in payload
    assert payload_meta["use_tool"] is False
    assert payload_meta["provider"] == expected_provider
    assert payload_meta["raw_text_exposed_to_model"] is False


@pytest.mark.asyncio
async def test_mcp_memory_recall_exposes_terminal_render_candidate_without_raw_text(tmp_path, monkeypatch):
    import src.mcp_server as mcp_mod
    from tests._mcp_auth import auth_token_for_agent, install_test_verified_auth

    mcp_mod.data_dir = str(tmp_path)
    mcp_mod.registry.clear()
    mcp_mod.courier_registry.clear()
    mcp_mod.connections.clear()
    mcp_mod.sub_to_conn.clear()
    install_test_verified_auth(monkeypatch)

    server = mcp_mod._get_memory("exact-copy-mcp")

    async def _fake_recall(*_args, **_kwargs):
        return {
            "context": "--- TERMINAL RENDER CANDIDATE ---\nRaw exact text is hidden from the model.",
            "retrieved": [],
            "query_type": "exact_copy",
            "complexity_hint": {"level": 1},
            "terminal_render_candidate": {
                "candidate_id": "candidate-mcp",
                "capability": "exact_copy",
                "raw_text_exposed_to_model": False,
                "render_text": "secret raw body must not leave server",
                "proof_summary": {"render_text": "nested secret raw body"},
            },
            "runtime_trace": {"terminal_render_candidate": {"candidate_id": "candidate-mcp"}},
        }

    server.recall = _fake_recall  # type: ignore[method-assign]

    result = await mcp_mod.memory_recall(
        key="exact-copy-mcp",
        query="copy exactly",
        query_type="exact_copy",
        agent_id="a",
        swarm_id="sw1",
        token=auth_token_for_agent("a"),
    )

    assert result["query_type"] == "exact_copy"
    assert result["terminal_render_candidate"]["candidate_id"] == "candidate-mcp"
    assert result["terminal_render_candidate"]["raw_text_exposed_to_model"] is False
    assert "render_text" not in result["terminal_render_candidate"]
    assert "render_text" not in result["terminal_render_candidate"]["proof_summary"]


def test_exact_copy_refuses_when_anchor_missing(tmp_path):
    profiles, profile_configs = _exact_copy_profiles()
    ms = MemoryServer(
        str(tmp_path),
        "container_exact_copy_anchor_missing",
        extract_model=None,
        profiles=profiles,
        profile_configs=profile_configs,
    )
    candidate = {
        "candidate_id": "candidate-missing-anchor",
        "capability": "exact_copy",
        "status": "available",
        "render_ref_id": "rr1",
        "container_id": "c1",
        "selected_container_ids": ["c1"],
        "selected_render_ref_ids": ["rr1"],
        "raw_text_exposed_to_model": False,
        "planner_proof": {
            "anchor_tokens_matched": ["song"],
            "anchor_tokens_missing": ["trainings"],
        },
        "output_constraints": {"prepend_prefix": "CANARY"},
    }
    ms._register_terminal_render_candidate({**candidate, "render_text": "selected raw body\n"})

    async def _fake_recall(*_args, **_kwargs):
        return {
            "context": (
                "--- TERMINAL RENDER CANDIDATE ---\n"
                "Planner proof:\n"
                "  anchor_tokens_matched: ['song']\n"
                "  anchor_tokens_missing: ['trainings']\n"
                "Raw exact text is hidden from the model."
            ),
            "terminal_render_candidate": candidate,
            "output_constraints": {"prepend_prefix": "CANARY"},
            "query_type": "exact_copy",
            "retrieved": [],
            "retrieval_families": ["document"],
            "search_family": "document",
            "recommended_profile": "fast",
            "runtime_trace": {"terminal_render_candidate": candidate},
        }

    def _assert_payload(prompt_payload):
        serialized_payload = json.dumps(prompt_payload, ensure_ascii=False)
        assert "anchor_tokens_missing: ['trainings']" in serialized_payload
        assert "selected raw body" not in serialized_payload

    ms.recall = _fake_recall  # type: ignore[method-assign]
    _set_fake_send_payload(ms, EXACT_COPY_REFUSAL, assert_payload=_assert_payload)

    result = asyncio.run(ms.ask("copy the 4th song about trainings exactly"))

    assert result["answer"] == EXACT_COPY_REFUSAL
    assert result["tool_results"] is None
    assert result["terminal_render_trace"]["error"] == "EXACT_COPY_MODEL_DID_NOT_SELECT_CANDIDATE"


def test_exact_copy_runtime_refuses_degraded_candidate_even_if_model_selects(tmp_path):
    profiles, profile_configs = _exact_copy_profiles()
    ms = MemoryServer(
        str(tmp_path),
        "container_exact_copy_degraded_proof",
        extract_model=None,
        profiles=profiles,
        profile_configs=profile_configs,
    )
    candidate = {
        "candidate_id": "candidate-degraded",
        "capability": "exact_copy",
        "status": "available",
        "render_ref_id": "rr1",
        "container_id": "c1",
        "selected_container_ids": ["c1"],
        "selected_render_ref_ids": ["rr1"],
        "raw_text_exposed_to_model": False,
        "render_ref_validated": False,
        "raw_source_present": True,
        "whole_or_fail": True,
        "degraded_render_source": "episode_join_fallback",
        "render_proof": {
            "render_ref_validated": False,
            "raw_source_present": True,
            "whole_or_fail": True,
            "degraded_render_source": "episode_join_fallback",
        },
        "output_constraints": {},
    }
    ms._register_terminal_render_candidate({**candidate, "render_text": "selected raw body\n"})

    async def _fake_recall(*_args, **_kwargs):
        return {
            "context": (
                "--- TERMINAL RENDER CANDIDATE ---\n"
                "Render proof:\n"
                "  degraded_render_source: episode_join_fallback\n"
                "Raw exact text is hidden from the model."
            ),
            "terminal_render_candidate": candidate,
            "output_constraints": {},
            "query_type": "exact_copy",
            "retrieved": [],
            "retrieval_families": ["document"],
            "search_family": "document",
            "recommended_profile": "fast",
            "runtime_trace": {"terminal_render_candidate": candidate},
        }

    def _assert_payload(prompt_payload):
        assert "selected raw body" not in json.dumps(prompt_payload, ensure_ascii=False)

    ms.recall = _fake_recall  # type: ignore[method-assign]
    _set_fake_send_payload(
        ms,
        '{"decision":"use_candidate","candidate_id":"candidate-degraded"}',
        assert_payload=_assert_payload,
    )

    result = asyncio.run(ms.ask("copy exactly"))

    assert result["answer"] == EXACT_COPY_REFUSAL
    assert result["terminal_render_trace"]["terminal_render_answer"] is False
    assert result["terminal_render_trace"]["error"] == "EXACT_COPY_SOURCE_DEGRADED"


def test_exact_render_has_no_sibling_contamination():
    raw_docs, episode_corpus, source_records = _sample_document(8)
    graph = build_document_container_graph(
        raw_docs=raw_docs,
        episode_corpus=episode_corpus,
        source_records=source_records,
    )
    query = (
        "Prepend CANARY to the 2nd (1 indexed) social media post about necks. "
        "Do not include any other text in your response."
    )

    plan = plan_document_structural_exact_copy(
        graph=graph,
        query=query,
        query_features=extract_query_features(query),
        seed_episode_ids=["DOC_e08"],
        fallback_source_ids=["DOC"],
    )

    assert plan is not None
    assert plan["status"] == "rendered"
    assert "unique-line-2" in plan["render_text"]
    assert "unique-line-1" not in plan["render_text"]
    assert "unique-line-3" not in plan["render_text"]
    assert plan["trace"]["no_cross_container_contamination"] is True


def test_structural_render_fails_closed_when_render_ref_unresolved():
    raw_docs, episode_corpus, source_records = _sample_document(2)
    graph = build_document_container_graph(
        raw_docs=raw_docs,
        episode_corpus=episode_corpus,
        source_records=source_records,
    )
    for row in graph["render_refs"]:
        row["status"] = "deleted"
    query = (
        "Prepend CANARY to the 1st (1 indexed) social media post about necks. "
        "Do not include any other text in your response."
    )

    plan = plan_document_structural_exact_copy(
        graph=graph,
        query=query,
        query_features=extract_query_features(query),
        seed_episode_ids=["DOC_e01"],
        fallback_source_ids=["DOC"],
    )

    assert plan is not None
    assert plan["status"] == "failed_closed"
    assert plan["reason"] == "render_ref_unresolved"
    assert plan["trace"]["fallback_allowed"] is False


def test_structural_render_fails_closed_when_render_source_degraded():
    raw_docs, episode_corpus, source_records = _sample_document(2)
    graph = build_document_container_graph(
        raw_docs=raw_docs,
        episode_corpus=episode_corpus,
        source_records=source_records,
    )
    for row in graph["render_refs"]:
        row["render_mode"] = "degraded_episode_join"
        row["ref_json"]["render_source"] = "episode_join_fallback"
    query = (
        "Prepend CANARY to the 1st (1 indexed) social media post about necks. "
        "Do not include any other text in your response."
    )

    plan = plan_document_structural_exact_copy(
        graph=graph,
        query=query,
        query_features=extract_query_features(query),
        seed_episode_ids=["DOC_e01"],
        fallback_source_ids=["DOC"],
    )

    assert plan is not None
    assert plan["status"] == "failed_closed"
    assert plan["reason"] == "EXACT_COPY_SOURCE_DEGRADED"
    assert plan["trace"]["degraded_render_source"] == "episode_join_fallback"
    assert plan["trace"]["render_ref_validated"] is False


def test_structural_render_fails_closed_when_ordinal_out_of_range():
    raw_docs, episode_corpus, source_records = _sample_document(2)
    graph = build_document_container_graph(
        raw_docs=raw_docs,
        episode_corpus=episode_corpus,
        source_records=source_records,
    )
    query = (
        "Prepend CANARY to the 3rd (1 indexed) social media post about necks. "
        "Do not include any other text in your response."
    )

    plan = plan_document_structural_exact_copy(
        graph=graph,
        query=query,
        query_features=extract_query_features(query),
        seed_episode_ids=["DOC_e01"],
        fallback_source_ids=["DOC"],
    )

    assert plan is not None
    assert plan["status"] == "failed_closed"
    assert plan["reason"] == "ordinal_out_of_range"
    assert plan["trace"]["ordinal_satisfied"] is False
    assert plan["trace"]["selected_container_ids"] == []


def test_episode_join_fallback_is_not_exact_copy():
    _raw_docs, episode_corpus, source_records = _sample_document(2)
    graph = build_document_container_graph(
        raw_docs={},
        episode_corpus=episode_corpus,
        source_records=source_records,
    )
    query = (
        "Prepend CANARY to the 1st (1 indexed) social media post about necks. "
        "Do not include any other text in your response."
    )

    plan = plan_document_structural_exact_copy(
        graph=graph,
        query=query,
        query_features=extract_query_features(query),
        seed_episode_ids=["DOC_e01"],
        fallback_source_ids=["DOC"],
    )

    assert plan is not None
    assert plan["status"] == "failed_closed"
    assert plan["reason"] == "no_scope_domain"
    assert graph["containers"] == []
    assert graph["render_refs"] == []
    assert graph["graph_revisions"][0]["status"] == "failed_closed"
    assert validate_container_exact_copy_render_refs(graph) == []


def test_episode_join_fallback_is_not_activated_when_raw_doc_has_no_artifact_spans():
    _raw_docs, episode_corpus, source_records = _sample_document(2)
    graph = build_document_container_graph(
        raw_docs={"DOC": "plain original source without artifact markers\n"},
        episode_corpus=episode_corpus,
        source_records=source_records,
    )
    query = (
        "Prepend CANARY to the 1st (1 indexed) social media post about necks. "
        "Do not include any other text in your response."
    )

    plan = plan_document_structural_exact_copy(
        graph=graph,
        query=query,
        query_features=extract_query_features(query),
        seed_episode_ids=["DOC_e01"],
        fallback_source_ids=["DOC"],
    )

    artifact_containers = [row for row in graph["containers"] if row["kind_fq"] == "document:artifact"]
    artifact_render_refs = [
        row
        for row in graph["render_refs"]
        if (row.get("ref_json") or {}).get("ref_type") == "document_artifact_response_text"
    ]
    assert plan is not None
    assert plan["status"] == "failed_closed"
    assert plan["reason"] == "no_scope_domain"
    assert artifact_containers == []
    assert artifact_render_refs == []
    assert validate_container_exact_copy_render_refs(graph) == []


def test_legacy_document_span_renderer_preserves_edge_whitespace():
    from src.episode_packet import _render_document_span_block

    header, raw = _render_document_span_block(
        span_id="DOC::artifact::0001",
        episode_ids=["DOC_e01"],
        episode_lookup={
            "DOC_e01": {
                "raw_original": "  lead line\nbody\n\n",
                "raw_text": "lead line\nbody",
            }
        },
    )

    assert header.startswith("[Document Span:")
    assert raw == "  lead line\nbody\n\n"
