# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

from __future__ import annotations

from uuid import uuid4

from ..container_graph import plan_document_structural_exact_copy
from ..episode_features import extract_query_features
from ..runtime_contracts import QueryExecutorV1


class ContainerGraphExecutor(QueryExecutorV1):
    name = "container_graph"
    priority = 450

    def supports(self, query_type: str, search_family: str | None, packet: dict) -> bool:
        return str(search_family or packet.get("search_family") or "").lower() in {"document", "auto", ""}

    def should_skip(
        self,
        *,
        packet: dict,
        query: str,
        query_type: str,
        episode_lookup: dict[str, dict],
        augmented_facts,
    ) -> bool:
        query_features = extract_query_features(query)
        ordinal = dict((query_features.get("operator_plan") or {}).get("ordinal") or {})
        output_constraints = dict(query_features.get("output_constraints") or {})
        if not ordinal.get("enabled"):
            return True
        if not (output_constraints.get("return_only") or output_constraints.get("prepend_prefix")):
            return True
        return not _packet_has_document_retrieval(packet, episode_lookup)

    async def augment(
        self,
        server,
        *,
        packet: dict,
        query: str,
        query_type: str,
        episode_lookup: dict[str, dict],
        fact_filter,
    ):
        graph = server._ensure_container_graph()
        query_features = extract_query_features(query)
        seed_episode_ids = list(packet.get("retrieved_episode_ids") or [])
        seed_episode_refs = _seed_episode_refs(seed_episode_ids, episode_lookup)
        if not _packet_has_document_retrieval(packet, episode_lookup):
            return packet, None
        fallback_source_ids = _document_fallback_source_ids(packet, seed_episode_ids, episode_lookup)
        plan = plan_document_structural_exact_copy(
            graph=graph,
            query=query,
            query_features=query_features,
            seed_episode_ids=seed_episode_ids,
            seed_episode_refs=seed_episode_refs,
            fallback_source_ids=fallback_source_ids,
            explicit_order_scope_ids=_packet_document_order_scope_ids(packet),
        )
        if plan is None:
            return packet, None

        packet = dict(packet)
        trace = dict(plan.get("trace") or {})
        packet["container_graph_trace"] = trace
        packet["container_path_attempted"] = True
        packet["container_graph_status"] = str(plan.get("status") or "")
        if plan.get("status") == "rendered":
            selected_episode_ids = list(trace.get("selected_episode_ids") or [])
            render_candidate, private_render_candidate = _terminal_render_candidate(plan, query_features)
            register_candidate = getattr(server, "_register_terminal_render_candidate", None)
            if callable(register_candidate):
                register_candidate(private_render_candidate)
            packet["context"] = _render_terminal_render_candidate_context(render_candidate)
            packet["terminal_render_candidate"] = render_candidate
            packet["output_constraints"] = dict(query_features.get("output_constraints") or {})
            packet["query_operator_plan"] = dict(query_features.get("operator_plan") or {})
            packet["actual_injected_episode_ids"] = selected_episode_ids
            packet["retrieved_episode_ids"] = selected_episode_ids
            packet["document_target_span_mode"] = "container_graph_exact_render"
            packet["document_target_span_snippet_mode"] = False
            packet["document_target_span_ids"] = list(trace.get("selected_artifact_span_ids") or [])
            packet["document_target_span_episode_ids"] = selected_episode_ids
        elif plan.get("status") == "failed_closed":
            packet["container_graph_failed_closed"] = True
            packet["context"] = ""
            packet["actual_injected_episode_ids"] = []
        return packet, None

    def should_halt(
        self,
        *,
        packet: dict,
        query: str,
        query_type: str,
        episode_lookup: dict[str, dict],
        augmented_facts,
    ) -> bool:
        return packet.get("container_graph_status") in {"rendered", "failed_closed"}


def _packet_has_document_retrieval(packet: dict, episode_lookup: dict[str, dict]) -> bool:
    search_family = str(packet.get("search_family") or "").lower()
    if search_family == "document":
        return True
    if _packet_document_source_scopes(packet):
        return True
    if _packet_document_order_scope_ids(packet):
        return True
    for episode_id in packet.get("retrieved_episode_ids") or []:
        episode = episode_lookup.get(str(episode_id)) or {}
        if str(episode.get("source_type") or episode.get("source_family") or episode.get("family") or "").lower() == "document":
            return True
    return False


def _packet_document_source_scopes(packet: dict) -> list[str]:
    scope_values: list[str] = []
    for key in ("document_source_ids", "document_scope_ids"):
        value = packet.get(key)
        if isinstance(value, str):
            scope_values.append(value)
        elif isinstance(value, list):
            scope_values.extend(str(item or "") for item in value)
    return [value for value in dict.fromkeys(scope_values) if value]


def _packet_document_order_scope_ids(packet: dict) -> list[str]:
    value = packet.get("document_order_scope_ids")
    if isinstance(value, str):
        scope_values = [value]
    elif isinstance(value, list):
        scope_values = [str(item or "") for item in value]
    else:
        scope_values = []
    return [scope_id for scope_id in dict.fromkeys(scope_values) if scope_id]


def _document_fallback_source_ids(packet: dict, seed_episode_ids: list[str], episode_lookup: dict[str, dict]) -> list[str]:
    source_ids: list[str] = []
    for episode_id in seed_episode_ids:
        episode = episode_lookup.get(str(episode_id)) or {}
        if str(episode.get("source_type") or episode.get("source_family") or episode.get("family") or "").lower() == "document":
            source_ids.append(str(episode.get("source_id") or ""))
    source_ids.extend(_packet_document_source_scopes(packet))
    return [value for value in dict.fromkeys(source_ids) if value]


def _seed_episode_refs(seed_episode_ids: list[str], episode_lookup: dict[str, dict]) -> list[dict]:
    refs: list[dict] = []
    for episode_id in seed_episode_ids:
        episode = episode_lookup.get(str(episode_id)) or {}
        refs.append(
            {
                "episode_id": str(episode_id),
                "doc_id": str(episode.get("doc_id") or episode.get("document_id") or ""),
                "source_id": str(episode.get("source_id") or ""),
            }
        )
    return refs


def _terminal_render_candidate(plan: dict, query_features: dict) -> tuple[dict, dict]:
    trace = dict(plan.get("trace") or {})
    selected_container = dict(plan.get("selected_container") or {})
    selected_render_ref = dict(plan.get("selected_render_ref") or {})
    render_ref_json = dict(selected_render_ref.get("ref_json") or {})
    candidate_id = f"exact_copy_{uuid4().hex}"
    selected_container_ids = list(trace.get("selected_container_ids") or [])
    selected_render_ref_ids = list(trace.get("selected_render_ref_ids") or [])
    selected_artifact_span_ids = list(trace.get("selected_artifact_span_ids") or [])
    output_constraints = dict(query_features.get("output_constraints") or {})
    query_operator_plan = dict(query_features.get("operator_plan") or {})
    requested_selector = {
        "query_type": "exact_copy",
        "operator_kind": trace.get("operator_kind") or "nth",
        "requested_index": trace.get("requested_index"),
        "indexing": trace.get("indexing") or "one_indexed",
        "target_kind": trace.get("target_kind"),
        "target_topic": trace.get("target_topic"),
        "target_anchor_text": trace.get("target_anchor_text"),
        "output_prepend_prefix": output_constraints.get("prepend_prefix"),
        "output_return_only": bool(output_constraints.get("return_only")),
    }
    planner_proof = {
        "planner": trace.get("planner") or "container_graph",
        "planner_contract_version": trace.get("planner_contract_version")
        or (trace.get("query_plan") or {}).get("planner_contract_version"),
        "planner_implementation": trace.get("planner_implementation") or "document_artifact_planner",
        "planner_implementation_scope": trace.get("planner_implementation_scope"),
        "candidate_family": trace.get("candidate_family") or selected_container.get("family") or "document",
        "candidate_kind_fq": trace.get("candidate_kind_fq") or selected_container.get("kind_fq"),
        "selected_index_in_matching_domain": trace.get("selected_index_in_matching_domain"),
        "matching_domain_count": trace.get("matching_domain_count"),
        "ordinal_satisfied": bool(trace.get("ordinal_satisfied")),
        "kind_satisfied": bool(trace.get("kind_satisfied")),
        "topic_satisfied": bool(trace.get("topic_satisfied")),
        "surface_anchor_tokens_requested": list(trace.get("surface_anchor_tokens_requested") or []),
        "surface_anchor_tokens_matched": list(trace.get("surface_anchor_tokens_matched") or []),
        "normalized_anchor_tokens_requested": list(trace.get("normalized_anchor_tokens_requested") or trace.get("anchor_tokens") or []),
        "normalized_anchor_tokens_matched": list(trace.get("normalized_anchor_tokens_matched") or trace.get("anchor_tokens_matched") or []),
        "anchor_tokens_matched": list(trace.get("anchor_tokens_matched") or []),
        "anchor_tokens_missing": list(trace.get("anchor_tokens_missing") or []),
        "proof_source_fields": list(trace.get("proof_source_fields") or []),
        "order_scope_id": trace.get("order_scope_id"),
        "order_basis": trace.get("order_basis"),
        "order_scope": trace.get("order_scope"),
        "no_cross_container_contamination": bool(trace.get("no_cross_container_contamination", True)),
        "candidate_domain_policy": trace.get("candidate_domain_policy")
        or ((trace.get("query_plan") or {}).get("candidate_domain") or {}).get("domain_policy"),
    }
    render_ref_validated = bool(trace.get("render_ref_validated") or trace.get("exact_copy_validated"))
    render_proof = {
        "render_mode": "exact_copy",
        "render_source": trace.get("render_source"),
        "render_ref_validated": render_ref_validated,
        "raw_source_present": bool(trace.get("raw_source_present")),
        "raw_source_provenance": trace.get("raw_source_provenance"),
        "raw_source_validated": bool(trace.get("exact_copy_validated")),
        "whole_or_fail": True,
        "degraded_render_source": trace.get("degraded_render_source"),
    }
    proof_summary = {
        "requested_selector": requested_selector,
        "planner_proof": planner_proof,
        "render_proof": render_proof,
    }
    public_candidate = {
        "candidate_id": candidate_id,
        "capability": "exact_copy",
        "status": "available",
        "query_type": "exact_copy",
        "decision_required": True,
        "whole_or_fail": True,
        "raw_text_exposed_to_model": False,
        "render_mode": "exact_copy",
        "render_source": trace.get("render_source"),
        "raw_source_present": trace.get("raw_source_present"),
        "raw_source_provenance": trace.get("raw_source_provenance"),
        "raw_source_validated": bool(trace.get("exact_copy_validated")),
        "render_ref_validated": render_ref_validated,
        "degraded_render_source": trace.get("degraded_render_source"),
        "operator_kind": requested_selector["operator_kind"],
        "requested_index": requested_selector["requested_index"],
        "indexing": requested_selector["indexing"],
        "target_kind": requested_selector["target_kind"],
        "target_topic": requested_selector["target_topic"],
        "target_anchor_text": requested_selector["target_anchor_text"],
        "output_prepend_prefix": requested_selector["output_prepend_prefix"],
        "output_return_only": requested_selector["output_return_only"],
        "planner": planner_proof["planner"],
        "planner_contract_version": planner_proof["planner_contract_version"],
        "planner_implementation": planner_proof["planner_implementation"],
        "planner_implementation_scope": planner_proof["planner_implementation_scope"],
        "candidate_family": planner_proof["candidate_family"],
        "candidate_kind_fq": planner_proof["candidate_kind_fq"],
        "selected_index_in_matching_domain": planner_proof["selected_index_in_matching_domain"],
        "matching_domain_count": planner_proof["matching_domain_count"],
        "ordinal_satisfied": planner_proof["ordinal_satisfied"],
        "kind_satisfied": planner_proof["kind_satisfied"],
        "topic_satisfied": planner_proof["topic_satisfied"],
        "surface_anchor_tokens_requested": planner_proof["surface_anchor_tokens_requested"],
        "surface_anchor_tokens_matched": planner_proof["surface_anchor_tokens_matched"],
        "normalized_anchor_tokens_requested": planner_proof["normalized_anchor_tokens_requested"],
        "normalized_anchor_tokens_matched": planner_proof["normalized_anchor_tokens_matched"],
        "anchor_tokens_matched": planner_proof["anchor_tokens_matched"],
        "anchor_tokens_missing": planner_proof["anchor_tokens_missing"],
        "proof_source_fields": planner_proof["proof_source_fields"],
        "order_basis": planner_proof["order_basis"],
        "order_scope": planner_proof["order_scope"],
        "candidate_domain_policy": planner_proof["candidate_domain_policy"],
        "requested_selector": requested_selector,
        "planner_proof": planner_proof,
        "render_proof": render_proof,
        "proof_summary": proof_summary,
        "output_constraints": output_constraints,
        "query_operator_plan": query_operator_plan,
        "family": selected_container.get("family") or trace.get("family") or "document",
        "selected_container_ids": selected_container_ids,
        "selected_render_ref_ids": selected_render_ref_ids,
        "selected_episode_ids": list(trace.get("selected_episode_ids") or []),
        "selected_artifact_span_ids": selected_artifact_span_ids,
        "order_scope_id": trace.get("order_scope_id"),
        "no_cross_container_contamination": bool(trace.get("no_cross_container_contamination", True)),
        "container_id": selected_container.get("container_id"),
        "render_ref_id": selected_render_ref.get("render_ref_id"),
        "render_ref_fingerprint": selected_render_ref.get("ref_fingerprint"),
        "render_ref": {
            "render_kind": selected_render_ref.get("render_kind"),
            "render_mode": selected_render_ref.get("render_mode"),
            "fidelity": selected_render_ref.get("fidelity"),
            "render_source": render_ref_json.get("render_source"),
        },
    }
    private_candidate = {
        **public_candidate,
        "render_text": str(plan.get("render_text") or ""),
    }
    return public_candidate, private_candidate


def _candidate_value(candidate: dict, section: str, key: str):
    nested = dict(candidate.get(section) or {})
    return nested.get(key, candidate.get(key))


def _candidate_list(candidate: dict, section: str, key: str) -> list:
    value = _candidate_value(candidate, section, key)
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def _render_terminal_render_candidate_context(candidate: dict) -> str:
    constraints = dict(candidate.get("output_constraints") or {})
    prefix = str(candidate.get("output_prepend_prefix") or constraints.get("prepend_prefix") or "")
    return_only = bool(candidate.get("output_return_only") or constraints.get("return_only"))
    metadata_lines = [
        "--- TERMINAL RENDER CANDIDATE ---",
        "Query type: exact_copy",
        "Capability: exact_copy",
        f"Status: {candidate.get('status') or 'available'}",
        "Contract: choose this candidate only if its metadata satisfies the request; runtime renders the exact text internally.",
        "",
        "Requested selector:",
        f"  operator: {_candidate_value(candidate, 'requested_selector', 'operator_kind')}",
        f"  requested_index: {_candidate_value(candidate, 'requested_selector', 'requested_index')}",
        f"  indexing: {_candidate_value(candidate, 'requested_selector', 'indexing')}",
        f"  target_kind: {_candidate_value(candidate, 'requested_selector', 'target_kind')}",
        f"  target_topic: {_candidate_value(candidate, 'requested_selector', 'target_topic')}",
        f"  target_anchor_text: {_candidate_value(candidate, 'requested_selector', 'target_anchor_text')}",
        f"  output_prefix: {prefix}",
        f"  return_only: {return_only}",
        "",
        "Planner proof:",
        f"  planner: {_candidate_value(candidate, 'planner_proof', 'planner')}",
        f"  planner_contract_version: {_candidate_value(candidate, 'planner_proof', 'planner_contract_version')}",
        f"  planner_implementation: {_candidate_value(candidate, 'planner_proof', 'planner_implementation')}",
        f"  planner_implementation_scope: {_candidate_value(candidate, 'planner_proof', 'planner_implementation_scope')}",
        f"  candidate_family: {_candidate_value(candidate, 'planner_proof', 'candidate_family')}",
        f"  candidate_kind: {_candidate_value(candidate, 'planner_proof', 'candidate_kind_fq')}",
        f"  selected_index_in_matching_domain: {_candidate_value(candidate, 'planner_proof', 'selected_index_in_matching_domain')}",
        f"  matching_domain_count: {_candidate_value(candidate, 'planner_proof', 'matching_domain_count')}",
        f"  ordinal_satisfied: {_candidate_value(candidate, 'planner_proof', 'ordinal_satisfied')}",
        f"  kind_satisfied: {_candidate_value(candidate, 'planner_proof', 'kind_satisfied')}",
        f"  topic_satisfied: {_candidate_value(candidate, 'planner_proof', 'topic_satisfied')}",
        f"  surface_anchor_tokens_requested: {_candidate_list(candidate, 'planner_proof', 'surface_anchor_tokens_requested')}",
        f"  surface_anchor_tokens_matched: {_candidate_list(candidate, 'planner_proof', 'surface_anchor_tokens_matched')}",
        f"  normalized_anchor_tokens_requested: {_candidate_list(candidate, 'planner_proof', 'normalized_anchor_tokens_requested')}",
        f"  normalized_anchor_tokens_matched: {_candidate_list(candidate, 'planner_proof', 'normalized_anchor_tokens_matched')}",
        f"  anchor_tokens_matched: {_candidate_list(candidate, 'planner_proof', 'anchor_tokens_matched')}",
        f"  anchor_tokens_missing: {_candidate_list(candidate, 'planner_proof', 'anchor_tokens_missing')}",
        f"  proof_source_fields: {_candidate_list(candidate, 'planner_proof', 'proof_source_fields')}",
        f"  order_scope_id: {_candidate_value(candidate, 'planner_proof', 'order_scope_id')}",
        f"  order_basis: {_candidate_value(candidate, 'planner_proof', 'order_basis')}",
        f"  order_scope: {_candidate_value(candidate, 'planner_proof', 'order_scope')}",
        f"  no_cross_container_contamination: {_candidate_value(candidate, 'planner_proof', 'no_cross_container_contamination')}",
        f"  candidate_domain_policy: {_candidate_value(candidate, 'planner_proof', 'candidate_domain_policy')}",
        "",
        "Render proof:",
        f"  render_mode: {_candidate_value(candidate, 'render_proof', 'render_mode')}",
        f"  render_source: {_candidate_value(candidate, 'render_proof', 'render_source')}",
        f"  render_ref_validated: {_candidate_value(candidate, 'render_proof', 'render_ref_validated')}",
        f"  raw_source_present: {_candidate_value(candidate, 'render_proof', 'raw_source_present')}",
        f"  raw_source_provenance: {_candidate_value(candidate, 'render_proof', 'raw_source_provenance')}",
        f"  raw_source_validated: {_candidate_value(candidate, 'render_proof', 'raw_source_validated')}",
        f"  whole_or_fail: {_candidate_value(candidate, 'render_proof', 'whole_or_fail')}",
        f"  degraded_render_source: {_candidate_value(candidate, 'render_proof', 'degraded_render_source')}",
        "",
        "Debug identifiers:",
        f"  candidate_id: {candidate.get('candidate_id')}",
        f"  container_id: {candidate.get('container_id')}",
        f"  render_ref_id: {candidate.get('render_ref_id')}",
        f"  selected_containers: {candidate.get('selected_container_ids') or []}",
        f"  selected_render_refs: {candidate.get('selected_render_ref_ids') or []}",
        f"  selected_artifact_spans: {candidate.get('selected_artifact_span_ids') or []}",
        "Raw exact text is hidden from the model.",
        "If this candidate satisfies the query, return exactly one JSON object:",
        f'{{"decision":"use_candidate","candidate_id":"{candidate.get("candidate_id")}"}}',
    ]
    return "\n".join(metadata_lines)
