# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

import pytest

# ── should_accept ──


def test_should_accept_passes_when_all_gates_met():
    from src.mal.eval_runner import should_accept
    old = {
        "episode_hit_rate": 0.58,
        "packet_support_rate": 0.54,
        "answer_accuracy_rate": 0.68,
        "operator_breakdown": {
            "lookup": {"answer_accuracy_rate": 0.74},
            "temporal": {"answer_accuracy_rate": 0.67},
            "ordinal": {"answer_accuracy_rate": 0.61},
            "commonality": {"answer_accuracy_rate": 0.60},
            "list_set": {"answer_accuracy_rate": 0.58},
            "compare_diff": {"answer_accuracy_rate": 0.55},
            "local_anchor": {"answer_accuracy_rate": 0.62},
            "bounded_chain": {"answer_accuracy_rate": 0.59},
        },
    }
    new = {
        "episode_hit_rate": 0.63,
        "packet_support_rate": 0.54,
        "answer_accuracy_rate": 0.68,
        "operator_breakdown": {
            "lookup": {"answer_accuracy_rate": 0.74},
            "temporal": {"answer_accuracy_rate": 0.67},
            "ordinal": {"answer_accuracy_rate": 0.61},
            "commonality": {"answer_accuracy_rate": 0.60},
            "list_set": {"answer_accuracy_rate": 0.58},
            "compare_diff": {"answer_accuracy_rate": 0.55},
            "local_anchor": {"answer_accuracy_rate": 0.62},
            "bounded_chain": {"answer_accuracy_rate": 0.59},
        },
    }
    assert should_accept(old, new) is True


def test_should_accept_rejects_insufficient_episode_hit_delta():
    from src.mal.eval_runner import should_accept
    old = {"episode_hit_rate": 0.58, "packet_support_rate": 0.54, "answer_accuracy_rate": 0.68,
           "operator_breakdown": {op: {"answer_accuracy_rate": 0.70} for op in
               ["lookup", "temporal", "ordinal", "commonality", "list_set", "compare_diff", "local_anchor", "bounded_chain"]}}
    new = dict(old)
    new["episode_hit_rate"] = 0.59  # delta only 0.01, need > 0.02
    assert should_accept(old, new) is False


def test_should_accept_rejects_packet_regression():
    from src.mal.eval_runner import should_accept
    old = {"episode_hit_rate": 0.58, "packet_support_rate": 0.54, "answer_accuracy_rate": 0.68,
           "operator_breakdown": {op: {"answer_accuracy_rate": 0.70} for op in
               ["lookup", "temporal", "ordinal", "commonality", "list_set", "compare_diff", "local_anchor", "bounded_chain"]}}
    new = dict(old)
    new["episode_hit_rate"] = 0.65
    new["packet_support_rate"] = 0.51  # regression > 0.02
    assert should_accept(old, new) is False


def test_should_accept_rejects_aggregate_answer_accuracy_regression():
    from src.mal.eval_runner import should_accept
    old = {"episode_hit_rate": 0.58, "packet_support_rate": 0.54, "answer_accuracy_rate": 0.68,
           "operator_breakdown": {op: {"answer_accuracy_rate": 0.70} for op in
               ["lookup", "temporal", "ordinal", "commonality", "list_set", "compare_diff", "local_anchor", "bounded_chain"]}}
    new = dict(old)
    new["episode_hit_rate"] = 0.65
    new["answer_accuracy_rate"] = 0.65  # regression > 0.02
    assert should_accept(old, new) is False


def test_should_accept_rejects_per_operator_regression():
    from src.mal.eval_runner import should_accept
    old = {"episode_hit_rate": 0.58, "packet_support_rate": 0.54, "answer_accuracy_rate": 0.68,
           "operator_breakdown": {op: {"answer_accuracy_rate": 0.70} for op in
               ["lookup", "temporal", "ordinal", "commonality", "list_set", "compare_diff", "local_anchor", "bounded_chain"]}}
    new = {
        "episode_hit_rate": 0.65,
        "packet_support_rate": 0.54,
        "answer_accuracy_rate": 0.68,
        "operator_breakdown": {op: {"answer_accuracy_rate": 0.70} for op in
            ["lookup", "temporal", "ordinal", "commonality", "list_set", "compare_diff", "local_anchor", "bounded_chain"]},
    }
    new["operator_breakdown"]["temporal"] = {"answer_accuracy_rate": 0.64}  # regression > 0.05
    assert should_accept(old, new) is False


def test_should_accept_allows_small_per_operator_regression():
    from src.mal.eval_runner import should_accept
    old = {"episode_hit_rate": 0.58, "packet_support_rate": 0.54, "answer_accuracy_rate": 0.68,
           "operator_breakdown": {op: {"answer_accuracy_rate": 0.70} for op in
               ["lookup", "temporal", "ordinal", "commonality", "list_set", "compare_diff", "local_anchor", "bounded_chain"]}}
    new = {
        "episode_hit_rate": 0.65,
        "packet_support_rate": 0.54,
        "answer_accuracy_rate": 0.68,
        "operator_breakdown": {op: {"answer_accuracy_rate": 0.70} for op in
            ["lookup", "temporal", "ordinal", "commonality", "list_set", "compare_diff", "local_anchor", "bounded_chain"]},
    }
    new["operator_breakdown"]["temporal"] = {"answer_accuracy_rate": 0.66}  # regression 0.04, within 0.05
    assert should_accept(old, new) is True


# ── TRACE_INCOMPLETE fairness ──


def test_trace_incomplete_baseline_excluded_from_both():
    from src.mal.eval_runner import apply_trace_incomplete_fairness
    baseline_results = [
        {"qid": "q1", "status": "ok", "episode_hit": 1.0},
        {"qid": "q2", "status": "TRACE_INCOMPLETE", "episode_hit": 0.0},
        {"qid": "q3", "status": "ok", "episode_hit": 0.0},
    ]
    candidate_results = [
        {"qid": "q1", "status": "ok", "episode_hit": 1.0},
        {"qid": "q2", "status": "ok", "episode_hit": 1.0},
        {"qid": "q3", "status": "ok", "episode_hit": 1.0},
    ]
    b, c = apply_trace_incomplete_fairness(baseline_results, candidate_results)
    assert len(b) == 2  # q2 excluded
    assert len(c) == 2
    assert all(r["qid"] != "q2" for r in b)


def test_trace_incomplete_candidate_scored_as_zero():
    from src.mal.eval_runner import apply_trace_incomplete_fairness
    baseline_results = [
        {"qid": "q1", "status": "ok", "episode_hit": 1.0, "packet_support": 1.0, "answer_accuracy": 0.8},
    ]
    candidate_results = [
        {"qid": "q1", "status": "TRACE_INCOMPLETE", "episode_hit": 0.0, "packet_support": 0.0, "answer_accuracy": 0.0},
    ]
    b, c = apply_trace_incomplete_fairness(baseline_results, candidate_results)
    assert len(c) == 1
    assert c[0]["episode_hit"] == 0.0
    assert c[0]["packet_support"] == 0.0
    assert c[0]["answer_accuracy"] == 0.0


# ── abort conditions ──


def test_baseline_skip_rate_above_15_percent_aborts():
    from src.mal.eval_runner import check_eval_coverage
    total = 20
    skipped = 4  # 20% > 15%
    result = check_eval_coverage(total_questions=total, skipped_questions=skipped, operator_counts={})
    assert result["error"] == "EVAL_INSUFFICIENT_COVERAGE"


def test_operator_below_3_evaluable_questions_aborts():
    from src.mal.eval_runner import check_eval_coverage
    operator_counts = {
        "lookup": 5, "temporal": 5, "ordinal": 2,  # < 3
        "commonality": 5, "list_set": 5, "compare_diff": 5,
        "local_anchor": 5, "bounded_chain": 5,
    }
    result = check_eval_coverage(total_questions=40, skipped_questions=0, operator_counts=operator_counts)
    assert result["error"] == "EVAL_OPERATOR_COVERAGE_INSUFFICIENT"


def test_coverage_ok_returns_none():
    from src.mal.eval_runner import check_eval_coverage
    operator_counts = dict.fromkeys(["lookup", "temporal", "ordinal", "commonality", "list_set", "compare_diff", "local_anchor", "bounded_chain"], 5)
    result = check_eval_coverage(total_questions=40, skipped_questions=0, operator_counts=operator_counts)
    assert result is None
