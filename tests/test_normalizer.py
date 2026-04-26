# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

import pytest

from src.normalizer import (
    dedup_domain_key,
    hamming_distance,
    normalize_text,
    simhash,
)


def test_normalize_text_removes_bom_and_lone_surrogates():
    raw = "\ufeffhello\ud800world"
    assert normalize_text(raw) == "hello\ufffdworld"


def test_normalize_text_normalizes_crlf_and_cr():
    assert normalize_text("a\r\nb\rc") == "a\nb\nc"


def test_normalize_text_applies_nfc():
    decomposed = "Cafe\u0301"
    composed = "Caf\u00e9"
    assert normalize_text(decomposed) == composed


def test_normalize_text_removes_only_safe_invisibles():
    raw = "ab\u200bcd\u2060ef\u00adh\u200e\u202eh"
    assert normalize_text(raw) == "abcdefhh"


def test_normalize_text_preserves_zwnj_and_zwj():
    raw = "a\u200cb\u200dc"
    assert normalize_text(raw) == raw


def test_content_aware_whitespace_compaction_for_conversation():
    raw = "\n\nhello  \n\n\n\nworld\t \n"
    assert normalize_text(raw, family="conversation") == "hello\n\n\nworld"


def test_content_aware_normalization_preserves_fenced_code_blocks():
    raw = "Intro  \n\n```python  \nprint('x')  \n```\n\n\nTail  \n"
    assert normalize_text(raw, family="document") == "Intro\n\n```python  \nprint('x')  \n```\n\n\nTail"


def test_codebase_family_skips_content_aware_transforms():
    raw = "  alpha  \n\n\nbeta—gamma\n"
    assert normalize_text(raw, family="codebase") == "  alpha  \n\n\nbeta—gamma\n"


def test_none_or_unknown_family_uses_universal_only():
    raw = "  alpha  \n\n\nbeta…\n"
    assert normalize_text(raw, family=None) == raw
    assert normalize_text(raw, family="unknown") == raw


def test_smart_punctuation_normalization_for_document():
    raw = "“quoted” ‘single’ foo—bar wow… 2020–2023 and word–word"
    assert normalize_text(raw, family="document") == "\"quoted\" 'single' foo -- bar wow... 2020–2023 and word-word"


def test_chat_alias_uses_conversation_rules():
    raw = "\n“hi”  \n"
    assert normalize_text(raw, family="chat") == "\"hi\""


def test_normalize_text_is_idempotent():
    raw = "\ufeff“Hello”\r\n\r\n\r\nworld\u200b"
    once = normalize_text(raw, family="conversation")
    twice = normalize_text(once, family="conversation")
    assert twice == once


def test_simhash_and_hamming_distance_are_deterministic():
    a = simhash("alpha beta gamma delta")
    b = simhash("alpha beta gamma delta")
    c = simhash("alpha beta gamma epsilon")
    assert a == b
    assert hamming_distance(a, b) == 0
    assert 0 <= hamming_distance(a, c) <= 64


@pytest.mark.parametrize(
    ("scope", "owner_id", "swarm_id", "expected"),
    [
        ("system-wide", "system", None, "system"),
        ("swarm-shared", "agent:alice", "sw1", "swarm:sw1"),
        ("agent-private", "agent:alice", "sw1", "agent:alice"),
    ],
)
def test_dedup_domain_key(scope, owner_id, swarm_id, expected):
    assert dedup_domain_key(scope, owner_id, swarm_id) == expected
