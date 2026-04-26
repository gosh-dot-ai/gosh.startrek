# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

from src.source_adapters import segment_document_text


def test_segment_document_text_drops_empty_separator_and_doc_id_only_blocks():
    text = """KAR_DOC_A1E

---

# Title

Paragraph with real content.

---

KAR_DOC_E18

| Col | Val |
|-----|-----|
| A   | B   |
"""
    block_dicts, blocks = segment_document_text(text, "doc1")

    rendered = [b["text"] for b in block_dicts]

    assert not any(t.strip() == "" for t in rendered)
    assert not any(t.strip() == "---" for t in rendered)
    assert not any(t.strip() == "KAR_DOC_A1E" for t in rendered)
    assert not any(t.strip() == "KAR_DOC_E18" for t in rendered)
    assert any("Paragraph with real content." in t for t in rendered)
    assert any("| Col | Val |" in t for t in rendered)


def test_segment_document_text_keeps_semantic_heading_blocks():
    text = """# Compliance Register

**Inspection measures:**

1. Monthly monitoring
2. Weekly review
"""
    block_dicts, blocks = segment_document_text(text, "doc2")
    rendered = [b["text"] for b in block_dicts]

    assert any("**Inspection measures:**" in t for t in rendered)
    assert any("Monthly monitoring" in t for t in rendered)
