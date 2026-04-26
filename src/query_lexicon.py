# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

from __future__ import annotations

HYDRATION_QUERY_MARKERS = (
    "exact code",
    "exact source",
    "show code",
    "show source",
    "show the source",
    "show the code",
    "signature",
    "parameter",
    "parameters",
    "field",
    "fields",
    "body",
    "implementation",
    "definition",
    "ast",
    "qualified name",
    "python qualified name",
    "code symbol",
    "code function",
    "line ",
    "lines ",
)

CODE_EVIDENCE_QUERY_MARKERS = HYDRATION_QUERY_MARKERS + (
    "callable",
    "function",
    "symbol",
    "python function",
)

PROSE_EVIDENCE_QUERY_MARKERS = (
    "chat",
    "conversation",
    "document",
    "runbook",
    "owner team",
    "mentioned in chat",
    "named in the document",
    "all available memory sources",
    "all available sources",
)

PROMPT_ROUTING_CODE_SLOT_MARKERS = (
    "qualified name",
    "python qualified name",
    "exact python qualified name",
    "which file defines",
    "file defines",
    "exact code",
    "exact source",
    "show the code",
    "show the source",
    "show code",
    "show source",
    "signature",
    "code symbol",
    "symbol name",
    "function name",
    "callable",
    "definition",
    "parameter list",
    "parameters",
)

PROMPT_ROUTING_CODE_CHAIN_MARKERS = (
    "what calls",
    "who calls",
    "called by",
    "calls this",
    "dependency path",
    "dependency chain",
    "call chain",
    "trace the dependency",
    "trace dependency",
    "trace the call",
    "where is this used",
    "what depends on",
    "depends on this",
    "impact path",
)

PROMPT_ROUTING_RISK_REVIEW_MARKERS = (
    "risk review",
    "review risk",
    "regression risk",
    "review this change",
    "review the diff",
    "review the patch",
    "what breaks if",
    "what would break if",
    "blast radius",
    "unsafe change",
    "what could break",
    "side effects",
    "security risk",
)


PROMPT_ROUTING_MIXED_CODE_PROSE_MARKERS = (
    "all available memory sources",
    "all available sources",
    "use chat",
    "use conversation",
    "use document",
    "mentioned in chat",
    "named in the document",
    "owner team",
)

CODEBASE_FILE_LOOKUP_REQUEST_MARKERS = (
    "which file",
    "what file",
    "file defines",
    "file contains",
    "defining file",
    "only the file path",
)

CODEBASE_FILE_LOOKUP_MULTI_PART_MARKERS = (
    "using all available",
    "exactly three labelled lines",
    "exactly three labeled lines",
    "labelled lines",
    "labeled lines",
    "chat_codename",
    "document_owner",
    "code_symbol",
    "document owner",
    "owner team",
    "codename",
    "chat:",
    "document:",
    "code:",
    "\n",
)

CODEBASE_FILE_LOOKUP_NORMALIZE_CLAUSES = (
    "answer with only the file path",
    "return only the file path",
    "just the file path",
    "file path only",
    "answer only with the file path",
)

CODEBASE_FILE_LOOKUP_PATTERNS = (
    r"^which file\b.+\b(?:define|defines|contains|has)\b.+$",
    r"^what file\b.+\b(?:define|defines|contains|has)\b.+$",
    r"^which file is\b.+\bin\b.+$",
    r"^what is the file path\b.+$",
)

CODE_ATTACHMENT_SECTION_LABEL = "--- SOURCE FILES ---"
