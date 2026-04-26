<!--
  Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
  SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0

  Licensed under the GOSH.AI Noncommercial License, Version 1.0.
  See LICENSE.md in the root of this repository for the full license
  text. Use of this file for any Commercial Purpose requires a separate
  written license. Contact: legal@gosh.sh
-->

# Documentation Audit Report

**Date:** 2026-03-17
**Scope:** All .md files in repo root and docs/ compared against actual code in src/

---

## Summary

- Files audited: 14 .md files (excluding benchmarks/, archive/, .claude/)
- Discrepancies found: 11
- Fixed in this PR: 2 (CLAUDE.md, specs/SPEC-memory-mcp-v1_4.md)
- Noted but not fixed: 9 (spec plans reference Rust code that was superseded by Python; editorial choices)

---

## Discrepancies Found

### 1. CLAUDE.md -- Source Files section is incomplete

**File:** `CLAUDE.md` lines 107-118

**Problem:** The "Production pipeline (src/)" listing mentions only 5 files:
- `src/common.py`, `src/librarian.py`, `src/retrieval.py`, `src/inference.py`, `src/judge.py`

**Actual src/ contents (18 files):** The following 13 files exist but are not listed:
- `src/cli.py` -- CLI entrypoint (start, import, status, setup commands)
- `src/mcp_server.py` -- MCP server (FastMCP + SSE, 17 tool definitions)
- `src/memory.py` -- MemoryServer class (core storage + extraction orchestration)
- `src/courier.py` -- Courier SSE push system
- `src/config.py` -- MemoryConfig dataclass
- `src/tools.py` -- date_diff(), count_items() deterministic tools
- `src/storage.py` -- disk persistence helpers
- `src/importers.py` -- parse_conversation_json, parse_text, parse_directory
- `src/providers.py` -- multi-provider routing (OpenAI/Groq/Anthropic/Google)
- `src/setup_store.py` -- ~/.gosh-memory/config.json management, API key storage
- `src/git_importer.py` -- git repo import (clone + parse)
- `src/prompt_registry.py` -- Librarian prompt registry (builtin + custom)
- `src/__init__.py` -- package init

**Status:** FIXED -- added complete listing to CLAUDE.md.

---

### 2. CLAUDE.md -- build_data_dict() location not specified

**File:** `CLAUDE.md` line 68

**Problem:** `build_data_dict()` is mentioned in the Wiring Rule narrative as the Sprint 19 root cause, but its actual location (`src/retrieval.py`) is not stated. Readers cannot find it without grepping.

**Actual location:** `src/retrieval.py` line 33.

**Status:** FIXED -- added location to CLAUDE.md Source Files section (retrieval.py line).

---

### 3. specs/SPEC-memory-mcp-v1_4.md -- Implementation Status table is stale

**File:** `specs/SPEC-memory-mcp-v1_4.md` lines 29-34

**Problem:** Six tools are marked "missing" in the implementation status table:
- `memory_store_secret` -- marked missing, actually implemented at `src/mcp_server.py:281`
- `memory_get_secret` -- marked missing, actually implemented at `src/mcp_server.py:295`
- `memory_import_history` -- marked missing, actually implemented at `src/mcp_server.py:307`
- `memory_list_prompts` -- marked missing, actually implemented at `src/mcp_server.py:358`
- `memory_get_prompt` -- marked missing, actually implemented at `src/mcp_server.py:365`
- `memory_set_prompt` -- marked missing, actually implemented at `src/mcp_server.py:381`

**Status:** FIXED -- updated all six to "implemented" in the spec.

---

### 4. README.md -- CLI reference missing `setup` command

**File:** `README.md`

**Problem:** The CLI reference documents three commands: `start`, `status`, `import`. The actual CLI (`src/cli.py`) has four commands -- `setup` is missing from the README.

`gosh-memory setup` is an interactive wizard for configuring providers, API keys, and models. It also supports non-interactive mode (`--provider` + `--api-key`) and `--show` for current config display.

**Status:** Noted. Not fixed (out of scope for this PR -- README changes are a separate concern).

---

### 5. README.md -- MCP tool count matches (17) but tool list is incomplete in description

**File:** `README.md` lines 253-274

**Problem:** The MCP Tools table lists 17 tools, which matches the 17 `@mcp.tool` decorators in code. However, tool descriptions are very brief and do not mention v1.3/v1.4 parameters (upsert_by_key, content_type, librarian_prompt, skip_extraction on memory_store).

**Status:** Noted. Editorial -- the brief table is intentional for a README.

---

### 6. docs/docs-index.md and docs/docs-openclaw.md -- tool count "17" is correct

**File:** `docs/docs-index.md` line 7, `docs/docs-openclaw.md` line 41

**Finding:** Both reference "17 tools". Actual count in `src/mcp_server.py` is 17. This is CORRECT.

**Status:** No action needed.

---

### 7. specs/PLAN-memory-implementation-v7.md -- references Rust architecture

**File:** `specs/PLAN-memory-implementation-v7.md`

**Problem:** This plan references Rust-based architecture (`src/lib.rs`, `gosh-memory-server.rs`, `Cargo.toml`, sqlite-vec, LanceDB). The actual implementation is entirely Python. The plan is a historical design document from before the Python prototype was built.

**Status:** Noted. Historical document, not actionable.

---

### 8. specs/PLAN-memory-tests-v1.md -- references Rust test structure

**File:** `specs/PLAN-memory-tests-v1.md`

**Problem:** Test plan references `mechanical/storage.rs`, `llm/mock.rs`, Rust test layers. Actual tests are Python (`tests/test_*.py`). The plan predates the Python implementation.

**Status:** Noted. Historical document, not actionable.

---

### 9. specs/SPEC-memory-context-v1_2.md and v1_3.md -- spec predates Python rewrite

**Files:** `specs/SPEC-memory-context-v1_2.md`, `specs/SPEC-memory-context-v1_3.md`

**Problem:** These specs describe the Rust memory engine design (SQL tables, LanceDB, etc.). The current Python implementation uses JSON files + numpy embeddings, not SQL/LanceDB.

**Status:** Noted. Historical specs; the active spec is `specs/SPEC-memory-mcp-v1_4.md`.

---

### 10. librarian-inputs-survey.md -- duplicate file at repo root

**File:** `librarian-inputs-survey.md` (root) vs `specs/librarian-inputs-survey.md`

**Problem:** Identical file exists at both locations. The root copy appears to be accidental.

**Status:** Noted. Not fixed (data safety -- not deleting files per project rules).

---

### 11. specs/status.md -- LongMemEval best score may be stale

**File:** `specs/status.md` line 46

**Problem:** Reports best LongMemEval score as 80.0% (Sprint 19, 50q subset, GPT-4o). Sprint 22 results exist in `benchmarks/longmemeval/results/sprint-22-baseline-fixes/` which may contain updated numbers. The status file's "Current Status" section has a "Last updated: 2026-03-13" date.

**Status:** Noted. Sprint 22 results should be reviewed and status.md updated when finalized.

---

## Files Verified as Correct

| File | Status |
|------|--------|
| `README.md` -- CLI flags (host, port, data-dir defaults) | Matches `src/cli.py` argparse definitions |
| `README.md` -- MCP endpoint paths (/mcp, /mcp/sse) | Matches `src/mcp_server.py` route definitions |
| `README.md` -- default port 8765 | Matches cli.py and mcp_server.py |
| `README.md` -- default host 127.0.0.1 | Matches cli.py and mcp_server.py |
| `README.md` -- storage format ({key}.json, {key}_embs.npz) | Consistent with implementation |
| `docs/docs-claude-code.md` | All examples use correct tool names and parameters |
| `docs/docs-gemini-cli.md` | Configuration path and examples correct |
| `docs/docs-openai-codex.md` | Configuration path and examples correct |
| `docs/docs-openclaw.md` | Configuration path, tool count (17), examples correct |
| `CLAUDE.md` -- embedding model (text-embedding-3-large) | Matches `src/common.py` default |
| `CLAUDE.md` -- architecture description (3-tier) | Matches `src/librarian.py` pipeline |
| `CLAUDE.md` -- retrieval components | All present in `src/retrieval.py` |
| `karnali/README.md` | Internal to karnali, consistent with karnali/ contents |
