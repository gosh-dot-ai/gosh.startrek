# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

from __future__ import annotations

import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HARNESS_SCRIPT = PROJECT_ROOT / "tests" / "live.e2e" / "run_remote_two_agents_atlas_retrieval.sh"
COMPOSE_FILE = PROJECT_ROOT / "tests" / "live.e2e" / "docker-compose.remote-atlas-two-agents.yml"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_live_harness_defines_split_context_and_work_keys() -> None:
    script = _text(HARNESS_SCRIPT)

    assert 'LIVE_E2E_NAMESPACE_BASE="${LIVE_E2E_NAMESPACE_BASE:-${LIVE_E2E_MEMORY_KEY:-atlas-march-review}}"' in script
    assert 'LIVE_E2E_CONTEXT_KEY="${LIVE_E2E_CONTEXT_KEY:-${LIVE_E2E_NAMESPACE_BASE}-context}"' in script
    assert 'LIVE_E2E_WORK_KEY_AGENT_A="${LIVE_E2E_WORK_KEY_AGENT_A:-${LIVE_E2E_NAMESPACE_BASE}-work-agent-a}"' in script
    assert 'LIVE_E2E_WORK_KEY_AGENT_B="${LIVE_E2E_WORK_KEY_AGENT_B:-${LIVE_E2E_NAMESPACE_BASE}-work-agent-b}"' in script
    assert 'log "namespace contract: base=$LIVE_E2E_NAMESPACE_BASE context=$LIVE_E2E_CONTEXT_KEY work_a=$LIVE_E2E_WORK_KEY_AGENT_A work_b=$LIVE_E2E_WORK_KEY_AGENT_B"' in script


def test_live_harness_uses_context_key_for_context_setup_and_recall() -> None:
    script = _text(HARNESS_SCRIPT)

    assert 'init --key "\\$CONTEXT_KEY" --owner-id "\\$CLI_PRINCIPAL_ID"' in script
    assert 'secret set groq "\\$GROQ_VALUE"' in script
    assert '--key "\\$CONTEXT_KEY"' in script
    assert 'secret set openai "\\$OPENAI_VALUE"' in script
    assert 'config set --key "\\$CONTEXT_KEY"' in script
    assert len(
        re.findall(r'ingest document \\\\\n\s+--key "\\\$CONTEXT_KEY" \\\\\n\s+--scope swarm-shared', script)
    ) == 3
    assert 'recall --key "\\$CONTEXT_KEY" "Atlas March summary metrics risks next steps"' in script
    assert 'recall --key "\\$CONTEXT_KEY" \'$TASK_A_TEXT\'' in script
    assert 'recall --key "\\$CONTEXT_KEY" \'$TASK_B_TEXT\'' in script


def test_live_harness_uses_work_key_for_agent_private_task_flow() -> None:
    script = _text(HARNESS_SCRIPT)

    assert 'init --key "\\$WORK_KEY_AGENT_A" --owner-id "agent:agent-a"' in script
    assert 'init --key "\\$WORK_KEY_AGENT_B" --owner-id "agent:agent-b"' in script
    assert 'secret set groq "\\$GROQ_VALUE" --key "\\$WORK_KEY_AGENT_A"' in script
    assert 'secret set groq "\\$GROQ_VALUE" --key "\\$WORK_KEY_AGENT_B"' in script
    assert 'secret set openai "\\$OPENAI_VALUE" --key "\\$WORK_KEY_AGENT_A"' in script
    assert 'secret set openai "\\$OPENAI_VALUE" --key "\\$WORK_KEY_AGENT_B"' in script
    assert 'config set --key "\\$WORK_KEY_AGENT_A"' in script
    assert 'config set --key "\\$WORK_KEY_AGENT_B"' in script
    assert len(
        re.findall(r'task create \\\\\n\s+--key "\\\$WORK_KEY_AGENT_[AB]" \\\\\n\s+--scope agent-private', script)
    ) == 2
    assert "task status '$task_id' --key '$work_key'" in script
    assert "task list --key '$LIVE_E2E_WORK_KEY_AGENT_A'" in script
    assert "task list --key '$LIVE_E2E_WORK_KEY_AGENT_B'" in script
    assert 'query_a_task=$(jq -nc --arg key "$LIVE_E2E_WORK_KEY_AGENT_A"' in script
    assert 'query_b_task=$(jq -nc --arg key "$LIVE_E2E_WORK_KEY_AGENT_B"' in script
    assert 'query_a_result=$(jq -nc --arg key "$LIVE_E2E_WORK_KEY_AGENT_A"' in script
    assert 'query_b_result=$(jq -nc --arg key "$LIVE_E2E_WORK_KEY_AGENT_B"' in script
    assert 'query_a_session=$(jq -nc --arg key "$LIVE_E2E_WORK_KEY_AGENT_A"' in script
    assert 'query_b_session=$(jq -nc --arg key "$LIVE_E2E_WORK_KEY_AGENT_B"' in script
    assert 'query_a_review=$(jq -nc --arg key "$LIVE_E2E_WORK_KEY_AGENT_A"' in script
    assert 'query_b_review=$(jq -nc --arg key "$LIVE_E2E_WORK_KEY_AGENT_B"' in script
    assert 'query_a_attempt=$(jq -nc --arg key "$LIVE_E2E_WORK_KEY_AGENT_A"' in script
    assert 'query_b_attempt=$(jq -nc --arg key "$LIVE_E2E_WORK_KEY_AGENT_B"' in script


def test_live_harness_compose_wires_agent_watch_keys_to_split_namespaces() -> None:
    compose = _text(COMPOSE_FILE)

    assert compose.count("WATCH_KEY: ${LIVE_E2E_WORK_KEY_AGENT_A}") == 1
    assert compose.count("WATCH_KEY: ${LIVE_E2E_WORK_KEY_AGENT_B}") == 1
    assert compose.count("WATCH_CONTEXT_KEY: ${LIVE_E2E_CONTEXT_KEY}") == 2


def test_postmortem_auth_context_exposes_split_namespace_fields() -> None:
    script = _text(HARNESS_SCRIPT)

    assert "namespace_base: $namespace_base," in script
    assert "context_key: $context_key," in script
    assert "work_key_agent_a: $work_key_agent_a," in script
    assert "work_key_agent_b: $work_key_agent_b," in script
    assert "memory_encryption_key: env.LIVE_E2E_MEMORY_ENCRYPTION_KEY," in script
