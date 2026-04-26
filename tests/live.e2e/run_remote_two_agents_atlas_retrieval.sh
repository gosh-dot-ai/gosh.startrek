#!/usr/bin/env bash
# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
FIXTURES_DIR="$SCRIPT_DIR/fixtures"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.remote-atlas-two-agents.yml"
RUN_STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN_ID="remote-atlas-two-agents-${RUN_STAMP}"
ARTIFACTS_DIR="${LIVE_E2E_ARTIFACTS_DIR:-$SCRIPT_DIR/artifacts/$RUN_ID}"
SUMMARY_FILE="$ARTIFACTS_DIR/summary.txt"
VERSIONS_FILE="$ARTIFACTS_DIR/versions.json"
OPERATOR_LOG="$ARTIFACTS_DIR/operator-cli.log"
OPERATOR_CONTAINER_LOG="$ARTIFACTS_DIR/operator-container.log"
MEMORY_LOG="$ARTIFACTS_DIR/memory.log"
AGENT_A_LOG="$ARTIFACTS_DIR/agent-a.log"
AGENT_B_LOG="$ARTIFACTS_DIR/agent-b.log"
COMPOSE_PS_FILE="$ARTIFACTS_DIR/docker-compose.ps.txt"
RUNTIME_ROOT="$ARTIFACTS_DIR/runtime"
GENERATED_DIR="$ARTIFACTS_DIR/generated"
VERIFY_DIR="$ARTIFACTS_DIR/verify"
BOOTSTRAP_DIR="$ARTIFACTS_DIR/bootstrap"
COMPOSE_PROJECT="atlas${RUN_STAMP,,}"
POSTMORTEM_DIR="$ARTIFACTS_DIR/postmortem"

SIBLING_ROOT=$(cd -- "$PROJECT_ROOT/.." && pwd)
DEFAULT_REVIEWED_CLI_REPO="/tmp/gosh-cli-pr24-worktree"
DEFAULT_REVIEWED_AGENT_REPO="/tmp/gosh-agent-pr29-worktree"

default_cli_repo() {
  if [[ -d "$DEFAULT_REVIEWED_CLI_REPO" ]]; then
    printf '%s\n' "$DEFAULT_REVIEWED_CLI_REPO"
  else
    printf '%s\n' "$SIBLING_ROOT/gosh.cli"
  fi
}

default_agent_repo() {
  if [[ -d "$DEFAULT_REVIEWED_AGENT_REPO" ]]; then
    printf '%s\n' "$DEFAULT_REVIEWED_AGENT_REPO"
  else
    printf '%s\n' "$SIBLING_ROOT/gosh.agent"
  fi
}

CLI_REPO="${LIVE_E2E_GOSH_CLI_REPO:-$(default_cli_repo)}"
AGENT_REPO="${LIVE_E2E_GOSH_AGENT_REPO:-$(default_agent_repo)}"
MEMORY_REPO="$PROJECT_ROOT"

LIVE_E2E_MEMORY_INSTANCE_NAME="${LIVE_E2E_MEMORY_INSTANCE_NAME:-atlas-live-e2e}"
LIVE_E2E_NAMESPACE_BASE="${LIVE_E2E_NAMESPACE_BASE:-${LIVE_E2E_MEMORY_KEY:-atlas-march-review}}"
LIVE_E2E_CONTEXT_KEY="${LIVE_E2E_CONTEXT_KEY:-${LIVE_E2E_NAMESPACE_BASE}-context}"
LIVE_E2E_WORK_KEY_AGENT_A="${LIVE_E2E_WORK_KEY_AGENT_A:-${LIVE_E2E_NAMESPACE_BASE}-work-agent-a}"
LIVE_E2E_WORK_KEY_AGENT_B="${LIVE_E2E_WORK_KEY_AGENT_B:-${LIVE_E2E_NAMESPACE_BASE}-work-agent-b}"
LIVE_E2E_SWARM_ID="${LIVE_E2E_SWARM_ID:-atlas-review}"
LIVE_E2E_TASK_TIMEOUT_SECS="${LIVE_E2E_TASK_TIMEOUT_SECS:-300}"
LIVE_E2E_AGENT_START_TIMEOUT_SECS="${LIVE_E2E_AGENT_START_TIMEOUT_SECS:-180}"
LIVE_E2E_INFERENCE_MODEL="${LIVE_E2E_INFERENCE_MODEL:-qwen/qwen3-32b}"
LIVE_E2E_EXTRACTION_MODEL="${LIVE_E2E_EXTRACTION_MODEL:-qwen/qwen3-32b}"
LIVE_E2E_JUDGE_MODEL="${LIVE_E2E_JUDGE_MODEL:-qwen/qwen3-32b}"
LIVE_E2E_EMBED_MODE="${LIVE_E2E_EMBED_MODE:-openai}"
LIVE_E2E_EMBED_MODEL="${LIVE_E2E_EMBED_MODEL:-text-embedding-3-large}"
LIVE_E2E_SERVER_TOKEN="${LIVE_E2E_SERVER_TOKEN:-atlas-server-token-${RUN_STAMP}}"
LIVE_E2E_ADMIN_TOKEN="${LIVE_E2E_ADMIN_TOKEN:-atlas-admin-token-${RUN_STAMP}}"
LIVE_E2E_MEMORY_ENCRYPTION_KEY="${LIVE_E2E_MEMORY_ENCRYPTION_KEY:-$(python3 - <<'EOF_PY'
import secrets
print(secrets.token_hex(32))
EOF_PY
)}"
LIVE_E2E_MEMORY_TAG="${LIVE_E2E_MEMORY_TAG:-gosh-memory-live-e2e:atlas-review}"
LIVE_E2E_RUNTIME_TAG="${LIVE_E2E_RUNTIME_TAG:-gosh-runtime-live-e2e:atlas-review}"
LIVE_E2E_PRESERVE_ON_FAILURE="${LIVE_E2E_PRESERVE_ON_FAILURE:-1}"

TASK_A_TEXT='Напиши короткое письмо для стейкхолдеров по итогам марта по проекту Atlas. В письме нужен: 1) общий итог месяца, 2) три ключевые метрики, 3) один риск или ограничение, 4) два следующих шага. Пиши профессионально и кратко.'
TASK_B_TEXT='Подготовь короткий internal status update по проекту Atlas за март. Нужны разделы: Highlights, Metrics, Risks, Next Steps. Используй конкретные данные и пиши без воды.'

MEMORY_CONFIG_FILE="$GENERATED_DIR/memory-config.json"
OPERATOR_SETUP_SCRIPT="$GENERATED_DIR/operator-setup.sh"
OPERATOR_TASK_SCRIPT="$GENERATED_DIR/operator-tasks.sh"
PRETASK_RECALL_JSON="$VERIFY_DIR/pretask-recall-atlas.json"
PRETASK_RECALL_TASK_A_JSON="$VERIFY_DIR/pretask-recall-agent-a-task.json"
PRETASK_RECALL_TASK_B_JSON="$VERIFY_DIR/pretask-recall-agent-b-task.json"
RECALL_EVIDENCE_FILE="$VERIFY_DIR/memory-recall-evidence.txt"
SECRET_RESOLVE_EVIDENCE_FILE="$VERIFY_DIR/secret-resolve-evidence.txt"
TASK_A_CREATE_JSON="$VERIFY_DIR/task-create-agent-a.json"
TASK_B_CREATE_JSON="$VERIFY_DIR/task-create-agent-b.json"
TASK_A_STATUS_JSON="$VERIFY_DIR/task-status-agent-a.json"
TASK_B_STATUS_JSON="$VERIFY_DIR/task-status-agent-b.json"
TASK_LIST_A_JSON="$VERIFY_DIR/task-list-agent-a.json"
TASK_LIST_B_JSON="$VERIFY_DIR/task-list-agent-b.json"
TASK_FACTS_JSON="$VERIFY_DIR/memory-query-task.json"
TASK_RESULT_FACTS_JSON="$VERIFY_DIR/memory-query-task_result.json"
TASK_SESSION_FACTS_JSON="$VERIFY_DIR/memory-query-task_session.json"
TASK_FACTS_A_JSON="$VERIFY_DIR/task-facts-agent-a.json"
TASK_FACTS_B_JSON="$VERIFY_DIR/task-facts-agent-b.json"
TASK_RESULTS_A_JSON="$VERIFY_DIR/task-results-agent-a.json"
TASK_RESULTS_B_JSON="$VERIFY_DIR/task-results-agent-b.json"
TASK_SESSIONS_A_JSON="$VERIFY_DIR/task-sessions-agent-a.json"
TASK_SESSIONS_B_JSON="$VERIFY_DIR/task-sessions-agent-b.json"
TASK_REVIEW_FACTS_JSON="$VERIFY_DIR/memory-query-task_review.json"
TASK_ATTEMPT_FACTS_JSON="$VERIFY_DIR/memory-query-task_attempt.json"
TASK_REVIEW_A_JSON="$VERIFY_DIR/task-review-agent-a.json"
TASK_REVIEW_B_JSON="$VERIFY_DIR/task-review-agent-b.json"
TASK_ATTEMPTS_A_JSON="$VERIFY_DIR/task-attempts-agent-a.json"
TASK_ATTEMPTS_B_JSON="$VERIFY_DIR/task-attempts-agent-b.json"
TASK_A_ID_FILE="$GENERATED_DIR/task-agent-a.id"
TASK_B_ID_FILE="$GENERATED_DIR/task-agent-b.id"
POSTMORTEM_AUTH_CONTEXT_JSON="$POSTMORTEM_DIR/auth-context.json"
POSTMORTEM_PROJECT_FILE="$POSTMORTEM_DIR/compose-project.txt"
POSTMORTEM_VOLUMES_FILE="$POSTMORTEM_DIR/volume-names.txt"
POSTMORTEM_MEMORY_CONTAINER_FILE="$POSTMORTEM_DIR/memory-container.txt"
POSTMORTEM_MEMORY_MOUNTS_JSON="$POSTMORTEM_DIR/memory-container-mounts.json"
POSTMORTEM_DATA_FILE_LIST="$POSTMORTEM_DIR/data-files.txt"
POSTMORTEM_DB_CANDIDATES_FILE="$POSTMORTEM_DIR/db-candidates.txt"
POSTMORTEM_DB_COPY_DIR="$POSTMORTEM_DIR/data-copy"
POSTMORTEM_COMMANDS_FILE="$POSTMORTEM_DIR/postmortem-commands.txt"

MEMORY_HOME_DIR="$RUNTIME_ROOT/memory-home"
OPERATOR_HOME_DIR="$RUNTIME_ROOT/operator-home"
AGENT_A_HOME_DIR="$RUNTIME_ROOT/agent-a-home"
AGENT_B_HOME_DIR="$RUNTIME_ROOT/agent-b-home"
AGENT_A_BOOTSTRAP_DIR="$BOOTSTRAP_DIR/agent-a"
AGENT_B_BOOTSTRAP_DIR="$BOOTSTRAP_DIR/agent-b"
AGENT_A_BOOTSTRAP_HOST_FILE="$AGENT_A_BOOTSTRAP_DIR/agent-a-bootstrap.json"
AGENT_B_BOOTSTRAP_HOST_FILE="$AGENT_B_BOOTSTRAP_DIR/agent-b-bootstrap.json"

mkdir -p \
  "$ARTIFACTS_DIR" \
  "$GENERATED_DIR" \
  "$VERIFY_DIR" \
  "$BOOTSTRAP_DIR" \
  "$POSTMORTEM_DIR" \
  "$MEMORY_HOME_DIR" \
  "$OPERATOR_HOME_DIR" \
  "$AGENT_A_HOME_DIR" \
  "$AGENT_B_HOME_DIR" \
  "$AGENT_A_BOOTSTRAP_DIR" \
  "$AGENT_B_BOOTSTRAP_DIR" \
  "$POSTMORTEM_DB_COPY_DIR"
: >"$SUMMARY_FILE"
: >"$OPERATOR_LOG"

log() {
  printf '[live.e2e] %s\n' "$*" | tee -a "$SUMMARY_FILE"
}

die() {
  log "ERROR: $*"
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

require_env() {
  local name="$1"
  [[ -n "${!name:-}" ]] || die "required env var is not set: $name"
}

load_keyring_secret() {
  local name="$1"
  python3 - "$name" <<'EOF_PY'
import sys

try:
    import keyring
except Exception:
    raise SystemExit(1)

name = sys.argv[1]
value = keyring.get_password("gosh-memory", name)
if not value:
    raise SystemExit(1)
sys.stdout.write(value)
EOF_PY
}

resolve_host_secret() {
  local name="$1"
  local current="${!name:-}"
  local resolved=""

  if resolved="$(load_keyring_secret "$name" 2>/dev/null)"; then
    export "$name=$resolved"
    log "resolved $name from host Secret Service"
    return 0
  fi

  if [[ -n "$current" ]]; then
    export "$name=$current"
    log "resolved $name from host environment"
    return 0
  fi

  die "required secret is not available in host Secret Service or environment: $name"
}

compose() {
  docker compose -p "$COMPOSE_PROJECT" -f "$COMPOSE_FILE" "$@"
}

service_container_id() {
  compose ps -q "$1" 2>/dev/null || true
}

service_is_running() {
  local cid
  cid=$(service_container_id "$1")
  [[ -n "$cid" ]] || return 1
  [[ "$(docker inspect --format '{{.State.Status}}' "$cid" 2>/dev/null || true)" == "running" ]]
}

assert_local_image_present() {
  local image_ref="$1"
  docker image inspect "$image_ref" >/dev/null 2>&1 || die "expected local Docker image to exist after build: $image_ref"
}

assert_compose_watch_env() {
  local resolved
  resolved=$(compose config)
  COMPOSE_CONFIG_RESOLVED="$resolved" python3 - "$LIVE_E2E_WORK_KEY_AGENT_A" "$LIVE_E2E_WORK_KEY_AGENT_B" "$LIVE_E2E_CONTEXT_KEY" "$LIVE_E2E_SWARM_ID" <<'EOF_PY'
import os
import re
import sys

resolved = os.environ["COMPOSE_CONFIG_RESOLVED"]
watch_key_a = sys.argv[1]
watch_key_b = sys.argv[2]
watch_context_key = sys.argv[3]
watch_swarm = sys.argv[4]

if not re.search(rf'WATCH_KEY:\s*"?{re.escape(watch_key_a)}"?', resolved):
    raise SystemExit(f"WATCH_KEY for agent-a did not resolve in docker compose config: {watch_key_a!r}")
if not re.search(rf'WATCH_KEY:\s*"?{re.escape(watch_key_b)}"?', resolved):
    raise SystemExit(f"WATCH_KEY for agent-b did not resolve in docker compose config: {watch_key_b!r}")
if not re.search(rf'WATCH_CONTEXT_KEY:\s*"?{re.escape(watch_context_key)}"?', resolved):
    raise SystemExit(f"WATCH_CONTEXT_KEY did not resolve in docker compose config: {watch_context_key!r}")
if not re.search(rf'WATCH_SWARM_ID:\s*"?{re.escape(watch_swarm)}"?', resolved):
    raise SystemExit(f"WATCH_SWARM_ID did not resolve in docker compose config: {watch_swarm!r}")
EOF_PY
}

resolve_binary() {
  local repo_dir="$1"
  local binary_name="$2"
  local debug_binary="$repo_dir/target/debug/$binary_name"
  local release_binary="$repo_dir/target/release/$binary_name"
  if [[ -x "$debug_binary" ]]; then
    printf '%s
' "$debug_binary"
    return 0
  fi
  if [[ -x "$release_binary" ]]; then
    printf '%s
' "$release_binary"
    return 0
  fi
  die "binary not found for $binary_name under $repo_dir/target/{debug,release}"
}

assert_cli_binary_contract() {
  local cli_bin="$1"
  local help_text
  help_text=$("$cli_bin" --help 2>&1 || true)
  [[ "$help_text" == *"memory"* ]] || die "selected gosh binary does not expose memory command: $cli_bin"
  [[ "$help_text" == *"agent"* ]] || die "selected gosh binary does not expose agent command: $cli_bin"
  [[ "$help_text" == *"--test-mode"* ]] || die "selected gosh binary does not expose --test-mode: $cli_bin"

  help_text=$("$cli_bin" memory --help 2>&1 || true)
  [[ "$help_text" == *"auth"* ]] || die "selected gosh binary does not expose reviewed memory auth surface: $cli_bin"
  [[ "$help_text" == *"data"* ]] || die "selected gosh binary does not expose reviewed memory data surface: $cli_bin"
}

assert_agent_binary_contract() {
  local agent_bin="$1"
  local help_text
  help_text=$("$agent_bin" --help 2>&1 || true)
  [[ "$help_text" == *"serve"* ]] || die "selected gosh-agent binary does not expose serve subcommand: $agent_bin"

  help_text=$("$agent_bin" serve --help 2>&1 || true)
  [[ "$help_text" == *"--bootstrap-file"* ]] || die "selected gosh-agent binary does not expose reviewed --bootstrap-file contract: $agent_bin"
}

sanitize_artifacts() {
  return 0
}

cleanup() {
  local rc=$?
  local preserve_on_failure=0
  set +e
  [[ "$LIVE_E2E_PRESERVE_ON_FAILURE" == "1" ]] && preserve_on_failure=1
  if [[ -f "$COMPOSE_FILE" ]]; then
    compose ps >"$COMPOSE_PS_FILE" 2>/dev/null || true
    compose logs memory >"$MEMORY_LOG" 2>&1 || true
	    compose logs operator >"$OPERATOR_CONTAINER_LOG" 2>&1 || true
	    compose logs agent-a >"$AGENT_A_LOG" 2>&1 || true
	    compose logs agent-b >"$AGENT_B_LOG" 2>&1 || true
	    write_postmortem_auth_context || true
	    best_effort_export_private_task_artifacts
	    best_effort_export_review_attempt_artifacts
	    if (( rc != 0 )); then
	      capture_postmortem_state
	      write_failure_postmortem_summary
    fi
    if (( rc == 0 )); then
      compose down -v --remove-orphans >/dev/null 2>&1 || true
    elif (( preserve_on_failure == 0 )); then
      compose down -v --remove-orphans >/dev/null 2>&1 || true
    fi
  fi
  if (( rc == 0 )); then
    log "SUCCESS: artifacts preserved at $ARTIFACTS_DIR"
  else
    if (( preserve_on_failure == 1 )); then
      log "FAILURE: stack preserved for postmortem at $ARTIFACTS_DIR"
    else
      log "FAILURE: inspect artifacts at $ARTIFACTS_DIR"
    fi
  fi
  exit "$rc"
}
trap cleanup EXIT

write_versions_file() {
  python3 - <<EOF_PY >"$VERSIONS_FILE"
import json
import subprocess

def sha(path: str) -> str:
    return subprocess.check_output(['git', '-C', path, 'rev-parse', 'HEAD'], text=True).strip()

print(json.dumps({
    'gosh.memory': sha('$MEMORY_REPO'),
    'gosh.cli': sha('$CLI_REPO'),
    'gosh.agent': sha('$AGENT_REPO'),
}, indent=2))
EOF_PY
}

provider_secret_name_for_model() {
  local model="$1"
  local prefix
  if [[ "$model" == */* ]]; then
    prefix=${model%%/*}
    case "$prefix" in
      openai|qwen)
        printf '%s\n' 'groq'
        ;;
      inception)
        printf '%s\n' 'inception'
        ;;
      *)
        printf '%s\n' "$prefix"
        ;;
    esac
  else
    printf '%s\n' 'openai'
  fi
}

write_memory_config_json() {
  local inference_secret_name
  local librarian_secret_name
  local judge_secret_name
  inference_secret_name=$(provider_secret_name_for_model "$LIVE_E2E_INFERENCE_MODEL")
  librarian_secret_name=$(provider_secret_name_for_model "$LIVE_E2E_EXTRACTION_MODEL")
  judge_secret_name=$(provider_secret_name_for_model "$LIVE_E2E_JUDGE_MODEL")
  [[ "$LIVE_E2E_EMBED_MODE" == "openai" ]] || die "current live harness requires LIVE_E2E_EMBED_MODE=openai"

  jq -n \
    --arg embed_model "$LIVE_E2E_EMBED_MODEL" \
    --arg inference_model "$LIVE_E2E_INFERENCE_MODEL" \
    --arg inference_secret_name "$inference_secret_name" \
    --arg librarian_secret_name "$librarian_secret_name" \
    --arg judge_secret_name "$judge_secret_name" \
    '{
      schema_version: 1,
      embedding_model: $embed_model,
      embedding_secret_ref: {name: "openai", scope: "system-wide"},
      librarian_profile: "balanced",
      librarian_secret_ref: {name: $librarian_secret_name, scope: "system-wide"},
      inference_secret_ref: {name: $inference_secret_name, scope: "system-wide"},
      judge_secret_ref: {name: $judge_secret_name, scope: "system-wide"},
      profiles: {"1": "fast", "2": "fast", "3": "balanced", "4": "strong", "5": "strong"},
      profile_configs: {
        fast: {model: $inference_model, context_window: 32000, max_output_tokens: 1024, input_cost_per_1k: 0.03, output_cost_per_1k: 0.12},
        balanced: {model: $inference_model, context_window: 128000, max_output_tokens: 2048, input_cost_per_1k: 0.30, output_cost_per_1k: 1.50},
        strong: {model: $inference_model, context_window: 128000, max_output_tokens: 4096, input_cost_per_1k: 1.50, output_cost_per_1k: 7.50}
      }
    }' >"$MEMORY_CONFIG_FILE"
}

write_operator_setup_script() {
  cat >"$OPERATOR_SETUP_SCRIPT" <<EOF_SETUP
#!/usr/bin/env bash
set -euo pipefail

GOSH=/opt/bin/gosh
MEMORY_INSTANCE='$LIVE_E2E_MEMORY_INSTANCE_NAME'
CONTEXT_KEY='$LIVE_E2E_CONTEXT_KEY'
WORK_KEY_AGENT_A='$LIVE_E2E_WORK_KEY_AGENT_A'
WORK_KEY_AGENT_B='$LIVE_E2E_WORK_KEY_AGENT_B'
SWARM_ID='$LIVE_E2E_SWARM_ID'
SERVER_TOKEN='$LIVE_E2E_SERVER_TOKEN'
ADMIN_TOKEN='$LIVE_E2E_ADMIN_TOKEN'
GROQ_VALUE='$GROQ_API_KEY'
OPENAI_VALUE='${OPENAI_API_KEY:-}'
CLI_PRINCIPAL_ID='agent:cli-operator'

mkdir -p /artifacts/fixtures /artifacts/bootstrap/agent-a /artifacts/bootstrap/agent-b /tmp/gosh_test_keychain
rm -rf /tmp/gosh_test_keychain

cp /workspace/tests/live.e2e/fixtures/project_march_summary.md /artifacts/fixtures/
cp /workspace/tests/live.e2e/fixtures/project_march_metrics.md /artifacts/fixtures/
cp /workspace/tests/live.e2e/fixtures/project_march_risks.md /artifacts/fixtures/

"\$GOSH" --test-mode memory setup remote \\
  --name "\$MEMORY_INSTANCE" \\
  --url http://memory:8765 \\
  --bootstrap-token "\$ADMIN_TOKEN" \\
  --server-token "\$SERVER_TOKEN"

"\$GOSH" --test-mode memory --instance "\$MEMORY_INSTANCE" auth provision-cli
"\$GOSH" --test-mode memory --instance "\$MEMORY_INSTANCE" auth swarm create "\$SWARM_ID" --owner "\$CLI_PRINCIPAL_ID"
"\$GOSH" --test-mode memory --instance "\$MEMORY_INSTANCE" auth membership grant "\$CLI_PRINCIPAL_ID" --swarm "\$SWARM_ID"
"\$GOSH" --test-mode memory --instance "\$MEMORY_INSTANCE" init --key "\$CONTEXT_KEY" --owner-id "\$CLI_PRINCIPAL_ID"

"\$GOSH" --test-mode memory --instance "\$MEMORY_INSTANCE" secret set groq "\$GROQ_VALUE" \\
  --key "\$CONTEXT_KEY"

[[ -n "\$OPENAI_VALUE" ]] || { echo 'OPENAI_API_KEY is required for the current live harness contract' >&2; exit 1; }
"\$GOSH" --test-mode memory --instance "\$MEMORY_INSTANCE" secret set openai "\$OPENAI_VALUE" \\
  --key "\$CONTEXT_KEY"

"\$GOSH" --test-mode memory --instance "\$MEMORY_INSTANCE" config set --key "\$CONTEXT_KEY" "\$(jq -c . /artifacts/generated/memory-config.json)"

"\$GOSH" --test-mode memory --instance "\$MEMORY_INSTANCE" data --swarm "\$SWARM_ID" ingest document \\
  --key "\$CONTEXT_KEY" \\
  --scope swarm-shared \\
  --source-id atlas-march-summary \\
  --file /workspace/tests/live.e2e/fixtures/project_march_summary.md

"\$GOSH" --test-mode memory --instance "\$MEMORY_INSTANCE" data --swarm "\$SWARM_ID" ingest document \\
  --key "\$CONTEXT_KEY" \\
  --scope swarm-shared \\
  --source-id atlas-march-metrics \\
  --file /workspace/tests/live.e2e/fixtures/project_march_metrics.md

"\$GOSH" --test-mode memory --instance "\$MEMORY_INSTANCE" data --swarm "\$SWARM_ID" ingest document \\
  --key "\$CONTEXT_KEY" \\
  --scope swarm-shared \\
  --source-id atlas-march-risks \\
  --file /workspace/tests/live.e2e/fixtures/project_march_risks.md

"\$GOSH" --test-mode memory --instance "\$MEMORY_INSTANCE" data --swarm "\$SWARM_ID" recall --key "\$CONTEXT_KEY" "Atlas March summary metrics risks next steps" > /artifacts/verify/pretask-recall-atlas.json
jq -e 'if (type == "object") and ((.error? != null) or (.code? != null)) then false else true end' /artifacts/verify/pretask-recall-atlas.json >/dev/null || {
  cat /artifacts/verify/pretask-recall-atlas.json >&2
  exit 1
}

"\$GOSH" --test-mode memory --instance "\$MEMORY_INSTANCE" data --swarm "\$SWARM_ID" recall --key "\$CONTEXT_KEY" '$TASK_A_TEXT' > /artifacts/verify/pretask-recall-agent-a-task.json
jq -e 'if (type == "object") and ((.error? != null) or (.code? != null)) then false else true end' /artifacts/verify/pretask-recall-agent-a-task.json >/dev/null || {
  cat /artifacts/verify/pretask-recall-agent-a-task.json >&2
  exit 1
}

"\$GOSH" --test-mode memory --instance "\$MEMORY_INSTANCE" data --swarm "\$SWARM_ID" recall --key "\$CONTEXT_KEY" '$TASK_B_TEXT' > /artifacts/verify/pretask-recall-agent-b-task.json
jq -e 'if (type == "object") and ((.error? != null) or (.code? != null)) then false else true end' /artifacts/verify/pretask-recall-agent-b-task.json >/dev/null || {
  cat /artifacts/verify/pretask-recall-agent-b-task.json >&2
  exit 1
}

"\$GOSH" --test-mode agent create agent-a \\
  --memory "\$MEMORY_INSTANCE" \\
  --swarm "\$SWARM_ID" \\
  --binary /opt/bin/gosh-agent \\
  --host agent-a \\
  --port 8767

"\$GOSH" --test-mode agent create agent-b \\
  --memory "\$MEMORY_INSTANCE" \\
  --swarm "\$SWARM_ID" \\
  --binary /opt/bin/gosh-agent \\
  --host agent-b \\
  --port 8768

"\$GOSH" --test-mode memory --instance "\$MEMORY_INSTANCE" init --key "\$WORK_KEY_AGENT_A" --owner-id "agent:agent-a"
"\$GOSH" --test-mode memory --instance "\$MEMORY_INSTANCE" init --key "\$WORK_KEY_AGENT_B" --owner-id "agent:agent-b"
"\$GOSH" --test-mode memory --instance "\$MEMORY_INSTANCE" secret set groq "\$GROQ_VALUE" --key "\$WORK_KEY_AGENT_A"
"\$GOSH" --test-mode memory --instance "\$MEMORY_INSTANCE" secret set groq "\$GROQ_VALUE" --key "\$WORK_KEY_AGENT_B"
"\$GOSH" --test-mode memory --instance "\$MEMORY_INSTANCE" secret set openai "\$OPENAI_VALUE" --key "\$WORK_KEY_AGENT_A"
"\$GOSH" --test-mode memory --instance "\$MEMORY_INSTANCE" secret set openai "\$OPENAI_VALUE" --key "\$WORK_KEY_AGENT_B"
"\$GOSH" --test-mode memory --instance "\$MEMORY_INSTANCE" config set --key "\$WORK_KEY_AGENT_A" "\$(jq -c . /artifacts/generated/memory-config.json)"
"\$GOSH" --test-mode memory --instance "\$MEMORY_INSTANCE" config set --key "\$WORK_KEY_AGENT_B" "\$(jq -c . /artifacts/generated/memory-config.json)"

"\$GOSH" --test-mode agent --instance agent-a bootstrap export --file /artifacts/bootstrap/agent-a/agent-a-bootstrap.json
"\$GOSH" --test-mode agent --instance agent-b bootstrap export --file /artifacts/bootstrap/agent-b/agent-b-bootstrap.json
"\$GOSH" --test-mode memory --instance "\$MEMORY_INSTANCE" auth membership grant --swarm "\$SWARM_ID" agent:agent-a
"\$GOSH" --test-mode memory --instance "\$MEMORY_INSTANCE" auth membership grant --swarm "\$SWARM_ID" agent:agent-b
EOF_SETUP
  chmod +x "$OPERATOR_SETUP_SCRIPT"
}

write_operator_task_script() {
  cat >"$OPERATOR_TASK_SCRIPT" <<EOF_TASKS
#!/usr/bin/env bash
set -euo pipefail

GOSH=/opt/bin/gosh
WORK_KEY_AGENT_A='$LIVE_E2E_WORK_KEY_AGENT_A'
WORK_KEY_AGENT_B='$LIVE_E2E_WORK_KEY_AGENT_B'
CONTEXT_KEY='$LIVE_E2E_CONTEXT_KEY'
TASK_A_TEXT=$(printf '%q' "$TASK_A_TEXT")
TASK_B_TEXT=$(printf '%q' "$TASK_B_TEXT")

"\$GOSH" --test-mode agent --instance agent-a task create \\
  --key "\$WORK_KEY_AGENT_A" \\
  --scope agent-private \\
  --target agent:agent-a \\
  "\$TASK_A_TEXT" | tee /artifacts/verify/task-create-agent-a.json >/dev/null

jq -er '.task_id' /artifacts/verify/task-create-agent-a.json > /artifacts/generated/task-agent-a.id

"\$GOSH" --test-mode agent --instance agent-b task create \\
  --key "\$WORK_KEY_AGENT_B" \\
  --scope agent-private \\
  --target agent:agent-b \\
  "\$TASK_B_TEXT" | tee /artifacts/verify/task-create-agent-b.json >/dev/null

jq -er '.task_id' /artifacts/verify/task-create-agent-b.json > /artifacts/generated/task-agent-b.id
EOF_TASKS
  chmod +x "$OPERATOR_TASK_SCRIPT"
}

wait_for_container_health() {
  local service="$1"
  local timeout_secs="$2"
  local deadline=$((SECONDS + timeout_secs))
  local cid status
  while (( SECONDS < deadline )); do
    cid=$(compose ps -q "$service" 2>/dev/null || true)
    if [[ -n "$cid" ]]; then
      status=$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' "$cid" 2>/dev/null || true)
      if [[ "$status" == "healthy" || "$status" == "running" ]]; then
        return 0
      fi
    fi
    sleep 2
  done
  return 1
}

wait_for_agent_http() {
  local host="$1"
  local port="$2"
  local timeout_secs="$3"
  local deadline=$((SECONDS + timeout_secs))
  while (( SECONDS < deadline )); do
    if compose exec -T operator bash -lc "curl -fsS http://$host:$port/health >/dev/null" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  return 1
}

operator_run_script() {
  local label="$1"
  local script_path="$2"
  local container_script="/artifacts/generated/$(basename "$script_path")"
  log "operator phase: $label"
  {
    printf '### %s\n' "$label"
    compose exec -T operator bash "$container_script"
    printf '\n'
  } 2>&1 | tee -a "$OPERATOR_LOG"
}

assert_agent_provider_env_clean() {
  local service="$1"
  local output
  output=$(compose exec -T "$service" env | grep -E '^(GROQ_API_KEY|OPENAI_API_KEY|ANTHROPIC_API_KEY|GOOGLE_API_KEY|INCEPTION_API_KEY|MERCURY_API_KEY)=' || true)
  [[ -z "$output" ]] || die "provider env leaked into $service: $output"
}

read_generated_task_id() {
  local path="$1"
  [[ -s "$path" ]] || die "expected generated task id file: $path"
  tr -d '\n' <"$path"
}

wait_for_task_done() {
  local agent_name="$1"
  local task_id="$2"
  local work_key="$3"
  local output_json="$4"
  local deadline=$((SECONDS + LIVE_E2E_TASK_TIMEOUT_SECS))
  local tmp_json="${output_json}.tmp"
  while (( SECONDS < deadline )); do
    compose exec -T operator bash -lc "/opt/bin/gosh --test-mode agent --instance '$agent_name' task status '$task_id' --key '$work_key'" >"$tmp_json" 2>>"$OPERATOR_LOG" || true
    if [[ -s "$tmp_json" ]]; then
      if jq -e '.status == "done"' "$tmp_json" >/dev/null 2>&1; then
        mv "$tmp_json" "$output_json"
        return 0
      fi
      if jq -e '(.status == "failed") or (.status == "failure")' "$tmp_json" >/dev/null 2>&1; then
        local error_text
        error_text=$(jq -r '.error // empty' "$tmp_json" 2>/dev/null || true)
        mv "$tmp_json" "$output_json"
        if [[ -n "$error_text" ]]; then
          die "task failed: $agent_name / $task_id: $error_text"
        fi
        die "task failed: $agent_name / $task_id"
      fi
    fi
    sleep 5
  done
  [[ -f "$tmp_json" ]] && mv "$tmp_json" "$output_json"
  die "task did not reach done before timeout: $agent_name / $task_id"
}

assert_agent_a_result() {
  python3 - "$1" <<'EOF_PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text(encoding='utf-8'))
if data.get('status') != 'done':
    raise SystemExit(f'agent-a status is not done: {data.get("status")}')
text = (data.get('result') or '').strip()
if len(text) < 80:
    raise SystemExit('agent-a result too short or empty for a completed task')
EOF_PY
}

assert_agent_b_result() {
  python3 - "$1" <<'EOF_PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text(encoding='utf-8'))
if data.get('status') != 'done':
    raise SystemExit(f'agent-b status is not done: {data.get("status")}')
text = (data.get('result') or '').strip()
if len(text) < 80:
    raise SystemExit('agent-b result too short or empty for a completed task')
EOF_PY
}

assert_agent_a_memory_coverage() {
  python3 - "$1" <<'EOF_PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text(encoding='utf-8'))
ctx = (data.get('context') or '').lower()
episodes = data.get('actual_injected_episode_ids') or []
if not any('atlas-march-metrics' in episode for episode in episodes):
    raise SystemExit('agent-a recall missing metrics episode in injected context')
if not any('atlas-march-risks' in episode for episode in episodes):
    raise SystemExit('agent-a recall missing risks episode in injected context')
if not any('atlas-march-summary' in episode for episode in episodes):
    raise SystemExit('agent-a recall missing summary episode in injected context')
metrics = ['99.95%', '73%', '1.8s', '12', '4']
hits = sum(1 for metric in metrics if metric.lower() in ctx)
if hits < 3:
    raise SystemExit(f'agent-a recall contains only {hits} KPI values; expected at least 3')
theme_group_1 = ['vendor api migration', 'partner testing']
theme_group_2 = ['analytics backlog', 'flaky pipeline', 'upstream data pipeline']
if not any(term in ctx for term in theme_group_1):
    raise SystemExit('agent-a recall missing vendor migration / partner testing theme')
if not any(term in ctx for term in theme_group_2):
    raise SystemExit('agent-a recall missing analytics backlog / flaky pipeline theme')
EOF_PY
}

assert_agent_b_memory_coverage() {
  python3 - "$1" <<'EOF_PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text(encoding='utf-8'))
ctx = (data.get('context') or '').lower()
episodes = data.get('actual_injected_episode_ids') or []
if not any('atlas-march-metrics' in episode for episode in episodes):
    raise SystemExit('agent-b recall missing metrics episode in injected context')
if not any('atlas-march-risks' in episode for episode in episodes):
    raise SystemExit('agent-b recall missing risks episode in injected context')
metrics = ['99.95%', '73%', '1.8s', '12', '4']
hits = sum(1 for metric in metrics if metric.lower() in ctx)
if hits < 3:
    raise SystemExit(f'agent-b recall contains only {hits} KPI values; expected at least 3')
if not any(term in ctx for term in ['vendor api migration', 'analytics backlog', 'flaky pipeline', 'upstream data pipeline']):
    raise SystemExit('agent-b recall missing supported operational risk theme')
EOF_PY
}

memory_admin_tool_call() {
  local output_file="$1"
  local tool_name="$2"
  local arguments_json="$3"
  local auth_field="${4:-operator_agent_token}"
  local response
  local init_payload
  local session_id
  local init_boundary="__MCP_HEADERS_BOUNDARY__"
  local server_token
  local agent_token

  [[ -f "$POSTMORTEM_AUTH_CONTEXT_JSON" ]] || write_postmortem_auth_context || return 1

  server_token=$(jq -r '.server_token // empty' "$POSTMORTEM_AUTH_CONTEXT_JSON")
  agent_token=$(jq -r --arg field "$auth_field" '.[$field] // empty' "$POSTMORTEM_AUTH_CONTEXT_JSON")
  [[ -n "$server_token" ]] || return 1
  [[ -n "$agent_token" ]] || return 1

  init_payload=$(compose exec -T operator bash -lc '
set -euo pipefail
headers=$(mktemp)
trap "rm -f \"$headers\"" EXIT
curl -fsS -D "$headers" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "x-server-token: $1" \
  -H "Authorization: Bearer $2" \
  http://memory:8765/mcp \
  -d "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{\"protocolVersion\":\"2025-03-26\",\"capabilities\":{},\"clientInfo\":{\"name\":\"live-e2e-review\",\"version\":\"0.1\"}}}"
printf "\n%s\n" "$3"
cat "$headers"
' bash "$server_token" "$agent_token" "$init_boundary") || return 1

  response=${init_payload%%$'\n'"$init_boundary"$'\n'*}
  session_id=$(printf '%s\n' "$init_payload" | awk -F': ' 'BEGIN {IGNORECASE=1} $1=="mcp-session-id" {gsub("\r","",$2); print $2}')
  [[ -n "$session_id" ]] || return 1

  compose exec -T operator curl -fsS \
    -H 'Content-Type: application/json' \
    -H 'Accept: application/json, text/event-stream' \
    -H "Mcp-Session-Id: $session_id" \
    -H "x-server-token: $server_token" \
    -H "Authorization: Bearer $agent_token" \
    http://memory:8765/mcp \
    -d '{"jsonrpc":"2.0","method":"notifications/initialized"}' >/dev/null

  response=$(compose exec -T operator bash -lc '
set -euo pipefail
tool_name="$1"
arguments_json="$2"
session_id="$3"
server_token="$4"
agent_token="$5"
request=$(jq -nc --arg name "$tool_name" --argjson arguments "$arguments_json" \
  "{jsonrpc:\"2.0\",id:2,method:\"tools/call\",params:{name:\$name,arguments:\$arguments}}")
curl -fsS \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "Mcp-Session-Id: ${session_id}" \
  -H "x-server-token: ${server_token}" \
  -H "Authorization: Bearer ${agent_token}" \
  http://memory:8765/mcp \
  -d "$request"
' bash "$tool_name" "$arguments_json" "$session_id" "$server_token" "$agent_token")

  python3 - "$output_file" "$response" <<'EOF_PY'
import json
import sys

output_path = sys.argv[1]
raw = sys.argv[2]
payload = None
for line in raw.splitlines():
    if line.startswith("data:"):
        data = line[5:].lstrip()
        if data:
            payload = json.loads(data)
            break

if payload is None:
    payload = json.loads(raw)

result = payload.get("result", {})
content = result.get("content") or []
parsed = result
if content:
    text = content[0].get("text", "")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = text

with open(output_path, "w", encoding="utf-8") as fh:
    json.dump(parsed, fh, ensure_ascii=False, indent=2)
    fh.write("\n")
EOF_PY
}

write_postmortem_auth_context() {
  local memory_auth
  local agent_a_auth
  local agent_b_auth

  service_is_running operator || return 0

  memory_auth=$(compose exec -T operator bash -lc "cat '/tmp/gosh_test_keychain/memory_${LIVE_E2E_MEMORY_INSTANCE_NAME}.json' 2>/dev/null || true")
  agent_a_auth=$(compose exec -T operator bash -lc "cat '/tmp/gosh_test_keychain/agent_agent-a.json' 2>/dev/null || true")
  agent_b_auth=$(compose exec -T operator bash -lc "cat '/tmp/gosh_test_keychain/agent_agent-b.json' 2>/dev/null || true")

  [[ -n "$memory_auth" ]] || memory_auth='{}'
  [[ -n "$agent_a_auth" ]] || agent_a_auth='{}'
  [[ -n "$agent_b_auth" ]] || agent_b_auth='{}'

  jq -n \
    --arg compose_project "$COMPOSE_PROJECT" \
    --arg memory_instance_name "$LIVE_E2E_MEMORY_INSTANCE_NAME" \
    --arg namespace_base "$LIVE_E2E_NAMESPACE_BASE" \
    --arg context_key "$LIVE_E2E_CONTEXT_KEY" \
    --arg work_key_agent_a "$LIVE_E2E_WORK_KEY_AGENT_A" \
    --arg work_key_agent_b "$LIVE_E2E_WORK_KEY_AGENT_B" \
    --arg swarm_id "$LIVE_E2E_SWARM_ID" \
    --argjson memory_auth "$memory_auth" \
    --argjson agent_a_auth "$agent_a_auth" \
    --argjson agent_b_auth "$agent_b_auth" \
    '{
      compose_project: $compose_project,
      memory_instance_name: $memory_instance_name,
      namespace_base: $namespace_base,
      context_key: $context_key,
      work_key_agent_a: $work_key_agent_a,
      work_key_agent_b: $work_key_agent_b,
      memory_encryption_key: env.LIVE_E2E_MEMORY_ENCRYPTION_KEY,
      swarm_id: $swarm_id,
      server_token: ($memory_auth.server_token // ""),
      operator_agent_token: ($memory_auth.agent_token // ""),
      agent_a_token: ($agent_a_auth.principal_token // ""),
      agent_b_token: ($agent_b_auth.principal_token // ""),
      agent_a_id: "agent-a",
      agent_b_id: "agent-b",
      operator_agent_id: "agent:cli-operator",
      agent_a_principal_id: "agent:agent-a",
      agent_b_principal_id: "agent:agent-b"
    }' >"$POSTMORTEM_AUTH_CONTEXT_JSON"
}

export_review_attempt_artifacts() {
  local task_a_id="$1"
  local task_b_id="$2"
  local mode="${3:-strict}"
  local query_a_review
  local query_b_review
  local query_a_attempt
  local query_b_attempt
  local response

  write_postmortem_auth_context || {
    [[ "$mode" == "best-effort" ]] && return 1
    die "failed to write plaintext auth context"
  }

  query_a_review=$(jq -nc --arg key "$LIVE_E2E_WORK_KEY_AGENT_A" --arg swarm "$LIVE_E2E_SWARM_ID" '{key:$key, agent_id:"agent-a", swarm_id:$swarm, filter:{kind:"task_review"}, sort_by:"created_at", sort_order:"asc", limit:100}')
  query_b_review=$(jq -nc --arg key "$LIVE_E2E_WORK_KEY_AGENT_B" --arg swarm "$LIVE_E2E_SWARM_ID" '{key:$key, agent_id:"agent-b", swarm_id:$swarm, filter:{kind:"task_review"}, sort_by:"created_at", sort_order:"asc", limit:100}')
  query_a_attempt=$(jq -nc --arg key "$LIVE_E2E_WORK_KEY_AGENT_A" --arg swarm "$LIVE_E2E_SWARM_ID" '{key:$key, agent_id:"agent-a", swarm_id:$swarm, filter:{kind:"task_attempt"}, sort_by:"created_at", sort_order:"asc", limit:100}')
  query_b_attempt=$(jq -nc --arg key "$LIVE_E2E_WORK_KEY_AGENT_B" --arg swarm "$LIVE_E2E_SWARM_ID" '{key:$key, agent_id:"agent-b", swarm_id:$swarm, filter:{kind:"task_attempt"}, sort_by:"created_at", sort_order:"asc", limit:100}')

  memory_admin_tool_call "$TASK_REVIEW_A_JSON" "memory_query" "$query_a_review" "agent_a_token" || {
    [[ "$mode" == "best-effort" ]] && return 1
    die "failed to query task_review facts for agent-a"
  }
  memory_admin_tool_call "$TASK_REVIEW_B_JSON" "memory_query" "$query_b_review" "agent_b_token" || {
    [[ "$mode" == "best-effort" ]] && return 1
    die "failed to query task_review facts for agent-b"
  }
  memory_admin_tool_call "$TASK_ATTEMPTS_A_JSON" "memory_query" "$query_a_attempt" "agent_a_token" || {
    [[ "$mode" == "best-effort" ]] && return 1
    die "failed to query task_attempt facts for agent-a"
  }
  memory_admin_tool_call "$TASK_ATTEMPTS_B_JSON" "memory_query" "$query_b_attempt" "agent_b_token" || {
    [[ "$mode" == "best-effort" ]] && return 1
    die "failed to query task_attempt facts for agent-b"
  }

  python3 - "$TASK_REVIEW_A_JSON" "$TASK_REVIEW_B_JSON" "$TASK_ATTEMPTS_A_JSON" "$TASK_ATTEMPTS_B_JSON" "$TASK_REVIEW_FACTS_JSON" "$TASK_ATTEMPT_FACTS_JSON" "$task_a_id" "$task_b_id" <<'EOF_PY'
import json
import sys
from pathlib import Path

review_a_path, review_b_path, attempt_a_path, attempt_b_path, review_out_path, attempt_out_path, task_a_id, task_b_id = sys.argv[1:9]


def load_payload(path_str: str):
    return json.loads(Path(path_str).read_text(encoding="utf-8"))


def filtered(payload: dict, task_id: str):
    facts = payload.get("facts", []) if isinstance(payload, dict) else []
    if not task_id:
        return {"total": len(facts), "facts": facts, "has_more": False}
    matched = [
        fact for fact in facts
        if str((fact or {}).get("metadata", {}).get("task_id", "")) == task_id
    ]
    return {"total": len(matched), "facts": matched, "has_more": False}


review_a = filtered(load_payload(review_a_path), task_a_id)
review_b = filtered(load_payload(review_b_path), task_b_id)
attempt_a = filtered(load_payload(attempt_a_path), task_a_id)
attempt_b = filtered(load_payload(attempt_b_path), task_b_id)

Path(review_a_path).write_text(json.dumps(review_a, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
Path(review_b_path).write_text(json.dumps(review_b, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
Path(attempt_a_path).write_text(json.dumps(attempt_a, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
Path(attempt_b_path).write_text(json.dumps(attempt_b, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

all_reviews = review_a["facts"] + review_b["facts"]
all_attempts = attempt_a["facts"] + attempt_b["facts"]
Path(review_out_path).write_text(json.dumps({"total": len(all_reviews), "facts": all_reviews, "has_more": False}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
Path(attempt_out_path).write_text(json.dumps({"total": len(all_attempts), "facts": all_attempts, "has_more": False}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
EOF_PY
}

export_private_task_artifacts() {
  local task_a_id="$1"
  local task_b_id="$2"
  local mode="${3:-strict}"
  local query_a_task
  local query_b_task
  local query_a_result
  local query_b_result
  local query_a_session
  local query_b_session

  write_postmortem_auth_context || {
    [[ "$mode" == "best-effort" ]] && return 1
    die "failed to write plaintext auth context"
  }

  query_a_task=$(jq -nc --arg key "$LIVE_E2E_WORK_KEY_AGENT_A" '{key:$key, agent_id:"agent-a", filter:{kind:"task"}, sort_by:"created_at", sort_order:"asc", limit:100}')
  query_b_task=$(jq -nc --arg key "$LIVE_E2E_WORK_KEY_AGENT_B" '{key:$key, agent_id:"agent-b", filter:{kind:"task"}, sort_by:"created_at", sort_order:"asc", limit:100}')
  query_a_result=$(jq -nc --arg key "$LIVE_E2E_WORK_KEY_AGENT_A" '{key:$key, agent_id:"agent-a", filter:{kind:"task_result"}, sort_by:"created_at", sort_order:"asc", limit:100}')
  query_b_result=$(jq -nc --arg key "$LIVE_E2E_WORK_KEY_AGENT_B" '{key:$key, agent_id:"agent-b", filter:{kind:"task_result"}, sort_by:"created_at", sort_order:"asc", limit:100}')
  query_a_session=$(jq -nc --arg key "$LIVE_E2E_WORK_KEY_AGENT_A" '{key:$key, agent_id:"agent-a", filter:{kind:"task_session"}, sort_by:"created_at", sort_order:"asc", limit:100}')
  query_b_session=$(jq -nc --arg key "$LIVE_E2E_WORK_KEY_AGENT_B" '{key:$key, agent_id:"agent-b", filter:{kind:"task_session"}, sort_by:"created_at", sort_order:"asc", limit:100}')

  memory_admin_tool_call "$TASK_FACTS_A_JSON" "memory_query" "$query_a_task" "agent_a_token" || {
    [[ "$mode" == "best-effort" ]] && return 1
    die "failed to query task facts for agent-a"
  }
  memory_admin_tool_call "$TASK_FACTS_B_JSON" "memory_query" "$query_b_task" "agent_b_token" || {
    [[ "$mode" == "best-effort" ]] && return 1
    die "failed to query task facts for agent-b"
  }
  memory_admin_tool_call "$TASK_RESULTS_A_JSON" "memory_query" "$query_a_result" "agent_a_token" || {
    [[ "$mode" == "best-effort" ]] && return 1
    die "failed to query task_result facts for agent-a"
  }
  memory_admin_tool_call "$TASK_RESULTS_B_JSON" "memory_query" "$query_b_result" "agent_b_token" || {
    [[ "$mode" == "best-effort" ]] && return 1
    die "failed to query task_result facts for agent-b"
  }
  memory_admin_tool_call "$TASK_SESSIONS_A_JSON" "memory_query" "$query_a_session" "agent_a_token" || {
    [[ "$mode" == "best-effort" ]] && return 1
    die "failed to query task_session facts for agent-a"
  }
  memory_admin_tool_call "$TASK_SESSIONS_B_JSON" "memory_query" "$query_b_session" "agent_b_token" || {
    [[ "$mode" == "best-effort" ]] && return 1
    die "failed to query task_session facts for agent-b"
  }

  python3 - \
    "$TASK_A_CREATE_JSON" "$TASK_B_CREATE_JSON" "$task_a_id" "$task_b_id" \
    "$TASK_FACTS_A_JSON" "$TASK_FACTS_B_JSON" \
    "$TASK_RESULTS_A_JSON" "$TASK_RESULTS_B_JSON" \
    "$TASK_SESSIONS_A_JSON" "$TASK_SESSIONS_B_JSON" \
    "$TASK_FACTS_JSON" "$TASK_RESULT_FACTS_JSON" "$TASK_SESSION_FACTS_JSON" <<'EOF_PY'
import json
import sys
from pathlib import Path

(
    task_a_create_path,
    task_b_create_path,
    task_a_id,
    task_b_id,
    task_a_path,
    task_b_path,
    result_a_path,
    result_b_path,
    session_a_path,
    session_b_path,
    task_out_path,
    result_out_path,
    session_out_path,
) = sys.argv[1:14]


def load_json(path_str: str):
    path = Path(path_str)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_facts(path_str: str):
    payload = load_json(path_str)
    if isinstance(payload, dict) and "facts" in payload:
        return payload
    if isinstance(payload, list):
        return {"total": len(payload), "facts": payload, "has_more": False}
    return {"total": 0, "facts": [], "has_more": False}


def load_task_ref(create_path: str, fallback_task_id: str):
    payload = load_json(create_path)
    if not isinstance(payload, dict):
        return {"task_id": fallback_task_id, "task_fact_id": ""}
    return {
        "task_id": str(payload.get("task_id") or fallback_task_id or ""),
        "task_fact_id": str(payload.get("task_fact_id") or ""),
    }


def matches_fact(fact: dict, task_ref: dict) -> bool:
    if not isinstance(fact, dict):
        return False
    metadata = fact.get("metadata", {}) or {}
    task_fact_id = task_ref["task_fact_id"]
    task_id = task_ref["task_id"]
    if task_fact_id:
        if str(fact.get("id", "")) == task_fact_id:
            return True
        if str(metadata.get("task_fact_id", "")) == task_fact_id:
            return True
    return bool(task_id and str(metadata.get("task_id", "")) == task_id)


def filtered_payload(path_str: str, task_ref: dict) -> dict:
    payload = load_facts(path_str)
    facts = payload.get("facts", [])
    matched = [fact for fact in facts if matches_fact(fact, task_ref)]
    return {"total": len(matched), "facts": matched, "has_more": False}


def write_payload(path_str: str, payload: dict):
    Path(path_str).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


task_a_ref = load_task_ref(task_a_create_path, task_a_id)
task_b_ref = load_task_ref(task_b_create_path, task_b_id)

task_a = filtered_payload(task_a_path, task_a_ref)
task_b = filtered_payload(task_b_path, task_b_ref)
result_a = filtered_payload(result_a_path, task_a_ref)
result_b = filtered_payload(result_b_path, task_b_ref)
session_a = filtered_payload(session_a_path, task_a_ref)
session_b = filtered_payload(session_b_path, task_b_ref)

write_payload(task_a_path, task_a)
write_payload(task_b_path, task_b)
write_payload(result_a_path, result_a)
write_payload(result_b_path, result_b)
write_payload(session_a_path, session_a)
write_payload(session_b_path, session_b)

all_tasks = task_a["facts"] + task_b["facts"]
all_results = result_a["facts"] + result_b["facts"]
all_sessions = session_a["facts"] + session_b["facts"]

write_payload(task_out_path, {"total": len(all_tasks), "facts": all_tasks, "has_more": False})
write_payload(result_out_path, {"total": len(all_results), "facts": all_results, "has_more": False})
write_payload(session_out_path, {"total": len(all_sessions), "facts": all_sessions, "has_more": False})
EOF_PY
}

best_effort_export_private_task_artifacts() {
  local task_a_id=""
  local task_b_id=""

  service_is_running operator || return 0
  service_is_running memory || return 0
  [[ -f "$TASK_A_ID_FILE" ]] && task_a_id=$(tr -d '\n' <"$TASK_A_ID_FILE" 2>/dev/null || true)
  [[ -f "$TASK_B_ID_FILE" ]] && task_b_id=$(tr -d '\n' <"$TASK_B_ID_FILE" 2>/dev/null || true)
  export_private_task_artifacts "$task_a_id" "$task_b_id" best-effort >>"$OPERATOR_LOG" 2>&1 || true
}

best_effort_export_review_attempt_artifacts() {
  local task_a_id=""
  local task_b_id=""

  service_is_running operator || return 0
  service_is_running memory || return 0
  [[ -f "$TASK_A_ID_FILE" ]] && task_a_id=$(tr -d '\n' <"$TASK_A_ID_FILE" 2>/dev/null || true)
  [[ -f "$TASK_B_ID_FILE" ]] && task_b_id=$(tr -d '\n' <"$TASK_B_ID_FILE" 2>/dev/null || true)
  export_review_attempt_artifacts "$task_a_id" "$task_b_id" best-effort >>"$OPERATOR_LOG" 2>&1 || true
}

capture_postmortem_state() {
  local memory_cid
  local memory_container_name

  write_postmortem_auth_context || true
  printf '%s\n' "$COMPOSE_PROJECT" >"$POSTMORTEM_PROJECT_FILE"
  docker volume ls --format '{{.Name}}' | grep "^${COMPOSE_PROJECT}_" >"$POSTMORTEM_VOLUMES_FILE" 2>/dev/null || true

  memory_cid=$(service_container_id memory)
  [[ -n "$memory_cid" ]] || return 0

  memory_container_name=$(docker inspect --format '{{.Name}}' "$memory_cid" 2>/dev/null | sed 's#^/##')
  printf '%s\n' "$memory_container_name" >"$POSTMORTEM_MEMORY_CONTAINER_FILE"
  docker inspect "$memory_cid" --format '{{json .Mounts}}' >"$POSTMORTEM_MEMORY_MOUNTS_JSON" 2>/dev/null || true
  compose exec -T memory bash -lc 'find /data -maxdepth 6 -type f | sort' >"$POSTMORTEM_DATA_FILE_LIST" 2>/dev/null || true
  compose exec -T memory bash -lc 'find /data -maxdepth 6 -type f \( -name "*.db" -o -name "*.sqlite" -o -name "*.sqlite3" -o -name "*.sqlcipher" \) | sort' >"$POSTMORTEM_DB_CANDIDATES_FILE" 2>/dev/null || true
  rm -rf "$POSTMORTEM_DB_COPY_DIR"
  mkdir -p "$POSTMORTEM_DB_COPY_DIR"
  docker cp "$memory_container_name:/data/." "$POSTMORTEM_DB_COPY_DIR" >/dev/null 2>&1 || true
}

write_failure_postmortem_summary() {
  local memory_container_name=""
  local db_path_hint="/data/<db-file>"

  [[ -f "$POSTMORTEM_MEMORY_CONTAINER_FILE" ]] && memory_container_name=$(tr -d '\n' <"$POSTMORTEM_MEMORY_CONTAINER_FILE")
  if [[ -f "$POSTMORTEM_DB_CANDIDATES_FILE" ]] && [[ -s "$POSTMORTEM_DB_CANDIDATES_FILE" ]]; then
    db_path_hint=$(head -n 1 "$POSTMORTEM_DB_CANDIDATES_FILE")
  fi

  cat >>"$SUMMARY_FILE" <<EOF_SUMMARY

POSTMORTEM
- compose project: $COMPOSE_PROJECT
- auth context: $POSTMORTEM_AUTH_CONTEXT_JSON
- memory container: ${memory_container_name:-<unknown>}
- compose ps: $COMPOSE_PS_FILE
- volume list: $POSTMORTEM_VOLUMES_FILE
- memory data path inside container: /data
- copied data directory: $POSTMORTEM_DB_COPY_DIR
- data file listing: $POSTMORTEM_DATA_FILE_LIST
- db candidates: $POSTMORTEM_DB_CANDIDATES_FILE
- review artifacts: $TASK_REVIEW_A_JSON, $TASK_REVIEW_B_JSON
- attempt artifacts: $TASK_ATTEMPTS_A_JSON, $TASK_ATTEMPTS_B_JSON
- task artifacts: $TASK_FACTS_A_JSON, $TASK_FACTS_B_JSON
- task_result artifacts: $TASK_RESULTS_A_JSON, $TASK_RESULTS_B_JSON
- task_session artifacts: $TASK_SESSIONS_A_JSON, $TASK_SESSIONS_B_JSON

NEXT STEPS
docker compose -p "$COMPOSE_PROJECT" -f "$COMPOSE_FILE" ps
docker compose -p "$COMPOSE_PROJECT" -f "$COMPOSE_FILE" exec memory bash
docker compose -p "$COMPOSE_PROJECT" -f "$COMPOSE_FILE" exec memory bash -lc 'find /data -maxdepth 6 -type f | sort'
docker cp "${memory_container_name:-$COMPOSE_PROJECT-memory-1}:$db_path_hint" "$ARTIFACTS_DIR/"
docker compose -p "$COMPOSE_PROJECT" -f "$COMPOSE_FILE" exec memory bash -lc 'python3 - <<'"'"'\"'\"'PY'"'"'\"'\"'
from pysqlcipher3 import dbapi2 as sqlite3
import os
db_path = "$db_path_hint"
key = os.environ["GOSH_MEMORY_ENCRYPTION_KEY"].replace("'"'"'", "''")
conn = sqlite3.connect(db_path)
conn.execute(f"PRAGMA key = '{key}'")
for row in conn.execute("select kind, json_extract(metadata, ''$.task_id''), json_extract(metadata, ''$.attempt''), substr(fact, 1, 200) from facts where kind in (''task_review'', ''task_attempt'') order by created_at desc limit 20"):
    print(row)
PY'
EOF_SUMMARY

  cat >"$POSTMORTEM_COMMANDS_FILE" <<EOF_COMMANDS
docker compose -p "$COMPOSE_PROJECT" -f "$COMPOSE_FILE" ps
docker compose -p "$COMPOSE_PROJECT" -f "$COMPOSE_FILE" exec memory bash
docker compose -p "$COMPOSE_PROJECT" -f "$COMPOSE_FILE" exec memory bash -lc 'find /data -maxdepth 6 -type f | sort'
docker cp "${memory_container_name:-$COMPOSE_PROJECT-memory-1}:$db_path_hint" "$ARTIFACTS_DIR/"
docker compose -p "$COMPOSE_PROJECT" -f "$COMPOSE_FILE" exec memory bash -lc 'python3 - <<'"'"'\"'\"'PY'"'"'\"'\"'
from pysqlcipher3 import dbapi2 as sqlite3
import os
db_path = "$db_path_hint"
key = os.environ["GOSH_MEMORY_ENCRYPTION_KEY"].replace("'"'"'", "''")
conn = sqlite3.connect(db_path)
conn.execute(f"PRAGMA key = '{key}'")
for row in conn.execute("select kind, json_extract(metadata, ''$.task_id''), json_extract(metadata, ''$.attempt''), substr(fact, 1, 200) from facts where kind in (''task_review'', ''task_attempt'') order by created_at desc limit 20"):
    print(row)
PY'
EOF_COMMANDS
}

verify_memory_kinds() {
  local task_a_id="$1"
  local task_b_id="$2"

  export_private_task_artifacts "$task_a_id" "$task_b_id"

  python3 - \
    "$TASK_A_CREATE_JSON" "$TASK_B_CREATE_JSON" \
    "$TASK_FACTS_A_JSON" "$TASK_FACTS_B_JSON" \
    "$TASK_RESULTS_A_JSON" "$TASK_RESULTS_B_JSON" \
    "$TASK_SESSIONS_A_JSON" "$TASK_SESSIONS_B_JSON" <<'EOF_PY'
import json
import sys
from pathlib import Path

(
    task_a_create_path,
    task_b_create_path,
    task_a_path,
    task_b_path,
    result_a_path,
    result_b_path,
    session_a_path,
    session_b_path,
) = sys.argv[1:9]

def load_payload(path_str: str):
    payload = json.loads(Path(path_str).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "facts" in payload:
        return payload
    if isinstance(payload, list):
        return {"total": len(payload), "facts": payload, "has_more": False}
    return {"total": 0, "facts": [], "has_more": False}

def load_task_ref(path_str: str):
    payload = json.loads(Path(path_str).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {"task_id": "", "task_fact_id": ""}
    return {
        "task_id": str(payload.get("task_id") or ""),
        "task_fact_id": str(payload.get("task_fact_id") or ""),
    }

def ensure_kind(payload: dict, label: str, task_ref: dict):
    facts = payload.get("facts", [])
    if not facts:
        raise SystemExit(f"missing {label} for {task_ref['task_fact_id'] or task_ref['task_id']}")

task_a_ref = load_task_ref(task_a_create_path)
task_b_ref = load_task_ref(task_b_create_path)

ensure_kind(load_payload(task_a_path), "agent-a task fact", task_a_ref)
ensure_kind(load_payload(task_b_path), "agent-b task fact", task_b_ref)
ensure_kind(load_payload(result_a_path), "agent-a task_result fact", task_a_ref)
ensure_kind(load_payload(result_b_path), "agent-b task_result fact", task_b_ref)
ensure_kind(load_payload(session_a_path), "agent-a task_session fact", task_a_ref)
ensure_kind(load_payload(session_b_path), "agent-b task_session fact", task_b_ref)
EOF_PY
}

capture_recall_evidence() {
  if jq -e -s 'any(.[]; [.. | strings | select(test("memory_recall"))] | length > 0)' \
    "$TASK_A_STATUS_JSON" "$TASK_B_STATUS_JSON" >/dev/null 2>&1; then
    jq -n --slurpfile a "$TASK_A_STATUS_JSON" --slurpfile b "$TASK_B_STATUS_JSON" '{agent_a: $a[0].tool_trace, agent_b: $b[0].tool_trace}' >"$RECALL_EVIDENCE_FILE"
    return 0
  fi
  if grep -n 'memory_recall' "$AGENT_A_LOG" "$AGENT_B_LOG" "$MEMORY_LOG" >"$RECALL_EVIDENCE_FILE" 2>/dev/null; then
    return 0
  fi
  die 'unable to capture evidence that memory_recall participated in the flow'
}

capture_secret_resolve_evidence() {
  compose logs memory >"$MEMORY_LOG" 2>&1 || true
  if grep -n '/api/v1/agent/secrets/resolve' "$MEMORY_LOG" >"$SECRET_RESOLVE_EVIDENCE_FILE" 2>/dev/null; then
    return 0
  fi
  die 'unable to capture evidence for /api/v1/agent/secrets/resolve in memory logs'
}

main() {
  local task_a_id
  local task_b_id

  require_cmd docker
  require_cmd jq
  require_cmd python3
  resolve_host_secret GROQ_API_KEY
  [[ "$LIVE_E2E_EMBED_MODE" == "openai" ]] || die "current live harness requires LIVE_E2E_EMBED_MODE=openai"
  resolve_host_secret OPENAI_API_KEY
  [[ -d "$CLI_REPO" ]] || die "gosh.cli sibling repo not found at $CLI_REPO"
  [[ -d "$AGENT_REPO" ]] || die "gosh.agent sibling repo not found at $AGENT_REPO"
  [[ -f "$FIXTURES_DIR/project_march_summary.md" ]] || die "fixture missing: project_march_summary.md"
  [[ -f "$FIXTURES_DIR/project_march_metrics.md" ]] || die "fixture missing: project_march_metrics.md"
  [[ -f "$FIXTURES_DIR/project_march_risks.md" ]] || die "fixture missing: project_march_risks.md"
  [[ -z "${GOSH_MEMORY_ALLOW_PLAINTEXT_SECRETS:-}" ]] || die 'GOSH_MEMORY_ALLOW_PLAINTEXT_SECRETS must stay unset for this harness'

  CLI_BINARY_HOST=$(resolve_binary "$CLI_REPO" gosh)
  AGENT_BINARY_HOST=$(resolve_binary "$AGENT_REPO" gosh-agent)
  assert_cli_binary_contract "$CLI_BINARY_HOST"
  assert_agent_binary_contract "$AGENT_BINARY_HOST"

  export LIVE_E2E_BUILD_CONTEXT="$PROJECT_ROOT"
  export MEMORY_WORKTREE="$PROJECT_ROOT"
  export CLI_BINARY_HOST
  export AGENT_BINARY_HOST
  export ARTIFACTS_DIR
  export MEMORY_HOME_DIR
  export OPERATOR_HOME_DIR
  export AGENT_A_HOME_DIR
  export AGENT_B_HOME_DIR
  export AGENT_A_BOOTSTRAP_DIR
  export AGENT_B_BOOTSTRAP_DIR
  export LIVE_E2E_MEMORY_INSTANCE_NAME
  export LIVE_E2E_NAMESPACE_BASE
  export LIVE_E2E_CONTEXT_KEY
  export LIVE_E2E_WORK_KEY_AGENT_A
  export LIVE_E2E_WORK_KEY_AGENT_B
  export LIVE_E2E_SWARM_ID
  export LIVE_E2E_SERVER_TOKEN
  export LIVE_E2E_ADMIN_TOKEN
  export LIVE_E2E_MEMORY_ENCRYPTION_KEY
  export LIVE_E2E_MEMORY_TAG
  export LIVE_E2E_RUNTIME_TAG

  log "namespace contract: base=$LIVE_E2E_NAMESPACE_BASE context=$LIVE_E2E_CONTEXT_KEY work_a=$LIVE_E2E_WORK_KEY_AGENT_A work_b=$LIVE_E2E_WORK_KEY_AGENT_B"

  write_versions_file
  write_memory_config_json
  write_operator_setup_script
  write_operator_task_script

  assert_compose_watch_env
  log "building local memory/runtime images"
  compose build memory operator agent-a agent-b
  assert_local_image_present "$LIVE_E2E_MEMORY_TAG"
  assert_local_image_present "$LIVE_E2E_RUNTIME_TAG"

  log "bringing up memory/operator/agent-a/agent-b containers"
  compose up -d memory operator agent-a agent-b

  wait_for_container_health memory 120 || die 'memory container did not become healthy'
  wait_for_container_health operator 30 || die 'operator container did not stay running'

  operator_run_script 'setup remote memory, provision CLI, ingest Atlas dataset, configure secrets, create agents, export bootstrap bundles' "$OPERATOR_SETUP_SCRIPT"
  write_postmortem_auth_context || die 'failed to write plaintext auth context'

  assert_agent_provider_env_clean agent-a
  assert_agent_provider_env_clean agent-b
  [[ -s "$AGENT_A_BOOTSTRAP_HOST_FILE" ]] || die "missing agent-a bootstrap artifact: $AGENT_A_BOOTSTRAP_HOST_FILE"
  [[ -s "$AGENT_B_BOOTSTRAP_HOST_FILE" ]] || die "missing agent-b bootstrap artifact: $AGENT_B_BOOTSTRAP_HOST_FILE"

  wait_for_agent_http agent-a 8767 "$LIVE_E2E_AGENT_START_TIMEOUT_SECS" || die 'agent-a did not become healthy'
  wait_for_agent_http agent-b 8768 "$LIVE_E2E_AGENT_START_TIMEOUT_SECS" || die 'agent-b did not become healthy'

  operator_run_script 'create natural Atlas tasks for agent-a and agent-b' "$OPERATOR_TASK_SCRIPT"

  task_a_id=$(read_generated_task_id "$TASK_A_ID_FILE")
  task_b_id=$(read_generated_task_id "$TASK_B_ID_FILE")

  wait_for_task_done agent-a "$task_a_id" "$LIVE_E2E_WORK_KEY_AGENT_A" "$TASK_A_STATUS_JSON"
  wait_for_task_done agent-b "$task_b_id" "$LIVE_E2E_WORK_KEY_AGENT_B" "$TASK_B_STATUS_JSON"
  assert_agent_a_memory_coverage "$PRETASK_RECALL_TASK_A_JSON"
  assert_agent_b_memory_coverage "$PRETASK_RECALL_TASK_B_JSON"
  export_review_attempt_artifacts "$task_a_id" "$task_b_id"

  compose exec -T operator bash -lc "/opt/bin/gosh --test-mode agent --instance agent-a task list --key '$LIVE_E2E_WORK_KEY_AGENT_A'" >"$TASK_LIST_A_JSON"
  compose exec -T operator bash -lc "/opt/bin/gosh --test-mode agent --instance agent-b task list --key '$LIVE_E2E_WORK_KEY_AGENT_B'" >"$TASK_LIST_B_JSON"

  assert_agent_a_result "$TASK_A_STATUS_JSON"
  assert_agent_b_result "$TASK_B_STATUS_JSON"
  verify_memory_kinds "$task_a_id" "$task_b_id"
  capture_recall_evidence
  capture_secret_resolve_evidence

  log "Atlas distributed retrieval scenario passed"
}

main "$@"
