#!/usr/bin/env bash
# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

set -euo pipefail

log() {
  printf '[agent-entry] %s\n' "$*"
}

deny_provider_env() {
  local name
  for name in GROQ_API_KEY OPENAI_API_KEY ANTHROPIC_API_KEY GOOGLE_API_KEY INCEPTION_API_KEY MERCURY_API_KEY; do
    if [[ -n "${!name:-}" ]]; then
      log "refusing to start because provider env leaked into agent container: $name"
      exit 1
    fi
  done
}

wait_for_bootstrap_file() {
  local file="$1"
  local timeout_secs="${AGENT_BOOTSTRAP_TIMEOUT_SECS:-180}"
  local deadline=$((SECONDS + timeout_secs))
  while [[ ! -s "$file" ]]; do
    if (( SECONDS >= deadline )); then
      log "bootstrap file did not appear in time: $file"
      exit 1
    fi
    sleep 1
  done
}

append_optional_flag() {
  local -n out_ref=$1
  local flag_name="$2"
  local value="$3"
  if [[ -n "$value" ]]; then
    out_ref+=("$flag_name" "$value")
  fi
}

main() {
  : "${AGENT_BOOTSTRAP_FILE:?AGENT_BOOTSTRAP_FILE is required}"
  : "${AGENT_PORT:?AGENT_PORT is required}"
  : "${WATCH_KEY:?WATCH_KEY is required}"
  : "${WATCH_SWARM_ID:?WATCH_SWARM_ID is required}"

  local agent_bin="${GOSH_AGENT_BIN:-/opt/bin/gosh-agent}"
  local host="${AGENT_HOST:-0.0.0.0}"
  local poll_interval="${WATCH_POLL_INTERVAL:-5}"
  local watch_budget="${WATCH_BUDGET:-10.0}"
  local watch_context_key="${WATCH_CONTEXT_KEY:-$WATCH_KEY}"

  deny_provider_env
  wait_for_bootstrap_file "$AGENT_BOOTSTRAP_FILE"

  if [[ ! -x "$agent_bin" ]]; then
    log "agent binary is not executable: $agent_bin"
    exit 1
  fi

  local -a cmd=(
    "$agent_bin"
    serve
    --bootstrap-file "$AGENT_BOOTSTRAP_FILE"
    --host "$host"
    --port "$AGENT_PORT"
    --watch
    --watch-key "$WATCH_KEY"
    --watch-context-key "$watch_context_key"
    --watch-swarm-id "$WATCH_SWARM_ID"
    --poll-interval "$poll_interval"
    --watch-budget "$watch_budget"
  )

  append_optional_flag cmd --watch-agent-id "${WATCH_AGENT_ID:-}"

  log "starting agent via reviewed bootstrap flow on ${host}:${AGENT_PORT}"
  exec "${cmd[@]}"
}

main "$@"
