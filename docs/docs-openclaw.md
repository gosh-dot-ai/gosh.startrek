<!--
  Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
  SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0

  Licensed under the GOSH.AI Noncommercial License, Version 1.0.
  See LICENSE.md in the root of this repository for the full license
  text. Use of this file for any Commercial Purpose requires a separate
  written license. Contact: legal@gosh.sh
-->

# Connecting gosh.memory to OpenClaw

OpenClaw has native MCP support. gosh.memory connects as an HTTP MCP server.

## Prerequisites

Server must be running:
```bash
gosh-memory start --data-dir ./data
# Listening on http://127.0.0.1:8765
```

The server is protected by token authentication. On first start, a token is
auto-generated and saved to `~/.gosh-memory/token` (chmod 600). All requests
(except `GET /health`) must include an `x-server-token` header.

---

## Configuration

Edit `~/.openclaw/mcp_config.json`:

```json
{
  "mcpServers": {
    "gosh-memory": {
      "type": "http",
      "url": "http://localhost:8765/mcp",
      "headers": {"x-server-token": "<token from ~/.gosh-memory/token>"}
    }
  }
}
```

Or via CLI:

```bash
openclaw mcp register gosh-memory \
  --transport http \
  --url http://localhost:8765/mcp
```

Verify:
```bash
openclaw mcp status
# gosh-memory: connected (17 tools)
```

---

## Usage pattern

Once connected, OpenClaw agents can call memory tools directly:

```
# Agent recalls context before responding
memory_recall(key="project", query="current task requirements")

# Agent stores outcomes after completing work
memory_store(key="project", content="<what happened>",
             agent_id="openclaw", swarm_id="proj",
             session_num=1, session_date="2026-03-16")
```

Recall queries must be English. If the user prompt or task is not English,
translate the recall query in OpenClaw before calling `memory_recall`; if the
server returns `NON_ENGLISH_QUERY`, retry with the translated English query.

The key benefit: agent replaces conversation history with `memory_recall` output — pays ~2K tokens instead of full history on every LLM call.

For the complete agent flow (loading data, building index, query types, scope
isolation), see [README — How agents use memory](../README.md#how-agents-use-memory).

---

## Import history before connecting

```bash
# Import your project docs
gosh-memory import \
  --format git \
  --source-uri https://github.com/you/repo \
  --content-type technical \
  --key project --scope swarm-shared

# Import past conversations
gosh-memory import \
  --format conversation_json \
  --file ~/Downloads/conversations.json \
  --key project --scope agent-private
```
