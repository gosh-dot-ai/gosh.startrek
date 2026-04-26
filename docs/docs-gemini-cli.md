<!--
  Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
  SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0

  Licensed under the GOSH.AI Noncommercial License, Version 1.0.
  See LICENSE.md in the root of this repository for the full license
  text. Use of this file for any Commercial Purpose requires a separate
  written license. Contact: legal@gosh.sh
-->

# Connecting gosh.memory to Gemini CLI

Gemini CLI supports HTTP MCP servers via `settings.json`.

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

Edit `~/.gemini/settings.json`:

```json
{
  "mcpServers": {
    "gosh-memory": {
      "url": "http://localhost:8765/mcp",
      "headers": {"x-server-token": "<token from ~/.gosh-memory/token>"}
    }
  }
}
```

For project scope, create `.gemini/settings.json` in your project root.

---

## Verify

Inside a Gemini CLI session:

```
/mcp
# Lists all connected servers and available tools

/mcp reload
# Forces re-query if tools not showing
```

---

## Usage pattern

```
# Recall relevant context
memory_recall(key="project", query="authentication decisions")

# Store session outcome
memory_store(key="project", content="<session>",
             agent_id="gemini", swarm_id="proj",
             session_num=1, session_date="2026-03-16")
```

Recall queries must be English. If the user prompt or task is not English,
translate the recall query in Gemini before calling `memory_recall`; if the
server returns `NON_ENGLISH_QUERY`, retry with the translated English query.

For the complete agent flow (loading data, building index, query types, scope
isolation), see [README — How agents use memory](../README.md#how-agents-use-memory).

---

## Import history before connecting

```bash
# Import your codebase
gosh-memory import \
  --format git \
  --source-uri https://github.com/you/repo \
  --content-type technical \
  --key project --scope swarm-shared

# Import documents
gosh-memory import \
  --format directory \
  --dir ./docs/ \
  --key project --scope swarm-shared
```
