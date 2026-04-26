<div align="center">

# gosh.startrek

**GOSH.AI Memory — Community Edition**

[![License: GOSH Noncommercial 1.0](https://img.shields.io/badge/license-GOSH%20Noncommercial%201.0-orange.svg)](LICENSE.md)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-green.svg)](https://python.org)
[![MCP](https://img.shields.io/badge/protocol-MCP-purple.svg)](https://modelcontextprotocol.io)

</div>

---

> [!IMPORTANT]
> **This repository is a temporary home.**
>
> `gosh.startrek` is a snapshot of [`gosh.memory`](https://github.com/gosh-dot-ai/gosh.memory)
> published under the [GOSH.AI Noncommercial License v1.0](LICENSE.md)
> while we refactor the upstream codebase to support a clean dual-
> license setup.
>
> Once the dual-license refactor lands, this repository's contents
> will be **merged back into `main` of
> [`gosh-dot-ai/gosh.memory`](https://github.com/gosh-dot-ai/gosh.memory)**,
> which will then ship under both licenses (commercial via separate
> written agreement; noncommercial under [LICENSE.md](LICENSE.md)).
>
> Until that merge happens, treat any code in this repo as licensed
> **only** under the noncommercial terms.

---

## What this is

`gosh.memory` is the semantic long-term memory engine for the GOSH AI
stack. It runs as an MCP server, ingests conversations / documents /
code / agent traces, extracts atomic facts via a format-aware
Librarian, and serves retrieval, inference, and agent-side execution
plans across a multi-agent swarm.

The companion repositories that form the rest of the stack:

- 📚 **Documentation** — full architecture, setup, ACL, telemetry,
  swarm protocol, and benchmarks:
  [`gosh-dot-ai/gosh.docs`](https://github.com/gosh-dot-ai/gosh.docs)
- 🛠 **Operator CLI** — installer, lifecycle, secrets, agent
  bootstrap: [`gosh-dot-ai/gosh.cli`](https://github.com/gosh-dot-ai/gosh.cli)
- 🧠 **Upstream memory source** (post-merge canonical home):
  [`gosh-dot-ai/gosh.memory`](https://github.com/gosh-dot-ai/gosh.memory)

## Installing

You do **not** install `gosh.memory` directly from this repository.
Use the operator CLI — it pulls the right memory image, generates
encryption keys, bootstraps the admin principal, and wires hooks for
your coding CLIs:

```sh
# install gosh CLI
curl -fsSL https://raw.githubusercontent.com/gosh-dot-ai/gosh.cli/main/install.sh | bash

# pull memory + agent components
gosh setup

# bring memory up locally
gosh memory setup local
gosh memory start
```

For the full ten-minute walkthrough — namespace bootstrap, secrets,
swarm creation, agent provisioning, coding-CLI capture — see:

- [`gosh.docs/SETUP.md`](https://github.com/gosh-dot-ai/gosh.docs/blob/main/SETUP.md)
- [`gosh.cli/docs/quickstart.md`](https://github.com/gosh-dot-ai/gosh.cli/blob/main/docs/quickstart.md)

## License

This repository is licensed under the **GOSH.AI Noncommercial License v1.0**.

You may use, modify, and redistribute the software for any
**Permitted Purpose** — personal use, research, evaluation, education,
and internal use within an organization that is not undertaken for
commercial advantage or monetary compensation.

Any **Commercial Purpose** — including selling, licensing,
sublicensing, hosted-as-a-service, or embedding into a commercial
product — requires a separate written license. Contact
`legal@gosh.sh`.

The full license text is in [LICENSE.md](LICENSE.md).

---

Copyright © 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky.
