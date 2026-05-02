# Changelog

## Unreleased

## v0.3.7 — 2026-05-02

### Added

- Require each private `dev -> main` release promotion to update
  `CHANGELOG.md` with a non-empty entry for the promoted version, and ensure
  the matching release tag already contains that changelog entry.

### Changed

- Relax `local_cli` profile config so memory only needs to persist the routing
  intent (`backend = "local_cli"`) plus memory-local context budgeting fields
  (`context_window`, `max_output_tokens`, `max_output_tokens_summarize`, and
  `thinking_overhead`). Host-local binary paths, command arguments,
  provider/model selection, sampling knobs, API secrets, and pricing are not
  valid `local_cli` memory config fields. Memory does not provide a legacy
  `cli_bin` execution mode for `local_cli`; it only returns an agent-executed
  planning payload, and the agent resolves the concrete local CLI on its own
  host.

## v0.3.5 — 2026-04-27

### Fixed

- Stop the private `main` gate from force-pushing a squash sync commit back to
  protected `main`. The gate still validates the filtered publish tree, while
  `main` itself is updated only through the normal `dev -> main` promotion.

## v0.3.4 — 2026-04-27

### Fixed

- Restore the private `main` gate after MAL-heavy e2e is intentionally skipped:
  filtered-main staging and private-main sync now run after skipped MAL-heavy
  jobs when all required checks passed.
- Apply the repository gitleaks allowlist to filtered release-stage scans so
  checked-in benchmark fixture IDs do not block release staging.
- Keep CodeQL uploads non-blocking for private repositories where GitHub code
  scanning is disabled.

## v0.3.3 — 2026-04-27

### Fixed

- Keep selected raw recall evidence in the finalized recall context so
  `memory_recall` and `memory_ask` do not lose raw evidence during context
  packet rendering. (PR #183)
- Add first-class recall continuation metadata and public
  `memory_recall(continuation_handle=..., page="next")` pagination for anchored
  retrieval, while keeping internal `get_more_context` behavior for
  `memory_ask`. (PR #183)
- Normalize schedule-style recall routes and relative temporal expressions such
  as `tonight` against ingestion context for dated release/deadline evidence.
  (PR #183)
- Redact startup join-token logging and keep token output to fingerprint/path
  diagnostics. (PR #183)
- Suppress SQLCipher native memlock log noise via supported SQLCipher logging
  PRAGMAs before keying encrypted connections, without disabling SQLCipher
  memory security. Add a one-shot diagnostic that points operators to
  `RLIMIT_MEMLOCK` / `CAP_IPC_LOCK` remediation. (PR #185)
- Restore green typecheck coverage for recall continuation paging and
  SQLCipher diagnostics. The unsafe `cipher_memory_security = OFF` attempt from
  PR #184 was reverted and replaced by the safe diagnostics path.

### Documentation

- Document the release invariant: create and push the version bump commit and
  matching `v<version>` tag on `dev` before promoting `dev -> main`.

## v0.3.2 — 2026-04-26

### Fixed

- Align inference prompt recency/conflict policy across recall/ask prompts while
  preserving the `get_more_context` tool affordance. (PR #180)
- Prevent raw-only / zero-fact conversation writes from poisoning exact dedup,
  so a later successful retry can still ingest extracted facts. (PR #182)
- Make raw recall respect inactive lifecycle status and avoid arbitrary raw
  fallback for kind-specific recall. (PR #182)
- Preserve supported document cross facts only when backed by the current
  active source/version, including reload behavior. (PR #182)
- Make `retract()` hide fact and raw evidence, clear relevant content indices,
  and allow explicit re-ingest after retraction. (PR #182)

## v0.3.1 — 2026-04-26

### Fixed

- Finalize recall evidence before `memory_plan_inference` builds the
  payload, and share the same finalized evidence view between
  `memory_recall`, `memory_ask`, and the planning surface. Prevents
  drift where the three surfaces could disagree on which facts had
  been admitted to the response. (PR #175)
- Build the recall answer contract from the original query rather
  than from a downstream-rewritten form, preserving query semantics
  through the recall pipeline.
- Expose the recall answer contract on the recall response so callers
  can see how the answer was constrained.

## v0.3.0 — 2026-04-25

### Changed

- `memory_recall` is now an evidence-only retrieval surface. It no longer
  returns executable inference planning fields such as `recommended_profile`,
  `payload`, `payload_meta`, or secret routing references.
- `memory_recall` accepts English queries only. Non-English queries fail closed
  with `NON_ENGLISH_QUERY`; callers must translate the query in their own
  agent/model and retry recall with the English query.
- Executable inference planning moved to the ACL-gated `memory_plan_inference`
  tool, which returns the planning payload, payload metadata, recommended
  profile, and opaque `secret_ref`.
