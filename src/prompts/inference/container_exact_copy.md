You are deciding whether a grounded exact-copy retrieval candidate satisfies the user's request.

Retrieved memory context:
{context}

Question: {question}

The context may include a TERMINAL RENDER CANDIDATE block with structured planner proof. The current
production planner is the document structural exact-copy slice; the API/trace contract is generic, but
you must judge only the candidate proof you are shown. The raw answer text is hidden from you. Do not write, reconstruct, summarize, or paraphrase the copied text yourself.

Decide from the proof fields, not from technical ids. The candidate is usable only when the planner
proof shows the requested selector, ordinal, kind/topic anchors, source/order scope, output constraints,
and exact render proof are satisfied. Use the semantic proof fields such as
selected_index_in_matching_domain, surface/normalized anchor tokens, and proof_source_fields; do not
infer the ordinal from artifact ids or other debug identifiers. Anchor tokens must have no missing
entries, render proof must not be degraded, and whole-or-fail must be true.

If the structured proof satisfies the request, return exactly one JSON object:
{{"decision":"use_candidate","candidate_id":"<candidate id from context>"}}

If the candidate is missing, degraded, unresolved, contradictory, has anchor_tokens_missing, or does
not satisfy the request, answer exactly:
Not enough grounded context.

CONFLICT RESOLUTION:
If multiple candidate facts or traces contradict each other on the same topic, use only the structured terminal render candidate proof shown in the current context. Do not resolve conflicts by guessing or copying raw text. If the proof is incomplete or contradictory, answer exactly:
Not enough grounded context.

Do not include explanations, citations, markdown fences, sibling context, or copied raw text.
