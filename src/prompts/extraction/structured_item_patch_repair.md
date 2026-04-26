You are repairing only the invalid parts of a structured extraction payload.

Return strict JSON only. No markdown. No explanation.

Return only this shape:
{{
  "operations": []
}}

Each operation must be one of:

- {{"path": "...", "action": "set", "value": ...}}
- {{"path": "...", "action": "replace", "value": ...}}
- {{"path": "...", "action": "remove", "reason": "..."}}
- {{"path": "...", "action": "append", "value": ...}}

Allowed target paths:
{allowed_paths_json}

Validation issues:
{issues_json}

Current payload:
{current_payload_json}

Schema snippets for broken targets only:
{target_schema_json}

Grounding context:
{context_payload}

Rules:

1. Only touch paths listed in Allowed target paths.
2. Do not resend the whole payload.
3. Do not modify untouched items.
4. If an item cannot be repaired without inventing unsupported content, remove it.
5. If a field value is unsupported, set it to null or remove the item if the field is required.
6. Preserve existing ids when possible.
7. If you append a new object, it must be fully valid.
8. Return only the operations object.
