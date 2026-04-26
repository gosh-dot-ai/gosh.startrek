You are repairing structured extraction output that failed before a valid JSON object could be parsed.

Return strict JSON only. No markdown. No explanation.

You must return exactly one JSON object that matches this root contract:
{root_contract_json}

Do not return a patch. Do not explain. Do not include extra keys.

Previous raw output:
{previous_raw}

Known parse issues:
{issues_json}

Rules:

1. Return valid JSON only.
2. Keep exactly the required top-level keys.
3. Every required top-level field must have the correct type.
4. Do not invent unsupported content.
5. If a value cannot be grounded, leave the relevant collection empty rather than guessing.
6. Do not output any commentary.

Context:
{context_payload}
