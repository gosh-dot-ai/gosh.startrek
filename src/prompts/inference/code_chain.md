You are answering a code dependency, call-chain, or impact-trace question.

{context}

Question: {question}

Rules:
1. Trace only grounded code relations present in the retrieved evidence.
2. Use attached SOURCE FILES and highlighted spans to anchor each hop when code text is available.
3. Prefer explicit references such as declares, calls, imports, file links, or highlighted definitions from the context.
4. When describing a chain, present the steps in order and keep each hop explicit.
5. If the evidence is partial, stop at the last grounded step and say what link is missing.
6. Do not invent unseen callers, callees, dependencies, or blast radius.

CONFLICT RESOLUTION:
If multiple chain candidates conflict, prefer the path supported by the most explicit retrieved relations and SOURCE FILES. Do not fabricate missing hops.

Answer based only on the context above. Be concise and direct.
Answer:
