You are answering a code change risk or review question using grounded memory evidence.

{context}

Question: {question}

Rules:
1. Focus on grounded change risk, review surface, regression risk, or blast radius that is explicit in the retrieved evidence.
2. Prefer exact files, symbols, dependencies, checks, reviews, and attached SOURCE FILES over generic advice.
3. Use highlighted spans inside SOURCE FILES to identify the most relevant code region.
4. Separate confirmed grounded risk from missing evidence.
5. If the context does not support a risk claim, say that the evidence is insufficient instead of guessing.

CONFLICT RESOLUTION:
If multiple retrieved signals disagree about risk, prefer the signal tied to the most explicit code object or attached SOURCE FILES. Do not escalate speculative risk as confirmed.

Answer based only on the context above. Be concise and direct.
Answer:
