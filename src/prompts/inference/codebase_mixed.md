You are answering a question that requires BOTH prose memory evidence and codebase evidence.

{context}

Question: {question}

Rules:
1. Use conversation/document evidence for prose facts.
2. Use grounded codebase facts plus attached SOURCE FILES for code facts.
3. SOURCE FILES are the canonical code attachment surface. Use highlighted spans to identify the most relevant symbol or implementation.
4. Keep prose evidence and code evidence separate when the answer asks for multiple fields.
5. If one lane is missing evidence, say exactly which lane is missing.
6. Do not let prose evidence overwrite exact code evidence, and do not let code evidence overwrite prose evidence.

CONFLICT RESOLUTION:
If prose evidence and code evidence disagree, prefer the lane that directly grounds the requested field. Use SOURCE FILES for exact code wording and chat/document evidence for prose facts. Do not invent a reconciliation.

Answer based only on the context above. Be concise and direct.
Answer:
