You are answering a question using ONLY the retrieved codebase memory evidence below.

{context}

Question: {question}

The context may contain:
- grounded codebase facts
- attached SOURCE FILES
- highlighted target spans inside those files
- windowed file views for large files

Rules:
1. Prefer explicit grounded code objects, file paths, qualified names, signatures, and highlighted spans.
2. Treat SOURCE FILES as the canonical code attachment surface. Use the highlighted spans to locate the most relevant definition or implementation.
3. If a file is attached in windowed form, do not claim visibility outside the shown windows.
4. Do not invent unseen code, omitted lines, or missing symbols.
5. If the retrieved evidence does not ground the answer, say which code object, file, or span is missing.

CONFLICT RESOLUTION:
If grounded code facts and SOURCE FILES disagree, prefer the attached file text for exact code wording and the grounded facts for object identity only when the file view is incomplete. Do not invent a reconciliation.

Answer:
