You are answering an exact code object lookup question using codebase evidence.

{context}

Question: {question}

Rules:
1. Return the exact grounded code object requested: qualified name, file path, signature, symbol, or defining location.
2. Prefer explicit codebase facts first, then use attached SOURCE FILES to confirm exact wording and location.
3. Treat highlighted spans inside SOURCE FILES as the primary locator for the requested symbol.
4. Preserve exact spelling, casing, module paths, and file paths.
5. If the question asks which file defines or contains the symbol, answer with the exact attached file path.
6. If the user asks for only the file path, return only that path and nothing else.
7. Never return an empty answer. If the requested code object is not grounded in the retrieved evidence, say exactly what is missing.

CONFLICT RESOLUTION:
If multiple code facts conflict, prefer the most explicit grounded code object reference together with the attached SOURCE FILES. Do not invent a symbol or file path to break a tie.

Answer based only on the context above. Be concise and direct.
Answer:
