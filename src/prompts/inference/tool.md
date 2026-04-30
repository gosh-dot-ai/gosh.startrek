You are answering a question using the retrieved memory below.

{context}

MEMORY METADATA:
- Sessions searched: {sessions_in_context} of {total_sessions} total

Question: {question}

The context has two sections:
- RETRIEVED FACTS: structured extracted facts
- RAW CONTEXT: source text excerpts with full details

Step-by-step:
1. Check RETRIEVED FACTS for the answer
2. If facts lack details (step numbers, coordinates, names, sequences), check RAW CONTEXT
3. If the context says RECALL CONTINUATION AVAILABLE and the answer is not in
   the current page, call get_more_context with the recall_continuation handle
   and page="next" to retrieve the next evidence page. Repeat only while the
   tool returns more continuation evidence.
4. If BOTH facts and raw context still don't contain enough to answer
   confidently and a specific session seems relevant, call get_more_context
   with that session number from the context (e.g. S12) to retrieve the FULL
   text of that session.
5. Answer based on all available information

CONFLICT RESOLUTION:
Never use session number alone to resolve contradictions.
Use a fact as newer/current only if the evidence explicitly says it is newer/current/replaces another fact, or if an explicit date/version/status proves it.
If the evidence does not prove which fact supersedes the other, report the conflict or answer from the strongest directly relevant evidence.

Answer concisely and directly.
If the facts don't contain the answer, say what you can infer.
