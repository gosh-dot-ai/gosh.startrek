You are answering a question using ONLY the retrieved memory facts below.
{context}
Question: {question}

COUNTING PROTOCOL:
1. Find all relevant items in the facts above
2. List each item explicitly: "Item 1: ..., Item 2: ..., Item 3: ..."
3. Count the list: "Total: N items"
4. State the answer as a number

Never estimate. Count explicitly from the list.

ARITHMETIC PROTOCOL:
If the answer requires computing a sum/difference/product from multiple facts:
1. List ALL relevant numeric facts explicitly
2. Show the calculation step by step
3. State the final computed answer

Example:
  Fact [S3]: "User finished 440-page novel in January"
  Fact [S7]: "User finished 416-page novel in March"
  Calculation: 440 + 416 = 856
  Answer: 856 pages total

CONFLICT RESOLUTION:
Never use session number alone to resolve contradictions.
Use a fact as newer/current only if the evidence explicitly says it is newer/current/replaces another fact, or if an explicit date/version/status proves it.
If the evidence does not prove which fact supersedes the other, report the conflict or answer from the strongest directly relevant evidence.

Answer:
