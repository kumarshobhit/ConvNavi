# Option 1: RAG-Based Memory Findings

## Works Well

- Direct coreference with keyword overlap works when the new query shares terms with the stored episode, such as "cheap like last time" retrieving a cheap Italian episode.
- Simple preference retrieval within the same domain can work when the stored summary contains explicit cuisine, price, rating, or selected-place information.
- Irrelevant memory pollution is not always a problem: a stored gym episode did not bleed into a later restaurant query.

## Fails or Is Unreliable

- Vague implicit references fail when the query has weak keyword overlap with stored memory, such as "that place I liked" or "another one like that".
- Preference updates fail because old and new memory entries are both stored as independent summaries, with no durable overwrite or contradiction handling.
- Adding a prompt instruction to prefer the most recent conflicting memory did not fix the state-shift failure. The retriever still passed a mixed set of memories, and the LLM filled generic restaurant constraints from other retrieved episodes.
- Conditional preferences are weak: the system stored "quiet places with family", but later converted this into "family-friendly" rather than preserving the quiet constraint.
- Some non-restaurant navigation requests do not receive food-memory pollution, but may still fail because the POI dataset/retrieval cannot find the target.

## Root Cause of Most Failures

The memory backend is embedding-based retrieval over natural-language summaries. It has no explicit model of recency, temporal priority, contradiction, user-specific state, or conditional logic. All stored episodes are treated as searchable text records, and the LLM is asked to infer constraints after retrieval.

This means failures can happen in two places:

- Retrieval can miss the right memory when the query is vague or semantically far from the stored summary.
- Injection can retrieve useful memory but still fail to convert it into the right structured constraints.


## Current Thesis Implication

The current implementation is a useful baseline for episodic RAG memory. It is strong enough to show that text-summary memory can help with simple references, but the failures suggest that robust context awareness likely needs structured memory or explicit state-management rules in addition to RAG retrieval.

Retrieval failure (wrong memory chunk retrieved)
Injection failure (right chunk retrieved but LLM ignored it)
Temporal failure (old and new memory both retrieved, no priority)
Semantic gap (query has no keyword overlap with stored memory)