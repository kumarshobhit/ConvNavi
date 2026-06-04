# Context Memory V1

This branch adds a first structured-memory implementation for context-aware ConvNavi testing.

## Goal

Support a minimal but testable version of the architecture discussed in `Context-Testing.pptx-1-18.pdf`:

1. Store durable context from user utterances.
2. Decide when a later request needs context memory.
3. Enrich the structured POI request before retrieval.
4. Expose debug fields in the API response so tests can inspect what happened.

## Memory Shape

Each `Session` now has `context_memory`:

```json
{
  "personal_places": {
    "home": null,
    "work": null,
    "usual_restaurant": null,
    "favorite_restaurant": null,
    "favorite_bar": null
  },
  "preferences": {
    "diet": null,
    "food": null,
    "restaurant": null,
    "bar": null,
    "price_level": null,
    "liked_cuisines": [],
    "disliked_cuisines": []
  },
  "events": []
}
```

The memory is currently in-memory per `user_id` and survives across turns while the server process is running.

## Runtime Flow

For each query:

1. `update_context_memory` asks the LLM whether the current utterance updates durable memory.
2. Normal ConvNavi intent/action classification runs.
3. For POI refinement/navigation requests, `decide_memory_retrieval` asks the LLM whether memory is needed.
4. Relevant memory keys are selected.
5. `enrich_constraints_with_memory` updates the parsed request before filtering and FAISS retrieval.

Example:

1. User says: `My usual restaurant is Barbuzzo on 110 S 13th St.`
2. Memory stores `personal_places.usual_restaurant`.
3. Later the same `user_id` says: `Take me to my usual restaurant.`
4. Parsed generic name `usual restaurant` is enriched to `Barbuzzo`.

## Debug Fields

The `/query` response now includes:

- `context_memory`
- `context_used`
- `memory_before`
- `memory_update`
- `memory_after`
- `parsed_constraints`
- `memory_retrieval_decision`
- `memory_used`
- `enriched_constraints`

These fields are intended for benchmarking and thesis testing, not final production UX.

## Suggested Manual Test

Restart the FastAPI server after changing code:

```bash
python app.py
```

Then use the same `user_id` for both turns:

```bash
curl -s -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "My usual restaurant is Barbuzzo on 110 S 13th St.",
    "user_location": [39.925, -75.17],
    "llm_type": "gpt-4o-mini",
    "user_id": 9001
  }' | python -m json.tool
```

```bash
curl -s -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Take me to my usual restaurant.",
    "user_location": [39.925, -75.17],
    "llm_type": "gpt-4o-mini",
    "user_id": 9001
  }' | python -m json.tool
```

Check that the second response has:

- `memory_used.personal_places.usual_restaurant.name = "Barbuzzo"`
- `enriched_constraints.name = "Barbuzzo"`
- retrieved POIs containing Barbuzzo, if it exists in the current city/cache.

