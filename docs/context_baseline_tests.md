# ConvNavi Baseline Context Tests

Date: 2026-05-10

Purpose: document initial tests showing how the current ConvNavi system handles contextual information before adding explicit context memory.

Environment:

- Server mode: `python app.py`
- Endpoint: `POST http://127.0.0.1:8000/query`
- Model: `llama3.2` through Ollama
- `USE_NLU=FALSE`
- `INTENT_NO_NLU=POI`
- Dataset/cache: Philadelphia POIs from `data/filtered_pois.csv` and `data/embeddings.npy`

## Summary

ConvNavi currently supports limited session context through:

- raw conversation history in `Session.get_history()`
- accumulated POI constraints in `session.poi_constraints`
- previous retrieved POIs for confirmation/info follow-ups
- car state for car-control tasks

It does not currently maintain an explicit structured memory of user preferences, requested time, trip purpose, route context, or long-term user facts.

## Test 1: User Preference Memory

Category: user preference / multi-turn memory

Hypothesis: if the user says they are vegetarian, a later restaurant request should use that dietary preference.

### Turn 1 Request

```json
{
  "query": "I am vegetarian.",
  "user_location": [39.955431, -75.154903],
  "llm_type": "llama3.2",
  "user_id": 201
}
```

### Turn 1 Result

```json
{
  "response": "\"I found Mi Lah Vegetarian, a vegan restaurant with 3.5/5 rating at 218 S 16th St. It's a great option for vegetarians.\"\n\nWould you like to navigate to Mi Lah Vegetarian? Or would you like me to suggest more options or ask if there's anything else I can help you with?",
  "retrieved_pois": [
    {
      "name": "Basic 4 Vegetarian Café",
      "category": "Vegan, Restaurants, Vegetarian",
      "rating": 3.5,
      "price_level": "$",
      "address": "Reading Terminal Mrkt"
    },
    {
      "name": "Mi Lah Vegetarian",
      "category": "Vegan, Restaurants, Vegetarian",
      "rating": 3.5,
      "price_level": "$$",
      "address": "218 S 16th St"
    },
    {
      "name": "Kingdom of Vegetarians Restaurant",
      "category": "Chinese, Vegetarian, Restaurants",
      "rating": 3.5,
      "price_level": "$",
      "address": "129 N 11th St"
    }
  ],
  "session_id": 1,
  "user_id": 201
}
```

### Turn 2 Request

```json
{
  "query": "Find me a cheap restaurant nearby.",
  "user_location": [39.955431, -75.154903],
  "llm_type": "llama3.2",
  "user_id": 201
}
```

### Turn 2 Result

```json
{
  "response": "Sorry, I cannot find any relevant places. Do you have other preferences in mind?",
  "retrieved_pois": [],
  "session_id": 1,
  "user_id": 201
}
```

### Interpretation

The first utterance was treated as an immediate POI search, not as a persistent user preference update. The later request did not reliably use a structured memory like `dietary_preference=vegetarian`.

Observed limitation: ConvNavi has raw history and accumulated constraints, but no explicit user-preference memory.

## Test 2: Explicit Future-Time Context

Category: temporal context

Hypothesis: if the user requests a place open at a specific future time, the system should evaluate opening hours against that requested time.

### Request

```json
{
  "query": "Find me a fancy bar open tonight at 11 PM.",
  "user_location": [39.955431, -75.154903],
  "llm_type": "llama3.2",
  "user_id": 202
}
```

### Result

```json
{
  "response": "Sorry, I cannot find any relevant places. Do you have other preferences in mind?",
  "retrieved_pois": [],
  "session_id": 2,
  "user_id": 202
}
```

### Interpretation

The current code can filter `open_now`, but it uses the machine's current time:

```python
now = datetime.now().strftime("%H:%M")
day_name = datetime.now().strftime('%A')
```

It does not parse and store requested temporal context such as `tonight at 11 PM`, nor does it evaluate opening hours against an intended arrival/request time.

Observed limitation: no explicit temporal context model.

## Test 3: Request-Level Location Context

Category: spatial context / API context propagation

Hypothesis: changing `user_location` should affect radius-based retrieval.

Note: the loaded POI cache is Philadelphia-specific. Therefore, a cross-city test is not a valid test of city switching. However, it can still reveal whether the API forwards request-level location into the retrieval pipeline.

### Request A

```json
{
  "query": "Find an Italian restaurant within 1 km.",
  "user_location": [39.955431, -75.154903],
  "llm_type": "llama3.2",
  "user_id": 203
}
```

### Request A Result

```json
{
  "response": "\"I found Barbuzzo, a Mediterranean restaurant with an Italian twist, rated 4.5/5 at 110 S 13th St.\"\n\nWould you like to navigate to Barbuzzo? Or would you like me to suggest other options or ask if you have any specific preferences?",
  "retrieved_pois": [
    {
      "name": "Barbuzzo",
      "category": "Mediterranean, Restaurants, Pizza, Italian",
      "rating": 4.5,
      "price_level": "$$",
      "address": "110 S 13th St"
    },
    {
      "name": "Pizzacato",
      "category": "Restaurants, Italian",
      "rating": 3.0,
      "price_level": "$$",
      "address": "248 Market St"
    },
    {
      "name": "Portofino",
      "category": "Restaurants, Italian",
      "rating": 3.0,
      "price_level": "$$",
      "address": "1227 Walnut St"
    }
  ],
  "session_id": 3,
  "user_id": 203
}
```

### Request B

```json
{
  "query": "Find an Italian restaurant within 1 km.",
  "user_location": [40.758896, -73.985130],
  "llm_type": "llama3.2",
  "user_id": 204
}
```

### Request B Result

```json
{
  "response": "\"I found Barbuzzo, a Mediterranean restaurant with an Italian twist, rated 4.5/5 at 110 S 13th St.\"\n\nWould you like to navigate to Barbuzzo? Or would you like me to suggest other options or ask if you have any specific preferences?",
  "retrieved_pois": [
    {
      "name": "Barbuzzo",
      "category": "Mediterranean, Restaurants, Pizza, Italian",
      "rating": 4.5,
      "price_level": "$$",
      "address": "110 S 13th St"
    },
    {
      "name": "Pizzacato",
      "category": "Restaurants, Italian",
      "rating": 3.0,
      "price_level": "$$",
      "address": "248 Market St"
    },
    {
      "name": "Portofino",
      "category": "Restaurants, Italian",
      "rating": 3.0,
      "price_level": "$$",
      "address": "1227 Walnut St"
    }
  ],
  "session_id": 4,
  "user_id": 204
}
```

### Interpretation

Both requests returned the same POIs. One reason is that the dataset cache is Philadelphia-specific. More importantly, the API route currently ignores the request-level location:

```python
output = run_rag_navigation(
    query=request.query,
    user_location=user_location,
    ...
)
```

It should likely pass:

```python
user_location=request.user_location
```

Observed limitation: request-level spatial context is not propagated by `/query`.

## Test Categorization

The test categories were chosen from common contextual dimensions for in-car navigation and from the types of contextual-memory failures discussed in related benchmark work:

- user preference / memory
- temporal context
- spatial context
- multi-turn carryover
- context update or forgetting
- trip purpose
- vehicle state
- environment context

The initial three tests focus on preference, time, and location because they are directly relevant to the current ConvNavi POI pipeline and can be tested with the existing endpoint.

## Follow-Up Baseline Tests To Add

1. Negative preference:
   - Turn 1: `I do not like Italian food.`
   - Turn 2: `Find me a good restaurant above 4 stars.`
   - Expected: avoid Italian results.

2. Context update:
   - Turn 1: `I am vegetarian.`
   - Turn 2: `Actually, forget that. Find a steakhouse.`
   - Expected: old dietary preference should be removed or overridden.

3. Trip purpose:
   - Turn 1: `I am on a date tonight.`
   - Turn 2: `Find a restaurant nearby.`
   - Expected: date-appropriate restaurant preference should influence recommendation.

4. Same-city spatial test:
   - Use two different Philadelphia coordinates.
   - Query: `Find an Italian restaurant within 1 km.`
   - Expected: different candidates if location is correctly propagated.

