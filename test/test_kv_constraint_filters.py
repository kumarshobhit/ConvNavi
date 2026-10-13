import os

import pandas as pd

os.environ.setdefault("LLM_MODEL", "gpt-4o")

from main import apply_kv_entries_to_constraints, apply_structured_filters


def test_kv_cuisine_preferences_match_any_listed_cuisine():
    constraints = apply_kv_entries_to_constraints(
        {"category": "cafe"},
        [{"key": "User_cuisine_preference", "value": "Italian, French"}],
    )
    pois = pd.DataFrame(
        [
            {"name": "Italian Cafe", "category": "Cafes, Italian", "rating": 4.5},
            {"name": "French Cafe", "category": "Cafes, French", "rating": 4.5},
            {"name": "Greek Cafe", "category": "Cafes, Greek", "rating": 4.5},
        ]
    )

    filtered = apply_structured_filters(pois, constraints, user_location=None)

    assert constraints["cuisine"] == ["Italian", "French"]
    assert filtered["name"].tolist() == ["Italian Cafe", "French Cafe"]