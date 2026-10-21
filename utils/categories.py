"""Canonical category aliases used when filtering Yelp POIs."""

CATEGORY_SEARCH_TERMS = {
    "bakery": ("bakery", "bakeries"),
}


def category_search_terms(category: str) -> tuple[str, ...]:
    """Return dataset labels that represent a requested canonical category."""
    normalized = category.strip().lower()
    return CATEGORY_SEARCH_TERMS.get(normalized, (normalized,))