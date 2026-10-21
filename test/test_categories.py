from utils.categories import category_search_terms


def test_bakery_category_includes_yelp_plural_label():
    assert category_search_terms("bakery") == ("bakery", "bakeries")


def test_unknown_category_uses_its_canonical_label():
    assert category_search_terms("supermarket") == ("supermarket",)