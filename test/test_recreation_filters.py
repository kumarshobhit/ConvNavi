import pandas as pd

from main import apply_structured_filters


def test_filters_recreation_pois_by_dog_and_wheelchair_accessibility():
    pois = pd.DataFrame([
        {
            "name": "Accessible Park",
            "category": "Parks",
            "dogs_allowed": True,
            "wheelchair_accessible": True,
            "rating": 4.5,
        },
        {
            "name": "No Dogs Garden",
            "category": "Gardens",
            "dogs_allowed": False,
            "wheelchair_accessible": True,
            "rating": 4.5,
        },
        {
            "name": "Inaccessible Park",
            "category": "Parks",
            "dogs_allowed": True,
            "wheelchair_accessible": False,
            "rating": 4.5,
        },
    ])

    filtered = apply_structured_filters(
        pois,
        {
            "category": "park",
            "dogs_allowed": True,
            "wheelchair_accessible": True,
        },
        user_location=(39.95, -75.15),
    )

    assert filtered["name"].tolist() == ["Accessible Park"]