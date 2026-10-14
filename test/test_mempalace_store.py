import numpy as np

from memory.mempalace_store import MemPalaceNavigationMemory


def embed_texts(texts):
    return np.asarray(
        [[1.0, 0.0] if "Italian" in text else [0.0, 1.0] for text in texts],
        dtype=np.float32,
    )


def test_replace_stores_verbatim_rows_and_retrieves_from_local_sqlite(tmp_path):
    store = MemPalaceNavigationMemory(tmp_path, max_entries=10, embed_texts=embed_texts)
    count = store.replace([
        {
            "time": "2026-10-14T12:00:00Z",
            "conversation_id": "driver-1",
            "summary": "Driver: I prefer quiet Italian restaurants.",
        },
        {
            "time": "2026-10-14T12:01:00Z",
            "conversation_id": "driver-2",
            "summary": "Driver: I prefer Mexican restaurants.",
        },
    ])

    results = store.search("Italian restaurant", 3)

    assert count == 2
    assert (store.palace_dir / "sqlite_exact.sqlite3").exists()
    assert results[0].conversation_id == "driver-1"
    assert results[0].summary == "Driver: I prefer quiet Italian restaurants."
    assert results[0].score == 1.0