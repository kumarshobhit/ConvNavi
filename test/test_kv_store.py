import pandas as pd

from memory import kv_store


def test_memory_add_creates_and_overwrites_per_user(tmp_path):
    memory_path = tmp_path / "kv_memory.csv"

    first = kv_store.memory_add(1, "User_cuisine_preference", "Italian", memory_path=memory_path)
    second = kv_store.memory_add(1, "User_cuisine_preference", "Japanese", memory_path=memory_path)
    other_user = kv_store.memory_add(2, "User_cuisine_preference", "Mexican", memory_path=memory_path)

    rows = pd.read_csv(memory_path, dtype=str).fillna("")

    assert first["value"] == "Italian"
    assert second["value"] == "Japanese"
    assert other_user["value"] == "Mexican"
    assert len(rows) == 2
    assert rows[rows["user_id"] == "1"].iloc[0]["value"] == "Japanese"
    assert rows[rows["user_id"] == "2"].iloc[0]["value"] == "Mexican"


def test_memory_add_preserves_different_conditions(tmp_path):
    memory_path = tmp_path / "kv_memory.csv"

    kv_store.memory_add(1, "User_noise_level_preference", "quiet", condition="with_family", memory_path=memory_path)
    kv_store.memory_add(1, "User_noise_level_preference", "lively", condition="with_friends", memory_path=memory_path)
    kv_store.memory_add(1, "User_noise_level_preference", "silent", condition="with_family", memory_path=memory_path)

    rows = kv_store.memory_search(1, "User_noise_level_preference", memory_path=memory_path)

    assert len(rows) == 2
    assert {row["condition"] for row in rows} == {"with_family", "with_friends"}
    assert {row["value"] for row in rows} == {"silent", "lively"}


def test_memory_add_truncates_value_to_50_chars(tmp_path):
    memory_path = tmp_path / "kv_memory.csv"
    long_value = "x" * 80

    kv_store.memory_add(1, "User_route_avoid", long_value, memory_path=memory_path)

    rows = kv_store.memory_list(1, memory_path=memory_path)
    assert rows[0]["value"] == "x" * 50


def test_memory_remove_and_search(tmp_path):
    memory_path = tmp_path / "kv_memory.csv"
    kv_store.memory_add(1, "User_cuisine_preference", "Italian", memory_path=memory_path)
    kv_store.memory_add(1, "User_price_preference", "cheap", memory_path=memory_path)

    assert kv_store.memory_search(1, "User_cuisine_preference", memory_path=memory_path)[0]["value"] == "Italian"
    assert kv_store.memory_search(1, "price", memory_path=memory_path)[0]["value"] == "cheap"
    assert kv_store.memory_remove(1, "User_price_preference", memory_path=memory_path) == 1
    assert kv_store.memory_search(1, "price", memory_path=memory_path) == []


def test_retrieve_filters_conditional_preferences(monkeypatch, tmp_path):
    memory_path = tmp_path / "kv_memory.csv"
    kv_store.memory_add(1, "User_noise_level_preference", "quiet", condition="with_family", memory_path=memory_path)

    def fake_pass_llm(**kwargs):
        return (
            '[{"tool": "memory_search", "arguments": {"key": "User_noise_level_preference"}}]',
            10,
            5,
        )

    monkeypatch.setattr("llm.llm_selector.pass_llm", fake_pass_llm)

    no_condition = kv_store.retrieve_from_kv_store(1, "Find a restaurant", memory_path=memory_path)
    with_condition = kv_store.retrieve_from_kv_store(1, "Find a restaurant for my kids", memory_path=memory_path)

    assert no_condition["entries"] == []
    assert with_condition["entries"][0]["value"] == "quiet"


def test_save_to_kv_store_executes_mocked_tool_calls(monkeypatch, tmp_path):
    memory_path = tmp_path / "kv_memory.csv"

    def fake_pass_llm(**kwargs):
        return (
            """
            [
              {"tool": "memory_search", "arguments": {"key": "User_cuisine_preference"}},
              {"tool": "memory_add", "arguments": {"key": "User_cuisine_preference", "value": "Italian", "condition": ""}}
            ]
            """,
            11,
            7,
        )

    monkeypatch.setattr("llm.llm_selector.pass_llm", fake_pass_llm)

    result = kv_store.save_to_kv_store(1, '[{"question": "Italian please"}]', session_id="s1", memory_path=memory_path)

    rows = kv_store.memory_list(1, memory_path=memory_path)
    assert [entry["tool"] for entry in result["executed"]] == ["memory_search", "memory_add"]
    assert "raw_response" in result
    assert rows[0]["key"] == "User_cuisine_preference"
    assert rows[0]["value"] == "Italian"
    assert rows[0]["session_id"] == "s1"


def test_retrieve_formats_context(monkeypatch, tmp_path):
    memory_path = tmp_path / "kv_memory.csv"
    kv_store.memory_add(1, "User_cuisine_preference", "Italian", memory_path=memory_path)
    kv_store.memory_add(1, "User_price_preference", "cheap", memory_path=memory_path)

    def fake_pass_llm(**kwargs):
        return (
            """
            [
              {"tool": "memory_search", "arguments": {"key": "User_cuisine_preference"}},
              {"tool": "memory_search", "arguments": {"key": "User_price_preference"}}
            ]
            """,
            12,
            6,
        )

    monkeypatch.setattr("llm.llm_selector.pass_llm", fake_pass_llm)

    result = kv_store.retrieve_from_kv_store(1, "Find dinner", memory_path=memory_path)

    assert "- cuisine: Italian" in result["context"]
    assert "- price: cheap" in result["context"]


def test_coreference_includes_latest_episodic_memory(monkeypatch, tmp_path):
    memory_path = tmp_path / "kv_memory.csv"
    episodic_path = tmp_path / "navigation_memory.csv"
    pd.DataFrame(
        [
            {"time": "2026-01-01T00:00:00Z", "conversation_id": "old", "summary": "Old destination."},
            {"time": "2026-01-02T00:00:00Z", "conversation_id": "new", "summary": "Went to Barbuzzo."},
        ]
    ).to_csv(episodic_path, index=False)

    def fake_pass_llm(**kwargs):
        return ("[]", 3, 2)

    monkeypatch.setattr("llm.llm_selector.pass_llm", fake_pass_llm)

    result = kv_store.retrieve_from_kv_store(
        1,
        "Find another one like last time",
        memory_path=memory_path,
        episodic_memory_path=episodic_path,
    )

    assert result["episodic_memory"]["summary"] == "Went to Barbuzzo."
    assert "Went to Barbuzzo." in result["context"]
