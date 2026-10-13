from __future__ import annotations

import json
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from json_repair import repair_json

from prompts import PROMPT_KV_MEMORY_CONSTRUCT, PROMPT_KV_MEMORY_RETRIEVE
from utils.format import extract_json_list


KV_COLUMNS = ["user_id", "key", "value", "condition", "timestamp", "session_id"]
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
KV_MEMORY_PATH = DATA_DIR / "kv_memory.csv"
EPISODIC_MEMORY_PATH = DATA_DIR / "navigation_memory.csv"
KV_MEMORY_LOCK = threading.Lock()
COREFERENCE_PHRASES = [
    "that place",
    "like last time",
    "same as before",
    "back to",
    "another one like",
    "where i went",
    "my usual",
    "similar to last time",
]


def _empty_kv_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=KV_COLUMNS)


def _normalize_user_id(user_id: Any) -> str:
    return str(user_id)


def _normalize_key(key: Any) -> str:
    return str(key or "").strip()


def _normalize_text(value: Any, limit: Optional[int] = None) -> str:
    text = str(value or "").strip()
    if limit is not None:
        text = text[:limit]
    return text


def _normalize_condition(condition: Any) -> str:
    """Keep only real applicability conditions, not preference values."""
    text = _normalize_text(condition).lower().replace(" ", "_")
    if not text:
        return ""

    invalid_conditions = {
        "cheap", "budget", "budget-friendly", "affordable", "inexpensive",
        "italian", "thai", "japanese", "mexican", "indian", "chinese",
        "korean", "vietnamese", "mediterranean", "american", "pizza",
        "$", "$$", "$$$", "$$$$",
    }
    if text in invalid_conditions:
        return ""

    condition_markers = (
        "with_", "for_", "during_", "at_", "when_", "near_", "on_",
    )
    condition_terms = {
        "family", "kids", "children", "child", "friends", "alone", "solo",
        "rush_hour", "commute", "work", "date", "night", "morning",
    }
    if text.startswith(condition_markers) or text in condition_terms:
        return text

    return ""


def _load_kv_memory(memory_path: Path = KV_MEMORY_PATH) -> pd.DataFrame:
    if not memory_path.exists():
        return _empty_kv_frame()
    try:
        memory_df = pd.read_csv(memory_path, dtype=str).fillna("")
    except Exception:
        return _empty_kv_frame()
    for column in KV_COLUMNS:
        if column not in memory_df.columns:
            memory_df[column] = ""
    return memory_df[KV_COLUMNS].fillna("")


def _save_kv_memory(memory_df: pd.DataFrame, memory_path: Path = KV_MEMORY_PATH) -> None:
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    memory_df[KV_COLUMNS].to_csv(memory_path, index=False)


def memory_add(
    user_id: Any,
    key: str,
    value: str,
    condition: str = "",
    session_id: str = "",
    memory_path: Path = KV_MEMORY_PATH,
) -> Optional[Dict[str, str]]:
    """Create or overwrite a KV memory entry for one user/key/condition."""
    normalized_key = _normalize_key(key)
    normalized_value = _normalize_text(value, limit=50)
    normalized_condition = _normalize_condition(condition)
    if not normalized_key or not normalized_value:
        return None

    row = {
        "user_id": _normalize_user_id(user_id),
        "key": normalized_key,
        "value": normalized_value,
        "condition": normalized_condition,
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "session_id": _normalize_text(session_id),
    }

    with KV_MEMORY_LOCK:
        memory_df = _load_kv_memory(memory_path)
        same_entry = (
            (memory_df["user_id"] == row["user_id"])
            & (memory_df["key"].str.lower() == normalized_key.lower())
            & (memory_df["condition"].str.lower() == normalized_condition.lower())
        )
        memory_df = memory_df.loc[~same_entry].copy()
        memory_df = pd.concat([memory_df, pd.DataFrame([row])], ignore_index=True)
        _save_kv_memory(memory_df, memory_path)
    return row


def memory_remove(
    user_id: Any,
    key: str,
    memory_path: Path = KV_MEMORY_PATH,
) -> int:
    """Delete matching entries for one user/key and return number removed."""
    normalized_key = _normalize_key(key)
    if not normalized_key:
        return 0

    with KV_MEMORY_LOCK:
        memory_df = _load_kv_memory(memory_path)
        same_entry = (
            (memory_df["user_id"] == _normalize_user_id(user_id))
            & (memory_df["key"].str.lower() == normalized_key.lower())
        )
        removed = int(same_entry.sum())
        if removed:
            _save_kv_memory(memory_df.loc[~same_entry].copy(), memory_path)
    return removed


def memory_list(user_id: Any, memory_path: Path = KV_MEMORY_PATH) -> List[Dict[str, str]]:
    memory_df = _load_kv_memory(memory_path)
    user_df = memory_df[memory_df["user_id"] == _normalize_user_id(user_id)]
    return user_df.to_dict(orient="records")


def memory_search(user_id: Any, key: str, memory_path: Path = KV_MEMORY_PATH) -> List[Dict[str, str]]:
    normalized_key = _normalize_key(key).lower()
    if not normalized_key:
        return []

    entries = memory_list(user_id, memory_path=memory_path)
    exact = [entry for entry in entries if entry["key"].lower() == normalized_key]
    if exact:
        return exact

    return [
        entry
        for entry in entries
        if normalized_key in entry["key"].lower() or normalized_key in entry["value"].lower()
    ]


def query_has_coreference(query: str) -> bool:
    normalized_query = str(query or "").lower()
    return any(phrase in normalized_query for phrase in COREFERENCE_PHRASES)


def _latest_episodic_memory(memory_path: Path = EPISODIC_MEMORY_PATH) -> Optional[Dict[str, str]]:
    if not memory_path.exists():
        return None
    try:
        memory_df = pd.read_csv(memory_path, dtype=str).fillna("")
    except Exception:
        return None
    if memory_df.empty:
        return None
    row = memory_df.iloc[-1]
    return {
        "conversation_id": str(row.get("conversation_id", "")),
        "time": str(row.get("time", "")),
        "summary": str(row.get("summary", "")),
    }


def _extract_tool_calls(llm_response: str) -> List[Dict[str, Any]]:
    try:
        parsed = extract_json_list(repair_json(llm_response))
    except Exception:
        return []
    return parsed if isinstance(parsed, list) else []


def _tool_name(call: Dict[str, Any]) -> str:
    return str(call.get("tool") or call.get("name") or call.get("function") or "").strip()


def _tool_args(call: Dict[str, Any]) -> Dict[str, Any]:
    args = call.get("arguments") or call.get("args")
    if isinstance(args, dict):
        return args
    if all(field in call for field in ("key", "value")):
        return {"key": call.get("key"), "value": call.get("value"), "condition": call.get("condition", "")}
    if "key" in call:
        return {"key": call.get("key")}
    return {}


def save_to_kv_store(
    user_id: Any,
    conversation_turn: Any,
    session_id: str = "",
    llm_model: str = "",
    memory_path: Path = KV_MEMORY_PATH,
) -> Dict[str, Any]:
    """Ask the LLM for memory tool calls and execute them against the KV CSV."""
    from llm.llm_selector import pass_llm

    current_entries = memory_list(user_id, memory_path=memory_path)
    prompt = PROMPT_KV_MEMORY_CONSTRUCT.format(
        current_keys=json.dumps([entry["key"] for entry in current_entries], ensure_ascii=False, indent=2),
        conversation_turn=json.dumps(conversation_turn, ensure_ascii=False, indent=2)
        if not isinstance(conversation_turn, str)
        else conversation_turn,
    )
    response, tokens_input, tokens_output = pass_llm(prompt=prompt, model=llm_model, max_tokens=500)
    print("[DEBUG] KV memory construction raw response:", response)
    calls = _extract_tool_calls(response)
    executed = []

    for call in calls:
        name = _tool_name(call)
        args = _tool_args(call)
        key = _normalize_key(args.get("key"))
        if not key:
            continue

        if name in ("memory_search", "search"):
            executed.append({"tool": "memory_search", "key": key, "result": memory_search(user_id, key, memory_path)})
            continue

        if name in ("memory_remove", "remove"):
            removed = memory_remove(user_id, key, memory_path=memory_path)
            executed.append({"tool": "memory_remove", "key": key, "removed": removed})
            continue

        if name in ("memory_add", "add") or "value" in args:
            existing = memory_search(user_id, key, memory_path)
            row = memory_add(
                user_id,
                key,
                args.get("value", ""),
                condition=args.get("condition", ""),
                session_id=session_id,
                memory_path=memory_path,
            )
            if row:
                executed.append({"tool": "memory_add", "key": key, "previous": existing, "row": row})

    return {
        "executed": executed,
        "raw_response": response,
        "tokens_input": tokens_input,
        "tokens_output": tokens_output,
    }


def replace_kv_memory(
    user_id: Any,
    rows: List[Dict[str, Any]],
    llm_model: str = "",
    memory_path: Path = KV_MEMORY_PATH,
) -> Dict[str, Any]:
    """Replace one user's KV entries from RAG-style episodic-memory rows."""
    with KV_MEMORY_LOCK:
        memory_df = _load_kv_memory(memory_path)
        remaining = memory_df[memory_df["user_id"] != _normalize_user_id(user_id)].copy()
        _save_kv_memory(remaining, memory_path)

    history = "\n".join(
        f"[{row.get('time', '')}] {row.get('summary', '')}"
        for row in rows
        if str(row.get("summary", "")).strip()
    )
    if not history:
        return {"executed": [], "raw_response": "", "tokens_input": 0, "tokens_output": 0}

    session_id = str(rows[-1].get("conversation_id", "injected")) if rows else "injected"
    return save_to_kv_store(
        user_id,
        history,
        session_id=session_id,
        llm_model=llm_model,
        memory_path=memory_path,
    )


def _condition_matches(query: str, condition: str) -> bool:
    condition = _normalize_text(condition).lower()
    if not condition:
        return True
    query = str(query or "").lower()
    normalized_condition = condition.replace("_", " ")
    if condition in query or normalized_condition in query:
        return True

    synonyms = {
        "family": ["family", "kids", "children", "child"],
        "kids": ["family", "kids", "children", "child"],
        "rush": ["rush hour", "traffic", "commute"],
        "hour": ["rush hour", "traffic", "commute"],
    }
    tokens = [token for token in re.split(r"[^a-z0-9]+", condition) if token and token not in {"with", "for", "at", "during"}]
    for token in tokens:
        if token in query:
            return True
        if any(alias in query for alias in synonyms.get(token, [])):
            return True
    return False


def _label_for_key(key: str) -> str:
    parts = [part for part in str(key).split("_") if part]
    if parts and parts[0].lower() == "user":
        parts = parts[1:]
    if parts and parts[-1].lower() == "preference":
        parts = parts[:-1]
    label = "_".join(parts).lower()
    label_map = {
        "cuisine": "cuisine",
        "price": "price",
        "dietary_restriction": "dietary",
        "route_avoid": "route avoid",
        "home_address": "home address",
        "noise_level": "noise",
        "good_for_kids": "kids",
    }
    return label_map.get(label, label.replace("_", " ") or key)


def format_kv_context(entries: List[Dict[str, str]], episodic_memory: Optional[Dict[str, str]] = None) -> str:
    lines = ["User preferences from memory:"]
    for entry in entries:
        lines.append(f"- {_label_for_key(entry.get('key', ''))}: {entry.get('value', '')}")
    if episodic_memory and episodic_memory.get("summary"):
        lines.append("Recent navigation episode from memory:")
        lines.append(f"- {episodic_memory['summary']}")
    return "\n".join(lines) if len(lines) > 1 else ""


def retrieve_from_kv_store(
    user_id: Any,
    query: str,
    llm_model: str = "",
    memory_path: Path = KV_MEMORY_PATH,
    episodic_memory_path: Path = EPISODIC_MEMORY_PATH,
    include_episodic_memory: bool = True,
) -> Dict[str, Any]:
    from llm.llm_selector import pass_llm

    entries = memory_list(user_id, memory_path=memory_path)
    if not entries and not query_has_coreference(query):
        return {"entries": [], "context": "", "episodic_memory": None, "tokens_input": 0, "tokens_output": 0}

    prompt = PROMPT_KV_MEMORY_RETRIEVE.format(
        query=query,
        memories=json.dumps(entries, ensure_ascii=False, indent=2),
    )
    response, tokens_input, tokens_output = pass_llm(prompt=prompt, model=llm_model, max_tokens=350)
    calls = _extract_tool_calls(response)

    selected = []
    seen = set()
    for call in calls:
        args = _tool_args(call)
        key = _normalize_key(args.get("key"))
        if not key:
            continue
        for entry in memory_search(user_id, key, memory_path=memory_path):
            entry_id = (entry.get("user_id", ""), entry.get("key", ""), entry.get("condition", ""))
            if entry_id in seen:
                continue
            if not _condition_matches(query, entry.get("condition", "")):
                continue
            selected.append(entry)
            seen.add(entry_id)

    episodic_memory = (
        _latest_episodic_memory(episodic_memory_path)
        if include_episodic_memory and query_has_coreference(query)
        else None
    )
    return {
        "entries": selected,
        "context": format_kv_context(selected, episodic_memory=episodic_memory),
        "episodic_memory": episodic_memory,
        "tokens_input": tokens_input,
        "tokens_output": tokens_output,
    }
