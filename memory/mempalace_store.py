import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np


@dataclass(frozen=True)
class MemoryResult:
    conversation_id: str
    time: str
    summary: str
    score: float


class MemPalaceNavigationMemory:
    WING = "convnavi"
    COLLECTION = "mempalace_drawers"

    def __init__(
        self,
        memory_dir: Path,
        max_entries: int,
        embed_texts: Callable[[List[str]], np.ndarray],
    ):
        self.memory_dir = memory_dir
        self.max_entries = max_entries
        self.embed_texts = embed_texts
        self.palace_dir = memory_dir / "mempalace"

    def replace(self, rows: List[Dict[str, Any]]) -> int:
        normalized_rows = [self._normalize_row(row, index) for index, row in enumerate(rows)]
        normalized_rows = normalized_rows[-self.max_entries:]
        shutil.rmtree(self.palace_dir, ignore_errors=True)
        if not normalized_rows:
            return 0

        collection = self._collection()
        texts = [row["summary"] for row in normalized_rows]
        collection.add(
            documents=texts,
            ids=[row["id"] for row in normalized_rows],
            metadatas=[self._metadata(row) for row in normalized_rows],
            embeddings=self.embed_texts(texts).tolist(),
        )
        return len(normalized_rows)

    def append(self, row: Dict[str, Any]) -> None:
        collection = self._collection()
        existing = collection.get(include=["metadatas"])
        if len(existing["ids"]) >= self.max_entries:
            collection.delete(ids=[existing["ids"][0]])

        normalized_row = self._normalize_row(row, len(existing["ids"]))
        collection.add(
            documents=[normalized_row["summary"]],
            ids=[normalized_row["id"]],
            metadatas=[self._metadata(normalized_row)],
            embeddings=self.embed_texts([normalized_row["summary"]]).tolist(),
        )

    def search(self, query: str, top_k: int) -> List[MemoryResult]:
        if not self.palace_dir.exists():
            return []

        collection = self._collection()
        available = collection.count()
        if not available:
            return []

        results = collection.query(
            query_embeddings=self.embed_texts([query]).tolist(),
            n_results=min(top_k, available),
            where={"wing": self.WING},
            include=["documents", "metadatas", "distances"],
        )
        return [
            MemoryResult(
                conversation_id=metadata["conversation_id"],
                time=metadata["time"],
                summary=document,
                score=max(0.0, 1.0 - float(distance)),
            )
            for document, metadata, distance in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

    def _collection(self):
        from mempalace.backends.sqlite_exact import SQLiteExactBackend

        return SQLiteExactBackend().get_collection(
            str(self.palace_dir),
            self.COLLECTION,
            create=True,
        )

    @staticmethod
    def _normalize_row(row: Dict[str, Any], index: int) -> Dict[str, str]:
        conversation_id = str(row.get("conversation_id", ""))
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", conversation_id).strip("-") or "unknown"
        return {
            "id": f"{index:06d}_{safe_id}",
            "conversation_id": conversation_id,
            "time": str(row.get("time", "")),
            "summary": str(row.get("summary", "")),
        }

    def _metadata(self, row: Dict[str, str]) -> Dict[str, str]:
        return {
            "wing": self.WING,
            "conversation_id": row["conversation_id"],
            "time": row["time"],
        }