from __future__ import annotations

from typing import Any, Optional

import numpy as np

from aifunckit._internal.memory.repository import SQLiteRepository
from aifunckit._internal.schemas.memory_schemas import RecordData


class AIRAGMemory:
    """Compatibility wrapper for tests that expect the legacy AIRAGMemory API."""

    def __init__(self, db_path: str = "aie_runtime.db") -> None:
        self.db_path = db_path
        self.repository = SQLiteRepository(db_path=db_path)
        self.db = None
        self.id_cache: dict[str, list[int]] = {"decomposition": [], "extraction": []}
        self.vector_list_cache: dict[str, list[np.ndarray]] = {
            "decomposition": [],
            "extraction": [],
        }
        self.compiled_matrix_cache: dict[str, Optional[np.ndarray]] = {
            "decomposition": None,
            "extraction": None,
        }

    @staticmethod
    def _normalize_vector(vector: Any) -> np.ndarray:
        arr = np.asarray(vector, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(arr))
        if norm > 1e-10:
            return (arr / norm).astype(np.float32)
        return arr.astype(np.float32)

    async def init_and_load_cache(self) -> None:
        await self.repository.init_db()
        self.db = self.repository.db

        for layer in ("decomposition", "extraction"):
            self.id_cache[layer] = []
            self.vector_list_cache[layer] = []
            self.compiled_matrix_cache[layer] = None

        records = await self.repository.load_all_active_vectors()
        for record in records:
            layer = record.layer
            self.id_cache.setdefault(layer, []).append(record.id)
            vector = np.asarray(record.vector, dtype=np.float32)
            self.vector_list_cache.setdefault(layer, []).append(vector)

        for layer in ("decomposition", "extraction"):
            vectors = self.vector_list_cache.get(layer, [])
            if vectors:
                self.compiled_matrix_cache[layer] = np.stack(vectors, axis=0).astype(np.float32)

    async def add_record(
        self,
        layer: str,
        user_prompt: str,
        context_snapshot: dict[str, Any],
        extracted_json: dict[str, Any],
        vector: Any,
    ) -> int:
        record_data = RecordData(
            layer=layer,
            user_prompt=user_prompt,
            context_dict=context_snapshot,
            args=extracted_json,
            raw_vector=np.asarray(vector, dtype=np.float32),
        )
        inserted_id = await self.repository.add_record(record_data=record_data)
        self.id_cache.setdefault(layer, []).append(inserted_id)
        normalized = self._normalize_vector(vector)
        self.vector_list_cache.setdefault(layer, []).append(normalized)

        current = self.compiled_matrix_cache.get(layer)
        if current is None:
            self.compiled_matrix_cache[layer] = normalized.reshape(1, -1)
        else:
            self.compiled_matrix_cache[layer] = np.vstack([current, normalized.reshape(1, -1)])
        return inserted_id

    async def search_relevant_example(
        self,
        layer: str,
        query_vector: Any,
    ) -> Optional[tuple[str, dict[str, Any], dict[str, Any]]]:
        vectors = self.vector_list_cache.get(layer, [])
        if not vectors:
            return None

        matrix = self.compiled_matrix_cache.get(layer)
        if matrix is None:
            return None

        query = self._normalize_vector(query_vector)
        similarities = np.dot(matrix, query)
        if similarities.size == 0:
            return None

        best_index = int(np.argmax(similarities))
        if similarities[best_index] < 0.4:
            return None

        target_id = self.id_cache.get(layer, [])[best_index]
        record = await self.repository.get_record_by_id(target_id)
        if record is None:
            return None
        return (record.user_prompt, record.context_snapshot, record.extracted_json)

    async def archive_old_records(self, days_limit: int = 365) -> None:
        await self.repository.archive_old_records(days_limit=days_limit)
        await self.init_and_load_cache()

    async def close(self) -> None:
        if self.repository.db is not None:
            await self.repository.close()
        self.db = None


__all__ = ["AIRAGMemory"]
