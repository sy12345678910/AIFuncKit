from __future__ import annotations

from typing import Optional, List
import numpy as np

from schemas.memory_schemas import ActiveVectors, RecordData, MemoryRecord
from memory.search_engine import VectorSearchEngine
from memory.repository import SQLiteRepository
from log_config import logger

VALID_LAYERS: tuple[str, str] = ("decomposition", "extraction")

class AIRAGMemoryManager:
    """SQLite-backed memory cache for decomposition and extraction embeddings."""

    def __init__(self, db_path: str = "aie_runtime.db") -> None:
        self.repository = SQLiteRepository(db_path=db_path)
        self.search_engine = VectorSearchEngine()
            
    async def initialize(self):
        await self.repository.init_db()
        records: List[ActiveVectors] = await self.repository.load_all_active_vectors()
        self.search_engine.load_initial_cache(
            layer="decomposition",
            ids=[r.id for r in records],
            vectors=[r.vector for r in records]
        )

    async def init_and_load_cache(self) -> None:
        await self.initialize()
        
    async def add_record(self, record_data: RecordData) -> None:
        inserted_id = await self.repository.add_record(record_data=record_data)
        self.search_engine.add_vector_to_cache(layer=record_data.layer, record_id=inserted_id, raw_vector=record_data.raw_vector)

    async def search_relevant_example(
        self,
        layer: str,
        current_task_embedding: np.ndarray,
    ) -> Optional[MemoryRecord]:
        """Return the most relevant historical example for a requested memory layer."""
        target_id = self.search_engine.find_best_match_id(layer=layer, query_raw_vector=current_task_embedding)
        if target_id is None:
            return None
        
        return await self.repository.get_record_by_id(target_id)
    
    async def close(self) -> None:
        return await self.repository.close()
        