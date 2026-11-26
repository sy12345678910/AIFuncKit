from __future__ import annotations

from typing import Optional, List
import numpy as np

from aifunckit._internal.schemas.memory_schemas import ActiveVectors, RecordData, MemoryRecord
from aifunckit._internal.memory.search_engine import VectorSearchEngine
from aifunckit._internal.memory.repository import SQLiteRepository
from aifunckit._internal.log_config import logger

VALID_LAYERS: tuple[str, str] = ("decomposition", "extraction")

class AIRAGMemoryManager:
    """SQLite-backed memory cache for decomposition and extraction embeddings."""

    def __init__(self, db_path: str = "aie_runtime.db") -> None:
        self.repository = SQLiteRepository(db_path=db_path)
        self.search_engine = VectorSearchEngine()
            
    async def initialize(self):
        try:
            await self.repository.init_db()
            records: List[ActiveVectors] = await self.repository.load_all_active_vectors()
            for layer_name in ("decomposition", "extraction"):
                layer_records = [r for r in records if r.layer == layer_name]
                if layer_records:
                    self.search_engine.load_initial_cache(
                        layer=layer_name,
                        ids=[r.id for r in layer_records],
                        vectors=[r.vector for r in layer_records]
                    )
            logger.info("RAG memory initialized with %d record(s)", len(records))
        except Exception as exc:
            logger.error("Failed to initialize RAG memory: %s", exc, exc_info=True)
            raise

    async def init_and_load_cache(self) -> None:
        try:
            await self.initialize()
            logger.info("RAG memory cache loaded")
        except Exception as exc:
            logger.error("Failed to load RAG memory cache: %s", exc, exc_info=True)
            raise

    async def add_record(self, record_data: RecordData) -> None:
        try:
            inserted_id = await self.repository.add_record(record_data=record_data)
            self.search_engine.add_vector_to_cache(layer=record_data.layer, record_id=inserted_id, raw_vector=record_data.raw_vector)
            logger.info("RAG record %d added to layer '%s'", inserted_id, record_data.layer)
        except Exception as exc:
            logger.error("Failed to add RAG record to layer '%s': %s", record_data.layer, exc, exc_info=True)
            raise

    async def search_relevant_example(
        self,
        layer: str,
        current_task_embedding: np.ndarray,
    ) -> Optional[MemoryRecord]:
        try:
            target_id = self.search_engine.find_best_match_id(layer=layer, query_raw_vector=current_task_embedding)
            if target_id is None:
                logger.info("No relevant RAG example found for layer '%s'", layer)
                return None

            example = await self.repository.get_record_by_id(target_id)
            if example is not None:
                logger.info("Relevant RAG example found for layer '%s'", layer)
            return example
        except Exception as exc:
            logger.error("Failed to search RAG examples on layer '%s': %s", layer, exc, exc_info=True)
            raise

    async def close(self) -> None:
        try:
            await self.repository.close()
            logger.info("RAG memory closed")
        except Exception as exc:
            logger.error("Failed to close RAG memory: %s", exc, exc_info=True)
            raise


AIRAGMemory = AIRAGMemoryManager
