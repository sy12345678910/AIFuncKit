from __future__ import annotations

import json
from typing import Any, Optional
import numpy as np
from numpy.typing import NDArray
import aiosqlite

from log_config import logger
from src.schemas.memory_schemas import MemoryRecord, ActiveVectors, RecordData
from exceptions.vector_exceptions import FetchVectorError
from exceptions.db_exceptions import DBArchiveError

VALID_LAYERS: tuple[str, str] = ("decomposition", "extraction")

class BaseRepository:
    def __init__(self, db_path: str):
        self.db_path = db_path

class SQLiteRepository(BaseRepository):
    
    def __init__(self, db_path: str):
        super().__init__(db_path)        
        self.db: Optional[aiosqlite.Connection] = None  
        
    async def init_db(self) -> None:
        try:
            if self.db is None:
                self.db = await aiosqlite.connect(self.db_path)
                await self.db.execute("PRAGMA journal_mode = WAL;")
                await self.db.execute("PRAGMA synchronous = NORMAL;")
                await self.db.execute(
                    """
                    CREATE TABLE IF NOT EXISTS memory_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        layer TEXT NOT NULL,
                        user_prompt TEXT NOT NULL,
                        context_snapshot TEXT NOT NULL,
                        extracted_json TEXT NOT NULL,
                        embedding BLOB NOT NULL,
                        is_archived INTEGER DEFAULT 0
                    )
                    """
                )
                await self.db.commit()
                logger.info("SQLite memory store initialized at '%s'", self.db_path)
        except aiosqlite.Error as exc:
            logger.error("Failed to initialize SQLite memory store: %s", exc, exc_info=True)
            raise

    async def load_all_active_vectors(self) -> list[ActiveVectors]:
        if self.db is None:
            raise RuntimeError("Database not initialized.")
        try:
            async with self.db.execute(
                "SELECT id, layer, embedding FROM memory_history WHERE is_archived = 0"
            ) as cursor:
                rows = await cursor.fetchall()
                return [ActiveVectors.model_validate(dict(row)) for row in rows]
        except aiosqlite.Error as exc:
            logger.error("Failed to fetch vectors from SQLite: %s", exc, exc_info=True)
            raise FetchVectorError("Vectors aren't fetched from SQLite database")

    async def add_record(
        self,
        record_data: RecordData,
    ) -> int:
        if record_data.layer not in VALID_LAYERS:
            raise ValueError("Invalid layer name. Choose 'decomposition' or 'extraction'.")

        if self.db is None:
            raise RuntimeError("Database not initialized. Call init_db first.")

        try:
            blob: bytes = record_data.raw_vector.tobytes()
            json_str: str = json.dumps(record_data.args, ensure_ascii=False)
            context_str: str = json.dumps(record_data.context_dict, ensure_ascii=False)

            async with self.db.execute(
                """
                    INSERT INTO memory_history
                    (layer, user_prompt, context_snapshot, extracted_json, embedding)
                    VALUES (?, ?, ?, ?, ?)
                """,    
                (record_data.layer, record_data.user_prompt, context_str, json_str, blob),
            ) as cursor:
                inserted_id: int = int(cursor.lastrowid)
            await self.db.commit()
            
            logger.info("Stored RAG record id=%s on layer='%s'", inserted_id, record_data.layer)
            return inserted_id
            
        except (aiosqlite.Error, TypeError, ValueError) as exc:
            logger.error("Failed to persist memory record on layer '%s': %s", record_data.layer, exc, exc_info=True)
            raise

    async def get_record_by_id(self, record_id: int) -> Optional[MemoryRecord]:
        if self.db is None:
            raise RuntimeError("Database not initialized.")
        try:
            async with self.db.execute(
                """
                SELECT user_prompt, context_snapshot, extracted_json
                FROM memory_history
                WHERE id = ? AND is_archived = 0
                """,
                (record_id,),
            ) as cursor:
                row = await cursor.fetchone()
                if not row:
                    return None
                
                return MemoryRecord(
                    user_prompt=str(row[0]),
                    context_snapshot=json.loads(row[1]),
                    extracted_json=json.loads(row[2])
                )
        except aiosqlite.Error as exc:
            logger.error("Failed to fetch RAG record id=%s: %s", record_id, exc, exc_info=True)
            return None

    async def archive_old_records(self, days_limit: int = 365) -> None:
        """Archive memory entries older than the configured number of days."""
        if self.db is None:
            raise RuntimeError("Database not initialized.")
    
        modifier: str = f"-{days_limit} days"
        try:
            await self.db.execute(
                "UPDATE memory_history SET is_archived = 1 WHERE timestamp < datetime('now', ?)",
                (modifier,),
            )
            await self.db.commit()
            logger.info("Archived memory records older than %s days", days_limit)
        except aiosqlite.Error as exc:
            logger.error("Failed to archive old records: %s", exc, exc_info=True)
            raise DBArchiveError("Archiving is failed")

    async def close(self) -> None:
        if self.db:
            await self.db.close()
            self.db = None
            logger.info("SQLite connection closed safely.")
