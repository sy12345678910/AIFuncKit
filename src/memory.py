from __future__ import annotations
import json
import os
from typing import Any, Optional
import aiosqlite
import numpy as np
from numpy.typing import NDArray
from log_config import logger
VALID_LAYERS: tuple[str, str] = ("decomposition", "extraction")

def _as_float32_vector(vec: NDArray[np.floating] | np.ndarray) -> NDArray[np.float32]:
    if hasattr(vec, "detach"):
        vec = vec.detach().cpu().numpy()
    return np.asarray(vec, dtype=np.float32).reshape(-1)


class AIRAGMemory:
    def __init__(self, db_path: str = "aie_runtime.db") -> None:
        self.db_path: str = os.path.abspath(db_path)
        self.id_cache: dict[str, list[int]] = {"decomposition": [], "extraction": []}
        self.vector_list_cache: dict[str, list[NDArray[np.float32]]] = {
            "decomposition": [],
            "extraction": [],
        }
        self.compiled_matrix_cache: dict[str, Optional[NDArray[np.float32]]] = {
            "decomposition": None,
            "extraction": None,
        }
        self.db: Optional[aiosqlite.Connection] = None

    async def init_and_load_cache(self) -> None:
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

            await self._reload_cache_from_db()
        except aiosqlite.Error as exc:
            logger.error("Failed to initialize SQLite memory store: %s", exc, exc_info=True)
            raise

    async def _reload_cache_from_db(self) -> None:
        if self.db is None:
            logger.warning("Cannot reload cache: database connection is missing")
            return

        try:
            async with self.db.execute(
                "SELECT id, layer, embedding FROM memory_history WHERE is_archived = 0"
            ) as cursor:
                rows = await cursor.fetchall()
        except aiosqlite.Error as exc:
            logger.error("Failed to reload memory cache from SQLite: %s", exc, exc_info=True)
            raise

        self.id_cache = {"decomposition": [], "extraction": []}
        self.vector_list_cache = {"decomposition": [], "extraction": []}
        self.compiled_matrix_cache = {"decomposition": None, "extraction": None}

        for row_id, layer, blob in rows:
            if layer in self.id_cache:
                self.id_cache[layer].append(row_id)
                vec_np = np.frombuffer(blob, dtype=np.float32)
                self.vector_list_cache[layer].append(vec_np)

        for layer in VALID_LAYERS:
            if self.vector_list_cache[layer]:
                self.compiled_matrix_cache[layer] = np.stack(
                    self.vector_list_cache[layer], axis=0
                )

        logger.info(
            "Memory cache reloaded (decomposition=%s, extraction=%s)",
            len(self.id_cache["decomposition"]),
            len(self.id_cache["extraction"]),
        )

    def _normalize_vector(self, vec: np.ndarray) -> NDArray[np.float32]:
        array = _as_float32_vector(vec)
        norm = np.linalg.norm(array)
        if norm > 1e-10:
            return (array / norm).astype(np.float32)
        return array.astype(np.float32)

    async def add_record(
        self,
        layer: str,
        user_prompt: str,
        context_dict: dict[str, Any],
        extracted_args: dict[str, Any],
        embedding_vector: np.ndarray,
    ) -> None:
        if layer not in VALID_LAYERS:
            raise ValueError("Invalid layer name. Choose 'decomposition' or 'extraction'.")

        if self.db is None:
            raise RuntimeError("Database not initialized. Call init_and_load_cache first.")

        try:
            norm_vec = self._normalize_vector(embedding_vector)
            blob = norm_vec.tobytes()
            json_str = json.dumps(extracted_args, ensure_ascii=False)
            context_str = json.dumps(context_dict, ensure_ascii=False)

            async with self.db.execute(
                """
                INSERT INTO memory_history
                    (layer, user_prompt, context_snapshot, extracted_json, embedding)
                VALUES (?, ?, ?, ?, ?)
                """,
                (layer, user_prompt, context_str, json_str, blob),
            ) as cursor:
                inserted_id = cursor.lastrowid
            await self.db.commit()
        except (aiosqlite.Error, TypeError, ValueError) as exc:
            logger.error(
                "Failed to persist memory record on layer '%s': %s",
                layer,
                exc,
                exc_info=True,
            )
            raise

        self.id_cache[layer].append(inserted_id)
        self.vector_list_cache[layer].append(norm_vec)

        if self.compiled_matrix_cache[layer] is not None:
            new_row = norm_vec.reshape(1, -1)
            self.compiled_matrix_cache[layer] = np.vstack(
                [self.compiled_matrix_cache[layer], new_row]
            )
        else:
            self.compiled_matrix_cache[layer] = np.stack(
                self.vector_list_cache[layer], axis=0
            )

        logger.info("Stored RAG record id=%s on layer='%s'", inserted_id, layer)

    async def search_relevant_example(
        self,
        layer: str,
        current_task_embedding: np.ndarray,
    ) -> Optional[tuple[str, str, str]]:
        if layer not in VALID_LAYERS:
            logger.warning("search_relevant_example called with invalid layer '%s'", layer)
            return None
        if not self.id_cache[layer] or not self.vector_list_cache[layer]:
            logger.info("No RAG examples available on layer '%s'", layer)
            return None

        if self.compiled_matrix_cache[layer] is None:
            self.compiled_matrix_cache[layer] = np.stack(
                self.vector_list_cache[layer], axis=0
            )

        matrix = self.compiled_matrix_cache[layer]
        query = self._normalize_vector(current_task_embedding)
        similarities = np.dot(matrix, query)
        best_match_idx = int(np.argmax(similarities))
        score = float(similarities[best_match_idx])

        if score < 0.4:
            logger.info(
                "No RAG match on layer='%s' (best index=%s, numpy score=%.4f below 0.4)",
                layer,
                best_match_idx,
                score,
            )
            return None

        target_id = self.id_cache[layer][best_match_idx]
        if self.db is None or target_id is None:
            logger.warning("RAG match found but database is unavailable")
            return None

        logger.info(
            "RAG match on layer='%s': index=%s, memory_id=%s, numpy score=%.4f",
            layer,
            best_match_idx,
            target_id,
            score,
        )

        try:
            async with self.db.execute(
                """
                SELECT user_prompt, context_snapshot, extracted_json
                FROM memory_history
                WHERE id = ? AND is_archived = 0
                """,
                (target_id,),
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return str(row[0]), str(row[1]), str(row[2])
        except aiosqlite.Error as exc:
            logger.error(
                "Failed to fetch RAG example id=%s: %s",
                target_id,
                exc,
                exc_info=True,
            )
            return None
        return None

    async def archive_old_records(self, days_limit: int = 365) -> None:
        if self.db is None:
            raise RuntimeError("Database not initialized.")
        modifier = f"-{days_limit} days"
        try:
            await self.db.execute(
                "UPDATE memory_history SET is_archived = 1 WHERE timestamp < datetime('now', ?)",
                (modifier,),
            )
            await self.db.commit()
            logger.info("Archived memory records older than %s days", days_limit)
            await self._reload_cache_from_db()
        except aiosqlite.Error as exc:
            logger.error("Failed to archive old records: %s", exc, exc_info=True)
            raise

    async def close(self) -> None:
        if self.db:
            try:
                await self.db.close()
                logger.info("SQLite memory connection closed")
            except aiosqlite.Error as exc:
                logger.error("Failed to close SQLite connection: %s", exc, exc_info=True)
            finally:
                self.db = None
