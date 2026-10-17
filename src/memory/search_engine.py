from typing import Optional
import numpy as np
from numpy.typing import NDArray
from memory.vector import normalize_vector, compute_best_match

class VectorSearchEngine:
    def __init__(self) -> None:
        self._matrices: dict[str, Optional[NDArray[np.float32]]] = {"decomposition": None, "extraction": None}
        self._ids: dict[str, list[int]] = {"decomposition": [], "extraction": []}

    def load_initial_cache(self, ids: list[int], layer: str, vectors: list[NDArray[np.float32]]) -> None:
        if not vectors:
            return
        self._ids[layer] = ids
        self._matrices[layer] = np.stack(vectors, axis=0)

    def add_vector_to_cache(self, layer: str, record_id: int, raw_vector: NDArray[np.float32]) -> None:
        norm_vector = normalize_vector(raw_vector)
        self._ids[layer].append(record_id)
        
        if self._matrices[layer] is not None:
            new_row = norm_vector.reshape(1, -1)
            self._matrices[layer] = np.vstack([self._matrices[layer], new_row])
        else: 
            self._matrices[layer] = norm_vector.reshape(1, -1)

    def find_best_match_id(self, layer: str, query_raw_vector: NDArray[np.float32], threshold: float = 0.4) -> Optional[int]:
        matrix = self._matrices[layer]
        if matrix is None or not self._ids[layer]:
            return None

        query = normalize_vector(query_raw_vector)
        best_idx = compute_best_match(matrix, query, threshold)
        
        return self._ids[layer][best_idx] if best_idx is not None else None
