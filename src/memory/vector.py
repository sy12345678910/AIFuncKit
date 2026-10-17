from typing import Any, Optional
import numpy as np
from numpy.typing import NDArray

def as_float32_vector(vec: Any) -> NDArray[np.float32]:
    return np.asarray(vec, dtype=np.float32).reshape(-1)

def normalize_vector(vec: Any) -> NDArray[np.float32]:
    array: NDArray[np.float32] = as_float32_vector(vec)
    norm: float = float(np.linalg.norm(array))
    if norm > 1e-10:
        return (array / norm).astype(np.float32)
    return array.astype(np.float32)

def compute_best_match(matrix: NDArray[np.float32], query_vector: NDArray[np.float32], threshold: float = 0.4) -> Optional[int]:
    similarities = np.dot(matrix, query_vector)
    best_idx = int(np.argmax(similarities))
    
    if similarities[best_idx] < threshold:
        return None
    return best_idx