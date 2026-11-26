from typing import Any, Optional
import numpy as np
from numpy.typing import NDArray
from aifunckit._internal.log_config import logger

def as_float32_vector(vec: Any) -> NDArray[np.float32]:
    try:
        return np.asarray(vec, dtype=np.float32).reshape(-1)
    except Exception as exc:
        logger.error("Failed to convert input to float32 vector: %s", exc, exc_info=True)
        raise

def normalize_vector(vec: Any) -> NDArray[np.float32]:
    array: NDArray[np.float32] = as_float32_vector(vec)
    norm: float = float(np.linalg.norm(array))
    if norm > 1e-10:
        return (array / norm).astype(np.float32)
    return array.astype(np.float32)

def compute_best_match(matrix: NDArray[np.float32], query_vector: NDArray[np.float32], threshold: float = 0.4) -> Optional[int]:
    try:
        similarities = np.dot(matrix, query_vector)
        best_idx = int(np.argmax(similarities))
        if similarities[best_idx] < threshold:
            return None
        return best_idx
    except Exception as exc:
        logger.error("compute_best_match failed: %s", exc, exc_info=True)
        raise