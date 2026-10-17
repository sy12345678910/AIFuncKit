# from typing import Any
# import numpy as np
# from numpy.typing import NDArray

# def as_float32_vector(vec: Any) -> NDArray[np.float32]:
#     """Normalize vector-like objects into a flat float32 NumPy array."""
#     if hasattr(vec, "detach"):
#         vec = vec.detach().cpu().numpy()
#     return np.asarray(vec, dtype=np.float32).reshape(-1)

# def metadata_action_name(meta: dict[str, Any]) -> str:
#     """Resolve the action name from a metadata record while handling callable names."""
#     action_name: Any = meta.get("action_name")
#     if action_name:
#         return str(action_name)
#     stored: Any = meta.get("name")
#     if callable(stored):
#         return stored.__name__
#     return str(stored or "")

# def normalize_vector(vec: np.ndarray) -> NDArray[np.float32]:
#         """L2-normalize the vector and cast it to float32 for semantic matching."""
#         array: NDArray[np.float32] = as_float32_vector(vec)
#         norm: float = float(np.linalg.norm(array))
#         if norm > 1e-10:
#             return (array / norm).astype(np.float32)
#         return array.astype(np.float32)