import numpy as np
from numpy.typing import NDArray
from log_config import logger
from exceptions.vector_exceptions import VectorGenerationError

class EmbeddingService:
    def __init__(self, model_name: str = "sentence_transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self._model = None
        self.model_name = model_name
        
    def _encode_text(self, text: str) -> NDArray[np.float32]:
        """Convert a text string into a normalized embedding vector."""
        self.ensure_embedding_model(self)

        try:
            embidding_generator = self._model.embed([text])
            raw_vector = next(embidding_generator)
            
            return np.asarray(raw_vector, dtype=np.float32).reshape(-1)
        except Exception as e:
            logger.error(f"Vector isn't generated correctly {e}")
            raise VectorGenerationError
        
    def ensure_embedding_model(self) -> None:
        """Ensure the configured embedding model is loaded for vector encoding."""
        if self._model is not None:
            return
        try:
            from fastembed import TextEmbedding
            self._model = TextEmbedding(model_name=self.model_name)
            logger.info("Loaded embedding model '%s' via ONNX", self.model_name)
        except Exception as exc:
            logger.error("Failed to load embedding model '%s': %s", self.model_name, exc, exc_info=True)
            raise 