from __future__ import annotations

import asyncio
from typing import Any, Optional

from log_config import logger

_llm: Optional[Any] = None


async def generate_with_llama_cpp(
    prompt: str,
    user_prompt: str,
    model_path: str,
    format: Any,
    context_window: int = 4096,
) -> str:
    global _llm
    try:
        if not _llm:
            from llama_cpp import Llama

            _llm = Llama(
                model_path=model_path,
                n_ctx=context_window,
                verbose=False,
                chat_format="chatml",
            )
            logger.info("Loaded llama.cpp model from '%s'", model_path)
    except ImportError:
        logger.error("llama_cpp isn't installed. Please, install llama_cpp_python")
        raise
    except Exception as exc:
        logger.error("Failed to load llama.cpp model '%s': %s", model_path, exc, exc_info=True)
        raise

    def run() -> str:
        schema = format.model_json_schema() if hasattr(format, "model_json_schema") else format
        res = _llm.create_chat_completion(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=256,
            temperature=0.0,
            response_format={
                "type": "json_object",
                "schema": schema,
            },
        )
        return str(res["choices"][0]["message"]["content"])

    try:
        content = await asyncio.to_thread(run)
        logger.info("llama.cpp generation succeeded for '%s'", model_path)
        return content
    except Exception as exc:
        logger.error("llama.cpp generation failed: %s", exc, exc_info=True)
        raise
