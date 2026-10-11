# src/model_provide/ollama.py
from __future__ import annotations
from typing import Any
import ollama
from log_config import logger

async def generate_with_ollama(
    prompt: str,
    user_prompt: str,
    model: str,
    format: Any,
    context_window: int = 4096,
) -> str:
    agent = ollama.AsyncClient()
    try:
        response = await agent.chat(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_prompt},
            ],
            format=format,  
            options={
                "num_ctx": context_window,
                "temperature": 0.0,
                "num_predict": 256,
            },
        )
        content = response["message"]["content"]
        logger.info("Ollama generation succeeded for model '%s'", model)
        return str(content)
    except Exception as exc:
        logger.error("Ollama API call failed for model '%s': %s", model, exc, exc_info=True)
        raise

