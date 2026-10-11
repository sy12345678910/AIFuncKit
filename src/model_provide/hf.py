from __future__ import annotations

import asyncio
from typing import Any, Optional

import huggingface_hub as hf

from log_config import logger


async def generate_with_hf_on_cloud(
    prompt: str,
    user_prompt: str,
    model_name: str,
    format: Any,
    token: Optional[str] = None,
) -> str:
    client = hf.InferenceClient(token=token)
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_prompt},
    ]
    schema = format.model_json_schema() if hasattr(format, "model_json_schema") else format

    try:
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=model_name,
            messages=messages,
            response_format={
                "type": "json_object",
                "schema": schema,
            },
        )
        content = response.choices[0].message.content
        logger.info("Hugging Face generation succeeded for model '%s'", model_name)
        return str(content or "")
    except Exception as exc:
        logger.error(
            "Hugging Face API call failed for model '%s': %s",
            model_name,
            exc,
            exc_info=True,
        )
        raise
