from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI

from log_config import logger


async def generate_with_openai(
    prompt: str,
    user_prompt: str,
    model: str,
    api_key: str,
    format: Any,
) -> str:
    client = AsyncOpenAI(api_key=api_key)
    schema = format.model_json_schema() if hasattr(format, "model_json_schema") else format

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={
                "type": "json_object",
                "schema": schema,
            },
        )
        content = response.choices[0].message.content
        logger.info("OpenAI generation succeeded for model '%s'", model)
        return str(content or "")
    except Exception as exc:
        logger.error("OpenAI API call failed for model '%s': %s", model, exc, exc_info=True)
        raise
