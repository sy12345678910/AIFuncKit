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
    """Generate a structured completion using the OpenAI chat API."""
    client: AsyncOpenAI = AsyncOpenAI(api_key=api_key)
    schema: Any = format.model_json_schema() if hasattr(format, "model_json_schema") else format

    try:
        response: Any = await client.chat.completions.create(
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
        content: str = str(response.choices[0].message.content or "")
        logger.info("OpenAI generation succeeded for model '%s'", model)
        return content
    except Exception as exc:
        logger.error("OpenAI API call failed for model '%s': %s", model, exc, exc_info=True)
        raise
