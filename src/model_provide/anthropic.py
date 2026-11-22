from __future__ import annotations

from typing import Any
from anthropic import AsyncAnthropic

from log_config import logger

async def generate_with_anthropic(
    prompt: str,        
    user_prompt: str,   
    model: str,         
    api_key: str,
    context_window: int,
    format: Any,        
) -> str:
    client = AsyncAnthropic(
        api_key=api_key,
        default_headers={"anthropic-beta": "structured-outputs-2025-11-13"}
    )
    
    schema = format.model_json_schema() if hasattr(format, "model_json_schema") else format

    try:
        response = await client.messages.create(
            model=model,
            max_tokens=context_window,  
            system=prompt,    
            messages=[
                {"role": "user", "content": user_prompt},
            ],
            output_format={
                "type": "json",
                "schema": schema,
            },
        )
        
        content = str(response.content[0].text or "")
        logger.info("Anthropic generation succeeded for model '%s'", model)
        return content
    except Exception as exc:
        logger.error("Anthropic API call failed for model '%s': %s", model, exc, exc_info=True)
        raise
