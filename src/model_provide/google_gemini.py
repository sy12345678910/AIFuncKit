from __future__ import annotations

from typing import Any
from google import genai
from google.genai import types

from log_config import logger

async def generate_with_gemini(
    prompt: str,        
    user_prompt: str,   
    model: str,         
    api_key: str,
    format: Any,        
) -> str:
    client = genai.Client(api_key=api_key)
    
    schema = format.model_json_schema() if hasattr(format, "model_json_schema") else format

    try:
        config = types.GenerateContentConfig(
            system_instruction=prompt,
            response_mime_type="application/json",
            response_schema=schema,
        )

        response = await client.aio.models.generate_content(
            model=model,
            contents=[user_prompt],
            config=config,
        )
        
        content = str(response.text or "")
        logger.info("Gemini generation succeeded for model '%s'", model)
        return content
    except Exception as exc:
        logger.error("Gemini API call failed for model '%s': %s", model, exc, exc_info=True)
        raise
