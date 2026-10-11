# src/model_provide/model_router.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Optional
from config import TaskList
from log_config import logger

async def get_data_generated_by_model(
    prompt: str,
    user_prompt: str,
    format: Any = TaskList,
    provider: str = "ollama",
    model: str = "gemma4:e4b",
    api_key: Optional[str] = None,
    context_window: int = 4096,
) -> Any:
    logger.info("Requesting generation from provider '%s' (model='%s')", provider, model)
    response: Any = None

    try:
        match provider:
            case "hf" | "hugging_face":
                from model_provide import hf as hf_provider
                response = await hf_provider.generate_with_hf_on_cloud(
                    prompt=prompt,
                    user_prompt=user_prompt,
                    model_name=model,
                    token=api_key,
                    format=format,
                )
            case "openai":
                from model_provide import openai as openai_provider
                response = await openai_provider.generate_with_openai(
                    prompt=prompt,
                    user_prompt=user_prompt,
                    model=model,
                    api_key=api_key or "",
                    format=format,
                )
            case "ollama":
                from model_provide import ollama as ollama_provider
                compiled_format = format
                if hasattr(format, "model_json_schema"):
                    compiled_format = format.model_json_schema()

                response = await ollama_provider.generate_with_ollama(
                    prompt=prompt,
                    user_prompt=user_prompt,
                    model=model,
                    context_window=context_window,
                    format=compiled_format,  
                )
            case "llama_cpp":
                from model_provide import llama_cpp as llama_cpp_provider
                try:
                    response = await llama_cpp_provider.generate_with_llama_cpp(
                        prompt=prompt,
                        user_prompt=user_prompt,
                        model_path=str(Path(model)),
                        format=format,
                        context_window=context_window,
                    )
                except ImportError:
                    logger.error("llama_cpp isn't installed. Please, install llama_cpp_python")
                    raise ImportError("llama_cpp isn't installed. Please, install llama_cpp_python")
            case _:
                raise ValueError(f"Unknown provider: {provider}")
    except Exception as exc:
        logger.error("Provider '%s' generation failed: %s", provider, exc, exc_info=True)
        raise

    if format == "json":
        if isinstance(response, (dict, list)):
            return response
        try:
            cleaned_response = str(response).strip().removeprefix("```json").removesuffix("```").strip()
            return json.loads(cleaned_response)
        except (json.JSONDecodeError, TypeError, AttributeError) as exc:
            logger.error("LLM returned invalid JSON: %s. Raw: %s", exc, response)
            raise RuntimeError(f"LLM returned invalid JSON: {exc}. Raw: {response}") from exc

    if hasattr(format, "model_validate") or hasattr(format, "model_validate_json"):
        try:
            if isinstance(response, format):
                logger.info("LLM response validated as %s", getattr(format, "__name__", format))
                return response
            cleaned_response = str(response).strip().removeprefix("```json").removesuffix("```").strip()
            data = json.loads(cleaned_response)
            validated = format.model_validate(data)
            logger.info("LLM response validated as %s", getattr(format, "__name__", format))
            return validated
        except Exception as exc:
            logger.error("LLM returned invalid structure for %s: %s. Raw: %s", getattr(format, "__name__", format), exc, response)
            raise RuntimeError(f"LLM returned invalid structure for TaskList: {exc}. Raw: {response}") from exc

    return response
