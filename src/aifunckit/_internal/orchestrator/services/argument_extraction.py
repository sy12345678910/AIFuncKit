from __future__ import annotations

import json
import re
from typing import Any, Optional, Type

from pydantic import BaseModel, ValidationError, create_model

from aifunckit._internal.log_config import logger
from aifunckit._internal.model_provide.model_router import get_data_generated_by_model
from aifunckit._internal.schemas.memory_schemas import MemoryRecord
from aifunckit._internal.schemas.orchestrator_schemas import LLMConfig


class ArgumentExtractionService:
    """Infer plugin argument values from live context and optional memory examples."""

    MAX_RETRIES: int = 2

    def _build_dynamic_schema(self, expected_args: dict[str, Any]) -> Type[BaseModel]:
        """Dynamically generate a Pydantic model from the target function's type annotations."""
        fields: dict[str, Any] = {}
        for key, field_type in expected_args.items():
            resolved_type = field_type if isinstance(field_type, type) or hasattr(field_type, "__origin__") else Any
            fields[key] = (resolved_type, ...)
        return create_model("DynamicActionArgs", **fields)

    def _clean_json_response(self, raw_response: Any) -> str:
        """Strip markdown code blocks and conversational filler around the JSON object."""
        if isinstance(raw_response, dict):
            return json.dumps(raw_response, ensure_ascii=False)
        if not isinstance(raw_response, str):
            return str(raw_response)

        cleaned = raw_response.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end >= start:
            return cleaned[start : end + 1]
        return cleaned

    async def extract(
        self,
        task: str,
        expected_args: dict[str, Any],
        current_context: dict[str, Any],
        llm_config: LLMConfig,
        memory_example: Optional[MemoryRecord],
    ) -> dict[str, Any]:
        """Infer and validate argument values for a plugin method from runtime context."""
        if not expected_args:
            logger.info("Task '%s' requires no arguments; skipping extraction", task)
            return {}

        dynamic_schema = self._build_dynamic_schema(expected_args)
        json_schema_str = json.dumps(dynamic_schema.model_json_schema(), ensure_ascii=False)

        memory_prompt_block: str = ""
        if memory_example:
            memory_prompt_block = (
                "PAST SUCCESSFUL EXPERIENCE:\n"
                f"- Context snapshot: {memory_example.context_snapshot}\n"
                f'- Task: "{memory_example.user_prompt}"\n'
                f"- Extracted arguments: {memory_example.extracted_json}\n\n"
            )

        base_prompt: str = (
            "You are a precise data extraction engine. Your ONLY job is to extract and compute "
            "function arguments from the user's task and the current runtime state.\n\n"
            f"TARGET JSON SCHEMA:\n{json_schema_str}\n\n"
            f"CURRENT RUNTIME STATE:\n{json.dumps(current_context, ensure_ascii=False, default=str)}\n\n"
            f"{memory_prompt_block}"
            "EXTRACTION RULES:\n"
            "1. Output ONLY a single valid JSON object matching the TARGET JSON SCHEMA.\n"
            "2. Every required key in the schema MUST be present in your output.\n"
            "3. Strictly respect data types (numbers as int/float without quotes, booleans as true/false, lists as arrays).\n"
            "4. Resolve relative or contextual references (e.g., 'increase by 10', 'previous file', 'louder') "
            "using values from CURRENT RUNTIME STATE.\n"
            "5. Preserve the original language of extracted text values (e.g., messages, names, queries) "
            "unless translation is explicitly requested.\n"
            "6. Do NOT include markdown formatting, explanations, or extra keys."
        )

        current_prompt = base_prompt

        for attempt in range(self.MAX_RETRIES + 1):
            try:
                response: Any = await get_data_generated_by_model(
                    prompt=current_prompt,
                    provider=llm_config.provider,
                    user_prompt=task,
                    format="json",
                    model=llm_config.model_name,
                    context_window=llm_config.context_window,
                )
            except Exception as exc:
                logger.error("Argument extraction LLM call failed (attempt %d): %s", attempt + 1, exc, exc_info=True)
                return {}

            try:
                if isinstance(response, BaseModel):
                    return response.model_dump()

                cleaned_json = self._clean_json_response(response)
                validated_args = dynamic_schema.model_validate_json(cleaned_json)
                return validated_args.model_dump()

            except (ValidationError, json.JSONDecodeError, TypeError) as exc:
                logger.warning(
                    "Argument validation failed on attempt %d/%d: %s",
                    attempt + 1,
                    self.MAX_RETRIES + 1,
                    exc,
                )
                if attempt < self.MAX_RETRIES:
                    current_prompt = (
                        f"{base_prompt}\n\n"
                        f"PREVIOUS OUTPUT FAILED VALIDATION:\n{response}\n"
                        f"VALIDATION ERROR DETAILS:\n{exc}\n"
                        "Fix the errors and return ONLY the corrected valid JSON object."
                    )

        logger.error("All argument extraction attempts failed for task: '%s'", task)
        return {}
