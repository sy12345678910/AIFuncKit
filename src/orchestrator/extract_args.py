import json
from typing import Optional, Any

from log_config import logger
from model_provide.model_router import get_data_generated_by_model
from schemas.memory_schemas import MemoryRecord
from src.schemas.orchestrator_schemas import LLMConfig

async def extract_args(
        task: str,
        expected_args: dict[str, Any],
        current_context: dict[str, Any],
        llm_config: LLMConfig,
        memory_example: Optional[MemoryRecord],
    ) -> dict[str, Any]:
        """Infer argument values for a plugin method from the active runtime context."""
        memory_prompt_block: str = ""
        if memory_example:
            past_prompt = memory_example.user_prompt
            past_context = memory_example.context_snapshot
            past_json = memory_example.extracted_json

            memory_prompt_block = (
                "PAST SUCCESSFUL EXPERIENCE:\n"
                f"- System state: {past_context}\n"
                f'- User said: "{past_prompt}"\n'
                f"- Calculated result: {past_json}\n\n"
            )

        readable_schema: dict[str, str] = {
            key: getattr(val, "__name__", str(val)) for key, val in expected_args.items()
        }

        prompt: str = (
            "You are a strict data extraction compiler for an automation environment (AIE).\n"
            f"REQUIRED ARGUMENTS AND TYPES: {json.dumps(readable_schema)}.\n"
            f"CURRENT LIVE ENVIRONMENT STATE: {json.dumps(current_context)}.\n\n"
            f"{memory_prompt_block}"
            "COMPILATION RULES:\n"
            "1. Extracted JSON keys MUST strictly match the REQUIRED ARGUMENTS.\n"
            "2. Convert types according to expectations (e.g., if type is 'int', provide a "
            "number, if 'bool', provide true/false).\n"
            "3. Resolve relative requests (e.g., 'louder', 'hotter') using calculations based "
            "on CURRENT LIVE ENVIRONMENT STATE.\n"
            "4. Output ONLY raw, valid JSON matching the exact target keys without markdown "
            "formatting or markdown code blocks."
        )

        try:
            response: Any = await get_data_generated_by_model(
                prompt=prompt,
                provider=llm_config.provider,
                user_prompt=task,
                format="json",
                model=llm_config.model_name,
                context_window=llm_config.context_window
            )
        except Exception as exc:
            logger.error("Argument extraction LLM call failed: %s", exc, exc_info=True)
            return {}

        try:
            if isinstance(response, dict):
                return response
            return json.loads(response)
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning("Could not parse extracted arguments JSON: %s", exc)
            return {}