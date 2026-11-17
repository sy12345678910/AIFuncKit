import re
from typing import List, Optional
from exceptions.prompt_exeption import PromptInjectionError, TooLongPromptError

class PromptSanitizer:
    DEFAULT_INJECTION_PATTERNS = [
        r"(?i)ignore\s+(all\s+)?(previous|prior)\s+instructions",
        r"(?i)forget\s+(all\s+)?(previous|prior)\s+rules",
        r"(?i)you\s+are\s+now\s+(DAN|unconstrained|jailbroken)",
        r"(?i)system\s+override",
        r"(?i)(output|print|show)\s+(your\s+)?(system\s+prompt|raw\s+instructions)",
        r"(?i)</?system>",
        r"(?i)```\s*system",
    ]

    def __init__(self, custom_patterns: Optional[List[str]] = None, max_length: int = 4096):
        self.max_length = max_length
        patterns = self.DEFAULT_INJECTION_PATTERNS + (custom_patterns or [])
        self._compiled_regex = re.compile("|".join(f"({p})" for p in patterns))

    def sanitize(self, text: str) -> str:
        if len(text) > self.max_length:
            raise TooLongPromptError(f"Prompt too long: {len(text)} > {self.max_length} symbols")

        match = self._compiled_regex.search(text)
        if match:
            raise PromptInjectionError(f"Prompt Injection: too suspicious patern '{match.group()}'")

        return text.strip()