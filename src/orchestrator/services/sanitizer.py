import re
from typing import List, Optional, Pattern

from exceptions.prompt_exeption import PromptInjectionError, TooLongPromptError


class PromptSanitizer:
    DEFAULT_INJECTION_PATTERNS = [
        r"ignore\s+(all\s+)?(previous|prior)\s+instructions",
        r"forget\s+(all\s+)?(previous|prior)\s+rules",
        r"you\s+are\s+now\s+(DAN|unconstrained|jailbroken)",
        r"system\s+override",
        r"(output|print|show)\s+(your\s+)?(system\s+prompt|raw\s+instructions)",
        r"</?system>",
        r"```\s*system",
    ]

    def __init__(self, custom_patterns: Optional[List[str]] = None, max_length: int = 4096):
        self.max_length = max_length
        raw_patterns = [
            pattern
            for pattern in (*self.DEFAULT_INJECTION_PATTERNS, *(custom_patterns or []))
            if pattern
        ]
        self._compiled: List[Pattern[str]] = [
            re.compile(pattern, re.IGNORECASE) for pattern in raw_patterns
        ]

    def sanitize(self, text: str) -> str:
        if not isinstance(text, str):
            raise TypeError("Prompt must be a string")
        if len(text) > self.max_length:
            raise TooLongPromptError(
                f"Prompt too long: {len(text)} > {self.max_length} symbols"
            )

        for pattern in self._compiled:
            match = pattern.search(text)
            if match:
                raise PromptInjectionError(
                    f"Prompt Injection: too suspicious pattern '{match.group()}'"
                )

        return text.strip()
