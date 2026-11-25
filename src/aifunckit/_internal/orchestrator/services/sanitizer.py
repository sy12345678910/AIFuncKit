import re
from typing import List, Optional, Pattern
import numpy as np

try:
    import onnxruntime as ort
    from tokenizers import Tokenizer
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

from aifunckit.exceptions.prompt_exeption import PromptInjectionError, TooLongPromptError


class PromptSanitizer:
    DEFAULT_PATTERNS = [
        r"ignore\s+(all\s+)?(previous|prior)\s+instructions",
        r"forget\s+(all\s+)?(previous|prior)\s+rules",
        r"you\s+are\s+now\s+(DAN|unconstrained|jailbroken)",
        r"(output|print|show)\s+(your\s+)?(system\s+prompt|raw\s+instructions)",
        r"</?system>",
        r"```\s*system",
    ]

    def __init__(
        self,
        model_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        custom_patterns: Optional[List[str]] = None,
        max_length: int = 4096,
        threshold: float = 0.85,
    ):
        self.max_length = max_length
        self.threshold = threshold
        raw_patterns = [*self.DEFAULT_PATTERNS, *(custom_patterns or [])]
        self._patterns: List[Pattern[str]] = [
            re.compile(p, re.IGNORECASE) for p in raw_patterns if p
        ]

        self._session = None
        self._tokenizer = None

        if model_path and tokenizer_path:
            if not HAS_ONNX:
                raise ImportError("If you want to use AI guard: pip install onnxruntime tokenizers")
            self._tokenizer = Tokenizer.from_file(tokenizer_path)
            self._tokenizer.enable_truncation(max_length=512)
            self._session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    def sanitize(self, text: str) -> str:
        if not isinstance(text, str):
            raise TypeError("Prompt must be a string")

        cleaned = text.strip()
        if not cleaned:
            return cleaned

        if len(cleaned) > self.max_length:
            raise TooLongPromptError(f"Prompt too long: {len(cleaned)} > {self.max_length}")

        for pattern in self._patterns:
            if match := pattern.search(cleaned):
                raise PromptInjectionError(f"Suspicious pattern: '{match.group()}'")

        if self._session and self._tokenizer:
            encoded = self._tokenizer.encode(cleaned)
            outputs = self._session.run(None, {
                "input_ids": np.array([encoded.ids], dtype=np.int64),
                "attention_mask": np.array([encoded.attention_mask], dtype=np.int64),
            })
            logits = outputs[0][0]  
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)
            
            injection_prob = float(probs[1]) if len(probs) > 1 else float(probs[0])
            if injection_prob >= self.threshold:
                raise PromptInjectionError(f"Prompt Injection detected by model ({injection_prob:.2f})")

        return cleaned