from __future__ import annotations

from typing import Protocol


class LlmClient(Protocol):
    """Completions. Optional JPEG(s) = vision models (Ollama llava, GPT-4o, …)."""

    name: str
    model: str

    def status(self) -> str:
        """ready (...) | unavailable: reason"""
        ...

    def complete(
        self,
        system: str,
        user: str,
        image_jpeg: bytes | None = None,
        images: list[bytes] | None = None,
    ) -> str:
        ...
