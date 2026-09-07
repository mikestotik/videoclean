from __future__ import annotations

from typing import Protocol

import numpy as np

from videoclean.domain.intent import Intent


class PromptParser(Protocol):
    name: str

    def parse(
        self,
        prompt: str | None,
        frame: np.ndarray | None = None,
        frames: list[np.ndarray] | None = None,
    ) -> Intent: ...
