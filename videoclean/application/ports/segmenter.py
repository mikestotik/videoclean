from __future__ import annotations

from typing import Protocol

import numpy as np

from videoclean.domain.tracks import Track


class Segmenter(Protocol):
    """Turns tracks into per-frame masks."""

    name: str

    def masks(self, frames: list[np.ndarray], tracks: list[Track]) -> list[np.ndarray]:
        ...
