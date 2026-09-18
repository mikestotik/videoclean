from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import numpy as np

from videoclean.domain.tracks import Track

# current, total, detail — heartbeat while segmenting frames
OnProgress = Callable[[int, int, str], None]


class Segmenter(Protocol):
    """Turns tracks into per-frame masks."""

    name: str

    def masks(
        self,
        frames: list[np.ndarray],
        tracks: list[Track],
        on_progress: OnProgress | None = None,
    ) -> list[np.ndarray]:
        ...
