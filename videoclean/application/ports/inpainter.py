from __future__ import annotations

from typing import Protocol

import numpy as np


class Inpainter(Protocol):
    """Fills masks. Per-frame adapters implement inpaint(); video adapters also inpaint_clip()."""

    name: str
    device_note: str
    video_aware: bool

    def inpaint(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray: ...

    def inpaint_clip(self, frames: list[np.ndarray], masks: list[np.ndarray]) -> list[np.ndarray]: ...
