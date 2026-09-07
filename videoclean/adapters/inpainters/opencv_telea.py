from __future__ import annotations

import cv2
import numpy as np


class OpenCvTeleaInpainter:
    """Per-frame TELEA. Always CPU. Fine for short clips; smears on large/moving holes."""

    name = "opencv-telea"
    device_note = "CPU only (OpenCV). --device does not apply."
    video_aware = False

    def __init__(self, radius: int = 9) -> None:
        self.radius = radius

    def status(self) -> str:
        return "ready (CPU)"

    def inpaint_clip(self, frames: list[np.ndarray], masks: list[np.ndarray]) -> list[np.ndarray]:
        return [self.inpaint(frame, mask) for frame, mask in zip(frames, masks)]

    def inpaint(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        if not np.any(mask):
            return frame
        binary = np.where(mask > 0, 255, 0).astype(np.uint8)
        return cv2.inpaint(frame, binary, self.radius, cv2.INPAINT_TELEA)
