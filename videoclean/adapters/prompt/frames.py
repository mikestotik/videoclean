from __future__ import annotations

import cv2
import numpy as np

from videoclean.application.frames_sample import sample_frame_indices

__all__ = ["sample_frame_indices", "bgr_to_jpeg"]


def bgr_to_jpeg(frame: np.ndarray, *, max_side: int = 768, quality: int = 85) -> bytes:
    """Downscale long side and encode JPEG for vision LLM payloads."""
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("expected HxWx3 BGR frame")
    h, w = frame.shape[:2]
    scale = 1.0
    long_side = max(h, w)
    if long_side > max_side:
        scale = max_side / float(long_side)
    if scale < 1.0:
        frame = cv2.resize(
            frame,
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise ValueError("JPEG encode failed")
    return buf.tobytes()
