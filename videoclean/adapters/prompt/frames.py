from __future__ import annotations

import cv2
import numpy as np


def sample_frame_indices(n_frames: int, stride: int, max_frames: int) -> list[int]:
    """Pick frame indices: 0, stride, 2*stride, ... then evenly cap to max_frames.

    stride <= 0 → no vision samples (text-only parse).
    """
    if n_frames <= 0 or stride <= 0 or max_frames <= 0:
        return []
    idxs = list(range(0, n_frames, stride))
    if not idxs:
        return [0]
    if len(idxs) <= max_frames:
        return idxs
    if max_frames == 1:
        return [idxs[0]]
    step = (len(idxs) - 1) / (max_frames - 1)
    picked = [idxs[int(round(i * step))] for i in range(max_frames)]
    # de-dupe while preserving order
    out: list[int] = []
    seen: set[int] = set()
    for i in picked:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


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
