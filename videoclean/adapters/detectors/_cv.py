from __future__ import annotations

import cv2
import numpy as np


def resize_max(image: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale == 1.0:
        return image, 1.0
    small = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return small, scale


# Full-frame / near-full-frame hits. A width-spanning caption bar is ~0.05, not 0.25.
MAX_BOX_AREA = 0.25
# Template-matching a huge crop locks onto the background.
MAX_TEMPLATE_AREA = 0.12


def box_area_frac(xyxy: tuple[int, int, int, int], width: int, height: int) -> float:
    x1, y1, x2, y2 = xyxy
    den = max(1, width) * max(1, height)
    return max(0, x2 - x1) * max(0, y2 - y1) / den


def keep_detection_box(
    xyxy: tuple[int, int, int, int],
    width: int,
    height: int,
    max_area: float = MAX_BOX_AREA,
) -> bool:
    x1, y1, x2, y2 = xyxy
    if x2 - x1 < 4 or y2 - y1 < 4:
        return False
    return box_area_frac(xyxy, width, height) <= max_area


def sample_indices(n: int, limit: int) -> list[int]:
    if n <= limit:
        return list(range(n))
    return sorted({int(round(i * (n - 1) / (limit - 1))) for i in range(limit)})


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    ub = max(0, bx2 - bx1) * max(0, by2 - by1)
    den = ua + ub - inter
    return inter / den if den else 0.0


def mean_iou(a: list[tuple[int, int, int, int] | None], b: list[tuple[int, int, int, int] | None]) -> float:
    vals = [iou(x, y) for x, y in zip(a, b) if x and y]
    return float(np.mean(vals)) if vals else 0.0


def nms(hits: list, iou_thr: float = 0.3) -> list:
    """Greedy score-ordered NMS over BoxHit-like objects with .score/.xyxy."""
    ordered = sorted(hits, key=lambda h: -h.score)
    kept: list = []
    for hit in ordered:
        if any(iou(hit.xyxy, other.xyxy) > iou_thr for other in kept):
            continue
        kept.append(hit)
    return kept


def match_template(
    frames: list[np.ndarray],
    template_bgr: np.ndarray,
    min_score: float = 0.62,
) -> list[tuple[int, int, int, int] | None]:
    templ_s, _ = resize_max(template_bgr, 96)
    gray_t = cv2.cvtColor(templ_s, cv2.COLOR_BGR2GRAY)
    if gray_t.shape[0] < 8 or gray_t.shape[1] < 8:
        return [None] * len(frames)
    hits: list[tuple[int, int, int, int] | None] = []
    for frame in frames:
        small, scale = resize_max(frame, 640)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if gray.shape[0] <= gray_t.shape[0] or gray.shape[1] <= gray_t.shape[1]:
            hits.append(None)
            continue
        res = cv2.matchTemplate(gray, gray_t, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val < min_score:
            hits.append(None)
            continue
        x1 = int(max_loc[0] / scale)
        y1 = int(max_loc[1] / scale)
        x2 = int((max_loc[0] + gray_t.shape[1]) / scale)
        y2 = int((max_loc[1] + gray_t.shape[0]) / scale)
        hits.append((x1, y1, x2, y2))
    return hits
