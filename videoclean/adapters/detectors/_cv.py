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


# Near-full-frame hits. Thin caption bars are ~0.05; tall lower-thirds need ~0.45.
MAX_BOX_AREA = 0.45
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


def _box_point_grid(box: tuple[int, int, int, int], grid: int = 3) -> np.ndarray:
    x1, y1, x2, y2 = box
    xs = np.linspace(x1 + 1, max(x1 + 1, x2 - 2), grid)
    ys = np.linspace(y1 + 1, max(y1 + 1, y2 - 2), grid)
    pts = np.array([[x, y] for y in ys for x in xs], dtype=np.float32).reshape(-1, 1, 2)
    return pts


def _points_to_box(
    pts: np.ndarray,
    status: np.ndarray,
    fallback: tuple[int, int, int, int],
    frame_shape: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    good = pts[status.reshape(-1) == 1].reshape(-1, 2)
    if len(good) < max(3, pts.shape[0] // 3):
        return None
    h, w = frame_shape
    x1 = int(np.clip(np.min(good[:, 0]), 0, w - 1))
    y1 = int(np.clip(np.min(good[:, 1]), 0, h - 1))
    x2 = int(np.clip(np.max(good[:, 0]) + 1, 1, w))
    y2 = int(np.clip(np.max(good[:, 1]) + 1, 1, h))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    # Reject huge jumps relative to seed size (lost track).
    sx1, sy1, sx2, sy2 = fallback
    seed_area = max(1, (sx2 - sx1) * (sy2 - sy1))
    area = (x2 - x1) * (y2 - y1)
    if area > seed_area * 4 or area < seed_area * 0.15:
        return None
    return x1, y1, x2, y2


def fill_boxes_optical_flow(
    frames: list[np.ndarray],
    boxes: list[tuple[int, int, int, int] | None],
) -> list[tuple[int, int, int, int] | None]:
    """Fill None slots by Lucas–Kanade flow from nearest known boxes; keep anchors."""
    if not frames or not boxes:
        return boxes
    out = list(boxes)
    n = len(frames)
    lk = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    known = [i for i, b in enumerate(out) if b is not None]
    if not known:
        return out

    def step(i_from: int, i_to: int, box: tuple[int, int, int, int]) -> tuple[int, int, int, int] | None:
        prev = cv2.cvtColor(frames[i_from], cv2.COLOR_BGR2GRAY)
        nxt = cv2.cvtColor(frames[i_to], cv2.COLOR_BGR2GRAY)
        pts = _box_point_grid(box)
        nxt_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev, nxt, pts, None, **lk)
        if nxt_pts is None or status is None:
            return None
        return _points_to_box(nxt_pts, status, box, frames[i_to].shape[:2])

    # Forward from each anchor into following holes.
    for start in known:
        box = out[start]
        assert box is not None
        cur = box
        for i in range(start + 1, n):
            if out[i] is not None:
                break
            nxt = step(i - 1, i, cur)
            if nxt is None:
                break
            out[i] = nxt
            cur = nxt

    # Backward from each anchor into preceding holes.
    for start in reversed(known):
        box = out[start]
        assert box is not None
        cur = box
        for i in range(start - 1, -1, -1):
            if out[i] is not None:
                break
            nxt = step(i + 1, i, cur)
            if nxt is None:
                break
            out[i] = nxt
            cur = nxt

    return out


def fill_boxes_csrt(
    frames: list[np.ndarray],
    boxes: list[tuple[int, int, int, int] | None],
) -> list[tuple[int, int, int, int] | None]:
    """Fill remaining None slots with OpenCV CSRT (or KCF) from nearest anchors."""
    if not frames or not boxes:
        return boxes
    create = _tracker_factory()
    if create is None:
        return boxes
    out = list(boxes)
    n = len(frames)
    known = [i for i, b in enumerate(out) if b is not None]
    if not known:
        return out

    def _run(direction: int) -> None:
        ordered = known if direction > 0 else list(reversed(known))
        for start in ordered:
            box = out[start]
            assert box is not None
            tracker = create()
            x1, y1, x2, y2 = box
            ok = tracker.init(frames[start], (float(x1), float(y1), float(x2 - x1), float(y2 - y1)))
            if not ok:
                continue
            i = start + direction
            while 0 <= i < n:
                if out[i] is not None:
                    break
                ok, rect = tracker.update(frames[i])
                if not ok:
                    break
                rx, ry, rw, rh = rect
                h, w = frames[i].shape[:2]
                nb = (
                    int(np.clip(rx, 0, w - 1)),
                    int(np.clip(ry, 0, h - 1)),
                    int(np.clip(rx + rw, 1, w)),
                    int(np.clip(ry + rh, 1, h)),
                )
                if nb[2] - nb[0] < 4 or nb[3] - nb[1] < 4:
                    break
                # Reject wild size jumps vs seed.
                seed_area = max(1, (x2 - x1) * (y2 - y1))
                area = (nb[2] - nb[0]) * (nb[3] - nb[1])
                if area > seed_area * 4 or area < seed_area * 0.15:
                    break
                out[i] = nb
                i += direction

    _run(+1)
    _run(-1)
    return out


def _tracker_factory():
    for path in (
        lambda: cv2.TrackerCSRT_create,
        lambda: cv2.legacy.TrackerCSRT_create,
        lambda: cv2.TrackerKCF_create,
        lambda: cv2.legacy.TrackerKCF_create,
    ):
        try:
            fn = path()
            if callable(fn):
                return fn
        except Exception:  # noqa: BLE001
            continue
    return None


def track_across_frames(
    frames: list[np.ndarray],
    boxes: list[tuple[int, int, int, int] | None],
    template_bgr: np.ndarray | None = None,
    min_score: float = 0.55,
) -> list[tuple[int, int, int, int] | None]:
    """Optical-flow → CSRT/KCF → template match for remaining holes."""
    filled = fill_boxes_optical_flow(frames, boxes)
    if any(b is None for b in filled):
        filled = fill_boxes_csrt(frames, filled)
    if template_bgr is None or not any(b is None for b in filled):
        return filled
    templ = match_template(frames, template_bgr, min_score=min_score)
    return [b if b is not None else t for b, t in zip(filled, templ)]
