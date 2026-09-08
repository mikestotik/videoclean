from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Detection:
    label: str
    coverage: float
    mask_kind: str
    notes: list[str] = field(default_factory=list)


@dataclass
class Track:
    track_id: int
    label: str
    boxes: list[tuple[int, int, int, int] | None]
    scores: list[float]
    motion: str = "static"
    part: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def n_frames(self) -> int:
        return len(self.boxes)

    @property
    def coverage(self) -> float:
        hit = sum(1 for b in self.boxes if b is not None)
        return hit / max(1, len(self.boxes))

    def observed_boxes(self) -> list[tuple[int, int, int, int]]:
        return [b for b in self.boxes if b is not None]


def box_center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def infer_motion(track: Track, width: int, height: int) -> str:
    pts = [box_center(b) for b in track.observed_boxes()]
    if len(pts) < 3:
        return "static"
    xs = np.array([p[0] for p in pts]) / max(1, width)
    ys = np.array([p[1] for p in pts]) / max(1, height)
    jitter = float(np.hypot(xs.std(), ys.std()))
    return "floating" if jitter > 0.025 else "static"


def interpolate_gaps(track: Track) -> Track:
    boxes = list(track.boxes)
    known = [i for i, b in enumerate(boxes) if b is not None]
    if len(known) < 2:
        return track
    for a, b in zip(known, known[1:]):
        if b == a + 1:
            continue
        ba, bb = boxes[a], boxes[b]
        assert ba is not None and bb is not None
        span = b - a
        for t in range(1, span):
            u = t / span
            boxes[a + t] = tuple(int(round(ba[k] + u * (bb[k] - ba[k]))) for k in range(4))  # type: ignore[misc]
    track.boxes = boxes
    return track


def apply_part(box: tuple[int, int, int, int], part: str | None) -> tuple[int, int, int, int]:
    if not part:
        return box
    x1, y1, x2, y2 = box
    mx, my = (x1 + x2) // 2, (y1 + y2) // 2
    if part == "left":
        return x1, y1, mx, y2
    if part == "right":
        return mx, y1, x2, y2
    if part == "top":
        return x1, y1, x2, my
    if part == "bottom":
        return x1, my, x2, y2
    if part == "icon":
        side = min(x2 - x1, y2 - y1)
        return x1, y1, x1 + side, y1 + side
    if part == "text":
        side = min(x2 - x1, y2 - y1)
        return x1 + side, y1, x2, y2
    return box


def tracks_to_json(tracks: list[Track]) -> list[dict]:
    out = []
    for tr in tracks:
        out.append(
            {
                "id": tr.track_id,
                "label": tr.label,
                "motion": tr.motion,
                "part": tr.part,
                "coverage": tr.coverage,
                "notes": tr.notes,
                "boxes": [list(b) if b else None for b in tr.boxes],
            }
        )
    return out


def tracks_from_json(data: list[dict] | None) -> list[Track]:
    """Inverse of tracks_to_json. Unusable rows are skipped; nothing left → ValueError."""
    out: list[Track] = []
    for item in data or []:
        if not isinstance(item, dict):
            continue
        boxes: list[tuple[int, int, int, int] | None] = []
        for b in item.get("boxes") or []:
            if not isinstance(b, (list, tuple)) or len(b) != 4:
                boxes.append(None)
                continue
            try:
                x1, y1, x2, y2 = (int(round(float(v))) for v in b)
            except (TypeError, ValueError):
                boxes.append(None)
                continue
            boxes.append((x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None)
        if not boxes or not any(b is not None for b in boxes):
            continue
        try:
            track_id = int(item.get("id") or len(out))
        except (TypeError, ValueError):
            track_id = len(out)
        out.append(
            Track(
                track_id=track_id,
                label=str(item.get("label") or f"track{len(out)}"),
                boxes=boxes,
                scores=[1.0] * len(boxes),
                motion=str(item.get("motion") or "static"),
                part=item.get("part"),
                notes=[str(n) for n in (item.get("notes") or [])],
            )
        )
    if not out:
        raise ValueError("tracks payload contains no usable tracks")
    return out
