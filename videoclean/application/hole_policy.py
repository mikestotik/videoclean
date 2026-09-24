"""Hole policy. Thresholds are engine constants, not sliders.

Coverage is mask pixels, not the box area. The box is only crop geometry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

SMALL_COVERAGE = 0.08
SMALL_SIDE = 0.35
FAST_SHIFT = 0.04
ALMOST_FULL_COVERAGE = 0.55
ALMOST_FULL_SIDE = 0.85
PROPAINTER_BYTES_AT_640 = 6 * 1024**3


@dataclass(frozen=True)
class RangePlan:
    start: int
    end: int
    policy: str
    mean_coverage: float
    fast_moving: bool
    almost_full: bool
    side: int
    batch: int
    feather_px: int
    context_px: int
    limited_by: str


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return float(ordered[len(ordered) // 2])


def mask_coverage(mask: np.ndarray) -> float:
    if mask.size == 0:
        return 0.0
    return float(np.count_nonzero(mask)) / float(mask.size)


def mask_box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def is_fast_moving(
    track_motions: list[str],
    union_centers: list[tuple[float, float] | None],
    diagonal: float,
) -> bool:
    if any(str(m) == "floating" for m in track_motions):
        return True
    shifts: list[float] = []
    prev: tuple[float, float] | None = None
    for center in union_centers:
        if center is None:
            prev = None
            continue
        if prev is not None:
            shifts.append(math.hypot(center[0] - prev[0], center[1] - prev[1]))
        prev = center
    if not shifts or diagonal <= 0:
        return False
    return _median(shifts) > FAST_SHIFT * diagonal


def crop_geometry(
    box: tuple[int, int, int, int] | None,
    frame_hw: tuple[int, int],
    *,
    inpaint_max_side: int | None,
    budget_bytes: int,
    family: str,
    neighbor_length: int = 10,
    subvideo_length: int = 80,
) -> tuple[int, int, int, str]:
    """Return side, context_px, feather_px, limited_by for one hole."""
    h, w = frame_hw
    if box is None:
        bw = bh = 8
    else:
        bw = max(1, box[2] - box[0])
        bh = max(1, box[3] - box[1])
    context = max(32, int(0.5 * max(bw, bh)))
    need = max(bw, bh) + 2 * context
    side = max(64, int(math.ceil(need / 8.0) * 8))
    frame_cap = max(8, (min(h, w) // 8) * 8)
    side = min(side, frame_cap)
    limited = "ceiling"
    if inpaint_max_side:
        cap = max(64, (int(inpaint_max_side) // 8) * 8)
        if side > cap:
            side = cap
            limited = "inpaint_max_side"
    if family == "propainter" and budget_bytes > 0:
        guard = 0
        while side > 64 and guard < 16:
            bytes_at = (
                PROPAINTER_BYTES_AT_640
                * (side / 640) ** 2
                * (max(1, neighbor_length) / 10)
                * math.sqrt(max(1, subvideo_length) / 80)
            )
            if bytes_at <= budget_bytes:
                break
            side = max(64, ((int(side * 0.75)) // 8) * 8)
            limited = "ceiling"
            guard += 1
    feather = min(16, side // 8)
    return side, context, feather, limited


def plan_holes(
    *,
    frame_hw: tuple[int, int],
    mask_coverage: list[float] | None = None,
    mask_boxes: list[tuple[int, int, int, int] | None] | None = None,
    track_motions: list[str] | None = None,
    union_centers: list[tuple[float, float] | None] | None = None,
    family: str = "lama",
    device: str = "cpu",
    budget_bytes: int = 0,
    inpaint_max_side: int | None = None,
    neighbor_length: int = 10,
    subvideo_length: int = 80,
    batch_cap: int | None = None,
    mask: np.ndarray | None = None,
    budget_side: int | None = None,
    budget_batch: int | None = None,
) -> list[RangePlan]:
    """One RangePlan per run of the same family and side.

    ``mask`` / ``budget_side`` / ``budget_batch`` are the single-frame test entry.
    """
    if mask is not None:
        mask_coverage = [mask_coverage_of(mask) if False else _cov(mask)]
        mask_boxes = [_box(mask)]
        if budget_side is not None:
            inpaint_max_side = budget_side
        if budget_batch is not None:
            batch_cap = budget_batch
    coverages = list(mask_coverage or [])
    boxes = list(mask_boxes or [None] * len(coverages))
    if len(boxes) < len(coverages):
        boxes.extend([None] * (len(coverages) - len(boxes)))
    n = len(coverages)
    if n == 0:
        return []
    h, w = frame_hw
    diag = math.hypot(h, w)
    motions = list(track_motions or [])
    centers = list(union_centers or [])
    if not centers:
        centers = []
        for box in boxes:
            if box is None:
                centers.append(None)
            else:
                centers.append(((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0))
    fast = is_fast_moving(motions, centers, diag)
    sides = [max(1, b[2] - b[0], b[3] - b[1]) if b else 0 for b in boxes]
    present = [s for s in sides if s > 0]
    median_side = _median([s / max(h, w) for s in present]) if present else 0.0
    mean_cov = float(sum(coverages) / n)
    union = _union_box([b for b in boxes if b is not None])
    almost = mean_cov >= ALMOST_FULL_COVERAGE
    if union is not None:
        uw = (union[2] - union[0]) / max(1, w)
        uh = (union[3] - union[1]) / max(1, h)
        if uw >= ALMOST_FULL_SIDE and uh >= ALMOST_FULL_SIDE:
            almost = True
    small = mean_cov < SMALL_COVERAGE and median_side < SMALL_SIDE and not fast
    use_pp = family == "propainter" and device == "cuda" and (not small or fast)
    policy = "propainter-crop" if use_pp else "lama-crop"
    geom_family = "propainter" if use_pp else "lama"
    per: list[tuple[int, int, int, str]] = []
    for box in boxes:
        per.append(
            crop_geometry(
                box,
                frame_hw,
                inpaint_max_side=inpaint_max_side,
                budget_bytes=budget_bytes,
                family=geom_family,
                neighbor_length=neighbor_length,
                subvideo_length=subvideo_length,
            )
        )
    plans: list[RangePlan] = []
    start = 0
    while start < n:
        side, context, feather, limited = per[start]
        end = start + 1
        while end < n and per[end][0] == side:
            end += 1
        length = end - start
        cap = length if not batch_cap or batch_cap <= 0 else min(length, int(batch_cap))
        if limited != "inpaint_max_side":
            limited = "frames" if cap == length else "ceiling"
        plans.append(
            RangePlan(
                start=start,
                end=end,
                policy=policy,
                mean_coverage=float(sum(coverages[start:end]) / length),
                fast_moving=fast,
                almost_full=almost,
                side=side,
                batch=max(1, cap),
                feather_px=feather,
                context_px=context,
                limited_by=limited,
            )
        )
        start = end
    return plans


def _cov(mask: np.ndarray) -> float:
    return mask_coverage(mask)


def _box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    return mask_box(mask)


def _union_box(
    boxes: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    if not boxes:
        return None
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def classify_lama_probe(half_error: BaseException | None, output_finite: bool) -> str:
    if half_error is not None or not output_finite:
        return "fp32"
    return "fp16"


def probe_lama_dtype(model=None, device: str = "cpu", *, raise_on_half: bool = False) -> str:
    """CUDA-only 64x64 batch-2 probe. Any exception or non-finite output locks fp32."""
    if raise_on_half or device != "cuda" or model is None:
        return "fp32"
    import torch

    try:
        half = model.half()
        image = torch.zeros(2, 3, 64, 64, device=device, dtype=torch.float16)
        mask = torch.zeros(2, 1, 64, 64, device=device, dtype=torch.float16)
        with torch.inference_mode():
            out = half(image, mask)
        finite = bool(torch.isfinite(out).all().item())
        kind = classify_lama_probe(None, finite)
    except Exception as exc:  # noqa: BLE001
        kind = classify_lama_probe(exc, False)
    if kind == "fp32":
        try:
            model.float()
        except Exception:  # noqa: BLE001
            pass
    return kind


def grow_batch(last: int, range_len: int, measured_delta: int, budget_bytes: int) -> tuple[str, int]:
    """Next batch step.

    Returns ("frames", last) when the range is already covered,
    ("try", nxt) when the doubled candidate should run,
    ("mid", mid) when the candidate does not fit and a midpoint should run,
    ("ceiling", last) when the midpoint is not larger than last.
    """
    if last >= range_len:
        return "frames", last
    nxt = min(last * 2, range_len)
    if nxt == last:
        return "frames", last
    if last > 0 and measured_delta > 0 and budget_bytes > 0:
        estimate = int(measured_delta * (nxt / last))
        if estimate > budget_bytes:
            mid = min((last + nxt) // 2, range_len)
            if mid > last:
                return "mid", mid
            return "ceiling", last
    return "try", nxt
