"""Post-inpaint leftover detection and local re-inpaint helpers."""

from __future__ import annotations

import cv2
import numpy as np


def residual_unchanged_mask(
    original: np.ndarray,
    cleaned: np.ndarray,
    prior_mask: np.ndarray,
    *,
    max_delta: int = 14,
    dilate_px: int = 6,
    min_area: int = 24,
) -> np.ndarray:
    """Pixels inside (dilated) prior mask that barely changed → likely leftover."""
    h, w = original.shape[:2]
    out = np.zeros((h, w), dtype=np.uint8)
    if prior_mask is None or not np.any(prior_mask):
        return out
    prior = prior_mask
    if prior.dtype != np.uint8:
        prior = prior.astype(np.uint8)
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))
        prior = cv2.dilate(prior, k)
    diff = cv2.absdiff(original, cleaned)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) if diff.ndim == 3 else diff
    unchanged = (gray <= int(max_delta)) & (prior > 0)
    out[unchanged] = 255
    if min_area > 0 and np.any(out):
        n, labels, stats, _ = cv2.connectedComponentsWithStats(out, connectivity=8)
        cleaned_cc = np.zeros_like(out)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                cleaned_cc[labels == i] = 255
        out = cleaned_cc
    return out


def or_masks(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a is None or not np.any(a):
        return b.copy() if b is not None else a
    if b is None or not np.any(b):
        return a.copy()
    return np.maximum(a, b)


def masks_grew(old: list[np.ndarray], new: list[np.ndarray], *, min_new_px: int = 8) -> list[bool]:
    """True where new has enough pixels that old did not."""
    flags: list[bool] = []
    for a, b in zip(old, new):
        if b is None or not np.any(b):
            flags.append(False)
            continue
        if a is None or not np.any(a):
            flags.append(int(np.count_nonzero(b)) >= min_new_px)
            continue
        added = (b > 0) & (a == 0)
        flags.append(int(np.count_nonzero(added)) >= min_new_px)
    return flags


def dirty_ranges(flags: list[bool], *, pad: int = 8) -> list[tuple[int, int]]:
    """Merge True indices into inclusive [start, end) ranges with padding."""
    n = len(flags)
    if n == 0:
        return []
    hits = [i for i, f in enumerate(flags) if f]
    if not hits:
        return []
    ranges: list[tuple[int, int]] = []
    start = max(0, hits[0] - pad)
    prev = hits[0]
    for i in hits[1:]:
        if i - prev <= pad * 2 + 1:
            prev = i
            continue
        end = min(n, prev + pad + 1)
        ranges.append((start, end))
        start = max(0, i - pad)
        prev = i
    ranges.append((start, min(n, prev + pad + 1)))
    return _merge_ranges(ranges)


def _merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not ranges:
        return []
    ordered = sorted(ranges)
    out = [ordered[0]]
    for a, b in ordered[1:]:
        pa, pb = out[-1]
        if a <= pb:
            out[-1] = (pa, max(pb, b))
        else:
            out.append((a, b))
    return out


def blend_overlap(
    base: list[np.ndarray],
    patch: list[np.ndarray],
    start: int,
    *,
    overlap: int,
) -> None:
    """Write patch into base[start:start+len(patch)] with linear edge blend."""
    n = len(patch)
    if n == 0:
        return
    ov = max(0, min(int(overlap), n // 2))
    for i, frame in enumerate(patch):
        idx = start + i
        if idx < 0 or idx >= len(base):
            continue
        if ov <= 0 or (i >= ov and i < n - ov):
            base[idx] = frame
            continue
        if i < ov:
            # fade in from base → patch
            alpha = (i + 1) / (ov + 1)
        else:
            # fade out patch → base toward the end
            alpha = (n - i) / (ov + 1)
        alpha = float(np.clip(alpha, 0.0, 1.0))
        base[idx] = cv2.addWeighted(frame, alpha, base[idx], 1.0 - alpha, 0)
