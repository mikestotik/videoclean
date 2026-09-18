"""Parallel framewise inpaint and chunked video-aware inpaint with overlap blend."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import numpy as np

from videoclean.application.verify_quality import blend_overlap


def resolve_workers(requested: int, *, device: str, video_aware: bool) -> int:
    """Pick worker count. GPU video models stay serial unless explicitly >1."""
    if video_aware:
        return 1
    if requested and requested > 0:
        return max(1, int(requested))
    cpu_n = os.cpu_count() or 2
    if (device or "cpu").strip().lower() == "cpu":
        return max(1, min(8, cpu_n))
    # CUDA/MPS: one worker avoids VRAM fights for neural framewise models.
    return 1


def inpaint_frames(
    inpainter,
    frames: list[np.ndarray],
    masks: list[np.ndarray],
    *,
    workers: int = 1,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[np.ndarray]:
    """Inpaint every frame; use threads when workers>1 (OpenCV/LaMa release GIL)."""
    n = len(frames)
    if n == 0:
        return []
    workers = max(1, int(workers))
    if workers == 1 or n == 1:
        out: list[np.ndarray] = []
        for i, (frame, mask) in enumerate(zip(frames, masks), start=1):
            out.append(inpainter.inpaint(frame, mask))
            if on_progress:
                on_progress(i, n)
        return out

    # Materialize — LazyFrames cache is not thread-safe.
    frame_list = [frames[i] for i in range(n)]
    mask_list = [masks[i] for i in range(n)]
    out = [None] * n  # type: ignore[list-item]

    def _one(i: int) -> tuple[int, np.ndarray]:
        return i, inpainter.inpaint(frame_list[i], mask_list[i])

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, frame in pool.map(_one, range(n)):
            out[i] = frame
            done += 1
            if on_progress:
                on_progress(done, n)
    return out  # type: ignore[return-value]


def inpaint_clip_chunked(
    inpainter,
    frames: list[np.ndarray],
    masks: list[np.ndarray],
    *,
    chunk_len: int,
    overlap: int,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[np.ndarray]:
    """Run video-aware inpaint in overlapping chunks and blend seams.

    For short clips (≤ chunk_len) delegates to a single inpaint_clip call.
    """
    n = len(frames)
    if n == 0:
        return []
    chunk_len = max(8, int(chunk_len))
    overlap = max(0, min(int(overlap), chunk_len // 2))
    if n <= chunk_len:
        out = inpainter.inpaint_clip(frames, masks)
        if on_progress:
            on_progress(n, n)
        return out

    result: list[np.ndarray | None] = [None] * n
    step = max(1, chunk_len - overlap)
    starts = list(range(0, n, step))
    # Ensure last chunk covers the end.
    if starts[-1] + chunk_len < n:
        starts.append(max(0, n - chunk_len))
    # Dedupe while preserving order.
    seen: set[int] = set()
    uniq_starts: list[int] = []
    for s in starts:
        if s not in seen:
            seen.add(s)
            uniq_starts.append(s)

    completed = 0
    for s in uniq_starts:
        e = min(n, s + chunk_len)
        patch = inpainter.inpaint_clip(frames[s:e], masks[s:e])
        if result[s] is None:
            for i, frame in enumerate(patch):
                result[s + i] = frame
        else:
            # Blend into already-written region.
            base_slice = [result[s + i] for i in range(len(patch))]
            # Use a temporary list for blend_overlap API.
            tmp = list(base_slice)
            blend_overlap(tmp, patch, 0, overlap=overlap)
            for i, frame in enumerate(tmp):
                result[s + i] = frame
        completed = max(completed, e)
        if on_progress:
            on_progress(min(completed, n), n)

    # Fill any holes (should not happen) with originals.
    for i, frame in enumerate(result):
        if frame is None:
            result[i] = frames[i]
    return result  # type: ignore[return-value]


def re_inpaint_ranges(
    inpainter,
    originals: list[np.ndarray],
    masks: list[np.ndarray],
    cleaned: list[np.ndarray],
    ranges: list[tuple[int, int]],
    *,
    video_aware: bool,
    chunk_overlap: int,
    workers: int,
) -> list[np.ndarray]:
    """Re-inpaint only dirty ranges; leave other frames untouched."""
    out = list(cleaned)
    if not ranges:
        return out
    for start, end in ranges:
        if start >= end:
            continue
        if video_aware:
            patch = inpainter.inpaint_clip(originals[start:end], masks[start:end])
            blend_overlap(out, patch, start, overlap=chunk_overlap)
        else:
            patch = inpaint_frames(
                inpainter,
                originals[start:end],
                masks[start:end],
                workers=workers,
            )
            for i, frame in enumerate(patch):
                out[start + i] = frame
    return out
