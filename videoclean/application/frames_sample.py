from __future__ import annotations


def default_detector_keyframes(n_frames: int, fps: float | None = None) -> int:
    """Auto keyframe count from clip length: ~2/s, clamped to [8, 24].

    Longer / dynamic clips need more than the old fixed 8. Profiles and an
    explicit detector_keyframes override still win when set.
    """
    if n_frames <= 0:
        return 8
    rate = float(fps) if fps and fps > 0 else 30.0
    duration_s = n_frames / rate
    return max(8, min(24, int(round(duration_s * 2))))


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
