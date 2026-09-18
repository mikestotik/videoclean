"""Profiles, verify residual helpers, and inpaint runtime."""

from __future__ import annotations

import numpy as np

from videoclean.adapters.detectors._cv import fill_boxes_optical_flow, track_across_frames
from videoclean.adapters.segmenters.sam2_video import _anchor_boxes
from videoclean.application.inpaint_runtime import (
    inpaint_frames,
    resolve_workers,
    re_inpaint_ranges,
)
from videoclean.application.profiles import apply_profile, profile_defaults
from videoclean.application.verify_quality import (
    dirty_ranges,
    masks_grew,
    residual_unchanged_mask,
)
from videoclean.domain.tracks import Track


def test_quality_profile_prefers_propainter_on_cuda():
    d = profile_defaults("quality", "cuda")
    assert d["inpainter"] == "propainter"
    assert d["segmenter"] == "sam2-video"
    assert d["verify_max_passes"] == 2


def test_quality_profile_falls_back_on_cpu():
    d = profile_defaults("quality", "cpu")
    assert d["inpainter"] == "lama"
    assert d["segmenter"] == "sam2"


def test_apply_profile_overwrites_owned_keys():
    out = apply_profile(
        {"profile": "fast", "device": "cpu", "inpainter": "propainter", "verify": True}
    )
    assert out["inpainter"] == "opencv-telea"
    assert out["verify"] is False
    assert out["profile"] == "fast"


def test_residual_flags_unchanged_region():
    h, w = 40, 40
    original = np.zeros((h, w, 3), dtype=np.uint8)
    original[10:20, 10:20] = (0, 0, 255)
    cleaned = original.copy()  # failed inpaint — still red
    prior = np.zeros((h, w), dtype=np.uint8)
    prior[10:20, 10:20] = 255
    residual = residual_unchanged_mask(original, cleaned, prior, max_delta=14, dilate_px=0, min_area=4)
    assert np.count_nonzero(residual) >= 50


def test_dirty_ranges_merge_with_pad():
    flags = [False] * 20
    flags[3] = flags[4] = flags[12] = True
    ranges = dirty_ranges(flags, pad=2)
    assert ranges[0] == (1, 7)
    assert ranges[1][0] == 10


def test_masks_grew():
    a = [np.zeros((8, 8), dtype=np.uint8)]
    b = [np.zeros((8, 8), dtype=np.uint8)]
    b[0][2:6, 2:6] = 255
    assert masks_grew(a, b) == [True]
    assert masks_grew(b, a) == [False]


def test_resolve_workers_auto():
    assert resolve_workers(0, device="cuda", video_aware=True) == 1
    assert resolve_workers(0, device="cuda", video_aware=False) == 1
    assert resolve_workers(4, device="cpu", video_aware=False) == 4
    assert resolve_workers(0, device="cpu", video_aware=False) >= 1


def test_inpaint_frames_parallel_telea():
    from videoclean.adapters.inpainters.opencv_telea import OpenCvTeleaInpainter

    inp = OpenCvTeleaInpainter(radius=3)
    frames = [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(6)]
    for f in frames:
        f[8:16, 8:16] = 200
    masks = [np.zeros((32, 32), dtype=np.uint8) for _ in range(6)]
    for m in masks:
        m[8:16, 8:16] = 255
    out = inpaint_frames(inp, frames, masks, workers=3)
    assert len(out) == 6
    assert out[0].shape == frames[0].shape


def test_re_inpaint_ranges_only_dirty():
    from videoclean.adapters.inpainters.opencv_telea import OpenCvTeleaInpainter

    inp = OpenCvTeleaInpainter(radius=3)
    originals = [np.full((16, 16, 3), 40, dtype=np.uint8) for _ in range(5)]
    for o in originals:
        o[4:12, 4:12] = 220
    masks = [np.zeros((16, 16), dtype=np.uint8) for _ in range(5)]
    masks[2][4:12, 4:12] = 255
    cleaned = [o.copy() for o in originals]
    out = re_inpaint_ranges(
        inp,
        originals,
        masks,
        cleaned,
        [(2, 3)],
        video_aware=False,
        chunk_overlap=0,
        workers=1,
    )
    assert not np.array_equal(out[2], cleaned[2])
    assert np.array_equal(out[0], cleaned[0])


def test_optical_flow_fills_gap():
    frames = []
    boxes: list[tuple[int, int, int, int] | None] = []
    for i in range(5):
        f = np.zeros((64, 64, 3), dtype=np.uint8)
        x = 10 + i * 3
        f[20:36, x : x + 12] = (0, 255, 0)
        frames.append(f)
        boxes.append((x, 20, x + 12, 36) if i in (0, 4) else None)
    filled = fill_boxes_optical_flow(frames, boxes)
    assert filled[0] is not None and filled[4] is not None
    # Middle frames should get a box from flow (best-effort; allow CSRT/template path too).
    tracked = track_across_frames(frames, boxes, template_bgr=frames[0][20:36, 10:22], min_score=0.4)
    assert sum(1 for b in tracked if b is not None) >= 3


def test_anchor_boxes_samples_multiple():
    boxes = [None] * 20
    for i in (0, 5, 10, 15, 19):
        boxes[i] = (10, 10, 30, 30)
    tr = Track(track_id=1, label="logo", boxes=boxes, scores=[1.0] * 20)
    anchors = _anchor_boxes(tr, 100, 100, max_anchors=3)
    assert len(anchors) == 3
    assert anchors[0][0] == 0
    assert anchors[-1][0] == 19
