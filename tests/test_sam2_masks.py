import numpy as np
import torch

from videoclean.adapters.segmenters.sam2 import _or_masks


def test_or_masks_keeps_every_box_mask():
    h, w = 16, 32
    planes = []
    for x0 in (2, 8, 20):
        p = np.zeros((1, h, w), dtype=np.float32)
        p[0, 2:5, x0 : x0 + 3] = 1.0
        planes.append(p)
    arr = np.stack(planes)  # (3, 1, H, W) — post_process_masks output shape
    out = _or_masks(torch.from_numpy(arr), (h, w))
    assert out[3, 3] == 255
    assert out[3, 9] == 255
    assert out[3, 21] == 255


def test_or_masks_single_plane():
    h, w = 8, 8
    one = np.zeros((1, h, w), dtype=np.float32)
    one[0, 1:3, 1:3] = 1.0
    out = _or_masks(torch.from_numpy(one), (h, w))
    assert out[1, 1] == 255
    assert out.shape == (h, w)


def test_or_masks_2d_input():
    h, w = 8, 8
    two = np.zeros((h, w), dtype=np.float32)
    two[5:7, 5:7] = 1.0
    out = _or_masks(torch.from_numpy(two), (h, w))
    assert out[5, 5] == 255


def test_or_masks_unsized_planes_are_resized():
    h, w = 8, 8
    small = np.full((1, 4, 4), 255, dtype=np.float32)  # off-size plane → resize
    out = _or_masks(torch.from_numpy(small), (h, w))
    assert out.shape == (h, w)
    assert out.max() == 255
