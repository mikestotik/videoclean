import numpy as np

from videoclean.adapters.inpainters.propainter import _pad_pair


def test_one_frame_is_repeated_so_raft_has_a_pair():
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    frames, masks, keep = _pad_pair([frame], [mask])
    assert keep == 1
    assert len(frames) == 2 and len(masks) == 2
    assert frames[0] is frame and frames[1] is frame


def test_two_frames_stay_untouched():
    frames = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(2)]
    masks = [np.zeros((4, 4), dtype=np.uint8) for _ in range(2)]
    out_f, out_m, keep = _pad_pair(frames, masks)
    assert keep == 2
    assert out_f is frames and out_m is masks
