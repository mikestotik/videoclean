import numpy as np

from videoclean.adapters.inpainters.opencv_telea import OpenCvTeleaInpainter


def test_telea_does_not_repaint_pixels_far_from_the_mask():
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    frame[:] = (70, 170, 70)  # BGR green, would match a cyan/green HSV hack
    mask = np.zeros((80, 80), dtype=np.uint8)
    mask[30:50, 30:50] = 255
    out = OpenCvTeleaInpainter().inpaint(frame, mask)
    # corners are far from the hole; they must stay the source pixels
    assert np.array_equal(out[0, 0], frame[0, 0])
    assert np.array_equal(out[0, 79], frame[0, 79])
    assert np.array_equal(out[79, 0], frame[79, 0])
    assert np.array_equal(out[79, 79], frame[79, 79])
