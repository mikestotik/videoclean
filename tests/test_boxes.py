from videoclean.adapters.detectors._cv import box_area_frac, keep_detection_box


def test_full_frame_box_is_rejected():
    assert keep_detection_box((6, 19, 1910, 1070), 1920, 1080) is False


def test_caption_sized_box_is_kept():
    # bottom banner on 1920x1080
    assert keep_detection_box((401, 902, 1512, 996), 1920, 1080) is True


def test_hud_sized_box_area():
    frac = box_area_frac((809, 463, 1003, 803), 1920, 1080)
    assert 0.02 < frac < 0.05
    assert keep_detection_box((809, 463, 1003, 803), 1920, 1080) is True
