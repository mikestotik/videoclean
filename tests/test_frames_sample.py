from videoclean.application.frames_sample import default_detector_keyframes, sample_frame_indices


def test_default_detector_keyframes_scales_with_duration():
    # 30 fps × 5 s → ~10 keyframes
    assert default_detector_keyframes(150, fps=30) == 10
    # short clip floors at 8
    assert default_detector_keyframes(30, fps=30) == 8
    # long clip caps at 24
    assert default_detector_keyframes(30 * 60, fps=30) == 24


def test_sample_frame_indices_caps():
    assert sample_frame_indices(100, stride=10, max_frames=3) == [0, 40, 90]
