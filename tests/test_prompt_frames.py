from videoclean.adapters.prompt.frames import sample_frame_indices


def test_sample_stride_four():
    assert sample_frame_indices(27, stride=4, max_frames=8) == [0, 4, 8, 12, 16, 20, 24]


def test_sample_stride_zero_means_no_vision():
    assert sample_frame_indices(27, stride=0, max_frames=8) == []


def test_sample_caps_and_spreads():
    idxs = sample_frame_indices(100, stride=2, max_frames=5)
    assert len(idxs) == 5
    assert idxs[0] == 0
    assert idxs[-1] == 98
    assert idxs == sorted(set(idxs))
