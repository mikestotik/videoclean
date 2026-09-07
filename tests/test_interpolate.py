from videoclean.domain.tracks import Track, interpolate_gaps


def test_interpolate_fills_interior_only():
    tr = Track(
        track_id=1,
        label="text",
        boxes=[None, (10, 10, 20, 20), None, (30, 10, 40, 20), None],
        scores=[0, 1, 0, 1, 0],
    )
    out = interpolate_gaps(tr)
    assert out.boxes[0] is None
    assert out.boxes[2] is not None
    assert out.boxes[4] is None
