"""tracks_from_json round-trips tracks_to_json and rejects garbage."""
import pytest

from videoclean.domain.tracks import Track, tracks_from_json, tracks_to_json


def test_round_trip():
    tr = Track(track_id=2, label="logo", boxes=[(1, 2, 30, 40), None, (2, 3, 31, 41)], scores=[1.0, 0.0, 1.0], motion="static", notes=["a"])
    data = tracks_to_json([tr])
    out = tracks_from_json(data)
    assert len(out) == 1
    t = out[0]
    assert t.track_id == 2
    assert t.label == "logo"
    assert t.boxes == [(1, 2, 30, 40), None, (2, 3, 31, 41)]
    assert t.motion == "static"
    assert t.notes == ["a"]


def test_skips_unusable_rows():
    data = [
        {"id": 0, "label": "ok", "boxes": [[1, 1, 5, 5]]},
        {"id": 1, "label": "no boxes", "boxes": []},
        {"id": 2, "label": "all nulls", "boxes": [None, None]},
        "not a dict",
        {"id": 3, "label": "bad coords", "boxes": [["x", "y", 5, 5]]},
    ]
    out = tracks_from_json(data)
    assert [t.label for t in out] == ["ok"]


def test_empty_raises_value_error():
    with pytest.raises(ValueError):
        tracks_from_json([])
    with pytest.raises(ValueError):
        tracks_from_json(None)
