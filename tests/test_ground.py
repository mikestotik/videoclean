from videoclean.application.select import select_tracks
from videoclean.domain.intent import Intent, Target
from videoclean.domain.tracks import Track


def test_detector_tracks_are_required_even_if_parser_had_a_box():
    # Parser boxes are not a location source. Select still needs detector tracks.
    intent = Intent(
        targets=[Target(kind="object", query="mug", box=(0.0, 0.0, 1.0, 1.0))],
    )
    det = Track(
        track_id=7,
        label="mug",
        boxes=[(10, 10, 40, 40)],
        scores=[1.0],
        notes=["open-vocab"],
    )
    chosen = select_tracks([det], intent, width=100, height=100)
    assert [t.track_id for t in chosen] == [7]
    assert select_tracks([], intent, width=100, height=100) == []
