from videoclean.application.select import select_tracks
from videoclean.domain.intent import Intent, Target
from videoclean.domain.tracks import Track


def _track(tid: int, label: str, box: tuple[int, int, int, int], motion: str = "static") -> Track:
    return Track(track_id=tid, label=label, boxes=[box], scores=[1.0], motion=motion)


def test_select_keeps_tracks_whose_label_matches_the_query():
    mug = _track(1, "red mug", (10, 10, 40, 40))
    stain = _track(2, "black stain", (80, 10, 120, 50))
    intent = Intent(targets=[Target(kind="object", query="red mug")])
    chosen = select_tracks([mug, stain], intent, width=200, height=100)
    assert [t.track_id for t in chosen] == [1]


def test_select_does_not_invent_matches_from_kind():
    blob = _track(0, "channel logo", (10, 10, 40, 40))
    intent = Intent(targets=[Target(kind="text_overlay", query="надписи")])
    assert select_tracks([blob], intent, width=200, height=100) == []


def test_two_targets_take_their_own_queries():
    mug = _track(1, "mug", (10, 10, 40, 40))
    logo = _track(2, "channel logo", (80, 80, 120, 99))
    intent = Intent(
        targets=[
            Target(kind="object", query="mug"),
            Target(kind="watermark", query="channel logo"),
        ]
    )
    chosen = select_tracks([mug, logo], intent, width=200, height=100)
    assert {t.track_id for t in chosen} == {1, 2}


def test_where_keeps_the_named_region():
    top = _track(1, "on-screen text", (10, 5, 80, 20))
    bottom = _track(2, "on-screen text", (10, 80, 80, 99))
    intent = Intent(targets=[Target(kind="text_overlay", query="on-screen text", where="bottom")])
    chosen = select_tracks([top, bottom], intent, width=100, height=100)
    assert [t.track_id for t in chosen] == [2]


def test_truncated_label_does_not_match_the_full_query():
    hud = _track(0, "надпи", (80, 40, 120, 80))
    intent = Intent(targets=[Target(kind="text_overlay", query="надписи")])
    assert select_tracks([hud], intent, width=200, height=100) == []


def test_object_word_still_matches_a_longer_query():
    mug = _track(1, "mug", (10, 10, 40, 40))
    intent = Intent(targets=[Target(kind="object", query="red mug")])
    chosen = select_tracks([mug], intent, width=200, height=100)
    assert [t.track_id for t in chosen] == [1]


def test_generic_object_label_matches_text_overlay():
    top = _track(1, "object", (10, 5, 80, 20))
    bottom = _track(2, "object", (10, 80, 80, 99))
    intent = Intent(targets=[Target(kind="text_overlay", query="text", where="top")])
    chosen = select_tracks([top, bottom], intent, width=100, height=100)
    assert [t.track_id for t in chosen] == [1]


def test_generic_object_label_does_not_match_physical_object_query():
    blob = _track(0, "object", (10, 10, 40, 40))
    intent = Intent(targets=[Target(kind="object", query="mug")])
    assert select_tracks([blob], intent, width=200, height=100) == []


def test_relax_drops_bad_where_when_nothing_matches():
    mid = _track(1, "text", (40, 40, 60, 60))
    intent = Intent(targets=[Target(kind="text_overlay", query="text", where="bottom-right")])
    assert select_tracks([mid], intent, width=100, height=100) == []
    chosen = select_tracks([mid], intent, width=100, height=100, relax=True)
    assert [t.track_id for t in chosen] == [1]


def test_ordinal_picks_third_from_the_left():
    left = _track(1, "red mug", (10, 10, 40, 40))
    mid = _track(2, "red mug", (80, 10, 110, 40))
    right = _track(3, "red mug", (200, 10, 230, 40))
    intent = Intent(
        targets=[Target(kind="object", query="red mug", ordinal=3, from_side="left")]
    )
    chosen = select_tracks([right, left, mid], intent, width=300, height=100)
    assert len(chosen) == 1
    assert chosen[0].track_id == 3


def _mk_track(tid: int, boxes: list, motion: str = "static") -> Track:
    return Track(
        track_id=tid,
        label="object",
        boxes=boxes,
        scores=[0.8] * len(boxes),
        motion=motion,
    )


def test_scoped_target_zeroes_boxes_outside_window():
    """Track boxes outside the target window must be dropped before interpolation."""
    n = 10
    # Track lives on frames 0..9; window covers only frames 5..9.
    boxes = [(10 * i, 10, 10 * i + 20, 30) for i in range(n)]
    tr = _mk_track(1, boxes)
    intent = Intent(
        targets=[Target(kind="object", query="object", frames=(5, 10))],
    )
    out = select_tracks([tr], intent, width=100, height=100)
    assert len(out) == 1
    kept = out[0]
    assert all(b is None for b in kept.boxes[:5])
    assert all(b is not None for b in kept.boxes[5:])


def test_scoped_target_none_means_whole_clip():
    n = 4
    boxes = [(10, 10, 20, 20)] * n
    tr = _mk_track(1, boxes)
    intent = Intent(targets=[Target(kind="object", query="object", frames=None)])
    out = select_tracks([tr], intent, width=100, height=100)
    assert out[0].boxes == boxes


def test_track_entirely_outside_window_is_dropped():
    tr = _mk_track(1, [(10, 10, 20, 20)] * 6)
    intent = Intent(targets=[Target(kind="object", query="object", frames=(7, 9))])
    out = select_tracks([tr], intent, width=100, height=100)
    assert out == []


def test_scoped_targets_from_different_windows_both_pass():
    n = 12
    a = _mk_track(1, [(10, 10, 20, 20)] * 6 + [None] * 6)  # frames 0..5
    b = _mk_track(2, [None] * 6 + [(30, 30, 50, 50)] * 6)  # frames 6..11
    intent = Intent(
        targets=[
            Target(kind="object", query="object", frames=(0, 6)),
            Target(kind="object", query="object", frames=(6, 12)),
        ]
    )
    out = select_tracks([a, b], intent, width=100, height=100)
    assert len(out) == 2
    assert all(x is not None for x in out[0].boxes[:6])
    assert all(x is not None for x in out[1].boxes[6:])
