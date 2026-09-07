from __future__ import annotations

from videoclean.domain.intent import CROP_PARTS, Intent, Target
from videoclean.domain.tracks import Track, box_center, interpolate_gaps


def select_tracks(
    tracks: list[Track],
    intent: Intent,
    *,
    width: int,
    height: int,
) -> list[Track]:
    chosen: list[Track] = []
    used: set[int] = set()
    for target in intent.targets:
        picked = _matches(tracks, target, width, height, require_motion=True)
        if not picked and target.motion != "any":
            picked = _matches(tracks, target, width, height, require_motion=False)
        picked = _pick_ordinal(picked, target)
        for tr in picked:
            if tr.track_id in used:
                continue
            used.add(tr.track_id)
            tr = interpolate_gaps(tr)
            tr.part = target.part if target.part in CROP_PARTS else None
            tr.notes = list(tr.notes) + [f"motion={tr.motion}"]
            if target.where:
                tr.notes.append(f"where={target.where}")
            if target.ordinal:
                tr.notes.append(f"ordinal={target.ordinal}:{target.from_side or 'left'}")
            if tr.part:
                tr.notes.append(f"part={tr.part}")
            chosen.append(tr)
    return chosen


def _matches(
    tracks: list[Track],
    target: Target,
    width: int,
    height: int,
    *,
    require_motion: bool,
) -> list[Track]:
    out: list[Track] = []
    for tr in tracks:
        if require_motion and target.motion != "any" and tr.motion != target.motion:
            continue
        if not _query_ok(tr, target):
            continue
        if not _where_ok(tr, target.where, width, height):
            continue
        out.append(tr)
    return out


def _pick_ordinal(tracks: list[Track], target: Target) -> list[Track]:
    if not target.ordinal or not tracks:
        return tracks
    side = target.from_side if target.from_side in {"left", "right"} else "left"

    def _cx(tr: Track) -> float:
        boxes = tr.observed_boxes()
        if not boxes:
            return 0.0
        xs = [box_center(b)[0] for b in boxes]
        return sum(xs) / len(xs)

    ordered = sorted(tracks, key=_cx, reverse=(side == "right"))
    idx = target.ordinal - 1
    if idx < 0 or idx >= len(ordered):
        return []
    return [ordered[idx]]


def _query_ok(tr: Track, target: Target) -> bool:
    return _overlap((tr.label or "").casefold(), (target.query or "").casefold())


def _overlap(label: str, query: str) -> bool:
    if not query:
        return False
    lab = " ".join(label.replace("-", " ").split())
    q = " ".join(query.replace("-", " ").split())
    if not lab:
        return False
    if q == lab or q in lab:
        return True
    q_words = [w for w in q.split() if len(w) > 2]
    lab_words = set(lab.split())
    if lab in q_words:
        return True
    return bool(q_words) and any(w in lab_words or w in lab for w in q_words)


def _where_ok(tr: Track, where: str | None, width: int, height: int) -> bool:
    if not where:
        return True
    boxes = tr.observed_boxes()
    if not boxes:
        return False
    xs, ys = zip(*(box_center(b) for b in boxes))
    cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
    nx, ny = cx / max(1, width), cy / max(1, height)
    if where == "bottom":
        return ny >= 0.62
    if where == "top":
        return ny <= 0.38
    if where == "left":
        return nx <= 0.38
    if where == "right":
        return nx >= 0.62
    if where == "top-right":
        return nx >= 0.55 and ny <= 0.45
    if where == "top-left":
        return nx <= 0.45 and ny <= 0.45
    if where == "bottom-right":
        return nx >= 0.55 and ny >= 0.55
    if where == "bottom-left":
        return nx <= 0.45 and ny >= 0.55
    return True

