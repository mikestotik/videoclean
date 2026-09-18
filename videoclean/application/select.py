from __future__ import annotations

from dataclasses import replace

from videoclean.domain.intent import CROP_PARTS, Intent, Target
from videoclean.domain.tracks import Track, box_center, interpolate_gaps

_GENERIC_LABELS = {"object", "thing", "stuff", "label", "entity", ""}


def select_tracks(
    tracks: list[Track],
    intent: Intent,
    *,
    width: int,
    height: int,
    relax: bool = False,
) -> list[Track]:
    _scoped = _apply_frames_windows(tracks, intent)
    chosen = _select_once(_scoped, intent, width, height)
    if chosen or not relax or not tracks:
        return chosen
    no_ord = _strip_intent(intent, ordinal=True)
    chosen = _select_once(_scoped, no_ord, width, height)
    if chosen:
        return chosen
    no_where = _strip_intent(intent, ordinal=True, where=True)
    return _select_once(_scoped, no_where, width, height)


def _apply_frames_windows(tracks: list[Track], intent: Intent) -> list[Track]:
    """Zero out track boxes that fall outside every target window with that query.

    A target with frames=(a, b) only hunts on frames [a, b). Tracks whose boxes live
    outside all matching windows are dropped entirely. Targets without windows keep
    the whole clip. Mutates nothing: tracks are copied with clipped box lists.
    """
    windows = [(t.query.casefold(), t.frames) for t in intent.targets if t.frames is not None]
    if not windows:
        return tracks
    out: list[Track] = []
    for tr in tracks:
        q = (tr.label or "").casefold()
        relevant = [w for wq, w in windows if wq and (wq in q or q in wq)]
        if not relevant:
            out.append(tr)
            continue
        boxes = list(tr.boxes)
        for i, box in enumerate(boxes):
            if box is None:
                continue
            if not any(a <= i < b for a, b in relevant):
                boxes[i] = None
        if not any(b is not None for b in boxes):
            continue
        clipped = replace(tr, boxes=boxes)
        clipped.motion = tr.motion
        out.append(clipped)
    return out


def _select_once(
    tracks: list[Track],
    intent: Intent,
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
            if target.frames is not None:
                a, b = target.frames
                boxes = list(tr.boxes)
                for i in range(len(boxes)):
                    if not (a <= i < b):
                        boxes[i] = None
                tr = replace(tr, boxes=boxes)
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


def _strip_intent(intent: Intent, *, ordinal: bool = False, where: bool = False) -> Intent:
    targets = []
    for t in intent.targets:
        kw: dict = {}
        if ordinal:
            kw["ordinal"] = None
            kw["from_side"] = None
        if where:
            kw["where"] = None
        targets.append(replace(t, **kw) if kw else t)
    return replace(intent, targets=targets)


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
        if target.frames is not None:
            a, b = target.frames
            # Require a real observation inside the window (ignore hold-filled gaps).
            if not any(box is not None for box in tr.boxes[a:b]):
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
    lab = (tr.label or "").casefold().strip()
    if target.kind in {"text_overlay", "watermark"} and lab in _GENERIC_LABELS:
        return True
    return _overlap(lab, (target.query or "").casefold())


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


def region_name(tr: Track, width: int, height: int) -> str:
    boxes = tr.observed_boxes()
    if not boxes:
        return "unknown"
    xs, ys = zip(*(box_center(b) for b in boxes))
    nx, ny = (sum(xs) / len(xs)) / max(1, width), (sum(ys) / len(ys)) / max(1, height)
    vert = "top" if ny <= 0.38 else "bottom" if ny >= 0.62 else "middle"
    horz = "left" if nx <= 0.38 else "right" if nx >= 0.62 else "center"
    if vert == "middle" and horz == "center":
        return "center"
    if horz == "center":
        return vert
    if vert == "middle":
        return horz
    return f"{vert}-{horz}"


def explain_unmatched(tracks: list[Track], intent: Intent, width: int, height: int) -> str:
    want = "; ".join(_want(t) for t in intent.targets) or "no targets"
    parts = [f"kept 0 of {len(tracks)} detected box(es) for {want}."]
    for tr in tracks[:6]:
        why = _drop_reason(tr, intent, width, height)
        parts.append(f"box label={tr.label!r} at {region_name(tr, width, height)}: {why}.")
    extra = len(tracks) - 6
    if extra > 0:
        parts.append(f"plus {extra} more box(es).")
    parts.append("Hint: Grounding DINO often labels overlays as 'object'; we match those to text/logo queries. If region is wrong, say top/bottom/corner in the prompt.")
    return " ".join(parts)


def _want(t: Target) -> str:
    loc = t.where or "any-region"
    extra = f" ordinal={t.ordinal} from {t.from_side or 'left'}" if t.ordinal else ""
    return f"{t.query!r} ({t.kind}, {loc}{extra})"


def _drop_reason(tr: Track, intent: Intent, width: int, height: int) -> str:
    reasons: list[str] = []
    for t in intent.targets:
        if not _query_ok(tr, t):
            reasons.append(f"label does not overlap {t.query!r}")
            continue
        if not _where_ok(tr, t.where, width, height):
            reasons.append(f"outside where={t.where}")
            continue
        if t.ordinal:
            reasons.append(f"not ordinal {t.ordinal} from {t.from_side or 'left'}")
            continue
        reasons.append(f"would match {t.query!r}")
    return "; ".join(reasons) if reasons else "no target accepted it"

