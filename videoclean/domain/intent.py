from __future__ import annotations

from dataclasses import dataclass, field, replace

CROP_PARTS = frozenset({"text", "icon", "word", "dot"})
SCREEN_WHERES = (
    "top-left",
    "top-right",
    "bottom-left",
    "bottom-right",
    "top",
    "bottom",
    "left",
    "right",
)


@dataclass
class Target:
    """One thing to remove. Parser never locates pixels — it names what to find."""

    kind: str  # watermark | text_overlay | object
    query: str  # short visual name for the detector
    part: str | None = None  # crop of a found box: text|icon|word|dot
    motion: str = "any"  # any | static | floating
    where: str | None = None  # screen region filter after detection
    ordinal: int | None = None  # 1-based, e.g. third
    from_side: str | None = None  # left | right, with ordinal
    box: tuple[float, float, float, float] | None = None  # unused by run; debug only
    frames: tuple[int, int] | None = None  # [start, end) validity window; None = whole clip


@dataclass
class Intent:
    targets: list[Target] = field(default_factory=list)
    parse_mode: str = "llm"
    raw: str = ""
    defaulted: bool = False

    @property
    def kinds(self) -> list[str]:
        return [t.kind for t in self.targets]

    @property
    def queries(self) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for t in self.targets:
            q = (t.query or "").strip()
            if q and q not in seen:
                seen.add(q)
                out.append(q)
        return out


def merge_scoped_targets(targets: list[Target]) -> list[Target]:
    """Merge targets that share (kind, query, where, ordinal, from_side) and overlap in time.

    windows (a1,b1) and (a2,b2) merge when b1 >= a2 (adjacent or overlapping chunks).
    Targets without windows pass through untouched.
    """
    buckets: dict[tuple, list[Target]] = {}
    order: list[tuple] = []
    for t in targets:
        key = (t.kind, t.query.casefold(), t.where, t.ordinal, t.from_side)
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append(t)
    out: list[Target] = []
    for key in order:
        group = buckets[key]
        if any(t.frames is None for t in group):
            first = group[0]
            out.append(replace(first, frames=None))
            continue
        group.sort(key=lambda t: t.frames[0])
        cur = group[0]
        for nxt in group[1:]:
            pa, pb = cur.frames
            na, nb = nxt.frames
            if pb >= na:
                cur = replace(cur, frames=(pa, max(pb, nb)))
            else:
                out.append(cur)
                cur = nxt
        out.append(cur)
    return out
