from __future__ import annotations

from dataclasses import dataclass, field

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
