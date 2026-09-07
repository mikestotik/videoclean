from __future__ import annotations

from videoclean.domain.intent import CROP_PARTS, SCREEN_WHERES


def normalize_where(value: str | None) -> str | None:
    if not value:
        return None
    v = str(value).strip().lower().replace("_", "-").replace(" ", "-")
    aliases = {
        "upper-right": "top-right",
        "upper-left": "top-left",
        "lower-right": "bottom-right",
        "lower-left": "bottom-left",
    }
    v = aliases.get(v, v)
    return v if v in SCREEN_WHERES else None


def part_or_where(part: str | None, where: str | None) -> tuple[str | None, str | None]:
    where = normalize_where(where)
    if part in SCREEN_WHERES:
        if where is None:
            where = normalize_where(part)
        part = None
    if part not in CROP_PARTS:
        if normalize_where(part):
            where = where or normalize_where(part)
        part = None
    return part, where
