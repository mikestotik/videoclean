from __future__ import annotations

import re
from dataclasses import replace

from videoclean.domain.intent import Intent, Target

_TEXT_ASK = re.compile(
    r"надпис|титры|текст|caption|subtitle|\btitle\b|watermark|логотип|\blogo\b|баннер|\bbanner\b",
    re.I,
)
_TEXT_QUERIES = {
    "text",
    "caption",
    "title",
    "subtitle",
    "banner",
    "headline",
    "lettering",
    "inscription",
    "sign",
    "osd",
    "on-screen text",
    "on screen text",
}
_MARK_QUERIES = {"logo", "watermark", "emblem", "bug", "channel logo", "mark"}

# Longer / more specific patterns first. Russian can be "правом нижнем" or "нижнем правом".
_WHERE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"верхн\w*.{0,16}прав|прав\w*.{0,16}верхн|top[-\s]?right|upper[-\s]?right", re.I), "top-right"),
    (re.compile(r"верхн\w*.{0,16}лев|лев\w*.{0,16}верхн|top[-\s]?left|upper[-\s]?left", re.I), "top-left"),
    (re.compile(r"нижн\w*.{0,16}прав|прав\w*.{0,16}нижн|bottom[-\s]?right|lower[-\s]?right", re.I), "bottom-right"),
    (re.compile(r"нижн\w*.{0,16}лев|лев\w*.{0,16}нижн|bottom[-\s]?left|lower[-\s]?left", re.I), "bottom-left"),
    (re.compile(r"сверху|вверху|наверху|(?<![a-z])top(?![a-z-])", re.I), "top"),
    (re.compile(r"снизу|внизу|(?<![a-z])bottom(?![a-z-])", re.I), "bottom"),
    (re.compile(r"слева|(?<![a-z])left(?![a-z-])", re.I), "left"),
    (re.compile(r"справа|(?<![a-z])right(?![a-z-])", re.I), "right"),
)

# Whitelist of words that explicitly name an on-screen overlay. Only these may flip an
# object target into a text overlay — the detector vocabulary stays open for everything
# else (ball, clock, drone, sticker on a wall, road sign…). A blacklist here would
# silently break unknown requests. Keep this list unambiguous: no words that also name
# physical things (sign, banner, sticker are deliberately absent).
_OVERLAY_WORDS = {
    "overlay",
    "writing",
    "lettering",
    "letters",
    "inscription",
    "caption",
    "subtitle",
    "title",
    "headline",
    "watermark",
    "logo",
    "osd",
}

_ORDINAL_ASK = re.compile(
    r"\b(1st|2nd|3rd|4th|first|second|third|fourth)\b"
    r"|трет(ь[юяе]|ий|ья)"
    r"|втор(ой|ая|ую|ое)"
    r"|перв(ый|ая|ую|ое|ую)"
    r"|четверт"
    r"|\b\d+\s*[- ]?(й|ый|ой|st|nd|rd|th)\b",
    re.I,
)


def where_from_prompt(raw: str) -> str | None:
    text = raw or ""
    for pat, where in _WHERE_PATTERNS:
        if pat.search(text):
            return where
    return None


def prompt_has_ordinal(raw: str) -> bool:
    return bool(_ORDINAL_ASK.search(raw or ""))


def prompt_asks_text(raw: str) -> bool:
    return bool(_TEXT_ASK.search(raw or ""))


def refine_intent(intent: Intent) -> Intent:
    """Prefer the user's words over a weak VLM: region, no invented ordinals, detector-friendly queries."""
    raw = intent.raw or ""
    user_where = where_from_prompt(raw)
    keep_ordinal = prompt_has_ordinal(raw)
    text_ask = prompt_asks_text(raw)
    out: list[Target] = []
    seen: set[tuple] = set()
    for t in intent.targets:
        kind = t.kind
        if text_ask and kind == "object" and _treat_object_as_overlay(t.query):
            kind = "text_overlay"
        query = _rewrite_query(kind, t.query)
        where = user_where or t.where
        ordinal = t.ordinal if keep_ordinal else None
        from_side = t.from_side if keep_ordinal else None
        key = (kind, query.casefold(), where, ordinal, from_side)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            replace(
                t,
                kind=kind,
                query=query,
                where=where,
                ordinal=ordinal,
                from_side=from_side,
            )
        )
    intent.targets = out
    return intent


def _treat_object_as_overlay(query: str) -> bool:
    words = (query or "").casefold().split()
    return bool(words) and any(w in _OVERLAY_WORDS for w in words)


def _rewrite_query(kind: str, query: str) -> str:
    q = " ".join((query or "").split())
    low = q.casefold()
    if kind == "text_overlay":
        if any(token in low for token in _TEXT_QUERIES):
            return q
        return "text"
    if kind == "watermark":
        if any(token in low for token in _MARK_QUERIES):
            return q
        return "logo"
    return q
