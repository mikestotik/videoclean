from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from videoclean.adapters.prompt.frames import bgr_to_jpeg
from videoclean.adapters.prompt.locations import normalize_where, part_or_where
from videoclean.adapters.prompt.refine import refine_intent
from videoclean.application.errors import AdapterUnavailable, PipelineError
from videoclean.domain.intent import Intent, Target

KINDS = {"watermark", "text_overlay", "object"}
MOTIONS = {"any", "static", "floating"}
SIDES = {"left", "right"}
BAD_QUERIES = {
    "the",
    "a",
    "an",
    "overlay",
    "image",
    "frame",
    "video",
    "object",
    "thing",
    "stuff",
}

SYSTEM = """You turn a user's removal request into detector queries for Grounding DINO / OWL-ViT.
You do NOT see frames. Do NOT invent a fixed menu of overlays.

Return ONLY JSON:
{"targets":[{"kind":"watermark|text_overlay|object","query":"short English visual name","where":null,"ordinal":null,"from_side":null,"motion":"any"}]}

Rules:
- query: short English name the detector can search. Never Russian. Never vague ("overlay", "stuff").
- For letters on screen use "text" / "caption" / "title". Do NOT OCR the overlay into the query (not the words written on screen).
- Derive queries from what the USER named (translate/paraphrase into English visuals). If they named specific text, describe that kind of overlay, not a generic pack.
- where follows the USER's location words. Do not point at a different overlay you noticed.
- If the request is vague and you have no frame notes, ask yourself what concrete English queries could match — but do NOT paste a canned list. Prefer fewer honest targets over a fake full HUD inventory.
- kind: watermark = logo/© mark; text_overlay = letters/captions; object = physical thing.
- where: only if the user named a region (top|bottom|left|right|top-left|top-right|bottom-left|bottom-right), else null.
- ordinal + from_side: only for explicit "third from the left". Else both null. Do NOT invent ordinals.
- motion: floating if they said moving/floating; static if fixed corner mark; else any.
- No HUD/reticle unless asked. No boxes. No prose.
"""

VISION_SYSTEM = """You see sampled video frames AND a user's removal request.
Build detector queries for Grounding DINO / OWL-ViT. No boxes, no masks, no prose.

Return ONLY JSON:
{"targets":[{"kind":"watermark|text_overlay|object","query":"short English visual name","where":null,"ordinal":null,"from_side":null,"motion":"any"}]}

Rules:
- Look at the frames. Emit a target only for something you can see that matches the user's ask.
- query: short English visual name of the overlay TYPE ("text", "caption", "logo"), never Russian, never the words written on screen.
- where follows the USER's location words, not some other overlay in the frame.
- Do NOT output a canned set like title+caption+side+watermark unless those are actually visible and requested.
- where from what you see (top|bottom|left|right|corners) or null.
- ordinal/from_side only if the user said so.
- motion=floating if it moves/morphs across the sampled frames; static for fixed marks; else any.
- Split distinct visible overlays into separate targets. Do not invent extras.
"""

BRIDGE_SYSTEM = """You convert vision-model frame notes + a user removal request into detector JSON.
Use ONLY overlays mentioned in the notes that match the request. Do NOT invent a standard news-graphic template.

Return ONLY JSON:
{"targets":[{"kind":"watermark|text_overlay|object","query":"short English visual name","where":null,"ordinal":null,"from_side":null,"motion":"any"}]}

query must be English, concrete, searchable by Grounding DINO. No Russian. Overlay type ("text","caption","logo"), not OCR of the letters.
"""


_PROMPTS_DIR = Path(__file__).parent / "prompts"


def _read_prompt(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8")


SYSTEM = _read_prompt("system.md")
VISION_SYSTEM = _read_prompt("vision_system.md")
BRIDGE_SYSTEM = _read_prompt("bridge_system.md")

# Fallback defaults when no --prompt-templates dir overrides them.
_BUILTIN_PROMPTS = {
    "system.md": SYSTEM,
    "vision_system.md": VISION_SYSTEM,
    "bridge_system.md": BRIDGE_SYSTEM,
}


def _load_prompts(templates_dir: str | None) -> dict[str, str]:
    if not templates_dir:
        return dict(_BUILTIN_PROMPTS)
    root = Path(templates_dir).expanduser()
    out: dict[str, str] = {}
    for name, default in _BUILTIN_PROMPTS.items():
        custom = root / name
        out[name] = custom.read_text(encoding="utf-8") if custom.is_file() else default
    return out


class LlmPromptParser:
    name = "llm"

    def __init__(self, llm, vision_batch: int = 2, templates_dir: str | None = None) -> None:
        self.llm = llm
        self.vision_batch = max(1, int(vision_batch))
        self._templates = _load_prompts(templates_dir)

    def status(self) -> str:
        return self.llm.status()

    def parse(
        self,
        prompt: str | None,
        frame: np.ndarray | None = None,
        frames: list[np.ndarray] | None = None,
    ) -> Intent:
        raw = (prompt or "").strip()
        if not raw:
            raise PipelineError("--prompt is required")
        st = self.llm.status()
        if not st.startswith("ready"):
            raise AdapterUnavailable(f"prompt-parser llm: {st}")

        sample = list(frames or [])
        if frame is not None:
            sample.append(frame)

        if sample:
            intent = self._try_vision(raw, sample)
            if intent is not None:
                return intent
        return self._parse_text(raw)

    def _parse_text(self, raw: str) -> Intent:
        text = self.llm.complete(self._templates["system.md"], raw, images=None)
        intent = intent_from_llm_json(text, raw=raw)
        intent.parse_mode = "llm"
        return intent

    def _try_vision(self, raw: str, frames: list[np.ndarray]) -> Intent | None:
        try:
            jpegs = [bgr_to_jpeg(f) for f in frames]
        except Exception:  # noqa: BLE001
            return None
        targets: list[Target] = []
        modes: list[str] = []
        for i in range(0, len(jpegs), self.vision_batch):
            batch = jpegs[i : i + self.vision_batch]
            batch_intent = self._vision_batch(raw, batch)
            if batch_intent is None:
                continue
            targets.extend(batch_intent.targets)
            modes.append(batch_intent.parse_mode)
        if not targets:
            return None
        deduped: list[Target] = []
        seen: set[tuple] = set()
        for t in targets:
            key = (t.kind, t.query.casefold(), t.where, t.ordinal, t.from_side)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(t)
        intent = Intent(targets=deduped, raw=raw, defaulted=False)
        intent.parse_mode = "llm-vision-bridged" if modes and all(m == "llm-vision-bridged" for m in modes) else "llm-vision"
        return intent

    def _vision_batch(self, raw: str, jpegs: list[bytes]) -> Intent | None:
        # Small VLMs (llava-phi3) often ignore system; put the contract in user text.
        vision_system = self._templates["vision_system.md"]
        user = (
            f"{vision_system}\n\n"
            f"User request:\n{raw}\n\n"
            f"Attached: {len(jpegs)} sampled frame(s) in time order. "
            "Reply with ONLY the JSON object for what you see that matches the request. "
            "No prose. No canned overlay list."
        )
        vision_notes = ""
        try:
            text = self.llm.complete(vision_system, user, images=jpegs)
            vision_notes = text
            intent = intent_from_llm_json(text, raw=raw)
        except (PipelineError, AdapterUnavailable):
            repair = (
                f"User request: {raw}\n"
                "From the attached frames, list matching overlays as JSON only:\n"
                '{"targets":[{"kind":"text_overlay|watermark|object","query":"short English name",'
                '"where":"top|bottom|left|right|top-left|top-right|bottom-left|bottom-right"|null,'
                '"ordinal":null,"from_side":null,"motion":"any|static|floating"}]}\n'
                "No other text."
            )
            try:
                text = self.llm.complete(vision_system, repair, images=jpegs)
                # Keep the first vision prose for bridging; repair may be shorter junk.
                if len((text or "").strip()) > len(vision_notes.strip()):
                    vision_notes = text
                intent = intent_from_llm_json(text, raw=raw)
            except (PipelineError, AdapterUnavailable, Exception):  # noqa: BLE001
                # Bridge: vision prose → text JSON (same model, no images).
                intent = self._intent_from_vision_notes(raw, vision_notes or "")
                if intent is None:
                    return None
                intent.parse_mode = "llm-vision-bridged"
                return intent
        except Exception:  # noqa: BLE001
            return None
        intent.parse_mode = "llm-vision"
        return intent

    def _intent_from_vision_notes(self, raw: str, notes: str) -> Intent | None:
        notes = (notes or "").strip()
        if len(notes) < 20:
            return None
        user = (
            f"Frame notes from a vision model:\n{notes[:2500]}\n\n"
            f"User removal request:\n{raw}\n\n"
            "Return ONLY targets JSON grounded in those notes. "
            "If the notes are vague, keep targets few and specific — never a canned HUD list."
        )
        try:
            text = self.llm.complete(self._templates["bridge_system.md"], user, images=None)
            return intent_from_llm_json(text, raw=raw)
        except (PipelineError, AdapterUnavailable, Exception):  # noqa: BLE001
            return None


def intent_from_llm_json(text: str, *, raw: str) -> Intent:
    data = _extract_json(text)
    items = data.get("targets")
    if not isinstance(items, list) or not items:
        raise PipelineError("prompt-parser llm JSON has no targets[]")
    targets: list[Target] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind") or "object").strip().lower()
        if kind not in KINDS:
            kind = "object"
        query = " ".join(str(item.get("query") or "").split())
        if not _query_ok(query):
            continue
        part = item.get("part")
        if part in ("", "null", "none", "whole"):
            part = None
        if part is not None:
            part = str(part).strip().lower()
        where = item.get("where")
        if where in ("", "null", "none"):
            where = None
        if where is not None:
            where = normalize_where(str(where))
        part, where = part_or_where(part, where)
        motion = str(item.get("motion") or "any").strip().lower()
        if motion not in MOTIONS:
            motion = "any"
        ordinal = _ordinal(item.get("ordinal"))
        from_side = str(item.get("from_side") or "").strip().lower() or None
        if from_side not in SIDES:
            from_side = None
        box = _box(item.get("box"))
        targets.append(
            Target(
                kind=kind,
                query=query,
                part=part,
                motion=motion,
                where=where,
                ordinal=ordinal,
                from_side=from_side,
                box=box,
            )
        )
    if not targets:
        raise PipelineError("prompt-parser llm JSON had no usable targets")
    return refine_intent(Intent(targets=targets, parse_mode="llm", raw=raw, defaulted=False))


def _query_ok(query: str) -> bool:
    q = (query or "").strip().lower()
    if len(q) < 3:
        return False
    if q in BAD_QUERIES:
        return False
    # Detector queries must be English; Cyrillic/etc. fail open-vocab search.
    if any(ord(ch) > 127 for ch in q):
        return False
    if "|" in q:  # model echoed "watermark|text_overlay|object"
        return False
    return True


def _ordinal(raw) -> int | None:
    if raw in (None, "", "null", "none"):
        return None
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return None
    return n if n >= 1 else None


def _box(raw) -> tuple[float, float, float, float] | None:
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in raw)
    except (TypeError, ValueError):
        return None
    x1, x2 = sorted((max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))))
    y1, y2 = sorted((max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))))
    if x2 - x1 < 0.005 or y2 - y1 < 0.005:
        return None
    return x1, y1, x2, y2


def _extract_json(text: str) -> dict:
    blob = (text or "").strip()
    fenced_obj = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", blob, re.DOTALL)
    fenced_arr = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", blob, re.DOTALL)
    if fenced_obj:
        blob = fenced_obj.group(1)
    elif fenced_arr:
        blob = fenced_arr.group(1)
    else:
        obj_start, obj_end = blob.find("{"), blob.rfind("}")
        arr_start, arr_end = blob.find("["), blob.rfind("]")
        if obj_start >= 0 and obj_end > obj_start and (arr_start < 0 or obj_start < arr_start):
            blob = blob[obj_start : obj_end + 1]
        elif arr_start >= 0 and arr_end > arr_start:
            blob = blob[arr_start : arr_end + 1]
    try:
        data = json.loads(blob)
    except json.JSONDecodeError as exc:
        raise PipelineError(f"prompt-parser llm did not return JSON: {exc}") from exc
    if isinstance(data, list):
        data = {"targets": data}
    if not isinstance(data, dict):
        raise PipelineError("prompt-parser llm JSON must be an object")
    return data
