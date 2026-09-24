"""Picture recipes fast / balanced / quality.

They choose the segmenter, inpainter family, verify passes, keyframes and dilate.
They do not decide how much of the GPU to use. An empty resource ceiling is the
whole machine. Explicit form/CLI fields win over recipe keys.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from videoclean.application.errors import PipelineError

PROFILES = ("fast", "balanced", "quality")


def profile_defaults(name: str, device: str) -> dict[str, Any]:
    """Return config overrides for a named profile on the given device."""
    key = (name or "").strip().lower()
    if key not in PROFILES:
        raise ValueError(f"unknown profile {name!r}; known: {', '.join(PROFILES)}")
    dev = (device or "cpu").strip().lower()
    cuda = dev == "cuda"

    if key == "fast":
        return {
            "verify": False,
            "verify_max_passes": 0,
            "detector_keyframes": 6,
            "mask_dilate_px": 2,
            "segmenter": "sam2",
            "inpainter": "lama",
            "inpaint_workers": 0,  # auto
            "inpaint_chunk_overlap": 0,
        }

    if key == "balanced":
        return {
            "verify": True,
            "verify_max_passes": 1,
            "detector_keyframes": 10,
            "mask_dilate_px": 3,
            "segmenter": "sam2",
            "inpainter": "lama" if not cuda else "lama",
            "inpaint_workers": 0,
            "inpaint_chunk_overlap": 8,
        }

    # quality
    return {
        "verify": True,
        "verify_max_passes": 2,
        "detector_keyframes": 16 if cuda else 12,
        "mask_dilate_px": 5,
        "segmenter": "sam2-video" if cuda else "sam2",
        "inpainter": "propainter" if cuda else "lama",
        "inpaint_workers": 0,
        "inpaint_chunk_overlap": 12,
        "propainter_subvideo_length": 80,
    }


# Knobs owned by a profile. Selecting a profile overwrites these; device/formats/llm stay.
PROFILE_KEYS = frozenset(
    {
        "verify",
        "verify_max_passes",
        "detector_keyframes",
        "mask_dilate_px",
        "segmenter",
        "inpainter",
        "inpaint_workers",
        "inpaint_chunk_overlap",
        "propainter_subvideo_length",
    }
)


def apply_profile(payload: dict[str, Any]) -> dict[str, Any]:
    """Apply built-in profile defaults onto payload.

    Profile-owned knobs are overwritten. Other fields (device, formats, llm, …) stay.
    After the user edits a knob in the UI, set profile=custom to stop re-applying.
    """
    out = dict(payload or {})
    raw = str(out.get("profile") or "").strip().lower()
    if not raw or raw == "custom":
        out["profile"] = "custom"
        return out
    device = str(out.get("device") or "cpu").strip().lower() or "cpu"
    defaults = profile_defaults(raw, device)
    for key, value in defaults.items():
        out[key] = value
    out["profile"] = raw
    return out


def profiles_payload(device: str = "cpu") -> list[dict[str, Any]]:
    """UI/API catalog of built-in profiles with resolved defaults."""
    labels = {
        "fast": "Быстро",
        "balanced": "Баланс",
        "quality": "Качество",
    }
    hints = {
        "fast": "LaMa, без проверки, 6 ключевых кадров.",
        "balanced": "LaMa, одна проверка остатка, 10 ключевых кадров.",
        "quality": (
            "Больше ключевых кадров, на CUDA sam2-video и ProPainter. "
            "Дырка заливается кропом до потолка памяти, не целым кадром. "
            "До 2 проходов проверки остатка."
        ),
    }
    return [
        {
            "id": name,
            "title": labels[name],
            "hint": hints[name],
            "defaults": profile_defaults(name, device),
        }
        for name in PROFILES
    ]


def explicit_from_form(fields: Mapping[str, str]) -> dict[str, Any]:
    """Keys the caller actually sent. Empty string is absent.

    Does not call serialize_clean_form and does not invent device=cpu.
    """
    out: dict[str, Any] = {}
    for key, raw in fields.items():
        if raw is None:
            continue
        text = raw if isinstance(raw, str) else str(raw)
        if text == "":
            continue
        out[key] = _coerce_form_value(key, text)
    return out


def merge_run_config(
    defaults: Mapping[str, Any],
    preset: Mapping[str, Any] | None,
    profile: str | None,
    explicit: Mapping[str, Any],
    *,
    resolve_device: Callable[[str], str],
) -> dict[str, Any]:
    """defaults, then preset, then PROFILE_KEYS, then explicit.

    ``profile`` is the explicit profile if the request sent one,
    otherwise the preset profile, otherwise defaults["profile"].
    Only fast|balanced|quality apply PROFILE_KEYS. ``custom`` skips
    that layer. Any other name raises PipelineError (HTTP 400).
    Device is resolved before profile_defaults. A preset field that
    is also a PROFILE_KEY does not survive its own built-in profile.
    """
    out = dict(defaults)
    preset_map = dict(preset or {})
    explicit_map = dict(explicit or {})
    for key, value in preset_map.items():
        if key == "profile":
            continue
        out[key] = value

    chosen = explicit_map.get("profile")
    if chosen in (None, ""):
        chosen = profile if profile not in (None, "") else preset_map.get("profile", out.get("profile"))
    name = str(chosen or "custom").strip().lower() or "custom"

    if "device" in explicit_map:
        requested = explicit_map.get("device")
    elif "device" in preset_map:
        requested = preset_map.get("device")
    else:
        requested = out.get("device")
    requested_text = "" if requested is None else str(requested).strip().lower()
    resolved = resolve_device(requested_text or "auto")
    out["device_requested"] = requested_text or "auto"
    out["device"] = resolved

    if name in PROFILES:
        for key, value in profile_defaults(name, resolved).items():
            out[key] = value
        out["profile"] = name
    elif name == "custom":
        out["profile"] = "custom"
    else:
        raise PipelineError(f"unknown profile {name!r}; known: custom, {', '.join(PROFILES)}")

    for key, value in explicit_map.items():
        if key == "device":
            continue
        out[key] = value
    if "device" in explicit_map:
        text = "" if explicit_map.get("device") is None else str(explicit_map.get("device")).strip().lower()
        out["device_requested"] = text or "auto"
        out["device"] = resolve_device(text or "auto")
    if str(out.get("device") or "") in {"", "auto"}:
        out["device"] = resolve_device("auto")
    return out


_INT_KEYS = {
    "mask_dilate_px",
    "prompt_frame_stride",
    "prompt_frame_max",
    "parse_chunk_frames",
    "vision_batch",
    "detector_keyframes",
    "verify_max_passes",
    "inpaint_workers",
    "inpaint_chunk_overlap",
    "propainter_mask_dilation",
    "propainter_ref_stride",
    "propainter_neighbor_length",
    "propainter_subvideo_length",
    "propainter_raft_iter",
    "webm_crf",
    "segment_seconds",
    "max_vram_mb",
    "cpu_threads",
    "inpaint_max_side",
}
_FLOAT_KEYS = {
    "detector_threshold",
    "min_mask_coverage",
    "verify_max_coverage",
    "detector_nms_iou",
    "detector_max_box_area",
    "tracker_min_score",
    "tracker_max_template_area",
}
_BOOL_KEYS = {"verify", "keep_workdir", "select_relax", "verify_redetect", "overwrite", "allow_download"}


def _coerce_form_value(key: str, text: str) -> Any:
    if key in _BOOL_KEYS:
        return text.strip().lower() in {"1", "true", "yes", "on"}
    if key in _INT_KEYS:
        return int(text)
    if key in _FLOAT_KEYS:
        return float(text)
    if key == "formats":
        return [part.strip().lower() for part in text.split(",") if part.strip()]
    return text.strip()
