"""Built-in speed/quality profiles (device-aware).

Explicit form/CLI fields always win over profile defaults.
"""

from __future__ import annotations

from typing import Any

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
            "inpainter": "opencv-telea",
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
        "inpaint_workers": 1 if cuda else 0,
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
        "fast": "Telea, без verify, мало keyframes — черновик.",
        "balanced": "LaMa + один verify-pass — обычный рабочий режим.",
        "quality": "Больше keyframes, sam2-video+ProPainter на CUDA, до 2 verify-pass.",
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
