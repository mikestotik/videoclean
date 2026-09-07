from __future__ import annotations

import os
from pathlib import Path


def _env_path(*names: str) -> Path | None:
    for name in names:
        raw = (os.environ.get(name) or "").strip()
        if raw:
            return Path(os.path.expandvars(raw)).expanduser()
    return None


def hf_hub_cache_root() -> Path:
    """Hub cache per huggingface_hub: HF_HUB_CACHE, else HF_HOME/hub, else ~/.cache/huggingface/hub."""
    override = _env_path("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE")
    if override is not None:
        return override
    hf_home = _env_path("HF_HOME")
    if hf_home is not None:
        return hf_home / "hub"
    xdg = _env_path("XDG_CACHE_HOME")
    if xdg is not None:
        return xdg / "huggingface" / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def hf_hub_dir(model_id: str) -> Path:
    return hf_hub_cache_root() / ("models--" + model_id.replace("/", "--"))


def hf_cached(model_id: str) -> bool:
    """True when a local hub snapshot exists and no in-flight *.incomplete blobs.

    Do not rglob the whole tree — hub caches are large and this runs on UI paint.
    """
    root = hf_hub_dir(model_id)
    if not root.is_dir():
        return False
    snaps = root / "snapshots"
    if not snaps.is_dir():
        return False
    try:
        if not any(p.is_dir() for p in snaps.iterdir()):
            return False
    except OSError:
        return False
    blobs = root / "blobs"
    if blobs.is_dir():
        try:
            if any(blobs.glob("*.incomplete")):
                return False
        except OSError:
            return False
    return True


def download_hint(model_id: str) -> str:
    return f"uv run huggingface-cli download {model_id}"
