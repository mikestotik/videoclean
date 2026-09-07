from __future__ import annotations

from pathlib import Path


def hf_hub_dir(model_id: str) -> Path:
    return Path.home() / ".cache" / "huggingface" / "hub" / ("models--" + model_id.replace("/", "--"))


def hf_cached(model_id: str) -> bool:
    root = hf_hub_dir(model_id)
    if not root.is_dir():
        return False
    if any(root.rglob("*.incomplete")):
        return False
    snaps = root / "snapshots"
    if not snaps.is_dir():
        return False
    try:
        return any(snaps.iterdir())
    except OSError:
        return False


def download_hint(model_id: str) -> str:
    return f"uv run huggingface-cli download {model_id}"
