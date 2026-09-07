from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request

from videoclean.adapters.hf_cache import hf_cached
from videoclean.adapters.inpainters.lama import LAMA_MODEL_URL, find_weights as find_lama_weights
from videoclean.adapters.inpainters.propainter import (
    DEFAULT_HF_REPO,
    find_vendor,
    find_weights as find_propainter_weights,
)
from videoclean.application.ports.model_catalog import ComponentInfo, ComponentStatus
from videoclean.store import JobIndex

OLLAMA_API = "http://127.0.0.1:11434"
OLLAMA_TAGS_TIMEOUT_S = 0.2
OLLAMA_NEG_TTL_S = 30.0

_ollama_neg_lock = threading.Lock()
_ollama_neg_until = 0.0

COMPONENTS: tuple[ComponentInfo, ...] = (
    ComponentInfo(
        id="detector:grounding-dino",
        title="Grounding DINO tiny",
        kind="detector",
        model_ref="IDEA-Research/grounding-dino-tiny",
        size_hint="~650 MB",
    ),
    ComponentInfo(
        id="detector:owlvit",
        title="OWL-ViT base",
        kind="detector",
        model_ref="google/owlvit-base-patch32",
        size_hint="~600 MB",
    ),
    ComponentInfo(
        id="segmenter:sam2-tiny",
        title="SAM2 Hiera tiny",
        kind="segmenter",
        model_ref="facebook/sam2-hiera-tiny",
        size_hint="~160 MB",
    ),
    ComponentInfo(
        id="segmenter:sam2-large",
        title="SAM2 Hiera large",
        kind="segmenter",
        model_ref="facebook/sam2-hiera-large",
        size_hint="~900 MB",
    ),
    ComponentInfo(
        id="inpainter:lama",
        title="LaMa (big-lama.pt)",
        kind="inpainter",
        model_ref=LAMA_MODEL_URL,
        size_hint="~200 MB",
    ),
    ComponentInfo(
        id="inpainter:propainter",
        title="ProPainter",
        kind="inpainter",
        model_ref=DEFAULT_HF_REPO,
        size_hint="~400 MB",
    ),
    ComponentInfo(
        id="llm:ollama-llama3.2",
        title="Ollama llama3.2",
        kind="llm",
        model_ref="llama3.2",
        size_hint="~2 GB",
    ),
    ComponentInfo(
        id="llm:ollama-llava-phi3",
        title="Ollama llava-phi3",
        kind="llm",
        model_ref="llava-phi3",
        size_hint="~2.3 GB",
    ),
)

COMPONENT_IDS: tuple[str, ...] = tuple(c.id for c in COMPONENTS)
COMPONENT_BY_ID: dict[str, ComponentInfo] = {c.id: c for c in COMPONENTS}

_BACKEND_TO_COMPONENT: dict[tuple[str, str], str] = {
    ("detector", "grounding-dino"): "detector:grounding-dino",
    ("detector", "owlvit"): "detector:owlvit",
    ("inpainter", "lama"): "inpainter:lama",
    ("inpainter", "propainter"): "inpainter:propainter",
    ("llm", "llama3.2"): "llm:ollama-llama3.2",
    ("llm", "llava-phi3"): "llm:ollama-llava-phi3",
}


def component_id_for(kind: str, name: str, segmenter_model: str = "") -> str | None:
    """Map a Clean-tab backend name to a catalog component id (None = always-ready)."""
    kind = (kind or "").strip().lower()
    name = (name or "").strip().lower()
    if kind == "inpainter" and name == "opencv-telea":
        return None
    if kind == "segmenter" and name in {"sam2", "sam2-video"}:
        model = (segmenter_model or "").strip().lower()
        if "large" in model:
            return "segmenter:sam2-large"
        return "segmenter:sam2-tiny"
    return _BACKEND_TO_COMPONENT.get((kind, name))


def backend_ready(
    kind: str,
    name: str,
    catalog: ModelCatalog | None = None,
    segmenter_model: str = "",
) -> bool:
    if kind == "inpainter" and name == "opencv-telea":
        return True
    cid = component_id_for(kind, name, segmenter_model=segmenter_model)
    if cid is None:
        return False
    return (catalog or ModelCatalog()).is_ready(cid)


class ModelCatalog:
    def __init__(self, jobs: JobIndex | None = None) -> None:
        self.jobs = jobs

    def list_status(self) -> list[ComponentStatus]:
        active, failed = self._download_overlay()
        tags = ollama_tags()
        rows: list[ComponentStatus] = []
        for info in COMPONENTS:
            if info.id in active:
                rows.append(ComponentStatus(info, "downloading", active[info.id] or "downloading"))
                continue
            state, message = self._probe(info, tags)
            if state != "ready" and info.id in failed:
                rows.append(ComponentStatus(info, "error", failed[info.id] or message))
                continue
            rows.append(ComponentStatus(info, state, message))
        return rows

    def is_ready(self, component_id: str) -> bool:
        info = COMPONENT_BY_ID.get(component_id)
        if info is None:
            return False
        tags = ollama_tags() if info.kind == "llm" else None
        state, _ = self._probe(info, tags)
        return state == "ready"

    def _probe(self, info: ComponentInfo, tags: set[str] | None) -> tuple[str, str]:
        if info.kind in {"detector", "segmenter"}:
            if hf_cached(info.model_ref):
                return "ready", f"{info.model_ref} cached"
            return "missing", f"{info.model_ref} not in local HF cache"
        if info.id == "inpainter:lama":
            path = find_lama_weights()
            if path is not None:
                return "ready", f"weights on disk ({path.name})"
            return "missing", "big-lama.pt not found"
        if info.id == "inpainter:propainter":
            vendor = find_vendor()
            weights = find_propainter_weights(info.model_ref)
            missing: list[str] = []
            if vendor is None:
                missing.append("vendor repo missing")
            if weights is None:
                missing.append("weights missing")
            if missing:
                return "missing", "; ".join(missing)
            return "ready", "vendor+weights on disk"
        if info.kind == "llm":
            if tags is None:
                return "error", "unavailable: start ollama"
            if info.model_ref in tags:
                return "ready", f"{info.model_ref} present"
            return "missing", f"{info.model_ref} not pulled"
        return "error", f"unknown component {info.id}"

    def _download_overlay(self) -> tuple[dict[str, str], dict[str, str]]:
        active: dict[str, str] = {}
        failed: dict[str, str] = {}
        if self.jobs is None:
            return active, failed
        for row in self.jobs.list_downloads():
            cid = row["component_id"]
            state = row["state"]
            message = row["message"] or ""
            if cid in active or cid in failed:
                continue
            if state in {"running", "queued"}:
                active[cid] = message
            elif state == "failed":
                failed[cid] = message
        return active, failed


def ollama_tags() -> set[str] | None:
    global _ollama_neg_until
    now = time.monotonic()
    with _ollama_neg_lock:
        if now < _ollama_neg_until:
            return None
    url = OLLAMA_API.rstrip("/") + "/api/tags"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=OLLAMA_TAGS_TIMEOUT_S) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError, ValueError):
        with _ollama_neg_lock:
            _ollama_neg_until = time.monotonic() + OLLAMA_NEG_TTL_S
        return None
    names: set[str] = set()
    for model in payload.get("models") or []:
        name = str(model.get("name") or model.get("model") or "").strip()
        if not name:
            continue
        names.add(name)
        names.add(name.split(":")[0])
    with _ollama_neg_lock:
        _ollama_neg_until = 0.0
    return names
