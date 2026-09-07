from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request

from pathlib import Path

from videoclean.adapters.hf_cache import hf_cached
from videoclean.adapters.inpainters.lama import LAMA_MODEL_URL, find_weights as find_lama_weights
from videoclean.adapters.inpainters.propainter import (
    DEFAULT_HF_REPO,
    find_vendor,
    find_weights as find_propainter_weights,
)
from videoclean.application.ports.model_catalog import ComponentInfo, ComponentStatus
from videoclean.store import JobIndex, resolve_data_dir

OLLAMA_API = (os.environ.get("VIDEOCLEAN_OLLAMA_URL") or "http://127.0.0.1:11434").rstrip("/")
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
        backend="ollama",
    ),
)

TELEA = ComponentInfo(
    id="inpainter:opencv-telea",
    title="OpenCV TELEA",
    kind="inpainter",
    model_ref="(builtin)",
    size_hint="CPU, always on",
    backend="opencv-telea",
    source="builtin",
)

COMPONENT_IDS: tuple[str, ...] = tuple(c.id for c in COMPONENTS)
COMPONENT_BY_ID: dict[str, ComponentInfo] = {c.id: c for c in (*COMPONENTS, TELEA)}
EXTRA_FILENAME = "catalog_extra.json"

_BACKEND_TO_COMPONENT: dict[tuple[str, str], str] = {
    ("detector", "grounding-dino"): "detector:grounding-dino",
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


def max_quality_ready(catalog: ModelCatalog | None = None) -> bool:
    cat = catalog or ModelCatalog()
    return (
        backend_ready("detector", "grounding-dino", catalog=cat)
        and backend_ready("segmenter", "sam2-video", catalog=cat)
        and backend_ready("inpainter", "propainter", catalog=cat)
    )


class ModelCatalog:
    def __init__(self, jobs: JobIndex | None = None) -> None:
        self.jobs = jobs

    def _data_dir(self) -> Path:
        if self.jobs is not None:
            return Path(self.jobs.db_path).parent
        return resolve_data_dir()

    def list_infos(self) -> list[ComponentInfo]:
        infos: list[ComponentInfo] = [TELEA, *COMPONENTS]
        seen = {info.id for info in infos}
        for extra in load_extras(self._data_dir()):
            if extra.id not in seen:
                infos.append(extra)
                seen.add(extra.id)
        names = ollama_model_names() or []
        covered = {info.model_ref for info in infos if info.kind == "llm"}
        for name in names:
            base = name.split(":")[0]
            if name in covered or base in covered:
                continue
            cid = extra_id("llm", "ollama", name)
            if cid in seen:
                continue
            infos.append(
                ComponentInfo(
                    id=cid,
                    title=name,
                    kind="llm",
                    model_ref=name,
                    size_hint="",
                    backend="ollama",
                    source="ollama",
                )
            )
            seen.add(cid)
            covered.add(name)
        return infos

    def list_status(self) -> list[ComponentStatus]:
        active, failed = self._download_overlay()
        tags = ollama_tags()
        rows: list[ComponentStatus] = []
        for info in self.list_infos():
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
        info = resolve_component(component_id, data_dir=self._data_dir())
        if info is None:
            return False
        tags = ollama_tags() if info.kind == "llm" else None
        state, _ = self._probe(info, tags)
        return state == "ready"

    def _probe(self, info: ComponentInfo, tags: set[str] | None) -> tuple[str, str]:
        if info.id == "inpainter:opencv-telea":
            return "ready", "built-in, CPU"
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
        if info.kind == "inpainter":
            if hf_cached(info.model_ref):
                return "ready", f"{info.model_ref} cached"
            return "missing", f"{info.model_ref} not in local HF cache"
        if info.kind == "llm":
            if tags is None:
                return "error", "unavailable: start ollama"
            if info.model_ref in tags or info.model_ref.split(":")[0] in tags:
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
    names = ollama_model_names()
    if names is None:
        return None
    out: set[str] = set()
    for name in names:
        out.add(name)
        out.add(name.split(":")[0])
    return out


def ollama_model_names(*, timeout: float | None = None, force: bool = False) -> list[str] | None:
    """Exact names from `ollama list` /api/tags (for UI dropdowns). None if unreachable."""
    global _ollama_neg_until
    now = time.monotonic()
    with _ollama_neg_lock:
        if not force and now < _ollama_neg_until:
            return None
    url = OLLAMA_API + "/api/tags"
    req = urllib.request.Request(url, method="GET")
    wait = OLLAMA_TAGS_TIMEOUT_S if timeout is None else float(timeout)
    try:
        with urllib.request.urlopen(req, timeout=wait) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError, ValueError):
        with _ollama_neg_lock:
            _ollama_neg_until = time.monotonic() + OLLAMA_NEG_TTL_S
        return None
    names: list[str] = []
    seen: set[str] = set()
    for model in payload.get("models") or []:
        name = str(model.get("name") or model.get("model") or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        names.append(name)
    names.sort()
    with _ollama_neg_lock:
        _ollama_neg_until = 0.0
    return names


def extra_id(kind: str, backend: str, model_ref: str) -> str:
    return f"extra:{kind}:{backend}:{model_ref}"


def extras_path(data_dir: Path | None = None) -> Path:
    return Path(data_dir or resolve_data_dir()) / EXTRA_FILENAME


def load_extras(data_dir: Path | None = None) -> list[ComponentInfo]:
    path = extras_path(data_dir)
    if not path.is_file():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    out: list[ComponentInfo] = []
    if not isinstance(raw, list):
        return []
    for item in raw:
        if not isinstance(item, dict):
            continue
        cid = str(item.get("id") or "").strip()
        kind = str(item.get("kind") or "").strip().lower()
        model_ref = str(item.get("model_ref") or "").strip()
        if not cid or not kind or not model_ref:
            continue
        out.append(
            ComponentInfo(
                id=cid,
                title=str(item.get("title") or model_ref),
                kind=kind,
                model_ref=model_ref,
                size_hint=str(item.get("size_hint") or "HF snapshot"),
                backend=str(item.get("backend") or ""),
                source="extra",
            )
        )
    return out


def save_extras(infos: list[ComponentInfo], data_dir: Path | None = None) -> None:
    path = extras_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "id": info.id,
            "title": info.title,
            "kind": info.kind,
            "model_ref": info.model_ref,
            "size_hint": info.size_hint,
            "backend": info.backend,
        }
        for info in infos
    ]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def add_extra(
    *,
    kind: str,
    backend: str,
    model_ref: str,
    title: str = "",
    data_dir: Path | None = None,
) -> ComponentInfo:
    from videoclean.application.errors import PipelineError

    kind = (kind or "").strip().lower()
    backend = (backend or "").strip().lower()
    model_ref = (model_ref or "").strip()
    if kind not in {"detector", "segmenter", "inpainter", "llm"}:
        raise PipelineError("kind must be detector, segmenter, inpainter, or llm")
    if kind != "llm" and not backend:
        raise PipelineError("backend is required")
    if kind == "llm":
        backend = backend or "ollama"
    if not model_ref:
        raise PipelineError("model_ref is required")
    if kind in {"detector", "segmenter"} and "/" not in model_ref:
        raise PipelineError("model_ref must be a Hugging Face id like org/name")
    cid = extra_id(kind, backend, model_ref)
    info = ComponentInfo(
        id=cid,
        title=(title or "").strip() or model_ref,
        kind=kind,
        model_ref=model_ref,
        size_hint="HF snapshot" if kind != "llm" else "",
        backend=backend,
        source="extra",
    )
    existing = load_extras(data_dir)
    if any(row.id == cid for row in existing):
        return info
    existing.append(info)
    save_extras(existing, data_dir)
    return info


def backend_name(info: ComponentInfo) -> str:
    if info.backend:
        return info.backend
    parts = info.id.split(":")
    if len(parts) >= 2 and parts[0] == "segmenter":
        return "sam2"
    if len(parts) >= 2 and parts[0] == "llm":
        return "ollama"
    if len(parts) >= 2:
        return parts[1]
    return ""


def resolve_component(component_id: str, data_dir: Path | None = None) -> ComponentInfo | None:
    component_id = (component_id or "").strip()
    if not component_id:
        return None
    hit = COMPONENT_BY_ID.get(component_id)
    if hit is not None:
        return hit
    for extra in load_extras(data_dir):
        if extra.id == component_id:
            return extra
    if component_id.startswith("extra:llm:") or component_id.startswith("llm:ollama:"):
        ref = component_id.split(":", 2)[-1]
        if component_id.startswith("extra:llm:"):
            # extra:llm:ollama:tag
            ref = component_id.split(":", 3)[-1] if component_id.count(":") >= 3 else ref
        return ComponentInfo(
            id=component_id,
            title=ref,
            kind="llm",
            model_ref=ref,
            size_hint="",
            backend="ollama",
            source="ollama",
        )
    return None
