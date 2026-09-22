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

# Segmenter weights: driver is sam2 | sam2-video; these are the HF checkpoints.
SEGMENTER_COMPONENTS: tuple[ComponentInfo, ...] = (
    ComponentInfo(
        id="segmenter:sam2-tiny",
        title="SAM2 tiny",
        kind="segmenter",
        model_ref="facebook/sam2-hiera-tiny",
        size_hint="~160 MB",
    ),
    ComponentInfo(
        id="segmenter:sam2-small",
        title="SAM2 small",
        kind="segmenter",
        model_ref="facebook/sam2-hiera-small",
        size_hint="~180 MB",
    ),
    ComponentInfo(
        id="segmenter:sam2-base-plus",
        title="SAM2 base+",
        kind="segmenter",
        model_ref="facebook/sam2-hiera-base-plus",
        size_hint="~320 MB",
    ),
    ComponentInfo(
        id="segmenter:sam2-large",
        title="SAM2 large",
        kind="segmenter",
        model_ref="facebook/sam2-hiera-large",
        size_hint="~900 MB",
    ),
    ComponentInfo(
        id="segmenter:sam21-tiny",
        title="SAM2.1 tiny",
        kind="segmenter",
        model_ref="facebook/sam2.1-hiera-tiny",
        size_hint="~160 MB",
    ),
    ComponentInfo(
        id="segmenter:sam21-small",
        title="SAM2.1 small",
        kind="segmenter",
        model_ref="facebook/sam2.1-hiera-small",
        size_hint="~185 MB",
    ),
    ComponentInfo(
        id="segmenter:sam21-base-plus",
        title="SAM2.1 base+",
        kind="segmenter",
        model_ref="facebook/sam2.1-hiera-base-plus",
        size_hint="~320 MB",
    ),
    ComponentInfo(
        id="segmenter:sam21-large",
        title="SAM2.1 large",
        kind="segmenter",
        model_ref="facebook/sam2.1-hiera-large",
        size_hint="~900 MB",
    ),
)

SEGMENTER_MODEL_REFS: tuple[str, ...] = tuple(c.model_ref for c in SEGMENTER_COMPONENTS)
_SEGMENTER_REF_TO_ID: dict[str, str] = {c.model_ref.casefold(): c.id for c in SEGMENTER_COMPONENTS}

COMPONENTS: tuple[ComponentInfo, ...] = (
    ComponentInfo(
        id="detector:grounding-dino",
        title="Grounding DINO tiny",
        kind="detector",
        model_ref="IDEA-Research/grounding-dino-tiny",
        size_hint="~650 MB",
    ),
    *SEGMENTER_COMPONENTS,
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
)

COMPONENT_IDS: tuple[str, ...] = tuple(c.id for c in COMPONENTS)
COMPONENT_BY_ID: dict[str, ComponentInfo] = {c.id: c for c in COMPONENTS}
EXTRA_FILENAME = "catalog_extra.json"
PROVIDERS_FILENAME = "llm_providers.json"

# Supported families for Settings → Add. Soft-check model_ref against hints.
FAMILIES: dict[str, list[dict[str, object]]] = {
    "detector": [
        {
            "id": "grounding-dino",
            "label": "Grounding DINO",
            "ref_kind": "hf",
            "hints": ("grounding-dino", "groundingdino"),
            "example": "IDEA-Research/grounding-dino-base",
        }
    ],
    "segmenter": [
        {
            "id": "sam2",
            "label": "SAM2 / SAM2.1",
            "ref_kind": "hf",
            "hints": ("sam2", "sam2.1", "sam21"),
            "example": "facebook/sam2.1-hiera-small",
        }
    ],
    "inpainter": [
        {
            "id": "propainter",
            "label": "ProPainter weights",
            "ref_kind": "hf",
            "hints": ("propainter",),
            "example": "camenduru/ProPainter",
        }
    ],
    "llm": [
        {
            "id": "ollama",
            "label": "Ollama",
            "ref_kind": "ollama",
            "hints": (),
            "example": "qwen2.5vl:3b",
        },
        {
            "id": "openai_compat",
            "label": "OpenAI-compatible",
            "ref_kind": "provider",
            "hints": (),
            "example": "gpt-4o-mini",
        },
    ],
}

_BACKEND_TO_COMPONENT: dict[tuple[str, str], str] = {
    ("detector", "grounding-dino"): "detector:grounding-dino",
    ("inpainter", "lama"): "inpainter:lama",
    ("inpainter", "propainter"): "inpainter:propainter",
}


def component_id_for(kind: str, name: str, segmenter_model: str = "") -> str | None:
    """Map a Clean-tab backend name to a catalog component id."""
    kind = (kind or "").strip().lower()
    name = (name or "").strip().lower()
    if kind == "segmenter" and name in {"sam2", "sam2-video"}:
        return segmenter_component_id(segmenter_model)
    return _BACKEND_TO_COMPONENT.get((kind, name))


def segmenter_component_id(segmenter_model: str = "") -> str:
    """Resolve HF segmenter checkpoint → catalog id (default tiny)."""
    model = (segmenter_model or "").strip().casefold()
    if model in _SEGMENTER_REF_TO_ID:
        return _SEGMENTER_REF_TO_ID[model]
    # Fuzzy match for custom / abbreviated refs.
    if "2.1" in model or "sam21" in model or "sam2.1" in model:
        if "large" in model:
            return "segmenter:sam21-large"
        if "base" in model:
            return "segmenter:sam21-base-plus"
        if "small" in model:
            return "segmenter:sam21-small"
        return "segmenter:sam21-tiny"
    if "large" in model:
        return "segmenter:sam2-large"
    if "base" in model:
        return "segmenter:sam2-base-plus"
    if "small" in model:
        return "segmenter:sam2-small"
    return "segmenter:sam2-tiny"


def backend_ready(
    kind: str,
    name: str,
    catalog: ModelCatalog | None = None,
    segmenter_model: str = "",
) -> bool:
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
        """Builtin seed + user extras. Ollama tags are not auto-listed; add via Settings."""
        infos: list[ComponentInfo] = list(COMPONENTS)
        seen = {info.id for info in infos}
        for extra in load_extras(self._data_dir()):
            if extra.id not in seen:
                infos.append(extra)
                seen.add(extra.id)
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
        if info.kind in {"detector", "segmenter"}:
            if hf_cached(info.model_ref):
                return "ready", f"{info.model_ref} cached"
            return "missing", f"{info.model_ref} not in local HF cache"
        backend = backend_name(info)
        if info.id == "inpainter:lama" or (info.kind == "inpainter" and backend == "lama"):
            path = find_lama_weights()
            if path is not None:
                return "ready", f"weights on disk ({path.name})"
            return "missing", "big-lama.pt not found"
        if info.id == "inpainter:propainter" or (info.kind == "inpainter" and backend == "propainter"):
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


def family_backends(kind: str) -> list[dict[str, object]]:
    return list(FAMILIES.get((kind or "").strip().lower(), []))


def validate_family_ref(kind: str, backend: str, model_ref: str) -> None:
    """Soft family check: HF ids must look like the adapter family we can load."""
    from videoclean.application.errors import PipelineError

    kind = (kind or "").strip().lower()
    backend = (backend or "").strip().lower()
    model_ref = (model_ref or "").strip()
    families = family_backends(kind)
    if not families:
        raise PipelineError(f"unsupported kind {kind!r}")
    match = next((f for f in families if str(f.get("id")) == backend), None)
    if match is None:
        known = ", ".join(str(f.get("id")) for f in families)
        raise PipelineError(f"unsupported backend {backend!r} for {kind}. known: {known}")
    ref_kind = str(match.get("ref_kind") or "")
    if ref_kind == "hf":
        if "/" not in model_ref:
            raise PipelineError("model_ref must be a Hugging Face id like org/name")
        hints = tuple(str(h).casefold() for h in (match.get("hints") or ()))  # type: ignore[arg-type]
        folded = model_ref.casefold()
        if hints and not any(h in folded for h in hints):
            example = match.get("example") or "org/name"
            raise PipelineError(
                f"{backend} expects a {backend}-family HF id (e.g. {example}), got {model_ref!r}"
            )
    elif ref_kind == "ollama":
        if "/" in model_ref and model_ref.count("/") == 1 and ":" not in model_ref.split("/")[-1]:
            # Reject accidental HF ids for ollama tags.
            raise PipelineError("ollama model_ref must be an Ollama tag, not a Hugging Face id")
    elif ref_kind == "provider":
        raise PipelineError("openai_compat models belong to a provider, not the model catalog")


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
    if kind not in FAMILIES:
        raise PipelineError("kind must be detector, segmenter, inpainter, or llm")
    if kind == "llm":
        backend = backend or "ollama"
    if not backend:
        raise PipelineError("backend is required")
    if not model_ref:
        raise PipelineError("model_ref is required")
    validate_family_ref(kind, backend, model_ref)
    cid = extra_id(kind, backend, model_ref)
    size = ""
    if kind in {"detector", "segmenter"} or (kind == "inpainter" and backend == "propainter"):
        size = "HF snapshot"
    info = ComponentInfo(
        id=cid,
        title=(title or "").strip() or model_ref,
        kind=kind,
        model_ref=model_ref,
        size_hint=size,
        backend=backend,
        source="extra",
    )
    existing = load_extras(data_dir)
    if any(row.id == cid for row in existing):
        return info
    existing.append(info)
    save_extras(existing, data_dir)
    return info


def remove_extra(component_id: str, data_dir: Path | None = None) -> bool:
    from videoclean.application.errors import PipelineError

    component_id = (component_id or "").strip()
    if not component_id:
        raise PipelineError("id is required")
    if component_id in COMPONENT_BY_ID:
        raise PipelineError("builtin catalog entries cannot be removed")
    existing = load_extras(data_dir)
    kept = [row for row in existing if row.id != component_id]
    if len(kept) == len(existing):
        return False
    save_extras(kept, data_dir)
    return True


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


def providers_path(data_dir: Path | None = None) -> Path:
    return Path(data_dir or resolve_data_dir()) / PROVIDERS_FILENAME


def load_providers(data_dir: Path | None = None) -> list[dict[str, object]]:
    path = providers_path(data_dir)
    if not path.is_file():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(raw, list):
        return []
    out: list[dict[str, object]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        pid = str(item.get("id") or "").strip()
        base_url = str(item.get("base_url") or "").strip()
        if not pid or not base_url:
            continue
        models_raw = item.get("models") or []
        models = [str(m).strip() for m in models_raw if str(m).strip()] if isinstance(models_raw, list) else []
        out.append(
            {
                "id": pid,
                "title": str(item.get("title") or pid).strip() or pid,
                "kind": "openai_compat",
                "base_url": base_url.rstrip("/"),
                "api_key": str(item.get("api_key") or ""),
                "models": models,
            }
        )
    return out


def save_providers(rows: list[dict[str, object]], data_dir: Path | None = None) -> None:
    path = providers_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "id": row["id"],
            "title": row.get("title") or row["id"],
            "base_url": row["base_url"],
            "api_key": row.get("api_key") or "",
            "models": list(row.get("models") or []),
        }
        for row in rows
    ]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def add_provider(
    *,
    title: str,
    base_url: str,
    api_key: str = "",
    models: list[str] | None = None,
    provider_id: str = "",
    data_dir: Path | None = None,
) -> dict[str, object]:
    from videoclean.application.errors import PipelineError

    base_url = (base_url or "").strip().rstrip("/")
    if not base_url:
        raise PipelineError("base_url is required")
    if "://" not in base_url:
        raise PipelineError("base_url must include scheme, e.g. http://127.0.0.1:8000/v1")
    title = (title or "").strip() or base_url
    if (provider_id or "").strip():
        pid = provider_id.strip()
    else:
        slug = "".join(ch if ch.isalnum() else "-" for ch in base_url.lower())
        slug = "-".join(p for p in slug.split("-") if p)[:48] or "provider"
        pid = f"openai:{slug}"
    clean_models = [m.strip() for m in (models or []) if m and str(m).strip()]
    row: dict[str, object] = {
        "id": pid,
        "title": title,
        "kind": "openai_compat",
        "base_url": base_url,
        "api_key": (api_key or "").strip(),
        "models": clean_models,
    }
    existing = load_providers(data_dir)
    for i, prev in enumerate(existing):
        if prev["id"] == pid or prev["base_url"] == base_url:
            existing[i] = row
            save_providers(existing, data_dir)
            return row
    existing.append(row)
    save_providers(existing, data_dir)
    return row


def remove_provider(provider_id: str, data_dir: Path | None = None) -> bool:
    from videoclean.application.errors import PipelineError

    provider_id = (provider_id or "").strip()
    if not provider_id:
        raise PipelineError("id is required")
    existing = load_providers(data_dir)
    kept = [row for row in existing if row["id"] != provider_id]
    if len(kept) == len(existing):
        return False
    save_providers(kept, data_dir)
    return True


def families_payload() -> dict[str, list[dict[str, object]]]:
    """UI metadata for Add-model forms."""
    out: dict[str, list[dict[str, object]]] = {}
    for kind, backends in FAMILIES.items():
        out[kind] = [
            {
                "id": b["id"],
                "label": b["label"],
                "ref_kind": b["ref_kind"],
                "example": b.get("example") or "",
            }
            for b in backends
        ]
    return out
