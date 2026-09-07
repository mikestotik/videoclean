from __future__ import annotations

import json
import os
import shutil
import threading
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from videoclean.adapters.models.catalog import (
    backend_name,
    backend_ready,
    max_quality_ready,
    ollama_model_names,
)
from videoclean.adapters.web.app_state import AppState
from videoclean.application.config import (
    DEFAULT_DETECTOR_MODEL,
    DEFAULT_GROUNDING_DINO_MODEL,
    DEFAULT_INPAINTER_MODEL,
    DEFAULT_SEGMENTER_MODEL,
    DETECTORS,
    INPAINTERS,
    LLM_PLACES,
    SEGMENTERS,
)
from videoclean.application.errors import PipelineError
from videoclean.progress import STAGES, _fmt_seconds
from videoclean.store import new_job_id

_doctor_cache: tuple[float, dict[str, str]] | None = None
_DOCTOR_TTL_S = 60.0


def auth_from_env(env: Mapping[str, str] | None = None) -> tuple[str, str]:
    env = os.environ if env is None else env
    password = str(env.get("VIDEOCLEAN_UI_PASSWORD") or "").strip()
    if not password:
        raise RuntimeError(
            "VIDEOCLEAN_UI_PASSWORD is required to launch the UI "
            "(safer default for RunPod). Set VIDEOCLEAN_UI_USER and "
            "VIDEOCLEAN_UI_PASSWORD before starting."
        )
    user = str(env.get("VIDEOCLEAN_UI_USER") or "admin").strip() or "admin"
    return (user, password)


def api_token(env: Mapping[str, str] | None = None) -> str:
    env = os.environ if env is None else env
    token = str(env.get("VIDEOCLEAN_API_TOKEN") or "").strip()
    if token:
        return token
    return auth_from_env(env)[1]


def default_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


def serialize_clean_form(payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
    data = dict(payload or {})
    detector = str(data.get("detector") or "grounding-dino").strip().lower()
    segmenter = str(data.get("segmenter") or "sam2").strip().lower()
    inpainter = str(data.get("inpainter") or "opencv-telea").strip().lower()
    detector_model = str(data.get("detector_model") or "").strip()
    segmenter_model = str(data.get("segmenter_model") or "").strip()
    inpainter_model = str(data.get("inpainter_model") or "").strip()
    if not detector_model:
        detector_model = (
            DEFAULT_DETECTOR_MODEL if detector == "owlvit" else DEFAULT_GROUNDING_DINO_MODEL
        )
    if not segmenter_model:
        segmenter_model = DEFAULT_SEGMENTER_MODEL
    if not inpainter_model:
        inpainter_model = DEFAULT_INPAINTER_MODEL
    fmt_name = str(data.get("fmt") or data.get("format") or "mp4").strip().lower() or "mp4"
    return {
        "device": str(data.get("device") or "cpu").strip().lower() or "cpu",
        "detector": detector,
        "detectors": [detector] if detector else [],
        "detector_model": detector_model,
        "detector_threshold": float(data.get("detector_threshold") or 0.15),
        "segmenter": segmenter,
        "segmenter_model": segmenter_model,
        "inpainter": inpainter,
        "inpainter_model": inpainter_model,
        "llm_place": str(data.get("llm_place") or "auto").strip().lower() or "auto",
        "llm_model": str(data.get("llm_model") or "").strip(),
        "formats": [fmt_name],
        "allow_download": False,
        "verify": _as_bool(data.get("verify"), True),
        "mask_dilate_px": int(data.get("mask_dilate_px") or 3),
        "telea_radius": int(data.get("telea_radius") or 9),
        "prompt_frame_stride": int(data.get("prompt_frame_stride") or 4),
        "prompt_frame_max": int(data.get("prompt_frame_max") or 8),
        "overwrite": _as_bool(data.get("overwrite"), True),
    }


def queue_clean_job(
    state: AppState,
    src: Path,
    prompt: str,
    request: dict[str, Any],
    original_name: str = "",
) -> str:
    prompt = (prompt or "").strip()
    if not prompt:
        raise PipelineError("prompt is required")
    if src is None or not src.is_file():
        raise PipelineError("upload a video file first")
    job_id = new_job_id()
    dest_dir = Path(state.data_dir) / "uploads" / job_id / "input"
    dest_dir.mkdir(parents=True, exist_ok=True)
    name = original_name or src.name
    dest = dest_dir / Path(name).name
    if src.resolve() != dest.resolve():
        shutil.copy2(src, dest)
    suffix = dest.suffix.lower() or ".mp4"
    output_path = Path(state.data_dir) / "jobs" / job_id / "output" / f"cleaned{suffix}"
    payload = dict(request or {})
    payload["allow_download"] = False
    payload["input_path"] = str(dest)
    payload["output_path"] = str(output_path)
    payload["prompt"] = prompt
    return state.manage.submit(payload, dest, output_path, prompt, job_id=job_id)


def job_dict(state: AppState, row) -> dict[str, Any]:
    job_id = row["id"]
    progress = _as_dict(row["progress_json"] if "progress_json" in row.keys() else None)
    request = _as_dict(row["request_json"] if "request_json" in row.keys() else None)
    output_path = Path(row["output_path"] or "") if row["output_path"] else None
    input_path = Path(row["input_path"] or "") if row["input_path"] else None
    state_name = row["state"] or ""
    has_output = bool(output_path and output_path.is_file() and state_name == "COMPLETED")
    has_input = bool(input_path and input_path.is_file())
    return {
        "id": job_id,
        "state": state_name,
        "prompt": row["prompt"] or "",
        "created_at": row["created_at"],
        "updated_at": row["updated_at"] if "updated_at" in row.keys() else row["created_at"],
        "error": (row["error"] if "error" in row.keys() else None) or "",
        "progress": progress,
        "stage": progress.get("stage") or "",
        "fraction": _frac(progress.get("fraction")),
        "detail": progress.get("detail") or "",
        "eta": eta_label(row),
        "stages": _stage_marks(progress.get("stage") or ""),
        "request": {
            "device": request.get("device"),
            "detector": request.get("detector"),
            "detector_model": request.get("detector_model"),
            "segmenter": request.get("segmenter"),
            "segmenter_model": request.get("segmenter_model"),
            "inpainter": request.get("inpainter"),
            "inpainter_model": request.get("inpainter_model"),
            "llm_place": request.get("llm_place"),
            "llm_model": request.get("llm_model"),
        },
        "has_output": has_output,
        "has_input": has_input,
        "output_url": f"/api/jobs/{job_id}/output" if has_output else None,
        "input_url": f"/api/jobs/{job_id}/input" if has_input else None,
        "status_url": f"/api/jobs/{job_id}",
    }


def eta_label(row, now: datetime | None = None) -> str:
    payload = _as_dict(row["progress_json"] if "progress_json" in row.keys() else None)
    frac = _frac(payload.get("fraction"))
    started = _parse_ts(payload.get("started_at")) or _parse_ts(row["created_at"])
    if started is None:
        return ""
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    elapsed = max(0.0, (current - started).total_seconds())
    if frac < 0.08:
        return ""
    remaining = max(0.0, elapsed / max(frac, 1e-6) - elapsed)
    return _fmt_seconds(remaining)


def grouped_models(state: AppState) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {
        "detector": [],
        "segmenter": [],
        "inpainter": [],
        "llm": [],
    }
    downloads = {row["component_id"]: row for row in _active_downloads(state)}
    for status in _safe_list_status(state.catalog):
        info = status.info
        row = downloads.get(info.id)
        frac = 0.0
        if row is not None:
            try:
                frac = float(row["progress"] or 0.0)
            except (TypeError, ValueError):
                frac = 0.0
        groups.setdefault(info.kind, []).append(
            {
                "id": info.id,
                "title": info.title,
                "kind": info.kind,
                "backend": backend_name(info),
                "model_ref": info.model_ref,
                "size_hint": info.size_hint,
                "state": status.state,
                "message": status.message,
                "source": info.source,
                "progress": max(0.0, min(frac, 1.0)),
                "downloadable": info.id != "inpainter:opencv-telea",
            }
        )
    return groups


def options_payload(state: AppState) -> dict[str, Any]:
    catalog = state.catalog
    llm_names = ollama_model_names(timeout=2.0) or []
    return {
        "device": default_device(),
        "devices": ["cpu", "cuda", "mps"],
        "formats": ["mp4", "mov", "mkv", "webm"],
        "llm_places": list(LLM_PLACES),
        "llm_models": llm_names,
        "detectors": [name for name in DETECTORS if backend_ready("detector", name, catalog=catalog)],
        "segmenters": [
            name for name in SEGMENTERS if backend_ready("segmenter", name, catalog=catalog)
        ],
        "inpainters": [
            name for name in INPAINTERS if backend_ready("inpainter", name, catalog=catalog)
        ],
        "max_quality_ready": max_quality_ready(catalog),
        "models": grouped_models(state),
    }


def doctor_payload(*, force: bool = False) -> dict[str, str]:
    global _doctor_cache
    import time

    now = time.monotonic()
    if not force and _doctor_cache is not None and (now - _doctor_cache[0]) < _DOCTOR_TTL_S:
        return _doctor_cache[1]
    from videoclean.composition import machine_facts

    facts = machine_facts()
    _doctor_cache = (now, facts)
    return facts


def start_download(state: AppState, component_id: str) -> str:
    if state.downloader is None:
        raise PipelineError("downloader is not configured")
    component_id = (component_id or "").strip()
    if not component_id:
        raise PipelineError("component_id is required")
    active = _active_downloads(state)
    if active:
        raise PipelineError(f"already downloading {active[0]['component_id']}")
    downloader = state.downloader

    def _run() -> None:
        try:
            downloader.execute(component_id, state.jobs, None)
        except Exception:  # noqa: BLE001
            return

    threading.Thread(target=_run, name=f"vc-dl-{component_id}", daemon=True).start()
    return f"starting {component_id}"


def cancel_downloads(state: AppState) -> list[str]:
    cancelled: list[str] = []
    for row in state.jobs.list_downloads(limit=20):
        if row["state"] in {"running", "queued"}:
            state.jobs.request_download_cancel(row["id"])
            state.jobs.upsert_download(
                row["id"],
                row["component_id"],
                "cancelled",
                progress=0.0,
                message="cancelled",
            )
            cancelled.append(row["component_id"])
    return cancelled


def downloads_payload(state: AppState) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in state.jobs.list_downloads(limit=20):
        out.append(
            {
                "id": row["id"],
                "component_id": row["component_id"],
                "state": row["state"],
                "progress": float(row["progress"] or 0.0),
                "bytes_done": row["bytes_done"],
                "bytes_total": row["bytes_total"],
                "message": row["message"] or "",
            }
        )
    return out


def _active_downloads(state: AppState) -> list:
    rows: list = []
    for st in ("running", "queued"):
        rows.extend(state.jobs.list_downloads(limit=20, state=st))
    return rows


def _safe_list_status(catalog) -> list:
    try:
        return list(catalog.list_status())
    except Exception:  # noqa: BLE001
        return []


def _stage_marks(current: str) -> list[dict[str, str]]:
    keys = [key for key, _title, _w in STAGES]
    out: list[dict[str, str]] = []
    for key, title, _w in STAGES:
        if key == current:
            mark = "current"
        elif current in keys and keys.index(key) < keys.index(current):
            mark = "done"
        else:
            mark = "todo"
        out.append({"id": key, "title": title, "mark": mark})
    return out


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        data = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _as_bool(value: Any, default: bool) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _frac(value: Any) -> float:
    try:
        return max(0.0, min(float(value or 0.0), 1.0))
    except (TypeError, ValueError):
        return 0.0


def _parse_ts(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        ts = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts
