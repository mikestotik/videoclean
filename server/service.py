from __future__ import annotations

import json
import os
import shutil
import threading
import uuid
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
from server.app_state import AppState
from videoclean.application.config import (
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
from videoclean.store import new_job_id, new_source_id, utc_now

VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}

_doctor_cache: tuple[float, dict[str, str]] | None = None
_DOCTOR_TTL_S = 60.0


def _profiles_for_options(device: str) -> list[dict[str, Any]]:
    from videoclean.application.profiles import profiles_payload

    return profiles_payload(device)


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
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


def serialize_clean_form(payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
    data = dict(payload or {})
    detector = str(data.get("detector") or "grounding-dino").strip().lower()
    segmenter = str(data.get("segmenter") or "sam2").strip().lower()
    inpainter = str(data.get("inpainter") or "lama").strip().lower()
    detector_model = str(data.get("detector_model") or "").strip()
    segmenter_model = str(data.get("segmenter_model") or "").strip()
    inpainter_model = str(data.get("inpainter_model") or "").strip()
    if not detector_model:
        detector_model = DEFAULT_GROUNDING_DINO_MODEL
    if not segmenter_model:
        segmenter_model = DEFAULT_SEGMENTER_MODEL
    if not inpainter_model:
        inpainter_model = DEFAULT_INPAINTER_MODEL
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
        "formats": [
            f.strip().lower()
            for f in str(data.get("formats") or data.get("fmt") or "mp4").split(",")
            if f.strip()
        ] or ["mp4"],
        "llm_base_url": str(data.get("llm_base_url") or "").strip(),
        "llm_api_key": str(data.get("llm_api_key") or "").strip(),
        "keep_workdir": _as_bool(data.get("keep_workdir"), False),
        "min_mask_coverage": (
            float(data["min_mask_coverage"]) if str(data.get("min_mask_coverage") or "").strip() else 0.0004
        ),
        "verify_max_coverage": (
            float(data["verify_max_coverage"]) if str(data.get("verify_max_coverage") or "").strip() else 0.12
        ),
        "allow_download": False,
        "verify": _as_bool(data.get("verify"), True),
        "mask_dilate_px": int(data.get("mask_dilate_px") or 3),
        "prompt_frame_stride": int(data.get("prompt_frame_stride") or 4),
        "prompt_frame_max": int(data.get("prompt_frame_max") or 8),
        "parse_chunk_frames": int(data.get("parse_chunk_frames") or 0),
        "vision_batch": int(data.get("vision_batch") or 2),
        "detector_keyframes": (
            int(data["detector_keyframes"]) if str(data.get("detector_keyframes") or "").strip() else None
        ),
        "detector_nms_iou": float(data.get("detector_nms_iou") or 0.3),
        "detector_max_box_area": float(data.get("detector_max_box_area") or 0.25),
        "tracker_min_score": float(data.get("tracker_min_score") or 0.55),
        "tracker_max_template_area": float(data.get("tracker_max_template_area") or 0.12),
        "propainter_mask_dilation": int(data.get("propainter_mask_dilation") or 4),
        "propainter_ref_stride": int(data.get("propainter_ref_stride") or 10),
        "propainter_neighbor_length": int(data.get("propainter_neighbor_length") or 10),
        "propainter_subvideo_length": int(data.get("propainter_subvideo_length") or 80),
        "propainter_raft_iter": int(data.get("propainter_raft_iter") or 20),
        "profile": str(data.get("profile") or "custom").strip().lower() or "custom",
        "verify_max_passes": (
            int(data["verify_max_passes"])
            if str(data.get("verify_max_passes") or "").strip()
            else 1
        ),
        "inpaint_workers": (
            int(data["inpaint_workers"])
            if str(data.get("inpaint_workers") or "").strip()
            else 0
        ),
        "inpaint_chunk_overlap": (
            int(data["inpaint_chunk_overlap"])
            if str(data.get("inpaint_chunk_overlap") or "").strip()
            else 8
        ),
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


def queue_preview_job(
    state: AppState,
    src: Path,
    prompt: str,
    request: dict[str, Any],
    original_name: str = "",
) -> str:
    prompt = (prompt or "").strip()
    mode = str((request or {}).get("mode") or "parse")
    if mode not in {"parse", "detect"}:
        raise PipelineError("preview mode must be parse | detect")
    if mode == "parse" and not prompt:
        raise PipelineError("prompt is required for mode=parse")
    if mode == "detect" and not request.get("targets"):
        raise PipelineError("targets are required for mode=detect")
    if src is None or not src.is_file():
        raise PipelineError("upload a video file first")
    job_id = new_job_id()
    dest_dir = Path(state.data_dir) / "uploads" / job_id / "input"
    dest_dir.mkdir(parents=True, exist_ok=True)
    name = original_name or src.name
    dest = dest_dir / Path(name).name
    if src.resolve() != dest.resolve():
        shutil.copy2(src, dest)
    payload = dict(request or {})
    payload["kind"] = "preview"
    payload["input_path"] = str(dest)
    payload["prompt"] = prompt
    output_dir = Path(state.data_dir) / "jobs" / job_id / "output"
    payload["output_path"] = str(output_dir / "preview")
    return state.manage.submit(payload, dest, output_dir, prompt, job_id=job_id)


def queue_source_run(
    state: AppState,
    source_row,
    prompt: str,
    request: dict[str, Any],
) -> str:
    """Queue a full cleanup referencing the source video in place."""
    src = Path(source_row["path"])
    if not src.is_file():
        raise PipelineError("source file missing")
    prompt = (prompt or "").strip()
    if not prompt and not request.get("targets_override") and not request.get("tracks_override") and not request.get("masks"):
        raise PipelineError("prompt is required (or targets/tracks/masks)")
    job_id = new_job_id()
    suffix = src.suffix.lower() or ".mp4"
    output_path = Path(state.data_dir) / "jobs" / job_id / "output" / f"cleaned{suffix}"
    payload = dict(request or {})
    payload["kind"] = "run"
    payload["allow_download"] = False
    payload["input_path"] = str(src)
    payload["output_path"] = str(output_path)
    payload["prompt"] = prompt
    payload["source_id"] = source_row["id"]
    frames = payload.pop("masks", None)
    if frames:
        masks_dir = Path(state.data_dir) / "sources" / source_row["id"] / "masks"
        paths = [masks_dir / f"{int(n):06d}.png" for n in frames]
        missing = next((p for p in paths if not p.is_file()), None)
        if missing is not None:
            raise PipelineError(f"mask for frame {missing.stem} not found on source")
        payload["masks_override"] = [str(p) for p in paths]
    return state.manage.submit(payload, src, output_path, prompt, job_id=job_id, source_id=source_row["id"])


def queue_source_preview(state: AppState, source_row, prompt: str, request: dict[str, Any]) -> str:
    """Queue a preview job against a registered source (in place, no copy)."""
    src = Path(source_row["path"])
    if not src.is_file():
        raise PipelineError("source file missing")
    prompt = (prompt or "").strip()
    mode = str((request or {}).get("mode") or "parse")
    targets = request.get("targets") or request.get("targets_override")
    if mode not in {"parse", "detect"}:
        raise PipelineError("preview mode must be parse | detect")
    if mode == "parse" and not prompt:
        raise PipelineError("prompt is required for mode=parse")
    if mode == "detect" and not targets:
        raise PipelineError("targets are required for mode=detect")
    job_id = new_job_id()
    payload = dict(request or {})
    payload["kind"] = "preview"
    if mode == "detect":
        payload["targets"] = targets
    payload["input_path"] = str(src)
    payload["prompt"] = prompt
    payload["source_id"] = source_row["id"]
    output_dir = Path(state.data_dir) / "jobs" / job_id / "output"
    payload["output_path"] = str(output_dir / "preview")
    if payload.pop("all", None):
        probe = _as_dict(source_row["probe_json"])
        payload["start"] = 0
        payload["count"] = int(probe.get("frame_count") or 0)
    payload["segmenter"] = "sam2"
    return state.manage.submit(payload, src, output_dir, prompt, job_id=job_id, source_id=source_row["id"])


def queue_source_prompt(state: AppState, source_row, prompt: str, request: dict[str, Any]) -> str:
    """Queue a prompt-interpretation job over the masks drawn on a source."""
    src = Path(source_row["path"])
    if not src.is_file():
        raise PipelineError("source file missing")
    masks_dir = Path(state.data_dir) / "sources" / source_row["id"] / "masks"
    annotations = [{"frame": int(p.stem), "mask": str(p)} for p in sorted(masks_dir.glob("*.png"))]
    prompt = (prompt or "").strip()
    if not prompt and not annotations:
        raise PipelineError("provide a text prompt or draw at least one mask on the source")
    job_id = new_job_id()
    payload = dict(request or {})
    payload["kind"] = "prompt"
    payload["input_path"] = str(src)
    payload["prompt"] = prompt
    payload["annotations"] = annotations
    payload["source_id"] = source_row["id"]
    output_dir = Path(state.data_dir) / "jobs" / job_id / "output"
    payload["output_path"] = str(output_dir)
    return state.manage.submit(payload, src, output_dir, prompt, job_id=job_id, source_id=source_row["id"])


def queue_preview_from_job(
    state: AppState,
    job_id: str,
    request: dict[str, Any],
    prompt: str = "",
) -> str:
    """Queue a preview run reusing the input video of an existing job."""
    row = state.jobs.get(job_id)
    if row is None:
        raise PipelineError(f"unknown job {job_id}")
    src = Path(row["input_path"] or "")
    if not src.is_file():
        raise PipelineError("source job has no input file")
    return queue_preview_job(state, src, prompt, request, original_name=src.name)


def preview_artifact_path(state: AppState, job_id: str, name: str) -> Path | None:
    """Resolve a preview artifact; reject traversal and unknown names."""
    if not name or "/" in name or "\\" in name or ".." in name:
        return None
    allowed = {"preview.json"}
    stem = name.rsplit(".", 1)[-1].lower()
    is_img = name.endswith(".jpg")
    is_mp4 = name.endswith(".mp4")
    if not is_img and not is_mp4 and name not in allowed:
        return None
    row = state.jobs.get(job_id)
    if row is None:
        return None
    root = Path(state.data_dir) / "jobs" / job_id / "preview"
    path = root / name
    if not path.is_file() or path.parent != root:
        return None
    return path


def _presets_file(data_dir: Path) -> Path:
    return Path(data_dir) / "presets.json"


def list_presets(data_dir: Path) -> list[dict[str, Any]]:
    f = _presets_file(data_dir)
    if not f.is_file():
        return []
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return data if isinstance(data, list) else []


def save_preset(data_dir: Path, name: str, payload: Any) -> dict[str, Any]:
    name = (name or "").strip()
    if not name:
        raise PipelineError("preset name is required")
    if not isinstance(payload, dict):
        raise PipelineError("preset payload must be an object of pipeline fields")
    presets = list_presets(data_dir)
    item = {
        "id": f"p_{uuid.uuid4().hex[:8]}",
        "name": name,
        "payload": payload,
        "createdAt": utc_now().isoformat(),
    }
    presets.append(item)
    _presets_file(data_dir).write_text(json.dumps(presets, ensure_ascii=False, indent=2), encoding="utf-8")
    return item


def delete_preset(data_dir: Path, preset_id: str) -> bool:
    presets = list_presets(data_dir)
    rest = [p for p in presets if p.get("id") != preset_id]
    if len(rest) == len(presets):
        return False
    _presets_file(data_dir).write_text(json.dumps(rest, ensure_ascii=False, indent=2), encoding="utf-8")
    return True


def source_dict(row) -> dict[str, Any]:
    probe = _as_dict(row["probe_json"])
    return {
        "id": row["id"],
        "name": row["name"],
        "createdAt": row["created_at"],
        "probe": probe,
        "video_url": f"/api/sources/{row['id']}/video",
        "annotations_url": f"/api/sources/{row['id']}/annotations",
    }


def register_source(state: AppState, tmp: Path, original_name: str) -> str:
    """Store an uploaded video under sources/{id}/ and probe it."""
    from videoclean.adapters.media.ffmpeg import FFmpegMedia

    suffix = Path(original_name).suffix.lower()
    if suffix not in VIDEO_SUFFIXES:
        raise PipelineError(f"unsupported video type {suffix}")
    source_id = new_source_id()
    dest_dir = Path(state.data_dir) / "sources" / source_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"input{suffix}"
    shutil.copy2(tmp, dest)
    try:
        m = FFmpegMedia().probe(dest)
    except Exception:  # noqa: BLE001
        shutil.rmtree(dest_dir, ignore_errors=True)
        raise
    probe = {
        "fps": m.fps, "duration_s": m.duration_s, "width": m.width, "height": m.height,
        "frame_count": m.frame_count, "has_audio": m.has_audio,
    }
    state.sources.register(source_id, Path(original_name).name or dest.name, str(dest), probe=probe)
    return source_id


def source_frame_path(state: AppState, source_row, n: int) -> Path | None:
    """Extract one frame as JPEG; cached on disk. None if out of range or ffmpeg fails."""
    probe = _as_dict(source_row["probe_json"])
    fc = int(probe.get("frame_count") or 0)
    if n < 0 or (fc and n >= fc):
        return None
    src = Path(source_row["path"])
    out = src.parent / "frames" / f"{n:06d}.jpg"
    if out.is_file():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    fps = float(probe.get("fps") or 0) or 25.0
    import subprocess

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{n / fps:.6f}", "-i", str(src),
        "-frames:v", "1", "-q:v", "2", str(out),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0 or not out.is_file():
        out.unlink(missing_ok=True)
        return None
    return out


PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def source_masks_dir(state: AppState, source_row) -> Path:
    return Path(state.data_dir) / "sources" / source_row["id"] / "masks"


def save_mask(state: AppState, source_row, n: int, png: bytes, strokes: str) -> None:
    probe = _as_dict(source_row["probe_json"])
    fc = int(probe.get("frame_count") or 0)
    if fc and not 0 <= n < fc:
        raise PipelineError(f"frame {n} out of range (0..{fc - 1})")
    if not png.startswith(PNG_MAGIC):
        raise PipelineError("mask must be a PNG image")
    directory = source_masks_dir(state, source_row)
    directory.mkdir(parents=True, exist_ok=True)
    try:
        strokes_data = json.loads(strokes) if strokes and strokes.strip() else []
    except json.JSONDecodeError as exc:
        raise PipelineError(f"strokes must be JSON: {exc}") from exc
    (directory / f"{n:06d}.png").write_bytes(png)
    (directory / f"{n:06d}.json").write_text(
        json.dumps(strokes_data, ensure_ascii=False), encoding="utf-8"
    )


def mask_path(state: AppState, source_row, n: int) -> Path | None:
    p = source_masks_dir(state, source_row) / f"{n:06d}.png"
    return p if p.is_file() else None


def delete_mask(state: AppState, source_row, n: int) -> bool:
    directory = source_masks_dir(state, source_row)
    png = directory / f"{n:06d}.png"
    meta = directory / f"{n:06d}.json"
    existed = png.is_file()
    png.unlink(missing_ok=True)
    meta.unlink(missing_ok=True)
    return existed


def annotations_payload(state: AppState, source_row) -> dict[str, Any]:
    out = []
    directory = source_masks_dir(state, source_row)
    for p in sorted(directory.glob("*.png")):
        try:
            frame = int(p.stem)
        except ValueError:
            continue
        meta = p.with_suffix(".json")
        try:
            strokes = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            strokes = []
        out.append({
            "frame": frame,
            "url": f"/api/sources/{source_row['id']}/masks/{frame}",
            "strokes": strokes if isinstance(strokes, list) else [],
            "updatedAt": datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat(),
        })
    return {"frames": out}


def _normalize_saved_tracks(raw: Any) -> list[dict[str, Any]]:
    """Validate editor track payload; keep boxes + optional keyframes."""
    if not isinstance(raw, list) or not raw:
        raise ValueError("tracks must be a non-empty list")
    out: list[dict[str, Any]] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"tracks[{i}] must be an object")
        boxes_raw = item.get("boxes")
        if not isinstance(boxes_raw, list) or not boxes_raw:
            raise ValueError(f"tracks[{i}].boxes must be a non-empty list")
        boxes: list[list[int] | None] = []
        for b in boxes_raw:
            if b is None:
                boxes.append(None)
                continue
            if not isinstance(b, (list, tuple)) or len(b) < 4:
                boxes.append(None)
                continue
            try:
                x1, y1, x2, y2 = (int(round(float(v))) for v in b[:4])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"tracks[{i}] has invalid box") from exc
            boxes.append([x1, y1, x2, y2] if x2 > x1 and y2 > y1 else None)
        if not any(b is not None for b in boxes):
            raise ValueError(f"tracks[{i}] has no usable boxes")
        try:
            track_id = int(item.get("id", i))
        except (TypeError, ValueError):
            track_id = i
        keys_raw = item.get("keyframes") or []
        keyframes: list[int] = []
        if isinstance(keys_raw, list):
            for k in keys_raw:
                try:
                    n = int(k)
                except (TypeError, ValueError):
                    continue
                if 0 <= n < len(boxes):
                    keyframes.append(n)
        keyframes = sorted(set(keyframes))
        row: dict[str, Any] = {
            "id": track_id,
            "label": str(item.get("label") or f"track{track_id}"),
            "boxes": boxes,
            "keyframes": keyframes,
        }
        if item.get("motion") is not None:
            row["motion"] = str(item["motion"])
        out.append(row)
    return out


def save_job_tracks(state: AppState, job_id: str, tracks_raw: Any) -> dict[str, Any]:
    """Persist edited tracks (+ keyframes) into a completed job report."""
    row = state.jobs.get(job_id)
    if row is None:
        raise LookupError(f"unknown job {job_id}")
    if row["state"] != "COMPLETED":
        raise RuntimeError(f"job is {row['state']}")
    report_json = row["report_json"] if "report_json" in row.keys() else None
    if not report_json:
        raise LookupError(f"job {job_id} has no report")
    try:
        report = json.loads(report_json)
    except (TypeError, ValueError) as exc:
        raise ValueError("report is unreadable") from exc
    if not isinstance(report, dict):
        raise ValueError("report is unreadable")
    request = _as_dict(row["request_json"] if "request_json" in row.keys() else None)
    kind = str(report.get("kind") or request.get("kind") or "")
    if kind and kind not in {"preview", "run"}:
        raise ValueError("only preview/run jobs accept track edits")
    tracks = _normalize_saved_tracks(tracks_raw)
    report = {**report, "tracks": tracks, "tracksEditedAt": utc_now().isoformat()}
    state.jobs.upsert(job_id, "COMPLETED", report=report)
    workdir = report.get("workdir")
    if isinstance(workdir, str) and workdir:
        root = Path(workdir)
        payload = json.dumps(report, ensure_ascii=False, indent=2)
        for rel in ("output/report.json", "preview/preview.json"):
            path = root / rel
            if path.parent.is_dir():
                path.write_text(payload, encoding="utf-8")
    return {"ok": True, "id": job_id, "tracks": len(tracks)}


def job_output_artifacts(row) -> dict[str, Path]:
    """Map format name → artifact path from report.outputs (file or package dir)."""
    report = _as_dict(row["report_json"] if "report_json" in row.keys() else None)
    raw = report.get("outputs")
    out: dict[str, Path] = {}
    if isinstance(raw, dict):
        for fmt, path_str in raw.items():
            if not fmt or path_str is None:
                continue
            if str(fmt).lower() == "mezzanine":
                continue
            path = Path(str(path_str))
            if path.exists():
                out[str(fmt)] = path
    if out:
        return out
    output_path = Path(row["output_path"] or "") if row["output_path"] else None
    if output_path and output_path.is_file():
        suffix = output_path.suffix.lstrip(".").lower() or "mp4"
        out[suffix] = output_path
    return out


def resolve_job_mezzanine(row) -> Path | None:
    """Master file for on-demand packaging (mezzanine, else a single-file delivery)."""
    report = _as_dict(row["report_json"] if "report_json" in row.keys() else None)
    candidates: list[Path] = []
    mezz = report.get("mezzanine")
    if isinstance(mezz, str) and mezz.strip():
        candidates.append(Path(mezz))
    workdir = report.get("workdir")
    if isinstance(workdir, str) and workdir.strip():
        root = Path(workdir)
        candidates.append(root / "output" / "mezzanine.mp4")
        candidates.append(root / "process" / "mezzanine.mp4")
    raw = report.get("outputs")
    if isinstance(raw, dict):
        for key in ("mp4", "mov", "mkv", "webm"):
            path_str = raw.get(key)
            if path_str:
                candidates.append(Path(str(path_str)))
    output_path = Path(row["output_path"] or "") if row["output_path"] else None
    if output_path and output_path.is_file():
        candidates.append(output_path)
    for path in candidates:
        if path.is_file():
            return path
    return None


def queue_package_job(
    state: AppState,
    parent_job_id: str,
    formats: list[str],
    *,
    webm_crf: int = 32,
    segment_seconds: int = 6,
    overwrite: bool = True,
) -> str:
    """Queue on-demand packaging from a completed cleanup job's mezzanine."""
    from videoclean.domain.formats import parse_formats

    row = state.jobs.get(parent_job_id)
    if row is None:
        raise PipelineError(f"unknown job {parent_job_id}")
    if row["state"] != "COMPLETED":
        raise PipelineError(f"job is {row['state']}")
    request = _as_dict(row["request_json"] if "request_json" in row.keys() else None)
    kind = str(request.get("kind") or "run")
    if kind != "run":
        raise PipelineError("only completed cleanup jobs can be packaged")
    mezz = resolve_job_mezzanine(row)
    if mezz is None:
        raise PipelineError("mezzanine missing; re-run removal to create a master file")
    try:
        fmts = parse_formats(formats)
    except ValueError as exc:
        raise PipelineError(str(exc)) from exc
    job_id = new_job_id()
    report = _as_dict(row["report_json"] if "report_json" in row.keys() else None)
    workdir = report.get("workdir")
    if isinstance(workdir, str) and workdir.strip():
        out_base = Path(workdir) / "output" / "cleaned"
    else:
        out_base = Path(state.data_dir) / "jobs" / parent_job_id / "output" / "cleaned"
    out_base.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": "package",
        "parent_job_id": parent_job_id,
        "formats": fmts,
        "webm_crf": int(webm_crf),
        "segment_seconds": int(segment_seconds),
        "overwrite": bool(overwrite),
        "input_path": str(mezz),
        "output_path": str(out_base),
        "source_id": (row["source_id"] if "source_id" in row.keys() else None) or request.get("source_id"),
    }
    return state.manage.submit(
        payload,
        mezz,
        out_base,
        row["prompt"] or "",
        job_id=job_id,
        source_id=payload.get("source_id"),
    )


def job_dict(state: AppState, row) -> dict[str, Any]:
    job_id = row["id"]
    progress = _as_dict(row["progress_json"] if "progress_json" in row.keys() else None)
    request = _as_dict(row["request_json"] if "request_json" in row.keys() else None)
    output_path = Path(row["output_path"] or "") if row["output_path"] else None
    input_path = Path(row["input_path"] or "") if row["input_path"] else None
    state_name = row["state"] or ""
    artifacts = job_output_artifacts(row) if state_name == "COMPLETED" else {}
    has_output = bool(artifacts) or bool(output_path and output_path.is_file() and state_name == "COMPLETED")
    has_input = bool(input_path and input_path.is_file())
    outputs = {
        fmt: f"/api/jobs/{job_id}/output?fmt={fmt}" for fmt in artifacts
    } if has_output and artifacts else {}
    primary_url = None
    if has_output:
        if artifacts:
            # Prefer a single-file container for the default download link.
            preferred = next((f for f in ("mp4", "mov", "mkv", "webm") if f in artifacts), None)
            primary_fmt = preferred or next(iter(artifacts))
            primary_url = f"/api/jobs/{job_id}/output?fmt={primary_fmt}"
        else:
            primary_url = f"/api/jobs/{job_id}/output"
    out = {
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
        "kind": request.get("kind") or "run",
        "source_id": (row["source_id"] if "source_id" in row.keys() else None) or request.get("source_id"),
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
        "can_package": bool(
            state_name == "COMPLETED"
            and (request.get("kind") or "run") == "run"
            and resolve_job_mezzanine(row) is not None
        ),
        "parent_job_id": request.get("parent_job_id"),
        "output_url": primary_url,
        "outputs": outputs,
        "input_url": f"/api/jobs/{job_id}/input" if has_input else None,
        "status_url": f"/api/jobs/{job_id}",
    }
    sid = out["source_id"]
    source_name = None
    if sid and state.sources is not None:
        srow = state.sources.get(sid)
        if srow is not None:
            source_name = srow["name"]
    out["source_name"] = source_name
    return out


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
                "downloadable": True,
            }
        )
    return groups


def options_payload(state: AppState) -> dict[str, Any]:
    catalog = state.catalog
    llm_names = ollama_model_names(timeout=2.0) or []
    from videoclean.adapters.models.catalog import SEGMENTER_COMPONENTS

    segmenter_models = [
        {
            "id": info.id,
            "title": info.title,
            "model_ref": info.model_ref,
            "size_hint": info.size_hint,
            "ready": catalog.is_ready(info.id),
        }
        for info in SEGMENTER_COMPONENTS
    ]
    from videoclean.domain.formats import known_format_names

    # Drivers are always listed; readiness is per weights (segmenter_models), not per driver name.
    return {
        "device": default_device(),
        "devices": ["cpu", "cuda", "mps"],
        "formats": known_format_names(),
        "llm_places": list(LLM_PLACES),
        "llm_models": llm_names,
        "detectors": [name for name in DETECTORS if backend_ready("detector", name, catalog=catalog)],
        "segmenters": list(SEGMENTERS),
        "segmenter_models": segmenter_models,
        "default_segmenter_model": DEFAULT_SEGMENTER_MODEL,
        "inpainters": [
            name for name in INPAINTERS if backend_ready("inpainter", name, catalog=catalog)
        ],
        "max_quality_ready": max_quality_ready(catalog),
        "models": grouped_models(state),
        "profiles": _profiles_for_options(default_device()),
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
