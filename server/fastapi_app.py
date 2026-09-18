from __future__ import annotations

import json
import os
import secrets
import shutil
import signal
from pathlib import Path
from typing import Any

import io
import zipfile

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Security, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from server.api_schema import APP_DESCRIPTION, ErrorBody, Job, JobCreated, apply_public_internal_tags

from videoclean.adapters.models.catalog import (
    add_extra,
    add_provider,
    ollama_model_names,
    remove_extra,
    remove_provider,
)
from server.app_state import AppState, build_app_state
from server.service import (
    VIDEO_SUFFIXES,
    annotations_payload,
    api_token,
    auth_enabled,
    auth_from_env,
    cancel_downloads,
    default_device,
    delete_mask,
    delete_preset,
    doctor_payload,
    downloads_payload,
    grouped_models,
    job_dict,
    job_output_artifacts,
    list_presets,
    mask_path,
    options_payload,
    providers_payload,
    queue_package_job,
    queue_preview_from_job,
    queue_preview_job,
    queue_source_preview,
    queue_source_prompt,
    queue_source_run,
    register_source,
    save_job_tracks,
    save_mask,
    save_preset,
    serialize_clean_form,
    source_dict,
    source_frame_path,
    start_download,
)
from videoclean.application.errors import PipelineError

DIST_DIR = Path(__file__).resolve().parent / "static_dist"

_PLACEHOLDER_HTML = """<!doctype html><html lang="en"><meta charset="utf-8">
<title>videoclean</title><body style="font-family:system-ui;max-width:40rem;margin:4rem auto">
<h1>videoclean UI</h1>
<p>React UI is not built yet. Run:</p>
<pre>cd webui &amp;&amp; bun install &amp;&amp; bun run build</pre>
<p>Then restart <code>videoclean serve</code>. API docs: <a href="/api/docs">/api/docs</a></p>
</body></html>"""


# Docs UIs fetch /openapi.json via JS without the browser Basic prompt credentials.
_AUTH_EXEMPT_PATHS = frozenset(
    {
        "/health",
        "/openapi.json",
        "/api/docs",
        "/api/redoc",
    }
)


def _auth_exempt(path: str) -> bool:
    if path in _AUTH_EXEMPT_PATHS:
        return True
    # Swagger UI / ReDoc static assets under the docs URLs.
    return path.startswith("/api/docs/") or path.startswith("/api/redoc/")


class BasicOrBearerAuth(BaseHTTPMiddleware):
    def __init__(self, app, user: str, password: str, token: str) -> None:
        super().__init__(app)
        self.user = user
        self.password = password
        self.token = token

    async def dispatch(self, request: Request, call_next):
        if request.method == "OPTIONS" or _auth_exempt(request.url.path):
            return await call_next(request)
        header = request.headers.get("authorization") or ""
        if self._ok(header):
            return await call_next(request)
        return JSONResponse(
            {"detail": "unauthorized"},
            status_code=401,
            headers={"WWW-Authenticate": 'Basic realm="videoclean"'},
        )

    def _ok(self, header: str) -> bool:
        if not header:
            return False
        kind, _, rest = header.partition(" ")
        kind = kind.strip().lower()
        rest = rest.strip()
        if kind == "bearer":
            return secrets.compare_digest(rest, self.token)
        if kind != "basic":
            return False
        import base64

        try:
            decoded = base64.b64decode(rest).decode("utf-8")
        except Exception:  # noqa: BLE001
            return False
        user, _, password = decoded.partition(":")
        return secrets.compare_digest(user, self.user) and secrets.compare_digest(
            password, self.password
        )


def create_app(state: AppState) -> FastAPI:
    try:
        from importlib.metadata import version as pkg_version

        app_version = pkg_version("videoclean")
    except Exception:  # noqa: BLE001
        app_version = "0.1.0"

    app = FastAPI(
        title="VideoClean",
        version=app_version,
        description=APP_DESCRIPTION,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        swagger_ui_parameters={"persistAuthorization": True},
    )
    app.state.vc = state
    bearer_scheme = HTTPBearer(auto_error=False)

    def _openapi() -> dict[str, Any]:
        if app.openapi_schema:
            return app.openapi_schema
        schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )
        schema.setdefault("components", {}).setdefault("securitySchemes", {})["BearerAuth"] = {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "API token",
            "description": "VIDEOCLEAN_API_TOKEN (or UI password if the token is unset).",
        }
        schema["security"] = [{"BearerAuth": []}]
        app.openapi_schema = apply_public_internal_tags(schema)
        return app.openapi_schema

    app.openapi = _openapi  # type: ignore[method-assign]

    origins = [
        item.strip()
        for item in str(os.environ.get("VIDEOCLEAN_CORS") or "*").split(",")
        if item.strip()
    ]
    wildcard = origins == ["*"] or origins == []
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if wildcard else origins,
        allow_credentials=not wildcard,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    if auth_enabled():
        user, password = auth_from_env()
        app.add_middleware(BasicOrBearerAuth, user=user, password=password, token=api_token())
    if DIST_DIR.is_dir():
        app.mount("/assets", StaticFiles(directory=str(DIST_DIR / "assets")), name="assets")

    def get_state() -> AppState:
        return app.state.vc

    def _source_or_404(st: AppState, source_id: str):
        row = st.sources.get(source_id)
        if row is None:
            raise HTTPException(404, f"unknown source {source_id}")
        return row

    @app.get("/health", summary="Liveness probe (no auth)")
    def health():
        return {"ok": True}

    @app.get("/api/auth-check", include_in_schema=False)
    def auth_check(_creds: HTTPAuthorizationCredentials | None = Security(bearer_scheme)):
        return {"ok": True}

    def _spa_index():
        spa_index = DIST_DIR / "index.html"
        if spa_index.is_file():
            return spa_index.read_text(encoding="utf-8")
        return _PLACEHOLDER_HTML

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    @app.get("/settings", response_class=HTMLResponse, include_in_schema=False)
    @app.get("/config", response_class=HTMLResponse, include_in_schema=False)
    def index():
        return _spa_index()

    @app.get("/v/{source_id}", response_class=HTMLResponse, include_in_schema=False)
    def workspace_source(source_id: str):
        return _spa_index()

    @app.get("/api", summary="API index")
    def api_index():
        return {
            "docs": "/api/docs",
            "redoc": "/api/redoc",
            "openapi": "/openapi.json",
            "auth": {
                "browser": "HTTP Basic (VIDEOCLEAN_UI_USER / VIDEOCLEAN_UI_PASSWORD)",
                "api": "Authorization: Bearer VIDEOCLEAN_API_TOKEN (falls back to UI password)",
            },
            "public": {
                "GET /health": "liveness",
                "GET /api/options": "profiles, devices, formats, catalog defaults",
                "POST /api/jobs": "one-shot cleanup: video|source_id + prompt; formats; webhook_url",
                "GET /api/jobs": "list",
                "GET /api/jobs/{id}": "poll status",
                "GET /api/jobs/{id}/output": "download (?fmt=)",
                "POST /api/jobs/{id}/cancel": "cancel queued/running",
            },
            "webhook": {
                "fields": ["webhook_url", "webhook_secret"],
                "events": ["COMPLETED", "FAILED"],
                "signature": "X-VideoClean-Signature: sha256=<hmac> when webhook_secret set",
                "absolute_urls": "set VIDEOCLEAN_PUBLIC_BASE_URL",
            },
            "internal": {
                "sources": "POST/GET/DELETE /api/sources…",
                "preview": "POST /api/preview, kind=preview|prompt",
                "package": "POST /api/jobs/{id}/package",
                "models": "GET/POST /api/models…",
                "presets": "GET/POST/DELETE /api/presets",
            },
        }

    @app.post("/api/sources")
    async def create_source(st: AppState = Depends(get_state), video: UploadFile = File(...)):
        if not (video.filename or "").strip():
            raise HTTPException(400, "video file is required")
        tmp_dir = Path(st.data_dir) / "uploads" / "_incoming"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        safe_name = Path(video.filename or "").name
        tmp = tmp_dir / f"src_{os.getpid()}_{safe_name}"
        try:
            await _save_upload(video, tmp)
            sid = register_source(st, tmp, video.filename)
        except PipelineError as exc:
            raise HTTPException(400, str(exc)) from exc
        finally:
            tmp.unlink(missing_ok=True)
        return JSONResponse(source_dict(st.sources.get(sid)), status_code=201)

    @app.get("/api/sources")
    def list_sources(st: AppState = Depends(get_state)):
        return [source_dict(row) for row in st.sources.list()]

    @app.get("/api/sources/{source_id}")
    def get_source(source_id: str, st: AppState = Depends(get_state)):
        return source_dict(_source_or_404(st, source_id))

    @app.delete("/api/sources/{source_id}")
    def delete_source(source_id: str, st: AppState = Depends(get_state)):
        row = _source_or_404(st, source_id)
        for job_row in st.jobs.list_jobs_full(limit=10_000):
            if job_row["source_id"] == source_id and job_row["state"] in {"QUEUED", "RUNNING"}:
                raise HTTPException(409, "source has active jobs; cancel them first")
        st.sources.delete(source_id)
        shutil.rmtree(Path(st.data_dir) / "sources" / source_id, ignore_errors=True)
        return {"ok": True, "id": source_id}

    @app.get("/api/sources/{source_id}/video")
    def source_video(source_id: str, st: AppState = Depends(get_state)):
        row = _source_or_404(st, source_id)
        path = Path(row["path"])
        if not path.is_file():
            raise HTTPException(404, "source file missing")
        media_type = "video/webm" if path.suffix.lower() == ".webm" else "video/mp4"
        return FileResponse(path, media_type=media_type, filename=path.name)

    @app.get("/api/sources/{source_id}/frames/{n}.jpg")
    @app.get("/api/sources/{source_id}/frames/{n}")
    def source_frame(source_id: str, n: str, st: AppState = Depends(get_state)):
        row = _source_or_404(st, source_id)
        try:
            frame = int(n.removesuffix(".jpg"))
        except ValueError as exc:
            raise HTTPException(404, f"frame {n} unavailable") from exc
        path = source_frame_path(st, row, frame)
        if path is None:
            raise HTTPException(404, f"frame {frame} unavailable")
        return FileResponse(path, media_type="image/jpeg")

    @app.put("/api/sources/{source_id}/masks/{n}")
    async def put_mask(
        source_id: str,
        n: int,
        st: AppState = Depends(get_state),
        mask: UploadFile = File(...),
        strokes: str = Form(""),
    ):
        row = _source_or_404(st, source_id)
        data = await mask.read()
        try:
            save_mask(st, row, n, data, strokes)
        except PipelineError as exc:
            raise HTTPException(400, str(exc)) from exc
        return JSONResponse({"ok": True, "frame": n}, status_code=201)

    @app.get("/api/sources/{source_id}/masks/{n}")
    def get_mask(source_id: str, n: int, st: AppState = Depends(get_state)):
        row = _source_or_404(st, source_id)
        path = mask_path(st, row, n)
        if path is None:
            raise HTTPException(404, f"mask for frame {n} not found")
        return FileResponse(path, media_type="image/png")

    @app.delete("/api/sources/{source_id}/masks/{n}")
    def remove_mask(source_id: str, n: int, st: AppState = Depends(get_state)):
        row = _source_or_404(st, source_id)
        if not delete_mask(st, row, n):
            raise HTTPException(404, f"mask for frame {n} not found")
        return {"ok": True, "frame": n}

    @app.get("/api/sources/{source_id}/annotations")
    def source_annotations(source_id: str, st: AppState = Depends(get_state)):
        row = _source_or_404(st, source_id)
        return annotations_payload(st, row)

    @app.get("/api/presets")
    def presets(st: AppState = Depends(get_state)):
        return list_presets(st.data_dir)

    @app.post("/api/presets")
    def create_preset(body: dict[str, Any], st: AppState = Depends(get_state)):
        data = body or {}
        try:
            item = save_preset(st.data_dir, str(data.get("name") or ""), data.get("payload"))
        except PipelineError as exc:
            raise HTTPException(400, str(exc)) from exc
        return JSONResponse(item, status_code=201)

    @app.delete("/api/presets/{preset_id}")
    def remove_preset(preset_id: str, st: AppState = Depends(get_state)):
        if not delete_preset(st.data_dir, preset_id):
            raise HTTPException(404, f"unknown preset {preset_id}")
        return {"ok": True, "id": preset_id}

    @app.get("/api/poll")
    def poll(st: AppState = Depends(get_state)):
        jobs = [job_dict(st, row) for row in st.jobs.list_jobs_full(limit=100)]
        return {
            "jobs": jobs,
            "models": grouped_models(st),
            "downloads": downloads_payload(st),
            "doctor": doctor_payload(),
            "options": options_payload(st),
            "ollama": _ollama_payload(),
            "providers": providers_payload(st),
            "device": default_device(),
        }

    @app.get(
        "/api/jobs",
        response_model=list[Job],
        summary="List jobs",
        responses={401: {"model": ErrorBody}},
    )
    def list_jobs(state_filter: str | None = None, st: AppState = Depends(get_state)):
        filt = None if not state_filter or state_filter == "all" else state_filter
        return [job_dict(st, row) for row in st.jobs.list_jobs_full(limit=100, state=filt)]

    @app.get(
        "/api/jobs/{job_id}",
        response_model=Job,
        summary="Get job status",
        responses={401: {"model": ErrorBody}, 404: {"model": ErrorBody}},
    )
    def get_job(job_id: str, st: AppState = Depends(get_state)):
        row = st.jobs.get(job_id)
        if row is None:
            raise HTTPException(404, f"unknown job {job_id}")
        return job_dict(st, row)

    @app.post(
        "/api/jobs",
        response_model=JobCreated,
        status_code=201,
        summary="Start cleanup job (one-shot)",
        responses={400: {"model": ErrorBody}, 401: {"model": ErrorBody}, 404: {"model": ErrorBody}},
    )
    async def create_job(
        st: AppState = Depends(get_state),
        video: UploadFile | None = File(
            None, description="Video file (required unless source_id). Public path: upload here."
        ),
        prompt: str = Form(
            "",
            description="What to remove (required for kind=run unless tracks/masks override).",
        ),
        kind: str = Form(
            "run",
            description="run (public) | preview | prompt. Public contract uses run.",
        ),
        source_id: str = Form("", description="Library source id instead of uploading video."),
        targets: str = Form("", description="JSON targets override (advanced)."),
        tracks: str = Form("", description="JSON tracks override (advanced)."),
        masks: str = Form("", description="Comma-separated annotated frame indices (advanced)."),
        mode: str = Form(""),
        start: str = Form(""),
        count: str = Form(""),
        stride: str = Form(""),
        indices: str = Form(""),
        all_: str = Form("", alias="all"),
        llm_base_url: str = Form(""),
        llm_api_key: str = Form(""),
        keep_workdir: str = Form(""),
        min_mask_coverage: str = Form(""),
        verify_max_coverage: str = Form(""),
        formats: str = Form(
            "",
            description="Delivery formats, comma-separated (default mp4). Example: mp4,webm",
        ),
        webhook_url: str = Form(
            "",
            description="Optional callback URL; POST JSON on COMPLETED or FAILED (kind=run).",
        ),
        webhook_secret: str = Form(
            "",
            description="Optional HMAC secret; sets X-VideoClean-Signature: sha256=<hex>.",
        ),
        webm_crf: str = Form("", description="WebM CRF when packaging webm (default 32)."),
        segment_seconds: str = Form("", description="HLS/DASH segment length in seconds (default 6)."),
        device: str = Form(""),
        detector: str = Form(""),
        segmenter: str = Form(""),
        inpainter: str = Form(""),
        llm_place: str = Form(""),
        llm_model: str = Form(""),
        fmt: str = Form(""),
        detector_model: str = Form(""),
        segmenter_model: str = Form(""),
        inpainter_model: str = Form(""),
        detector_threshold: str = Form(""),
        mask_dilate_px: str = Form(""),
        verify: str = Form(""),
        prompt_frame_stride: str = Form(""),
        prompt_frame_max: str = Form(""),
        parse_chunk_frames: str = Form(""),
        vision_batch: str = Form(""),
        detector_keyframes: str = Form(""),
        detector_nms_iou: str = Form(""),
        detector_max_box_area: str = Form(""),
        tracker_min_score: str = Form(""),
        tracker_max_template_area: str = Form(""),
        propainter_mask_dilation: str = Form(""),
        propainter_ref_stride: str = Form(""),
        propainter_neighbor_length: str = Form(""),
        propainter_subvideo_length: str = Form(""),
        propainter_raft_iter: str = Form(""),
        profile: str = Form("", description="fast | balanced | quality | custom"),
        verify_max_passes: str = Form(""),
        inpaint_workers: str = Form(""),
        inpaint_chunk_overlap: str = Form(""),
        mask_policy: str = Form(""),
        overwrite: str = Form(""),
    ):
        kind = (kind or "run").strip().lower() or "run"
        if kind not in {"run", "preview", "prompt"}:
            raise HTTPException(400, "kind must be run | preview | prompt")
        fields = {
            "kind": kind,
            "source_id": source_id,
            "targets": targets,
            "tracks": tracks,
            "masks": masks,
            "mode": mode,
            "start": start,
            "count": count,
            "stride": stride,
            "indices": indices,
            "all": all_,
            "llm_base_url": llm_base_url,
            "llm_api_key": llm_api_key,
            "keep_workdir": keep_workdir,
            "min_mask_coverage": min_mask_coverage,
            "verify_max_coverage": verify_max_coverage,
            "formats": formats,
            "webm_crf": webm_crf,
            "segment_seconds": segment_seconds,
            "device": device,
            "detector": detector,
            "segmenter": segmenter,
            "inpainter": inpainter,
            "llm_place": llm_place,
            "llm_model": llm_model,
            "fmt": fmt,
            "detector_model": detector_model,
            "segmenter_model": segmenter_model,
            "inpainter_model": inpainter_model,
            "detector_threshold": detector_threshold,
            "mask_dilate_px": mask_dilate_px,
            "verify": verify,
            "prompt_frame_stride": prompt_frame_stride,
            "prompt_frame_max": prompt_frame_max,
            "parse_chunk_frames": parse_chunk_frames,
            "vision_batch": vision_batch,
            "detector_keyframes": detector_keyframes,
            "detector_nms_iou": detector_nms_iou,
            "detector_max_box_area": detector_max_box_area,
            "tracker_min_score": tracker_min_score,
            "tracker_max_template_area": tracker_max_template_area,
            "propainter_mask_dilation": propainter_mask_dilation,
            "propainter_ref_stride": propainter_ref_stride,
            "propainter_neighbor_length": propainter_neighbor_length,
            "propainter_subvideo_length": propainter_subvideo_length,
            "propainter_raft_iter": propainter_raft_iter,
            "profile": profile,
            "verify_max_passes": verify_max_passes,
            "inpaint_workers": inpaint_workers,
            "inpaint_chunk_overlap": inpaint_chunk_overlap,
            "mask_policy": mask_policy,
            "overwrite": overwrite,
        }
        payload = serialize_clean_form({k: v for k, v in fields.items() if v != ""})
        if (webhook_url or "").strip():
            payload["webhook_url"] = webhook_url.strip()
        if (webhook_secret or "").strip():
            payload["webhook_secret"] = webhook_secret.strip()
        if (mask_policy or "").strip():
            policy = mask_policy.strip().lower()
            if policy not in {"static", "propagate"}:
                raise HTTPException(400, "mask_policy must be static | propagate")
            payload["mask_policy"] = policy
        for raw, key in ((targets, "targets_override"), (tracks, "tracks_override")):
            raw = (raw or "").strip()
            if raw:
                try:
                    payload[key] = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise HTTPException(400, f"{key} must be JSON: {exc}") from exc
        if (masks or "").strip():
            try:
                payload["masks"] = [int(x) for x in masks.split(",") if x.strip()]
            except ValueError as exc:
                raise HTTPException(400, f"masks must be comma-separated frame numbers: {exc}") from exc
        if mode.strip():
            payload["mode"] = mode.strip()
        for name in ("start", "count", "stride"):
            val = {"start": start, "count": count, "stride": stride}[name]
            if val.strip():
                payload[name] = int(val)
        if indices.strip():
            payload["indices"] = [int(i) for i in indices.split(",") if i.strip()]
        if all_.strip():
            payload["all"] = True
        if sum(1 for k in ("targets_override", "tracks_override", "masks") if payload.get(k)) > 1:
            raise HTTPException(400, "targets, tracks and masks are mutually exclusive")

        source_row = None
        if source_id.strip():
            source_row = st.sources.get(source_id.strip())
            if source_row is None:
                raise HTTPException(404, f"unknown source {source_id.strip()}")
        try:
            if kind == "run":
                if source_row is not None:
                    job_id = queue_source_run(st, source_row, prompt, payload)
                elif video is not None and (video.filename or "").strip():
                    suffix = Path(video.filename).suffix.lower()
                    if suffix not in VIDEO_SUFFIXES:
                        raise HTTPException(400, f"unsupported video type {suffix}")
                    tmp_dir = Path(st.data_dir) / "uploads" / "_incoming"
                    tmp_dir.mkdir(parents=True, exist_ok=True)
                    safe_name = Path(video.filename or "").name
                    tmp = tmp_dir / f"up_{os.getpid()}_{safe_name}"
                    try:
                        await _save_upload(video, tmp)
                        sid = register_source(st, tmp, video.filename)
                    finally:
                        tmp.unlink(missing_ok=True)
                    job_id = queue_source_run(st, st.sources.get(sid), prompt, payload)
                else:
                    raise HTTPException(400, "provide a video file or source_id")
            elif kind == "preview":
                if source_row is None:
                    raise HTTPException(400, "preview from the editor requires source_id")
                job_id = queue_source_preview(st, source_row, prompt, payload)
            else:
                if source_row is None:
                    raise HTTPException(400, "prompt interpretation requires source_id")
                job_id = queue_source_prompt(st, source_row, prompt, payload)
        except PipelineError as exc:
            raise HTTPException(400, str(exc)) from exc
        row = st.jobs.get(job_id)
        body = job_dict(st, row) if row is not None else {"id": job_id, "state": "QUEUED"}
        body["poll"] = f"/api/jobs/{job_id}"
        body["download"] = f"/api/jobs/{job_id}/output" if kind == "run" else None
        return JSONResponse(body, status_code=201)

    @app.get("/api/jobs/{job_id}/report")
    def job_report(job_id: str, st: AppState = Depends(get_state)):
        row = st.jobs.get(job_id)
        if row is None:
            raise HTTPException(404, f"unknown job {job_id}")
        report_json = row["report_json"] if "report_json" in row.keys() else None
        if not report_json:
            raise HTTPException(404, f"job {job_id} has no report")
        try:
            payload = json.loads(report_json)
        except (TypeError, ValueError) as exc:
            raise HTTPException(404, f"job {job_id} report is unreadable") from exc
        return JSONResponse(payload)

    @app.put("/api/jobs/{job_id}/report/tracks")
    def job_report_tracks(job_id: str, body: dict[str, Any], st: AppState = Depends(get_state)):
        try:
            return save_job_tracks(st, job_id, body.get("tracks"))
        except LookupError as exc:
            raise HTTPException(404, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/jobs/{job_id}/package")
    async def job_package(
        job_id: str,
        st: AppState = Depends(get_state),
        formats: str = Form(""),
        webm_crf: str = Form(""),
        segment_seconds: str = Form(""),
        overwrite: str = Form("1"),
    ):
        """Queue on-demand conversion from the job mezzanine into delivery formats."""
        fmts = [f.strip().lower() for f in formats.split(",") if f.strip()]
        if not fmts:
            raise HTTPException(400, "formats is required (e.g. webm,mov)")
        try:
            crf = int(webm_crf) if str(webm_crf).strip() else 32
        except ValueError as exc:
            raise HTTPException(400, "webm_crf must be an integer") from exc
        try:
            seg = int(segment_seconds) if str(segment_seconds).strip() else 6
        except ValueError as exc:
            raise HTTPException(400, "segment_seconds must be an integer") from exc
        try:
            new_id = queue_package_job(
                st,
                job_id,
                fmts,
                webm_crf=crf,
                segment_seconds=seg,
                overwrite=str(overwrite).strip().lower() not in {"0", "false", "no"},
            )
        except PipelineError as exc:
            raise HTTPException(400, str(exc)) from exc
        row = st.jobs.get(new_id)
        body = job_dict(st, row) if row is not None else {"id": new_id, "state": "QUEUED"}
        body["poll"] = f"/api/jobs/{new_id}"
        body["parent_job_id"] = job_id
        return JSONResponse(body, status_code=201)

    @app.get(
        "/api/jobs/{job_id}/output",
        summary="Download cleaned output",
        responses={401: {"model": ErrorBody}, 404: {"model": ErrorBody}, 409: {"model": ErrorBody}},
    )
    def job_output(
        job_id: str,
        fmt: str | None = Query(None, description="Delivery format key from job.outputs (e.g. mp4, webm)"),
        st: AppState = Depends(get_state),
    ):
        row = st.jobs.get(job_id)
        if row is None:
            raise HTTPException(404, f"unknown job {job_id}")
        if row["state"] != "COMPLETED":
            raise HTTPException(409, f"job is {row['state']}")
        artifacts = job_output_artifacts(row)
        path: Path | None = None
        if fmt:
            key = fmt.strip().lower()
            path = artifacts.get(key)
            if path is None:
                raise HTTPException(404, f"format {key!r} not in job outputs")
        elif artifacts:
            preferred = next((f for f in ("mp4", "mov", "mkv", "webm") if f in artifacts), None)
            path = artifacts[preferred or next(iter(artifacts))]
        else:
            candidate = Path(row["output_path"] or "")
            path = candidate if candidate.is_file() else None
        if path is None or not path.exists():
            raise HTTPException(404, "output file missing")
        if path.is_dir():
            return _zip_directory_response(path)
        return FileResponse(path, filename=path.name, media_type="application/octet-stream")

    @app.post("/api/preview")
    async def create_preview(
        st: AppState = Depends(get_state),
        video: UploadFile | None = File(None),
        prompt: str = Form(""),
        mode: str = Form("parse"),
        targets: str = Form(""),
        device: str = Form(""),
        detector: str = Form(""),
        detector_model: str = Form(""),
        detector_threshold: str = Form(""),
        segmenter: str = Form(""),
        segmenter_model: str = Form(""),
        mask_dilate_px: str = Form(""),
        start: str = Form(""),
        count: str = Form(""),
        stride: str = Form(""),
        indices: str = Form(""),
    ):
        payload = serialize_clean_form({k: v for k, v in {
            "device": device,
            "detector": detector,
            "detector_model": detector_model,
            "detector_threshold": detector_threshold,
            "segmenter": segmenter,
            "segmenter_model": segmenter_model,
            "mask_dilate_px": mask_dilate_px,
        }.items() if v != ""})
        payload["kind"] = "preview"
        payload["mode"] = mode
        payload["start"] = int(start) if start.strip() else None
        payload["count"] = int(count) if count.strip() else None
        payload["stride"] = int(stride) if stride.strip() else None
        payload["indices"] = [int(i) for i in indices.split(",") if i.strip()] if indices.strip() else None
        if targets.strip():
            try:
                payload["targets"] = json.loads(targets)
            except json.JSONDecodeError as exc:
                raise HTTPException(400, f"targets must be JSON: {exc}") from exc
        if video is None or not (video.filename or "").strip():
            raise HTTPException(400, "video file is required")
        suffix = Path(video.filename or "input.mp4").suffix.lower() or ".mp4"
        if suffix not in VIDEO_SUFFIXES:
            raise HTTPException(400, f"unsupported video type {suffix}")
        tmp_dir = Path(st.data_dir) / "uploads" / "_incoming"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        safe_name = Path(video.filename or "").name
        tmp = tmp_dir / f"up_{os.getpid()}_{safe_name}"
        try:
            await _save_upload(video, tmp)
            job_id = queue_preview_job(
                st, tmp, prompt, payload, original_name=video.filename or tmp.name
            )
        except PipelineError as exc:
            raise HTTPException(400, str(exc)) from exc
        finally:
            tmp.unlink(missing_ok=True)
        row = st.jobs.get(job_id)
        body = job_dict(st, row) if row is not None else {"id": job_id, "state": "QUEUED"}
        body["poll"] = f"/api/jobs/{job_id}"
        return JSONResponse(body, status_code=201)

    @app.post("/api/preview/from-job")
    async def create_preview_from_job(body: dict[str, Any], st: AppState = Depends(get_state)):
        data = body or {}
        job_id = str(data.get("job_id") or "").strip()
        if not job_id:
            raise HTTPException(400, "job_id is required")
        payload: dict[str, Any] = {"kind": "preview"}
        for key in (
            "mode", "device", "detector", "detector_model", "detector_threshold",
            "segmenter", "segmenter_model", "mask_dilate_px",
        ):
            if str(data.get(key) or "").strip():
                payload[key] = data[key]
        for key in ("start", "count", "stride"):
            if data.get(key) is not None:
                payload[key] = int(data[key])
        if data.get("indices"):
            payload["indices"] = [int(i) for i in data["indices"]]
        if data.get("targets"):
            payload["targets"] = data["targets"]
        try:
            new_id = queue_preview_from_job(st, job_id, payload, prompt=str(data.get("prompt") or ""))
        except PipelineError as exc:
            raise HTTPException(400, str(exc)) from exc
        row = st.jobs.get(new_id)
        body_out = job_dict(st, row) if row is not None else {"id": new_id, "state": "QUEUED"}
        body_out["poll"] = f"/api/jobs/{new_id}"
        return JSONResponse(body_out, status_code=201)

    @app.get("/api/jobs/{job_id}/preview/{name}")
    def preview_artifact(job_id: str, name: str, st: AppState = Depends(get_state)):
        from server.service import preview_artifact_path

        path = preview_artifact_path(st, job_id, name)
        if path is None:
            raise HTTPException(404, "artifact not found")
        media_type = "application/json" if name.endswith(".json") else (
            "video/mp4" if name.endswith(".mp4") else "image/jpeg"
        )
        return FileResponse(path, media_type=media_type)

    @app.get("/api/jobs/{job_id}/input")
    def job_input(job_id: str, st: AppState = Depends(get_state)):
        row = st.jobs.get(job_id)
        if row is None:
            raise HTTPException(404, f"unknown job {job_id}")
        path = Path(row["input_path"] or "")
        if not path.is_file():
            raise HTTPException(404, "input file missing")
        return FileResponse(path, filename=path.name, media_type="application/octet-stream")

    @app.get("/api/jobs/{job_id}/probe")
    def job_probe(job_id: str, st: AppState = Depends(get_state)):
        row = st.jobs.get(job_id)
        if row is None:
            raise HTTPException(404, f"unknown job {job_id}")
        path = Path(row["input_path"] or "")
        if not path.is_file():
            raise HTTPException(404, "input file missing")
        from videoclean.adapters.media.ffmpeg import FFmpegMedia

        try:
            m = FFmpegMedia().probe(path)
        except PipelineError as exc:
            raise HTTPException(502, str(exc)) from exc
        return {
            "fps": m.fps,
            "duration_s": m.duration_s,
            "width": m.width,
            "height": m.height,
            "frame_count": m.frame_count,
        }

    @app.post(
        "/api/jobs/{job_id}/cancel",
        response_model=Job,
        summary="Cancel a queued or running job",
        responses={401: {"model": ErrorBody}, 404: {"model": ErrorBody}},
    )
    def cancel_job(job_id: str, st: AppState = Depends(get_state)):
        row = st.jobs.get(job_id)
        if row is None:
            raise HTTPException(404, f"unknown job {job_id}")
        st.manage.cancel(job_id)
        after = st.jobs.get(job_id)
        return job_dict(st, after) if after is not None else {"id": job_id, "state": "CANCELLED"}

    @app.post("/api/jobs/{job_id}/retry")
    def retry_job(job_id: str, st: AppState = Depends(get_state)):
        try:
            new_id = st.manage.retry(job_id)
        except PipelineError as exc:
            raise HTTPException(400, str(exc)) from exc
        row = st.jobs.get(new_id)
        return job_dict(st, row) if row is not None else {"id": new_id}

    @app.delete("/api/jobs/{job_id}")
    def delete_job(job_id: str, st: AppState = Depends(get_state)):
        row = st.jobs.get(job_id)
        if row is None:
            raise HTTPException(404, f"unknown job {job_id}")
        st.manage.delete(job_id, st.data_dir)
        return {"ok": True, "id": job_id}

    @app.get("/api/models")
    def models(st: AppState = Depends(get_state)):
        return grouped_models(st)

    @app.post("/api/models/download")
    def download_model(body: dict[str, Any], st: AppState = Depends(get_state)):
        cid = str((body or {}).get("id") or "").strip()
        try:
            msg = start_download(st, cid)
        except PipelineError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"ok": True, "message": msg, "id": cid}

    @app.post("/api/models/custom")
    def custom_model(body: dict[str, Any], st: AppState = Depends(get_state)):
        data = body or {}
        try:
            info = add_extra(
                kind=str(data.get("kind") or ""),
                backend=str(data.get("backend") or ""),
                model_ref=str(data.get("model_ref") or ""),
                title=str(data.get("title") or ""),
                data_dir=st.data_dir,
            )
        except PipelineError as exc:
            raise HTTPException(400, str(exc)) from exc
        download = bool(data.get("download", True))
        started = None
        if download:
            try:
                started = start_download(st, info.id)
            except PipelineError as exc:
                started = str(exc)
        return {
            "ok": True,
            "id": info.id,
            "title": info.title,
            "download": started,
        }

    @app.delete("/api/models/{component_id:path}")
    def delete_model(component_id: str, st: AppState = Depends(get_state)):
        try:
            ok = remove_extra(component_id, data_dir=st.data_dir)
        except PipelineError as exc:
            raise HTTPException(400, str(exc)) from exc
        if not ok:
            raise HTTPException(404, f"unknown extra {component_id}")
        return {"ok": True, "id": component_id}

    @app.get("/api/providers")
    def list_providers(st: AppState = Depends(get_state)):
        return {"providers": providers_payload(st), "ollama": _ollama_payload()}

    @app.post("/api/providers")
    def create_provider(body: dict[str, Any], st: AppState = Depends(get_state)):
        data = body or {}
        models_raw = data.get("models") or []
        if isinstance(models_raw, str):
            models = [p.strip() for p in models_raw.replace(",", "\n").splitlines() if p.strip()]
        elif isinstance(models_raw, list):
            models = [str(p).strip() for p in models_raw if str(p).strip()]
        else:
            models = []
        try:
            row = add_provider(
                title=str(data.get("title") or ""),
                base_url=str(data.get("base_url") or ""),
                api_key=str(data.get("api_key") or ""),
                models=models,
                provider_id=str(data.get("id") or ""),
                data_dir=st.data_dir,
            )
        except PipelineError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"ok": True, "provider": {**row, "api_key": "", "has_api_key": bool(row.get("api_key"))}}

    @app.delete("/api/providers/{provider_id:path}")
    def delete_provider(provider_id: str, st: AppState = Depends(get_state)):
        try:
            ok = remove_provider(provider_id, data_dir=st.data_dir)
        except PipelineError as exc:
            raise HTTPException(400, str(exc)) from exc
        if not ok:
            raise HTTPException(404, f"unknown provider {provider_id}")
        return {"ok": True, "id": provider_id}

    @app.post("/api/models/cancel")
    def cancel_model(st: AppState = Depends(get_state)):
        return {"cancelled": cancel_downloads(st)}

    @app.get("/api/doctor")
    def doctor(force: bool = False):
        return doctor_payload(force=force)

    @app.get(
        "/api/options",
        summary="Profiles, devices, formats, catalog defaults",
        responses={401: {"model": ErrorBody}},
    )
    def options(st: AppState = Depends(get_state)):
        return options_payload(st)

    @app.get("/api/ollama")
    def ollama():
        return _ollama_payload()

    return app


def _zip_directory_response(path: Path) -> Response:
    """Zip an HLS/DASH package directory for download."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(path.rglob("*")):
            if file_path.is_file():
                zf.write(file_path, file_path.relative_to(path).as_posix())
    data = buf.getvalue()
    return Response(
        content=data,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{path.name}.zip"',
            "Content-Length": str(len(data)),
        },
    )


def _ollama_payload() -> dict[str, Any]:
    names = ollama_model_names(timeout=2.0, force=True)
    return {
        "id": "ollama",
        "kind": "openai_compat",
        "base_url": os.environ.get("VIDEOCLEAN_OLLAMA_URL") or "http://127.0.0.1:11434",
        "ok": names is not None,
        "models": names or [],
        "note": "builtin OpenAI-compatible provider; add more via POST /api/providers",
    }


async def _save_upload(upload: UploadFile, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as handle:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    await upload.close()


def shutdown_serve(state: AppState, join_s: float = 30) -> None:
    for row in state.jobs.list_jobs(state="RUNNING", limit=10_000):
        state.jobs.request_cancel(row["id"])
    if state.worker is not None:
        state.worker.stop(timeout=join_s)


def install_serve_signal_handlers(state: AppState, join_s: float = 30):
    previous = {
        signal.SIGINT: signal.getsignal(signal.SIGINT),
        signal.SIGTERM: signal.getsignal(signal.SIGTERM),
    }

    def handler(signum, frame):
        shutdown_serve(state, join_s=min(float(join_s), 5.0))
        prev = previous.get(signum)
        if callable(prev):
            prev(signum, frame)
            return
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)

    def restore() -> None:
        for sig, prev in previous.items():
            if prev is None:
                continue
            signal.signal(sig, prev)

    return restore


def launch_ui(state: AppState, host: str, port: int) -> None:
    import uvicorn

    auth_from_env()
    app = create_app(state)
    uvicorn.run(app, host=host, port=int(port), log_level="info", timeout_keep_alive=120)


def launch_from_env(
    *,
    host: str = "0.0.0.0",
    port: int | None = None,
    data_dir: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    env_map = os.environ if env is None else env
    from videoclean.adapters.hf_cache import relax_hf_transfer_flag

    relax_hf_transfer_flag()
    auth_from_env(env_map)
    port_i = int(port if port is not None else env_map.get("VIDEOCLEAN_PORT") or 7860)
    root = Path(data_dir or env_map.get("VIDEOCLEAN_DATA_DIR") or (Path.home() / ".videoclean"))
    state = build_app_state(root)
    state.manage.recover_orphans()
    if state.worker is not None:
        state.worker.start()
    restore = install_serve_signal_handlers(state)
    try:
        launch_ui(state, host, port_i)
    finally:
        shutdown_serve(state)
        restore()
