from __future__ import annotations

import os
import secrets
import signal
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from videoclean.adapters.models.catalog import add_extra, ollama_model_names
from videoclean.adapters.web.app_state import AppState, build_app_state
from videoclean.adapters.web.service import (
    api_token,
    auth_from_env,
    cancel_downloads,
    default_device,
    doctor_payload,
    downloads_payload,
    grouped_models,
    job_dict,
    options_payload,
    queue_clean_job,
    serialize_clean_form,
    start_download,
)
from videoclean.application.errors import PipelineError

STATIC_DIR = Path(__file__).resolve().parent / "static"
INDEX_HTML = STATIC_DIR / "index.html"
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}


class BasicOrBearerAuth(BaseHTTPMiddleware):
    def __init__(self, app, user: str, password: str, token: str) -> None:
        super().__init__(app)
        self.user = user
        self.password = password
        self.token = token

    async def dispatch(self, request: Request, call_next):
        if request.method == "OPTIONS" or request.url.path == "/health":
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
    app = FastAPI(title="videoclean", docs_url="/api/docs", redoc_url=None)
    app.state.vc = state
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
    user, password = auth_from_env()
    app.add_middleware(BasicOrBearerAuth, user=user, password=password, token=api_token())
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    def get_state() -> AppState:
        return app.state.vc

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.get("/", response_class=HTMLResponse)
    @app.get("/config", response_class=HTMLResponse)
    def index():
        return INDEX_HTML.read_text(encoding="utf-8")

    @app.get("/api")
    def api_index():
        return {
            "ui": ["/", "/config"],
            "auth": {
                "browser": "HTTP Basic (VIDEOCLEAN_UI_USER / VIDEOCLEAN_UI_PASSWORD)",
                "api": "Authorization: Bearer VIDEOCLEAN_API_TOKEN (falls back to UI password)",
            },
            "jobs": {
                "POST /api/jobs": "multipart: video + prompt (+ optional pipeline fields)",
                "GET /api/jobs": "list",
                "GET /api/jobs/{id}": "status",
                "GET /api/jobs/{id}/output": "download cleaned file when COMPLETED",
                "GET /api/jobs/{id}/input": "source file",
                "POST /api/jobs/{id}/cancel": "",
                "POST /api/jobs/{id}/retry": "",
                "DELETE /api/jobs/{id}": "",
            },
            "models": {
                "GET /api/models": "grouped by kind",
                "POST /api/models/download": '{"id": "detector:grounding-dino"}',
                "POST /api/models/custom": '{"kind","backend","model_ref"}',
                "POST /api/models/cancel": "",
            },
        }

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
            "device": default_device(),
        }

    @app.get("/api/jobs")
    def list_jobs(state_filter: str | None = None, st: AppState = Depends(get_state)):
        filt = None if not state_filter or state_filter == "all" else state_filter
        return [job_dict(st, row) for row in st.jobs.list_jobs_full(limit=100, state=filt)]

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str, st: AppState = Depends(get_state)):
        row = st.jobs.get(job_id)
        if row is None:
            raise HTTPException(404, f"unknown job {job_id}")
        return job_dict(st, row)

    @app.post("/api/jobs")
    async def create_job(
        st: AppState = Depends(get_state),
        video: UploadFile | None = File(None),
        prompt: str = Form(""),
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
        telea_radius: str = Form(""),
        verify: str = Form(""),
        prompt_frame_stride: str = Form(""),
        prompt_frame_max: str = Form(""),
        overwrite: str = Form(""),
    ):
        fields = {
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
            "telea_radius": telea_radius,
            "verify": verify,
            "prompt_frame_stride": prompt_frame_stride,
            "prompt_frame_max": prompt_frame_max,
            "overwrite": overwrite,
        }
        payload = serialize_clean_form({k: v for k, v in fields.items() if v != ""})
        if video is None or not (video.filename or "").strip():
            raise HTTPException(400, "video file is required")
        suffix = Path(video.filename or "input.mp4").suffix.lower() or ".mp4"
        if suffix not in VIDEO_SUFFIXES:
            raise HTTPException(400, f"unsupported video type {suffix}")
        tmp_dir = Path(st.data_dir) / "uploads" / "_incoming"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp = tmp_dir / f"up_{os.getpid()}_{video.filename}"
        try:
            await _save_upload(video, tmp)
            job_id = queue_clean_job(
                st, tmp, prompt, payload, original_name=video.filename or tmp.name
            )
        except PipelineError as exc:
            raise HTTPException(400, str(exc)) from exc
        finally:
            tmp.unlink(missing_ok=True)
        row = st.jobs.get(job_id)
        body = job_dict(st, row) if row is not None else {"id": job_id, "state": "QUEUED"}
        body["poll"] = f"/api/jobs/{job_id}"
        body["download"] = f"/api/jobs/{job_id}/output"
        return JSONResponse(body, status_code=201)

    @app.get("/api/jobs/{job_id}/output")
    def job_output(job_id: str, st: AppState = Depends(get_state)):
        row = st.jobs.get(job_id)
        if row is None:
            raise HTTPException(404, f"unknown job {job_id}")
        if row["state"] != "COMPLETED":
            raise HTTPException(409, f"job is {row['state']}")
        path = Path(row["output_path"] or "")
        if not path.is_file():
            raise HTTPException(404, "output file missing")
        return FileResponse(path, filename=path.name, media_type="application/octet-stream")

    @app.get("/api/jobs/{job_id}/input")
    def job_input(job_id: str, st: AppState = Depends(get_state)):
        row = st.jobs.get(job_id)
        if row is None:
            raise HTTPException(404, f"unknown job {job_id}")
        path = Path(row["input_path"] or "")
        if not path.is_file():
            raise HTTPException(404, "input file missing")
        return FileResponse(path, filename=path.name, media_type="application/octet-stream")

    @app.post("/api/jobs/{job_id}/cancel")
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

    @app.post("/api/models/cancel")
    def cancel_model(st: AppState = Depends(get_state)):
        return {"cancelled": cancel_downloads(st)}

    @app.get("/api/doctor")
    def doctor(force: bool = False):
        return doctor_payload(force=force)

    @app.get("/api/options")
    def options(st: AppState = Depends(get_state)):
        return options_payload(st)

    @app.get("/api/ollama")
    def ollama():
        return _ollama_payload()

    return app


def _ollama_payload() -> dict[str, Any]:
    names = ollama_model_names(timeout=2.0, force=True)
    return {
        "id": "ollama",
        "kind": "openai_compat",
        "base_url": os.environ.get("VIDEOCLEAN_OLLAMA_URL") or "http://127.0.0.1:11434",
        "ok": names is not None,
        "models": names or [],
        "note": "first LLM provider. extra OpenAI-compatible endpoints can be added later.",
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
