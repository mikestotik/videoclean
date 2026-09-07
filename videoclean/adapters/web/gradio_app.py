from __future__ import annotations

import json
import os
import shutil
import signal
import threading
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gradio as gr
from gradio.events import SelectData

from videoclean.adapters.models.catalog import COMPONENTS, backend_ready
from videoclean.adapters.web.app_state import AppState, build_app_state
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
from videoclean.store import JobIndex, new_job_id

NONE_READY = "(none ready — see Models)"
JOB_TABLE_HEADERS = ["id", "state", "prompt", "created", "updated", "progress", "backends"]
MODEL_TABLE_HEADERS = ["id", "title", "status", "size", "message"]
JOB_STATES = ("all", "QUEUED", "RUNNING", "COMPLETED", "FAILED", "CANCELLED")
PROMPT_MAX = 40

_PORT_NAMES: dict[str, tuple[str, ...]] = {
    "detector": DETECTORS,
    "segmenter": SEGMENTERS,
    "inpainter": INPAINTERS,
}

_CSS = """
.gradio-container { max-width: 1100px !important; }
footer { display: none !important; }
#vc-progress textarea, #vc-doctor textarea, #vc-download-msg textarea {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 13px !important;
}
.tab-nav button { font-weight: 600 !important; }
"""

_doctor_cache: tuple[float, str] | None = None
_DOCTOR_TTL_S = 60.0


def ready_choices(port: str, catalog) -> list[str]:
    port = (port or "").strip().lower()
    if port == "llm":
        return list(LLM_PLACES)
    names = _PORT_NAMES.get(port)
    if not names:
        return []
    return [name for name in names if backend_ready(port, name, catalog=catalog)]


def missing_hint(port: str, catalog) -> str:
    port = (port or "").strip().lower()
    names = _PORT_NAMES.get(port) or ()
    ready = set(ready_choices(port, catalog))
    missing = [name for name in names if name not in ready]
    if not missing:
        return ""
    return "To enable " + ", ".join(missing) + " → Models"


def max_quality_ready(catalog) -> bool:
    return (
        backend_ready("detector", "grounding-dino", catalog=catalog)
        and backend_ready("segmenter", "sam2-video", catalog=catalog)
        and backend_ready("inpainter", "propainter", catalog=catalog)
    )


def serialize_clean_form(
    device: str = "cpu",
    detector: str = "grounding-dino",
    segmenter: str = "sam2",
    inpainter: str = "opencv-telea",
    llm_place: str = "auto",
    llm_model: str = "",
    fmt: str = "mp4",
    detector_threshold: float = 0.15,
    mask_dilate_px: int = 3,
    telea_radius: int = 9,
    verify: bool = True,
    prompt_frame_stride: int = 4,
    prompt_frame_max: int = 8,
    overwrite: bool = True,
    detector_model: str = "",
    segmenter_model: str = "",
    inpainter_model: str = "",
    **_extra: Any,
) -> dict[str, Any]:
    detector = (detector or "").strip().lower()
    segmenter = (segmenter or "").strip().lower()
    inpainter = (inpainter or "").strip().lower()
    if not detector_model:
        detector_model = (
            DEFAULT_DETECTOR_MODEL if detector == "owlvit" else DEFAULT_GROUNDING_DINO_MODEL
        )
    if not segmenter_model:
        segmenter_model = DEFAULT_SEGMENTER_MODEL
    if not inpainter_model:
        inpainter_model = DEFAULT_INPAINTER_MODEL
    fmt_name = (fmt or "mp4").strip().lower() or "mp4"
    return {
        "device": (device or "cpu").strip().lower() or "cpu",
        "detector": detector,
        "detectors": [detector] if detector else [],
        "detector_model": detector_model,
        "detector_threshold": float(detector_threshold),
        "segmenter": segmenter,
        "segmenter_model": segmenter_model,
        "inpainter": inpainter,
        "inpainter_model": inpainter_model,
        "llm_place": (llm_place or "auto").strip().lower() or "auto",
        "llm_model": (llm_model or "").strip(),
        "formats": [fmt_name],
        "allow_download": False,
        "verify": bool(verify),
        "mask_dilate_px": int(mask_dilate_px),
        "telea_radius": int(telea_radius),
        "prompt_frame_stride": int(prompt_frame_stride),
        "prompt_frame_max": int(prompt_frame_max),
        "overwrite": bool(overwrite),
    }


def format_jobs_table(rows) -> list[list]:
    table: list[list] = []
    for row in rows:
        prompt = _truncate(_row_get(row, "prompt"), PROMPT_MAX)
        progress = _progress_label(_row_get(row, "progress_json"))
        backends = _backends_label(_row_get(row, "request_json"))
        table.append(
            [
                _row_get(row, "id"),
                _row_get(row, "state"),
                prompt,
                _short_ts(_row_get(row, "created_at")),
                _short_ts(_row_get(row, "updated_at")),
                progress,
                backends,
            ]
        )
    return table


def format_models_table(statuses) -> list[list]:
    rows: list[list] = []
    for status in statuses:
        info = status.info
        rows.append([info.id, info.title, status.state, info.size_hint, status.message])
    return rows


def format_active_progress(row, now: datetime | None = None) -> str:
    if row is None:
        return "No running job."
    job_id = _row_get(row, "id") or "—"
    state = _row_get(row, "state") or "—"
    payload = _as_dict(_row_get(row, "progress_json"))
    stage = str(payload.get("stage") or "—")
    detail = str(payload.get("detail") or "")
    try:
        frac = float(payload.get("fraction") or 0.0)
    except (TypeError, ValueError):
        frac = 0.0
    titles = {key: title for key, title, _weight in STAGES}
    title = titles.get(stage, stage)
    pct = f"{max(0.0, min(frac, 1.0)) * 100:.1f}%"
    headline = f"{pct} · {title}" + (f" — {detail}" if detail else "")
    eta = eta_label(row, now=now)
    if eta:
        headline = f"{headline} · {eta}"
    lines = [
        f"**{job_id}** `{state}`",
        headline,
        "",
    ]
    for key, stage_title, _weight in STAGES:
        mark = "→" if key == stage else ("✓" if _stage_before(key, stage) else "·")
        lines.append(f"{mark} {stage_title}")
    error = _row_get(row, "error")
    if error:
        lines.extend(["", f"Error: {error}"])
    return "\n".join(lines)


def eta_label(row, now: datetime | None = None) -> str:
    if row is None:
        return ""
    payload = _as_dict(_row_get(row, "progress_json"))
    try:
        frac = float(payload.get("fraction") or 0.0)
    except (TypeError, ValueError):
        frac = 0.0
    started = _parse_ts(payload.get("started_at")) or _parse_ts(_row_get(row, "created_at"))
    if started is None:
        return "ETA —"
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    elapsed = max(0.0, (current - started).total_seconds())
    if frac < 0.08:
        return "ETA —"
    remaining = max(0.0, elapsed / max(frac, 1e-6) - elapsed)
    return f"ETA {_fmt_seconds(remaining)}"


def as_path(value: Any) -> Path | None:
    if value is None or value == "":
        return None
    if isinstance(value, Path):
        return value
    if isinstance(value, dict):
        for key in ("path", "name", "video"):
            if value.get(key):
                return as_path(value[key])
        return None
    if isinstance(value, (list, tuple)) and value:
        return as_path(value[0])
    return Path(str(value))


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


def default_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # noqa: BLE001 — UI default must not crash without torch
        pass
    return "cpu"


def is_job_stale(
    row,
    *,
    now: datetime | None = None,
    stale_seconds: int | None = None,
) -> bool:
    if _row_get(row, "state") != "RUNNING":
        return False
    seconds = int(stale_seconds if stale_seconds is not None else _stale_seconds())
    heartbeat = ""
    payload = _as_dict(_row_get(row, "progress_json"))
    heartbeat = str(payload.get("heartbeat_at") or "") or _row_get(row, "updated_at")
    if not heartbeat:
        return True
    try:
        ts = datetime.fromisoformat(str(heartbeat).replace("Z", "+00:00"))
    except ValueError:
        return True
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return (current - ts).total_seconds() > seconds


def queue_clean_job(state: AppState, video: Any, prompt: str, request: dict[str, Any]) -> str:
    prompt = (prompt or "").strip()
    if not prompt:
        raise PipelineError("prompt is required")
    src = as_path(video)
    if src is None or not src.is_file():
        raise PipelineError("upload a video file first")
    job_id = new_job_id()
    dest_dir = Path(state.data_dir) / "uploads" / job_id / "input"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    if src.resolve() != dest.resolve():
        shutil.copy2(src, dest)
    output_path = Path(state.data_dir) / "jobs" / job_id / "output" / "cleaned.mp4"
    payload = dict(request or {})
    payload["allow_download"] = False
    payload["input_path"] = str(dest)
    payload["output_path"] = str(output_path)
    payload["prompt"] = prompt
    return state.manage.submit(payload, dest, output_path, prompt, job_id=job_id)


def list_full_jobs(jobs: JobIndex, state: str | None = None, limit: int = 100) -> list:
    return list(jobs.list_jobs_full(limit=limit, state=state))


def format_doctor_text(*, force: bool = False) -> str:
    """Cached — torch/cuda probe is ~1s cold and must not run on every Timer tick."""
    global _doctor_cache
    import time

    now = time.monotonic()
    if not force and _doctor_cache is not None and (now - _doctor_cache[0]) < _DOCTOR_TTL_S:
        return _doctor_cache[1]
    from videoclean.composition import machine_facts

    facts = machine_facts()
    order = ("python", "ffmpeg", "ffprobe", "opencv", "torch", "cuda", "mps")
    text = "\n".join(f"{key}: {facts.get(key, '—')}" for key in order)
    _doctor_cache = (now, text)
    return text


def _ui_busy(state: AppState) -> bool:
    if _active_downloads(state):
        return True
    for st in ("RUNNING", "QUEUED"):
        if state.jobs.list_jobs(state=st, limit=1):
            return True
    return False


def build_ui(state: AppState):
    catalog = state.catalog
    det0 = _choice_list(ready_choices("detector", catalog), "grounding-dino")
    seg0 = _choice_list(ready_choices("segmenter", catalog), "sam2")
    inp0 = _choice_list(ready_choices("inpainter", catalog), "opencv-telea")

    with gr.Blocks(title="videoclean") as demo:
        gr.Markdown(
            "# videoclean\n"
            "Remove a named object from video. One GPU cleanup at a time — extras stay "
            "**QUEUED**. Download weights on **Models** before they appear in the selects."
        )
        with gr.Tabs():
            with gr.Tab("Clean"):
                with gr.Row():
                    with gr.Column(scale=3):
                        video = gr.File(
                            label="Video",
                            file_types=[".mp4", ".mov", ".mkv", ".webm"],
                            file_count="single",
                        )
                        prompt = gr.Textbox(
                            label="Prompt",
                            placeholder="remove the watermark in the corner",
                            lines=2,
                        )
                        with gr.Row():
                            device = gr.Dropdown(
                                label="Device",
                                choices=["cpu", "cuda", "mps"],
                                value=default_device(),
                            )
                            fmt = gr.Dropdown(
                                label="Format",
                                choices=["mp4", "mov", "mkv", "webm"],
                                value="mp4",
                            )
                        with gr.Row():
                            detector = gr.Dropdown(
                                label="Detector",
                                choices=det0[0],
                                value=det0[1],
                            )
                            segmenter = gr.Dropdown(
                                label="Segmenter",
                                choices=seg0[0],
                                value=seg0[1],
                            )
                            inpainter = gr.Dropdown(
                                label="Inpainter",
                                choices=inp0[0],
                                value=inp0[1],
                            )
                        det_hint = gr.Markdown(missing_hint("detector", catalog))
                        seg_hint = gr.Markdown(missing_hint("segmenter", catalog))
                        inp_hint = gr.Markdown(missing_hint("inpainter", catalog))
                        with gr.Row():
                            llm_place = gr.Dropdown(
                                label="LLM",
                                choices=list(LLM_PLACES),
                                value="auto",
                            )
                            llm_model = gr.Textbox(
                                label="LLM model",
                                placeholder="llama3.2 / llava-phi3 / empty for default",
                            )
                        with gr.Accordion("Advanced", open=False):
                            detector_threshold = gr.Slider(
                                0.01, 0.9, value=0.15, step=0.01, label="Detector threshold"
                            )
                            mask_dilate_px = gr.Slider(
                                0, 32, value=3, step=1, label="Mask dilate (px)"
                            )
                            telea_radius = gr.Slider(
                                1, 21, value=9, step=1, label="TELEA radius"
                            )
                            verify = gr.Checkbox(value=True, label="Verify")
                            prompt_frame_stride = gr.Number(
                                value=4, precision=0, label="Prompt frame stride"
                            )
                            prompt_frame_max = gr.Number(
                                value=8, precision=0, label="Prompt frame max"
                            )
                            overwrite = gr.Checkbox(value=True, label="Overwrite output")
                        with gr.Row():
                            max_btn = gr.Button(
                                "Max quality",
                                interactive=max_quality_ready(catalog),
                            )
                            submit_btn = gr.Button("Submit", variant="primary")
                        submit_status = gr.Markdown()
                    with gr.Column(scale=2):
                        gr.Markdown("### Live job")
                        live_progress = gr.Markdown(
                            value=_live_progress_text(state),
                            elem_id="vc-progress",
                        )
            with gr.Tab("Models"):
                models_table = gr.Dataframe(
                    headers=MODEL_TABLE_HEADERS,
                    value=format_models_table(_safe_list_status(catalog)),
                    wrap=True,
                    interactive=False,
                    label="Catalog",
                )
                download_bar = gr.Slider(
                    minimum=0,
                    maximum=1,
                    value=0,
                    interactive=False,
                    label="Active download %",
                )
                download_msg = gr.Textbox(
                    label="Download status",
                    value=_download_message(state),
                    lines=2,
                    interactive=False,
                    elem_id="vc-download-msg",
                )
                gr.Markdown("Download a component (one at a time; OK during a cleanup job):")
                download_buttons: dict[str, Any] = {}
                with gr.Column():
                    for info in COMPONENTS:
                        with gr.Row():
                            gr.Markdown(f"**{info.title}** `{info.id}` · {info.size_hint}")
                            download_buttons[info.id] = gr.Button("Download", scale=0, size="sm")
                with gr.Row():
                    refresh_models = gr.Button("Refresh status")
                    cancel_dl = gr.Button("Cancel download")
                doctor = gr.Textbox(
                    label="Doctor (ffmpeg / torch / cuda)",
                    value=format_doctor_text(),
                    lines=8,
                    interactive=False,
                    elem_id="vc-doctor",
                )
            with gr.Tab("Jobs"):
                with gr.Row():
                    state_filter = gr.Dropdown(
                        label="Filter",
                        choices=list(JOB_STATES),
                        value="all",
                    )
                    refresh_jobs = gr.Button("Refresh")
                jobs_table = gr.Dataframe(
                    headers=JOB_TABLE_HEADERS,
                    value=format_jobs_table(list_full_jobs(state.jobs)),
                    wrap=True,
                    interactive=False,
                    label="Queue",
                )
                job_id_box = gr.Textbox(label="Job id", placeholder="select a row or paste an id")
                with gr.Row():
                    cancel_btn = gr.Button("Cancel")
                    retry_btn = gr.Button("Retry")
                    delete_btn = gr.Button("Delete")
                    fail_btn = gr.Button("Mark failed")
                jobs_status = gr.Markdown()
                output_file = gr.File(label="Download output", interactive=False)

        # Idle by default. User clicks must not sit behind a 1Hz full-UI rewrite.
        timer = gr.Timer(1.0, active=_ui_busy(state))

        def _refresh_clean_selects(current_det, current_seg, current_inp):
            d = _choice_list(ready_choices("detector", state.catalog), "grounding-dino", current_det)
            s = _choice_list(ready_choices("segmenter", state.catalog), "sam2", current_seg)
            i = _choice_list(
                ready_choices("inpainter", state.catalog), "opencv-telea", current_inp
            )
            return (
                gr.update(choices=d[0], value=d[1]),
                gr.update(choices=s[0], value=s[1]),
                gr.update(choices=i[0], value=i[1]),
                missing_hint("detector", state.catalog),
                missing_hint("segmenter", state.catalog),
                missing_hint("inpainter", state.catalog),
                gr.update(interactive=max_quality_ready(state.catalog)),
            )

        def _on_max_quality():
            return "cuda", "grounding-dino", "sam2-video", "propainter"

        def _on_submit(
            video_val,
            prompt_val,
            device_val,
            detector_val,
            segmenter_val,
            inpainter_val,
            llm_place_val,
            llm_model_val,
            fmt_val,
            thr,
            dilate,
            radius,
            verify_val,
            stride,
            frame_max,
            overwrite_val,
        ):
            try:
                _validate_backend_choice("detector", detector_val, state.catalog)
                _validate_backend_choice("segmenter", segmenter_val, state.catalog)
                _validate_backend_choice("inpainter", inpainter_val, state.catalog)
                payload = serialize_clean_form(
                    device=device_val,
                    detector=detector_val,
                    segmenter=segmenter_val,
                    inpainter=inpainter_val,
                    llm_place=llm_place_val,
                    llm_model=llm_model_val,
                    fmt=fmt_val,
                    detector_threshold=float(thr),
                    mask_dilate_px=int(dilate),
                    telea_radius=int(radius),
                    verify=bool(verify_val),
                    prompt_frame_stride=int(stride or 0),
                    prompt_frame_max=int(frame_max or 1),
                    overwrite=bool(overwrite_val),
                )
                job_id = queue_clean_job(state, video_val, prompt_val, payload)
            except Exception as exc:  # noqa: BLE001 — surface as UI text, not traceback
                return _ui_error(exc), _live_progress_text(state)
            extra = ""
            running = state.jobs.list_jobs(state="RUNNING", limit=1)
            if running and running[0]["id"] != job_id:
                extra = " A job is already running — this one waits in the FIFO queue."
            return (
                f"Queued **{job_id}**. Watch it on the Jobs tab.{extra}",
                _live_progress_text(state),
            )

        def _models_status_panel():
            """Catalog + download progress + doctor. No dropdown updates (avoids Gradio 422)."""
            table = format_models_table(_safe_list_status(state.catalog))
            frac, msg = _download_progress(state)
            return table, frac, msg, format_doctor_text()

        def _on_refresh_models(current_det, current_seg, current_inp):
            selects = _refresh_clean_selects(current_det, current_seg, current_inp)
            return (*_models_status_panel(), *selects)

        def _on_cancel_download(current_det, current_seg, current_inp):
            msg = _cancel_download(state)
            table, _, _, doctor, *selects = _on_refresh_models(
                current_det, current_seg, current_inp
            )
            # Always clear the bar on cancel — do not keep a mid-download %.
            return (table, 0.0, msg, doctor, *selects, gr.update(active=_ui_busy(state)))
        def _jobs_refresh(filter_val, selected_id):
            st = None if not filter_val or filter_val == "all" else str(filter_val)
            table = format_jobs_table(list_full_jobs(state.jobs, state=st))
            status = f"{len(table)} job(s)." if table else "No jobs yet."
            out = _output_file(state, selected_id)
            # gr.File rejects bare None in some Gradio 6 queue validations.
            if out is None:
                out = gr.update(value=None)
            return table, status, out, _live_progress_text(state)

        def _on_job_select(evt: SelectData):
            row_value = getattr(evt, "row_value", None)
            if isinstance(row_value, (list, tuple)) and row_value:
                return str(row_value[0])
            index = getattr(evt, "index", None)
            value = getattr(evt, "value", None)
            col = index[1] if isinstance(index, (list, tuple)) and len(index) > 1 else 0
            if value is not None and col == 0:
                return str(value)
            return gr.update()

        def _act(fn, job_id, filter_val):
            try:
                msg = fn(state, (job_id or "").strip())
            except Exception as exc:  # noqa: BLE001
                msg = _ui_error(exc)
            table, status, out, live = _jobs_refresh(filter_val, job_id)
            return table, f"{msg}\n\n{status}", out, live

        max_btn.click(
            _on_max_quality,
            outputs=[device, detector, segmenter, inpainter],
        )

        # Full refresh (includes Clean selects) — only on button / download actions.
        model_outputs = [
            models_table,
            download_bar,
            download_msg,
            doctor,
            detector,
            segmenter,
            inpainter,
            det_hint,
            seg_hint,
            inp_hint,
            max_btn,
        ]
        # Lightweight poll: progress text + bar. Heavy Dataframes/doctor only while busy.
        # Timer is an output so it can sleep when the queue is idle (keeps tabs snappy).
        poll_outputs = [
            download_bar,
            download_msg,
            live_progress,
            models_table,
            jobs_table,
            jobs_status,
            timer,
        ]
        refresh_models.click(
            _on_refresh_models,
            inputs=[detector, segmenter, inpainter],
            outputs=model_outputs,
        )
        # queue=False: cancel must run even if a download/poll event is in flight.
        cancel_dl.click(
            _on_cancel_download,
            inputs=[detector, segmenter, inpainter],
            outputs=[*model_outputs, timer],
            queue=False,
        )
        for cid, btn in download_buttons.items():
            btn.click(
                make_download_click_handler(state, cid),
                inputs=[detector, segmenter, inpainter],
                outputs=[*model_outputs, timer],
            )

        jobs_outputs = [jobs_table, jobs_status, output_file, live_progress]
        refresh_jobs.click(_jobs_refresh, inputs=[state_filter, job_id_box], outputs=jobs_outputs)
        state_filter.change(_jobs_refresh, inputs=[state_filter, job_id_box], outputs=jobs_outputs)
        jobs_table.select(_on_job_select, outputs=[job_id_box])

        def _on_poll(filter_val, selected_id):
            frac, msg = _download_progress(state)
            live = _live_progress_text(state)
            busy = _ui_busy(state)
            if _active_downloads(state):
                models = format_models_table(_safe_list_status(state.catalog))
            else:
                models = gr.update()
            if any(state.jobs.list_jobs(state=st, limit=1) for st in ("RUNNING", "QUEUED")):
                st = None if not filter_val or filter_val == "all" else str(filter_val)
                jobs = format_jobs_table(list_full_jobs(state.jobs, state=st))
                status = f"{len(jobs)} job(s)." if jobs else "No jobs yet."
            else:
                jobs = gr.update()
                status = gr.update()
            return frac, msg, live, models, jobs, status, gr.update(active=busy)

        def _wrap_job_act(fn):
            def _inner(job_id, filt):
                return (*_act(fn, job_id, filt), gr.update(active=_ui_busy(state)))

            return _inner

        cancel_btn.click(
            _wrap_job_act(_cancel_job),
            inputs=[job_id_box, state_filter],
            outputs=[*jobs_outputs, timer],
        )
        retry_btn.click(
            _wrap_job_act(_retry_job),
            inputs=[job_id_box, state_filter],
            outputs=[*jobs_outputs, timer],
        )
        delete_btn.click(
            lambda job_id, filt: _act(delete_job, job_id, filt),
            inputs=[job_id_box, state_filter],
            outputs=jobs_outputs,
        )
        fail_btn.click(
            lambda job_id, filt: _act(_mark_failed_job, job_id, filt),
            inputs=[job_id_box, state_filter],
            outputs=jobs_outputs,
        )

        timer.tick(
            _on_poll,
            inputs=[state_filter, job_id_box],
            outputs=poll_outputs,
            show_progress="hidden",
            concurrency_limit=1,
        )
        # No heavy demo.load refresh — build_ui already filled tables/selects.
        # A full post-login reload was blocking the Gradio queue for seconds.

        # Submit wakes the timer while a job is queued/running.
        submit_btn.click(
            lambda *args: (*_on_submit(*args), gr.update(active=True)),
            inputs=[
                video,
                prompt,
                device,
                detector,
                segmenter,
                inpainter,
                llm_place,
                llm_model,
                fmt,
                detector_threshold,
                mask_dilate_px,
                telea_radius,
                verify,
                prompt_frame_stride,
                prompt_frame_max,
                overwrite,
            ],
            outputs=[submit_status, live_progress, timer],
        )
    return demo


def launch_ui(state: AppState, host: str, port: int, auth: tuple[str, str] | None) -> None:
    import warnings

    # Gradio 6 + current Starlette spam this on every Timer tick; not actionable.
    try:
        from starlette.exceptions import StarletteDeprecationWarning
    except ImportError:  # pragma: no cover
        StarletteDeprecationWarning = UserWarning  # type: ignore[misc, assignment]
    warnings.filterwarnings(
        "ignore",
        message=".*HTTP_422_UNPROCESSABLE_ENTITY.*",
        category=StarletteDeprecationWarning,
    )
    password = "" if auth is None else str(auth[1] or "")
    if not password.strip():
        raise RuntimeError(
            "UI password is required to launch (safer default for RunPod). "
            "Set VIDEOCLEAN_UI_PASSWORD and pass auth=(user, password)."
        )
    user = str(auth[0] or "admin") if auth else "admin"
    # Warm torch/cuda probe once so the first Models paint is not a 1s hitch.
    format_doctor_text(force=True)
    demo = build_ui(state)
    demo.launch(
        server_name=host,
        server_port=int(port),
        auth=(user, password),
        show_error=True,
        max_file_size="4gb",
        css=_CSS,
        allowed_paths=[str(state.data_dir)],
        theme=gr.themes.Soft(),
    )


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
        # Keep handler work short: cancel + bounded join, then exit.
        shutdown_serve(state, join_s=min(float(join_s), 5.0))
        prev = previous.get(signum)
        if callable(prev):
            prev(signum, frame)
            return
        # SIG_DFL / SIG_IGN are not callable — re-arm default and re-raise so
        # Docker/RunPod SIGTERM actually terminates the process.
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


def launch_from_env(
    *,
    host: str = "0.0.0.0",
    port: int | None = None,
    data_dir: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> None:
    env_map = os.environ if env is None else env
    auth = auth_from_env(env_map)
    port_i = int(port if port is not None else env_map.get("VIDEOCLEAN_PORT") or 7860)
    root = Path(data_dir or env_map.get("VIDEOCLEAN_DATA_DIR") or (Path.home() / ".videoclean"))
    state = build_app_state(root)
    state.manage.recover_orphans()
    if state.worker is not None:
        state.worker.start()
    restore = install_serve_signal_handlers(state)
    try:
        launch_ui(state, host, port_i, auth)
    finally:
        shutdown_serve(state)
        restore()


def _choice_list(
    names: list[str], preferred: str, current: str | None = None
) -> tuple[list[str], str]:
    choices = list(names) if names else [NONE_READY]
    if current in choices:
        return choices, current
    if preferred in choices:
        return choices, preferred
    return choices, choices[0]


def _validate_backend_choice(port: str, name: str, catalog) -> None:
    name = (name or "").strip()
    if not name or name == NONE_READY:
        raise PipelineError(f"no ready {port}; download weights on the Models tab")
    if not backend_ready(port, name, catalog=catalog):
        raise PipelineError(f"{port} {name} is not ready; download it on the Models tab")


def _ui_error(exc: BaseException) -> str:
    return str(exc)[:400]


def _live_progress_text(state: AppState) -> str:
    running = list_full_jobs(state.jobs, state="RUNNING", limit=1)
    if running:
        return format_active_progress(running[0])
    queued = list_full_jobs(state.jobs, state="QUEUED", limit=1)
    if queued:
        job_id = queued[0]["id"]
        return f"**{job_id}** `QUEUED`\n\nWaiting for the GPU worker."
    return format_active_progress(None)


def _safe_list_status(catalog) -> list:
    try:
        return list(catalog.list_status())
    except Exception:  # noqa: BLE001 — doctor/status must still render
        return []


def _active_downloads(state: AppState) -> list:
    rows: list = []
    for st in ("running", "queued"):
        rows.extend(state.jobs.list_downloads(limit=20, state=st))
    return rows


def _format_download_line(
    component_id: str,
    frac: float,
    message: str = "",
    bytes_done: int | None = None,
    bytes_total: int | None = None,
) -> str:
    frac = max(0.0, min(float(frac), 1.0))
    counts = ""
    if bytes_done is not None and bytes_total not in (None, 0):
        counts = f" ({bytes_done}/{bytes_total})"
    msg = (message or "").strip()
    return f"{component_id}: {int(round(frac * 100))}%{counts} {msg}".strip()


def _download_progress(state: AppState) -> tuple[float, str]:
    for row in _active_downloads(state):
        if row["cancel_requested"]:
            state.last_download_frac = 0.0
            state.last_download_msg = f"Cancelling {row['component_id']}…"
            return 0.0, state.last_download_msg
        try:
            frac = float(row["progress"] or 0.0)
        except (TypeError, ValueError):
            frac = 0.0
        line = _format_download_line(
            row["component_id"],
            frac,
            row["message"] or "",
            row["bytes_done"],
            row["bytes_total"],
        )
        state.last_download_frac = max(0.0, min(frac, 1.0))
        state.last_download_msg = line
        return state.last_download_frac, line

    # No active download: sync terminal status so the bar does not stick mid-%.
    recent = state.jobs.list_downloads(limit=1)
    if recent:
        row = recent[0]
        cid = row["component_id"]
        st = row["state"]
        if st == "done":
            state.last_download_frac = 1.0
            state.last_download_msg = f"Finished {cid} — ready"
        elif st == "cancelled":
            state.last_download_frac = 0.0
            state.last_download_msg = f"Cancelled {cid}"
        elif st == "failed":
            state.last_download_frac = 0.0
            detail = (row["message"] or "failed").strip()
            state.last_download_msg = f"Failed {cid}: {detail}"

    return float(state.last_download_frac or 0.0), state.last_download_msg or "No download in progress."


def _download_message(state: AppState) -> str:
    for row in _active_downloads(state):
        if row["cancel_requested"]:
            return f"Cancelling {row['component_id']}…"
        return f"Downloading {row['component_id']}…"
    return state.last_download_msg or "No download in progress."


def _start_download(state: AppState, component_id: str) -> str:
    """Start download on a daemon thread; UI progress comes from the Timer poll."""
    if state.downloader is None:
        return "Downloader is not configured."
    component_id = (component_id or "").strip()
    if not component_id:
        return "Pick a component."
    for row in _active_downloads(state):
        return f"Already downloading {row['component_id']}. Wait or cancel."
    downloader = state.downloader

    def _run() -> None:
        try:
            downloader.execute(component_id, state.jobs, None)
        except Exception:  # noqa: BLE001 — status is stored on the download row
            return

    threading.Thread(target=_run, name=f"vc-dl-{component_id}", daemon=True).start()
    state.last_download_frac = 0.01
    state.last_download_msg = f"Starting {component_id}…"
    return state.last_download_msg


def make_download_click_handler(state: AppState, component_id: str):
    """Bind component_id; return one model_outputs tuple (do not hold the Gradio queue)."""

    def _handler(current_det, current_seg, current_inp):
        msg = _start_download(state, component_id)
        table = format_models_table(_safe_list_status(state.catalog))
        doctor = format_doctor_text()
        # Keep Clean dropdowns unchanged until download finishes (Timer refreshes bar/msg).
        hold = (gr.update(),) * 7
        frac = 0.01 if msg.startswith("Starting") else float(state.last_download_frac or 0.0)
        if msg.startswith("Starting"):
            state.last_download_frac = 0.01
            state.last_download_msg = msg
            wake = gr.update(active=True)
        else:
            frac, _ = _download_progress(state)
            state.last_download_msg = msg
            wake = gr.update(active=_ui_busy(state))
        return (table, frac, msg, doctor, *hold, wake)

    return _handler

def _cancel_download(state: AppState) -> str:
    cancelled = []
    for row in state.jobs.list_downloads(limit=20):
        if row["state"] in {"running", "queued"}:
            state.jobs.request_download_cancel(row["id"])
            # Flip row to cancelled immediately so the Timer stops painting mid-%.
            state.jobs.upsert_download(
                row["id"],
                row["component_id"],
                "cancelled",
                progress=0.0,
                message="cancelled",
            )
            cancelled.append(row["component_id"])
    state.last_download_frac = 0.0
    if not cancelled:
        state.last_download_msg = "No running download."
        return state.last_download_msg
    state.last_download_msg = "Cancelled " + ", ".join(cancelled)
    return state.last_download_msg


def _cancel_job(state: AppState, job_id: str) -> str:
    if not job_id:
        return "Enter a job id."
    row = state.jobs.get(job_id)
    if row is None:
        return f"Unknown job {job_id}."
    if row["state"] not in {"QUEUED", "RUNNING"}:
        return f"Job {job_id} is {row['state']}; nothing to cancel."
    state.manage.cancel(job_id)
    after = state.jobs.get(job_id)
    if after is not None and after["state"] == "RUNNING":
        return f"Cancel requested for {job_id}; the worker stops at the next progress tick."
    return f"Job {job_id} → {after['state'] if after else 'CANCELLED'}."


def _retry_job(state: AppState, job_id: str) -> str:
    if not job_id:
        return "Enter a job id."
    new_id = state.manage.retry(job_id)
    return f"Queued retry {new_id} (from {job_id})."


def delete_job(state: AppState, job_id: str) -> str:
    if not job_id:
        return "Enter a job id."
    row = state.jobs.get(job_id)
    if row is None:
        return f"Unknown job {job_id}."
    input_path = row["input_path"] or ""
    state.manage.delete(job_id, state.data_dir)
    _maybe_remove_upload(state, input_path)
    return f"Deleted {job_id}."


def _maybe_remove_upload(state: AppState, input_path: str) -> None:
    upload_dir = upload_root_for(state.data_dir, input_path)
    if upload_dir is None or not upload_dir.exists():
        return
    if input_still_referenced(state.jobs, input_path):
        return
    shutil.rmtree(upload_dir, ignore_errors=True)


def upload_root_for(data_dir: Path, input_path: str | Path | None) -> Path | None:
    if not input_path:
        return None
    uploads = (Path(data_dir) / "uploads").resolve()
    try:
        resolved = Path(input_path).resolve()
        rel = resolved.relative_to(uploads)
    except (OSError, ValueError):
        return None
    if not rel.parts:
        return None
    return uploads / rel.parts[0]


def input_still_referenced(jobs: JobIndex, input_path: str | Path | None) -> bool:
    if not input_path:
        return False
    wanted = {str(input_path), str(Path(input_path))}
    try:
        wanted.add(str(Path(input_path).resolve()))
    except OSError:
        pass
    for row in jobs.list_jobs(limit=10_000):
        other = row["input_path"] or ""
        if not other:
            continue
        if other in wanted:
            return True
        try:
            if str(Path(other).resolve()) in wanted:
                return True
        except OSError:
            continue
    return False


def _mark_failed_job(state: AppState, job_id: str) -> str:
    if not job_id:
        return "Enter a job id."
    row = state.jobs.get(job_id)
    if row is None:
        return f"Unknown job {job_id}."
    if row["state"] != "RUNNING":
        return f"Job {job_id} is {row['state']}, not RUNNING."
    if not is_job_stale(row):
        return f"Job {job_id} still has a recent heartbeat. Wait or Cancel instead."
    state.manage.mark_failed(job_id, "stale heartbeat")
    return f"Marked {job_id} failed (stale heartbeat)."


def _output_file(state: AppState, job_id: str | None) -> str | None:
    job_id = (job_id or "").strip()
    if not job_id:
        return None
    row = state.jobs.get(job_id)
    if row is None or row["state"] != "COMPLETED":
        return None
    path = Path(row["output_path"] or "")
    if path.is_file():
        return str(path)
    report = _as_dict(row["report_json"] if "report_json" in row.keys() else None)
    for item in (report.get("outputs") or {}).values():
        candidate = Path(str(item))
        if candidate.is_file():
            return str(candidate)
    return None


def _stale_seconds() -> int:
    raw = os.environ.get("VIDEOCLEAN_STALE_SECONDS", "900")
    try:
        return max(1, int(raw))
    except ValueError:
        return 900


def _row_get(row, key: str, default: str = "") -> str:
    if row is None:
        return default
    if isinstance(row, dict):
        value = row.get(key, default)
        return default if value is None else str(value) if not isinstance(value, str) else value
    try:
        value = row[key]
    except (KeyError, IndexError, TypeError):
        return default
    if value is None:
        return default
    return value if isinstance(value, str) else str(value)


def _as_dict(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _truncate(text: str, n: int) -> str:
    text = (text or "").replace("\n", " ").strip()
    if len(text) <= n:
        return text
    return text[: n - 1] + "…"


def _short_ts(value: str) -> str:
    return (value or "").replace("T", " ")[:19]


def _parse_ts(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        ts = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts


def _progress_label(raw: str) -> str:
    payload = _as_dict(raw)
    stage = str(payload.get("stage") or "")
    frac = payload.get("fraction")
    if frac is None:
        return stage or "—"
    try:
        pct = f"{max(0.0, min(float(frac), 1.0)) * 100:.0f}%"
    except (TypeError, ValueError):
        return stage or "—"
    return f"{stage} {pct}".strip()


def _backends_label(raw: str) -> str:
    payload = _as_dict(raw)
    device = str(payload.get("device") or "")
    detector = payload.get("detector")
    if not detector:
        dets = payload.get("detectors") or []
        detector = ",".join(dets) if isinstance(dets, list) else str(dets or "")
    segmenter = str(payload.get("segmenter") or "")
    inpainter = str(payload.get("inpainter") or "")
    core = " / ".join(part for part in (str(detector), segmenter, inpainter) if part)
    if device and core:
        return f"{device} · {core}"
    return device or core or "—"


def _stage_before(key: str, current: str) -> bool:
    keys = [item[0] for item in STAGES]
    if key not in keys or current not in keys:
        return False
    return keys.index(key) < keys.index(current)
