import json
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from videoclean.adapters.web.app_state import AppState, build_app_state
from videoclean.adapters.web.gradio_app import (
    as_path,
    auth_from_env,
    default_device,
    delete_job,
    eta_label,
    format_active_progress,
    format_jobs_table,
    format_models_table,
    install_serve_signal_handlers,
    is_job_stale,
    launch_from_env,
    launch_ui,
    list_full_jobs,
    max_quality_ready,
    missing_hint,
    queue_clean_job,
    ready_choices,
    serialize_clean_form,
    shutdown_serve,
)
from videoclean.application.use_cases.manage_jobs import ManageJobs, pipeline_config_from_dict
from videoclean.store import JobIndex


class FakeCat:
    def is_ready(self, cid: str) -> bool:
        return cid in {"detector:grounding-dino", "inpainter:propainter"}


def test_ready_detector_choices():
    assert ready_choices("detector", FakeCat()) == ["grounding-dino"]


def test_ready_inpainter_includes_telea():
    assert ready_choices("inpainter", FakeCat()) == ["opencv-telea", "propainter"]


def test_ready_segmenter_empty_without_weights():
    assert ready_choices("segmenter", FakeCat()) == []


def test_missing_hint_points_at_models():
    hint = missing_hint("segmenter", FakeCat())
    assert "sam2" in hint
    assert "Models" in hint
    assert "owlvit" in missing_hint("detector", FakeCat())
    assert "lama" in missing_hint("inpainter", FakeCat())


def test_max_quality_ready_requires_three():
    assert max_quality_ready(FakeCat()) is False

    class Ready:
        def is_ready(self, cid: str) -> bool:
            return cid in {
                "detector:grounding-dino",
                "segmenter:sam2-tiny",
                "inpainter:propainter",
            }

    assert max_quality_ready(Ready()) is True


def test_serialize_clean_form_disables_download():
    payload = serialize_clean_form(
        device="cuda",
        detector="grounding-dino",
        segmenter="sam2-video",
        inpainter="propainter",
        llm_place="local",
        llm_model="llama3.2",
        fmt="mp4",
    )
    assert payload["allow_download"] is False
    assert payload["device"] == "cuda"
    assert payload["detectors"] == ["grounding-dino"]
    assert payload["segmenter"] == "sam2-video"
    assert payload["inpainter"] == "propainter"
    assert payload["formats"] == ["mp4"]
    assert payload["overwrite"] is True
    cfg = pipeline_config_from_dict(payload)
    assert cfg.allow_download is False
    assert cfg.detectors == ["grounding-dino"]
    cfg.validate()


def test_format_jobs_table():
    rows = [
        {
            "id": "j1",
            "state": "RUNNING",
            "prompt": "remove the watermark from the lower third please",
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:10+00:00",
            "progress_json": json.dumps({"stage": "detect", "fraction": 0.4}),
            "request_json": json.dumps(
                {
                    "device": "cuda",
                    "detector": "grounding-dino",
                    "segmenter": "sam2-video",
                    "inpainter": "propainter",
                }
            ),
        }
    ]
    table = format_jobs_table(rows)
    assert isinstance(table, str)
    assert "| j1 |" in table
    assert "RUNNING" in table
    assert "detect" in table
    assert "40%" in table


def test_format_jobs_table_empty():
    assert "No jobs" in format_jobs_table([])


def test_auth_from_env_requires_password():
    with pytest.raises(RuntimeError, match="VIDEOCLEAN_UI_PASSWORD"):
        auth_from_env({"VIDEOCLEAN_UI_USER": "admin"})
    with pytest.raises(RuntimeError, match="VIDEOCLEAN_UI_PASSWORD"):
        auth_from_env({"VIDEOCLEAN_UI_USER": "admin", "VIDEOCLEAN_UI_PASSWORD": "  "})


def test_auth_from_env_ok():
    assert auth_from_env(
        {"VIDEOCLEAN_UI_USER": "admin", "VIDEOCLEAN_UI_PASSWORD": "secret"}
    ) == ("admin", "secret")


def test_auth_from_env_defaults_user():
    assert auth_from_env({"VIDEOCLEAN_UI_PASSWORD": "secret"}) == ("admin", "secret")


def test_shutdown_serve_cancels_running_and_stops_worker(tmp_path: Path):
    jobs = JobIndex(tmp_path / "jobs.sqlite")
    jobs.upsert("r1", "RUNNING")
    jobs.upsert("q1", "QUEUED")

    class FakeWorker:
        def __init__(self):
            self.stop_timeout = None

        def stop(self, timeout=5):
            self.stop_timeout = timeout

    worker = FakeWorker()
    state = AppState(
        data_dir=tmp_path,
        jobs=jobs,
        manage=ManageJobs(jobs),
        catalog=FakeCat(),
        worker=worker,
    )
    shutdown_serve(state, join_s=30)
    assert jobs.is_cancel_requested("r1") is True
    assert jobs.is_cancel_requested("q1") is False
    assert worker.stop_timeout == 30


def test_install_serve_signal_handlers_wires_sigterm(monkeypatch, tmp_path: Path):
    import signal

    from videoclean.adapters.web import gradio_app as ga

    jobs = JobIndex(tmp_path / "jobs.sqlite")
    jobs.upsert("r1", "RUNNING")
    handlers: dict[int, object] = {}
    killed: list[tuple[int, int]] = []

    def fake_signal(sig, handler):
        handlers[sig] = handler
        return signal.SIG_DFL

    monkeypatch.setattr(ga.signal, "signal", fake_signal)
    monkeypatch.setattr(ga.signal, "getsignal", lambda sig: signal.SIG_DFL)
    monkeypatch.setattr(ga.os, "kill", lambda pid, sig: killed.append((pid, sig)))

    class FakeWorker:
        def stop(self, timeout=5):
            self.timeout = timeout

    worker = FakeWorker()
    state = AppState(
        data_dir=tmp_path,
        jobs=jobs,
        manage=ManageJobs(jobs),
        catalog=FakeCat(),
        worker=worker,
    )
    restore = install_serve_signal_handlers(state, join_s=30)
    assert signal.SIGTERM in handlers
    assert signal.SIGINT in handlers
    handlers[signal.SIGTERM](signal.SIGTERM, None)
    assert jobs.is_cancel_requested("r1") is True
    assert worker.timeout == 5  # signal path caps join at 5s
    assert killed and killed[0][1] == signal.SIGTERM
    restore()


def test_launch_from_env_stops_worker_on_return(monkeypatch, tmp_path: Path):
    pytest.importorskip("gradio")
    jobs = JobIndex(tmp_path / "jobs.sqlite")

    class FakeWorker:
        def __init__(self):
            self.started = False
            self.stop_timeout = None

        def start(self):
            self.started = True

        def stop(self, timeout=5):
            self.stop_timeout = timeout

    worker = FakeWorker()

    def fake_build(root, **kwargs):
        return AppState(
            data_dir=root,
            jobs=jobs,
            manage=ManageJobs(jobs),
            catalog=FakeCat(),
            worker=worker,
        )

    monkeypatch.setattr("videoclean.adapters.web.gradio_app.build_app_state", fake_build)
    monkeypatch.setattr(
        "videoclean.adapters.web.gradio_app.launch_ui",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "videoclean.adapters.web.gradio_app.install_serve_signal_handlers",
        lambda state, join_s=30: (lambda: None),
    )
    monkeypatch.setenv("VIDEOCLEAN_UI_PASSWORD", "secret")
    launch_from_env(host="127.0.0.1", port=7860, data_dir=tmp_path)
    assert worker.started is True
    assert worker.stop_timeout == 30


def test_launch_ui_refuses_missing_password(tmp_path: Path):
    jobs = JobIndex(tmp_path / "jobs.sqlite")
    state = AppState(
        data_dir=tmp_path,
        jobs=jobs,
        manage=ManageJobs(jobs),
        catalog=FakeCat(),
    )
    with pytest.raises(RuntimeError, match="password"):
        launch_ui(state, "127.0.0.1", 7860, None)
    with pytest.raises(RuntimeError, match="password"):
        launch_ui(state, "127.0.0.1", 7860, ("admin", ""))


def test_queue_clean_job_paths(tmp_path: Path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake-video")
    state = build_app_state(tmp_path, worker=False)
    payload = serialize_clean_form(device="cpu", detector="owlvit", inpainter="opencv-telea")
    job_id = queue_clean_job(state, video, "remove logo", payload)
    row = state.jobs.get(job_id)
    assert row is not None
    assert row["state"] == "QUEUED"
    input_path = Path(row["input_path"])
    assert input_path.is_file()
    assert input_path.read_bytes() == b"fake-video"
    assert input_path.parent == tmp_path / "uploads" / job_id / "input"
    assert input_path.name == "clip.mp4"
    assert Path(row["output_path"]) == tmp_path / "jobs" / job_id / "output" / "cleaned.mp4"
    request = json.loads(row["request_json"])
    assert request["allow_download"] is False
    assert request["prompt"] == "remove logo"


def test_queue_clean_job_requires_prompt_and_file(tmp_path: Path):
    from videoclean.application.errors import PipelineError

    state = build_app_state(tmp_path, worker=False)
    with pytest.raises(PipelineError, match="prompt"):
        queue_clean_job(state, None, "  ", {})
    with pytest.raises(PipelineError, match="video"):
        queue_clean_job(state, None, "remove logo", {})


def test_as_path_accepts_gradio_shapes(tmp_path: Path):
    clip = tmp_path / "a.mp4"
    clip.write_bytes(b"x")
    assert as_path(clip) == clip
    assert as_path(str(clip)) == clip
    assert as_path({"path": str(clip)}) == clip
    assert as_path({"video": str(clip)}) == clip
    assert as_path(None) is None


def test_format_models_table():
    from videoclean.application.ports.model_catalog import ComponentInfo, ComponentStatus

    info = ComponentInfo("detector:owlvit", "OWL-ViT", "detector", "google/owlvit", "~600 MB")
    rows = format_models_table([ComponentStatus(info, "missing", "not cached")])
    assert isinstance(rows, str)
    assert "detector:owlvit" in rows
    assert "missing" in rows


def test_format_active_progress_running():
    now = datetime(2026, 1, 1, 0, 1, 0, tzinfo=timezone.utc)
    text = format_active_progress(
        {
            "id": "job-1",
            "state": "RUNNING",
            "created_at": "2026-01-01T00:00:00+00:00",
            "progress_json": json.dumps(
                {"stage": "inpaint", "fraction": 0.55, "detail": "frame 10"}
            ),
        },
        now=now,
    )
    assert "job-1" in text
    assert "55" in text
    assert "inpaint" in text.lower() or "Inpaint" in text or "inpaint" in text
    assert "frame 10" in text
    assert "ETA" in text


def test_eta_label_too_early():
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    row = {
        "state": "RUNNING",
        "created_at": (now - timedelta(seconds=2)).isoformat(),
        "progress_json": json.dumps({"fraction": 0.02}),
    }
    assert eta_label(row, now=now) == "ETA —"


def test_eta_label_from_elapsed_fraction():
    now = datetime(2026, 1, 1, 0, 1, 40, tzinfo=timezone.utc)
    row = {
        "state": "RUNNING",
        "created_at": "2026-01-01T00:00:00+00:00",
        "progress_json": json.dumps({"fraction": 0.5}),
    }
    # 100s elapsed at 50% → ~100s remaining
    assert eta_label(row, now=now) == "ETA 01:40"


def test_format_active_progress_idle():
    assert "No" in format_active_progress(None)


def test_is_job_stale():
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    fresh = {
        "state": "RUNNING",
        "updated_at": (now - timedelta(seconds=10)).isoformat(),
        "progress_json": json.dumps({"heartbeat_at": (now - timedelta(seconds=10)).isoformat()}),
    }
    stale = {
        "state": "RUNNING",
        "updated_at": (now - timedelta(seconds=1000)).isoformat(),
        "progress_json": json.dumps({"heartbeat_at": (now - timedelta(seconds=1000)).isoformat()}),
    }
    queued = {"state": "QUEUED", "updated_at": (now - timedelta(seconds=1000)).isoformat()}
    assert is_job_stale(fresh, now=now, stale_seconds=900) is False
    assert is_job_stale(stale, now=now, stale_seconds=900) is True
    assert is_job_stale(queued, now=now, stale_seconds=900) is False


def test_default_device_is_cpu_or_cuda():
    assert default_device() in {"cpu", "cuda"}


def test_delete_keeps_shared_upload(tmp_path: Path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"shared")
    state = build_app_state(tmp_path, worker=False)
    payload = serialize_clean_form(device="cpu", detector="owlvit", inpainter="opencv-telea")
    original = queue_clean_job(state, video, "remove logo", payload)
    retry = state.manage.retry(original)
    upload = tmp_path / "uploads" / original / "input" / "clip.mp4"
    assert upload.is_file()
    assert Path(state.jobs.get(retry)["input_path"]) == upload

    msg = delete_job(state, original)
    assert "Deleted" in msg
    assert state.jobs.get(original) is None
    assert upload.is_file()
    assert Path(state.jobs.get(retry)["input_path"]).is_file()

    delete_job(state, retry)
    assert state.jobs.get(retry) is None
    assert not upload.exists()


def test_delete_unique_upload(tmp_path: Path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"solo")
    state = build_app_state(tmp_path, worker=False)
    payload = serialize_clean_form(device="cpu", detector="owlvit", inpainter="opencv-telea")
    job_id = queue_clean_job(state, video, "remove logo", payload)
    upload_dir = tmp_path / "uploads" / job_id
    assert upload_dir.is_dir()
    delete_job(state, job_id)
    assert not upload_dir.exists()


def test_build_ui_timer_refreshes_models(tmp_path: Path):
    pytest.importorskip("gradio")
    from videoclean.adapters.web.gradio_app import build_ui

    state = build_app_state(tmp_path, worker=False, catalog=FakeCat())
    demo = build_ui(state)
    tick_fns = [
        handler
        for handler in demo.fns.values()
        if handler.fn and any(target[1] == "tick" for target in (handler.targets or []))
    ]
    assert tick_fns, "expected a Timer.tick handler"
    # Lightweight poll: bar/msg/live + conditional tables + timer active flag.
    assert any(len(handler.outputs) == 7 for handler in tick_fns)
    assert all(handler.fn.__name__ == "_on_poll" for handler in tick_fns)


def test_download_progress_reads_running_row(tmp_path: Path):
    from videoclean.adapters.web.gradio_app import _download_progress

    state = build_app_state(tmp_path, worker=False, catalog=FakeCat(), downloader=False)
    state.jobs.upsert_download(
        "d1",
        "detector:grounding-dino",
        "running",
        progress=0.42,
        bytes_done=3,
        bytes_total=9,
        message="files",
    )
    frac, msg = _download_progress(state)
    assert frac == pytest.approx(0.42)
    assert "42%" in msg
    assert "3/9" in msg
    assert "detector:grounding-dino" in msg


def test_download_progress_keeps_last_status_when_idle(tmp_path: Path):
    from videoclean.adapters.web.gradio_app import _download_progress

    state = build_app_state(tmp_path, worker=False, catalog=FakeCat(), downloader=False)
    state.last_download_frac = 1.0
    state.last_download_msg = "Finished detector:owlvit — ready"
    frac, msg = _download_progress(state)
    assert frac == 1.0
    assert "Finished" in msg


def test_download_progress_syncs_finished_from_store(tmp_path: Path):
    from videoclean.adapters.web.gradio_app import _download_progress

    state = build_app_state(tmp_path, worker=False, catalog=FakeCat(), downloader=False)
    state.last_download_frac = 0.78
    state.last_download_msg = "segmenter:sam2-large: 78% Fetching"
    state.jobs.upsert_download("d1", "segmenter:sam2-large", "done", progress=1.0, message="ready")
    frac, msg = _download_progress(state)
    assert frac == 1.0
    assert msg == "Finished segmenter:sam2-large — ready"


def test_cancel_download_clears_bar_immediately(tmp_path: Path):
    from videoclean.adapters.web.gradio_app import _cancel_download, _download_progress

    state = build_app_state(tmp_path, worker=False, catalog=FakeCat(), downloader=False)
    state.jobs.upsert_download(
        "d1",
        "segmenter:sam2-large",
        "running",
        progress=0.78,
        message="Fetching",
    )
    state.last_download_frac = 0.78
    msg = _cancel_download(state)
    assert msg.startswith("Cancelled")
    assert state.last_download_frac == 0.0
    row = state.jobs.get_download("d1")
    assert row is not None
    assert row["state"] == "cancelled"
    assert row["cancel_requested"] == 1
    frac, status = _download_progress(state)
    assert frac == 0.0
    assert "Cancelled" in status


def test_download_click_handler_returns_tuple_without_blocking(tmp_path: Path):
    """Download click must return immediately — a generator holds Gradio's queue and blocks Cancel."""
    pytest.importorskip("gradio")
    import types

    from videoclean.adapters.web.gradio_app import make_download_click_handler
    from videoclean.application.use_cases.download_component import DownloadComponent

    started = threading.Event()
    release = threading.Event()

    def fake_runner(component_id, on_progress=None, is_cancelled=None):
        started.set()
        release.wait(timeout=2)
        if on_progress:
            on_progress(1.0, "ready")

    state = build_app_state(
        tmp_path,
        worker=False,
        catalog=FakeCat(),
        downloader=DownloadComponent(fake_runner),
    )
    handler = make_download_click_handler(state, "detector:owlvit")
    result = handler(None, None, "opencv-telea")
    assert not isinstance(result, types.GeneratorType)
    assert isinstance(result, tuple)
    assert len(result) == 11  # model_outputs only
    assert result[2].startswith("Starting")
    assert started.wait(timeout=1)
    release.set()

def test_doctor_text_is_cached(monkeypatch):
    import videoclean.adapters.web.gradio_app as ga

    ga._doctor_cache = None
    calls = {"n": 0}

    def fake_facts():
        calls["n"] += 1
        return {
            "python": "3",
            "ffmpeg": "f",
            "ffprobe": "p",
            "opencv": "o",
            "torch": "t",
            "cuda": "no",
            "mps": "no",
        }

    monkeypatch.setattr("videoclean.composition.machine_facts", fake_facts)
    assert "python: 3" in ga.format_doctor_text()
    assert "python: 3" in ga.format_doctor_text()
    assert calls["n"] == 1
    assert "python: 3" in ga.format_doctor_text(force=True)
    assert calls["n"] == 2


def test_list_full_jobs_one_query(tmp_path: Path):
    jobs = JobIndex(tmp_path / "j.sqlite")
    jobs.upsert("a", "QUEUED", prompt="one")
    jobs.upsert("b", "RUNNING", prompt="two")
    rows = list_full_jobs(jobs)
    assert {r["id"] for r in rows} == {"a", "b"}
    assert rows[0]["prompt"] in {"one", "two"}
