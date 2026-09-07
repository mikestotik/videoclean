from pathlib import Path

import pytest
from typer.testing import CliRunner

from videoclean.application.config import PipelineConfig
from videoclean.application.use_cases.manage_jobs import ManageJobs
from videoclean.cli import app, data_dir
from videoclean.composition import (
    default_serve_port,
    extra_doctor_warnings,
    parse_torch_version,
    resolve_data_dir,
    sam2_video_torch_warning,
)
from videoclean.store import JobIndex

runner = CliRunner()


@pytest.fixture(autouse=True)
def _mock_ollama_tags(monkeypatch):
    monkeypatch.setattr("videoclean.adapters.models.catalog.ollama_tags", lambda: None)


def test_resolve_data_dir_env_and_default(tmp_path: Path):
    assert resolve_data_dir({"VIDEOCLEAN_DATA_DIR": str(tmp_path / "vc")}) == tmp_path / "vc"
    assert resolve_data_dir({"VIDEOCLEAN_DATA_DIR": "  "}) == Path.home() / ".videoclean"
    assert resolve_data_dir({}) == Path.home() / ".videoclean"


def test_data_dir_honors_env(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("VIDEOCLEAN_DATA_DIR", str(tmp_path / "store"))
    assert data_dir() == tmp_path / "store"


def test_default_serve_port():
    assert default_serve_port({}) == 7860
    assert default_serve_port({"VIDEOCLEAN_PORT": "7999"}) == 7999
    assert default_serve_port({"VIDEOCLEAN_PORT": "nope"}) == 7860


def test_parse_torch_version():
    assert parse_torch_version("2.2.2") == (2, 2)
    assert parse_torch_version("2.5.0+cu124") == (2, 5)
    assert parse_torch_version("2.6.0.dev20250101") == (2, 6)
    assert parse_torch_version("not imported") is None
    assert parse_torch_version("") is None


def test_sam2_video_torch_warning():
    msg = sam2_video_torch_warning("sam2-video", "2.2.2")
    assert msg is not None
    assert "2.5" in msg
    assert "2.2.2" in msg
    assert sam2_video_torch_warning("sam2", "2.2.2") is None
    assert sam2_video_torch_warning("sam2-video", "2.5.0+cu124") is None
    assert sam2_video_torch_warning("sam2-video", "2.6.0") is None
    assert sam2_video_torch_warning("sam2-video", "not imported") is None


def test_extra_doctor_warnings_for_sam2_video():
    cfg = PipelineConfig(segmenter="sam2-video")
    notes = extra_doctor_warnings(cfg, {"torch": "2.2.2"})
    assert notes
    assert "torch>=2.5" in notes[0]
    assert extra_doctor_warnings(PipelineConfig(segmenter="sam2"), {"torch": "2.2.2"}) == []


def test_serve_help():
    result = runner.invoke(app, ["serve", "--help"])
    assert result.exit_code == 0
    assert "--host" in result.output
    assert "--port" in result.output


def test_models_help():
    result = runner.invoke(app, ["models", "--help"])
    assert result.exit_code == 0
    assert "list" in result.output
    assert "download" in result.output


def test_jobs_help_has_control_commands():
    result = runner.invoke(app, ["jobs", "--help"])
    assert result.exit_code == 0
    assert "cancel" in result.output
    assert "retry" in result.output
    assert "delete" in result.output


def test_serve_calls_launch_from_env(monkeypatch, tmp_path: Path):
    pytest.importorskip("fastapi")
    seen: dict = {}

    def fake_launch(*, host, port, data_dir, env=None):
        seen["host"] = host
        seen["port"] = port
        seen["data_dir"] = Path(data_dir)
        seen["env"] = env

    monkeypatch.setenv("VIDEOCLEAN_DATA_DIR", str(tmp_path))
    monkeypatch.setattr("videoclean.adapters.web.fastapi_app.launch_from_env", fake_launch)
    result = runner.invoke(app, ["serve", "--host", "127.0.0.1", "--port", "7999"])
    assert result.exit_code == 0, result.output
    assert seen["host"] == "127.0.0.1"
    assert seen["port"] == 7999
    assert seen["data_dir"] == tmp_path


def test_serve_missing_password_exits(monkeypatch, tmp_path: Path):
    pytest.importorskip("fastapi")
    monkeypatch.setenv("VIDEOCLEAN_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("VIDEOCLEAN_UI_PASSWORD", raising=False)
    result = runner.invoke(app, ["serve", "--host", "127.0.0.1", "--port", "7860"])
    assert result.exit_code != 0
    text = result.output + str(result.exception or "")
    assert "VIDEOCLEAN_UI_PASSWORD" in text


def test_serve_busy_port_message(monkeypatch, tmp_path: Path):
    pytest.importorskip("fastapi")

    def boom(*, host, port, data_dir, env=None):
        raise OSError(
            "Cannot find empty port in range: 7860-7860. You can specify a different port"
        )

    monkeypatch.setenv("VIDEOCLEAN_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("VIDEOCLEAN_UI_PASSWORD", "x")
    monkeypatch.setattr("videoclean.adapters.web.fastapi_app.launch_from_env", boom)
    result = runner.invoke(app, ["serve", "--host", "127.0.0.1", "--port", "7860"])
    assert result.exit_code != 0
    text = result.output + str(result.exception or "")
    assert "already in use" in text
    assert "--port 7861" in text


def test_models_list_prints_component_ids(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("VIDEOCLEAN_DATA_DIR", str(tmp_path))
    result = runner.invoke(app, ["models", "list"])
    assert result.exit_code == 0, result.output
    assert "detector:grounding-dino" in result.output
    assert "segmenter:sam2-tiny" in result.output


def test_models_download_invokes_component(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("VIDEOCLEAN_DATA_DIR", str(tmp_path))
    seen: list[str] = []

    def fake_run(component_id, on_progress=None, is_cancelled=None):
        seen.append(component_id)
        if on_progress:
            on_progress(1.0, "ready")

    monkeypatch.setattr("videoclean.adapters.models.downloaders.run_download", fake_run)
    result = runner.invoke(app, ["models", "download", "detector:owlvit"])
    assert result.exit_code == 0, result.output
    assert seen == ["detector:owlvit"]
    assert "detector:owlvit" in result.output
    rows = JobIndex(tmp_path / "jobs.sqlite").list_downloads()
    assert rows
    assert rows[0]["state"] == "done"


def test_models_download_unknown_exits(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("VIDEOCLEAN_DATA_DIR", str(tmp_path))
    result = runner.invoke(app, ["models", "download", "nope"])
    assert result.exit_code != 0
    assert "unknown component" in result.output


def test_jobs_cancel_retry_delete(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("VIDEOCLEAN_DATA_DIR", str(tmp_path))
    jobs = JobIndex(tmp_path / "jobs.sqlite")
    mgr = ManageJobs(jobs)
    jid = mgr.submit({"device": "cpu"}, tmp_path / "in.mp4", tmp_path / "out.mp4", "x")
    result = runner.invoke(app, ["jobs", "cancel", jid])
    assert result.exit_code == 0, result.output
    assert jobs.get(jid)["state"] == "CANCELLED"

    new_result = runner.invoke(app, ["jobs", "retry", jid])
    assert new_result.exit_code == 0, new_result.output
    assert "Queued retry" in new_result.output
    queued = [row for row in jobs.list_jobs() if row["state"] == "QUEUED"]
    assert len(queued) == 1
    new_id = queued[0]["id"]

    deleted = runner.invoke(app, ["jobs", "delete", jid])
    assert deleted.exit_code == 0, deleted.output
    assert jobs.get(jid) is None
    assert jobs.get(new_id) is not None


def test_jobs_cancel_unknown(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("VIDEOCLEAN_DATA_DIR", str(tmp_path))
    JobIndex(tmp_path / "jobs.sqlite")
    result = runner.invoke(app, ["jobs", "cancel", "missing"])
    assert result.exit_code != 0
    assert "unknown job" in result.output
