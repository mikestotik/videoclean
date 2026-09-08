import base64
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from server import fastapi_app as fa
from server.app_state import build_app_state
from server.service import auth_from_env


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VIDEOCLEAN_UI_USER", "admin")
    monkeypatch.setenv("VIDEOCLEAN_UI_PASSWORD", "admin")
    auth_from_env()
    state = build_app_state(tmp_path, worker=False, downloader=False)
    http = TestClient(fa.create_app(state))
    auth = {"Authorization": "Basic " + base64.b64encode(b"admin:admin").decode()}
    http.headers.update(auth)
    return http, state, tmp_path


def _make_video(path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc=duration=1:size=160x120:rate=10",
            "-pix_fmt", "yuv420p", str(path),
        ],
        check=True, capture_output=True,
    )


def test_probe_returns_manifest(client, tmp_path: Path):
    http, state, _ = client
    video = tmp_path / "in.mp4"
    _make_video(video)
    state.jobs.upsert("j-probe", "COMPLETED", input_path=str(video))
    res = http.get("/api/jobs/j-probe/probe")
    assert res.status_code == 200
    body = res.json()
    assert body["width"] == 160
    assert body["height"] == 120
    assert abs(body["fps"] - 10.0) < 0.01
    assert body["frame_count"] == 10


def test_probe_unknown_job(client):
    http, _, _ = client
    assert http.get("/api/jobs/nope/probe").status_code == 404
