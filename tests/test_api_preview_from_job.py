from pathlib import Path

import base64
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


def test_from_job_requires_existing_input(client, tmp_path: Path):
    http, state, _ = client
    state.jobs.upsert("j-src", "COMPLETED", input_path=str(tmp_path / "missing.mp4"))
    res = http.post("/api/preview/from-job", json={"job_id": "j-src", "prompt": "text"})
    assert res.status_code == 400


def test_from_job_queues_preview(client, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    http, state, tmp = client
    video = tmp / "in.mp4"
    video.write_bytes(b"stub")
    state.jobs.upsert("j-src3", "COMPLETED", input_path=str(video))
    captured: dict = {}

    def fake_queue(state_, src, prompt, request, original_name=""):
        captured.update(request)
        captured["src"] = src
        return "j-preview"

    monkeypatch.setattr("server.service.queue_preview_job", fake_queue)
    res = http.post(
        "/api/preview/from-job",
        json={"job_id": "j-src3", "prompt": "logo", "indices": [3, 7]},
    )
    assert res.status_code == 201
    assert res.json()["id"] == "j-preview"
    assert captured["indices"] == [3, 7]
    assert captured["kind"] == "preview"
    assert captured["src"].name == "in.mp4"
