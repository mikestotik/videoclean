"""Preview API endpoints (auth comes from the shared BasicOrBearerAuth middleware)."""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from server import fastapi_app as fa
from server.app_state import AppState
from server.service import auth_from_env
from videoclean.store import JobIndex


class FakePreview:
    last_req = None

    def execute(self, req, data_dir):
        FakePreview.last_req = req
        # emulate artifact
        (Path(data_dir) / "jobs" / req.job_id / "preview").mkdir(parents=True, exist_ok=True)
        return {"state": "COMPLETED", "jobId": req.job_id, "frames": []}


@pytest.fixture
def client(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("VIDEOCLEAN_UI_USER", "admin")
    monkeypatch.setenv("VIDEOCLEAN_UI_PASSWORD", "pw")
    jobs = JobIndex(tmp_path / "jobs.sqlite")
    from server.app_state import build_app_state

    state = build_app_state(tmp_path, worker=False, downloader=False)
    app = fa.create_app(state)
    token = auth_from_env()
    import base64

    auth = {"Authorization": "Basic " + base64.b64encode(b"admin:pw").decode()}
    client = TestClient(app)
    client.headers.update(auth)
    yield client, jobs, tmp_path


VIDEO_BYTES = (
    b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom"
)


def test_preview_job_submission_and_artifact_route(client):
    client, jobs, tmp_path = client
    resp = client.post(
        "/api/preview",
        files={"video": ("in.mp4", VIDEO_BYTES, "video/mp4")},
        data={"prompt": "удали текст", "start": "0", "count": "8", "stride": "2", "mode": "parse"},
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    job_id = body["id"]
    row = jobs.get(job_id)
    assert row is not None and row["state"] == "QUEUED"
    import json

    payload = json.loads(row["request_json"])
    assert payload["kind"] == "preview"
    assert payload["start"] == 0 and payload["count"] == 8 and payload["stride"] == 2


def test_preview_detect_mode_requires_targets(client):
    client, jobs, tmp_path = client
    resp = client.post(
        "/api/preview",
        files={"video": ("in.mp4", VIDEO_BYTES, "video/mp4")},
        data={"prompt": "x", "mode": "detect", "targets": ""},
    )
    assert resp.status_code == 400
    assert "targets" in resp.json()["detail"].lower()


def test_preview_detect_mode_with_targets(client):
    client, jobs, tmp_path = client
    import json as _json

    targets = _json.dumps([{"kind": "text_overlay", "query": "text", "where": None, "motion": "any"}])
    resp = client.post(
        "/api/preview",
        files={"video": ("in.mp4", VIDEO_BYTES, "video/mp4")},
        data={"prompt": "x", "mode": "detect", "targets": targets},
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    payload = _json.loads(jobs.get(body["id"])["request_json"])
    assert payload["mode"] == "detect"
    assert payload["targets"] == [{"kind": "text_overlay", "query": "text", "where": None, "motion": "any"}]


def test_preview_artifact_route_serves_files_and_blocks_traversal(client):
    client, jobs, tmp_path = client
    job_id = "p-art"
    art_dir = tmp_path / "jobs" / job_id / "preview"
    art_dir.mkdir(parents=True)
    (art_dir / "000000_boxes.jpg").write_bytes(b"jpegdata")
    (art_dir / "preview.json").write_text("{}")
    jobs.upsert(job_id, "COMPLETED", input_path="x", prompt="p")

    r = client.get(f"/api/jobs/{job_id}/preview/000000_boxes.jpg")
    assert r.status_code == 200
    assert r.content == b"jpegdata"

    r = client.get(f"/api/jobs/{job_id}/preview/preview.json")
    assert r.status_code == 200

    r = client.get(f"/api/jobs/{job_id}/preview/..%2Finput%2Finput_manifest.json")
    assert r.status_code in (400, 404)

    r = client.get("/api/jobs/unknown/preview/000000_boxes.jpg")
    assert r.status_code == 404
