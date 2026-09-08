"""POST /api/jobs: kinds run/preview/prompt, source_id, manual overrides."""
import base64
import json
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from server import fastapi_app as fa
from server.service import auth_from_env


def _make_clip(path: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=duration=1:size=64x64:rate=5",
         "-pix_fmt", "yuv420p", str(path)],
        check=True, capture_output=True, timeout=60,
    )


@pytest.fixture
def client(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("VIDEOCLEAN_UI_USER", "admin")
    monkeypatch.setenv("VIDEOCLEAN_UI_PASSWORD", "pw")
    from server.app_state import build_app_state

    state = build_app_state(tmp_path, worker=False, downloader=False)
    app = fa.create_app(state)
    client = TestClient(app)
    client.headers.update({"Authorization": "Basic " + base64.b64encode(b"admin:pw").decode()})
    yield client, state, tmp_path


def _source(client, tmp_path: Path) -> dict:
    clip = tmp_path / "clip.mp4"
    _make_clip(clip)
    with clip.open("rb") as f:
        resp = client.post("/api/sources", files={"video": ("clip.mp4", f, "video/mp4")})
    assert resp.status_code == 201
    return resp.json()


def _payload(row) -> dict:
    return json.loads(row["request_json"])


def test_run_from_source_with_tracks_override(client):
    client, state, tmp_path = client
    src = _source(client, tmp_path)
    resp = client.post(
        "/api/jobs",
        data={
            "kind": "run", "source_id": src["id"],
            "tracks": json.dumps([{"id": 0, "label": "logo", "motion": "static", "boxes": [[1, 1, 4, 4]]}]),
            "prompt": "",
        },
    )
    assert resp.status_code == 201, resp.text
    row = state.jobs.get(resp.json()["id"])
    payload = _payload(row)
    assert payload["kind"] == "run"
    assert payload["tracks_override"] == [{"id": 0, "label": "logo", "motion": "static", "boxes": [[1, 1, 4, 4]]}]
    assert row["source_id"] == src["id"]
    assert Path(payload["input_path"]).parent.name == src["id"], "input lives under the source dir"


def test_overrides_are_mutually_exclusive(client):
    client, state, tmp_path = client
    src = _source(client, tmp_path)
    resp = client.post(
        "/api/jobs",
        data={
            "kind": "run", "source_id": src["id"],
            "targets": json.dumps([{"kind": "object", "query": "mug"}]),
            "tracks": json.dumps([{"id": 0, "label": "x", "boxes": [[1, 1, 4, 4]]}]),
            "prompt": "p",
        },
    )
    assert resp.status_code == 400
    assert "mutually exclusive" in resp.json()["detail"]


def test_preview_from_source_all_frames(client):
    client, state, tmp_path = client
    src = _source(client, tmp_path)
    resp = client.post(
        "/api/jobs",
        data={"kind": "preview", "source_id": src["id"], "prompt": "убери", "all": "1"},
    )
    assert resp.status_code == 201, resp.text
    payload = _payload(state.jobs.get(resp.json()["id"]))
    assert payload["kind"] == "preview"
    assert payload["start"] == 0 and payload["count"] == 5, "all-frames expanded"
    assert payload["segmenter"] == "sam2", "detect stage forces the per-frame segmenter"


def test_preview_detect_from_source_with_targets(client):
    client, state, tmp_path = client
    src = _source(client, tmp_path)
    resp = client.post(
        "/api/jobs",
        data={
            "kind": "preview", "source_id": src["id"], "mode": "detect",
            "targets": json.dumps([{"kind": "object", "query": "mug"}]),
        },
    )
    assert resp.status_code == 201, resp.text
    payload = _payload(state.jobs.get(resp.json()["id"]))
    assert payload["kind"] == "preview"
    assert payload["mode"] == "detect"
    assert payload["targets"] == [{"kind": "object", "query": "mug"}]
    assert payload["segmenter"] == "sam2"


def test_prompt_job_collects_source_masks(client):
    client, state, tmp_path = client
    src = _source(client, tmp_path)
    import cv2
    import numpy as np

    ok, buf = cv2.imencode(".png", np.zeros((64, 64), np.uint8))
    assert ok
    masks_dir = state.data_dir / "sources" / src["id"] / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    (masks_dir / "000002.png").write_bytes(buf.tobytes())
    resp = client.post("/api/jobs", data={"kind": "prompt", "source_id": src["id"], "prompt": ""})
    assert resp.status_code == 201, resp.text
    payload = _payload(state.jobs.get(resp.json()["id"]))
    assert payload["kind"] == "prompt"
    assert payload["annotations"] == [{
        "frame": 2, "mask": str(masks_dir / "000002.png"),
    }]


def test_run_upload_registers_source(client):
    client, state, tmp_path = client
    clip = tmp_path / "up.mp4"
    _make_clip(clip)
    with clip.open("rb") as f:
        resp = client.post("/api/jobs", files={"video": ("up.mp4", f, "video/mp4")}, data={"prompt": "убери логотип"})
    assert resp.status_code == 201, resp.text
    job = resp.json()
    row = state.jobs.get(job["id"])
    assert row["source_id"], "upload created an implicit source"
    assert state.sources.get(row["source_id"]) is not None


def test_job_dict_carries_source_fields(client):
    client, state, tmp_path = client
    src = _source(client, tmp_path)
    resp = client.post("/api/jobs", data={"kind": "run", "source_id": src["id"], "prompt": "p"})
    assert resp.status_code == 201
    job = resp.json()
    assert job["source_id"] == src["id"]
    assert job["source_name"] == "clip.mp4"
