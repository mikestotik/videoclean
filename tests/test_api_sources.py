"""Sources API: upload, list, delete, video stream, single-frame extraction."""
import base64
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
    token = auth_from_env()
    client = TestClient(app)
    client.headers.update({"Authorization": "Basic " + base64.b64encode(b"admin:pw").decode()})
    yield client, state, tmp_path


def _upload(client, tmp_path: Path, name="clip.mp4"):
    clip = tmp_path / name
    _make_clip(clip)
    with clip.open("rb") as f:
        resp = client.post("/api/sources", files={"video": (name, f, "video/mp4")})
    assert resp.status_code == 201, resp.text
    return resp.json(), clip


def test_upload_list_get_delete(client, tmp_path: Path):
    client, state, _ = client
    src, _ = _upload(client, tmp_path)
    assert src["name"] == "clip.mp4"
    assert src["probe"]["frame_count"] == 5
    assert src["probe"]["width"] == 64
    lst = client.get("/api/sources").json()
    assert [s["id"] for s in lst] == [src["id"]]
    got = client.get(f"/api/sources/{src['id']}").json()
    assert got["id"] == src["id"]
    assert client.get("/api/sources/s_missing").status_code == 404
    assert client.delete(f"/api/sources/{src['id']}").status_code == 200
    assert client.get(f"/api/sources/{src['id']}").status_code == 404
    assert not (state.data_dir / "sources" / src["id"]).exists(), "source dir removed with the row"


def test_rejects_non_video(client, tmp_path: Path):
    client, state, _ = client
    bad = tmp_path / "x.txt"
    bad.write_text("nope")
    with bad.open("rb") as f:
        resp = client.post("/api/sources", files={"video": ("x.txt", f, "text/plain")})
    assert resp.status_code == 400


def test_video_stream_and_frame(client, tmp_path: Path):
    client, state, _ = client
    src, _ = _upload(client, tmp_path)
    resp = client.get(f"/api/sources/{src['id']}/video")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("video/")
    for path in (f"/api/sources/{src['id']}/frames/2", f"/api/sources/{src['id']}/frames/2.jpg"):
        r = client.get(path)
        assert r.status_code == 200, path
        assert r.headers["content-type"] == "image/jpeg"
    assert client.get(f"/api/sources/{src['id']}/frames/99").status_code == 404
