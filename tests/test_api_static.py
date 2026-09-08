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
    return http


def test_placeholder_when_dist_missing(client, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(fa, "DIST_DIR", Path("/nonexistent-dist"))
    res = client.get("/")
    assert res.status_code == 200
    assert "bun run build" in res.text


def test_spa_served_when_dist_exists(client, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<html><body>SPA OK</body></html>", encoding="utf-8")
    monkeypatch.setattr(fa, "DIST_DIR", dist)
    assert client.get("/").status_code == 200
    assert "SPA OK" in client.get("/config").text
