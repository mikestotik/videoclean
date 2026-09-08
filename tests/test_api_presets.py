"""Presets CRUD backed by data_dir/presets.json."""
import base64
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from server import fastapi_app as fa
from server.service import auth_from_env


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


def test_preset_crud(client):
    client, state, tmp_path = client
    created = client.post("/api/presets", json={"name": "Логотип", "payload": {"inpainter": "lama", "mask_dilate_px": 5}})
    assert created.status_code == 201, created.text
    item = created.json()
    assert item["name"] == "Логотип" and item["payload"]["inpainter"] == "lama"
    lst = client.get("/api/presets").json()
    assert [p["id"] for p in lst] == [item["id"]]
    assert (tmp_path / "presets.json").is_file()
    assert client.delete(f"/api/presets/{item['id']}").status_code == 200
    assert client.get("/api/presets").json() == []
    assert client.delete(f"/api/presets/{item['id']}").status_code == 404


def test_preset_validation(client):
    client, state, tmp_path = client
    assert client.post("/api/presets", json={"name": " ", "payload": {}}).status_code == 400
    assert client.post("/api/presets", json={"name": "x", "payload": [1, 2]}).status_code == 400
    assert (tmp_path / "presets.json").exists() is False
