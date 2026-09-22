import base64
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from server import fastapi_app as fa
from server.app_state import build_app_state
from server.service import auth_from_env
from videoclean.adapters.llm import ollama_setup


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


def test_ollama_payload_reports_installed_flag(client, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(fa, "ollama_installed", lambda: True)
    assert client.get("/api/ollama").json()["installed"] is True
    monkeypatch.setattr(fa, "ollama_installed", lambda: False)
    assert client.get("/api/ollama").json()["installed"] is False


def test_ollama_install_starts_tracked_download(client, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(ollama_setup, "ollama_installed", lambda: False)
    seen = {}

    def fake_install(*, on_progress=None, is_cancelled=None):
        seen["ran"] = True
        if on_progress is not None:
            on_progress(0.5, "downloading ollama", bytes_done=1, bytes_total=2)
        return Path("/tmp/ollama")

    monkeypatch.setattr(ollama_setup, "install_ollama_binary", fake_install)
    res = client.post("/api/ollama/install")
    assert res.status_code == 200
    assert res.json()["ok"] is True
    for _ in range(100):
        rows = client.get("/api/poll").json()["downloads"]
        match = [r for r in rows if r["component_id"] == "system:ollama"]
        if match and match[0]["state"] == "done":
            break
        time.sleep(0.05)
    assert seen.get("ran") is True
    assert match[0]["state"] == "done"
    assert match[0]["progress"] == 1.0


def test_ollama_install_rejected_when_already_downloading(client, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(ollama_setup, "ollama_installed", lambda: False)
    client.app.state.vc.jobs.upsert_download("busy-1", "llm:ollama-llama3.2", "running")
    res = client.post("/api/ollama/install")
    assert res.status_code == 400


def test_ollama_start_reports_status(client, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(ollama_setup, "_ollama_serving", lambda timeout=2.0: False)
    fake_bin = client.app.state.vc.data_dir / "ollama"
    fake_bin.touch()
    monkeypatch.setattr(ollama_setup, "ollama_binary_path", lambda: fake_bin)
    started = {}

    class FakeProc:
        pass

    def fake_popen(*args, **kwargs):
        started["ran"] = True
        return FakeProc()

    monkeypatch.setattr(ollama_setup.subprocess, "Popen", fake_popen)
    res = client.post("/api/ollama/start")
    assert res.status_code == 200
    assert res.json()["status"] == "started"
    assert started.get("ran") is True
