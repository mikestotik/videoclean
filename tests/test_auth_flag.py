"""VIDEOCLEAN_AUTH=off disables Basic/Bearer middleware."""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from server import fastapi_app as fa
from server.service import auth_enabled, auth_from_env


def test_auth_enabled_parses_off_values(monkeypatch):
    for raw in ("off", "0", "false", "NO", "Off"):
        monkeypatch.setenv("VIDEOCLEAN_AUTH", raw)
        assert auth_enabled() is False
    monkeypatch.setenv("VIDEOCLEAN_AUTH", "on")
    assert auth_enabled() is True
    monkeypatch.delenv("VIDEOCLEAN_AUTH", raising=False)
    assert auth_enabled() is True


def test_auth_off_allows_start_without_password(monkeypatch):
    monkeypatch.setenv("VIDEOCLEAN_AUTH", "off")
    monkeypatch.delenv("VIDEOCLEAN_UI_PASSWORD", raising=False)
    user, password = auth_from_env()
    assert user == "admin"
    assert password == "off"


def test_api_open_when_auth_off(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("VIDEOCLEAN_AUTH", "off")
    monkeypatch.delenv("VIDEOCLEAN_UI_PASSWORD", raising=False)
    from server.app_state import build_app_state

    app = fa.create_app(build_app_state(tmp_path, worker=False, downloader=False))
    client = TestClient(app)
    assert client.get("/api/options").status_code == 200
    assert client.get("/health").status_code == 200


def test_api_requires_auth_when_on(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("VIDEOCLEAN_AUTH", "on")
    monkeypatch.setenv("VIDEOCLEAN_UI_USER", "admin")
    monkeypatch.setenv("VIDEOCLEAN_UI_PASSWORD", "secret")
    from server.app_state import build_app_state

    app = fa.create_app(build_app_state(tmp_path, worker=False, downloader=False))
    bare = TestClient(app)
    assert bare.get("/api/options").status_code == 401
