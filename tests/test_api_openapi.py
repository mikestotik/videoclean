"""OpenAPI public contract: Bearer scheme and Public-tagged routes."""
import base64
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from server import fastapi_app as fa
from server.api_schema import PUBLIC_OPS


@pytest.fixture
def client(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("VIDEOCLEAN_UI_USER", "admin")
    monkeypatch.setenv("VIDEOCLEAN_UI_PASSWORD", "pw")
    monkeypatch.setenv("VIDEOCLEAN_API_TOKEN", "tok-test")
    from server.app_state import build_app_state

    state = build_app_state(tmp_path, worker=False, downloader=False)
    app = fa.create_app(state)
    client = TestClient(app)
    client.headers.update({"Authorization": "Basic " + base64.b64encode(b"admin:pw").decode()})
    yield client


def test_openapi_has_bearer_and_public_ops(client):
    resp = client.get("/openapi.json")
    assert resp.status_code == 200, resp.text
    schema = resp.json()
    schemes = schema["components"]["securitySchemes"]
    assert "BearerAuth" in schemes
    assert schemes["BearerAuth"]["scheme"] == "bearer"
    assert schema.get("security") == [{"BearerAuth": []}]

    paths = schema["paths"]
    for method, path in PUBLIC_OPS:
        assert path in paths, path
        op = paths[path][method]
        assert op.get("tags") == ["Public"], (method, path, op.get("tags"))

    # Sample Internal route still present but tagged.
    assert paths["/api/sources"]["get"]["tags"] == ["Internal"]
    assert paths["/health"]["get"].get("security") == []


def test_api_index_points_to_docs(client):
    resp = client.get("/api")
    assert resp.status_code == 200
    body = resp.json()
    assert body["docs"] == "/api/docs"
    assert "POST /api/jobs" in body["public"]


def test_bearer_auth_accepted(client, monkeypatch, tmp_path: Path):
    monkeypatch.setenv("VIDEOCLEAN_API_TOKEN", "tok-test")
    from server.app_state import build_app_state

    state = build_app_state(tmp_path / "b", worker=False, downloader=False)
    app = fa.create_app(state)
    bare = TestClient(app)
    ok = bare.get("/api/options", headers={"Authorization": "Bearer tok-test"})
    assert ok.status_code == 200
    bad = bare.get("/api/options", headers={"Authorization": "Bearer wrong"})
    assert bad.status_code == 401


def test_docs_and_openapi_are_public(monkeypatch, tmp_path: Path):
    """Swagger/ReDoc fetch /openapi.json via JS without Basic credentials."""
    monkeypatch.setenv("VIDEOCLEAN_UI_USER", "admin")
    monkeypatch.setenv("VIDEOCLEAN_UI_PASSWORD", "pw")
    from server.app_state import build_app_state

    state = build_app_state(tmp_path / "docs", worker=False, downloader=False)
    bare = TestClient(fa.create_app(state))
    spec = bare.get("/openapi.json")
    assert spec.status_code == 200, spec.text
    body = spec.json()
    assert isinstance(body, dict)
    assert str(body.get("openapi", "")).startswith("3.")
    assert bare.get("/api/docs").status_code == 200
    assert bare.get("/api/redoc").status_code == 200
    # API itself stays protected.
    assert bare.get("/api/options").status_code == 401
