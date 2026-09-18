"""Webhook signing and delivery."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from server.webhook import build_webhook_payload, maybe_deliver_job_webhook, post_webhook, sign_body


def test_sign_body_stable():
    sig = sign_body("secret", b'{"id":"1"}')
    assert sig.startswith("sha256=")
    assert sig == sign_body("secret", b'{"id":"1"}')
    assert sig != sign_body("other", b'{"id":"1"}')


def test_build_webhook_payload_absolute(monkeypatch):
    monkeypatch.setenv("VIDEOCLEAN_PUBLIC_BASE_URL", "https://example.test/")
    payload = build_webhook_payload(
        {
            "id": "j1",
            "state": "COMPLETED",
            "error": "",
            "kind": "run",
            "output_url": "/api/jobs/j1/output?fmt=mp4",
            "outputs": {"mp4": "/api/jobs/j1/output?fmt=mp4"},
        }
    )
    assert payload["absolute_output_url"] == "https://example.test/api/jobs/j1/output?fmt=mp4"
    assert payload["absolute_outputs"]["mp4"].startswith("https://example.test/")


def test_post_webhook_ok_and_signature(monkeypatch):
    seen: dict = {}

    class Resp:
        status = 204

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(req, timeout=10.0):
        seen["url"] = req.full_url
        seen["body"] = req.data
        seen["headers"] = dict(req.headers)
        return Resp()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    result = post_webhook(
        "http://hook.test/cb",
        {"id": "j1", "state": "COMPLETED"},
        secret="s3",
        attempts=1,
        sleep_fn=lambda _s: None,
    )
    assert result["ok"] is True
    # urllib.Request stores header names in a case-insensitive dict / may rewrite casing.
    sig = next(
        (v for k, v in seen["headers"].items() if k.lower() == "x-videoclean-signature"),
        None,
    )
    assert sig == sign_body("s3", seen["body"])


def test_post_webhook_failure_does_not_raise(monkeypatch):
    def boom(*args, **kwargs):
        raise TimeoutError("slow")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    result = post_webhook(
        "http://hook.test/cb",
        {"id": "j1", "state": "FAILED", "error": "x"},
        attempts=2,
        sleep_fn=lambda _s: None,
    )
    assert result["ok"] is False
    assert result["attempts"] == 2


def test_maybe_deliver_skips_without_url(tmp_path: Path, monkeypatch):
    from server.app_state import build_app_state

    state = build_app_state(tmp_path, worker=False, downloader=False)
    jid = state.manage.submit(
        {"kind": "run", "prompt": "p"},
        tmp_path / "in.mp4",
        tmp_path / "out.mp4",
        "p",
    )
    (tmp_path / "in.mp4").write_bytes(b"x")
    called = []

    monkeypatch.setattr(
        "server.webhook.post_webhook",
        lambda *a, **k: called.append(1) or {"ok": True, "attempts": 1, "last_status": 200, "last_error": ""},
    )
    maybe_deliver_job_webhook(state, jid, "COMPLETED")
    assert called == []


def test_maybe_deliver_posts_and_records(tmp_path: Path, monkeypatch):
    from server.app_state import build_app_state

    state = build_app_state(tmp_path, worker=False, downloader=False)
    (tmp_path / "in.mp4").write_bytes(b"x")
    jid = state.manage.submit(
        {
            "kind": "run",
            "prompt": "p",
            "webhook_url": "http://hook.test/cb",
            "webhook_secret": "sec",
        },
        tmp_path / "in.mp4",
        tmp_path / "out.mp4",
        "p",
    )
    state.jobs.upsert(jid, "COMPLETED", report={"outputs": {"mp4": str(tmp_path / "out.mp4")}})

    monkeypatch.setattr(
        "server.webhook.post_webhook",
        lambda *a, **k: {"ok": True, "attempts": 1, "last_status": 200, "last_error": ""},
    )
    maybe_deliver_job_webhook(state, jid, "COMPLETED")
    row = state.jobs.get(jid)
    report = json.loads(row["report_json"])
    assert report["webhook"]["ok"] is True
