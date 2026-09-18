"""Terminal job webhook delivery for external integrations."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

_TERMINAL = frozenset({"COMPLETED", "FAILED"})


def public_base_url() -> str:
    return str(os.environ.get("VIDEOCLEAN_PUBLIC_BASE_URL") or "").strip().rstrip("/")


def sign_body(secret: str, body: bytes) -> str:
    digest = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def build_webhook_payload(job: dict[str, Any]) -> dict[str, Any]:
    """Build JSON payload from a job_dict-like mapping."""
    base = public_base_url()
    output_url = job.get("output_url")
    outputs = dict(job.get("outputs") or {})
    payload: dict[str, Any] = {
        "id": job.get("id"),
        "state": job.get("state"),
        "error": job.get("error") or "",
        "kind": job.get("kind") or "run",
        "output_url": output_url,
        "outputs": outputs,
    }
    if base:
        abs_out = f"{base}{output_url}" if output_url else None
        payload["absolute_output_url"] = abs_out
        payload["absolute_outputs"] = {
            fmt: f"{base}{path}" for fmt, path in outputs.items() if path
        }
    else:
        payload["absolute_output_url"] = None
        payload["absolute_outputs"] = None
    return payload


def post_webhook(
    url: str,
    payload: dict[str, Any],
    *,
    secret: str = "",
    attempts: int = 3,
    timeout_s: float = 10.0,
    sleep_fn=time.sleep,
) -> dict[str, Any]:
    """POST JSON with optional HMAC. Retries on failure. Never raises to callers."""
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json", "User-Agent": "videoclean-webhook/1.0"}
    if secret:
        headers["X-VideoClean-Signature"] = sign_body(secret, body)
    last_status: int | None = None
    last_error = ""
    for i in range(max(1, int(attempts))):
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                last_status = int(getattr(resp, "status", 200) or 200)
                if 200 <= last_status < 300:
                    return {
                        "ok": True,
                        "attempts": i + 1,
                        "last_status": last_status,
                        "last_error": "",
                    }
                last_error = f"HTTP {last_status}"
        except urllib.error.HTTPError as exc:
            last_status = int(exc.code)
            last_error = f"HTTP {exc.code}"
        except Exception as exc:  # noqa: BLE001 — delivery must not break the worker
            last_status = None
            last_error = str(exc)[:500]
        if i + 1 < attempts:
            sleep_fn(min(8.0, 2**i))
    return {
        "ok": False,
        "attempts": max(1, int(attempts)),
        "last_status": last_status,
        "last_error": last_error,
    }


def maybe_deliver_job_webhook(state: Any, job_id: str, terminal_state: str) -> None:
    """Fire webhook for kind=run jobs that requested webhook_url. Safe no-op otherwise."""
    if terminal_state not in _TERMINAL:
        return
    row = state.jobs.get(job_id)
    if row is None:
        return
    try:
        request = json.loads(row["request_json"] or "{}")
    except (TypeError, ValueError):
        request = {}
    if not isinstance(request, dict):
        request = {}
    kind = str(request.get("kind") or "run").strip().lower() or "run"
    if kind != "run":
        return
    url = str(request.get("webhook_url") or "").strip()
    if not url:
        return
    secret = str(request.get("webhook_secret") or "").strip()
    from server.service import job_dict

    job = job_dict(state, row)
    payload = build_webhook_payload(job)
    result = post_webhook(url, payload, secret=secret)
    try:
        report = json.loads(row["report_json"] or "{}")
    except (TypeError, ValueError):
        report = {}
    if not isinstance(report, dict):
        report = {}
    report["webhook"] = result
    state.jobs.upsert(job_id, terminal_state, report=report)
