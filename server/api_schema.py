"""OpenAPI models and docs helpers for the public integration contract."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

JobState = Literal["QUEUED", "RUNNING", "COMPLETED", "FAILED", "CANCELLED"]

PUBLIC_OPS: set[tuple[str, str]] = {
    ("get", "/health"),
    ("get", "/api/options"),
    ("post", "/api/jobs"),
    ("get", "/api/jobs"),
    ("get", "/api/jobs/{job_id}"),
    ("get", "/api/jobs/{job_id}/output"),
    ("post", "/api/jobs/{job_id}/cancel"),
    ("get", "/api/presets"),
    ("post", "/api/presets"),
    ("put", "/api/presets/{preset_id}"),
    ("delete", "/api/presets/{preset_id}"),
}

APP_DESCRIPTION = """
VideoClean removes named or outlined objects (text, logo, thing) from video.

## Public happy path (integration)

1. `POST /api/jobs` with multipart `video` + `prompt` (`kind=run` by default).
   Optional: `formats` (e.g. `mp4,webm`), `profile`, `preset`, `webhook_url`, `webhook_secret`.
2. Poll `GET /api/jobs/{id}` **or** wait for the terminal webhook.
3. Download `GET /api/jobs/{id}/output` (use `?fmt=` when multiple outputs).

Cancel with `POST /api/jobs/{id}/cancel`. Discover profiles/devices/formats via `GET /api/options`.

## Auth

- Internal contour: `VIDEOCLEAN_AUTH=off` disables Basic/Bearer (compose default).
- When auth is on: Bearer `<VIDEOCLEAN_API_TOKEN>` (or UI password) / HTTP Basic.
- `GET /health` is always open; `/openapi.json` and `/api/docs` are open for Swagger/ReDoc.

## Tags

- **Public** — stable integration contract.
- **Internal** — WebUI/admin helpers; no stability promise.
""".strip()


class JobProgress(BaseModel):
    stage: str = ""
    fraction: float | None = None
    detail: str = ""


class Job(BaseModel):
    id: str
    state: JobState
    prompt: str = ""
    created_at: str | None = None
    updated_at: str | None = None
    error: str = ""
    progress: dict[str, Any] | JobProgress | None = None
    stage: str = ""
    fraction: float | None = None
    detail: str = ""
    eta: str | None = None
    stages: dict[str, Any] | list[Any] | None = None
    kind: str = "run"
    source_id: str | None = None
    source_name: str | None = None
    has_output: bool = False
    has_input: bool = False
    can_package: bool = False
    parent_job_id: str | None = None
    output_url: str | None = None
    outputs: dict[str, str] = Field(default_factory=dict)
    input_url: str | None = None
    status_url: str | None = None
    request: dict[str, Any] | None = None
    poll: str | None = None
    download: str | None = None


class JobCreated(Job):
    poll: str
    download: str | None = None


class ErrorBody(BaseModel):
    detail: str


def apply_public_internal_tags(schema: dict[str, Any]) -> dict[str, Any]:
    paths = schema.get("paths") or {}
    for path, methods in paths.items():
        if not isinstance(methods, dict):
            continue
        for method, op in methods.items():
            if method.startswith("x-") or not isinstance(op, dict):
                continue
            key = (method.lower(), path)
            if key in PUBLIC_OPS:
                op["tags"] = ["Public"]
            else:
                op["tags"] = ["Internal"]
                note = "WebUI/admin; not part of the public stability contract."
                desc = (op.get("description") or "").strip()
                if note not in desc:
                    op["description"] = f"{desc}\n\n{note}".strip() if desc else note
            if path == "/health":
                op["security"] = []
    schema["tags"] = [
        {"name": "Public", "description": "Stable integration contract for external clients."},
        {
            "name": "Internal",
            "description": "WebUI/admin endpoints. May change without notice.",
        },
    ]
    return schema
