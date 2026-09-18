# Public API + OpenAPI Implementation Plan

> **For agentic workers:** Inline execution in this session (user asked to implement).

**Goal:** Stable Public OpenAPI contract with poll, same-job auto-package, and terminal webhooks.

**Architecture:** Honor `formats` inside `RunCleanup`; deliver webhooks from server via `JobWorker.on_terminal`; document Public/Internal tags and Bearer in FastAPI OpenAPI.

**Tech Stack:** FastAPI, Pydantic, urllib for webhook POST, pytest TestClient.

**Spec:** `docs/superpowers/specs/2026-09-18-public-api-openapi-design.md`

## Tasks

1. Auto-package: `PipelineConfig.webm_crf` / `segment_seconds`; `run_cleanup` packages all `cfg.formats`.
2. `server/webhook.py` + `JobWorker.on_terminal` wired from `create_app`.
3. OpenAPI: schemas, tags, security, Public form fields (`webhook_*`).
4. Refresh `GET /api` + `docs/RUNPOD.md`.
5. Thin tests: openapi Bearer/Public; webhook HMAC; formats in cleanup report.
