# Public API + OpenAPI (integration contract)

Date: 2026-09-18  
Status: approved for planning (chat); awaiting file review

## Goal

Make VideoClean usable as an embeddable cleanup service for external pipelines (Telegram bot → VideoClean → next service) and for manual WebUI use, without a second URL prefix.

Deliver:

1. Stable **Public** HTTP contract on existing `/api/*` paths.
2. Interactive OpenAPI at `/api/docs` (Swagger UI) with Bearer auth.
3. One-shot cleanup with **poll**, **webhook**, and **auto-package** of requested formats.
4. Remaining endpoints tagged **Internal** (WebUI/admin; no stability promise).

## Non-goals

- `/api/v1` or a parallel public gateway.
- Separate SDK / client library.
- Changing WebUI stage semantics (prompt / preview / track edit).
- Webhook on every progress tick.
- Pushing binary video into the webhook body.

## Context (as of today)

- FastAPI already exposes `docs_url="/api/docs"`; most routes lack `response_model`, tags, field descriptions, and an OpenAPI security scheme.
- Auth is middleware `BasicOrBearerAuth`: HTTP Basic (UI user/password) or `Authorization: Bearer` (`VIDEOCLEAN_API_TOKEN`, else UI password). `/health` is open.
- One-shot path exists: `POST /api/jobs` (`kind=run`) → poll `GET /api/jobs/{id}` → `GET /api/jobs/{id}/output`.
- Cleanup always writes mezzanine + baseline `mp4`. Extra containers go through on-demand `POST /api/jobs/{id}/package` (`kind=package`). Form field `formats` is accepted into the job request but ignored for delivery during `run`.
- Index: `GET /api`. Short notes: `docs/RUNPOD.md`, `docs/PARAMS.md`.

## Public surface

Tag **`Public`**. These are the stable integration endpoints:

| Method | Path | Role |
|---|---|---|
| GET | `/health` | Liveness (no auth) |
| GET | `/api/options` | Profiles, devices, format names, catalog defaults |
| POST | `/api/jobs` | Start cleanup (`kind=run` default); multipart |
| GET | `/api/jobs` | List (optional `state` filter) |
| GET | `/api/jobs/{id}` | Status / progress / output URLs |
| GET | `/api/jobs/{id}/output` | Download (`?fmt=` when multiple) |
| POST | `/api/jobs/{id}/cancel` | Cancel queued/running job |

Everything else stays available for WebUI but is tagged **`Internal`** with description: *WebUI/admin; not part of the public stability contract.*

Happy path documented in OpenAPI description and operation examples:

```
POST /api/jobs  (video + prompt [+ formats] [+ webhook_*] [+ profile])
→ poll GET /api/jobs/{id}  and/or wait for webhook
→ GET /api/jobs/{id}/output?fmt=…
```

## Request: `POST /api/jobs` (Public fields)

Multipart form. Existing pipeline knobs remain accepted. Public docs highlight:

| Field | Required | Notes |
|---|---|---|
| `video` | one of video / `source_id` | Upload for external clients |
| `source_id` | one of video / `source_id` | Library id (WebUI / Internal-friendly) |
| `prompt` | yes for `kind=run` | What to remove |
| `kind` | no | Default `run`. Public contract = `run`. `preview` / `prompt` stay Internal-oriented |
| `formats` | no | Comma list, default `mp4`. Job reaches `COMPLETED` only when these delivery artifacts exist |
| `profile` | no | `fast` / `balanced` / `quality` / `custom` |
| `device` | no | From options / doctor |
| `webhook_url` | no | HTTPS/HTTP callback on terminal state |
| `webhook_secret` | no | If set, HMAC-SHA256 over raw body |
| `webm_crf` | no | Used when packaging `webm` |
| `segment_seconds` | no | Used when packaging HLS/DASH |

Internal overrides (`targets`, `tracks`, `masks`, model ids, …) remain on the same endpoint; documented under Internal / advanced, not as the primary Public example.

Mutual exclusion of `targets` / `tracks` / `masks` stays as today.

## Auto-package

For `kind=run`:

1. Run cleanup as today (mezzanine + baseline `mp4`).
2. If `formats` is only `mp4` (or empty → default mp4), mark job `COMPLETED` with `outputs.mp4`.
3. If `formats` requests more (or only non-mp4), package the missing formats **inside the same job** before terminal `COMPLETED`. Progress stage can show `package`. Integrators keep one `job_id`.
4. `POST /api/jobs/{id}/package` remains for WebUI re-convert; tagged Internal (or Public-adjacent only if we later promote it). First cut: Internal.

Failure during packaging → job `FAILED` with stage-named error (same style as pipeline errors). Baseline mp4 may exist on disk; `has_output` / `outputs` reflect what was successfully registered.

Known format names come from `videoclean.domain.formats` / `GET /api/options` → `formats`.

## Webhook

Triggers once per job on terminal state: `COMPLETED` or `FAILED`. Only for jobs that set `webhook_url` and `kind=run` (including auto-package completion).

### Payload

JSON body (relative paths match `job_dict`; absolute URLs when base is configured):

```json
{
  "id": "…",
  "state": "COMPLETED",
  "error": "",
  "kind": "run",
  "output_url": "/api/jobs/{id}/output?fmt=mp4",
  "outputs": {
    "mp4": "/api/jobs/{id}/output?fmt=mp4",
    "webm": "/api/jobs/{id}/output?fmt=webm"
  },
  "absolute_output_url": "https://host/api/jobs/{id}/output?fmt=mp4",
  "absolute_outputs": {
    "mp4": "https://host/api/jobs/{id}/output?fmt=mp4",
    "webm": "https://host/api/jobs/{id}/output?fmt=webm"
  }
}
```

`absolute_*` filled when `VIDEOCLEAN_PUBLIC_BASE_URL` is set (no trailing slash). If unset, omit absolute fields or set them `null`; relative paths always present.

On `FAILED`, `output_url` / `outputs` may be empty; `error` is set.

### Security and delivery

- Method: `POST`, `Content-Type: application/json`.
- If `webhook_secret` set: header `X-VideoClean-Signature: sha256=<hex>` where hex is HMAC-SHA256 of the raw request body with the secret.
- Retries: small fixed policy (e.g. 3 attempts, exponential backoff starting ~1s). Webhook failure does **not** flip job state; record `webhook` summary in job report (`ok` / `attempts` / `last_status` / `last_error`).
- Timeouts: short connect/read timeout (e.g. 10s) so worker threads are not blocked long.
- Implementation home: server layer after job terminal upsert (worker completion hook or manage callback), not inside domain use cases. `videoclean` library stays free of HTTP callbacks.

### Out of scope for webhook v1

- Custom headers map.
- Choosing which events fire beyond terminal.
- Delivering file bytes in the callback.
- Signed download URLs with expiry (Bearer on GET remains).

## Job response shape (Public)

Align OpenAPI `Job` model with current `job_dict`, documenting stable fields:

- `id`, `state`, `prompt`, `created_at`, `updated_at`, `error`
- `progress`, `stage`, `fraction`, `detail`, `eta`, `stages`
- `kind`, `source_id`, `has_output`, `has_input`
- `output_url`, `outputs`, `input_url`, `status_url`
- `poll` / `download` on create `201` (already returned)

Job `state` enum used by the worker today: `QUEUED`, `RUNNING`, `COMPLETED`, `FAILED`, `CANCELLED`. Public clients must handle all five.

## OpenAPI / Swagger

- Keep `/api/docs`. Enable `/api/redoc` as a second view of the same OpenAPI.
- App metadata: title `VideoClean`, version from package metadata (fallback `0.1.0`), description with Public happy path and auth.
- Security scheme: HTTP Bearer. Document that Basic also works for browsers; Swagger Authorize uses Bearer.
- Middleware auth unchanged; OpenAPI scheme is for documentation and Try it out.
- Tag all routes `Public` or `Internal`.
- Public operations: summaries, parameter descriptions, request examples (multipart), response models, error responses (`401`, `400`, `404`).
- Internal operations: short summary + note that the contract is not stable; `include_in_schema=True` (visible).
- Refresh `GET /api` index: point to `/api/docs`, list Public endpoints, mention webhook/formats.

## Auth and env

| Variable | Role |
|---|---|
| `VIDEOCLEAN_UI_USER` / `VIDEOCLEAN_UI_PASSWORD` | Basic auth; Bearer fallback if token unset |
| `VIDEOCLEAN_API_TOKEN` | Preferred Bearer token for API clients |
| `VIDEOCLEAN_PUBLIC_BASE_URL` | Origin for absolute webhook URLs (e.g. `https://pod-7860.proxy.runpod.net`) |
| `VIDEOCLEAN_CORS` | Existing CORS allowlist |

No new auth mode in this work.

## Files (expected)

- `server/fastapi_app.py` — tags, schemas, OpenAPI security, new form fields, docs metadata
- `server/service.py` — auto-package after run; job report webhook fields; wire formats into completion path
- New small module e.g. `server/webhook.py` — HMAC, POST, retries
- Worker / manage completion hook (wherever jobs flip to `COMPLETED`/`FAILED`) — fire webhook + ensure formats packaged before COMPLETED when requested on run
- `videoclean/application/use_cases/run_cleanup.py` and/or server post-step — either package inside cleanup when `formats` set, or server chains package into same job before terminal state (prefer **server post-step or managed same-job package** so WebUI default `mp4`-only path stays fast and unchanged)
- Tests: OpenAPI contains Public paths + Bearer scheme; formats auto-package; webhook signature + retry/no state flip
- Docs touch: `docs/RUNPOD.md` API blurb; optionally one paragraph in `docs/PARAMS.md` for `webhook_*` / formats semantics

Prefer not bloating domain with HTTP. If packaging must stay in the library, reuse `PackageMedia` from the server completion path with the parent job id.

## Testing

- Prototype-first per project preference; add thin tests for the contract points:
  - `/openapi.json` has `components.securitySchemes` Bearer and Public-tagged job routes.
  - Create run with `formats=mp4,webm` (mocked media) ends `COMPLETED` with both keys in `outputs`.
  - Webhook: mock HTTP server receives one terminal POST; signature matches; failed webhook leaves job `COMPLETED`.
- Manual checklist for the owner: `make dev` / serve, open `/api/docs`, Authorize with token, Try it out upload + poll; optional webhook.site URL.

## Rollout

1. OpenAPI tagging + Job/Options schemas + Bearer scheme (docs usable immediately).
2. Honor `formats` on run via same-job auto-package.
3. Webhook fields + delivery + report status.
4. Update `GET /api` and RUNPOD blurb.

## Risks

- Packaging inside the run worker extends wall-clock for HLS/DASH; document that heavy formats increase time-to-COMPLETED.
- Absolute URLs wrong if `VIDEOCLEAN_PUBLIC_BASE_URL` mismatches the reverse proxy; relative URLs always work with the same host the client already called.
- OpenAPI multipart with many Form fields is noisy; Public example should show the minimal field set, not every advanced knob.
