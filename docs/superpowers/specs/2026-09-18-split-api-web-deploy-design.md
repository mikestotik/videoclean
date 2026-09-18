# Split API / Web deploy (internal contour)

Date: 2026-09-18  
Status: approved in chat; awaiting file review

## Goal

Run backend and frontend as **two services** (different instances / images) for internal CI/CD and docker-compose. Auth stays off on the internal contour via env flag; accounts come later. Keep an optional all-in-one image for RunPod-style one-box deploys.

## Non-goals

- User accounts, JWT, OAuth, multi-tenant auth.
- Runtime `config.json` / `window.__ENV` for API URL (build-time `VITE_API_BASE_URL` only).
- Splitting the job worker into a third service.
- Production TLS / ingress beyond what compose already needs locally.

## Decisions

| Topic | Choice |
|---|---|
| Compose services | `api` + `web` + optional `monolith` (compose profile) |
| Frontend → API | Build-time `VITE_API_BASE_URL` (empty = same-origin / vite proxy) |
| Web server | nginx, host port **8080** → container 80 |
| Auth (internal) | `VIDEOCLEAN_AUTH=off` (middleware off; UI password not required) |
| CORS | `VIDEOCLEAN_CORS` (compose internal: `*` or web origin) |

## Current state

- Single `docker-compose.yml` service `videoclean` from one `Dockerfile` (bun build + CUDA api + `server/static_dist`).
- WebUI calls relative `/api/...`; vite proxies to `:7860` in dev; prod is same-origin from FastAPI static.
- Auth: Basic/Bearer middleware; `VIDEOCLEAN_UI_PASSWORD` required to start.

## Target topology

```
Browser
  ├─ http://host:8080  →  web (nginx static SPA)
  └─ http://host:7860  →  api (FastAPI + worker + GPU)
       (VITE_API_BASE_URL points at the public/base URL of api)
```

Optional:

```
Browser → :7860 → monolith (api + embedded static_dist)
```

## Backend changes

### Auth flag

- Env `VIDEOCLEAN_AUTH`: `off` / `0` / `false` / `no` → auth disabled; otherwise enabled (default **on** for backward-compatible local/prod habits, compose internal sets **off**).
- When off:
  - Do not require `VIDEOCLEAN_UI_PASSWORD` in `auth_from_env` / `launch_from_env`.
  - Do not install `BasicOrBearerAuth` (or install a no-op pass-through).
  - `/health`, `/openapi.json`, `/api/docs` remain reachable as today.
- When on: current Basic + Bearer behavior unchanged.

### API-only image

- New `Dockerfile.api`: same CUDA/Python stack as today **without** the webui bun stage and **without** copying `static_dist`.
- Serve works with placeholder HTML if dist missing (already present).
- `scripts/start.sh` reused (doctor + serve).

### CORS

- Keep `VIDEOCLEAN_CORS`. Document for split: set to web origin or `*` on internal networks.

## Frontend changes

### API base URL

- `import.meta.env.VITE_API_BASE_URL` (trim trailing `/`).
- `apiUrl(path)` in `webui/src/shared/api/` joins base + path (`path` always starts with `/`).
- Route all HTTP through it: `api()` client, `downloadJobOutput`, frame/video/preview URLs that the browser loads directly.
- Empty base → current relative URLs (dev proxy + monolith unchanged).

### Web image

- New `Dockerfile.web`:
  1. `oven/bun` build with `ARG VITE_API_BASE_URL`.
  2. `nginx:alpine` (or stable) with SPA `try_files $uri /index.html` for `/`, `/settings`, `/config`, `/v/*`.
  3. Listen 80; compose maps `8080:80`.

### Vite build output

- Prefer minimal churn: keep building into `server/static_dist` for monolith Dockerfile COPY path **or** build to `webui/dist` and adjust both Dockerfiles to COPY the same artifact. Spec requirement: **one** build output consumed by web image and by monolith. Implementation may switch `outDir` to `dist` and update monolith `COPY` accordingly if cleaner.

## docker-compose.yml

```yaml
services:
  api:
    build:
      context: .
      dockerfile: Dockerfile.api
    image: videoclean-api:local
    ports:
      - "${VIDEOCLEAN_PORT:-7860}:7860"
    env_file: [.env]
    environment:
      VIDEOCLEAN_AUTH: "off"
      VIDEOCLEAN_CORS: "*"
      VIDEOCLEAN_DATA_DIR: /root/.videoclean
      HF_HOME: /root/.cache/huggingface
      VIDEOCLEAN_PORT: "7860"
    volumes:
      - videoclean-data:/root/.videoclean
      - hf-cache:/root/.cache/huggingface
    gpus: all
    restart: unless-stopped

  web:
    build:
      context: .
      dockerfile: Dockerfile.web
      args:
        VITE_API_BASE_URL: ${VITE_API_BASE_URL:-http://localhost:7860}
    image: videoclean-web:local
    ports:
      - "${VIDEOCLEAN_WEB_PORT:-8080}:80"
    depends_on: [api]
    restart: unless-stopped

  monolith:
    profiles: ["monolith"]
    build:
      context: .
      dockerfile: Dockerfile
    image: videoclean:local
    # …existing monolith env/volumes/gpus…
```

`.env.example` gains: `VIDEOCLEAN_AUTH`, `VITE_API_BASE_URL`, `VIDEOCLEAN_WEB_PORT`, `VIDEOCLEAN_CORS`.

## Make / docs

- `make docker` builds `videoclean-api:local` and `videoclean-web:local` (and documents `docker compose --profile monolith build` for one-box).
- README + AGENTS: two-service deploy; auth flag; `VITE_API_BASE_URL`.
- OpenAPI/auth blurb: note auth can be disabled for internal.

## CI (stub)

- GitHub Actions (or existing CI path if any): on PR — `uv run pytest -q` (subset ok), `cd webui && bun run lint && bun run typecheck`.
- Optional job: `docker build -f Dockerfile.api` / `Dockerfile.web` (no push required in v1).
- No deploy stage.

## Testing

- Backend: auth off starts without password; `/api/options` 200 without Authorization; auth on still 401 without creds.
- Frontend: unit/helper test or small assert that `apiUrl('/api/jobs')` prefixes base.
- Compose smoke (manual): `docker compose up --build`, open `:8080`, confirm network calls hit `:7860`.

## Rollout

1. Auth flag + `apiUrl` + wire fetches.
2. `Dockerfile.api`, `Dockerfile.web`, nginx conf, rewrite compose.
3. Adjust monolith Dockerfile to still embed UI.
4. `.env.example`, Make, README/AGENTS.
5. CI stub workflow.

## Risks

- Wrong `VITE_API_BASE_URL` (e.g. Docker-internal `http://api:7860`) breaks **browser** calls; browser needs a host-reachable URL (`localhost` / public hostname).
- CORS misconfig blocks web→api; internal `*` is acceptable until accounts.
- Absolute media URLs in UI must use `apiUrl` or video/frames break cross-origin.
