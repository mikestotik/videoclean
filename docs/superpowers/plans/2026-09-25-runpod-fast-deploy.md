# RunPod Fast Deploy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (native inline). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fast CUDA-dev loop on RunPod RTX 4090 via Network Volume + SSH sync, without rebuilding torch images.

**Architecture:** Base image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`; volume holds checkout/venv/HF; GitHub Actions builds webui, rsyncs code, soft-restarts serve. Push deploys only if pod is RUNNING; workflow_dispatch supports start/deploy/stop(terminate).

**Tech Stack:** bash, uv, GitHub Actions, RunPod REST `https://rest.runpod.io/v1`, rsync/SSH over exposed TCP.

**Spec:** `docs/superpowers/specs/2026-09-25-runpod-fast-deploy-design.md`

## Global Constraints

- Never install/reinstall `torch` / `torchvision` via uv on the pod.
- Volume mount path `/workspace`; pod name default `videoclean-dev`.
- GPU: `NVIDIA GeForce RTX 4090`; image tag pinned as in spec.
- Data center must match volume (`EU-RO-1` for volume `nye0ldi1vs`).
- `VIDEOCLEAN_AUTH=off` for this dev path.
- Existing Dockerfiles unchanged.

## Review Focus

- Push with pod offline must not fail the workflow noisily.
- First cold start before any rsync: bootstrap clone must run without pre-existing `/workspace/videoclean/scripts/...`.
- Venv must see image torch (`--system-site-packages`).
- Terminate (not merely stop) on `stop` action so GPU billing ends.
- SSH uses public IP + mapped port 22 (rsync needs full TCP SSH).

---

### Task 1: Pod shell scripts

**Files:**
- Create: `scripts/runpod-env.sh`
- Create: `scripts/runpod-start.sh`
- Create: `scripts/runpod-restart.sh`
- Create: `scripts/runpod-bootstrap.sh`

**Interfaces:**
- Produces: env helpers; start (foreground exec serve); restart (background serve + pid); bootstrap (clone-then-start for empty volume).

- [ ] **Step 1:** Write the four scripts per spec (uv sync skips torch; venv `--system-site-packages`; sam2 git install after sync; pid under `/workspace/run`).
- [ ] **Step 2:** `bash -n` on all four scripts.
- [ ] **Step 3:** Commit (if git identity available).

### Task 2: GitHub Actions workflow + helper

**Files:**
- Create: `scripts/runpod-api.sh` (list/find/create/terminate + wait SSH)
- Create: `.github/workflows/runpod-deploy.yml`

- [ ] **Step 1:** API helper using REST v1 + jq.
- [ ] **Step 2:** Workflow: build web → deploy/start/stop; push to main with path filters; offline pod = success notice on push.
- [ ] **Step 3:** `bash -n scripts/runpod-api.sh`; validate YAML with Python/`actionlint` if available.
- [ ] **Step 4:** Commit.

### Task 3: Docs

**Files:**
- Create: `docs/RUNPOD.md`
- Modify: `README.md` (one link under UI/API deploy)

- [ ] **Step 1:** Document secrets, one-time setup, start/deploy/stop, URLs, troubleshooting.
- [ ] **Step 2:** Link from README.
- [ ] **Step 3:** Commit.
