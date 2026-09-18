from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Callable

from videoclean.application.errors import JobCancelled
from videoclean.application.use_cases.manage_jobs import cleanup_request_from_row
from videoclean.store import JobIndex, utc_now


class JobWorker:
    def __init__(
        self,
        data_dir: Path,
        jobs: JobIndex,
        build_runner: Callable[..., Any],
        idle_s: float = 0.05,
        build_preview_runner: Callable[..., Any] | None = None,
        build_prompt_runner: Callable[..., Any] | None = None,
        on_terminal: Callable[[str, str], None] | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.jobs = jobs
        self.build_runner = build_runner
        self.build_preview_runner = build_preview_runner or build_runner
        self.build_prompt_runner = build_prompt_runner
        self.on_terminal = on_terminal
        self._idle_s = idle_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _emit_terminal(self, job_id: str) -> None:
        if self.on_terminal is None:
            return
        row = self.jobs.get(job_id)
        if row is None:
            return
        state = str(row["state"] or "")
        if state not in {"COMPLETED", "FAILED", "CANCELLED"}:
            return
        try:
            self.on_terminal(job_id, state)
        except Exception:  # noqa: BLE001 — terminal hooks must not kill the worker
            return

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="videoclean-job-worker", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 5) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)

    def _loop(self) -> None:
        while not self._stop.is_set():
            job_id = self._claim()
            if job_id is None:
                self._stop.wait(self._idle_s)
                continue
            self._run_one(job_id)

    def _claim(self) -> str | None:
        queued = list(self.jobs.list_jobs(state="QUEUED", limit=10_000))
        if not queued:
            return None
        queued.sort(key=lambda row: (row["created_at"], row["id"]))
        job_id = queued[0]["id"]
        row = self.jobs.get(job_id)
        if row is None or row["state"] != "QUEUED":
            return None
        if self.jobs.is_cancel_requested(job_id):
            self.jobs.upsert(job_id, "CANCELLED", error="cancelled")
            self._emit_terminal(job_id)
            return None
        now = utc_now().isoformat()
        self.jobs.upsert(
            job_id,
            "RUNNING",
            progress={
                "stage": "validate",
                "fraction": 0.0,
                "detail": "",
                "heartbeat_at": now,
                "started_at": now,
            },
        )
        return job_id

    def _run_one(self, job_id: str) -> None:
        if self.jobs.is_cancel_requested(job_id):
            self.jobs.upsert(job_id, "CANCELLED", error="cancelled")
            self._emit_terminal(job_id)
            return
        row = self.jobs.get(job_id)
        if row is None:
            return
        payload: dict = {}
        try:
            payload = json.loads(row["request_json"] or "{}")
        except (TypeError, ValueError):
            payload = {}
        if payload.get("kind") == "preview":
            self._run_preview(job_id, row, payload)
            return
        if payload.get("kind") == "prompt":
            self._run_prompt(job_id, row, payload)
            return
        if payload.get("kind") == "package":
            self._run_package(job_id, row, payload)
            return
        try:
            req = cleanup_request_from_row(row)
            runner = self.build_runner(req.config, None, self.jobs, job_id)
            report = runner.execute(req, self.data_dir)
        except JobCancelled:
            self.jobs.upsert(job_id, "CANCELLED", error="cancelled")
            self._emit_terminal(job_id)
            return
        except Exception as exc:  # noqa: BLE001 — queue must isolate job failures
            current = self.jobs.get(job_id)
            if current is None:
                return
            # RunCleanup may already flip to FAILED/CANCELLED before re-raising.
            if current["state"] in {"FAILED", "CANCELLED"}:
                if not current["error"]:
                    self.jobs.upsert(job_id, current["state"], error=str(exc)[:500])
                self._emit_terminal(job_id)
                return
            if current["state"] != "RUNNING":
                self._emit_terminal(job_id)
                return
            if self.jobs.is_cancel_requested(job_id):
                self.jobs.upsert(job_id, "CANCELLED", error=str(exc)[:500])
            else:
                self.jobs.upsert(job_id, "FAILED", error=str(exc)[:500])
            self._emit_terminal(job_id)
            return
        current = self.jobs.get(job_id)
        if current is not None and current["state"] == "COMPLETED":
            # RunCleanup already marked COMPLETED.
            self._emit_terminal(job_id)
            return
        if current is None or current["state"] != "RUNNING":
            self._emit_terminal(job_id)
            return
        if self.jobs.is_cancel_requested(job_id):
            self.jobs.upsert(job_id, "CANCELLED", error="cancelled")
            self._emit_terminal(job_id)
            return
        payload = report if isinstance(report, dict) else None
        self.jobs.upsert(job_id, "COMPLETED", report=payload)
        self._emit_terminal(job_id)

    def _run_preview(self, job_id: str, row, payload: dict) -> None:
        from videoclean.application.use_cases.run_preview import PreviewRequest

        try:
            config = self._config_from_payload(payload)
            input_path = Path(row["input_path"] or payload.get("input_path") or "")
            req = PreviewRequest(
                input_path=input_path,
                prompt=payload.get("prompt") or "",
                config=config,
                start=payload.get("start"),
                count=payload.get("count"),
                stride=payload.get("stride"),
                indices=payload.get("indices"),
                mode=payload.get("mode") or "parse",
                targets=payload.get("targets"),
                job_id=job_id,
            )
            runner = self.build_preview_runner(config, None, self.jobs, job_id)
            report = runner.execute(req, self.data_dir)
        except JobCancelled:
            self.jobs.upsert(job_id, "CANCELLED", error="cancelled")
            self._emit_terminal(job_id)
            return
        except Exception as exc:  # noqa: BLE001 — queue must isolate job failures
            current = self.jobs.get(job_id)
            if current is None:
                return
            if current["state"] in {"FAILED", "CANCELLED"}:
                self._emit_terminal(job_id)
                return
            if self.jobs.is_cancel_requested(job_id):
                self.jobs.upsert(job_id, "CANCELLED", error=str(exc)[:500])
            else:
                self.jobs.upsert(job_id, "FAILED", error=str(exc)[:800])
            self._emit_terminal(job_id)
            return
        current = self.jobs.get(job_id)
        if current is not None and current["state"] == "COMPLETED":
            self._emit_terminal(job_id)
            return
        if current is not None and current["state"] == "RUNNING":
            self.jobs.upsert(job_id, "COMPLETED", report=report)
            self._emit_terminal(job_id)

    def _run_package(self, job_id: str, row, payload: dict) -> None:
        from videoclean.composition import build_packager
        from videoclean.domain.formats import parse_formats

        started = utc_now().isoformat()

        def set_progress(fraction: float, detail: str = "", stage: str = "package") -> None:
            self.jobs.upsert(
                job_id,
                "RUNNING",
                progress={
                    "stage": stage,
                    "fraction": max(0.0, min(0.99, float(fraction))),
                    "detail": detail,
                    "heartbeat_at": utc_now().isoformat(),
                    "started_at": started,
                },
            )

        try:
            if self.jobs.is_cancel_requested(job_id):
                raise JobCancelled("cancelled")
            parent_id = str(payload.get("parent_job_id") or "").strip()
            if not parent_id:
                raise RuntimeError("parent_job_id is required for package jobs")
            parent = self.jobs.get(parent_id)
            if parent is None:
                raise RuntimeError(f"unknown parent job {parent_id}")
            src = Path(row["input_path"] or payload.get("input_path") or "")
            if not src.is_file():
                raise RuntimeError(f"mezzanine missing: {src}")
            out_base = Path(row["output_path"] or payload.get("output_path") or "")
            fmts = parse_formats(payload.get("formats") or ["mp4"])
            webm_crf = int(payload.get("webm_crf") or 32)
            segment_seconds = int(payload.get("segment_seconds") or 6)
            overwrite = bool(payload.get("overwrite", True))
            log_file = out_base.parent / "ffmpeg-package.log"
            log_file.parent.mkdir(parents=True, exist_ok=True)
            set_progress(0.05, f"0/{len(fmts)}")

            def on_format(fmt: str, i: int, total: int, _path: Path) -> None:
                if self.jobs.is_cancel_requested(job_id):
                    raise JobCancelled("cancelled")
                set_progress(i / max(total, 1), f"{fmt} ({i}/{total})")

            artifacts = build_packager().execute(
                src,
                out_base,
                fmts,
                overwrite,
                log_file,
                segment_seconds=segment_seconds,
                webm_crf=webm_crf,
                on_format=on_format,
            )
            # Merge into parent report so downloads stay on the cleanup job.
            try:
                parent_report = json.loads(parent["report_json"] or "{}")
            except (TypeError, ValueError):
                parent_report = {}
            if not isinstance(parent_report, dict):
                parent_report = {}
            outputs = dict(parent_report.get("outputs") or {})
            for fmt, path in artifacts.items():
                outputs[fmt] = str(path)
            parent_report["outputs"] = outputs
            parent_report["formats"] = sorted({*list(parent_report.get("formats") or []), *fmts})
            parent_report["packagedAt"] = utc_now().isoformat()
            self.jobs.upsert(parent_id, "COMPLETED", report=parent_report)
            workdir = parent_report.get("workdir")
            if isinstance(workdir, str) and workdir:
                report_path = Path(workdir) / "output" / "report.json"
                if report_path.parent.is_dir():
                    report_path.write_text(
                        json.dumps(parent_report, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
            report = {
                "kind": "package",
                "parent_job_id": parent_id,
                "formats": fmts,
                "outputs": {fmt: str(path) for fmt, path in artifacts.items()},
                "state": "COMPLETED",
                "finishedAt": utc_now().isoformat(),
            }
        except JobCancelled:
            self.jobs.upsert(job_id, "CANCELLED", error="cancelled")
            self._emit_terminal(job_id)
            return
        except Exception as exc:  # noqa: BLE001 — queue must isolate job failures
            current = self.jobs.get(job_id)
            if current is None:
                return
            if current["state"] in {"FAILED", "CANCELLED"}:
                self._emit_terminal(job_id)
                return
            if self.jobs.is_cancel_requested(job_id):
                self.jobs.upsert(job_id, "CANCELLED", error=str(exc)[:500])
            else:
                self.jobs.upsert(job_id, "FAILED", error=str(exc)[:800])
            self._emit_terminal(job_id)
            return
        current = self.jobs.get(job_id)
        if current is not None and current["state"] == "RUNNING":
            self.jobs.upsert(
                job_id,
                "COMPLETED",
                report=report,
                progress={
                    "stage": "package",
                    "fraction": 1.0,
                    "detail": "done",
                    "heartbeat_at": utc_now().isoformat(),
                    "started_at": started,
                },
            )
            self._emit_terminal(job_id)

    def _run_prompt(self, job_id: str, row, payload: dict) -> None:
        from videoclean.application.use_cases.build_prompt import BuildPromptRequest

        try:
            if self.build_prompt_runner is None:
                raise RuntimeError("prompt runner is not configured")
            config = self._config_from_payload(payload)
            input_path = Path(row["input_path"] or payload.get("input_path") or "")
            annotations = [
                {"frame": int(a["frame"]), "mask": Path(str(a.get("mask") or ""))}
                for a in (payload.get("annotations") or [])
                if str(a.get("mask") or "").strip()
            ]
            req = BuildPromptRequest(
                input_path=input_path,
                prompt=payload.get("prompt") or "",
                annotations=annotations,
                config=config,
                job_id=job_id,
            )
            runner = self.build_prompt_runner(config, None, self.jobs, job_id)
            report = runner.execute(req, self.data_dir)
        except JobCancelled:
            self.jobs.upsert(job_id, "CANCELLED", error="cancelled")
            self._emit_terminal(job_id)
            return
        except Exception as exc:  # noqa: BLE001 — queue must isolate job failures
            current = self.jobs.get(job_id)
            if current is None:
                return
            if current["state"] in {"FAILED", "CANCELLED"}:
                self._emit_terminal(job_id)
                return
            if self.jobs.is_cancel_requested(job_id):
                self.jobs.upsert(job_id, "CANCELLED", error=str(exc)[:500])
            else:
                self.jobs.upsert(job_id, "FAILED", error=str(exc)[:800])
            self._emit_terminal(job_id)
            return
        current = self.jobs.get(job_id)
        if current is not None and current["state"] == "COMPLETED":
            self._emit_terminal(job_id)
            return
        if current is not None and current["state"] == "RUNNING":
            self.jobs.upsert(job_id, "COMPLETED", report=report)
            self._emit_terminal(job_id)

    def _config_from_payload(self, payload: dict):
        from videoclean.application.use_cases.manage_jobs import pipeline_config_from_dict

        return pipeline_config_from_dict(payload)
