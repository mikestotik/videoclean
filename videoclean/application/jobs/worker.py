from __future__ import annotations

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
    ) -> None:
        self.data_dir = Path(data_dir)
        self.jobs = jobs
        self.build_runner = build_runner
        self._idle_s = idle_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

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
            return
        row = self.jobs.get(job_id)
        if row is None:
            return
        try:
            req = cleanup_request_from_row(row)
            runner = self.build_runner(req.config, None, self.jobs, job_id)
            report = runner.execute(req, self.data_dir)
        except JobCancelled:
            self.jobs.upsert(job_id, "CANCELLED", error="cancelled")
            return
        except Exception as exc:  # noqa: BLE001 — queue must isolate job failures
            current = self.jobs.get(job_id)
            if current is None:
                return
            # RunCleanup may already flip to FAILED/CANCELLED before re-raising.
            if current["state"] in {"FAILED", "CANCELLED"}:
                if not current["error"]:
                    self.jobs.upsert(job_id, current["state"], error=str(exc)[:500])
                return
            if current["state"] != "RUNNING":
                return
            if self.jobs.is_cancel_requested(job_id):
                self.jobs.upsert(job_id, "CANCELLED", error=str(exc)[:500])
            else:
                self.jobs.upsert(job_id, "FAILED", error=str(exc)[:500])
            return
        current = self.jobs.get(job_id)
        if current is None or current["state"] != "RUNNING":
            return
        if self.jobs.is_cancel_requested(job_id):
            self.jobs.upsert(job_id, "CANCELLED", error="cancelled")
            return
        payload = report if isinstance(report, dict) else None
        self.jobs.upsert(job_id, "COMPLETED", report=payload)
