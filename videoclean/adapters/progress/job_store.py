from __future__ import annotations

import time

from videoclean.application.errors import JobCancelled
from videoclean.progress import STAGES
from videoclean.store import JobIndex, utc_now


class ProgressBridge:
    def __init__(self, jobs: JobIndex, job_id: str) -> None:
        self.jobs = jobs
        self.job_id = job_id
        self._started_at = utc_now().isoformat()
        self._stages = {
            key: {
                "title": title,
                "weight": weight,
                "status": "pending",
                "current": 0,
                "total": 0,
                "detail": "",
            }
            for key, title, weight in STAGES
        }
        self._clock: dict[str, float] = {}
        self._seconds: dict[str, float] = {}

    def start(self, key: str, total: int = 0, detail: str = "") -> None:
        self._raise_if_cancelled()
        stage = self._ensure(key)
        stage["status"] = "run"
        stage["total"] = total
        stage["current"] = 0
        stage["detail"] = detail
        self._clock[key] = time.monotonic()
        self._flush(key)

    def tick(self, key: str, current: int, total: int | None = None, detail: str = "") -> None:
        self._raise_if_cancelled()
        stage = self._ensure(key)
        stage["current"] = current
        if total is not None:
            stage["total"] = total
        if detail:
            stage["detail"] = detail
        self._flush(key)

    def finish(self, key: str, detail: str = "") -> None:
        self._raise_if_cancelled()
        stage = self._ensure(key)
        stage["status"] = "done"
        stage["detail"] = detail
        if stage["total"]:
            stage["current"] = stage["total"]
        started = self._clock.pop(key, None)
        if started is not None:
            self._seconds[key] = self._seconds.get(key, 0.0) + (time.monotonic() - started)
        self._flush(key)

    def stage_seconds(self) -> dict[str, float]:
        return dict(self._seconds)

    def stage_title(self, key: str) -> str:
        stage = self._stages.get(key) or {}
        return str(stage.get("title") or key)

    def fraction_done(self) -> float:
        weighted = [stage for stage in self._stages.values() if stage["weight"]]
        if weighted and all(stage["status"] == "done" for stage in weighted):
            return 1.0
        done = 0.0
        for stage in self._stages.values():
            if stage["status"] == "done":
                done += stage["weight"]
            elif stage["status"] == "run":
                if stage["total"]:
                    done += stage["weight"] * min(1.0, stage["current"] / stage["total"])
                else:
                    done += stage["weight"] * 0.15
        return min(0.99, done)

    def _ensure(self, key: str) -> dict:
        if key not in self._stages:
            self._stages[key] = {
                "title": key,
                "weight": 0.0,
                "status": "pending",
                "current": 0,
                "total": 0,
                "detail": "",
            }
        return self._stages[key]

    def _raise_if_cancelled(self) -> None:
        if self.jobs.is_cancel_requested(self.job_id):
            raise JobCancelled(self.job_id)

    def _flush(self, key: str) -> None:
        stage = self._stages.get(key) or {}
        self.jobs.update_progress(
            self.job_id,
            {
                "stage": key,
                "stageTitle": stage.get("title") or key,
                "fraction": self.fraction_done(),
                "detail": stage.get("detail") or "",
                "heartbeat_at": utc_now().isoformat(),
                "started_at": self._started_at,
            },
        )
