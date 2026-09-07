from __future__ import annotations

import json
import shutil
from dataclasses import fields
from pathlib import Path
from typing import Any

from videoclean.application.config import (
    DEFAULT_DETECTORS,
    DETECTORS,
    PipelineConfig,
    RunCleanupRequest,
    parse_name_list,
)
from videoclean.application.errors import PipelineError
from videoclean.store import JobIndex, new_job_id

_CONFIG_FIELDS = {f.name for f in fields(PipelineConfig)}


def pipeline_config_from_dict(data: dict[str, Any] | None) -> PipelineConfig:
    data = data or {}
    kwargs = {key: data[key] for key in _CONFIG_FIELDS if key in data}
    if "detectors" not in kwargs and "detector" in data:
        kwargs["detectors"] = parse_name_list(
            data["detector"], DETECTORS, default=list(DEFAULT_DETECTORS)
        )
    return PipelineConfig(**kwargs)


def cleanup_request_from_row(row: Any) -> RunCleanupRequest:
    payload = json.loads(row["request_json"] or "{}")
    cfg = pipeline_config_from_dict(payload)
    input_path = Path(row["input_path"] or payload.get("input_path") or "")
    output_path = Path(row["output_path"] or payload.get("output_path") or "")
    prompt = row["prompt"] if row["prompt"] is not None else payload.get("prompt") or ""
    return RunCleanupRequest(
        input_path=input_path,
        output_path=output_path,
        prompt=prompt,
        config=cfg,
        keep_workdir=bool(payload.get("keep_workdir", False)),
        overwrite=bool(payload.get("overwrite", True)),
        job_id=row["id"],
    )


class ManageJobs:
    def __init__(self, jobs: JobIndex) -> None:
        self.jobs = jobs

    def submit(
        self,
        request_dict: dict[str, Any],
        input_path: Path,
        output_path: Path,
        prompt: str,
        job_id: str | None = None,
    ) -> str:
        job_id = job_id or new_job_id()
        payload = dict(request_dict or {})
        payload.setdefault("input_path", str(input_path))
        payload.setdefault("output_path", str(output_path))
        payload.setdefault("prompt", prompt)
        self.jobs.upsert(
            job_id,
            "QUEUED",
            input_path=str(input_path),
            output_path=str(output_path),
            prompt=prompt,
            request=payload,
        )
        return job_id

    def cancel(self, job_id: str) -> None:
        row = self.jobs.get(job_id)
        if row is None:
            return
        if row["state"] == "QUEUED":
            self.jobs.upsert(job_id, "CANCELLED", error="cancelled", cancel_requested=True)
            return
        if row["state"] == "RUNNING":
            self.jobs.request_cancel(job_id)

    def retry(self, job_id: str) -> str:
        row = self.jobs.get(job_id)
        if row is None:
            raise PipelineError(f"unknown job {job_id}")
        payload = json.loads(row["request_json"] or "{}")
        input_path = Path(row["input_path"] or payload.get("input_path") or "")
        output_path = Path(row["output_path"] or payload.get("output_path") or "")
        prompt = row["prompt"] if row["prompt"] is not None else payload.get("prompt") or ""
        return self.submit(payload, input_path, output_path, prompt)

    def delete(self, job_id: str, data_dir: Path) -> None:
        self.jobs.delete(job_id)
        shutil.rmtree(Path(data_dir) / "jobs" / job_id, ignore_errors=True)

    def mark_failed(self, job_id: str, reason: str) -> None:
        self.jobs.upsert(job_id, "FAILED", error=reason)

    def recover_orphans(self) -> int:
        return self.jobs.mark_orphans_failed("interrupted")
