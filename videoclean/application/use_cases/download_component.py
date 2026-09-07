from __future__ import annotations

import uuid
from collections.abc import Callable

from videoclean.application.errors import DownloadCancelled, PipelineError
from videoclean.store import JobIndex

OnProgress = Callable[..., None]
IsCancelled = Callable[[], bool]
DownloadRunner = Callable[..., None]


class DownloadComponent:
    """Orchestrate one catalog download, writing progress to JobIndex.downloads."""

    def __init__(self, runner: DownloadRunner) -> None:
        self._runner = runner

    def execute(
        self,
        component_id: str,
        jobs: JobIndex,
        progress_cb: OnProgress | None,
    ) -> None:
        if not (component_id or "").strip():
            raise PipelineError("component_id is required")
        component_id = component_id.strip()
        download_id = uuid.uuid4().hex
        jobs.upsert_download(
            download_id,
            component_id,
            "running",
            progress=0.0,
            message="starting",
        )

        def is_cancelled() -> bool:
            row = jobs.get_download(download_id)
            return bool(row and row["cancel_requested"])

        def on_progress(
            fraction: float,
            message: str = "",
            bytes_done: int | None = None,
            bytes_total: int | None = None,
        ) -> None:
            if is_cancelled():
                raise DownloadCancelled("download cancelled")
            frac = max(0.0, min(float(fraction), 1.0))
            jobs.upsert_download(
                download_id,
                component_id,
                "running",
                progress=frac,
                bytes_done=bytes_done,
                bytes_total=bytes_total,
                message=message or None,
            )
            if progress_cb is not None:
                progress_cb(frac, message)

        try:
            self._runner(
                component_id,
                on_progress=on_progress,
                is_cancelled=is_cancelled,
            )
        except DownloadCancelled:
            jobs.upsert_download(
                download_id,
                component_id,
                "cancelled",
                message="cancelled",
            )
            raise
        except Exception as exc:
            jobs.upsert_download(
                download_id,
                component_id,
                "failed",
                message=str(exc)[:400],
            )
            raise
        jobs.upsert_download(
            download_id,
            component_id,
            "done",
            progress=1.0,
            message="ready",
        )
