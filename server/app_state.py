from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from videoclean.application.jobs.worker import JobWorker
from videoclean.application.use_cases.download_component import DownloadComponent
from videoclean.application.use_cases.manage_jobs import ManageJobs
from videoclean.store import JobIndex, SourceIndex


@dataclass
class AppState:
    data_dir: Path
    jobs: JobIndex
    manage: ManageJobs
    catalog: Any
    sources: SourceIndex | None = None
    worker: JobWorker | None = None
    downloader: DownloadComponent | None = None
    # Last finished download line so the UI does not snap back to 0%/idle.
    last_download_frac: float = 0.0
    last_download_msg: str = "No download in progress."


def build_app_state(
    data_dir: Path,
    *,
    worker: JobWorker | None | bool = None,
    catalog: Any = None,
    downloader: DownloadComponent | None | bool = None,
    on_terminal: Any | None = None,
) -> AppState:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    jobs = JobIndex(data_dir / "jobs.sqlite")
    sources = SourceIndex(data_dir / "jobs.sqlite")
    manage = ManageJobs(jobs)
    if catalog is None:
        from videoclean.adapters.models.catalog import ModelCatalog

        catalog = ModelCatalog(jobs=jobs)
    state = AppState(
        data_dir=data_dir,
        jobs=jobs,
        manage=manage,
        catalog=catalog,
        sources=sources,
        worker=None,
        downloader=None,
    )

    def _default_on_terminal(job_id: str, terminal_state: str) -> None:
        from server.webhook import maybe_deliver_job_webhook

        maybe_deliver_job_webhook(state, job_id, terminal_state)

    terminal_hook = on_terminal if on_terminal is not None else _default_on_terminal

    if worker is False:
        worker_obj: JobWorker | None = None
    elif worker is None:
        from videoclean.composition import build_job_worker

        worker_obj = build_job_worker(data_dir, jobs, on_terminal=terminal_hook)
    else:
        worker_obj = worker
        if getattr(worker_obj, "on_terminal", None) is None and terminal_hook is not None:
            worker_obj.on_terminal = terminal_hook
    if downloader is False:
        downloader_obj: DownloadComponent | None = None
    elif downloader is None:
        from videoclean.adapters.models.downloaders import run_download

        downloader_obj = DownloadComponent(run_download)
    else:
        downloader_obj = downloader
    state.worker = worker_obj
    state.downloader = downloader_obj
    return state
