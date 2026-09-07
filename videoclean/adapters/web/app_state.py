from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from videoclean.application.jobs.worker import JobWorker
from videoclean.application.use_cases.download_component import DownloadComponent
from videoclean.application.use_cases.manage_jobs import ManageJobs
from videoclean.store import JobIndex


@dataclass
class AppState:
    data_dir: Path
    jobs: JobIndex
    manage: ManageJobs
    catalog: Any
    worker: JobWorker | None = None
    downloader: DownloadComponent | None = None


def build_app_state(
    data_dir: Path,
    *,
    worker: JobWorker | None | bool = None,
    catalog: Any = None,
    downloader: DownloadComponent | None | bool = None,
) -> AppState:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    jobs = JobIndex(data_dir / "jobs.sqlite")
    manage = ManageJobs(jobs)
    if catalog is None:
        from videoclean.adapters.models.catalog import ModelCatalog

        catalog = ModelCatalog(jobs=jobs)
    if worker is False:
        worker_obj: JobWorker | None = None
    elif worker is None:
        from videoclean.composition import build_job_worker

        worker_obj = build_job_worker(data_dir, jobs)
    else:
        worker_obj = worker
    if downloader is False:
        downloader_obj: DownloadComponent | None = None
    elif downloader is None:
        from videoclean.adapters.models.downloaders import run_download

        downloader_obj = DownloadComponent(run_download)
    else:
        downloader_obj = downloader
    return AppState(
        data_dir=data_dir,
        jobs=jobs,
        manage=manage,
        catalog=catalog,
        worker=worker_obj,
        downloader=downloader_obj,
    )
