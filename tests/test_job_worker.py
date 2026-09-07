import json
import threading
import time
from pathlib import Path

import pytest

from videoclean.adapters.web.progress_bridge import ProgressBridge
from videoclean.application.errors import JobCancelled
from videoclean.application.jobs.worker import JobWorker
from videoclean.application.use_cases.manage_jobs import ManageJobs
from videoclean.store import JobIndex


class FakeRunner:
    def __init__(self):
        self.ran = []

    def execute(self, req, data_dir):
        self.ran.append(req.job_id)
        time.sleep(0.05)
        return {"jobId": req.job_id, "state": "COMPLETED", "outputs": {}}


def _wait_state(jobs: JobIndex, job_id: str, *states: str, timeout: float = 3.0) -> str:
    deadline = time.time() + timeout
    while time.time() < deadline:
        row = jobs.get(job_id)
        if row is not None and row["state"] in states:
            return row["state"]
        time.sleep(0.02)
    row = jobs.get(job_id)
    actual = row["state"] if row is not None else None
    raise AssertionError(f"job {job_id} state={actual!r} not in {states}")


def test_submit_queued(tmp_path: Path):
    jobs = JobIndex(tmp_path / "j.sqlite")
    mgr = ManageJobs(jobs)
    jid = mgr.submit({"device": "cpu"}, tmp_path / "in.mp4", tmp_path / "out.mp4", "remove logo")
    assert jobs.get(jid)["state"] == "QUEUED"


def test_cancel_queued(tmp_path: Path):
    jobs = JobIndex(tmp_path / "j.sqlite")
    mgr = ManageJobs(jobs)
    jid = mgr.submit({}, tmp_path / "a.mp4", tmp_path / "b.mp4", "x")
    mgr.cancel(jid)
    assert jobs.get(jid)["state"] == "CANCELLED"


def test_worker_fifo(tmp_path: Path):
    jobs = JobIndex(tmp_path / "j.sqlite")
    mgr = ManageJobs(jobs)
    runner = FakeRunner()

    def build_runner(cfg, progress, jobs, job_id):
        return runner

    worker = JobWorker(tmp_path, jobs, build_runner)
    j1 = mgr.submit({"device": "cpu"}, tmp_path / "a.mp4", tmp_path / "out1.mp4", "one")
    j2 = mgr.submit({"device": "cpu"}, tmp_path / "b.mp4", tmp_path / "out2.mp4", "two")
    worker.start()
    try:
        _wait_state(jobs, j1, "COMPLETED")
        _wait_state(jobs, j2, "COMPLETED")
        assert runner.ran == [j1, j2]
    finally:
        worker.stop()


def test_one_running_at_a_time(tmp_path: Path):
    jobs = JobIndex(tmp_path / "j.sqlite")
    mgr = ManageJobs(jobs)
    started = threading.Event()
    release = threading.Event()

    class BlockingRunner:
        def execute(self, req, data_dir):
            started.set()
            assert release.wait(timeout=2)
            return {"jobId": req.job_id, "state": "COMPLETED", "outputs": {}}

    worker = JobWorker(tmp_path, jobs, lambda cfg, progress, jobs, job_id: BlockingRunner())
    j1 = mgr.submit({"device": "cpu"}, tmp_path / "a.mp4", tmp_path / "o1.mp4", "one")
    j2 = mgr.submit({"device": "cpu"}, tmp_path / "b.mp4", tmp_path / "o2.mp4", "two")
    worker.start()
    try:
        assert started.wait(timeout=2)
        time.sleep(0.05)
        assert jobs.get(j1)["state"] == "RUNNING"
        assert jobs.get(j2)["state"] == "QUEUED"
    finally:
        release.set()
        worker.stop()


def test_cancel_queued_not_executed(tmp_path: Path):
    jobs = JobIndex(tmp_path / "j.sqlite")
    mgr = ManageJobs(jobs)
    runner = FakeRunner()
    worker = JobWorker(tmp_path, jobs, lambda cfg, progress, jobs, job_id: runner)
    jid = mgr.submit({}, tmp_path / "a.mp4", tmp_path / "b.mp4", "x")
    mgr.cancel(jid)
    worker.start()
    try:
        time.sleep(0.2)
        assert jobs.get(jid)["state"] == "CANCELLED"
        assert runner.ran == []
    finally:
        worker.stop()


def test_successful_execute_keeps_completed_despite_cancel_flag(tmp_path: Path):
    jobs = JobIndex(tmp_path / "j.sqlite")
    mgr = ManageJobs(jobs)

    class CompletingRunner:
        def execute(self, req, data_dir):
            jobs.upsert(req.job_id, "COMPLETED", report={"jobId": req.job_id, "state": "COMPLETED"})
            jobs.request_cancel(req.job_id)
            return {"jobId": req.job_id, "state": "COMPLETED"}

    worker = JobWorker(tmp_path, jobs, lambda cfg, progress, jobs, job_id: CompletingRunner())
    jid = mgr.submit({}, tmp_path / "a.mp4", tmp_path / "b.mp4", "x")
    worker.start()
    try:
        _wait_state(jobs, jid, "COMPLETED", "CANCELLED")
        assert jobs.get(jid)["state"] == "COMPLETED"
        assert jobs.is_cancel_requested(jid) is True
    finally:
        worker.stop()


def test_cancel_running_sets_flag(tmp_path: Path):
    jobs = JobIndex(tmp_path / "j.sqlite")
    mgr = ManageJobs(jobs)
    started = threading.Event()
    release = threading.Event()

    class BlockingRunner:
        def execute(self, req, data_dir):
            started.set()
            assert release.wait(timeout=2)
            return {"jobId": req.job_id, "state": "COMPLETED", "outputs": {}}

    worker = JobWorker(tmp_path, jobs, lambda cfg, progress, jobs, job_id: BlockingRunner())
    jid = mgr.submit({}, tmp_path / "a.mp4", tmp_path / "b.mp4", "x")
    worker.start()
    try:
        assert started.wait(timeout=2)
        mgr.cancel(jid)
        assert jobs.is_cancel_requested(jid) is True
        assert jobs.get(jid)["state"] == "RUNNING"
    finally:
        release.set()
        worker.stop()


def test_retry_clones_request(tmp_path: Path):
    jobs = JobIndex(tmp_path / "j.sqlite")
    mgr = ManageJobs(jobs)
    src = tmp_path / "in.mp4"
    out = tmp_path / "out.mp4"
    jid = mgr.submit({"device": "cpu", "inpainter": "lama"}, src, out, "remove logo")
    mgr.mark_failed(jid, "boom")
    new_id = mgr.retry(jid)
    assert new_id != jid
    row = jobs.get(new_id)
    assert row["state"] == "QUEUED"
    assert row["prompt"] == "remove logo"
    assert row["input_path"] == str(src)
    request = json.loads(row["request_json"])
    assert request["device"] == "cpu"
    assert request["inpainter"] == "lama"


def test_recover_orphans(tmp_path: Path):
    jobs = JobIndex(tmp_path / "j.sqlite")
    mgr = ManageJobs(jobs)
    jobs.upsert("run", "RUNNING")
    jobs.upsert("q", "QUEUED")
    jobs.upsert_download("dl1", "segmenter:sam2-large", "running", progress=0.5)
    n = mgr.recover_orphans()
    assert n == 2
    assert jobs.get("run")["state"] == "FAILED"
    assert jobs.get("q")["state"] == "QUEUED"
    assert jobs.get_download("dl1")["state"] == "cancelled"


def test_delete_job(tmp_path: Path):
    jobs = JobIndex(tmp_path / "j.sqlite")
    mgr = ManageJobs(jobs)
    jid = mgr.submit({}, tmp_path / "a.mp4", tmp_path / "b.mp4", "x")
    job_dir = tmp_path / "jobs" / jid
    job_dir.mkdir(parents=True)
    (job_dir / "note.txt").write_text("x", encoding="utf-8")
    mgr.delete(jid, tmp_path)
    assert jobs.get(jid) is None
    assert not job_dir.exists()


def test_progress_bridge_writes_heartbeat(tmp_path: Path):
    jobs = JobIndex(tmp_path / "j.sqlite")
    jobs.upsert("j1", "RUNNING")
    bridge = ProgressBridge(jobs, "j1")
    bridge.start("detect", total=10, detail="boxes")
    bridge.tick("detect", 4, 10, "frame 4")
    row = jobs.get("j1")
    progress = json.loads(row["progress_json"])
    assert progress["stage"] == "detect"
    assert 0 < progress["fraction"] < 1
    assert progress["detail"] == "frame 4"
    assert progress["heartbeat_at"]
    assert progress["started_at"]
    started = progress["started_at"]
    bridge.finish("detect", "done")
    progress = json.loads(jobs.get("j1")["progress_json"])
    assert progress["stage"] == "detect"
    assert progress["detail"] == "done"
    assert progress["started_at"] == started


def test_progress_bridge_raises_on_cancel(tmp_path: Path):
    jobs = JobIndex(tmp_path / "j.sqlite")
    jobs.upsert("j1", "RUNNING")
    jobs.request_cancel("j1")
    bridge = ProgressBridge(jobs, "j1")
    with pytest.raises(JobCancelled):
        bridge.start("inpaint")


def test_build_job_worker_wires_factory(tmp_path: Path):
    from videoclean.composition import build_job_worker

    jobs = JobIndex(tmp_path / "j.sqlite")
    worker = build_job_worker(tmp_path, jobs)
    assert isinstance(worker, JobWorker)
    assert worker.build_runner is not None
