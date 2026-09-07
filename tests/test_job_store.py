from pathlib import Path

from videoclean.store import SQLITE_TIMEOUT_S, JobIndex


def test_schema_has_queue_fields(tmp_path: Path):
    idx = JobIndex(tmp_path / "jobs.sqlite")
    idx.upsert("j1", "QUEUED", prompt="x", request={"device": "cuda"})
    row = idx.get("j1")
    assert row["state"] == "QUEUED"
    assert row["request_json"]
    assert row["updated_at"]


def test_mark_orphans_failed(tmp_path: Path):
    idx = JobIndex(tmp_path / "jobs.sqlite")
    idx.upsert("a", "RUNNING")
    idx.upsert("b", "QUEUED")
    n = idx.mark_orphans_failed("interrupted")
    assert n == 1
    assert idx.get("a")["state"] == "FAILED"
    assert idx.get("b")["state"] == "QUEUED"


def test_cancel_flag(tmp_path: Path):
    idx = JobIndex(tmp_path / "jobs.sqlite")
    idx.upsert("c", "RUNNING")
    assert idx.request_cancel("c") is True
    assert idx.is_cancel_requested("c") is True


def test_sqlite_wal_and_timeout(tmp_path: Path):
    idx = JobIndex(tmp_path / "jobs.sqlite")
    assert SQLITE_TIMEOUT_S > 0
    with idx._connect() as con:
        mode = con.execute("PRAGMA journal_mode").fetchone()[0]
        assert str(mode).lower() == "wal"
