"""SourceIndex registry: sources survive restarts, delete removes the row."""
from pathlib import Path

from videoclean.store import SourceIndex, new_source_id


def test_register_get_list_delete(tmp_path: Path):
    idx = SourceIndex(tmp_path / "jobs.sqlite")
    sid = new_source_id()
    idx.register(sid, "clip.mp4", "/tmp/x/clip.mp4", probe={"fps": 25.0, "frame_count": 100})
    row = idx.get(sid)
    assert row is not None
    assert row["name"] == "clip.mp4"
    assert row["path"] == "/tmp/x/clip.mp4"
    import json
    probe = json.loads(row["probe_json"])
    assert probe["frame_count"] == 100
    assert len(idx.list()) == 1
    assert idx.delete(sid) is True
    assert idx.get(sid) is None
    assert idx.delete(sid) is False


def test_missing_source_returns_none(tmp_path: Path):
    idx = SourceIndex(tmp_path / "jobs.sqlite")
    assert idx.get("s_missing") is None


def test_new_source_id_format():
    sid = new_source_id()
    assert sid.startswith("s_") and len(sid.split("_")) == 4
