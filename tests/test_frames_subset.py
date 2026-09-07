from pathlib import Path

import pytest

from videoclean.adapters.media.ffmpeg import FFmpegMedia


class FakeRun:
    """Records the ffmpeg command list passed by the adapter."""

    def __init__(self):
        self.calls: list[list[str]] = []

    def __call__(self, cmd: list[str], log_file: Path) -> None:
        self.calls.append(cmd)
        # Simulate real ffmpeg: selected outputs are numbered sequentially.
        out = Path(cmd[-1])
        out.parent.mkdir(parents=True, exist_ok=True)
        vf = cmd[cmd.index("-vf") + 1]
        inner = vf.split("select=")[1].strip("'")
        idxs = [int(tok) for tok in inner.replace("eq(n,", "").replace(")", "").split("+") if tok]
        for k in range(len(idxs)):
            (out.parent / f"frame_{k + 1:06d}.jpg").write_bytes(b"\xff\xd8fake")


@pytest.fixture
def fake_run(monkeypatch):
    fr = FakeRun()
    monkeypatch.setattr("videoclean.adapters.media.ffmpeg._run", fr)
    return fr


def test_subset_extracts_only_requested_indices(fake_run, tmp_path):
    media = FFmpegMedia()
    out = media.extract_frames_subset(
        Path("in.mp4"), [0, 4, 8], tmp_path / "frames", tmp_path / "log.txt"
    )
    names = [p.name for p in out]
    assert names == ["frame_000000.jpg", "frame_000004.jpg", "frame_000008.jpg"]
    vf = fake_run.calls[0][fake_run.calls[0].index("-vf") + 1]
    assert "eq(n,0)" in vf and "eq(n,4)" in vf and "eq(n,8)" in vf


def test_subset_sorts_and_dedups(fake_run, tmp_path):
    media = FFmpegMedia()
    out = media.extract_frames_subset(
        Path("in.mp4"), [8, 0, 4, 4, 0], tmp_path / "frames", tmp_path / "log.txt"
    )
    assert [p.stem for p in out] == ["frame_000000", "frame_000004", "frame_000008"]


def test_subset_empty_indices_is_error(tmp_path):
    media = FFmpegMedia()
    from videoclean.application.errors import PipelineError

    try:
        media.extract_frames_subset(Path("in.mp4"), [], tmp_path / "f", tmp_path / "log.txt")
    except PipelineError as exc:
        assert "no frames" in str(exc).lower() or "indices" in str(exc).lower()
    else:
        raise AssertionError("expected PipelineError")


def test_subset_missing_output_raises(monkeypatch, tmp_path):
    from videoclean.adapters.media import ffmpeg as ff

    def run_no_output(cmd, log_file):
        pass

    monkeypatch.setattr(ff, "_run", run_no_output)
    media = FFmpegMedia()
    from videoclean.application.errors import PipelineError

    try:
        media.extract_frames_subset(Path("in.mp4"), [3], tmp_path / "f", tmp_path / "log.txt")
    except PipelineError as exc:
        assert "no frames" in str(exc).lower()
    else:
        raise AssertionError("expected PipelineError")


def test_subset_rename_survives_overlapping_names(fake_run, tmp_path):
    """ffmpeg names outputs sequentially; rename must not clobber (2→4 while 4 exists)."""
    media = FFmpegMedia()
    out = media.extract_frames_subset(
        Path("in.mp4"), [0, 4, 8, 12], tmp_path / "frames", tmp_path / "log.txt"
    )
    assert [p.name for p in out] == [
        "frame_000000.jpg", "frame_000004.jpg", "frame_000008.jpg", "frame_000012.jpg"
    ]
    for p in out:
        assert p.is_file() and p.stat().st_size > 0
