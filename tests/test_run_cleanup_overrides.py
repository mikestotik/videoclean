"""Manual entry points into RunCleanup: tracks_override and masks_override."""
from pathlib import Path

import cv2
import numpy as np

from tests.test_run_cleanup import (
    FakeDetector, FakeInpainter, FakeJobs, FakeMedia, FakeParser, FakeSegmenter,
)
from videoclean.adapters.progress.silent import SilentProgress
from videoclean.application.config import PipelineConfig, RunCleanupRequest
from videoclean.application.errors import PipelineError
from videoclean.application.use_cases.run_cleanup import RunCleanup
from videoclean.store import JobPaths


def _uc(tmp_path: Path, detector=None, segmenter=None, parser=None, progress=None) -> RunCleanup:
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    return RunCleanup(
        media=FakeMedia(),
        parser=parser or FakeParser(),
        detectors=[detector] if detector else [],
        segmenter=segmenter or FakeSegmenter(),
        inpainter=FakeInpainter(),
        jobs=FakeJobs(),
        progress=progress or SilentProgress(),
        new_job_id=lambda: "j-manual",
        make_paths=JobPaths.create,
        utc_now=lambda: __import__("datetime").datetime(2026, 1, 1),
        read_image=lambda path: frame,
        write_image=lambda path, image: path.write_bytes(b"img"),
    )


class RecordingProgress(SilentProgress):
    def __init__(self):
        self.finishes: dict[str, str] = {}

    def finish(self, key, detail=""):
        self.finishes[key] = detail


class CountingDetector(FakeDetector):
    def __init__(self):
        self.calls = 0

    def discover(self, frames, queries, on_progress=None):
        self.calls += 1
        return []


class CountingParser(FakeParser):
    def __init__(self):
        self.calls = 0

    def parse(self, *args, **kwargs):
        self.calls += 1
        return super().parse(*args, **kwargs)


class CountingSegmenter(FakeSegmenter):
    def __init__(self):
        self.calls = 0

    def masks(self, frames, tracks):
        self.calls += 1
        return super().masks(frames, tracks)


def _req(tmp_path: Path, **overrides) -> RunCleanupRequest:
    base = dict(
        input_path=tmp_path / "in.mp4",
        output_path=tmp_path / "out.mp4",
        prompt="",
        config=PipelineConfig(detectors=["grounding-dino"], formats=["mp4"], verify=False),
        overwrite=True,
        keep_workdir=True,
        job_id="j-manual",
    )
    base.update(overrides)
    return RunCleanupRequest(**base)


def _write_mask(tmp_path: Path, name: str, w: int = 8, h: int = 8) -> Path:
    m = np.zeros((h, w), dtype=np.uint8)
    m[1:4, 1:4] = 255
    p = tmp_path / name
    cv2.imwrite(str(p), m)
    return p


def test_tracks_override_skips_parse_and_detect(tmp_path: Path):
    src = tmp_path / "in.mp4"
    src.write_bytes(b"fake")
    out = tmp_path / "out.mp4"
    det, par, seg = CountingDetector(), CountingParser(), CountingSegmenter()
    uc = _uc(tmp_path, detector=det, segmenter=seg, parser=par)
    report = uc.execute(_req(tmp_path, tracks_override=[
        {"id": 0, "label": "logo", "motion": "static", "boxes": [[1, 1, 4, 4], [1, 1, 4, 4]]},
    ]), tmp_path)
    assert report["state"] == "COMPLETED"
    assert par.calls == 0 and det.calls == 0, "parse+detect must be skipped"
    assert seg.calls == 1, "segmenter still runs on the given tracks"
    assert report["promptParseMode"] == "manual-tracks"
    assert report["detectorUsed"] == "manual"


def test_masks_override_skips_parse_detect_segment(tmp_path: Path):
    src = tmp_path / "in.mp4"
    src.write_bytes(b"fake")
    out = tmp_path / "out.mp4"
    det, par, seg = CountingDetector(), CountingParser(), CountingSegmenter()
    mask = _write_mask(tmp_path, "m.png")
    uc = _uc(tmp_path, detector=det, segmenter=seg, parser=par)
    report = uc.execute(_req(tmp_path, masks_override=[str(mask)]), tmp_path)
    assert report["state"] == "COMPLETED"
    assert par.calls == 0 and det.calls == 0 and seg.calls == 0
    assert report["promptParseMode"] == "manual-masks"
    assert report["detectorUsed"] == "manual"
    masks_dir = tmp_path / "jobs" / "j-manual" / "masks" / "processed"
    assert len(list(masks_dir.iterdir())) == 2, "mask tiled to every frame"


def test_manual_overrides_satisfy_prompt_check(tmp_path: Path):
    src = tmp_path / "in.mp4"
    src.write_bytes(b"fake")
    det = CountingDetector()
    uc = _uc(tmp_path, detector=det)
    try:
        uc.execute(_req(tmp_path), tmp_path)
    except PipelineError as exc:
        assert "--prompt" in str(exc)
    else:
        raise AssertionError("expected PipelineError without prompt or overrides")


def _cfg(verify: bool = False) -> PipelineConfig:
    return PipelineConfig(detectors=["grounding-dino"], formats=["mp4"], verify=verify)


def test_masks_override_with_default_verify_skips_verify(tmp_path: Path):
    src = tmp_path / "in.mp4"
    src.write_bytes(b"fake")
    det = CountingDetector()
    mask = _write_mask(tmp_path, "m.png")
    prog = RecordingProgress()
    uc = _uc(tmp_path, detector=det, progress=prog)
    report = uc.execute(_req(tmp_path, config=_cfg(verify=True), masks_override=[str(mask)]), tmp_path)
    assert report["state"] == "COMPLETED"
    assert det.calls == 0, "verify must not re-detect when masks are manual"
    assert report["verify"] is True and report["verifyPasses"] == 0
    assert "skipped" in prog.finishes.get("verify", "")


def test_tracks_override_with_verify_runs_verify_and_stays_clean(tmp_path: Path):
    src = tmp_path / "in.mp4"
    src.write_bytes(b"fake")
    det = CountingDetector()
    prog = RecordingProgress()
    uc = _uc(tmp_path, detector=det, progress=prog)
    report = uc.execute(_req(tmp_path, config=_cfg(verify=True), tracks_override=[
        {"id": 0, "label": "logo", "motion": "static", "boxes": [[1, 1, 4, 4], [1, 1, 4, 4]]},
    ]), tmp_path)
    assert report["state"] == "COMPLETED"
    assert det.calls == 1, "verify re-detects once on inpainted frames"
    assert report["verify"] is True and report["verifyPasses"] == 0
    assert prog.finishes.get("verify", "").startswith("clean")


def test_unusable_tracks_raise(tmp_path: Path):
    src = tmp_path / "in.mp4"
    src.write_bytes(b"fake")
    uc = _uc(tmp_path)
    try:
        uc.execute(_req(tmp_path, tracks_override=[{"id": 0, "label": "x", "boxes": []}]), tmp_path)
    except PipelineError:
        pass
    else:
        raise AssertionError("expected PipelineError for unusable tracks")
