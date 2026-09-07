"""RunPreview: detect + masks on a frame subset, no inpaint/encode."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from videoclean.application.config import PipelineConfig
from videoclean.application.errors import PipelineError
from videoclean.application.use_cases.run_preview import RunPreview, PreviewRequest
from videoclean.domain.intent import Intent, Target
from videoclean.domain.media import MediaManifest
from videoclean.domain.tracks import Track


class FakeMedia:
    name = "fake-ffmpeg"

    def __init__(self, n_frames: int = 16):
        self.n = n_frames
        self.subsets: list[list[int]] = []

    def probe(self, path: Path) -> MediaManifest:
        return MediaManifest(
            path=path,
            duration_s=self.n / 8.0,
            width=64,
            height=64,
            fps=8.0,
            fps_ratio="8/1",
            video_codec="h264",
            pix_fmt="yuv420p",
            frame_count=self.n,
            has_audio=False,
            audio_codec=None,
        )

    def extract_frames_subset(self, src, indices, dest_dir: Path, log_file):
        self.subsets.append(list(indices))
        dest_dir.mkdir(parents=True, exist_ok=True)
        out = []
        for i in sorted(set(indices)):
            p = dest_dir / f"frame_{i:06d}.jpg"
            p.write_bytes(b"jpg")
            out.append(p)
        return out


class FakeParser:
    name = "llm"

    def __init__(self, intent: Intent | None = None):
        self.intent = intent or Intent(targets=[Target(kind="text_overlay", query="text")])
        self.calls: list[dict] = []

    def status(self) -> str:
        return "ready (fake)"

    def parse(self, prompt, frame=None, frames=None, frame_indices=None, parse_chunk_frames=0):
        self.calls.append({"prompt": prompt, "n": len(frames or []), "idx": frame_indices})
        return self.intent


class FakeDetector:
    name = "grounding-dino"

    def __init__(self, boxes: dict[int, tuple] | None = None):
        # keyed by position within the preview frame list
        self.boxes = boxes or {}
        self.calls: list[int] = []

    def status(self) -> str:
        return "ready (fake)"

    def discover(self, frames, queries, on_progress=None):
        self.calls.append(len(frames))
        tracks = []
        for i in range(len(frames)):
            box = self.boxes.get(i)
            if box is None:
                continue
            boxes = [None] * len(frames)
            boxes[i] = box
            tr = Track(track_id=100 + i, label="object", boxes=boxes, scores=[0.9] * len(frames))
            tr.motion = "static"
            tracks.append(tr)
        return tracks


class FakeSegmenter:
    name = "sam2"

    def __init__(self, coverage: float = 0.02):
        self.coverage = coverage
        self.calls = 0

    def masks(self, frames, tracks):
        self.calls += 1
        n = frames[0].shape[0], frames[0].shape[1]
        m = np.zeros(n, dtype=np.uint8)
        m[10:30, 10:30] = 255
        return [m.copy() for _ in frames]


class Paths:
    def __init__(self, root: Path):
        self.root = root
        self.frames_dir = root / "frames"
        self.preview_dir = root / "preview"
        self.logs_dir = root / "logs"
        self.ffmpeg_log = root / "logs" / "ffmpeg.log"
        self.report_file = root / "output" / "report.json"
        self.input_dir = root / "input"


def _make_paths(root: Path) -> Paths:
    root.mkdir(parents=True, exist_ok=True)
    return Paths(root)


class SilentProgress:
    def start(self, *a, **k): ...

    def tick(self, *a, **k): ...

    def finish(self, *a, **k): ...


def _build(tmp_path: Path, n_frames: int = 16):
    media = FakeMedia(n_frames)
    parser = FakeParser()
    detector = FakeDetector(boxes={i: (10, 10, 40, 40) for i in range(n_frames)})
    segmenter = FakeSegmenter()
    uc = RunPreview(
        media=media,
        parser=parser,
        detectors=[detector],
        segmenter=segmenter,
        jobs=_FakeJobs(),
        progress=SilentProgress(),
        new_job_id=lambda: "p1",
        make_paths=_make_paths,
        read_image=lambda p: np.zeros((64, 64, 3), dtype=np.uint8),
        write_image=lambda p, img: Path(p).parent.mkdir(parents=True, exist_ok=True) or Path(p).write_bytes(b"img"),
    )
    return uc, media, parser, detector, segmenter


class _FakeJobs:
    def __init__(self):
        self.rows: dict = {}

    def upsert(self, job_id, state, **kw):
        self.rows[job_id] = {"state": state, **kw}


def _req(tmp_path: Path, **kw) -> PreviewRequest:
    src = tmp_path / "in.mp4"
    src.write_bytes(b"video")
    defaults = dict(
        input_path=src,
        prompt="удали текст",
        config=PipelineConfig(device="cpu", detectors=["grounding-dino"], formats=["mp4"]),
        start=0,
        count=8,
        stride=2,
        mode="parse",
        targets=None,
        job_id=None,
    )
    defaults.update(kw)
    return PreviewRequest(**defaults)


def test_preview_happy_path(tmp_path: Path):
    uc, media, parser, detector, segmenter = _build(tmp_path)
    req = _req(tmp_path)
    report = uc.execute(req, tmp_path / "data")
    assert report["state"] == "COMPLETED"
    # 8 frames requested: 0,2,4,...,14
    assert media.subsets == [[0, 2, 4, 6, 8, 10, 12, 14]]
    # detector saw every preview frame
    assert detector.calls and detector.calls[0] == 8
    # parser was called with vision frames
    assert parser.calls and parser.calls[0]["n"] == 8
    # artifacts
    wdir = tmp_path / "data" / "jobs" / "p1"
    assert (wdir / "preview" / "preview.json").is_file()
    assert (wdir / "preview" / "000000_boxes.jpg").is_file()
    assert (wdir / "preview" / "000000_mask.jpg").is_file()
    pj = json.loads((wdir / "preview" / "preview.json").read_text())
    assert pj["targets"][0]["query"] == "text"
    assert len(pj["frames"]) == 8
    assert all(f["maskCoverage"] > 0 for f in pj["frames"])


def test_preview_detect_mode_skips_parser(tmp_path: Path):
    uc, media, parser, detector, segmenter = _build(tmp_path)
    targets = [{"kind": "text_overlay", "query": "text", "where": None, "motion": "any"}]
    req = _req(tmp_path, mode="detect", targets=targets)
    report = uc.execute(req, tmp_path / "data")
    assert report["state"] == "COMPLETED"
    assert parser.calls == []
    assert report["targets"][0]["query"] == "text"


def test_preview_detect_mode_requires_targets(tmp_path: Path):
    uc, *_ = _build(tmp_path)
    with pytest.raises(PipelineError):
        uc.execute(_req(tmp_path, mode="detect", targets=None), tmp_path / "data")


def test_preview_no_detections_is_error(tmp_path: Path):
    uc, media, parser, detector, segmenter = _build(tmp_path)
    detector.boxes = {}
    with pytest.raises(PipelineError):
        uc.execute(_req(tmp_path), tmp_path / "data")


def test_preview_frame_selection_clamps_to_video(tmp_path: Path):
    uc, media, *_ = _build(tmp_path, n_frames=10)
    uc.execute(_req(tmp_path, start=8, count=8, stride=2), tmp_path / "data")
    assert media.subsets[0] == [8]


def test_preview_explicit_indices(tmp_path: Path):
    uc, media, *_ = _build(tmp_path)
    req = _req(tmp_path, start=None, count=None, stride=None, indices=[3, 7, 7, 15])
    uc.execute(req, tmp_path / "data")
    assert media.subsets == [[3, 7, 15]]
