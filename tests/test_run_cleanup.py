from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from videoclean.adapters.progress.silent import SilentProgress
from videoclean.application.config import PipelineConfig, RunCleanupRequest
from videoclean.application.errors import JobCancelled
from videoclean.application.use_cases.run_cleanup import RunCleanup
from videoclean.domain.intent import Intent, Target
from videoclean.domain.media import MediaManifest
from videoclean.domain.tracks import Track
from videoclean.store import JobPaths


class FakeMedia:
    name = "ffmpeg"

    def probe(self, path: Path) -> MediaManifest:
        return MediaManifest(
            path=path,
            duration_s=1.0,
            width=8,
            height=8,
            fps=1.0,
            fps_ratio="1/1",
            video_codec="h264",
            pix_fmt="yuv420p",
            frame_count=2,
            has_audio=False,
            audio_codec=None,
        )

    def extract_frames(self, src: Path, dest_dir: Path, log_file: Path) -> list[Path]:
        dest_dir.mkdir(parents=True, exist_ok=True)
        paths = [dest_dir / "frame_000001.png", dest_dir / "frame_000002.png"]
        for p in paths:
            p.write_bytes(b"x")
        return paths

    def encode_mezzanine(self, *args, **kwargs) -> None:
        dest: Path = args[2]
        dest.write_bytes(b"mezz")

    def package(self, src, dest: Path, fmt, **kwargs) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"out")
        return dest


class FakeParser:
    name = "llm"

    def status(self) -> str:
        return "ready (fake)"

    def parse(self, prompt: str | None, frame=None, frames=None, frame_indices=None, parse_chunk_frames=0) -> Intent:
        return Intent(targets=[Target(kind="object", query="mug")], raw=prompt or "")


class FakeDetector:
    name = "grounding-dino"

    def status(self) -> str:
        return "ready"

    def discover(self, frames, queries, on_progress=None):
        n = len(frames)
        box = (1, 1, 4, 4)
        if on_progress:
            on_progress(1, 1, "grounding-dino keyframe 1/1")
        return [Track(track_id=0, label="mug", boxes=[box] * n, scores=[1.0] * n)]


class FakeSegmenter:
    name = "sam2"

    def masks(self, frames, tracks):
        return [np.zeros(frames[0].shape[:2], dtype=np.uint8) + 255 for _ in frames]


class FakeInpainter:
    name = "opencv-telea"
    device_note = "CPU only"
    video_aware = False

    def inpaint(self, frame, mask):
        return frame

    def inpaint_clip(self, frames, masks):
        return [self.inpaint(f, m) for f, m in zip(frames, masks)]


class FakeJobs:
    def upsert(self, *args, **kwargs) -> None:
        return None


class RecordingJobs:
    def __init__(self) -> None:
        self.states: list[str] = []

    def upsert(self, job_id, state, **kwargs) -> None:
        self.states.append(state)


def test_run_cleanup_with_fakes(tmp_path: Path):
    src = tmp_path / "in.mp4"
    src.write_bytes(b"fake")
    out = tmp_path / "out.mp4"
    frame = np.zeros((8, 8, 3), dtype=np.uint8)

    uc = RunCleanup(
        media=FakeMedia(),
        parser=FakeParser(),
        detectors=[FakeDetector()],
        segmenter=FakeSegmenter(),
        inpainter=FakeInpainter(),
        jobs=FakeJobs(),
        progress=SilentProgress(),
        new_job_id=lambda: "job1",
        make_paths=JobPaths.create,
        utc_now=lambda: __import__("datetime").datetime(2026, 1, 1),
        read_image=lambda path: frame,
        write_image=lambda path, image: path.write_bytes(b"img"),
    )
    req = RunCleanupRequest(
        input_path=src,
        output_path=out,
        prompt="убери надписи",
        config=PipelineConfig(detectors=["grounding-dino"], formats=["mp4"], verify=False),
        overwrite=True,
        keep_workdir=True,
    )
    report = uc.execute(req, tmp_path)
    assert report["state"] == "COMPLETED"
    assert report["detectorUsed"] == "grounding-dino"
    assert report["segmenter"] == "sam2"
    assert report["inpainter"] == "opencv-telea"
    assert report["media"] == "ffmpeg"
    assert out.is_file()


class BoomDetector:
    name = "grounding-dino"

    def status(self) -> str:
        return "ready"

    def discover(self, frames, queries, on_progress=None):
        raise RuntimeError("The size of tensor a (25) must match the size of tensor b (16)")


def test_discover_falls_back_when_first_detector_raises(tmp_path: Path):
    src = tmp_path / "in.mp4"
    src.write_bytes(b"fake")
    out = tmp_path / "out.mp4"
    frame = np.zeros((8, 8, 3), dtype=np.uint8)

    uc = RunCleanup(
        media=FakeMedia(),
        parser=FakeParser(),
        detectors=[BoomDetector(), FakeDetector()],
        segmenter=FakeSegmenter(),
        inpainter=FakeInpainter(),
        jobs=FakeJobs(),
        progress=SilentProgress(),
        new_job_id=lambda: "job2",
        make_paths=JobPaths.create,
        utc_now=lambda: __import__("datetime").datetime(2026, 1, 1),
        read_image=lambda path: frame,
        write_image=lambda path, image: path.write_bytes(b"img"),
    )
    req = RunCleanupRequest(
        input_path=src,
        output_path=out,
        prompt="убери вотермарку и текстовые overlay",
        config=PipelineConfig(detectors=["grounding-dino", "grounding-dino"], formats=["mp4"], verify=False),
        overwrite=True,
        keep_workdir=True,
    )
    report = uc.execute(req, tmp_path)
    assert report["state"] == "COMPLETED"
    assert report["detectorUsed"] == "grounding-dino"
    assert any("grounding-dino: error" in a for a in report["detectorAttempts"])


class CancelDetector:
    name = "grounding-dino"

    def status(self) -> str:
        return "ready"

    def discover(self, frames, queries, on_progress=None):
        raise JobCancelled("job-cancel")


def test_execute_upserts_cancelled_not_failed(tmp_path: Path):
    src = tmp_path / "in.mp4"
    src.write_bytes(b"fake")
    out = tmp_path / "out.mp4"
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    jobs = RecordingJobs()
    uc = RunCleanup(
        media=FakeMedia(),
        parser=FakeParser(),
        detectors=[CancelDetector()],
        segmenter=FakeSegmenter(),
        inpainter=FakeInpainter(),
        jobs=jobs,
        progress=SilentProgress(),
        new_job_id=lambda: "job-cancel",
        make_paths=JobPaths.create,
        utc_now=lambda: __import__("datetime").datetime(2026, 1, 1),
        read_image=lambda path: frame,
        write_image=lambda path, image: path.write_bytes(b"img"),
    )
    req = RunCleanupRequest(
        input_path=src,
        output_path=out,
        prompt="убери надписи",
        config=PipelineConfig(detectors=["grounding-dino"], formats=["mp4"], verify=False),
        overwrite=True,
        keep_workdir=True,
    )
    with pytest.raises(JobCancelled):
        uc.execute(req, tmp_path)
    assert "FAILED" not in jobs.states
    assert jobs.states[-1] == "CANCELLED"


def test_discover_reraises_job_cancelled():
    uc = RunCleanup(
        media=FakeMedia(),
        parser=FakeParser(),
        detectors=[CancelDetector(), FakeDetector()],
        segmenter=FakeSegmenter(),
        inpainter=FakeInpainter(),
        jobs=FakeJobs(),
        progress=SilentProgress(),
        new_job_id=lambda: "job-cancel",
        make_paths=JobPaths.create,
        utc_now=lambda: __import__("datetime").datetime(2026, 1, 1),
        read_image=lambda path: None,
        write_image=lambda path, image: None,
    )
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    try:
        uc._discover([frame], ["mug"], stage="detect")
    except JobCancelled:
        return
    raise AssertionError("expected JobCancelled to propagate from _discover")


class CountingDetector:
    name = "grounding-dino"

    def __init__(self):
        self.calls = 0

    def status(self) -> str:
        return "ready"

    def discover(self, frames, queries, on_progress=None):
        self.calls += 1
        n = len(frames)
        return [
            Track(
                track_id=0,
                label="mug",
                boxes=[(1, 1, 4, 4)] * n,
                scores=[1.0] * n,
                notes=["open-vocab"],
            )
        ]


class BoxedParser:
    name = "llm"

    def parse(self, prompt: str | None, frame=None, frames=None, frame_indices=None, parse_chunk_frames=0):
        return Intent(
            targets=[Target(kind="object", query="mug", box=(0.1, 0.1, 0.4, 0.4))],
            parse_mode="llm",
            raw=prompt or "",
        )


def test_parser_boxes_do_not_skip_the_detector(tmp_path: Path):
    src = tmp_path / "in.mp4"
    src.write_bytes(b"fake")
    out = tmp_path / "out.mp4"
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    det = CountingDetector()
    uc = RunCleanup(
        media=FakeMedia(),
        parser=BoxedParser(),
        detectors=[det],
        segmenter=FakeSegmenter(),
        inpainter=FakeInpainter(),
        jobs=FakeJobs(),
        progress=SilentProgress(),
        new_job_id=lambda: "job3",
        make_paths=JobPaths.create,
        utc_now=lambda: __import__("datetime").datetime(2026, 1, 1),
        read_image=lambda path: frame,
        write_image=lambda path, image: path.write_bytes(b"img"),
    )
    req = RunCleanupRequest(
        input_path=src,
        output_path=out,
        prompt="убери кружку",
        config=PipelineConfig(detectors=["grounding-dino"], formats=["mp4"], verify=False),
        overwrite=True,
        keep_workdir=True,
    )
    report = uc.execute(req, tmp_path)
    assert det.calls == 1
    assert report["detectorUsed"] == "grounding-dino"


def test_empty_prompt_fails_before_detect(tmp_path: Path):
    from videoclean.application.errors import PipelineError

    src = tmp_path / "in.mp4"
    src.write_bytes(b"fake")
    uc = RunCleanup(
        media=FakeMedia(),
        parser=FakeParser(),
        detectors=[FakeDetector()],
        segmenter=FakeSegmenter(),
        inpainter=FakeInpainter(),
        jobs=FakeJobs(),
        progress=SilentProgress(),
        new_job_id=lambda: "job-empty",
        make_paths=JobPaths.create,
        utc_now=lambda: __import__("datetime").datetime(2026, 1, 1),
        read_image=lambda path: np.zeros((8, 8, 3), dtype=np.uint8),
        write_image=lambda path, image: path.write_bytes(b"img"),
    )
    req = RunCleanupRequest(
        input_path=src,
        output_path=tmp_path / "out.mp4",
        prompt="  ",
        config=PipelineConfig(detectors=["grounding-dino"], formats=["mp4"], verify=False),
        overwrite=True,
    )
    try:
        uc.execute(req, tmp_path)
    except PipelineError as exc:
        assert "--prompt" in str(exc)
    else:
        raise AssertionError("expected PipelineError")


class OrderMedia(FakeMedia):
    def __init__(self):
        self.events: list[str] = []

    def probe(self, path: Path) -> MediaManifest:
        self.events.append("probe")
        return super().probe(path)

    def extract_frames(self, src: Path, dest_dir: Path, log_file: Path) -> list[Path]:
        self.events.append("extract")
        return super().extract_frames(src, dest_dir, log_file)


class OrderParser(FakeParser):
    def __init__(self, events: list[str]):
        self.events = events

    def parse(self, prompt: str | None, frame=None, frames=None, frame_indices=None, parse_chunk_frames=0) -> Intent:
        self.events.append("parse")
        return super().parse(prompt, frame)


def _uc(tmp_path: Path, media, parser, detectors=None):
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    return RunCleanup(
        media=media,
        parser=parser,
        detectors=detectors or [FakeDetector()],
        segmenter=FakeSegmenter(),
        inpainter=FakeInpainter(),
        jobs=FakeJobs(),
        progress=SilentProgress(),
        new_job_id=lambda: "job-order",
        make_paths=JobPaths.create,
        utc_now=lambda: __import__("datetime").datetime(2026, 1, 1),
        read_image=lambda path: frame,
        write_image=lambda path, image: path.write_bytes(b"img"),
    )


def test_extract_runs_before_parse_for_vision_and_input_is_not_copied(tmp_path: Path):
    src = tmp_path / "in.mp4"
    src.write_bytes(b"fake-video")
    media = OrderMedia()
    parser = OrderParser(media.events)
    uc = _uc(tmp_path, media, parser)
    req = RunCleanupRequest(
        input_path=src,
        output_path=tmp_path / "out.mp4",
        prompt="убери надписи",
        config=PipelineConfig(detectors=["grounding-dino"], formats=["mp4"], verify=False),
        overwrite=True,
        keep_workdir=True,
        manifest=media.probe(src),
    )
    media.events.clear()
    report = uc.execute(req, tmp_path)
    # Vision parse needs frames on disk first.
    assert media.events.index("extract") < media.events.index("parse")
    assert "probe" not in media.events
    copied = tmp_path / "jobs" / "job-order" / "input" / "in.mp4"
    assert not copied.exists()
    assert report["state"] == "COMPLETED"


def test_unavailable_parser_fails_in_ensure_ready_before_extract(tmp_path: Path):
    from videoclean.application.errors import AdapterUnavailable

    class DeadParser(FakeParser):
        def status(self) -> str:
            return "unavailable: llm unreachable at http://127.0.0.1:11434/v1/models"

    src = tmp_path / "in.mp4"
    src.write_bytes(b"fake")
    media = OrderMedia()
    uc = _uc(tmp_path, media, DeadParser())
    req = RunCleanupRequest(
        input_path=src,
        output_path=tmp_path / "out.mp4",
        prompt="убери надписи",
        config=PipelineConfig(detectors=["grounding-dino"], formats=["mp4"], verify=False),
        overwrite=True,
        manifest=media.probe(src),
    )
    media.events.clear()
    try:
        uc.execute(req, tmp_path)
    except AdapterUnavailable as exc:
        assert "llm unreachable" in str(exc)
    else:
        raise AssertionError("expected AdapterUnavailable")
    # status() is checked in _ensure_ready before ffmpeg extract
    assert "extract" not in media.events


def test_targets_override_skips_parser(tmp_path: Path):
    from videoclean.domain.tracks import Track

    calls = []
    parser = FakeParser()
    orig_parse = parser.parse

    def spy_parse(*a, **kw):
        calls.append(a)
        return orig_parse(*a, **kw)

    parser.parse = spy_parse
    det = FakeDetector()
    jobs = FakeJobs()
    src = tmp_path / "in.mp4"
    src.write_bytes(b"fake")
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    uc = RunCleanup(
        media=FakeMedia(),
        parser=parser,
        detectors=[det],
        segmenter=FakeSegmenter(),
        inpainter=FakeInpainter(),
        jobs=jobs,
        progress=SilentProgress(),
        new_job_id=lambda: "j-override",
        make_paths=JobPaths.create,
        utc_now=lambda: __import__("datetime").datetime(2026, 1, 1),
        read_image=lambda path: frame,
        write_image=lambda path, image: path.write_bytes(b"img"),
    )
    req = RunCleanupRequest(
        input_path=src,
        output_path=tmp_path / "out.mp4",
        prompt="",
        config=PipelineConfig(detectors=["grounding-dino"], formats=["mp4"], verify=False),
        overwrite=True,
        keep_workdir=True,
        job_id="j-override",
        targets_override=[{"kind": "object", "query": "mug", "where": None, "motion": "any"}],
    )
    report = uc.execute(req, tmp_path)
    assert calls == [], "parser must not run when targets_override is given"
    assert report["promptParseMode"] == "manual"
    assert report["targets"][0]["query"] == "mug"
