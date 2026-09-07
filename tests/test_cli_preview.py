"""CLI preview command: --frames selection and --queries override."""
from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

import videoclean.cli as cli


class FakeMedia:
    name = "fake"

    def probe(self, path):
        return SimpleNamespace(
            path=path, duration_s=2.0, width=64, height=64, fps=8.0, fps_ratio="8/1",
            video_codec="h264", pix_fmt="yuv420p", frame_count=16, has_audio=False, audio_codec=None,
        )

    def extract_frames_subset(self, src, indices, dest_dir, log_file):
        dest_dir.mkdir(parents=True, exist_ok=True)
        out = []
        for i in sorted(set(indices)):
            p = dest_dir / f"frame_{i:06d}.jpg"
            p.write_bytes(b"jpg")
            out.append(p)
        return out


class FakePreview:
    last_req = None

    def execute(self, req, data_dir):
        FakePreview.last_req = req
        return {"state": "COMPLETED", "frames": []}


def _patch(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "FFmpegMedia", lambda: FakeMedia())

    def fake_build(cfg, progress, jobs=None, job_id=None):
        return FakePreview()

    monkeypatch.setattr(cli, "build_run_preview", fake_build)


def test_preview_command_passes_frame_window(monkeypatch, tmp_path: Path):
    _patch(monkeypatch, tmp_path)
    video = tmp_path / "in.mp4"
    video.write_bytes(b"v")
    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        [
            "preview",
            "--input", str(video),
            "--output", str(tmp_path / "out"),
            "--prompt", "удали текст",
            "--frames", "0:8:2",
            "--llm", "local", "--llm-model", "llava-phi3",
        ],
    )
    assert result.exit_code == 0, result.output
    req = FakePreview.last_req
    assert (req.start, req.count, req.stride) == (0, 8, 2)
    assert req.mode == "parse"


def test_preview_explicit_indices(monkeypatch, tmp_path: Path):
    _patch(monkeypatch, tmp_path)
    video = tmp_path / "in.mp4"
    video.write_bytes(b"v")
    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        ["preview", "--input", str(video), "--output", str(tmp_path / "out"),
         "--prompt", "x", "--frames", "3,7,15", "--llm", "local", "--llm-model", "llava-phi3"],
    )
    assert result.exit_code == 0, result.output
    assert FakePreview.last_req.indices == [3, 7, 15]


def test_preview_with_queries_only_no_prompt(monkeypatch, tmp_path: Path):
    _patch(monkeypatch, tmp_path)
    video = tmp_path / "in.mp4"
    video.write_bytes(b"v")
    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        ["preview", "--input", str(video), "--output", str(tmp_path / "out"),
         "--queries", "text [bottom], logo"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    req = FakePreview.last_req
    assert req.mode == "detect"
    queries = [t["query"] for t in req.targets]
    assert queries == ["text", "logo"]
    assert req.targets[0]["where"] == "bottom"
    assert req.targets[1]["kind"] == "watermark"


def test_queries_parser_syntax():
    from videoclean.application.use_cases.run_preview import parse_queries_arg

    targets = parse_queries_arg("text [bottom], logo, red mug")
    assert [(t.query, t.kind, t.where) for t in targets] == [
        ("text", "text_overlay", "bottom"),
        ("logo", "watermark", None),
        ("red mug", "object", None),
    ]
