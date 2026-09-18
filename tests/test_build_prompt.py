# tests/test_build_prompt.py
"""BuildPrompt: frames + drawn masks → VLM → editable prompt + detector targets."""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from videoclean.adapters.progress.silent import SilentProgress
from videoclean.application.config import PipelineConfig
from videoclean.application.errors import AdapterUnavailable, PipelineError
from videoclean.application.use_cases.build_prompt import BuildPrompt, BuildPromptRequest, PromptPaths
from videoclean.domain.media import MediaManifest
from videoclean.store import JobPaths

W, H = 8, 8


class FakeMedia:
    name = "ffmpeg"

    def __init__(self, n: int = 5):
        self.n = n

    def probe(self, path: Path) -> MediaManifest:
        return MediaManifest(
            path=path, duration_s=1.0, width=W, height=H, fps=5.0, fps_ratio="5/1",
            video_codec="h264", pix_fmt="yuv420p", frame_count=self.n, has_audio=False, audio_codec=None,
        )

    def extract_frames_subset(self, src, indices, dest_dir, log_file) -> list[Path]:
        dest_dir.mkdir(parents=True, exist_ok=True)
        out = []
        for i in indices:
            p = dest_dir / f"{i:06d}.jpg"
            cv2.imwrite(str(p), np.zeros((H, W, 3), dtype=np.uint8))
            out.append(p)
        return out


class FakeLlm:
    name = "fake"
    model = "fake-vision"

    def __init__(self, reply: str):
        self.reply = reply
        self.calls: list[dict] = []

    def status(self) -> str:
        return "ready (fake)"

    def complete(self, system, user, image_jpeg=None, images=None):
        self.calls.append({"system": system, "user": user, "images": images})
        return self.reply


REPLY = (
    'Прекрасный выбор! {"prompt": "убери логотип в правом нижнем углу", '
    '"targets": [{"kind": "watermark", "query": "channel logo", "where": "bottom-right", "motion": "static"}]}'
)


def _build_prompt(tmp_path: Path, llm: FakeLlm, make_paths=JobPaths.create) -> BuildPrompt:
    frame = np.zeros((H, W, 3), dtype=np.uint8)

    def to_jpeg(img) -> bytes:
        ok, buf = cv2.imencode(".jpg", img)
        assert ok
        return buf.tobytes()

    return BuildPrompt(
        media=FakeMedia(),
        llm=llm,
        system_prompt="SYSTEM",
        jobs=FakeJobs(),
        progress=SilentProgress(),
        new_job_id=lambda: "j-prompt",
        make_paths=make_paths,
        read_image=lambda path: frame,
        to_jpeg=to_jpeg,
    )


class FakeJobs:
    def upsert(self, *args, **kwargs) -> None:
        return None


def _write_mask(tmp_path: Path, name: str) -> Path:
    m = np.zeros((H, W), dtype=np.uint8)
    m[1:4, 1:4] = 255
    p = tmp_path / name
    cv2.imwrite(str(p), m)
    return p


def _req(tmp_path: Path, prompt: str = "", annotations=None) -> BuildPromptRequest:
    return BuildPromptRequest(
        input_path=tmp_path / "in.mp4",
        prompt=prompt,
        annotations=annotations or [],
        config=PipelineConfig(prompt_frame_stride=4, prompt_frame_max=8),
        job_id="j-prompt",
    )


def test_masks_and_text_are_interpreted(tmp_path: Path):
    src = tmp_path / "in.mp4"
    src.write_bytes(b"fake")
    mask = _write_mask(tmp_path, "m.png")
    llm = FakeLlm(REPLY)
    out = _build_prompt(tmp_path, llm).execute(_req(tmp_path, prompt="убери это", annotations=[
        {"frame": 3, "mask": str(mask)},
    ]), tmp_path)
    assert out["state"] == "COMPLETED"
    assert out["prompt"] == "убери логотип в правом нижнем углу"
    assert out["targets"][0]["query"] == "channel logo"
    assert out["framesUsed"] == [3]
    assert out["annotatedFrames"] == [3]
    assert out["llmModel"] == "fake-vision"
    assert len(llm.calls[0]["images"]) == 1, "one jpeg per requested frame"
    assert "убери это" in llm.calls[0]["user"]
    assert "frame 3" in llm.calls[0]["user"]
    assert "annotation marker" in llm.calls[0]["user"]
    assert "MUST be English" in llm.calls[0]["user"]
    assert "never call the object red" in llm.calls[0]["user"]


def test_cjk_prompt_repaired_when_user_wrote_nothing(tmp_path: Path):
    """Gemma-like models invent Chinese; pipeline prompt must stay usable for DINO."""
    src = tmp_path / "in.mp4"
    src.write_bytes(b"fake")
    mask = _write_mask(tmp_path, "m.png")
    reply = (
        '{"prompt":"请移除视频顶部的水印和底部的文字叠加层。",'
        '"targets":[{"kind":"watermark","query":"channel logo","where":"top","motion":"static"},'
        '{"kind":"text_overlay","query":"bottom caption","where":"bottom","motion":"static"}]}'
    )
    llm = FakeLlm(reply)
    out = _build_prompt(tmp_path, llm).execute(
        _req(tmp_path, prompt="", annotations=[{"frame": 0, "mask": str(mask)}]),
        tmp_path,
    )
    assert out["state"] == "COMPLETED"
    assert "请" not in out["prompt"]
    assert out["prompt"].startswith("Remove:")
    assert "channel logo" in out["prompt"]
    assert "bottom caption" in out["prompt"]


def test_mask_geometry_overrides_vlm_where(tmp_path: Path):
    """Painted top-right must not become VLM 'top' / center."""
    src = tmp_path / "in.mp4"
    src.write_bytes(b"fake")
    m = np.zeros((H, W), dtype=np.uint8)
    m[0:2, 5:8] = 255  # top-right on 8x8
    mask = tmp_path / "tr.png"
    cv2.imwrite(str(mask), m)
    reply = (
        '{"prompt":"Remove the logo at the top",'
        '"targets":[{"kind":"watermark","query":"logo","where":"top","motion":"static"}]}'
    )
    llm = FakeLlm(reply)
    out = _build_prompt(tmp_path, llm).execute(
        _req(tmp_path, prompt="", annotations=[{"frame": 0, "mask": str(mask)}]),
        tmp_path,
    )
    assert out["targets"][0]["where"] == "top-right"
    assert out["maskRegions"][0]["where"] == "top-right"
    assert "Mask geometry is AUTHORITATIVE" in llm.calls[0]["user"]


def test_annotation_paint_color_stripped_from_prompt_and_query(tmp_path: Path):
    """Red tint on masks must not become 'red logo' when the object is not red."""
    src = tmp_path / "in.mp4"
    src.write_bytes(b"fake")
    mask = _write_mask(tmp_path, "m.png")
    reply = (
        '{"prompt":"Remove the red logo at the top and the text overlay at the bottom.",'
        '"targets":[{"kind":"watermark","query":"red logo","where":"top","motion":"static"},'
        '{"kind":"text_overlay","query":"text overlay","where":"bottom","motion":"static"}]}'
    )
    out = _build_prompt(tmp_path, FakeLlm(reply)).execute(
        _req(tmp_path, prompt="", annotations=[{"frame": 0, "mask": str(mask)}]),
        tmp_path,
    )
    assert "red" not in out["prompt"].casefold()
    assert out["targets"][0]["query"] == "logo"
    assert "red" not in out["targets"][0]["query"].casefold()


def test_user_named_red_is_kept(tmp_path: Path):
    src = tmp_path / "in.mp4"
    src.write_bytes(b"fake")
    mask = _write_mask(tmp_path, "m.png")
    reply = (
        '{"prompt":"Remove the red logo",'
        '"targets":[{"kind":"watermark","query":"red logo","where":"top","motion":"static"}]}'
    )
    out = _build_prompt(tmp_path, FakeLlm(reply)).execute(
        _req(tmp_path, prompt="remove the red logo", annotations=[{"frame": 0, "mask": str(mask)}]),
        tmp_path,
    )
    assert "red logo" in out["prompt"].casefold()
    assert out["targets"][0]["query"] == "red logo"


def test_text_only_uses_sampled_frames(tmp_path: Path):
    src = tmp_path / "in.mp4"
    src.write_bytes(b"fake")
    llm = FakeLlm(REPLY)
    out = _build_prompt(tmp_path, llm).execute(_req(tmp_path, prompt="чисти"), tmp_path)
    assert out["framesUsed"] == [0, 4], "stride 4 over 5 frames"
    assert len(llm.calls[0]["images"]) == 2


def test_empty_input_fails(tmp_path: Path):
    src = tmp_path / "in.mp4"
    src.write_bytes(b"fake")
    with pytest.raises(PipelineError):
        _build_prompt(tmp_path, FakeLlm(REPLY)).execute(_req(tmp_path), tmp_path)


def test_unavailable_llm_fails(tmp_path: Path):
    src = tmp_path / "in.mp4"
    src.write_bytes(b"fake")
    mask = _write_mask(tmp_path, "m.png")

    class Dead(FakeLlm):
        def status(self):
            return "unavailable: down"

    with pytest.raises(AdapterUnavailable):
        _build_prompt(tmp_path, Dead(REPLY)).execute(_req(tmp_path, annotations=[
            {"frame": 0, "mask": str(mask)},
        ]), tmp_path)


def test_nonjson_reply_fails(tmp_path: Path):
    src = tmp_path / "in.mp4"
    src.write_bytes(b"fake")
    mask = _write_mask(tmp_path, "m.png")
    with pytest.raises(PipelineError):
        _build_prompt(tmp_path, FakeLlm("no json here")).execute(_req(tmp_path, annotations=[
            {"frame": 0, "mask": str(mask)},
        ]), tmp_path)


def test_failure_writes_report(tmp_path: Path):
    src = tmp_path / "in.mp4"
    src.write_bytes(b"fake")
    mask = _write_mask(tmp_path, "m.png")
    with pytest.raises(PipelineError):
        _build_prompt(tmp_path, FakeLlm("no json here"), make_paths=PromptPaths.create).execute(_req(tmp_path, annotations=[
            {"frame": 0, "mask": str(mask)},
        ]), tmp_path)
    report_file = tmp_path / "jobs" / "j-prompt" / "output" / "report.json"
    assert report_file.is_file(), "failure report must be written even when output/ did not exist"
    data = json.loads(report_file.read_text(encoding="utf-8"))
    assert data["state"] == "FAILED"
    assert data["jobId"] == "j-prompt"
