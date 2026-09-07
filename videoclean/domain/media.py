from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MediaManifest:
    path: Path
    duration_s: float
    width: int
    height: int
    fps: float
    fps_ratio: str
    video_codec: str
    pix_fmt: str
    frame_count: int
    has_audio: bool
    audio_codec: str | None
