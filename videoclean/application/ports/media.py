from __future__ import annotations

from pathlib import Path
from typing import Protocol

from videoclean.domain.media import MediaManifest


class MediaGateway(Protocol):
    """Always FFmpeg in this product. Port exists so the use case never imports ffmpeg."""

    name: str

    def probe(self, path: Path) -> MediaManifest: ...

    def extract_frames(self, src: Path, dest_dir: Path, log_file: Path) -> list[Path]: ...

    def extract_frames_subset(
        self, src: Path, indices: list[int], dest_dir: Path, log_file: Path
    ) -> list[Path]: ...

    def encode_mezzanine(
        self,
        frames_dir: Path,
        src: Path,
        dest: Path,
        fps_ratio: str,
        has_audio: bool,
        frame_count: int,
        log_file: Path,
    ) -> None: ...

    def package(
        self,
        src: Path,
        dest: Path,
        fmt: str,
        *,
        width: int,
        height: int,
        fps: float,
        log_file: Path,
        segment_seconds: int = 6,
        webm_crf: int = 32,
    ) -> Path: ...
