from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

from videoclean.application.errors import PipelineError
from videoclean.application.ports.media import MediaGateway
from videoclean.domain.formats import parse_formats, resolve_dest

OnFormatDone = Callable[[str, int, int, Path], None]


class PackageMedia:
    def __init__(self, media: MediaGateway) -> None:
        self.media = media

    def execute(
        self,
        src: Path,
        output: Path | None,
        formats: list[str] | None,
        overwrite: bool,
        log_file: Path,
        *,
        segment_seconds: int = 6,
        webm_crf: int = 32,
        on_format: OnFormatDone | None = None,
    ) -> dict[str, Path]:
        fmts = parse_formats(formats or ["mp4"])
        src = src.expanduser().resolve()
        if not src.is_file():
            raise PipelineError(f"not a file: {src}")
        base = output.expanduser().resolve() if output else src
        manifest = self.media.probe(src)
        artifacts: dict[str, Path] = {}
        total = len(fmts)
        for i, fmt in enumerate(fmts, start=1):
            dest = resolve_dest(base, fmt, fmts)
            if dest.exists() and not overwrite:
                raise FileExistsError(f"{dest} exists (pass --overwrite)")
            if dest.exists():
                if dest.is_dir():
                    shutil.rmtree(dest)
                else:
                    dest.unlink()
            artifacts[fmt] = self.media.package(
                src,
                dest,
                fmt,
                width=manifest.width,
                height=manifest.height,
                fps=manifest.fps,
                log_file=log_file,
                segment_seconds=segment_seconds,
                webm_crf=webm_crf,
            )
            if on_format is not None:
                on_format(fmt, i, total, artifacts[fmt])
        return artifacts
