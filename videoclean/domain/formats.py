from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FormatSpec:
    name: str
    kind: str  # file | package
    suffix: str
    master_name: str | None = None


FORMATS: dict[str, FormatSpec] = {
    "mp4": FormatSpec("mp4", "file", ".mp4"),
    "mov": FormatSpec("mov", "file", ".mov"),
    "mkv": FormatSpec("mkv", "file", ".mkv"),
    "webm": FormatSpec("webm", "file", ".webm"),
    "hls-fmp4": FormatSpec("hls-fmp4", "package", ".hls-fmp4", "master.m3u8"),
    "hls-ts": FormatSpec("hls-ts", "package", ".hls-ts", "master.m3u8"),
    "dash": FormatSpec("dash", "package", ".dash", "manifest.mpd"),
}

ALIASES = {
    "hls": "hls-fmp4",
    "mpegdash": "dash",
    "mpd": "dash",
}


def known_format_names() -> list[str]:
    return list(FORMATS)


def parse_formats(raw: list[str] | str | None) -> list[str]:
    if not raw:
        return ["mp4"]
    items = raw if isinstance(raw, list) else [raw]
    names: list[str] = []
    for item in items:
        for part in str(item).split(","):
            name = part.strip().lower()
            if not name:
                continue
            if name == "both":
                raise ValueError("'both' is not a format. Example: --format mp4,webm,hls-fmp4")
            name = ALIASES.get(name, name)
            if name not in FORMATS:
                raise ValueError(f"unknown format {name!r}. known: {', '.join(known_format_names())}")
            if name not in names:
                names.append(name)
    return names or ["mp4"]


def output_stem(output: Path) -> Path:
    file_suffixes = {spec.suffix for spec in FORMATS.values() if spec.kind == "file"}
    if output.suffix.lower() in file_suffixes:
        return output.with_suffix("")
    return output


def resolve_dest(output: Path, fmt: str, requested: list[str]) -> Path:
    spec = FORMATS[fmt]
    if spec.kind == "file" and requested == [fmt] and output.suffix.lower() == spec.suffix:
        return output
    return output_stem(output).with_name(output_stem(output).name + spec.suffix)
