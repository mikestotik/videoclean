from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from videoclean.application.errors import PipelineError
from videoclean.domain.media import MediaManifest


class FFmpegMedia:
    name = "ffmpeg"

    def probe(self, path: Path) -> MediaManifest:
        cmd = [
            _ffprobe(),
            "-hide_banner",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise PipelineError(result.stderr.strip() or "ffprobe failed")
        data = json.loads(result.stdout)
        video = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
        audio = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), None)
        if not video:
            raise PipelineError("no video stream")

        rate = video.get("r_frame_rate") or video.get("avg_frame_rate") or "30/1"
        num, den = _split_rate(rate)
        fps = num / den if den else 30.0
        duration = float(data.get("format", {}).get("duration") or video.get("duration") or 0)
        nb = video.get("nb_frames")
        try:
            frame_count = int(nb) if nb and nb != "N/A" else max(1, round(duration * fps))
        except ValueError:
            frame_count = max(1, round(duration * fps))

        return MediaManifest(
            path=path,
            duration_s=duration,
            width=int(video["width"]),
            height=int(video["height"]),
            fps=fps,
            fps_ratio=rate,
            video_codec=str(video.get("codec_name") or "unknown"),
            pix_fmt=str(video.get("pix_fmt") or "unknown"),
            frame_count=frame_count,
            has_audio=audio is not None,
            audio_codec=(audio or {}).get("codec_name"),
        )

    def extract_frames(self, src: Path, dest_dir: Path, log_file: Path) -> list[Path]:
        dest_dir.mkdir(parents=True, exist_ok=True)
        pattern = dest_dir / "frame_%06d.jpg"
        _run(
            [
                _ffmpeg(),
                "-y",
                "-hide_banner",
                "-i",
                str(src),
                "-fps_mode",
                "passthrough",
                "-q:v",
                "2",
                str(pattern),
            ],
            log_file,
        )
        frames = sorted(dest_dir.glob("frame_*.jpg")) or sorted(dest_dir.glob("frame_*.png"))
        if not frames:
            raise PipelineError("FFmpeg produced no frames")
        return frames

    def extract_frames_subset(
        self, src: Path, indices: list[int], dest_dir: Path, log_file: Path
    ) -> list[Path]:
        """Extract only the requested frame numbers. Output order matches sorted indices.

        ffmpeg names selected outputs sequentially (frame_000001 = first selected),
        so files are renamed to their source frame numbers afterwards.
        """
        idxs = sorted({int(i) for i in indices})
        if not idxs:
            raise PipelineError("extract_frames_subset: no frame indices requested")
        dest_dir.mkdir(parents=True, exist_ok=True)
        selector = "+".join(f"eq(n,{i})" for i in idxs)
        _run(
            [
                _ffmpeg(),
                "-y",
                "-hide_banner",
                "-i",
                str(src),
                "-vf",
                f"select='{selector}'",
                "-vsync",
                "0",
                "-q:v",
                "2",
                str(dest_dir / "frame_%06d.jpg"),
            ],
            log_file,
        )
        produced = sorted(dest_dir.glob("frame_*.jpg"))
        if not produced:
            raise PipelineError("FFmpeg produced no frames for the requested subset")
        if len(produced) != len(idxs):
            raise PipelineError(
                f"FFmpeg produced {len(produced)} of {len(idxs)} requested frames"
            )
        # Sequential output k-th file == k-th requested source frame index.
        # Two-phase rename: produced names overlap targets (frame_000004 is both
        # produced #4 and the target of produced #2) — renaming in place loses files.
        staged: list[tuple[Path, Path]] = []
        for p in produced:
            tmp = p.with_name(f".stage_{p.name}")
            p.rename(tmp)
            staged.append((tmp, p))
        for (tmp, _orig), i in zip(staged, idxs):
            tmp.rename(dest_dir / f"frame_{i:06d}.jpg")
        return [dest_dir / f"frame_{i:06d}.jpg" for i in idxs]

    def encode_mezzanine(
        self,
        frames_dir: Path,
        src: Path,
        dest: Path,
        fps_ratio: str,
        has_audio: bool,
        frame_count: int,
        log_file: Path,
    ) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        pattern = _frame_pattern(frames_dir)
        cmd = [_ffmpeg(), "-y", "-hide_banner", "-framerate", fps_ratio, "-i", str(pattern)]
        if has_audio:
            cmd += ["-i", str(src)]
        cmd += ["-map", "0:v:0", "-frames:v", str(frame_count)]
        if has_audio:
            cmd += ["-map", "1:a:0", "-c:a", "aac", "-b:a", "192k"]
        cmd += ["-pix_fmt", "yuv420p", *encoder_argv(False), "-movflags", "+faststart", str(dest)]
        _run(cmd, log_file)

    def decode_bgr(self, src: Path, store, log_file: Path) -> int:
        """Stream bgr24 into a FrameStore. Returns the number of frames written."""
        import subprocess

        cmd = [
            _ffmpeg(),
            "-hide_banner",
            "-i",
            str(src),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "pipe:1",
        ]
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "ab") as log:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=log)
            assert proc.stdout is not None
            written = 0
            try:
                for i in range(len(store)):
                    raw = proc.stdout.read(store.frame_bytes)
                    if len(raw) < store.frame_bytes:
                        break
                    store.write_bytes_at(i, raw)
                    written += 1
            finally:
                proc.stdout.close()
                code = proc.wait()
            if code != 0 and written == 0:
                raise PipelineError(f"ffmpeg decode failed (exit {code}); see {log_file}")
        return written

    def encode_from_store(
        self,
        store,
        src: Path,
        dest: Path,
        fps_ratio: str,
        has_audio: bool,
        log_file: Path,
        *,
        nvenc: bool,
    ) -> None:
        """Encode bgr24 frames from a store on stdin. No JPEG directory."""
        import subprocess

        import numpy as np

        dest.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            _ffmpeg(),
            "-y",
            "-hide_banner",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{store.w}x{store.h}",
            "-framerate",
            fps_ratio,
            "-i",
            "pipe:0",
        ]
        if has_audio:
            cmd += ["-i", str(src)]
        cmd += ["-map", "0:v:0", "-frames:v", str(len(store)), "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
        cmd += encoder_argv(bool(nvenc))
        if has_audio:
            cmd += ["-map", "1:a:0", "-c:a", "aac", "-b:a", "192k"]
        cmd += [str(dest)]
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "ab") as log:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=log)
            assert proc.stdin is not None
            try:
                for i in range(len(store)):
                    proc.stdin.write(np.ascontiguousarray(store[i]).tobytes())
            finally:
                proc.stdin.close()
                code = proc.wait()
        if code != 0 or not dest.is_file():
            raise PipelineError(f"ffmpeg encode failed (exit {code}); see {log_file}")

    def scale_clip(
        self,
        src: Path,
        dest: Path,
        *,
        width: int,
        height: int,
        log_file: Path | None = None,
    ) -> MediaManifest:
        """Re-encode a library source at a smaller even frame size."""
        manifest = self.probe(src)
        if width < 2 or height < 2 or width % 2 or height % 2:
            raise PipelineError(f"scale size must be even and >= 2, got {width}x{height}")
        if width > manifest.width or height > manifest.height:
            raise PipelineError("scale cannot enlarge the source")
        dest.parent.mkdir(parents=True, exist_ok=True)
        log = log_file or (dest.parent / "ffmpeg_scale.log")
        cmd = [
            _ffmpeg(), "-y", "-hide_banner", "-i", str(src),
            "-vf", f"scale={width}:{height}",
            "-pix_fmt", "yuv420p",
            *encoder_argv(nvenc_available()),
        ]
        if manifest.has_audio:
            cmd += ["-c:a", "aac", "-b:a", "128k"]
        else:
            cmd += ["-an"]
        cmd += ["-movflags", "+faststart", str(dest)]
        _run(cmd, log)
        if not dest.is_file():
            raise PipelineError("scale produced no output")
        return self.probe(dest)

    def crop_clip(
        self,
        src: Path,
        dest: Path,
        *,
        start_s: float | None = None,
        end_s: float | None = None,
        left: int = 0,
        right: int = 0,
        top: int = 0,
        bottom: int = 0,
        log_file: Path | None = None,
    ) -> MediaManifest:
        """Trim time and/or crop frame edges into a new mp4. Re-encodes when needed."""
        manifest = self.probe(src)
        left_i = max(0, int(left))
        right_i = max(0, int(right))
        top_i = max(0, int(top))
        bottom_i = max(0, int(bottom))
        out_w = manifest.width - left_i - right_i
        out_h = manifest.height - top_i - bottom_i
        if out_w < 2 or out_h < 2:
            raise PipelineError(
                f"crop leaves {out_w}x{out_h}; need at least 2x2 "
                f"(source {manifest.width}x{manifest.height})"
            )
        # yuv420p needs even dimensions
        out_w -= out_w % 2
        out_h -= out_h % 2
        if out_w < 2 or out_h < 2:
            raise PipelineError("crop dimensions must stay even and >= 2")

        duration = float(manifest.duration_s or 0.0)
        ss = 0.0 if start_s is None else float(start_s)
        ee = duration if end_s is None else float(end_s)
        if ss < 0:
            raise PipelineError("start_s must be >= 0")
        if duration > 0 and ss >= duration:
            raise PipelineError(f"start_s {ss} is past duration {duration:.3f}s")
        if ee <= ss:
            raise PipelineError("end_s must be greater than start_s")
        if duration > 0:
            ee = min(ee, duration)
        trim_dur = ee - ss
        spatial = left_i or right_i or top_i or bottom_i
        temporal = ss > 0 or (duration > 0 and ee < duration - 0.001)
        if not spatial and not temporal:
            raise PipelineError("nothing to crop: set time range and/or edge pixels")

        dest.parent.mkdir(parents=True, exist_ok=True)
        log = log_file or (dest.parent / "ffmpeg_crop.log")
        # Always re-encode: stream-copy trims are keyframe-snapped and inaccurate.
        cmd = [_ffmpeg(), "-y", "-hide_banner", "-i", str(src)]
        if ss > 0:
            cmd += ["-ss", f"{ss:.6f}"]
        cmd += ["-t", f"{trim_dur:.6f}"]
        if spatial:
            cmd += ["-vf", f"crop={out_w}:{out_h}:{left_i}:{top_i}"]
        cmd += ["-pix_fmt", "yuv420p", *encoder_argv(nvenc_available())]
        if manifest.has_audio:
            cmd += ["-c:a", "aac", "-b:a", "128k"]
        else:
            cmd += ["-an"]
        cmd += ["-movflags", "+faststart", str(dest)]
        _run(cmd, log)
        if not dest.is_file():
            raise PipelineError("crop produced no output")
        return self.probe(dest)

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
    ) -> Path:
        if fmt == "mp4":
            dest.parent.mkdir(parents=True, exist_ok=True)
            if src.resolve() != dest.resolve():
                _run(
                    [_ffmpeg(), "-y", "-hide_banner", "-i", str(src), "-c", "copy", "-movflags", "+faststart", str(dest)],
                    log_file,
                )
            return dest
        if fmt == "mov":
            dest.parent.mkdir(parents=True, exist_ok=True)
            _run(
                [_ffmpeg(), "-y", "-hide_banner", "-i", str(src), "-c:v", "copy", "-c:a", "copy", "-f", "mov", str(dest)],
                log_file,
            )
            return dest
        if fmt == "mkv":
            dest.parent.mkdir(parents=True, exist_ok=True)
            _run([_ffmpeg(), "-y", "-hide_banner", "-i", str(src), "-c", "copy", str(dest)], log_file)
            return dest
        if fmt == "webm":
            crf = max(10, min(63, int(webm_crf)))
            dest.parent.mkdir(parents=True, exist_ok=True)
            _run(
                [
                    _ffmpeg(),
                    "-y",
                    "-hide_banner",
                    "-i",
                    str(src),
                    "-c:v",
                    "libvpx-vp9",
                    "-b:v",
                    "0",
                    "-crf",
                    str(crf),
                    "-c:a",
                    "libopus",
                    "-b:a",
                    "128k",
                    str(dest),
                ],
                log_file,
            )
            return dest
        if fmt in {"hls-fmp4", "hls-ts"}:
            return _package_hls(
                src,
                dest,
                width=width,
                height=height,
                fps=fps,
                log_file=log_file,
                segment_seconds=max(1, int(segment_seconds)),
                segment_type="fmp4" if fmt == "hls-fmp4" else "mpegts",
            )
        if fmt == "dash":
            return _package_dash(
                src,
                dest,
                fps=fps,
                log_file=log_file,
                segment_seconds=max(1, int(segment_seconds)),
            )
        raise PipelineError(f"no packager for {fmt}")


_NVENC_OK: bool | None = None


def encoder_argv(nvenc: bool) -> list[str]:
    """Video codec flags. NVENC must not receive -crf."""
    if nvenc:
        return ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "18", "-b:v", "0"]
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18"]


def nvenc_available() -> bool:
    """Process-cached probe: encoder listed, then a 16x16 null encode."""
    global _NVENC_OK
    if _NVENC_OK is not None:
        return _NVENC_OK
    _NVENC_OK = _probe_nvenc()
    return _NVENC_OK


def _probe_nvenc() -> bool:
    import subprocess

    binary = _ffmpeg()
    try:
        listed = subprocess.run(
            [binary, "-hide_banner", "-encoders"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return False
    blob = (listed.stdout or "") + (listed.stderr or "")
    if "h264_nvenc" not in blob:
        return False
    cmd = [
        binary,
        "-y",
        "-hide_banner",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        "16x16",
        "-r",
        "1",
        "-i",
        "pipe:0",
        "-frames:v",
        "1",
        *encoder_argv(True),
        "-f",
        "null",
        "-",
    ]
    try:
        proc = subprocess.run(cmd, input=b"\x00" * (16 * 16 * 3), check=False, capture_output=True)
    except OSError:
        return False
    return proc.returncode == 0


def _frame_pattern(frames_dir: Path) -> Path:
    if next(frames_dir.glob("frame_*.jpg"), None):
        return frames_dir / "frame_%06d.jpg"
    return frames_dir / "frame_%06d.png"


def _ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise PipelineError("FFmpeg not found in PATH. Install ffmpeg and retry.")
    return path


def _ffprobe() -> str:
    path = shutil.which("ffprobe")
    if not path:
        raise PipelineError("ffprobe not found in PATH. Install ffmpeg and retry.")
    return path


def _package_hls(
    src: Path,
    dest_dir: Path,
    *,
    width: int,
    height: int,
    fps: float,
    log_file: Path,
    segment_seconds: int = 6,
    segment_type: str = "fmp4",
) -> Path:
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    variant = dest_dir / f"{height}p"
    variant.mkdir(parents=True, exist_ok=True)
    gop = max(1, int(round(fps * segment_seconds)))
    playlist = variant / "index.m3u8"
    if segment_type == "fmp4":
        seg_args = [
            "-hls_segment_type",
            "fmp4",
            "-hls_fmp4_init_filename",
            "init.mp4",
            "-hls_segment_filename",
            str(variant / "segment_%05d.m4s"),
        ]
    else:
        seg_args = [
            "-hls_segment_type",
            "mpegts",
            "-hls_segment_filename",
            str(variant / "segment_%05d.ts"),
        ]
    codecs = "avc1.640028,mp4a.40.2"
    cmd = [
        _ffmpeg(),
        "-y",
        "-hide_banner",
        "-i",
        str(src),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "19",
        "-pix_fmt",
        "yuv420p",
        "-g",
        str(gop),
        "-keyint_min",
        str(gop),
        "-sc_threshold",
        "0",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-ar",
        "48000",
        "-f",
        "hls",
        "-hls_time",
        str(segment_seconds),
        "-hls_playlist_type",
        "vod",
        "-hls_flags",
        "independent_segments",
        *seg_args,
        str(playlist),
    ]
    _run(cmd, log_file)

    size = src.stat().st_size if src.is_file() else 0
    duration = max(FFmpegMedia().probe(src).duration_s, 0.001)
    bandwidth = max(300_000, int(size * 8 / duration))
    master = dest_dir / "master.m3u8"
    master.write_text(
        "\n".join(
            [
                "#EXTM3U",
                "#EXT-X-VERSION:7",
                (
                    f"#EXT-X-STREAM-INF:BANDWIDTH={bandwidth},"
                    f"RESOLUTION={width}x{height},"
                    f'CODECS="{codecs}"'
                ),
                f"{height}p/index.m3u8",
                "",
            ]
        ),
        encoding="utf-8",
    )
    if not playlist.is_file():
        raise PipelineError("HLS playlist was not created")
    # Return the package directory so downloads can zip the whole tree.
    return dest_dir


def _package_dash(
    src: Path,
    dest_dir: Path,
    *,
    fps: float,
    log_file: Path,
    segment_seconds: int = 6,
) -> Path:
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    gop = max(1, int(round(fps * segment_seconds)))
    manifest = dest_dir / "manifest.mpd"
    _run(
        [
            _ffmpeg(),
            "-y",
            "-hide_banner",
            "-i",
            str(src),
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "19",
            "-pix_fmt",
            "yuv420p",
            "-g",
            str(gop),
            "-keyint_min",
            str(gop),
            "-sc_threshold",
            "0",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-f",
            "dash",
            "-seg_duration",
            str(segment_seconds),
            "-use_timeline",
            "1",
            "-use_template",
            "1",
            str(manifest),
        ],
        log_file,
    )
    if not manifest.is_file():
        raise PipelineError("DASH manifest was not created")
    # Return the package directory so downloads can zip the whole tree.
    return dest_dir


def _run(cmd: list[str], log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as fh:
        fh.write("\n$ " + " ".join(cmd) + "\n")
        fh.flush()
        result = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise PipelineError(f"command failed ({result.returncode}): {' '.join(cmd)}")


def _split_rate(rate: str) -> tuple[float, float]:
    if "/" in rate:
        a, b = rate.split("/", 1)
        return float(a), float(b) or 1.0
    return float(rate), 1.0
