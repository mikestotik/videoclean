from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from videoclean.application.config import PipelineConfig, RunCleanupRequest
from videoclean.application.errors import AdapterUnavailable, JobCancelled, PipelineError
from videoclean.application.ports.detector import Detector
from videoclean.application.ports.inpainter import Inpainter
from videoclean.application.ports.jobs import JobStore
from videoclean.application.ports.media import MediaGateway
from videoclean.application.ports.progress import ProgressPort
from videoclean.application.ports.prompt import PromptParser
from videoclean.application.ports.segmenter import Segmenter
from videoclean.application.frames_sample import sample_frame_indices
from videoclean.application.frames import LazyFrames
from videoclean.application.select import select_tracks
from videoclean.domain.formats import resolve_dest
from videoclean.domain.tracks import Detection, tracks_to_json


class RunCleanup:
    def __init__(
        self,
        *,
        media: MediaGateway,
        parser: PromptParser,
        detectors: list[Detector],
        segmenter: Segmenter,
        inpainter: Inpainter,
        jobs: JobStore,
        progress: ProgressPort,
        new_job_id: callable,
        make_paths: callable,
        utc_now: callable,
        read_image: callable,
        write_image: callable,
    ) -> None:
        self.media = media
        self.parser = parser
        self.detectors = detectors
        self.segmenter = segmenter
        self.inpainter = inpainter
        self.jobs = jobs
        self.progress = progress
        self._new_job_id = new_job_id
        self._make_paths = make_paths
        self._utc_now = utc_now
        self._read_image = read_image
        self._write_image = write_image

    def execute(self, req: RunCleanupRequest, data_dir: Path) -> dict:
        cfg = req.config
        cfg.validate()
        if not (req.prompt or "").strip():
            raise PipelineError("--prompt is required")
        if not req.input_path.is_file():
            raise FileNotFoundError(req.input_path)
        for fmt in cfg.formats:
            dest = resolve_dest(req.output_path, fmt, cfg.formats)
            if dest.exists() and not req.overwrite:
                raise FileExistsError(f"{dest} exists (pass --overwrite)")

        job_id = req.job_id or self._new_job_id()
        paths = self._make_paths(data_dir / "jobs" / job_id)
        self.jobs.upsert(
            job_id,
            "RUNNING",
            input_path=str(req.input_path),
            output_path=str(req.output_path),
            prompt=req.prompt,
        )

        manifest = req.manifest or self.media.probe(req.input_path)
        report: dict = {
            "jobId": job_id,
            "state": "RUNNING",
            "startedAt": self._utc_now().isoformat(),
            "prompt": req.prompt,
            "device": cfg.device,
            "media": self.media.name,
            "detectors": [d.name for d in self.detectors],
            "detectorModel": cfg.detector_model,
            "segmenter": self.segmenter.name,
            "inpainter": self.inpainter.name,
        }

        try:
            return self._run(req, cfg, job_id, paths, manifest, report)
        except Exception as exc:
            report["state"] = "FAILED"
            report["error"] = str(exc)
            report["finishedAt"] = self._utc_now().isoformat()
            try:
                paths.report_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
            except OSError:
                pass
            self.jobs.upsert(job_id, "FAILED", report=report)
            raise

    def _run(self, req: RunCleanupRequest, cfg: PipelineConfig, job_id: str, paths, manifest, report: dict) -> dict:
        self._ensure_ready()
        self.progress.start("validate")
        (paths.input_dir / "input_manifest.json").write_text(
            json.dumps(
                {
                    "inputPath": str(req.input_path),
                    "durationMs": int(manifest.duration_s * 1000),
                    "video": {
                        "codec": manifest.video_codec,
                        "width": manifest.width,
                        "height": manifest.height,
                        "fps": manifest.fps,
                        "pixelFormat": manifest.pix_fmt,
                    },
                    "audio": {"present": manifest.has_audio, "codec": manifest.audio_codec},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        self.progress.finish("validate", f"{manifest.video_codec} {manifest.width}x{manifest.height}")

        self.progress.start("normalize", total=manifest.frame_count, detail="ffmpeg extract frames")
        frames = self.media.extract_frames(req.input_path, paths.frames_dir, paths.ffmpeg_log)
        self.progress.tick("normalize", len(frames), len(frames))
        self.progress.finish("normalize", f"{len(frames)} frames")

        (paths.root / "analysis").mkdir(exist_ok=True)
        cache_n = min(48, max(8, len(frames)))
        images = LazyFrames(frames, self._read_image, cache_size=cache_n)
        first = images[0]
        h, w = first.shape[:2]

        parse_st = self.parser.status() if hasattr(self.parser, "status") else "llm"
        sample_idxs = sample_frame_indices(len(frames), cfg.prompt_frame_stride, cfg.prompt_frame_max)
        sample_frames = [images[i] for i in sample_idxs] if sample_idxs else None
        detail = f"llm  {parse_st}"
        if sample_idxs:
            detail += f"  vision frames={len(sample_idxs)} stride={cfg.prompt_frame_stride}"
        self.progress.start("parse", detail=detail)
        intent = self.parser.parse(req.prompt, frames=sample_frames)
        (paths.root / "analysis" / "prompt.json").write_text(
            json.dumps(
                {
                    "targets": [_target_json(t) for t in intent.targets],
                    "queries": intent.queries,
                    "parseMode": intent.parse_mode,
                    "defaulted": intent.defaulted,
                    "raw": intent.raw,
                    "visionFrameIndices": sample_idxs,
                    "promptFrameStride": cfg.prompt_frame_stride,
                    "promptFrameMax": cfg.prompt_frame_max,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        label = ", ".join(_target_label(t) for t in intent.targets)
        if intent.defaulted:
            label += " (default)"
        if intent.parse_mode == "llm-vision":
            label += f"  (vision×{len(sample_idxs)})"
        self.progress.finish("parse", label)

        self.progress.start("detect", total=len(frames), detail="read frames")
        used_detector, attempts, raw_tracks = self._discover(images, intent.queries, stage="detect")
        tracks = select_tracks(raw_tracks, intent, width=w, height=h)
        if raw_tracks and not tracks:
            labels = sorted({(tr.label or "?") for tr in raw_tracks})
            raise PipelineError(
                f"detector found {len(raw_tracks)} tracks labeled {labels} "
                f"but none matched queries {intent.queries} "
                f"({', '.join(_target_label(t) for t in intent.targets)}). "
                f"where/ordinal may have dropped them. detectors tried: {attempts}"
            )
        (paths.root / "analysis" / "tracks_raw.json").write_text(
            json.dumps(tracks_to_json(raw_tracks), indent=2),
            encoding="utf-8",
        )
        (paths.root / "analysis" / "tracks.json").write_text(
            json.dumps(tracks_to_json(tracks), indent=2),
            encoding="utf-8",
        )
        (paths.root / "analysis" / "detector.json").write_text(
            json.dumps({"used": used_detector, "attempts": attempts}, indent=2),
            encoding="utf-8",
        )
        masks = self.segmenter.masks(images, tracks)
        for frame_path, mask in zip(frames, masks):
            self._write_image(paths.masks_dir / frame_path.name, mask)
        mean_cov = float(np.mean([np.count_nonzero(m) / m.size for m in masks])) if masks else 0.0
        if mean_cov < cfg.min_mask_coverage:
            raise PipelineError(
                f"mask coverage {mean_cov:.4%} is below --min-mask-coverage "
                f"{cfg.min_mask_coverage:.4%}. queries={intent.queries}. "
                f"detectors tried: {attempts}"
            )
        detections = [
            Detection(label=tr.label, coverage=tr.coverage, mask_kind=tr.motion, notes=tr.notes)
            for tr in tracks
        ]
        self.progress.finish("detect", f"{used_detector}: {len(detections)} tracks, coverage {mean_cov:.2%}")

        self.progress.start("inpaint", total=len(frames), detail=f"{self.inpainter.name} 0/{len(frames)}")
        if getattr(self.inpainter, "video_aware", False):
            self.progress.tick("inpaint", 0, len(frames), f"{self.inpainter.name} clip")
            cleaned_frames = self.inpainter.inpaint_clip(images, masks)
            for i, (frame_path, cleaned) in enumerate(zip(frames, cleaned_frames), start=1):
                self._write_image(paths.inpainted_dir / frame_path.name, cleaned)
                self.progress.tick("inpaint", i, len(frames), f"{self.inpainter.name} write {i}/{len(frames)}")
        else:
            for i, (frame_path, image) in enumerate(zip(frames, images), start=1):
                cleaned = self.inpainter.inpaint(image, masks[i - 1])
                self._write_image(paths.inpainted_dir / frame_path.name, cleaned)
                self.progress.tick("inpaint", i, len(frames), f"{self.inpainter.name} {i}/{len(frames)}")
        self.progress.finish("inpaint", self.inpainter.name)

        verify_passes = 0
        if cfg.verify:
            self.progress.start("verify", total=len(frames), detail="read inpainted frames")
            cleaned_paths = [paths.inpainted_dir / p.name for p in frames]
            cleaned = LazyFrames(cleaned_paths, self._read_image, cache_size=cache_n)
            self.progress.tick("verify", 0, 1, "re-detect leftover")
            _, v_attempts, leftover = self._discover(cleaned, intent.queries, stage="verify")
            leftover = select_tracks(leftover, intent, width=w, height=h)
            v_masks = self.segmenter.masks(cleaned, leftover) if leftover else []
            still = float(np.mean([np.count_nonzero(m) / m.size for m in v_masks])) if v_masks else 0.0
            cap = max(cfg.verify_max_coverage, mean_cov * 1.5)
            if leftover and cfg.min_mask_coverage <= still <= cap:
                verify_passes = 1
                grown = [_or_mask(a, b) for a, b in zip(masks, v_masks)]
                n = len(frames)
                if getattr(self.inpainter, "video_aware", False):
                    self.progress.tick("verify", 0, n, f"re-inpaint leftover {still:.2%}  clip")
                    redone = self.inpainter.inpaint_clip(images, grown)
                    for frame_path, frame in zip(frames, redone):
                        self._write_image(paths.inpainted_dir / frame_path.name, frame)
                else:
                    for i in range(n):
                        frame = self.inpainter.inpaint(images[i], grown[i])
                        self._write_image(paths.inpainted_dir / frames[i].name, frame)
                        self.progress.tick(
                            "verify", i + 1, n, f"re-inpaint leftover {still:.2%}  {i + 1}/{n}"
                        )
                self.progress.finish("verify", f"pass 1, leftover {still:.2%}")
            else:
                why = "clean" if not leftover else f"leftover {still:.2%} outside [{cfg.min_mask_coverage:.4%}, {cap:.2%}] ({v_attempts})"
                self.progress.finish("verify", why)
        else:
            self.progress.start("verify")
            self.progress.finish("verify", "skipped")

        self.progress.start("encode", detail="ffmpeg mezzanine")
        mezz = paths.inpainted_dir.parent / "mezzanine.mp4"
        self.media.encode_mezzanine(
            paths.inpainted_dir,
            req.input_path,
            mezz,
            manifest.fps_ratio,
            manifest.has_audio,
            len(frames),
            paths.ffmpeg_log,
        )
        self.progress.finish("encode", mezz.name)

        self.progress.start("package", total=len(cfg.formats))
        outputs: dict[str, str] = {}
        for i, fmt in enumerate(cfg.formats, start=1):
            dest = resolve_dest(req.output_path, fmt, cfg.formats)
            if dest.exists():
                if dest.is_dir():
                    shutil.rmtree(dest)
                else:
                    dest.unlink()
            artifact = self.media.package(
                mezz,
                dest,
                fmt,
                width=manifest.width,
                height=manifest.height,
                fps=manifest.fps,
                log_file=paths.ffmpeg_log,
            )
            outputs[fmt] = str(artifact)
            self.progress.tick("package", i, len(cfg.formats), fmt)
        self.progress.finish("package", ", ".join(cfg.formats))

        self.progress.start("report")
        report.update(
            {
                "state": "COMPLETED",
                "finishedAt": self._utc_now().isoformat(),
                "targets": [_target_json(t) for t in intent.targets],
                "verify": cfg.verify,
                "verifyPasses": verify_passes,
                "promptParseMode": intent.parse_mode,
                "detectorUsed": used_detector,
                "detectorAttempts": attempts,
                "frames": len(frames),
                "meanMaskCoverage": mean_cov,
                "detections": [
                    {"label": d.label, "coverage": d.coverage, "kind": d.mask_kind, "notes": d.notes}
                    for d in detections
                ],
                "formats": cfg.formats,
                "outputs": outputs,
                "output": next(iter(outputs.values())),
                "workdir": str(paths.root),
            }
        )
        paths.report_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        self.progress.finish("report", "ok")
        self.jobs.upsert(job_id, "COMPLETED", report=report)
        if not req.keep_workdir:
            shutil.rmtree(paths.frames_dir, ignore_errors=True)
            shutil.rmtree(paths.inpainted_dir, ignore_errors=True)
            shutil.rmtree(paths.masks_dir, ignore_errors=True)
            shutil.rmtree(paths.input_dir, ignore_errors=True)
            process = paths.root / "process"
            if process.is_dir():
                shutil.rmtree(process, ignore_errors=True)
        return report

    def _discover(self, images, queries: list[str], *, stage: str) -> tuple[str, list[str], list]:
        attempts: list[str] = []
        last_tracks: list = []
        q = ", ".join(queries)

        def on_progress(current: int, total: int, detail: str = "") -> None:
            self.progress.tick(stage, current, total, detail)

        for detector in self.detectors:
            status = detector.status()
            self.progress.tick(stage, 0, 1, f"{detector.name}: {status}")
            if not status.startswith("ready"):
                attempts.append(f"{detector.name}: skip ({status})")
                continue
            self.progress.tick(stage, 0, 1, f"{detector.name}  queries: {q}")
            try:
                tracks = detector.discover(images, queries, on_progress=on_progress)
            except (MemoryError, JobCancelled):
                raise
            except Exception as exc:  # noqa: BLE001 — optional backend; next in chain
                msg = f"{type(exc).__name__}: {exc}".replace("\n", " ")[:400]
                attempts.append(f"{detector.name}: error ({msg})")
                continue
            attempts.append(f"{detector.name}: {len(tracks)} tracks")
            last_tracks = tracks
            if tracks:
                return detector.name, attempts, tracks
        if last_tracks:
            return self.detectors[-1].name, attempts, last_tracks
        if attempts and all("skip" in a for a in attempts):
            raise AdapterUnavailable(
                "no detector could run. " + " | ".join(attempts)
            )
        return "none", attempts, []

    def _ensure_ready(self) -> None:
        if hasattr(self.parser, "status"):
            st = self.parser.status()
            if not st.startswith("ready"):
                raise AdapterUnavailable(f"prompt-parser llm: {st}")
        notes: list[str] = []
        ready = False
        for detector in self.detectors:
            st = detector.status()
            notes.append(f"{detector.name}: {st}")
            if st.startswith("ready"):
                ready = True
        if self.detectors and not ready:
            raise AdapterUnavailable("no detector could run. " + " | ".join(notes))


def _target_json(t) -> dict:
    return {
        "kind": t.kind,
        "query": t.query,
        "part": t.part,
        "motion": t.motion,
        "where": t.where,
        "ordinal": t.ordinal,
        "from_side": t.from_side,
    }


def _target_label(t) -> str:
    loc = t.where or t.part or "whole"
    extra = f":{t.ordinal}{t.from_side or ''}" if t.ordinal else ""
    return f"{t.query} [{loc}{extra}]"


def _or_mask(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros_like(a)
    out[a > 0] = 255
    out[b > 0] = 255
    return out
