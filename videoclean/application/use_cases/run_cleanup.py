from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from videoclean.application.config import PipelineConfig, RunCleanupRequest
from videoclean.application.errors import AdapterUnavailable, JobCancelled, PipelineError
from videoclean.application.inpaint_runtime import (
    inpaint_clip_chunked,
    inpaint_frames,
    re_inpaint_ranges,
    resolve_workers,
)
from videoclean.application.ports.detector import Detector
from videoclean.application.ports.inpainter import Inpainter
from videoclean.application.ports.jobs import JobStore
from videoclean.application.ports.media import MediaGateway
from videoclean.application.ports.progress import ProgressPort
from videoclean.application.ports.prompt import PromptParser
from videoclean.application.ports.segmenter import Segmenter
from videoclean.application.frames_sample import sample_frame_indices
from videoclean.application.frames import LazyFrames
from videoclean.application.select import explain_unmatched, select_tracks
from videoclean.application.verify_quality import (
    dirty_ranges,
    masks_grew,
    or_masks,
    residual_unchanged_mask,
)
from videoclean.domain.formats import parse_formats, resolve_dest
from videoclean.domain.intent import Intent, Target
from videoclean.domain.tracks import Detection, Track, interpolate_gaps, tracks_from_json, tracks_to_json


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
        if (
            not (req.prompt or "").strip()
            and not req.targets_override
            and not req.tracks_override
            and not req.masks_override
        ):
            raise PipelineError("--prompt is required (or pass manual targets/tracks/masks)")
        if not req.input_path.is_file():
            raise FileNotFoundError(req.input_path)
        # Cleanup always emits baseline mp4; extra delivery formats are packaged on demand.
        dest = resolve_dest(req.output_path, "mp4", ["mp4"])
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
        except JobCancelled:
            report["state"] = "CANCELLED"
            report["error"] = "cancelled"
            report["finishedAt"] = self._utc_now().isoformat()
            try:
                paths.report_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
            except OSError:
                pass
            self.jobs.upsert(job_id, "CANCELLED", report=report, error="cancelled")
            raise
        except Exception as exc:
            report["state"] = "FAILED"
            report["error"] = str(exc)
            report["finishedAt"] = self._utc_now().isoformat()
            try:
                paths.report_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
            except OSError:
                pass
            self.jobs.upsert(job_id, "FAILED", report=report, error=str(exc)[:1500])
            raise
        finally:
            prev = getattr(self, "_thread_prev", None)
            if prev is not None:
                from videoclean.application.budget import restore_thread_caps

                restore_thread_caps(prev)
                self._thread_prev = None

    def _run(self, req: RunCleanupRequest, cfg: PipelineConfig, job_id: str, paths, manifest, report: dict) -> dict:
        self._ensure_ready(req)
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

        self.progress.start("normalize", total=manifest.frame_count, detail="ffmpeg raw frames")
        images = self._open_frames(req, paths, manifest)
        frames = images
        self.progress.tick("normalize", len(frames), len(frames))
        self.progress.finish("normalize", f"{len(frames)} frames")

        (paths.root / "analysis").mkdir(exist_ok=True)
        first = images[0]
        h, w = first.shape[:2]
        self._prepare_runtime(cfg, paths, manifest)

        used_detector: str = "manual"
        attempts: list[str] = []
        if req.masks_override:
            policy = (req.mask_policy or "static").strip().lower()
            if policy not in {"static", "propagate"}:
                raise PipelineError(f"mask_policy must be static | propagate, got {policy!r}")
            intent = self._intent_for_masks(req, images, frames, cfg)
            self.progress.start("detect", total=len(frames), detail=f"manual masks ({policy})")
            if policy == "propagate":
                tracks = self._tracks_from_mask_anchors(req.masks_override, len(frames), (h, w))
                if not tracks:
                    raise PipelineError("masks_override: no usable mask anchors for propagate")
                _label_tracks_from_intent(tracks, intent)
                for tr in tracks:
                    interpolate_gaps(tr)
                masks = self._segment_masks(images, tracks, stage="detect")
                if cfg.mask_dilate_px > 0:
                    kernel = cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE, (cfg.mask_dilate_px * 2 + 1, cfg.mask_dilate_px * 2 + 1)
                    )
                    masks = [cv2.dilate(m, kernel) if m is not None else m for m in masks]
                self.progress.finish("detect", f"propagate: {len(tracks)} tracks")
            else:
                base = self._or_masks(req.masks_override, (h, w))
                if base is None:
                    raise PipelineError("masks_override: no readable mask images")
                if cfg.mask_dilate_px > 0:
                    kernel = cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE, (cfg.mask_dilate_px * 2 + 1, cfg.mask_dilate_px * 2 + 1)
                    )
                    base = cv2.dilate(base, kernel)
                masks = [base.copy() for _ in frames]
                tracks = []
                self.progress.finish("detect", "manual masks (static)")
        elif req.tracks_override:
            self.progress.start("parse", detail="manual tracks")
            try:
                tracks = tracks_from_json(req.tracks_override)
            except ValueError as exc:
                raise PipelineError(f"tracks_override: {exc}") from exc
            n_frames = len(frames)
            for tr in tracks:
                if len(tr.boxes) != n_frames:
                    raise PipelineError(
                        f"tracks_override: track {tr.track_id} has {len(tr.boxes)} boxes, "
                        f"video has {n_frames} frames (full-length tracks required)"
                    )
                interpolate_gaps(tr)
            labels = list(dict.fromkeys(tr.label for tr in tracks))
            intent = Intent(
                targets=[Target(kind="object", query=lb) for lb in labels] or [Target(kind="object", query="target")],
                parse_mode="manual-tracks",
                raw=req.prompt or "",
            )
            self.progress.finish("parse", ", ".join(labels))
            self.progress.start("detect", total=len(frames), detail="manual tracks")
            masks = self._segment_masks(images, tracks, stage="detect")
            self.progress.finish("detect", f"manual: {len(tracks)} tracks")
        else:
            parse_st = self.parser.status() if hasattr(self.parser, "status") else "llm"
            sample_idxs = sample_frame_indices(len(frames), cfg.prompt_frame_stride, cfg.prompt_frame_max)
            sample_frames = [images[i] for i in sample_idxs] if sample_idxs else None
            if req.targets_override:
                from videoclean.application.use_cases.run_preview import targets_from_json

                targets = targets_from_json(req.targets_override)
                intent = Intent(targets=targets, parse_mode="manual", raw=req.prompt or "")
                detail = f"manual targets: {len(targets)}"
                label = ", ".join(_target_label(t) for t in intent.targets)
            else:
                detail = f"llm  {parse_st}"
                if sample_idxs:
                    detail += f"  vision frames={len(sample_idxs)} stride={cfg.prompt_frame_stride}"
                self.progress.start("parse", detail=detail)
                intent = self.parser.parse(
                    req.prompt,
                    frames=sample_frames,
                    frame_indices=sample_idxs,
                    parse_chunk_frames=cfg.parse_chunk_frames,
                )
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
            if intent.parse_mode != "manual":
                if intent.defaulted:
                    label += " (default)"
                if intent.parse_mode == "llm-vision":
                    label += f"  (vision×{len(sample_idxs)})"
            self.progress.finish("parse", label)

            self.progress.start("detect", total=len(frames), detail="read frames")
            self._apply_caption_box_area(intent)
            used_detector, attempts, raw_tracks = self._discover(images, intent.queries, stage="detect")
            strict = select_tracks(raw_tracks, intent, width=w, height=h, relax=False)
            if strict:
                tracks = strict
                select_relaxed = False
            elif cfg.select_relax:
                tracks = select_tracks(raw_tracks, intent, width=w, height=h, relax=True)
                select_relaxed = bool(tracks)
            else:
                tracks = []
                select_relaxed = False
            if select_relaxed:
                for tr in tracks:
                    tr.notes = list(tr.notes) + ["relaxed-match"]
                report["selectRelaxed"] = True
            if raw_tracks and not tracks:
                raise PipelineError(
                    explain_unmatched(raw_tracks, intent, w, h) + f" detectors tried: {attempts}"
                )
            if not raw_tracks:
                raise PipelineError(
                    f"detector found no boxes for queries {intent.queries} "
                    f"({', '.join(_target_label(t) for t in intent.targets)}). "
                    f"detectors tried: {attempts}"
                )
            (paths.root / "analysis" / "tracks_raw.json").write_text(
                json.dumps(tracks_to_json(raw_tracks), indent=2),
                encoding="utf-8",
            )
            (paths.root / "analysis" / "detector.json").write_text(
                json.dumps(
                    {"used": used_detector, "attempts": attempts, "selectRelaxed": select_relaxed},
                    indent=2,
                ),
                encoding="utf-8",
            )
            masks = self._segment_masks(images, tracks, stage="detect")

        if tracks:
            (paths.root / "analysis" / "tracks.json").write_text(
                json.dumps(tracks_to_json(tracks), indent=2), encoding="utf-8"
            )
        for i, mask in enumerate(masks):
            if req.keep_workdir:
                self._write_image(paths.masks_dir / f"frame_{i:06d}.png", mask)
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

        overlap = int(cfg.inpaint_chunk_overlap)
        if not callable(getattr(self.inpainter, "inpaint_masked", None)):
            raise AdapterUnavailable(
                f"{getattr(self.inpainter, 'name', 'inpainter')} has no inpaint_masked; "
                "full-frame inpaint is not used"
            )
        self.progress.start("inpaint", total=len(frames), detail=f"{self.inpainter.name} crop")
        cleaned = self._paint_holes(cfg, images, masks, tracks, paths)
        self.progress.finish("inpaint", f"{self.inpainter.name} crop")

        verify_passes = 0
        verify_note = "skipped"
        if cfg.verify and not req.masks_override and cfg.verify_max_passes > 0:
            self.progress.start("verify", total=len(frames), detail="leftover pass")
            working_masks = list(masks)
            cap = max(cfg.verify_max_coverage, mean_cov * 1.5)
            max_passes = int(cfg.verify_max_passes)
            v_attempts = "residual"
            for pass_i in range(1, max_passes + 1):
                self.progress.tick("verify", 0, 1, f"pass {pass_i}/{max_passes} residual")
                residual = [
                    residual_unchanged_mask(images[i], cleaned[i], working_masks[i])
                    for i in range(len(frames))
                ]
                if cfg.verify_redetect:
                    _, v_attempts, leftover = self._discover(cleaned, intent.queries, stage="verify")
                    leftover_strict = select_tracks(leftover, intent, width=w, height=h, relax=False)
                    leftover = leftover_strict or (
                        select_tracks(leftover, intent, width=w, height=h, relax=True)
                        if cfg.select_relax
                        else []
                    )
                    detect_masks = (
                        self._segment_masks(cleaned, leftover, stage="verify")
                        if leftover
                        else [np.zeros_like(working_masks[0]) for _ in working_masks]
                    )
                else:
                    detect_masks = [np.zeros_like(working_masks[0]) for _ in working_masks]
                grown = [
                    or_masks(or_masks(prev, det), res)
                    for prev, det, res in zip(working_masks, detect_masks, residual)
                ]
                still = float(np.mean([np.count_nonzero(m) / m.size for m in grown])) if grown else 0.0
                grew_flags = masks_grew(working_masks, grown)
                if not any(grew_flags):
                    verify_note = f"clean after {pass_i - 1} re-inpaint ({v_attempts})"
                    break
                if still > cap:
                    verify_note = (
                        f"leftover {still:.2%} above cap {cap:.2%} — skip re-inpaint ({v_attempts})"
                    )
                    break
                ranges = dirty_ranges(grew_flags, pad=max(4, overlap))
                self.progress.tick(
                    "verify",
                    0,
                    len(frames),
                    f"pass {pass_i} re-inpaint {len(ranges)} range(s) cov={still:.2%}",
                )
                self._repaint_ranges(cfg, images, grown, cleaned, ranges, tracks)
                working_masks = grown
                verify_passes = pass_i
                verify_note = f"pass {pass_i}, leftover {still:.2%}, ranges={len(ranges)}"
            if req.keep_workdir and verify_passes > 0:
                for i, mask in enumerate(working_masks):
                    self._write_image(paths.masks_dir / f"verify_frame_{i:06d}.png", mask)
            self.progress.finish("verify", verify_note)
        else:
            if req.masks_override:
                verify_note = "skipped (manual masks)"
            elif not cfg.verify:
                verify_note = "skipped"
            else:
                verify_note = "skipped (verify_max_passes=0)"
            self.progress.start("verify")
            self.progress.finish("verify", verify_note)

        self.progress.start("encode", detail="ffmpeg mezzanine")
        mezz_work = paths.output_dir / "mezzanine.mp4"
        encode_raw = getattr(type(self.media), "encode_from_store", None)
        if encode_raw is not None and hasattr(cleaned, "frame_bytes"):
            self.media.encode_from_store(
                cleaned,
                req.input_path,
                mezz_work,
                manifest.fps_ratio,
                manifest.has_audio,
                paths.ffmpeg_log,
                nvenc=bool(getattr(self._budget, "nvenc", False)),
            )
        else:
            for i in range(len(cleaned)):
                self._write_image(paths.inpainted_dir / f"frame_{i:06d}.jpg", cleaned[i])
            self.media.encode_mezzanine(
                paths.inpainted_dir,
                req.input_path,
                mezz_work,
                manifest.fps_ratio,
                manifest.has_audio,
                len(frames),
                paths.ffmpeg_log,
            )
        # Keep master outside process/ so keep_workdir=false does not wipe it.
        mezz = paths.output_dir / "mezzanine.mp4"
        if mezz_work.resolve() != mezz.resolve():
            shutil.copy2(mezz_work, mezz)
        self.progress.finish("encode", mezz.name)

        # Always keep baseline mp4; also package any extra formats requested on this run.
        requested = parse_formats(cfg.formats or ["mp4"])
        if "mp4" not in requested:
            # Baseline mp4 stays available for playback / mezzanine consumers.
            delivery = ["mp4", *requested]
        else:
            delivery = list(requested)
        self.progress.start("package", total=len(delivery), detail=delivery[0])
        outputs: dict[str, str] = {}
        for i, fmt in enumerate(delivery, start=1):
            dest = resolve_dest(req.output_path, fmt, delivery)
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
                segment_seconds=int(getattr(cfg, "segment_seconds", 6) or 6),
                webm_crf=int(getattr(cfg, "webm_crf", 32) or 32),
            )
            outputs[fmt] = str(artifact)
            self.progress.tick("package", i, len(delivery), fmt)
        self.progress.finish("package", ",".join(delivery))

        self.progress.start("report")
        report.update(
            {
                "state": "COMPLETED",
                "finishedAt": self._utc_now().isoformat(),
                "targets": [_target_json(t) for t in intent.targets],
                "verify": cfg.verify,
                "verifyPasses": verify_passes,
                "verifyNote": verify_note,
                "profile": cfg.profile,
                "presetId": getattr(cfg, "preset_id", None),
                "promptParseMode": intent.parse_mode,
                "detectorUsed": used_detector,
                "detectorAttempts": attempts,
                "frames": len(frames),
                "meanMaskCoverage": mean_cov,
                "detections": [
                    {"label": d.label, "coverage": d.coverage, "kind": d.mask_kind, "notes": d.notes}
                    for d in detections
                ],
                "mezzanine": str(mezz),
                "formats": delivery,
                "outputs": outputs,
                "output": next(iter(outputs.values())),
                "workdir": str(paths.root),
                "timings": self._timings(paths),
                "budget": self._budget_report(),
                "hole": getattr(self, "_hole_report", {}),
            }
        )
        paths.report_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        self.progress.finish("report", "ok")
        self.jobs.upsert(job_id, "COMPLETED", report=report)
        if not req.keep_workdir:
            for store in (images, cleaned):
                close = getattr(store, "close", None)
                if close:
                    close()
            for name in ("frames.bgr", "cleaned.bgr", "sam2.f32"):
                target = paths.root / name
                if target.is_file():
                    target.unlink(missing_ok=True)
            shutil.rmtree(paths.frames_dir, ignore_errors=True)
            shutil.rmtree(paths.inpainted_dir, ignore_errors=True)
            shutil.rmtree(paths.masks_dir, ignore_errors=True)
            shutil.rmtree(paths.input_dir, ignore_errors=True)
            process = paths.root / "process"
            if process.is_dir():
                shutil.rmtree(process, ignore_errors=True)
        return report

    def _open_frames(self, req: RunCleanupRequest, paths, manifest):
        decode = getattr(type(self.media), "decode_bgr", None)
        n = int(manifest.frame_count or 0)
        h = int(manifest.height or 0)
        w = int(manifest.width or 0)
        if decode is None or n < 1 or h < 1 or w < 1:
            extracted = self.media.extract_frames(req.input_path, paths.frames_dir, paths.ffmpeg_log)
            cache_n = min(48, max(8, len(extracted)))
            return LazyFrames(extracted, self._read_image, cache_size=cache_n)
        from videoclean.adapters.media.raw_store import FrameStore, assert_local_data_dir

        assert_local_data_dir(paths.root)
        raw_bytes = n * h * w * 3
        self._assert_disk(paths.root, raw_bytes * 2)
        store = FrameStore(paths.root / "frames.bgr", n, h, w)
        written = self.media.decode_bgr(req.input_path, store, paths.ffmpeg_log)
        if written != n:
            raise PipelineError(f"decode wrote {written} frames, probe says {n}")
        return store

    def _assert_disk(self, root: Path, nbytes: int) -> None:
        usage = shutil.disk_usage(root)
        if nbytes > usage.free // 2:
            raise PipelineError(
                f"need {nbytes} bytes under {root}, free {usage.free}. "
                "A second full-frame copy is not allocated."
            )

    def _prepare_runtime(self, cfg: PipelineConfig, paths, manifest) -> None:
        import time

        from videoclean.application.budget import TorchProbe, apply_thread_caps, resolve_budget

        t0 = time.monotonic()
        loaded_now = False
        for owner in (*self.detectors, self.segmenter, self.inpainter):
            ensure = getattr(owner, "_ensure", None) or getattr(owner, "_try_load", None)
            if ensure is None:
                continue
            try:
                ensure()
            except Exception:
                pass
            if getattr(owner, "loaded_now", False):
                loaded_now = True
        self._load_s = time.monotonic() - t0
        self._warm = not loaded_now
        self._budget = resolve_budget(cfg, probe=TorchProbe())
        self._thread_prev = apply_thread_caps(self._budget.cpu_threads)
        if cfg.device == "cuda":
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                    self.inpainter.activation_baseline_bytes = int(torch.cuda.memory_allocated())
            except Exception:  # noqa: BLE001
                pass
        self.inpainter.vram_budget_bytes = self._budget.vram_budget_bytes
        self.inpainter.inpaint_max_side = self._budget.inpaint_max_side
        if getattr(self.segmenter, "name", "") == "sam2-video":
            image_size = 1024
            n = int(manifest.frame_count or 0)
            tensor_bytes = n * 3 * image_size * image_size * 4
            self._assert_disk(paths.root, tensor_bytes)
            self.segmenter.tensor_path = paths.root / "sam2.f32"
        if self._budget.nvenc:
            self._encoder = ("h264_nvenc", "p4", 18)
        else:
            self._encoder = ("libx264", "veryfast", 18)

    def _paint_holes(self, cfg, images, masks, tracks, paths):
        from videoclean.adapters.media.raw_store import FrameStore
        from videoclean.application.hole_policy import mask_box, mask_coverage, plan_holes

        n = len(images)
        h, w = images[0].shape[:2]
        if hasattr(images, "frame_bytes"):
            cleaned = FrameStore(paths.root / "cleaned.bgr", n, h, w)
        else:
            cleaned = [None] * n
        boxes = [mask_box(m) for m in masks]
        coverages = [mask_coverage(m) for m in masks]
        motions = [getattr(tr, "motion", "static") for tr in tracks]
        family = "propainter" if cfg.inpainter == "propainter" else "lama"
        plans = plan_holes(
            frame_hw=(h, w),
            mask_coverage=coverages,
            mask_boxes=boxes,
            track_motions=motions,
            family=family,
            device=cfg.device,
            budget_bytes=int(getattr(self._budget, "vram_budget_bytes", 0) or 0),
            inpaint_max_side=getattr(cfg, "inpaint_max_side", None),
            neighbor_length=int(cfg.propainter_neighbor_length),
            subvideo_length=int(cfg.propainter_subvideo_length),
            batch_cap=self._budget.inpaint_workers_cap,
        )
        if not plans:
            raise PipelineError("hole policy produced no range")
        stats = []
        for plan in plans:
            step = plan.end - plan.start if plan.policy == "propainter-crop" else 16
            cursor = plan.start
            while cursor < plan.end:
                stop = min(plan.end, cursor + step)
                window_f = [images[i] for i in range(cursor, stop)]
                window_m = masks[cursor:stop]
                painted = self.inpainter.inpaint_masked(window_f, window_m, plan)
                if len(painted) != stop - cursor:
                    raise PipelineError("inpaint_masked returned the wrong number of frames")
                for offset, frame in enumerate(painted):
                    cleaned[cursor + offset] = frame
                stats.append(getattr(self.inpainter, "last_hole_stats", {}))
                del window_f, painted
                cursor = stop
        primary = plans[0]
        last = stats[-1] if stats else {}
        self._hole_report = {
            "policy": primary.policy,
            "meanCoverage": primary.mean_coverage,
            "fastMoving": primary.fast_moving,
            "almostFull": primary.almost_full,
            "side": primary.side,
            "batch": max((int(s.get("batch") or primary.batch) for s in stats), default=primary.batch),
            "frameCount": primary.end - primary.start,
            "featherPx": primary.feather_px,
            "contextPx": primary.context_px,
            "limitedBy": last.get("limitedBy") or primary.limited_by,
            "oomRetry": any(bool(s.get("oomRetry")) for s in stats),
            "lamaDtype": last.get("lamaDtype") or "fp32",
            "ranges": [
                {
                    "start": p.start,
                    "end": p.end,
                    "policy": p.policy,
                    "side": p.side,
                    "batch": p.batch,
                    "limitedBy": p.limited_by,
                }
                for p in plans
            ],
        }
        return cleaned

    def _repaint_ranges(self, cfg, images, masks, cleaned, ranges, tracks) -> None:
        from videoclean.application.hole_policy import mask_box, mask_coverage, plan_holes

        h, w = images[0].shape[:2]
        family = "propainter" if cfg.inpainter == "propainter" else "lama"
        for start, end in ranges:
            window_m = masks[start:end]
            plans = plan_holes(
                frame_hw=(h, w),
                mask_coverage=[mask_coverage(m) for m in window_m],
                mask_boxes=[mask_box(m) for m in window_m],
                track_motions=[getattr(tr, "motion", "static") for tr in tracks],
                family=family,
                device=cfg.device,
                budget_bytes=int(getattr(self._budget, "vram_budget_bytes", 0) or 0),
                inpaint_max_side=getattr(cfg, "inpaint_max_side", None),
                batch_cap=self._budget.inpaint_workers_cap,
            )
            for plan in plans:
                abs_start = start + plan.start
                abs_end = start + plan.end
                painted = self.inpainter.inpaint_masked(
                    [images[i] for i in range(abs_start, abs_end)],
                    masks[abs_start:abs_end],
                    plan,
                )
                for offset, frame in enumerate(painted):
                    cleaned[abs_start + offset] = frame

    def _timings(self, paths) -> dict:
        seconds = {}
        stage_seconds = getattr(self.progress, "stage_seconds", None)
        if stage_seconds:
            seconds = stage_seconds()
        detect = 0.0
        track = 0.0
        for detector in self.detectors:
            detect += float(getattr(detector, "last_detect_s", 0.0) or 0.0)
            track += float(getattr(detector, "last_track_s", 0.0) or 0.0)
        segment = float(getattr(self, "_segment_s", 0.0) or 0.0)
        return {
            "load": round(float(getattr(self, "_load_s", 0.0) or 0.0), 4),
            "decode": seconds.get("normalize"),
            "parse": seconds.get("parse", 0.0),
            "detect": round(detect, 4),
            "track": round(track, 4),
            "segment": round(segment, 4),
            "inpaint": seconds.get("inpaint"),
            "verify": seconds.get("verify"),
            "encode": seconds.get("encode"),
            "package": seconds.get("package"),
        }

    def _budget_report(self) -> dict:
        budget = getattr(self, "_budget", None)
        if budget is None:
            return {}
        encoder, preset, crf = getattr(self, "_encoder", ("libx264", "veryfast", 18))
        peak_load = int(getattr(self.inpainter, "activation_baseline_bytes", 0) or 0)
        peak_inpaint = int(getattr(self.inpainter, "activation_delta_bytes", 0) or 0)

        def mb(n: int) -> int:
            return int(n / (1024 * 1024))

        return {
            "deviceRequested": budget.device_requested,
            "device": budget.device,
            "vramTotalMb": mb(budget.vram_total_bytes),
            "vramFreeMb": mb(budget.vram_free_bytes),
            "vramCeilingMb": None if budget.vram_ceiling_bytes is None else mb(budget.vram_ceiling_bytes),
            "vramBudgetMb": mb(budget.vram_budget_bytes),
            "vramPeakLoadMb": mb(peak_load),
            "vramPeakInpaintMb": mb(peak_inpaint),
            "vramSource": budget.vram_source,
            "cpuCount": budget.cpu_count,
            "cpuThreads": budget.cpu_threads,
            "inpaintWorkersCap": budget.inpaint_workers_cap,
            "inpaintMaxSide": budget.inpaint_max_side,
            "nvenc": budget.nvenc,
            "encoder": encoder,
            "encoderPreset": preset,
            "crf": crf,
            "warm": bool(getattr(self, "_warm", False)),
        }

    def _intent_for_masks(self, req: RunCleanupRequest, images, frames, cfg: PipelineConfig) -> Intent:
        """Labels for mask anchors: targets_override, else parse prompt, else empty (entry B)."""
        policy = (req.mask_policy or "static").strip().lower()
        if req.targets_override:
            from videoclean.application.use_cases.run_preview import targets_from_json

            targets = targets_from_json(req.targets_override)
            intent = Intent(
                targets=targets,
                parse_mode="manual-masks+targets",
                raw=req.prompt or "",
            )
            self.progress.start("parse", detail=f"masks+targets ({policy})")
            self.progress.finish("parse", f"{len(req.masks_override)} masks · {len(targets)} targets")
            return intent
        if (req.prompt or "").strip():
            parse_st = self.parser.status() if hasattr(self.parser, "status") else "llm"
            sample_idxs = sample_frame_indices(len(frames), cfg.prompt_frame_stride, cfg.prompt_frame_max)
            sample_frames = [images[i] for i in sample_idxs] if sample_idxs else None
            self.progress.start("parse", detail=f"masks+prompt  {parse_st}")
            intent = self.parser.parse(
                req.prompt,
                frames=sample_frames,
                frame_indices=sample_idxs,
                parse_chunk_frames=cfg.parse_chunk_frames,
            )
            intent = Intent(
                targets=intent.targets,
                parse_mode=f"masks+{intent.parse_mode}",
                raw=intent.raw or req.prompt,
                defaulted=intent.defaulted,
            )
            self.progress.finish(
                "parse",
                f"{len(req.masks_override)} masks · " + (", ".join(_target_label(t) for t in intent.targets) or "no targets"),
            )
            return intent
        self.progress.start("parse", detail=f"manual masks ({policy})")
        intent = Intent(targets=[], parse_mode="manual-masks", raw=req.prompt or "")
        self.progress.finish("parse", f"{len(req.masks_override)} masks")
        return intent

    def _apply_caption_box_area(self, intent: Intent) -> None:
        """Raise detector max_box_area when hunting text/captions (wide lower-thirds)."""
        kinds = {t.kind for t in intent.targets}
        if not kinds.intersection({"text_overlay", "watermark"}):
            return
        for detector in self.detectors:
            cur = float(getattr(detector, "max_box_area", 0.45) or 0.45)
            detector.max_box_area = max(cur, 0.55)

    def _segment_masks(self, images, tracks, *, stage: str):
        import time

        self.progress.tick(stage, 0, max(len(images), 1), f"{self.segmenter.name} masks")
        t0 = time.monotonic()
        masks = self.segmenter.masks(
            images,
            tracks,
            on_progress=lambda cur, tot, detail="": self.progress.tick(stage, cur, tot, detail),
        )
        self._segment_s = float(getattr(self, "_segment_s", 0.0) or 0.0) + (time.monotonic() - t0)
        return masks

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

    def _ensure_ready(self, req: RunCleanupRequest) -> None:
        if req.masks_override:
            policy = (req.mask_policy or "static").strip().lower()
            if policy == "propagate":
                if not self.segmenter or not getattr(self.segmenter, "name", ""):
                    raise AdapterUnavailable("segmenter unavailable for mask propagate")
            return
        if req.tracks_override:
            if not self.segmenter or not getattr(self.segmenter, "name", ""):
                raise AdapterUnavailable("segmenter unavailable for manual tracks")
            return
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

    def _or_masks(self, mask_paths: list[str], shape: tuple[int, int]) -> np.ndarray | None:
        """OR all user masks; resize to frame shape."""
        out: np.ndarray | None = None
        for raw in mask_paths or []:
            m = cv2.imdecode(np.fromfile(str(raw), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            if m.shape != (shape[0], shape[1]):
                m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
            out = m if out is None else np.maximum(out, m)
        return out

    def _tracks_from_mask_anchors(
        self,
        mask_paths: list[str],
        n_frames: int,
        shape: tuple[int, int],
    ) -> list[Track]:
        """Build one track from per-frame mask PNGs (stem = absolute frame index)."""
        boxes: list[tuple[int, int, int, int] | None] = [None] * n_frames
        for raw in mask_paths or []:
            path = Path(str(raw))
            try:
                idx = int(path.stem)
            except ValueError:
                continue
            if idx < 0 or idx >= n_frames:
                continue
            m = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            if m.shape != (shape[0], shape[1]):
                m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
            box = _bbox_from_mask(m)
            if box is not None:
                boxes[idx] = box
        if not any(b is not None for b in boxes):
            return []
        return [
            Track(
                track_id=0,
                label="mask",
                boxes=boxes,
                scores=[1.0 if b is not None else 0.0 for b in boxes],
                motion="static",
                notes=["mask-propagate"],
            )
        ]


def _label_tracks_from_intent(tracks: list[Track], intent: Intent) -> None:
    """Apply entry-C target query/kind onto mask-propagated tracks."""
    if not tracks or not intent.targets:
        return
    t0 = intent.targets[0]
    label = (t0.query or (intent.queries[0] if intent.queries else "mask")) or "mask"
    for i, tr in enumerate(tracks):
        t = intent.targets[i] if i < len(intent.targets) else t0
        tr.label = (t.query or label).strip() or label
        notes = list(tr.notes)
        notes.append(f"kind={t.kind}")
        if t.where:
            notes.append(f"where={t.where}")
        tr.notes = notes


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


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
