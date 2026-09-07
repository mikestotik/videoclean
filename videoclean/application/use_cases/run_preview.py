from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from videoclean.application.config import PipelineConfig
from videoclean.application.errors import JobCancelled, PipelineError
from videoclean.application.ports.detector import Detector
from videoclean.application.ports.jobs import JobStore
from videoclean.application.ports.media import MediaGateway
from videoclean.application.ports.progress import ProgressPort
from videoclean.application.ports.prompt import PromptParser
from videoclean.application.ports.segmenter import Segmenter
from videoclean.application.select import explain_unmatched, select_tracks
from videoclean.domain.intent import Target
from videoclean.domain.tracks import tracks_to_json


@dataclass
class PreviewRequest:
    input_path: Path
    prompt: str
    config: PipelineConfig
    start: int | None = 0
    count: int | None = 16
    stride: int | None = None
    indices: list[int] | None = None
    mode: str = "parse"  # parse | detect
    targets: list[dict] | None = None
    job_id: str | None = None


@dataclass
class PreviewPaths:
    root: Path
    frames_dir: Path
    preview_dir: Path
    logs_dir: Path
    ffmpeg_log: Path
    report_file: Path
    input_dir: Path

    @classmethod
    def create(cls, root: Path) -> "PreviewPaths":
        return cls(
            root=root,
            frames_dir=root / "frames",
            preview_dir=root / "preview",
            logs_dir=root / "logs",
            ffmpeg_log=root / "logs" / "ffmpeg.log",
            report_file=root / "output" / "report.json",
            input_dir=root / "input",
        )


def indices_from_request(req: PreviewRequest, n_video: int) -> list[int]:
    if req.indices:
        return sorted({int(i) for i in req.indices if 0 <= int(i) < n_video})
    start = max(0, int(req.start or 0))
    count = max(1, int(req.count or 16))
    stride = max(1, int(req.stride or 1)) if req.stride else 1
    return [i for i in range(start, min(n_video, start + count * stride), stride)][:count]


def targets_from_json(data: list[dict]) -> list[Target]:
    out: list[Target] = []
    for item in data or []:
        if not isinstance(item, dict):
            continue
        query = " ".join(str(item.get("query") or "").split())
        if not query:
            continue
        kind = str(item.get("kind") or "object")
        if kind not in {"watermark", "text_overlay", "object"}:
            kind = "object"
        out.append(
            Target(
                kind=kind,
                query=query,
                motion=str(item.get("motion") or "any"),
                where=item.get("where") or None,
                ordinal=item.get("ordinal"),
                from_side=item.get("from_side"),
            )
        )
    if not out:
        raise PipelineError("no usable targets in preview request")
    return out


_TEXT_WORDS = {"text", "caption", "title", "subtitle", "headline", "lettering", "inscription"}
_MARK_WORDS = {"logo", "watermark", "emblem", "bug"}


def _kind_for_query(query: str) -> str:
    low = query.casefold()
    if any(w in low for w in _MARK_WORDS):
        return "watermark"
    if any(w in low for w in _TEXT_WORDS):
        return "text_overlay"
    return "object"


def parse_queries_arg(raw: str) -> list[Target]:
    """Parse `text [bottom], logo, red mug` into targets. Where is optional."""
    from videoclean.domain.intent import SCREEN_WHERES

    out: list[Target] = []
    for chunk in (raw or "").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        where = None
        query = chunk
        if chunk.endswith("]") and "[" in chunk:
            query, bracket = chunk.rsplit("[", 1)
            candidate = bracket.rstrip("]").strip().casefold()
            if candidate in SCREEN_WHERES:
                where = candidate
        query = " ".join(query.split())
        if not query:
            continue
        out.append(Target(kind=_kind_for_query(query.casefold()), query=query, where=where))
    if not out:
        raise PipelineError("--queries: nothing parsed")
    return out


class RunPreview:
    """Detect + segment on a frame subset. No inpaint, no encode — masks only."""

    def __init__(
        self,
        *,
        media: MediaGateway,
        parser: PromptParser,
        detectors: list[Detector],
        segmenter: Segmenter,
        jobs: JobStore,
        progress: ProgressPort,
        new_job_id: callable,
        make_paths: callable,
        read_image: callable,
        write_image: callable,
    ) -> None:
        self.media = media
        self.parser = parser
        self.detectors = detectors
        self.segmenter = segmenter
        self.jobs = jobs
        self.progress = progress
        self._new_job_id = new_job_id
        self._make_paths = make_paths
        self._read_image = read_image
        self._write_image = write_image

    def execute(self, req: PreviewRequest, data_dir: Path) -> dict:
        if req.mode == "detect" and not req.targets:
            raise PipelineError("preview mode=detect requires targets[]")
        if not req.input_path.is_file():
            raise FileNotFoundError(req.input_path)
        job_id = req.job_id or self._new_job_id()
        paths = self._make_paths(data_dir / "jobs" / job_id)
        self.jobs.upsert(job_id, "RUNNING", input_path=str(req.input_path), prompt=req.prompt)
        report: dict = {
            "jobId": job_id,
            "kind": "preview",
            "state": "RUNNING",
            "prompt": req.prompt,
            "mode": req.mode,
        }
        try:
            out = self._run(req, job_id, paths, report)
            self.jobs.upsert(job_id, "COMPLETED", report=out)
            return out
        except (JobCancelled, Exception) as exc:
            report["state"] = "CANCELLED" if isinstance(exc, JobCancelled) else "FAILED"
            report["error"] = str(exc)
            try:
                paths.report_file.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
            except OSError:
                pass
            self.jobs.upsert(job_id, report["state"], report=report, error=str(exc)[:1500])
            raise

    def _run(self, req: PreviewRequest, job_id: str, paths: PreviewPaths, report: dict) -> dict:
        manifest = self.media.probe(req.input_path)
        n_video = manifest.frame_count
        idxs = indices_from_request(req, n_video)
        if not idxs:
            raise PipelineError("preview: requested frames are out of range")

        paths.preview_dir.mkdir(parents=True, exist_ok=True)
        self.progress.start("extract", total=len(idxs))
        frame_paths = self.media.extract_frames_subset(
            req.input_path, idxs, paths.frames_dir, paths.ffmpeg_log
        )
        images = [self._read_image(p) for p in frame_paths]
        self.progress.finish("extract", f"{len(images)} frames")

        if req.mode == "detect":
            targets = targets_from_json(req.targets)
            intent = None
            queries: list[str] = []
            report["targets"] = [_t_json(t) for t in targets]
        else:
            parse_st = self.parser.status() if hasattr(self.parser, "status") else "llm"
            self.progress.start("parse", detail=f"llm {parse_st}")
            intent = self.parser.parse(
                req.prompt,
                frames=images,
                frame_indices=idxs,
                parse_chunk_frames=req.config.parse_chunk_frames,
            )
            targets = intent.targets
            queries = intent.queries
            report["targets"] = [_t_json(t) for t in targets]
            report["parseMode"] = intent.parse_mode
            self.progress.finish("parse", ", ".join(queries) or "(none)")

        self.progress.start("detect", total=len(images))
        used, attempts, raw_tracks = self._discover(images, queries or [t.query for t in targets])
        selected = select_tracks(raw_tracks, _targets_as_intent(targets), width=manifest.width, height=manifest.height, relax=True)
        if raw_tracks and not selected:
            raise PipelineError(explain_unmatched(raw_tracks, _targets_as_intent(targets), manifest.width, manifest.height))
        if not raw_tracks:
            raise PipelineError(f"detector found no boxes for queries {queries or [t.query for t in targets]}")

        masks = self.segmenter.masks(images, selected)
        dilate = req.config.mask_dilate_px

        per_frame: list[dict] = []
        for pos, (img, idx, mask) in enumerate(zip(images, idxs, masks)):
            cov = float(np.count_nonzero(mask) / mask.size) if mask.size else 0.0
            boxes = [tr.boxes[pos] for tr in selected]
            per_frame.append({"index": idx, "maskCoverage": cov, "boxes": [list(b) if b else None for b in boxes]})
            self._write_overlays(paths.preview_dir, idx, img, mask, boxes, selected, dilate)
        self.progress.finish("detect", f"{used}: {len(selected)} tracks")

        report.update(
            {
                "state": "COMPLETED",
                "framesRequested": idxs,
                "frames": per_frame,
                "tracks": tracks_to_json(selected),
                "detectorUsed": used,
                "detectorAttempts": attempts,
                "meanMaskCoverage": float(np.mean([f["maskCoverage"] for f in per_frame])) if per_frame else 0.0,
                "workdir": str(paths.root),
                "note": "Masks only. Final quality comes from the full run (inpainter + encode).",
            }
        )
        (paths.preview_dir / "preview.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        shutil.rmtree(paths.frames_dir, ignore_errors=True)
        return report

    def _discover(self, images, queries: list[str]):
        attempts: list[str] = []
        for detector in self.detectors:
            status = detector.status()
            if not status.startswith("ready"):
                attempts.append(f"{detector.name}: skip ({status})")
                continue
            tracks = detector.discover(images, queries)
            attempts.append(f"{detector.name}: {len(tracks)} tracks")
            if tracks:
                return detector.name, attempts, tracks
        raise PipelineError("no detector could run. " + " | ".join(attempts))

    def _write_overlays(self, preview_dir: Path, idx: int, img, mask, boxes, tracks, dilate: int) -> None:
        base = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img.ndim == 3 and img.shape[2] == 3 else img
        frame_bgr = base
        boxes_img = frame_bgr.copy()
        for tr, box in zip(tracks, boxes):
            if box is None:
                continue
            x1, y1, x2, y2 = (int(v) for v in box)
            cv2.rectangle(boxes_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(boxes_img, f"{tr.track_id} {tr.label}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        self._write_image(preview_dir / f"{idx:06d}_boxes.jpg", boxes_img[:, :, ::-1])

        m = mask.copy()
        if dilate > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate * 2 + 1, dilate * 2 + 1))
            m = cv2.dilate(m, k)
        mask_img = frame_bgr.copy()
        mask_img[m > 0] = (0.5 * mask_img[m > 0] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
        self._write_image(preview_dir / f"{idx:06d}_mask.jpg", mask_img[:, :, ::-1])
        self._write_image(preview_dir / f"{idx:06d}_raw.jpg", frame_bgr[:, :, ::-1])


def _targets_as_intent(targets: list[Target]):
    from videoclean.domain.intent import Intent

    return Intent(targets=targets)


def _t_json(t: Target) -> dict:
    return {
        "kind": t.kind,
        "query": t.query,
        "where": t.where,
        "motion": t.motion,
        "ordinal": t.ordinal,
        "from_side": t.from_side,
        "frames": list(t.frames) if t.frames else None,
    }
