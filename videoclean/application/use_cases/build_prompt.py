from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

from videoclean.adapters.prompt.locations import MaskRegion, mask_regions
from videoclean.application.config import PipelineConfig
from videoclean.application.errors import AdapterUnavailable, JobCancelled, PipelineError
from videoclean.application.frames_sample import sample_frame_indices
from videoclean.application.ports.jobs import JobStore
from videoclean.application.ports.media import MediaGateway
from videoclean.application.ports.progress import ProgressPort
from videoclean.application.use_cases.run_preview import targets_from_json
from videoclean.domain.intent import Target


@dataclass
class BuildPromptRequest:
    input_path: Path
    prompt: str
    annotations: list[dict]  # [{"frame": int, "mask": str | Path}]
    config: PipelineConfig
    job_id: str | None = None


@dataclass
class PromptPaths:
    root: Path
    frames_dir: Path
    logs_dir: Path
    ffmpeg_log: Path
    report_file: Path

    @classmethod
    def create(cls, root: Path) -> "PromptPaths":
        return cls(
            root=root,
            frames_dir=root / "process" / "frames",
            logs_dir=root / "logs",
            ffmpeg_log=root / "logs" / "ffmpeg.log",
            report_file=root / "output" / "report.json",
        )


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> dict | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass
    match = _JSON_RE.search(text)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


class BuildPrompt:
    """User prompt and/or drawn masks → VLM → editable prompt + detector targets."""

    def __init__(
        self,
        *,
        media: MediaGateway,
        llm,
        system_prompt: str,
        jobs: JobStore,
        progress: ProgressPort,
        new_job_id: callable,
        make_paths: callable,
        read_image: callable,
        to_jpeg: callable,
    ) -> None:
        self.media = media
        self.llm = llm
        self._system_prompt = system_prompt
        self.jobs = jobs
        self.progress = progress
        self._new_job_id = new_job_id
        self._make_paths = make_paths
        self._read_image = read_image
        self._to_jpeg = to_jpeg

    def execute(self, req: BuildPromptRequest, data_dir: Path) -> dict:
        prompt = (req.prompt or "").strip()
        if not prompt and not req.annotations:
            raise PipelineError("build-prompt: provide text or at least one mask annotation")
        if not req.input_path.is_file():
            raise FileNotFoundError(req.input_path)
        job_id = req.job_id or self._new_job_id()
        paths = self._make_paths(data_dir / "jobs" / job_id)
        self.jobs.upsert(job_id, "RUNNING", input_path=str(req.input_path), prompt=prompt)
        report: dict = {"jobId": job_id, "kind": "prompt", "state": "RUNNING", "prompt": prompt}
        try:
            out = self._run(req, job_id, paths, report)
            self.jobs.upsert(job_id, "COMPLETED", report=out)
            return out
        except JobCancelled:
            report["state"] = "CANCELLED"
            report["error"] = "cancelled"
            self.jobs.upsert(job_id, "CANCELLED", report=report, error="cancelled")
            raise
        except Exception as exc:
            report["state"] = "FAILED"
            report["error"] = str(exc)
            try:
                paths.report_file.parent.mkdir(parents=True, exist_ok=True)
                paths.report_file.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
            except OSError:
                pass
            self.jobs.upsert(job_id, "FAILED", report=report, error=str(exc)[:1500])
            raise

    def _run(self, req: BuildPromptRequest, job_id: str, paths, report: dict) -> dict:
        manifest = self.media.probe(req.input_path)
        ann_by_frame = self._annotations(req)
        idxs = (
            sorted(ann_by_frame)
            if ann_by_frame
            else sample_frame_indices(manifest.frame_count, req.config.prompt_frame_stride, req.config.prompt_frame_max)
        )
        if not idxs:
            raise PipelineError("build-prompt: no frames to look at")

        paths.frames_dir.mkdir(parents=True, exist_ok=True)
        self.progress.start("normalize", total=len(idxs))
        frame_paths = self.media.extract_frames_subset(req.input_path, idxs, paths.frames_dir, paths.ffmpeg_log)
        images = [self._read_image(p) for p in frame_paths]
        bad = [str(p) for p, img in zip(frame_paths, images) if img is None or not getattr(img, "size", 1)]
        if bad:
            raise PipelineError(f"build-prompt: failed to read {len(bad)} frame(s), first: {bad[0]}")
        jpegs = [
            self._to_jpeg(self._overlay(img, ann_by_frame[idx]) if idx in ann_by_frame else img)
            for img, idx in zip(images, idxs)
        ]
        geo_regions = _geometry_regions(ann_by_frame)
        self.progress.finish("normalize", f"{len(jpegs)} frames")

        st = self.llm.status()
        if not st.startswith("ready"):
            raise AdapterUnavailable(f"prompt-parser llm: {st}")
        self.progress.start("parse", detail=f"llm {self.llm.model}")
        raw = self.llm.complete(
            self._system_prompt,
            self._user_message(req.prompt, ann_by_frame, idxs, geo_regions),
            images=jpegs,
        )
        data = _extract_json(raw)
        if data is None:
            snippet = " ".join((raw or "").split())[:200]
            raise PipelineError(f"build-prompt: model returned no JSON object; raw: {snippet or '(empty)'}")
        targets_json = data.get("targets") or []
        targets = targets_from_json(targets_json) if targets_json else []
        targets = apply_mask_wheres(targets, geo_regions)
        user_prompt = (req.prompt or "").strip()
        if ann_by_frame:
            targets = strip_annotation_paint_colors(targets, user_prompt)
        out_prompt = str(data.get("prompt") or "").strip()
        if not out_prompt or _prompt_language_mismatch(user_prompt, out_prompt):
            out_prompt = _english_prompt_from_targets(targets) or user_prompt
        elif ann_by_frame:
            out_prompt = strip_annotation_paint_colors_text(out_prompt, user_prompt)
        self.progress.finish("parse", ", ".join(t.query for t in targets) or "(no targets)")

        report.update(
            {
                "state": "COMPLETED",
                "prompt": out_prompt,
                "userPrompt": user_prompt,
                "targets": [_t_json(t) for t in targets],
                "maskRegions": [
                    {"where": r.where, "nx": round(r.nx, 3), "ny": round(r.ny, 3), "area": r.area}
                    for r in geo_regions
                ],
                "framesUsed": idxs,
                "annotatedFrames": sorted(ann_by_frame),
                "imagesSent": len(jpegs),
                "llmModel": self.llm.model,
                "parseMode": "interpret",
            }
        )
        paths.report_file.parent.mkdir(parents=True, exist_ok=True)
        paths.report_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        shutil.rmtree(paths.frames_dir, ignore_errors=True)
        return report

    def _annotations(self, req: BuildPromptRequest) -> dict[int, Path]:
        out: dict[int, Path] = {}
        for a in req.annotations or []:
            try:
                frame = int(a.get("frame"))
            except (TypeError, ValueError):
                continue
            mask = Path(str(a.get("mask") or ""))
            if frame >= 0 and str(mask):
                out[frame] = mask
        return out

    @staticmethod
    def _overlay(img: np.ndarray, mask_path: Path) -> np.ndarray:
        m = cv2.imdecode(np.fromfile(mask_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if m is None:
            return img
        if m.shape[:2] != img.shape[:2]:
            m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        out = img.copy()
        out[m > 0] = (0.5 * out[m > 0] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
        return out

    @staticmethod
    def _user_message(
        prompt: str,
        ann_by_frame: dict[int, Path],
        idxs: list[int],
        geo_regions: list[MaskRegion] | None = None,
    ) -> str:
        lines = [
            f"User request: {prompt.strip()}"
            if prompt.strip()
            else "User request: (none — the user only marked areas on frames)",
            "Output requirements: JSON only. The prompt field MUST be English "
            "(normalized for Grounding DINO). targets[].query MUST be English. "
            "Do not write Chinese/Japanese/Korean in prompt. "
            "The bright red tint on marked pixels is annotation paint ONLY — "
            "never call the object red/crimson because of it.",
        ]
        if geo_regions:
            bits = [
                f"{r.where or 'center'} (nx={r.nx:.2f}, ny={r.ny:.2f}, area={r.area})"
                for r in geo_regions
            ]
            lines.append(
                "Mask geometry is AUTHORITATIVE for targets[].where (do not guess center "
                f"when the mask is in a corner): {'; '.join(bits)}."
            )
        for i, idx in enumerate(idxs):
            if idx in ann_by_frame:
                lines.append(
                    f"Image {i + 1}: frame {idx}. Red tint = user annotation marker only; "
                    "describe the underlying logo/text/object, not the paint color."
                )
            else:
                lines.append(f"Image {i + 1}: frame {idx}.")
        return "\n".join(lines)


def _read_mask(path: Path) -> np.ndarray | None:
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)


def _geometry_regions(ann_by_frame: dict[int, Path]) -> list[MaskRegion]:
    out: list[MaskRegion] = []
    for path in ann_by_frame.values():
        m = _read_mask(path)
        if m is None:
            continue
        out.extend(mask_regions(m))
    out.sort(key=lambda r: -r.area)
    return out


def apply_mask_wheres(targets: list[Target], regions: list[MaskRegion]) -> list[Target]:
    """Painted-mask geometry wins over VLM guesses for `where`."""
    if not targets or not regions:
        return targets
    if len(regions) == 1:
        w = regions[0].where
        return [replace(t, where=w) for t in targets]
    regs = list(regions)
    if len(targets) == len(regs):
        regs_sorted = sorted(regs, key=lambda r: (r.ny, r.nx))
        return [replace(t, where=regs_sorted[i].where) for i, t in enumerate(targets)]
    # More/fewer targets than blobs: assign largest blobs first, keep extras as-is.
    out: list[Target] = []
    for i, t in enumerate(targets):
        if i < len(regs):
            out.append(replace(t, where=regs[i].where))
        else:
            out.append(t)
    return out


def _has_cjk(text: str) -> bool:
    return any(
        "\u3040" <= ch <= "\u30ff"  # Hiragana/Katakana
        or "\u3400" <= ch <= "\u9fff"  # CJK unified
        or "\uac00" <= ch <= "\ud7af"  # Hangul
        for ch in text
    )


def _prompt_language_mismatch(user_prompt: str, out_prompt: str) -> bool:
    """Model invented CJK (or similar) when the user did not write in CJK."""
    if not out_prompt:
        return True
    return _has_cjk(out_prompt) and not _has_cjk(user_prompt)


def _english_prompt_from_targets(targets: list) -> str:
    if not targets:
        return "Remove the marked areas"
    parts: list[str] = []
    for t in targets:
        q = (getattr(t, "query", None) or "").strip()
        if not q:
            continue
        where = getattr(t, "where", None)
        parts.append(f"{q} ({where})" if where else q)
    return f"Remove: {', '.join(parts)}" if parts else "Remove the marked areas"


# Overlay in BuildPrompt._overlay is BGR red tint — models often invent "red logo".
_ANNOTATION_PAINT_COLORS = ("red", "crimson", "scarlet", "ruby")


def strip_annotation_paint_colors_text(text: str, user_prompt: str) -> str:
    """Drop paint-color adjectives unless the user themselves named that color."""
    raw = (text or "").strip()
    if not raw:
        return raw
    user = (user_prompt or "").casefold()
    out = raw
    for color in _ANNOTATION_PAINT_COLORS:
        if color in user:
            continue
        out = re.sub(rf"\b{color}\b[\s-]*", "", out, flags=re.IGNORECASE)
    return " ".join(out.split())


def strip_annotation_paint_colors(targets: list[Target], user_prompt: str) -> list[Target]:
    out: list[Target] = []
    for t in targets:
        q = strip_annotation_paint_colors_text(t.query, user_prompt)
        out.append(replace(t, query=q or t.query))
    return out


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
