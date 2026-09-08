from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

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
        self.progress.finish("normalize", f"{len(jpegs)} frames")

        st = self.llm.status()
        if not st.startswith("ready"):
            raise AdapterUnavailable(f"prompt-parser llm: {st}")
        self.progress.start("parse", detail=f"llm {self.llm.model}")
        raw = self.llm.complete(self._system_prompt, self._user_message(req.prompt, ann_by_frame, idxs), images=jpegs)
        data = _extract_json(raw)
        if data is None:
            raise PipelineError("build-prompt: model returned no JSON object")
        targets_json = data.get("targets") or []
        targets = targets_from_json(targets_json) if targets_json else []
        out_prompt = str(data.get("prompt") or "").strip()
        if not out_prompt:
            out_prompt = ", ".join(t.query for t in targets) or (req.prompt or "").strip()
        self.progress.finish("parse", ", ".join(t.query for t in targets) or "(no targets)")

        report.update(
            {
                "state": "COMPLETED",
                "prompt": out_prompt,
                "userPrompt": (req.prompt or "").strip(),
                "targets": [_t_json(t) for t in targets],
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
    def _user_message(prompt: str, ann_by_frame: dict[int, Path], idxs: list[int]) -> str:
        lines = [
            f"User request: {prompt.strip()}" if prompt.strip() else "User request: (none — the user only marked areas on frames)"
        ]
        for i, idx in enumerate(idxs):
            if idx in ann_by_frame:
                lines.append(f"Image {i + 1}: frame {idx}. The red overlay marks what the user wants removed.")
            else:
                lines.append(f"Image {i + 1}: frame {idx}.")
        return "\n".join(lines)


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
