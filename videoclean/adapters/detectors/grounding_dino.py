from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from videoclean.adapters.detectors._cv import (
    MAX_TEMPLATE_AREA,
    box_area_frac,
    iou,
    keep_detection_box,
    match_template,
    sample_indices,
)
from videoclean.adapters.hf_cache import download_hint, hf_cached
from videoclean.domain.tracks import Track, infer_motion

DEFAULT_MODEL = "IDEA-Research/grounding-dino-tiny"


@dataclass
class BoxHit:
    label: str
    score: float
    xyxy: tuple[int, int, int, int]


class GroundingDinoDetector:
    """Open-vocab boxes from a text query (mug, logo, person — same model)."""

    name = "grounding-dino"

    def __init__(
        self,
        model_id: str,
        device: str,
        threshold: float = 0.25,
        allow_download: bool = False,
    ) -> None:
        self.model_id = model_id.strip() or DEFAULT_MODEL
        self.device = device
        self.threshold = threshold
        self.allow_download = allow_download
        self._model = None
        self._processor = None
        self._load_error: str | None = None

    def status(self) -> str:
        if self._model is not None:
            return f"ready ({self.model_id} on {self.device})"
        if self._load_error:
            return f"unavailable: {self._load_error}"
        cached = hf_cached(self.model_id)
        if not cached and not self.allow_download:
            return (
                f"unavailable: {self.model_id} not in local HF cache. "
                f"{download_hint(self.model_id)}"
            )
        return f"ready (weights={'disk' if cached else 'download-allowed'}; loads on first detect, device={self.device})"

    def discover(
        self,
        frames: list[np.ndarray],
        queries: list[str],
        on_progress=None,
    ) -> list[Track]:
        if not frames or not queries:
            return []
        if on_progress:
            on_progress(0, 1, f"{self.name} load")
        ok, err = self._try_load()
        if not ok:
            return []
        phrases = _phrases(queries)
        if not phrases:
            return []
        key_idx = sample_indices(len(frames), min(8, len(frames)))
        n_keys = len(key_idx)
        per_frame: list[list[tuple[str, float, tuple[int, int, int, int]]]] = [[] for _ in frames]
        for n, i in enumerate(key_idx, start=1):
            if on_progress:
                on_progress(
                    n,
                    n_keys,
                    f"{self.name} keyframe {n}/{n_keys}  frame {i + 1}/{len(frames)}",
                )
            for hit in self._detect_frame(frames[i], phrases):
                per_frame[i].append((hit.label, hit.score, hit.xyxy))
        if on_progress:
            on_progress(n_keys, n_keys, f"{self.name} template-track")
        tracks: list[Track] = []
        tid = 3000
        used = [[False] * len(per_frame[i]) for i in range(len(frames))]
        for i in key_idx:
            for j, (label, score, box) in enumerate(per_frame[i]):
                if used[i][j]:
                    continue
                boxes: list[tuple[int, int, int, int] | None] = [None] * len(frames)
                boxes[i] = box
                used[i][j] = True
                for k in key_idx:
                    if k <= i:
                        continue
                    best, best_iou = -1, 0.3
                    for u, (_, _, b2) in enumerate(per_frame[k]):
                        if used[k][u]:
                            continue
                        v = iou(box if boxes[k] is None else boxes[max(i, k - 1)] or box, b2)
                        if v > best_iou:
                            best, best_iou = u, v
                    if best >= 0:
                        used[k][best] = True
                        boxes[k] = per_frame[k][best][2]
                x1, y1, x2, y2 = box
                crop = frames[i][y1:y2, x1:x2]
                fh, fw = frames[i].shape[:2]
                if crop.size and box_area_frac(box, fw, fh) <= MAX_TEMPLATE_AREA:
                    searched = match_template(frames, crop, min_score=0.55)
                    boxes = [s if s is not None else b for s, b in zip(searched, boxes)]
                tr = Track(
                    track_id=tid,
                    label=label,
                    boxes=boxes,
                    scores=[score] * len(frames),
                    notes=["open-vocab", self.model_id],
                )
                tr.motion = infer_motion(tr, frames[0].shape[1], frames[0].shape[0])
                tracks.append(tr)
                tid += 1
        return tracks

    def _try_load(self) -> tuple[bool, str]:
        if self._model is not None:
            return True, ""
        if self._load_error is not None:
            return False, self._load_error
        if not hf_cached(self.model_id) and not self.allow_download:
            self._load_error = f"{self.model_id} not in local HF cache. {download_hint(self.model_id)}"
            return False, self._load_error
        try:
            from transformers import AutoProcessor, GroundingDinoForObjectDetection

            kwargs = {"local_files_only": not self.allow_download}
            self._processor = AutoProcessor.from_pretrained(self.model_id, **kwargs)
            self._model = GroundingDinoForObjectDetection.from_pretrained(self.model_id, **kwargs)
            self._model.eval()
            self._model.to(self.device)
            return True, ""
        except Exception as exc:  # noqa: BLE001
            self._load_error = f"{type(exc).__name__}: {exc}"[:240]
            self._model = None
            self._processor = None
            return False, self._load_error

    def _detect_frame(self, bgr: np.ndarray, phrases: list[str]) -> list[BoxHit]:
        import torch
        from PIL import Image

        rgb = Image.fromarray(bgr[:, :, ::-1])
        caption = " . ".join(phrases) + " ."
        inputs = self._processor(images=rgb, text=caption, return_tensors="pt")
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs)
        h, w = bgr.shape[:2]
        target_sizes = torch.tensor([(h, w)], device=self.device)
        try:
            results = self._processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                box_threshold=self.threshold,
                text_threshold=max(0.15, self.threshold - 0.05),
                target_sizes=target_sizes,
            )[0]
        except TypeError:
            results = self._processor.post_process_grounded_object_detection(
                outputs,
                input_ids=inputs["input_ids"],
                threshold=self.threshold,
                target_sizes=target_sizes,
            )[0]
        hits: list[BoxHit] = []
        boxes = results.get("boxes")
        scores = results.get("scores")
        labels = results.get("labels") or results.get("text_labels") or []
        if boxes is None:
            return []
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = (int(v) for v in box.tolist())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if not keep_detection_box((x1, y1, x2, y2), w, h):
                continue
            score = float(scores[i]) if scores is not None and i < len(scores) else 0.0
            raw = labels[i] if i < len(labels) else "object"
            name = str(raw).strip() or "object"
            if name.startswith("##"):
                continue
            name = " ".join(name.replace("-", " ").split()) or "object"
            hits.append(BoxHit(label=name, score=score, xyxy=(x1, y1, x2, y2)))
        return _nms(hits)


def _nms(hits: list[BoxHit], iou_thr: float = 0.3) -> list[BoxHit]:
    ordered = sorted(hits, key=lambda h: -h.score)
    kept: list[BoxHit] = []
    for hit in ordered:
        if any(iou(hit.xyxy, other.xyxy) > iou_thr for other in kept):
            continue
        kept.append(hit)
    return kept


def _phrases(queries: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for q in queries:
        p = " ".join((q or "").split()).casefold()
        if not p or p in seen:
            continue
        if not p.endswith("."):
            p = p + "."
        seen.add(p)
        out.append(p)
    return out
