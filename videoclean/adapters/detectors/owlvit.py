from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

import re

from videoclean.adapters.detectors._cv import (
    MAX_TEMPLATE_AREA,
    box_area_frac,
    iou,
    keep_detection_box,
    match_template,
    sample_indices,
)
from videoclean.domain.tracks import Track, infer_motion

# OwlViTTextConfig.max_position_embeddings. Per visual class, not the user prompt.
OWL_VIT_MAX_QUERY_TOKENS = 16
_PHRASE_SPLIT = re.compile(r"[\n;•]+|(?:, |\. )|\s+и\s+", re.IGNORECASE)


def split_visual_phrases(text: str) -> list[str]:
    raw = " ".join((text or "").split())
    if not raw:
        return []
    parts = [p.strip() for p in _PHRASE_SPLIT.split(raw) if p and p.strip()]
    return parts or [raw]


def fit_owlvit_queries(
    queries: list[str],
    token_len,
    max_tokens: int = OWL_VIT_MAX_QUERY_TOKENS,
) -> list[str]:
    """Keep queries that fit OWL-ViT's 16-token class encoder. Split over-long
    strings; do not invent replacement classes.
    """
    kept: list[str] = []
    seen: set[str] = set()

    def add(phrase: str) -> None:
        p = phrase.strip()
        if not p or p in seen:
            return
        if token_len(p) > max_tokens:
            return
        seen.add(p)
        kept.append(p)

    for query in queries:
        q = " ".join((query or "").split())
        if not q:
            continue
        if token_len(q) <= max_tokens:
            add(q)
            continue
        for part in split_visual_phrases(q):
            add(part)
    return kept


@dataclass
class BoxHit:
    label: str
    score: float
    xyxy: tuple[int, int, int, int]


class OwlVitDetector:
    """Open-vocabulary boxes from a Hugging Face OWL-ViT checkpoint, then template-track across frames."""

    name = "owlvit"

    def __init__(
        self,
        model_id: str,
        device: str,
        threshold: float = 0.15,
        allow_download: bool = False,
    ) -> None:
        self.model_id = model_id
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
        cache = Path.home() / ".cache" / "huggingface" / "hub" / ("models--" + self.model_id.replace("/", "--"))
        if cache.exists():
            return f"ready (weights on disk: {self.model_id}; loads on first detect, device={self.device})"
        return (
            f"unavailable: {self.model_id} not in local HF cache. "
            f"Download: uv run huggingface-cli download {self.model_id}  "
            f"or pass --download-models"
        )

    def discover(self, frames: list[np.ndarray], queries: list[str], on_progress=None) -> list[Track]:
        if not frames or not queries:
            return []
        if on_progress:
            on_progress(0, 1, f"{self.name} load")
        ok, err = self._try_load()
        if not ok:
            return []
        class_queries = fit_owlvit_queries(queries, self._token_len)
        if not class_queries:
            return []
        key_idx = sample_indices(len(frames), min(12, len(frames)))
        n_keys = len(key_idx)
        per_frame: list[list[tuple[str, float, tuple[int, int, int, int]]]] = [[] for _ in frames]
        for n, i in enumerate(key_idx, start=1):
            if on_progress:
                on_progress(
                    n,
                    n_keys,
                    f"{self.name} keyframe {n}/{n_keys}  frame {i + 1}/{len(frames)}",
                )
            for hit in self._detect_frame(frames[i], class_queries):
                per_frame[i].append((hit.label, hit.score, hit.xyxy))
        if on_progress:
            on_progress(n_keys, n_keys, f"{self.name} template-track")
        tracks: list[Track] = []
        tid = 1000
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
        try:
            if int(np.__version__.split(".", 1)[0]) >= 2:
                self._load_error = "numpy2/torch2.2 mismatch"
                return False, self._load_error
            from transformers import OwlViTForObjectDetection, OwlViTProcessor
            from transformers.utils import is_torch_available

            if not is_torch_available():
                self._load_error = "torch not usable"
                return False, self._load_error

            kwargs = {"local_files_only": not self.allow_download}
            self._processor = OwlViTProcessor.from_pretrained(self.model_id, **kwargs)
            self._model = OwlViTForObjectDetection.from_pretrained(self.model_id, **kwargs)
            self._model.eval()
            self._model.to(self.device)
            return True, ""
        except Exception as exc:  # noqa: BLE001 — optional backend
            self._load_error = f"{type(exc).__name__}: {exc}"[:240]
            self._model = None
            self._processor = None
            return False, self._load_error

    def _token_len(self, text: str) -> int:
        ids = self._processor.tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]
        return len(ids)

    def _detect_frame(self, bgr: np.ndarray, queries: list[str]) -> list[BoxHit]:
        import torch
        from PIL import Image

        rgb = bgr[:, :, ::-1]
        image = Image.fromarray(rgb)
        queries = [q.strip() for q in queries if q and str(q).strip()]
        if not queries:
            return []
        # Queries are already fitted to OWL_VIT_MAX_QUERY_TOKENS. Pad only.
        text_inputs = self._processor.tokenizer(
            queries,
            padding="max_length",
            truncation=True,
            max_length=OWL_VIT_MAX_QUERY_TOKENS,
            return_tensors="pt",
        )
        image_inputs = self._processor.image_processor(images=image, return_tensors="pt")
        inputs = {**text_inputs, **image_inputs}
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]], device=self.device)
        results = self._processor.post_process_object_detection(
            outputs=outputs, threshold=self.threshold, target_sizes=target_sizes
        )[0]
        hits: list[BoxHit] = []
        h, w = bgr.shape[:2]
        for score, label_id, box in zip(results["scores"], results["labels"], results["boxes"]):
            x1, y1, x2, y2 = (int(v) for v in box.tolist())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if not keep_detection_box((x1, y1, x2, y2), w, h):
                continue
            idx = int(label_id)
            name = queries[idx] if 0 <= idx < len(queries) else "object"
            hits.append(BoxHit(label=name, score=float(score), xyxy=(x1, y1, x2, y2)))
        return hits
