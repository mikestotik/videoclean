from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from videoclean.adapters.hf_cache import download_hint, hf_cached
from videoclean.application.errors import AdapterUnavailable
from videoclean.domain.tracks import Track, apply_part


class Sam2VideoSegmenter:
    """Propagate a box prompt across the clip (official sam2 video predictor)."""

    name = "sam2-video"

    def __init__(
        self,
        model_id: str,
        device: str,
        allow_download: bool = False,
        dilate_px: int = 3,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.allow_download = allow_download
        self.dilate_px = dilate_px
        self._predictor = None
        self._load_error: str | None = None

    def status(self) -> str:
        if self._predictor is not None:
            return f"ready ({self.model_id} video on {self.device})"
        if self._load_error:
            return f"unavailable: {self._load_error}"
        try:
            from sam2.sam2_video_predictor import SAM2VideoPredictor  # noqa: F401
        except Exception:  # noqa: BLE001
            return (
                "unavailable: sam2 video predictor missing. "
                "uv pip install git+https://github.com/facebookresearch/sam2.git (needs torch>=2.5)"
            )
        cached = hf_cached(self.model_id)
        if not cached and not self.allow_download:
            return f"unavailable: {self.model_id} not in local HF cache. {download_hint(self.model_id)}"
        return f"ready (sam2 video, weights={'disk' if cached else 'download-allowed'}, device={self.device})"

    def masks(self, frames: list[np.ndarray], tracks: list[Track]) -> list[np.ndarray]:
        if not frames:
            return []
        self._ensure()
        h, w = frames[0].shape[:2]
        if not tracks:
            return [np.zeros((h, w), dtype=np.uint8) for _ in frames]
        try:
            return self._propagate(frames, tracks)
        except Exception as exc:  # noqa: BLE001
            raise AdapterUnavailable(f"sam2-video failed: {type(exc).__name__}: {exc}"[:240]) from exc

    def _ensure(self) -> None:
        if self._predictor is not None:
            return
        if self._load_error:
            raise AdapterUnavailable(self._load_error)
        if not hf_cached(self.model_id) and not self.allow_download:
            self._load_error = f"{self.model_id} not in local HF cache. {download_hint(self.model_id)}"
            raise AdapterUnavailable(self._load_error)
        try:
            from sam2.sam2_video_predictor import SAM2VideoPredictor

            self._predictor = SAM2VideoPredictor.from_pretrained(self.model_id)
            if hasattr(self._predictor, "model"):
                self._predictor.model.to(self.device)
        except Exception as exc:  # noqa: BLE001
            self._load_error = f"{type(exc).__name__}: {exc}"[:240]
            raise AdapterUnavailable(self._load_error) from exc

    def _propagate(self, frames: list[np.ndarray], tracks: list[Track]) -> list[np.ndarray]:
        import cv2
        import torch

        h, w = frames[0].shape[:2]
        acc = [np.zeros((h, w), dtype=np.uint8) for _ in frames]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i, frame in enumerate(frames):
                cv2.imwrite(str(root / f"{i:05d}.jpg"), frame)
            state = self._predictor.init_state(video_path=str(root))
            obj_id = 1
            for tr in tracks:
                frame_idx, box = _first_box(tr, w, h)
                if box is None:
                    continue
                self._predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    box=np.array(box, dtype=np.float32),
                )
                obj_id += 1
            if obj_id == 1:
                return acc

            def _consume() -> None:
                for frame_idx, _obj_ids, mask_logits in self._predictor.propagate_in_video(state):
                    if frame_idx >= len(acc):
                        continue
                    for plane in mask_logits:
                        sl = plane
                        if hasattr(sl, "cpu"):
                            sl = sl.cpu().numpy()
                        while getattr(sl, "ndim", 0) > 2:
                            sl = sl[0]
                        acc[frame_idx][sl > 0] = 255

            with torch.inference_mode():
                if self.device == "cuda":
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        _consume()
                else:
                    _consume()
        if self.dilate_px > 0:
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (self.dilate_px * 2 + 1, self.dilate_px * 2 + 1)
            )
            acc = [cv2.dilate(m, k) if np.any(m) else m for m in acc]
        return acc


def _first_box(tr: Track, w: int, h: int) -> tuple[int, tuple[int, int, int, int] | None]:
    for i, box in enumerate(tr.boxes):
        if box is None:
            continue
        x1, y1, x2, y2 = apply_part(box, tr.part)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 >= 4 and y2 - y1 >= 4:
            return i, (x1, y1, x2, y2)
    return 0, None
