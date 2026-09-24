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

    def masks(self, frames: list[np.ndarray], tracks: list[Track], on_progress=None) -> list[np.ndarray]:
        if not frames:
            return []
        if on_progress:
            on_progress(0, 1, f"{self.name} load")
        self._ensure()
        h, w = frames[0].shape[:2]
        if not tracks:
            return [np.zeros((h, w), dtype=np.uint8) for _ in frames]
        try:
            return self._propagate(frames, tracks, on_progress=on_progress)
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

    def _propagate(self, frames: list[np.ndarray], tracks: list[Track], on_progress=None) -> list[np.ndarray]:
        import cv2
        import torch

        h, w = frames[0].shape[:2]
        n = len(frames)
        acc = [np.zeros((h, w), dtype=np.uint8) for _ in frames]
        if on_progress:
            on_progress(0, n + len(tracks), f"{self.name} init")
        state = self._init_state_from_frames(frames)
        obj_id = 1
        for tr_i, tr in enumerate(tracks):
            if on_progress:
                on_progress(n + tr_i, n + len(tracks), f"{self.name} track {tr_i + 1}/{len(tracks)}")
            anchors = _anchor_boxes(tr, w, h, max_anchors=8)
            if not anchors:
                continue
            for frame_idx, box in anchors:
                self._predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    box=np.array(box, dtype=np.float32),
                )
            obj_id += 1
        if obj_id == 1:
            return acc

        prop_total = max(n, 1)

        def _consume() -> None:
            for frame_idx, _obj_ids, mask_logits in self._predictor.propagate_in_video(state):
                if on_progress:
                    on_progress(
                        frame_idx + 1,
                        prop_total,
                        f"{self.name} propagate {frame_idx + 1}/{n}",
                    )
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
        if on_progress:
            on_progress(1, 1, f"{self.name} done")
        return acc

    def _init_state_from_frames(self, frames) -> dict:
        """Copy of SAM2VideoPredictor.init_state at commit 2b90b9f5.

        Only load_video_frames is replaced. reset_state is not used: on that
        commit it clears keys that do not exist yet.
        """
        import torch
        from collections import OrderedDict

        predictor = self._predictor
        image_size = int(getattr(predictor, "image_size", 1024) or 1024)
        n = len(frames)
        h, w = frames[0].shape[:2]
        images = _image_tensor(frames, image_size, getattr(self, "tensor_path", None))
        device = torch.device(getattr(predictor, "device", self.device))
        state = {
            "images": images,
            "num_frames": n,
            "offload_video_to_cpu": True,
            "offload_state_to_cpu": True,
            "video_height": h,
            "video_width": w,
            "device": device,
            "storage_device": torch.device("cpu"),
            "point_inputs_per_obj": {},
            "mask_inputs_per_obj": {},
            "cached_features": {},
            "constants": {},
            "obj_id_to_idx": OrderedDict(),
            "obj_idx_to_id": OrderedDict(),
            "obj_ids": [],
            "output_dict_per_obj": {},
            "temp_output_dict_per_obj": {},
            "frames_tracked_per_obj": {},
        }
        warmup = getattr(predictor, "_get_image_feature", None)
        if warmup is not None:
            warmup(state, frame_idx=0, batch_size=1)
        return state


def _image_tensor(frames, image_size: int, tensor_path):
    """Normalized NCHW tensor. File-backed when tensor_path is set."""
    import torch

    n = len(frames)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    if tensor_path:
        path = Path(tensor_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        count = n * 3 * image_size * image_size
        with open(path, "wb") as handle:
            handle.truncate(count * 4)
        storage = torch.from_file(str(path), shared=True, size=count, dtype=torch.float32)
        images = storage.view(n, 3, image_size, image_size)
        for i, frame in enumerate(frames):
            images[i].copy_(torch.from_numpy(_normalize_frame(frame, image_size, mean, std)))
        return images
    packed = np.empty((n, 3, image_size, image_size), dtype=np.float32)
    for i, frame in enumerate(frames):
        packed[i] = _normalize_frame(frame, image_size, mean, std)
    return torch.from_numpy(packed)


def _normalize_frame(frame: np.ndarray, image_size: int, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    import cv2

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if rgb.shape[0] != image_size or rgb.shape[1] != image_size:
        rgb = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    x = rgb.astype(np.float32) / 255.0
    x = (x - mean) / std
    return np.transpose(x, (2, 0, 1))


def _clamp_box(
    box: tuple[int, int, int, int],
    part,
    w: int,
    h: int,
) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = apply_part(box, part)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 >= 4 and y2 - y1 >= 4:
        return x1, y1, x2, y2
    return None


def _first_box(tr: Track, w: int, h: int) -> tuple[int, tuple[int, int, int, int] | None]:
    for i, box in enumerate(tr.boxes):
        if box is None:
            continue
        clamped = _clamp_box(box, tr.part, w, h)
        if clamped is not None:
            return i, clamped
    return 0, None


def _anchor_boxes(
    tr: Track,
    w: int,
    h: int,
    *,
    max_anchors: int = 8,
) -> list[tuple[int, tuple[int, int, int, int]]]:
    """Sample up to max_anchors non-null boxes across the track for multi-frame prompts."""
    hits: list[tuple[int, tuple[int, int, int, int]]] = []
    for i, box in enumerate(tr.boxes):
        if box is None:
            continue
        clamped = _clamp_box(box, tr.part, w, h)
        if clamped is not None:
            hits.append((i, clamped))
    if not hits:
        return []
    if len(hits) <= max_anchors:
        return hits
    # Evenly sample across available keyframes (keep first and last).
    idxs = sorted(
        {
            int(round(j * (len(hits) - 1) / (max_anchors - 1)))
            for j in range(max_anchors)
        }
    )
    return [hits[i] for i in idxs]
