from __future__ import annotations

import numpy as np
from PIL import Image

from videoclean.adapters.hf_cache import download_hint, hf_cached
from videoclean.adapters.torch_compat import ensure_torch_compiler_compat
from videoclean.application.errors import AdapterUnavailable
from videoclean.domain.tracks import Track, apply_part


class Sam2Segmenter:
    """Pixel masks from track boxes. Uses transformers Sam2Model if present, else the sam2 package."""

    name = "sam2"

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
        self._backend: str | None = None
        self._model = None
        self._processor = None
        self._predictor = None
        self._load_error: str | None = None

    def status(self) -> str:
        if self._backend:
            return f"ready ({self._backend}, {self.model_id} on {self.device})"
        if self._load_error:
            return f"unavailable: {self._load_error}"
        runtime = _runtime_available()
        cached = hf_cached(self.model_id)
        if runtime is None:
            weights = (
                f"weights on disk ({self.model_id})"
                if cached
                else f"weights: {download_hint(self.model_id)}"
            )
            return (
                "unavailable: SAM2 runtime missing. Need transformers>=4.56 "
                "(Sam2Model) or the facebookresearch/sam2 package "
                f"(that one wants torch>=2.5). {weights}"
            )
        if not cached and not self.allow_download:
            return (
                f"unavailable: {self.model_id} not in local HF cache. "
                f"{download_hint(self.model_id)}"
            )
        return f"ready (runtime={runtime}, weights={'disk' if cached else 'download-allowed'}, device={self.device})"

    def masks(self, frames: list[np.ndarray], tracks: list[Track]) -> list[np.ndarray]:
        if not frames:
            return []
        self._ensure()
        h, w = frames[0].shape[:2]
        out: list[np.ndarray] = []
        for i, frame in enumerate(frames):
            boxes: list[tuple[int, int, int, int]] = []
            for tr in tracks:
                box = tr.boxes[i] if i < len(tr.boxes) else None
                if box is None:
                    continue
                x1, y1, x2, y2 = apply_part(box, tr.part)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 - x1 >= 4 and y2 - y1 >= 4:
                    boxes.append((x1, y1, x2, y2))
            if not boxes:
                out.append(np.zeros((h, w), dtype=np.uint8))
                continue
            mask = self._predict(frame, boxes)
            if self.dilate_px > 0:
                mask = _dilate(mask, self.dilate_px)
            out.append(mask)
        return out

    def _ensure(self) -> None:
        if self._backend:
            return
        ok, err = self._try_load()
        if not ok:
            raise AdapterUnavailable(err)

    def _try_load(self) -> tuple[bool, str]:
        if self._backend:
            return True, ""
        if self._load_error:
            return False, self._load_error
        if not hf_cached(self.model_id) and not self.allow_download:
            self._load_error = (
                f"{self.model_id} not in local HF cache. {download_hint(self.model_id)}"
            )
            return False, self._load_error
        ensure_torch_compiler_compat()
        kwargs = {"local_files_only": not self.allow_download}
        try:
            from transformers import Sam2Model, Sam2Processor

            self._processor = Sam2Processor.from_pretrained(self.model_id, **kwargs)
            self._model = Sam2Model.from_pretrained(self.model_id, **kwargs)
            self._model.eval()
            self._model.to(self.device)
            self._backend = "transformers.Sam2Model"
            return True, ""
        except Exception as hf_exc:  # noqa: BLE001
            hf_err = f"{type(hf_exc).__name__}: {hf_exc}"[:200]
        try:
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            self._predictor = SAM2ImagePredictor.from_pretrained(self.model_id)
            self._predictor.model.to(self.device)
            self._backend = "sam2.SAM2ImagePredictor"
            return True, ""
        except Exception as pkg_exc:  # noqa: BLE001
            pkg_err = f"{type(pkg_exc).__name__}: {pkg_exc}"[:200]
            self._load_error = (
                "SAM2 failed to load. transformers: "
                f"{hf_err}. sam2 package: {pkg_err}. "
                "Prefer transformers>=4.56 (works on torch 2.2+). "
                "Official sam2 pkg needs torch>=2.5. "
                f"Weights: {download_hint(self.model_id)}"
            )
            return False, self._load_error

    def _predict(self, bgr: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
        if self._backend == "transformers.Sam2Model":
            return self._predict_transformers(bgr, boxes)
        return self._predict_sam2pkg(bgr, boxes)

    def _predict_transformers(self, bgr: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
        import torch

        ensure_torch_compiler_compat()
        rgb = Image.fromarray(bgr[:, :, ::-1])
        input_boxes = [[list(b) for b in boxes]]
        inputs = self._processor(images=rgb, input_boxes=input_boxes, return_tensors="pt")
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs, multimask_output=False)
        original = inputs.get("original_sizes")
        masks = self._processor.post_process_masks(outputs.pred_masks.cpu(), original)[0]
        return _or_masks(masks, bgr.shape[:2])

    def _predict_sam2pkg(self, bgr: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
        import torch

        rgb = bgr[:, :, ::-1]
        h, w = bgr.shape[:2]
        acc = np.zeros((h, w), dtype=np.uint8)

        def _run() -> None:
            self._predictor.set_image(rgb)
            for box in boxes:
                masks, _, _ = self._predictor.predict(
                    box=np.array(box, dtype=np.float32),
                    multimask_output=False,
                )
                acc[masks[0] > 0] = 255

        with torch.inference_mode():
            if self.device == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _run()
            else:
                _run()
        return acc


def _runtime_available() -> str | None:
    try:
        from transformers import Sam2Model  # noqa: F401

        return "transformers.Sam2Model"
    except Exception:  # noqa: BLE001
        pass
    try:
        from sam2.sam2_image_predictor import SAM2ImagePredictor  # noqa: F401

        return "sam2.SAM2ImagePredictor"
    except Exception:  # noqa: BLE001
        return None


def _or_masks(masks, shape: tuple[int, int]) -> np.ndarray:
    import torch

    h, w = shape
    acc = np.zeros((h, w), dtype=np.uint8)
    tensor = masks if isinstance(masks, torch.Tensor) else torch.as_tensor(masks)
    arr = tensor.detach().cpu().numpy()
    while arr.ndim > 3:
        arr = arr[0]
    if arr.ndim == 2:
        acc[arr > 0] = 255
        return acc
    for plane in arr:
        sl = plane
        while sl.ndim > 2:
            sl = sl[0]
        if sl.shape != (h, w):
            sl = _resize_mask(sl.astype(np.float32), (h, w))
        acc[sl > 0] = 255
    return acc


def _resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    import cv2

    h, w = size
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)


def _dilate(mask: np.ndarray, px: int) -> np.ndarray:
    import cv2

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (px * 2 + 1, px * 2 + 1))
    return cv2.dilate(mask, k)
