from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from videoclean.application.errors import AdapterUnavailable
from videoclean.store import resolve_data_dir

# Torch Hub checkpoint used by simple-lama-inpainting (big-lama.pt).
LAMA_MODEL_URL = (
    "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt"
)


def _data_dir_model_path() -> Path:
    return resolve_data_dir() / "weights" / "lama" / "big-lama.pt"


def _torch_hub_model_path() -> Path:
    # Do not import torch here — cold import is ~1s and catalog status probes this path.
    torch_home = (os.environ.get("TORCH_HOME") or "").strip()
    root = Path(torch_home).expanduser() if torch_home else (Path.home() / ".cache" / "torch")
    return root / "hub" / "checkpoints" / "big-lama.pt"


def _default_model_path() -> Path:
    env = (os.environ.get("LAMA_MODEL") or "").strip()
    if env:
        return Path(env).expanduser()
    return _data_dir_model_path()


def find_weights() -> Path | None:
    seen: set[Path] = set()
    for path in (_default_model_path(), _data_dir_model_path(), _torch_hub_model_path()):
        if path in seen:
            continue
        seen.add(path)
        if path.is_file():
            return path
    return None


class LamaInpainter:
    """Per-frame LaMa (TorchScript big-lama). CPU and CUDA; no temporal consistency.

    Loads the JIT weights ourselves with map_location. PyPI simple-lama-inpainting 0.1.0 calls
    torch.jit.load without map_location, which fails on machines without CUDA.
    """

    name = "lama"
    device_note = "Uses --device. CPU works; CUDA/MPS faster."
    video_aware = False

    def __init__(self, device: str = "cpu", allow_download: bool = False) -> None:
        self.device = device
        self.allow_download = allow_download
        self._model = None
        self._torch_device = None
        self._prepare = None
        self._load_error: str | None = None

    def status(self) -> str:
        if self._model is not None:
            return f"ready (loaded on {self.device})"
        if self._load_error:
            return f"unavailable: {self._load_error}"
        try:
            import torch  # noqa: F401
        except ImportError:
            return "unavailable: torch missing"
        weights = find_weights()
        if weights is None and not self.allow_download:
            dest = _default_model_path()
            return (
                f"unavailable: big-lama.pt not found at {dest}. "
                f"Pass --download-models once, or: curl -L {LAMA_MODEL_URL} -o {dest}"
            )
        if weights is None:
            return f"ready (will download big-lama.pt on first inpaint, device={self.device})"
        return f"ready (weights on disk, loads on first inpaint, device={self.device})"

    def inpaint_clip(self, frames: list[np.ndarray], masks: list[np.ndarray]) -> list[np.ndarray]:
        from videoclean.application.hole_policy import plan_holes

        if not frames:
            return []
        plans = plan_holes(
            frame_hw=frames[0].shape[:2],
            mask_coverage=[float(np.count_nonzero(m)) / float(m.size) for m in masks],
            mask_boxes=[_box_of(m) for m in masks],
            family="lama",
            device=self.device,
            budget_bytes=int(getattr(self, "vram_budget_bytes", 0) or 0),
            inpaint_max_side=getattr(self, "inpaint_max_side", None),
        )
        out: list[np.ndarray] = []
        for plan in plans:
            out.extend(self.inpaint_masked(frames[plan.start : plan.end], masks[plan.start : plan.end], plan))
        return out

    def inpaint(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        if not np.any(mask):
            return frame
        return self.inpaint_clip([frame], [mask])[0]

    def inpaint_masked(self, frames: list[np.ndarray], masks: list[np.ndarray], plan) -> list[np.ndarray]:
        """Crop, batch under the VRAM budget, feather the paste. No full-frame fallback."""
        self._ensure()
        side = int(plan.side)
        feather = int(getattr(plan, "feather_px", 16) or 0)
        out = [np.ascontiguousarray(frame) for frame in frames]
        pending: list[tuple[int, np.ndarray, np.ndarray, tuple[int, int]]] = []
        for i, (frame, mask) in enumerate(zip(frames, masks)):
            if mask.dtype != np.uint8:
                mask = mask.astype(np.uint8)
            if not np.any(mask):
                continue
            crop, cmask, origin = _extract_square(frame, mask, side)
            pending.append((i, crop, cmask, origin))
        if not pending:
            self._note_stats(batch=0, limited_by="frames", oom_retry=False)
            return out
        budget = int(getattr(self, "vram_budget_bytes", 0) or 0)
        cap = max(1, min(int(getattr(plan, "batch", 1) or 1), len(pending)))
        batch, limited, oom = self._run_batches(pending, cap, budget)
        for index, painted, _mask, origin in self._painted:
            _paste(out[index], painted, _mask, origin, feather)
        self._note_stats(batch=batch, limited_by=limited, oom_retry=oom)
        return out

    def _note_stats(self, *, batch: int, limited_by: str, oom_retry: bool) -> None:
        self.last_hole_stats = {
            "batch": int(batch),
            "limitedBy": limited_by,
            "oomRetry": bool(oom_retry),
            "lamaDtype": getattr(self, "_dtype", "fp32"),
        }

    def _run_batches(self, pending, cap: int, budget: int) -> tuple[int, str, bool]:
        from videoclean.application.hole_policy import grow_batch

        self._painted = []
        cursor = 0
        last = 1
        measured = 0
        oom = False
        limited = "frames"
        while cursor < len(pending):
            remain = len(pending) - cursor
            action, nxt = grow_batch(min(last, remain), remain, measured, budget or (1 << 62))
            if action == "frames" and last >= remain:
                take = remain
                limited = "frames"
            elif action == "try":
                take = min(nxt, cap, remain)
            elif action == "mid":
                take = min(nxt, cap, remain)
            else:
                take = min(last, remain)
                limited = "ceiling"
            if take < 1:
                take = 1
            chunk = pending[cursor : cursor + take]
            try:
                painted, delta = self._forward_crops([(c, m) for _, c, m, _ in chunk])
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower() or take == 1:
                    if take == 1:
                        raise
                    oom = True
                    take = max(1, (1 + take) // 2)
                    chunk = pending[cursor : cursor + take]
                    painted, delta = self._forward_crops([(c, m) for _, c, m, _ in chunk])
                else:
                    oom = True
                    mid = max(1, (last + take) // 2) if last < take else max(1, take // 2)
                    chunk = pending[cursor : cursor + mid]
                    painted, delta = self._forward_crops([(c, m) for _, c, m, _ in chunk])
                    take = mid
                    limited = "ceiling"
            measured = delta or measured
            last = take
            for (index, _c, mask, origin), frame in zip(chunk, painted):
                self._painted.append((index, frame, mask, origin))
            cursor += take
            if take < remain and limited == "ceiling":
                continue
            if cursor >= len(pending):
                limited = "frames" if last == cap or cap >= len(pending) else limited
        if cap < len(pending) and limited == "frames":
            limited = "ceiling"
        return last, limited, oom

    def _forward_crops(self, pairs: list[tuple[np.ndarray, np.ndarray]]) -> tuple[list[np.ndarray], int]:
        import torch

        images_t = []
        masks_t = []
        for crop, mask in pairs:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            binary = np.where(mask > 0, 255, 0).astype(np.uint8)
            image_t, mask_t = self._prepare(
                Image.fromarray(rgb),
                Image.fromarray(binary, mode="L"),
                self._torch_device,
            )
            if getattr(self, "_dtype", "fp32") == "fp16":
                image_t = image_t.to(dtype=torch.float16)
                mask_t = mask_t.to(dtype=torch.float16)
            images_t.append(image_t)
            masks_t.append(mask_t)
        image_b = torch.cat(images_t, dim=0)
        mask_b = torch.cat(masks_t, dim=0)
        baseline = 0
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            baseline = int(torch.cuda.memory_allocated())
        with torch.inference_mode():
            painted = self._model(image_b, mask_b)
        delta = 0
        if self.device == "cuda" and torch.cuda.is_available():
            delta = max(0, int(torch.cuda.max_memory_allocated()) - baseline)
            self.activation_delta_bytes = max(int(getattr(self, "activation_delta_bytes", 0)), delta)
        frames: list[np.ndarray] = []
        for i in range(painted.shape[0]):
            arr = painted[i].permute(1, 2, 0).detach().float().cpu().numpy()
            arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
            frames.append(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
        return frames, delta

    def _ensure(self) -> None:
        if self._model is not None:
            return
        ok, err = self._try_load()
        if not ok:
            raise AdapterUnavailable(err)

    def _try_load(self) -> tuple[bool, str]:
        if self._model is not None:
            return True, ""
        if self._load_error:
            return False, self._load_error
        try:
            import torch
            from simple_lama_inpainting.utils.util import prepare_img_and_mask
        except ImportError as exc:
            # prepare helpers ship with the package; fall back message if missing.
            try:
                import torch  # noqa: F401
            except ImportError as torch_exc:
                self._load_error = f"torch not installed ({torch_exc})"
                return False, self._load_error
            self._load_error = (
                f"simple-lama-inpainting helpers missing ({exc}). "
                "Install: uv sync --extra web --extra lama"
            )
            return False, self._load_error

        weights = find_weights()
        if weights is None:
            if not self.allow_download:
                self._load_error = self.status().removeprefix("unavailable: ")
                return False, self._load_error
            try:
                from simple_lama_inpainting.utils.util import download_model

                weights = Path(download_model(LAMA_MODEL_URL))
            except Exception as exc:  # noqa: BLE001
                self._load_error = f"failed to download big-lama.pt: {exc}"
                return False, self._load_error

        try:
            from videoclean.adapters.models.weights_cache import WeightKey, get_or_load
            from videoclean.application.hole_policy import probe_lama_dtype

            device = torch.device(self.device)

            def _load():
                # map_location is required: the public big-lama.pt was saved on CUDA.
                model = torch.jit.load(str(weights), map_location=device)
                model.eval()
                model.to(device)
                dtype = probe_lama_dtype(model, self.device)
                return model, dtype

            (model, dtype), loaded_now = get_or_load(
                WeightKey("lama", "big-lama", self.device, "fp16" if self.device == "cuda" else "fp32"),
                _load,
            )
            self._model = model
            self._dtype = dtype if loaded_now or getattr(model, "dtype", None) is not None else dtype
            self._dtype = dtype
            self.loaded_now = loaded_now
            self._torch_device = device
            self._prepare = prepare_img_and_mask
            return True, ""
        except Exception as exc:  # noqa: BLE001
            self._load_error = f"failed to load LaMa: {exc}"
            return False, self._load_error


def _box_of(mask: np.ndarray):
    from videoclean.application.hole_policy import mask_box

    return mask_box(mask)


def _extract_square(frame: np.ndarray, mask: np.ndarray, side: int):
    box = _box_of(mask)
    h, w = frame.shape[:2]
    if box is None:
        return frame[:side, :side], mask[:side, :side], (0, 0)
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    left = max(0, min(max(0, w - side), cx - side // 2))
    top = max(0, min(max(0, h - side), cy - side // 2))
    crop = np.zeros((side, side, 3), dtype=np.uint8)
    cmask = np.zeros((side, side), dtype=np.uint8)
    src = frame[top : top + side, left : left + side]
    sm = mask[top : top + side, left : left + side]
    crop[: src.shape[0], : src.shape[1]] = src
    cmask[: sm.shape[0], : sm.shape[1]] = sm
    return crop, cmask, (top, left)


def _paste(dst: np.ndarray, src: np.ndarray, mask: np.ndarray, origin: tuple[int, int], feather: int) -> None:
    top, left = origin
    h, w = dst.shape[:2]
    sh, sw = src.shape[:2]
    y2, x2 = min(h, top + sh), min(w, left + sw)
    if y2 <= top or x2 <= left:
        return
    patch = src[: y2 - top, : x2 - left]
    weight = (mask[: y2 - top, : x2 - left] > 0).astype(np.float32)
    if feather > 1 and weight.any():
        k = int(feather) * 2 + 1
        if k % 2 == 0:
            k += 1
        weight = cv2.GaussianBlur(weight, (k, k), 0)
    weight = weight[..., None]
    roi = dst[top:y2, left:x2].astype(np.float32)
    blended = roi * (1.0 - weight) + patch.astype(np.float32) * weight
    dst[top:y2, left:x2] = np.clip(blended, 0, 255).astype(np.uint8)
