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
        return [self.inpaint(frame, mask) for frame, mask in zip(frames, masks)]

    def inpaint(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        if not np.any(mask):
            return frame
        self._ensure()
        import torch

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        binary = np.where(mask > 0, 255, 0).astype(np.uint8)
        image_t, mask_t = self._prepare(
            Image.fromarray(rgb),
            Image.fromarray(binary, mode="L"),
            self._torch_device,
        )
        with torch.inference_mode():
            inpainted = self._model(image_t, mask_t)
        out = inpainted[0].permute(1, 2, 0).detach().cpu().numpy()
        out = np.clip(out * 255, 0, 255).astype(np.uint8)
        out = out[:h, :w]
        return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

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
            device = torch.device(self.device)
            # map_location is required: the public big-lama.pt was saved on CUDA.
            model = torch.jit.load(str(weights), map_location=device)
            model.eval()
            model.to(device)
            self._model = model
            self._torch_device = device
            self._prepare = prepare_img_and_mask
            return True, ""
        except Exception as exc:  # noqa: BLE001
            self._load_error = f"failed to load LaMa: {exc}"
            return False, self._load_error
