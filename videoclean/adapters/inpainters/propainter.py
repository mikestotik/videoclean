from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from videoclean.adapters.hf_cache import download_hint, hf_cached, hf_hub_dir
from videoclean.application.errors import AdapterUnavailable

WEIGHT_FILES = ("ProPainter.pth", "raft-things.pth", "recurrent_flow_completion.pth")
DEFAULT_HF_REPO = "camenduru/ProPainter"


class ProPainterInpainter:
    """Video-aware fill. Needs the official ProPainter repo on disk plus the three .pth weights."""

    name = "propainter"
    device_note = "Uses --device. CUDA strongly recommended."
    video_aware = True

    def __init__(
        self,
        model_id: str,
        device: str,
        allow_download: bool = False,
        mask_dilation: int = 4,
        ref_stride: int = 10,
        neighbor_length: int = 10,
        subvideo_length: int = 80,
        raft_iter: int = 20,
    ) -> None:
        self.model_id = model_id.strip() or DEFAULT_HF_REPO
        self.device = device
        self.allow_download = allow_download
        self.mask_dilation = mask_dilation
        self.ref_stride = ref_stride
        self.neighbor_length = neighbor_length
        self.subvideo_length = subvideo_length
        self.raft_iter = raft_iter
        self._loaded = False
        self._raft = None
        self._flow_complete = None
        self._model = None
        self._to_tensors = None
        self._load_error: str | None = None

    def status(self) -> str:
        if self._loaded:
            return f"ready (vendor+weights on {self.device})"
        if self._load_error:
            return f"unavailable: {self._load_error}"
        root = find_vendor()
        weights = find_weights(self.model_id)
        missing: list[str] = []
        if root is None:
            missing.append(
                "clone https://github.com/sczhou/ProPainter.git to ~/.videoclean/vendor/ProPainter"
            )
        if weights is None:
            missing.append(
                f"weights ({', '.join(WEIGHT_FILES)}). "
                f"{download_hint(self.model_id)}  --local-dir ~/.videoclean/weights/propainter"
            )
        if missing:
            return "unavailable: " + " | ".join(missing)
        return f"ready (code+weights on disk, loads on first inpaint, device={self.device})"

    def inpaint(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return self.inpaint_clip([frame], [mask])[0]

    def inpaint_clip(self, frames: list[np.ndarray], masks: list[np.ndarray]) -> list[np.ndarray]:
        if not frames:
            return []
        self._ensure()
        return self._run(frames, masks)

    def _ensure(self) -> None:
        if self._loaded:
            return
        ok, err = self._try_load()
        if not ok:
            raise AdapterUnavailable(err)

    def _try_load(self) -> tuple[bool, str]:
        if self._loaded:
            return True, ""
        if self._load_error:
            return False, self._load_error
        root = find_vendor()
        weights = find_weights(self.model_id)
        if weights is None and self.allow_download:
            weights = _download_weights(self.model_id)
        if root is None or weights is None:
            self._load_error = self.status().removeprefix("unavailable: ")
            return False, self._load_error
        try:
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            import torch
            from core.utils import to_tensors
            from model.modules.flow_comp_raft import RAFT_bi
            from model.propainter import InpaintGenerator
            from model.recurrent_flow_completion import RecurrentFlowCompleteNet

            device = torch.device(self.device)
            self._raft = RAFT_bi(str(weights / "raft-things.pth"), device)
            self._flow_complete = RecurrentFlowCompleteNet(str(weights / "recurrent_flow_completion.pth"))
            for p in self._flow_complete.parameters():
                p.requires_grad = False
            self._flow_complete.to(device)
            self._flow_complete.eval()
            self._model = InpaintGenerator(model_path=str(weights / "ProPainter.pth")).to(device)
            self._model.eval()
            self._to_tensors = to_tensors
            self._loaded = True
            return True, ""
        except Exception as exc:  # noqa: BLE001
            self._load_error = (
                f"{type(exc).__name__}: {exc}"[:240]
                + ". Install ProPainter deps on the GPU box: uv pip install scipy einops timm"
            )
            return False, self._load_error

    def _run(self, frames_bgr: list[np.ndarray], masks_u8: list[np.ndarray]) -> list[np.ndarray]:
        import torch

        device = torch.device(self.device)
        use_half = self.device == "cuda"
        orig_h, orig_w = frames_bgr[0].shape[:2]
        process_w = orig_w - orig_w % 8
        process_h = orig_h - orig_h % 8
        pil_frames = [
            Image.fromarray(cv2.cvtColor(cv2.resize(f, (process_w, process_h)), cv2.COLOR_BGR2RGB))
            for f in frames_bgr
        ]
        flow_masks, masks_dilated = _prepare_masks(
            [_resize_mask(m, process_h, process_w) for m in masks_u8],
            self.mask_dilation,
        )
        frames_inp = [np.array(f).astype(np.uint8) for f in pil_frames]
        frames = self._to_tensors()(pil_frames).unsqueeze(0) * 2 - 1
        flow_masks_t = self._to_tensors()(flow_masks).unsqueeze(0)
        masks_dilated_t = self._to_tensors()(masks_dilated).unsqueeze(0)
        frames = frames.to(device)
        flow_masks_t = flow_masks_t.to(device)
        masks_dilated_t = masks_dilated_t.to(device)

        video_length = frames.size(1)
        h, w = process_h, process_w
        with torch.no_grad():
            gt_flows_bi = self._raft_flows(frames, video_length)
            if use_half:
                frames = frames.half()
                flow_masks_t = flow_masks_t.half()
                masks_dilated_t = masks_dilated_t.half()
                gt_flows_bi = (gt_flows_bi[0].half(), gt_flows_bi[1].half())
                self._flow_complete = self._flow_complete.half()
                self._model = self._model.half()
            pred_flows_bi = self._complete_flow(gt_flows_bi, flow_masks_t)
            updated_frames, updated_masks = self._propagate(frames, pred_flows_bi, masks_dilated_t, h, w)
            comp = self._transformer(
                frames_inp,
                updated_frames,
                updated_masks,
                masks_dilated_t,
                pred_flows_bi,
                video_length,
                h,
                w,
            )
        out: list[np.ndarray] = []
        for rgb in comp:
            bgr = cv2.cvtColor(cv2.resize(rgb, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC), cv2.COLOR_RGB2BGR)
            out.append(bgr)
        if self.device == "cuda":
            torch.cuda.empty_cache()
        return out

    def _raft_flows(self, frames, video_length):
        import torch

        if frames.size(-1) <= 640:
            short_clip_len = 12
        elif frames.size(-1) <= 720:
            short_clip_len = 8
        elif frames.size(-1) <= 1280:
            short_clip_len = 4
        else:
            short_clip_len = 2
        if frames.size(1) > short_clip_len:
            gt_flows_f_list, gt_flows_b_list = [], []
            for f in range(0, video_length, short_clip_len):
                end_f = min(video_length, f + short_clip_len)
                chunk = frames[:, f:end_f] if f == 0 else frames[:, f - 1 : end_f]
                flows_f, flows_b = self._raft(chunk, iters=self.raft_iter)
                gt_flows_f_list.append(flows_f)
                gt_flows_b_list.append(flows_b)
            return torch.cat(gt_flows_f_list, dim=1), torch.cat(gt_flows_b_list, dim=1)
        return self._raft(frames, iters=self.raft_iter)

    def _complete_flow(self, gt_flows_bi, flow_masks_t):
        import torch

        flow_length = gt_flows_bi[0].size(1)
        if flow_length > self.subvideo_length:
            pred_flows_f, pred_flows_b = [], []
            pad_len = 5
            for f in range(0, flow_length, self.subvideo_length):
                s_f = max(0, f - pad_len)
                e_f = min(flow_length, f + self.subvideo_length + pad_len)
                pad_len_s = max(0, f) - s_f
                pad_len_e = e_f - min(flow_length, f + self.subvideo_length)
                pred_flows_bi_sub, _ = self._flow_complete.forward_bidirect_flow(
                    (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]),
                    flow_masks_t[:, s_f : e_f + 1],
                )
                pred_flows_bi_sub = self._flow_complete.combine_flow(
                    (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]),
                    pred_flows_bi_sub,
                    flow_masks_t[:, s_f : e_f + 1],
                )
                pred_flows_f.append(pred_flows_bi_sub[0][:, pad_len_s : e_f - s_f - pad_len_e])
                pred_flows_b.append(pred_flows_bi_sub[1][:, pad_len_s : e_f - s_f - pad_len_e])
            return torch.cat(pred_flows_f, dim=1), torch.cat(pred_flows_b, dim=1)
        pred_flows_bi, _ = self._flow_complete.forward_bidirect_flow(gt_flows_bi, flow_masks_t)
        return self._flow_complete.combine_flow(gt_flows_bi, pred_flows_bi, flow_masks_t)

    def _propagate(self, frames, pred_flows_bi, masks_dilated_t, h, w):
        import torch

        video_length = frames.size(1)
        masked_frames = frames * (1 - masks_dilated_t)
        sub = min(100, self.subvideo_length)
        if video_length > sub:
            updated_frames, updated_masks = [], []
            pad_len = 10
            for f in range(0, video_length, sub):
                s_f = max(0, f - pad_len)
                e_f = min(video_length, f + sub + pad_len)
                pad_len_s = max(0, f) - s_f
                pad_len_e = e_f - min(video_length, f + sub)
                b, t, _, _, _ = masks_dilated_t[:, s_f:e_f].size()
                pred_flows_bi_sub = (pred_flows_bi[0][:, s_f : e_f - 1], pred_flows_bi[1][:, s_f : e_f - 1])
                prop_imgs_sub, updated_local_masks_sub = self._model.img_propagation(
                    masked_frames[:, s_f:e_f], pred_flows_bi_sub, masks_dilated_t[:, s_f:e_f], "nearest"
                )
                updated_frames_sub = frames[:, s_f:e_f] * (1 - masks_dilated_t[:, s_f:e_f]) + prop_imgs_sub.view(
                    b, t, 3, h, w
                ) * masks_dilated_t[:, s_f:e_f]
                updated_frames.append(updated_frames_sub[:, pad_len_s : e_f - s_f - pad_len_e])
                updated_masks.append(updated_local_masks_sub.view(b, t, 1, h, w)[:, pad_len_s : e_f - s_f - pad_len_e])
            return torch.cat(updated_frames, dim=1), torch.cat(updated_masks, dim=1)
        b, t, _, _, _ = masks_dilated_t.size()
        prop_imgs, updated_local_masks = self._model.img_propagation(
            masked_frames, pred_flows_bi, masks_dilated_t, "nearest"
        )
        updated_frames = frames * (1 - masks_dilated_t) + prop_imgs.view(b, t, 3, h, w) * masks_dilated_t
        return updated_frames, updated_local_masks.view(b, t, 1, h, w)

    def _transformer(
        self,
        ori_frames,
        updated_frames,
        updated_masks,
        masks_dilated_t,
        pred_flows_bi,
        video_length,
        h,
        w,
    ):
        import torch

        comp_frames = [None] * video_length
        neighbor_stride = max(1, self.neighbor_length // 2)
        ref_num = self.subvideo_length // self.ref_stride if video_length > self.subvideo_length else -1
        for f in range(0, video_length, neighbor_stride):
            neighbor_ids = list(range(max(0, f - neighbor_stride), min(video_length, f + neighbor_stride + 1)))
            ref_ids = _ref_index(f, neighbor_ids, video_length, self.ref_stride, ref_num)
            selected_imgs = updated_frames[:, neighbor_ids + ref_ids, :, :, :]
            selected_masks = masks_dilated_t[:, neighbor_ids + ref_ids, :, :, :]
            selected_update_masks = updated_masks[:, neighbor_ids + ref_ids, :, :, :]
            selected_pred_flows_bi = (
                pred_flows_bi[0][:, neighbor_ids[:-1], :, :, :],
                pred_flows_bi[1][:, neighbor_ids[:-1], :, :, :],
            )
            with torch.no_grad():
                l_t = len(neighbor_ids)
                pred_img = self._model(selected_imgs, selected_pred_flows_bi, selected_masks, selected_update_masks, l_t)
                pred_img = pred_img.view(-1, 3, h, w)
                pred_img = (pred_img + 1) / 2
                pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy() * 255
                binary_masks = (
                    masks_dilated_t[0, neighbor_ids, :, :, :].cpu().permute(0, 2, 3, 1).numpy().astype(np.uint8)
                )
                for i, idx in enumerate(neighbor_ids):
                    img = np.array(pred_img[i]).astype(np.uint8) * binary_masks[i] + ori_frames[idx] * (
                        1 - binary_masks[i]
                    )
                    if comp_frames[idx] is None:
                        comp_frames[idx] = img
                    else:
                        comp_frames[idx] = comp_frames[idx].astype(np.float32) * 0.5 + img.astype(np.float32) * 0.5
                    comp_frames[idx] = comp_frames[idx].astype(np.uint8)
        return comp_frames


def find_vendor() -> Path | None:
    env = os.environ.get("VIDEOCLEAN_PROPAINTER_ROOT", "").strip()
    candidates = []
    if env:
        candidates.append(Path(env).expanduser())
    candidates.append(Path.home() / ".videoclean" / "vendor" / "ProPainter")
    for path in candidates:
        if (path / "model" / "propainter.py").is_file():
            return path
    return None


def find_weights(model_id: str) -> Path | None:
    env = os.environ.get("VIDEOCLEAN_PROPAINTER_WEIGHTS", "").strip()
    candidates = []
    if env:
        candidates.append(Path(env).expanduser())
    local = Path(model_id).expanduser()
    if local.is_dir():
        candidates.append(local)
    candidates.append(Path.home() / ".videoclean" / "weights" / "propainter")
    vendor = find_vendor()
    if vendor:
        candidates.append(vendor / "weights")
    if "/" in model_id and not local.is_dir():
        hub = hf_hub_dir(model_id)
        if hub.is_dir():
            snapshots = hub / "snapshots"
            if snapshots.is_dir():
                for snap in snapshots.iterdir():
                    candidates.append(snap)
            candidates.append(hub)
    for path in candidates:
        if path.is_dir() and all((path / name).is_file() for name in WEIGHT_FILES):
            return path
    return None


def _download_weights(model_id: str) -> Path | None:
    dest = Path.home() / ".videoclean" / "weights" / "propainter"
    dest.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=model_id, local_dir=str(dest))
    except Exception:  # noqa: BLE001
        return None
    return dest if all((dest / name).is_file() for name in WEIGHT_FILES) else None


def _prepare_masks(masks: list[np.ndarray], dilate: int) -> tuple[list[Image.Image], list[Image.Image]]:
    flow_masks: list[Image.Image] = []
    dilated: list[Image.Image] = []
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate * 2 + 1, dilate * 2 + 1)) if dilate > 0 else None
    for mask in masks:
        binary = np.where(mask > 0, 255, 0).astype(np.uint8)
        flow = cv2.dilate(binary, k) if k is not None else binary
        hole = cv2.dilate(binary, k) if k is not None else binary
        flow_masks.append(Image.fromarray(flow))
        dilated.append(Image.fromarray(hole))
    return flow_masks, dilated


def _resize_mask(mask: np.ndarray, h: int, w: int) -> np.ndarray:
    if mask.shape[0] == h and mask.shape[1] == w:
        return mask
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)


def _ref_index(mid: int, neighbor_ids: list[int], length: int, ref_stride: int, ref_num: int) -> list[int]:
    ref_index: list[int] = []
    if ref_num == -1:
        for i in range(0, length, ref_stride):
            if i not in neighbor_ids:
                ref_index.append(i)
        return ref_index
    start_idx = max(0, mid - ref_stride * (ref_num // 2))
    end_idx = min(length, mid + ref_stride * (ref_num // 2))
    for i in range(start_idx, end_idx, ref_stride):
        if i not in neighbor_ids:
            if len(ref_index) >= ref_num:
                break
            ref_index.append(i)
    return ref_index
