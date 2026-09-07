from __future__ import annotations

import shutil
import sys
from pathlib import Path

from videoclean.adapters.detectors.grounding_dino import DEFAULT_MODEL as DEFAULT_GROUNDING_DINO_MODEL
from videoclean.adapters.detectors.grounding_dino import GroundingDinoDetector
from videoclean.adapters.detectors.owlvit import OwlVitDetector
from videoclean.adapters.images import read_bgr, write_bgr
from videoclean.adapters.inpainters.lama import LamaInpainter
from videoclean.adapters.inpainters.opencv_telea import OpenCvTeleaInpainter
from videoclean.adapters.inpainters.propainter import ProPainterInpainter
from videoclean.adapters.media.ffmpeg import FFmpegMedia
from videoclean.adapters.llm.resolve import resolve_llm
from videoclean.adapters.prompt.llm import LlmPromptParser
from videoclean.adapters.segmenters.sam2 import Sam2Segmenter
from videoclean.adapters.segmenters.sam2_video import Sam2VideoSegmenter
from videoclean.application.config import (
    BACKENDS,
    DEFAULT_DETECTOR_MODEL,
    DEFAULT_DETECTORS,
    DETECTORS,
    PipelineConfig,
    parse_name_list,
)
from videoclean.application.errors import AdapterUnavailable, DeviceUnavailable, PipelineError
from videoclean.application.ports.detector import Detector
from videoclean.application.ports.progress import ProgressPort
from videoclean.application.use_cases.package_media import PackageMedia
from videoclean.application.use_cases.run_cleanup import RunCleanup
from videoclean.store import JobIndex, JobPaths, new_job_id, utc_now


def config_from_flags(
    *,
    device: str,
    detector: str,
    detector_model: str,
    detector_threshold: float,
    segmenter: str,
    segmenter_model: str,
    inpainter: str,
    inpainter_model: str,
    formats: list[str] | None,
    allow_download: bool,
    require_device: bool = False,
    llm_place: str = "auto",
    llm_model: str = "",
    llm_base_url: str = "",
    llm_api_key: str = "",
    verify: bool = True,
    mask_dilate_px: int = 3,
    telea_radius: int = 9,
    min_mask_coverage: float = 0.0004,
    verify_max_coverage: float = 0.12,
    prompt_frame_stride: int = 4,
    prompt_frame_max: int = 8,
    require_runtime: bool = False,
) -> PipelineConfig:
    cfg = PipelineConfig(
        device=device.strip().lower(),
        detectors=parse_name_list(detector, DETECTORS, default=list(DEFAULT_DETECTORS)),
        detector_model=detector_model.strip(),
        detector_threshold=detector_threshold,
        segmenter=segmenter.strip().lower(),
        segmenter_model=segmenter_model.strip(),
        inpainter=inpainter.strip().lower(),
        inpainter_model=inpainter_model.strip(),
        llm_place=llm_place.strip().lower() or "auto",
        llm_model=llm_model.strip(),
        llm_base_url=llm_base_url.strip(),
        llm_api_key=llm_api_key.strip(),
        formats=formats or ["mp4"],
        allow_download=allow_download,
        verify=verify,
        mask_dilate_px=int(mask_dilate_px),
        telea_radius=int(telea_radius),
        min_mask_coverage=float(min_mask_coverage),
        verify_max_coverage=float(verify_max_coverage),
        prompt_frame_stride=int(prompt_frame_stride),
        prompt_frame_max=int(prompt_frame_max),
    )
    cfg.validate()
    if require_device:
        resolve_device(cfg.device)
    if require_runtime:
        ensure_runtime(cfg)
    return cfg


def resolve_device(device: str) -> str:
    if device == "cpu":
        return "cpu"
    try:
        import torch
    except ImportError as exc:
        raise DeviceUnavailable("torch is required for --device cuda|mps") from exc
    if device == "cuda":
        if not torch.cuda.is_available():
            raise DeviceUnavailable("CUDA requested, but torch.cuda.is_available() is False. Use --device cpu.")
        return "cuda"
    if device == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise DeviceUnavailable("MPS requested, but it is not available on this machine. Use --device cpu.")
        return "mps"
    raise PipelineError(f"--device must be cpu | cuda | mps, got {device!r}")


def _detector_model_id(name: str, cfg: PipelineConfig) -> str:
    mid = (cfg.detector_model or "").strip()
    if name == "grounding-dino":
        if mid and "owlvit" not in mid.lower():
            return mid
        return DEFAULT_GROUNDING_DINO_MODEL
    if name == "owlvit":
        if mid and "grounding" not in mid.lower() and "dino" not in mid.lower():
            return mid
        return DEFAULT_DETECTOR_MODEL
    return mid


def make_detector(name: str, cfg: PipelineConfig) -> Detector:
    if name == "grounding-dino":
        return GroundingDinoDetector(
            model_id=_detector_model_id(name, cfg),
            device=cfg.device,
            threshold=cfg.detector_threshold,
            allow_download=cfg.allow_download,
        )
    if name == "owlvit":
        return OwlVitDetector(
            model_id=_detector_model_id(name, cfg),
            device=cfg.device,
            threshold=cfg.detector_threshold,
            allow_download=cfg.allow_download,
        )
    raise PipelineError(f"no factory for detector {name!r}")


def make_segmenter(cfg: PipelineConfig):
    if cfg.segmenter == "sam2":
        return Sam2Segmenter(
            model_id=cfg.segmenter_model,
            device=cfg.device,
            allow_download=cfg.allow_download,
            dilate_px=cfg.mask_dilate_px,
        )
    if cfg.segmenter == "sam2-video":
        return Sam2VideoSegmenter(
            model_id=cfg.segmenter_model,
            device=cfg.device,
            allow_download=cfg.allow_download,
            dilate_px=cfg.mask_dilate_px,
        )
    raise PipelineError(f"no factory for segmenter {cfg.segmenter!r}")


def make_inpainter(cfg: PipelineConfig):
    if cfg.inpainter == "opencv-telea":
        return OpenCvTeleaInpainter(radius=cfg.telea_radius)
    if cfg.inpainter == "lama":
        return LamaInpainter(device=cfg.device, allow_download=cfg.allow_download)
    if cfg.inpainter == "propainter":
        return ProPainterInpainter(
            model_id=cfg.inpainter_model,
            device=cfg.device,
            allow_download=cfg.allow_download,
        )
    raise PipelineError(f"no factory for inpainter {cfg.inpainter!r}")


def make_parser(cfg: PipelineConfig):
    return LlmPromptParser(resolve_llm(cfg))


def ensure_runtime(cfg: PipelineConfig) -> None:
    """Ping LLM and check detector weights before extract_frames."""
    parser = make_parser(cfg)
    st = parser.status()
    if not st.startswith("ready"):
        raise AdapterUnavailable(f"prompt-parser llm: {st}")
    notes: list[str] = []
    ready = False
    for name in cfg.detectors:
        det = make_detector(name, cfg)
        ds = det.status()
        notes.append(f"{name}: {ds}")
        if ds.startswith("ready"):
            ready = True
    if not ready:
        raise AdapterUnavailable("no detector could run. " + " | ".join(notes))


def build_run_cleanup(
    cfg: PipelineConfig,
    progress: ProgressPort,
    jobs: JobIndex | None = None,
    job_id: str | None = None,
) -> RunCleanup:
    id_factory = (lambda: job_id) if job_id else new_job_id
    return RunCleanup(
        media=FFmpegMedia(),
        parser=make_parser(cfg),
        detectors=[make_detector(name, cfg) for name in cfg.detectors],
        segmenter=make_segmenter(cfg),
        inpainter=make_inpainter(cfg),
        jobs=jobs or JobIndex(Path.home() / ".videoclean" / "jobs.sqlite"),
        progress=progress,
        new_job_id=id_factory,
        make_paths=JobPaths.create,
        utc_now=utc_now,
        read_image=read_bgr,
        write_image=write_bgr,
    )


def build_packager() -> PackageMedia:
    return PackageMedia(FFmpegMedia())


def estimate_seconds(frame_count: int, width: int, height: int, device: str = "cpu") -> float:
    megapixels = (width * height) / 1_000_000
    per_frame = megapixels * (0.05 if device in {"cuda", "mps"} else 0.12)
    decode = frame_count * 0.04 + 2.0
    encode = 3.0 + megapixels * 0.3
    return max(8.0, decode + frame_count * per_frame + encode)


def machine_facts() -> dict[str, str]:
    cuda = "no"
    mps = "no"
    torch_ver = "not imported"
    try:
        import torch

        torch_ver = torch.__version__
        cuda = "yes" if torch.cuda.is_available() else "no"
        mps_mod = getattr(torch.backends, "mps", None)
        mps = "yes" if mps_mod is not None and mps_mod.is_available() else "no"
    except Exception:  # noqa: BLE001
        pass
    import cv2

    return {
        "python": sys.version.split()[0],
        "ffmpeg": shutil.which("ffmpeg") or "missing",
        "ffprobe": shutil.which("ffprobe") or "missing",
        "opencv": cv2.__version__,
        "torch": torch_ver,
        "cuda": cuda,
        "mps": mps,
    }


def doctor_sections(cfg: PipelineConfig) -> list[tuple[str, list[str]]]:
    facts = machine_facts()
    machine = [
        f"Python     {facts['python']}",
        f"FFmpeg     {facts['ffmpeg']}",
        f"ffprobe    {facts['ffprobe']}",
        f"OpenCV     {facts['opencv']}",
        f"torch      {facts['torch']}   cuda={facts['cuda']}  mps={facts['mps']}",
    ]
    device_line = f"device           {cfg.device}   (ML adapters only)"
    try:
        resolve_device(cfg.device)
        device_line += "  available"
    except DeviceUnavailable as exc:
        device_line += f"  NOT available — {exc}"
    requested = [
        device_line,
        f"detector         {', '.join(cfg.detectors)}",
        f"detector-model   {cfg.detector_model}",
        f"detector-thr     {cfg.detector_threshold}",
        f"segmenter        {cfg.segmenter}",
        f"segmenter-model  {cfg.segmenter_model or '—'}",
        f"inpainter        {cfg.inpainter}",
        f"inpainter-model  {cfg.inpainter_model or '—'}",
        f"prompt-parser    llm",
        f"verify           {cfg.verify}",
        f"llm              {cfg.llm_place}  model={cfg.llm_model or '—'}  url={cfg.llm_base_url or '—'}",
        f"vision-frames    stride={cfg.prompt_frame_stride}  max={cfg.prompt_frame_max}",
        f"format           {', '.join(cfg.formats)}",
        "media            ffmpeg   (always, not a flag)",
    ]
    adapters: list[str] = []
    adapters.append(f"ffmpeg           {'ready' if facts['ffmpeg'] != 'missing' else 'MISSING'}")
    for name in cfg.detectors:
        det = make_detector(name, cfg)
        adapters.append(f"{name:<16} {det.status()}")
    seg = make_segmenter(cfg)
    adapters.append(f"{seg.name:<16} {seg.status()}")
    inp = make_inpainter(cfg)
    adapters.append(f"{inp.name:<16} {inp.status()}  ({inp.device_note})")
    parser = make_parser(cfg)
    pst = parser.status() if hasattr(parser, "status") else "ready"
    adapters.append(f"{parser.name:<16} {pst}")
    return [
        ("This machine", machine),
        ("This command (independent flags, not a profile)", requested),
        ("Adapters that will actually run", adapters),
    ]


def backends_rows() -> list[dict[str, str]]:
    return [dict(row) for row in BACKENDS]
