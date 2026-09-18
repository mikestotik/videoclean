from __future__ import annotations

import os
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path

from videoclean.adapters.detectors.grounding_dino import DEFAULT_MODEL as DEFAULT_GROUNDING_DINO_MODEL
from videoclean.adapters.detectors.grounding_dino import GroundingDinoDetector
from videoclean.adapters.images import read_bgr, write_bgr
from videoclean.adapters.inpainters.lama import LamaInpainter
from videoclean.adapters.inpainters.propainter import ProPainterInpainter
from videoclean.adapters.media.ffmpeg import FFmpegMedia
from videoclean.adapters.llm.resolve import resolve_llm
from videoclean.adapters.prompt.llm import LlmPromptParser
from videoclean.adapters.segmenters.sam2 import Sam2Segmenter
from videoclean.adapters.segmenters.sam2_video import Sam2VideoSegmenter
from videoclean.application.config import (
    BACKENDS,
    DEFAULT_DETECTORS,
    DETECTORS,
    PipelineConfig,
    parse_name_list,
)
from videoclean.application.errors import AdapterUnavailable, DeviceUnavailable, PipelineError
from videoclean.application.ports.detector import Detector
from videoclean.application.ports.progress import ProgressPort
from videoclean.application.jobs.worker import JobWorker
from videoclean.application.use_cases.download_component import DownloadComponent
from videoclean.application.use_cases.package_media import PackageMedia
from videoclean.application.use_cases.run_cleanup import RunCleanup
from videoclean.store import JobIndex, JobPaths, new_job_id, resolve_data_dir, utc_now


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
    min_mask_coverage: float = 0.0004,
    verify_max_coverage: float = 0.12,
    prompt_frame_stride: int = 4,
    prompt_frame_max: int = 8,
    parse_chunk_frames: int = 0,
    vision_batch: int = 2,
    detector_keyframes: int | None = None,
    detector_nms_iou: float = 0.3,
    detector_max_box_area: float = 0.25,
    tracker_min_score: float = 0.55,
    tracker_max_template_area: float = 0.12,
    propainter_mask_dilation: int = 4,
    propainter_ref_stride: int = 10,
    propainter_neighbor_length: int = 10,
    propainter_subvideo_length: int = 80,
    propainter_raft_iter: int = 20,
    prompt_templates: str | None = None,
    profile: str = "custom",
    verify_max_passes: int | None = None,
    inpaint_workers: int | None = None,
    inpaint_chunk_overlap: int | None = None,
    require_runtime: bool = False,
) -> PipelineConfig:
    from videoclean.application.profiles import apply_profile

    raw = {
        "device": device.strip().lower(),
        "detector": detector,
        "detector_model": detector_model.strip(),
        "detector_threshold": detector_threshold,
        "segmenter": segmenter.strip().lower(),
        "segmenter_model": segmenter_model.strip(),
        "inpainter": inpainter.strip().lower(),
        "inpainter_model": inpainter_model.strip(),
        "llm_place": llm_place.strip().lower() or "auto",
        "llm_model": llm_model.strip(),
        "llm_base_url": llm_base_url.strip(),
        "llm_api_key": llm_api_key.strip(),
        "formats": formats or ["mp4"],
        "allow_download": allow_download,
        "verify": verify,
        "mask_dilate_px": int(mask_dilate_px),
        "min_mask_coverage": float(min_mask_coverage),
        "verify_max_coverage": float(verify_max_coverage),
        "prompt_frame_stride": int(prompt_frame_stride),
        "prompt_frame_max": int(prompt_frame_max),
        "parse_chunk_frames": int(parse_chunk_frames),
        "vision_batch": int(vision_batch),
        "detector_keyframes": detector_keyframes,
        "detector_nms_iou": float(detector_nms_iou),
        "detector_max_box_area": float(detector_max_box_area),
        "tracker_min_score": float(tracker_min_score),
        "tracker_max_template_area": float(tracker_max_template_area),
        "propainter_mask_dilation": int(propainter_mask_dilation),
        "propainter_ref_stride": int(propainter_ref_stride),
        "propainter_neighbor_length": int(propainter_neighbor_length),
        "propainter_subvideo_length": int(propainter_subvideo_length),
        "propainter_raft_iter": int(propainter_raft_iter),
        "prompt_templates": prompt_templates,
        "profile": (profile or "custom").strip().lower() or "custom",
    }
    if verify_max_passes is not None:
        raw["verify_max_passes"] = int(verify_max_passes)
    if inpaint_workers is not None:
        raw["inpaint_workers"] = int(inpaint_workers)
    if inpaint_chunk_overlap is not None:
        raw["inpaint_chunk_overlap"] = int(inpaint_chunk_overlap)
    merged = apply_profile(raw)
    cfg = PipelineConfig(
        device=str(merged["device"]),
        detectors=parse_name_list(merged.get("detector") or detector, DETECTORS, default=list(DEFAULT_DETECTORS)),
        detector_model=str(merged["detector_model"]),
        detector_threshold=float(merged["detector_threshold"]),
        segmenter=str(merged["segmenter"]),
        segmenter_model=str(merged["segmenter_model"]),
        inpainter=str(merged["inpainter"]),
        inpainter_model=str(merged["inpainter_model"]),
        llm_place=str(merged["llm_place"]),
        llm_model=str(merged["llm_model"]),
        llm_base_url=str(merged["llm_base_url"]),
        llm_api_key=str(merged["llm_api_key"]),
        formats=list(merged.get("formats") or ["mp4"]),
        allow_download=bool(merged["allow_download"]),
        verify=bool(merged["verify"]),
        mask_dilate_px=int(merged["mask_dilate_px"]),
        min_mask_coverage=float(merged["min_mask_coverage"]),
        verify_max_coverage=float(merged["verify_max_coverage"]),
        prompt_frame_stride=int(merged["prompt_frame_stride"]),
        prompt_frame_max=int(merged["prompt_frame_max"]),
        parse_chunk_frames=int(merged["parse_chunk_frames"]),
        vision_batch=int(merged["vision_batch"]),
        detector_keyframes=(
            None
            if merged.get("detector_keyframes") is None
            else int(merged["detector_keyframes"])
        ),
        detector_nms_iou=float(merged["detector_nms_iou"]),
        detector_max_box_area=float(merged["detector_max_box_area"]),
        tracker_min_score=float(merged["tracker_min_score"]),
        tracker_max_template_area=float(merged["tracker_max_template_area"]),
        propainter_mask_dilation=int(merged["propainter_mask_dilation"]),
        propainter_ref_stride=int(merged["propainter_ref_stride"]),
        propainter_neighbor_length=int(merged["propainter_neighbor_length"]),
        propainter_subvideo_length=int(merged["propainter_subvideo_length"]),
        propainter_raft_iter=int(merged["propainter_raft_iter"]),
        prompt_templates=merged.get("prompt_templates"),
        profile=str(merged.get("profile") or "custom"),
        verify_max_passes=int(merged.get("verify_max_passes", 1)),
        inpaint_workers=int(merged.get("inpaint_workers", 0)),
        inpaint_chunk_overlap=int(merged.get("inpaint_chunk_overlap", 8)),
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
    if mid:
        return mid
    return DEFAULT_GROUNDING_DINO_MODEL


def make_detector(name: str, cfg: PipelineConfig) -> Detector:
    if name == "grounding-dino":
        return GroundingDinoDetector(
            model_id=_detector_model_id(name, cfg),
            device=cfg.device,
            threshold=cfg.detector_threshold,
            allow_download=cfg.allow_download,
            keyframes=cfg.detector_keyframes,
            nms_iou=cfg.detector_nms_iou,
            max_box_area=cfg.detector_max_box_area,
            tracker_min_score=cfg.tracker_min_score,
            tracker_max_template_area=cfg.tracker_max_template_area,
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
    if cfg.inpainter == "lama":
        return LamaInpainter(device=cfg.device, allow_download=cfg.allow_download)
    if cfg.inpainter == "propainter":
        return ProPainterInpainter(
            model_id=cfg.inpainter_model,
            device=cfg.device,
            allow_download=cfg.allow_download,
            mask_dilation=cfg.propainter_mask_dilation,
            ref_stride=cfg.propainter_ref_stride,
            neighbor_length=cfg.propainter_neighbor_length,
            subvideo_length=cfg.propainter_subvideo_length,
            raft_iter=cfg.propainter_raft_iter,
        )
    raise PipelineError(f"no factory for inpainter {cfg.inpainter!r}")


def make_parser(cfg: PipelineConfig):
    return LlmPromptParser(
        resolve_llm(cfg),
        vision_batch=cfg.vision_batch,
        templates_dir=cfg.prompt_templates,
    )


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


def build_run_preview(
    cfg: PipelineConfig,
    progress: ProgressPort,
    jobs: JobIndex | None = None,
    job_id: str | None = None,
):
    from videoclean.application.use_cases.run_preview import RunPreview

    id_factory = (lambda: job_id) if job_id else new_job_id
    return RunPreview(
        media=FFmpegMedia(),
        parser=make_parser(cfg),
        detectors=[make_detector(name, cfg) for name in cfg.detectors],
        segmenter=make_segmenter(cfg),
        jobs=jobs or JobIndex(Path.home() / ".videoclean" / "jobs.sqlite"),
        progress=progress,
        new_job_id=id_factory,
        make_paths=PreviewPathsFactory,
        read_image=read_bgr,
        write_image=write_bgr,
    )


def PreviewPathsFactory(root: Path):
    from videoclean.application.use_cases.run_preview import PreviewPaths

    return PreviewPaths.create(root)


def PromptPathsFactory(root: Path):
    from videoclean.application.use_cases.build_prompt import PromptPaths

    return PromptPaths.create(root)


def build_build_prompt(
    cfg: PipelineConfig,
    progress: ProgressPort | None,
    jobs: JobIndex | None = None,
    job_id: str | None = None,
):
    from videoclean.adapters.prompt.frames import bgr_to_jpeg
    from videoclean.adapters.prompt.llm import interpret_system_prompt
    from videoclean.application.use_cases.build_prompt import BuildPrompt

    id_factory = (lambda: job_id) if job_id else new_job_id
    return BuildPrompt(
        media=FFmpegMedia(),
        llm=resolve_llm(cfg),
        system_prompt=interpret_system_prompt(cfg.prompt_templates),
        jobs=jobs or JobIndex(Path.home() / ".videoclean" / "jobs.sqlite"),
        progress=progress,
        new_job_id=id_factory,
        make_paths=PromptPathsFactory,
        read_image=read_bgr,
        to_jpeg=bgr_to_jpeg,
    )


def build_packager() -> PackageMedia:
    return PackageMedia(FFmpegMedia())


def build_job_worker(data_dir: Path, jobs: JobIndex) -> JobWorker:
    from videoclean.adapters.progress.job_store import ProgressBridge

    def factory(cfg, progress, jobs, job_id):
        return build_run_cleanup(
            cfg,
            progress or ProgressBridge(jobs, job_id),
            jobs,
            job_id=job_id,
        )

    def preview_factory(cfg, progress, jobs, job_id):
        return build_run_preview(
            cfg,
            progress or ProgressBridge(jobs, job_id),
            jobs,
            job_id=job_id,
        )

    def prompt_factory(cfg, progress, jobs, job_id):
        return build_build_prompt(
            cfg,
            progress or ProgressBridge(jobs, job_id),
            jobs,
            job_id=job_id,
        )

    return JobWorker(
        data_dir=data_dir,
        jobs=jobs,
        build_runner=factory,
        build_preview_runner=preview_factory,
        build_prompt_runner=prompt_factory,
    )


def build_catalog(jobs: JobIndex | None = None):
    from videoclean.adapters.models.catalog import ModelCatalog

    return ModelCatalog(jobs=jobs)


def build_downloader() -> DownloadComponent:
    from videoclean.adapters.models.downloaders import run_download

    return DownloadComponent(run_download)


def default_serve_port(env: Mapping[str, str] | None = None) -> int:
    env_map = os.environ if env is None else env
    raw = str(env_map.get("VIDEOCLEAN_PORT") or "7860").strip()
    try:
        return int(raw)
    except ValueError:
        return 7860


def parse_torch_version(version: str) -> tuple[int, int] | None:
    text = (version or "").strip()
    if not text or text in {"not imported", "missing"}:
        return None
    core = text.split("+", 1)[0].split(".dev", 1)[0]
    parts = core.split(".")
    try:
        return int(parts[0]), int(parts[1])
    except (IndexError, ValueError):
        return None


def sam2_video_torch_warning(segmenter: str, torch_ver: str) -> str | None:
    if (segmenter or "").strip().lower() != "sam2-video":
        return None
    parsed = parse_torch_version(torch_ver)
    if parsed is None or parsed >= (2, 5):
        return None
    return f"warning: sam2-video typically needs torch>=2.5 (this machine has {torch_ver})"


def extra_doctor_warnings(cfg: PipelineConfig, facts: Mapping[str, str]) -> list[str]:
    warning = sam2_video_torch_warning(cfg.segmenter, facts.get("torch", ""))
    return [warning] if warning else []


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
    machine.extend(extra_doctor_warnings(cfg, facts))
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
        f"vision-frames    stride={cfg.prompt_frame_stride}  max={cfg.prompt_frame_max}  batch={cfg.vision_batch}",
        f"detector-tune    keyframes={cfg.detector_keyframes or 'default'}  nms-iou={cfg.detector_nms_iou}  max-box-area={cfg.detector_max_box_area}",
        f"tracker-tune     min-score={cfg.tracker_min_score}  max-template-area={cfg.tracker_max_template_area}",
        f"propainter-tune  dilate={cfg.propainter_mask_dilation}  ref-stride={cfg.propainter_ref_stride}  neighbor={cfg.propainter_neighbor_length}  subvideo={cfg.propainter_subvideo_length}  raft={cfg.propainter_raft_iter}",
        f"prompt-templates {cfg.prompt_templates or 'builtin'}",
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
