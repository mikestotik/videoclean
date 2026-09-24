from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from videoclean.application.errors import AdapterNotImplemented, PipelineError
from videoclean.domain.formats import parse_formats

DEVICES = ("cpu", "cuda", "mps")

# Names the product understands. Wiring (or "not implemented") lives in composition.
DETECTORS = ("grounding-dino",)
SEGMENTERS = ("sam2", "sam2-video")
INPAINTERS = ("lama", "propainter")
LLM_PLACES = ("auto", "local", "cloud")

READY_ADAPTERS = {
    "detector": frozenset({"grounding-dino"}),
    "segmenter": frozenset({"sam2", "sam2-video"}),
    "inpainter": frozenset({"lama", "propainter"}),
}

DEFAULT_GROUNDING_DINO_MODEL = "IDEA-Research/grounding-dino-tiny"
DEFAULT_SEGMENTER_MODEL = "facebook/sam2-hiera-tiny"
DEFAULT_INPAINTER_MODEL = "camenduru/ProPainter"
DEFAULT_DETECTORS = ["grounding-dino"]

# What each flag actually controls. Doctor / backends print this.
PORT_HELP = (
    ("media", "(fixed)", "ffmpeg", "Read and write video. Not a switch."),
    ("device", "--device", "cpu | cuda | mps", "Where ML adapters run."),
    ("detector", "--detector", "grounding-dino", "Text → boxes from the prompt queries."),
    ("segmenter", "--segmenter", "sam2 | sam2-video", "Boxes → pixel masks."),
    ("inpainter", "--inpainter", "lama | propainter", "How the hole is filled."),
    ("parser", "(fixed)", "llm", "Prompt → detector queries."),
    ("llm", "--llm", "auto | local | cloud", "Where the prompt parser runs."),
    ("package", "--format", "mp4, mov, mkv, …", "Delivery containers after the mezzanine."),
)

# Catalog for `videoclean backends`. `ready` means a factory exists in composition.
BACKENDS: tuple[dict[str, str], ...] = (
    {
        "port": "media",
        "flag": "(fixed)",
        "name": "ffmpeg",
        "status": "ready",
        "device": "no",
        "model": "—",
        "does": "Decode frames, encode mezzanine, mux --format. Always on.",
    },
    {
        "port": "detector",
        "flag": "--detector",
        "name": "grounding-dino",
        "status": "ready (needs local HF cache)",
        "device": "yes",
        "model": DEFAULT_GROUNDING_DINO_MODEL,
        "does": "Open-vocab boxes from the parser queries. Same model for any named thing.",
    },
    {
        "port": "segmenter",
        "flag": "--segmenter",
        "name": "sam2",
        "status": "ready (needs transformers>=4.56 + local HF cache)",
        "device": "yes",
        "model": DEFAULT_SEGMENTER_MODEL,
        "does": "Pixel mask from each track box, per frame (transformers Sam2 or sam2 pkg).",
    },
    {
        "port": "segmenter",
        "flag": "--segmenter",
        "name": "sam2-video",
        "status": "ready (needs sam2 pkg + torch>=2.5 + local HF cache)",
        "device": "yes",
        "model": DEFAULT_SEGMENTER_MODEL,
        "does": "Box on one frame, propagate the silhouette through the clip.",
    },
    {
        "port": "inpainter",
        "flag": "--inpainter",
        "name": "lama",
        "status": "ready (needs simple-lama-inpainting + big-lama.pt)",
        "device": "yes",
        "model": "big-lama.pt (torch hub)",
        "does": "Neural per-frame fill. Works on CPU and CUDA; no temporal consistency.",
    },
    {
        "port": "inpainter",
        "flag": "--inpainter",
        "name": "propainter",
        "status": "ready (needs vendor repo + 3 .pth weights)",
        "device": "yes",
        "model": DEFAULT_INPAINTER_MODEL,
        "does": "Fill using neighboring frames. CUDA recommended.",
    },
    {
        "port": "parser",
        "flag": "(fixed)",
        "name": "llm",
        "status": "ready (needs --llm local or cloud)",
        "device": "no",
        "model": "--llm-model",
        "does": "--prompt → JSON queries for the detector. Local or cloud text model.",
    },
)


def parse_name_list(raw: str | list[str] | None, allowed: tuple[str, ...], *, default: list[str]) -> list[str]:
    if not raw:
        return list(default)
    items = raw if isinstance(raw, list) else [raw]
    names: list[str] = []
    for item in items:
        for part in str(item).split(","):
            name = part.strip().lower()
            if not name:
                continue
            if name not in allowed:
                raise PipelineError(
                    f"unknown adapter {name!r}. known: {', '.join(allowed)}"
                )
            if name not in names:
                names.append(name)
    return names or list(default)


def require_ready(port: str, name: str) -> None:
    ready = READY_ADAPTERS.get(port, frozenset())
    if name not in ready:
        raise AdapterNotImplemented(
            f"{port} {name!r} is a real port, but this adapter is not wired yet. "
            f"Working now: {', '.join(sorted(ready)) or '(none)'}. "
            f"Add an adapter under videoclean/adapters/{port}s/ and register it in composition."
        )


@dataclass
class PipelineConfig:
    """CLI-facing choices. One field per independent axis. No 'profile'."""

    device: str = "cpu"
    detectors: list[str] = field(default_factory=lambda: list(DEFAULT_DETECTORS))
    detector_model: str = DEFAULT_GROUNDING_DINO_MODEL
    detector_threshold: float = 0.15
    segmenter: str = "sam2"
    segmenter_model: str = DEFAULT_SEGMENTER_MODEL
    inpainter: str = "lama"
    inpainter_model: str = DEFAULT_INPAINTER_MODEL
    llm_place: str = "auto"
    llm_model: str = ""
    llm_base_url: str = ""
    llm_api_key: str = ""
    formats: list[str] = field(default_factory=lambda: ["mp4"])
    # Delivery packaging knobs (used when formats include webm / HLS / DASH).
    webm_crf: int = 32
    segment_seconds: int = 6
    allow_download: bool = False
    verify: bool = True
    mask_dilate_px: int = 3
    min_mask_coverage: float = 0.0004
    verify_max_coverage: float = 0.12
    # Built-in picture recipe (fast|balanced|quality|custom); informational after merge.
    profile: str = "custom"
    # What the caller asked for before auto resolved to cpu|cuda|mps.
    device_requested: str = "auto"
    # Empty ceiling means the whole machine. 0 is a validation error, not auto.
    max_vram_mb: int | None = None
    cpu_threads: int | None = None
    inpaint_max_side: int | None = None
    # Second GroundingDINO+SAM pass. Off unless the caller set it. Not a recipe key.
    verify_redetect: bool = False
    preset_id: str | None = None
    # Verify 2.0: residual+detect re-inpaint passes (0 disables even if verify=True leftover path).
    verify_max_passes: int = 1
    # Framewise inpaint threads; 0 = auto (CPU→cores, GPU→1). Video-aware always 1.
    inpaint_workers: int = 0
    # Overlap frames when chunking video-aware / local re-inpaint.
    inpaint_chunk_overlap: int = 8

    # Vision prompt parse: send every Nth frame (0 = text-only). Cap keeps Ollama payloads small.
    prompt_frame_stride: int = 4
    prompt_frame_max: int = 8
    # Scoped parse: split the sampled timeline into chunks of this many frames and parse
    # each chunk separately; targets get frame windows. 0 = one chunk for the whole clip.
    parse_chunk_frames: int = 0
    # Frames per vision-LLM request. Small VLMs (llava-phi3) degrade past 1-2; stronger
    # multi-image models can take more. Raise --vision-batch only for models proven multi-image.
    vision_batch: int = 2
    # Detector tuning. keyframes=None keeps the detector's own default (dino 8).
    detector_keyframes: int | None = None
    detector_nms_iou: float = 0.3
    detector_max_box_area: float = 0.45
    # When True (default), select_tracks may drop where/ordinal if nothing matches.
    select_relax: bool = True
    # Template tracker tuning (boxes between keyframes).
    tracker_min_score: float = 0.55
    tracker_max_template_area: float = 0.12
    # ProPainter knobs, forwarded 1:1 to the vendor model.
    propainter_mask_dilation: int = 4
    propainter_ref_stride: int = 10
    propainter_neighbor_length: int = 10
    propainter_subvideo_length: int = 80
    propainter_raft_iter: int = 20
    # Dir with custom prompt templates (system.md, vision_system.md, bridge_system.md).
    prompt_templates: str | None = None

    def validate(self) -> None:
        if self.device not in DEVICES:
            raise PipelineError(f"--device must be {' | '.join(DEVICES)}, got {self.device!r}")
        if self.prompt_frame_stride < 0:
            raise PipelineError(f"--prompt-frame-stride must be >= 0, got {self.prompt_frame_stride}")
        if self.prompt_frame_max < 1:
            raise PipelineError(f"--prompt-frame-max must be >= 1, got {self.prompt_frame_max}")
        if self.parse_chunk_frames < 0:
            raise PipelineError(f"--parse-chunk-frames must be >= 0, got {self.parse_chunk_frames}")
        if self.vision_batch < 1:
            raise PipelineError(f"--vision-batch must be >= 1, got {self.vision_batch}")
        if self.detector_keyframes is not None and self.detector_keyframes < 1:
            raise PipelineError(f"--detector-keyframes must be >= 1, got {self.detector_keyframes}")
        if not 0.0 < self.detector_nms_iou < 1.0:
            raise PipelineError(f"--detector-nms-iou must be in (0, 1), got {self.detector_nms_iou}")
        if not 0.0 < self.detector_max_box_area <= 1.0:
            raise PipelineError(f"--detector-max-box-area must be in (0, 1], got {self.detector_max_box_area}")
        if not 0.0 < self.tracker_min_score < 1.0:
            raise PipelineError(f"--tracker-min-score must be in (0, 1), got {self.tracker_min_score}")
        if not 0.0 < self.tracker_max_template_area <= 1.0:
            raise PipelineError(
                f"--tracker-max-template-area must be in (0, 1], got {self.tracker_max_template_area}"
            )
        if self.propainter_mask_dilation < 0:
            raise PipelineError(f"--propainter-mask-dilation must be >= 0, got {self.propainter_mask_dilation}")
        if min(
            self.propainter_ref_stride,
            self.propainter_neighbor_length,
            self.propainter_subvideo_length,
            self.propainter_raft_iter,
        ) < 1:
            raise PipelineError("--propainter-ref-stride/neighbor-length/subvideo-length/raft-iter must be >= 1")
        if self.verify_max_passes < 0:
            raise PipelineError(f"--verify-max-passes must be >= 0, got {self.verify_max_passes}")
        if self.inpaint_workers < 0:
            raise PipelineError(f"--inpaint-workers must be >= 0, got {self.inpaint_workers}")
        if self.inpaint_chunk_overlap < 0:
            raise PipelineError(f"--inpaint-chunk-overlap must be >= 0, got {self.inpaint_chunk_overlap}")
        if self.max_vram_mb is not None and not 256 <= int(self.max_vram_mb) <= 262144:
            raise PipelineError(f"--max-vram-mb must be 256…262144, got {self.max_vram_mb}")
        if self.cpu_threads is not None and not 1 <= int(self.cpu_threads) <= 256:
            raise PipelineError(f"--cpu-threads must be 1…256, got {self.cpu_threads}")
        if self.inpaint_max_side is not None:
            side = int(self.inpaint_max_side)
            if side < 64 or side > 8192 or side % 8 != 0:
                raise PipelineError(f"--inpaint-max-side must be 64…8192 and a multiple of 8, got {side}")
        if self.inpaint_workers > 32:
            raise PipelineError(f"--inpaint-workers must be 0…32, got {self.inpaint_workers}")
        prof = (self.profile or "custom").strip().lower() or "custom"
        if prof not in {"custom", "fast", "balanced", "quality"}:
            raise PipelineError(f"--profile must be custom|fast|balanced|quality, got {self.profile!r}")
        self.profile = prof
        if self.prompt_templates is not None:
            root = Path(self.prompt_templates).expanduser()
            if not root.is_dir():
                raise PipelineError(f"--prompt-templates: dir not found: {root}")
        for name in self.detectors:
            if name not in DETECTORS:
                raise PipelineError(f"unknown --detector {name!r}. known: {', '.join(DETECTORS)}")
            require_ready("detector", name)
        if self.segmenter not in SEGMENTERS:
            raise PipelineError(f"unknown --segmenter {self.segmenter!r}. known: {', '.join(SEGMENTERS)}")
        require_ready("segmenter", self.segmenter)
        if self.inpainter not in INPAINTERS:
            raise PipelineError(f"unknown --inpainter {self.inpainter!r}. known: {', '.join(INPAINTERS)}")
        require_ready("inpainter", self.inpainter)
        place = (self.llm_place or "auto").strip().lower()
        if place not in LLM_PLACES:
            raise PipelineError(f"--llm must be {' | '.join(LLM_PLACES)}, got {self.llm_place!r}")
        self.llm_place = place
        self.formats = parse_formats(self.formats)
        if int(self.webm_crf) < 0:
            raise PipelineError(f"--webm-crf must be >= 0, got {self.webm_crf}")
        self.webm_crf = int(self.webm_crf)
        if int(self.segment_seconds) < 1:
            raise PipelineError(f"--segment-seconds must be >= 1, got {self.segment_seconds}")
        self.segment_seconds = int(self.segment_seconds)


@dataclass
class RunCleanupRequest:
    input_path: Path
    output_path: Path
    prompt: str
    config: PipelineConfig
    keep_workdir: bool = False
    overwrite: bool = False
    job_id: str | None = None
    manifest: object | None = None
    targets_override: list[dict] | None = None  # manual targets: skips the prompt parser
    tracks_override: list[dict] | None = None  # manual tracks: skips parse + detect
    masks_override: list[str] | None = None  # mask PNG paths: skips parse + detect + segment
    mask_policy: str = "static"  # static: tile one mask; propagate: anchors → tracks → segment
