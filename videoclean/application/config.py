from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from videoclean.application.errors import AdapterNotImplemented, PipelineError
from videoclean.domain.formats import parse_formats

DEVICES = ("cpu", "cuda", "mps")

# Names the product understands. Wiring (or "not implemented") lives in composition.
DETECTORS = ("grounding-dino", "owlvit")
SEGMENTERS = ("sam2", "sam2-video")
INPAINTERS = ("opencv-telea", "lama", "propainter")
LLM_PLACES = ("auto", "local", "cloud")

READY_ADAPTERS = {
    "detector": frozenset({"grounding-dino", "owlvit"}),
    "segmenter": frozenset({"sam2", "sam2-video"}),
    "inpainter": frozenset({"opencv-telea", "lama", "propainter"}),
}

DEFAULT_DETECTOR_MODEL = "google/owlvit-base-patch32"
DEFAULT_GROUNDING_DINO_MODEL = "IDEA-Research/grounding-dino-tiny"
DEFAULT_SEGMENTER_MODEL = "facebook/sam2-hiera-tiny"
DEFAULT_INPAINTER_MODEL = "camenduru/ProPainter"
DEFAULT_DETECTORS = ["grounding-dino"]

# What each flag actually controls. Doctor / backends print this.
PORT_HELP = (
    ("media", "(fixed)", "ffmpeg", "Read and write video. Not a switch."),
    ("device", "--device", "cpu | cuda | mps", "Where ML adapters run. OpenCV TELEA stays on CPU."),
    ("detector", "--detector", "grounding-dino | owlvit", "Text → boxes from the prompt queries."),
    ("segmenter", "--segmenter", "sam2 | sam2-video", "Boxes → pixel masks."),
    ("inpainter", "--inpainter", "opencv-telea | lama | propainter", "How the hole is filled."),
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
        "port": "detector",
        "flag": "--detector",
        "name": "owlvit",
        "status": "ready (needs local HF cache)",
        "device": "yes",
        "model": DEFAULT_DETECTOR_MODEL,
        "does": "Open-vocab boxes from short queries, then template-track.",
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
        "name": "opencv-telea",
        "status": "ready",
        "device": "CPU only",
        "model": "—",
        "does": "Fill the hole one frame at a time.",
    },
    {
        "port": "inpainter",
        "flag": "--inpainter",
        "name": "lama",
        "status": "ready (needs simple-lama-inpainting + big-lama.pt)",
        "device": "yes",
        "model": "big-lama.pt (torch hub)",
        "does": "Neural per-frame fill. Better than TELEA; no temporal consistency.",
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
    inpainter: str = "opencv-telea"
    inpainter_model: str = DEFAULT_INPAINTER_MODEL
    llm_place: str = "auto"
    llm_model: str = ""
    llm_base_url: str = ""
    llm_api_key: str = ""
    formats: list[str] = field(default_factory=lambda: ["mp4"])
    allow_download: bool = False
    verify: bool = True
    mask_dilate_px: int = 3
    telea_radius: int = 9
    min_mask_coverage: float = 0.0004
    verify_max_coverage: float = 0.12
    # Vision prompt parse: send every Nth frame (0 = text-only). Cap keeps Ollama payloads small.
    prompt_frame_stride: int = 4
    prompt_frame_max: int = 8

    def validate(self) -> None:
        if self.device not in DEVICES:
            raise PipelineError(f"--device must be {' | '.join(DEVICES)}, got {self.device!r}")
        if self.prompt_frame_stride < 0:
            raise PipelineError(f"--prompt-frame-stride must be >= 0, got {self.prompt_frame_stride}")
        if self.prompt_frame_max < 1:
            raise PipelineError(f"--prompt-frame-max must be >= 1, got {self.prompt_frame_max}")
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
