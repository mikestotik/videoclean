from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from videoclean.application.config import (
    DEFAULT_DETECTORS,
    DEFAULT_GROUNDING_DINO_MODEL,
    DEFAULT_INPAINTER_MODEL,
    DEFAULT_SEGMENTER_MODEL,
    PORT_HELP,
    PipelineConfig,
    RunCleanupRequest,
)
from videoclean.application.errors import AdapterUnavailable, DownloadCancelled, PipelineError
from videoclean.composition import (
    backends_rows,
    build_catalog,
    build_downloader,
    build_packager,
    build_run_cleanup,
    build_run_preview,
    config_from_flags,
    default_serve_port,
    doctor_sections,
    estimate_seconds,
    make_parser,
    resolve_data_dir,
)
from videoclean.domain.formats import known_format_names
from videoclean.adapters.media.ffmpeg import FFmpegMedia
from videoclean.progress import JobProgress
from videoclean.store import JobIndex

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Remove a named object from video. FFmpeg reads and writes files. See: videoclean backends",
)
jobs_app = typer.Typer(no_args_is_help=True, help="Job history")
models_app = typer.Typer(no_args_is_help=True, help="Model catalog")
app.add_typer(jobs_app, name="jobs")
app.add_typer(models_app, name="models")
console = Console()


def data_dir() -> Path:
    return resolve_data_dir()


def _job_index() -> JobIndex:
    return JobIndex(data_dir() / "jobs.sqlite")


def _die(msg: str, code: int = 1) -> None:
    console.print(msg)
    raise typer.Exit(code)


class _SilentProgress:
    def start(self, *args, **kwargs) -> None: ...

    def tick(self, *args, **kwargs) -> None: ...

    def finish(self, *args, **kwargs) -> None: ...


def _flags(
    device: str,
    detector: str,
    detector_model: str,
    detector_threshold: float,
    segmenter: str,
    segmenter_model: str,
    inpainter: str,
    inpainter_model: str,
    format: list[str] | None,
    download_models: bool,
    require_device: bool = False,
    llm_place: str = "auto",
    llm_model: str = "",
    llm_base_url: str = "",
    verify: bool = True,
    require_runtime: bool = False,
    mask_dilate_px: int = 3,
    telea_radius: int = 9,
    min_mask_coverage: float = 0.0004,
    verify_max_coverage: float = 0.12,
    prompt_frame_stride: int = 4,
    prompt_frame_max: int = 8,
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
) -> PipelineConfig:
    try:
        return config_from_flags(
            device=device,
            detector=detector,
            detector_model=detector_model,
            detector_threshold=detector_threshold,
            segmenter=segmenter,
            segmenter_model=segmenter_model,
            inpainter=inpainter,
            inpainter_model=inpainter_model,
            formats=format,
            allow_download=download_models,
            require_device=require_device,
            llm_place=llm_place,
            llm_model=llm_model,
            llm_base_url=llm_base_url,
            verify=verify,
            require_runtime=require_runtime,
            mask_dilate_px=mask_dilate_px,
            telea_radius=telea_radius,
            min_mask_coverage=min_mask_coverage,
            verify_max_coverage=verify_max_coverage,
            prompt_frame_stride=prompt_frame_stride,
            prompt_frame_max=prompt_frame_max,
            vision_batch=vision_batch,
            detector_keyframes=detector_keyframes,
            detector_nms_iou=detector_nms_iou,
            detector_max_box_area=detector_max_box_area,
            tracker_min_score=tracker_min_score,
            tracker_max_template_area=tracker_max_template_area,
            propainter_mask_dilation=propainter_mask_dilation,
            propainter_ref_stride=propainter_ref_stride,
            propainter_neighbor_length=propainter_neighbor_length,
            propainter_subvideo_length=propainter_subvideo_length,
            propainter_raft_iter=propainter_raft_iter,
            prompt_templates=prompt_templates,
        )
    except PipelineError as exc:
        raise typer.Exit(str(exc)) from exc


def _prompt_frame_stride_opt() -> int:
    return typer.Option(
        4,
        "--prompt-frame-stride",
        help="Vision parse: take frame 0, then every Nth (4 → 0,4,8,…). 0 = text-only, no frames.",
    )


def _prompt_frame_max_opt() -> int:
    return typer.Option(
        8,
        "--prompt-frame-max",
        help="Max frames sent to the vision LLM (after stride). Keeps Ollama payloads small.",
    )


def _device_opt() -> str:
    return typer.Option(
        "cpu",
        "--device",
        help="cpu | cuda | mps. Where Grounding DINO / SAM2 / ProPainter run. TELEA stays CPU.",
    )


def _detector_opt() -> str:
    return typer.Option(
        ",".join(DEFAULT_DETECTORS),
        "--detector",
        help="grounding-dino. Searches the parser queries. Chain with a comma.",
    )


def _detector_model_opt() -> str:
    return typer.Option(
        DEFAULT_GROUNDING_DINO_MODEL,
        "--detector-model",
        help="HF id for grounding-dino. Empty/default follows --detector.",
    )


def _segmenter_opt() -> str:
    return typer.Option(
        "sam2",
        "--segmenter",
        help="sam2 (mask per frame) | sam2-video (propagate through the clip).",
    )


def _inpainter_opt() -> str:
    return typer.Option(
        "opencv-telea",
        "--inpainter",
        help="opencv-telea (CPU) | lama (neural, CPU/CUDA) | propainter (video-aware, CUDA).",
    )


def _llm_place_opt() -> str:
    return typer.Option(
        "auto",
        "--llm",
        help="auto | local | cloud. Where the prompt parser LLM runs.",
    )


def _llm_model_opt() -> str:
    return typer.Option(
        "",
        "--llm-model",
        help="Ollama tag (llama3.2 / llava-phi3), cloud id, or .gguf path. Env: VIDEOCLEAN_LLM_MODEL.",
    )


def _llm_url_opt() -> str:
    return typer.Option(
        "",
        "--llm-base-url",
        help="OpenAI-compatible /v1 root. Local default http://127.0.0.1:11434/v1. Cloud: api.x.ai or api.openai.com.",
    )


def _verify_opt() -> bool:
    return typer.Option(
        True,
        "--verify/--no-verify",
        help="After inpaint, run the detector again. If the object is still there, expand the mask and fill once more.",
    )


@app.command()
def backends() -> None:
    """What each port is, which adapters exist, which are wired."""
    console.print("[bold]Ports[/bold]  — independent. Not a single profile.\n")
    ports = Table()
    ports.add_column("port")
    ports.add_column("flag")
    ports.add_column("values")
    ports.add_column("role")
    for port, flag, values, role in PORT_HELP:
        ports.add_row(port, flag, values, role)
    console.print(ports)
    console.print()
    table = Table(title="adapters")
    table.add_column("port")
    table.add_column("name")
    table.add_column("status")
    table.add_column("uses --device")
    table.add_column("model flag")
    table.add_column("does")
    for row in backends_rows():
        table.add_row(row["port"], row["name"], row["status"], row["device"], row["model"], row["does"])
    console.print(table)
    console.print(
        "\nSwap later by adding a class under videoclean/adapters/<port>/ "
        "and a factory in composition.py. The use case does not import adapters."
    )


@app.command()
def doctor(
    device: str = _device_opt(),
    detector: str = _detector_opt(),
    detector_model: str = _detector_model_opt(),
    detector_threshold: float = typer.Option(0.15, "--detector-threshold"),
    segmenter: str = _segmenter_opt(),
    segmenter_model: str = typer.Option(DEFAULT_SEGMENTER_MODEL, "--segmenter-model"),
    inpainter: str = _inpainter_opt(),
    inpainter_model: str = typer.Option(DEFAULT_INPAINTER_MODEL, "--inpainter-model"),
    llm: str = _llm_place_opt(),
    llm_model: str = _llm_model_opt(),
    llm_base_url: str = _llm_url_opt(),
    format: list[str] | None = typer.Option(None, "--format", "-f"),
    download_models: bool = typer.Option(False, "--download-models"),
    verify: bool = _verify_opt(),
    prompt_frame_stride: int = _prompt_frame_stride_opt(),
    prompt_frame_max: int = _prompt_frame_max_opt(),
) -> None:
    """Machine facts + the adapters this exact command would use."""
    cfg = _flags(
        device,
        detector,
        detector_model,
        detector_threshold,
        segmenter,
        segmenter_model,
        inpainter,
        inpainter_model,
        format,
        download_models,
        llm_place=llm,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        verify=verify,
        prompt_frame_stride=prompt_frame_stride,
        prompt_frame_max=prompt_frame_max,
    )
    for title, lines in doctor_sections(cfg):
        console.print(f"[bold]{title}[/bold]")
        for line in lines:
            console.print(f"  {line}")
        console.print()


@app.command()
def inspect(
    input: Path = typer.Option(..., "--input", exists=True, readable=True),
    prompt: str = typer.Option(..., "--prompt", help="What to remove from the video."),
    device: str = _device_opt(),
    detector: str = _detector_opt(),
    detector_model: str = _detector_model_opt(),
    detector_threshold: float = typer.Option(0.15, "--detector-threshold"),
    segmenter: str = _segmenter_opt(),
    segmenter_model: str = typer.Option(DEFAULT_SEGMENTER_MODEL, "--segmenter-model"),
    inpainter: str = _inpainter_opt(),
    inpainter_model: str = typer.Option(DEFAULT_INPAINTER_MODEL, "--inpainter-model"),
    llm: str = _llm_place_opt(),
    llm_model: str = _llm_model_opt(),
    llm_base_url: str = _llm_url_opt(),
    format: list[str] | None = typer.Option(None, "--format", "-f"),
    download_models: bool = typer.Option(False, "--download-models"),
    verify: bool = _verify_opt(),
    prompt_frame_stride: int = _prompt_frame_stride_opt(),
    prompt_frame_max: int = _prompt_frame_max_opt(),
) -> None:
    """Probe the file and print the plan. Does not write an output video."""
    import tempfile

    from videoclean.adapters.images import read_bgr
    from videoclean.adapters.prompt.frames import sample_frame_indices

    cfg = _flags(
        device,
        detector,
        detector_model,
        detector_threshold,
        segmenter,
        segmenter_model,
        inpainter,
        inpainter_model,
        format,
        download_models,
        llm_place=llm,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        verify=verify,
        prompt_frame_stride=prompt_frame_stride,
        prompt_frame_max=prompt_frame_max,
    )
    media = FFmpegMedia()
    manifest = media.probe(input)
    sample_frames = None
    sample_idxs: list[int] = []
    if cfg.prompt_frame_stride > 0:
        with tempfile.TemporaryDirectory(prefix="videoclean-inspect-") as tmp:
            frames_dir = Path(tmp) / "frames"
            log = Path(tmp) / "ffmpeg.log"
            frame_paths = media.extract_frames(input, frames_dir, log)
            sample_idxs = sample_frame_indices(
                len(frame_paths), cfg.prompt_frame_stride, cfg.prompt_frame_max
            )
            sample_frames = []
            for i in sample_idxs:
                img = read_bgr(frame_paths[i])
                if img is not None:
                    sample_frames.append(img)
    try:
        intent = make_parser(cfg).parse(prompt, frames=sample_frames)
    except (PipelineError, AdapterUnavailable) as exc:
        raise typer.Exit(str(exc)) from exc
    eta = estimate_seconds(manifest.frame_count, manifest.width, manifest.height)
    console.print(f"[bold]Input[/bold]  {input}")
    console.print(f"  {manifest.video_codec}  {manifest.width}x{manifest.height}  {manifest.fps:.3f} fps")
    console.print(f"  {manifest.duration_s:.3f}s  {manifest.frame_count} frames  audio={manifest.audio_codec or 'none'}")
    console.print()
    console.print(f"[bold]Prompt[/bold]  parse={intent.parse_mode}  vision_frames={len(sample_idxs) or 0}")
    for t in intent.targets:
        console.print(
            f"  kind={t.kind}  motion={t.motion}  where={t.where or 'any'}  "
            f"ordinal={t.ordinal or '—'}  from={t.from_side or '—'}  "
            f"part={t.part or 'whole'}  query={t.query!r}"
        )
    console.print()
    console.print("[bold]Pipeline for this command[/bold]")
    console.print("  media        ffmpeg  (always)")
    console.print(f"  device       {cfg.device}")
    console.print(f"  detector     {', '.join(cfg.detectors)}   model={cfg.detector_model}")
    console.print(f"  segmenter    {cfg.segmenter}")
    console.print(f"  inpainter    {cfg.inpainter}")
    console.print(f"  parser       llm  place={cfg.llm_place}  model={cfg.llm_model or '—'}")
    console.print(
        f"  vision       stride={cfg.prompt_frame_stride}  max={cfg.prompt_frame_max}  "
        f"idxs={sample_idxs or '—'}"
    )
    console.print(f"  verify       {cfg.verify}")
    console.print(f"  format       {', '.join(cfg.formats)}")
    console.print(f"  eta (CPU, rough)  ~{int(eta)}s")
    console.print(f"  known formats     {', '.join(known_format_names())}")


@app.command()
def preview(
    input: Path = typer.Option(..., "--input", exists=True, readable=True),
    output: Path = typer.Option(..., "--output", help="Directory for preview artifacts"),
    prompt: str = typer.Option("", "--prompt"),
    queries: str = typer.Option(
        "",
        "--queries",
        help='Manual targets instead of LLM parse: "text [bottom], logo, red mug". Skips the LLM.',
    ),
    frames: str = typer.Option(
        "0:16:1",
        "--frames",
        help="Frame selection: start:count:stride or a comma list of indices (3,7,15).",
    ),
    device: str = _device_opt(),
    detector: str = _detector_opt(),
    detector_model: str = _detector_model_opt(),
    detector_threshold: float = typer.Option(0.15, "--detector-threshold"),
    segmenter: str = _segmenter_opt(),
    segmenter_model: str = typer.Option(DEFAULT_SEGMENTER_MODEL, "--segmenter-model"),
    mask_dilate: int = typer.Option(3, "--mask-dilate"),
    llm: str = _llm_place_opt(),
    llm_model: str = _llm_model_opt(),
    llm_base_url: str = _llm_url_opt(),
    prompt_frame_stride: int = _prompt_frame_stride_opt(),
    prompt_frame_max: int = _prompt_frame_max_opt(),
    parse_chunk_frames: int = typer.Option(
        0,
        "--parse-chunk-frames",
        help="Scoped parse: split the timeline into chunks of N frames; each chunk gets its own targets (frame windows). 0 = one list for the whole clip.",
        min=0,
    ),
    vision_batch: int = typer.Option(2, "--vision-batch"),
    detector_keyframes: int | None = typer.Option(None, "--detector-keyframes", min=1),
    detector_nms_iou: float = typer.Option(0.3, "--detector-nms-iou"),
    detector_max_box_area: float = typer.Option(0.25, "--detector-max-box-area"),
    tracker_min_score: float = typer.Option(0.55, "--tracker-min-score"),
    tracker_max_template_area: float = typer.Option(0.12, "--tracker-max-template-area"),
    prompt_templates: str | None = typer.Option(None, "--prompt-templates", exists=True, file_okay=False),
    download_models: bool = typer.Option(False, "--download-models"),
) -> None:
    """Detect + masks on a frame subset. No inpaint, no encode. For tuning parameters."""
    cfg = _flags(
        device,
        detector,
        detector_model,
        detector_threshold,
        segmenter,
        segmenter_model,
        "opencv-telea",
        "",
        None,
        download_models,
        require_device=True,
        llm_place=llm,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        mask_dilate_px=mask_dilate,
        prompt_frame_stride=prompt_frame_stride,
        prompt_frame_max=prompt_frame_max,
        vision_batch=vision_batch,
        detector_keyframes=detector_keyframes,
        detector_nms_iou=detector_nms_iou,
        detector_max_box_area=detector_max_box_area,
        tracker_min_score=tracker_min_score,
        tracker_max_template_area=tracker_max_template_area,
        prompt_templates=prompt_templates,
    )
    from videoclean.application.use_cases.run_preview import PreviewRequest, parse_queries_arg

    if queries.strip():
        if prompt.strip():
            _die("--queries and --prompt are mutually exclusive")
        targets = [t for t in parse_queries_arg(queries)]
        mode, target_dicts = "detect", [
            {"kind": t.kind, "query": t.query, "where": t.where, "motion": t.motion}
            for t in targets
        ]
    else:
        if not prompt.strip():
            _die("--prompt is required (or pass --queries)")
        mode, target_dicts = "parse", None

    parts = frames.split(":")
    if len(parts) == 3:
        start, count, stride = (int(p) if p.strip() else None for p in parts)
        indices = None
    else:
        start = count = stride = None
        indices = [int(p) for p in frames.split(",") if p.strip()]

    media = FFmpegMedia()
    manifest = media.probe(input)
    from videoclean.store import new_job_id

    job_id = new_job_id()
    req = PreviewRequest(
        input_path=input.expanduser().resolve(),
        prompt=prompt,
        config=cfg,
        start=start,
        count=count,
        stride=stride,
        indices=indices,
        mode=mode,
        targets=target_dicts,
        job_id=job_id,
    )
    output.expanduser().resolve().mkdir(parents=True, exist_ok=True)
    try:
        uc = build_run_preview(cfg, _SilentProgress(), JobIndex(data_dir() / "jobs.sqlite"), job_id=job_id)
        report = uc.execute(req, output.expanduser().resolve())
    except (PipelineError, AdapterUnavailable, FileNotFoundError) as exc:
        raise typer.Exit(str(exc)) from exc
    console.print(f"[green]{report['state']}[/green]  artifacts: {report.get('workdir')}")


@app.command()
def run(
    input: Path = typer.Option(..., "--input", exists=True, readable=True),
    output: Path = typer.Option(..., "--output"),
    prompt: str = typer.Option(..., "--prompt", help="What to remove from the video."),
    format: list[str] | None = typer.Option(
        None,
        "--format",
        "-f",
        help="Repeat or comma-list. Known: mp4, mov, mkv, webm, hls-fmp4, hls-ts, dash",
    ),
    device: str = _device_opt(),
    detector: str = _detector_opt(),
    detector_model: str = _detector_model_opt(),
    detector_threshold: float = typer.Option(
        0.15,
        "--detector-threshold",
        help="Box score cutoff. Applied as-is to grounding-dino.",
    ),
    segmenter: str = _segmenter_opt(),
    segmenter_model: str = typer.Option(DEFAULT_SEGMENTER_MODEL, "--segmenter-model"),
    inpainter: str = _inpainter_opt(),
    inpainter_model: str = typer.Option(DEFAULT_INPAINTER_MODEL, "--inpainter-model"),
    mask_dilate: int = typer.Option(3, "--mask-dilate", help="Pixels to dilate SAM masks after segmentation."),
    telea_radius: int = typer.Option(9, "--telea-radius", help="OpenCV TELEA radius."),
    min_mask_coverage: float = typer.Option(
        0.0004,
        "--min-mask-coverage",
        help="Fail if mean mask area fraction is below this.",
    ),
    verify_max_coverage: float = typer.Option(
        0.12,
        "--verify-max-coverage",
        help="Skip verify re-inpaint if leftover coverage is above this (or 1.5× original).",
    ),
    llm: str = _llm_place_opt(),
    llm_model: str = _llm_model_opt(),
    llm_base_url: str = _llm_url_opt(),
    prompt_frame_stride: int = _prompt_frame_stride_opt(),
    prompt_frame_max: int = _prompt_frame_max_opt(),
    parse_chunk_frames: int = typer.Option(
        0,
        "--parse-chunk-frames",
        help="Scoped parse: split the timeline into chunks of N frames; each chunk gets its own targets (frame windows). 0 = one list for the whole clip.",
        min=0,
    ),
    vision_batch: int = typer.Option(
        2,
        "--vision-batch",
        help="Frames per vision-LLM request. llava-phi3: keep 2. Strong multi-image VLMs: 4-8.",
    ),
    detector_keyframes: int | None = typer.Option(
        None,
        "--detector-keyframes",
        help="Frames the detector actually runs on (default: grounding-dino 8). More = slower, better recall.",
        min=1,
    ),
    detector_nms_iou: float = typer.Option(
        0.3,
        "--detector-nms-iou",
        help="Merge duplicate boxes overlapping more than this IoU.",
        min=0.01,
        max=0.99,
    ),
    detector_max_box_area: float = typer.Option(
        0.25,
        "--detector-max-box-area",
        help="Drop detections covering more than this fraction of the frame (0-1].",
        min=0.01,
        max=1.0,
    ),
    tracker_min_score: float = typer.Option(
        0.55,
        "--tracker-min-score",
        help="Template-match score to accept a box between keyframes. Lower = more aggressive fill.",
        min=0.01,
        max=0.99,
    ),
    tracker_max_template_area: float = typer.Option(
        0.12,
        "--tracker-max-template-area",
        help="Skip template tracking for boxes above this frame fraction (background dominates).",
        min=0.01,
        max=1.0,
    ),
    propainter_mask_dilation: int = typer.Option(
        4,
        "--propainter-mask-dilation",
        help="ProPainter: grow masks by N px before filling.",
        min=0,
    ),
    propainter_ref_stride: int = typer.Option(
        10,
        "--propainter-ref-stride",
        help="ProPainter: spacing of global reference frames.",
        min=1,
    ),
    propainter_neighbor_length: int = typer.Option(
        10,
        "--propainter-neighbor-length",
        help="ProPainter: temporal window around each keyframe.",
        min=1,
    ),
    propainter_subvideo_length: int = typer.Option(
        80,
        "--propainter-subvideo-length",
        help="ProPainter: chunk length for flow completion / propagation.",
        min=1,
    ),
    propainter_raft_iter: int = typer.Option(
        20,
        "--propainter-raft-iter",
        help="ProPainter: RAFT flow iterations. More = slower, sharper flow.",
        min=1,
    ),
    prompt_templates: str | None = typer.Option(
        None,
        "--prompt-templates",
        help="Dir with custom prompt templates: system.md, vision_system.md, bridge_system.md.",
        exists=True,
        file_okay=False,
    ),
    download_models: bool = typer.Option(
        False,
        "--download-models",
        help="Allow Hugging Face download. Off by default. Prefer: huggingface-cli download <id>",
    ),
    keep_workdir: bool = typer.Option(False, "--keep-workdir"),
    overwrite: bool = typer.Option(False, "--overwrite"),
    verify: bool = _verify_opt(),
) -> None:
    """Run the cleanup pipeline with live progress."""
    cfg = _flags(
        device,
        detector,
        detector_model,
        detector_threshold,
        segmenter,
        segmenter_model,
        inpainter,
        inpainter_model,
        format,
        download_models,
        require_device=True,
        require_runtime=True,
        llm_place=llm,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        verify=verify,
        mask_dilate_px=mask_dilate,
        telea_radius=telea_radius,
        min_mask_coverage=min_mask_coverage,
        verify_max_coverage=verify_max_coverage,
        prompt_frame_stride=prompt_frame_stride,
        prompt_frame_max=prompt_frame_max,
        parse_chunk_frames=parse_chunk_frames,
        vision_batch=vision_batch,
        detector_keyframes=detector_keyframes,
        detector_nms_iou=detector_nms_iou,
        detector_max_box_area=detector_max_box_area,
        tracker_min_score=tracker_min_score,
        tracker_max_template_area=tracker_max_template_area,
        propainter_mask_dilation=propainter_mask_dilation,
        propainter_ref_stride=propainter_ref_stride,
        propainter_neighbor_length=propainter_neighbor_length,
        propainter_subvideo_length=propainter_subvideo_length,
        propainter_raft_iter=propainter_raft_iter,
        prompt_templates=prompt_templates,
    )
    media = FFmpegMedia()
    manifest = media.probe(input)
    headline = (
        f"{input.name}  •  {manifest.width}×{manifest.height}  •  {manifest.duration_s:.1f}s  •  "
        f"{manifest.frame_count} frames  •  device {cfg.device}  •  "
        f"det={','.join(cfg.detectors)}  seg={cfg.segmenter}  inpaint={cfg.inpainter}"
    )
    eta0 = estimate_seconds(manifest.frame_count, manifest.width, manifest.height, cfg.device)
    from videoclean.store import new_job_id

    job_id = new_job_id()
    req = RunCleanupRequest(
        input_path=input.expanduser().resolve(),
        output_path=output.expanduser().resolve(),
        prompt=prompt,
        config=cfg,
        keep_workdir=keep_workdir,
        overwrite=overwrite,
        job_id=job_id,
        manifest=manifest,
    )
    try:
        with JobProgress(job_id, headline, eta0) as progress:
            uc = build_run_cleanup(cfg, progress, JobIndex(data_dir() / "jobs.sqlite"), job_id=job_id)
            report = uc.execute(req, data_dir())
    except (PipelineError, AdapterUnavailable, FileExistsError, FileNotFoundError) as exc:
        raise typer.Exit(str(exc)) from exc

    console.print()
    for fmt_name, path in (report.get("outputs") or {}).items():
        console.print(f"[green]{fmt_name}[/green]  {path}")
    console.print(f"report  {report['workdir']}/output/report.json")


@app.command("package")
def package_cmd(
    input: Path = typer.Option(..., "--input", exists=True, readable=True, help="Mezzanine / cleaned file"),
    output: Path | None = typer.Option(None, "--output", help="Output path stem"),
    format: list[str] | None = typer.Option(None, "--format", "-f"),
    overwrite: bool = typer.Option(False, "--overwrite"),
) -> None:
    """Repackage an existing cleaned file. FFmpeg only; no detector."""
    log = (output.expanduser().resolve().parent if output else input.expanduser().resolve().parent) / ".videoclean-package.log"
    try:
        artifacts = build_packager().execute(input, output, format, overwrite, log)
    except (PipelineError, FileExistsError) as exc:
        raise typer.Exit(str(exc)) from exc
    for fmt, path in artifacts.items():
        console.print(f"[green]{fmt}[/green]  {path}")


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host"),
    port: int | None = typer.Option(None, "--port", help="Default VIDEOCLEAN_PORT or 7860"),
) -> None:
    """Launch the web UI (FastAPI) and background job worker."""
    try:
        from server.fastapi_app import launch_from_env
    except ImportError:
        _die("Web UI is not installed. Install with: uv sync --extra web")
    port_i = port if port is not None else default_serve_port()
    try:
        launch_from_env(host=host, port=port_i, data_dir=data_dir())
    except RuntimeError as exc:
        _die(str(exc))
    except OSError as exc:
        msg = str(exc)
        if "empty port" in msg.lower() or "address already in use" in msg.lower():
            _die(
                f"port {port_i} is already in use (another videoclean serve?). "
                f"Stop it, or pick a free port: videoclean serve --port 7861\n"
                f"On macOS: lsof -nP -iTCP:{port_i} -sTCP:LISTEN"
            )
        _die(msg)


@models_app.command("list")
def models_list() -> None:
    """Show local catalog status (no network)."""
    catalog = build_catalog(jobs=_job_index())
    table = Table(title="models")
    table.add_column("id", overflow="fold", no_wrap=True)
    table.add_column("title")
    table.add_column("status")
    table.add_column("size")
    table.add_column("message", overflow="fold")
    for status in catalog.list_status():
        info = status.info
        table.add_row(info.id, info.title, status.state, info.size_hint, status.message)
    console.print(table)


@models_app.command("download")
def models_download(component_id: str) -> None:
    """Download one catalog component (HF / git / Ollama / LaMa)."""
    jobs = _job_index()

    def on_progress(fraction: float, message: str = "") -> None:
        pct = int(max(0.0, min(float(fraction), 1.0)) * 100)
        console.print(f"{pct:3d}%  {message}".rstrip())

    try:
        build_downloader().execute(component_id, jobs, on_progress)
    except (PipelineError, DownloadCancelled) as exc:
        _die(str(exc))
    console.print(f"[green]ready[/green]  {component_id}")


@jobs_app.command("list")
def jobs_list() -> None:
    index = _job_index()
    rows = index.list_jobs()
    if not rows:
        console.print("No jobs yet.")
        return
    table = Table(title="jobs")
    table.add_column("id")
    table.add_column("state")
    table.add_column("created")
    table.add_column("input")
    for row in rows:
        table.add_row(row["id"], row["state"], row["created_at"], Path(row["input_path"] or "").name)
    console.print(table)


@jobs_app.command("show")
def jobs_show(job_id: str) -> None:
    index = _job_index()
    row = index.get(job_id)
    if not row:
        _die(f"unknown job {job_id}")
    console.print_json(row["report_json"] or "{}")


@jobs_app.command("cancel")
def jobs_cancel(job_id: str) -> None:
    from videoclean.application.use_cases.manage_jobs import ManageJobs

    index = _job_index()
    row = index.get(job_id)
    if row is None:
        _die(f"unknown job {job_id}")
    if row["state"] not in {"QUEUED", "RUNNING"}:
        _die(f"job {job_id} is {row['state']}; nothing to cancel")
    ManageJobs(index).cancel(job_id)
    after = index.get(job_id)
    if after is not None and after["state"] == "RUNNING":
        console.print(f"Cancel requested for {job_id}; the worker stops at the next progress tick.")
        return
    console.print(f"Job {job_id} → {after['state'] if after else 'CANCELLED'}.")


@jobs_app.command("retry")
def jobs_retry(job_id: str) -> None:
    from videoclean.application.use_cases.manage_jobs import ManageJobs

    index = _job_index()
    try:
        new_id = ManageJobs(index).retry(job_id)
    except PipelineError as exc:
        _die(str(exc))
    console.print(f"Queued retry {new_id} (from {job_id}).")


@jobs_app.command("delete")
def jobs_delete(job_id: str) -> None:
    from videoclean.application.use_cases.manage_jobs import ManageJobs

    index = _job_index()
    if index.get(job_id) is None:
        _die(f"unknown job {job_id}")
    ManageJobs(index).delete(job_id, data_dir())
    console.print(f"Deleted {job_id}.")
