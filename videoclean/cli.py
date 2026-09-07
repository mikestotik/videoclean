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
    RunCleanupRequest,
)
from videoclean.application.errors import AdapterUnavailable, PipelineError
from videoclean.composition import (
    backends_rows,
    build_packager,
    build_run_cleanup,
    config_from_flags,
    doctor_sections,
    estimate_seconds,
    make_parser,
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
app.add_typer(jobs_app, name="jobs")
console = Console()


def data_dir() -> Path:
    return Path.home() / ".videoclean"


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
        help="cpu | cuda | mps. Where Grounding DINO / OWL-ViT / SAM2 / ProPainter run. TELEA stays CPU.",
    )


def _detector_opt() -> str:
    return typer.Option(
        ",".join(DEFAULT_DETECTORS),
        "--detector",
        help="grounding-dino | owlvit. Searches the parser queries. Chain with a comma.",
    )


def _detector_model_opt() -> str:
    return typer.Option(
        DEFAULT_GROUNDING_DINO_MODEL,
        "--detector-model",
        help="HF id for grounding-dino or owlvit. Empty/default follows --detector.",
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
        help="Box score cutoff. Applied as-is to grounding-dino and owlvit.",
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


@jobs_app.command("list")
def jobs_list() -> None:
    index = JobIndex(data_dir() / "jobs.sqlite")
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
    index = JobIndex(data_dir() / "jobs.sqlite")
    row = index.get(job_id)
    if not row:
        raise typer.Exit(f"unknown job {job_id}")
    console.print_json(row["report_json"] or "{}")
