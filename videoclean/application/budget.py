"""Resource ceiling. An empty ceiling is the whole machine."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

from videoclean.application.errors import PipelineError

RESERVE_BYTES = 1536 * 1024 * 1024
_MIN_VRAM_BYTES = 512 * 1024 * 1024


class MachineProbe(Protocol):
    def cuda_mem(self) -> tuple[int, int] | None:
        """(free_bytes, total_bytes) or None when CUDA is not the device."""

    def unified_total(self) -> int:
        """Host RAM bytes used as the MPS estimate."""

    def cpu_count(self) -> int: ...

    def nvenc(self) -> bool: ...


@dataclass(frozen=True)
class Budget:
    device_requested: str
    device: str
    vram_total_bytes: int
    vram_free_bytes: int
    vram_ceiling_bytes: int | None
    vram_budget_bytes: int
    cpu_count: int
    cpu_threads: int
    inpaint_workers_cap: int | None
    inpaint_max_side: int | None
    nvenc: bool
    vram_source: str = "cuda"


def pick_device(requested: str | None) -> str:
    """Resolve auto/empty to cuda, else mps, else cpu. Explicit names pass through."""
    key = (requested or "").strip().lower()
    if key in {"cpu", "cuda", "mps"}:
        return key
    if key not in {"", "auto"}:
        raise PipelineError(f"--device must be auto | cpu | cuda | mps, got {requested!r}")
    try:
        import torch
    except Exception:  # noqa: BLE001
        return "cpu"
    try:
        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # noqa: BLE001
        pass
    try:
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


def resolve_budget(cfg, *, probe: MachineProbe) -> Budget:
    """Empty ceilings mean the whole machine. Never invent a smaller default."""
    requested = str(getattr(cfg, "device_requested", None) or cfg.device or "auto")
    device = str(cfg.device or "cpu").strip().lower()
    if device not in {"cpu", "cuda", "mps"}:
        device = pick_device(device)

    ceiling_mb = getattr(cfg, "max_vram_mb", None)
    ceiling_bytes = None if ceiling_mb in (None, "") else int(ceiling_mb) * 1024 * 1024
    threads_field = getattr(cfg, "cpu_threads", None)
    side_field = getattr(cfg, "inpaint_max_side", None)
    workers = int(getattr(cfg, "inpaint_workers", 0) or 0)
    cpu_n = max(1, int(probe.cpu_count() or 1))

    if device == "cuda":
        mem = probe.cuda_mem()
        free, total = (0, 0) if mem is None else mem
        budget = int(free) - RESERVE_BYTES
        if ceiling_bytes is not None:
            budget = min(budget, ceiling_bytes)
        if budget < _MIN_VRAM_BYTES:
            raise PipelineError(
                f"VRAM budget {budget} bytes is below 512 MiB after the {RESERVE_BYTES} byte reserve "
                f"(free {free}). Raise the card or lower other tenants; a silent tiny batch is not used."
            )
        source = "cuda"
        vram_budget = budget
        vram_free, vram_total = int(free), int(total)
    elif device == "mps":
        total = int(probe.unified_total() or 0)
        budget = int(total * 0.60)
        if ceiling_bytes is not None:
            budget = min(budget, ceiling_bytes)
        if budget < _MIN_VRAM_BYTES:
            raise PipelineError(
                f"unified memory budget {budget} bytes is below 512 MiB (total {total})."
            )
        source = "unified-estimate"
        vram_budget = budget
        vram_free, vram_total = budget, total
    else:
        source = "cpu"
        vram_budget = 0
        vram_free, vram_total = 0, 0

    if threads_field in (None, ""):
        cpu_threads = cpu_n
    else:
        cpu_threads = int(threads_field)
        if cpu_threads < 1:
            raise PipelineError(f"cpu_threads must be >= 1, got {cpu_threads}")

    return Budget(
        device_requested=requested,
        device=device,
        vram_total_bytes=vram_total,
        vram_free_bytes=vram_free,
        vram_ceiling_bytes=ceiling_bytes,
        vram_budget_bytes=vram_budget,
        cpu_count=cpu_n,
        cpu_threads=cpu_threads,
        inpaint_workers_cap=None if workers <= 0 else workers,
        inpaint_max_side=None if side_field in (None, "") else int(side_field),
        nvenc=bool(probe.nvenc()),
        vram_source=source,
    )


class TorchProbe:
    """Live probe. Tests pass a fake MachineProbe instead."""

    def cuda_mem(self) -> tuple[int, int] | None:
        try:
            import torch

            if not torch.cuda.is_available():
                return None
            free, total = torch.cuda.mem_get_info()
            return int(free), int(total)
        except Exception:  # noqa: BLE001
            return None

    def unified_total(self) -> int:
        import psutil

        return int(psutil.virtual_memory().total)

    def cpu_count(self) -> int:
        return os.cpu_count() or 1

    def nvenc(self) -> bool:
        from videoclean.adapters.media.ffmpeg import nvenc_available

        return nvenc_available()


def apply_thread_caps(cpu_threads: int) -> tuple[int, int]:
    """Set torch and cv2 thread counts. Returns the previous pair for finally."""
    import cv2

    prev_cv = int(cv2.getNumThreads())
    prev_torch = 0
    try:
        import torch

        prev_torch = int(torch.get_num_threads())
        torch.set_num_threads(max(1, int(cpu_threads)))
    except Exception:  # noqa: BLE001
        prev_torch = 0
    cv2.setNumThreads(max(1, int(cpu_threads)))
    return prev_torch, prev_cv


def restore_thread_caps(prev: tuple[int, int]) -> None:
    prev_torch, prev_cv = prev
    try:
        import torch

        if prev_torch > 0:
            torch.set_num_threads(prev_torch)
    except Exception:  # noqa: BLE001
        pass
    try:
        import cv2

        cv2.setNumThreads(max(0, int(prev_cv)))
    except Exception:  # noqa: BLE001
        pass
