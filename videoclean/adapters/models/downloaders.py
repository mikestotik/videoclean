from __future__ import annotations

import json
import os
import shutil
import subprocess
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

from videoclean.adapters.inpainters.lama import LAMA_MODEL_URL, _default_model_path
from videoclean.adapters.inpainters.propainter import DEFAULT_HF_REPO, find_vendor
from videoclean.adapters.models.catalog import COMPONENT_BY_ID, OLLAMA_API
from videoclean.application.errors import DownloadCancelled, PipelineError
from videoclean.store import resolve_data_dir

OnProgress = Callable[..., None]
IsCancelled = Callable[[], bool]

PROPAINTER_GIT = "https://github.com/sczhou/ProPainter.git"


def run_download(
    component_id: str,
    on_progress: OnProgress | None = None,
    is_cancelled: IsCancelled | None = None,
) -> None:
    info = COMPONENT_BY_ID.get(component_id)
    if info is None:
        raise PipelineError(f"unknown component {component_id!r}")
    _check(is_cancelled)
    if info.kind in {"detector", "segmenter"}:
        download_hf(info.model_ref, on_progress=on_progress, is_cancelled=is_cancelled)
        return
    if info.id == "inpainter:lama":
        download_lama(on_progress=on_progress, is_cancelled=is_cancelled)
        return
    if info.id == "inpainter:propainter":
        download_propainter(on_progress=on_progress, is_cancelled=is_cancelled)
        return
    if info.kind == "llm":
        download_ollama(info.model_ref, on_progress=on_progress, is_cancelled=is_cancelled)
        return
    raise PipelineError(f"no downloader for {component_id!r}")


def download_hf(
    repo_id: str,
    *,
    local_dir: str | None = None,
    on_progress: OnProgress | None = None,
    is_cancelled: IsCancelled | None = None,
) -> None:
    _check(is_cancelled)
    kwargs: dict = {
        "local_files_only": False,
        "tqdm_class": _make_tqdm(on_progress, is_cancelled),
    }
    if local_dir:
        kwargs["local_dir"] = local_dir
    _snapshot_download(repo_id, **kwargs)
    _emit(on_progress, 1.0, f"{repo_id} ready")


def download_lama(
    on_progress: OnProgress | None = None,
    is_cancelled: IsCancelled | None = None,
) -> None:
    _check(is_cancelled)
    dest = _default_model_path()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")

    def hook(blocknum: int, blocksize: int, totalsize: int) -> None:
        _check(is_cancelled)
        done = blocknum * blocksize
        total = totalsize if totalsize and totalsize > 0 else None
        frac = (done / total) if total else 0.0
        _emit(
            on_progress,
            min(frac, 0.99),
            "downloading big-lama.pt",
            bytes_done=done,
            bytes_total=total,
        )

    try:
        _urlretrieve(LAMA_MODEL_URL, tmp, hook)
        tmp.replace(dest)
    except DownloadCancelled:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise
    except Exception as exc:  # noqa: BLE001
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise PipelineError(f"failed to download big-lama.pt: {exc}") from exc
    size = dest.stat().st_size if dest.is_file() else None
    _emit(on_progress, 1.0, "ready", bytes_done=size, bytes_total=size)


def download_propainter(
    on_progress: OnProgress | None = None,
    is_cancelled: IsCancelled | None = None,
) -> None:
    _check(is_cancelled)
    vendor = find_vendor()
    if vendor is None:
        dest = _propainter_vendor_dest()
        _emit(on_progress, 0.05, "cloning ProPainter")
        _git_clone(PROPAINTER_GIT, dest, is_cancelled=is_cancelled)
        _check(is_cancelled)
    weights_dest = _propainter_weights_dest()
    weights_dest.mkdir(parents=True, exist_ok=True)
    _emit(on_progress, 0.2, "downloading ProPainter weights")

    def wrapped(fraction: float, message: str = "", **kwargs) -> None:
        _emit(on_progress, 0.2 + 0.8 * min(max(fraction, 0.0), 1.0), message or "weights", **kwargs)

    download_hf(
        DEFAULT_HF_REPO,
        local_dir=str(weights_dest),
        on_progress=wrapped,
        is_cancelled=is_cancelled,
    )


def download_ollama(
    tag: str,
    on_progress: OnProgress | None = None,
    is_cancelled: IsCancelled | None = None,
) -> None:
    _check(is_cancelled)
    payload = json.dumps({"name": tag, "model": tag, "stream": True}).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_API.rstrip("/") + "/api/pull",
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with _http_open(req, timeout=None) as resp:
            while True:
                _check(is_cancelled)
                line = resp.readline()
                if not line:
                    break
                raw = line.decode("utf-8") if isinstance(line, bytes) else line
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if obj.get("error"):
                    raise PipelineError(str(obj["error"]))
                total = int(obj.get("total") or 0)
                completed = int(obj.get("completed") or 0)
                status = str(obj.get("status") or "pulling")
                frac = (completed / total) if total else 0.0
                _emit(
                    on_progress,
                    min(frac, 0.99),
                    status,
                    bytes_done=completed or None,
                    bytes_total=total or None,
                )
    except DownloadCancelled:
        raise
    except urllib.error.URLError as exc:
        raise PipelineError(f"unavailable: start ollama ({exc.reason})") from exc
    except PipelineError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise PipelineError(f"ollama pull failed: {exc}") from exc
    _emit(on_progress, 1.0, "ready")


def _snapshot_download(repo_id: str, **kwargs):
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=repo_id, **kwargs)


def _urlretrieve(url: str, dest: Path, reporthook) -> None:
    urllib.request.urlretrieve(url, filename=str(dest), reporthook=reporthook)


def _http_open(req: urllib.request.Request, timeout: float | None = None):
    return urllib.request.urlopen(req, timeout=timeout)


def _git_clone(url: str, dest: Path, is_cancelled: IsCancelled | None = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        shutil.rmtree(dest)
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    try:
        proc = subprocess.Popen(
            ["git", "clone", "--depth", "1", url, str(dest)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
        )
    except FileNotFoundError as exc:
        raise PipelineError("git is required to clone ProPainter") from exc
    try:
        while proc.poll() is None:
            _check(is_cancelled)
            try:
                proc.wait(timeout=0.25)
            except subprocess.TimeoutExpired:
                continue
    except DownloadCancelled:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=1)
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
        raise
    if proc.returncode:
        raise PipelineError(f"git clone failed: exit {proc.returncode}")


def _propainter_vendor_dest() -> Path:
    env = os.environ.get("VIDEOCLEAN_PROPAINTER_ROOT", "").strip()
    if env:
        return Path(env).expanduser()
    return resolve_data_dir() / "vendor" / "ProPainter"


def _propainter_weights_dest() -> Path:
    env = os.environ.get("VIDEOCLEAN_PROPAINTER_WEIGHTS", "").strip()
    if env:
        return Path(env).expanduser()
    return resolve_data_dir() / "weights" / "propainter"


def _check(is_cancelled: IsCancelled | None) -> None:
    if is_cancelled is not None and is_cancelled():
        raise DownloadCancelled("download cancelled")


def _emit(
    on_progress: OnProgress | None,
    fraction: float,
    message: str = "",
    bytes_done: int | None = None,
    bytes_total: int | None = None,
) -> None:
    if on_progress is None:
        return
    on_progress(fraction, message, bytes_done=bytes_done, bytes_total=bytes_total)


def _make_tqdm(on_progress: OnProgress | None, is_cancelled: IsCancelled | None):
    from tqdm.auto import tqdm as BaseTqdm

    class ProgressTqdm(BaseTqdm):
        def update(self, n=1):
            _check(is_cancelled)
            result = super().update(n)
            total = self.total or 0
            current = self.n or 0
            frac = (current / total) if total else 0.0
            _emit(
                on_progress,
                min(max(frac, 0.0), 0.99),
                self.desc or "downloading",
                bytes_done=int(current),
                bytes_total=int(total) if total else None,
            )
            return result

    return ProgressTqdm
