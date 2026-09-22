"""On-demand Ollama setup: install the binary and launch `ollama serve`.

Used by the Config UI so test pods can boot without the ~1 GB ollama
download and install it later on demand. Installs to ~/.local/bin
(no root required); progress follows the (fraction, message,
bytes_done, bytes_total) convention from models.downloaders so the
install shows up in the regular downloads SSE channel.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import tarfile
import urllib.request
from collections.abc import Callable
from pathlib import Path

from videoclean.application.errors import DownloadCancelled, PipelineError

OnProgress = Callable[..., None]
IsCancelled = Callable[[], bool]

OLLAMA_SYSTEM_ID = "system:ollama"
OLLAMA_DOWNLOAD_BASE = "https://ollama.com/download"


def ollama_bin_dir() -> Path:
    return Path.home() / ".local" / "bin"


def ollama_binary_path() -> Path:
    return ollama_bin_dir() / "ollama"


def ollama_installed() -> bool:
    return ollama_binary_path().is_file() or shutil.which("ollama") is not None


def ollama_download_url() -> str:
    """Download URL for the current Linux host. Only Linux is supported."""
    if platform.system() != "Linux":
        raise PipelineError(
            "automatic ollama install supports Linux only; "
            "on macOS install the Ollama app manually"
        )
    machine = platform.machine().lower()
    if machine in {"x86_64", "amd64"}:
        arch = "amd64"
    elif machine in {"aarch64", "arm64"}:
        arch = "arm64"
    else:
        raise PipelineError(f"unsupported architecture for ollama: {machine}")
    return f"{OLLAMA_DOWNLOAD_BASE}/ollama-linux-{arch}.tar.zst"


def install_ollama_binary(
    on_progress: OnProgress | None = None,
    is_cancelled: IsCancelled | None = None,
) -> Path:
    """Download and unpack the ollama binary to ~/.local/bin."""
    _check(is_cancelled)
    url = ollama_download_url()
    dest_dir = ollama_bin_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)
    tmp = dest_dir / "ollama-install.part"
    if tmp.exists():
        tmp.unlink()

    def hook(blocknum: int, blocksize: int, totalsize: int) -> None:
        _check(is_cancelled)
        done = blocknum * blocksize
        total = totalsize if totalsize and totalsize > 0 else None
        frac = (done / total) if total else 0.0
        _emit(on_progress, min(frac, 0.90), "downloading ollama", done, total)

    try:
        urllib.request.urlretrieve(url, filename=str(tmp), reporthook=hook)
    except DownloadCancelled:
        tmp.unlink(missing_ok=True)
        raise
    except Exception as exc:  # noqa: BLE001
        tmp.unlink(missing_ok=True)
        raise PipelineError(f"failed to download ollama: {exc}") from exc
    _check(is_cancelled)
    _emit(on_progress, 0.92, "unpacking ollama")
    try:
        _unpack_zst(tmp, dest_dir)
    except DownloadCancelled:
        raise
    except Exception as exc:  # noqa: BLE001
        raise PipelineError(f"failed to unpack ollama: {exc}") from exc
    finally:
        tmp.unlink(missing_ok=True)
    binary = ollama_binary_path()
    if not binary.is_file():
        raise PipelineError("ollama archive did not contain the binary")
    binary.chmod(0o755)
    _emit(on_progress, 1.0, "ollama ready")
    return binary


def start_ollama_serve(log_path: Path | None = None) -> str:
    """Launch `ollama serve` detached. Returns 'started' or 'already running'."""
    if _ollama_serving():
        return "already running"
    binary = ollama_binary_path()
    if not binary.is_file():
        resolved = shutil.which("ollama")
        if resolved is None:
            raise PipelineError("ollama is not installed")
        binary = Path(resolved)
    log_path = log_path or (Path.home() / ".cache" / "videoclean-ollama.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        log_file = log_path.open("ab")
    except OSError as exc:
        raise PipelineError(f"cannot open ollama log: {exc}") from exc
    with log_file:
        try:
            subprocess.Popen(
                [str(binary), "serve"],
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except FileNotFoundError as exc:
            raise PipelineError("ollama is not installed") from exc
    return "started"


def _ollama_serving(timeout: float = 2.0) -> bool:
    base = (os.environ.get("VIDEOCLEAN_OLLAMA_URL") or "http://127.0.0.1:11434").rstrip("/")
    try:
        with urllib.request.urlopen(base + "/api/tags", timeout=timeout) as resp:
            return resp.status < 400
    except Exception:  # noqa: BLE001
        return False


def _unpack_zst(archive: Path, dest_dir: Path) -> None:
    zstd = shutil.which("zstd") or shutil.which("unzstd")
    if zstd is not None:
        proc = subprocess.run(
            ["tar", "--use-compress-program=unzstd", "-xf", str(archive), "-C", str(dest_dir)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=False,
        )
        if proc.returncode:
            raise PipelineError(proc.stderr.decode("utf-8", "replace")[-400:] or "tar failed")
        return
    try:
        import zstandard as zstd_lib
    except ImportError as exc:
        raise PipelineError(
            "zstd is required to unpack ollama (apt install zstd / brew install zstd)"
        ) from exc
    with archive.open("rb") as src, (dest_dir / "ollama.tar").open("wb") as dst:
        zstd_lib.ZstdDecompressor().copy_stream(src, dst)
    with tarfile.open(dest_dir / "ollama.tar") as tar:
        tar.extractall(dest_dir, filter="data")
    (dest_dir / "ollama.tar").unlink(missing_ok=True)


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
