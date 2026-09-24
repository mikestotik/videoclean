"""BGR24 frame mmap. One frame in RAM at a time."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from videoclean.application.errors import PipelineError

_NETWORK_FS = {
    "nfs",
    "nfs4",
    "cifs",
    "smb",
    "smbfs",
    "sshfs",
    "fuse.sshfs",
    "virtiofs",
    "9p",
    "lustre",
    "glusterfs",
}


def filesystem_kind(path: Path) -> str:
    """Best-effort fstype name. Empty string if the platform will not say."""
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    probe = target if target.is_dir() else target.parent
    try:
        if os.uname().sysname == "Darwin":
            import subprocess

            out = subprocess.check_output(
                ["stat", "-f", "%T", str(probe)],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            return out.strip().lower()
    except Exception:  # noqa: BLE001
        pass
    mountinfo = Path("/proc/self/mountinfo")
    if mountinfo.is_file():
        best = ""
        best_len = -1
        try:
            resolved = str(probe.resolve())
        except OSError:
            resolved = str(probe)
        for line in mountinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.split()
            if len(parts) < 10 or "-" not in parts:
                continue
            dash = parts.index("-")
            mount = parts[4]
            fstype = parts[dash + 1] if dash + 1 < len(parts) else ""
            if resolved == mount or resolved.startswith(mount.rstrip("/") + "/"):
                if len(mount) > best_len:
                    best = fstype.lower()
                    best_len = len(mount)
        return best
    return ""


def assert_local_data_dir(path: Path) -> None:
    kind = filesystem_kind(path)
    if kind in _NETWORK_FS:
        raise PipelineError(
            f"data_dir is on a network filesystem ({kind}): {path}. "
            "Put VIDEOCLEAN_DATA_DIR on the container disk (sqlite and frame mmap)."
        )


class FrameStore:
    """Random-access bgr24 frames in one file. __getitem__ returns a copy."""

    def __init__(self, path: Path, n: int, h: int, w: int, *, create: bool = True) -> None:
        if n < 1 or h < 1 or w < 1:
            raise PipelineError(f"FrameStore needs n,h,w >= 1, got {n}x{h}x{w}")
        self.path = Path(path)
        self.n = int(n)
        self.h = int(h)
        self.w = int(w)
        self.frame_bytes = self.h * self.w * 3
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if create and (not self.path.exists() or self.path.stat().st_size != self.n * self.frame_bytes):
            with open(self.path, "wb") as fh:
                fh.truncate(self.n * self.frame_bytes)
        self._fh = open(self.path, "r+b", buffering=0)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int) -> np.ndarray:
        i = int(index)
        if i < 0 or i >= self.n:
            raise IndexError(i)
        self._fh.seek(i * self.frame_bytes)
        raw = self._fh.read(self.frame_bytes)
        if len(raw) != self.frame_bytes:
            raise PipelineError(f"short read at frame {i} of {self.path}")
        return np.frombuffer(raw, dtype=np.uint8).reshape(self.h, self.w, 3).copy()

    def __setitem__(self, index: int, frame: np.ndarray) -> None:
        i = int(index)
        if i < 0 or i >= self.n:
            raise IndexError(i)
        arr = np.ascontiguousarray(frame, dtype=np.uint8)
        if arr.shape != (self.h, self.w, 3):
            raise PipelineError(f"frame {i} shape {arr.shape} != {(self.h, self.w, 3)}")
        self._fh.seek(i * self.frame_bytes)
        self._fh.write(arr.tobytes())

    def write_bytes_at(self, index: int, raw: bytes) -> None:
        i = int(index)
        if len(raw) != self.frame_bytes:
            raise PipelineError(f"frame {i} has {len(raw)} bytes, want {self.frame_bytes}")
        self._fh.seek(i * self.frame_bytes)
        self._fh.write(raw)

    def close(self) -> None:
        try:
            self._fh.close()
        except OSError:
            pass

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass
