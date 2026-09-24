"""Build the SLA clip next to this file and print its sha256.

The mp4 is not committed. Commit only sla_1080p30_60s.sha256.
"""

from __future__ import annotations

import hashlib
import struct
import subprocess
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PNG = ROOT / "logo_160x48.png"
MP4 = ROOT / "sla_1080p30_60s.mp4"
HASH = ROOT / "sla_1080p30_60s.sha256"


def _png(width: int, height: int) -> bytes:
    raw = bytearray()
    for _y in range(height):
        raw.append(0)
        for x in range(width):
            black = 16 <= (x % 32) < 24
            pixel = b"\x00\x00\x00" if black else b"\xff\xff\xff"
            raw.extend(pixel)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(bytes(raw), 9)) + chunk(b"IEND", b"")


def main() -> None:
    PNG.write_bytes(_png(160, 48))
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=1920x1080:rate=30:duration=60",
        "-i",
        str(PNG),
        "-filter_complex",
        "overlay=x=1720:y=40",
        "-frames:v",
        "1800",
        "-pix_fmt",
        "yuv420p",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        str(MP4),
    ]
    subprocess.run(cmd, check=True)
    digest = hashlib.sha256(MP4.read_bytes()).hexdigest()
    HASH.write_text(digest + "\n", encoding="utf-8")
    print(digest)


if __name__ == "__main__":
    main()
