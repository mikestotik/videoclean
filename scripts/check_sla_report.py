#!/usr/bin/env python3
"""Release gate for the balanced 1080p30 one-minute clip. Exit 0 only when every SLA rule holds."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

TIMING_KEYS = (
    "load",
    "decode",
    "parse",
    "detect",
    "track",
    "segment",
    "inpaint",
    "verify",
    "encode",
    "package",
)


def check(report: dict, *, max_wall_s: float | None, require_warm: bool, require_cuda: bool) -> int:
    budget = report.get("budget") or {}
    hole = report.get("hole") or {}
    timings = report.get("timings") or {}
    if require_warm and budget.get("warm") is not True:
        return 1
    if require_cuda and budget.get("device") != "cuda":
        return 1
    if budget.get("vramCeilingMb") is not None:
        return 1
    if budget.get("inpaintMaxSide") is not None or hole.get("limitedBy") == "inpaint_max_side":
        return 1
    if budget.get("cpuThreads") != budget.get("cpuCount"):
        return 1
    if hole.get("policy") != "lama-crop":
        return 1
    for key in TIMING_KEYS:
        if not isinstance(timings.get(key), (int, float)):
            return 1
    try:
        wall = float(report["finishedAt"]) - float(report["startedAt"])
    except (KeyError, TypeError, ValueError):
        started = report.get("startedAt")
        finished = report.get("finishedAt")
        if isinstance(started, str) or isinstance(finished, str):
            from datetime import datetime

            wall = (
                datetime.fromisoformat(str(finished).replace("Z", "+00:00"))
                - datetime.fromisoformat(str(started).replace("Z", "+00:00"))
            ).total_seconds()
        else:
            return 1
    if max_wall_s is not None and wall > max_wall_s:
        return 1
    gap = abs(wall - sum(float(timings[key]) for key in TIMING_KEYS))
    if gap > 5:
        return 1
    limited = hole.get("limitedBy")
    peak = budget.get("vramPeakInpaintMb")
    vram_budget = budget.get("vramBudgetMb")
    if not isinstance(peak, (int, float)) or not isinstance(vram_budget, (int, float)) or vram_budget <= 0:
        return 1
    if limited == "frames":
        if hole.get("batch") != hole.get("frameCount"):
            return 1
    elif limited == "ceiling":
        if not (0.60 * float(vram_budget) <= float(peak) <= float(vram_budget)):
            return 1
    else:
        return 1
    return 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-wall-s", type=float, default=None)
    parser.add_argument("--require-warm", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument(
        "--hash-file",
        type=Path,
        default=Path("tests/fixtures/sla/sla_1080p30_60s.sha256"),
    )
    args = parser.parse_args(argv)
    if not args.hash_file.is_file():
        print("missing hash file", file=sys.stderr)
        return 2
    expect = args.hash_file.read_text(encoding="utf-8").strip().split()[0]
    if _sha256(args.clip) != expect:
        print("clip hash mismatch", file=sys.stderr)
        return 3
    report = json.loads(args.report.read_text(encoding="utf-8"))
    code = check(
        report,
        max_wall_s=args.max_wall_s,
        require_warm=args.require_warm or args.max_wall_s is not None,
        require_cuda=args.require_cuda or args.max_wall_s is not None,
    )
    print("eye-check frames at indices 0, 899, 1799 (display 1, 900, 1800)")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
