from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from videoclean.domain.intent import CROP_PARTS, SCREEN_WHERES

# Same bands as application.select.region_name / _where_ok.
_TOP, _BOTTOM = 0.38, 0.62
_LEFT, _RIGHT = 0.38, 0.62


@dataclass(frozen=True)
class MaskRegion:
    where: str | None
    nx: float
    ny: float
    area: int


def where_from_norm(nx: float, ny: float) -> str | None:
    """Map normalized centroid (0..1) to a SCREEN_WHERES label, or None for center."""
    vert = "top" if ny <= _TOP else "bottom" if ny >= _BOTTOM else None
    horz = "left" if nx <= _LEFT else "right" if nx >= _RIGHT else None
    if vert and horz:
        return f"{vert}-{horz}"
    return vert or horz


def mask_regions(mask: np.ndarray, *, min_area: int | None = None) -> list[MaskRegion]:
    """Connected components of a painted mask → screen regions (geometry, not VLM)."""
    import cv2

    if mask is None or mask.size == 0:
        return []
    binary = (mask > 0).astype(np.uint8)
    if not np.any(binary):
        return []
    h, w = binary.shape[:2]
    # Tiny preview masks (tests / low-res) still count; full-HD ignores 1px noise.
    area_floor = min_area if min_area is not None else max(4, (h * w) // 50_000)
    n, _labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out: list[MaskRegion] = []
    for i in range(1, n):  # 0 = background
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < area_floor:
            continue
        cx, cy = float(centroids[i][0]), float(centroids[i][1])
        nx, ny = cx / max(1, w), cy / max(1, h)
        out.append(MaskRegion(where=where_from_norm(nx, ny), nx=nx, ny=ny, area=area))
    out.sort(key=lambda r: -r.area)
    return out


def normalize_where(value: str | None) -> str | None:
    if not value:
        return None
    v = str(value).strip().lower().replace("_", "-").replace(" ", "-")
    aliases = {
        "upper-right": "top-right",
        "upper-left": "top-left",
        "lower-right": "bottom-right",
        "lower-left": "bottom-left",
    }
    v = aliases.get(v, v)
    return v if v in SCREEN_WHERES else None


def part_or_where(part: str | None, where: str | None) -> tuple[str | None, str | None]:
    where = normalize_where(where)
    if part in SCREEN_WHERES:
        if where is None:
            where = normalize_where(part)
        part = None
    if part not in CROP_PARTS:
        if normalize_where(part):
            where = where or normalize_where(part)
        part = None
    return part, where
