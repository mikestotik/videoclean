import numpy as np

from videoclean.adapters.prompt.locations import mask_regions, where_from_norm
from videoclean.application.use_cases.build_prompt import apply_mask_wheres
from videoclean.domain.intent import Target


def test_where_from_norm_corners_and_edges():
    assert where_from_norm(0.1, 0.1) == "top-left"
    assert where_from_norm(0.9, 0.1) == "top-right"
    assert where_from_norm(0.9, 0.9) == "bottom-right"
    assert where_from_norm(0.1, 0.9) == "bottom-left"
    assert where_from_norm(0.5, 0.1) == "top"
    assert where_from_norm(0.5, 0.5) is None


def test_mask_regions_top_right():
    m = np.zeros((100, 200), dtype=np.uint8)
    m[5:25, 160:195] = 255  # top-right blob
    regs = mask_regions(m)
    assert len(regs) == 1
    assert regs[0].where == "top-right"
    assert regs[0].nx > 0.7
    assert regs[0].ny < 0.3


def test_apply_mask_wheres_overrides_vlm_center_guess():
    from videoclean.adapters.prompt.locations import MaskRegion

    targets = [Target(kind="watermark", query="logo", where="top")]
    regs = [MaskRegion(where="top-right", nx=0.85, ny=0.12, area=400)]
    out = apply_mask_wheres(targets, regs)
    assert out[0].where == "top-right"
