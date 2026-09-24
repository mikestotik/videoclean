import numpy as np

from videoclean.application.hole_policy import plan_holes, probe_lama_dtype
from videoclean.application.profiles import profile_defaults


def box_mask(bw: int, bh: int, frame=(1080, 1920)) -> np.ndarray:
    mask = np.zeros(frame, dtype=np.uint8)
    mask[40 : 40 + bh, 40 : 40 + bw] = 255
    return mask


def test_small_mask_plans_a_crop_under_the_ceiling():
    plans = plan_holes(frame_hw=(1080, 1920), mask=box_mask(80, 40), budget_side=256, budget_batch=2)
    assert plans
    assert plans[0].side <= 256 and plans[0].side % 8 == 0
    assert plans[0].batch <= 2
    assert plans[0].policy == "lama-crop"


def test_lama_family_stays_lama_on_a_large_hole():
    plans = plan_holes(
        frame_hw=(1080, 1920),
        mask=box_mask(900, 900),
        family="lama",
        budget_side=512,
        budget_batch=1,
    )
    assert plans[0].policy == "lama-crop"
    assert plans[0].side <= 512


def test_coverage_is_pixels_not_box_area():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[0:10, 0:10] = 255
    plans = plan_holes(frame_hw=(100, 100), mask=mask, family="lama")
    assert plans[0].mean_coverage == 0.01


def test_half_probe_failure_locks_fp32():
    assert probe_lama_dtype(raise_on_half=True) == "fp32"


def test_quality_cuda_workers_are_auto_and_balanced_stays_sam2():
    quality = profile_defaults("quality", "cuda")
    assert quality["inpaint_workers"] == 0
    assert quality["segmenter"] == "sam2-video"
    assert quality["inpainter"] == "propainter"
    assert profile_defaults("balanced", "cuda")["segmenter"] == "sam2"
