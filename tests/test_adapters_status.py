import pytest

from videoclean.adapters.inpainters.lama import LamaInpainter
from videoclean.adapters.inpainters.propainter import ProPainterInpainter, find_vendor, find_weights
from videoclean.adapters.segmenters.sam2 import Sam2Segmenter
from videoclean.application.config import PipelineConfig
from videoclean.composition import make_inpainter, make_segmenter


def test_sam2_status_does_not_crash_without_runtime():
    status = Sam2Segmenter(model_id="facebook/sam2-hiera-tiny", device="cpu").status()
    assert "sam2" in status.lower() or "unavailable" in status or "ready" in status


def test_propainter_status_without_vendor():
    status = ProPainterInpainter(model_id="camenduru/ProPainter", device="cpu").status()
    assert "unavailable" in status or "ready" in status


def test_find_vendor_missing_by_default():
    # May exist on a GPU box; the function must not raise.
    find_vendor()
    find_weights("camenduru/ProPainter")


def test_factories_accept_sam2_and_propainter():
    cfg = PipelineConfig(segmenter="sam2", inpainter="propainter", device="cpu")
    cfg.validate()
    assert make_segmenter(cfg).name == "sam2"
    assert make_inpainter(cfg).name == "propainter"


def test_lama_status_does_not_crash():
    status = LamaInpainter(device="cpu").status()
    assert "ready" in status or "unavailable" in status


def test_lama_loads_with_map_location_cpu():
    """PyPI SimpleLama omits map_location; our adapter must still load big-lama.pt on CPU-only Macs."""
    pytest.importorskip("simple_lama_inpainting")
    from videoclean.adapters.inpainters.lama import find_weights

    if find_weights() is None:
        pytest.skip("big-lama.pt not on disk")
    import numpy as np

    inp = LamaInpainter(device="cpu")
    frame = np.full((32, 32, 3), 80, dtype=np.uint8)
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:24, 8:24] = 255
    out = inp.inpaint(frame, mask)
    assert out.shape == frame.shape


def test_factory_accepts_lama():
    cfg = PipelineConfig(inpainter="lama", device="cpu")
    cfg.validate()
    assert make_inpainter(cfg).name == "lama"


def test_grounding_dino_uses_flag_threshold():
    from videoclean.composition import make_detector

    cfg = PipelineConfig(detectors=["grounding-dino"], detector_threshold=0.15, device="cpu")
    cfg.validate()
    det = make_detector("grounding-dino", cfg)
    assert det.threshold == 0.15


def test_factory_accepts_sam2_video_and_grounding_dino():
    from videoclean.composition import make_detector

    cfg = PipelineConfig(segmenter="sam2-video", detectors=["grounding-dino"], device="cpu")
    cfg.validate()
    assert make_segmenter(cfg).name == "sam2-video"
    assert make_detector("grounding-dino", cfg).name == "grounding-dino"
