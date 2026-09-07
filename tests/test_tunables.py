import numpy as np

from videoclean.adapters.detectors.grounding_dino import GroundingDinoDetector
from videoclean.adapters.detectors.owlvit import OwlVitDetector
from videoclean.adapters.inpainters.propainter import ProPainterInpainter
from videoclean.adapters.prompt.llm import LlmPromptParser
from videoclean.application.config import PipelineConfig
from videoclean.composition import config_from_flags, make_detector, make_inpainter, make_parser


class FakeLlm:
    name = "fake"

    def status(self) -> str:
        return "ready (fake)"


def _cfg(**kw) -> PipelineConfig:
    flags: dict = dict(
        device="cpu",
        detector="grounding-dino",
        detector_model="",
        detector_threshold=0.15,
        segmenter="sam2",
        segmenter_model="",
        inpainter="propainter",
        inpainter_model="",
        formats=None,
        allow_download=False,
    )
    flags.update(kw)
    return config_from_flags(**flags)


def test_config_from_flags_plumbs_tunables():
    cfg = _cfg(
        detector_keyframes=12,
        detector_nms_iou=0.4,
        detector_max_box_area=0.3,
        tracker_min_score=0.6,
        tracker_max_template_area=0.2,
        vision_batch=3,
        propainter_mask_dilation=8,
        propainter_ref_stride=15,
        propainter_neighbor_length=12,
        propainter_subvideo_length=100,
        propainter_raft_iter=25,
    )
    assert cfg.detector_keyframes == 12
    assert cfg.detector_nms_iou == 0.4
    assert cfg.detector_max_box_area == 0.3
    assert cfg.tracker_min_score == 0.6
    assert cfg.tracker_max_template_area == 0.2
    assert cfg.vision_batch == 3
    assert cfg.propainter_mask_dilation == 8
    assert cfg.propainter_ref_stride == 15
    assert cfg.propainter_neighbor_length == 12
    assert cfg.propainter_subvideo_length == 100
    assert cfg.propainter_raft_iter == 25


def test_make_detector_gets_tunables():
    cfg = _cfg(
        detector_keyframes=16,
        detector_nms_iou=0.45,
        detector_max_box_area=0.4,
        tracker_min_score=0.7,
        tracker_max_template_area=0.3,
    )
    dino = make_detector("grounding-dino", cfg)
    assert isinstance(dino, GroundingDinoDetector)
    assert dino.keyframes == 16
    assert dino.nms_iou == 0.45
    assert dino.max_box_area == 0.4
    assert dino.tracker_min_score == 0.7
    assert dino.tracker_max_template_area == 0.3


def test_make_detector_defaults_match_adapter_history():
    cfg = _cfg()
    dino = make_detector("grounding-dino", cfg)
    assert dino.keyframes is None  # adapter applies its own 8
    assert dino.nms_iou == 0.3
    assert dino.tracker_min_score == 0.55

    owlcfg = _cfg(detector="owlvit")
    owl = make_detector("owlvit", owlcfg)
    assert isinstance(owl, OwlVitDetector)
    assert owl.keyframes is None  # adapter applies its own 12


def test_make_inpainter_gets_propainter_tunables():
    cfg = _cfg(propainter_mask_dilation=6, propainter_ref_stride=20, propainter_raft_iter=30)
    inp = make_inpainter(cfg)
    assert isinstance(inp, ProPainterInpainter)
    assert inp.mask_dilation == 6
    assert inp.ref_stride == 20
    assert inp.raft_iter == 30


def test_make_parser_gets_vision_batch():
    cfg = _cfg(vision_batch=5)
    parser = make_parser(cfg)
    assert isinstance(parser, LlmPromptParser)
    assert parser.vision_batch == 5


def test_llm_parser_vision_respects_batch_size():
    calls: list[int] = []

    class CountingLlm(FakeLlm):
        def complete(self, system, user, image_jpeg=None, images=None):
            calls.append(len(images or []))
            return '{"targets":[{"kind":"object","query":"mug","motion":"any"}]}'

    frames = [np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(5)]
    parser = LlmPromptParser(CountingLlm(), vision_batch=3)
    intent = parser.parse("убери кружку", frames=frames)
    assert calls == [3, 2]
    assert intent.parse_mode == "llm-vision"


def test_serialize_clean_form_passes_tunables():
    from videoclean.adapters.web.service import serialize_clean_form

    form = serialize_clean_form(
        {
            "vision_batch": "4",
            "detector_keyframes": "16",
            "detector_nms_iou": "0.45",
            "detector_max_box_area": "0.4",
            "tracker_min_score": "0.6",
            "tracker_max_template_area": "0.2",
            "propainter_mask_dilation": "6",
            "propainter_ref_stride": "20",
            "propainter_neighbor_length": "12",
            "propainter_subvideo_length": "100",
            "propainter_raft_iter": "25",
        }
    )
    assert form["vision_batch"] == 4
    assert form["detector_keyframes"] == 16  # int, not None
    assert form["detector_nms_iou"] == 0.45
    assert form["detector_max_box_area"] == 0.4
    assert form["tracker_min_score"] == 0.6
    assert form["tracker_max_template_area"] == 0.2
    assert form["propainter_mask_dilation"] == 6
    assert form["propainter_ref_stride"] == 20
    assert form["propainter_neighbor_length"] == 12
    assert form["propainter_subvideo_length"] == 100
    assert form["propainter_raft_iter"] == 25


def test_serialize_clean_form_empty_keyframes_means_default():
    from videoclean.adapters.web.service import serialize_clean_form

    form = serialize_clean_form({"detector_keyframes": ""})
    assert form["detector_keyframes"] is None


def test_manage_jobs_roundtrip_keeps_tunables():
    from videoclean.application.use_cases.manage_jobs import pipeline_config_from_dict

    cfg = pipeline_config_from_dict(
        {
            "device": "cpu",
            "vision_batch": 5,
            "detector_keyframes": 20,
            "tracker_min_score": 0.6,
            "propainter_raft_iter": 30,
        }
    )
    assert cfg.vision_batch == 5
    assert cfg.detector_keyframes == 20
    assert cfg.tracker_min_score == 0.6
    assert cfg.propainter_raft_iter == 30


def test_prompt_templates_override_system(tmp_path):
    custom = tmp_path / "system.md"
    custom.write_text('{"targets":[{"kind":"object","query":"CUSTOM-MARKER"}]}', encoding="utf-8")
    seen: dict = {}

    class CaptureLlm(FakeLlm):
        def complete(self, system, user, image_jpeg=None, images=None):
            seen["system"] = system
            seen["images"] = len(images or [])
            return '{"targets":[{"kind":"object","query":"mug","motion":"any"}]}'

    parser = LlmPromptParser(CaptureLlm(), templates_dir=str(tmp_path))
    parser.parse("убери кружку")
    assert "CUSTOM-MARKER" in seen["system"]
    assert seen["images"] == 0


def test_prompt_templates_fallback_for_missing_files(tmp_path):
    from videoclean.adapters.prompt.llm import SYSTEM

    parser = LlmPromptParser(FakeLlm(), templates_dir=str(tmp_path))
    assert parser._templates["system.md"] == SYSTEM
