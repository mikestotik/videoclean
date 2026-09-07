from videoclean.application.config import PipelineConfig
from videoclean.application.errors import PipelineError
from videoclean.application.config import parse_name_list, DETECTORS


def test_default_config_validates():
    PipelineConfig().validate()


def test_device_must_be_explicit():
    cfg = PipelineConfig(device="gpu")
    try:
        cfg.validate()
    except PipelineError as exc:
        assert "--device" in str(exc)
    else:
        raise AssertionError("expected PipelineError")


def test_sam2_is_a_valid_segmenter():
    PipelineConfig(segmenter="sam2").validate()
    PipelineConfig(segmenter="sam2-video").validate()


def test_grounding_dino_is_a_valid_detector():
    PipelineConfig(detectors=["grounding-dino"]).validate()


def test_propainter_is_a_valid_inpainter():
    PipelineConfig(inpainter="propainter").validate()


def test_lama_is_a_valid_inpainter():
    PipelineConfig(inpainter="lama").validate()


def test_detector_chain():
    names = parse_name_list("grounding-dino,grounding-dino", DETECTORS, default=["grounding-dino"])
    assert names == ["grounding-dino"]


def test_unknown_detector():
    try:
        parse_name_list("yolo", DETECTORS, default=["grounding-dino"])
    except PipelineError as exc:
        assert "yolo" in str(exc)
    else:
        raise AssertionError("expected PipelineError")


def test_recurrence_is_not_a_detector():
    try:
        parse_name_list("recurrence", DETECTORS, default=["grounding-dino"])
    except PipelineError as exc:
        assert "recurrence" in str(exc)
    else:
        raise AssertionError("expected PipelineError")


def test_default_detectors_are_grounding_dino():
    from videoclean.application.config import DEFAULT_DETECTORS

    assert DEFAULT_DETECTORS == ["grounding-dino"]


def test_no_profile_field():
    assert not hasattr(PipelineConfig(), "profile")


def test_llm_place_must_be_known():
    cfg = PipelineConfig(llm_place="edge")
    try:
        cfg.validate()
    except PipelineError as exc:
        assert "--llm" in str(exc)
    else:
        raise AssertionError("expected PipelineError")


def test_tunable_params_have_defaults():
    cfg = PipelineConfig()
    assert cfg.detector_keyframes is None  # adapter default: dino 8
    assert cfg.detector_nms_iou == 0.3
    assert cfg.detector_max_box_area == 0.25
    assert cfg.tracker_min_score == 0.55
    assert cfg.tracker_max_template_area == 0.12
    assert cfg.vision_batch == 2
    assert cfg.propainter_mask_dilation == 4
    assert cfg.propainter_ref_stride == 10
    assert cfg.propainter_neighbor_length == 10
    assert cfg.propainter_subvideo_length == 80
    assert cfg.propainter_raft_iter == 20
    assert cfg.prompt_templates is None
    cfg.validate()


def test_tunable_params_validate_bounds():
    for bad in (
        {"detector_keyframes": 0},
        {"detector_keyframes": -3},
        {"detector_nms_iou": 0.0},
        {"detector_nms_iou": 1.0},
        {"detector_max_box_area": 0.0},
        {"detector_max_box_area": 1.5},
        {"tracker_min_score": 0.0},
        {"tracker_min_score": 1.0},
        {"tracker_max_template_area": -0.1},
        {"vision_batch": 0},
        {"vision_batch": -1},
        {"propainter_mask_dilation": -1},
        {"propainter_ref_stride": 0},
        {"propainter_neighbor_length": 0},
        {"propainter_subvideo_length": 0},
        {"propainter_raft_iter": 0},
    ):
        try:
            PipelineConfig(**bad).validate()
        except PipelineError:
            continue
        raise AssertionError(f"expected PipelineError for {bad}")


def test_prompt_templates_must_exist_or_be_empty():
    PipelineConfig(prompt_templates=None).validate()
    try:
        PipelineConfig(prompt_templates="/nonexistent/dir/xyz").validate()
    except PipelineError as exc:
        assert "prompt-templates" in str(exc)
    else:
        raise AssertionError("expected PipelineError for missing dir")
