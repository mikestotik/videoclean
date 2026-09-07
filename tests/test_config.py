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
    names = parse_name_list("owlvit,grounding-dino", DETECTORS, default=["grounding-dino"])
    assert names == ["owlvit", "grounding-dino"]


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
