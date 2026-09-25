import pytest

from videoclean.application.errors import PipelineError
from server.service import fit_short_side


def test_fit_short_side_landscape_1080_to_720_and_480():
    assert fit_short_side(1920, 1080, 720) == (1280, 720)
    assert fit_short_side(1920, 1080, 480) == (852, 480)


def test_fit_short_side_portrait_uses_width():
    assert fit_short_side(1080, 1920, 720) == (720, 1280)


def test_fit_short_side_refuses_upscale_and_unknown():
    with pytest.raises(PipelineError):
        fit_short_side(1280, 720, 720)
    with pytest.raises(PipelineError):
        fit_short_side(1920, 1080, 360)
