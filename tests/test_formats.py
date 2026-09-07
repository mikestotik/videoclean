from pathlib import Path

from videoclean.formats import parse_formats, resolve_dest


def test_default_mp4():
    assert parse_formats(None) == ["mp4"]


def test_comma_and_repeat():
    assert parse_formats(["mp4,webm", "dash"]) == ["mp4", "webm", "dash"]


def test_hls_alias():
    assert parse_formats("hls") == ["hls-fmp4"]


def test_both_is_not_a_format():
    try:
        parse_formats("both")
    except ValueError as exc:
        assert "not a format" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_unknown():
    try:
        parse_formats("avi")
    except ValueError as exc:
        assert "unknown" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_resolve_single_mp4_keeps_path():
    out = Path("/tmp/cleaned.mp4")
    assert resolve_dest(out, "mp4", ["mp4"]) == out


def test_resolve_multi_uses_suffixes():
    out = Path("/tmp/cleaned.mp4")
    assert resolve_dest(out, "webm", ["mp4", "webm"]).name == "cleaned.webm"
    assert resolve_dest(out, "hls-fmp4", ["mp4", "hls-fmp4"]).name == "cleaned.hls-fmp4"
