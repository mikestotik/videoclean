from videoclean.adapters.media.ffmpeg import encoder_argv


def test_nvenc_argv_has_no_crf():
    argv = encoder_argv(nvenc=True)
    assert argv[:6] == ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr"]
    assert "-cq" in argv and "18" in argv
    assert "-crf" not in argv


def test_missing_nvenc_uses_veryfast():
    argv = encoder_argv(nvenc=False)
    assert "libx264" in argv and "veryfast" in argv and "-crf" in argv
