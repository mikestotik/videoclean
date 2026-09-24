from videoclean.application.profiles import merge_run_config


def _merge(defaults, preset, profile, explicit):
    return merge_run_config(
        defaults,
        preset,
        profile,
        explicit,
        resolve_device=lambda name: "cpu" if name in {"", "auto", "cpu"} else name,
    )


def test_merge_order_dilate():
    defaults = {"mask_dilate_px": 3, "profile": "custom", "device": "cpu"}
    assert _merge(defaults, {"mask_dilate_px": 8}, None, {})["mask_dilate_px"] == 8
    assert _merge(defaults, {"mask_dilate_px": 8}, "quality", {})["mask_dilate_px"] == 5
    assert _merge(defaults, {"mask_dilate_px": 8}, "quality", {"mask_dilate_px": 9})["mask_dilate_px"] == 9


def test_preset_profile_quality_does_not_keep_its_own_dilate():
    defaults = {"mask_dilate_px": 3, "profile": "custom", "device": "cpu"}
    out = _merge(defaults, {"profile": "quality", "mask_dilate_px": 8}, None, {})
    assert out["mask_dilate_px"] == 5


def test_absent_device_is_not_explicit_cpu():
    defaults = {"device": "auto", "profile": "custom", "mask_dilate_px": 3}
    out = _merge(defaults, None, "balanced", {})
    assert out["device"] == "cpu"
    assert out["device_requested"] == "auto"
    assert "cpu" != out["device_requested"]
