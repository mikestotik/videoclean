from videoclean.adapters.segmenters.sam2_video import (
    _is_cuda_oom,
    _live_vram_budget_bytes,
    choose_sam2_storage,
)


def test_choose_sam2_storage_full_gpu_when_budget_has_wide_headroom():
    # 100 frames @1024 ≈ 1.17 GiB; need ~8 GiB headroom for state → 12 GiB fits full GPU.
    decision = choose_sam2_storage(
        n_frames=100,
        image_size=1024,
        device="cuda",
        vram_budget_bytes=12 * 1024**3,
    )
    assert decision.offload_video_to_cpu is False
    assert decision.offload_state_to_cpu is False
    assert decision.storage_device == "cuda"


def test_choose_sam2_storage_hybrid_keeps_video_on_gpu_state_on_cpu():
    # 100 frames ≈ 1.17 GiB; 8 GiB covers video+6 GiB hybrid but not full 8 GiB state headroom.
    decision = choose_sam2_storage(
        n_frames=100,
        image_size=1024,
        device="cuda",
        vram_budget_bytes=8 * 1024**3,
    )
    assert decision.offload_video_to_cpu is False
    assert decision.offload_state_to_cpu is True
    assert decision.storage_device == "cuda"


def test_choose_sam2_storage_offloads_long_clip_that_nearly_fills_the_card():
    # 1500 frames ≈ 17.6 GiB; hybrid needs +6 GiB working room → 22 GiB must full-offload.
    decision = choose_sam2_storage(
        n_frames=1500,
        image_size=1024,
        device="cuda",
        vram_budget_bytes=22 * 1024**3,
    )
    assert decision.offload_video_to_cpu is True
    assert decision.offload_state_to_cpu is True
    assert decision.storage_device == "cpu"


def test_choose_sam2_storage_hybrid_medium_clip_with_room_for_encode():
    # 800 frames ≈ 9.4 GiB; 16 GiB fits hybrid (+6) but not full (+8).
    decision = choose_sam2_storage(
        n_frames=800,
        image_size=1024,
        device="cuda",
        vram_budget_bytes=16 * 1024**3,
    )
    assert decision.offload_video_to_cpu is False
    assert decision.offload_state_to_cpu is True
    assert decision.storage_device == "cuda"


def test_choose_sam2_storage_offloads_when_budget_is_tight():
    # 1800 frames @1024 ≈ 21 GiB images alone; 10 GiB budget must offload.
    decision = choose_sam2_storage(
        n_frames=1800,
        image_size=1024,
        device="cuda",
        vram_budget_bytes=10 * 1024**3,
    )
    assert decision.offload_video_to_cpu is True
    assert decision.offload_state_to_cpu is True
    assert decision.storage_device == "cpu"


def test_choose_sam2_storage_cpu_device_always_offloads():
    decision = choose_sam2_storage(
        n_frames=10,
        image_size=1024,
        device="cpu",
        vram_budget_bytes=80 * 1024**3,
    )
    assert decision.offload_video_to_cpu is True
    assert decision.storage_device == "cpu"


def test_live_vram_budget_caps_configured_when_cuda_reports_less(monkeypatch):
    import videoclean.adapters.segmenters.sam2_video as mod

    class _Cuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def mem_get_info():
            # 4 GiB free → live budget 4 - 1.5 = 2.5 GiB
            return 4 * 1024**3, 24 * 1024**3

    class _Torch:
        cuda = _Cuda

    monkeypatch.setitem(__import__("sys").modules, "torch", _Torch)
    monkeypatch.setattr(mod, "RESERVE_BYTES", 1536 * 1024 * 1024)

    # Re-import helpers bind RESERVE at call time from mod — call through mod.
    capped = mod._live_vram_budget_bytes(20 * 1024**3)
    assert capped == 4 * 1024**3 - 1536 * 1024 * 1024


def test_is_cuda_oom_detects_runtime_message():
    assert _is_cuda_oom(RuntimeError("CUDA out of memory. Tried to allocate 24.00 MiB"))
    assert not _is_cuda_oom(RuntimeError("something else"))
