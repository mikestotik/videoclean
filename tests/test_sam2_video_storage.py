from videoclean.adapters.segmenters.sam2_video import choose_sam2_storage


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
    # Same 100 frames; 8 GiB is enough for video+2 GiB but not full 8 GiB state headroom.
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
    # 1500 frames ≈ 17.6 GiB; +3 GiB old headroom falsely chose full GPU and OOMed.
    decision = choose_sam2_storage(
        n_frames=1500,
        image_size=1024,
        device="cuda",
        vram_budget_bytes=22 * 1024**3,
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
