from videoclean.adapters.segmenters.sam2_video import choose_sam2_storage


def test_choose_sam2_storage_keeps_frames_on_cuda_when_budget_fits():
    # 100 frames @1024 float32 ≈ 1.17 GiB; 8 GiB budget with 3 GiB headroom fits.
    decision = choose_sam2_storage(
        n_frames=100,
        image_size=1024,
        device="cuda",
        vram_budget_bytes=8 * 1024**3,
    )
    assert decision.offload_video_to_cpu is False
    assert decision.offload_state_to_cpu is False
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
