from scripts.check_sla_report import check


def sla_report(**kw):
    wall = kw.get("wall_s", 90)
    budget_mb = kw["vram_budget_mb"]
    peak = kw["vram_peak_inpaint_mb"]
    limited = kw.get("limited_by", "ceiling")
    batch = kw.get("batch", 8)
    frames = kw.get("frame_count", 1800 if limited != "frames" else batch)
    return {
        "startedAt": 0,
        "finishedAt": wall,
        "budget": {
            "warm": kw.get("warm", True),
            "device": kw.get("device", "cuda"),
            "vramCeilingMb": None,
            "inpaintMaxSide": kw.get("inpaint_max_side"),
            "cpuThreads": 8,
            "cpuCount": 8,
            "vramBudgetMb": budget_mb,
            "vramPeakInpaintMb": peak,
        },
        "hole": {
            "policy": kw.get("policy", "lama-crop"),
            "limitedBy": limited,
            "batch": batch,
            "frameCount": frames,
        },
        "timings": {
            "load": 1,
            "decode": 4,
            "parse": 3,
            "detect": 6,
            "track": 2,
            "segment": 30,
            "inpaint": 25,
            "verify": 6,
            "encode": 7,
            "package": 1,
        },
    }


def test_checker_rejects_weight_peak_as_if_it_were_activations():
    report = sla_report(vram_budget_mb=20480, vram_peak_inpaint_mb=4000, limited_by="ceiling")
    assert check(report, max_wall_s=120, require_warm=True, require_cuda=True) != 0


def test_checker_accepts_activation_delta_inside_the_band():
    report = sla_report(
        vram_budget_mb=20480,
        vram_peak_inpaint_mb=14000,
        limited_by="ceiling",
        warm=True,
        device="cuda",
        policy="lama-crop",
        wall_s=90,
    )
    assert check(report, max_wall_s=120, require_warm=True, require_cuda=True) == 0
