from videoclean.application.budget import RESERVE_BYTES, resolve_budget


class _Cfg:
    def __init__(self, **kw):
        self.device = kw.get("device", "cuda")
        self.device_requested = kw.get("device_requested", "auto")
        self.max_vram_mb = kw.get("max_vram_mb")
        self.cpu_threads = kw.get("cpu_threads")
        self.inpaint_max_side = kw.get("inpaint_max_side")
        self.inpaint_workers = kw.get("inpaint_workers", 0)


class _Probe:
    def __init__(self, free, total=24 * 1024**3, cpus=8):
        self._free = free
        self._total = total
        self._cpus = cpus

    def cuda_mem(self):
        return self._free, self._total

    def unified_total(self):
        return self._total

    def cpu_count(self):
        return self._cpus

    def nvenc(self):
        return False


def test_empty_ceiling_is_free_minus_reserve():
    free = 20 * 1024**3
    budget = resolve_budget(_Cfg(max_vram_mb=None), probe=_Probe(free))
    assert budget.vram_budget_bytes == free - RESERVE_BYTES
    assert budget.vram_ceiling_bytes is None
    assert budget.inpaint_workers_cap is None
