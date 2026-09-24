from videoclean.adapters.models.weights_cache import clear_cache, get_or_load


def test_second_get_or_load_does_not_call_loader():
    clear_cache()
    calls = {"n": 0}

    def loader():
        calls["n"] += 1
        return object()

    get_or_load(("lama", "big-lama", "cpu", "fp32"), loader)
    get_or_load(("lama", "big-lama", "cpu", "fp32"), loader)
    assert calls["n"] == 1
