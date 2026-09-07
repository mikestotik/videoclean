from videoclean.adapters.detectors.owlvit import fit_owlvit_queries


def test_fit_keeps_short_phrases_and_splits_overlong_instead_of_truncating():
    lengths = {
        "вотермарку и текстовые overlay": 25,
        "вотермарку": 12,
        "текстовые overlay": 13,
        "ИЗНАНКА": 9,
        "watermark": 4,
    }

    out = fit_owlvit_queries(
        ["вотермарку и текстовые overlay", "ИЗНАНКА", "watermark"],
        token_len=lambda s: lengths.get(s, 5),
    )
    assert "ИЗНАНКА" in out
    assert "watermark" in out
    assert "вотермарку" in out
    assert "текстовые overlay" in out
    assert "вотермарку и текстовые overlay" not in out


def test_fit_drops_queries_that_do_not_fit_the_encoder():
    out = fit_owlvit_queries(["x" * 500], token_len=lambda s: 99)
    assert out == []
