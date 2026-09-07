from videoclean.adapters.prompt.llm import intent_from_llm_json
from videoclean.adapters.prompt.refine import refine_intent, where_from_prompt
from videoclean.domain.intent import Intent, Target


def test_where_from_russian_top():
    assert where_from_prompt("удали текстовую надпись сверху") == "top"
    assert where_from_prompt("убери логотип в правом нижнем углу") == "bottom-right"


def test_refine_rewrites_ocr_query_and_user_where():
    intent = Intent(
        targets=[
            Target(kind="text_overlay", query="wrong", where="bottom-right", ordinal=2, from_side="right"),
            Target(kind="text_overlay", query="side", where="bottom-right", ordinal=3, from_side="right"),
        ],
        raw="удали текстовую надпись сверху",
    )
    out = refine_intent(intent)
    assert [t.query for t in out.targets] == ["text"]
    assert out.targets[0].where == "top"
    assert out.targets[0].ordinal is None


def test_intent_from_llm_json_trusts_object_kind_and_query():
    """New policy: VLM's object target is never silently rewritten. Junk queries now
    fail loudly at the detector stage (explainable) instead of silently removing text."""
    intent = intent_from_llm_json(
        '{"targets":[{"kind":"object","query":"wrong","where":"bottom-right","ordinal":2,"from_side":"right"}]}',
        raw="удали текстовую надпись сверху",
    )
    t = intent.targets[0]
    assert t.kind == "object"
    assert t.query == "wrong"
    assert t.where == "top"
    assert t.ordinal is None


def test_refine_keeps_named_caption_query():
    intent = Intent(
        targets=[Target(kind="text_overlay", query="news title", where=None)],
        raw="убери заголовок",
    )
    out = refine_intent(intent)
    assert out.targets[0].query == "news title"


def test_object_target_is_never_silently_rewritten_to_text():
    """Public service: any named thing must stay searchable. No blacklist games."""
    for query in ("ball", "clock", "cage", "sticker", "drone", "sign", "banner"):
        intent = Intent(
            targets=[Target(kind="object", query=query)],
            raw="удали надпись и мяч",
        )
        out = refine_intent(intent)
        assert out.targets[0].kind == "object", query
        assert out.targets[0].query == query, query


def test_object_becomes_text_only_when_word_is_explicitly_overlay():
    for query in ("writing", "on-screen caption", "channel watermark"):
        intent = Intent(
            targets=[Target(kind="object", query=query)],
            raw="удали надпись",
        )
        out = refine_intent(intent)
        assert out.targets[0].kind == "text_overlay", query


def test_multiword_object_with_noun_stays_object():
    intent = Intent(
        targets=[Target(kind="object", query="parked car")],
        raw="удали надпись и машину",
    )
    out = refine_intent(intent)
    assert out.targets[0].kind == "object"
    assert out.targets[0].query == "parked car"
