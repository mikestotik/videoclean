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


def test_intent_from_llm_json_grounds_user_prompt():
    intent = intent_from_llm_json(
        '{"targets":[{"kind":"object","query":"wrong","where":"bottom-right","ordinal":2,"from_side":"right"}]}',
        raw="удали текстовую надпись сверху",
    )
    t = intent.targets[0]
    assert t.kind == "text_overlay"
    assert t.query == "text"
    assert t.where == "top"
    assert t.ordinal is None


def test_refine_keeps_named_caption_query():
    intent = Intent(
        targets=[Target(kind="text_overlay", query="news title", where=None)],
        raw="убери заголовок",
    )
    out = refine_intent(intent)
    assert out.targets[0].query == "news title"
