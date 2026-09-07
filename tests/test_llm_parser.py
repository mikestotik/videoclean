from videoclean.adapters.prompt.llm import LlmPromptParser, intent_from_llm_json
from videoclean.application.errors import AdapterUnavailable, PipelineError


class FakeLlm:
    name = "fake"

    def __init__(self, reply: str | None = None, status: str = "ready (fake)", replies: list[str] | None = None):
        self._replies = list(replies) if replies is not None else [reply or ""]
        self._status = status
        self.calls: list[tuple[str, str, int]] = []

    def status(self) -> str:
        return self._status

    def complete(
        self,
        system: str,
        user: str,
        image_jpeg: bytes | None = None,
        images: list[bytes] | None = None,
    ) -> str:
        n = len(images or []) + (1 if image_jpeg else 0)
        self.calls.append((system, user, n))
        if len(self._replies) == 1:
            return self._replies[0]
        return self._replies.pop(0)


def test_intent_from_json_object_is_a_mug_not_text():
    intent = intent_from_llm_json(
        '{"targets":[{"kind":"object","query":"coffee mug","part":null,"motion":"any"}]}',
        raw="убери кружку на столе",
    )
    assert len(intent.targets) == 1
    t = intent.targets[0]
    assert t.kind == "object"
    assert t.query == "coffee mug"
    assert intent.queries == ["coffee mug"]
    assert intent.raw == "убери кружку на столе"


def test_queries_reject_cyrillic_and_keep_english():
    try:
        intent_from_llm_json(
            '{"targets":[{"kind":"text_overlay","query":"надписи","motion":"any"}]}',
            raw="убери надписи",
        )
    except PipelineError as exc:
        assert "no usable targets" in str(exc)
    else:
        raise AssertionError("expected Cyrillic query to be rejected")

    intent = intent_from_llm_json(
        '{"targets":[{"kind":"text_overlay","query":"news title","motion":"any"}]}',
        raw="убери надписи",
    )
    assert intent.queries == ["news title"]


def test_intent_from_json_strips_markdown_fence():
    text = """here
```json
{"targets":[{"kind":"watermark","query":"channel logo","part":"icon","motion":"static"}]}
```
"""
    intent = intent_from_llm_json(text, raw="logo")
    assert intent.targets[0].kind == "watermark"
    assert intent.targets[0].part == "icon"
    assert intent.targets[0].motion == "static"


def test_intent_from_json_rejects_garbage():
    try:
        intent_from_llm_json("sorry I cannot", raw="x")
    except PipelineError as exc:
        assert "JSON" in str(exc) or "json" in str(exc)
    else:
        raise AssertionError("expected PipelineError")


def test_llm_parser_empty_prompt_is_an_error():
    llm = FakeLlm(reply="should not be used")
    parser = LlmPromptParser(llm)
    try:
        parser.parse("")
    except PipelineError as exc:
        assert "--prompt" in str(exc)
    else:
        raise AssertionError("expected PipelineError")
    assert llm.calls == []


def test_llm_parser_uses_model_for_long_prompt():
    llm = FakeLlm(
        reply='{"targets":[{"kind":"object","query":"mug","motion":"any"},'
        '{"kind":"watermark","query":"telegram logo","motion":"static"}]}'
    )
    parser = LlmPromptParser(llm)
    intent = parser.parse("убери кружку и логотип телеграма, прицел не трогай")
    assert [t.kind for t in intent.targets] == ["object", "watermark"]
    assert "mug" in intent.queries
    assert "telegram logo" in intent.queries
    assert len(llm.calls) == 1


def test_llm_does_not_invent_targets_from_the_sentence():
    llm = FakeLlm(
        reply='{"targets":[{"kind":"text_overlay","query":"right caption","part":null,"motion":"any"}]}'
    )
    parser = LlmPromptParser(llm)
    intent = parser.parse(
        "убери надпись внизу экрана с логотипом и плавающий текст в правом верхнем углу"
    )
    assert len(intent.targets) == 1
    assert intent.targets[0].query == "right caption"


def test_llm_reads_ordinal_not_as_a_box():
    intent = intent_from_llm_json(
        '{"targets":[{"kind":"object","query":"red mug","ordinal":3,"from_side":"left","motion":"any"}]}',
        raw="убери третью слева красную кружку",
    )
    assert intent.targets[0].kind == "object"
    assert intent.targets[0].ordinal == 3
    assert intent.targets[0].from_side == "left"


def test_llm_parser_vision_sends_frames_and_sets_mode():
    import numpy as np
    from videoclean.adapters.prompt.llm import VISION_SYSTEM

    llm = FakeLlm(reply='{"targets":[{"kind":"object","query":"mug","motion":"any"}]}')
    parser = LlmPromptParser(llm)
    frames = [np.zeros((32, 32, 3), dtype=np.uint8), np.ones((32, 32, 3), dtype=np.uint8) * 40]
    intent = parser.parse("убери кружку", frames=frames)
    assert intent.parse_mode == "llm-vision"
    assert llm.calls[0][0] == VISION_SYSTEM
    assert llm.calls[0][2] == 2
    assert intent.queries == ["mug"]


def test_llm_parser_vision_repair_recovers_json():
    import numpy as np

    llm = FakeLlm(
        replies=[
            "not json at all",
            '{"targets":[{"kind":"object","query":"mug","motion":"any"}]}',
        ]
    )
    parser = LlmPromptParser(llm)
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    intent = parser.parse("убери кружку", frames=[frame])
    assert intent.parse_mode == "llm-vision"
    assert intent.queries == ["mug"]
    assert llm.calls[0][2] == 1
    assert llm.calls[1][2] == 1


def test_llm_parser_bridges_vision_prose_to_text_json():
    import numpy as np

    llm = FakeLlm(
        replies=[
            "I see a ceramic mug on the table near the laptop.",
            "still not json",
            '{"targets":[{"kind":"object","query":"mug","motion":"any"}]}',
        ]
    )
    parser = LlmPromptParser(llm)
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    intent = parser.parse("убери кружку", frames=[frame])
    assert intent.parse_mode == "llm-vision-bridged"
    assert intent.queries == ["mug"]
    assert llm.calls[2][2] == 0


def test_llm_parser_text_only_when_no_frames():
    from videoclean.adapters.prompt.llm import SYSTEM

    llm = FakeLlm(reply='{"targets":[{"kind":"object","query":"mug","motion":"any"}]}')
    parser = LlmPromptParser(llm)
    intent = parser.parse("убери кружку")
    assert intent.parse_mode == "llm"
    assert llm.calls[0][0] == SYSTEM
    assert llm.calls[0][2] == 0


def test_llm_parser_unavailable_raises():
    llm = FakeLlm(reply="{}", status="unavailable: no key")
    parser = LlmPromptParser(llm)
    try:
        parser.parse("убери кружку")
    except AdapterUnavailable as exc:
        assert "no key" in str(exc)
    else:
        raise AssertionError("expected AdapterUnavailable")


def test_system_prompt_does_not_hand_canned_overlay_lists():
    from videoclean.adapters.prompt.llm import BRIDGE_SYSTEM, SYSTEM, VISION_SYSTEM

    for blob in (SYSTEM, VISION_SYSTEM, BRIDGE_SYSTEM):
        assert '"query":"on-screen text"' not in blob
        assert "убери надписи" not in blob
        # No few-shot that teaches a fixed title/caption/side/watermark pack.
        assert '"query":"news title"' not in blob
        assert '"query":"caption banner"' not in blob
        assert '"query":"side label"' not in blob
