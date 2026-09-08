"""interpret_system.md is a builtin template with override support."""
import tempfile
from pathlib import Path

from videoclean.adapters.prompt.llm import interpret_system_prompt


def test_builtin_interpret_prompt():
    text = interpret_system_prompt(None)
    assert "JSON" in text
    assert "targets" in text


def test_custom_override():
    with tempfile.TemporaryDirectory() as d:
        custom = Path(d) / "interpret_system.md"
        custom.write_text("CUSTOM PROMPT", encoding="utf-8")
        assert interpret_system_prompt(d) == "CUSTOM PROMPT"


def test_missing_custom_falls_back():
    with tempfile.TemporaryDirectory() as d:
        assert "JSON" in interpret_system_prompt(d)
