from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "videoclean"


def _py_files(folder: str) -> list[Path]:
    return list((ROOT / folder).rglob("*.py"))


def test_domain_does_not_import_outer_circles():
    banned = ("videoclean.adapters", "videoclean.application", "videoclean.cli", "videoclean.composition")
    for path in _py_files("domain"):
        text = path.read_text(encoding="utf-8")
        for name in banned:
            assert name not in text, f"{path} imports {name}"


def test_application_does_not_import_adapters_or_cli():
    banned = ("videoclean.adapters", "videoclean.cli", "videoclean.composition")
    for path in _py_files("application"):
        text = path.read_text(encoding="utf-8")
        for name in banned:
            assert name not in text, f"{path} imports {name}"
