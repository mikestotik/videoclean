from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "videoclean"
SERVER = Path(__file__).resolve().parents[1] / "server"


def _py_files(folder: str) -> list[Path]:
    return list((ROOT / folder).rglob("*.py"))


def _py_files_in(root: Path) -> list[Path]:
    return list(root.rglob("*.py"))


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


def test_server_does_not_import_webui():
    for path in _py_files_in(SERVER):
        assert "webui" not in path.read_text(encoding="utf-8"), f"{path} imports webui"


def test_library_does_not_import_server():
    banned = ("from server", "import server")
    for path in _py_files("domain") + _py_files("application") + _py_files_in(ROOT / "adapters"):
        text = path.read_text(encoding="utf-8")
        for name in banned:
            assert name not in text, f"{path} imports {name}"
