from pathlib import Path

from videoclean.adapters.detectors.owlvit import OwlVitDetector
from videoclean.adapters.hf_cache import hf_cached, hf_hub_cache_root, hf_hub_dir


def _clear_hf_env(monkeypatch) -> None:
    for name in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME", "XDG_CACHE_HOME"):
        monkeypatch.delenv(name, raising=False)


def _write_snapshot(root: Path, model_id: str) -> Path:
    snap = root / ("models--" + model_id.replace("/", "--")) / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}", encoding="utf-8")
    return snap


def test_hf_hub_dir_honors_hf_home(monkeypatch, tmp_path: Path):
    _clear_hf_env(monkeypatch)
    hf_home = tmp_path / "custom-hf-home"
    monkeypatch.setenv("HF_HOME", str(hf_home))
    model_id = "org/not-the-default-cache"
    assert hf_hub_cache_root() == hf_home / "hub"
    assert hf_hub_dir(model_id) == hf_home / "hub" / "models--org--not-the-default-cache"
    assert hf_hub_dir(model_id) != Path.home() / ".cache" / "huggingface" / "hub" / "models--org--not-the-default-cache"
    assert hf_cached(model_id) is False
    _write_snapshot(hf_home / "hub", model_id)
    assert hf_cached(model_id) is True


def test_hf_hub_cache_overrides_hf_home(monkeypatch, tmp_path: Path):
    _clear_hf_env(monkeypatch)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub-cache"))
    assert hf_hub_dir("a/b") == tmp_path / "hub-cache" / "models--a--b"
    assert hf_cached("a/b") is False
    _write_snapshot(tmp_path / "hub-cache", "a/b")
    assert hf_cached("a/b") is True


def test_owlvit_status_uses_hf_home(monkeypatch, tmp_path: Path):
    _clear_hf_env(monkeypatch)
    hf_home = tmp_path / "owl-hf"
    monkeypatch.setenv("HF_HOME", str(hf_home))
    model_id = "google/owlvit-base-patch32"
    missing = OwlVitDetector(model_id=model_id, device="cpu").status()
    assert "unavailable" in missing
    _write_snapshot(hf_home / "hub", model_id)
    ready = OwlVitDetector(model_id=model_id, device="cpu").status()
    assert "ready" in ready
    assert model_id in ready
