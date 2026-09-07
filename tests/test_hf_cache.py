import builtins
import os
import sys
from pathlib import Path

from videoclean.adapters.detectors.owlvit import OwlVitDetector
from videoclean.adapters.hf_cache import (
    hf_cached,
    hf_hub_cache_root,
    hf_hub_dir,
    relax_hf_transfer_flag,
)


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


def test_relax_hf_transfer_flag_clears_env_without_package(monkeypatch):
    monkeypatch.setenv("HF_HUB_ENABLE_HF_TRANSFER", "1")
    monkeypatch.delitem(sys.modules, "hf_transfer", raising=False)
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "hf_transfer":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    relax_hf_transfer_flag()
    assert os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") == "0"


def test_hf_cached_false_when_blob_incomplete(monkeypatch, tmp_path: Path):
    _clear_hf_env(monkeypatch)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    model_id = "org/model"
    _write_snapshot(tmp_path / "hub", model_id)
    blobs = tmp_path / "hub" / "models--org--model" / "blobs"
    blobs.mkdir(parents=True)
    (blobs / "abc.incomplete").write_bytes(b"x")
    assert hf_cached(model_id) is False
