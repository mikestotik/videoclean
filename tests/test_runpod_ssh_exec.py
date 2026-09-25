import importlib.util
from pathlib import Path


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "runpod_ssh_exec.py"
    spec = importlib.util.spec_from_file_location("runpod_ssh_exec", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_detect_container_not_found():
    mod = _load()
    msg = mod.detect_ssh_proxy_failure("Warning: Permanently added...\ncontainer not found\n")
    assert msg == "container not found"


def test_detect_none_on_normal_banner():
    mod = _load()
    assert mod.detect_ssh_proxy_failure("Welcome to Ubuntu\nroot@pod:~# ") is None


def test_proxy_user_candidates_prefer_live_over_stale_suffix(monkeypatch):
    mod = _load()
    monkeypatch.setenv("RUNPOD_SSH_PROXY_SUFFIX", "deadbeef")
    monkeypatch.delenv("RUNPOD_SSH_PROXY_USER", raising=False)
    monkeypatch.setattr(mod, "fetch_proxy_user_runpodctl", lambda pod_id: f"{pod_id}-liveuser1")
    monkeypatch.setattr(mod, "fetch_proxy_user_from_api", lambda pod_id: "")
    users = mod.proxy_user_candidates("cpy65b5zj24xab")
    assert users[0] == "cpy65b5zj24xab-liveuser1"
    assert "cpy65b5zj24xab-deadbeef" in users
