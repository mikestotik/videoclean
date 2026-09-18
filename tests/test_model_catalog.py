from pathlib import Path

import pytest

from videoclean.adapters.models.catalog import ModelCatalog, COMPONENT_IDS
from videoclean.application.errors import DownloadCancelled, PipelineError
from videoclean.store import JobIndex


@pytest.fixture(autouse=True)
def _mock_ollama_tags(monkeypatch, request):
    if request.node.name.startswith("test_ollama_tags_"):
        return
    monkeypatch.setattr("videoclean.adapters.models.catalog.ollama_tags", lambda: None)


def test_registry_ids():
    assert "detector:grounding-dino" in COMPONENT_IDS
    assert "inpainter:propainter" in COMPONENT_IDS


def test_all_registry_ids():
    expected = (
        "detector:grounding-dino",
        "segmenter:sam2-tiny",
        "segmenter:sam2-small",
        "segmenter:sam2-base-plus",
        "segmenter:sam2-large",
        "segmenter:sam21-tiny",
        "segmenter:sam21-small",
        "segmenter:sam21-base-plus",
        "segmenter:sam21-large",
        "inpainter:lama",
        "inpainter:propainter",
        "llm:ollama-llama3.2",
        "llm:ollama-llava-phi3",
    )
    for cid in expected:
        assert cid in COMPONENT_IDS
    assert len(COMPONENT_IDS) == 13


def test_list_status_smoke(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    cat = ModelCatalog()
    rows = cat.list_status()
    assert len(rows) >= 6
    assert all(r.state in {"ready", "missing", "downloading", "error"} for r in rows)


def test_component_id_mapping():
    from videoclean.adapters.models.catalog import component_id_for

    assert component_id_for("detector", "grounding-dino") == "detector:grounding-dino"
    assert component_id_for("detector", "owlvit") is None
    assert component_id_for("segmenter", "sam2") == "segmenter:sam2-tiny"
    assert component_id_for("segmenter", "sam2-video") == "segmenter:sam2-tiny"
    assert (
        component_id_for("segmenter", "sam2", segmenter_model="facebook/sam2-hiera-large")
        == "segmenter:sam2-large"
    )
    assert (
        component_id_for("segmenter", "sam2", segmenter_model="facebook/sam2-hiera-small")
        == "segmenter:sam2-small"
    )
    assert (
        component_id_for("segmenter", "sam2-video", segmenter_model="facebook/sam2.1-hiera-base-plus")
        == "segmenter:sam21-base-plus"
    )
    assert (
        component_id_for("segmenter", "sam2", segmenter_model="facebook/sam2.1-hiera-tiny")
        == "segmenter:sam21-tiny"
    )
    assert component_id_for("inpainter", "propainter") == "inpainter:propainter"
    assert component_id_for("inpainter", "lama") == "inpainter:lama"
    assert component_id_for("inpainter", "opencv-telea") is None


def test_backend_ready_uses_catalog():
    from videoclean.adapters.models.catalog import backend_ready

    class FakeCat:
        def is_ready(self, cid: str) -> bool:
            return cid in {"detector:grounding-dino", "inpainter:propainter", "inpainter:lama"}

    fake = FakeCat()
    assert backend_ready("detector", "grounding-dino", catalog=fake) is True
    assert backend_ready("detector", "unknown-detector", catalog=fake) is False
    assert backend_ready("inpainter", "propainter", catalog=fake) is True
    assert backend_ready("inpainter", "lama", catalog=fake) is True
    assert backend_ready("inpainter", "opencv-telea", catalog=fake) is False
    assert backend_ready("segmenter", "sam2-video", catalog=fake) is False


def test_list_status_downloading(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("HOME", str(tmp_path))
    jobs = JobIndex(tmp_path / "jobs.sqlite")
    jobs.upsert_download("d1", "detector:grounding-dino", "running", progress=0.3, message="fetch")
    cat = ModelCatalog(jobs=jobs)
    by_id = {r.info.id: r for r in cat.list_status()}
    assert by_id["detector:grounding-dino"].state == "downloading"
    assert "fetch" in by_id["detector:grounding-dino"].message


def test_ollama_tags_caches_negative_and_uses_short_timeout(monkeypatch):
    from videoclean.adapters.models import catalog as cat

    cat._ollama_neg_until = 0.0
    calls: list[float | None] = []

    def boom(req, timeout=None):
        calls.append(timeout)
        raise TimeoutError("down")

    monkeypatch.setattr(cat.urllib.request, "urlopen", boom)
    assert cat.ollama_tags() is None
    assert cat.ollama_tags() is None
    assert calls == [cat.OLLAMA_TAGS_TIMEOUT_S]
    assert cat.OLLAMA_TAGS_TIMEOUT_S <= 0.2
    assert cat._ollama_neg_until > 0
    cat._ollama_neg_until = 0.0
    assert cat.ollama_tags() is None
    assert len(calls) == 2


def test_list_status_caches_ollama_tags(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("HOME", str(tmp_path))
    calls = {"n": 0}

    def fake_tags():
        calls["n"] += 1
        return None

    monkeypatch.setattr("videoclean.adapters.models.catalog.ollama_tags", fake_tags)
    rows = ModelCatalog().list_status()
    assert calls["n"] == 1
    assert rows
    llm = {r.info.id: r for r in rows}
    assert llm["llm:ollama-llama3.2"].state == "error"
    assert "start ollama" in llm["llm:ollama-llama3.2"].message


def test_download_component_records_progress(tmp_path: Path):
    from videoclean.application.use_cases.download_component import DownloadComponent

    jobs = JobIndex(tmp_path / "j.sqlite")
    cb: list[tuple[float, str]] = []

    def runner(component_id, on_progress, is_cancelled):
        assert component_id == "detector:grounding-dino"
        on_progress(0.4, "fetch")
        assert is_cancelled() is False

    DownloadComponent(runner).execute(
        "detector:grounding-dino", jobs, lambda f, m="": cb.append((f, m))
    )
    row = jobs.list_downloads()[0]
    assert row["state"] == "done"
    assert row["progress"] == 1.0
    assert row["component_id"] == "detector:grounding-dino"
    assert cb[0] == (0.4, "fetch")


def test_download_component_cancel(tmp_path: Path):
    from videoclean.application.use_cases.download_component import DownloadComponent

    jobs = JobIndex(tmp_path / "j.sqlite")

    def runner(component_id, on_progress, is_cancelled):
        on_progress(0.1, "start")
        jobs.request_download_cancel(jobs.list_downloads()[0]["id"])
        assert is_cancelled() is True
        on_progress(0.2, "more")

    with pytest.raises(DownloadCancelled):
        DownloadComponent(runner).execute("detector:grounding-dino", jobs, None)
    assert jobs.list_downloads()[0]["state"] == "cancelled"


def test_run_download_unknown():
    from videoclean.adapters.models.downloaders import run_download

    with pytest.raises(PipelineError, match="unknown component"):
        run_download("nope")


def test_run_download_dispatches_hf(monkeypatch):
    from videoclean.adapters.models import downloaders as d

    seen: list[str] = []
    monkeypatch.setattr(d, "download_hf", lambda repo_id, **kw: seen.append(repo_id))
    d.run_download("detector:grounding-dino")
    assert seen == ["IDEA-Research/grounding-dino-tiny"]
    seen.clear()
    d.run_download("segmenter:sam2-tiny")
    assert seen == ["facebook/sam2-hiera-tiny"]


def test_run_download_dispatches_others(monkeypatch):
    from videoclean.adapters.models import downloaders as d

    calls: list[str] = []
    monkeypatch.setattr(d, "download_lama", lambda **kw: calls.append("lama"))
    monkeypatch.setattr(d, "download_propainter", lambda **kw: calls.append("propainter"))
    monkeypatch.setattr(d, "download_ollama", lambda tag, **kw: calls.append(tag))
    d.run_download("inpainter:lama")
    d.run_download("inpainter:propainter")
    d.run_download("llm:ollama-llama3.2")
    d.run_download("llm:ollama-llava-phi3")
    assert calls == ["lama", "propainter", "llama3.2", "llava-phi3"]


def test_download_hf_snapshot_and_cancel(monkeypatch):
    from videoclean.adapters.models import downloaders as d

    seen: dict = {}

    def fake_snap(repo_id, **kwargs):
        seen["repo_id"] = repo_id
        seen["local_files_only"] = kwargs.get("local_files_only")
        seen["tqdm_class"] = kwargs.get("tqdm_class")

    monkeypatch.setattr(d, "_snapshot_download", fake_snap)
    d.download_hf("IDEA-Research/grounding-dino-tiny", is_cancelled=lambda: False)
    assert seen["repo_id"] == "IDEA-Research/grounding-dino-tiny"
    assert seen["local_files_only"] is False
    assert seen["tqdm_class"] is not None

    def boom(*args, **kwargs):
        raise AssertionError("should not download when cancelled")

    monkeypatch.setattr(d, "_snapshot_download", boom)
    with pytest.raises(DownloadCancelled):
        d.download_hf("x", is_cancelled=lambda: True)


def test_download_lama_writes_file(monkeypatch, tmp_path: Path):
    from videoclean.adapters.models import downloaders as d

    dest = tmp_path / "hub" / "checkpoints" / "big-lama.pt"
    monkeypatch.setattr(d, "_default_model_path", lambda: dest)

    def fake_retrieve(url, path, reporthook):
        assert "big-lama.pt" in url
        path.write_bytes(b"weights")
        reporthook(1, 7, 7)

    monkeypatch.setattr(d, "_urlretrieve", fake_retrieve)
    d.download_lama(is_cancelled=lambda: False)
    assert dest.is_file()
    assert dest.read_bytes() == b"weights"


def test_download_ollama_streams_progress(monkeypatch):
    from videoclean.adapters.models import downloaders as d

    class FakeResp:
        def __init__(self):
            self._lines = [
                b'{"status":"pulling","completed":10,"total":100}\n',
                b'{"status":"success"}\n',
            ]
            self._i = 0

        def readline(self):
            if self._i >= len(self._lines):
                return b""
            line = self._lines[self._i]
            self._i += 1
            return line

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(d, "_http_open", lambda req, timeout=None: FakeResp())
    ticks: list[tuple[float, str]] = []
    d.download_ollama(
        "llama3.2",
        on_progress=lambda f, m="", **k: ticks.append((f, m)),
        is_cancelled=lambda: False,
    )
    assert ticks[0][0] == pytest.approx(0.1)
    assert ticks[-1][0] == 1.0


def test_download_propainter_honors_data_dir(monkeypatch, tmp_path: Path):
    from videoclean.adapters.models import downloaders as d

    data = tmp_path / "custom-data"
    monkeypatch.setenv("VIDEOCLEAN_DATA_DIR", str(data))
    monkeypatch.delenv("VIDEOCLEAN_PROPAINTER_ROOT", raising=False)
    monkeypatch.delenv("VIDEOCLEAN_PROPAINTER_WEIGHTS", raising=False)
    monkeypatch.setattr(d, "find_vendor", lambda: None)
    cloned: list[Path] = []
    hf: list[str] = []
    monkeypatch.setattr(d, "_git_clone", lambda url, dest, **kw: cloned.append(dest))
    monkeypatch.setattr(
        d, "download_hf", lambda repo_id, **kw: hf.append(str(kw.get("local_dir")))
    )
    d.download_propainter(is_cancelled=lambda: False)
    assert cloned[0] == data / "vendor" / "ProPainter"
    assert Path(hf[0]) == data / "weights" / "propainter"


def test_download_lama_honors_data_dir(monkeypatch, tmp_path: Path):
    from videoclean.adapters.models import downloaders as d

    data = tmp_path / "custom-data"
    dest = data / "weights" / "lama" / "big-lama.pt"
    monkeypatch.setenv("VIDEOCLEAN_DATA_DIR", str(data))
    monkeypatch.delenv("LAMA_MODEL", raising=False)

    def fake_retrieve(url, path, reporthook):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"weights")
        reporthook(1, 7, 7)

    monkeypatch.setattr(d, "_urlretrieve", fake_retrieve)
    d.download_lama(is_cancelled=lambda: False)
    assert dest.is_file()
    assert dest.read_bytes() == b"weights"


def test_download_propainter_clones_then_weights(monkeypatch, tmp_path: Path):
    from videoclean.adapters.models import downloaders as d

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(d, "find_vendor", lambda: None)
    cloned: list[tuple[str, Path]] = []
    hf: list[str] = []
    monkeypatch.setattr(
        d, "_git_clone", lambda url, dest, **kw: cloned.append((url, dest))
    )
    monkeypatch.setattr(
        d, "download_hf", lambda repo_id, **kw: hf.append(repo_id + "|" + str(kw.get("local_dir")))
    )
    d.download_propainter(is_cancelled=lambda: False)
    assert cloned[0][0] == "https://github.com/sczhou/ProPainter.git"
    assert cloned[0][1] == tmp_path / ".videoclean" / "vendor" / "ProPainter"
    assert hf[0].startswith("camenduru/ProPainter|")
    assert str(tmp_path / ".videoclean" / "weights" / "propainter") in hf[0]


def test_progress_tqdm_matches_thread_map_api():
    import io

    from tqdm.contrib.concurrent import thread_map

    from videoclean.adapters.models.downloaders import _make_tqdm

    ticks: list[float] = []
    cls = _make_tqdm(lambda f, m="", **k: ticks.append(f), is_cancelled=lambda: False)
    lock = cls.get_lock()
    cls.set_lock(lock)
    items = ["a", "b", "c"]
    sink = io.StringIO()
    out = list(cls(items, desc="files", file=sink, miniters=1, mininterval=0, delay=0))
    assert out == items
    assert ticks

    ticks.clear()
    doubled = thread_map(
        lambda x: x + x,
        items,
        max_workers=2,
        tqdm_class=cls,
        file=sink,
        miniters=1,
        mininterval=0,
    )
    assert doubled == ["aa", "bb", "cc"]
    assert ticks

    cancelled = _make_tqdm(None, is_cancelled=lambda: True)
    with pytest.raises(DownloadCancelled):
        with cancelled(total=1, file=sink) as bar:
            bar.update()


def test_git_clone_honours_cancel(monkeypatch, tmp_path: Path):
    import subprocess

    from videoclean.adapters.models import downloaders as d

    class FakeProc:
        def __init__(self):
            self.returncode = None
            self.terminated = False

        def poll(self):
            return 0 if self.terminated else None

        def wait(self, timeout=None):
            if not self.terminated:
                raise subprocess.TimeoutExpired(cmd="git", timeout=timeout)
            return self.returncode

        def terminate(self):
            self.terminated = True
            self.returncode = -15

        def kill(self):
            self.terminated = True
            self.returncode = -9

    proc = FakeProc()
    seen: dict = {}

    def fake_popen(*args, **kwargs):
        seen.update(kwargs)
        return proc

    monkeypatch.setattr(d.subprocess, "Popen", fake_popen)
    dest = tmp_path / "ProPainter"
    dest.mkdir()
    with pytest.raises(DownloadCancelled):
        d._git_clone("https://github.com/sczhou/ProPainter.git", dest, is_cancelled=lambda: True)
    assert proc.terminated
    assert not dest.exists()
    assert seen["stdout"] is subprocess.DEVNULL
    assert seen["stderr"] is subprocess.DEVNULL
