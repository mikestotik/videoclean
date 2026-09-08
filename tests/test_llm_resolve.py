from videoclean.adapters.llm.resolve import resolve_llm
from videoclean.application.config import PipelineConfig


def test_explicit_cloud_uses_openai_when_key_present(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "oai-test")
    monkeypatch.delenv("VIDEOCLEAN_LLM_BASE_URL", raising=False)
    cfg = PipelineConfig(llm_place="cloud")
    client = resolve_llm(cfg)
    assert client.status().startswith("ready")
    assert "cloud" in client.status()
    assert "api.openai.com" in client.status()
    assert client.model == "gpt-4o-mini"


def test_explicit_local_defaults_to_ollama(monkeypatch):
    monkeypatch.delenv("VIDEOCLEAN_LLM_BASE_URL", raising=False)
    cfg = PipelineConfig(llm_place="local", llm_model="phi4")
    client = resolve_llm(cfg)
    monkeypatch.setattr(client, "_ping", lambda: (True, ""))
    assert "local" in client.status()
    assert "11434" in client.status()
    assert client.model == "phi4"


def test_local_status_is_unavailable_when_ollama_is_down(monkeypatch):
    cfg = PipelineConfig(llm_place="local", llm_model="llama3.2")
    client = resolve_llm(cfg)
    monkeypatch.setattr(client, "_ping", lambda: (False, "llm unreachable at http://127.0.0.1:11434/v1/models: Connection refused"))
    st = client.status()
    assert st.startswith("unavailable")
    assert "11434" in st


def test_local_gguf_path(tmp_path, monkeypatch):
    weights = tmp_path / "model.gguf"
    weights.write_bytes(b"GGUF")
    cfg = PipelineConfig(llm_place="local", llm_model=str(weights))
    client = resolve_llm(cfg)
    st = client.status()
    assert "gguf" in st.lower() or "llama.cpp" in st.lower() or "unavailable" in st


def test_auto_prefers_local_url_over_cloud_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "oai-test")
    cfg = PipelineConfig(
        llm_place="auto",
        llm_base_url="http://127.0.0.1:8080/v1",
        llm_model="local-model",
    )
    client = resolve_llm(cfg)
    monkeypatch.setattr(client, "_ping", lambda: (True, ""))
    assert "local" in client.status()
    assert "8080" in client.status()
