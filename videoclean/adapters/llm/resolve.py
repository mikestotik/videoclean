from __future__ import annotations

import os
from pathlib import Path

from videoclean.adapters.llm.llama_cpp import LlamaCppLlm
from videoclean.adapters.llm.openai_compat import OpenAiCompatLlm
from videoclean.application.config import PipelineConfig

DEFAULT_LOCAL_URL = "http://127.0.0.1:11434/v1"
DEFAULT_LOCAL_MODEL = "llama3.2"
DEFAULT_OPENAI_URL = "https://api.openai.com/v1"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


class UnconfiguredLlm:
    name = "unconfigured"
    model = ""

    def __init__(self, reason: str):
        self._reason = reason

    def status(self) -> str:
        return f"unavailable: {self._reason}"

    def complete(
        self,
        system: str,
        user: str,
        image_jpeg: bytes | None = None,
        images: list[bytes] | None = None,
    ) -> str:
        from videoclean.application.errors import AdapterUnavailable

        raise AdapterUnavailable(self.status())


def is_local_url(url: str) -> bool:
    u = (url or "").lower()
    return any(h in u for h in ("127.0.0.1", "localhost", "0.0.0.0", "[::1]"))


def _looks_gguf(model: str) -> bool:
    if not model:
        return False
    p = Path(model).expanduser()
    return model.lower().endswith(".gguf") or p.suffix.lower() == ".gguf"


def _provider_credentials(model: str, base: str) -> tuple[str, str]:
    """Fill base_url / api_key from Settings → OpenAI-compatible providers when matched."""
    try:
        from videoclean.adapters.models.catalog import load_providers
    except Exception:  # noqa: BLE001
        return base, ""
    model = (model or "").strip()
    base_norm = (base or "").strip().rstrip("/")
    for row in load_providers():
        row_base = str(row.get("base_url") or "").rstrip("/")
        row_key = str(row.get("api_key") or "")
        models = {str(m).strip() for m in (row.get("models") or []) if str(m).strip()}
        if base_norm and row_base == base_norm:
            return row_base, row_key
        if model and model in models:
            return row_base or base_norm, row_key
    return base, ""


def resolve_llm(cfg: PipelineConfig):
    place = (cfg.llm_place or "auto").strip().lower()
    model = (cfg.llm_model or os.environ.get("VIDEOCLEAN_LLM_MODEL") or "").strip()
    base = (
        cfg.llm_base_url
        or os.environ.get("VIDEOCLEAN_LLM_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or ""
    ).strip()
    key = (
        cfg.llm_api_key
        or os.environ.get("VIDEOCLEAN_LLM_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or ""
    ).strip()
    prov_base, prov_key = _provider_credentials(model, base)
    if prov_base and not base:
        base = prov_base
    if prov_key and not key:
        key = prov_key
    openai_key = (os.environ.get("OPENAI_API_KEY") or "").strip()

    if place == "auto":
        if _looks_gguf(model):
            place = "local"
        elif base and is_local_url(base):
            place = "local"
        elif base and not is_local_url(base):
            place = "cloud"
        elif openai_key or (key and not is_local_url(base)):
            place = "cloud"
        else:
            place = "local"

    if place == "local":
        if _looks_gguf(model):
            return LlamaCppLlm(model)
        url = base or DEFAULT_LOCAL_URL
        tag = model or DEFAULT_LOCAL_MODEL
        return OpenAiCompatLlm(place="local", base_url=url, model=tag, api_key=key or "local")

    if place == "cloud":
        if not base:
            base = DEFAULT_OPENAI_URL
            model = model or DEFAULT_OPENAI_MODEL
        elif not model:
            model = DEFAULT_OPENAI_MODEL
        if not key:
            return UnconfiguredLlm(
                "cloud LLM needs an API key. Set OPENAI_API_KEY or VIDEOCLEAN_LLM_API_KEY, "
                "or switch to --llm local"
            )
        return OpenAiCompatLlm(place="cloud", base_url=base, model=model, api_key=key)

    return UnconfiguredLlm(f"unknown --llm {place!r}. use auto | local | cloud")
