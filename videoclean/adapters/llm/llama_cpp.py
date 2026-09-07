from __future__ import annotations

from pathlib import Path

from videoclean.application.errors import AdapterUnavailable


class LlamaCppLlm:
    """Local GGUF via llama-cpp-python. Optional extra: uv sync --extra local-llm"""

    name = "llama.cpp"

    def __init__(self, model_path: str):
        self.model = model_path
        self._path = Path(model_path).expanduser()
        self._llm = None

    def status(self) -> str:
        if not self._path.is_file():
            return f"unavailable: GGUF not found: {self._path}"
        try:
            import llama_cpp  # noqa: F401
        except ImportError:
            return (
                "unavailable: llama-cpp-python not installed. "
                "uv sync --extra local-llm   or use --llm local with Ollama /v1"
            )
        return f"ready (local llama.cpp {self._path.name})"

    def complete(
        self,
        system: str,
        user: str,
        image_jpeg: bytes | None = None,
        images: list[bytes] | None = None,
    ) -> str:
        if image_jpeg or images:
            raise AdapterUnavailable(
                "llama.cpp GGUF path is text-only. Use Ollama with a vision model "
                "(e.g. llava-phi3) for frame-aware prompt parsing."
            )
        st = self.status()
        if not st.startswith("ready"):
            raise AdapterUnavailable(f"prompt-parser llm: {st}")
        llm = self._load()
        try:
            out = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.0,
            )
        except Exception as exc:  # noqa: BLE001
            raise AdapterUnavailable(f"llama.cpp failed: {exc}") from exc
        try:
            text = out["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise AdapterUnavailable("llama.cpp returned no content") from exc
        if not isinstance(text, str) or not text.strip():
            raise AdapterUnavailable("llama.cpp returned empty content")
        return text

    def _load(self):
        if self._llm is None:
            from llama_cpp import Llama

            self._llm = Llama(model_path=str(self._path), n_ctx=4096, verbose=False)
        return self._llm
