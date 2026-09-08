from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request

from videoclean.application.errors import AdapterUnavailable


class OpenAiCompatLlm:
    """Any OpenAI-compatible /v1/chat/completions: Ollama, vLLM, llama.cpp server, OpenAI."""

    name = "openai-compat"

    def __init__(self, *, place: str, base_url: str, model: str, api_key: str | None, timeout_s: float = 120.0):
        self.place = place
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or "not-needed"
        self.timeout_s = timeout_s

    def status(self) -> str:
        if _is_loopback(self.base_url):
            ok, err = self._ping()
            if not ok:
                return f"unavailable: {err}"
        return f"ready ({self.place} {self.model} @ {self.base_url})"

    def _ping(self) -> tuple[bool, str]:
        url = self.base_url.rstrip("/") + "/models"
        req = urllib.request.Request(
            url,
            method="GET",
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=2.5) as resp:
                resp.read(256)
        except urllib.error.HTTPError as exc:
            return False, f"llm HTTP {exc.code} at {url}"
        except urllib.error.URLError as exc:
            return False, f"llm unreachable at {url}: {exc.reason}"
        except TimeoutError:
            return False, f"llm timeout at {url}"
        return True, ""

    def complete(
        self,
        system: str,
        user: str,
        image_jpeg: bytes | None = None,
        images: list[bytes] | None = None,
    ) -> str:
        url = self.base_url + "/chat/completions"
        jpegs: list[bytes] = []
        if images:
            jpegs.extend(images)
        if image_jpeg:
            jpegs.append(image_jpeg)
        if jpegs:
            # Images first: some Ollama vision tags attend better that way.
            user_content: object = []
            for jpeg in jpegs:
                b64 = base64.b64encode(jpeg).decode("ascii")
                user_content.append(
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                )
            user_content.append({"type": "text", "text": user})
        else:
            user_content = user
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        timeout = self.timeout_s if not jpegs else max(self.timeout_s, 180.0)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            raise AdapterUnavailable(f"llm HTTP {exc.code} at {url}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise AdapterUnavailable(f"llm unreachable at {url}: {exc.reason}") from exc
        try:
            text = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise AdapterUnavailable(f"llm response missing choices[0].message.content: {body!r}"[:400]) from exc
        if not isinstance(text, str) or not text.strip():
            raise AdapterUnavailable("llm returned empty content")
        return text


def _is_loopback(url: str) -> bool:
    u = (url or "").lower()
    return any(h in u for h in ("127.0.0.1", "localhost", "0.0.0.0", "[::1]"))
