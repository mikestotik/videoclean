"""Process-lifetime weight cache. Keyed by kind, model id, device, dtype."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

_LOCK = threading.Lock()
_CACHE: dict[tuple[str, str, str, str], Any] = {}


@dataclass(frozen=True)
class WeightKey:
    kind: str
    model_id: str
    device: str
    dtype: str

    def as_tuple(self) -> tuple[str, str, str, str]:
        return (self.kind, self.model_id, self.device, self.dtype)


def _coerce(key: WeightKey | tuple[str, str, str, str]) -> tuple[str, str, str, str]:
    if isinstance(key, WeightKey):
        return key.as_tuple()
    if isinstance(key, tuple) and len(key) == 4:
        return key
    raise TypeError(f"weight cache key must be WeightKey or 4-tuple, got {key!r}")


def get_or_load(key: WeightKey | tuple[str, str, str, str], loader: Callable[[], Any]) -> tuple[Any, bool]:
    """Return (module, loaded_now). loaded_now is false on a warm hit."""
    token = _coerce(key)
    with _LOCK:
        hit = _CACHE.get(token)
        if hit is not None:
            return hit, False
    loaded = loader()
    with _LOCK:
        hit = _CACHE.get(token)
        if hit is not None:
            return hit, False
        _CACHE[token] = loaded
        return loaded, True


def cache_contains(key: WeightKey | tuple[str, str, str, str]) -> bool:
    return _coerce(key) in _CACHE


def clear_cache() -> None:
    with _LOCK:
        _CACHE.clear()
