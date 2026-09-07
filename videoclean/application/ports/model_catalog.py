from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ComponentInfo:
    id: str
    title: str
    kind: str
    model_ref: str
    size_hint: str
    backend: str = ""
    source: str = "builtin"


@dataclass(frozen=True)
class ComponentStatus:
    info: ComponentInfo
    state: str  # ready | missing | downloading | error
    message: str


class ModelCatalog(Protocol):
    def list_status(self) -> list[ComponentStatus]: ...

    def is_ready(self, component_id: str) -> bool: ...
