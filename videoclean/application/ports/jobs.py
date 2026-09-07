from __future__ import annotations

from typing import Any, Protocol


class JobStore(Protocol):
    def upsert(
        self,
        job_id: str,
        state: str,
        input_path: str | None = None,
        output_path: str | None = None,
        prompt: str | None = None,
        report: dict[str, Any] | None = None,
    ) -> None: ...
