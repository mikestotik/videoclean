from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol


class JobControl(Protocol):
    def submit(
        self,
        request_dict: dict[str, Any],
        input_path: Path,
        output_path: Path,
        prompt: str,
    ) -> str: ...

    def cancel(self, job_id: str) -> None: ...

    def retry(self, job_id: str) -> str: ...

    def delete(self, job_id: str, data_dir: Path) -> None: ...

    def mark_failed(self, job_id: str, reason: str) -> None: ...

    def recover_orphans(self) -> int: ...
