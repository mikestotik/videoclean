from __future__ import annotations

import json
import os
import sqlite3
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SQLITE_TIMEOUT_S = 30.0


def resolve_data_dir(env: Mapping[str, str] | None = None) -> Path:
    env_map = os.environ if env is None else env
    raw = str(env_map.get("VIDEOCLEAN_DATA_DIR") or "").strip()
    if raw:
        return Path(raw).expanduser()
    return Path.home() / ".videoclean"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_job_id() -> str:
    stamp = utc_now().strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{uuid.uuid4().hex[:8]}"


@dataclass
class JobPaths:
    root: Path
    input_dir: Path
    frames_dir: Path
    masks_dir: Path
    inpainted_dir: Path
    output_dir: Path
    logs_dir: Path
    events_file: Path
    ffmpeg_log: Path
    report_file: Path

    @classmethod
    def create(cls, root: Path) -> "JobPaths":
        paths = cls(
            root=root,
            input_dir=root / "input",
            frames_dir=root / "process" / "frames",
            masks_dir=root / "masks" / "processed",
            inpainted_dir=root / "process" / "inpainted",
            output_dir=root / "output",
            logs_dir=root / "logs",
            events_file=root / "logs" / "events.jsonl",
            ffmpeg_log=root / "logs" / "ffmpeg.log",
            report_file=root / "output" / "report.json",
        )
        for folder in (
            paths.input_dir,
            paths.frames_dir,
            paths.masks_dir,
            paths.inpainted_dir,
            paths.output_dir,
            paths.logs_dir,
        ):
            folder.mkdir(parents=True, exist_ok=True)
        return paths


_JOB_COLUMNS = (
    ("updated_at", "TEXT"),
    ("request_json", "TEXT"),
    ("progress_json", "TEXT"),
    ("error", "TEXT"),
    ("cancel_requested", "INTEGER NOT NULL DEFAULT 0"),
    ("source_id", "TEXT"),
)


class JobIndex:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        # Optional UI/SSE hook: called with "jobs" | "downloads" after mutations.
        self.on_change: Callable[[str], None] | None = None
        with self._connect() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    state TEXT NOT NULL,
                    input_path TEXT,
                    output_path TEXT,
                    prompt TEXT,
                    report_json TEXT
                )
                """
            )
            for name, decl in _JOB_COLUMNS:
                try:
                    con.execute(f"ALTER TABLE jobs ADD COLUMN {name} {decl}")
                except sqlite3.OperationalError as exc:
                    if "duplicate column" not in str(exc).lower():
                        raise
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS downloads (
                    id TEXT PRIMARY KEY,
                    component_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    progress REAL,
                    bytes_done INTEGER,
                    bytes_total INTEGER,
                    message TEXT,
                    updated_at TEXT NOT NULL,
                    cancel_requested INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            try:
                con.execute(
                    "ALTER TABLE downloads ADD COLUMN cancel_requested INTEGER NOT NULL DEFAULT 0"
                )
            except sqlite3.OperationalError as exc:
                if "duplicate column" not in str(exc).lower():
                    raise

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path, timeout=SQLITE_TIMEOUT_S)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        return con

    def _notify(self, kind: str) -> None:
        cb = self.on_change
        if cb is None:
            return
        try:
            cb(kind)
        except Exception:  # noqa: BLE001 — hooks must not break store writes
            return

    def upsert(
        self,
        job_id: str,
        state: str,
        input_path: str | None = None,
        output_path: str | None = None,
        prompt: str | None = None,
        report: dict[str, Any] | None = None,
        request: dict[str, Any] | None = None,
        progress: dict[str, Any] | None = None,
        error: str | None = None,
        cancel_requested: bool | None = None,
        source_id: str | None = None,
    ) -> None:
        now = utc_now().isoformat()
        with self._connect() as con:
            existing = con.execute("SELECT id FROM jobs WHERE id = ?", (job_id,)).fetchone()
            report_payload = json.dumps(report, ensure_ascii=False) if report is not None else None
            request_payload = (
                json.dumps(request, ensure_ascii=False) if request is not None else None
            )
            progress_payload = (
                json.dumps(progress, ensure_ascii=False) if progress is not None else None
            )
            cancel_val = None if cancel_requested is None else (1 if cancel_requested else 0)
            if existing:
                con.execute(
                    """
                    UPDATE jobs
                    SET state = ?,
                        updated_at = ?,
                        input_path = COALESCE(?, input_path),
                        output_path = COALESCE(?, output_path),
                        prompt = COALESCE(?, prompt),
                        report_json = COALESCE(?, report_json),
                        request_json = COALESCE(?, request_json),
                        progress_json = COALESCE(?, progress_json),
                        error = COALESCE(?, error),
                        cancel_requested = COALESCE(?, cancel_requested),
                        source_id = COALESCE(?, source_id)
                    WHERE id = ?
                    """,
                    (
                        state,
                        now,
                        input_path,
                        output_path,
                        prompt,
                        report_payload,
                        request_payload,
                        progress_payload,
                        error,
                        cancel_val,
                        source_id,
                        job_id,
                    ),
                )
            else:
                con.execute(
                    """
                    INSERT INTO jobs (
                        id, created_at, updated_at, state, input_path, output_path,
                        prompt, report_json, request_json, progress_json, error,
                        cancel_requested, source_id
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        job_id,
                        now,
                        now,
                        state,
                        input_path,
                        output_path,
                        prompt,
                        report_payload,
                        request_payload,
                        progress_payload,
                        error,
                        cancel_val if cancel_val is not None else 0,
                        source_id,
                    ),
                )
        self._notify("jobs")

    def list_jobs(self, limit: int = 100, state: str | None = None) -> list[sqlite3.Row]:
        with self._connect() as con:
            if state is None:
                return list(
                    con.execute(
                        "SELECT id, created_at, state, input_path, output_path FROM jobs "
                        "ORDER BY created_at DESC LIMIT ?",
                        (limit,),
                    )
                )
            return list(
                con.execute(
                    "SELECT id, created_at, state, input_path, output_path FROM jobs "
                    "WHERE state = ? ORDER BY created_at DESC LIMIT ?",
                    (state, limit),
                )
            )

    def list_jobs_full(self, limit: int = 100, state: str | None = None) -> list[sqlite3.Row]:
        """One-query full rows for UI tables (avoids N+1 get())."""
        with self._connect() as con:
            if state is None:
                return list(
                    con.execute(
                        "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?",
                        (limit,),
                    )
                )
            return list(
                con.execute(
                    "SELECT * FROM jobs WHERE state = ? ORDER BY created_at DESC LIMIT ?",
                    (state, limit),
                )
            )

    def get(self, job_id: str) -> sqlite3.Row | None:
        with self._connect() as con:
            return con.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()

    def delete(self, job_id: str) -> bool:
        with self._connect() as con:
            cur = con.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
            ok = cur.rowcount > 0
        if ok:
            self._notify("jobs")
        return ok

    def request_cancel(self, job_id: str) -> bool:
        now = utc_now().isoformat()
        with self._connect() as con:
            cur = con.execute(
                "UPDATE jobs SET cancel_requested = 1, updated_at = ? WHERE id = ?",
                (now, job_id),
            )
            ok = cur.rowcount > 0
        if ok:
            self._notify("jobs")
        return ok

    def is_cancel_requested(self, job_id: str) -> bool:
        with self._connect() as con:
            row = con.execute(
                "SELECT cancel_requested FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
            return bool(row and row["cancel_requested"])

    def mark_orphans_failed(self, reason: str = "interrupted") -> int:
        now = utc_now().isoformat()
        with self._connect() as con:
            cur = con.execute(
                """
                UPDATE jobs
                SET state = 'FAILED', error = ?, updated_at = ?
                WHERE state = 'RUNNING'
                """,
                (reason, now),
            )
            return cur.rowcount

    def update_progress(self, job_id: str, progress: dict[str, Any]) -> None:
        now = utc_now().isoformat()
        payload = json.dumps(progress, ensure_ascii=False)
        with self._connect() as con:
            con.execute(
                "UPDATE jobs SET progress_json = ?, updated_at = ? WHERE id = ?",
                (payload, now, job_id),
            )
        self._notify("jobs")

    def upsert_download(
        self,
        download_id: str,
        component_id: str,
        state: str,
        progress: float | None = None,
        bytes_done: int | None = None,
        bytes_total: int | None = None,
        message: str | None = None,
        cancel_requested: bool | None = None,
    ) -> None:
        now = utc_now().isoformat()
        cancel_val = None if cancel_requested is None else (1 if cancel_requested else 0)
        with self._connect() as con:
            existing = con.execute(
                "SELECT id FROM downloads WHERE id = ?", (download_id,)
            ).fetchone()
            if existing:
                con.execute(
                    """
                    UPDATE downloads
                    SET component_id = ?,
                        state = ?,
                        progress = COALESCE(?, progress),
                        bytes_done = COALESCE(?, bytes_done),
                        bytes_total = COALESCE(?, bytes_total),
                        message = COALESCE(?, message),
                        updated_at = ?,
                        cancel_requested = COALESCE(?, cancel_requested)
                    WHERE id = ?
                    """,
                    (
                        component_id,
                        state,
                        progress,
                        bytes_done,
                        bytes_total,
                        message,
                        now,
                        cancel_val,
                        download_id,
                    ),
                )
            else:
                con.execute(
                    """
                    INSERT INTO downloads (
                        id, component_id, state, progress, bytes_done, bytes_total,
                        message, updated_at, cancel_requested
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        download_id,
                        component_id,
                        state,
                        progress,
                        bytes_done,
                        bytes_total,
                        message,
                        now,
                        cancel_val if cancel_val is not None else 0,
                    ),
                )
        self._notify("downloads")

    def get_download(self, download_id: str) -> sqlite3.Row | None:
        with self._connect() as con:
            return con.execute(
                "SELECT * FROM downloads WHERE id = ?", (download_id,)
            ).fetchone()

    def list_downloads(self, limit: int = 100, state: str | None = None) -> list[sqlite3.Row]:
        with self._connect() as con:
            if state is None:
                return list(
                    con.execute(
                        "SELECT * FROM downloads ORDER BY updated_at DESC LIMIT ?",
                        (limit,),
                    )
                )
            return list(
                con.execute(
                    "SELECT * FROM downloads WHERE state = ? ORDER BY updated_at DESC LIMIT ?",
                    (state, limit),
                )
            )

    def request_download_cancel(self, download_id: str) -> bool:
        now = utc_now().isoformat()
        with self._connect() as con:
            cur = con.execute(
                "UPDATE downloads SET cancel_requested = 1, updated_at = ? WHERE id = ?",
                (now, download_id),
            )
            ok = cur.rowcount > 0
        if ok:
            self._notify("downloads")
        return ok


def new_source_id() -> str:
    stamp = utc_now().strftime("%Y%m%d_%H%M%S")
    return f"s_{stamp}_{uuid.uuid4().hex[:8]}"


class SourceIndex:
    """Uploaded videos independent of jobs. Same sqlite file as JobIndex."""

    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self.on_change: Callable[[str], None] | None = None
        with self._connect() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS sources (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    path TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    probe_json TEXT
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path, timeout=SQLITE_TIMEOUT_S)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        return con

    def _notify(self, kind: str) -> None:
        cb = self.on_change
        if cb is None:
            return
        try:
            cb(kind)
        except Exception:  # noqa: BLE001
            return

    def register(self, source_id: str, name: str, path: str, probe: dict[str, Any] | None = None) -> None:
        now = utc_now().isoformat()
        with self._connect() as con:
            con.execute(
                "INSERT OR REPLACE INTO sources (id, name, path, created_at, probe_json) VALUES (?, ?, ?, ?, ?)",
                (source_id, name, path, now, json.dumps(probe or {}, ensure_ascii=False)),
            )
        self._notify("sources")

    def list(self, limit: int = 200) -> list[sqlite3.Row]:
        with self._connect() as con:
            return list(con.execute("SELECT * FROM sources ORDER BY created_at DESC LIMIT ?", (limit,)))

    def get(self, source_id: str) -> sqlite3.Row | None:
        with self._connect() as con:
            return con.execute("SELECT * FROM sources WHERE id = ?", (source_id,)).fetchone()

    def delete(self, source_id: str) -> bool:
        with self._connect() as con:
            cur = con.execute("DELETE FROM sources WHERE id = ?", (source_id,))
            ok = cur.rowcount > 0
        if ok:
            self._notify("sources")
        return ok
