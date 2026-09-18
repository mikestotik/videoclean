"""In-process pub/sub for WebUI SSE (server → browser)."""

from __future__ import annotations

import json
import queue
import threading
import time
from collections.abc import Callable
from typing import Any


class EventHub:
    """Fan-out named events to SSE subscribers. Thread-safe."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subs: list[queue.Queue[str]] = []
        self._last_emit: dict[str, float] = {}
        self._pending: set[str] = set()
        self._flush_timer: threading.Timer | None = None

    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subs)

    def subscribe(self) -> queue.Queue[str]:
        q: queue.Queue[str] = queue.Queue(maxsize=64)
        with self._lock:
            self._subs.append(q)
        return q

    def unsubscribe(self, q: queue.Queue[str]) -> None:
        with self._lock:
            try:
                self._subs.remove(q)
            except ValueError:
                pass

    def publish(self, event: str, *, min_interval: float = 0.0) -> None:
        """Notify subscribers. Optional coalesce/throttle for high-frequency kinds."""
        if min_interval <= 0:
            self._fanout(event)
            return
        now = time.monotonic()
        with self._lock:
            last = self._last_emit.get(event, 0.0)
            if now - last >= min_interval:
                self._last_emit[event] = now
                self._pending.discard(event)
                subs = list(self._subs)
            else:
                self._pending.add(event)
                if self._flush_timer is None:
                    delay = max(0.01, min_interval - (now - last))
                    timer = threading.Timer(delay, self._flush_pending)
                    timer.daemon = True
                    self._flush_timer = timer
                    timer.start()
                return
        for q in subs:
            self._put(q, event)

    def _flush_pending(self) -> None:
        with self._lock:
            self._flush_timer = None
            pending = sorted(self._pending)
            self._pending.clear()
            now = time.monotonic()
            for event in pending:
                self._last_emit[event] = now
            subs = list(self._subs)
        for event in pending:
            for q in subs:
                self._put(q, event)

    def _fanout(self, event: str) -> None:
        with self._lock:
            subs = list(self._subs)
        for q in subs:
            self._put(q, event)

    @staticmethod
    def _put(q: queue.Queue[str], event: str) -> None:
        try:
            q.put_nowait(event)
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass
            try:
                q.put_nowait(event)
            except queue.Full:
                pass


def format_sse(event: str, data: Any) -> str:
    payload = json.dumps(data, ensure_ascii=False, default=str)
    lines = [f"event: {event}", *(f"data: {line}" for line in payload.splitlines() or [""]), ""]
    return "\n".join(lines) + "\n"


def wire_store_hooks(
    hub: EventHub,
    *,
    jobs_on_change: Callable[[str], None] | None = None,
) -> Callable[[str], None]:
    """Return a JobIndex/SourceIndex on_change callback that publishes to hub."""

    def on_change(kind: str) -> None:
        if jobs_on_change is not None:
            jobs_on_change(kind)
        if kind == "jobs":
            hub.publish("jobs", min_interval=0.2)
        elif kind == "downloads":
            hub.publish("downloads", min_interval=0.25)
        elif kind == "sources":
            hub.publish("sources", min_interval=0.0)
        else:
            hub.publish(kind, min_interval=0.0)

    return on_change
