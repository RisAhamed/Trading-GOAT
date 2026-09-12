"""Optional enrichment that can never block the critical loop.

Whale (Kraken), political (QuiverQuant), and similar signals run ONLY here:
background daemon refresh with bounded timeout, per-source circuit breaker
(cooldown after N consecutive failures; longer for auth errors), cached
last-good values, and neutral fallback with explicit UNAVAILABLE reason.

Critical path calls `snapshot()` (non-blocking, cached-or-neutral) and never
waits on network. Health: ok / DEGRADED(source=reason).
"""
from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

AUTH_COOLDOWN_S = 3600.0
FAIL_COOLDOWN_S = 300.0
FAIL_THRESHOLD = 3


@dataclass
class SourceState:
    name: str
    value: dict = field(default_factory=dict)
    available: bool = False
    reason: str = "not_refreshed_yet"
    failures: int = 0
    cooldown_until: float = 0.0
    updated_at: float = 0.0


class EnrichmentHub:
    def __init__(self) -> None:
        self._sources: dict[str, SourceState] = {}
        self._fetchers: dict[str, Callable[[], dict]] = {}
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def register(self, name: str, fetcher: Callable[[], dict]) -> None:
        with self._lock:
            self._fetchers[name] = fetcher
            self._sources.setdefault(name, SourceState(name=name))

    def snapshot(self) -> dict[str, dict]:
        """Non-blocking: cached-or-neutral, never network."""
        with self._lock:
            return {k: {"available": v.available, "reason": v.reason, "value": dict(v.value)}
                    for k, v in self._sources.items()}

    def health(self) -> dict[str, str]:
        with self._lock:
            return {k: ("ok" if v.available else f"UNAVAILABLE:{v.reason}") for k, v in self._sources.items()}

    def refresh_once(self, timeout_s: float = 8.0) -> None:
        """One bounded refresh pass; each source isolated (one slow source
        cannot starve others). Called from the background thread."""
        with self._lock:
            names = list(self._fetchers)
        now = time.time()
        for name in names:
            with self._lock:
                st = self._sources[name]
                if now < st.cooldown_until:
                    continue
            box: dict[str, Any] = {}
            t = threading.Thread(target=self._run_fetch, args=(name, box), daemon=True)
            t.start()
            t.join(timeout=timeout_s)
            with self._lock:
                st = self._sources[name]
                if box.get("ok"):
                    st.value = box["value"]
                    st.available = True
                    st.reason = "fresh"
                    st.failures = 0
                    st.updated_at = time.time()
                else:
                    st.failures += 1
                    err = str(box.get("error", "timeout"))[:120]
                    if "401" in err or "unauthorized" in err.lower() or "auth" in err.lower():
                        st.reason = "AUTHENTICATION_FAILURE"
                        st.cooldown_until = time.time() + AUTH_COOLDOWN_S
                    elif st.failures >= FAIL_THRESHOLD:
                        st.reason = f"CIRCUIT_OPEN:{err}"
                        st.cooldown_until = time.time() + FAIL_COOLDOWN_S
                    else:
                        st.reason = err
                    st.available = False

    def _run_fetch(self, name: str, box: dict) -> None:
        try:
            with self._lock:
                fn = self._fetchers[name]
            box["value"] = fn() or {}
            box["ok"] = True
        except Exception as e:  # enrichment boundary: isolate, never raise
            box["ok"] = False
            box["error"] = str(e)[:200]

    def start_background(self, interval_s: float = 120.0) -> None:
        if self._thread and self._thread.is_alive():
            return

        def _loop() -> None:
            while True:
                try:
                    self.refresh_once()
                except Exception:
                    pass
                time.sleep(interval_s)

        self._thread = threading.Thread(target=_loop, daemon=True, name="enrichment")
        self._thread.start()
