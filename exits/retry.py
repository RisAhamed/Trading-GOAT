"""Bounded exit retry: a failed close must not spin every cycle.

Records attempt/error/retry_count/next_retry_at. Caller asks
`should_retry(symbol, now)`; Ultrasonic every-cycle retries become
1st retry ~30s, then 2m, 5m, 15m caps. Broker-flat positions clear state
instead of retrying.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

BACKOFF_S = (30, 120, 300, 900)
MAX_ATTEMPTS = 6


@dataclass
class ExitAttempt:
    symbol: str
    attempts: int = 0
    last_error: str = ""
    last_attempt_at: str = ""
    next_retry_at: float = 0.0  # epoch seconds


@dataclass
class ExitRetryTracker:
    attempts: dict[str, ExitAttempt] = field(default_factory=dict)

    def record_failure(self, symbol: str, error: str, now_s: float | None = None) -> ExitAttempt:
        now = now_s if now_s is not None else datetime.now(timezone.utc).timestamp()
        a = self.attempts.get(symbol) or ExitAttempt(symbol=symbol)
        a.attempts += 1
        a.last_error = str(error)[:300]
        a.last_attempt_at = datetime.now(timezone.utc).isoformat()
        step = BACKOFF_S[min(a.attempts - 1, len(BACKOFF_S) - 1)]
        a.next_retry_at = now + step
        self.attempts[symbol] = a
        return a

    def record_success(self, symbol: str) -> None:
        self.attempts.pop(symbol, None)

    def clear_if_flat(self, symbol: str, broker_open_symbols: set[str]) -> bool:
        """Broker says flat -> clear local retry state, never retry again."""
        if symbol not in broker_open_symbols:
            self.attempts.pop(symbol, None)
            return True
        return False

    def should_retry(self, symbol: str, now_s: float | None = None) -> bool:
        now = now_s if now_s is not None else datetime.now(timezone.utc).timestamp()
        a = self.attempts.get(symbol)
        if a is None:
            return True
        if a.attempts >= MAX_ATTEMPTS:
            return False  # needs human attention; surfaced via metrics/events
        return now >= a.next_retry_at
