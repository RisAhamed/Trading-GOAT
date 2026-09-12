"""MarketDataProvider: Alpaca adapter with retry, cache-awareness, quality gates.

Wraps the REAL legacy API (verified against core/market_data.py):
  fetch_bars(symbol, interval, lookback_bars) -> DataFrame | None
  fetch_latest_quote(symbol)                 -> QuoteData | None

Freshness semantics (§17-18): a 15s cycle reuses the same 5Min candle for many
cycles. That is cache reuse, not staleness. We report separately:
  bar_timestamp / retrieved_at / bar_age_s / is_new_bar / is_fresh.
A bar is stale only when its age exceeds max_age_s for the timeframe
(default: 3x bar interval + grace), not merely because it is cached.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol

import pandas as pd

from data.quality import QualityReport, check_bars
from domain.models import MarketSnapshot

TIMEFRAME_SECONDS = {
    "1Min": 60, "5Min": 300, "10Min": 600, "15Min": 900,
    "30Min": 1800, "1Hour": 3600, "1Day": 86400,
}


def default_max_age_s(timeframe: str) -> float:
    """3x bar interval + 60s grace. A 5Min bar is fresh for ~16 min."""
    return TIMEFRAME_SECONDS.get(timeframe, 300) * 3 + 60.0


@dataclass
class BarsResult:
    df: pd.DataFrame
    quality: QualityReport
    latency_ms: float
    source: str = "alpaca"
    retrieved_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    is_new_bar: bool = True

    @property
    def bar_age_s(self) -> float:
        return self.quality.age_seconds


class MarketDataProvider(Protocol):
    def fetch_bars(self, symbol: str, timeframe: str, lookback: int) -> BarsResult: ...
    def latest_price(self, symbol: str) -> MarketSnapshot: ...


class AlpacaProvider:
    """Alpaca-backed provider. Recoverable (retry) vs fatal errors distinguished."""

    def __init__(self, legacy_fetcher, max_age_s: float | None = None) -> None:
        self._fetcher = legacy_fetcher
        self._max_age_override = max_age_s
        self._seen: dict[tuple[str, str], datetime] = {}

    def _max_age(self, timeframe: str) -> float:
        return self._max_age_override if self._max_age_override is not None else default_max_age_s(timeframe)

    def fetch_bars(self, symbol: str, timeframe: str, lookback: int) -> BarsResult:
        t0 = time.perf_counter()
        last_exc: Exception | None = None
        df = None
        for attempt in range(3):  # exponential backoff: 1s, 2s, 4s (bounded)
            try:
                df = self._fetcher.fetch_bars(symbol, timeframe, lookback)
                break
            except Exception as e:  # recoverable: network/rate-limit/transient
                last_exc = e
                time.sleep(1.0 * (2**attempt))
        latency_ms = (time.perf_counter() - t0) * 1000.0
        now = datetime.now(timezone.utc)
        if df is None:
            # None = fetcher failed fatally (incl. invalid-symbol rejection upstream).
            raise RuntimeError(f"market_data_unavailable after retries: {last_exc}")
        prev = self._seen.get((symbol, timeframe))
        quality = check_bars(df, timeframe=timeframe, max_age_s=self._max_age(timeframe), seen_last_ts=prev)
        is_new = bool(quality.bar_timestamp and (prev is None or quality.bar_timestamp > prev))
        if quality.bar_timestamp:
            self._seen[(symbol, timeframe)] = quality.bar_timestamp
        return BarsResult(df=df, quality=quality, latency_ms=latency_ms, retrieved_at=now, is_new_bar=is_new)

    def latest_price(self, symbol: str) -> MarketSnapshot:
        quote = self._fetcher.fetch_latest_quote(symbol)  # real legacy name
        now = datetime.now(timezone.utc)
        if quote is None:
            return MarketSnapshot(symbol=symbol, price=0.0, timestamp=now, stale=True,
                                  issues=["quote_unavailable"])
        price = float(getattr(quote, "mid_price", 0.0) or 0.0)
        return MarketSnapshot(symbol=symbol, price=price, timestamp=now,
                              bar_timestamp=getattr(quote, "timestamp", None),
                              stale=price <= 0,
                              issues=[] if price > 0 else ["invalid_quote_price"])
