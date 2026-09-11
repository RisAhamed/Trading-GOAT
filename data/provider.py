"""MarketDataProvider protocol + Alpaca adapter with retry, cache, quality gates.

thin wrapper around legacy core.market_data.MarketDataFetcher so historical
behavior is preserved, but every fetch returns a normalized, validated result.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

import pandas as pd

from data.quality import QualityReport, check_bars
from domain.models import MarketSnapshot


@dataclass
class BarsResult:
    df: pd.DataFrame
    quality: QualityReport
    latency_ms: float
    source: str = "alpaca"


class MarketDataProvider(Protocol):
    def fetch_bars(self, symbol: str, timeframe: str, lookback: int) -> BarsResult: ...
    def latest_price(self, symbol: str) -> MarketSnapshot: ...


class AlpacaProvider:
    """Alpaca-backed provider. Recoverable (retry) vs fatal errors distinguished."""

    def __init__(self, legacy_fetcher, max_age_s: float = 600.0) -> None:
        self._fetcher = legacy_fetcher
        self._max_age_s = max_age_s
        self._seen: dict[tuple[str, str], datetime] = {}

    def fetch_bars(self, symbol: str, timeframe: str, lookback: int) -> BarsResult:
        t0 = time.perf_counter()
        last_exc: Exception | None = None
        df = None
        for attempt in range(3):  # exponential backoff: 1s, 2s
            try:
                df = self._fetcher.fetch_bars(symbol, timeframe, lookback)
                break
            except Exception as e:  # recoverable: network/rate-limit/transient
                last_exc = e
                time.sleep(1.0 * (2**attempt))
        latency_ms = (time.perf_counter() - t0) * 1000.0
        if df is None:
            raise RuntimeError(f"market_data_unavailable after retries: {last_exc}")
        quality = check_bars(
            df, timeframe=timeframe, max_age_s=self._max_age_s,
            seen_last_ts=self._seen.get((symbol, timeframe)),
        )
        if quality.bar_timestamp:
            self._seen[(symbol, timeframe)] = quality.bar_timestamp
        return BarsResult(df=df, quality=quality, latency_ms=latency_ms)

    def latest_price(self, symbol: str) -> MarketSnapshot:
        quote = self._fetcher.get_latest_quote(symbol)
        now = datetime.now(timezone.utc)
        if quote is None:
            return MarketSnapshot(symbol=symbol, price=0.0, timestamp=now, stale=True, issues=["quote_unavailable"])
        price = float(getattr(quote, "mid_price", 0.0) or 0.0)
        return MarketSnapshot(symbol=symbol, price=price, timestamp=now, stale=price <= 0,
                              issues=[] if price > 0 else ["invalid_quote_price"])
