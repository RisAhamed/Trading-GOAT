"""Feature snapshot builder: wraps legacy IndicatorCalculator, adds validation.

Documents per-feature contract in FEATURE_NOTES below.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

import pandas as pd

from domain.models import FeatureSnapshot

FEATURE_NOTES = {
    "rsi": "Momentum oscillator (0-100). <35 oversold bounce watch, >65 overbought fade watch. Used in entry context + confluence.",
    "macd": "Trend-momentum: macd_line vs signal + histogram slope. Rising histogram supports entries in trend direction.",
    "ema": "Trend filter: price vs ema_short/long, golden/death cross. Higher-TF EMA stack defines TrendContext.",
    "bollinger": "Position within volatility band via %B. >=0.8 stretched, <=0.2 washed out. Not a standalone signal.",
    "atr": "Volatility ruler for stops/sizing: stop_distance = 1.5xATR floored by stop_loss_pct. Also regime input.",
    "adx": "Trend strength: >20 trending, <20 chop. Hardened: ADXIndicator window=14.",
    "volume": "Participation: volume_ratio vs 20-bar SMA + rising flag. Confirms, never triggers alone.",
}


def _finite(x: float, lo: float, hi: float) -> bool:
    return x is not None and lo <= x <= hi and math.isfinite(x)


def build_snapshot(symbol: str, timeframe: str, df: pd.DataFrame, legacy_values) -> FeatureSnapshot:
    """Convert legacy IndicatorValues -> typed snapshot with explicit quality info."""
    now = datetime.now(timezone.utc)
    snap = FeatureSnapshot(
        symbol=symbol, timeframe=timeframe, timestamp=now,
        price=float(getattr(legacy_values, "current_price", 0.0) or 0.0),
        rsi=float(getattr(legacy_values, "rsi", 50.0)),
        macd_line=float(getattr(legacy_values, "macd_line", 0.0)),
        macd_signal=float(getattr(legacy_values, "macd_signal", 0.0)),
        macd_histogram=float(getattr(legacy_values, "macd_histogram", 0.0)),
        ema_short=float(getattr(legacy_values, "ema_short", 0.0)),
        ema_long=float(getattr(legacy_values, "ema_long", 0.0)),
        bb_upper=float(getattr(legacy_values, "bb_upper", 0.0)),
        bb_middle=float(getattr(legacy_values, "bb_middle", 0.0)),
        bb_lower=float(getattr(legacy_values, "bb_lower", 0.0)),
        bb_percent=float(getattr(legacy_values, "bb_percent", 0.5)),
        atr=float(getattr(legacy_values, "atr", 0.0)),
        atr_percent=float(getattr(legacy_values, "atr_percent", 0.0)),
        adx=float(getattr(legacy_values, "adx", 0.0)),
        volume_ratio=float(getattr(legacy_values, "volume_ratio", 1.0)),
        overall_trend=str(getattr(legacy_values, "overall_trend", "SIDEWAYS")),
        data_points=int(getattr(legacy_values, "data_points", len(df) if df is not None else 0)),
    )
    issues: list[str] = []
    if df is None or df.empty:
        issues.append("missing_ohlcv_data")
    if snap.data_points < 30:
        issues.append(f"insufficient_history n={snap.data_points}<30")
    if not _finite(snap.rsi, 0, 100):
        issues.append(f"unrealistic_rsi {snap.rsi}")
    if not _finite(snap.bb_percent, 0, 1):
        issues.append(f"unrealistic_bb_percent {snap.bb_percent}")
    if not _finite(snap.adx, 0, 100):
        issues.append(f"unrealistic_adx {snap.adx}")
    if snap.price <= 0:
        issues.append("invalid_price")
    if snap.atr < 0 or snap.atr_percent < 0 or snap.atr_percent > 50:
        issues.append(f"suspicious_atr {snap.atr}/{snap.atr_percent:.2f}%")
    snap.quality_issues = issues
    snap.valid = not any(i.startswith(("missing", "insufficient", "invalid", "unrealistic")) for i in issues)
    return snap
