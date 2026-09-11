"""Market-data quality gates: timezone-aware, stale/duplicate/gap/price/volume checks."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd

REQUIRED_COLS = ("open", "high", "low", "close", "volume")


@dataclass
class QualityReport:
    ok: bool
    stale: bool = False
    issues: list[str] = field(default_factory=list)
    bar_timestamp: datetime | None = None
    age_seconds: float = 0.0


def _to_utc(ts) -> datetime | None:
    try:
        t = pd.to_datetime(ts, utc=True).to_pydatetime()
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        return t
    except Exception:
        return None


def check_bars(
    df: pd.DataFrame | None,
    *,
    timeframe: str,
    max_age_s: float,
    now: datetime | None = None,
    seen_last_ts: datetime | None = None,
) -> QualityReport:
    now = now or datetime.now(timezone.utc)
    if df is None or df.empty:
        return QualityReport(ok=False, issues=["missing_ohlcv_data"])
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        return QualityReport(ok=False, issues=[f"missing_columns:{','.join(missing)}"])
    if df[["open", "high", "low", "close"]].isna().any().any():
        return QualityReport(ok=False, issues=["nan_in_ohlc"])
    if ((df["close"] <= 0) | (df["high"] <= 0) | (df["low"] <= 0)).any():
        return QualityReport(ok=False, issues=["invalid_nonpositive_price"])
    if ((df["high"] < df["low"]) | (df["high"] < df["close"]) | (df["low"] > df["close"])).any():
        return QualityReport(ok=False, issues=["invalid_high_low_range"])
    if (df["volume"] < 0).any():
        return QualityReport(ok=False, issues=["invalid_negative_volume"])

    idx = df.index
    bar_ts = _to_utc(idx[-1])
    age = (now - bar_ts).total_seconds() if bar_ts else 0.0
    issues: list[str] = []
    stale = age > max_age_s
    if bar_ts is None:
        issues.append("bar_timestamp_unparseable")
    if stale:
        issues.append(f"stale_data age_s={age:.0f} > max_{max_age_s:.0f}")
    if seen_last_ts and bar_ts and bar_ts <= seen_last_ts:
        issues.append("duplicate_bar no_new_data")
    # Missing-bar gap detection (last two bars)
    if len(df) >= 2:
        t1, t2 = _to_utc(idx[-2]), _to_utc(idx[-1])
        if t1 and t2:
            gap = (t2 - t1).total_seconds()
            # crude expected step per timeframe
            expect = {"1Min": 60, "5Min": 300, "15Min": 900, "1Hour": 3600}.get(timeframe, 300)
            if gap > expect * 3:
                issues.append(f"missing_bars gap_s={gap:.0f}")
    ok = not stale and not any(i.startswith(("missing", "nan", "invalid")) for i in issues)
    # duplicates/gaps alone don't hard-fail, but are surfaced
    if issues and ok and any(i.startswith(("duplicate", "missing_bars")) for i in issues):
        ok = True
    return QualityReport(ok=ok, stale=stale, issues=issues, bar_timestamp=bar_ts, age_seconds=age)
