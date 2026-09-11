"""Explicit market-regime detector. Describes conditions; predicts nothing.

Inputs: EMA relationship, ADX, ATR%, price structure. Outputs Regime + reasons
shared with AI context and risk policy.
"""
from __future__ import annotations

from domain.models import FeatureSnapshot, Regime


def detect_regime(trend: FeatureSnapshot, entry: FeatureSnapshot) -> tuple[Regime, list[str]]:
    reasons: list[str] = []
    adx = trend.adx
    ema_stack_bull = trend.ema_short > trend.ema_long > 0
    ema_stack_bear = trend.ema_short < trend.ema_long
    atr_pct = max(trend.atr_percent, entry.atr_percent)

    if ema_stack_bull:
        reasons.append("ema_short>ema_long (bullish stack)")
    elif ema_stack_bear:
        reasons.append("ema_short<ema_long (bearish stack)")
    reasons.append(f"adx={adx:.1f}")
    reasons.append(f"atr%={atr_pct:.2f}")

    if atr_pct > 2.5:
        return Regime.HIGH_VOLATILITY, reasons + ["volatility_expanded"]
    if adx >= 20 and ema_stack_bull and trend.overall_trend == "BULLISH":
        return Regime.TRENDING_BULLISH, reasons
    if adx >= 20 and ema_stack_bear and trend.overall_trend == "BEARISH":
        return Regime.TRENDING_BEARISH, reasons
    if adx < 20 and atr_pct < 0.4:
        return Regime.LOW_VOLATILITY, reasons + ["compression"]
    if adx < 20:
        return Regime.SIDEWAYS, reasons + ["no_trend_strength"]
    return Regime.TRANSITION, reasons + ["mixed_signals"]
