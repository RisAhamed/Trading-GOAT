"""Deterministic confluence: AI proposal vs technical validation -> final signal.

Separates confirmations from rejections for explainability.
"""
from __future__ import annotations

from domain.models import Action, AIAnalysis, FeatureSnapshot, Regime, SignalDecision


def evaluate(symbol: str, ai: AIAnalysis, trend: FeatureSnapshot, entry: FeatureSnapshot,
             regime: Regime, min_confidence: float, has_position: bool) -> SignalDecision:
    confs: list[str] = []
    rejects: list[str] = []
    score = 0.0

    if ai.raw_invalid:
        rejects.append(f"ai_invalid:{ai.failure_reason or 'schema'} -> HOLD")
        return SignalDecision(symbol, Action.HOLD, ai.action, 0.0, confs, rejects, 0.0)
    if ai.confidence < min_confidence:
        rejects.append(f"ai_confidence {ai.confidence:.2f} < min {min_confidence:.2f}")

    # Trend alignment (higher TF)
    if ai.action == Action.BUY and trend.overall_trend == "BULLISH":
        confs.append("htf_trend_bullish"); score += 1
    elif ai.action == Action.SELL and trend.overall_trend == "BEARISH":
        confs.append("htf_trend_bearish"); score += 1
    elif ai.action in (Action.BUY, Action.SELL):
        rejects.append(f"htf_trend_mismatch ({trend.overall_trend})")

    # Entry momentum
    if ai.action == Action.BUY:
        if entry.rsi <= 65:
            confs.append(f"rsi_ok {entry.rsi:.1f}"); score += 1
        else:
            rejects.append(f"rsi_overbought {entry.rsi:.1f}")
        if entry.macd_histogram >= 0:
            confs.append("macd_hist>=0"); score += 0.5
        else:
            rejects.append("macd_hist<0")
    elif ai.action == Action.SELL:
        if entry.rsi >= 35:
            confs.append(f"rsi_ok {entry.rsi:.1f}"); score += 1
        else:
            rejects.append(f"rsi_oversold {entry.rsi:.1f}")
        if entry.macd_histogram <= 0:
            confs.append("macd_hist<=0"); score += 0.5
        else:
            rejects.append("macd_hist>0")

    # Regime guard
    if regime in (Regime.HIGH_VOLATILITY,) and ai.action in (Action.BUY, Action.SELL):
        rejects.append(f"regime_{regime.value}_caution")
    if regime in (Regime.TRENDING_BULLISH,) and ai.action == Action.SELL:
        rejects.append("regime_bullish_counter_sell")
    if regime in (Regime.TRENDING_BEARISH,) and ai.action == Action.BUY:
        rejects.append("regime_bearish_counter_buy")

    # Volume participation (confirm only)
    if entry.volume_ratio >= 1.0:
        confs.append(f"volume_x{entry.volume_ratio:.2f}"); score += 0.5
    else:
        rejects.append(f"volume_thin_x{entry.volume_ratio:.2f}")

    # Existing position: avoid doubling same direction blindly
    if has_position and ai.action in (Action.BUY, Action.SELL):
        rejects.append("position_exists review_exit_not_new_entry")

    if ai.action in (Action.BUY, Action.SELL):
        score += ai.confidence * 2.0
        fatal = [r for r in rejects if r.startswith(("ai_", "htf_trend_mismatch", "position_exists"))]
        final = ai.action if not fatal else Action.HOLD
        if final == Action.HOLD and ai.action != Action.HOLD:
            rejects.append(f"vetoed ai_{ai.action.value} -> HOLD")
    else:
        final = ai.action  # HOLD / CLOSE pass through (CLOSE handled by exit layer)
        score = ai.confidence

    return SignalDecision(symbol, final, ai.action, ai.confidence, confs, rejects, round(score, 2))
