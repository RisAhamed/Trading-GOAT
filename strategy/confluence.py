"""Explainable signal validation: hard safety gates + soft evidence scoring.

Pipeline: AI_PROPOSAL -> TECHNICAL_EVIDENCE -> SIGNAL_DECISION (-> RISK ->
EXECUTION downstream). Weak evidence adjusts the score; only genuine
safety items veto:
  hard gates: invalid AI schema, low confidence, stale/invalid features,
              conflicting open position, extreme-volatility regime block.
  soft evidence (score): HTF trend alignment, RSI zone, MACD histogram side,
              Bollinger %B position, ADX strength, volume participation,
              regime alignment.

Every HOLD carries a machine-readable HOLD_REASON (AI_HOLD,
AI_LOW_CONFIDENCE, AI_INVALID, TECHNICAL_VETO, REGIME_VETO,
POSITION_EXISTS, STALE_DATA, INVALID_FEATURES) counted in metrics.
"""
from __future__ import annotations

from domain.models import Action, AIAnalysis, FeatureSnapshot, Regime, SignalDecision

# Soft-evidence weights (sum of positives ~= 5.0 before confidence term).
W_HTF_ALIGN = 1.5
W_RSI_ZONE = 1.0
W_MACD_SIDE = 0.75
W_BB_POSITION = 0.5
W_ADX_STRENGTH = 0.75
W_VOLUME = 0.5
W_REGIME_ALIGN = 0.75
ACCEPT_SCORE = 2.5  # proposal needs at least this evidence score to stand


def _rsi_supports(action: Action, rsi: float) -> tuple[bool, str]:
    if action == Action.BUY:
        return (rsi <= 70, f"rsi {rsi:.1f} not overbought") if rsi <= 70 else (False, f"rsi overbought {rsi:.1f}")
    if action == Action.SELL:
        return (rsi >= 30, f"rsi {rsi:.1f} not oversold") if rsi >= 30 else (False, f"rsi oversold {rsi:.1f}")
    return False, "no_directional_action"


def evaluate(
    symbol: str,
    ai: AIAnalysis,
    trend: FeatureSnapshot,
    entry: FeatureSnapshot,
    regime: Regime,
    min_confidence: float,
    has_position: bool,
    features_valid: bool = True,
    market_stale: bool = False,
) -> SignalDecision:
    confs: list[str] = []
    rejects: list[str] = []
    score = 0.0

    # ---- hard gates (safety-critical, deterministic veto) ----
    if market_stale:
        return SignalDecision(symbol, Action.HOLD, ai.action, ai.confidence, confs,
                              ["stale_market_data -> no new entry"], 0.0, hold_reason="STALE_DATA")
    if not features_valid:
        return SignalDecision(symbol, Action.HOLD, ai.action, ai.confidence, confs,
                              ["invalid_features -> no new entry"], 0.0, hold_reason="INVALID_FEATURES")
    if ai.raw_invalid:
        return SignalDecision(symbol, Action.HOLD, ai.action, 0.0, confs,
                              [f"ai_invalid:{ai.failure_reason or 'schema'} -> HOLD"],
                              0.0, hold_reason="AI_INVALID")
    if ai.action == Action.HOLD or ai.action == Action.CLOSE:
        reason = "AI_HOLD" if ai.action == Action.HOLD else "AI_CLOSE"
        return SignalDecision(symbol, ai.action, ai.action, ai.confidence, confs,
                              [f"ai_proposed_{ai.action.value} conf={ai.confidence:.2f}"],
                              round(ai.confidence, 2), hold_reason=reason)
    if ai.confidence < min_confidence:
        return SignalDecision(symbol, Action.HOLD, ai.action, ai.confidence, confs,
                              [f"ai_confidence {ai.confidence:.2f} < min {min_confidence:.2f}"],
                              round(ai.confidence, 2), hold_reason="AI_LOW_CONFIDENCE")
    if has_position:
        return SignalDecision(symbol, Action.HOLD, ai.action, ai.confidence, confs,
                              ["strategy_position_exists: manage exit, no new entry"],
                              round(ai.confidence, 2), hold_reason="POSITION_EXISTS")
    if regime == Regime.HIGH_VOLATILITY:
        return SignalDecision(symbol, Action.HOLD, ai.action, ai.confidence, confs,
                              [f"regime_{regime.value}: new entries blocked"],
                              round(ai.confidence, 2), hold_reason="REGIME_VETO")

    # ---- soft evidence (explainable score) ----
    want_bull = ai.action == Action.BUY
    htf_ok = (trend.overall_trend == "BULLISH") if want_bull else (trend.overall_trend == "BEARISH")
    if htf_ok:
        score += W_HTF_ALIGN
        confs.append(f"htf_{trend.overall_trend.lower()}_aligned")
    else:
        rejects.append(f"htf_mismatch ({trend.overall_trend}, need {'BULLISH' if want_bull else 'BEARISH'})")

    rsi_ok, rsi_msg = _rsi_supports(ai.action, entry.rsi)
    if rsi_ok:
        score += W_RSI_ZONE
        confs.append(rsi_msg)
    else:
        rejects.append(rsi_msg)

    macd_ok = (entry.macd_histogram >= 0) if want_bull else (entry.macd_histogram <= 0)
    if macd_ok:
        score += W_MACD_SIDE
        confs.append(f"macd_hist_{'pos' if entry.macd_histogram >= 0 else 'neg'}")
    else:
        rejects.append("macd_hist_wrong_side")

    bb_mid = 0.2 <= entry.bb_percent <= 0.8
    bb_edge_ok = (entry.bb_percent <= 0.35) if want_bull else (entry.bb_percent >= 0.65)
    if bb_mid or bb_edge_ok:
        score += W_BB_POSITION
        confs.append(f"bb_%B={entry.bb_percent:.2f}")
    else:
        rejects.append(f"bb_stretched_%B={entry.bb_percent:.2f}")

    if trend.adx >= 20:
        score += W_ADX_STRENGTH
        confs.append(f"adx_trending_{trend.adx:.0f}")
    else:
        rejects.append(f"adx_chop_{trend.adx:.0f}")

    if entry.volume_ratio >= 1.0:
        score += W_VOLUME
        confs.append(f"volume_x{entry.volume_ratio:.2f}")
    else:
        rejects.append(f"volume_thin_x{entry.volume_ratio:.2f}")

    regime_aligned = (regime == Regime.TRENDING_BULLISH and want_bull) or \
                     (regime == Regime.TRENDING_BEARISH and not want_bull) or \
                     regime in (Regime.SIDEWAYS, Regime.LOW_VOLATILITY, Regime.TRANSITION, Regime.UNKNOWN)
    if regime_aligned:
        score += W_REGIME_ALIGN
        confs.append(f"regime_{regime.value.lower()}_ok")
    else:
        rejects.append(f"regime_counter_{regime.value}")

    score += ai.confidence * 2.0
    score = round(score, 2)
    if score >= ACCEPT_SCORE:
        confs.append(f"evidence_score {score} >= {ACCEPT_SCORE}")
        return SignalDecision(symbol, ai.action, ai.action, ai.confidence, confs, rejects, score,
                              hold_reason="")
    rejects.append(f"evidence_score {score} < {ACCEPT_SCORE} -> HOLD")
    return SignalDecision(symbol, Action.HOLD, ai.action, ai.confidence, confs, rejects, score,
                          hold_reason="TECHNICAL_VETO")
