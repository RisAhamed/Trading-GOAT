"""Unit: confluence scoring, hard gates, HOLD reasons."""
from datetime import datetime, timezone

from domain.models import Action, AIAnalysis, FeatureSnapshot, Regime
from strategy.confluence import ACCEPT_SCORE, evaluate


def _ai(action="BUY", conf=0.8):
    return AIAnalysis("BTC/USD", Action(action), conf)


def _snaps(trend="BULLISH", rsi=55.0, hist=1.0, bb=0.5, adx=25.0, vol=1.5):
    now = datetime.now(timezone.utc)
    t = FeatureSnapshot("BTC/USD", "5Min", now, overall_trend=trend,
                        ema_short=2, ema_long=1, adx=adx, macd_histogram=hist)
    e = FeatureSnapshot("BTC/USD", "1Min", now, rsi=rsi, macd_histogram=hist,
                        bb_percent=bb, volume_ratio=vol)
    return t, e


def test_strong_setup_accepts():
    t, e = _snaps()
    s = evaluate("BTC/USD", _ai(), t, e, Regime.TRENDING_BULLISH, 0.5, False)
    assert s.final_action == Action.BUY and s.hold_reason == "" and s.score >= ACCEPT_SCORE
    assert s.confirmations


def test_hard_gates():
    t, e = _snaps()
    assert evaluate("BTC/USD", _ai(conf=0.4), t, e, Regime.TRENDING_BULLISH, 0.5, False).hold_reason == "AI_LOW_CONFIDENCE"
    assert evaluate("BTC/USD", _ai(), t, e, Regime.TRENDING_BULLISH, 0.5, True).hold_reason == "POSITION_EXISTS"
    assert evaluate("BTC/USD", _ai(), t, e, Regime.HIGH_VOLATILITY, 0.5, False).hold_reason == "REGIME_VETO"
    bad = AIAnalysis("BTC/USD", Action.BUY, 0.0, raw_invalid=True, failure_reason="x")
    assert evaluate("BTC/USD", bad, t, e, Regime.SIDEWAYS, 0.5, False).hold_reason == "AI_INVALID"
    assert evaluate("BTC/USD", _ai(), t, e, Regime.SIDEWAYS, 0.5, False, market_stale=True).hold_reason == "STALE_DATA"


def test_weak_evidence_vetoes_without_hard_gate():
    t, e = _snaps(trend="BEARISH", rsi=80.0, hist=-2.0, bb=0.95, adx=8.0, vol=0.3)
    s = evaluate("BTC/USD", _ai(), t, e, Regime.SIDEWAYS, 0.5, False)
    assert s.final_action == Action.HOLD and s.hold_reason == "TECHNICAL_VETO"
    assert s.rejections  # explainable, not silent


def test_ai_hold_passes_through():
    t, e = _snaps()
    s = evaluate("BTC/USD", _ai("HOLD", 0.4), t, e, Regime.SIDEWAYS, 0.5, False)
    assert s.final_action == Action.HOLD and s.hold_reason == "AI_HOLD"
