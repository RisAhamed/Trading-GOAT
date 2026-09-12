"""Unit: broker is paper-only + idempotent; confluence explains accept/reject."""
from unittest.mock import MagicMock

from domain.models import Action, AIAnalysis, FeatureSnapshot, OrderIntent, Regime
from execution.broker import AlpacaPaperBroker, PaperOnlyError
from strategy.confluence import evaluate


def test_paper_assertion():
    ex = MagicMock()
    ex.config.bot.mode = "live"
    try:
        AlpacaPaperBroker(ex)
        assert False, "must refuse live"
    except PaperOnlyError:
        pass


def test_duplicate_intent_blocked(tmp_path):
    ex = MagicMock()
    ex.config.bot.mode = "paper"
    b = AlpacaPaperBroker(ex, dry_run=True, store_path=tmp_path / "intents.json")
    intent = OrderIntent("id1", "c1", "s1", "BTC/USD", "buy", 1.0)
    r1 = b.submit(intent)
    r2 = b.submit(intent)
    assert r1.status == "DRY_RUN"
    assert r2.status == "REJECTED" and "duplicate" in r2.message


def test_confluence_reasons():
    ai = AIAnalysis("BTC/USD", Action.BUY, 0.8)
    trend = FeatureSnapshot("BTC/USD", "5Min", __import__("datetime").datetime.now(__import__("datetime").timezone.utc),
                            overall_trend="BULLISH", ema_short=2, ema_long=1, adx=25)
    entry = FeatureSnapshot("BTC/USD", "1Min", trend.timestamp, rsi=50, macd_histogram=1.0,
                            bb_percent=0.5, volume_ratio=1.5)
    sig = evaluate("BTC/USD", ai, trend, entry, Regime.TRENDING_BULLISH, 0.5, False)
    assert sig.final_action == Action.BUY and sig.confirmations
    bad = evaluate("BTC/USD", ai, trend, entry, Regime.TRENDING_BULLISH, 0.95, False)
    assert bad.final_action == Action.HOLD and bad.rejections
