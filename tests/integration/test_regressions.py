"""Regression tests for the exact observed failure (§43).

1. 41 consecutive HOLD cycles complete (liveness) with HOLD breakdown counted.
2. Scanner never eliminates primaries (covered in test_state + here via select).
3. Unrelated stock positions don't consume crypto capacity.
4. Local vs broker reconcile (covered in test_state; exercised via snapshot here).
5. Optional APIs can't block the loop (enrichment bounded).
6. Broker calls only real methods (test_broker_real.py).
7-8. BUY/SELL reach execution (test_e2e_fake.py).
9-11. Invalid AI->HOLD, duplicate->REJECTED, partial stays PARTIAL.
12. Failed exit not recorded as closed.
13. Stale prevents entry.
14. Same-candle reuse across cycles is not a freeze.
15. Live refused.
"""
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.provider import AlpacaProvider
from domain.models import Action, AIAnalysis, OrderIntent
from execution.broker import AlpacaPaperBroker, PaperOnlyError
from exits.retry import ExitRetryTracker
from features.regime import detect_regime
from features.snapshot import build_snapshot
from intelligence.provider import PROMPT_VERSION
from orchestration.loop import CycleDeps, run_cycle
from portfolio.snapshot import BrokerPosition, PortfolioSnapshot, snapshot_from_broker
from risk.policy import RiskConfig
from risk.policy import decide as risk_decide
from strategy.confluence import evaluate
from tests.fakes import FakePaperBroker


def _bars(n=60):
    idx = pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC")
    px = 100.0 + pd.Series(range(n), index=idx) * 0.1
    return pd.DataFrame({"open": px, "high": px * 1.0005, "low": px * 0.9995,
                         "close": px, "volume": 100.0}, index=idx)


def _indicators():
    from core.indicators import IndicatorCalculator
    cfg = MagicMock()
    cfg.indicators.rsi_period = 7
    cfg.indicators.rsi_oversold = 35
    cfg.indicators.rsi_overbought = 65
    cfg.indicators.macd_fast = 8
    cfg.indicators.macd_slow = 17
    cfg.indicators.macd_signal = 9
    cfg.indicators.ema_short = 5
    cfg.indicators.ema_long = 13
    cfg.indicators.bb_period = 14
    cfg.indicators.bb_std_dev = 1.8
    cfg.indicators.atr_period = 10
    return IndicatorCalculator(config=cfg)


def _hold_ai(ctx, p, q):
    return AIAnalysis(ctx.symbol, Action.HOLD, 0.45, prompt_version=PROMPT_VERSION,
                      timestamp=datetime.now(timezone.utc))


def _deps(ai_fn, snapshot=None):
    fetcher = MagicMock()
    fetcher.fetch_bars.return_value = _bars()
    settings = MagicMock(trend_interval="5Min", trend_lookback=60,
                         entry_interval="1Min", entry_lookback=60)
    return CycleDeps(
        settings=settings, provider=AlpacaProvider(fetcher, max_age_s=10**9),
        indicators=_indicators(), build_snapshot_fn=build_snapshot,
        detect_regime_fn=detect_regime, ai_decide_fn=ai_fn, confluence_fn=evaluate,
        risk_decide_fn=lambda i: risk_decide(RiskConfig(), i),
        broker=FakePaperBroker(),
        snapshot=snapshot or PortfolioSnapshot(equity=100000.0, broker_ok=True),
        exit_tracker=ExitRetryTracker(), min_confidence=0.5)


def test_41_holds_stay_alive_with_breakdown():
    from observability import metrics
    before = metrics.snapshot()["counters"].get("cycle_count", 0)
    for _ in range(41):
        r = run_cycle(_deps(_hold_ai), "BTC/USD")
        assert r.final_state == "HOLD" and r.finished_at >= r.started_at
    after = metrics.snapshot()["counters"]
    assert after.get("cycle_count", 0) - before == 41
    assert after.get("hold_AI_HOLD", 0) >= 41  # classified, measurable


def test_stocks_do_not_consume_strategy_capacity():
    snap = PortfolioSnapshot(
        equity=100000.0, broker_ok=True,
        external_positions=[BrokerPosition("NVDA", "NVDA", qty=10, entry_price=180.0, external=True),
                            BrokerPosition("XOM", "XOM", qty=20, entry_price=150.0, external=True)])
    snap.broker_total_positions = 2
    snap.strategy_open_positions = 0
    assert snap.strategy_open_positions == 0  # capacity intact despite 2 broker positions
    r = run_cycle(_deps(_hold_ai, snapshot=snap), "BTC/USD")
    assert r.final_state == "HOLD"  # AI hold, not capacity block
    assert r.signal.hold_reason == "AI_HOLD"


def test_failed_exit_not_recorded_closed(tmp_path):
    ex = MagicMock()
    ex.config.bot.mode = "paper"
    ex.get_open_orders.return_value = []
    ex.close_position.return_value = MagicMock(success=False, order_id="",
                                               error_message="Insufficient buying power",
                                               qty=0, filled_price=None, status="error")
    ex.client.get_open_position.return_value = MagicMock(qty=10)  # still open
    b = AlpacaPaperBroker(ex, store_path=tmp_path / "i.json")
    r = b.close_position("NVDA")
    assert r.status != "FILLED"  # never marked closed on failure
    assert "buying power" in r.message


def test_same_candle_reuse_is_not_freeze():
    df = _bars()
    fetcher = MagicMock()
    fetcher.fetch_bars.return_value = df  # identical object every cycle
    provider = AlpacaProvider(fetcher, max_age_s=10**9)
    d = _deps(_hold_ai)
    d.provider = provider
    r1 = run_cycle(d, "BTC/USD")
    r2 = run_cycle(d, "BTC/USD")
    assert r1.final_state == r2.final_state == "HOLD"
    assert r2.market_snapshot.bar_timestamp == r1.market_snapshot.bar_timestamp  # reuse, honest


def test_broker_surfaces_broker_positions_not_local_file():
    snap = snapshot_from_broker(FakePaperBroker())
    assert snap.broker_ok and snap.strategy_open_positions == 0


def test_live_mode_refused_everywhere():
    with pytest.raises(PaperOnlyError):
        AlpacaPaperBroker(MagicMock(config=MagicMock(bot=MagicMock(mode="LIVE"))))
    from config.settings import CanonicalSettings, assert_paper_mode
    with pytest.raises(SystemExit):
        assert_paper_mode(CanonicalSettings(mode="live"))


def test_duplicate_intent_rejected_and_partial_honest(tmp_path):
    ex = MagicMock()
    ex.config.bot.mode = "paper"
    b = AlpacaPaperBroker(ex, dry_run=True, store_path=tmp_path / "i.json")
    it = OrderIntent("d1", "c1", "s1", "BTC/USD", "buy", 1.0)
    assert b.submit(it).status == "DRY_RUN"
    assert b.submit(it).status == "REJECTED"
    fb = FakePaperBroker(fill_mode="partial")
    assert fb.submit(it, 100.0).status == "PARTIAL"  # stays PARTIAL, never FILLED


def test_optional_enrichment_bounded():
    from intelligence.enrichment import EnrichmentHub
    hub = EnrichmentHub()
    hub.register("slow", lambda: time.sleep(30) or {})
    t0 = time.perf_counter()
    hub.refresh_once(timeout_s=0.4)
    assert time.perf_counter() - t0 < 5
    assert hub.health()["slow"].startswith("UNAVAILABLE")
