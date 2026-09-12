"""Deterministic end-to-end harness: fake broker + fake market + fake AI.

market -> feature -> regime -> AI BUY -> signal ACCEPT -> risk APPROVE ->
broker SUBMIT -> FILL -> portfolio update -> exit trigger -> CLOSE ->
reconcile -> ledger. No credentials.
"""
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.provider import AlpacaProvider
from domain.models import Action, AIAnalysis
from exits.engine import ExitCheck, ExitConfig
from exits.engine import check as exit_check
from features.regime import detect_regime
from features.snapshot import build_snapshot
from intelligence.provider import PROMPT_VERSION
from orchestration.loop import CycleDeps, run_cycle
from portfolio.snapshot import normalize_symbol, reconcile, snapshot_from_broker
from risk.policy import RiskConfig
from risk.policy import decide as risk_decide
from strategy.confluence import evaluate
from tests.fakes import FakePaperBroker


def _bars(n=80, px=100.0, drift=0.4):
    idx = pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC")
    close = px + pd.Series(range(n), index=idx) * drift  # steady uptrend
    return pd.DataFrame({"open": close, "high": close * 1.0005, "low": close * 0.9995,
                         "close": close, "volume": 200.0}, index=idx)


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


def _ai_buy(ctx, p, q):
    return AIAnalysis(ctx.symbol, Action.BUY, 0.75, trend="BULLISH",
                      entry_quality="STRONG", reasoning_summary="uptrend + volume",
                      model_requested="fake", model_used="fake",
                      prompt_version=PROMPT_VERSION,
                      timestamp=datetime.now(timezone.utc))


def _deps(broker, fetcher, settings, ai_fn=_ai_buy, equity=100000.0):
    snap = snapshot_from_broker(broker)
    return CycleDeps(
        settings=settings, provider=AlpacaProvider(fetcher, max_age_s=10**9),
        indicators=_indicators(), build_snapshot_fn=build_snapshot,
        detect_regime_fn=detect_regime, ai_decide_fn=ai_fn, confluence_fn=evaluate,
        risk_decide_fn=lambda i: risk_decide(RiskConfig(), i), broker=broker,
        snapshot=snap, min_confidence=0.5)


def test_full_journey_buy_fill_exit_close():
    broker = FakePaperBroker()
    fetcher = MagicMock()
    fetcher.fetch_bars.side_effect = [_bars(), _bars()]
    settings = MagicMock(trend_interval="5Min", trend_lookback=80,
                         entry_interval="1Min", entry_lookback=80)
    res = run_cycle(_deps(broker, fetcher, settings), "BTC/USD")
    assert res.signal and res.signal.accepted, res.signal.rejections if res.signal else res.errors
    assert res.risk and res.risk.allowed, res.risk.rejection_reason if res.risk else None
    assert res.execution.status == "FILLED", res.execution.message
    assert res.final_state == "BUY"

    # portfolio reflects broker truth
    snap = snapshot_from_broker(broker)
    assert snap.strategy_open_positions == 1
    assert normalize_symbol("BTCUSD") == "BTC/USD"

    # exit trigger -> close -> reconcile flat
    pos = snap.strategy_positions[0]
    reason, _ = exit_check(
        ExitConfig(max_hold_seconds=0),
        ExitCheck("BTC/USD", "long", pos.entry_price, pos.current_price * 1.05,
                  pos.entry_price * 0.995, pos.entry_price * 1.01,
                  pos.current_price * 1.05, hold_s=99999, signal_action="HOLD"))
    assert reason.value in ("TAKE_PROFIT", "TIMEOUT")  # TP prioritized over timeout
    closed = broker.close_position("BTC/USD", reason=reason.value)
    assert closed.status == "FILLED"
    snap2 = snapshot_from_broker(broker)
    assert snap2.strategy_open_positions == 0
    rec = reconcile(snap2, {"BTC/USD": {"status": "OPEN"}})
    assert any("local_OPEN_but_broker_flat" in m for m in rec.mismatches)


def test_sell_reaches_broker_where_supported():
    broker = FakePaperBroker()

    def _ai_sell(ctx, p, q):
        return AIAnalysis(ctx.symbol, Action.SELL, 0.8, prompt_version=PROMPT_VERSION,
                          timestamp=datetime.now(timezone.utc))

    fetcher = MagicMock()
    fetcher.fetch_bars.side_effect = [_bars(drift=-0.4), _bars(drift=-0.4)]
    settings = MagicMock(trend_interval="5Min", trend_lookback=80,
                         entry_interval="1Min", entry_lookback=80)
    res = run_cycle(_deps(broker, fetcher, settings, ai_fn=_ai_sell), "BTC/USD")
    assert res.execution.status == "FILLED" and res.final_state == "SELL"


def test_rejected_order_no_position():
    broker = FakePaperBroker(fill_mode="reject")
    fetcher = MagicMock()
    fetcher.fetch_bars.side_effect = [_bars(), _bars()]
    settings = MagicMock(trend_interval="5Min", trend_lookback=80,
                         entry_interval="1Min", entry_lookback=80)
    res = run_cycle(_deps(broker, fetcher, settings), "BTC/USD")
    assert res.execution.status == "REJECTED"
    assert res.final_state == "HOLD"
    assert snapshot_from_broker(broker).strategy_open_positions == 0
