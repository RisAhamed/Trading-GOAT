"""Integration: market->features->signal->risk->broker(dry-run) with fakes. No credentials."""
from unittest.mock import MagicMock

import pandas as pd

from data.provider import AlpacaProvider
from execution.broker import AlpacaPaperBroker
from features.regime import detect_regime
from features.snapshot import build_snapshot
from intelligence.schemas import SCHEMA_VERSION
from orchestration.loop import CycleDeps, run_cycle
from portfolio.state import PortfolioState
from risk.policy import RiskConfig
from risk.policy import decide as risk_decide
from strategy.confluence import evaluate


def _bars(n=60, px=50000.0):
    idx = pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC")
    df = pd.DataFrame({"open": px, "high": px * 1.001, "low": px * 0.999,
                       "close": px, "volume": 100.0}, index=idx)
    df["close"] = px + pd.Series(range(n), index=idx) * 2.0  # gentle uptrend
    df["high"] = df["close"] * 1.0005
    df["low"] = df["close"] * 0.9995
    return df


def test_pipeline_end_to_end_dry_run():
    fetcher = MagicMock()
    fetcher.fetch_bars.side_effect = [_bars(60), _bars(40)]
    provider = AlpacaProvider(fetcher, max_age_s=10**9)

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
    indicators = IndicatorCalculator(config=cfg)

    primary = MagicMock()
    primary.complete.return_value = MagicMock(
        text='{"action":"HOLD","confidence":0.4,"trend":"SIDEWAYS","entry_quality":"WEAK","reasoning_summary":"mixed","risk_notes":[]}',
        model_used="fake", fallback_used=False, latency_ms=5.0, failure_reason="")

    ex = MagicMock()
    ex.config.bot.mode = "paper"
    broker = AlpacaPaperBroker(ex, dry_run=True)

    settings = MagicMock()
    settings.trend_interval = "5Min"
    settings.trend_lookback = 60
    settings.entry_interval = "1Min"
    settings.entry_lookback = 40

    def ai_fn(ctx, p, q):
        from domain.models import AIAnalysis
        from intelligence.provider import PROMPT_VERSION
        from intelligence.schemas import parse_ai_json
        parsed = parse_ai_json(primary.complete("", 1).text)
        return AIAnalysis(ctx.symbol, parsed.action, parsed.confidence, prompt_version=PROMPT_VERSION)

    deps = CycleDeps(settings, provider, indicators, build_snapshot, detect_regime, ai_fn,
                     evaluate, lambda i: risk_decide(RiskConfig(), i), broker,
                     lambda: PortfolioState(equity=100000.0), tracker=MagicMock(get_positions=dict),
                     min_confidence=0.5)
    res = run_cycle(deps, "BTC/USD")
    assert res.final_state == "HOLD"  # HOLD proposal -> no order
    assert res.market_snapshot and res.features and res.ai and res.signal
    assert SCHEMA_VERSION == "1"


def test_stale_data_holds_without_trading():
    df = _bars(60)
    df.index = df.index - pd.Timedelta(days=2)  # force stale
    fetcher = MagicMock()
    fetcher.fetch_bars.return_value = df
    provider = AlpacaProvider(fetcher, max_age_s=60)
    deps = CycleDeps(MagicMock(trend_interval="5Min", trend_lookback=60, entry_interval="1Min", entry_lookback=40),
                     provider, MagicMock(), build_snapshot, detect_regime,
                     MagicMock(), evaluate, MagicMock(), MagicMock(dry_run=True),
                     lambda: PortfolioState(equity=100000.0), tracker=None, min_confidence=0.5)
    res = run_cycle(deps, "BTC/USD")
    assert res.final_state == "HOLD" and any("stale" in e or "market_quality" in e for e in res.errors)
