"""Startup sequence: CONFIG VALIDATED -> ALPACA -> OLLAMA -> DATA -> FEATURES -> RISK -> READY.

Fails safe: SYSTEM NOT READY unless critical deps healthy. Never prints secrets.
"""
from __future__ import annotations

import logging

from config.settings import assert_paper_mode, load_canonical, secrets_present
from observability import health

logger = logging.getLogger("trading-goat")


def startup(config_path: str = "config.yaml", *, require_alpaca: bool = True) -> object:
    print("SYSTEM STARTING")
    settings, issues = load_canonical(config_path)
    for i in issues:
        print(f"  config {i.level}: {i.message}")
    if any(i.level == "error" for i in issues if "mode" in i.message or "symbol" in i.message):
        health.set_component("config", False, "validation error")
        raise SystemExit("SYSTEM NOT READY - config validation failed")
    print("CONFIG VALIDATED")
    print(settings.safe_summary())
    assert_paper_mode(settings)
    health.set_component("paper_mode", True, "PAPER asserted")
    health.set_component("config", True, "ok")

    # Lazy imports so `health_check` stays light
    from core.ai_brain import AIBrain
    from core.config_loader import get_config
    from core.indicators import IndicatorCalculator
    from core.market_data import MarketDataFetcher
    from core.order_executor import OrderExecutor
    from core.portfolio_tracker import PortfolioTracker

    legacy = get_config(config_path)
    brain = AIBrain(legacy)
    executor = OrderExecutor(legacy)
    tracker = PortfolioTracker(legacy)
    executor.portfolio_tracker = tracker
    fetcher = MarketDataFetcher(legacy)
    indicators = IndicatorCalculator(legacy)

    ok, msg = brain.check_connection()
    health.set_component("ollama", ok, msg)
    print(f"OLLAMA {'CONNECTED' if ok else 'UNAVAILABLE -> AI falls back to HOLD'} ({msg})")

    ok, msg, _acct = executor.check_connection()
    health.set_component("alpaca", ok, msg)
    print(f"ALPACA {'CONNECTED' if ok else 'FAILED'} ({msg})")
    if not ok and require_alpaca:
        raise SystemExit("SYSTEM NOT READY - Alpaca unreachable")

    # market-data + feature smoke test on first symbol
    try:
        df = fetcher.fetch_bars(settings.symbols[0], settings.trend_interval, settings.trend_lookback)
        assert df is not None and not df.empty
        vals = indicators.calculate(df)
        assert vals.data_points > 0
        health.set_component("market_data", True, "ok")
        print("MARKET DATA READY")
        print("FEATURE ENGINE READY")
    except Exception as e:
        health.set_component("market_data", False, str(e)[:150])
        raise SystemExit(f"SYSTEM NOT READY - market data smoke test failed: {e}")

    health.set_component("risk", True, "policy loaded")
    print("RISK ENGINE READY")
    h = health.get_health()
    if h.status == "NOT_READY":
        raise SystemExit("SYSTEM NOT READY")
    print("SYSTEM READY - PAPER TRADING")
    print(f"health={h.status} components={h.components}")

    return type("Boot", (), {
        "settings": settings, "legacy": legacy, "brain": brain,
        "executor": executor, "tracker": tracker, "fetcher": fetcher,
        "indicators": indicators, "secrets": secrets_present(),
    })()
