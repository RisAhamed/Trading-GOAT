"""Structured real-time trading cycle.

MARKET DATA -> quality -> FEATURES -> REGIME -> AI CONTEXT -> AI DECISION
-> SIGNAL CONFLUENCE -> RISK -> PAPER EXECUTION -> EXITS -> METRICS/EVENTS
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from domain.models import (
    Action,
    EntryContext,
    MarketContext,
    MarketSnapshot,
    OrderIntent,
    RiskDecision,
    SignalDecision,
    TradingCycleResult,
    TrendContext,
)
from observability import events as bus
from observability import metrics


@dataclass
class CycleDeps:
    settings: object
    provider: object  # data.provider.AlpacaProvider
    indicators: object  # legacy IndicatorCalculator
    build_snapshot_fn: object
    detect_regime_fn: object
    ai_decide_fn: object  # (ctx, pos_desc, port_desc) -> AIAnalysis
    confluence_fn: object
    risk_decide_fn: object  # (inputs) -> RiskDecision
    broker: object
    portfolio_fn: object  # () -> PortfolioState
    tracker: object = None  # legacy portfolio tracker for position lookup
    min_confidence: float = 0.5


def _ctx_from_snapshots(symbol: str, price: float, trend, entry, regime) -> MarketContext:
    trend_ctx = TrendContext(
        direction=trend.overall_trend if trend.overall_trend in ("BULLISH", "BEARISH") else "SIDEWAYS",
        strength="STRONG" if trend.adx >= 25 else ("MODERATE" if trend.adx >= 20 else "WEAK"),
        ema_trend="BULLISH" if trend.ema_short > trend.ema_long else ("BEARISH" if trend.ema_short < trend.ema_long else "NEUTRAL"),
        macd_trend="BULLISH" if trend.macd_histogram > 0 else ("BEARISH" if trend.macd_histogram < 0 else "NEUTRAL"),
        adx=trend.adx,
        reasons=[f"adx={trend.adx:.1f}", f"ema {trend.ema_short:.1f} vs {trend.ema_long:.1f}", f"htf={trend.overall_trend}"],
    )
    entry_ctx = EntryContext(
        direction="BULLISH" if entry.macd_histogram > 0 and entry.rsi > 50 else ("BEARISH" if entry.macd_histogram < 0 and entry.rsi < 50 else "SIDEWAYS"),
        rsi=entry.rsi,
        rsi_state="OVERSOLD" if entry.rsi <= 35 else ("OVERBOUGHT" if entry.rsi >= 65 else "NEUTRAL"),
        macd_histogram_rising=entry.macd_histogram > 0,
        bb_percent=entry.bb_percent,
        volume_ratio=entry.volume_ratio,
        reasons=[f"rsi={entry.rsi:.1f}", f"%B={entry.bb_percent:.2f}", f"vol_x{entry.volume_ratio:.2f}"],
    )
    return MarketContext(symbol=symbol, price=price, trend=trend_ctx, entry=entry_ctx,
                         regime=regime, atr_percent=entry.atr_percent)


def run_cycle(deps: CycleDeps, symbol: str) -> TradingCycleResult:
    cycle_id = f"{datetime.now(timezone.utc):%Y%m%d%H%M%S}-{uuid.uuid4().hex[:6]}"
    started = datetime.now(timezone.utc)
    res = TradingCycleResult(cycle_id=cycle_id, symbol=symbol, started_at=started)
    lat: dict[str, float] = {}
    metrics.inc("cycle_count")

    def timed(name: str, fn, *a, **k):
        t0 = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            ms = (time.perf_counter() - t0) * 1000.0
            lat[name] = ms
            metrics.observe_latency(name, ms)

    try:
        # 1. market data
        bars = timed("market_data_ms", deps.provider.fetch_bars, symbol,
                     deps.settings.trend_interval, deps.settings.trend_lookback)
        res.market_snapshot = MarketSnapshot(symbol=symbol, price=float(bars.df["close"].iloc[-1]),
                                             timestamp=datetime.now(timezone.utc),
                                             bar_timestamp=bars.quality.bar_timestamp,
                                             timeframe=deps.settings.trend_interval,
                                             bars=len(bars.df), stale=bars.quality.stale,
                                             issues=list(bars.quality.issues))
        bus.publish("MARKET_DATA_UPDATED", {"symbol": symbol, "cycle_id": cycle_id, "stale": res.market_snapshot.stale})
        metrics.inc("stale_market_data_events" if res.market_snapshot.stale else "fresh_market_data_events")
        if not bars.quality.ok:
            res.errors.append(f"market_quality: {bars.quality.issues}")
            res.final_state = "HOLD"
            res.latency_ms = lat
            return res  # uncertain -> do not trade
        entry_bars = timed("market_data_entry_ms", deps.provider.fetch_bars, symbol,
                           deps.settings.entry_interval, deps.settings.entry_lookback)

        # 2. features
        tvals_trend = timed("feature_ms", deps.indicators.calculate, bars.df)
        tvals_entry = deps.indicators.calculate(entry_bars.df)
        trend_snap = deps.build_snapshot_fn(symbol, deps.settings.trend_interval, bars.df, tvals_trend)
        entry_snap = deps.build_snapshot_fn(symbol, deps.settings.entry_interval, entry_bars.df, tvals_entry)
        res.features = entry_snap
        if not trend_snap.valid or not entry_snap.valid:
            res.errors.append(f"feature_quality: {trend_snap.quality_issues + entry_snap.quality_issues}")
            res.final_state = "HOLD"
            res.latency_ms = lat
            return res
        bus.publish("FEATURES_UPDATED", {"symbol": symbol, "cycle_id": cycle_id})

        # 3. regime
        regime, _reasons = deps.detect_regime_fn(trend_snap, entry_snap)
        res.regime = regime

        # 4-5. AI context + decision
        ctx = _ctx_from_snapshots(symbol, res.market_snapshot.price, trend_snap, entry_snap, regime)
        port = deps.portfolio_fn()
        positions = deps.tracker.get_positions() if deps.tracker else {}
        has_pos = symbol in (positions or {})
        ai = timed("llm_ms", deps.ai_decide_fn, ctx,
                   f"open={has_pos}", f"equity={port.equity:.0f} exposure={port.total_exposure_notional:.0f}")
        res.ai = ai
        bus.publish("AI_DECISION_CREATED", {"symbol": symbol, "cycle_id": cycle_id,
                                            "action": ai.action.value, "confidence": ai.confidence,
                                            "model": ai.model_used, "fallback": ai.fallback_used})
        metrics.observe_latency("llm_ms", ai.latency_ms)

        # 6. signal confluence
        sig: SignalDecision = timed("signal_ms", deps.confluence_fn, symbol, ai, trend_snap, entry_snap,
                                    regime, deps.min_confidence, has_pos)
        res.signal = sig
        metrics.inc("buy_signals" if sig.final_action == Action.BUY else
                    "sell_signals" if sig.final_action == Action.SELL else "hold_signals")
        if sig.accepted:
            bus.publish("SIGNAL_GENERATED", {"symbol": symbol, "cycle_id": cycle_id, "action": sig.final_action.value})
        else:
            bus.publish("SIGNAL_REJECTED", {"symbol": symbol, "cycle_id": cycle_id, "reasons": sig.rejections})
            metrics.inc("rejected_signals")
            res.final_state = "HOLD"
            res.latency_ms = lat
            return res

        # 7. risk
        from risk.policy import RiskInputs
        risk_in = RiskInputs(symbol=symbol, action=sig.final_action, price=res.market_snapshot.price,
                             atr=entry_snap.atr, portfolio_value=port.equity or 100000.0,
                             open_positions=port.open_positions,
                             symbol_exposure_notional=port.symbol_exposure.get(symbol, 0.0),
                             total_exposure_notional=port.total_exposure_notional,
                             daily_pnl_pct=port.daily_pnl_pct)
        risk: RiskDecision = timed("risk_ms", deps.risk_decide_fn, risk_in)
        res.risk = risk
        if not risk.allowed:
            bus.publish("RISK_REJECTED", {"symbol": symbol, "cycle_id": cycle_id, "reason": risk.rejection_reason})
            metrics.inc("rejected_signals")
            res.final_state = "HOLD"
            res.latency_ms = lat
            return res
        bus.publish("RISK_APPROVED", {"symbol": symbol, "cycle_id": cycle_id, "qty": risk.position_size})
        metrics.inc("risk_approved")

        # 8. execution (idempotent intent)
        intent = OrderIntent(intent_id=f"{cycle_id}-{symbol}", cycle_id=cycle_id, signal_id=cycle_id,
                             symbol=symbol, side="buy" if sig.final_action == Action.BUY else "sell",
                             qty=risk.position_size, stop_price=risk.stop_price,
                             take_profit_price=risk.take_profit_price)
        bus.publish("ORDER_SUBMITTED", {"symbol": symbol, "cycle_id": cycle_id, "qty": intent.qty})
        metrics.inc("orders_submitted")
        ex = timed("execution_ms", deps.broker.submit, intent)
        res.execution = ex
        if ex.status in ("FILLED", "PARTIAL"):
            bus.publish("ORDER_FILLED", {"symbol": symbol, "cycle_id": cycle_id, "status": ex.status})
            metrics.inc("orders_filled")
        else:
            metrics.inc("orders_failed")
            res.errors.append(f"execution_{ex.status}: {ex.message}")
        res.final_state = sig.final_action.value if ex.status in ("FILLED", "PARTIAL", "DRY_RUN") else "HOLD"
        res.latency_ms = lat
        return res
    except Exception as e:
        res.errors.append(f"cycle_exception: {e}"[:300])
        metrics.inc("api_errors")
        bus.publish("ERROR_OCCURRED", {"symbol": symbol, "cycle_id": cycle_id, "error": str(e)[:200]})
        res.final_state = "HOLD"
        res.latency_ms = lat
        return res
    finally:
        total = sum(lat.values())
        metrics.observe_latency("cycle_ms", total)
        bus.publish("CYCLE_COMPLETED", res.to_event())
