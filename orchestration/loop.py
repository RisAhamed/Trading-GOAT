"""Canonical real-time trading cycle.

START CYCLE -> broker state (authoritative) -> market data -> freshness ->
features -> regime -> AI context -> AI decision -> signal evaluation ->
risk policy -> execution -> reconcile -> exits -> persist/events -> dashboard.

Every cycle: cycle_id, timestamps, signal/risk/execution results, latency,
errors, final_state, HOLD reason. Uncertain -> HOLD. Invalid AI -> HOLD.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

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
    settings: Any
    provider: Any            # data.provider.AlpacaProvider
    indicators: Any          # legacy IndicatorCalculator
    build_snapshot_fn: Any
    detect_regime_fn: Any
    ai_decide_fn: Any        # (ctx, pos_desc, port_desc) -> AIAnalysis
    confluence_fn: Any
    risk_decide_fn: Any
    broker: Any              # execution.broker.AlpacaPaperBroker
    snapshot: Any            # portfolio.snapshot.PortfolioSnapshot (per-cycle)
    universe_rows: list = None
    exit_tracker: Any = None  # exits.retry.ExitRetryTracker
    enrichment: dict = None   # optional-enrichment snapshot (neutral if absent)
    min_confidence: float = 0.5


def _ctx_from_snapshots(symbol: str, price: float, trend, entry, regime) -> MarketContext:
    trend_ctx = TrendContext(
        direction=trend.overall_trend if trend.overall_trend in ("BULLISH", "BEARISH") else "SIDEWAYS",
        strength="STRONG" if trend.adx >= 25 else ("MODERATE" if trend.adx >= 20 else "WEAK"),
        ema_trend="BULLISH" if trend.ema_short > trend.ema_long else ("BEARISH" if trend.ema_short < trend.ema_long else "NEUTRAL"),
        macd_trend="BULLISH" if trend.macd_histogram > 0 else ("BEARISH" if trend.macd_histogram < 0 else "NEUTRAL"),
        adx=trend.adx,
        reasons=[f"adx={trend.adx:.1f}", f"ema {trend.ema_short:.1f} vs {trend.ema_long:.1f}",
                 f"htf={trend.overall_trend}"],
    )
    entry_ctx = EntryContext(
        direction="BULLISH" if entry.macd_histogram > 0 and entry.rsi > 50 else (
            "BEARISH" if entry.macd_histogram < 0 and entry.rsi < 50 else "SIDEWAYS"),
        rsi=entry.rsi,
        rsi_state="OVERSOLD" if entry.rsi <= 35 else ("OVERBOUGHT" if entry.rsi >= 65 else "NEUTRAL"),
        macd_histogram_rising=entry.macd_histogram > 0,
        bb_percent=entry.bb_percent,
        volume_ratio=entry.volume_ratio,
        reasons=[f"rsi={entry.rsi:.1f}", f"%B={entry.bb_percent:.2f}", f"vol_x{entry.volume_ratio:.2f}"],
    )
    return MarketContext(symbol=symbol, price=price, trend=trend_ctx, entry=entry_ctx,
                         regime=regime, atr_percent=entry.atr_percent)


def _hold(deps: CycleDeps, res: TradingCycleResult, lat: dict, reason_metric: str) -> TradingCycleResult:
    res.final_state = "HOLD"
    res.latency_ms = lat
    metrics.inc("hold_total")
    metrics.inc(f"hold_{reason_metric}")
    return res


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

    snap = deps.snapshot
    strat_pos = next((p for p in (snap.strategy_positions or []) if p.symbol == symbol), None)
    has_position = strat_pos is not None
    signal_id = f"{cycle_id}-{symbol}"

    try:
        # 1-2. market data + freshness (unchanged bar is reuse, not staleness)
        bars = timed("market_data_ms", deps.provider.fetch_bars, symbol,
                     deps.settings.trend_interval, deps.settings.trend_lookback)
        res.market_snapshot = MarketSnapshot(
            symbol=symbol, price=float(bars.df["close"].iloc[-1]),
            timestamp=datetime.now(timezone.utc), bar_timestamp=bars.quality.bar_timestamp,
            timeframe=deps.settings.trend_interval, bars=len(bars.df),
            stale=bars.quality.stale, issues=list(bars.quality.issues))
        bus.publish("MARKET_DATA_UPDATED", {
            "symbol": symbol, "cycle_id": cycle_id, "stale": res.market_snapshot.stale,
            "is_new_bar": bars.is_new_bar, "bar_age_s": round(bars.bar_age_s, 1)})
        if bars.is_new_bar:
            metrics.inc("fresh_bars")
        else:
            metrics.inc("duplicate_bars")  # same valid candle reused: normal, not a freeze
        if any(i.startswith("missing_bars") for i in bars.quality.issues):
            metrics.inc("missing_bar_events")
        if not bars.quality.ok:
            metrics.inc("stale_bars")
            res.errors.append(f"market_quality: {bars.quality.issues}")
            res.signal = SignalDecision(
                symbol, Action.HOLD, Action.HOLD, 0.0, [],
                [f"market_quality: {bars.quality.issues}"], 0.0, hold_reason="STALE_DATA")
            bus.publish("SIGNAL_REJECTED", {"symbol": symbol, "cycle_id": cycle_id,
                                            "hold_reason": "STALE_DATA", "reasons": res.errors})
            return _hold(deps, res, lat, "STALE_DATA")
        entry_bars = timed("market_data_entry_ms", deps.provider.fetch_bars, symbol,
                           deps.settings.entry_interval, deps.settings.entry_lookback)

        # 3. features
        tvals_trend = timed("feature_ms", deps.indicators.calculate, bars.df)
        tvals_entry = deps.indicators.calculate(entry_bars.df)
        trend_snap = deps.build_snapshot_fn(
            symbol, deps.settings.trend_interval, bars.df, tvals_trend,
            min_bars=int(getattr(deps.settings, "trend_lookback", 30)))
        entry_snap = deps.build_snapshot_fn(
            symbol, deps.settings.entry_interval, entry_bars.df, tvals_entry,
            min_bars=int(getattr(deps.settings, "entry_lookback", 20)))
        res.features = entry_snap
        feats_ok = bool(trend_snap.valid and entry_snap.valid)
        if not feats_ok:
            res.errors.append(f"feature_quality: {trend_snap.quality_issues + entry_snap.quality_issues}")
        bus.publish("FEATURES_UPDATED", {"symbol": symbol, "cycle_id": cycle_id, "valid": feats_ok})

        # 4. regime
        regime, _reasons = deps.detect_regime_fn(trend_snap, entry_snap)
        res.regime = regime
        bus.publish("REGIME_DETECTED", {"symbol": symbol, "cycle_id": cycle_id, "regime": regime.value})

        # 5-6. AI context + decision (structured in, strict JSON out)
        ctx = _ctx_from_snapshots(symbol, res.market_snapshot.price, trend_snap, entry_snap, regime)
        pos_desc = (f"strategy_position qty={strat_pos.qty} entry={strat_pos.entry_price}" if strat_pos
                    else "no strategy position")
        port_desc = f"equity={snap.equity:.0f} strategy_pos={snap.strategy_open_positions} exposure={snap.total_exposure_notional:.0f}"
        ai = timed("llm_ms", deps.ai_decide_fn, ctx, pos_desc, port_desc)
        res.ai = ai
        bus.publish("AI_DECISION_CREATED", {
            "symbol": symbol, "cycle_id": cycle_id, "action": ai.action.value,
            "confidence": ai.confidence, "model_requested": ai.model_requested,
            "model_used": ai.model_used, "fallback_used": ai.fallback_used,
            "invalid": ai.raw_invalid})
        metrics.inc(f"ai_{ai.action.value.lower()}_proposals")
        if ai.raw_invalid:
            metrics.inc("ai_invalid_outputs")
        metrics.observe_latency("llm_ms", ai.latency_ms)

        # 7. signal evaluation (hard gates + scored evidence)
        sig: SignalDecision = timed(
            "signal_ms", deps.confluence_fn, symbol, ai, trend_snap, entry_snap,
            regime, deps.min_confidence, has_position,
            feats_ok, res.market_snapshot.stale)
        res.signal = sig
        if sig.accepted:
            metrics.inc("signals_accepted")
            bus.publish("SIGNAL_GENERATED", {"symbol": symbol, "cycle_id": cycle_id,
                                             "action": sig.final_action.value, "score": sig.score,
                                             "confirmations": sig.confirmations})
        else:
            metrics.inc("signals_rejected")
            bus.publish("SIGNAL_REJECTED", {"symbol": symbol, "cycle_id": cycle_id,
                                            "hold_reason": sig.hold_reason or "NO_VALID_SETUP",
                                            "reasons": sig.rejections})
            return _hold(deps, res, lat, sig.hold_reason or "NO_VALID_SETUP")

        # 8. risk (deterministic, AI cannot override)
        from risk.policy import RiskInputs

        risk_in = RiskInputs(
            symbol=symbol, action=sig.final_action, price=res.market_snapshot.price,
            atr=entry_snap.atr, portfolio_value=snap.equity or 100000.0,
            open_positions=snap.strategy_open_positions,  # strategy only (§10)
            symbol_exposure_notional=snap.symbol_exposure.get(symbol, 0.0),
            total_exposure_notional=snap.total_exposure_notional,
            daily_pnl_pct=snap.daily_pnl_pct)
        risk: RiskDecision = timed("risk_ms", deps.risk_decide_fn, risk_in)
        res.risk = risk
        if not risk.allowed:
            bus.publish("RISK_REJECTED", {"symbol": symbol, "cycle_id": cycle_id,
                                          "reason": risk.rejection_reason})
            return _hold(deps, res, lat, "RISK_LIMIT")
        bus.publish("RISK_APPROVED", {"symbol": symbol, "cycle_id": cycle_id,
                                      "qty": risk.position_size, "stop": risk.stop_price,
                                      "target": risk.take_profit_price})
        metrics.inc("risk_approved")

        # 9. execution (idempotent intent, verified fill)
        intent = OrderIntent(intent_id=f"{cycle_id}-{symbol}", cycle_id=cycle_id,
                             signal_id=signal_id, symbol=symbol,
                             side="buy" if sig.final_action == Action.BUY else "sell",
                             qty=risk.position_size, stop_price=risk.stop_price,
                             take_profit_price=risk.take_profit_price)
        bus.publish("ORDER_INTENT_CREATED", {"symbol": symbol, "cycle_id": cycle_id,
                                             "qty": intent.qty, "side": intent.side})
        bus.publish("ORDER_SUBMITTED", {"symbol": symbol, "cycle_id": cycle_id, "qty": intent.qty})
        metrics.inc("orders_submitted")
        ex = timed("execution_ms", deps.broker.submit_order, intent, res.market_snapshot.price)
        res.execution = ex
        if ex.status == "FILLED":
            bus.publish("ORDER_FILLED", {"symbol": symbol, "cycle_id": cycle_id,
                                         "qty": ex.filled_qty, "avg": ex.avg_fill_price})
            bus.publish("POSITION_OPENED", {"symbol": symbol, "cycle_id": cycle_id})
            metrics.inc("orders_filled")
        elif ex.status == "PARTIAL":
            bus.publish("ORDER_PARTIAL", {"symbol": symbol, "cycle_id": cycle_id,
                                          "qty": ex.filled_qty})
            metrics.inc("orders_partial")
        elif ex.status in ("REJECTED", "CANCELED"):
            bus.publish("ORDER_FAILED", {"symbol": symbol, "cycle_id": cycle_id,
                                         "status": ex.status, "message": ex.message})
            metrics.inc("orders_rejected")
            res.errors.append(f"execution_{ex.status}: {ex.message}")
        else:
            bus.publish("ORDER_FAILED", {"symbol": symbol, "cycle_id": cycle_id,
                                         "status": ex.status, "message": ex.message})
            metrics.inc("orders_failed")
            res.errors.append(f"execution_{ex.status}: {ex.message}")
        res.final_state = sig.final_action.value if ex.status in ("FILLED", "PARTIAL", "DRY_RUN") else "HOLD"
        if res.final_state == "HOLD":
            metrics.inc("hold_BROKER_ISSUE")
        res.latency_ms = lat
        return res
    except Exception as e:  # cycle boundary: fail safe to HOLD, record event
        res.errors.append(f"cycle_exception: {e}"[:300])
        metrics.inc("api_errors")
        bus.publish("ERROR_OCCURRED", {"symbol": symbol, "cycle_id": cycle_id, "error": str(e)[:200]})
        return _hold(deps, res, lat, "BROKER_ISSUE" if not res.signal else (res.signal.hold_reason or "NO_VALID_SETUP"))
    finally:
        metrics.observe_latency("cycle_ms", sum(lat.values()))
        bus.publish("CYCLE_COMPLETED", res.to_event())


def evaluate_exits_for_position(deps: CycleDeps, symbol: str, signal_action: str,
                                 cycle_id: str, max_hold_s: float) -> dict | None:
    """Exit check for one open strategy position with bounded retry.

    Returns an exit record dict on trigger attempt, else None. Failed closes
    record backoff state; broker-flat positions clear state (never marked
    closed merely because a request was sent).
    """
    from domain.models import ExitReason as _ExitReason
    from exits.engine import ExitCheck, ExitConfig
    from exits.engine import check as exit_check

    snap = deps.snapshot
    pos = next((p for p in (snap.strategy_positions or []) if p.symbol == symbol), None)
    if pos is None:
        return None
    broker_syms = {p.symbol for p in (snap.strategy_positions or [])} | \
                  {p.symbol for p in (snap.external_positions or [])}
    tracker = deps.exit_tracker
    if tracker is not None:
        if tracker.clear_if_flat(symbol, broker_syms) is True and symbol not in broker_syms:
            return None
        if not tracker.should_retry(symbol):
            metrics.inc("exit_retry_deferred")
            return {"symbol": symbol, "deferred": True,
                    "reason": "retry_backoff_pending"}
    price = pos.current_price or pos.entry_price
    if price <= 0 or pos.entry_price <= 0:
        return None
    direction = 1 if str(pos.side).lower() == "long" else -1
    # Conservative stop/target reconstruction from broker entry when the
    # position was adopted (entry unknown): percent-based guard rails.
    cfg = ExitConfig(max_hold_seconds=int(max_hold_s))
    stop = price * (1 - direction * 0.005)
    target = price * (1 + direction * 0.01)
    peak = max(pos.entry_price, price) if direction == 1 else min(pos.entry_price, price)
    reason, msg = exit_check(cfg, ExitCheck(
        symbol=symbol, side="long" if direction == 1 else "short",
        entry=pos.entry_price, price=price, stop=stop, target=target,
        peak=peak, hold_s=0.0, signal_action=signal_action))
    if reason is _ExitReason.NONE:
        return None
    reason_value = reason.value
    bus.publish("EXIT_TRIGGERED", {"symbol": symbol, "cycle_id": cycle_id,
                                   "reason": reason_value, "message": msg})
    metrics.inc("exits_triggered")
    result = deps.broker.close_position(symbol, reason=reason_value)
    if result.status == "FILLED":
        if tracker is not None:
            tracker.record_success(symbol)
        bus.publish("POSITION_CLOSED", {"symbol": symbol, "cycle_id": cycle_id,
                                        "reason": reason_value})
        metrics.inc("positions_closed")
        return {"symbol": symbol, "reason": reason_value, "closed": True}
    if tracker is not None:
        tracker.record_failure(symbol, result.message)
    bus.publish("ERROR_OCCURRED", {"symbol": symbol, "cycle_id": cycle_id,
                                   "error": f"exit_failed:{result.message}"[:200]})
    metrics.inc("exit_failures")
    return {"symbol": symbol, "reason": reason_value, "closed": False,
            "error": result.message}
