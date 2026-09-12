#!/usr/bin/env python3
"""Canonical trading runtime: python scripts/run_bot.py [--once] [--dry-run].

ONE canonical runtime (main.py is a deprecated wrapper around this).
ONE canonical configuration (config/settings.py).
BROKER STATE = source of truth (portfolio/snapshot.py + reconcile).
Optional enrichment never blocks the critical loop.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.bootstrap import startup
from data.provider import AlpacaProvider
from domain.models import Action
from execution.broker import AlpacaPaperBroker
from exits.retry import ExitRetryTracker
from features.regime import detect_regime
from features.snapshot import build_snapshot
from intelligence.enrichment import EnrichmentHub
from intelligence.provider import decide as ai_decide
from observability import events as bus
from observability import health, metrics
from observability.logging_config import log_event, setup_logging
from observability.state import publish_runtime_state
from orchestration.loop import CycleDeps, evaluate_exits_for_position, run_cycle
from portfolio.snapshot import normalize_symbol, reconcile, snapshot_from_broker
from risk.policy import RiskConfig
from risk.policy import decide as risk_decide
from strategy.confluence import evaluate
from strategy.universe import select_symbols

logger = setup_logging()
_ = Action


def _hold_breakdown() -> dict:
    snap = metrics.snapshot()
    counters = snap.get("counters", {})
    return {k.replace("hold_", ""): v for k, v in counters.items() if k.startswith("hold_")}


def build_enrichment(boot) -> tuple:
    """Optional sources registered with neutral fallback; background only.

    The legacy SymbolScanner performs network I/O inline (bars + Kraken +
    political per candidate), so it must NEVER run on the critical path:
    it refreshes here in the background and the loop reads cached rows.
    """
    from core.symbol_scanner import SymbolScanner

    scanner = SymbolScanner(boot.fetcher, boot.legacy)
    hub = EnrichmentHub()
    hub.register("universe", lambda: {"rows": scanner.scan_and_rank()})
    try:
        from core.market_intelligence import MarketIntelligence

        mi = MarketIntelligence(boot.legacy)
        hub.register("whale", lambda: mi.get_whale_signal("BTC/USD"))
    except Exception as _whale_init_e:
        _whale_err = str(_whale_init_e)[:200]
        hub.register("whale", lambda _e=_whale_err: (_ for _ in ()).throw(
            RuntimeError(f"init_failed: {_e}")))
    try:
        from core.political_signal_scanner import PoliticalSignalScanner

        ps = PoliticalSignalScanner(boot.legacy)
        hub.register("political", lambda: {"summary": ps.get_signal_summary()})
    except Exception as _pol_init_e:
        _pol_err = str(_pol_init_e)[:200]
        hub.register("political", lambda _e=_pol_err: (_ for _ in ()).throw(
            RuntimeError(f"init_failed: {_e}")))
    hub.start_background(interval_s=120.0)
    return hub, scanner


def main() -> None:
    ap = argparse.ArgumentParser(description="Trading-GOAT canonical paper runtime")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="decide everything, submit nothing")
    args = ap.parse_args()

    boot = startup()
    risk_cfg = RiskConfig.from_canonical(boot.settings)
    broker = AlpacaPaperBroker(boot.executor, dry_run=args.dry_run,
                               has_strategy_position=lambda s: False)  # replaced per-cycle below
    tracker = ExitRetryTracker()
    enrichment, _scanner = build_enrichment(boot)

    # Startup reconciliation: broker wins, local repaired, never fabricated.
    try:
        local_positions = {}
        rp = Path("logs/results/positions.json")
        if rp.exists():
            local_positions = (json.loads(rp.read_text(encoding="utf-8")) or {}).get("positions", {})
    except Exception:
        local_positions = {}
    snap0 = snapshot_from_broker(broker)
    rec = reconcile(snap0, local_positions)
    for m in rec.mismatches:
        bus.publish("RECONCILIATION_MISMATCH", {"message": m})
        logger.warning("RECONCILIATION_MISMATCH | %s", m)
    print(f"RECONCILIATION READY (mismatches={len(rec.mismatches)} repaired={len(rec.repaired)})")

    n = 0
    while True:
        n += 1
        t_cycle = time.perf_counter()
        snap = snapshot_from_broker(broker)
        metrics.set_gauge("strategy_positions", snap.strategy_open_positions)
        metrics.set_gauge("broker_positions", snap.broker_total_positions)
        metrics.set_gauge("equity", snap.equity)
        strat_syms = {p.symbol for p in snap.strategy_positions}
        broker.has_strategy_position = lambda s, _ss=strat_syms: normalize_symbol(s) in _ss

        # Universe: cached background ranking (never scanned inline on the
        # critical path). Advisory only: primaries + open positions always stay.
        rankings: dict[str, float] = {}
        try:
            uni = (enrichment.snapshot().get("universe", {}) or {}).get("value", {}) or {}
            for row in uni.get("rows", []) or []:
                if isinstance(row, dict) and row.get("symbol"):
                    rankings[str(row["symbol"])] = float(row.get("total_score", 0) or 0)
        except Exception as e:
            logger.warning("universe cache unreadable, using configured universe: %s", str(e)[:150])
        raw_cfg = boot.settings.raw.get("symbol_scanner", {}) if isinstance(boot.settings.raw, dict) else {}
        rows = select_symbols(
            configured=list(boot.settings.symbols),
            rankings=rankings,
            open_strategy_symbols=strat_syms,
            max_ranked=int(raw_cfg.get("max_symbols_to_trade", 3) or 3),
            min_score=float(raw_cfg.get("min_score_to_trade", 40) or 40),
        )
        symbols = [r.symbol for r in rows if r.tradeable]
        print(f"\nCYCLE #{n} universe=" + ", ".join(
            f"{r.symbol}(rank#{r.rank},{r.priority},{r.score:.0f})" for r in rows if r.tradeable))

        ai_states, sig_states, risk_states, exec_states = {}, {}, {}, {}
        for sym in symbols:
            deps = CycleDeps(
                settings=boot.settings, provider=AlpacaProvider(boot.fetcher),
                indicators=boot.indicators, build_snapshot_fn=build_snapshot,
                detect_regime_fn=detect_regime,
                ai_decide_fn=lambda ctx, p, q: ai_decide(
                    ctx, p, q, boot.brain, boot.settings.ai_model,
                    getattr(boot.legacy.ai, "local_fallback_model", ""),
                    boot.settings.ai_timeout_s),
                confluence_fn=evaluate,
                risk_decide_fn=lambda i: risk_decide(risk_cfg, i),
                broker=broker, snapshot=snap, universe_rows=rows,
                exit_tracker=tracker,
                enrichment=enrichment.snapshot(),
                min_confidence=boot.settings.min_signal_confidence,
            )
            print(f"\nCYCLE #{n} | {sym}")
            res = run_cycle(deps, sym)
            ai = f"{res.ai.action.value} {res.ai.confidence:.2f}" if res.ai else "n/a"
            sig = res.signal.final_action.value if res.signal else "n/a"
            reason = (res.signal.hold_reason or "") if res.signal else ""
            risk = "APPROVED" if (res.risk and res.risk.allowed) else (res.risk.rejection_reason if res.risk else "-")
            ex = res.execution.status if res.execution else "-"
            print(f"  BAR age={res.market_snapshot.bar_timestamp} new={res.market_snapshot.stale is False} | "
                  f"REGIME={res.regime.value} AI={ai} SIGNAL={sig} REASON={reason} RISK={risk} EXEC={ex} FINAL={res.final_state}")
            print(f"  latency_ms={ {k: round(v, 1) for k, v in res.latency_ms.items()} } errors={res.errors or []}")
            ai_states[sym] = {"action": res.ai.action.value if res.ai else "?", "confidence": res.ai.confidence if res.ai else 0,
                              "model": res.ai.model_used if res.ai else "", "fallback": res.ai.fallback_used if res.ai else False}
            sig_states[sym] = {"final": sig, "reason": reason, "score": res.signal.score if res.signal else 0}
            risk_states[sym] = {"approved": bool(res.risk and res.risk.allowed)}
            exec_states[sym] = {"status": ex}
            log_event(logger, logging.INFO, "CYCLE_COMPLETED", f"cycle {res.cycle_id} -> {res.final_state}",
                      cycle_id=res.cycle_id, symbol=sym, final_state=res.final_state)

            # Exit management for the open strategy position (bounded retry inside).
            if sym in strat_syms:
                rec_exit = evaluate_exits_for_position(
                    deps, sym, res.signal.final_action.value if res.signal else "HOLD",
                    res.cycle_id, float(boot.settings.raw.get("risk", {}).get("max_hold_seconds", 600)
                                         if isinstance(boot.settings.raw, dict) else 600))
                if rec_exit and not rec_exit.get("deferred"):
                    print(f"  EXIT {rec_exit.get('reason')} closed={rec_exit.get('closed')} err={rec_exit.get('error', '')}")

        # External positions: tracked, never crypto-managed, never force-closed.
        for p in snap.external_positions:
            print(f"  EXTERNAL_UNSUPPORTED_POSITION {p.raw_symbol} qty={p.qty} (tracked, not strategy)")
        print(f"  HOLD BREAKDOWN {_hold_breakdown()}")
        print(f"  CYCLE #{n} done in {(time.perf_counter() - t_cycle):.1f}s | "
              f"broker={snap.broker_total_positions} strategy={snap.strategy_open_positions} "
              f"enrichment={enrichment.health()}")
        publish_runtime_state({
            "cycle": n, "universe": [r.__dict__ for r in rows],
            "ai": ai_states, "signals": sig_states, "risk": risk_states,
            "exec": exec_states, "hold_breakdown": _hold_breakdown(),
            "strategy_positions": [p.__dict__ for p in snap.strategy_positions],
            "external_positions": [p.__dict__ for p in snap.external_positions],
            "equity": snap.equity, "health": health.get_health().status,
        })
        if args.once:
            break
        time.sleep(max(1, boot.settings.loop_interval_s))


if __name__ == "__main__":
    main()
