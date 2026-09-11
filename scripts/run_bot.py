#!/usr/bin/env python3
"""Run the real-time paper-trading loop with structured cycles.

Usage: python scripts/run_bot.py [--once] [--dry-run]
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.bootstrap import startup
from data.provider import AlpacaProvider
from execution.broker import AlpacaPaperBroker
from features.regime import detect_regime
from features.snapshot import build_snapshot
from intelligence.provider import LocalOllamaProvider, OllamaCloudProvider, decide
from observability import health, metrics
from observability.logging_config import log_event, setup_logging
from orchestration.loop import CycleDeps, run_cycle
from portfolio.state import from_legacy
from risk.policy import RiskConfig
from risk.policy import decide as risk_decide
from strategy.confluence import evaluate

logger = setup_logging()


def build_deps(boot, dry_run: bool) -> CycleDeps:
    risk_cfg = RiskConfig.from_canonical(boot.settings)
    primary = OllamaCloudProvider(boot.brain)
    fallback = LocalOllamaProvider(boot.brain)

    def ai_fn(ctx, pos_desc, port_desc):
        return decide(ctx, pos_desc, port_desc, primary, fallback,
                      boot.settings.ai_timeout_s, boot.settings.ai_model)

    def port_fn():
        try:
            _, _, acct = boot.executor.check_connection()
        except Exception:
            acct = {}
        return from_legacy(boot.tracker, acct)

    def risk_fn(inputs):
        return risk_decide(risk_cfg, inputs)

    return CycleDeps(
        settings=boot.settings,
        provider=AlpacaProvider(boot.fetcher),
        indicators=boot.indicators,
        build_snapshot_fn=build_snapshot,
        detect_regime_fn=detect_regime,
        ai_decide_fn=ai_fn,
        confluence_fn=evaluate,
        risk_decide_fn=risk_fn,
        broker=AlpacaPaperBroker(boot.executor, dry_run=dry_run),
        portfolio_fn=port_fn,
        tracker=boot.tracker,
        min_confidence=boot.settings.min_signal_confidence,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="run one cycle per symbol then exit")
    ap.add_argument("--dry-run", action="store_true", help="decide everything but never submit orders")
    args = ap.parse_args()

    boot = startup()
    deps = build_deps(boot, dry_run=args.dry_run)
    n = 0
    while True:
        n += 1
        for sym in boot.settings.symbols:
            print(f"\nCYCLE #{n} | {sym}")
            res = run_cycle(deps, sym)
            ai = f"{res.ai.action.value} {res.ai.confidence:.2f}" if res.ai else "n/a"
            sig = res.signal.final_action.value if res.signal else "n/a"
            risk = "APPROVED" if (res.risk and res.risk.allowed) else (res.risk.rejection_reason if res.risk else "-")
            ex = res.execution.status if res.execution else "-"
            print(f"  REGIME={res.regime.value} AI={ai} SIGNAL={sig} RISK={risk} EXEC={ex} FINAL={res.final_state}")
            print(f"  latency_ms={ {k: round(v,1) for k,v in res.latency_ms.items()} } errors={res.errors or []}")
            log_event(logger, logging.INFO, "CYCLE_COMPLETED", f"cycle {res.cycle_id} -> {res.final_state}",
                      cycle_id=res.cycle_id, symbol=sym, final_state=res.final_state)
        h = health.get_health()
        metrics.set_gauge("open_positions", 0)
        _ = h
        if args.once:
            break
        time.sleep(boot.settings.loop_interval_s)


if __name__ == "__main__":
    main()
