# Trading-GOAT — Real-Time AI-Assisted Paper-Trading Platform

**PAPER TRADING ONLY. No live-money path exists by design.** Startup asserts
`MODE=PAPER`; `execution/broker.py` raises `PaperOnlyError` for anything else.

Core idea: **LLM reasons, technicals contextualize, signal engine validates,
risk controls capital, broker paper-trades, broker state is truth.**

## Canonical runtime (one engine)

```bash
python scripts/run_bot.py --dry-run   # decide everything, submit nothing
python scripts/run_bot.py --once --dry-run
```

`main.py` is a deprecated wrapper forwarding here. `config/settings.py` is the
single configuration model (legacy `core/config_loader` values are read from
the same `config.yaml`, normalized once at startup with overlap warnings).

Startup prints: `SYSTEM STARTING → CONFIG VALIDATED → PAPER MODE ASSERTED →
ALPACA → OLLAMA → MARKET DATA → FEATURES → RISK → RECONCILIATION →
SYSTEM READY — PAPER TRADING`.

## Cycle

```
broker sync (truth) -> market data -> freshness -> features -> regime
-> AI context -> AI decision (strict JSON) -> signal scoring -> risk
-> paper execution (idempotent, verified) -> reconcile -> exits
-> events/metrics/dashboard
```

Every cycle: `cycle_id`, per-stage latency, typed state (`domain/models.py`),
errors, final state, HOLD reason. Uncertain → HOLD. Invalid AI → HOLD.

## Why no trades for hours is normal

- All 41 HOLDs stay alive: each HOLD carries a reason (`AI_HOLD`,
  `AI_LOW_CONFIDENCE`, `TECHNICAL_VETO`, `REGIME_VETO`, `POSITION_EXISTS`,
  `RISK_LIMIT`, `STALE_DATA`, …) counted under HOLD BREAKDOWN.
- Scanner is advisory: primaries (BTC/ETH) + open positions always evaluated.
- Unchanged 5Min candle across 15s cycles is cache reuse (`duplicate_bars`),
  not a freeze; staleness uses 3×interval+grace thresholds.
- Whale/political enrichment runs in background with circuit breaker; loop
  never blocks on it (`UNAVAILABLE:reason` + neutral fallback when down).
- Legacy stock positions (e.g. NVDA/XOM) are `EXTERNAL_UNSUPPORTED_POSITION`:
  tracked, never counted in strategy capacity, exits retried with backoff.

## Verify

```bash
.\.verify\Scripts\python.exe -m pytest tests -q   # 42 tests, no credentials
.\.verify\Scripts\python.exe scripts/health_check.py
.\.verify\Scripts\python.exe scripts/run_backtest.py --csv tests/fixtures/bars.csv --symbol BTC/USD
```

Dashboard: `/health`, `/ready`, `/metrics`, `/api/runtime` (unknown = NOT
AVAILABLE, never invented). Backtests are labeled HISTORICAL SIMULATION.
No win-rate/ROI/Sharpe claims: not yet measured; instrumentation in place.

See `docs/architecture.md`, `docs/strategy.md`, `docs/risk.md`,
`docs/operations.md`, `docs/execution.md`, `docs/migration.md`.
