# Trading-GOAT — Real-Time AI-Assisted Paper-Trading Platform

**PAPER TRADING ONLY. No live-money path exists by design** (`execution/broker.py` asserts `mode==paper` and fail-closes otherwise).

Separation of concerns is the core idea: **LLM reasons, technicals contextualize, signal engine validates, risk controls capital, broker paper-trades.**

## End-to-end cycle

```
MARKET DATA -> quality gates -> FEATURES -> REGIME -> AI CONTEXT -> AI DECISION (strict JSON)
-> SIGNAL CONFLUENCE (confirmations/rejections) -> RISK POLICY -> PAPER EXECUTION (idempotent)
-> EXITS -> METRICS/EVENTS -> next cycle
```

Every cycle carries `cycle_id`, per-stage latency, and typed state (`domain/models.py`: MarketSnapshot, FeatureSnapshot, AIAnalysis, SignalDecision, RiskDecision, OrderIntent, ExecutionResult, TradingCycleResult). Uncertain -> HOLD. Invalid AI -> HOLD.

## Why this stack

- Dual timeframe: 5Min x30 trend context + 1Min x20 entry timing.
- Indicators (RSI/MACD/EMA/Bollinger/ATR/ADX/volume): different views of momentum/trend/volatility/position/participation; documented in `features/snapshot.py` and `docs/strategy.md`.
- LLM (Ollama Cloud + local fallback): structured `MarketContext` in, strict JSON out (`intelligence/prompts/trading_analysis_v1.txt`, `schemas.py`). Unavailable/invalid -> HOLD.
- Risk (deterministic, `risk/policy.py`): `risk=equity*risk%`, `stop=max(pct,1.5xATR)`, `qty=risk/stop`, exposure caps, daily-loss halt, kill switch.
- Exits (`exits/engine.py`): stop / TP / trailing / reversal / timeout.

## Run

```bash
pip install -r requirements.txt
cp .env.example .env   # fill keys; .env never committed
python scripts/run_bot.py --dry-run   # decide everything, submit nothing
python scripts/run_bot.py --once --dry-run
python main.py                          # legacy full loop (terminal + web dashboard)
python scripts/health_check.py
python scripts/run_backtest.py --csv tests/fixtures/bars.csv --symbol BTC/USD
pytest -q
```

Legacy `main.py` / `core/` / `dashboard/` / `backtest.py` still work; the new `scripts/run_bot.py` runs the same components through the structured pipeline (`orchestration/loop.py`) with startup checks (`app/bootstrap.py`: CONFIG VALIDATED -> ALPACA -> OLLAMA -> DATA -> FEATURES -> RISK -> SYSTEM READY).

## Audit fixes (this refactor)

- **Bollinger `int(std)` truncation**: `core/indicators.py` did `window_dev=int(1.8)->1`; now `float()`.
- **Config drift**: `config_loader` defaults (RSI 14, EMA 9/21...) disagreed with `config.yaml` (RSI 7, EMA 5/13...); new `config/settings.py` canonical model + warnings for overlapping keys (`risk_per_trade_pct` vs `base_risk_pct`, `exit_engine` risk clamps).
- **Silent failures**: broad `except: pass` paths now surface `quality_issues`, `errors[]`, events, and HOLD.
- **Partial fills**: explicitly flagged, never treated as full.
- **Duplicates**: `intent_id = cycle_id-symbol` registry + position-aware checks.

## Observability / safety

- Structured logs (rotating, secrets redacted), metrics (`observability/metrics.py`, Prometheus text), event bus (`observability/events.py`), health (`scripts/health_check.py`).
- Backtests labeled **HISTORICAL SIMULATION**, no look-ahead (signal[i] -> open[i+1]), fees/slippage; never presented as live P&L.
- No measured win-rate/ROI/Sharpe/latency claims: **not yet measured** — instrumentation is in place (`cycle_ms`, `llm_ms`, ...).

## Layout

`app/` bootstrap · `config/settings.py` · `domain/` · `data/` · `features/` · `intelligence/` · `strategy/` · `risk/` · `execution/` (paper-only) · `portfolio/` · `exits/` · `orchestration/` · `observability/` · `dashboard/` · `scripts/` · `tests/` · `docs/` · `Dockerfile` · `docker-compose.yml` · `Makefile`

## Interview demo (30s)

`python scripts/run_bot.py --once --dry-run` shows CYCLE -> REGIME -> AI -> SIGNAL (+reasons) -> RISK -> EXEC -> latency. Then: where does data come from? (`data/provider.py`), what does the LLM get/return? (`intelligence/`), how validated? (`schemas.py`+`strategy/confluence.py`), sizing? (`risk/policy.py`), duplicates? (`execution/broker.py`), exits? (`exits/engine.py`), failures? (`docs/operations.md`).
