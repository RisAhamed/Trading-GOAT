# Legacy migration map

One canonical runtime: `scripts/run_bot.py`. One canonical config: `config/settings.py`.
Broker truth: `portfolio/snapshot.py`. Legacy `core/` is WRAPPED, not forked.

| Component | Verdict | Why |
|---|---|---|
| `main.py` (legacy AITrader engine) | DEPRECATE (wrapper) | Competing engine; now forwards to canonical runtime |
| `core/market_data.py` | WRAP | Real fetch/retry/cache kept; `data/provider.py` adds quality + freshness semantics |
| `core/indicators.py` | WRAP (+1 fix) | Real TA kept; Bollinger `int(std)` truncation fixed; `features/snapshot.py` validates |
| `core/ai_brain.py` | WRAP | Real cloud/local calls kept; `intelligence/provider.py` delegates to verified methods with explicit fallback outcomes |
| `core/signal_engine.py` | MIGRATE (superseded) | Opaque booleans replaced by `strategy/confluence.py` scored evidence + HOLD reasons |
| `core/risk_manager.py` | WRAP (sizing) + canonical policy | Legacy `RiskParameters` reused as broker handoff; `risk/policy.py` is the single formula |
| `core/order_executor.py` | WRAP | Real `execute_buy/sell`, `close_position`, `get_open_orders`, `cancel_order` used by `execution/broker.py`; nonexistent `place_bracket_order` dependency removed |
| `core/portfolio_tracker.py` | MIGRATE (history only) | Live truth moved to broker snapshot; tracker kept for dashboard history |
| `core/trailing_stop_manager.py`, `position_monitor.py`, `trade_exit_engine.py` | KEEP (legacy path) / canonical uses `exits/engine.py` + `exits/retry.py` | Exit rules consolidated; bounded retry replaces every-cycle close spam |
| `core/market_regime.py` | WRAP | Kept for session/context; canonical regime = `features/regime.py` + reasons |
| `core/symbol_scanner.py` | WRAP (background only) | `scan_and_rank()` runs in enrichment thread; loop reads cached rows; `strategy/universe.py` guarantees primaries |
| `core/market_intelligence.py`, `core/political_signal_scanner.py` | WRAP (optional) | Background `intelligence/enrichment.py` with circuit breaker; neutral fallback; never block loop |
| `core/bearish_scalp_strategy.py` | KEEP (dormant) | Not on canonical path; no removal to avoid losing behavior |
| `core/backtester.py`, `backtest.py` | KEEP (legacy) | Historical simulation; labeled as such; canonical offline harness = `scripts/run_backtest.py` (plumbing only, shares feature code) |
| `core/trade_results.py` (positions.json) | MIGRATE (audit only) | No longer live truth; broker wins via `portfolio/snapshot.reconcile` |
| `dashboard/terminal_ui.py`, `dashboard/web_ui.py` | KEEP + extend | Web UI gained `/health`, `/ready`, `/metrics`, `/api/runtime` (truthful UNKNOWN, never invented 0s) |
| `check_status.py` | KEEP | Manual ops helper |

Removed from git tracking (kept on disk): `.env`, `logs/trading.log` (secrets/history must not be committed).
