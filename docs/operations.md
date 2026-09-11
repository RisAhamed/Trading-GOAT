# Operations: startup, health, troubleshooting, paper safety

Startup: `python scripts/run_bot.py` -> CONFIG VALIDATED -> PAPER asserted -> ALPACA -> OLLAMA (degraded ok)
-> MARKET DATA READY -> FEATURE ENGINE READY -> RISK ENGINE READY -> SYSTEM READY - PAPER TRADING.

- Health: scripts/health_check.py; web UI adds /health /ready /metrics (Prometheus text).
- Metrics: cycle/latency, signals, orders, exposures, P&L, api_errors, stale events.
- Troubleshooting:
  - Alpaca down -> SYSTEM NOT READY (critical), never trades blind.
  - Ollama down -> DEGRADED, AI->HOLD, exits still manage positions.
  - Stale data -> HOLD + stale_market_data_events inc.
  - Order rejected/partial -> recorded; partials never treated as full.
  - Dashboard down -> loop unaffected (separate thread).
- Paper safety: AlpacaPaperBroker asserts mode==paper, fail-closes on live, dry-run flag for demos.
- Secrets: env only (.env never committed); logs redact API_KEY/SECRET/TOKEN.
