# SYSTEM
# ├── DATA (data/provider.py + quality.py over core/market_data.py)
# │   INPUT: Alpaca bars/quotes | PROCESSING: retry+backoff, tz-aware normalize,
# │   stale/duplicate/gap/price/volume gates | OUTPUT: BarsResult+QualityReport
# │   DEPS: Alpaca API | FAIL: stale->HOLD, missing->HOLD, retries then fatal
# ├── FEATURES (features/snapshot.py + core/indicators.py: RSI,MACD,EMA,BB,ATR,ADX,vol)
# │   INPUT: OHLCV | PROCESSING: ta-lib calc + NaN/history/range validation
# │   OUTPUT: FeatureSnapshot(valid, quality_issues) | FAIL: invalid->HOLD
# ├── MARKET REGIME (features/regime.py): EMA stack + ADX + ATR% -> Regime + reasons
# ├── AI INTELLIGENCE (intelligence/provider.py + schemas.py + prompts/v1)
# │   INPUT: MarketContext | OUTPUT: strict JSON AIAnalysis | FAIL: invalid/unavailable->HOLD
# ├── STRATEGY (strategy/confluence.py): AI proposal x HTF/LTF/RSI/MACD/BB/ADX/vol/regime/position
# │   OUTPUT: SignalDecision(confirmations[], rejections[]) deterministic
# ├── RISK (risk/policy.py): caps, sizing qty=risk/stop, R:R; AI cannot override
# ├── EXECUTION (execution/broker.py AlpacaPaperBroker): PAPER-only, idempotent intents, fill verify
# ├── PORTFOLIO (portfolio/state.py): equity/cash/exposure/P&L, reconciled vs broker
# ├── EXITS (exits/engine.py): STOP/TP/TRAILING/REVERSAL/TIMEOUT
# ├── ORCHESTRATION (orchestration/loop.py): per-symbol cycle with cycle_id + per-stage latency
# ├── OBSERVABILITY (observability/): structured logs, metrics, event bus, health
# └── DASHBOARD (dashboard/): terminal + web ops views consuming the same events
