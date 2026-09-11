# Strategy: what each input contributes (views, not predictions)

- RSI (7): momentum state. Entry context + confluence gate (BUY needs <=65, SELL >=35).
- MACD (8,17,9): trend-momentum + histogram slope for entry timing.
- EMA (5/13): trend stack. Higher-TF EMA defines TrendContext direction.
- Bollinger (14, 1.8 float): position-in-volatility (%B). Stretched>=0.8 / washed<=0.2; never standalone.
- ATR (10): volatility ruler. stop_distance=max(pct_stop, 1.5xATR); sizing denominator; regime input.
- ADX (14): trend strength. >=20 trending, <20 chop/sideways. Regime + strength label.
- Volume (20-bar SMA): participation confirm only.
- Higher TF (5Min x30): trend context. Lower TF (1Min x20): entry timing.
- Market regime: describes conditions for AI + risk caution flags.
- AI: reasons over combined MarketContext, returns strict JSON proposal.
- Signal engine: deterministic veto. AI proposes, technicals dispose; every accept/reject explained.
