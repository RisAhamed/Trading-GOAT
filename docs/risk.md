# Risk: capital controls the AI cannot override

Order: AI -> Signal -> RISK POLICY -> execution. Risk veto is final.

- risk_amount = equity * risk_per_trade_pct/100
- stop_distance = max(price*stop_loss_pct/100, ATR*1.5)
- qty = risk_amount/stop_distance, capped by min(symbol 35%, portfolio 60%/max_positions)
- take_profit = price +/- stop*R:R multiplier; R:R recorded on every RiskDecision
- Guards: max_positions, exposure caps, daily loss halt, kill switch, zero-stop reject.
- Dynamic exit_engine min/max risk clamps are sizing clamps only, not the base risk (see config warning).
- Trailing (0.25%, activation 0.15%, tiers) and exits (stop/TP/breakeven/reversal/timeout) protect after entry.
