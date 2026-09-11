"""Exit engine: single home for stop/TP/trailing/breakeven/reversal/timeout exits."""
from __future__ import annotations

from dataclasses import dataclass

from domain.models import ExitReason


@dataclass
class ExitConfig:
    stop_loss_pct: float = 0.5
    take_profit_multiplier: float = 2.0
    trailing_pct: float = 0.25
    max_hold_seconds: int = 600
    breakeven_r: float = 1.0


@dataclass
class ExitCheck:
    symbol: str
    side: str  # long | short
    entry: float
    price: float
    stop: float
    target: float
    peak: float
    hold_s: float
    signal_action: str = "HOLD"  # for reversal exits


def check(cfg: ExitConfig, c: ExitCheck) -> tuple[ExitReason, str]:
    if c.entry <= 0 or c.price <= 0:
        return ExitReason.NONE, ""
    direction = 1 if c.side == "long" else -1
    pnl_pct = (c.price - c.entry) / c.entry * 100 * direction
    # hard stop / target first
    if direction == 1 and c.price <= c.stop:
        return ExitReason.STOP_LOSS, f"price {c.price} <= stop {c.stop}"
    if direction == -1 and c.price >= c.stop:
        return ExitReason.STOP_LOSS, f"price {c.price} >= stop {c.stop}"
    if direction == 1 and c.price >= c.target:
        return ExitReason.TAKE_PROFIT, f"price {c.price} >= target {c.target}"
    if direction == -1 and c.price <= c.target:
        return ExitReason.TAKE_PROFIT, f"price {c.price} <= target {c.target}"
    # trailing: floor = peak*(1 - trail%) for longs
    floor = c.peak * (1 - cfg.trailing_pct / 100.0) if direction == 1 else c.peak * (1 + cfg.trailing_pct / 100.0)
    gained = (c.peak - c.entry) / c.entry * 100 * direction
    if gained >= 0.15:  # activation avoids premature trailing
        if direction == 1 and c.price <= floor:
            return ExitReason.TRAILING_STOP, f"trailing floor {floor:.2f} hit"
        if direction == -1 and c.price >= floor:
            return ExitReason.TRAILING_STOP, f"trailing ceiling {floor:.2f} hit"
    # reversal
    if (c.side == "long" and c.signal_action == "SELL") or (c.side == "short" and c.signal_action == "BUY"):
        return ExitReason.SIGNAL_REVERSAL, f"signal flipped to {c.signal_action}"
    # timeout / stale
    if c.hold_s >= cfg.max_hold_seconds:
        return ExitReason.TIMEOUT, f"held {c.hold_s:.0f}s >= max {cfg.max_hold_seconds}s"
    _ = pnl_pct
    return ExitReason.NONE, ""
