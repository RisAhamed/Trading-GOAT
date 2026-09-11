"""Canonical risk policy. AI can never override this.

Formulas (documented, deterministic):
  risk_amount   = portfolio_value * (risk_per_trade_pct / 100)
  stop_distance = max(price * stop_loss_pct/100, atr * atr_stop_multiplier)
  qty           = risk_amount / stop_distance
  notional      = qty * price, capped by max_symbol_exposure_pct and
                  max_portfolio_exposure_pct / max_positions
  take_profit   = price +/- stop_distance * take_profit_multiplier  (R:R)
"""
from __future__ import annotations

from dataclasses import dataclass

from domain.models import Action, RiskDecision


@dataclass
class RiskInputs:
    symbol: str
    action: Action
    price: float
    atr: float
    portfolio_value: float
    open_positions: int
    symbol_exposure_notional: float = 0.0
    total_exposure_notional: float = 0.0
    daily_pnl_pct: float = 0.0
    kill_switch: bool = False


@dataclass
class RiskConfig:
    max_positions: int = 2
    risk_per_trade_pct: float = 1.0
    stop_loss_pct: float = 0.5
    take_profit_multiplier: float = 2.0
    max_portfolio_exposure_pct: float = 60.0
    max_symbol_exposure_pct: float = 35.0
    max_daily_loss_pct: float = 3.0
    atr_stop_multiplier: float = 1.5

    @classmethod
    def from_canonical(cls, s) -> RiskConfig:
        raw = s.raw.get("risk", {}) if isinstance(getattr(s, "raw", None), dict) else {}
        return cls(
            max_positions=s.max_positions,
            risk_per_trade_pct=s.risk_per_trade_pct,
            stop_loss_pct=s.stop_loss_pct,
            take_profit_multiplier=s.take_profit_multiplier,
            max_daily_loss_pct=s.max_daily_loss_pct,
            max_portfolio_exposure_pct=float(raw.get("max_portfolio_exposure_pct", 60.0)),
            max_symbol_exposure_pct=float(raw.get("max_symbol_exposure_pct", 35.0)),
            atr_stop_multiplier=float(raw.get("atr_stop_multiplier", 1.5)),
        )


def decide(cfg: RiskConfig, i: RiskInputs) -> RiskDecision:
    if i.kill_switch:
        return RiskDecision(i.symbol, False, "kill_switch_engaged")
    if i.action not in (Action.BUY, Action.SELL):
        return RiskDecision(i.symbol, False, f"no_trade_action_{i.action.value}")
    if i.price <= 0 or i.portfolio_value <= 0:
        return RiskDecision(i.symbol, False, "invalid_price_or_equity")
    if i.open_positions >= cfg.max_positions:
        return RiskDecision(i.symbol, False, f"max_positions_{cfg.max_positions}_reached")
    if i.daily_pnl_pct <= -abs(cfg.max_daily_loss_pct):
        return RiskDecision(i.symbol, False, f"daily_loss_limit_{i.daily_pnl_pct:.2f}%")

    risk_amount = i.portfolio_value * (cfg.risk_per_trade_pct / 100.0)
    pct_stop = i.price * (cfg.stop_loss_pct / 100.0)
    atr_stop = i.atr * cfg.atr_stop_multiplier if i.atr > 0 else 0.0
    stop_distance = max(pct_stop, atr_stop)
    if stop_distance <= 0:
        return RiskDecision(i.symbol, False, "zero_stop_distance")
    qty = risk_amount / stop_distance
    notional = qty * i.price

    cap_symbol = i.portfolio_value * (cfg.max_symbol_exposure_pct / 100.0)
    cap_port = i.portfolio_value * (cfg.max_portfolio_exposure_pct / 100.0) / max(cfg.max_positions, 1)
    cap = min(cap_symbol, cap_port)
    if notional > cap > 0:
        notional = cap
        qty = notional / i.price

    if i.symbol_exposure_notional + notional > cap_symbol + 1e-9:
        return RiskDecision(i.symbol, False, "symbol_exposure_cap")
    if i.total_exposure_notional + notional > i.portfolio_value * (cfg.max_portfolio_exposure_pct / 100.0) + 1e-9:
        return RiskDecision(i.symbol, False, "portfolio_exposure_cap")

    direction = 1 if i.action == Action.BUY else -1
    stop = i.price - direction * stop_distance
    target = i.price + direction * stop_distance * cfg.take_profit_multiplier
    rr = abs(target - i.price) / stop_distance if stop_distance else 0.0
    return RiskDecision(i.symbol, True, "", round(qty, 6), round(notional, 2),
                        round(risk_amount, 2), round(stop, 2), round(target, 2), round(rr, 2))
