"""Unit: risk math is deterministic and explainable; AI cannot override."""
from domain.models import Action
from risk.policy import RiskConfig, RiskInputs, decide


def test_position_sizing_formula():
    cfg = RiskConfig(max_positions=2, risk_per_trade_pct=1.0, stop_loss_pct=0.5,
                     take_profit_multiplier=2.0, max_portfolio_exposure_pct=60.0,
                     max_symbol_exposure_pct=35.0, max_daily_loss_pct=3.0, atr_stop_multiplier=1.5)
    i = RiskInputs(symbol="BTC/USD", action=Action.BUY, price=100.0, atr=1.0,
                   portfolio_value=100000.0, open_positions=0)
    d = decide(cfg, i)
    # risk_amount=1000, stop=max(0.5, 1.5)=1.5, qty=666.6, capped by exposure
    assert d.allowed
    assert d.risk_amount == 1000.0
    assert d.risk_reward_ratio == 2.0
    assert d.stop_price < 100.0 < d.take_profit_price


def test_risk_blocks():
    cfg = RiskConfig()
    assert not decide(cfg, RiskInputs("BTC/USD", Action.HOLD, 100, 1, 100000, 0)).allowed
    assert not decide(cfg, RiskInputs("BTC/USD", Action.BUY, 100, 1, 100000, 2)).allowed
    kill = decide(cfg, RiskInputs("BTC/USD", Action.BUY, 100, 1, 100000, 0, kill_switch=True))
    assert not kill.allowed and "kill" in kill.rejection_reason
    dl = decide(cfg, RiskInputs("BTC/USD", Action.BUY, 100, 1, 100000, 0, daily_pnl_pct=-5.0))
    assert not dl.allowed
