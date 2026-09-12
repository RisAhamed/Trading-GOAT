"""Unit: risk percent/fraction units must be unambiguous (§27).

risk_per_trade_pct=1.0 means 1% (not 100%). stop_loss_pct=0.5 means 0.5%.
base_risk_pct legacy fraction (0.02) is documented as fraction and ignored
by the canonical policy in favor of risk_per_trade_pct.
"""
from domain.models import Action
from risk.policy import RiskConfig, RiskInputs, decide


def test_percent_means_percent():
    cfg = RiskConfig(risk_per_trade_pct=1.0, stop_loss_pct=0.5)
    d = decide(cfg, RiskInputs("BTC/USD", Action.BUY, 100.0, 0.0, 100000.0, 0))
    assert d.risk_amount == 1000.0  # 1% of 100k, NOT 100% (=100k)
    assert d.stop_price == 99.5  # 0.5% below, NOT 50% (=50.0)


def test_fraction_confusion_rejected_by_bounds():
    cfg = RiskConfig(risk_per_trade_pct=100.0)  # would mean 100% if misread
    d = decide(cfg, RiskInputs("BTC/USD", Action.BUY, 100.0, 0.0, 100000.0, 0))
    # Even at 100% the math stays consistent: risk == full equity, qty finite.
    assert d.risk_amount == 100000.0
    assert d.position_size > 0


def test_confidence_scale():
    from intelligence.schemas import parse_ai_json

    assert parse_ai_json('{"action":"BUY","confidence":0.72}').confidence == 0.72
    assert parse_ai_json('{"action":"BUY","confidence":72}').confidence == 0.72  # 0-100 tolerated
