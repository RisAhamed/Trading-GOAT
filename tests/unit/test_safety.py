"""Unit: AI schema validation fails safe to HOLD; exits fire correctly."""
from domain.models import Action, ExitReason
from exits.engine import ExitCheck, ExitConfig, check
from intelligence.schemas import parse_ai_json


def test_invalid_ai_holds():
    assert parse_ai_json("not json").action == Action.HOLD
    assert parse_ai_json('{"action":"MOON","confidence":0.9}').raw_invalid
    assert parse_ai_json('{"action":"BUY","confidence":99}').confidence == 0.99  # 0-100 tolerated
    assert parse_ai_json('{"action":"BUY","confidence":-1}').raw_invalid
    assert parse_ai_json('{"action":"BUY"}').raw_invalid


def test_valid_ai_parses():
    p = parse_ai_json('{"action":"BUY","confidence":0.72,"trend":"BULLISH","entry_quality":"MODERATE","reasoning_summary":"ok","risk_notes":["x"]}')
    assert p.action == Action.BUY and p.confidence == 0.72 and not p.raw_invalid


def test_exit_engine():
    cfg = ExitConfig(stop_loss_pct=0.5, take_profit_multiplier=2.0, trailing_pct=0.25, max_hold_seconds=600)
    r, _ = check(cfg, ExitCheck("BTC/USD", "long", 100, 99.0, 99.5, 101.0, 100.0, 10))
    assert r == ExitReason.STOP_LOSS
    r, _ = check(cfg, ExitCheck("BTC/USD", "long", 100, 101.5, 99.5, 101.0, 101.5, 10))
    assert r == ExitReason.TAKE_PROFIT
    r, _ = check(cfg, ExitCheck("BTC/USD", "long", 100, 100.0, 99.5, 101.0, 100.0, 9999))
    assert r == ExitReason.TIMEOUT
    r, _ = check(cfg, ExitCheck("BTC/USD", "long", 100, 100.2, 99.5, 101.0, 100.2, 10, signal_action="SELL"))
    assert r == ExitReason.SIGNAL_REVERSAL
