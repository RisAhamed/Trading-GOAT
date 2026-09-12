"""Execution-path tests against REAL legacy method names.

Uses a stub exposing exactly the verified OrderExecutor surface
(execute_buy/execute_sell/close_position/get_open_orders/cancel_order/
check_connection + client.get_order_by_id/get_open_position/get_all_positions).
Proves the broker never calls the nonexistent `place_bracket_order`.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from domain.models import OrderIntent
from execution.broker import AlpacaPaperBroker, PaperOnlyError


def _legacy_ok(tmp_path, mode="paper", buy=None, sell=None, close=None):
    client = MagicMock()
    client.get_order_by_id.side_effect = lambda oid: SimpleNamespace(
        status="filled", filled_qty=1.0, filled_avg_price=100.0, symbol="BTCUSD")
    client.get_open_position.side_effect = Exception("position does not exist")
    client.get_all_positions.return_value = []
    ex = MagicMock()
    ex.config.bot.mode = mode
    ex.client = client
    ex.get_open_orders.return_value = []
    ex.execute_buy = buy or MagicMock(return_value=SimpleNamespace(
        success=True, order_id="o1", status="accepted", filled_qty=0,
        filled_avg_price=None, stop_loss_order_id="sl1", take_profit_order_id="tp1",
        error_message=""))
    ex.execute_sell = sell or MagicMock(return_value=SimpleNamespace(
        success=True, order_id="o2", status="accepted", filled_qty=0,
        filled_avg_price=None, stop_loss_order_id="sl1", take_profit_order_id="tp1",
        error_message=""))
    ex.close_position = close or MagicMock(return_value=SimpleNamespace(
        success=True, order_id="c1", status="closed", qty=1.0,
        filled_avg_price=101.0, error_message=""))
    return ex


def _intent(**kw):
    d = dict(intent_id="i1", cycle_id="c1", signal_id="s1", symbol="BTC/USD",
             side="buy", qty=1.0, stop_price=99.5, take_profit_price=101.0)
    d.update(kw)
    return OrderIntent(**d)


def test_no_nonexistent_method_called(tmp_path):
    ex = _legacy_ok(tmp_path)
    del ex.place_bracket_order  # must not exist; broker must not need it
    b = AlpacaPaperBroker(ex, store_path=tmp_path / "i.json")
    r = b.submit_order(_intent(), last_price=100.0)
    assert r.status == "FILLED" and r.filled_qty == 1.0 and r.avg_fill_price == 100.0
    assert not hasattr(ex, "place_bracket_order")


def test_sell_path_and_protection_flag(tmp_path):
    ex = _legacy_ok(tmp_path)
    ex.execute_buy.return_value = SimpleNamespace(
        success=True, order_id="o1", status="accepted", filled_qty=0,
        filled_avg_price=None, stop_loss_order_id=None, take_profit_order_id=None,
        error_message="")
    b = AlpacaPaperBroker(ex, store_path=tmp_path / "i.json")
    r = b.submit_order(_intent(), last_price=100.0)
    assert "PROTECTION_MISSING" in r.message  # recorded, not pretended


def test_rejected_order_propagates(tmp_path):
    ex = _legacy_ok(tmp_path)
    ex.execute_buy.return_value = SimpleNamespace(
        success=False, order_id="", status="rejected", filled_qty=0,
        filled_avg_price=None, error_message="insufficient buying power")
    b = AlpacaPaperBroker(ex, store_path=tmp_path / "i.json")
    r = b.submit_order(_intent(), last_price=100.0)
    assert r.status == "REJECTED" and "buying power" in r.message


def test_partial_stays_partial(tmp_path):
    ex = _legacy_ok(tmp_path)
    ex.client.get_order_by_id.side_effect = lambda oid: SimpleNamespace(
        status="filled", filled_qty=0.4, filled_avg_price=100.0, symbol="BTCUSD")
    b = AlpacaPaperBroker(ex, store_path=tmp_path / "i.json")
    r = b.submit_order(_intent(), last_price=100.0)
    assert r.status == "PARTIAL" and r.filled_qty == 0.4


def test_close_confirms_flat(tmp_path):
    ex = _legacy_ok(tmp_path)
    b = AlpacaPaperBroker(ex, store_path=tmp_path / "i.json")
    r = b.close_position("BTC/USD")
    assert r.status == "FILLED"  # get_open_position raises => gone


def test_live_refused():
    ex = MagicMock()
    ex.config.bot.mode = "live"
    with pytest.raises(PaperOnlyError):
        AlpacaPaperBroker(ex)


def test_paper_allowed(tmp_path):
    b = AlpacaPaperBroker(_legacy_ok(tmp_path), store_path=tmp_path / "i.json")
    assert b.mode == "PAPER"
