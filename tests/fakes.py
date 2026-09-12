"""Deterministic fake paper broker for tests: fills, partials, rejects, delays.

Implements the Broker protocol (execution/broker.py) without credentials.
"""
from __future__ import annotations

import time
import uuid


class FakePaperBroker:
    mode = "PAPER"

    def __init__(self, equity: float = 100000.0, fill_mode: str = "fill",
                 fill_delay_s: float = 0.0) -> None:
        self.equity = equity
        self.fill_mode = fill_mode  # fill | partial | reject
        self.fill_delay_s = fill_delay_s
        self.orders: dict[str, dict] = {}
        self.positions_map: dict[str, dict] = {}
        self.submits = 0
        self.has_strategy_position = lambda s: s in self.positions_map

    def account(self) -> dict:
        return {"ok": True, "portfolio_value": self.equity, "cash": self.equity,
                "buying_power": self.equity * 2, "equity": self.equity}

    def positions(self) -> list[dict]:
        return [dict(v) for v in self.positions_map.values()]

    def open_orders(self) -> list[dict]:
        return [dict(o) for o in self.orders.values() if o["status"] in ("new", "accepted")]

    def submit_order(self, intent, last_price: float = 0.0):
        from domain.models import ExecutionResult

        self.submits += 1
        if self.fill_mode == "reject":
            return ExecutionResult(intent.intent_id, intent.symbol, "REJECTED",
                                   0, 0.0, "", "fake_insufficient_funds")
        oid = f"fake-{uuid.uuid4().hex[:8]}"
        if self.fill_mode == "partial":
            self.orders[oid] = {"id": oid, "symbol": intent.symbol, "side": intent.side,
                                "status": "partially_filled", "filled_qty": intent.qty / 2,
                                "avg": last_price}
            return ExecutionResult(intent.intent_id, intent.symbol, "PARTIAL",
                                   intent.qty / 2, last_price, oid, "partially_filled")
        if self.fill_delay_s:
            time.sleep(self.fill_delay_s)
        self.orders[oid] = {"id": oid, "symbol": intent.symbol, "side": intent.side,
                            "status": "filled", "filled_qty": intent.qty, "avg": last_price}
        self.positions_map[intent.symbol] = {
            "symbol": intent.symbol.replace("/", ""), "qty": intent.qty,
            "side": "long" if intent.side == "buy" else "short",
            "entry_price": last_price, "current_price": last_price,
            "market_value": intent.qty * last_price, "unrealized_pl": 0.0}
        return ExecutionResult(intent.intent_id, intent.symbol, "FILLED",
                               intent.qty, last_price, oid, "filled")

    def submit(self, intent, last_price: float = 0.0):
        return self.submit_order(intent, last_price)

    def get_order(self, order_id: str) -> dict:
        o = self.orders.get(order_id, {})
        return {"order_id": order_id, "status": o.get("status", "unknown"),
                "filled_qty": o.get("filled_qty", 0), "avg_fill_price": o.get("avg", 0.0)}

    def cancel_order(self, order_id: str) -> bool:
        if order_id in self.orders:
            self.orders[order_id]["status"] = "canceled"
            return True
        return False

    def close_position(self, symbol: str, reason: str = "MANUAL"):
        from domain.models import ExecutionResult

        if symbol not in self.positions_map:
            return ExecutionResult("close-x", symbol, "FAILED", 0, 0.0, "", "no_position")
        p = self.positions_map.pop(symbol)
        return ExecutionResult("close-x", symbol, "FILLED", p["qty"],
                               p["current_price"], "fake-close", reason)

    def reconcile(self, intent, order_id: str, timeout_s: float = 5.0):
        o = self.get_order(order_id)
        from domain.models import ExecutionResult

        st = {"filled": "FILLED", "partially_filled": "PARTIAL"}.get(o["status"], "FAILED")
        return ExecutionResult(intent.intent_id, intent.symbol, st,
                               o["filled_qty"], o["avg_fill_price"], order_id, o["status"])
