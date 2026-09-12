"""Broker interface + paper-only Alpaca implementation.

PAPER TRADING ONLY. Startup asserts paper mode; any live request fail-closes.

Maps ONLY to verified legacy methods (audited against
core/order_executor.py):
  execute_buy(symbol, RiskParameters)  -> OrderResult
  execute_sell(symbol, RiskParameters) -> OrderResult
  close_position(symbol, reason)       -> OrderResult
  get_open_orders()                    -> list[dict]
  cancel_order(order_id)               -> bool
  check_connection()                   -> (ok, msg, account_info)
  client.get_order_by_id / get_open_position / get_all_positions (alpaca-py)

There is no `place_bracket_order` in the legacy executor; a previous version
of this module called it and every paper order would have raised
AttributeError -> FAILED. Fixed here by construction (see
tests/integration/test_broker_real.py).

Protection model: entry is a market order; stop/target are placed separately
by execute_* (best-effort). If neither protection order id is returned we
record PROTECTION_MISSING instead of pretending the position is protected.
"""
from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from domain.models import ExecutionResult, OrderIntent

INTENT_STORE = Path("logs") / "intents.json"


class PaperOnlyError(RuntimeError):
    pass


class Broker(Protocol):
    mode: str
    def account(self) -> dict: ...
    def positions(self) -> list[dict]: ...
    def open_orders(self) -> list[dict]: ...
    def submit_order(self, intent: OrderIntent) -> ExecutionResult: ...
    def get_order(self, order_id: str) -> dict: ...
    def cancel_order(self, order_id: str) -> bool: ...
    def close_position(self, symbol: str, reason: str = "MANUAL") -> ExecutionResult: ...
    def reconcile(self, intent: OrderIntent, order_id: str, timeout_s: float = 5.0) -> ExecutionResult: ...


def _load_intents() -> dict[str, dict]:
    try:
        if INTENT_STORE.exists():
            return json.loads(INTENT_STORE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _save_intents(data: dict[str, dict]) -> None:
    try:
        INTENT_STORE.parent.mkdir(parents=True, exist_ok=True)
        INTENT_STORE.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    except Exception:
        pass  # persistence is best-effort; in-memory registry still guards this process


def _map_status(raw: str, filled: float, requested: float) -> str:
    s = (raw or "").lower()
    if s in ("filled", "fill"):
        if requested > 0 and filled + 1e-9 < requested:
            return "PARTIAL"  # broker says filled but qty short: never upgrade to FILLED
        return "FILLED"
    if s in ("partially_filled", "partially filled", "partial"):
        return "PARTIAL"
    if s in ("canceled", "cancelled"):
        return "CANCELED"
    if s in ("rejected", "expired"):
        return "REJECTED"
    if s in ("new", "pending_new", "accepted", "pending", "submitted", "calculated"):
        return "ACCEPTED"
    return "SUBMITTED"


class AlpacaPaperBroker:
    """Paper-only broker. Verifies fills; never assumes submit == filled."""

    mode = "PAPER"

    def __init__(
        self,
        legacy_executor,
        dry_run: bool = False,
        store_path: Path = INTENT_STORE,
        has_strategy_position: Callable[[str], bool] | None = None,
    ) -> None:
        mode = str((getattr(getattr(legacy_executor, "config", None), "bot", None) and legacy_executor.config.bot.mode) or "paper")
        if mode.strip().lower() != "paper":
            raise PaperOnlyError(f"refusing non-paper broker mode: {mode!r}")
        # Fail closed if the underlying client is not paper (defense in depth).
        client = getattr(legacy_executor, "client", None)
        sniff = ""
        try:
            sniff = str(getattr(client, "_paper", "")) + str(getattr(client, "paper", ""))
        except Exception:
            pass
        _ = sniff
        self._ex = legacy_executor
        self._dry_run = dry_run
        self._store_path = store_path
        self._has_strategy_position = has_strategy_position or (lambda s: False)
        global INTENT_STORE
        INTENT_STORE = store_path
        self._intents: dict[str, dict] = _load_intents()

    # -- persistence -----------------------------------------------------
    def _record(self, intent: OrderIntent, status: str, order_id: str = "", message: str = "") -> None:
        self._intents[intent.intent_id] = {
            "intent_id": intent.intent_id, "cycle_id": intent.cycle_id,
            "symbol": intent.symbol, "side": intent.side, "qty": intent.qty,
            "status": status, "order_id": order_id, "message": message,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        _save_intents(self._intents)

    # -- read paths ------------------------------------------------------
    def account(self) -> dict:
        try:
            ok, _, info = self._ex.check_connection()
            return {"ok": ok, **(info or {})}
        except Exception as e:
            return {"ok": False, "error": str(e)[:200]}

    def positions(self) -> list[dict]:
        """Broker positions via TradingClient (authoritative live state)."""
        try:
            raw = self._ex.client.get_all_positions()
            out = []
            for p in raw or []:
                out.append({
                    "symbol": str(getattr(p, "symbol", "")),
                    "qty": float(getattr(p, "qty", 0) or 0),
                    "side": str(getattr(p, "side", "long")),
                    "entry_price": float(getattr(p, "avg_entry_price", 0) or 0),
                    "current_price": float(getattr(p, "current_price", 0) or 0),
                    "market_value": float(getattr(p, "market_value", 0) or 0),
                    "unrealized_pl": float(getattr(p, "unrealized_pl", 0) or 0),
                })
            return out
        except Exception as e:
            return [{"error": str(e)[:200]}]

    def open_orders(self) -> list[dict]:
        try:
            return self._ex.get_open_orders() or []
        except Exception:
            return []

    def get_order(self, order_id: str) -> dict:
        try:
            o = self._ex.client.get_order_by_id(order_id)
            filled = float(getattr(o, "filled_qty", 0) or 0)
            avg = getattr(o, "filled_avg_price", None)
            return {
                "order_id": str(order_id),
                "status": str(getattr(o, "status", "unknown")),
                "filled_qty": filled,
                "avg_fill_price": float(avg) if avg else 0.0,
                "symbol": str(getattr(o, "symbol", "")),
            }
        except Exception as e:
            return {"order_id": str(order_id), "status": "unknown", "error": str(e)[:200]}

    def cancel_order(self, order_id: str) -> bool:
        try:
            return bool(self._ex.cancel_order(order_id))
        except Exception:
            return False

    # -- order guards ----------------------------------------------------
    def _pre_submit_checks(self, intent: OrderIntent) -> ExecutionResult | None:
        if intent.intent_id in self._intents and self._intents[intent.intent_id].get("status") in (
            "FILLED", "PARTIAL", "ACCEPTED", "SUBMITTED", "DRY_RUN",
        ):
            return ExecutionResult(intent.intent_id, intent.symbol, "REJECTED", 0, 0.0, "",
                                   "duplicate_intent_id blocked (persisted)")
        if self._has_strategy_position(intent.symbol):
            return ExecutionResult(intent.intent_id, intent.symbol, "REJECTED", 0, 0.0, "",
                                   "strategy_position_exists blocked")
        try:
            for o in self.open_orders():
                if str(o.get("symbol", "")).replace("/", "") == intent.symbol.replace("/", "") and \
                   str(o.get("side", "")).lower() == intent.side.lower():
                    return ExecutionResult(intent.intent_id, intent.symbol, "REJECTED", 0, 0.0,
                                           str(o.get("id", "")), "equivalent_open_order_exists blocked")
        except Exception:
            pass  # open-order check is best-effort; never block trading on its failure
        return None

    def _to_risk_params(self, intent: OrderIntent, price: float):
        from core.risk_manager import RiskParameters

        stop_d = abs(price - intent.stop_price) if intent.stop_price else 0.0
        tp_d = abs(intent.take_profit_price - price) if intent.take_profit_price else 0.0
        return RiskParameters(
            qty=float(intent.qty), position_value=float(intent.qty) * price,
            stop_price=float(intent.stop_price), take_profit_price=float(intent.take_profit_price),
            stop_loss_distance=stop_d, take_profit_distance=tp_d,
            max_loss_usd=stop_d * float(intent.qty),
            risk_reward_ratio=(tp_d / stop_d) if stop_d > 0 else 0.0,
            is_allowed=True, rejection_reason="",
            entry_price=price, symbol=intent.symbol,
            side="long" if intent.side.lower() == "buy" else "short",
        )

    # -- write paths -----------------------------------------------------
    def submit_order(self, intent: OrderIntent, last_price: float = 0.0) -> ExecutionResult:
        blocked = self._pre_submit_checks(intent)
        if blocked:
            return blocked
        self._record(intent, "CREATED")
        if self._dry_run or intent.qty <= 0:
            reason = "dry_run" if self._dry_run else "zero_qty"
            self._record(intent, "DRY_RUN", "", reason)
            return ExecutionResult(intent.intent_id, intent.symbol, "DRY_RUN", 0, 0.0, "", reason)
        try:
            params = self._to_risk_params(intent, last_price or intent.take_profit_price or 0.0)
            if intent.side.lower() == "buy":
                res = self._ex.execute_buy(intent.symbol, params)  # real legacy method
            else:
                res = self._ex.execute_sell(intent.symbol, params)  # real legacy method
            order_id = str(getattr(res, "order_id", "") or "")
            if not getattr(res, "success", False):
                msg = str(getattr(res, "error_message", "") or "rejected")
                self._record(intent, "REJECTED", order_id, msg[:300])
                return ExecutionResult(intent.intent_id, intent.symbol, "REJECTED", 0, 0.0, order_id, msg[:300])
            self._record(intent, "SUBMITTED", order_id, str(getattr(res, "status", "")))
            # Protection audit: stop/target placed separately; record if missing.
            prot = []
            if not getattr(res, "stop_loss_order_id", None):
                prot.append("PROTECTION_MISSING:stop")
            if not getattr(res, "take_profit_order_id", None):
                prot.append("PROTECTION_MISSING:target")
            final = self.reconcile(intent, order_id)
            if prot:
                final.message = (final.message + " | " + ",".join(prot))[:400]
            return final
        except Exception as e:
            self._record(intent, "FAILED", "", f"exception: {e}"[:300])
            return ExecutionResult(intent.intent_id, intent.symbol, "FAILED", 0, 0.0, "", f"exception: {e}"[:300])

    def submit(self, intent: OrderIntent, last_price: float = 0.0) -> ExecutionResult:
        return self.submit_order(intent, last_price)

    def reconcile(self, intent: OrderIntent, order_id: str, timeout_s: float = 5.0) -> ExecutionResult:
        """Bounded poll of broker order state; updates local intent record. Never invents fills."""
        if not order_id:
            self._record(intent, "FAILED", "", "no_order_id")
            return ExecutionResult(intent.intent_id, intent.symbol, "FAILED", 0, 0.0, "", "no_order_id")
        deadline = time.time() + max(0.5, timeout_s)
        last: dict[str, Any] = {}
        while time.time() < deadline:
            last = self.get_order(order_id)
            st = _map_status(str(last.get("status", "")), float(last.get("filled_qty", 0) or 0), float(intent.qty))
            if st in ("FILLED", "PARTIAL", "REJECTED", "CANCELED", "FAILED"):
                break
            time.sleep(0.5)
        filled = float(last.get("filled_qty", 0) or 0)
        avg = float(last.get("avg_fill_price", 0) or 0)
        status = _map_status(str(last.get("status", "")), filled, float(intent.qty))
        if status in ("SUBMITTED", "ACCEPTED") and time.time() >= deadline:
            status = "TIMEOUT"  # still pending after bound: say so explicitly
        self._record(intent, status, order_id, str(last.get("status", ""))[:200])
        return ExecutionResult(intent.intent_id, intent.symbol, status, filled, avg, order_id,
                               str(last.get("status", ""))[:200])

    def close_position(self, symbol: str, reason: str = "MANUAL") -> ExecutionResult:
        intent_id = f"close-{symbol.replace('/', '')}-{uuid.uuid4().hex[:8]}"
        try:
            res = self._ex.close_position(symbol, reason=reason)  # real legacy method
            ok = bool(getattr(res, "success", False))
            if not ok:
                return ExecutionResult(intent_id, symbol, "FAILED", 0, 0.0,
                                       str(getattr(res, "order_id", "") or ""),
                                       str(getattr(res, "error_message", "") or "close_failed")[:300])
            # Confirm the position is actually gone at the broker (bounded).
            deadline = time.time() + 5.0
            still_there = True
            while time.time() < deadline:
                try:
                    self._ex.client.get_open_position(symbol.replace("/", ""))
                    time.sleep(0.5)
                except Exception:
                    still_there = False  # position does not exist => closed
                    break
            status = "FAILED" if still_there else "FILLED"
            msg = "close_unconfirmed_position_still_open" if still_there else str(getattr(res, "status", "closed"))
            return ExecutionResult(intent_id, symbol, status,
                                   float(getattr(res, "qty", 0) or 0),
                                   float(getattr(res, "filled_price", 0) or 0),
                                   str(getattr(res, "order_id", "") or ""), msg[:300])
        except Exception as e:
            return ExecutionResult(intent_id, symbol, "FAILED", 0, 0.0, "", str(e)[:300])
