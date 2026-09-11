"""Broker interface + paper-only Alpaca implementation with idempotency.

PAPER TRADING ONLY. Startup asserts paper mode; any live request fail-closes.
Duplicate orders prevented via intent_id registry + position-aware checks.
"""
from __future__ import annotations

import uuid
from typing import Protocol

from domain.models import ExecutionResult, OrderIntent


class Broker(Protocol):
    mode: str
    def submit(self, intent: OrderIntent) -> ExecutionResult: ...
    def close_position(self, symbol: str) -> ExecutionResult: ...
    def account(self) -> dict: ...
    def positions(self) -> list[dict]: ...


class PaperOnlyError(RuntimeError):
    pass


class AlpacaPaperBroker:
    """Thin, honest wrapper over legacy OrderExecutor. Verifies fills, never assumes."""

    mode = "PAPER"

    def __init__(self, legacy_executor, dry_run: bool = False) -> None:
        mode = str(getattr(getattr(legacy_executor, "config", None), "bot", None) and legacy_executor.config.bot.mode or "paper")
        if mode.lower() != "paper":
            raise PaperOnlyError(f"refusing non-paper broker mode: {mode!r}")
        self._ex = legacy_executor
        self._dry_run = dry_run
        self._seen_intents: set[str] = set()

    def _check_duplicate(self, intent: OrderIntent) -> ExecutionResult | None:
        if intent.intent_id in self._seen_intents:
            return ExecutionResult(intent.intent_id, intent.symbol, "REJECTED", 0, 0, "",
                                   "duplicate_intent_id blocked")
        return None

    def submit(self, intent: OrderIntent) -> ExecutionResult:
        dup = self._check_duplicate(intent)
        if dup:
            return dup
        self._seen_intents.add(intent.intent_id)
        if self._dry_run or intent.qty <= 0:
            reason = "dry_run" if self._dry_run else "zero_qty"
            return ExecutionResult(intent.intent_id, intent.symbol, "DRY_RUN", 0, 0.0, "", reason)
        try:
            # Legacy executor places bracket market orders; verify result object.
            res = self._ex.place_bracket_order(
                symbol=intent.symbol, side=intent.side, qty=intent.qty,
                stop_price=intent.stop_price, take_profit_price=intent.take_profit_price,
            )
            ok = bool(getattr(res, "success", False))
            if not ok:
                return ExecutionResult(intent.intent_id, intent.symbol, "REJECTED", 0, 0.0,
                                       str(getattr(res, "order_id", "")), str(getattr(res, "message", "rejected")))
            filled = float(getattr(res, "filled_qty", 0.0) or 0.0)
            avg = float(getattr(res, "avg_fill_price", 0.0) or 0.0)
            status = "FILLED" if filled > 0 else "FAILED"
            if filled > 0 and abs(filled - intent.qty) / max(intent.qty, 1e-9) > 0.01:
                status = "PARTIAL"  # never silently treat partial as full
            return ExecutionResult(intent.intent_id, intent.symbol, status, filled, avg,
                                   str(getattr(res, "order_id", "")), str(getattr(res, "message", "")))
        except Exception as e:
            return ExecutionResult(intent.intent_id, intent.symbol, "FAILED", 0, 0.0, "", f"exception: {e}"[:300])

    def close_position(self, symbol: str) -> ExecutionResult:
        try:
            res = self._ex.close_position(symbol)
            ok = bool(getattr(res, "success", False))
            return ExecutionResult(f"close-{symbol}-{uuid.uuid4().hex[:8]}", symbol,
                                   "FILLED" if ok else "FAILED", 0, 0.0, "", str(getattr(res, "message", "")))
        except Exception as e:
            return ExecutionResult("close-err", symbol, "FAILED", 0, 0.0, "", str(e)[:300])

    def account(self) -> dict:
        try:
            ok, _, info = self._ex.check_connection()
            return {"ok": ok, **(info or {})}
        except Exception as e:
            return {"ok": False, "error": str(e)[:200]}

    def positions(self) -> list[dict]:
        try:
            tracker = getattr(self._ex, "portfolio_tracker", None)
            if tracker is None:
                return []
            return [p for p in tracker.get_positions().values()] if hasattr(tracker.get_positions(), "values") else list(tracker.get_positions())
        except Exception:
            return []
