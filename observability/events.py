"""Tiny event bus: pipeline stages publish, dashboard/logs subscribe."""
from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Callable
from typing import Any

_subs: dict[str, list[Callable[[dict], None]]] = defaultdict(list)
_history: deque[dict] = deque(maxlen=500)

EVENTS = (
    "MARKET_DATA_UPDATED", "FEATURES_UPDATED", "AI_DECISION_CREATED",
    "SIGNAL_GENERATED", "SIGNAL_REJECTED", "RISK_APPROVED", "RISK_REJECTED",
    "ORDER_SUBMITTED", "ORDER_FILLED", "POSITION_OPENED", "POSITION_UPDATED",
    "POSITION_CLOSED", "EXIT_TRIGGERED", "ERROR_OCCURRED", "CYCLE_COMPLETED",
)


def subscribe(event: str, fn: Callable[[dict], None]) -> None:
    _subs[event].append(fn)


def publish(event: str, payload: dict[str, Any]) -> None:
    msg = {"event": event, **payload}
    _history.append(msg)
    for fn in list(_subs.get(event, [])):
        try:
            fn(msg)
        except Exception:
            pass


def recent(limit: int = 100) -> list[dict]:
    return list(_history)[-limit:]
