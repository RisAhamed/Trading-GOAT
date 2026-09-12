"""Broker-authoritative portfolio state + deterministic reconciliation.

BROKER STATE = SOURCE OF TRUTH for current positions/orders/account.
Local JSON (logs/results/*) is history/audit only, never live truth.

- PortfolioSnapshot: built from broker every cycle (account/equity/cash/
  buying power, broker positions, open orders), split into strategy vs
  external positions.
- Unsupported assets (e.g. NVDA/XOM stock in a crypto-only strategy) are
  classified EXTERNAL_UNSUPPORTED_POSITION: tracked separately, never counted
  in strategy capacity, never fed to crypto market-data, exits retried with
  backoff instead of every-cycle spam. Never auto-liquidated.
- reconcile(): startup + periodic compare of broker vs local persisted state;
  broker wins; discrepancies emitted as RECONCILIATION_MISMATCH events and
  repaired locally without fabricating trades.
- Discovered positions get entry_time_unknown=True (never invent entry_time).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

RECONCILE_STORE = Path("logs") / "reconciliation.json"

# Strategy universe: BASE/QUOTE crypto symbols. Anything else is external.
CRYPTO_BASES = ("BTC", "ETH", "SOL", "AVAX", "DOGE", "LTC", "XRP", "ADA")


def normalize_symbol(raw: str) -> str:
    s = str(raw or "").strip().upper()
    if "/" in s:
        return s
    for base in CRYPTO_BASES:
        if s == base + "USD":
            return f"{base}/USD"
    return s  # e.g. NVDA, XOM, EURUSD stay as-is -> external


def is_strategy_symbol(symbol: str) -> bool:
    s = normalize_symbol(symbol)
    return "/" in s and s.split("/")[0] in CRYPTO_BASES and s.endswith("/USD")


@dataclass
class BrokerPosition:
    symbol: str  # normalized
    raw_symbol: str
    qty: float = 0.0
    side: str = "long"
    entry_price: float = 0.0
    current_price: float = 0.0
    market_value: float = 0.0
    unrealized_pl: float = 0.0
    external: bool = False  # True => EXTERNAL_UNSUPPORTED_POSITION
    entry_time_unknown: bool = True


@dataclass
class PortfolioSnapshot:
    equity: float = 0.0
    cash: float = 0.0
    buying_power: float = 0.0
    broker_total_positions: int = 0
    strategy_open_positions: int = 0
    external_positions: list[BrokerPosition] = field(default_factory=list)
    strategy_positions: list[BrokerPosition] = field(default_factory=list)
    open_orders: list[dict] = field(default_factory=list)
    total_exposure_notional: float = 0.0
    symbol_exposure: dict[str, float] = field(default_factory=dict)
    daily_pnl_pct: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    broker_ok: bool = False
    retrieved_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def snapshot_from_broker(broker) -> PortfolioSnapshot:
    """Build the authoritative snapshot. Unknown values stay 0/empty + broker_ok=False."""
    snap = PortfolioSnapshot()
    try:
        acct = broker.account() or {}
        snap.broker_ok = bool(acct.get("ok", False))
        snap.equity = float(acct.get("portfolio_value", 0) or acct.get("equity", 0) or 0)
        snap.cash = float(acct.get("cash", 0) or 0)
        snap.buying_power = float(acct.get("buying_power", 0) or 0)
    except (TypeError, ValueError, AttributeError):
        snap.broker_ok = False
    try:
        for p in broker.positions() or []:
            if not isinstance(p, dict) or "error" in p:
                continue
            norm = normalize_symbol(p.get("symbol", ""))
            bp = BrokerPosition(
                symbol=norm, raw_symbol=str(p.get("symbol", "")),
                qty=float(p.get("qty", 0) or 0), side=str(p.get("side", "long")),
                entry_price=float(p.get("entry_price", 0) or 0),
                current_price=float(p.get("current_price", 0) or 0),
                market_value=float(p.get("market_value", 0) or 0),
                unrealized_pl=float(p.get("unrealized_pl", 0) or 0),
                external=not is_strategy_symbol(norm),
                entry_time_unknown=True,
            )
            if bp.external:
                snap.external_positions.append(bp)
            else:
                snap.strategy_positions.append(bp)
                notional = abs(bp.qty) * (bp.current_price or bp.entry_price)
                snap.symbol_exposure[norm] = snap.symbol_exposure.get(norm, 0.0) + notional
                snap.total_exposure_notional += notional
                snap.unrealized_pnl += bp.unrealized_pl
    except (TypeError, ValueError, AttributeError):
        pass
    try:
        snap.open_orders = broker.open_orders() or []
    except Exception:
        snap.open_orders = []
    snap.broker_total_positions = len(snap.strategy_positions) + len(snap.external_positions)
    snap.strategy_open_positions = len(snap.strategy_positions)
    return snap


@dataclass
class ReconcileResult:
    mismatches: list[str] = field(default_factory=list)
    repaired: list[str] = field(default_factory=list)
    broker_wins: bool = True


def reconcile(broker_snapshot: PortfolioSnapshot, local_positions: dict) -> ReconcileResult:
    """Compare broker truth vs local persisted state. Broker wins; repair locally.

    Never fabricates trades: repairs only mark local records to match broker
    (e.g. drop locally-OPEN symbols the broker no longer holds).
    """
    res = ReconcileResult()
    broker_syms = {p.symbol for p in broker_snapshot.strategy_positions}
    local_syms = {str(k) for k, v in (local_positions or {}).items()
                  if isinstance(v, dict) and str(v.get("status", "")).upper() == "OPEN"}
    for sym in sorted(local_syms - broker_syms):
        res.mismatches.append(f"local_OPEN_but_broker_flat:{sym}")
        res.repaired.append(f"local_{sym}_marked_closed_to_match_broker")
    for sym in sorted(broker_syms - local_syms):
        res.mismatches.append(f"broker_OPEN_but_local_missing:{sym}")
        res.repaired.append(f"local_{sym}_adopted_from_broker_entry_unknown")
    for p in broker_snapshot.external_positions:
        res.mismatches.append(f"external_unsupported_position:{p.raw_symbol}_qty={p.qty}")
    try:
        RECONCILE_STORE.parent.mkdir(parents=True, exist_ok=True)
        RECONCILE_STORE.write_text(json.dumps({
            "at": datetime.now(timezone.utc).isoformat(),
            "mismatches": res.mismatches, "repaired": res.repaired,
        }, indent=2), encoding="utf-8")
    except Exception:
        pass
    return res


def from_legacy(tracker, account_info: dict | None = None):
    """Legacy adapter kept for dashboard/tests (broker snapshot is preferred)."""
    from portfolio.state import PortfolioState as LegacyState

    st = LegacyState()
    try:
        positions = tracker.get_positions() if tracker else {}
        items = positions.values() if hasattr(positions, "values") else positions
        items = list(items or [])
        st.open_positions = len(items)
        for p in items:
            if isinstance(p, dict):
                sym, qty = p.get("symbol", "?"), p.get("qty", 0)
                price = p.get("current_price", 0) or p.get("entry_price", 0)
                st.unrealized_pnl += float(p.get("unrealized_pnl", 0) or 0)
            else:
                sym, qty = getattr(p, "symbol", "?"), getattr(p, "qty", 0)
                price = getattr(p, "current_price", 0) or getattr(p, "entry_price", 0)
            notional = abs(float(qty or 0)) * float(price or 0)
            st.symbol_exposure[sym] = st.symbol_exposure.get(sym, 0.0) + notional
            st.total_exposure_notional += notional
    except Exception:
        pass
    if account_info:
        try:
            st.equity = float(account_info.get("portfolio_value", 0) or 0)
            st.cash = float(account_info.get("cash", 0) or 0)
        except (TypeError, ValueError):
            pass
    return st
