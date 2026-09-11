"""Portfolio state: single source of truth, reconcilable against broker."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PortfolioState:
    equity: float = 0.0
    cash: float = 0.0
    open_positions: int = 0
    total_exposure_notional: float = 0.0
    symbol_exposure: dict[str, float] = field(default_factory=dict)
    daily_pnl_pct: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0


def from_legacy(tracker, account_info: dict | None = None) -> PortfolioState:
    st = PortfolioState()
    try:
        positions = tracker.get_positions() if tracker else {}
        items = positions.values() if hasattr(positions, "values") else positions
        items = list(items or [])
        st.open_positions = len(items)
        for p in items:
            sym = p.get("symbol", "?") if isinstance(p, dict) else getattr(p, "symbol", "?")
            qty = p.get("qty", 0) if isinstance(p, dict) else getattr(p, "qty", 0)
            price = p.get("current_price", 0) or p.get("entry_price", 0) if isinstance(p, dict) else 0
            notional = abs(float(qty or 0)) * float(price or 0)
            st.symbol_exposure[sym] = st.symbol_exposure.get(sym, 0.0) + notional
            st.total_exposure_notional += notional
            st.unrealized_pnl += float(p.get("unrealized_pnl", 0) or 0) if isinstance(p, dict) else 0.0
    except Exception:
        pass
    if account_info:
        try:
            st.equity = float(account_info.get("portfolio_value", 0) or 0)
            st.cash = float(account_info.get("cash", 0) or 0)
        except (TypeError, ValueError):
            pass
    return st
