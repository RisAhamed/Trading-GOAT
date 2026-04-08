import logging
import time

import pandas as pd

from .config_loader import ConfigLoader


logger = logging.getLogger(__name__)


class TradeExitEngine:
    def __init__(self, config: ConfigLoader) -> None:
        self.config = config
        self._position_entry_times: dict = {}  # symbol → entry timestamp
        self._position_entry_prices: dict = {}  # symbol → entry price
        self._partial_exits_done: set = set()  # symbols where 50% already taken

    def evaluate_position_exit(
        self,
        position: dict,
        current_bars: pd.DataFrame,
        current_signal: str,
        entry_time: float,
    ) -> dict:
        hold_default = {
            "action": "HOLD",
            "reason": "No exit condition met",
            "urgency": "NORMAL",
            "exit_pct": 0.0,
        }
        try:
            if not isinstance(position, dict):
                return hold_default

            symbol = str(position.get("symbol", ""))
            side = str(position.get("side", "long")).lower()

            # 1) SIGNAL REVERSAL EXIT
            signal_reversal_enabled = bool(
                getattr(getattr(self.config, "exit_engine", None), "signal_reversal_exit", True)
            )
            if signal_reversal_enabled:
                if current_signal == "SELL" and side == "long":
                    return {
                        "action": "EXIT_FULL",
                        "reason": "Signal reversed to SELL — thesis invalidated",
                        "urgency": "IMMEDIATE",
                        "exit_pct": 1.0,
                    }
                if current_signal == "BUY" and side == "short":
                    return {
                        "action": "EXIT_FULL",
                        "reason": "Signal reversed to BUY — thesis invalidated",
                        "urgency": "IMMEDIATE",
                        "exit_pct": 1.0,
                    }

            # 2) TIME-BASED EXIT
            hours_open = (time.time() - float(entry_time)) / 3600
            max_hours = float(getattr(getattr(self.config, "exit_engine", None), "max_trade_hours", 8))
            unrealized_pnl_pct = float(position.get("unrealized_plpc", 0.0)) * 100
            if hours_open > max_hours and abs(unrealized_pnl_pct) < 0.3:
                return {
                    "action": "EXIT_FULL",
                    "reason": f"Dead trade: {hours_open:.1f}h open, <0.3% move",
                    "urgency": "NORMAL",
                    "exit_pct": 1.0,
                }

            entry_price = float(position.get("avg_entry_price", 0.0))
            stop_price = float(position.get("stop_price", 0.0))
            if entry_price <= 0:
                entry_price = float(position.get("entry_price", 0.0))

            if current_bars is None or current_bars.empty:
                return hold_default

            close = pd.to_numeric(current_bars["close"], errors="coerce").dropna()
            if close.empty:
                return hold_default
            current_price = float(close.iloc[-1])

            r_multiple = 0.0
            if stop_price > 0 and entry_price > 0:
                risk_distance = abs(entry_price - stop_price)
                if risk_distance > 0:
                    if side == "short":
                        current_gain = entry_price - current_price
                    else:
                        current_gain = current_price - entry_price
                    r_multiple = current_gain / risk_distance

            # 3) PARTIAL PROFIT TAKING
            partial_trigger_r = float(
                getattr(getattr(self.config, "exit_engine", None), "partial_exit_r", 1.5)
            )
            if r_multiple >= partial_trigger_r and symbol and symbol not in self._partial_exits_done:
                self._partial_exits_done.add(symbol)
                return {
                    "action": "EXIT_PARTIAL",
                    "reason": (
                        f"Hit {r_multiple:.1f}R — taking 50% profit, "
                        f"moving stop to breakeven"
                    ),
                    "urgency": "NORMAL",
                    "exit_pct": 0.5,
                }

            # 4) MOVE STOP TO BREAKEVEN
            breakeven_trigger_r = float(
                getattr(getattr(self.config, "exit_engine", None), "breakeven_r", 1.0)
            )
            if r_multiple >= breakeven_trigger_r:
                return {
                    "action": "MOVE_STOP_BREAKEVEN",
                    "reason": f"At {r_multiple:.1f}R — move stop to breakeven",
                    "urgency": "NORMAL",
                    "exit_pct": 0.0,
                }

            return hold_default
        except Exception as e:
            logger.error(f"[EXIT ENGINE] evaluate_position_exit error: {e}")
            return hold_default

    def calculate_dynamic_position_size(
        self,
        base_risk_pct: float,
        recent_trades: list,
    ) -> float:
        try:
            if not isinstance(recent_trades, list) or len(recent_trades) < 3:
                return float(base_risk_pct)

            last5 = recent_trades[-5:]
            if not last5:
                return float(base_risk_pct)

            wins = [t for t in last5 if float(t.get("pnl", 0)) > 0]
            losses = [t for t in last5 if float(t.get("pnl", 0)) < 0]
            _ = losses
            win_rate = len(wins) / len(last5)

            consecutive_losses = 0
            for t in reversed(recent_trades):
                if float(t.get("pnl", 0)) < 0:
                    consecutive_losses += 1
                else:
                    break

            consecutive_wins = 0
            for t in reversed(recent_trades):
                if float(t.get("pnl", 0)) > 0:
                    consecutive_wins += 1
                else:
                    break

            if consecutive_losses >= 3:
                multiplier = 0.5
            elif consecutive_losses == 2:
                multiplier = 0.75
            elif consecutive_wins >= 3:
                multiplier = 1.25
            elif win_rate > 0.65:
                multiplier = 1.1
            else:
                multiplier = 1.0

            adjusted = float(base_risk_pct) * multiplier
            min_risk = float(getattr(getattr(self.config, "exit_engine", None), "min_risk_pct", 0.25))
            max_risk = float(getattr(getattr(self.config, "exit_engine", None), "max_risk_pct", 2.5))
            return max(min_risk, min(max_risk, adjusted))
        except Exception as e:
            logger.error(f"[EXIT ENGINE] calculate_dynamic_position_size error: {e}")
            return float(base_risk_pct)
