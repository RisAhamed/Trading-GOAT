"""
Deterministic backtesting engine for the trading strategy.

Uses existing indicator/signal/risk modules to simulate trades on historical OHLCV data.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import pandas as pd

from .config_loader import ConfigLoader, get_config
from .indicators import IndicatorCalculator
from .signal_engine import SignalEngine
from .risk_manager import RiskManager


@dataclass
class BacktestTrade:
    symbol: str
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    qty: float
    pnl: float
    pnl_pct: float
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BacktestResult:
    symbol: str
    bars: int
    initial_balance: float
    ending_balance: float
    total_return_pct: float
    max_drawdown_pct: float
    trades: int
    wins: int
    losses: int
    win_rate_pct: float
    profit_factor: float
    avg_trade_return_pct: float
    total_fees: float
    total_slippage_cost: float
    trade_log: List[BacktestTrade]

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["trade_log"] = [t.to_dict() for t in self.trade_log]
        return data


class Backtester:
    """Runs a single-symbol long-only backtest."""

    # Extra bars beyond lookback windows to ensure stable indicator warmup.
    MIN_WARMUP_BUFFER = 5
    # Sentinel value used when there are profits but no realized losses.
    INFINITE_PROFIT_FACTOR = 999.0

    def __init__(self, config: Optional[ConfigLoader] = None) -> None:
        self.config = config or get_config()
        self.indicators = IndicatorCalculator(self.config)
        self.signal_engine = SignalEngine(self.config)
        self.risk_manager = RiskManager(self.config)

    def run(
        self,
        symbol: str,
        bars_df: pd.DataFrame,
        initial_balance: float = 100000.0,
        fee_bps: float = 5.0,
        slippage_bps: float = 2.0,
    ) -> BacktestResult:
        """
        Run deterministic backtest on OHLCV bars.

        bars_df must include: open, high, low, close, volume and datetime index.
        """
        df = bars_df.copy()
        required_cols = {"open", "high", "low", "close", "volume"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
        required_bars = (
            max(
                self.config.timeframes.trend_lookback_bars,
                self.config.timeframes.entry_lookback_bars,
            )
            + self.MIN_WARMUP_BUFFER
        )
        if len(df) < required_bars:
            raise ValueError("Not enough bars for backtest warmup")
        if slippage_bps < 0:
            raise ValueError("slippage_bps must be >= 0")
        if fee_bps < 0:
            raise ValueError("fee_bps must be >= 0")

        cash = initial_balance
        qty = 0.0
        entry_price = 0.0
        entry_time = ""
        stop_price: Optional[float] = None
        take_profit_price: Optional[float] = None
        fees_total = 0.0
        slippage_total = 0.0
        trade_log: List[BacktestTrade] = []
        equity_curve: List[float] = []

        trend_lb = self.config.timeframes.trend_lookback_bars
        entry_lb = self.config.timeframes.entry_lookback_bars
        warmup = max(trend_lb, entry_lb)
        fee_rate = fee_bps / 10000.0
        slippage_rate = slippage_bps / 10000.0

        for i in range(warmup, len(df)):
            window = df.iloc[: i + 1]
            trend_df = window.tail(trend_lb)
            entry_df = window.tail(entry_lb)
            current_bar = df.iloc[i]
            current_close = float(current_bar["close"])
            current_high = float(current_bar["high"])
            current_low = float(current_bar["low"])
            current_time = str(df.index[i])

            symbol_indicators = self.indicators.calculate_for_symbol(
                symbol=symbol,
                trend_df=trend_df,
                entry_df=entry_df,
                current_price=current_close,
            )
            quick_signal = self.signal_engine.quick_signal(symbol_indicators)

            if qty <= 0.0:
                if quick_signal == "BUY":
                    risk_params = self.risk_manager.calculate_position_size(
                        symbol=symbol,
                        entry_price=current_close,
                        portfolio_value=cash,
                        side="long",
                        atr=symbol_indicators.entry_tf.atr,
                    )
                    risk_params = self.risk_manager.validate_risk_parameters(risk_params)
                    if not risk_params.is_allowed:
                        equity_curve.append(cash)
                        continue

                    is_allowed, _ = self.risk_manager.check_trade_allowed(
                        symbol=symbol,
                        action="BUY",
                        current_positions=[],
                        portfolio_value=cash,
                        cash_available=cash,
                        position_value=risk_params.position_value,
                    )
                    if not is_allowed:
                        equity_curve.append(cash)
                        continue

                    fill_price = current_close * (1 + slippage_rate)
                    slippage_total += max(0.0, fill_price - current_close) * risk_params.qty
                    notional = fill_price * risk_params.qty
                    fee = notional * fee_rate
                    total_cost = notional + fee
                    if total_cost > cash:
                        equity_curve.append(cash)
                        continue

                    cash -= total_cost
                    fees_total += fee
                    qty = risk_params.qty
                    entry_price = fill_price
                    entry_time = current_time
                    stop_price = risk_params.stop_price
                    take_profit_price = risk_params.take_profit_price
            else:
                exit_reason = ""
                exit_raw_price: Optional[float] = None

                if stop_price is not None and current_low <= stop_price:
                    exit_reason = "STOP_LOSS"
                    exit_raw_price = stop_price
                elif take_profit_price is not None and current_high >= take_profit_price:
                    exit_reason = "TAKE_PROFIT"
                    exit_raw_price = take_profit_price
                elif quick_signal == "SELL":
                    exit_reason = "SIGNAL_SELL"
                    exit_raw_price = current_close

                if exit_raw_price is not None:
                    fill_price = exit_raw_price * (1 - slippage_rate)
                    slippage_total += max(0.0, exit_raw_price - fill_price) * qty
                    notional = fill_price * qty
                    fee = notional * fee_rate
                    fees_total += fee
                    cash += notional - fee

                    gross_pnl = (fill_price - entry_price) * qty
                    net_pnl = gross_pnl - ((entry_price * qty) * fee_rate) - fee
                    pnl_pct = (net_pnl / (entry_price * qty)) * 100 if entry_price > 0 else 0.0

                    trade_log.append(
                        BacktestTrade(
                            symbol=symbol,
                            entry_time=entry_time,
                            exit_time=current_time,
                            entry_price=entry_price,
                            exit_price=fill_price,
                            qty=qty,
                            pnl=net_pnl,
                            pnl_pct=pnl_pct,
                            reason=exit_reason,
                        )
                    )

                    qty = 0.0
                    entry_price = 0.0
                    entry_time = ""
                    stop_price = None
                    take_profit_price = None

            equity = cash + (qty * current_close if qty > 0 else 0.0)
            equity_curve.append(equity)

        # Force-close any open trade at final close
        if qty > 0.0:
            final_close = float(df["close"].iloc[-1])
            final_time = str(df.index[-1])
            fill_price = final_close * (1 - slippage_rate)
            slippage_total += max(0.0, final_close - fill_price) * qty
            notional = fill_price * qty
            fee = notional * fee_rate
            fees_total += fee
            cash += notional - fee

            gross_pnl = (fill_price - entry_price) * qty
            net_pnl = gross_pnl - ((entry_price * qty) * fee_rate) - fee
            pnl_pct = (net_pnl / (entry_price * qty)) * 100 if entry_price > 0 else 0.0
            trade_log.append(
                BacktestTrade(
                    symbol=symbol,
                    entry_time=entry_time,
                    exit_time=final_time,
                    entry_price=entry_price,
                    exit_price=fill_price,
                    qty=qty,
                    pnl=net_pnl,
                    pnl_pct=pnl_pct,
                    reason="FORCED_CLOSE",
                )
            )
            qty = 0.0
            equity_curve.append(cash)

        ending_balance = cash
        total_return_pct = ((ending_balance - initial_balance) / initial_balance * 100) if initial_balance > 0 else 0.0
        max_drawdown_pct = self._max_drawdown_pct(equity_curve)
        wins = sum(1 for t in trade_log if t.pnl > 0)
        losses = sum(1 for t in trade_log if t.pnl < 0)
        trades = len(trade_log)
        win_rate = (wins / trades * 100) if trades > 0 else 0.0
        avg_trade = (sum(t.pnl_pct for t in trade_log) / trades) if trades > 0 else 0.0

        gross_wins = sum(t.pnl for t in trade_log if t.pnl > 0)
        gross_losses = abs(sum(t.pnl for t in trade_log if t.pnl < 0))
        profit_factor = (
            (gross_wins / gross_losses)
            if gross_losses > 0
            else (self.INFINITE_PROFIT_FACTOR if gross_wins > 0 else 0.0)
        )

        return BacktestResult(
            symbol=symbol,
            bars=len(df),
            initial_balance=initial_balance,
            ending_balance=ending_balance,
            total_return_pct=total_return_pct,
            max_drawdown_pct=max_drawdown_pct,
            trades=trades,
            wins=wins,
            losses=losses,
            win_rate_pct=win_rate,
            profit_factor=profit_factor,
            avg_trade_return_pct=avg_trade,
            total_fees=fees_total,
            total_slippage_cost=slippage_total,
            trade_log=trade_log,
        )

    @staticmethod
    def _max_drawdown_pct(equity_curve: List[float]) -> float:
        if not equity_curve:
            return 0.0
        peak = equity_curve[0]
        max_dd = 0.0
        for v in equity_curve:
            if v > peak:
                peak = v
            if peak > 0:
                dd = ((peak - v) / peak) * 100
                if dd > max_dd:
                    max_dd = dd
        return max_dd
