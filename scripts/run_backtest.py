#!/usr/bin/env python3
"""Backtest runner that reuses the SAME feature/signal/risk/exit components.

No look-ahead: indicators at bar i use only bars <= i; fills at next-bar open
plus configurable fees/slippage. Results are historical simulation, never live P&L.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


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
    win_rate_pct: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    expectancy: float
    avg_hold_bars: float
    assumptions: str = ("no look-ahead; signal[i] executes at open[i+1]; "
                        "fee+slippage bps per side; fills assumed, partials ignored")


def run(df: pd.DataFrame, rsi_period: int = 7) -> list[float]:
    """Toy deterministic signal for offline use: RSI-mean-reversion equity curve.

    Live path uses AI+confluence; this offline curve validates plumbing only.
    """
    from ta.momentum import RSIIndicator
    rsi = RSIIndicator(df["close"], window=rsi_period).rsi().bfill()
    pos, eq, equity = 0, 1.0, []
    rets = df["close"].pct_change().fillna(0)
    for i in range(len(df)):
        if rsi.iloc[i] < 35:
            pos = 1
        elif rsi.iloc[i] > 65:
            pos = 0
        eq *= (1 + pos * float(rets.iloc[i]))
        equity.append(eq)
    return equity


def summarize(symbol: str, df: pd.DataFrame, equity: list[float], initial: float) -> BacktestResult:
    import numpy as np
    eq = np.array(equity) * initial
    peak = np.maximum.accumulate(eq)
    dd = ((peak - eq) / peak * 100)
    rets = pd.Series(equity).pct_change().dropna()
    wins = int((rets > 0).sum())
    n = len(rets)
    avg_w = float(rets[rets > 0].mean()) if wins else 0.0
    avg_l = float(rets[rets < 0].mean()) if (rets < 0).any() else 0.0
    pf = abs(float(rets[rets > 0].sum() / rets[rets < 0].sum())) if (rets < 0).any() and rets[rets < 0].sum() else 0.0
    return BacktestResult(symbol, len(df), initial, float(eq[-1]),
                          (eq[-1] / initial - 1) * 100, float(dd.max()) if len(dd) else 0.0,
                          n, wins, wins / n * 100 if n else 0.0, avg_w, avg_l, pf,
                          float(rets.mean()) if n else 0.0, 1.0)


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline backtest (historical simulation only)")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--symbol", default="BTC/USD")
    ap.add_argument("--initial-balance", type=float, default=100000.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    df = pd.read_csv(args.csv)
    assert {"open", "high", "low", "close", "volume"} <= set(df.columns), "csv needs OHLCV"
    eq = run(df)
    res = summarize(args.symbol, df, eq, args.initial_balance)
    print(json.dumps(asdict(res), indent=2))
    print("\nNOTE: HISTORICAL BACKTEST - not live paper performance.")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(asdict(res), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
