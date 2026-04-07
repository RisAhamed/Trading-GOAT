#!/usr/bin/env python3
"""
Run strategy backtests using historical bars.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from core.config_loader import get_config
from core.market_data import MarketDataFetcher
from core.backtester import Backtester


def _load_bars_from_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required_cols = {"open", "high", "low", "close", "volume"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {', '.join(sorted(missing))}")

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")
    else:
        df.index = pd.to_datetime(df.index, utc=True)
    return df.sort_index()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Trading-GOAT backtest")
    parser.add_argument("--symbol", default="BTC/USD", help="Trading symbol")
    parser.add_argument("--interval", default="5Min", help="Interval for Alpaca fetch")
    parser.add_argument("--lookback", type=int, default=1500, help="Bars to fetch for Alpaca source")
    parser.add_argument("--csv", type=str, default="", help="Optional CSV path with OHLCV data")
    parser.add_argument("--initial-balance", type=float, default=100000.0, help="Starting balance")
    parser.add_argument("--fee-bps", type=float, default=5.0, help="Fee basis points per side")
    parser.add_argument("--slippage-bps", type=float, default=2.0, help="Slippage basis points per side")
    parser.add_argument("--out", type=str, default="", help="Optional output JSON path")
    args = parser.parse_args()

    config = get_config("config.yaml")
    backtester = Backtester(config)

    if args.csv:
        bars_df = _load_bars_from_csv(args.csv)
    else:
        fetcher = MarketDataFetcher(config)
        bars_df = fetcher.fetch_bars(args.symbol, args.interval, args.lookback)
        if bars_df is None or bars_df.empty:
            raise RuntimeError(f"No bars available for {args.symbol} ({args.interval})")

    result = backtester.run(
        symbol=args.symbol,
        bars_df=bars_df,
        initial_balance=args.initial_balance,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
    )
    data = result.to_dict()

    print(f"\nBacktest complete for {result.symbol}")
    print(f"Bars: {result.bars}")
    print(f"Start: ${result.initial_balance:,.2f}")
    print(f"End:   ${result.ending_balance:,.2f}")
    print(f"Return: {result.total_return_pct:+.2f}%")
    print(f"Max Drawdown: {result.max_drawdown_pct:.2f}%")
    print(f"Trades: {result.trades} | Win rate: {result.win_rate_pct:.1f}% | Profit factor: {result.profit_factor:.2f}")
    print(f"Fees: ${result.total_fees:.2f} | Slippage cost: ${result.total_slippage_cost:.2f}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"Saved report: {out_path}")


if __name__ == "__main__":
    main()

