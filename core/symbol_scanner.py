"""
Dynamic symbol scanner and ranker.
"""

import logging
import time
from typing import List, Optional

import pandas as pd

from .config_loader import ConfigLoader
from .market_data import MarketDataFetcher
from .political_signal_scanner import PoliticalSignalScanner


logger = logging.getLogger(__name__)
EPSILON = 1e-9


class SymbolScanner:
    """Scores and ranks candidate symbols for dynamic trade selection."""

    def __init__(self, market_data: MarketDataFetcher, config: ConfigLoader) -> None:
        try:
            self.market_data = market_data
            self.config = config

            scanner_cfg = getattr(self.config, "symbol_scanner", None)
            default_pool = list(getattr(self.config.markets, "crypto_pairs", ["BTC/USD", "ETH/USD"]))
            self._candidate_pool: List[str] = list(
                getattr(scanner_cfg, "candidate_pool", default_pool)
            )
            self._ranked_symbols: List[dict] = []
            self._last_scan_time: float = 0.0
            self._scan_interval: int = int(
                getattr(scanner_cfg, "scan_interval_seconds", 120)
            )
            self._btc_price_cache: dict = {}

            self.political_scanner = PoliticalSignalScanner(config)
            political_cfg = getattr(config, "political_signals", None)
            if political_cfg is None or bool(getattr(political_cfg, "enabled", False)):
                self._political_enabled = True
            else:
                self._political_enabled = False
        except Exception as e:
            logger.error(f"[SCANNER] init error: {e}")
            self.market_data = market_data
            self.config = config
            self._candidate_pool = ["BTC/USD", "ETH/USD"]
            self._ranked_symbols = []
            self._last_scan_time = 0.0
            self._scan_interval = 120
            self._btc_price_cache = {}
            self.political_scanner = PoliticalSignalScanner(config)
            self._political_enabled = False

    def _get_btc_change_pct(self) -> float:
        """Return short-term BTC change percent with scan-interval cache."""
        try:
            cache = self._btc_price_cache.get("btc_change")
            if cache and (time.time() - cache.get("time", 0)) < self._scan_interval:
                return float(cache.get("value", 0.0))

            btc_bars = self.market_data.fetch_bars("BTC/USD", "5Min", 10)
            if btc_bars is None or btc_bars.empty or len(btc_bars) < 3:
                return 0.0
            btc_close = pd.to_numeric(btc_bars["close"], errors="coerce").dropna()
            if len(btc_close) < 3 or float(btc_close.iloc[-3]) == 0:
                return 0.0
            change = float((btc_close.iloc[-1] - btc_close.iloc[-3]) / btc_close.iloc[-3] * 100)
            self._btc_price_cache["btc_change"] = {"value": change, "time": time.time()}
            return change
        except Exception as e:
            logger.debug(f"[SCANNER] BTC change calc failed: {e}")
            return 0.0

    def scan_and_rank(self) -> List[dict]:
        """Score candidate symbols and return sorted ranking list."""
        try:
            min_score_to_trade = int(
                getattr(getattr(self.config, "symbol_scanner", None), "min_score_to_trade", 40)
            )
            btc_guard_enabled = bool(
                getattr(
                    getattr(self.config, "symbol_scanner", None),
                    "btc_correlation_guard",
                    True,
                )
            )
            log_rankings = bool(
                getattr(getattr(self.config, "symbol_scanner", None), "log_rankings", True)
            )

            btc_change_pct = self._get_btc_change_pct() if btc_guard_enabled else 0.0
            results: List[dict] = []

            for symbol in self._candidate_pool:
                try:
                    bars = self.market_data.fetch_bars(symbol, "5Min", 50)
                    if bars is None or bars.empty or len(bars) < 20:
                        logger.debug(f"[SCANNER] {symbol}: insufficient bars")
                        continue

                    required = {"high", "low", "close", "volume"}
                    if not required.issubset(set(bars.columns)):
                        logger.debug(f"[SCANNER] {symbol}: missing OHLCV columns")
                        continue

                    df = bars.copy()
                    for col in ["high", "low", "close", "volume"]:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                    df = df.dropna(subset=["high", "low", "close", "volume"])
                    if len(df) < 20:
                        continue

                    close = df["close"]
                    volume = df["volume"]
                    if float(close.iloc[-10]) <= 0:
                        continue

                    # Score A — Momentum
                    momentum_pct = float((close.iloc[-1] - close.iloc[-10]) / close.iloc[-10] * 100)
                    if momentum_pct < -0.5:
                        logger.debug(f"[SCANNER] {symbol}: momentum too negative ({momentum_pct:.2f}%)")
                        continue

                    momentum_score = 0
                    if momentum_pct > 2.0:
                        momentum_score = 30
                    elif momentum_pct > 1.0:
                        momentum_score = 20
                    elif momentum_pct > 0.3:
                        momentum_score = 10

                    # Score B — Volume Surge
                    vol_base = float(volume.tail(20).mean()) if len(volume) >= 20 else 0.0
                    vol_ratio = float(volume.iloc[-1] / max(vol_base, EPSILON))
                    volume_score = 0
                    if vol_ratio >= 2.0:
                        volume_score = 25
                    elif vol_ratio >= 1.5:
                        volume_score = 18
                    elif vol_ratio >= 1.2:
                        volume_score = 10

                    # Score C — Volatility Quality
                    atr = self._calculate_atr(df, 14)
                    atr_pct = float((atr / max(float(close.iloc[-1]), EPSILON)) * 100)
                    volatility_score = 0
                    if 0.3 <= atr_pct <= 1.5:
                        volatility_score = 20
                    elif 1.5 < atr_pct <= 2.5:
                        volatility_score = 10
                    elif atr_pct < 0.3:
                        volatility_score = 5

                    # Score E — RSI Entry Quality
                    rsi = self._calculate_rsi(close, 14)
                    rsi_score = 0
                    if 35 <= rsi <= 55:
                        rsi_score = 10
                    elif 25 <= rsi < 35:
                        rsi_score = 8
                    elif 55 < rsi <= 65:
                        rsi_score = 5
                    elif rsi < 25:
                        rsi_score = 3

                    total_score = momentum_score + volume_score + volatility_score + rsi_score

                    # Score D — BTC correlation guard
                    if btc_guard_enabled and symbol != "BTC/USD":
                        if btc_change_pct < -1.5:
                            total_score = int(total_score * 0.3)
                        elif btc_change_pct < -0.5:
                            total_score = int(total_score * 0.7)
                        elif btc_change_pct >= 0.0:
                            total_score += 15

                    total_score = max(0, min(100, int(total_score)))

                    symbol_result = {
                        "symbol": symbol,
                        "total_score": total_score,
                        "momentum_pct": float(momentum_pct),
                        "vol_ratio": float(vol_ratio),
                        "atr_pct": float(atr_pct),
                        "rsi": float(rsi),
                        "btc_change_pct": float(btc_change_pct),
                        "rank": 0,
                        "tradeable": total_score >= min_score_to_trade,
                    }

                    # Political bias bonus
                    if self._political_enabled:
                        try:
                            bias = self.political_scanner.get_crypto_bias(symbol)
                            political_bonus = int((bias - 1.0) * 30)
                            total_score = max(0, min(100, total_score + political_bonus))
                            symbol_result["total_score"] = int(total_score)
                            symbol_result["political_bias"] = float(bias)
                            symbol_result["political_bonus"] = int(political_bonus)
                            symbol_result["tradeable"] = total_score >= min_score_to_trade
                            if political_bonus > 0:
                                logger.info(f"[POLITICAL] {symbol}: +{political_bonus}pts from political buys")
                        except Exception as e:
                            logger.debug(f"[POLITICAL] Bias error for {symbol}: {e}")

                    results.append(symbol_result)

                except Exception as e:
                    logger.error(f"[SCANNER] Error scoring {symbol}: {e}")
                    continue

            results.sort(key=lambda item: int(item.get("total_score", 0)), reverse=True)
            for i, item in enumerate(results, start=1):
                item["rank"] = i

            self._ranked_symbols = results
            self._last_scan_time = time.time()

            top3 = results[:3]
            if top3:
                top_str = " | ".join(
                    f"{row.get('symbol')}={row.get('total_score')}" for row in top3
                )
                logger.info(f"[SCANNER] Top symbols: {top_str}")

            if log_rankings and results:
                table = " | ".join(
                    f"#{row.get('rank')} {row.get('symbol')}({row.get('total_score')})"
                    for row in results
                )
                logger.info(f"[SCANNER] Ranking: {table}")

            return results

        except Exception as e:
            logger.error(f"[SCANNER] scan_and_rank error: {e}")
            return list(self._ranked_symbols)

    def get_tradeable_symbols(self) -> List[str]:
        """Return currently tradeable symbols based on cached/fresh ranking."""
        try:
            if (time.time() - self._last_scan_time) >= self._scan_interval or not self._ranked_symbols:
                self.scan_and_rank()

            tradeable = [
                row.get("symbol")
                for row in self._ranked_symbols
                if bool(row.get("tradeable", False))
            ]

            if not tradeable and self._ranked_symbols:
                return [str(self._ranked_symbols[0].get("symbol"))]
            if not tradeable and self._candidate_pool:
                return [str(self._candidate_pool[0])]
            return [str(sym) for sym in tradeable if sym]

        except Exception as e:
            logger.error(f"[SCANNER] get_tradeable_symbols error: {e}")
            if self._ranked_symbols:
                return [str(self._ranked_symbols[0].get("symbol"))]
            return list(self._candidate_pool[:1])

    def get_best_symbol(self) -> Optional[str]:
        """Return best currently tradeable symbol."""
        try:
            ranked = self.get_tradeable_symbols()
            return ranked[0] if ranked else None
        except Exception as e:
            logger.error(f"[SCANNER] get_best_symbol error: {e}")
            return None

    def get_ranking_summary(self) -> List[dict]:
        """Return current ranking cache."""
        try:
            return list(self._ranked_symbols)
        except Exception as e:
            logger.error(f"[SCANNER] get_ranking_summary error: {e}")
            return []

    def _calculate_rsi(self, close: pd.Series, period: int = 14) -> float:
        """Calculate Wilder RSI."""
        try:
            delta = close.diff()
            gains = delta.clip(lower=0.0)
            losses = -delta.clip(upper=0.0)
            avg_gain = gains.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
            avg_loss = losses.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
            rs = avg_gain / avg_loss.clip(lower=EPSILON)
            rsi = 100 - (100 / (1 + rs))
            if rsi.empty or pd.isna(rsi.iloc[-1]):
                return 50.0
            return float(rsi.iloc[-1])
        except Exception:
            return 50.0

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ATR from high/low/close."""
        try:
            high = pd.to_numeric(df["high"], errors="coerce")
            low = pd.to_numeric(df["low"], errors="coerce")
            close = pd.to_numeric(df["close"], errors="coerce")

            tr1 = high - low
            tr2 = (high - close.shift(1)).abs()
            tr3 = (low - close.shift(1)).abs()
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

            if atr.empty or pd.isna(atr.iloc[-1]):
                return 0.0
            return float(atr.iloc[-1])
        except Exception:
            return 0.0
