import logging
import time
from typing import Optional

import pandas as pd
import requests

from .config_loader import ConfigLoader


logger = logging.getLogger(__name__)

# Kraken symbol map: base coin → Kraken pair name
# Kraken public trades endpoint needs no API key, no account
KRAKEN_SYMBOL_MAP = {
    "BTC": "XBTUSD",
    "ETH": "ETHUSD",
    "SOL": "SOLUSD",
    "AVAX": "AVAXUSD",
    "DOGE": "XDGUSD",
    "LTC": "LTCUSD",
    "LINK": "LINKUSD",
    "MATIC": "MATICUSD",
    "XRP": "XRPUSD",
}


class MarketIntelligence:
    """
    Fetches real-time market intelligence signals and converts them
    to score modifiers for the SymbolScanner.

    Data sources used (all 100% free, no API key required):
      - alternative.me/fng  → Fear & Greed index
      - api.kraken.com      → Large-trade buy/sell pressure (public REST, no auth)
      - Existing bar data   → Relative Strength vs BTC, Breakout detection
    """

    def __init__(self, config: ConfigLoader) -> None:
        self.config = config
        self._fg_cache: dict = {}
        self._whale_cache: dict = {}  # keyed by symbol
        self._last_fg_fetch: float = 0
        self._fg_interval: int = 900  # 15 min
        self._whale_interval: int = 300  # 5 min per symbol

    def _default_fg(self) -> dict:
        return {
            "score": 50,
            "classification": "Neutral",
            "trade_bias": "NEUTRAL",
            "score_modifier": 0,
        }

    def _default_whale(self, symbol: str) -> dict:
        return {
            "symbol": symbol,
            "large_buys": 0,
            "large_sells": 0,
            "net_flow": "NEUTRAL",
            "score_modifier": 0,
        }

    def _default_rs(self, outperforming: bool = False) -> dict:
        return {
            "coin_return_pct": 0.0,
            "btc_return_pct": 0.0,
            "relative_strength": 0.0,
            "outperforming": outperforming,
            "score_modifier": 0,
        }

    def _default_breakout(self) -> dict:
        return {
            "breakout_type": "NONE",
            "resistance": 0.0,
            "support": 0.0,
            "range_pct": 0.0,
            "vol_ratio": 1.0,
            "score_modifier": 0,
        }

    def _neutral_combined(self, symbol: str) -> dict:
        return {
            "symbol": symbol,
            "fear_greed_score": 50,
            "fear_greed_class": "Neutral",
            "whale_flow": "NEUTRAL",
            "relative_strength": 0.0,
            "breakout_type": "NONE",
            "total_modifier": 0,
            "summary": "",
        }

    def get_fear_greed_score(self, symbol: str = "BTC") -> dict:
        try:
            _ = symbol  # kept for interface symmetry and potential future per-symbol logic
            if self._fg_cache and (time.time() - self._last_fg_fetch) < self._fg_interval:
                return dict(self._fg_cache)

            url = "https://api.alternative.me/fng/?limit=1"
            response = requests.get(url, timeout=8)
            response.raise_for_status()
            data = response.json()

            score = int(data["data"][0]["value"])
            classification = str(data["data"][0]["value_classification"])

            if score <= 25:
                trade_bias = "BUY_BIAS"
                score_modifier = 10
            elif 26 <= score <= 45:
                trade_bias = "BUY_BIAS"
                score_modifier = 5
            elif 46 <= score <= 60:
                trade_bias = "NEUTRAL"
                score_modifier = 0
            elif 61 <= score <= 75:
                trade_bias = "SELL_BIAS"
                score_modifier = -5
            else:
                trade_bias = "SELL_BIAS"
                score_modifier = -15

            result = {
                "score": score,
                "classification": classification,
                "trade_bias": trade_bias,
                "score_modifier": score_modifier,
            }
            self._fg_cache = dict(result)
            self._last_fg_fetch = time.time()
            return result
        except Exception as e:
            logger.debug(f"[INTEL] Fear/Greed fetch failed: {e}")
            return self._default_fg()

    def get_whale_signal(self, symbol: str) -> dict:
        try:
            base_coin = symbol.split("/")[0].upper()
            kraken_pair = KRAKEN_SYMBOL_MAP.get(base_coin)
            if not kraken_pair:
                return self._default_whale(symbol)

            if symbol in self._whale_cache:
                cached = self._whale_cache[symbol]
                if time.time() - cached.get("timestamp", 0) < self._whale_interval:
                    return dict(cached.get("data", self._default_whale(symbol)))

            url = "https://api.kraken.com/0/public/Trades"
            params = {"pair": kraken_pair, "count": 1000}
            response = requests.get(url, params=params, timeout=8)
            response.raise_for_status()
            data = response.json()

            result_section = data.get("result", {})
            trades: Optional[list] = result_section.get(kraken_pair)
            if trades is None:
                for key, value in result_section.items():
                    if key != "last" and isinstance(value, list):
                        trades = value
                        break
            if trades is None:
                return self._default_whale(symbol)

            intel_cfg = getattr(self.config, "market_intelligence", None)
            min_usd = float(getattr(intel_cfg, "min_whale_usd", 500000))

            large_buys = 0
            large_sells = 0

            for trade in trades:
                if not isinstance(trade, list) or len(trade) < 4:
                    continue
                price = float(trade[0])
                volume = float(trade[1])
                usd_value = price * volume
                side = trade[3]  # "b" or "s"
                if usd_value >= min_usd:
                    if side == "b":
                        large_buys += 1
                    else:
                        large_sells += 1

            if large_buys >= 3 and large_buys > large_sells * 2:
                net_flow = "ACCUMULATION"
                modifier = 15
            elif large_buys >= 2 and large_buys > large_sells:
                net_flow = "ACCUMULATION"
                modifier = 8
            elif large_sells >= 3 and large_sells > large_buys * 2:
                net_flow = "DISTRIBUTION"
                modifier = -10
            elif large_sells > large_buys:
                net_flow = "DISTRIBUTION"
                modifier = -5
            else:
                net_flow = "NEUTRAL"
                modifier = 0

            result = {
                "symbol": symbol,
                "large_buys": int(large_buys),
                "large_sells": int(large_sells),
                "net_flow": net_flow,
                "score_modifier": int(modifier),
            }
            self._whale_cache[symbol] = {"data": dict(result), "timestamp": time.time()}
            return result
        except Exception as e:
            logger.debug(f"[INTEL] Whale signal fetch failed for {symbol}: {e}")
            return self._default_whale(symbol)

    def calculate_relative_strength(
        self,
        symbol: str,
        symbol_bars: pd.DataFrame,
        btc_bars: pd.DataFrame,
        lookback: int = 10,
    ) -> dict:
        try:
            intel_cfg = getattr(self.config, "market_intelligence", None)
            lookback = int(getattr(intel_cfg, "relative_strength_lookback", lookback))

            if symbol == "BTC/USD":
                return self._default_rs(outperforming=True)

            if (
                symbol_bars is None
                or btc_bars is None
                or len(symbol_bars) < lookback
                or len(btc_bars) < lookback
            ):
                return self._default_rs()

            coin_close = pd.to_numeric(symbol_bars["close"], errors="coerce").dropna()
            btc_close = pd.to_numeric(btc_bars["close"], errors="coerce").dropna()

            if len(coin_close) < lookback or len(btc_close) < lookback:
                return self._default_rs()

            base_coin = float(coin_close.iloc[-lookback])
            base_btc = float(btc_close.iloc[-lookback])
            if base_coin == 0 or base_btc == 0:
                return self._default_rs()

            coin_return = (float(coin_close.iloc[-1]) - base_coin) / base_coin * 100
            btc_return = (float(btc_close.iloc[-1]) - base_btc) / base_btc * 100
            relative_strength = coin_return - btc_return

            if relative_strength > 2.0:
                modifier = 15
            elif relative_strength > 1.0:
                modifier = 10
            elif relative_strength > 0.5:
                modifier = 5
            elif relative_strength < -1.0:
                modifier = -10
            elif relative_strength < -0.5:
                modifier = -5
            else:
                modifier = 0

            outperforming = relative_strength > 0.5

            return {
                "coin_return_pct": float(coin_return),
                "btc_return_pct": float(btc_return),
                "relative_strength": float(relative_strength),
                "outperforming": bool(outperforming),
                "score_modifier": int(modifier),
            }
        except Exception as e:
            logger.debug(f"[INTEL] Relative strength calc failed for {symbol}: {e}")
            return self._default_rs()

    def get_breakout_signal(self, bars: pd.DataFrame, lookback_bars: int = 10) -> dict:
        try:
            intel_cfg = getattr(self.config, "market_intelligence", None)
            lookback_bars = int(getattr(intel_cfg, "breakout_lookback_bars", lookback_bars))

            if bars is None or len(bars) < lookback_bars + 1:
                return self._default_breakout()

            high = pd.to_numeric(bars["high"], errors="coerce")
            low = pd.to_numeric(bars["low"], errors="coerce")
            close = pd.to_numeric(bars["close"], errors="coerce")
            volume = pd.to_numeric(bars["volume"], errors="coerce")

            resistance = float(high.iloc[-lookback_bars:-1].max())
            support = float(low.iloc[-lookback_bars:-1].min())
            current_close = float(close.iloc[-1])
            current_vol = float(volume.iloc[-1])
            avg_vol = float(volume.iloc[-lookback_bars:].mean())
            vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0
            range_pct = (resistance - support) / support * 100 if support > 0 else 0.0

            breakout_up = current_close > resistance and vol_ratio >= 1.5
            breakout_down = current_close < support and vol_ratio >= 1.5

            if breakout_up:
                breakout_type = "BULLISH"
                score_modifier = 20
                logger.info(
                    f"[BREAKOUT] BULLISH detected: close={current_close:.4f} > "
                    f"resist={resistance:.4f}, vol_ratio={vol_ratio:.2f}"
                )
            elif breakout_down:
                breakout_type = "BEARISH"
                score_modifier = -20
                logger.info(
                    f"[BREAKOUT] BEARISH detected: close={current_close:.4f} < "
                    f"support={support:.4f}, vol_ratio={vol_ratio:.2f}"
                )
            else:
                breakout_type = "NONE"
                score_modifier = 0

            return {
                "breakout_type": breakout_type,
                "resistance": float(resistance),
                "support": float(support),
                "range_pct": float(range_pct),
                "vol_ratio": float(vol_ratio),
                "score_modifier": int(score_modifier),
            }
        except Exception as e:
            logger.debug(f"[INTEL] Breakout detection failed: {e}")
            return self._default_breakout()

    def get_combined_intelligence(
        self, symbol: str, bars: pd.DataFrame, btc_bars: pd.DataFrame
    ) -> dict:
        try:
            fg = self.get_fear_greed_score()
            whale = self.get_whale_signal(symbol)
            rs = self.calculate_relative_strength(symbol, bars, btc_bars)
            bo = self.get_breakout_signal(bars)
        except Exception as e:
            logger.error(f"[INTEL] Combined intelligence error for {symbol}: {e}")
            return self._neutral_combined(symbol)

        fg_mod = int(fg.get("score_modifier", 0))
        whale_mod = int(whale.get("score_modifier", 0))
        rs_mod = int(rs.get("score_modifier", 0))
        bo_mod = int(bo.get("score_modifier", 0))

        total = fg_mod + whale_mod + rs_mod + bo_mod
        total = max(-30, min(+30, total))

        fg_class = fg.get("classification", "Neutral")
        whale_flow = whale.get("net_flow", "NEUTRAL")
        rs_val = rs.get("relative_strength", 0.0)
        bo_type = bo.get("breakout_type", "NONE")
        fg_score = int(fg.get("score", 50))

        whale_label = {
            "ACCUMULATION": "ACCUM",
            "DISTRIBUTION": "DIST",
            "NEUTRAL": "NEUTRAL",
        }.get(whale_flow, "NEUTRAL")

        summary = (
            f"FG={fg_class}({fg_mod:+d}) | "
            f"Flow={whale_label}({whale_mod:+d}) | "
            f"RS={float(rs_val):+.1f}%({rs_mod:+d}) | "
            f"BO={bo_type}({bo_mod:+d}) | "
            f"Total={total:+d}"
        )

        logger.debug(f"[INTEL] {symbol}: {summary}")

        return {
            "symbol": symbol,
            "fear_greed_score": fg_score,
            "fear_greed_class": fg_class,
            "whale_flow": whale_flow,
            "relative_strength": float(rs_val),
            "breakout_type": bo_type,
            "total_modifier": int(total),
            "summary": summary,
        }
