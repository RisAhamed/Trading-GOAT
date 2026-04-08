# core/market_intelligence.py
"""
Market Intelligence module — fetches real-time external signals:
  1. Fear & Greed Index (alternative.me — free, no key)
  2. Large-trade buy/sell pressure (Kraken public REST — free, no key)
  3. Relative Strength vs BTC (bar data — zero API cost)
  4. Breakout Detection (bar data — zero API cost)

Converts each into score modifiers for the SymbolScanner.
"""

import logging
import time
from typing import Optional

import pandas as pd
import requests

from .config_loader import ConfigLoader

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Kraken symbol map  (symbol base → Kraken pair string)
# ---------------------------------------------------------------------------
KRAKEN_SYMBOL_MAP: dict[str, str] = {
    "BTC":  "XBTUSD",
    "ETH":  "ETHUSD",
    "SOL":  "SOLUSD",
    "AVAX": "AVAXUSD",
    "DOGE": "XDGUSD",
    "LTC":  "LTCUSD",
    "LINK": "LINKUSD",
    "MATIC":"MATICUSD",
    "XRP":  "XRPUSD",
    "ADA":  "ADAUSD",
    "DOT":  "DOTUSD",
}

_NEUTRAL_WHALE: dict = {
    "large_buys": 0,
    "large_sells": 0,
    "net_flow": "NEUTRAL",
    "score_modifier": 0,
}
_NEUTRAL_FG: dict = {
    "score": 50,
    "classification": "Neutral",
    "trade_bias": "NEUTRAL",
    "score_modifier": 0,
}


class MarketIntelligence:
    """
    Aggregates four intelligence signals and exposes a single
    get_combined_intelligence() call for the SymbolScanner.

    All methods are safe — they return neutral defaults on any error
    and NEVER raise, so paper-trading mode is never interrupted.
    """

    def __init__(self, config: ConfigLoader) -> None:
        self.config = config

        self._fg_cache: dict = {}
        self._last_fg_fetch: float = 0
        self._fg_interval: int = 900            # refresh F&G every 15 min

        self._whale_cache: dict[str, dict] = {}  # keyed by symbol
        self._whale_time: dict[str, float] = {}
        self._whale_interval: int = 300          # refresh per-symbol every 5 min

    # ------------------------------------------------------------------ #
    #  METHOD 1 — Fear & Greed Index                                       #
    # ------------------------------------------------------------------ #
    def get_fear_greed_score(self, symbol: str = "BTC") -> dict:
        """
        Fetch the Crypto Fear & Greed Index from alternative.me.
        Completely free — no API key required.

        Returns a dict with keys: score, classification, trade_bias,
        score_modifier.
        """
        # Serve from cache if fresh
        if self._fg_cache and (time.time() - self._last_fg_fetch) < self._fg_interval:
            return self._fg_cache

        try:
            resp = requests.get(
                "https://api.alternative.me/fng/?limit=1",
                timeout=8,
            )
            resp.raise_for_status()
            data = resp.json()["data"][0]
            score = int(data["value"])
            classification = data["value_classification"]

            if score <= 25:
                trade_bias = "BUY_BIAS"
                modifier = +10
            elif score <= 45:
                trade_bias = "BUY_BIAS"
                modifier = +5
            elif score <= 60:
                trade_bias = "NEUTRAL"
                modifier = 0
            elif score <= 75:
                trade_bias = "SELL_BIAS"
                modifier = -5
            else:
                trade_bias = "SELL_BIAS"
                modifier = -15

            result = {
                "score": score,
                "classification": classification,
                "trade_bias": trade_bias,
                "score_modifier": modifier,
            }
            self._fg_cache = result
            self._last_fg_fetch = time.time()
            logger.debug(
                f"[FG] Fear/Greed={score} ({classification}) → modifier={modifier:+d}"
            )
            return result

        except Exception as exc:
            logger.warning(f"[FG] Failed to fetch Fear/Greed index: {exc}")
            return dict(_NEUTRAL_FG)

    # ------------------------------------------------------------------ #
    #  METHOD 2 — Large-trade buy/sell pressure via Kraken public REST     #
    # ------------------------------------------------------------------ #
    def get_whale_signal(self, symbol: str) -> dict:
        """
        Detect smart-money accumulation / distribution using Kraken's
        public trade endpoint — no API key required, 1 req/s rate limit.

        Kraken recent-trades endpoint:
            GET https://api.kraken.com/0/public/Trades?pair=XBTUSD&since=<unix_ns>
        Each trade: [price, volume, time, side, orderType, misc, tradeId]
            side: "b" = buy, "s" = sell

        A large "buy" (aggressive market buy, taker) = accumulation.
        A large "sell" (aggressive market sell, taker) = distribution.
        """
        # Serve from per-symbol cache if fresh
        now = time.time()
        if (
            symbol in self._whale_cache
            and (now - self._whale_time.get(symbol, 0)) < self._whale_interval
        ):
            return self._whale_cache[symbol]

        neutral = dict(_NEUTRAL_WHALE)
        neutral["symbol"] = symbol

        try:
            base_coin = symbol.split("/")[0].upper()
            kraken_pair = KRAKEN_SYMBOL_MAP.get(base_coin)
            if not kraken_pair:
                logger.debug(f"[WHALE] No Kraken pair mapping for {base_coin} — skipping")
                return neutral

            # Get config values safely
            mi_cfg = getattr(self.config, "market_intelligence", None)
            min_usd = float(getattr(mi_cfg, "min_whale_usd", 500_000))
            whale_enabled = getattr(mi_cfg, "whale_tracking_enabled", True)

            if not whale_enabled:
                return neutral

            # Kraken: 'since' is a Unix nanosecond timestamp
            since_ns = int((now - 3600) * 1e9)   # last 60 minutes
            resp = requests.get(
                "https://api.kraken.com/0/public/Trades",
                params={"pair": kraken_pair, "since": since_ns},
                timeout=10,
            )
            resp.raise_for_status()
            body = resp.json()

            if body.get("error"):
                logger.warning(f"[WHALE] Kraken API error for {kraken_pair}: {body['error']}")
                return neutral

            # Kraken returns  { "result": { "<pair>": [[price, vol, time, side, ...]] } }
            result_key = list(body["result"].keys())[0]   # first key is the pair
            trades = body["result"][result_key]

            large_buys = 0
            large_sells = 0

            for trade in trades:
                try:
                    price  = float(trade[0])
                    volume = float(trade[1])
                    side   = trade[3]   # "b" or "s"
                    usd_val = price * volume
                    if usd_val >= min_usd:
                        if side == "b":
                            large_buys += 1
                        else:
                            large_sells += 1
                except (IndexError, ValueError):
                    continue

            # Determine net flow and modifier
            if large_buys >= 3 and large_buys > large_sells * 2:
                net_flow = "ACCUMULATION"
                modifier = +15
            elif large_buys >= 2 and large_buys > large_sells:
                net_flow = "ACCUMULATION"
                modifier = +8
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
                "large_buys": large_buys,
                "large_sells": large_sells,
                "net_flow": net_flow,
                "score_modifier": modifier,
            }
            self._whale_cache[symbol] = result
            self._whale_time[symbol] = now

            logger.debug(
                f"[WHALE] {symbol}: buys={large_buys} sells={large_sells} "
                f"flow={net_flow} modifier={modifier:+d}"
            )
            return result

        except Exception as exc:
            logger.warning(f"[WHALE] Error fetching Kraken trades for {symbol}: {exc}")
            return neutral

    # ------------------------------------------------------------------ #
    #  METHOD 3 — Relative Strength vs BTC (no external API)              #
    # ------------------------------------------------------------------ #
    def calculate_relative_strength(
        self,
        symbol: str,
        symbol_bars: pd.DataFrame,
        btc_bars: pd.DataFrame,
        lookback: int = 10,
    ) -> dict:
        """
        Measure whether this coin is outperforming BTC over the last
        `lookback` bars. Uses existing market data — zero API cost.

        Returns dict with: coin_return_pct, btc_return_pct,
        relative_strength, outperforming, score_modifier.
        """
        neutral = {
            "coin_return_pct": 0.0,
            "btc_return_pct": 0.0,
            "relative_strength": 0.0,
            "outperforming": True,
            "score_modifier": 0,
        }

        try:
            # BTC is its own benchmark — always neutral RS
            if symbol in ("BTC/USD", "BTC/USDT"):
                return neutral

            # Need enough bars
            if symbol_bars is None or len(symbol_bars) < lookback + 1:
                return neutral
            if btc_bars is None or len(btc_bars) < lookback + 1:
                return neutral

            mi_cfg = getattr(self.config, "market_intelligence", None)
            lookback = int(getattr(mi_cfg, "relative_strength_lookback", lookback))

            coin_now  = float(symbol_bars["close"].iloc[-1])
            coin_then = float(symbol_bars["close"].iloc[-lookback])
            btc_now   = float(btc_bars["close"].iloc[-1])
            btc_then  = float(btc_bars["close"].iloc[-lookback])

            if coin_then == 0 or btc_then == 0:
                return neutral

            coin_return = (coin_now - coin_then) / coin_then * 100
            btc_return  = (btc_now  - btc_then)  / btc_then  * 100
            rs = coin_return - btc_return

            if rs > 2.0:
                modifier = +15
            elif rs > 1.0:
                modifier = +10
            elif rs > 0.5:
                modifier = +5
            elif rs < -1.0:
                modifier = -10
            elif rs < -0.5:
                modifier = -5
            else:
                modifier = 0

            result = {
                "coin_return_pct": round(coin_return, 3),
                "btc_return_pct":  round(btc_return,  3),
                "relative_strength": round(rs, 3),
                "outperforming": rs > 0.5,
                "score_modifier": modifier,
            }
            logger.debug(
                f"[RS] {symbol}: coin={coin_return:+.2f}% BTC={btc_return:+.2f}% "
                f"RS={rs:+.2f}% modifier={modifier:+d}"
            )
            return result

        except Exception as exc:
            logger.warning(f"[RS] Error calculating relative strength for {symbol}: {exc}")
            return neutral

    # ------------------------------------------------------------------ #
    #  METHOD 4 — Breakout Detection (no external API)                    #
    # ------------------------------------------------------------------ #
    def get_breakout_signal(
        self,
        bars: pd.DataFrame,
        lookback_bars: int = 10,
    ) -> dict:
        """
        Detect if price has just broken above/below a consolidation range
        with volume confirmation. Uses existing bar data — zero API cost.

        Returns dict with: breakout_type, resistance, support,
        range_pct, vol_ratio, score_modifier.
        """
        neutral = {
            "breakout_type": "NONE",
            "resistance": 0.0,
            "support": 0.0,
            "range_pct": 0.0,
            "vol_ratio": 1.0,
            "score_modifier": 0,
        }

        try:
            if bars is None or len(bars) < lookback_bars + 2:
                return neutral

            mi_cfg = getattr(self.config, "market_intelligence", None)
            lookback_bars = int(
                getattr(mi_cfg, "breakout_lookback_bars", lookback_bars)
            )

            # Exclude current (last) candle from the range definition
            prior = bars.iloc[-lookback_bars:-1]
            resistance = float(prior["high"].max())
            support    = float(prior["low"].min())

            current_close = float(bars["close"].iloc[-1])
            current_vol   = float(bars["volume"].iloc[-1])
            avg_vol       = float(bars["volume"].iloc[-lookback_bars:].mean())

            vol_ratio = (current_vol / avg_vol) if avg_vol > 0 else 1.0
            range_pct = (
                (resistance - support) / support * 100 if support > 0 else 0.0
            )

            breakout_up   = current_close > resistance and vol_ratio >= 1.5
            breakout_down = current_close < support    and vol_ratio >= 1.5

            if breakout_up:
                breakout_type = "BULLISH"
                modifier = +20
            elif breakout_down:
                breakout_type = "BEARISH"
                modifier = -20
            else:
                breakout_type = "NONE"
                modifier = 0

            result = {
                "breakout_type": breakout_type,
                "resistance": round(resistance, 6),
                "support":    round(support, 6),
                "range_pct":  round(range_pct, 3),
                "vol_ratio":  round(vol_ratio, 3),
                "score_modifier": modifier,
            }
            if breakout_type != "NONE":
                logger.debug(
                    f"[BO] Breakout detected: {breakout_type} "
                    f"vol_ratio={vol_ratio:.2f} modifier={modifier:+d}"
                )
            return result

        except Exception as exc:
            logger.warning(f"[BO] Error in breakout detection: {exc}")
            return neutral

    # ------------------------------------------------------------------ #
    #  METHOD 5 — Combined Intelligence                                   #
    # ------------------------------------------------------------------ #
    def get_combined_intelligence(
        self,
        symbol: str,
        bars: pd.DataFrame,
        btc_bars: Optional[pd.DataFrame],
    ) -> dict:
        """
        Call all four methods and aggregate into a single modifier + summary.
        Total modifier is clamped to ±30.

        Returns dict with: symbol, fear_greed_score, fear_greed_class,
        whale_flow, relative_strength, breakout_type, total_modifier,
        summary.
        """
        safe_btc = btc_bars if btc_bars is not None else pd.DataFrame()

        fg    = self.get_fear_greed_score(symbol)
        whale = self.get_whale_signal(symbol)
        rs    = self.calculate_relative_strength(symbol, bars, safe_btc)
        bo    = self.get_breakout_signal(bars)

        fg_mod    = fg["score_modifier"]
        whale_mod = whale["score_modifier"]
        rs_mod    = rs["score_modifier"]
        bo_mod    = bo["score_modifier"]

        total = fg_mod + whale_mod + rs_mod + bo_mod
        total = max(-30, min(+30, total))

        fg_abbr  = fg["classification"].replace("Extreme ", "Ext.")
        wh_abbr  = whale["net_flow"][:5]          # ACCUM / DISTR / NEUTR
        rs_val   = rs["relative_strength"]
        bo_type  = bo["breakout_type"]

        summary = (
            f"FG={fg_abbr}({fg_mod:+d})"
            f" | Whale={wh_abbr}({whale_mod:+d})"
            f" | RS={rs_val:+.1f}%({rs_mod:+d})"
            f" | BO={bo_type}({bo_mod:+d})"
            f" | Total={total:+d}"
        )

        return {
            "symbol":           symbol,
            "fear_greed_score": fg["score"],
            "fear_greed_class": fg["classification"],
            "whale_flow":       whale["net_flow"],
            "relative_strength": rs["relative_strength"],
            "breakout_type":    bo["breakout_type"],
            "total_modifier":   total,
            "summary":          summary,
        }
