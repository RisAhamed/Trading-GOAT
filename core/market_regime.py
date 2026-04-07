"""
Market regime detector for bullish/bearish/sideways classification.
"""

import logging
import time
from typing import Optional, TypedDict

import pandas as pd

from .config_loader import ConfigLoader, get_config
from .market_data import MarketDataFetcher


logger = logging.getLogger(__name__)
EPSILON = 1e-9


class RegimeSummary(TypedDict):
    """Typed payload for market regime summary."""
    regime: str
    ema20: float
    ema50: float
    adx: float
    rsi: float
    safe_to_buy: bool


class MarketRegimeDetector:
    """
    Detects market regime using 1H EMA and ADX filters.

    Regimes:
    - BULLISH: EMA fast > EMA slow and ADX > threshold
    - BEARISH: EMA fast < EMA slow and ADX > threshold
    - SIDEWAYS: ADX <= threshold
    """

    def __init__(
        self,
        market_data: MarketDataFetcher,
        config: Optional[ConfigLoader] = None,
    ) -> None:
        """Initialize detector with market data and config dependencies."""
        self.market_data = market_data
        self.config = config or get_config()

        regime_cfg = getattr(self.config, "regime_detection", None)
        self.primary_symbol = getattr(regime_cfg, "primary_symbol", "BTC/USD")
        self.timeframe = getattr(regime_cfg, "timeframe", "1Hour")
        self.lookback_bars = int(getattr(regime_cfg, "lookback_bars", 100))
        self.ema_fast = int(getattr(regime_cfg, "ema_fast", 20))
        self.ema_slow = int(getattr(regime_cfg, "ema_slow", 50))
        self.adx_period = int(getattr(regime_cfg, "adx_period", 14))
        self.adx_threshold = float(getattr(regime_cfg, "adx_trending_threshold", 20.0))
        self.cache_seconds = int(getattr(regime_cfg, "cache_seconds", 300))

        self._cached_at: float = 0.0
        self._cached_summary: RegimeSummary = {
            "regime": "SIDEWAYS",
            "ema20": 0.0,
            "ema50": 0.0,
            "adx": 0.0,
            "rsi": 50.0,
            "safe_to_buy": False,
        }

    def _calculate_rsi(self, close: pd.Series, period: int = 14) -> float:
        """Calculate RSI value from close prices using pandas."""
        delta = close.diff()
        gains = delta.clip(lower=0)
        losses = -delta.clip(upper=0)

        avg_gain = gains.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
        avg_loss = losses.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
        rs = avg_gain / avg_loss.clip(lower=EPSILON)
        rsi = 100 - (100 / (1 + rs))

        if rsi.empty or pd.isna(rsi.iloc[-1]):
            return 50.0
        return float(rsi.iloc[-1])

    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ADX manually from high/low/close using pandas."""
        high = pd.to_numeric(df["high"], errors="coerce")
        low = pd.to_numeric(df["low"], errors="coerce")
        close = pd.to_numeric(df["close"], errors="coerce")

        up_move = high.diff()
        down_move = -low.diff()

        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
        atr_safe = atr.clip(lower=EPSILON)
        plus_di = 100 * (
            plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
            / atr_safe
        )
        minus_di = 100 * (
            minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
            / atr_safe
        )

        di_sum = (plus_di + minus_di).clip(lower=EPSILON)
        dx = ((plus_di - minus_di).abs() / di_sum) * 100
        adx = dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

        if adx.empty or pd.isna(adx.iloc[-1]):
            return 0.0
        return float(adx.iloc[-1])

    def _detect_regime(self) -> RegimeSummary:
        """Fetch bars, compute indicators, and classify current market regime."""
        bars = self.market_data.fetch_bars(
            self.primary_symbol,
            self.timeframe,
            self.lookback_bars,
        )
        if bars is None or bars.empty:
            logger.warning("Regime detection fallback: no market data available")
            return {
                "regime": "SIDEWAYS",
                "ema20": 0.0,
                "ema50": 0.0,
                "adx": 0.0,
                "rsi": 50.0,
                "safe_to_buy": False,
            }

        required_cols = {"high", "low", "close"}
        if not required_cols.issubset(set(bars.columns)):
            logger.warning("Regime detection fallback: missing required OHLC columns")
            return {
                "regime": "SIDEWAYS",
                "ema20": 0.0,
                "ema50": 0.0,
                "adx": 0.0,
                "rsi": 50.0,
                "safe_to_buy": False,
            }

        close = pd.to_numeric(bars["close"], errors="coerce")
        ema_fast_val = float(close.ewm(span=self.ema_fast, adjust=False).mean().iloc[-1])
        ema_slow_val = float(close.ewm(span=self.ema_slow, adjust=False).mean().iloc[-1])
        adx_val = self._calculate_adx(bars, self.adx_period)
        rsi_val = self._calculate_rsi(close, 14)

        if adx_val <= self.adx_threshold:
            regime = "SIDEWAYS"
        elif ema_fast_val > ema_slow_val:
            regime = "BULLISH"
        else:
            regime = "BEARISH"

        summary = {
            "regime": regime,
            "ema20": float(ema_fast_val),
            "ema50": float(ema_slow_val),
            "adx": float(adx_val),
            "rsi": float(rsi_val),
            "safe_to_buy": regime == "BULLISH",
        }

        logger.info(
            "Market regime detected: %s | EMA%d=%.2f EMA%d=%.2f ADX=%.2f RSI=%.2f",
            regime,
            self.ema_fast,
            ema_fast_val,
            self.ema_slow,
            ema_slow_val,
            adx_val,
            rsi_val,
        )
        return summary

    def _get_summary(self) -> RegimeSummary:
        """Return cached regime summary or refresh if cache has expired."""
        if (time.time() - self._cached_at) < self.cache_seconds and self._cached_at > 0:
            return dict(self._cached_summary)

        self._cached_summary = self._detect_regime()
        self._cached_at = time.time()
        return dict(self._cached_summary)

    def get_regime(self) -> str:
        """Get current market regime string."""
        return str(self._get_summary()["regime"])

    def is_safe_to_buy(self) -> bool:
        """Return True only when regime is BULLISH."""
        return bool(self._get_summary()["safe_to_buy"])

    def get_regime_summary(self) -> dict:
        """Get full regime summary payload."""
        summary = self._get_summary()
        return {
            "regime": str(summary["regime"]),
            "ema20": float(summary["ema20"]),
            "ema50": float(summary["ema50"]),
            "adx": float(summary["adx"]),
            "rsi": float(summary["rsi"]),
            "safe_to_buy": bool(summary["safe_to_buy"]),
        }
