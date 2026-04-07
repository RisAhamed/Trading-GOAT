# core/indicators.py
"""
Technical indicator calculator using 'ta' library (Technical Analysis Library).
Calculates RSI, MACD, EMA, Bollinger Bands, ATR, and more.

Uses 'ta' library instead of 'pandas-ta' for Python 3.14+ compatibility.
The 'ta' library is pure Python with no numba dependency.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator, MACD, EMAIndicator, SMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange

from .config_loader import get_config, ConfigLoader


logger = logging.getLogger(__name__)


@dataclass
class IndicatorValues:
    """Container for all indicator values for a single timeframe."""
    # Price data
    current_price: float = 0.0
    prev_close: float = 0.0
    
    # RSI
    rsi: float = 50.0
    rsi_trend: str = "NEUTRAL"  # OVERSOLD, OVERBOUGHT, NEUTRAL
    
    # MACD
    macd_line: float = 0.0
    macd_signal: float = 0.0
    macd_histogram: float = 0.0
    macd_trend: str = "NEUTRAL"  # BULLISH, BEARISH, NEUTRAL
    macd_histogram_rising: bool = False
    
    # EMA
    ema_short: float = 0.0
    ema_long: float = 0.0
    ema_trend: str = "NEUTRAL"  # BULLISH, BEARISH, NEUTRAL
    ema_crossover: str = "NONE"  # GOLDEN_CROSS, DEATH_CROSS, NONE
    price_vs_ema_short: str = "NEUTRAL"  # ABOVE, BELOW, NEUTRAL
    price_vs_ema_long: str = "NEUTRAL"  # ABOVE, BELOW, NEUTRAL
    
    # Bollinger Bands
    bb_upper: float = 0.0
    bb_middle: float = 0.0
    bb_lower: float = 0.0
    bb_percent: float = 0.5  # 0-1, where 0 is at lower band, 1 is at upper band
    bb_trend: str = "NEUTRAL"  # OVERBOUGHT, OVERSOLD, NEUTRAL
    
    # ATR (for stop loss sizing)
    atr: float = 0.0
    atr_percent: float = 0.0  # ATR as percentage of price
    
    # Volume
    volume: float = 0.0
    volume_sma: float = 0.0
    volume_ratio: float = 1.0  # current volume / avg volume
    volume_increasing: bool = False  # True if last candle volume > prior candle volume
    
    # Trend strength
    adx: float = 0.0  # 14-period ADX
    
    # Price action summary
    high_5bars: float = 0.0
    low_5bars: float = 0.0
    price_range_5bars: float = 0.0
    
    # Overall assessment
    overall_trend: str = "SIDEWAYS"  # BULLISH, BEARISH, SIDEWAYS
    trend_strength: str = "WEAK"  # STRONG, MODERATE, WEAK
    
    # Data quality
    has_data: bool = True
    data_points: int = 0


@dataclass
class SymbolIndicators:
    """Container for indicators on both timeframes for a symbol."""
    symbol: str
    trend_tf: IndicatorValues = field(default_factory=IndicatorValues)
    entry_tf: IndicatorValues = field(default_factory=IndicatorValues)
    current_price: float = 0.0
    timestamp: str = ""


class IndicatorCalculator:
    """
    Calculates technical indicators using the 'ta' library.
    Supports dual-timeframe analysis with trend and entry timeframes.
    
    Uses 'ta' library for Python 3.14+ compatibility (no numba dependency).
    """
    
    def __init__(self, config: Optional[ConfigLoader] = None) -> None:
        """Initialize the indicator calculator."""
        self.config = config or get_config()
        
        # Get indicator settings from config
        self.rsi_period = self.config.indicators.rsi_period
        self.rsi_oversold = self.config.indicators.rsi_oversold
        self.rsi_overbought = self.config.indicators.rsi_overbought
        
        self.macd_fast = self.config.indicators.macd_fast
        self.macd_slow = self.config.indicators.macd_slow
        self.macd_signal = self.config.indicators.macd_signal
        
        self.ema_short = self.config.indicators.ema_short
        self.ema_long = self.config.indicators.ema_long
        
        self.bb_period = self.config.indicators.bb_period
        self.bb_std = self.config.indicators.bb_std_dev
        
        self.atr_period = self.config.indicators.atr_period
        
        logger.info("IndicatorCalculator initialized with 'ta' library")
    
    def calculate(self, df: pd.DataFrame) -> IndicatorValues:
        """
        Calculate all indicators for a DataFrame of OHLCV data.
        
        Args:
            df: DataFrame with columns: open, high, low, close, volume
                Index should be datetime
                
        Returns:
            IndicatorValues with all calculated indicators
        """
        result = IndicatorValues()
        
        if df is None or df.empty:
            result.has_data = False
            logger.warning("Empty DataFrame provided to indicator calculator")
            return result
        
        try:
            result.data_points = len(df)
            
            # Ensure we have numeric data
            df = df.copy()
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Current price
            result.current_price = float(df['close'].iloc[-1])
            result.prev_close = float(df['close'].iloc[-2]) if len(df) > 1 else result.current_price
            
            # Calculate RSI
            self._calculate_rsi(df, result)
            
            # Calculate MACD
            self._calculate_macd(df, result)
            
            # Calculate EMA
            self._calculate_ema(df, result)
            
            # Calculate Bollinger Bands
            self._calculate_bollinger(df, result)
            
            # Calculate ATR
            self._calculate_atr(df, result)

            # Calculate ADX
            self._calculate_adx(df, result)
            
            # Calculate volume metrics
            self._calculate_volume(df, result)
            
            # Calculate price action summary
            self._calculate_price_action(df, result)
            
            # Determine overall trend
            self._determine_overall_trend(result)
            
            result.has_data = True
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            result.has_data = False
        
        return result
    
    def _calculate_rsi(self, df: pd.DataFrame, result: IndicatorValues) -> None:
        """Calculate RSI indicator using ta library."""
        try:
            rsi_indicator = RSIIndicator(
                close=df['close'],
                window=self.rsi_period,
                fillna=False
            )
            rsi = rsi_indicator.rsi()
            
            if rsi is not None and not rsi.empty:
                rsi_value = rsi.iloc[-1]
                if not pd.isna(rsi_value):
                    result.rsi = float(rsi_value)
                    
                    if result.rsi <= self.rsi_oversold:
                        result.rsi_trend = "OVERSOLD"
                    elif result.rsi >= self.rsi_overbought:
                        result.rsi_trend = "OVERBOUGHT"
                    else:
                        result.rsi_trend = "NEUTRAL"
        except Exception as e:
            logger.debug(f"RSI calculation error: {e}")
    
    def _calculate_macd(self, df: pd.DataFrame, result: IndicatorValues) -> None:
        """Calculate MACD indicator using ta library."""
        try:
            macd = MACD(
                close=df['close'],
                window_fast=self.macd_fast,
                window_slow=self.macd_slow,
                window_sign=self.macd_signal,
                fillna=False
            )
            
            # MACD line
            macd_line = macd.macd()
            if macd_line is not None and not macd_line.empty:
                val = macd_line.iloc[-1]
                if not pd.isna(val):
                    result.macd_line = float(val)
            
            # Signal line
            macd_signal = macd.macd_signal()
            if macd_signal is not None and not macd_signal.empty:
                val = macd_signal.iloc[-1]
                if not pd.isna(val):
                    result.macd_signal = float(val)
            
            # Histogram
            macd_hist = macd.macd_diff()
            if macd_hist is not None and not macd_hist.empty:
                val = macd_hist.iloc[-1]
                if not pd.isna(val):
                    result.macd_histogram = float(val)
                
                # Check if histogram is rising
                if len(macd_hist) >= 2:
                    prev_hist = macd_hist.iloc[-2]
                    if not pd.isna(prev_hist):
                        result.macd_histogram_rising = result.macd_histogram > float(prev_hist)
            
            # Determine MACD trend
            if result.macd_line > result.macd_signal and result.macd_histogram > 0:
                result.macd_trend = "BULLISH"
            elif result.macd_line < result.macd_signal and result.macd_histogram < 0:
                result.macd_trend = "BEARISH"
            else:
                result.macd_trend = "NEUTRAL"
                
        except Exception as e:
            logger.debug(f"MACD calculation error: {e}")
    
    def _calculate_ema(self, df: pd.DataFrame, result: IndicatorValues) -> None:
        """Calculate EMA indicators and crossovers using ta library."""
        try:
            ema_short_ind = EMAIndicator(
                close=df['close'],
                window=self.ema_short,
                fillna=False
            )
            ema_long_ind = EMAIndicator(
                close=df['close'],
                window=self.ema_long,
                fillna=False
            )
            
            ema_short = ema_short_ind.ema_indicator()
            ema_long = ema_long_ind.ema_indicator()
            
            if ema_short is not None and not ema_short.empty:
                val = ema_short.iloc[-1]
                if not pd.isna(val):
                    result.ema_short = float(val)
            
            if ema_long is not None and not ema_long.empty:
                val = ema_long.iloc[-1]
                if not pd.isna(val):
                    result.ema_long = float(val)
            
            # EMA trend
            if result.ema_short > result.ema_long:
                result.ema_trend = "BULLISH"
            elif result.ema_short < result.ema_long:
                result.ema_trend = "BEARISH"
            else:
                result.ema_trend = "NEUTRAL"
            
            # Check for crossover
            if ema_short is not None and ema_long is not None and len(ema_short) >= 2 and len(ema_long) >= 2:
                prev_short = float(ema_short.iloc[-2]) if not pd.isna(ema_short.iloc[-2]) else result.ema_short
                prev_long = float(ema_long.iloc[-2]) if not pd.isna(ema_long.iloc[-2]) else result.ema_long
                curr_short = result.ema_short
                curr_long = result.ema_long
                
                if prev_short <= prev_long and curr_short > curr_long:
                    result.ema_crossover = "GOLDEN_CROSS"
                elif prev_short >= prev_long and curr_short < curr_long:
                    result.ema_crossover = "DEATH_CROSS"
                else:
                    result.ema_crossover = "NONE"
            
            # Price vs EMA
            if result.current_price > result.ema_short * 1.001:  # 0.1% buffer
                result.price_vs_ema_short = "ABOVE"
            elif result.current_price < result.ema_short * 0.999:
                result.price_vs_ema_short = "BELOW"
            else:
                result.price_vs_ema_short = "NEUTRAL"
            
            if result.current_price > result.ema_long * 1.001:
                result.price_vs_ema_long = "ABOVE"
            elif result.current_price < result.ema_long * 0.999:
                result.price_vs_ema_long = "BELOW"
            else:
                result.price_vs_ema_long = "NEUTRAL"
                
        except Exception as e:
            logger.debug(f"EMA calculation error: {e}")
    
    def _calculate_bollinger(self, df: pd.DataFrame, result: IndicatorValues) -> None:
        """Calculate Bollinger Bands using ta library."""
        try:
            bb = BollingerBands(
                close=df['close'],
                window=self.bb_period,
                window_dev=int(self.bb_std),  # ta uses integer std dev multiplier
                fillna=False
            )
            
            bb_upper = bb.bollinger_hband()
            bb_middle = bb.bollinger_mavg()
            bb_lower = bb.bollinger_lband()
            
            if bb_upper is not None and not bb_upper.empty:
                val = bb_upper.iloc[-1]
                if not pd.isna(val):
                    result.bb_upper = float(val)
            
            if bb_middle is not None and not bb_middle.empty:
                val = bb_middle.iloc[-1]
                if not pd.isna(val):
                    result.bb_middle = float(val)
            
            if bb_lower is not None and not bb_lower.empty:
                val = bb_lower.iloc[-1]
                if not pd.isna(val):
                    result.bb_lower = float(val)
            
            # Calculate %B (percent bandwidth)
            if result.bb_upper > result.bb_lower:
                result.bb_percent = (result.current_price - result.bb_lower) / (result.bb_upper - result.bb_lower)
                result.bb_percent = max(0, min(1, result.bb_percent))
            
            # BB trend
            if result.bb_percent >= 0.8:
                result.bb_trend = "OVERBOUGHT"
            elif result.bb_percent <= 0.2:
                result.bb_trend = "OVERSOLD"
            else:
                result.bb_trend = "NEUTRAL"
                
        except Exception as e:
            logger.debug(f"Bollinger Bands calculation error: {e}")
    
    def _calculate_atr(self, df: pd.DataFrame, result: IndicatorValues) -> None:
        """Calculate ATR for dynamic stop loss sizing using ta library."""
        try:
            atr = AverageTrueRange(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=self.atr_period,
                fillna=False
            )
            
            atr_values = atr.average_true_range()
            
            if atr_values is not None and not atr_values.empty:
                val = atr_values.iloc[-1]
                if not pd.isna(val):
                    result.atr = float(val)
                    
                    # ATR as percentage of price
                    if result.current_price > 0:
                        result.atr_percent = (result.atr / result.current_price) * 100
                        
        except Exception as e:
            logger.debug(f"ATR calculation error: {e}")
    
    def _calculate_adx(self, df: pd.DataFrame, result: IndicatorValues) -> None:
        """Calculate ADX indicator using ta library."""
        try:
            adx_indicator = ADXIndicator(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=14,
                fillna=False,
            )
            adx_series = adx_indicator.adx()
            if adx_series is not None and not adx_series.empty:
                adx_value = adx_series.iloc[-1]
                if not pd.isna(adx_value):
                    result.adx = float(adx_value)
        except Exception as e:
            logger.debug(f"ADX calculation error: {e}")
    
    def _calculate_volume(self, df: pd.DataFrame, result: IndicatorValues) -> None:
        """Calculate volume metrics."""
        try:
            if 'volume' not in df.columns:
                return
            
            volume = df['volume']
            result.volume = float(volume.iloc[-1]) if not pd.isna(volume.iloc[-1]) else 0.0
            
            # Calculate volume SMA using ta library
            vol_sma_ind = SMAIndicator(
                close=volume,
                window=20,
                fillna=False
            )
            vol_sma = vol_sma_ind.sma_indicator()
            
            if vol_sma is not None and not vol_sma.empty:
                val = vol_sma.iloc[-1]
                if not pd.isna(val) and val > 0:
                    result.volume_sma = float(val)
                    result.volume_ratio = result.volume / result.volume_sma

            if len(volume) >= 2:
                last_volume = volume.iloc[-1]
                prev_volume = volume.iloc[-2]
                if not pd.isna(last_volume) and not pd.isna(prev_volume):
                    result.volume_increasing = bool(last_volume > prev_volume)
                     
        except Exception as e:
            logger.debug(f"Volume calculation error: {e}")
    
    def _calculate_price_action(self, df: pd.DataFrame, result: IndicatorValues) -> None:
        """Calculate price action summary for last 5 bars."""
        try:
            last_5 = df.tail(5)
            
            if len(last_5) >= 5:
                result.high_5bars = float(last_5['high'].max())
                result.low_5bars = float(last_5['low'].min())
                result.price_range_5bars = result.high_5bars - result.low_5bars
                
        except Exception as e:
            logger.debug(f"Price action calculation error: {e}")
    
    def _determine_overall_trend(self, result: IndicatorValues) -> None:
        """Determine the overall trend based on all indicators."""
        bullish_signals = 0
        bearish_signals = 0
        total_signals = 0
        
        # RSI signal
        if result.rsi_trend == "OVERSOLD":
            bullish_signals += 1
        elif result.rsi_trend == "OVERBOUGHT":
            bearish_signals += 1
        total_signals += 1
        
        # MACD signal
        if result.macd_trend == "BULLISH":
            bullish_signals += 1
        elif result.macd_trend == "BEARISH":
            bearish_signals += 1
        total_signals += 1
        
        # MACD histogram rising adds weight
        if result.macd_histogram_rising and result.macd_histogram > 0:
            bullish_signals += 0.5
        elif not result.macd_histogram_rising and result.macd_histogram < 0:
            bearish_signals += 0.5
        
        # EMA signal
        if result.ema_trend == "BULLISH":
            bullish_signals += 1
        elif result.ema_trend == "BEARISH":
            bearish_signals += 1
        total_signals += 1
        
        # EMA crossover adds weight
        if result.ema_crossover == "GOLDEN_CROSS":
            bullish_signals += 1
        elif result.ema_crossover == "DEATH_CROSS":
            bearish_signals += 1
        
        # Price vs EMA
        if result.price_vs_ema_short == "ABOVE" and result.price_vs_ema_long == "ABOVE":
            bullish_signals += 0.5
        elif result.price_vs_ema_short == "BELOW" and result.price_vs_ema_long == "BELOW":
            bearish_signals += 0.5
        
        # Bollinger Band signal
        if result.bb_trend == "OVERSOLD":
            bullish_signals += 0.5
        elif result.bb_trend == "OVERBOUGHT":
            bearish_signals += 0.5
        
        # Determine overall trend
        if bullish_signals > bearish_signals + 1:
            result.overall_trend = "BULLISH"
        elif bearish_signals > bullish_signals + 1:
            result.overall_trend = "BEARISH"
        else:
            result.overall_trend = "SIDEWAYS"
        
        # Determine trend strength
        signal_diff = abs(bullish_signals - bearish_signals)
        if signal_diff >= 3:
            result.trend_strength = "STRONG"
        elif signal_diff >= 1.5:
            result.trend_strength = "MODERATE"
        else:
            result.trend_strength = "WEAK"
    
    def calculate_for_symbol(
        self,
        symbol: str,
        trend_df: Optional[pd.DataFrame],
        entry_df: Optional[pd.DataFrame],
        current_price: Optional[float] = None,
    ) -> SymbolIndicators:
        """
        Calculate indicators for both timeframes for a symbol.
        
        Args:
            symbol: Trading pair symbol
            trend_df: DataFrame for trend timeframe (10-min)
            entry_df: DataFrame for entry timeframe (5-min)
            current_price: Optional current price override
            
        Returns:
            SymbolIndicators with data for both timeframes
        """
        result = SymbolIndicators(symbol=symbol)
        
        # Calculate trend timeframe indicators
        if trend_df is not None and not trend_df.empty:
            result.trend_tf = self.calculate(trend_df)
        else:
            result.trend_tf = IndicatorValues(has_data=False)
        
        # Calculate entry timeframe indicators
        if entry_df is not None and not entry_df.empty:
            result.entry_tf = self.calculate(entry_df)
        else:
            result.entry_tf = IndicatorValues(has_data=False)
        
        # Set current price
        if current_price is not None:
            result.current_price = current_price
        elif result.entry_tf.has_data:
            result.current_price = result.entry_tf.current_price
        elif result.trend_tf.has_data:
            result.current_price = result.trend_tf.current_price
        
        # Set timestamp
        result.timestamp = datetime.now().isoformat()
        
        return result
    
    def calculate_batch(
        self,
        data: Dict[str, Dict[str, pd.DataFrame]],
        current_prices: Optional[Dict[str, float]] = None,
    ) -> Dict[str, SymbolIndicators]:
        """
        Calculate indicators for multiple symbols.
        
        Args:
            data: Dict of {symbol: {'trend': DataFrame, 'entry': DataFrame}}
            current_prices: Optional dict of {symbol: price}
            
        Returns:
            Dict of {symbol: SymbolIndicators}
        """
        results = {}
        
        for symbol, dfs in data.items():
            trend_df = dfs.get('trend')
            entry_df = dfs.get('entry')
            price = current_prices.get(symbol) if current_prices else None
            
            results[symbol] = self.calculate_for_symbol(
                symbol=symbol,
                trend_df=trend_df,
                entry_df=entry_df,
                current_price=price,
            )
        
        return results
