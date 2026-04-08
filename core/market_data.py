# core/market_data.py
"""
Market data fetcher using Alpaca API.
Fetches OHLCV data for both crypto and forex pairs with caching and retry logic.
"""

import logging
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

import pandas as pd
from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest, CryptoLatestQuoteRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from .config_loader import get_config, ConfigLoader


logger = logging.getLogger(__name__)

# Regex pattern for valid crypto/forex symbols (BASE/QUOTE format)
VALID_SYMBOL_PATTERN = re.compile(r"^[A-Z]+/[A-Z]+$")


@dataclass
class QuoteData:
    """Latest quote data for a symbol."""
    symbol: str
    bid_price: float
    ask_price: float
    mid_price: float
    timestamp: datetime


class MarketDataFetcher:
    """
    Fetches market data from Alpaca API.
    Supports both crypto and forex pairs with caching and exponential backoff retry.
    """
    
    # Cache TTL in seconds
    CACHE_TTL_BARS = 60  # Cache bars for 60 seconds
    CACHE_TTL_QUOTES = 10  # Cache quotes for 10 seconds
    
    # Retry settings
    MAX_RETRIES = 3
    INITIAL_BACKOFF = 1.0  # seconds
    
    # Timeframe mapping
    TIMEFRAME_MAP = {
        "1Min": TimeFrame(1, TimeFrameUnit.Minute),
        "5Min": TimeFrame(5, TimeFrameUnit.Minute),
        "10Min": TimeFrame(10, TimeFrameUnit.Minute),
        "15Min": TimeFrame(15, TimeFrameUnit.Minute),
        "30Min": TimeFrame(30, TimeFrameUnit.Minute),
        "1Hour": TimeFrame(1, TimeFrameUnit.Hour),
        "1Day": TimeFrame(1, TimeFrameUnit.Day),
    }
    
    # Forex pairs (these are traded as stocks in Alpaca)
    FOREX_SYMBOLS = {"EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD", "USD/CHF"}
    
    def __init__(self, config: Optional[ConfigLoader] = None) -> None:
        """Initialize the market data fetcher."""
        self.config = config or get_config()
        
        # Initialize Alpaca clients
        # Crypto client doesn't require API keys for market data
        self.crypto_client = CryptoHistoricalDataClient()
        
        # Stock client requires API keys (for forex pairs)
        self.stock_client = StockHistoricalDataClient(
            api_key=self.config.env.alpaca_api_key,
            secret_key=self.config.env.alpaca_api_secret,
        )
        
        # Cache storage: {(symbol, timeframe): (data, timestamp)}
        self._bars_cache: Dict[Tuple[str, str], Tuple[pd.DataFrame, float]] = {}
        self._quotes_cache: Dict[str, Tuple[QuoteData, float]] = {}
        
        logger.info("MarketDataFetcher initialized")
    
    def _is_forex(self, symbol: str) -> bool:
        """Check if a symbol is a forex pair."""
        return symbol in self.FOREX_SYMBOLS
    
    def _is_valid_symbol_format(self, symbol: str) -> bool:
        """Check if symbol matches valid BASE/QUOTE format (e.g., BTC/USD)."""
        if not isinstance(symbol, str):
            return False
        return bool(VALID_SYMBOL_PATTERN.match(symbol.strip()))
    
    def _convert_symbol_for_crypto(self, symbol: str) -> str:
        """Convert symbol format for Alpaca Crypto API (keeps slash)."""
        # Alpaca crypto API requires BTC/USD format (WITH slash)
        return symbol
    
    def _convert_symbol_for_stock(self, symbol: str) -> str:
        """Convert symbol format for Alpaca Stock API (removes slash)."""
        # Alpaca stock API uses EURUSD format (no slash)
        return symbol.replace("/", "")
    
    def _get_timeframe(self, interval: str) -> TimeFrame:
        """Convert string interval to Alpaca TimeFrame."""
        if interval not in self.TIMEFRAME_MAP:
            logger.warning(f"Unknown interval {interval}, defaulting to 5Min")
            return self.TIMEFRAME_MAP["5Min"]
        return self.TIMEFRAME_MAP[interval]
    
    def _retry_with_backoff(self, func, *args, **kwargs) -> Any:
        """Execute a function with exponential backoff retry."""
        last_exception = None
        backoff = self.INITIAL_BACKOFF
        
        for attempt in range(self.MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                logger.warning(f"Attempt {attempt + 1}/{self.MAX_RETRIES} failed: {e}")
                
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(backoff)
                    backoff *= 2  # Exponential backoff
        
        logger.error(f"All {self.MAX_RETRIES} attempts failed: {last_exception}")
        raise last_exception
    
    def _is_cache_valid(self, cache_time: float, ttl: float) -> bool:
        """Check if cached data is still valid."""
        return (time.time() - cache_time) < ttl
    
    def fetch_bars(
        self,
        symbol: str,
        interval: str,
        lookback_bars: int,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV bars for a symbol.
        
        Args:
            symbol: Trading pair (e.g., "BTC/USD", "EUR/USD")
            interval: Bar interval (e.g., "5Min", "10Min")
            lookback_bars: Number of bars to fetch
            
        Returns:
            DataFrame with columns: open, high, low, close, volume, vwap
            Index is datetime
        """
        # Validate symbol format before making any API request
        if not self._is_valid_symbol_format(symbol):
            logger.warning(f"[MARKET_DATA] Rejecting invalid crypto symbol: {symbol}")
            return None
        
        cache_key = (symbol, interval)
        
        # Check cache
        if cache_key in self._bars_cache:
            cached_data, cache_time = self._bars_cache[cache_key]
            if self._is_cache_valid(cache_time, self.CACHE_TTL_BARS):
                logger.debug(f"Using cached bars for {symbol} {interval}")
                return cached_data.tail(lookback_bars).copy()
        
        try:
            timeframe = self._get_timeframe(interval)
            
            # Calculate time range (fetch extra bars for indicator warmup)
            end_time = datetime.now(timezone.utc)
            # Estimate start time based on interval
            interval_minutes = self._interval_to_minutes(interval)
            start_time = end_time - timedelta(minutes=interval_minutes * (lookback_bars + 50))
            
            if self._is_forex(symbol):
                alpaca_symbol = self._convert_symbol_for_stock(symbol)
                df = self._fetch_stock_bars(alpaca_symbol, timeframe, start_time, end_time)
            else:
                alpaca_symbol = self._convert_symbol_for_crypto(symbol)
                df = self._fetch_crypto_bars(alpaca_symbol, timeframe, start_time, end_time)
            
            if df is not None and not df.empty:
                # Cache the data
                self._bars_cache[cache_key] = (df, time.time())
                logger.debug(f"Fetched {len(df)} bars for {symbol} {interval}")
                return df.tail(lookback_bars).copy()
            else:
                logger.warning(f"No data returned for {symbol} {interval}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching bars for {symbol}: {e}")
            return None
    
    def _interval_to_minutes(self, interval: str) -> int:
        """Convert interval string to minutes."""
        mapping = {
            "1Min": 1,
            "5Min": 5,
            "10Min": 10,
            "15Min": 15,
            "30Min": 30,
            "1Hour": 60,
            "1Day": 1440,
        }
        return mapping.get(interval, 5)
    
    def _fetch_crypto_bars(
        self,
        symbol: str,
        timeframe: TimeFrame,
        start: datetime,
        end: datetime,
    ) -> Optional[pd.DataFrame]:
        """Fetch crypto bars from Alpaca."""
        def fetch():
            request = CryptoBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=timeframe,
                start=start,
                end=end,
            )
            bars = self.crypto_client.get_crypto_bars(request)
            return bars
        
        bars = self._retry_with_backoff(fetch)
        
        if bars and symbol in bars.data:
            df = bars.df
            if isinstance(df.index, pd.MultiIndex):
                df = df.xs(symbol, level='symbol')
            
            # Standardize column names
            df = df.rename(columns={
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume',
                'vwap': 'vwap',
                'trade_count': 'trade_count',
            })
            
            # Ensure we have required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in df.columns:
                    df[col] = 0.0
            
            return df
        
        return None
    
    def _fetch_stock_bars(
        self,
        symbol: str,
        timeframe: TimeFrame,
        start: datetime,
        end: datetime,
    ) -> Optional[pd.DataFrame]:
        """Fetch stock/forex bars from Alpaca."""
        def fetch():
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=timeframe,
                start=start,
                end=end,
            )
            bars = self.stock_client.get_stock_bars(request)
            return bars
        
        bars = self._retry_with_backoff(fetch)
        
        if bars and symbol in bars.data:
            df = bars.df
            if isinstance(df.index, pd.MultiIndex):
                df = df.xs(symbol, level='symbol')
            
            # Standardize column names
            df = df.rename(columns={
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume',
                'vwap': 'vwap',
                'trade_count': 'trade_count',
            })
            
            # Ensure we have required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in df.columns:
                    df[col] = 0.0
            
            return df
        
        return None
    
    def fetch_latest_quote(self, symbol: str) -> Optional[QuoteData]:
        """
        Fetch the latest quote for a symbol.
        
        Args:
            symbol: Trading pair (e.g., "BTC/USD")
            
        Returns:
            QuoteData with bid, ask, and mid prices
        """
        # Validate symbol format before making any API request
        if not self._is_valid_symbol_format(symbol):
            logger.warning(f"[MARKET_DATA] Rejecting invalid symbol for quote: {symbol}")
            return None
        
        # Check cache
        if symbol in self._quotes_cache:
            cached_quote, cache_time = self._quotes_cache[symbol]
            if self._is_cache_valid(cache_time, self.CACHE_TTL_QUOTES):
                logger.debug(f"Using cached quote for {symbol}")
                return cached_quote
        
        try:
            if self._is_forex(symbol):
                alpaca_symbol = self._convert_symbol_for_stock(symbol)
                quote = self._fetch_stock_quote(alpaca_symbol)
            else:
                alpaca_symbol = self._convert_symbol_for_crypto(symbol)
                quote = self._fetch_crypto_quote(alpaca_symbol)
            
            if quote:
                quote.symbol = symbol  # Use original symbol format
                self._quotes_cache[symbol] = (quote, time.time())
                logger.debug(f"Fetched quote for {symbol}: {quote.mid_price}")
                return quote
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching quote for {symbol}: {e}")
            return None
    
    def _fetch_crypto_quote(self, symbol: str) -> Optional[QuoteData]:
        """Fetch crypto quote from Alpaca."""
        def fetch():
            request = CryptoLatestQuoteRequest(symbol_or_symbols=symbol)
            quotes = self.crypto_client.get_crypto_latest_quote(request)
            return quotes
        
        quotes = self._retry_with_backoff(fetch)
        
        if quotes and symbol in quotes:
            q = quotes[symbol]
            bid = float(q.bid_price) if q.bid_price else 0.0
            ask = float(q.ask_price) if q.ask_price else 0.0
            mid = (bid + ask) / 2 if bid and ask else bid or ask
            
            return QuoteData(
                symbol=symbol,
                bid_price=bid,
                ask_price=ask,
                mid_price=mid,
                timestamp=q.timestamp if hasattr(q, 'timestamp') else datetime.now(timezone.utc),
            )
        
        return None
    
    def _fetch_stock_quote(self, symbol: str) -> Optional[QuoteData]:
        """Fetch stock/forex quote from Alpaca."""
        def fetch():
            request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quotes = self.stock_client.get_stock_latest_quote(request)
            return quotes
        
        quotes = self._retry_with_backoff(fetch)
        
        if quotes and symbol in quotes:
            q = quotes[symbol]
            bid = float(q.bid_price) if q.bid_price else 0.0
            ask = float(q.ask_price) if q.ask_price else 0.0
            mid = (bid + ask) / 2 if bid and ask else bid or ask
            
            return QuoteData(
                symbol=symbol,
                bid_price=bid,
                ask_price=ask,
                mid_price=mid,
                timestamp=q.timestamp if hasattr(q, 'timestamp') else datetime.now(timezone.utc),
            )
        
        return None
    
    def fetch_all_bars(
        self,
        symbols: List[str],
        interval: str,
        lookback_bars: int,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch bars for multiple symbols.
        
        Args:
            symbols: List of trading pairs
            interval: Bar interval
            lookback_bars: Number of bars to fetch
            
        Returns:
            Dictionary mapping symbol to DataFrame
        """
        results = {}
        
        for symbol in symbols:
            df = self.fetch_bars(symbol, interval, lookback_bars)
            if df is not None:
                results[symbol] = df
            else:
                logger.warning(f"Failed to fetch bars for {symbol}")
        
        return results
    
    def fetch_all_quotes(self, symbols: List[str]) -> Dict[str, QuoteData]:
        """
        Fetch latest quotes for multiple symbols.
        
        Args:
            symbols: List of trading pairs
            
        Returns:
            Dictionary mapping symbol to QuoteData
        """
        results = {}
        
        for symbol in symbols:
            quote = self.fetch_latest_quote(symbol)
            if quote:
                results[symbol] = quote
            else:
                logger.warning(f"Failed to fetch quote for {symbol}")
        
        return results
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get the current mid price for a symbol.
        
        Args:
            symbol: Trading pair
            
        Returns:
            Current mid price or None
        """
        quote = self.fetch_latest_quote(symbol)
        return quote.mid_price if quote else None
    
    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._bars_cache.clear()
        self._quotes_cache.clear()
        logger.info("Market data cache cleared")
