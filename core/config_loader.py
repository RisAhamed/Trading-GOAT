# core/config_loader.py
"""
Configuration loader that combines config.yaml and .env settings.
"""

import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

import yaml
from dotenv import load_dotenv


@dataclass
class BotConfig:
    """Bot general settings."""
    name: str = "AI Crypto Trader"
    mode: str = "paper"
    base_currency: str = "USD"
    loop_interval_seconds: int = 30
    log_level: str = "INFO"
    log_file: str = "logs/trading.log"


@dataclass
class AIConfig:
    """AI/LLM settings for Ollama Cloud."""
    provider: str = "ollama_cloud"
    model: str = "minimax-m2.7"
    base_url: str = "https://api.ollama.com/v1"
    temperature: float = 0.3
    max_tokens: int = 1000
    reasoning_prompt_style: str = "chain_of_thought"
    timeout_seconds: int = 120
    fallback_models: List[str] = field(default_factory=lambda: ["glm-4-plus", "gpt-120b", "deepseek-r1", "qwen-max"])


@dataclass
class MarketConfig:
    """Market configuration for crypto and forex."""
    crypto_enabled: bool = True
    crypto_pairs: List[str] = field(default_factory=lambda: ["BTC/USD", "ETH/USD", "SOL/USD", "AVAX/USD"])
    crypto_exchange: str = "alpaca"
    forex_enabled: bool = True
    forex_pairs: List[str] = field(default_factory=lambda: ["EUR/USD", "GBP/USD"])
    forex_exchange: str = "alpaca"
    
    def get_all_pairs(self) -> List[str]:
        """Get all enabled trading pairs."""
        pairs = []
        if self.crypto_enabled:
            pairs.extend(self.crypto_pairs)
        if self.forex_enabled:
            pairs.extend(self.forex_pairs)
        return pairs


@dataclass
class TimeframeConfig:
    """Timeframe settings for dual-timeframe analysis."""
    trend_interval: str = "10Min"
    trend_lookback_bars: int = 50
    entry_interval: str = "5Min"
    entry_lookback_bars: int = 30


@dataclass
class IndicatorConfig:
    """Technical indicator settings."""
    rsi_period: int = 14
    rsi_oversold: int = 30
    rsi_overbought: int = 70
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    ema_short: int = 9
    ema_long: int = 21
    bb_period: int = 20
    bb_std_dev: float = 2.0
    atr_period: int = 14


@dataclass
class RiskConfig:
    """Risk management settings."""
    max_positions: int = 3
    risk_per_trade_pct: float = 1.5
    stop_loss_pct: float = 2.0
    take_profit_multiplier: float = 3.0
    max_daily_loss_pct: float = 5.0
    min_signal_confidence: float = 0.65


@dataclass
class SignalConfig:
    """Signal generation settings."""
    require_trend_confirmation: bool = True
    require_volume_confirmation: bool = False
    min_rsi_for_buy: int = 35
    max_rsi_for_sell: int = 65


@dataclass
class EnvConfig:
    """Environment variables from .env file."""
    ollama_api_key: str = ""
    taapi_api_key: str = ""
    alpaca_api_key: str = ""
    alpaca_api_secret: str = ""
    alpaca_base_url: str = "https://paper-api.alpaca.markets"


class ConfigLoader:
    """Loads and manages all configuration from config.yaml and .env."""
    
    _instance: Optional['ConfigLoader'] = None
    
    def __new__(cls, config_path: str = "config.yaml") -> 'ConfigLoader':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config_path: str = "config.yaml") -> None:
        if self._initialized:
            return
        
        self._initialized = True
        self.config_path = Path(config_path)
        self._raw_config: Dict[str, Any] = {}
        
        # Load .env file
        self._load_env()
        
        # Load config.yaml
        self._load_yaml()
        
        # Parse into dataclasses
        self._parse_config()
        
        # Setup logging
        self._setup_logging()
    
    def _load_env(self) -> None:
        """Load environment variables from .env file."""
        env_path = Path(".env")
        if env_path.exists():
            load_dotenv(env_path)
        
        self.env = EnvConfig(
            ollama_api_key=os.getenv("OLLAMA_API_KEY", ""),
            taapi_api_key=os.getenv("TAAPI_API_KEY", ""),
            alpaca_api_key=os.getenv("ALPACA_API_KEY", ""),
            alpaca_api_secret=os.getenv("ALPACA_API_SECRET", ""),
            alpaca_base_url=os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets"),
        )
    
    def _load_yaml(self) -> None:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            logging.warning(f"Config file {self.config_path} not found, using defaults")
            return
        
        with open(self.config_path, "r", encoding="utf-8") as f:
            self._raw_config = yaml.safe_load(f) or {}
    
    def _parse_config(self) -> None:
        """Parse raw config into typed dataclasses."""
        # Bot config
        bot_cfg = self._raw_config.get("bot", {})
        self.bot = BotConfig(
            name=bot_cfg.get("name", "AI Crypto Trader"),
            mode=bot_cfg.get("mode", "paper"),
            base_currency=bot_cfg.get("base_currency", "USD"),
            loop_interval_seconds=bot_cfg.get("loop_interval_seconds", 30),
            log_level=bot_cfg.get("log_level", "INFO"),
            log_file=bot_cfg.get("log_file", "logs/trading.log"),
        )
        
        # AI config for Ollama Cloud
        ai_cfg = self._raw_config.get("ai", {})
        self.ai = AIConfig(
            provider=ai_cfg.get("provider", "ollama_cloud"),
            model=ai_cfg.get("model", "minimax-m2.7"),
            base_url=ai_cfg.get("base_url", "https://api.ollama.com/v1"),
            temperature=ai_cfg.get("temperature", 0.3),
            max_tokens=ai_cfg.get("max_tokens", 1000),
            reasoning_prompt_style=ai_cfg.get("reasoning_prompt_style", "chain_of_thought"),
            timeout_seconds=ai_cfg.get("timeout_seconds", 120),
            fallback_models=ai_cfg.get("fallback_models", ["glm-4-plus", "gpt-120b", "deepseek-r1", "qwen-max"]),
        )
        
        # Markets config
        markets_cfg = self._raw_config.get("markets", {})
        crypto_cfg = markets_cfg.get("crypto", {})
        forex_cfg = markets_cfg.get("forex", {})
        self.markets = MarketConfig(
            crypto_enabled=crypto_cfg.get("enabled", True),
            crypto_pairs=crypto_cfg.get("pairs", ["BTC/USD", "ETH/USD", "SOL/USD", "AVAX/USD"]),
            crypto_exchange=crypto_cfg.get("exchange", "alpaca"),
            forex_enabled=forex_cfg.get("enabled", True),
            forex_pairs=forex_cfg.get("pairs", ["EUR/USD", "GBP/USD"]),
            forex_exchange=forex_cfg.get("exchange", "alpaca"),
        )
        
        # Timeframes config
        tf_cfg = self._raw_config.get("timeframes", {})
        trend_cfg = tf_cfg.get("trend", {})
        entry_cfg = tf_cfg.get("entry", {})
        self.timeframes = TimeframeConfig(
            trend_interval=trend_cfg.get("interval", "10Min"),
            trend_lookback_bars=trend_cfg.get("lookback_bars", 50),
            entry_interval=entry_cfg.get("interval", "5Min"),
            entry_lookback_bars=entry_cfg.get("lookback_bars", 30),
        )
        
        # Indicators config
        ind_cfg = self._raw_config.get("indicators", {})
        rsi_cfg = ind_cfg.get("rsi", {})
        macd_cfg = ind_cfg.get("macd", {})
        ema_cfg = ind_cfg.get("ema", {})
        bb_cfg = ind_cfg.get("bollinger_bands", {})
        atr_cfg = ind_cfg.get("atr", {})
        self.indicators = IndicatorConfig(
            rsi_period=rsi_cfg.get("period", 14),
            rsi_oversold=rsi_cfg.get("oversold", 30),
            rsi_overbought=rsi_cfg.get("overbought", 70),
            macd_fast=macd_cfg.get("fast", 12),
            macd_slow=macd_cfg.get("slow", 26),
            macd_signal=macd_cfg.get("signal", 9),
            ema_short=ema_cfg.get("short", 9),
            ema_long=ema_cfg.get("long", 21),
            bb_period=bb_cfg.get("period", 20),
            bb_std_dev=bb_cfg.get("std_dev", 2.0),
            atr_period=atr_cfg.get("period", 14),
        )
        
        # Risk config
        risk_cfg = self._raw_config.get("risk", {})
        self.risk = RiskConfig(
            max_positions=risk_cfg.get("max_positions", 3),
            risk_per_trade_pct=risk_cfg.get("risk_per_trade_pct", 1.5),
            stop_loss_pct=risk_cfg.get("stop_loss_pct", 2.0),
            take_profit_multiplier=risk_cfg.get("take_profit_multiplier", 3.0),
            max_daily_loss_pct=risk_cfg.get("max_daily_loss_pct", 5.0),
            min_signal_confidence=risk_cfg.get("min_signal_confidence", 0.65),
        )
        
        # Signals config
        sig_cfg = self._raw_config.get("signals", {})
        self.signals = SignalConfig(
            require_trend_confirmation=sig_cfg.get("require_trend_confirmation", True),
            require_volume_confirmation=sig_cfg.get("require_volume_confirmation", False),
            min_rsi_for_buy=sig_cfg.get("min_rsi_for_buy", 35),
            max_rsi_for_sell=sig_cfg.get("max_rsi_for_sell", 65),
        )
    
    def _setup_logging(self) -> None:
        """Setup logging based on configuration with daily log files."""
        from datetime import datetime
        from logging.handlers import TimedRotatingFileHandler
        
        log_dir = Path(self.bot.log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_level = getattr(logging, self.bot.log_level.upper(), logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # Daily log file with date in filename
        today = datetime.now().strftime("%Y-%m-%d")
        daily_log_file = log_dir / f"trading_{today}.log"
        
        # TimedRotatingFileHandler for daily rotation
        file_handler = TimedRotatingFileHandler(
            daily_log_file,
            when="midnight",
            interval=1,
            backupCount=30,  # Keep 30 days of logs
            encoding="utf-8"
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        file_handler.suffix = "%Y-%m-%d.log"  # Date suffix for rotated files
        
        # Also write to main trading.log for current session
        from logging.handlers import RotatingFileHandler
        main_log_handler = RotatingFileHandler(
            self.bot.log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding="utf-8"
        )
        main_log_handler.setLevel(log_level)
        main_log_handler.setFormatter(formatter)
        
        # Console handler (for debugging)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        console_handler.setFormatter(formatter)
        
        # Root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(main_log_handler)
        root_logger.addHandler(console_handler)
        
        # Log startup info
        logging.info(f"Logging initialized - Daily log: {daily_log_file}")
    
    def get_raw(self, key: str, default: Any = None) -> Any:
        """Get a raw config value by dot-separated key path."""
        keys = key.split(".")
        value = self._raw_config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
        return value
    
    def reload(self) -> None:
        """Reload configuration from files."""
        self._initialized = False
        self.__init__(str(self.config_path))


# Global config instance
_config: Optional[ConfigLoader] = None


def get_config(config_path: str = "config.yaml") -> ConfigLoader:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = ConfigLoader(config_path)
    return _config
