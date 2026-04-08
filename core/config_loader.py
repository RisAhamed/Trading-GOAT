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
    max_portfolio_exposure_pct: float = 60.0
    max_symbol_exposure_pct: float = 25.0
    # Scalping settings
    quick_profit_threshold: float = 0.3  # Take profit at 0.3% gain
    trailing_stop_pct: float = 0.25      # Trail stop at 0.25%
    max_hold_minutes: int = 30           # Max hold time
    min_profit_to_exit: float = 3.0      # Exit if profit >= $3
    max_hold_seconds: int = 600          # Force exit stale positions (seconds)
    loss_cooldown_minutes: int = 10
    profit_lock_trigger_pct: float = 1.5
    profit_lock_ratio: float = 0.5
    min_profit_lock_pct: float = 0.3
    confluence_min_score: int = 70
    atr_volatility_max_multiplier: float = 2.5


@dataclass
class SignalConfig:
    """Signal generation settings."""
    require_trend_confirmation: bool = True
    require_volume_confirmation: bool = False
    min_rsi_for_buy: int = 35
    max_rsi_for_sell: int = 65
    scalping_mode: bool = False  # Enable aggressive scalping


@dataclass
class RegimeDetectionConfig:
    """Market regime detection settings."""
    primary_symbol: str = "BTC/USD"
    timeframe: str = "1Hour"
    lookback_bars: int = 100
    ema_fast: int = 20
    ema_slow: int = 50
    adx_period: int = 14
    adx_trending_threshold: float = 20.0
    cache_seconds: int = 300


@dataclass
class BearishScalpConfig:
    """Bearish bounce scalp settings."""
    enabled: bool = True
    rsi_entry_max: float = 25.0
    adx_entry_max: float = 30.0
    profit_target_pct: float = 0.8
    stop_loss_pct: float = 0.4
    max_hold_seconds: int = 300
    position_size_multiplier: float = 0.5


@dataclass
class SessionFilterConfig:
    """Session preference settings."""
    prefer_high_volume_sessions: bool = True
    allowed_sessions: List[str] = field(default_factory=lambda: ["ASIAN", "LONDON", "US"])
    off_peak_mode: str = "HOLD_ONLY"


@dataclass
class SymbolScannerConfig:
    """Symbol scanner and ranking configuration."""
    enabled: bool = True
    scan_interval_seconds: int = 120
    min_score_to_trade: int = 40
    max_symbols_to_trade: int = 3
    candidate_pool: List[str] = field(
        default_factory=lambda: ["BTC/USD", "ETH/USD", "SOL/USD"]
    )
    btc_correlation_guard: bool = True
    log_rankings: bool = True


@dataclass
class PoliticalSignalConfig:
    """Political signal scanner configuration."""
    enabled: bool = True
    fetch_interval_seconds: int = 3600
    lookback_days: int = 30
    min_trade_size: int = 1000
    apply_to_crypto: bool = True
    log_signals: bool = True


@dataclass
class MarketIntelligenceConfig:
    """Market intelligence external signal layer configuration."""
    enabled: bool = True
    fear_greed_enabled: bool = True
    whale_tracking_enabled: bool = True
    min_whale_usd: float = 500000.0
    relative_strength_enabled: bool = True
    relative_strength_lookback: int = 10
    breakout_detection_enabled: bool = True
    breakout_lookback_bars: int = 10


@dataclass
class ExitEngineConfig:
    """Trade exit engine configuration."""
    enabled: bool = True
    max_trade_hours: float = 8
    partial_exit_r: float = 1.5
    breakeven_r: float = 1.0
    signal_reversal_exit: bool = True
    dynamic_sizing: bool = True
    min_risk_pct: float = 0.25
    max_risk_pct: float = 2.5


@dataclass
class WebUIConfig:
    """Web dashboard settings."""
    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 5000
    auto_open_browser: bool = True


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
            max_portfolio_exposure_pct=risk_cfg.get("max_portfolio_exposure_pct", 60.0),
            max_symbol_exposure_pct=risk_cfg.get("max_symbol_exposure_pct", 25.0),
            # Scalping settings
            quick_profit_threshold=risk_cfg.get("quick_profit_threshold", 0.3),
            trailing_stop_pct=risk_cfg.get("trailing_stop_pct", 0.25),
            max_hold_minutes=risk_cfg.get("max_hold_minutes", 30),
            min_profit_to_exit=risk_cfg.get("min_profit_to_exit", 3.0),
            max_hold_seconds=risk_cfg.get("max_hold_seconds", 600),
            loss_cooldown_minutes=risk_cfg.get("loss_cooldown_minutes", 10),
            profit_lock_trigger_pct=risk_cfg.get("profit_lock_trigger_pct", 1.5),
            profit_lock_ratio=risk_cfg.get("profit_lock_ratio", 0.5),
            min_profit_lock_pct=risk_cfg.get("min_profit_lock_pct", 0.3),
            confluence_min_score=risk_cfg.get("confluence_min_score", 70),
            atr_volatility_max_multiplier=risk_cfg.get("atr_volatility_max_multiplier", 2.5),
        )
        
        # Signals config
        sig_cfg = self._raw_config.get("signals", {})
        self.signals = SignalConfig(
            require_trend_confirmation=sig_cfg.get("require_trend_confirmation", True),
            require_volume_confirmation=sig_cfg.get("require_volume_confirmation", False),
            min_rsi_for_buy=sig_cfg.get("min_rsi_for_buy", 35),
            max_rsi_for_sell=sig_cfg.get("max_rsi_for_sell", 65),
            scalping_mode=sig_cfg.get("scalping_mode", False),
        )

        # Regime detection config
        regime_cfg = self._raw_config.get("regime_detection", {})
        self.regime_detection = RegimeDetectionConfig(
            primary_symbol=regime_cfg.get("primary_symbol", "BTC/USD"),
            timeframe=regime_cfg.get("timeframe", "1Hour"),
            lookback_bars=regime_cfg.get("lookback_bars", 100),
            ema_fast=regime_cfg.get("ema_fast", 20),
            ema_slow=regime_cfg.get("ema_slow", 50),
            adx_period=regime_cfg.get("adx_period", 14),
            adx_trending_threshold=regime_cfg.get("adx_trending_threshold", 20.0),
            cache_seconds=regime_cfg.get("cache_seconds", 300),
        )

        # Bearish scalp config
        bearish_scalp_cfg = self._raw_config.get("bearish_scalp", {})
        self.bearish_scalp = BearishScalpConfig(
            enabled=bearish_scalp_cfg.get("enabled", True),
            rsi_entry_max=bearish_scalp_cfg.get("rsi_entry_max", 25.0),
            adx_entry_max=bearish_scalp_cfg.get("adx_entry_max", 30.0),
            profit_target_pct=bearish_scalp_cfg.get("profit_target_pct", 0.8),
            stop_loss_pct=bearish_scalp_cfg.get("stop_loss_pct", 0.4),
            max_hold_seconds=bearish_scalp_cfg.get("max_hold_seconds", 300),
            position_size_multiplier=bearish_scalp_cfg.get("position_size_multiplier", 0.5),
        )

        # Session filter config
        session_filter_cfg = self._raw_config.get("session_filter", {})
        self.session_filter = SessionFilterConfig(
            prefer_high_volume_sessions=session_filter_cfg.get("prefer_high_volume_sessions", True),
            allowed_sessions=session_filter_cfg.get("allowed_sessions", ["ASIAN", "LONDON", "US"]),
            off_peak_mode=session_filter_cfg.get("off_peak_mode", "HOLD_ONLY"),
        )

        # Symbol scanner config
        scanner_cfg = self._raw_config.get("symbol_scanner", {})
        self.symbol_scanner = SymbolScannerConfig(
            enabled=scanner_cfg.get("enabled", True),
            scan_interval_seconds=scanner_cfg.get("scan_interval_seconds", 120),
            min_score_to_trade=scanner_cfg.get("min_score_to_trade", 40),
            max_symbols_to_trade=scanner_cfg.get("max_symbols_to_trade", 3),
            candidate_pool=scanner_cfg.get("candidate_pool", ["BTC/USD", "ETH/USD"]),
            btc_correlation_guard=scanner_cfg.get("btc_correlation_guard", True),
            log_rankings=scanner_cfg.get("log_rankings", True),
        )

        # Political signals config
        pol_cfg = self._raw_config.get("political_signals", {})
        self.political_signals = PoliticalSignalConfig(
            enabled=pol_cfg.get("enabled", True),
            fetch_interval_seconds=pol_cfg.get("fetch_interval_seconds", 3600),
            lookback_days=pol_cfg.get("lookback_days", 30),
            min_trade_size=pol_cfg.get("min_trade_size", 1000),
            apply_to_crypto=pol_cfg.get("apply_to_crypto", True),
            log_signals=pol_cfg.get("log_signals", True),
        )

        # Market intelligence config
        intel_cfg = self._raw_config.get("market_intelligence", {})
        self.market_intelligence = MarketIntelligenceConfig(
            enabled=intel_cfg.get("enabled", True),
            fear_greed_enabled=intel_cfg.get("fear_greed_enabled", True),
            whale_tracking_enabled=intel_cfg.get("whale_tracking_enabled", True),
            min_whale_usd=intel_cfg.get("min_whale_usd", 500000),
            relative_strength_enabled=intel_cfg.get("relative_strength_enabled", True),
            relative_strength_lookback=intel_cfg.get("relative_strength_lookback", 10),
            breakout_detection_enabled=intel_cfg.get("breakout_detection_enabled", True),
            breakout_lookback_bars=intel_cfg.get("breakout_lookback_bars", 10),
        )

        # Exit engine config
        exit_cfg = self._raw_config.get("exit_engine", {})
        self.exit_engine = ExitEngineConfig(
            enabled=exit_cfg.get("enabled", True),
            max_trade_hours=exit_cfg.get("max_trade_hours", 8),
            partial_exit_r=exit_cfg.get("partial_exit_r", 1.5),
            breakeven_r=exit_cfg.get("breakeven_r", 1.0),
            signal_reversal_exit=exit_cfg.get("signal_reversal_exit", True),
            dynamic_sizing=exit_cfg.get("dynamic_sizing", True),
            min_risk_pct=exit_cfg.get("min_risk_pct", 0.25),
            max_risk_pct=exit_cfg.get("max_risk_pct", 2.5),
        )
        
        # Web UI config
        web_cfg = self._raw_config.get("web_ui", {})
        self.web_ui = WebUIConfig(
            enabled=web_cfg.get("enabled", True),
            host=web_cfg.get("host", "127.0.0.1"),
            port=web_cfg.get("port", 5000),
            auto_open_browser=web_cfg.get("auto_open_browser", True),
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
