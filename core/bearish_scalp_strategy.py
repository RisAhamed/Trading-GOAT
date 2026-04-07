"""
Bearish micro-scalp strategy for oversold bounce entries.
"""

import logging
from typing import Optional

from .config_loader import ConfigLoader, get_config


logger = logging.getLogger(__name__)


class BearishScalpStrategy:
    """
    Defines a constrained bearish-market bounce scalp filter.

    This strategy only allows high-selectivity BUY entries in bearish conditions
    when RSI is below the configured oversold threshold (default 25) and ADX is
    below the configured trend-strength threshold (default 30).
    """

    def __init__(self, config: Optional[ConfigLoader] = None) -> None:
        """Initialize bearish scalp strategy config."""
        self.config = config or get_config()
        scalp_cfg = getattr(self.config, "bearish_scalp", None)

        self.enabled = bool(getattr(scalp_cfg, "enabled", True))
        self.rsi_entry_max = float(getattr(scalp_cfg, "rsi_entry_max", 25.0))
        self.adx_entry_max = float(getattr(scalp_cfg, "adx_entry_max", 30.0))
        self.profit_target_pct = float(getattr(scalp_cfg, "profit_target_pct", 0.8))
        self.stop_loss_pct = float(getattr(scalp_cfg, "stop_loss_pct", 0.4))
        self.max_hold_seconds = int(getattr(scalp_cfg, "max_hold_seconds", 300))
        self.position_size_multiplier = float(
            getattr(scalp_cfg, "position_size_multiplier", 0.5)
        )

    def should_enter_bearish_scalp(self, rsi: float, adx: float) -> bool:
        """Return True only when RSI < rsi_entry_max and ADX < adx_entry_max."""
        try:
            allowed = self.enabled and rsi < self.rsi_entry_max and adx < self.adx_entry_max
            logger.info(
                "Bearish scalp check: enabled=%s rsi=%.2f<%.2f adx=%.2f<%.2f => %s",
                self.enabled,
                rsi,
                self.rsi_entry_max,
                adx,
                self.adx_entry_max,
                allowed,
            )
            return allowed
        except Exception as e:
            logger.error(f"Bearish scalp check error: {e}")
            return False

    def get_bearish_scalp_params(self) -> dict:
        """Get bearish scalp execution parameters."""
        try:
            return {
                "profit_target_pct": self.profit_target_pct,
                "stop_loss_pct": self.stop_loss_pct,
                "max_hold_seconds": self.max_hold_seconds,
                "position_size_multiplier": self.position_size_multiplier,
            }
        except Exception as e:
            logger.error(f"Get bearish scalp params error: {e}")
            return {
                "profit_target_pct": 0.8,
                "stop_loss_pct": 0.4,
                "max_hold_seconds": 300,
                "position_size_multiplier": 0.5,
            }
