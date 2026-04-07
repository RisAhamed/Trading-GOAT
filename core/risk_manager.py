# core/risk_manager.py
"""
Risk management module for position sizing, stop loss, and take profit calculations.
Enforces trading limits and daily loss limits.
Integrates with trailing stop manager for dynamic stop loss management.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, date
from typing import Any, Dict, List, Optional

from .config_loader import get_config, ConfigLoader


logger = logging.getLogger(__name__)


@dataclass
class RiskParameters:
    """Container for calculated risk parameters."""
    # Position sizing
    qty: float  # Quantity to trade
    position_value: float  # Dollar value of position
    
    # Stop loss and take profit
    stop_price: float
    take_profit_price: float
    stop_loss_distance: float  # In dollars
    take_profit_distance: float  # In dollars
    
    # Risk metrics
    max_loss_usd: float  # Maximum loss in USD
    risk_reward_ratio: float  # Expected R:R
    
    # Approval status
    is_allowed: bool
    rejection_reason: str  # Empty if allowed
    
    # Metadata
    entry_price: float
    symbol: str
    side: str  # "long" or "short"
    position_size_override: float = 1.0


@dataclass
class DailyStats:
    """Container for daily trading statistics."""
    date: date
    starting_value: float
    current_value: float
    realized_pnl: float
    unrealized_pnl: float
    trades_count: int
    wins: int
    losses: int
    
    @property
    def total_pnl(self) -> float:
        return self.realized_pnl + self.unrealized_pnl
    
    @property
    def total_pnl_pct(self) -> float:
        if self.starting_value > 0:
            return (self.total_pnl / self.starting_value) * 100
        return 0.0
    
    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses
        return (self.wins / total * 100) if total > 0 else 0.0


class RiskManager:
    """
    Manages risk for all trading operations.
    Calculates position sizes, stop losses, and enforces trading limits.
    """
    
    def __init__(self, config: Optional[ConfigLoader] = None) -> None:
        """Initialize the risk manager."""
        self.config = config or get_config()
        
        # Get risk settings from config
        self.max_positions = self.config.risk.max_positions
        self.risk_per_trade_pct = self.config.risk.risk_per_trade_pct
        self.stop_loss_pct = self.config.risk.stop_loss_pct
        self.take_profit_multiplier = self.config.risk.take_profit_multiplier
        self.max_daily_loss_pct = self.config.risk.max_daily_loss_pct
        self.max_portfolio_exposure_pct = self.config.risk.max_portfolio_exposure_pct
        self.max_symbol_exposure_pct = self.config.risk.max_symbol_exposure_pct
        
        # Trading state
        self._is_halted = False
        self._halt_reason = ""
        
        # Daily stats
        self._daily_stats = DailyStats(
            date=date.today(),
            starting_value=0.0,
            current_value=0.0,
            realized_pnl=0.0,
            unrealized_pnl=0.0,
            trades_count=0,
            wins=0,
            losses=0,
        )
        
        logger.info("RiskManager initialized")
    
    def calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        portfolio_value: float,
        side: str = "long",
        atr: Optional[float] = None,
    ) -> RiskParameters:
        """
        Calculate position size and risk parameters for a trade.
        
        Formula: position_size = (portfolio_value × risk_per_trade_pct) / stop_loss_pct
        
        Args:
            symbol: Trading pair
            entry_price: Expected entry price
            portfolio_value: Total portfolio value
            side: "long" or "short"
            atr: Optional ATR for dynamic stop loss
            
        Returns:
            RiskParameters with all calculated values
        """
        # Initialize result
        result = RiskParameters(
            qty=0.0,
            position_value=0.0,
            stop_price=0.0,
            take_profit_price=0.0,
            stop_loss_distance=0.0,
            take_profit_distance=0.0,
            max_loss_usd=0.0,
            risk_reward_ratio=0.0,
            is_allowed=False,
            rejection_reason="",
            entry_price=entry_price,
            symbol=symbol,
            side=side,
        )
        
        # Validate inputs
        if entry_price <= 0:
            result.rejection_reason = "Invalid entry price"
            return result
        
        if portfolio_value <= 0:
            result.rejection_reason = "Invalid portfolio value"
            return result
        
        try:
            # ═══ TRAILING STOP INTEGRATION ═══════════════════════════════════════
            # Check if trailing stop is enabled - use those settings for stop loss
            trailing_config = self.config._raw_config.get("trailing_stop", {})
            trailing_enabled = trailing_config.get("enabled", False)

            # Calculate risk amount in USD
            # risk_per_trade_pct is the % of portfolio we're willing to lose on this trade
            risk_amount_usd = portfolio_value * (self.risk_per_trade_pct / 100)

            # Calculate stop loss distance
            if trailing_enabled:
                # Use hard_stop_pct from trailing stop config as the worst-case stop
                hard_stop_pct = trailing_config.get("hard_stop_pct", 1.5)
                stop_loss_pct = hard_stop_pct
                stop_loss_distance = entry_price * (stop_loss_pct / 100)
                logger.debug(
                    f"Using trailing stop hard_stop as risk calc: {hard_stop_pct}% "
                    f"(trailing stop will manage dynamically)"
                )
            elif atr and atr > 0:
                # Use 2x ATR for stop loss distance
                stop_loss_distance = atr * 2
                stop_loss_pct = (stop_loss_distance / entry_price) * 100
            else:
                # Use fixed percentage from config
                stop_loss_pct = self.stop_loss_pct
                stop_loss_distance = entry_price * (stop_loss_pct / 100)
            
            # Calculate position size
            # position_size = risk_amount / (stop_loss_pct / 100)
            # This ensures if we hit stop loss, we lose exactly risk_amount
            if stop_loss_pct > 0:
                position_value = risk_amount_usd / (stop_loss_pct / 100)
            else:
                position_value = risk_amount_usd / 0.02  # Default 2%
            
            # Calculate quantity
            qty = position_value / entry_price
            
            # Calculate stop loss price
            if side == "long":
                stop_price = entry_price - stop_loss_distance
            else:  # short
                stop_price = entry_price + stop_loss_distance
            
            # Calculate take profit price
            # Take profit = stop loss distance × multiplier
            take_profit_distance = stop_loss_distance * self.take_profit_multiplier
            
            if side == "long":
                take_profit_price = entry_price + take_profit_distance
            else:  # short
                take_profit_price = entry_price - take_profit_distance
            
            # Calculate maximum loss
            max_loss_usd = qty * stop_loss_distance
            
            # Calculate risk/reward ratio
            risk_reward_ratio = self.take_profit_multiplier  # By definition
            
            # Cap position value to max symbol exposure
            MAX_PER_TRADE = portfolio_value * (self.max_symbol_exposure_pct / 100)
            
            if position_value > MAX_PER_TRADE:
                logger.info(
                    f"Position value ${position_value:.2f} exceeds max ${MAX_PER_TRADE:.2f}, "
                    f"capping to max"
                )
                position_value = MAX_PER_TRADE
                qty = position_value / entry_price
                max_loss_usd = qty * stop_loss_distance
            
            # Update result
            result.qty = qty
            result.position_value = position_value
            result.stop_price = stop_price
            result.take_profit_price = take_profit_price
            result.stop_loss_distance = stop_loss_distance
            result.take_profit_distance = take_profit_distance
            result.max_loss_usd = max_loss_usd
            result.risk_reward_ratio = risk_reward_ratio
            result.is_allowed = True
            
            logger.debug(
                f"Position size for {symbol}: qty={qty:.6f}, "
                f"value=${position_value:.2f}, stop=${stop_price:.4f}, "
                f"tp=${take_profit_price:.4f}"
            )
            
        except Exception as e:
            result.rejection_reason = f"Calculation error: {str(e)}"
            logger.error(f"Error calculating position size: {e}")
        
        return result
    
    def check_trade_allowed(
        self,
        symbol: str,
        action: str,
        current_positions: List[Dict[str, Any]],
        portfolio_value: float,
        cash_available: float,
        position_value: float,
    ) -> tuple[bool, str]:
        """
        Check if a new trade is allowed based on risk rules.
        
        Args:
            symbol: Trading pair
            action: Trade action (BUY, SELL)
            current_positions: List of current open positions
            portfolio_value: Total portfolio value
            cash_available: Available cash for trading
            position_value: Value of proposed position
            
        Returns:
            Tuple of (is_allowed, rejection_reason)
        """
        # Check if trading is halted
        if self._is_halted:
            return False, f"Trading halted: {self._halt_reason}"
        
        # Check daily loss limit
        if self._daily_stats.total_pnl_pct <= -self.max_daily_loss_pct:
            self._halt_trading(f"Daily loss limit exceeded ({self._daily_stats.total_pnl_pct:.1f}%)")
            return False, self._halt_reason
        
        # Check max positions
        if action in ["BUY", "SELL"]:
            open_position_count = len(current_positions)
            
            # Check if we already have a position in this symbol
            existing_position = None
            for pos in current_positions:
                if pos.get('symbol') == symbol:
                    existing_position = pos
                    break
            
            if existing_position:
                # Already have position in this symbol
                if action == "BUY" and existing_position.get('side') == 'long':
                    return False, f"Already have long position in {symbol}"
                elif action == "SELL" and existing_position.get('side') == 'short':
                    return False, f"Already have short position in {symbol}"
            else:
                # New position
                if open_position_count >= self.max_positions:
                    return False, f"Maximum positions ({self.max_positions}) reached"
        
        # Check cash available
        if action == "BUY" and position_value > cash_available:
            return False, f"Insufficient cash (${cash_available:.2f} < ${position_value:.2f})"

        # Check portfolio-level exposure
        total_current_exposure = sum(
            abs(float(pos.get('market_value', 0))) for pos in current_positions
        )
        proposed_total_exposure = total_current_exposure + position_value
        if portfolio_value > 0:
            proposed_exposure_pct = (proposed_total_exposure / portfolio_value) * 100
            if proposed_exposure_pct > self.max_portfolio_exposure_pct:
                return (
                    False,
                    f"Portfolio exposure {proposed_exposure_pct:.1f}% exceeds max "
                    f"{self.max_portfolio_exposure_pct:.1f}%"
                )

        # Check symbol-level exposure
        if portfolio_value > 0:
            symbol_exposure_pct = (position_value / portfolio_value) * 100
            if symbol_exposure_pct > self.max_symbol_exposure_pct:
                return (
                    False,
                    f"Symbol exposure {symbol_exposure_pct:.1f}% exceeds max "
                    f"{self.max_symbol_exposure_pct:.1f}%"
                )

        # Check minimum position value
        min_position = 10.0  # Minimum $10 position
        if position_value < min_position:
            return False, f"Position value ${position_value:.2f} below minimum ${min_position}"
        
        return True, ""
    
    def update_daily_stats(
        self,
        portfolio_value: float,
        realized_pnl: float,
        unrealized_pnl: float,
        trades_count: int,
        wins: int,
        losses: int,
    ) -> None:
        """
        Update daily trading statistics.
        
        Args:
            portfolio_value: Current portfolio value
            realized_pnl: Today's realized P&L
            unrealized_pnl: Current unrealized P&L
            trades_count: Number of trades today
            wins: Number of winning trades
            losses: Number of losing trades
        """
        today = date.today()
        
        # Reset stats if it's a new day
        if self._daily_stats.date != today:
            self._daily_stats = DailyStats(
                date=today,
                starting_value=portfolio_value - realized_pnl - unrealized_pnl,
                current_value=portfolio_value,
                realized_pnl=realized_pnl,
                unrealized_pnl=unrealized_pnl,
                trades_count=trades_count,
                wins=wins,
                losses=losses,
            )
            # Reset halt status for new day
            self._is_halted = False
            self._halt_reason = ""
            logger.info("Daily stats reset for new trading day")
        else:
            self._daily_stats.current_value = portfolio_value
            self._daily_stats.realized_pnl = realized_pnl
            self._daily_stats.unrealized_pnl = unrealized_pnl
            self._daily_stats.trades_count = trades_count
            self._daily_stats.wins = wins
            self._daily_stats.losses = losses
        
        # Check daily loss limit
        if self._daily_stats.total_pnl_pct <= -self.max_daily_loss_pct:
            self._halt_trading(
                f"Daily loss limit exceeded: {self._daily_stats.total_pnl_pct:.1f}% "
                f"(max: -{self.max_daily_loss_pct}%)"
            )
    
    def _halt_trading(self, reason: str) -> None:
        """Halt trading for the day."""
        if not self._is_halted:
            self._is_halted = True
            self._halt_reason = reason
            logger.warning(f"TRADING HALTED: {reason}")
    
    def resume_trading(self) -> None:
        """Resume trading (manual override)."""
        self._is_halted = False
        self._halt_reason = ""
        logger.info("Trading resumed")
    
    def is_halted(self) -> bool:
        """Check if trading is halted."""
        return self._is_halted
    
    def get_halt_reason(self) -> str:
        """Get the reason for trading halt."""
        return self._halt_reason
    
    def get_daily_stats(self) -> DailyStats:
        """Get current daily statistics."""
        return self._daily_stats
    
    def record_trade_result(self, is_win: bool, pnl: float) -> None:
        """
        Record the result of a closed trade.
        
        Args:
            is_win: True if trade was profitable
            pnl: Realized P&L from the trade
        """
        self._daily_stats.trades_count += 1
        self._daily_stats.realized_pnl += pnl
        
        if is_win:
            self._daily_stats.wins += 1
        else:
            self._daily_stats.losses += 1
        
        logger.info(
            f"Trade recorded: {'WIN' if is_win else 'LOSS'} ${pnl:+.2f} | "
            f"Daily: {self._daily_stats.wins}W/{self._daily_stats.losses}L"
        )
    
    def get_position_summary(
        self,
        positions: List[Dict[str, Any]],
        portfolio_value: float,
    ) -> Dict[str, Any]:
        """
        Get a summary of current position risk.
        
        Args:
            positions: List of open positions
            portfolio_value: Total portfolio value
            
        Returns:
            Dictionary with risk summary
        """
        total_position_value = sum(
            abs(float(p.get('market_value', 0))) for p in positions
        )
        
        total_unrealized_pnl = sum(
            float(p.get('unrealized_pnl', 0)) for p in positions
        )
        
        position_exposure_pct = (total_position_value / portfolio_value * 100) if portfolio_value > 0 else 0
        
        return {
            "open_positions": len(positions),
            "max_positions": self.max_positions,
            "total_position_value": total_position_value,
            "position_exposure_pct": position_exposure_pct,
            "total_unrealized_pnl": total_unrealized_pnl,
            "is_halted": self._is_halted,
            "halt_reason": self._halt_reason,
            "daily_pnl": self._daily_stats.total_pnl,
            "daily_pnl_pct": self._daily_stats.total_pnl_pct,
            "win_rate": self._daily_stats.win_rate,
        }
    
    def validate_risk_parameters(self, params: RiskParameters) -> RiskParameters:
        """
        Validate and potentially adjust risk parameters.
        
        Args:
            params: Calculated risk parameters
            
        Returns:
            Validated (possibly adjusted) risk parameters
        """
        if not params.is_allowed:
            return params
        
        # Ensure stop price is reasonable
        if params.side == "long":
            if params.stop_price >= params.entry_price:
                params.is_allowed = False
                params.rejection_reason = "Stop price must be below entry for long"
                return params
            
            if params.take_profit_price <= params.entry_price:
                params.is_allowed = False
                params.rejection_reason = "Take profit must be above entry for long"
                return params
        
        else:  # short
            if params.stop_price <= params.entry_price:
                params.is_allowed = False
                params.rejection_reason = "Stop price must be above entry for short"
                return params
            
            if params.take_profit_price >= params.entry_price:
                params.is_allowed = False
                params.rejection_reason = "Take profit must be below entry for short"
                return params
        
        # Ensure quantity is positive
        if params.qty <= 0:
            params.is_allowed = False
            params.rejection_reason = "Invalid quantity calculated"
            return params
        
        return params
