# core/portfolio_tracker.py
"""
Portfolio tracker module for monitoring positions and P&L.
Fetches positions from Alpaca and calculates metrics.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, date, timezone
from typing import Any, Dict, List, Optional

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide

from .config_loader import get_config, ConfigLoader


logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Container for a single position."""
    symbol: str
    qty: float
    side: str  # "long" or "short"
    entry_price: float
    current_price: float
    market_value: float
    cost_basis: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    
    # Stop loss and take profit tracking
    stop_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    
    # Time tracking
    entry_time: Optional[datetime] = None
    time_in_trade: str = ""
    
    # Status
    status: str = "HOLD"  # HOLD, WATCH, NEAR_STOP, NEAR_TP


@dataclass
class TradeRecord:
    """Container for a completed trade."""
    symbol: str
    side: str
    qty: float
    entry_price: float
    exit_price: float
    pnl: float
    pnl_pct: float
    entry_time: datetime
    exit_time: datetime
    duration: str
    is_win: bool


@dataclass
class PortfolioState:
    """Container for overall portfolio state."""
    total_value: float
    cash: float
    buying_power: float
    equity: float
    
    # P&L metrics
    today_realized_pnl: float
    today_unrealized_pnl: float
    today_total_pnl: float
    today_pnl_pct: float
    
    # Position metrics
    open_positions: int
    total_position_value: float
    
    # Trading metrics
    today_trades: int
    today_wins: int
    today_losses: int
    win_rate: float
    
    # Status
    is_halted: bool
    halt_reason: str
    
    # Timestamp
    last_updated: str


class PortfolioTracker:
    """
    Tracks portfolio state, positions, and P&L.
    Interfaces with Alpaca to get real-time position data.
    """
    
    def __init__(self, config: Optional[ConfigLoader] = None) -> None:
        """Initialize the portfolio tracker."""
        self.config = config or get_config()
        
        # Initialize Alpaca trading client
        self.client = TradingClient(
            api_key=self.config.env.alpaca_api_key,
            secret_key=self.config.env.alpaca_api_secret,
            paper=True,
        )
        
        # Position cache
        self._positions: Dict[str, Position] = {}
        self._last_positions_update: Optional[datetime] = None
        
        # Trade history
        self._trade_history: List[TradeRecord] = []
        self._max_history = 100
        
        # Daily stats
        self._today = date.today()
        self._starting_value: Optional[float] = None
        self._today_realized_pnl = 0.0
        self._today_trades = 0
        self._today_wins = 0
        self._today_losses = 0
        
        logger.info("PortfolioTracker initialized")
    
    def _reset_daily_stats(self, current_value: float) -> None:
        """Reset daily statistics for a new trading day."""
        self._today = date.today()
        self._starting_value = current_value
        self._today_realized_pnl = 0.0
        self._today_trades = 0
        self._today_wins = 0
        self._today_losses = 0
        logger.info("Daily stats reset")
    
    def update(self) -> PortfolioState:
        """
        Update portfolio state from Alpaca.
        
        Returns:
            PortfolioState with current portfolio data
        """
        try:
            # Get account info
            account = self.client.get_account()
            
            total_value = float(account.portfolio_value)
            cash = float(account.cash)
            buying_power = float(account.buying_power)
            equity = float(account.equity)
            
            # Check for new day
            if date.today() != self._today:
                self._reset_daily_stats(total_value)
            
            # Initialize starting value if not set
            if self._starting_value is None:
                self._starting_value = total_value
            
            # Update positions
            self._update_positions()
            
            # Calculate unrealized P&L
            total_unrealized_pnl = sum(
                p.unrealized_pnl for p in self._positions.values()
            )
            
            # Calculate total position value
            total_position_value = sum(
                abs(p.market_value) for p in self._positions.values()
            )
            
            # Calculate today's P&L
            today_total_pnl = self._today_realized_pnl + total_unrealized_pnl
            today_pnl_pct = (today_total_pnl / self._starting_value * 100) if self._starting_value else 0
            
            # Calculate win rate
            total_trades = self._today_wins + self._today_losses
            win_rate = (self._today_wins / total_trades * 100) if total_trades > 0 else 0.0
            
            state = PortfolioState(
                total_value=total_value,
                cash=cash,
                buying_power=buying_power,
                equity=equity,
                today_realized_pnl=self._today_realized_pnl,
                today_unrealized_pnl=total_unrealized_pnl,
                today_total_pnl=today_total_pnl,
                today_pnl_pct=today_pnl_pct,
                open_positions=len(self._positions),
                total_position_value=total_position_value,
                today_trades=self._today_trades,
                today_wins=self._today_wins,
                today_losses=self._today_losses,
                win_rate=win_rate,
                is_halted=False,
                halt_reason="",
                last_updated=datetime.now().isoformat(),
            )
            
            return state
            
        except Exception as e:
            logger.error(f"Error updating portfolio: {e}")
            # Return empty state on error
            return PortfolioState(
                total_value=0,
                cash=0,
                buying_power=0,
                equity=0,
                today_realized_pnl=0,
                today_unrealized_pnl=0,
                today_total_pnl=0,
                today_pnl_pct=0,
                open_positions=0,
                total_position_value=0,
                today_trades=0,
                today_wins=0,
                today_losses=0,
                win_rate=0,
                is_halted=False,
                halt_reason="Error fetching portfolio",
                last_updated=datetime.now().isoformat(),
            )
    
    def _update_positions(self) -> None:
        """Update positions from Alpaca."""
        try:
            positions = self.client.get_all_positions()
            
            # Track which symbols we've seen
            seen_symbols = set()
            
            for pos in positions:
                symbol = pos.symbol
                
                # Convert symbol format (BTCUSD -> BTC/USD)
                if len(symbol) > 3 and "USD" in symbol:
                    base = symbol.replace("USD", "")
                    symbol = f"{base}/USD"
                
                seen_symbols.add(symbol)
                
                qty = float(pos.qty)
                side = "long" if qty > 0 else "short"
                qty = abs(qty)
                
                entry_price = float(pos.avg_entry_price)
                current_price = float(pos.current_price)
                market_value = float(pos.market_value)
                cost_basis = float(pos.cost_basis)
                unrealized_pnl = float(pos.unrealized_pl)
                unrealized_pnl_pct = float(pos.unrealized_plpc) * 100
                
                # Determine status based on P&L
                if unrealized_pnl_pct < -1.5:
                    status = "NEAR_STOP"
                elif unrealized_pnl_pct > 4:
                    status = "NEAR_TP"
                elif abs(unrealized_pnl_pct) > 1:
                    status = "WATCH"
                else:
                    status = "HOLD"
                
                # Get or create position
                if symbol in self._positions:
                    position = self._positions[symbol]
                    position.qty = qty
                    position.current_price = current_price
                    position.market_value = market_value
                    position.unrealized_pnl = unrealized_pnl
                    position.unrealized_pnl_pct = unrealized_pnl_pct
                    position.status = status
                else:
                    position = Position(
                        symbol=symbol,
                        qty=qty,
                        side=side,
                        entry_price=entry_price,
                        current_price=current_price,
                        market_value=market_value,
                        cost_basis=cost_basis,
                        unrealized_pnl=unrealized_pnl,
                        unrealized_pnl_pct=unrealized_pnl_pct,
                        entry_time=datetime.now(timezone.utc),
                        status=status,
                    )
                    self._positions[symbol] = position
                
                # Calculate time in trade
                if position.entry_time:
                    delta = datetime.now(timezone.utc) - position.entry_time
                    hours = delta.total_seconds() / 3600
                    if hours < 1:
                        position.time_in_trade = f"{int(delta.total_seconds() / 60)}m"
                    elif hours < 24:
                        position.time_in_trade = f"{hours:.1f}h"
                    else:
                        position.time_in_trade = f"{hours / 24:.1f}d"
            
            # Remove positions that no longer exist
            closed_symbols = set(self._positions.keys()) - seen_symbols
            for symbol in closed_symbols:
                closed_pos = self._positions.pop(symbol)
                logger.info(f"Position closed: {symbol} P&L: ${closed_pos.unrealized_pnl:.2f}")
                
                # Record the trade
                self.record_trade(
                    symbol=symbol,
                    side=closed_pos.side,
                    qty=closed_pos.qty,
                    entry_price=closed_pos.entry_price,
                    exit_price=closed_pos.current_price,
                    entry_time=closed_pos.entry_time,
                )
            
            self._last_positions_update = datetime.now()
            
        except Exception as e:
            logger.error(f"Error updating positions: {e}")
    
    def get_positions(self) -> List[Position]:
        """Get all open positions."""
        return list(self._positions.values())
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get a specific position by symbol."""
        return self._positions.get(symbol)
    
    def get_position_dict(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get position as dictionary for other modules."""
        pos = self._positions.get(symbol)
        if pos:
            return {
                "symbol": pos.symbol,
                "qty": pos.qty,
                "side": pos.side,
                "entry_price": pos.entry_price,
                "current_price": pos.current_price,
                "market_value": pos.market_value,
                "unrealized_pnl": pos.unrealized_pnl,
                "unrealized_pnl_pct": pos.unrealized_pnl_pct,
                "stop_price": pos.stop_price,
                "take_profit_price": pos.take_profit_price,
            }
        return None
    
    def get_all_positions_dict(self) -> List[Dict[str, Any]]:
        """Get all positions as list of dictionaries."""
        return [
            {
                "symbol": pos.symbol,
                "qty": pos.qty,
                "side": pos.side,
                "entry_price": pos.entry_price,
                "current_price": pos.current_price,
                "market_value": pos.market_value,
                "unrealized_pnl": pos.unrealized_pnl,
                "unrealized_pnl_pct": pos.unrealized_pnl_pct,
            }
            for pos in self._positions.values()
        ]
    
    def record_trade(
        self,
        symbol: str,
        side: str,
        qty: float,
        entry_price: float,
        exit_price: float,
        entry_time: Optional[datetime] = None,
    ) -> None:
        """
        Record a completed trade.
        
        Args:
            symbol: Trading pair
            side: "long" or "short"
            qty: Trade quantity
            entry_price: Entry price
            exit_price: Exit price
            entry_time: When trade was opened
        """
        # Calculate P&L
        if side == "long":
            pnl = (exit_price - entry_price) * qty
        else:
            pnl = (entry_price - exit_price) * qty
        
        pnl_pct = (pnl / (entry_price * qty)) * 100 if entry_price > 0 else 0
        is_win = pnl > 0
        
        exit_time = datetime.now(timezone.utc)
        entry_time = entry_time or exit_time
        
        # Calculate duration
        delta = exit_time - entry_time
        hours = delta.total_seconds() / 3600
        if hours < 1:
            duration = f"{int(delta.total_seconds() / 60)}m"
        elif hours < 24:
            duration = f"{hours:.1f}h"
        else:
            duration = f"{hours / 24:.1f}d"
        
        record = TradeRecord(
            symbol=symbol,
            side=side,
            qty=qty,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl=pnl,
            pnl_pct=pnl_pct,
            entry_time=entry_time,
            exit_time=exit_time,
            duration=duration,
            is_win=is_win,
        )
        
        # Update daily stats
        self._today_realized_pnl += pnl
        self._today_trades += 1
        if is_win:
            self._today_wins += 1
        else:
            self._today_losses += 1
        
        # Add to history
        self._trade_history.append(record)
        if len(self._trade_history) > self._max_history:
            self._trade_history.pop(0)
        
        logger.info(
            f"Trade recorded: {side.upper()} {symbol} | "
            f"P&L: ${pnl:+.2f} ({pnl_pct:+.1f}%) | "
            f"{'WIN' if is_win else 'LOSS'}"
        )
    
    def get_trade_history(self, limit: int = 10) -> List[TradeRecord]:
        """Get recent trade history."""
        return self._trade_history[-limit:]
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get a summary for display purposes."""
        state = self.update()
        
        return {
            "total_value": state.total_value,
            "cash": state.cash,
            "buying_power": state.buying_power,
            "today_pnl": state.today_total_pnl,
            "today_pnl_pct": state.today_pnl_pct,
            "open_positions": state.open_positions,
            "today_trades": state.today_trades,
            "win_rate": state.win_rate,
            "positions": [
                {
                    "symbol": p.symbol,
                    "entry": p.entry_price,
                    "current": p.current_price,
                    "qty": p.qty,
                    "pnl": p.unrealized_pnl,
                    "pnl_pct": p.unrealized_pnl_pct,
                    "status": p.status,
                }
                for p in self._positions.values()
            ],
            "recent_trades": [
                {
                    "time": t.exit_time.strftime("%H:%M"),
                    "action": "BUY" if t.side == "long" else "SELL",
                    "symbol": t.symbol,
                    "qty": t.qty,
                    "price": t.exit_price,
                    "pnl": t.pnl,
                    "result": "WIN" if t.is_win else "LOSS",
                }
                for t in self._trade_history[-5:]
            ],
        }
    
    def set_position_stops(
        self,
        symbol: str,
        stop_price: Optional[float] = None,
        take_profit_price: Optional[float] = None,
    ) -> None:
        """Set stop loss and take profit prices for a position."""
        if symbol in self._positions:
            if stop_price is not None:
                self._positions[symbol].stop_price = stop_price
            if take_profit_price is not None:
                self._positions[symbol].take_profit_price = take_profit_price

    def get_position_stop(self, symbol: str) -> Optional[float]:
        """Return current stop price for a symbol, or None if not tracked."""
        try:
            pos = self._positions.get(symbol)
            return pos.stop_price if pos else None
        except Exception:
            return None
