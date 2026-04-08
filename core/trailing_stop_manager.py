# core/trailing_stop_manager.py
"""
Trailing Stop Manager - Brain of the trailing stop system.

Manages trailing stops, profit protection tiers, and DCA (ladder-in) logic.
Thread-safe for concurrent access from main thread and position monitor thread.
"""

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from .config_loader import get_config, ConfigLoader


logger = logging.getLogger(__name__)


@dataclass
class TrailingPosition:
    """Container for a position being tracked with trailing stop."""
    symbol: str
    entry_price: float
    entry_time: datetime
    qty: float
    current_price: float
    peak_price: float              # Highest price seen since entry
    floor_price: float             # Current trailing floor (ONLY GOES UP)
    hard_stop_price: float         # Absolute hard floor (entry × hard_stop_pct)
    current_trail_pct: float       # Active trail % (may tighten via tiers)
    unrealized_pnl_usd: float
    unrealized_pnl_pct: float
    max_profit_seen_pct: float     # Highest profit % reached
    hold_minutes: float            # How long position has been open
    ladder_count: int              # How many DCA additions made
    last_dip_buy_time: Optional[datetime]
    status: str                    # INIT | TRAILING | PROFIT_LOCK | WATCH | HARD | EXITING
    original_qty: float = 0.0      # Original quantity before DCA
    original_entry: float = 0.0    # Original entry before DCA
    tier1_taken: bool = False
    tier2_taken: bool = False


@dataclass
class TrailingAction:
    """Action result from update_position()."""
    action: str           # HOLD | SELL
    reason: str           # Why this action
    urgency: str          # CRITICAL | HIGH | MEDIUM | LOW | NONE
    floor_price: float
    current_trail_pct: float
    pnl_usd: float = 0.0
    pnl_pct: float = 0.0


@dataclass
class DCAAction:
    """Action result from check_dca_opportunity()."""
    symbol: str
    additional_qty: float
    reason: str
    dip_pct_from_entry: float
    current_price: float


class TrailingStopManager:
    """
    Manages trailing stops for all open positions.
    
    Key rules:
    1. Floor ONLY goes UP, never down
    2. Hard stop is absolute emergency floor
    3. All config values read from config.yaml
    4. Thread-safe with locks
    """
    
    def __init__(self, config: Optional[ConfigLoader] = None) -> None:
        """Initialize the trailing stop manager."""
        self.config = config or get_config()
        
        # Thread lock for positions dict
        self._lock = threading.Lock()
        
        # Positions being tracked: {symbol: TrailingPosition}
        self._positions: Dict[str, TrailingPosition] = {}
        
        # Load trailing_stop config
        ts_config = self.config._raw_config.get("trailing_stop", {})
        self.enabled = ts_config.get("enabled", True)
        self.initial_stop_loss_pct = ts_config.get("initial_stop_loss_pct", 0.5)
        self.trail_pct = ts_config.get("trail_pct", 0.25)
        self.trail_activation_pct = ts_config.get("trail_activation_pct", 0.15)
        self.profit_tiers = ts_config.get("profit_tiers", [])
        self.min_profit_to_exit_usd = ts_config.get("min_profit_to_exit_usd", 3.0)
        self.quick_profit_pct = ts_config.get("quick_profit_pct", 0.3)
        self.max_hold_minutes = ts_config.get("max_hold_minutes", 30)
        self.hard_stop_pct = ts_config.get("hard_stop_pct", 1.5)
        
        # Load ladder_in config
        ladder_config = self.config._raw_config.get("ladder_in", {})
        self.ladder_enabled = ladder_config.get("enabled", True)
        self.dip_threshold_pct = ladder_config.get("dip_threshold_pct", 20.0)
        self.additional_size_pct = ladder_config.get("additional_size_pct", 50.0)
        self.max_ladder_count = ladder_config.get("max_ladder_count", 2)
        self.min_time_between_dips = ladder_config.get("min_time_between_dips_minutes", 5)
        
        # Load position_monitor config
        monitor_config = self.config._raw_config.get("position_monitor", {})
        self.log_trail_updates = monitor_config.get("log_trail_updates", True)
        self.alert_on_floor_hit = monitor_config.get("alert_on_floor_hit", True)
        
        logger.info(f"TrailingStopManager initialized (enabled={self.enabled})")
        logger.info(f"  Initial stop: {self.initial_stop_loss_pct}%, Trail: {self.trail_pct}%")
        logger.info(f"  Hard stop: {self.hard_stop_pct}%, Quick profit: {self.quick_profit_pct}%")
        logger.info(f"  DCA enabled: {self.ladder_enabled}, Max ladders: {self.max_ladder_count}")
    
    def is_enabled(self) -> bool:
        """Check if trailing stop is enabled."""
        return self.enabled
    
    def register_new_position(
        self,
        symbol: str,
        entry_price: float,
        qty: float,
        market_regime: str = "BULLISH",
    ) -> Optional[TrailingPosition]:
        """
        Register a new position for trailing stop tracking.
        
        Args:
            symbol: Trading pair (e.g., "BTC/USD")
            entry_price: Fill price of the entry order
            qty: Quantity bought
            market_regime: Current regime context to tune trail aggressiveness
            
        Returns:
            TrailingPosition object, or None if trailing is disabled
        """
        if not self.enabled:
            logger.debug(f"Trailing stop disabled, not registering {symbol}")
            return None
        
        try:
            multiplier = 1.0
            if market_regime == "BEARISH":
                multiplier = 0.5
            elif market_regime == "SIDEWAYS":
                multiplier = 0.7

            trail_distance = self.trail_pct * multiplier
            logger.info(
                f"Trail distance: {trail_distance:.4f} (regime={market_regime}, mult={multiplier})"
            )

            # Calculate initial floor = entry × (1 - initial_stop_loss_pct/100)
            initial_floor = entry_price * (1 - self.initial_stop_loss_pct / 100)
            
            # Calculate hard stop = entry × (1 - hard_stop_pct/100)
            hard_stop = entry_price * (1 - self.hard_stop_pct / 100)
            
            position = TrailingPosition(
                symbol=symbol,
                entry_price=entry_price,
                entry_time=datetime.now(),
                qty=qty,
                current_price=entry_price,
                peak_price=entry_price,
                floor_price=initial_floor,
                hard_stop_price=hard_stop,
                current_trail_pct=trail_distance,
                unrealized_pnl_usd=0.0,
                unrealized_pnl_pct=0.0,
                max_profit_seen_pct=0.0,
                hold_minutes=0.0,
                ladder_count=0,
                last_dip_buy_time=None,
                status="INIT",
                original_qty=qty,
                original_entry=entry_price,
            )
            
            with self._lock:
                self._positions[symbol] = position
            
            logger.info(
                f"📊 Trailing stop registered: {symbol} "
                f"entry=${entry_price:,.4f} floor=${initial_floor:,.4f} "
                f"hard_stop=${hard_stop:,.4f}"
            )
            
            return position
            
        except Exception as e:
            logger.error(f"Error registering position {symbol}: {e}")
            return None

    def update_floor_price(self, symbol: str, new_floor: float) -> bool:
        """Update tracked floor price only when new floor is higher."""
        try:
            with self._lock:
                if symbol not in self._positions:
                    logger.debug(f"Floor update skipped for {symbol}: not tracked")
                    return False

                pos = self._positions[symbol]
                current_floor = pos.floor_price
                if new_floor <= current_floor:
                    logger.info(
                        f"Floor update skipped for {symbol}: "
                        f"new={new_floor:.4f} <= current={current_floor:.4f}"
                    )
                    return False

                pos.floor_price = float(new_floor)
                logger.info(
                    f"Floor updated for {symbol}: {current_floor:.4f} -> {pos.floor_price:.4f}"
                )
                return True
        except Exception as e:
            logger.error(f"Error updating floor for {symbol}: {e}")
            return False
    
    def update_position(self, symbol: str, current_price: float) -> TrailingAction:
        """
        Update position with current price and check exit conditions.
        
        This is the core trailing stop logic:
        1. Update current_price and calculate P&L
        2. If price > peak: update peak and potentially raise floor
        3. Apply profit tier tightening
        4. Check all exit conditions in priority order
        
        Args:
            symbol: Trading pair
            current_price: Current market price
            
        Returns:
            TrailingAction with action=HOLD or action=SELL and reason
        """
        # Default action
        default_action = TrailingAction(
            action="HOLD",
            reason="NO_POSITION",
            urgency="NONE",
            floor_price=0.0,
            current_trail_pct=0.0,
        )
        
        if not self.enabled:
            return default_action
        
        try:
            with self._lock:
                if symbol not in self._positions:
                    return default_action
                
                pos = self._positions[symbol]
                
                # Step 1: Update current price and calculate P&L
                pos.current_price = current_price
                pos.unrealized_pnl_usd = (current_price - pos.entry_price) * pos.qty
                pos.unrealized_pnl_pct = ((current_price - pos.entry_price) / pos.entry_price) * 100
                
                # Update hold time
                pos.hold_minutes = (datetime.now() - pos.entry_time).total_seconds() / 60
                
                # Step 2: Check if price made new peak
                if current_price > pos.peak_price:
                    pos.peak_price = current_price
                    
                    # Track max profit seen
                    if pos.unrealized_pnl_pct > pos.max_profit_seen_pct:
                        pos.max_profit_seen_pct = pos.unrealized_pnl_pct
                    
                    # Check if trailing is activated (price rose enough from entry)
                    price_gain_pct = ((current_price - pos.entry_price) / pos.entry_price) * 100
                    if price_gain_pct >= self.trail_activation_pct:
                        # Calculate new floor based on peak
                        new_floor = pos.peak_price * (1 - pos.current_trail_pct / 100)
                        
                        # CRITICAL: Floor ONLY goes UP, never down
                        if new_floor > pos.floor_price:
                            old_floor = pos.floor_price
                            pos.floor_price = new_floor
                            
                            # Update status
                            if pos.floor_price > pos.entry_price:
                                pos.status = "PROFIT_LOCK"
                            else:
                                pos.status = "TRAILING"
                            
                            if self.log_trail_updates:
                                logger.info(
                                    f"📈 Trail raised {symbol}: "
                                    f"floor ${old_floor:,.4f} → ${new_floor:,.4f} "
                                    f"(peak=${pos.peak_price:,.4f}, trail={pos.current_trail_pct}%)"
                                )
                
                # Step 3: Apply profit tier tightening
                for tier in self.profit_tiers:
                    tier_profit = tier.get("profit_pct", 999)
                    tier_trail = tier.get("tighten_trail_to_pct", pos.current_trail_pct)
                    
                    if pos.unrealized_pnl_pct >= tier_profit:
                        if tier_trail < pos.current_trail_pct:
                            old_trail = pos.current_trail_pct
                            pos.current_trail_pct = tier_trail
                            
                            # Recalculate floor with tighter trail
                            new_floor = pos.peak_price * (1 - pos.current_trail_pct / 100)
                            if new_floor > pos.floor_price:
                                pos.floor_price = new_floor
                            
                            logger.info(
                                f"🔒 Trail tightened {symbol}: "
                                f"{old_trail}% → {tier_trail}% at {pos.unrealized_pnl_pct:.2f}% profit"
                            )
                
                # Update status based on conditions
                floor_gap_pct = ((current_price - pos.floor_price) / pos.floor_price) * 100
                hard_gap_pct = ((current_price - pos.hard_stop_price) / pos.hard_stop_price) * 100
                
                if hard_gap_pct < 0.3:
                    pos.status = "HARD"
                elif floor_gap_pct < 0.5:
                    pos.status = "WATCH"
                elif pos.unrealized_pnl_pct >= self.quick_profit_pct * 0.8:
                    pos.status = "PROFIT"
                
                # Step 4: Check exit conditions in priority order
                
                # a) HARD STOP: absolute emergency floor
                if current_price <= pos.hard_stop_price:
                    if self.alert_on_floor_hit:
                        logger.warning(
                            f"🛑 HARD STOP HIT {symbol}: "
                            f"price=${current_price:,.4f} <= hard_stop=${pos.hard_stop_price:,.4f}"
                        )
                    return TrailingAction(
                        action="SELL",
                        reason="HARD_STOP_HIT",
                        urgency="CRITICAL",
                        floor_price=pos.floor_price,
                        current_trail_pct=pos.current_trail_pct,
                        pnl_usd=pos.unrealized_pnl_usd,
                        pnl_pct=pos.unrealized_pnl_pct,
                    )
                
                # b) TRAIL STOP HIT: price dropped below trailing floor
                if current_price <= pos.floor_price:
                    if self.alert_on_floor_hit:
                        logger.warning(
                            f"🚨 TRAIL STOP HIT {symbol}: "
                            f"price=${current_price:,.4f} <= floor=${pos.floor_price:,.4f} "
                            f"P&L=${pos.unrealized_pnl_usd:,.2f}"
                        )
                    return TrailingAction(
                        action="SELL",
                        reason="TRAIL_STOP_HIT",
                        urgency="HIGH",
                        floor_price=pos.floor_price,
                        current_trail_pct=pos.current_trail_pct,
                        pnl_usd=pos.unrealized_pnl_usd,
                        pnl_pct=pos.unrealized_pnl_pct,
                    )
                
                # c) QUICK PROFIT: take profit immediately at target
                if pos.unrealized_pnl_pct >= self.quick_profit_pct:
                    logger.info(
                        f"💰 QUICK PROFIT TARGET {symbol}: "
                        f"{pos.unrealized_pnl_pct:.2f}% >= {self.quick_profit_pct}% "
                        f"P&L=${pos.unrealized_pnl_usd:,.2f}"
                    )
                    return TrailingAction(
                        action="SELL",
                        reason="QUICK_PROFIT_TARGET",
                        urgency="MEDIUM",
                        floor_price=pos.floor_price,
                        current_trail_pct=pos.current_trail_pct,
                        pnl_usd=pos.unrealized_pnl_usd,
                        pnl_pct=pos.unrealized_pnl_pct,
                    )
                
                # d) MIN PROFIT USD: take profit if minimum dollar profit reached
                if (pos.unrealized_pnl_usd >= self.min_profit_to_exit_usd and 
                    pos.unrealized_pnl_pct > 0):
                    logger.info(
                        f"💵 MIN PROFIT HIT {symbol}: "
                        f"${pos.unrealized_pnl_usd:,.2f} >= ${self.min_profit_to_exit_usd}"
                    )
                    return TrailingAction(
                        action="SELL",
                        reason="MIN_PROFIT_HIT",
                        urgency="MEDIUM",
                        floor_price=pos.floor_price,
                        current_trail_pct=pos.current_trail_pct,
                        pnl_usd=pos.unrealized_pnl_usd,
                        pnl_pct=pos.unrealized_pnl_pct,
                    )
                
                # e) TIME LIMIT: force exit if held too long
                if pos.hold_minutes >= self.max_hold_minutes:
                    logger.info(
                        f"⏰ MAX HOLD TIME {symbol}: "
                        f"{pos.hold_minutes:.0f}min >= {self.max_hold_minutes}min"
                    )
                    return TrailingAction(
                        action="SELL",
                        reason="MAX_HOLD_TIME",
                        urgency="LOW",
                        floor_price=pos.floor_price,
                        current_trail_pct=pos.current_trail_pct,
                        pnl_usd=pos.unrealized_pnl_usd,
                        pnl_pct=pos.unrealized_pnl_pct,
                    )
                
                # f) No exit condition met - continue trailing
                return TrailingAction(
                    action="HOLD",
                    reason="TRAILING",
                    urgency="NONE",
                    floor_price=pos.floor_price,
                    current_trail_pct=pos.current_trail_pct,
                    pnl_usd=pos.unrealized_pnl_usd,
                    pnl_pct=pos.unrealized_pnl_pct,
                )
                
        except Exception as e:
            logger.error(f"Error updating position {symbol}: {e}")
            return default_action
    
    def check_dca_opportunity(
        self, 
        symbol: str, 
        current_price: float
    ) -> Optional[DCAAction]:
        """
        Check if price has dropped enough to trigger a ladder-in (DCA).
        
        Conditions (ALL must be true):
        1. ladder_in.enabled == True
        2. ladder_count < max_ladder_count
        3. price dropped >= dip_threshold_pct% from ENTRY price
        4. Time since last_dip_buy > min_time_between_dips_minutes
        
        Args:
            symbol: Trading pair
            current_price: Current market price
            
        Returns:
            DCAAction if conditions met, None otherwise
        """
        if not self.ladder_enabled:
            return None
        
        try:
            with self._lock:
                if symbol not in self._positions:
                    return None
                
                pos = self._positions[symbol]
                
                # Check ladder count limit
                if pos.ladder_count >= self.max_ladder_count:
                    return None
                
                # Calculate drop from original entry
                drop_pct = ((pos.original_entry - current_price) / pos.original_entry) * 100
                
                # Check if dropped enough
                if drop_pct < self.dip_threshold_pct:
                    return None
                
                # Check time since last DCA
                if pos.last_dip_buy_time:
                    time_since_last = (datetime.now() - pos.last_dip_buy_time).total_seconds() / 60
                    if time_since_last < self.min_time_between_dips:
                        return None
                
                # Calculate additional quantity
                additional_qty = pos.original_qty * (self.additional_size_pct / 100)
                
                logger.info(
                    f"📉 DCA OPPORTUNITY {symbol}: "
                    f"price dropped {drop_pct:.2f}% from entry "
                    f"(current=${current_price:,.4f}, entry=${pos.original_entry:,.4f})"
                )
                
                return DCAAction(
                    symbol=symbol,
                    additional_qty=additional_qty,
                    reason="DIP_BUY_TRIGGERED",
                    dip_pct_from_entry=drop_pct,
                    current_price=current_price,
                )
                
        except Exception as e:
            logger.error(f"Error checking DCA for {symbol}: {e}")
            return None
    
    def add_dca_fill(
        self, 
        symbol: str, 
        additional_qty: float, 
        fill_price: float
    ) -> bool:
        """
        Record a DCA order fill and recalculate position parameters.
        
        - Recalculates weighted average entry price
        - Updates total quantity
        - Recalculates hard_stop based on new entry
        - Does NOT reset floor if it's above new entry
        - Increments ladder_count
        
        Args:
            symbol: Trading pair
            additional_qty: Quantity added
            fill_price: Fill price of DCA order
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self._lock:
                if symbol not in self._positions:
                    logger.warning(f"DCA fill for unknown position: {symbol}")
                    return False
                
                pos = self._positions[symbol]
                
                # Calculate weighted average entry
                total_cost = (pos.qty * pos.entry_price) + (additional_qty * fill_price)
                new_qty = pos.qty + additional_qty
                new_entry = total_cost / new_qty
                
                old_entry = pos.entry_price
                old_qty = pos.qty
                
                # Update position
                pos.entry_price = new_entry
                pos.qty = new_qty
                pos.ladder_count += 1
                pos.last_dip_buy_time = datetime.now()
                
                # Recalculate hard_stop based on new entry
                new_hard_stop = new_entry * (1 - self.hard_stop_pct / 100)
                pos.hard_stop_price = new_hard_stop
                
                # Recalculate initial floor based on new entry
                new_initial_floor = new_entry * (1 - self.initial_stop_loss_pct / 100)
                
                # IMPORTANT: Only raise floor, never lower it
                if new_initial_floor > pos.floor_price:
                    pos.floor_price = new_initial_floor
                
                logger.info(
                    f"📊 DCA FILL {symbol}: "
                    f"+{additional_qty:.6f} @ ${fill_price:,.4f} "
                    f"| Entry: ${old_entry:,.4f} → ${new_entry:,.4f} "
                    f"| Qty: {old_qty:.6f} → {new_qty:.6f} "
                    f"| DCA #{pos.ladder_count}/{self.max_ladder_count}"
                )
                
                return True
                
        except Exception as e:
            logger.error(f"Error recording DCA fill for {symbol}: {e}")
            return False

    def check_partial_exits(
        self, symbol: str, current_price: float, position_qty: float
    ) -> List[Dict[str, Any]]:
        """
        Returns list of partial exit orders to execute.
        Each dict: {"qty_fraction": float, "reason": str}

        Ladder:
          Tier 1: Exit 33% at +1.0% profit
          Tier 2: Exit 33% at +2.5% profit
          Tier 3: Remaining position handled by trailing stop.
        """
        if position_qty <= 0:
            return []
        if not getattr(self.config.risk, "partial_exit_enabled", True):
            return []
        pos = self.get_position(symbol)
        if not pos or pos.entry_price <= 0:
            return []

        exit_orders: List[Dict[str, Any]] = []
        profit_pct = (current_price - pos.entry_price) / pos.entry_price * 100

        tier1_pct = getattr(self.config.risk, "partial_exit_tier1_pct", 1.0)
        tier2_pct = getattr(self.config.risk, "partial_exit_tier2_pct", 2.5)
        tier1_qty_fraction = getattr(self.config.risk, "partial_exit_tier1_fraction", 0.33)
        tier2_qty_fraction = getattr(self.config.risk, "partial_exit_tier2_fraction", 0.33)

        with self._lock:
            tracked_pos = self._positions.get(symbol)
            if tracked_pos and profit_pct >= tier1_pct and not tracked_pos.tier1_taken:
                tracked_pos.tier1_taken = True
                exit_orders.append(
                    {"qty_fraction": tier1_qty_fraction, "reason": f"Tier1 +{profit_pct:.2f}%"}
                )

            if tracked_pos and profit_pct >= tier2_pct and not tracked_pos.tier2_taken:
                tracked_pos.tier2_taken = True
                exit_orders.append(
                    {"qty_fraction": tier2_qty_fraction, "reason": f"Tier2 +{profit_pct:.2f}%"}
                )

        return exit_orders
    
    def remove_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Remove a position from tracking (called when position is closed).
        
        Args:
            symbol: Trading pair
            
        Returns:
            Final position stats dict, or None if not found
        """
        try:
            with self._lock:
                if symbol not in self._positions:
                    logger.debug(f"remove_position: {symbol} not in tracking")
                    return None
                
                pos = self._positions.pop(symbol)
                
                final_stats = {
                    "symbol": symbol,
                    "entry_price": pos.entry_price,
                    "exit_price": pos.current_price,
                    "qty": pos.qty,
                    "pnl_usd": pos.unrealized_pnl_usd,
                    "pnl_pct": pos.unrealized_pnl_pct,
                    "peak_price": pos.peak_price,
                    "max_profit_pct": pos.max_profit_seen_pct,
                    "hold_minutes": pos.hold_minutes,
                    "ladder_count": pos.ladder_count,
                    "final_floor": pos.floor_price,
                    "final_trail_pct": pos.current_trail_pct,
                }
                
                logger.info(
                    f"📋 Position removed {symbol}: "
                    f"P&L=${pos.unrealized_pnl_usd:,.2f} ({pos.unrealized_pnl_pct:+.2f}%) "
                    f"held {pos.hold_minutes:.0f}min, peak=${pos.peak_price:,.4f}"
                )
                
                return final_stats
                
        except Exception as e:
            logger.error(f"Error removing position {symbol}: {e}")
            return None
    
    def get_position(self, symbol: str) -> Optional[TrailingPosition]:
        """Get a single position by symbol (thread-safe copy)."""
        with self._lock:
            if symbol in self._positions:
                pos = self._positions[symbol]
                # Return a copy to avoid race conditions
                return TrailingPosition(
                    symbol=pos.symbol,
                    entry_price=pos.entry_price,
                    entry_time=pos.entry_time,
                    qty=pos.qty,
                    current_price=pos.current_price,
                    peak_price=pos.peak_price,
                    floor_price=pos.floor_price,
                    hard_stop_price=pos.hard_stop_price,
                    current_trail_pct=pos.current_trail_pct,
                    unrealized_pnl_usd=pos.unrealized_pnl_usd,
                    unrealized_pnl_pct=pos.unrealized_pnl_pct,
                    max_profit_seen_pct=pos.max_profit_seen_pct,
                    hold_minutes=pos.hold_minutes,
                    ladder_count=pos.ladder_count,
                    last_dip_buy_time=pos.last_dip_buy_time,
                    status=pos.status,
                    original_qty=pos.original_qty,
                    original_entry=pos.original_entry,
                    tier1_taken=pos.tier1_taken,
                    tier2_taken=pos.tier2_taken,
                )
            return None
    
    def get_all_positions(self) -> List[TrailingPosition]:
        """Get all tracked positions (thread-safe copies)."""
        result = []
        with self._lock:
            for symbol, pos in self._positions.items():
                result.append(TrailingPosition(
                    symbol=pos.symbol,
                    entry_price=pos.entry_price,
                    entry_time=pos.entry_time,
                    qty=pos.qty,
                    current_price=pos.current_price,
                    peak_price=pos.peak_price,
                    floor_price=pos.floor_price,
                    hard_stop_price=pos.hard_stop_price,
                    current_trail_pct=pos.current_trail_pct,
                    unrealized_pnl_usd=pos.unrealized_pnl_usd,
                    unrealized_pnl_pct=pos.unrealized_pnl_pct,
                    max_profit_seen_pct=pos.max_profit_seen_pct,
                    hold_minutes=pos.hold_minutes,
                    ladder_count=pos.ladder_count,
                    last_dip_buy_time=pos.last_dip_buy_time,
                    status=pos.status,
                    original_qty=pos.original_qty,
                    original_entry=pos.original_entry,
                    tier1_taken=pos.tier1_taken,
                    tier2_taken=pos.tier2_taken,
                ))
        return result
    
    def get_position_summary(self) -> Dict[str, Any]:
        """Get summary of all positions for dashboard display."""
        with self._lock:
            positions_data = []
            for symbol, pos in self._positions.items():
                positions_data.append({
                    "symbol": symbol,
                    "entry_price": pos.entry_price,
                    "current_price": pos.current_price,
                    "floor_price": pos.floor_price,
                    "hard_stop_price": pos.hard_stop_price,
                    "peak_price": pos.peak_price,
                    "trail_pct": pos.current_trail_pct,
                    "pnl_usd": pos.unrealized_pnl_usd,
                    "pnl_pct": pos.unrealized_pnl_pct,
                    "max_profit_pct": pos.max_profit_seen_pct,
                    "hold_minutes": pos.hold_minutes,
                    "ladder_count": pos.ladder_count,
                    "max_ladder": self.max_ladder_count,
                    "status": pos.status,
                    "qty": pos.qty,
                })
            
            # Calculate next DCA trigger prices
            for p in positions_data:
                next_dca_price = p["entry_price"] * (1 - self.dip_threshold_pct / 100)
                p["next_dca_price"] = next_dca_price
            
            return {
                "enabled": self.enabled,
                "positions": positions_data,
                "config": {
                    "initial_stop_pct": self.initial_stop_loss_pct,
                    "trail_pct": self.trail_pct,
                    "hard_stop_pct": self.hard_stop_pct,
                    "quick_profit_pct": self.quick_profit_pct,
                    "max_hold_minutes": self.max_hold_minutes,
                    "dca_enabled": self.ladder_enabled,
                    "dca_threshold_pct": self.dip_threshold_pct,
                    "max_ladder_count": self.max_ladder_count,
                },
            }
    
    def has_position(self, symbol: str) -> bool:
        """Check if a symbol is being tracked."""
        with self._lock:
            return symbol in self._positions
    
    def position_count(self) -> int:
        """Get number of positions being tracked."""
        with self._lock:
            return len(self._positions)


# Module-level singleton instance
_trailing_manager: Optional[TrailingStopManager] = None


def get_trailing_manager(config: Optional[ConfigLoader] = None) -> TrailingStopManager:
    """Get the global trailing stop manager instance."""
    global _trailing_manager
    if _trailing_manager is None:
        _trailing_manager = TrailingStopManager(config)
    return _trailing_manager
