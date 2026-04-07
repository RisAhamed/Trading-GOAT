# core/position_monitor.py
"""
Position Monitor - Background thread for trailing stop management.

Runs independently of the main trading loop to check positions more frequently
and execute trailing stop logic in real-time.
"""

import logging
import threading
import time
from datetime import datetime
from typing import Optional

from .config_loader import get_config, ConfigLoader
from .trailing_stop_manager import TrailingStopManager
from .market_data import MarketDataFetcher


logger = logging.getLogger(__name__)


class PositionMonitor:
    """
    Background thread that monitors open positions and enforces trailing stops.

    Runs at a faster interval than the main loop (default: every 10 seconds)
    to ensure timely execution of trailing stops and DCA opportunities.
    """

    def __init__(
        self,
        trailing_manager: TrailingStopManager,
        order_executor: 'OrderExecutor',  # Forward reference to avoid circular import
        market_data: MarketDataFetcher,
        config: Optional[ConfigLoader] = None,
    ) -> None:
        """
        Initialize the position monitor.

        Args:
            trailing_manager: TrailingStopManager instance
            order_executor: OrderExecutor instance for closing positions
            market_data: MarketDataFetcher for getting current prices
            config: Optional config loader
        """
        self.config = config or get_config()
        self.trailing_manager = trailing_manager
        self.order_executor = order_executor
        self.market_data = market_data

        # Load position_monitor config
        monitor_config = self.config._raw_config.get("position_monitor", {})
        self.check_interval = monitor_config.get("check_interval_seconds", 10)
        self.price_update_method = monitor_config.get("price_update_method", "alpaca")
        self.trail_update_on_every_tick = monitor_config.get("trail_update_on_every_tick", True)

        # Load ladder_in config for DCA
        ladder_config = self.config._raw_config.get("ladder_in", {})
        self.dca_enabled = ladder_config.get("enabled", True)
        self.additional_size_pct = ladder_config.get("additional_size_pct", 50.0)

        # Thread control
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False

        logger.info(f"PositionMonitor initialized (interval={self.check_interval}s)")

    def start(self) -> None:
        """Start the position monitor background thread."""
        if self._running:
            logger.warning("Position monitor already running")
            return

        if not self.trailing_manager.is_enabled():
            logger.info("Trailing stop disabled, position monitor not started")
            return

        self._stop_event.clear()
        self._running = True

        self._thread = threading.Thread(
            target=self._monitor_loop,
            name="PositionMonitor",
            daemon=True,
        )
        self._thread.start()

        logger.info(f"✓ Position monitor started (checking every {self.check_interval}s)")

    def stop(self) -> None:
        """Stop the position monitor thread gracefully."""
        if not self._running:
            return

        logger.info("Stopping position monitor...")
        self._stop_event.set()
        self._running = False

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)

        logger.info("✓ Position monitor stopped")

    def _monitor_loop(self) -> None:
        """
        Main monitoring loop - runs until stop() is called.

        Never crashes - catches all exceptions and continues running.
        """
        logger.info("Position monitor loop started")

        while not self._stop_event.is_set():
            try:
                self._check_all_positions()
            except Exception as e:
                logger.error(f"Error in position monitor loop: {e}")
                # Continue running despite errors

            # Sleep for check_interval, but wake up if stop is signaled
            self._stop_event.wait(timeout=self.check_interval)

        logger.info("Position monitor loop exited")

    def _check_all_positions(self) -> None:
        """
        Check all open positions for trailing stops and DCA opportunities.

        This is called on every monitoring cycle.
        """
        # Get all tracked positions
        positions = self.trailing_manager.get_all_positions()

        if not positions:
            return  # No positions to monitor

        for pos in positions:
            try:
                self._check_single_position(pos.symbol)
            except Exception as e:
                logger.error(f"Error checking position {pos.symbol}: {e}")
                # Continue with other positions

    def _check_single_position(self, symbol: str) -> None:
        """
        Check a single position for exit signals and DCA opportunities.

        Args:
            symbol: Trading pair to check
        """
        # 1. Fetch current price
        current_price = self._get_current_price(symbol)
        if current_price is None or current_price <= 0:
            logger.warning(f"Failed to get valid price for {symbol}, skipping check")
            return

        # 2. Update trailing position and check for exit
        action = self.trailing_manager.update_position(symbol, current_price)

        if action.action == "SELL":
            # Exit condition triggered
            self._execute_exit(symbol, action)
            return  # Position closed, no need to check DCA

        # 3. Check for DCA opportunity (only if position still open)
        if self.dca_enabled:
            dca_action = self.trailing_manager.check_dca_opportunity(symbol, current_price)
            if dca_action:
                self._execute_dca(dca_action)

    def _get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get current market price for a symbol.

        Args:
            symbol: Trading pair

        Returns:
            Current price, or None if unavailable
        """
        try:
            price = self.market_data.get_current_price(symbol)
            if price and price > 0:
                return price

            logger.debug(f"get_current_price returned invalid price for {symbol}: {price}")
            return None

        except Exception as e:
            logger.error(f"Error fetching price for {symbol}: {e}")
            return None

    def _execute_exit(self, symbol: str, action) -> None:
        """
        Execute an exit order triggered by trailing stop.

        Args:
            symbol: Trading pair to close
            action: TrailingAction with exit reason and details
        """
        try:
            logger.warning(
                f"🚨 TRAILING STOP EXIT: {symbol} | "
                f"Reason: {action.reason} | "
                f"Price: ${action.floor_price:,.4f} | "
                f"P&L: ${action.pnl_usd:+,.2f} ({action.pnl_pct:+.2f}%) | "
                f"Urgency: {action.urgency}"
            )

            # Close the position via order executor
            result = self.order_executor.close_position(symbol, reason=action.reason)

            if result.success:
                logger.info(f"✅ Position closed successfully: {symbol}")

                # Remove from trailing manager
                final_stats = self.trailing_manager.remove_position(symbol)

                if final_stats:
                    logger.info(
                        f"📊 Final stats for {symbol}: "
                        f"P&L=${final_stats['pnl_usd']:,.2f} "
                        f"({final_stats['pnl_pct']:+.2f}%), "
                        f"held {final_stats['hold_minutes']:.0f}min, "
                        f"peak=${final_stats['peak_price']:,.4f}, "
                        f"DCA count={final_stats['ladder_count']}"
                    )
            else:
                logger.error(
                    f"❌ Failed to close position {symbol}: {result.error_message}"
                )
                # Don't remove from trailing manager if close failed

        except Exception as e:
            logger.error(f"Error executing exit for {symbol}: {e}")

    def _execute_dca(self, dca_action) -> None:
        """
        Execute a DCA (ladder-in) buy order.

        Args:
            dca_action: DCAAction with details of the dip buy
        """
        try:
            symbol = dca_action.symbol
            additional_qty = dca_action.additional_qty
            current_price = dca_action.current_price

            logger.info(
                f"📉 DCA BUY TRIGGERED: {symbol} | "
                f"Dip: {dca_action.dip_pct_from_entry:.2f}% | "
                f"Price: ${current_price:,.4f} | "
                f"Qty: +{additional_qty:.6f}"
            )

            # Calculate position value for this DCA buy
            position_value = additional_qty * current_price

            # Create a minimal RiskParameters for the DCA order
            # We'll use a simplified approach since this is a follow-on buy
            from .risk_manager import RiskParameters

            # For DCA, we don't use stop loss/take profit - trailing manager handles it
            dca_risk_params = RiskParameters(
                qty=additional_qty,
                position_value=position_value,
                stop_price=0.0,  # No separate stop for DCA
                take_profit_price=0.0,  # No separate TP for DCA
                stop_loss_distance=0.0,
                take_profit_distance=0.0,
                max_loss_usd=0.0,
                risk_reward_ratio=0.0,
                is_allowed=True,
                rejection_reason="",
                entry_price=current_price,
                symbol=symbol,
                side="long",
            )

            # Execute buy order
            result = self.order_executor.execute_buy(symbol, dca_risk_params)

            if result.success:
                # Record the DCA fill in trailing manager
                fill_price = result.filled_price or current_price
                success = self.trailing_manager.add_dca_fill(
                    symbol=symbol,
                    additional_qty=additional_qty,
                    fill_price=fill_price,
                )

                if success:
                    logger.info(
                        f"✅ DCA order filled: {symbol} | "
                        f"Qty: +{additional_qty:.6f} @ ${fill_price:,.4f}"
                    )
                else:
                    logger.error(f"Failed to record DCA fill for {symbol}")
            else:
                logger.error(
                    f"❌ DCA order failed for {symbol}: {result.error_message}"
                )

        except Exception as e:
            logger.error(f"Error executing DCA for {dca_action.symbol}: {e}")

    def is_running(self) -> bool:
        """Check if the monitor is currently running."""
        return self._running

    def get_status(self) -> dict:
        """
        Get current monitor status.

        Returns:
            Dict with status information
        """
        return {
            "running": self._running,
            "check_interval": self.check_interval,
            "positions_tracked": self.trailing_manager.position_count(),
            "dca_enabled": self.dca_enabled,
            "trailing_enabled": self.trailing_manager.is_enabled(),
        }


# Module-level singleton
_position_monitor: Optional[PositionMonitor] = None


def get_position_monitor(
    trailing_manager: Optional[TrailingStopManager] = None,
    order_executor = None,
    market_data: Optional[MarketDataFetcher] = None,
    config: Optional[ConfigLoader] = None,
) -> Optional[PositionMonitor]:
    """
    Get the global position monitor instance.

    Must be initialized with dependencies before first use.
    """
    global _position_monitor

    if _position_monitor is None and all([trailing_manager, order_executor, market_data]):
        _position_monitor = PositionMonitor(
            trailing_manager=trailing_manager,
            order_executor=order_executor,
            market_data=market_data,
            config=config,
        )

    return _position_monitor
