# core/order_executor.py
"""
Order executor module for placing trades on Alpaca.
Handles market orders, stop losses, and take profit orders.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    LimitOrderRequest,
    StopOrderRequest,
    StopLimitOrderRequest,
    ClosePositionRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType, OrderClass
from alpaca.common.exceptions import APIError

from .config_loader import get_config, ConfigLoader
from .risk_manager import RiskParameters
from .trade_results import get_results_tracker


logger = logging.getLogger(__name__)


@dataclass
class OrderResult:
    """Container for order execution result."""
    success: bool
    order_id: Optional[str]
    symbol: str
    side: str
    qty: float
    filled_price: Optional[float]
    status: str
    error_message: str
    timestamp: str
    
    # Stop loss and take profit order IDs
    stop_loss_order_id: Optional[str] = None
    take_profit_order_id: Optional[str] = None


class OrderExecutor:
    """
    Executes trades on Alpaca paper trading API.
    Handles market orders, stop losses, and take profit orders.
    """
    
    def __init__(self, config: Optional[ConfigLoader] = None) -> None:
        """Initialize the order executor."""
        self.config = config or get_config()
        
        # Initialize Alpaca trading client
        # IMPORTANT: Always use paper=True for paper trading
        self.client = TradingClient(
            api_key=self.config.env.alpaca_api_key,
            secret_key=self.config.env.alpaca_api_secret,
            paper=True,  # NEVER set to False
        )
        
        # Order history
        self._recent_orders: List[OrderResult] = []
        self._max_history = 50
        self.portfolio_tracker = None
        
        logger.info("OrderExecutor initialized (PAPER TRADING)")

    def calculate_dynamic_position_size(
        self,
        symbol: str,
        current_price: float,
        atr_pct: float,
        portfolio_cash: float,
        intel_modifier: int = 0,
    ) -> float:
        """
        ATR-normalized position sizing.
        Higher volatility = smaller position = same dollar risk per trade.
        """
        if current_price <= 0 or portfolio_cash <= 0:
            return 0.0

        base_risk_pct = getattr(self.config.risk, "base_risk_pct", 0.02)
        max_position_pct = getattr(self.config.risk, "max_position_pct", 0.10)
        atr_mult = getattr(self.config.risk, "atr_stop_multiplier", 1.5)
        strong_intel_threshold = getattr(self.config.risk, "intel_positive_threshold", 20)
        weak_intel_threshold = getattr(self.config.risk, "intel_negative_threshold", -10)
        strong_intel_size_multiplier = getattr(self.config.risk, "intel_positive_size_multiplier", 1.25)
        weak_intel_size_multiplier = getattr(self.config.risk, "intel_negative_size_multiplier", 0.5)

        dollar_risk = portfolio_cash * base_risk_pct

        atr_pct_value = float(atr_pct or 0.0)
        # Standardized input: atr_pct is numeric percent points from indicators/scanner
        # (example: 0.8 means 0.8%, and is converted to 0.008 for calculations).
        atr_pct_frac = max(0.0, atr_pct_value / 100.0)
        stop_distance_pct = max(atr_pct_frac * atr_mult, 0.005)

        raw_size = dollar_risk / (current_price * stop_distance_pct)

        if intel_modifier >= strong_intel_threshold:
            raw_size *= strong_intel_size_multiplier
        elif intel_modifier <= weak_intel_threshold:
            raw_size *= weak_intel_size_multiplier

        max_size = (portfolio_cash * max_position_pct) / current_price
        final_size = min(raw_size, max_size)

        logger.info(
            f"[SIZING] {symbol}: ATR={atr_pct_frac*100:.2f}%, "
            f"StopDist={stop_distance_pct*100:.2f}%, "
            f"Size={final_size:.4f} (risk=${dollar_risk:.0f})"
        )
        return round(max(final_size, 0.0), 4)
    
    def _convert_symbol(self, symbol: str) -> str:
        """Convert symbol format for Alpaca API."""
        # Crypto: BTC/USD -> BTCUSD, or BTC/USD -> BTC/USD (Alpaca accepts both)
        # For crypto, Alpaca prefers the slash format
        if "/" in symbol:
            # Check if it's crypto (common crypto pairs)
            crypto_bases = ["BTC", "ETH", "SOL", "AVAX", "DOGE", "LTC", "XRP", "ADA"]
            base = symbol.split("/")[0]
            if base in crypto_bases:
                return symbol  # Keep slash for crypto
            else:
                # Forex - remove slash
                return symbol.replace("/", "")
        return symbol
    
    def check_connection(self) -> tuple[bool, str, Dict[str, Any]]:
        """
        Check connection to Alpaca and get account info.
        
        Returns:
            Tuple of (is_connected, message, account_info)
        """
        try:
            account = self.client.get_account()
            
            account_info = {
                "account_number": account.account_number,
                "status": account.status,
                "cash": float(account.cash),
                "portfolio_value": float(account.portfolio_value),
                "buying_power": float(account.buying_power),
                "equity": float(account.equity),
                "currency": account.currency,
                "trading_blocked": account.trading_blocked,
                "pattern_day_trader": account.pattern_day_trader,
            }
            
            if account.trading_blocked:
                return False, "Trading is blocked on this account", account_info
            
            return True, f"Connected to Alpaca (Account: {account.account_number})", account_info
            
        except APIError as e:
            return False, f"Alpaca API error: {str(e)}", {}
        except Exception as e:
            return False, f"Connection error: {str(e)}", {}
    
    def get_account(self) -> Optional[Dict[str, Any]]:
        """Get current account information."""
        try:
            account = self.client.get_account()
            return {
                "cash": float(account.cash),
                "portfolio_value": float(account.portfolio_value),
                "buying_power": float(account.buying_power),
                "equity": float(account.equity),
                "currency": account.currency,
            }
        except Exception as e:
            logger.error(f"Error getting account: {e}")
            return None
    
    def execute_buy(
        self,
        symbol: str,
        risk_params: RiskParameters,
    ) -> OrderResult:
        """
        Execute a BUY order with stop loss and take profit.
        
        Args:
            symbol: Trading pair
            risk_params: Risk parameters from risk manager
            
        Returns:
            OrderResult with order details
        """
        timestamp = datetime.now().isoformat()
        alpaca_symbol = self._convert_symbol(symbol)
        
        result = OrderResult(
            success=False,
            order_id=None,
            symbol=symbol,
            side="buy",
            qty=risk_params.qty,
            filled_price=None,
            status="pending",
            error_message="",
            timestamp=timestamp,
        )
        
        # Check if trade is allowed
        if not risk_params.is_allowed:
            result.error_message = risk_params.rejection_reason
            result.status = "rejected"
            logger.warning(f"BUY {symbol} rejected: {risk_params.rejection_reason}")
            return result
        
        try:
            # Place market order
            order_request = MarketOrderRequest(
                symbol=alpaca_symbol,
                qty=risk_params.qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.GTC,  # Good till cancelled
            )
            
            order = self.client.submit_order(order_request)
            
            result.order_id = str(order.id)
            result.status = str(order.status)
            
            if order.filled_avg_price:
                result.filled_price = float(order.filled_avg_price)
            
            logger.info(
                f"BUY order submitted: {symbol} qty={risk_params.qty:.6f} "
                f"order_id={order.id} status={order.status}"
            )
            
            # Place stop loss order
            try:
                sl_result = self._place_stop_loss(
                    alpaca_symbol,
                    risk_params.qty,
                    risk_params.stop_price,
                    "sell",  # Stop loss for long is a sell
                )
                if sl_result:
                    result.stop_loss_order_id = sl_result
                    logger.info(f"Stop loss placed at ${risk_params.stop_price:.4f}")
            except Exception as e:
                logger.error(f"Failed to place stop loss: {e}")
            
            # Place take profit order
            try:
                tp_result = self._place_take_profit(
                    alpaca_symbol,
                    risk_params.qty,
                    risk_params.take_profit_price,
                    "sell",  # Take profit for long is a sell
                )
                if tp_result:
                    result.take_profit_order_id = tp_result
                    logger.info(f"Take profit placed at ${risk_params.take_profit_price:.4f}")
            except Exception as e:
                logger.error(f"Failed to place take profit: {e}")
            
            result.success = True
            
            # Record trade entry in results tracker
            try:
                results_tracker = get_results_tracker()
                filled_price = result.filled_price or risk_params.entry_price
                results_tracker.record_entry(
                    symbol=symbol,
                    side="BUY",
                    quantity=risk_params.qty,
                    entry_price=filled_price,
                    order_type="market",
                    ai_confidence=getattr(risk_params, 'ai_confidence', None),
                    ai_reasoning=getattr(risk_params, 'ai_reasoning', None),
                    order_id=result.order_id,
                )
                logger.info(f"Trade entry recorded in results tracker")
            except Exception as e:
                logger.warning(f"Failed to record trade result: {e}")
            
        except APIError as e:
            result.error_message = self._handle_api_error(e)
            result.status = "error"
            logger.error(f"Alpaca API error on BUY {symbol}: {result.error_message}")
        
        except Exception as e:
            result.error_message = str(e)
            result.status = "error"
            logger.error(f"Error executing BUY {symbol}: {e}")
        
        # Store in history
        self._recent_orders.append(result)
        if len(self._recent_orders) > self._max_history:
            self._recent_orders.pop(0)
        
        return result
    
    def execute_sell(
        self,
        symbol: str,
        risk_params: RiskParameters,
    ) -> OrderResult:
        """
        Execute a SELL/SHORT order with stop loss and take profit.
        
        Args:
            symbol: Trading pair
            risk_params: Risk parameters from risk manager
            
        Returns:
            OrderResult with order details
        """
        timestamp = datetime.now().isoformat()
        alpaca_symbol = self._convert_symbol(symbol)
        
        result = OrderResult(
            success=False,
            order_id=None,
            symbol=symbol,
            side="sell",
            qty=risk_params.qty,
            filled_price=None,
            status="pending",
            error_message="",
            timestamp=timestamp,
        )
        
        # Check if trade is allowed
        if not risk_params.is_allowed:
            result.error_message = risk_params.rejection_reason
            result.status = "rejected"
            logger.warning(f"SELL {symbol} rejected: {risk_params.rejection_reason}")
            return result
        
        try:
            # Place market order
            order_request = MarketOrderRequest(
                symbol=alpaca_symbol,
                qty=risk_params.qty,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.GTC,
            )
            
            order = self.client.submit_order(order_request)
            
            result.order_id = str(order.id)
            result.status = str(order.status)
            
            if order.filled_avg_price:
                result.filled_price = float(order.filled_avg_price)
            
            logger.info(
                f"SELL order submitted: {symbol} qty={risk_params.qty:.6f} "
                f"order_id={order.id} status={order.status}"
            )
            
            # Place stop loss order (for short, stop loss is a buy)
            try:
                sl_result = self._place_stop_loss(
                    alpaca_symbol,
                    risk_params.qty,
                    risk_params.stop_price,
                    "buy",
                )
                if sl_result:
                    result.stop_loss_order_id = sl_result
                    logger.info(f"Stop loss placed at ${risk_params.stop_price:.4f}")
            except Exception as e:
                logger.error(f"Failed to place stop loss: {e}")
            
            # Place take profit order (for short, take profit is a buy)
            try:
                tp_result = self._place_take_profit(
                    alpaca_symbol,
                    risk_params.qty,
                    risk_params.take_profit_price,
                    "buy",
                )
                if tp_result:
                    result.take_profit_order_id = tp_result
                    logger.info(f"Take profit placed at ${risk_params.take_profit_price:.4f}")
            except Exception as e:
                logger.error(f"Failed to place take profit: {e}")
            
            result.success = True
            
            # Record trade entry in results tracker
            try:
                results_tracker = get_results_tracker()
                filled_price = result.filled_price or risk_params.entry_price
                results_tracker.record_entry(
                    symbol=symbol,
                    side="SELL",
                    quantity=risk_params.qty,
                    entry_price=filled_price,
                    order_type="market",
                    ai_confidence=getattr(risk_params, 'ai_confidence', None),
                    ai_reasoning=getattr(risk_params, 'ai_reasoning', None),
                    order_id=result.order_id,
                )
                logger.info(f"Trade entry recorded in results tracker")
            except Exception as e:
                logger.warning(f"Failed to record trade result: {e}")
            
        except APIError as e:
            result.error_message = self._handle_api_error(e)
            result.status = "error"
            logger.error(f"Alpaca API error on SELL {symbol}: {result.error_message}")
        
        except Exception as e:
            result.error_message = str(e)
            result.status = "error"
            logger.error(f"Error executing SELL {symbol}: {e}")
        
        # Store in history
        self._recent_orders.append(result)
        if len(self._recent_orders) > self._max_history:
            self._recent_orders.pop(0)
        
        return result
    
    def close_position(self, symbol: str, reason: str = "MANUAL") -> OrderResult:
        """
        Close an existing position.

        Args:
            symbol: Trading pair to close
            reason: Reason for closing (e.g., "TRAIL_STOP_HIT", "HARD_STOP", "MANUAL")

        Returns:
            OrderResult with close order details
        """
        timestamp = datetime.now().isoformat()
        # For close_position, Alpaca requires symbol WITHOUT slash
        alpaca_symbol = symbol.replace("/", "")
        
        result = OrderResult(
            success=False,
            order_id=None,
            symbol=symbol,
            side="close",
            qty=0,
            filled_price=None,
            status="pending",
            error_message="",
            timestamp=timestamp,
        )
        
        try:
            # Close the position
            order = self.client.close_position(alpaca_symbol)
            
            result.order_id = str(order.id) if hasattr(order, 'id') else None
            result.status = str(order.status) if hasattr(order, 'status') else "closed"
            
            if hasattr(order, 'qty'):
                result.qty = float(order.qty)
            
            if hasattr(order, 'filled_avg_price') and order.filled_avg_price:
                result.filled_price = float(order.filled_avg_price)
            
            result.success = True

            if "STOP" in str(reason).upper() and self.portfolio_tracker:
                try:
                    self.portfolio_tracker.record_stop_loss_exit(symbol)
                except Exception as e:
                    logger.warning(f"[COOLDOWN] Failed to record stop-loss exit for {symbol}: {e}")
            
            # Record trade exit in results tracker
            try:
                results_tracker = get_results_tracker()
                if result.filled_price:
                    results_tracker.record_exit(
                        symbol=symbol,
                        exit_price=result.filled_price,
                        quantity=result.qty if result.qty > 0 else None,
                        reason=reason,
                    )
                    logger.info(f"Trade exit recorded in results tracker")
            except Exception as e:
                logger.warning(f"Failed to record trade exit: {e}")
            
            logger.info(f"Position closed: {symbol} (reason: {reason})")
            
        except APIError as e:
            error_msg = self._handle_api_error(e)
            if "position does not exist" in error_msg.lower():
                result.error_message = "No position to close"
                result.status = "no_position"
            else:
                result.error_message = error_msg
                result.status = "error"
            logger.error(f"Error closing position {symbol}: {result.error_message}")
        
        except Exception as e:
            result.error_message = str(e)
            result.status = "error"
            logger.error(f"Error closing position {symbol}: {e}")
        
        # Store in history
        self._recent_orders.append(result)
        if len(self._recent_orders) > self._max_history:
            self._recent_orders.pop(0)
        
        return result

    def close_position_partial(
        self,
        symbol: str,
        exit_pct: float,
        reason: str = "PARTIAL_EXIT",
    ) -> OrderResult:
        """
        Close a fraction of an open position with a market order.
        """
        timestamp = datetime.now().isoformat()
        result = OrderResult(
            success=False,
            order_id=None,
            symbol=symbol,
            side="close_partial",
            qty=0.0,
            filled_price=None,
            status="pending",
            error_message="",
            timestamp=timestamp,
        )
        try:
            if exit_pct <= 0:
                result.error_message = "exit_pct must be > 0"
                result.status = "rejected"
                return result

            pct = min(1.0, max(0.0, float(exit_pct)))
            alpaca_symbol = symbol.replace("/", "")
            open_pos = self.client.get_open_position(alpaca_symbol)

            pos_qty_signed = float(getattr(open_pos, "qty", 0) or 0)
            pos_side = str(getattr(open_pos, "side", "long")).lower()
            pos_qty = abs(pos_qty_signed)
            if pos_qty <= 0:
                result.error_message = "No position quantity available"
                result.status = "no_position"
                return result

            close_qty = max(0.0, pos_qty * pct)
            if close_qty <= 0:
                result.error_message = "Computed close quantity is zero"
                result.status = "rejected"
                return result

            order_side = OrderSide.SELL if pos_side == "long" else OrderSide.BUY
            order_request = MarketOrderRequest(
                symbol=alpaca_symbol,
                qty=close_qty,
                side=order_side,
                time_in_force=TimeInForce.GTC,
            )
            order = self.client.submit_order(order_request)

            result.order_id = str(order.id) if hasattr(order, "id") else None
            result.status = str(order.status) if hasattr(order, "status") else "submitted"
            result.qty = float(close_qty)
            if hasattr(order, "filled_avg_price") and order.filled_avg_price:
                result.filled_price = float(order.filled_avg_price)
            result.success = True

            logger.info(
                f"Partial close submitted: {symbol} qty={close_qty:.6f} "
                f"({pct*100:.1f}%) reason={reason}"
            )
        except APIError as e:
            result.error_message = self._handle_api_error(e)
            result.status = "error"
            logger.error(f"Error partially closing {symbol}: {result.error_message}")
        except Exception as e:
            result.error_message = str(e)
            result.status = "error"
            logger.error(f"Error partially closing {symbol}: {e}")

        self._recent_orders.append(result)
        if len(self._recent_orders) > self._max_history:
            self._recent_orders.pop(0)
        return result

    def move_stop_to_breakeven(self, symbol: str, breakeven_price: float) -> bool:
        """
        Best-effort stop update: cancel open stop orders and place a new stop at breakeven.
        """
        try:
            if breakeven_price <= 0:
                return False

            alpaca_symbol = symbol.replace("/", "")
            open_pos = self.client.get_open_position(alpaca_symbol)
            pos_qty_signed = float(getattr(open_pos, "qty", 0) or 0)
            pos_side = str(getattr(open_pos, "side", "long")).lower()
            pos_qty = abs(pos_qty_signed)
            if pos_qty <= 0:
                return False

            try:
                orders = self.client.get_orders()
                for o in orders:
                    o_symbol = str(getattr(o, "symbol", ""))
                    o_type = str(getattr(o, "type", "")).lower()
                    o_status = str(getattr(o, "status", "")).lower()
                    if (
                        o_symbol.replace("/", "") == alpaca_symbol
                        and o_type == "stop"
                        and o_status in {"new", "accepted", "pending_new", "partially_filled"}
                    ):
                        try:
                            self.client.cancel_order_by_id(str(o.id))
                        except Exception:
                            pass
            except Exception:
                pass

            stop_side = "sell" if pos_side == "long" else "buy"
            new_stop_id = self._place_stop_loss(
                symbol=alpaca_symbol,
                qty=pos_qty,
                stop_price=breakeven_price,
                side=stop_side,
            )
            if new_stop_id:
                logger.info(
                    f"Breakeven stop updated for {symbol}: stop=${breakeven_price:.4f} "
                    f"(order={new_stop_id})"
                )
                return True
            return False
        except Exception as e:
            logger.error(f"Error moving stop to breakeven for {symbol}: {e}")
            return False
    
    def _place_stop_loss(
        self,
        symbol: str,
        qty: float,
        stop_price: float,
        side: str,
    ) -> Optional[str]:
        """
        Place a stop loss order.
        
        Args:
            symbol: Alpaca-formatted symbol
            qty: Quantity
            stop_price: Stop trigger price
            side: "buy" or "sell"
            
        Returns:
            Order ID if successful, None otherwise
        """
        try:
            order_request = StopOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                stop_price=round(stop_price, 2),
                time_in_force=TimeInForce.GTC,
            )
            
            order = self.client.submit_order(order_request)
            return str(order.id)
            
        except Exception as e:
            logger.error(f"Error placing stop loss for {symbol}: {e}")
            return None
    
    def _place_take_profit(
        self,
        symbol: str,
        qty: float,
        limit_price: float,
        side: str,
    ) -> Optional[str]:
        """
        Place a take profit (limit) order.
        
        Args:
            symbol: Alpaca-formatted symbol
            qty: Quantity
            limit_price: Target limit price
            side: "buy" or "sell"
            
        Returns:
            Order ID if successful, None otherwise
        """
        try:
            order_request = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                limit_price=round(limit_price, 2),
                time_in_force=TimeInForce.GTC,
            )
            
            order = self.client.submit_order(order_request)
            return str(order.id)
            
        except Exception as e:
            logger.error(f"Error placing take profit for {symbol}: {e}")
            return None
    
    def _handle_api_error(self, error: APIError) -> str:
        """Handle and format Alpaca API errors."""
        error_str = str(error)
        
        # Common error messages
        if "insufficient" in error_str.lower():
            return "Insufficient buying power"
        elif "market" in error_str.lower() and "closed" in error_str.lower():
            return "Market is closed"
        elif "not tradeable" in error_str.lower():
            return "Symbol is not tradeable"
        elif "position does not exist" in error_str.lower():
            return "No position exists"
        elif "invalid" in error_str.lower():
            return f"Invalid order: {error_str}"
        else:
            return error_str
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an open order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if cancelled successfully
        """
        try:
            self.client.cancel_order_by_id(order_id)
            logger.info(f"Order cancelled: {order_id}")
            return True
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    def cancel_all_orders(self) -> int:
        """
        Cancel all open orders.
        
        Returns:
            Number of orders cancelled
        """
        try:
            cancelled = self.client.cancel_orders()
            count = len(cancelled) if cancelled else 0
            logger.info(f"Cancelled {count} open orders")
            return count
        except Exception as e:
            logger.error(f"Error cancelling all orders: {e}")
            return 0
    
    def get_open_orders(self) -> List[Dict[str, Any]]:
        """Get all open orders."""
        try:
            orders = self.client.get_orders()
            return [
                {
                    "id": str(o.id),
                    "symbol": o.symbol,
                    "side": str(o.side),
                    "qty": float(o.qty) if o.qty else 0,
                    "type": str(o.type),
                    "status": str(o.status),
                    "created_at": str(o.created_at),
                }
                for o in orders
            ]
        except Exception as e:
            logger.error(f"Error getting open orders: {e}")
            return []
    
    def get_recent_orders(self, limit: int = 10) -> List[OrderResult]:
        """Get recent order history."""
        return self._recent_orders[-limit:]
