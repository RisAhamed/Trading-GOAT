# main.py
"""
AI Crypto Trader - Main Entry Point

An autonomous AI-powered crypto and forex paper trading bot.
Uses Ollama LLM for market analysis and Alpaca for order execution.

PAPER TRADING ONLY - Never uses real money.
"""

import logging
import signal
import sys
import time
import threading
import webbrowser
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from core.config_loader import get_config, ConfigLoader
from core.market_data import MarketDataFetcher
from core.indicators import IndicatorCalculator, SymbolIndicators
from core.ai_brain import AIBrain, AIDecision
from core.signal_engine import SignalEngine, SignalResult
from core.risk_manager import RiskManager, RiskParameters
from core.order_executor import OrderExecutor, OrderResult
from core.portfolio_tracker import PortfolioTracker, PortfolioState
from core.trade_results import get_results_tracker, TradeResultsTracker
from core.trailing_stop_manager import TrailingStopManager
from core.position_monitor import PositionMonitor
from core.market_regime import MarketRegimeDetector
from core.bearish_scalp_strategy import BearishScalpStrategy
from dashboard.terminal_ui import TerminalUI
from dashboard.web_ui import app as web_app


# Setup logging
logger = logging.getLogger(__name__)


class AITrader:
    """
    Main AI Trading Bot class.
    Orchestrates all components and runs the trading loop.
    """
    
    def __init__(self, config_path: str = "config.yaml") -> None:
        """Initialize the AI Trader."""
        # Load configuration
        self.config = get_config(config_path)
        
        logger.info("=" * 60)
        logger.info("AI CRYPTO TRADER STARTING")
        logger.info("=" * 60)
        logger.info(f"Bot Name: {self.config.bot.name}")
        logger.info(f"Mode: {self.config.bot.mode.upper()}")
        logger.info(f"AI Model: {self.config.ai.model}")
        logger.info(f"Loop Interval: {self.config.bot.loop_interval_seconds}s")
        
        # Initialize components
        self.market_data = MarketDataFetcher(self.config)
        self.indicators = IndicatorCalculator(self.config)
        self.ai_brain = AIBrain(self.config)
        self.signal_engine = SignalEngine(self.config)
        self.risk_manager = RiskManager(self.config)
        self.order_executor = OrderExecutor(self.config)
        self.portfolio_tracker = PortfolioTracker(self.config)
        self.results_tracker = get_results_tracker()  # Trade results tracking
        self.ui = TerminalUI(self.config)

        # ═══ TRAILING STOP COMPONENTS ═══════════════════════════════════════
        # Initialize trailing stop manager
        self.trailing_manager = TrailingStopManager(self.config)

        # Market regime detector
        self.regime_detector = MarketRegimeDetector(self.market_data, self.config)
        self.bearish_scalp = BearishScalpStrategy(self.config)
        self._market_regime = "UNKNOWN"
        self._regime_summary = {}

        # Initialize position monitor (will be started in run())
        self.position_monitor = PositionMonitor(
            trailing_manager=self.trailing_manager,
            order_executor=self.order_executor,
            market_data=self.market_data,
            config=self.config,
        )
        # ═══ END TRAILING STOP COMPONENTS ═══════════════════════════════════
        
        # Get trading pairs
        self.trading_pairs = self.config.markets.get_all_pairs()
        logger.info(f"Trading Pairs: {', '.join(self.trading_pairs)}")
        
        # Loop control
        self._running = False
        self._loop_count = 0
        self._position_entry_times: Dict[str, datetime] = {}
        
        # ═══ WEB UI THREAD ═══════════════════════════════════════════════════
        self._web_thread: Optional[threading.Thread] = None
        # ═══ END WEB UI THREAD ═══════════════════════════════════════════════
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame) -> None:
        """Handle shutdown signals."""
        logger.info("Shutdown signal received")
        self._running = False

        # ═══ STOP POSITION MONITOR ═════════════════════════════════════════
        # Stop the position monitor thread
        try:
            self.position_monitor.stop()
            logger.info("Position monitor stopped")
        except Exception as e:
            logger.error(f"Error stopping position monitor: {e}")
        # ═══ END STOP POSITION MONITOR ═════════════════════════════════════

        # Generate final report on shutdown
        try:
            report_path = self.results_tracker.write_full_report()
            logger.info(f"Final report saved: {report_path}")
            self.results_tracker.close_session()
        except Exception as e:
            logger.error(f"Error generating final report: {e}")
    
    # ═══ WEB UI METHODS ═══════════════════════════════════════════════════════
    def _start_web_ui(self) -> None:
        """Start the web UI in a background thread."""
        if not self.config.web_ui.enabled:
            logger.info("Web UI disabled in config")
            return
        
        host = self.config.web_ui.host
        port = self.config.web_ui.port
        
        def run_flask():
            """Run Flask in a separate thread."""
            import logging as flask_logging
            # Suppress Flask's default logging to keep terminal clean
            flask_log = flask_logging.getLogger('werkzeug')
            flask_log.setLevel(flask_logging.WARNING)
            
            try:
                web_app.run(
                    host=host,
                    port=port,
                    debug=False,
                    threaded=True,
                    use_reloader=False,  # Important: disable reloader in thread
                )
            except Exception as e:
                logger.error(f"Web UI error: {e}")
        
        self._web_thread = threading.Thread(target=run_flask, daemon=True)
        self._web_thread.start()
        
        # Give Flask a moment to start
        time.sleep(1)
        
        # Open browser automatically if configured
        url = f"http://{host}:{port}"
        logger.info(f"Web dashboard available at: {url}")
        
        if self.config.web_ui.auto_open_browser:
            try:
                webbrowser.open(url)
            except Exception as e:
                logger.warning(f"Could not open browser automatically: {e}")
                print(f"\n📊 Web Dashboard: {url}")
        else:
            print(f"\n📊 Web Dashboard: {url}")
    # ═══ END WEB UI METHODS ═══════════════════════════════════════════════════
    
    def _verify_connections(self) -> bool:
        """Verify all API connections."""
        all_ok = True
        
        # Check Ollama
        ollama_ok, ollama_msg = self.ai_brain.check_connection()
        if ollama_ok:
            logger.info(f"✓ Ollama: {ollama_msg}")
            self.ui.set_ollama_status(True, self.ai_brain.get_active_model())
        else:
            logger.warning(f"✗ Ollama: {ollama_msg}")
            logger.warning("AI will use rule-based signals as fallback")
            self.ui.set_ollama_status(False)
        
        # Check Alpaca
        alpaca_ok, alpaca_msg, account_info = self.order_executor.check_connection()
        if alpaca_ok:
            logger.info(f"✓ Alpaca: {alpaca_msg}")
            if account_info:
                logger.info(f"  Portfolio Value: ${account_info.get('portfolio_value', 0):,.2f}")
                logger.info(f"  Cash Available: ${account_info.get('cash', 0):,.2f}")
            self.ui.set_alpaca_status(True)
        else:
            logger.error(f"✗ Alpaca: {alpaca_msg}")
            self.ui.set_alpaca_status(False)
            all_ok = False
        
        return all_ok
    
    def _check_quick_profit_exits(self, portfolio_state: PortfolioState) -> int:
        """
        Check all open positions for quick profit opportunities.
        Exits positions that meet profit thresholds from config.
        
        Returns:
            Number of positions closed for quick profit
        """
        exits_made = 0
        positions = self.portfolio_tracker.get_positions()
        
        if not positions:
            return 0
        
        # Get scalping settings from config
        quick_profit_pct = self.config.risk.quick_profit_threshold
        min_profit_dollars = self.config.risk.min_profit_to_exit
        max_hold_minutes = self.config.risk.max_hold_minutes
        scalping_mode = self.config.signals.scalping_mode
        
        if not scalping_mode:
            return 0
        
        from datetime import datetime, timedelta
        
        for pos in positions:
            symbol = pos.symbol
            entry_price = float(pos.entry_price)
            current_price = float(pos.current_price)
            qty = float(pos.qty)
            unrealized_pnl = float(pos.unrealized_pnl)
            
            # Calculate profit percentage
            profit_pct = ((current_price - entry_price) / entry_price) * 100
            
            should_exit = False
            exit_reason = ""
            
            # Check 1: Quick profit threshold reached
            if profit_pct >= quick_profit_pct:
                should_exit = True
                exit_reason = f"Quick profit: +{profit_pct:.2f}% >= {quick_profit_pct}%"
            
            # Check 2: Minimum dollar profit reached
            elif unrealized_pnl >= min_profit_dollars:
                should_exit = True
                exit_reason = f"Min profit: ${unrealized_pnl:.2f} >= ${min_profit_dollars}"
            
            # Check 3: Max hold time exceeded (exit regardless of profit/loss)
            # Note: We'd need entry time tracking for this - skip for now if not available
            
            if should_exit:
                logger.info(f"🎯 SCALP EXIT {symbol}: {exit_reason}")
                self.ui.add_log("INFO", f"🎯 Quick exit: {symbol} - {exit_reason}")
                
                result = self.order_executor.close_position(symbol)
                
                if result.success:
                    exits_made += 1
                    if symbol in self._position_entry_times:
                        del self._position_entry_times[symbol]
                    logger.info(f"✅ Quick profit taken: {symbol} P&L=${unrealized_pnl:.2f}")
                    self.ui.add_log("SUCCESS", f"Closed {symbol} for ${unrealized_pnl:.2f} profit")
                    
                    # Record in results
                    if result.filled_price:
                        self.results_tracker.record_exit(
                            symbol=symbol,
                            exit_price=result.filled_price,
                            reason="QUICK_PROFIT",
                        )
                else:
                    logger.warning(f"Failed to close {symbol}: {result.error_message}")
        
        return exits_made

    def _update_market_regime(self) -> str:
        """Detect current market regime and update internal state."""
        try:
            self._market_regime = self.regime_detector.get_regime()
            self._regime_summary = self.regime_detector.get_regime_summary()
            regime_emoji = {"BULLISH": "🟢", "BEARISH": "🔴", "SIDEWAYS": "🟡"}.get(self._market_regime, "⚪")
            self.ui.add_log("INFO", f"Market Regime: {regime_emoji} {self._market_regime} | ADX={self._regime_summary.get('adx', 0):.1f}")
            logger.info(f"Market regime: {self._market_regime} | {self._regime_summary}")
        except Exception as e:
            logger.error(f"Regime detection error: {e}")
            self._market_regime = "UNKNOWN"
        return self._market_regime

    def _check_max_hold_exits(self) -> int:
        """Force-exit positions held beyond max hold time (bearish protection)."""
        exits = 0
        max_hold_seconds = getattr(self.config.risk, "max_hold_seconds", 600)
        now = datetime.now()
        open_symbols = {pos.symbol for pos in self.portfolio_tracker.get_positions()}

        for symbol in list(self._position_entry_times.keys()):
            if symbol not in open_symbols:
                del self._position_entry_times[symbol]

        for symbol, entry_time in list(self._position_entry_times.items()):
            held_seconds = (now - entry_time).total_seconds()
            if held_seconds > max_hold_seconds:
                logger.warning(f"[MAX HOLD] {symbol} held {held_seconds:.0f}s > {max_hold_seconds}s — force closing")
                self.ui.add_log("WARN", f"⏱️ Force exit: {symbol} max hold exceeded ({held_seconds:.0f}s)")
                result = self.order_executor.close_position(symbol)
                if result.success:
                    exits += 1
                    del self._position_entry_times[symbol]
        return exits

    def _process_symbol(
        self,
        symbol: str,
        portfolio_state: PortfolioState,
    ) -> tuple[Optional[OrderResult], Optional[SymbolIndicators]]:
        """
        Process a single trading symbol through the full pipeline.
        
        Args:
            symbol: Trading pair to process
            portfolio_state: Current portfolio state
            
        Returns:
            Tuple of (OrderResult if executed, SymbolIndicators for scan display)
        """
        try:
            # 1. Fetch market data for both timeframes
            self.ui.set_data_fetch_status(symbol, "Fetching...")
            
            trend_interval = self.config.timeframes.trend_interval
            trend_lookback = self.config.timeframes.trend_lookback_bars
            entry_interval = self.config.timeframes.entry_interval
            entry_lookback = self.config.timeframes.entry_lookback_bars
            
            trend_bars = self.market_data.fetch_bars(symbol, trend_interval, trend_lookback)
            entry_bars = self.market_data.fetch_bars(symbol, entry_interval, entry_lookback)
            
            if trend_bars is None or entry_bars is None:
                self.ui.set_data_fetch_status(symbol, "❌ NO DATA")
                self.ui.add_error(f"{symbol}: Failed to fetch market data")
                logger.warning(f"Failed to fetch data for {symbol}")
                return None, None
            
            self.ui.set_data_fetch_status(symbol, f"✓ {len(trend_bars)} bars")
            
            # Get current price
            current_price = self.market_data.get_current_price(symbol)
            
            # 2. Calculate indicators on both timeframes
            symbol_indicators = self.indicators.calculate_for_symbol(
                symbol, trend_bars, entry_bars, current_price
            )
            
            # 3. Get current position if any
            current_position = self.portfolio_tracker.get_position_dict(symbol)
            
            # 4. Get AI decision
            self.ui.set_ai_thinking(True, symbol)
            
            portfolio_dict = {
                "total_value": portfolio_state.total_value,
                "cash": portfolio_state.cash,
                "open_positions": portfolio_state.open_positions,
            }
            
            ai_decision = self.ai_brain.get_decision(
                symbol,
                symbol_indicators,
                current_position,
                portfolio_dict,
            )
            
            self.ui.set_ai_thinking(False)
            
            # Update UI with AI reasoning
            self.ui.add_ai_reasoning(
                symbol,
                ai_decision.action,
                ai_decision.confidence,
                ai_decision.reasoning,
            )
            
            # Update signal detail for debug
            self.ui.update_signal_detail(
                symbol,
                ai_decision.action,
                ai_decision.confidence,
                ai_decision.reasoning[:100] if ai_decision.reasoning else "No reasoning"
            )
            
            # 5. Generate validated signal
            signal = self.signal_engine.generate_signal(
                symbol,
                symbol_indicators,
                ai_decision,
                current_position,
            )
            
            logger.info(
                f"{symbol}: {signal.action} (conf: {signal.confidence:.0%}) - "
                f"Confirmed: {signal.is_confirmed}"
            )
            
            # 6. Check if action is required
            if signal.action == "HOLD" or not signal.is_confirmed:
                return None, symbol_indicators
            
            # 7. Calculate risk parameters
            if signal.action in ["BUY", "SELL"]:
                # Get current price for entry
                entry_price = current_price or symbol_indicators.current_price
                position_size_override = 1.0

                # ═══ REGIME GATE — BLOCK BAD ENTRIES ════════════════════════
                if signal.action == "BUY" and self._market_regime == "BEARISH":
                    # Check if bearish scalp entry is valid instead
                    rsi_val = symbol_indicators.entry_tf.rsi
                    adx_val = getattr(symbol_indicators.entry_tf, 'adx', 25.0)
                    if self.bearish_scalp.should_enter_bearish_scalp(rsi_val, adx_val):
                        # Allow a small scalp entry with reduced size
                        logger.info(f"[BEARISH SCALP] {symbol}: RSI={rsi_val:.1f} — allowing micro-scalp entry")
                        self.ui.add_log("WARN", f"⚡ Bearish scalp entry: {symbol} RSI={rsi_val:.1f}")
                        scalp_params = self.bearish_scalp.get_bearish_scalp_params()
                        # Override position size multiplier
                        position_size_override = float(scalp_params["position_size_multiplier"])
                    else:
                        logger.info(f"[REGIME BLOCK] Skipping BUY for {symbol} — market is BEARISH, not extreme oversold")
                        self.ui.add_log("WARN", f"🔴 BUY blocked ({symbol}) — bearish market, no scalp conditions")
                        return None, symbol_indicators
                # ═══ END REGIME GATE ══════════════════════════════════════════
                
                risk_params = self.risk_manager.calculate_position_size(
                    symbol=symbol,
                    entry_price=entry_price,
                    portfolio_value=portfolio_state.total_value,
                    side="long" if signal.action == "BUY" else "short",
                    atr=symbol_indicators.entry_tf.atr,
                )
                
                # Apply optional position-size override (used for bearish scalp entries)
                risk_params.position_size_override = position_size_override
                if position_size_override != 1.0:
                    risk_params.qty *= position_size_override
                    risk_params.position_value *= position_size_override
                    risk_params.max_loss_usd *= position_size_override
                    logger.info(
                        f"{symbol} position size override applied: x{position_size_override:.2f} "
                        f"(qty={risk_params.qty:.6f})"
                    )

                # Validate risk parameters
                risk_params = self.risk_manager.validate_risk_parameters(risk_params)
                
                # Check if trade is allowed
                positions_list = self.portfolio_tracker.get_all_positions_dict()
                is_allowed, rejection = self.risk_manager.check_trade_allowed(
                    symbol=symbol,
                    action=signal.action,
                    current_positions=positions_list,
                    portfolio_value=portfolio_state.total_value,
                    cash_available=portfolio_state.cash,
                    position_value=risk_params.position_value,
                )
                
                if not is_allowed:
                    logger.warning(f"{symbol} trade rejected: {rejection}")
                    self.ui.add_log("WARN", f"{symbol}: {rejection}")
                    risk_params.is_allowed = False
                    risk_params.rejection_reason = rejection
                    return None, symbol_indicators
                
                # 8. Execute order
                if signal.action == "BUY":
                    result = self.order_executor.execute_buy(symbol, risk_params)
                else:
                    result = self.order_executor.execute_sell(symbol, risk_params)
                
                if result.success:
                    self.ui.record_trade_attempt(True)
                    logger.info(f"Order executed: {signal.action} {symbol} qty={risk_params.qty:.6f}")
                    self.ui.add_trade(
                        action=signal.action,
                        symbol=symbol,
                        qty=risk_params.qty,
                        price=entry_price,
                        stop_loss=risk_params.stop_price,
                        take_profit=risk_params.take_profit_price,
                    )
                    self.ui.add_log("INFO", f"Order executed: {signal.action} {symbol}")

                    # ═══ REGISTER WITH TRAILING STOP MANAGER ═══════════════════
                    # Register the new position for trailing stop tracking
                    if signal.action == "BUY":
                        fill_price = result.filled_price or entry_price
                        trailing_pos = self.trailing_manager.register_new_position(
                            symbol=symbol,
                            entry_price=fill_price,
                            qty=risk_params.qty,
                            market_regime=self._market_regime,
                        )
                        if trailing_pos:
                            logger.info(
                                f"Trailing stop registered for {symbol}: "
                                f"floor=${trailing_pos.floor_price:,.4f}, "
                                f"hard_stop=${trailing_pos.hard_stop_price:,.4f}"
                            )
                            self.ui.add_log(
                                "INFO",
                                f"Trailing stop active for {symbol} "
                                f"(floor=${trailing_pos.floor_price:,.2f})"
                            )

                        # Track entry time for max-hold enforcement
                        self._position_entry_times[symbol] = datetime.now()
                    # ═══ END REGISTER WITH TRAILING STOP MANAGER ═══════════════

                    # Update position stops in tracker
                    self.portfolio_tracker.set_position_stops(
                        symbol,
                        risk_params.stop_price,
                        risk_params.take_profit_price,
                    )
                else:
                    self.ui.record_trade_attempt(False)
                    self.ui.add_error(f"Order failed: {result.error_message}")
                    logger.error(f"Order failed: {result.error_message}")
                    self.ui.add_log("ERROR", f"Order failed: {result.error_message}")
                
                return result, symbol_indicators
            
            elif signal.action == "CLOSE":
                # Close existing position
                result = self.order_executor.close_position(symbol)
                
                if result.success:
                    logger.info(f"Position closed: {symbol}")
                    if symbol in self._position_entry_times:
                        del self._position_entry_times[symbol]
                    if current_position:
                        self.ui.add_trade(
                            action="CLOSE",
                            symbol=symbol,
                            qty=current_position.get("qty", 0),
                            price=current_position.get("current_price", 0),
                            pnl=current_position.get("unrealized_pnl", 0),
                            result="WIN" if current_position.get("unrealized_pnl", 0) > 0 else "LOSS",
                        )
                    self.ui.add_log("INFO", f"Position closed: {symbol}")
                
                return result, symbol_indicators
            
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            self.ui.add_log("ERROR", f"Error processing {symbol}: {str(e)[:50]}")
            return None, None
        
        return None, None
    
    def _update_market_scan(self, indicators_map: Dict[str, SymbolIndicators]) -> None:
        """Update the market scan display."""
        scan_data = []
        
        for symbol, ind in indicators_map.items():
            # Get quick signal
            quick_signal = self.signal_engine.quick_signal(ind)
            macd_arrow = self.signal_engine.get_macd_arrow(ind.entry_tf)
            
            scan_data.append({
                "symbol": symbol,
                "trend": ind.trend_tf.overall_trend,
                "rsi": ind.entry_tf.rsi,
                "macd_arrow": macd_arrow,
                "signal": quick_signal,
            })
        
        self.ui.update_market_scan(scan_data)
    
    def run(self) -> None:
        """Run the main trading loop."""
        # Print startup banner
        self.ui.print_startup_banner()
        
        # Verify connections
        logger.info("Verifying API connections...")
        if not self._verify_connections():
            logger.error("Cannot start: Alpaca connection failed")
            print("\n❌ Cannot start: Alpaca API connection failed")
            print("Please check your API keys in .env file")
            return
        
        # Initial portfolio update
        portfolio_state = self.portfolio_tracker.update()
        
        # Start the UI
        self.ui.start()
        self.ui.set_bot_status("RUNNING")
        self.ui.add_log("INFO", "AI Trader started successfully")

        # ═══ START POSITION MONITOR ═════════════════════════════════════════
        # Start the position monitor thread
        self.position_monitor.start()
        if self.trailing_manager.is_enabled():
            self.ui.add_log("INFO", "Trailing stop monitor started")
        # ═══ END START POSITION MONITOR ═════════════════════════════════════

        # ═══ START WEB UI ════════════════════════════════════════════════════
        # Start the web dashboard in background thread and open browser
        self._start_web_ui()
        if self.config.web_ui.enabled:
            self.ui.add_log(
                "INFO", 
                f"Web dashboard: http://{self.config.web_ui.host}:{self.config.web_ui.port}"
            )
        # ═══ END START WEB UI ════════════════════════════════════════════════
        
        # Initial portfolio display
        self.ui.update_portfolio({
            "total_value": portfolio_state.total_value,
            "cash": portfolio_state.cash,
            "today_pnl": portfolio_state.today_total_pnl,
            "today_pnl_pct": portfolio_state.today_pnl_pct,
            "open_positions": portfolio_state.open_positions,
            "today_trades": portfolio_state.today_trades,
            "win_rate": portfolio_state.win_rate,
        })
        
        self._running = True
        
        try:
            while self._running:
                self._loop_count += 1
                loop_start = time.time()
                
                logger.info(f"--- Loop #{self._loop_count} started ---")
                
                try:
                    # Update portfolio state
                    portfolio_state = self.portfolio_tracker.update()
                    
                    # Update results tracker with portfolio value
                    self.results_tracker.update_portfolio_value(
                        total_value=portfolio_state.total_value,
                        cash=portfolio_state.cash,
                    )
                    
                    # Update risk manager with daily stats
                    self.risk_manager.update_daily_stats(
                        portfolio_value=portfolio_state.total_value,
                        realized_pnl=portfolio_state.today_realized_pnl,
                        unrealized_pnl=portfolio_state.today_unrealized_pnl,
                        trades_count=portfolio_state.today_trades,
                        wins=portfolio_state.today_wins,
                        losses=portfolio_state.today_losses,
                    )
                    
                    # Check if trading is halted
                    if self.risk_manager.is_halted():
                        self.ui.add_log("WARN", f"Trading halted: {self.risk_manager.get_halt_reason()}")
                        self.ui.set_bot_status("HALTED")
                    else:
                        self.ui.set_bot_status("RUNNING")
                    
                    # Update portfolio display
                    self.ui.update_portfolio({
                        "total_value": portfolio_state.total_value,
                        "cash": portfolio_state.cash,
                        "today_pnl": portfolio_state.today_total_pnl,
                        "today_pnl_pct": portfolio_state.today_pnl_pct,
                        "open_positions": portfolio_state.open_positions,
                        "today_trades": portfolio_state.today_trades,
                        "win_rate": portfolio_state.win_rate,
                    })
                    
                    # Update positions display
                    positions_summary = self.portfolio_tracker.get_portfolio_summary()
                    self.ui.update_positions(positions_summary.get("positions", []))

                    # ═══ UPDATE TRAILING STOP DISPLAY ═══════════════════════════
                    # Update trailing stop panel with current positions
                    trailing_summary = self.trailing_manager.get_position_summary()
                    self.ui.update_trailing_stops(trailing_summary)
                    # ═══ END UPDATE TRAILING STOP DISPLAY ═══════════════════════
                    
                    # ═══════════════════════════════════════════════════════════════
                    # Update market regime at start of each loop
                    self._update_market_regime()

                    # ═══════════════════════════════════════════════════════════════
                    # Force-exit stale positions
                    stale_exits = self._check_max_hold_exits()
                    if stale_exits > 0:
                        portfolio_state = self.portfolio_tracker.update()

                    # ═══════════════════════════════════════════════════════════════
                    # SCALPING: Check for quick profit exits FIRST (before new trades)
                    # ═══════════════════════════════════════════════════════════════
                    quick_exits = self._check_quick_profit_exits(portfolio_state)
                    if quick_exits > 0:
                        logger.info(f"🎯 Quick profit exits: {quick_exits} positions closed")
                        # Refresh portfolio after exits
                        portfolio_state = self.portfolio_tracker.update()
                    
                    # Process each trading pair
                    indicators_map: Dict[str, SymbolIndicators] = {}
                    trades_executed = quick_exits  # Count quick exits as trades
                    
                    for symbol in self.trading_pairs:
                        if not self._running:
                            break
                        
                        # Skip if trading is halted (but still collect indicators for display)
                        if not self.risk_manager.is_halted():
                            result, symbol_indicators = self._process_symbol(symbol, portfolio_state)
                            if result and result.success:
                                trades_executed += 1
                            if symbol_indicators is not None:
                                indicators_map[symbol] = symbol_indicators
                        else:
                            # Still collect indicators for display while halted
                            trend_bars = self.market_data.fetch_bars(
                                symbol,
                                self.config.timeframes.trend_interval,
                                self.config.timeframes.trend_lookback_bars,
                            )
                            entry_bars = self.market_data.fetch_bars(
                                symbol,
                                self.config.timeframes.entry_interval,
                                self.config.timeframes.entry_lookback_bars,
                            )
                            
                            if trend_bars is not None and entry_bars is not None:
                                indicators_map[symbol] = self.indicators.calculate_for_symbol(
                                    symbol, trend_bars, entry_bars
                                )
                    
                    # Update market scan display
                    self._update_market_scan(indicators_map)
                    
                    # Log loop summary
                    loop_duration = time.time() - loop_start
                    next_run = datetime.now() + timedelta(seconds=self.config.bot.loop_interval_seconds)
                    next_run_str = next_run.strftime("%H:%M:%S")
                    
                    self.ui.set_loop_info(self._loop_count, next_run_str)
                    self.ui.add_log(
                        "INFO",
                        f"Loop #{self._loop_count} complete. "
                        f"Scanned {len(self.trading_pairs)} symbols. "
                        f"{trades_executed} trades executed."
                    )
                    
                    logger.info(
                        f"Loop #{self._loop_count} complete in {loop_duration:.1f}s. "
                        f"Trades: {trades_executed}. Next run: {next_run_str}"
                    )
                    
                except Exception as e:
                    logger.error(f"Error in main loop: {e}")
                    self.ui.add_log("ERROR", f"Loop error: {str(e)[:50]}")
                
                # Sleep until next loop
                if self._running:
                    time.sleep(self.config.bot.loop_interval_seconds)
        
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        
        finally:
            # Cleanup
            self.ui.set_bot_status("STOPPING")
            self.ui.add_log("INFO", "Shutting down...")
            logger.info("Shutting down AI Trader...")
            
            # Don't close positions on shutdown
            logger.info("Positions remain open. Goodbye!")
            
            time.sleep(1)  # Give UI time to update
            self.ui.stop()
            
            print("\n✓ AI Trader stopped cleanly")
            print("  All positions remain open")
            print("  Check logs for details: logs/trading.log")


def main() -> None:
    """Main entry point."""
    try:
        trader = AITrader()
        trader.run()
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
