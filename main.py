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
from dashboard.terminal_ui import TerminalUI


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
        
        # Get trading pairs
        self.trading_pairs = self.config.markets.get_all_pairs()
        logger.info(f"Trading Pairs: {', '.join(self.trading_pairs)}")
        
        # Loop control
        self._running = False
        self._loop_count = 0
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame) -> None:
        """Handle shutdown signals."""
        logger.info("Shutdown signal received")
        self._running = False
        
        # Generate final report on shutdown
        try:
            report_path = self.results_tracker.write_full_report()
            logger.info(f"Final report saved: {report_path}")
            self.results_tracker.close_session()
        except Exception as e:
            logger.error(f"Error generating final report: {e}")
    
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
    
    def _process_symbol(
        self,
        symbol: str,
        portfolio_state: PortfolioState,
    ) -> Optional[OrderResult]:
        """
        Process a single trading symbol through the full pipeline.
        
        Args:
            symbol: Trading pair to process
            portfolio_state: Current portfolio state
            
        Returns:
            OrderResult if a trade was executed, None otherwise
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
                return None
            
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
                return None
            
            # 7. Calculate risk parameters
            if signal.action in ["BUY", "SELL"]:
                # Get current price for entry
                entry_price = current_price or symbol_indicators.current_price
                
                risk_params = self.risk_manager.calculate_position_size(
                    symbol=symbol,
                    entry_price=entry_price,
                    portfolio_value=portfolio_state.total_value,
                    side="long" if signal.action == "BUY" else "short",
                    atr=symbol_indicators.entry_tf.atr,
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
                    return None
                
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
                
                return result
            
            elif signal.action == "CLOSE":
                # Close existing position
                result = self.order_executor.close_position(symbol)
                
                if result.success:
                    logger.info(f"Position closed: {symbol}")
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
                
                return result
            
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            self.ui.add_log("ERROR", f"Error processing {symbol}: {str(e)[:50]}")
            return None
        
        return None
    
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
                    
                    # Process each trading pair
                    indicators_map: Dict[str, SymbolIndicators] = {}
                    trades_executed = 0
                    
                    for symbol in self.trading_pairs:
                        if not self._running:
                            break
                        
                        # Skip if trading is halted (but still collect indicators for display)
                        if not self.risk_manager.is_halted():
                            result = self._process_symbol(symbol, portfolio_state)
                            if result and result.success:
                                trades_executed += 1
                        
                        # Collect indicators for market scan
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
