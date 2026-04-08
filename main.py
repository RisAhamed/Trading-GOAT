# main.py
"""
AI Crypto Trader - Main Entry Point

An autonomous AI-powered crypto and forex paper trading bot.
Uses Ollama LLM for market analysis and Alpaca for order execution.

PAPER TRADING ONLY - Never uses real money.
"""

import logging
import re
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
from core.market_regime import MarketRegimeDetector, MarketRegime
from core.bearish_scalp_strategy import BearishScalpStrategy
from core.trade_exit_engine import TradeExitEngine
from dashboard.terminal_ui import TerminalUI
from dashboard.web_ui import app as web_app


# Setup logging
logger = logging.getLogger(__name__)

# Regex pattern for valid crypto symbols (BASE/QUOTE format)
CRYPTO_SYMBOL_PATTERN = re.compile(r"^[A-Z]+/[A-Z]+$")
REGIME_BLOCKED_BUY = ("CRASH", "EXTREME_FEAR", "HIGH_VOLATILITY")
BEARISH_CONFIDENCE_BONUS = 0.20


def _is_valid_crypto_symbol(symbol: str) -> bool:
    """Check if symbol matches valid crypto format (BASE/QUOTE like BTC/USD)."""
    if not isinstance(symbol, str):
        return False
    return bool(CRYPTO_SYMBOL_PATTERN.match(symbol.strip()))


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
        self.order_executor.portfolio_tracker = self.portfolio_tracker
        self.exit_engine = TradeExitEngine(self.config)
        self.results_tracker = get_results_tracker()  # Trade results tracking
        self.ui = TerminalUI(self.config)

        # ═══ TRAILING STOP COMPONENTS ═══════════════════════════════════════
        # Initialize trailing stop manager
        self.trailing_manager = TrailingStopManager(self.config)

        # Market regime detector
        self.regime_detector = MarketRegimeDetector(self.market_data, self.config)
        from core.symbol_scanner import SymbolScanner
        self.symbol_scanner = SymbolScanner(self.market_data, self.config)
        self.bearish_scalp = BearishScalpStrategy(self.config)
        self.market_regime = MarketRegime(self.config)
        self._market_regime: str = "UNKNOWN"
        self._regime_summary: dict = {}

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
        self._active_trading_pairs: List[str] = list(self.trading_pairs)
        logger.info(f"Trading Pairs: {', '.join(self.trading_pairs)}")
        
        # Loop control
        self._running = False
        self._loop_count = 0
        self._loss_cooldowns: Dict[str, datetime] = {}
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
                    self._cleanup_exit_engine_state(symbol)
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
        try:
            self._market_regime = self.regime_detector.get_regime()
            self._regime_summary = self.regime_detector.get_regime_summary()
            emoji = {"BULLISH": "🟢", "BEARISH": "🔴", "SIDEWAYS": "🟡"}.get(
                self._market_regime, "⚪"
            )
            self.ui.add_log(
                "INFO",
                f"Regime: {emoji} {self._market_regime} | "
                f"ADX={self._regime_summary.get('adx', 0):.1f} | "
                f"Session={self._regime_summary.get('session', 'UNKNOWN')}"
            )
        except Exception as e:
            logger.error(f"Regime detection error: {e}")
            self._market_regime = "UNKNOWN"
        return self._market_regime

    def _check_regime_change_exits(self) -> int:
        """
        On BEARISH regime: exit losing longs immediately.
        Leave profitable longs alone — let trailing stop handle them.
        """
        exits = 0
        try:
            if self._market_regime != "BEARISH":
                return 0

            positions = self.portfolio_tracker.get_positions()
            for pos in positions:
                symbol = pos.symbol
                try:
                    side = getattr(pos, 'side', 'long')
                    unrealized_pnl = float(pos.unrealized_pnl)

                    if side != 'long':
                        continue

                    if unrealized_pnl >= 0:
                        logger.info(
                            f"[REGIME EXIT] {symbol}: Profitable long (+${unrealized_pnl:.2f}) "
                            f"— letting trailing stop protect it"
                        )
                        self.ui.add_log(
                            "INFO",
                            f"🔒 {symbol} profitable in bearish market — trailing stop protecting"
                        )
                        continue

                    logger.warning(
                        f"[REGIME EXIT] {symbol}: BEARISH regime, losing long "
                        f"(P&L=${unrealized_pnl:.2f}) — force closing"
                    )
                    self.ui.add_log(
                        "WARN",
                        f"🔴 Regime exit: {symbol} (P&L=${unrealized_pnl:.2f})"
                    )

                    result = self.order_executor.close_position(symbol)
                    if result.success:
                        exits += 1
                        if result.filled_price:
                            self.results_tracker.record_exit(
                                symbol=symbol,
                                exit_price=result.filled_price,
                                reason="REGIME_CHANGE_BEARISH",
                            )
                        cooldown_minutes = getattr(self.config.risk, 'loss_cooldown_minutes', 10)
                        self._loss_cooldowns[symbol] = datetime.now() + timedelta(minutes=cooldown_minutes)
                        logger.info(f"[COOLDOWN] {symbol}: {cooldown_minutes}min after regime loss exit")
                        self.ui.add_log("WARN", f"⏳ {symbol} cooldown: {cooldown_minutes}min")
                        self._position_entry_times.pop(symbol, None)
                        self._cleanup_exit_engine_state(symbol)

                except Exception as e:
                    logger.error(f"[REGIME EXIT] Error processing {symbol}: {e}")
                    continue

        except Exception as e:
            logger.error(f"[REGIME EXIT] Fatal error: {e}")
        return exits

    def _update_active_symbols(self) -> None:
        """
        Use the SymbolScanner to dynamically update which symbols
        we're actively trading this loop cycle.
        Only updates if scanner is enabled in config.
        """
        try:
            scanner_cfg = getattr(self.config, 'symbol_scanner', None)
            if not getattr(scanner_cfg, 'enabled', False):
                return

            crypto_only = getattr(scanner_cfg, 'crypto_only', True)
            tradeable = self.symbol_scanner.get_tradeable_symbols()
            max_symbols = getattr(scanner_cfg, 'max_symbols_to_trade', 3)
            
            # Filter and validate tradeable symbols
            validated_tradeable: List[str] = []
            for sym in tradeable[:max_symbols]:
                if _is_valid_crypto_symbol(sym):
                    validated_tradeable.append(sym)
                else:
                    logger.warning(f"[MAIN] Removed non-crypto symbol from active pairs: {sym}")
            
            self._active_trading_pairs = validated_tradeable

            # Always include symbols with open positions (don't abandon open trades)
            # But only if they are valid crypto symbols (or crypto_only is disabled)
            open_symbols = {pos.symbol for pos in self.portfolio_tracker.get_positions()}
            for sym in open_symbols:
                if sym not in self._active_trading_pairs:
                    if not crypto_only or _is_valid_crypto_symbol(sym):
                        self._active_trading_pairs.append(sym)
                    else:
                        logger.warning(f"[MAIN] Skipping non-crypto open position symbol: {sym}")

            rankings = self.symbol_scanner.get_ranking_summary()
            top3 = rankings[:3]
            rank_str = " | ".join(
                [f"{r['symbol']}={r['total_score']}" for r in top3]
            )
            self.ui.add_log(
                "INFO",
                f"📊 Symbol scan: Top3 → {rank_str} | Active: {self._active_trading_pairs}"
            )
            logger.info(f"[SCANNER] Active pairs updated: {self._active_trading_pairs}")

            if hasattr(self, 'symbol_scanner') and hasattr(
                self.symbol_scanner, 'political_scanner'
            ):
                try:
                    summary = self.symbol_scanner.political_scanner.get_signal_summary()
                    if summary:
                        self.ui.add_log("INFO", summary)
                except Exception as e:
                    logger.debug(f"[SCANNER] Political summary unavailable: {e}")

        except Exception as e:
            logger.error(f"[SCANNER] Symbol update error: {e}")
            # Keep existing pairs on error — never crash the main loop

    def _check_max_hold_exits(self) -> int:
        exits = 0
        try:
            max_hold_seconds = getattr(self.config.risk, 'max_hold_seconds', 600)
            now = datetime.now()
            for symbol, entry_time in list(self._position_entry_times.items()):
                try:
                    held_seconds = (now - entry_time).total_seconds()
                    if held_seconds <= max_hold_seconds:
                        continue

                    pos = self.portfolio_tracker.get_position_dict(symbol)
                    unrealized_pnl = float(pos.get('unrealized_pnl', 0)) if pos else 0

                    logger.warning(
                        f"[MAX HOLD] {symbol}: held {held_seconds:.0f}s > {max_hold_seconds}s "
                        f"(P&L=${unrealized_pnl:.2f}) — force closing"
                    )
                    self.ui.add_log(
                        "WARN",
                        f"⏱️ Max hold exit: {symbol} ({held_seconds:.0f}s, P&L=${unrealized_pnl:.2f})"
                    )

                    result = self.order_executor.close_position(symbol)
                    if result.success:
                        exits += 1
                        self._position_entry_times.pop(symbol, None)
                        self._cleanup_exit_engine_state(symbol)

                        if unrealized_pnl < 0:
                            cooldown_minutes = getattr(self.config.risk, 'loss_cooldown_minutes', 10)
                            self._loss_cooldowns[symbol] = datetime.now() + timedelta(minutes=cooldown_minutes)
                            logger.info(f"[COOLDOWN] {symbol}: {cooldown_minutes}min after max-hold loss")
                            self.ui.add_log("WARN", f"⏳ {symbol} cooldown: {cooldown_minutes}min")

                except Exception as e:
                    logger.error(f"[MAX HOLD] Error for {symbol}: {e}")
        except Exception as e:
            logger.error(f"[MAX HOLD] Fatal error: {e}")
        return exits

    def _check_profit_lock_stops(self) -> int:
        """
        Profit-lock stop: once profit >= profit_lock_trigger_pct,
        move stop to entry + (current_gain * profit_lock_ratio).
        Stop only moves UP, NEVER down.
        """
        updates = 0
        try:
            profit_lock_trigger_pct = getattr(self.config.risk, 'profit_lock_trigger_pct', 1.5)
            profit_lock_ratio = getattr(self.config.risk, 'profit_lock_ratio', 0.5)
            min_profit_lock_pct = getattr(self.config.risk, 'min_profit_lock_pct', 0.3)

            positions = self.portfolio_tracker.get_positions()
            for pos in positions:
                try:
                    symbol = pos.symbol
                    entry_price = float(pos.entry_price)
                    current_price = float(pos.current_price)

                    if entry_price <= 0 or current_price <= entry_price:
                        continue

                    current_gain = current_price - entry_price
                    profit_pct = (current_gain / entry_price) * 100

                    if profit_pct < profit_lock_trigger_pct:
                        continue

                    locked_gain = current_gain * profit_lock_ratio
                    min_gain = entry_price * (min_profit_lock_pct / 100)
                    new_stop = entry_price + max(locked_gain, min_gain)

                    current_stop = None
                    try:
                        current_stop = self.portfolio_tracker.get_position_stop(symbol)
                    except AttributeError:
                        pass

                    if current_stop is not None and current_stop >= new_stop:
                        continue

                    self.portfolio_tracker.set_position_stops(symbol, new_stop, None)

                    if hasattr(self.trailing_manager, 'update_floor_price'):
                        self.trailing_manager.update_floor_price(symbol, new_stop)

                    updates += 1
                    locked_dollars = new_stop - entry_price
                    logger.info(
                        f"[PROFIT-LOCK] {symbol}: Entry=${entry_price:,.4f} | "
                        f"Price=${current_price:,.4f} (+{profit_pct:.2f}%) | "
                        f"Stop→${new_stop:,.4f} | Min profit≥${locked_dollars:,.4f}"
                    )
                    self.ui.add_log(
                        "INFO",
                        f"🔒 Profit lock {symbol}: stop=${new_stop:,.2f} (+${locked_dollars:.2f} guaranteed)"
                    )

                except Exception as e:
                    logger.error(f"[PROFIT-LOCK] Error for {symbol}: {e}")
        except Exception as e:
            logger.error(f"[PROFIT-LOCK] Fatal error: {e}")
        return updates

    def _cleanup_exit_engine_state(self, symbol: str) -> None:
        try:
            self.exit_engine.cleanup_position(symbol)
        except Exception:
            pass

    def _evaluate_open_position_exits(self) -> int:
        """
        Re-evaluate open positions each loop and execute exit-engine actions.
        """
        actions_taken = 0
        try:
            exit_cfg = getattr(self.config, "exit_engine", None)
            if not bool(getattr(exit_cfg, "enabled", True)):
                return 0

            positions = self.portfolio_tracker.get_positions()
            for pos in positions:
                try:
                    symbol = pos.symbol
                    side = getattr(pos, "side", "long")
                    entry_price = float(getattr(pos, "entry_price", 0) or 0)
                    stop_price = float(getattr(pos, "stop_price", 0) or 0)
                    unrealized_pnl_pct = float(getattr(pos, "unrealized_pnl_pct", 0) or 0)

                    if symbol not in self._position_entry_times:
                        self._position_entry_times[symbol] = datetime.now()
                    entry_dt = self._position_entry_times.get(symbol, datetime.now())
                    entry_ts = entry_dt.timestamp()

                    self.exit_engine.register_position_entry(
                        symbol=symbol,
                        entry_timestamp=entry_ts,
                        entry_price=entry_price,
                    )

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
                    if trend_bars is None or entry_bars is None or entry_bars.empty:
                        continue

                    indicators = self.indicators.calculate_for_symbol(
                        symbol=symbol,
                        trend_bars=trend_bars,
                        entry_bars=entry_bars,
                    )
                    current_signal = self.signal_engine.quick_signal(indicators)
                    position_payload = {
                        "symbol": symbol,
                        "side": side,
                        "avg_entry_price": entry_price,
                        "stop_price": stop_price,
                        "unrealized_plpc": unrealized_pnl_pct / 100.0,
                    }
                    decision = self.exit_engine.evaluate_position_exit(
                        position=position_payload,
                        current_bars=entry_bars,
                        current_signal=current_signal,
                        entry_time=entry_ts,
                    )

                    action = str(decision.get("action", "HOLD"))
                    reason = str(decision.get("reason", "EXIT_ENGINE"))
                    exit_pct = float(decision.get("exit_pct", 0.0) or 0.0)

                    if action == "EXIT_FULL":
                        result = self.order_executor.close_position(symbol, reason=reason)
                        if result.success:
                            actions_taken += 1
                            self._position_entry_times.pop(symbol, None)
                            self._cleanup_exit_engine_state(symbol)
                            logger.info(f"[EXIT ENGINE] Full exit {symbol}: {reason}")
                            self.ui.add_log("WARN", f"🚪 Exit full {symbol}: {reason}")

                    elif action == "EXIT_PARTIAL":
                        result = self.order_executor.close_position_partial(
                            symbol=symbol,
                            exit_pct=exit_pct if exit_pct > 0 else 0.5,
                            reason=reason,
                        )
                        if result.success:
                            actions_taken += 1
                            if entry_price > 0:
                                moved = self.order_executor.move_stop_to_breakeven(symbol, entry_price)
                                if moved:
                                    self.portfolio_tracker.set_position_stops(symbol, entry_price, None)
                                    if hasattr(self.trailing_manager, "update_floor_price"):
                                        self.trailing_manager.update_floor_price(symbol, entry_price)
                            logger.info(f"[EXIT ENGINE] Partial exit {symbol}: {reason}")
                            self.ui.add_log("INFO", f"✂️ Exit partial {symbol}: {reason}")

                    elif action == "MOVE_STOP_BREAKEVEN":
                        if entry_price > 0:
                            moved = self.order_executor.move_stop_to_breakeven(symbol, entry_price)
                            if moved:
                                actions_taken += 1
                                self.portfolio_tracker.set_position_stops(symbol, entry_price, None)
                                if hasattr(self.trailing_manager, "update_floor_price"):
                                    self.trailing_manager.update_floor_price(symbol, entry_price)
                                logger.info(f"[EXIT ENGINE] Breakeven stop moved {symbol}: {reason}")
                                self.ui.add_log("INFO", f"🛡️ Stop→breakeven {symbol}: {reason}")
                except Exception as e:
                    logger.error(f"[EXIT ENGINE] Error evaluating {getattr(pos, 'symbol', '?')}: {e}")
                    continue
        except Exception as e:
            logger.error(f"[EXIT ENGINE] Fatal cycle error: {e}")
        return actions_taken

    def _pre_entry_analysis(
        self,
        symbol: str,
        signal_action: str,
        symbol_indicators: SymbolIndicators,
    ) -> tuple[bool, str]:
        """
        Confluence-scored pre-entry gate.
        Each check contributes a weighted score (max 100).
        Trade only approved if score >= confluence_min_score (default 70).
        """
        try:
            score = 0
            max_score = 100
            details = []
            hard_failures = []

            cooldown_until = self._loss_cooldowns.get(symbol)
            if cooldown_until and datetime.now() < cooldown_until:
                remaining = int((cooldown_until - datetime.now()).total_seconds())
                hard_failures.append(f"⏳ {symbol} in loss cooldown — {remaining}s remaining")

            session = self._regime_summary.get("session", "UNKNOWN")
            off_peak_mode = getattr(self.config, 'session_filter', None)
            prefer_sessions = True
            if off_peak_mode:
                prefer_sessions = getattr(off_peak_mode, 'prefer_high_volume_sessions', True)
            if prefer_sessions and session == "OFF_PEAK" and signal_action == "BUY":
                hard_failures.append(f"🌙 Off-peak session — no new entries (session={session})")

            if hard_failures:
                summary = " | ".join(hard_failures)
                logger.info(f"[PRE-ENTRY] {symbol} HARD BLOCKED: {summary}")
                self.ui.add_log("WARN", f"Pre-entry {symbol}: HARD BLOCKED")
                return False, summary

            regime = self._market_regime
            if signal_action == "BUY":
                if regime == "BULLISH":
                    score += 30
                    details.append("✅ Regime BULLISH (+30)")
                elif regime == "SIDEWAYS":
                    score += 15
                    details.append("⚠️ Regime SIDEWAYS (+15)")
                elif regime == "BEARISH":
                    rsi_val = symbol_indicators.entry_tf.rsi
                    if rsi_val < 25:
                        score += 10
                        details.append(f"⚡ Bearish scalp RSI={rsi_val:.1f} (+10)")
                    else:
                        details.append(f"🔴 Regime BEARISH RSI={rsi_val:.1f} (+0)")

            adx = getattr(symbol_indicators.entry_tf, 'adx', 25.0)
            if adx >= 25:
                score += 20
                details.append(f"✅ ADX={adx:.1f} strong trend (+20)")
            elif adx >= 20:
                score += 12
                details.append(f"✅ ADX={adx:.1f} trending (+12)")
            else:
                details.append(f"🟡 ADX={adx:.1f} choppy (+0)")

            rsi = symbol_indicators.entry_tf.rsi
            if signal_action == "BUY":
                if 35 <= rsi <= 55:
                    score += 15
                    details.append(f"✅ RSI={rsi:.1f} ideal zone (+15)")
                elif rsi < 35:
                    score += 10
                    details.append(f"✅ RSI={rsi:.1f} oversold (+10)")
                elif rsi <= 65:
                    score += 8
                    details.append(f"⚠️ RSI={rsi:.1f} neutral (+8)")
                else:
                    details.append(f"🔴 RSI={rsi:.1f} overbought (+0)")

            ema_trend = symbol_indicators.entry_tf.ema_trend
            if signal_action == "BUY":
                if ema_trend == "BULLISH":
                    score += 20
                    details.append("✅ EMA BULLISH (+20)")
                elif ema_trend == "SIDEWAYS":
                    score += 10
                    details.append("⚠️ EMA SIDEWAYS (+10)")
                else:
                    details.append("🔴 EMA BEARISH (+0)")

            volume_increasing = getattr(symbol_indicators.entry_tf, 'volume_increasing', True)
            volume_ratio = getattr(symbol_indicators.entry_tf, 'volume_ratio', 1.0)
            if volume_increasing and volume_ratio >= 1.2:
                score += 15
                details.append("✅ Volume spike confirmed (+15)")
            elif volume_increasing:
                score += 8
                details.append("✅ Volume increasing (+8)")
            else:
                details.append("⚠️ Volume declining (+0)")

            confluence_min = getattr(self.config.risk, 'confluence_min_score', 70)
            is_approved = score >= confluence_min
            status = f"{'APPROVED ✅' if is_approved else 'REJECTED ❌'} Score={score}/{max_score}"
            summary = f"{status} | " + " | ".join(details)

            logger.info(f"[PRE-ENTRY] {symbol} {signal_action} → {summary}")
            self.ui.add_log(
                "INFO" if is_approved else "WARN",
                f"Pre-entry {symbol}: Score={score}/{max_score} {'✅' if is_approved else '❌'}"
            )
            return is_approved, summary

        except Exception as e:
            logger.error(f"[PRE-ENTRY] Error for {symbol}: {e}")
            return False, f"Pre-entry analysis error: {e}"

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
            # Global daily drawdown kill switch
            self.risk_manager.update_daily_equity(portfolio_state.total_value)
            if self.risk_manager.is_kill_switch_active():
                self.ui.add_log("ERROR", "🛑 Daily kill switch active — no new trades")
                return None, None

            # Re-entry cooldown after stop-loss exits
            if self.portfolio_tracker.is_in_cooldown(symbol):
                logger.info(f"[COOLDOWN] {symbol}: Post-stop-loss cooldown active — skipping")
                return None, None

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

            symbol_meta = {}
            if hasattr(self, 'symbol_scanner'):
                rankings = self.symbol_scanner.get_ranking_summary()
                for r in rankings:
                    if r['symbol'] == symbol:
                        symbol_meta = dict(r)
                        symbol_meta['pool_size'] = len(rankings)
                        break

            # Regime gate before AI decision
            try:
                btc_bars = self.market_data.fetch_bars("BTC/USD", trend_interval, trend_lookback)
                current_regime = self.market_regime.detect_regime(btc_bars)
                if current_regime in REGIME_BLOCKED_BUY and not current_position:
                    logger.warning(
                        f"[REGIME GATE] {symbol}: Blocking new BUY — regime={current_regime}"
                    )
                    return None, symbol_indicators
                if current_regime == "BEARISH":
                    effective_min_confidence = (
                        getattr(self.config.risk, "min_signal_confidence", 0.50) + BEARISH_CONFIDENCE_BONUS
                    )
                    symbol_meta["regime"] = current_regime
                    symbol_meta["min_confidence_override"] = effective_min_confidence
                elif current_regime:
                    symbol_meta["regime"] = current_regime
            except Exception as e:
                logger.error(f"[REGIME GATE] Error checking regime: {e}")

            # Triple confirmation pre-filter before AI (BUY hints only)
            action_hint = self.signal_engine.quick_signal(symbol_indicators)
            if action_hint == "BUY":
                confirmed, reason = self.signal_engine._triple_confirmation_check(
                    symbol, symbol_indicators, symbol_meta
                )
                if not confirmed:
                    logger.info(f"[FILTER] {symbol}: BUY blocked — {reason}")
                    return None, symbol_indicators
            
            ai_decision = self.ai_brain.get_decision(
                symbol,
                symbol_indicators,
                current_position,
                portfolio_dict,
                symbol_meta=symbol_meta,
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

                # Dynamic ATR-normalized quantity sizing override
                # Prefer scanner ATR% when available (same bars used for ranking);
                # fallback to live entry timeframe ATR%.
                atr_pct_from_meta = symbol_meta.get("atr_pct")
                atr_pct = (
                    atr_pct_from_meta
                    if atr_pct_from_meta is not None
                    else getattr(symbol_indicators.entry_tf, "atr_percent", 0.0)
                )
                intel_modifier = int(symbol_meta.get("intel_modifier", 0) or 0)
                dynamic_qty = self.order_executor.calculate_dynamic_position_size(
                    symbol=symbol,
                    current_price=entry_price,
                    atr_pct=float(atr_pct or 0.0),
                    portfolio_cash=portfolio_state.cash,
                    intel_modifier=intel_modifier,
                )
                if dynamic_qty > 0:
                    risk_params.qty = dynamic_qty
                    risk_params.position_value = dynamic_qty * entry_price
                    risk_params.max_loss_usd = risk_params.stop_loss_distance * dynamic_qty
                
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
                    approved, reason = self._pre_entry_analysis(symbol, signal.action, symbol_indicators)
                    if not approved:
                        logger.info(f"[PRE-ENTRY BLOCKED] {symbol}: {reason}")
                        return None, symbol_indicators
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
                        self.exit_engine.register_position_entry(
                            symbol=symbol,
                            entry_timestamp=self._position_entry_times[symbol].timestamp(),
                            entry_price=fill_price,
                        )
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
                    if current_position:
                        unrealized_pnl = current_position.get('unrealized_pnl', 0)
                        if float(unrealized_pnl) < 0:
                            cooldown_minutes = getattr(self.config.risk, 'loss_cooldown_minutes', 10)
                            self._loss_cooldowns[symbol] = datetime.now() + timedelta(minutes=cooldown_minutes)
                            logger.info(f"[COOLDOWN] {symbol}: {cooldown_minutes}min cooldown after loss close")
                            self.ui.add_log("WARN", f"⏳ {symbol} cooldown: {cooldown_minutes}min after loss")
                    self._position_entry_times.pop(symbol, None)
                    self._cleanup_exit_engine_state(symbol)
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

                    # 4) Update market regime + session
                    self._update_market_regime()

                    # 4b) Dynamically rank and select best trading symbols
                    self._update_active_symbols()

                    # 5) Exit losing longs on BEARISH regime
                    regime_exits = self._check_regime_change_exits()
                    if regime_exits > 0:
                        portfolio_state = self.portfolio_tracker.update()

                    # 6) Exit engine re-evaluation (full/partial/breakeven)
                    exit_actions = self._evaluate_open_position_exits()
                    if exit_actions > 0:
                        portfolio_state = self.portfolio_tracker.update()

                    # 7) Force-exit stale positions
                    stale_exits = self._check_max_hold_exits()
                    if stale_exits > 0:
                        portfolio_state = self.portfolio_tracker.update()

                    # 9) Profit-lock stop updates
                    self._check_profit_lock_stops()

                    # 10) Existing quick-profit exits
                    quick_exits = self._check_quick_profit_exits(portfolio_state)
                    if quick_exits > 0:
                        logger.info(f"🎯 Quick profit exits: {quick_exits} positions closed")
                        portfolio_state = self.portfolio_tracker.update()

                    # 12) Update UI displays after portfolio/actions are finalized
                    self.ui.update_portfolio({
                        "total_value": portfolio_state.total_value,
                        "cash": portfolio_state.cash,
                        "today_pnl": portfolio_state.today_total_pnl,
                        "today_pnl_pct": portfolio_state.today_pnl_pct,
                        "open_positions": portfolio_state.open_positions,
                        "today_trades": portfolio_state.today_trades,
                        "win_rate": portfolio_state.win_rate,
                    })

                    positions_summary = self.portfolio_tracker.get_portfolio_summary()
                    self.ui.update_positions(positions_summary.get("positions", []))

                    trailing_summary = self.trailing_manager.get_position_summary()
                    self.ui.update_trailing_stops(trailing_summary)
                    
                    # Process each trading pair
                    indicators_map: Dict[str, SymbolIndicators] = {}
                    trades_executed = quick_exits  # Count quick exits as trades
                    
                    for symbol in self._active_trading_pairs:
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
                        f"Scanned {len(self._active_trading_pairs)} symbols. "
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
