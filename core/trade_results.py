# core/trade_results.py
"""
Trade Results Tracker - Professional P&L reporting system.

Creates comprehensive trade reports including:
- Individual trade details (entry/exit prices, timing, P&L)
- Daily/session summaries
- Running statistics
- JSON data for programmatic access
- Human-readable reports
"""

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from typing import Any, Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """Record of a single completed trade."""
    # Trade identification
    trade_id: str
    symbol: str
    
    # Order details
    side: str  # BUY or SELL
    order_type: str  # market, limit
    
    # Entry details
    entry_time: str
    entry_price: float
    quantity: float
    entry_value: float  # quantity * entry_price
    
    # Exit details (None if position still open)
    exit_time: Optional[str] = None
    exit_price: Optional[float] = None
    exit_value: Optional[float] = None
    
    # P&L calculations
    realized_pnl: Optional[float] = None
    pnl_percent: Optional[float] = None
    
    # Trade duration
    hold_duration_minutes: Optional[float] = None
    
    # AI decision info
    ai_confidence: Optional[float] = None
    ai_reasoning: Optional[str] = None
    
    # Status
    status: str = "OPEN"  # OPEN, CLOSED, CANCELLED
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SessionSummary:
    """Summary of a trading session."""
    session_date: str
    start_time: str
    end_time: Optional[str] = None
    
    # Portfolio
    starting_balance: float = 100000.0
    ending_balance: float = 100000.0
    
    # Trade counts
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    break_even_trades: int = 0
    
    # P&L
    total_realized_pnl: float = 0.0
    total_unrealized_pnl: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    
    # Ratios
    win_rate: float = 0.0
    profit_factor: float = 0.0
    
    # Symbols traded
    symbols_traded: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TradeResultsTracker:
    """
    Comprehensive trade results tracking and reporting.
    
    Maintains:
    - trades.json: All trade records
    - summary.json: Session summaries
    - report_YYYY-MM-DD.txt: Human-readable daily reports
    - positions.json: Current open positions
    """
    
    def __init__(self, results_dir: str = "logs/results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # File paths
        self.trades_file = self.results_dir / "trades.json"
        self.summary_file = self.results_dir / "summary.json"
        self.positions_file = self.results_dir / "positions.json"
        
        # In-memory data
        self.trades: List[TradeRecord] = []
        self.open_positions: Dict[str, TradeRecord] = {}
        self.session_summary: Optional[SessionSummary] = None
        
        # Load existing data
        self._load_data()
        
        # Start new session
        self._start_session()
        
        logger.info(f"TradeResultsTracker initialized: {self.results_dir}")
    
    def _load_data(self) -> None:
        """Load existing trade data from files."""
        # Load trades
        if self.trades_file.exists():
            try:
                with open(self.trades_file, 'r') as f:
                    data = json.load(f)
                    self.trades = [TradeRecord(**t) for t in data.get('trades', [])]
                logger.info(f"Loaded {len(self.trades)} historical trades")
            except Exception as e:
                logger.warning(f"Error loading trades: {e}")
                self.trades = []
        
        # Load open positions
        if self.positions_file.exists():
            try:
                with open(self.positions_file, 'r') as f:
                    data = json.load(f)
                    for symbol, pos_data in data.get('positions', {}).items():
                        self.open_positions[symbol] = TradeRecord(**pos_data)
                logger.info(f"Loaded {len(self.open_positions)} open positions")
            except Exception as e:
                logger.warning(f"Error loading positions: {e}")
                self.open_positions = {}
    
    def _save_data(self) -> None:
        """Save all data to files."""
        try:
            # Save trades
            trades_data = {
                'last_updated': datetime.now().isoformat(),
                'total_trades': len(self.trades),
                'trades': [t.to_dict() for t in self.trades]
            }
            with open(self.trades_file, 'w') as f:
                json.dump(trades_data, f, indent=2, default=str)
            
            # Save open positions
            positions_data = {
                'last_updated': datetime.now().isoformat(),
                'count': len(self.open_positions),
                'positions': {sym: pos.to_dict() for sym, pos in self.open_positions.items()}
            }
            with open(self.positions_file, 'w') as f:
                json.dump(positions_data, f, indent=2, default=str)
            
            # Save session summary
            if self.session_summary:
                summary_data = {
                    'last_updated': datetime.now().isoformat(),
                    'current_session': self.session_summary.to_dict()
                }
                with open(self.summary_file, 'w') as f:
                    json.dump(summary_data, f, indent=2, default=str)
                    
        except Exception as e:
            logger.error(f"Error saving trade data: {e}")
    
    def _start_session(self) -> None:
        """Start a new trading session."""
        now = datetime.now()
        self.session_summary = SessionSummary(
            session_date=now.strftime("%Y-%m-%d"),
            start_time=now.strftime("%H:%M:%S"),
        )
        logger.info(f"New session started: {self.session_summary.session_date}")
    
    def record_entry(
        self,
        symbol: str,
        side: str,
        quantity: float,
        entry_price: float,
        order_type: str = "market",
        ai_confidence: Optional[float] = None,
        ai_reasoning: Optional[str] = None,
        order_id: Optional[str] = None,
    ) -> TradeRecord:
        """
        Record a new trade entry (position opened).
        
        Args:
            symbol: Trading symbol (e.g., "BTC/USD")
            side: "BUY" or "SELL"
            quantity: Number of shares/coins
            entry_price: Entry price
            order_type: Order type (market, limit)
            ai_confidence: AI confidence level (0-1)
            ai_reasoning: AI reasoning text
            order_id: Broker order ID
            
        Returns:
            TradeRecord for the new position
        """
        now = datetime.now()
        trade_id = order_id or f"{symbol.replace('/', '')}_{now.strftime('%Y%m%d_%H%M%S')}"
        
        trade = TradeRecord(
            trade_id=trade_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            entry_time=now.isoformat(),
            entry_price=entry_price,
            quantity=quantity,
            entry_value=quantity * entry_price,
            ai_confidence=ai_confidence,
            ai_reasoning=ai_reasoning[:200] if ai_reasoning else None,
            status="OPEN",
        )
        
        # Store in open positions
        self.open_positions[symbol] = trade
        
        # Save to files
        self._save_data()
        
        # Write to daily report
        self._append_to_daily_report(
            f"📈 ENTRY: {side} {quantity:.6f} {symbol} @ ${entry_price:.2f}\n"
            f"   Value: ${trade.entry_value:.2f} | AI Confidence: {(ai_confidence or 0)*100:.0f}%\n"
            f"   Time: {now.strftime('%H:%M:%S')}\n"
        )
        
        logger.info(f"Trade entry recorded: {side} {quantity:.6f} {symbol} @ ${entry_price:.2f}")
        
        return trade
    
    def record_exit(
        self,
        symbol: str,
        exit_price: float,
        quantity: Optional[float] = None,
        reason: str = "SIGNAL",
    ) -> Optional[TradeRecord]:
        """
        Record a trade exit (position closed).
        
        Args:
            symbol: Trading symbol
            exit_price: Exit price
            quantity: Exit quantity (defaults to full position)
            reason: Exit reason (SIGNAL, STOP_LOSS, TAKE_PROFIT, MANUAL)
            
        Returns:
            Updated TradeRecord or None if no position
        """
        if symbol not in self.open_positions:
            logger.warning(f"No open position for {symbol} to close")
            return None
        
        trade = self.open_positions[symbol]
        now = datetime.now()
        
        exit_qty = quantity or trade.quantity
        
        # Calculate P&L
        trade.exit_time = now.isoformat()
        trade.exit_price = exit_price
        trade.exit_value = exit_qty * exit_price
        
        # For BUY positions: profit = exit - entry
        # For SELL (short) positions: profit = entry - exit
        if trade.side == "BUY":
            trade.realized_pnl = (exit_price - trade.entry_price) * exit_qty
        else:
            trade.realized_pnl = (trade.entry_price - exit_price) * exit_qty
        
        trade.pnl_percent = (trade.realized_pnl / trade.entry_value) * 100
        
        # Calculate hold duration
        entry_time = datetime.fromisoformat(trade.entry_time)
        trade.hold_duration_minutes = (now - entry_time).total_seconds() / 60
        
        trade.status = "CLOSED"
        
        # Move to completed trades
        self.trades.append(trade)
        del self.open_positions[symbol]
        
        # Update session summary
        self._update_session_summary(trade)
        
        # Save to files
        self._save_data()
        
        # Determine win/loss emoji
        pnl_emoji = "✅" if trade.realized_pnl >= 0 else "❌"
        
        # Write to daily report
        self._append_to_daily_report(
            f"{pnl_emoji} EXIT: CLOSE {exit_qty:.6f} {symbol} @ ${exit_price:.2f}\n"
            f"   Entry: ${trade.entry_price:.2f} → Exit: ${exit_price:.2f}\n"
            f"   P&L: ${trade.realized_pnl:.2f} ({trade.pnl_percent:+.2f}%)\n"
            f"   Hold Time: {trade.hold_duration_minutes:.1f} minutes | Reason: {reason}\n"
        )
        
        logger.info(
            f"Trade exit recorded: {symbol} P&L=${trade.realized_pnl:.2f} ({trade.pnl_percent:+.2f}%)"
        )
        
        return trade
    
    def _update_session_summary(self, trade: TradeRecord) -> None:
        """Update session summary with completed trade."""
        if not self.session_summary:
            return
        
        s = self.session_summary
        s.total_trades += 1
        
        pnl = trade.realized_pnl or 0
        s.total_realized_pnl += pnl
        
        if pnl > 0:
            s.winning_trades += 1
            s.largest_win = max(s.largest_win, pnl)
        elif pnl < 0:
            s.losing_trades += 1
            s.largest_loss = min(s.largest_loss, pnl)
        else:
            s.break_even_trades += 1
        
        if trade.symbol not in s.symbols_traded:
            s.symbols_traded.append(trade.symbol)
        
        # Calculate ratios
        if s.total_trades > 0:
            s.win_rate = (s.winning_trades / s.total_trades) * 100
        
        if s.winning_trades > 0:
            s.average_win = s.total_realized_pnl / s.winning_trades if s.total_realized_pnl > 0 else 0
        
        if s.losing_trades > 0:
            total_losses = sum(t.realized_pnl for t in self.trades if (t.realized_pnl or 0) < 0)
            s.average_loss = total_losses / s.losing_trades
            
            total_wins = sum(t.realized_pnl for t in self.trades if (t.realized_pnl or 0) > 0)
            if total_losses != 0:
                s.profit_factor = abs(total_wins / total_losses)
    
    def update_portfolio_value(self, total_value: float, cash: float) -> None:
        """Update portfolio values in session summary."""
        if self.session_summary:
            self.session_summary.ending_balance = total_value
            self._save_data()
    
    def update_unrealized_pnl(self, symbol: str, current_price: float) -> None:
        """Update unrealized P&L for an open position."""
        if symbol in self.open_positions:
            trade = self.open_positions[symbol]
            if trade.side == "BUY":
                unrealized = (current_price - trade.entry_price) * trade.quantity
            else:
                unrealized = (trade.entry_price - current_price) * trade.quantity
            
            if self.session_summary:
                # Recalculate total unrealized P&L
                total_unrealized = 0
                for sym, pos in self.open_positions.items():
                    if sym == symbol:
                        total_unrealized += unrealized
                    else:
                        # Keep existing unrealized calculation
                        pass
                self.session_summary.total_unrealized_pnl = unrealized
    
    def _append_to_daily_report(self, text: str) -> None:
        """Append text to the daily human-readable report."""
        today = date.today().strftime("%Y-%m-%d")
        report_file = self.results_dir / f"report_{today}.txt"
        
        try:
            with open(report_file, 'a', encoding='utf-8') as f:
                f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {text}\n")
        except Exception as e:
            logger.error(f"Error writing to daily report: {e}")
    
    def generate_summary_report(self) -> str:
        """Generate a comprehensive summary report."""
        if not self.session_summary:
            return "No active session"
        
        s = self.session_summary
        
        report = f"""
╔══════════════════════════════════════════════════════════════════╗
║                    TRADING SESSION REPORT                        ║
║                    {s.session_date}                                     ║
╠══════════════════════════════════════════════════════════════════╣
║ SESSION TIME                                                     ║
║   Start: {s.start_time}                                               ║
║   End:   {s.end_time or 'In Progress'}                                               ║
╠══════════════════════════════════════════════════════════════════╣
║ PORTFOLIO PERFORMANCE                                            ║
║   Starting Balance:  ${s.starting_balance:>12,.2f}                      ║
║   Current Balance:   ${s.ending_balance:>12,.2f}                      ║
║   Net P&L:           ${s.ending_balance - s.starting_balance:>+12,.2f}                      ║
║   Return:            {((s.ending_balance - s.starting_balance) / s.starting_balance) * 100:>+11.2f}%                       ║
╠══════════════════════════════════════════════════════════════════╣
║ TRADE STATISTICS                                                 ║
║   Total Trades:      {s.total_trades:>12}                              ║
║   Winning Trades:    {s.winning_trades:>12}                              ║
║   Losing Trades:     {s.losing_trades:>12}                              ║
║   Win Rate:          {s.win_rate:>11.1f}%                              ║
╠══════════════════════════════════════════════════════════════════╣
║ P&L BREAKDOWN                                                    ║
║   Realized P&L:      ${s.total_realized_pnl:>+12,.2f}                      ║
║   Unrealized P&L:    ${s.total_unrealized_pnl:>+12,.2f}                      ║
║   Largest Win:       ${s.largest_win:>+12,.2f}                      ║
║   Largest Loss:      ${s.largest_loss:>+12,.2f}                      ║
║   Profit Factor:     {s.profit_factor:>12.2f}                              ║
╠══════════════════════════════════════════════════════════════════╣
║ SYMBOLS TRADED                                                   ║
║   {', '.join(s.symbols_traded) if s.symbols_traded else 'None'}
╚══════════════════════════════════════════════════════════════════╝
"""
        return report
    
    def write_full_report(self) -> str:
        """Write a complete report to file and return the path."""
        today = date.today().strftime("%Y-%m-%d")
        report_file = self.results_dir / f"full_report_{today}.txt"
        
        report = self.generate_summary_report()
        
        # Add trade details
        report += "\n\n═══════════════════ TRADE HISTORY ═══════════════════\n\n"
        
        today_trades = [t for t in self.trades if t.entry_time.startswith(today)]
        
        if today_trades:
            for i, trade in enumerate(today_trades, 1):
                pnl_str = f"${trade.realized_pnl:+.2f}" if trade.realized_pnl else "N/A"
                report += f"""
Trade #{i}
─────────────────────────────────────────
  Symbol:         {trade.symbol}
  Side:           {trade.side}
  Status:         {trade.status}
  
  Entry Time:     {trade.entry_time}
  Entry Price:    ${trade.entry_price:.2f}
  Quantity:       {trade.quantity:.6f}
  Entry Value:    ${trade.entry_value:.2f}
  
  Exit Time:      {trade.exit_time or 'N/A'}
  Exit Price:     ${trade.exit_price:.2f if trade.exit_price else 'N/A'}
  Exit Value:     ${trade.exit_value:.2f if trade.exit_value else 'N/A'}
  
  Realized P&L:   {pnl_str}
  P&L %:          {trade.pnl_percent:+.2f}% if trade.pnl_percent else 'N/A'
  Hold Duration:  {trade.hold_duration_minutes:.1f} min if trade.hold_duration_minutes else 'N/A'
  
  AI Confidence:  {(trade.ai_confidence or 0) * 100:.0f}%
"""
        else:
            report += "  No completed trades today.\n"
        
        # Add open positions
        report += "\n\n═══════════════════ OPEN POSITIONS ═══════════════════\n\n"
        
        if self.open_positions:
            for symbol, pos in self.open_positions.items():
                report += f"""
Position: {symbol}
─────────────────────────────────────────
  Side:           {pos.side}
  Entry Time:     {pos.entry_time}
  Entry Price:    ${pos.entry_price:.2f}
  Quantity:       {pos.quantity:.6f}
  Entry Value:    ${pos.entry_value:.2f}
  AI Confidence:  {(pos.ai_confidence or 0) * 100:.0f}%
"""
        else:
            report += "  No open positions.\n"
        
        # Write to file with UTF-8 encoding
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Full report written to: {report_file}")
        
        return str(report_file)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current trading statistics as a dictionary."""
        s = self.session_summary
        if not s:
            return {}
        
        return {
            "session_date": s.session_date,
            "total_trades": s.total_trades,
            "winning_trades": s.winning_trades,
            "losing_trades": s.losing_trades,
            "win_rate": s.win_rate,
            "total_realized_pnl": s.total_realized_pnl,
            "total_unrealized_pnl": s.total_unrealized_pnl,
            "largest_win": s.largest_win,
            "largest_loss": s.largest_loss,
            "profit_factor": s.profit_factor,
            "open_positions": len(self.open_positions),
            "symbols_traded": s.symbols_traded,
        }
    
    def close_session(self) -> None:
        """Close the current session and write final report."""
        if self.session_summary:
            self.session_summary.end_time = datetime.now().strftime("%H:%M:%S")
            self._save_data()
            self.write_full_report()
            logger.info("Trading session closed")


# Global instance
_results_tracker: Optional[TradeResultsTracker] = None


def get_results_tracker() -> TradeResultsTracker:
    """Get or create the global results tracker instance."""
    global _results_tracker
    if _results_tracker is None:
        _results_tracker = TradeResultsTracker()
    return _results_tracker
