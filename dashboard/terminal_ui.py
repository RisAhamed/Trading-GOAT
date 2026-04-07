# dashboard/terminal_ui.py
"""
Rich terminal dashboard for the AI trading bot.
Displays live portfolio, positions, signals, and system status.
"""

import logging
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.spinner import Spinner
from rich.progress import Progress, SpinnerColumn, TextColumn

from core.config_loader import get_config, ConfigLoader


logger = logging.getLogger(__name__)


class TerminalUI:
    """
    Rich terminal dashboard for the AI trading bot.
    Displays real-time information in a beautiful terminal UI.
    """
    
    def __init__(self, config: Optional[ConfigLoader] = None) -> None:
        """Initialize the terminal UI."""
        self.config = config or get_config()
        self.console = Console()
        
        # State data
        self._bot_status = "STARTING"
        self._ollama_status = "CHECKING..."
        self._alpaca_status = "CHECKING..."
        self._ollama_model = ""
        self._loop_count = 0
        self._next_run_time = ""
        
        # Portfolio data
        self._portfolio: Dict[str, Any] = {
            "total_value": 0,
            "cash": 0,
            "today_pnl": 0,
            "today_pnl_pct": 0,
            "open_positions": 0,
            "today_trades": 0,
            "win_rate": 0,
            "daily_loss_pct": 0,
        }
        
        # Positions
        self._positions: List[Dict[str, Any]] = []
        
        # Market scan data
        self._market_scan: List[Dict[str, Any]] = []
        
        # AI reasoning history
        self._ai_reasoning: List[Dict[str, Any]] = []
        self._max_reasoning = 5
        
        # Recent trades
        self._recent_trades: List[Dict[str, Any]] = []
        self._max_trades = 5
        
        # System logs
        self._system_logs: List[str] = []
        self._max_logs = 5
        
        # AI thinking state
        self._ai_thinking = False
        self._ai_thinking_symbol = ""
        
        # Debug/status tracking
        self._data_fetch_status: Dict[str, str] = {}  # {symbol: status}
        self._last_errors: List[str] = []
        self._max_errors = 10
        self._signal_details: Dict[str, Dict[str, Any]] = {}  # {symbol: {action, confidence, reason}}
        self._trade_attempts: int = 0
        self._trade_successes: int = 0
        self._trade_failures: int = 0
        self._current_phase: str = "Initializing"
        
        # Thread control
        self._running = False
        self._live: Optional[Live] = None
        
        logger.info("TerminalUI initialized")
    
    def _make_layout(self) -> Layout:
        """Create the dashboard layout."""
        layout = Layout()
        
        layout.split_column(
            Layout(name="header", size=5),
            Layout(name="body"),
            Layout(name="footer", size=8),
        )
        
        layout["body"].split_row(
            Layout(name="left", ratio=1),
            Layout(name="right", ratio=1),
        )
        
        layout["left"].split_column(
            Layout(name="portfolio", size=7),
            Layout(name="positions"),
            Layout(name="market_scan"),
        )
        
        layout["right"].split_column(
            Layout(name="ai_reasoning"),
            Layout(name="recent_trades"),
        )
        
        return layout
    
    def _generate_header(self) -> Panel:
        """Generate the header panel."""
        time_str = datetime.now().strftime("%H:%M:%S")
        
        # Status colors
        status_color = "green" if self._bot_status == "RUNNING" else "yellow"
        ollama_color = "green" if "CONNECTED" in self._ollama_status else "red"
        alpaca_color = "green" if "CONNECTED" in self._alpaca_status else "red"
        
        header_text = Text()
        header_text.append("Mode: ", style="dim")
        header_text.append("PAPER", style="bold cyan")
        header_text.append(" | Status: ", style="dim")
        header_text.append(self._bot_status, style=f"bold {status_color}")
        header_text.append(" | Time: ", style="dim")
        header_text.append(time_str, style="bold white")
        header_text.append("\n")
        header_text.append("Ollama: ", style="dim")
        header_text.append(self._ollama_status, style=ollama_color)
        if self._ollama_model:
            header_text.append(f" ({self._ollama_model})", style="dim cyan")
        header_text.append(" | Alpaca: ", style="dim")
        header_text.append(self._alpaca_status, style=alpaca_color)
        
        if self._ai_thinking:
            header_text.append("\n")
            header_text.append("🤔 AI analyzing ", style="yellow")
            header_text.append(self._ai_thinking_symbol, style="bold yellow")
            header_text.append("...", style="yellow")
        
        return Panel(
            header_text,
            title="[bold blue]🤖 AI CRYPTO TRADER[/bold blue]",
            border_style="blue",
        )
    
    def _generate_portfolio(self) -> Panel:
        """Generate the portfolio panel."""
        p = self._portfolio
        
        # Format P&L with color
        pnl = p.get("today_pnl", 0)
        pnl_pct = p.get("today_pnl_pct", 0)
        pnl_color = "green" if pnl >= 0 else "red"
        pnl_sign = "+" if pnl >= 0 else ""
        
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Label", style="dim")
        table.add_column("Value", style="bold")
        
        table.add_row(
            "Total Value:",
            f"[bold white]${p.get('total_value', 0):,.2f}[/]"
        )
        table.add_row(
            "Cash:",
            f"${p.get('cash', 0):,.2f}"
        )
        table.add_row(
            "Today P&L:",
            f"[{pnl_color}]{pnl_sign}${pnl:,.2f} ({pnl_sign}{pnl_pct:.2f}%)[/]"
        )
        table.add_row(
            "Open Positions:",
            f"{p.get('open_positions', 0)}"
        )
        table.add_row(
            "Today Trades:",
            f"{p.get('today_trades', 0)} | Win Rate: {p.get('win_rate', 0):.0f}%"
        )
        
        return Panel(table, title="[bold]📊 PORTFOLIO[/bold]", border_style="green")
    
    def _generate_positions(self) -> Panel:
        """Generate the positions panel."""
        table = Table(show_header=True, header_style="bold", box=None)
        table.add_column("Symbol", style="cyan", width=10)
        table.add_column("Entry", justify="right", width=10)
        table.add_column("Current", justify="right", width=10)
        table.add_column("Qty", justify="right", width=8)
        table.add_column("P&L", justify="right", width=10)
        table.add_column("Status", width=6)
        
        if self._positions:
            for pos in self._positions[:5]:  # Max 5 positions
                pnl = pos.get("pnl", 0)
                pnl_pct = pos.get("pnl_pct", 0)
                pnl_color = "green" if pnl >= 0 else "red"
                pnl_sign = "+" if pnl >= 0 else ""
                
                status = pos.get("status", "HOLD")
                status_color = {
                    "HOLD": "white",
                    "WATCH": "yellow",
                    "NEAR_STOP": "red",
                    "NEAR_TP": "green",
                }.get(status, "white")
                
                table.add_row(
                    pos.get("symbol", ""),
                    f"${pos.get('entry', 0):,.2f}",
                    f"${pos.get('current', 0):,.2f}",
                    f"{pos.get('qty', 0):.4f}",
                    f"[{pnl_color}]{pnl_sign}${pnl:,.2f}[/]",
                    f"[{status_color}]{status}[/]",
                )
        else:
            table.add_row("", "[dim]No open positions[/]", "", "", "", "")
        
        return Panel(table, title="[bold]📈 OPEN POSITIONS[/bold]", border_style="cyan")
    
    def _generate_market_scan(self) -> Panel:
        """Generate the market scan panel."""
        table = Table(show_header=True, header_style="bold", box=None)
        table.add_column("Symbol", style="cyan", width=10)
        table.add_column("Trend", width=10)
        table.add_column("RSI", justify="center", width=6)
        table.add_column("MACD", justify="center", width=5)
        table.add_column("Signal", width=8)
        
        for scan in self._market_scan:
            trend = scan.get("trend", "SIDEWAYS")
            trend_color = {
                "BULLISH": "green",
                "BEARISH": "red",
                "SIDEWAYS": "yellow",
            }.get(trend, "white")
            
            signal = scan.get("signal", "HOLD")
            signal_color = {
                "BUY": "green",
                "SELL": "red",
                "HOLD": "dim",
                "WATCH": "yellow",
            }.get(signal, "white")
            
            macd_arrow = scan.get("macd_arrow", "→")
            
            table.add_row(
                scan.get("symbol", ""),
                f"[{trend_color}]{trend}[/]",
                f"{scan.get('rsi', 50):.0f}",
                macd_arrow,
                f"[{signal_color}]{signal}[/]",
            )
        
        return Panel(
            table,
            title="[bold]🔍 MARKET SCAN (10-min trend / 5-min entry)[/bold]",
            border_style="magenta",
        )
    
    def _generate_ai_reasoning(self) -> Panel:
        """Generate the AI reasoning panel."""
        content = Text()
        
        if not self._ai_reasoning:
            content.append("[dim]No AI decisions yet...[/dim]")
        else:
            for i, reasoning in enumerate(self._ai_reasoning[-self._max_reasoning:]):
                if i > 0:
                    content.append("\n\n")
                
                timestamp = reasoning.get("timestamp", "")
                symbol = reasoning.get("symbol", "")
                action = reasoning.get("action", "HOLD")
                confidence = reasoning.get("confidence", 0) * 100
                text = reasoning.get("reasoning", "")
                
                action_color = {
                    "BUY": "green",
                    "SELL": "red",
                    "HOLD": "yellow",
                    "CLOSE": "cyan",
                }.get(action, "white")
                
                content.append(f"[dim][{timestamp}][/dim] ")
                content.append(f"[bold cyan]{symbol}[/bold cyan] → ")
                content.append(f"[bold {action_color}]{action}[/bold {action_color}]")
                content.append(f" [dim]({confidence:.0f}% confidence)[/dim]")
                content.append(f"\n[italic]{text}[/italic]")
        
        return Panel(
            content,
            title="[bold]🧠 AI REASONING[/bold]",
            border_style="yellow",
        )
    
    def _generate_recent_trades(self) -> Panel:
        """Generate the recent trades panel."""
        content = Text()
        
        if not self._recent_trades:
            content.append("[dim]No trades yet...[/dim]")
        else:
            for trade in self._recent_trades[-self._max_trades:]:
                time_str = trade.get("time", "")
                action = trade.get("action", "")
                symbol = trade.get("symbol", "")
                qty = trade.get("qty", 0)
                price = trade.get("price", 0)
                pnl = trade.get("pnl", None)
                result = trade.get("result", "")
                stop_loss = trade.get("stop_loss", None)
                take_profit = trade.get("take_profit", None)
                
                action_color = "green" if action == "BUY" else "red"
                
                content.append(f"[dim]{time_str}[/dim] ")
                content.append(f"[{action_color}]{action}[/{action_color}] ")
                content.append(f"[cyan]{symbol}[/cyan] ")
                content.append(f"{qty:.4f} @ ${price:,.2f}")
                
                if stop_loss:
                    content.append(f" [dim]SL:${stop_loss:,.2f}[/dim]")
                if take_profit:
                    content.append(f" [dim]TP:${take_profit:,.2f}[/dim]")
                
                if pnl is not None:
                    pnl_color = "green" if pnl >= 0 else "red"
                    result_text = "WIN" if result == "WIN" else "LOSS"
                    content.append(f"\n    P&L: [{pnl_color}]${pnl:+,.2f}[/{pnl_color}] ({result_text})")
                
                content.append("\n")
        
        return Panel(
            content,
            title="[bold]📝 RECENT TRADES[/bold]",
            border_style="blue",
        )
    
    def _generate_footer(self) -> Panel:
        """Generate the system log footer."""
        content = Text()
        
        for log in self._system_logs[-self._max_logs:]:
            content.append(log + "\n")
        
        if self._loop_count > 0:
            content.append(f"\n[dim]Loop #{self._loop_count} | Next run: {self._next_run_time}[/dim]")
        
        return Panel(
            content,
            title="[bold]📋 SYSTEM LOG[/bold]",
            border_style="dim",
        )
    
    def _render(self) -> Layout:
        """Render the full dashboard."""
        layout = self._make_layout()
        
        layout["header"].update(self._generate_header())
        layout["portfolio"].update(self._generate_portfolio())
        layout["positions"].update(self._generate_positions())
        layout["market_scan"].update(self._generate_market_scan())
        layout["ai_reasoning"].update(self._generate_ai_reasoning())
        layout["recent_trades"].update(self._generate_recent_trades())
        layout["footer"].update(self._generate_footer())
        
        return layout
    
    def start(self) -> None:
        """Start the live dashboard."""
        self._running = True
        self._live = Live(
            self._render(),
            console=self.console,
            refresh_per_second=1,
            screen=True,
        )
        self._live.start()
        logger.info("Terminal UI started")
    
    def stop(self) -> None:
        """Stop the live dashboard."""
        self._running = False
        if self._live:
            self._live.stop()
            self._live = None
        logger.info("Terminal UI stopped")
    
    def refresh(self) -> None:
        """Refresh the dashboard."""
        if self._live:
            self._live.update(self._render())
    
    # Update methods
    def set_bot_status(self, status: str) -> None:
        """Set the bot status."""
        self._bot_status = status
        self.refresh()
    
    def set_ollama_status(self, connected: bool, model: str = "") -> None:
        """Set Ollama connection status."""
        self._ollama_status = "CONNECTED" if connected else "DISCONNECTED"
        self._ollama_model = model
        self.refresh()
    
    def set_alpaca_status(self, connected: bool) -> None:
        """Set Alpaca connection status."""
        self._alpaca_status = "CONNECTED" if connected else "DISCONNECTED"
        self.refresh()
    
    def set_ai_thinking(self, thinking: bool, symbol: str = "") -> None:
        """Set AI thinking state."""
        self._ai_thinking = thinking
        self._ai_thinking_symbol = symbol
        self.refresh()
    
    def update_portfolio(self, data: Dict[str, Any]) -> None:
        """Update portfolio data."""
        self._portfolio.update(data)
        self.refresh()
    
    def update_positions(self, positions: List[Dict[str, Any]]) -> None:
        """Update positions data."""
        self._positions = positions
        self.refresh()
    
    def update_market_scan(self, scan_data: List[Dict[str, Any]]) -> None:
        """Update market scan data."""
        self._market_scan = scan_data
        self.refresh()
    
    def add_ai_reasoning(
        self,
        symbol: str,
        action: str,
        confidence: float,
        reasoning: str,
    ) -> None:
        """Add an AI reasoning entry."""
        self._ai_reasoning.append({
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "symbol": symbol,
            "action": action,
            "confidence": confidence,
            "reasoning": reasoning,
        })
        
        # Keep only recent entries
        if len(self._ai_reasoning) > self._max_reasoning * 2:
            self._ai_reasoning = self._ai_reasoning[-self._max_reasoning:]
        
        self.refresh()
    
    def add_trade(
        self,
        action: str,
        symbol: str,
        qty: float,
        price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        pnl: Optional[float] = None,
        result: Optional[str] = None,
    ) -> None:
        """Add a trade entry."""
        self._recent_trades.append({
            "time": datetime.now().strftime("%H:%M"),
            "action": action,
            "symbol": symbol,
            "qty": qty,
            "price": price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "pnl": pnl,
            "result": result,
        })
        
        # Keep only recent entries
        if len(self._recent_trades) > self._max_trades * 2:
            self._recent_trades = self._recent_trades[-self._max_trades:]
        
        self.refresh()
    
    def add_log(self, level: str, message: str) -> None:
        """Add a system log entry."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        level_style = {
            "INFO": "blue",
            "WARN": "yellow",
            "ERROR": "red",
            "DEBUG": "dim",
        }.get(level.upper(), "white")
        
        log_entry = f"[dim][{timestamp}][/dim] [{level_style}][{level.upper()}][/{level_style}] {message}"
        self._system_logs.append(log_entry)
        
        # Keep only recent entries
        if len(self._system_logs) > self._max_logs * 2:
            self._system_logs = self._system_logs[-self._max_logs:]
        
        self.refresh()
    
    def set_loop_info(self, loop_count: int, next_run_time: str) -> None:
        """Set loop information."""
        self._loop_count = loop_count
        self._next_run_time = next_run_time
        self.refresh()
    
    def print_startup_banner(self) -> None:
        """Print the startup banner."""
        banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     🤖 AI CRYPTO TRADER - PAPER TRADING MODE 🤖              ║
║                                                              ║
║     Autonomous AI-Powered Trading Bot                        ║
║     Using Ollama LLM for Market Analysis                     ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""
        self.console.print(banner, style="bold blue")
        self.console.print(f"  Bot Name: {self.config.bot.name}", style="cyan")
        self.console.print(f"  AI Model: {self.config.ai.model}", style="cyan")
        self.console.print(f"  Mode: PAPER TRADING (Safe)", style="green")
        self.console.print(f"  Loop Interval: {self.config.bot.loop_interval_seconds}s", style="cyan")
        self.console.print()
    
    # Debug/Status methods
    def set_current_phase(self, phase: str) -> None:
        """Set the current phase (e.g., 'Fetching data', 'Analyzing', 'Trading')."""
        self._current_phase = phase
        self.refresh()
    
    def set_data_fetch_status(self, symbol: str, status: str) -> None:
        """Update data fetch status for a symbol."""
        self._data_fetch_status[symbol] = status
        self.refresh()
    
    def add_error(self, error: str) -> None:
        """Add an error to the error list."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._last_errors.append(f"[{timestamp}] {error}")
        if len(self._last_errors) > self._max_errors:
            self._last_errors = self._last_errors[-self._max_errors:]
        self.refresh()
    
    def update_signal_detail(self, symbol: str, action: str, confidence: float, reason: str) -> None:
        """Update signal details for a symbol."""
        self._signal_details[symbol] = {
            "action": action,
            "confidence": confidence,
            "reason": reason,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        }
        self.refresh()
    
    def record_trade_attempt(self, success: bool) -> None:
        """Record a trade attempt."""
        self._trade_attempts += 1
        if success:
            self._trade_successes += 1
        else:
            self._trade_failures += 1
        self.refresh()
    
    def get_trade_stats(self) -> Dict[str, int]:
        """Get trade statistics."""
        return {
            "attempts": self._trade_attempts,
            "successes": self._trade_successes,
            "failures": self._trade_failures,
        }
    
    def clear_data_fetch_status(self) -> None:
        """Clear all data fetch statuses."""
        self._data_fetch_status.clear()
        self.refresh()
