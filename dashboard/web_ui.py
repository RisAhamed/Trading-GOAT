# dashboard/web_ui.py
"""
Web-based monitoring dashboard for AI Crypto Trader.
Provides real-time visualization of bot status, trades, and portfolio.
"""

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
import re

from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from alpaca.trading.client import TradingClient
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame

import yaml

# Load environment variables
load_dotenv()

app = Flask(__name__, 
    template_folder='templates',
    static_folder='static'
)

# Configuration
CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
LOGS_DIR = Path(__file__).parent.parent / "logs"


def load_config() -> Dict[str, Any]:
    """Load bot configuration."""
    try:
        with open(CONFIG_PATH, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        return {"error": str(e)}


def get_alpaca_client() -> Optional[TradingClient]:
    """Get Alpaca trading client."""
    try:
        api_key = os.getenv("ALPACA_API_KEY")
        api_secret = os.getenv("ALPACA_API_SECRET")
        if api_key and api_secret:
            return TradingClient(api_key, api_secret, paper=True)
    except Exception:
        pass
    return None


def get_crypto_client() -> CryptoHistoricalDataClient:
    """Get Alpaca crypto data client."""
    return CryptoHistoricalDataClient()


def get_account_info() -> Dict[str, Any]:
    """Get current account information from Alpaca."""
    client = get_alpaca_client()
    if not client:
        return {"error": "Alpaca client not configured"}
    
    try:
        account = client.get_account()
        return {
            "account_number": account.account_number,
            "status": account.status.value if hasattr(account.status, 'value') else str(account.status),
            "cash": float(account.cash),
            "portfolio_value": float(account.portfolio_value),
            "buying_power": float(account.buying_power),
            "equity": float(account.equity),
            "last_equity": float(account.last_equity) if account.last_equity else 0,
            "currency": account.currency,
            "trading_blocked": account.trading_blocked,
            "pattern_day_trader": account.pattern_day_trader,
        }
    except Exception as e:
        return {"error": str(e)}


def get_positions() -> List[Dict[str, Any]]:
    """Get current open positions from Alpaca."""
    client = get_alpaca_client()
    if not client:
        return []
    
    try:
        positions = client.get_all_positions()
        result = []
        for pos in positions:
            pnl = float(pos.unrealized_pl) if pos.unrealized_pl else 0
            pnl_pct = float(pos.unrealized_plpc) * 100 if pos.unrealized_plpc else 0
            market_value = float(pos.market_value) if pos.market_value else 0
            avg_entry = float(pos.avg_entry_price) if pos.avg_entry_price else 0
            current_price = float(pos.current_price) if pos.current_price else 0
            qty = float(pos.qty) if pos.qty else 0
            
            result.append({
                "symbol": pos.symbol,
                "qty": qty,
                "avg_entry_price": avg_entry,
                "current_price": current_price,
                "market_value": market_value,
                "unrealized_pnl": pnl,
                "unrealized_pnl_pct": pnl_pct,
                "side": pos.side.value if hasattr(pos.side, 'value') else str(pos.side),
            })
        return result
    except Exception as e:
        return [{"error": str(e)}]


def get_recent_orders(limit: int = 20) -> List[Dict[str, Any]]:
    """Get recent orders from Alpaca."""
    client = get_alpaca_client()
    if not client:
        return []
    
    try:
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
        
        request = GetOrdersRequest(
            status=QueryOrderStatus.ALL,
            limit=limit,
        )
        orders = client.get_orders(request)
        
        result = []
        for order in orders:
            filled_price = float(order.filled_avg_price) if order.filled_avg_price else None
            filled_qty = float(order.filled_qty) if order.filled_qty else 0
            qty = float(order.qty) if order.qty else 0
            
            result.append({
                "id": str(order.id)[:8],
                "symbol": order.symbol,
                "side": order.side.value if hasattr(order.side, 'value') else str(order.side),
                "type": order.type.value if hasattr(order.type, 'value') else str(order.type),
                "qty": qty,
                "filled_qty": filled_qty,
                "filled_price": filled_price,
                "status": order.status.value if hasattr(order.status, 'value') else str(order.status),
                "created_at": order.created_at.strftime("%Y-%m-%d %H:%M:%S") if order.created_at else "",
                "filled_at": order.filled_at.strftime("%Y-%m-%d %H:%M:%S") if order.filled_at else "",
            })
        return result
    except Exception as e:
        return [{"error": str(e)}]


def get_market_prices() -> Dict[str, Dict[str, Any]]:
    """Get current market prices for trading pairs."""
    client = get_crypto_client()
    symbols = ["BTC/USD", "ETH/USD"]
    
    result = {}
    try:
        for symbol in symbols:
            request = CryptoBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Minute,
                limit=1,
            )
            bars = client.get_crypto_bars(request)
            
            if symbol in bars and len(bars[symbol]) > 0:
                bar = bars[symbol][-1]
                result[symbol] = {
                    "price": float(bar.close),
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "volume": float(bar.volume),
                    "timestamp": bar.timestamp.strftime("%H:%M:%S"),
                }
    except Exception as e:
        result["error"] = str(e)
    
    return result


def parse_log_line(line: str) -> Optional[Dict[str, Any]]:
    """Parse a log line into structured data."""
    # Format: 2026-04-07 08:01:19 | INFO     | module | message
    pattern = r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \| (\w+)\s*\| ([\w.]+) \| (.+)$'
    match = re.match(pattern, line.strip())
    
    if match:
        return {
            "timestamp": match.group(1),
            "level": match.group(2),
            "module": match.group(3),
            "message": match.group(4),
        }
    return None


def get_recent_logs(lines: int = 50) -> List[Dict[str, Any]]:
    """Get recent log entries."""
    today = datetime.now().strftime("%Y-%m-%d")
    log_file = LOGS_DIR / f"trading_{today}.log"
    
    if not log_file.exists():
        log_file = LOGS_DIR / "trading.log"
    
    if not log_file.exists():
        return []
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
        
        recent = all_lines[-lines:] if len(all_lines) > lines else all_lines
        
        result = []
        for line in recent:
            parsed = parse_log_line(line)
            if parsed:
                result.append(parsed)
        
        return result
    except Exception as e:
        return [{"error": str(e)}]


def get_trading_signals() -> List[Dict[str, Any]]:
    """Extract recent trading signals from logs."""
    logs = get_recent_logs(200)
    signals = []
    
    for log in logs:
        if isinstance(log, dict) and 'message' in log:
            msg = log['message']
            
            # Parse signal messages
            if 'Signal for' in msg or 'BUY' in msg or 'SELL' in msg or 'HOLD' in msg:
                # Extract symbol and signal info
                signal_match = re.search(r'(\w+/\w+): (BUY|SELL|HOLD) \(conf: (\d+)%\)', msg)
                if signal_match:
                    signals.append({
                        "timestamp": log['timestamp'],
                        "symbol": signal_match.group(1),
                        "action": signal_match.group(2),
                        "confidence": int(signal_match.group(3)),
                        "message": msg,
                    })
            
            # Parse indicator messages
            if 'RSI=' in msg:
                indicator_match = re.search(
                    r'\[(\w+/\w+)\] RSI=([\d.]+), MACD=(\w+), EMA=(\w+), Trend=(\w+)',
                    msg
                )
                if indicator_match:
                    signals.append({
                        "timestamp": log['timestamp'],
                        "symbol": indicator_match.group(1),
                        "type": "indicators",
                        "rsi": float(indicator_match.group(2)),
                        "macd": indicator_match.group(3),
                        "ema": indicator_match.group(4),
                        "trend": indicator_match.group(5),
                    })
    
    return signals[-20:]  # Last 20 signals


def get_bot_status() -> Dict[str, Any]:
    """Get current bot status from logs."""
    logs = get_recent_logs(50)
    
    status = {
        "running": False,
        "last_loop": None,
        "loop_count": 0,
        "last_update": None,
        "ollama_status": "unknown",
        "alpaca_status": "unknown",
    }
    
    for log in reversed(logs):
        if isinstance(log, dict) and 'message' in log:
            msg = log['message']
            
            if 'Loop #' in msg and 'started' in msg:
                status["running"] = True
                status["last_update"] = log['timestamp']
                loop_match = re.search(r'Loop #(\d+)', msg)
                if loop_match:
                    status["loop_count"] = int(loop_match.group(1))
            
            if 'Loop #' in msg and 'complete' in msg:
                status["last_loop"] = log['timestamp']
            
            if 'Ollama' in msg:
                if 'Connected' in msg:
                    status["ollama_status"] = "connected"
                elif 'unavailable' in msg or 'failed' in msg:
                    status["ollama_status"] = "fallback"
            
            if 'Alpaca' in msg and 'Connected' in msg:
                status["alpaca_status"] = "connected"
    
    return status


def calculate_daily_stats() -> Dict[str, Any]:
    """Calculate daily trading statistics."""
    orders = get_recent_orders(50)
    account = get_account_info()
    
    today = datetime.now().strftime("%Y-%m-%d")
    today_orders = [o for o in orders if isinstance(o, dict) and o.get('created_at', '').startswith(today)]
    
    filled_orders = [o for o in today_orders if o.get('status') == 'filled']
    buy_orders = [o for o in filled_orders if o.get('side') == 'buy']
    sell_orders = [o for o in filled_orders if o.get('side') == 'sell']
    
    # Calculate approximate P&L from positions
    positions = get_positions()
    unrealized_pnl = sum(p.get('unrealized_pnl', 0) for p in positions if isinstance(p, dict) and 'unrealized_pnl' in p)
    
    starting_equity = account.get('last_equity', 100000)
    current_equity = account.get('equity', 100000)
    day_pnl = current_equity - starting_equity if starting_equity else 0
    day_pnl_pct = (day_pnl / starting_equity * 100) if starting_equity else 0
    
    return {
        "total_trades": len(filled_orders),
        "buy_trades": len(buy_orders),
        "sell_trades": len(sell_orders),
        "unrealized_pnl": unrealized_pnl,
        "day_pnl": day_pnl,
        "day_pnl_pct": day_pnl_pct,
        "open_positions": len(positions),
    }


# Routes
@app.route('/')
def index():
    """Main dashboard page."""
    config = load_config()
    return render_template('index.html', config=config)


@app.route('/api/status')
def api_status():
    """Get current bot status."""
    return jsonify(get_bot_status())


@app.route('/api/account')
def api_account():
    """Get account information."""
    return jsonify(get_account_info())


@app.route('/api/positions')
def api_positions():
    """Get current positions."""
    return jsonify(get_positions())


@app.route('/api/orders')
def api_orders():
    """Get recent orders."""
    limit = request.args.get('limit', 20, type=int)
    return jsonify(get_recent_orders(limit))


@app.route('/api/prices')
def api_prices():
    """Get current market prices."""
    return jsonify(get_market_prices())


@app.route('/api/signals')
def api_signals():
    """Get recent trading signals."""
    return jsonify(get_trading_signals())


@app.route('/api/logs')
def api_logs():
    """Get recent log entries."""
    lines = request.args.get('lines', 50, type=int)
    return jsonify(get_recent_logs(lines))


@app.route('/api/stats')
def api_stats():
    """Get daily statistics."""
    return jsonify(calculate_daily_stats())


@app.route('/api/dashboard')
def api_dashboard():
    """Get all dashboard data in one call."""
    return jsonify({
        "status": get_bot_status(),
        "account": get_account_info(),
        "positions": get_positions(),
        "orders": get_recent_orders(10),
        "prices": get_market_prices(),
        "signals": get_trading_signals(),
        "stats": calculate_daily_stats(),
        "logs": get_recent_logs(30),
        "timestamp": datetime.now().isoformat(),
    })


def run_dashboard(host: str = "127.0.0.1", port: int = 5000, debug: bool = False):
    """Run the web dashboard."""
    print(f"\n{'='*60}")
    print("🖥️  AI CRYPTO TRADER - MONITORING DASHBOARD")
    print(f"{'='*60}")
    print(f"Dashboard URL: http://{host}:{port}")
    print(f"API Endpoint:  http://{host}:{port}/api/dashboard")
    print(f"{'='*60}\n")
    
    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Crypto Trader Monitoring Dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    run_dashboard(host=args.host, port=args.port, debug=args.debug)
