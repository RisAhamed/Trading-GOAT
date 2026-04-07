# AI Crypto Trader

An autonomous AI-powered crypto and forex paper trading bot that uses **Ollama Cloud models** (MiniMax M2.7, GLM-4-Plus, GPT-120B, DeepSeek-R1) for market analysis and Alpaca for order execution.

**⚠️ PAPER TRADING ONLY - This bot never uses real money.**

## Features

- 🤖 **Ollama Cloud AI Models**: Uses cloud-hosted thinking models (MiniMax M2.7, GLM-4-Plus, GPT-120B) for deep market reasoning
- 📊 **Dual-Timeframe Analysis**: 10-minute trend + 5-minute entry precision
- 📈 **Technical Indicators**: RSI, MACD, EMA, Bollinger Bands, ATR
- 💰 **Risk Management**: Position sizing, stop loss, take profit, daily loss limits
- 🖥️ **Rich Terminal Dashboard**: Beautiful real-time display of portfolio, positions, and signals
- 📝 **Comprehensive Logging**: Rotating daily log files for auditing
- ⚙️ **Fully Configurable**: Single config.yaml file for all settings

## Supported Ollama Cloud Models

| Model | Context | Best For |
|-------|---------|----------|
| `minimax-m2.7` | 80k | Strong reasoning, fast inference (Default) |
| `glm-4-plus` | 128k | Analytical capabilities, long context |
| `gpt-120b` | 64k | Deep reasoning, complex analysis |
| `deepseek-r1` | 64k | Structured thinking, chain-of-thought |
| `qwen-max` | 32k | General reasoning, balanced performance |

## Prerequisites

- **Python 3.10+** (tested with Python 3.10 - 3.14)
- **Ollama Cloud API Key** from [ollama.com](https://ollama.com)
- **Alpaca Account** with paper trading enabled ([alpaca.markets](https://alpaca.markets))

## Quick Start

### 1. Get Your Ollama Cloud API Key

1. Sign up at [ollama.com](https://ollama.com)
2. Navigate to your account settings
3. Generate an API key
4. Save the key - you'll need it for the `.env` file

### 2. Configure Environment

The `.env` file should already be configured with your API keys:

```env
OLLAMA_API_KEY=your_ollama_cloud_api_key
TAAPI_API_KEY=your_taapi_key
ALPACA_API_KEY=your_alpaca_key
ALPACA_API_SECRET=your_alpaca_secret
APCA_API_BASE_URL=https://paper-api.alpaca.markets
```

**Important**: The `OLLAMA_API_KEY` is used for authenticating with Ollama Cloud API.

### 3. Create and Activate Virtual Environment

**Windows (PowerShell):**
```powershell
# Navigate to project directory
cd c:\Users\riswa\Desktop\mybot

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# You should see (venv) in your prompt
```

**Windows (Command Prompt):**
```cmd
# Navigate to project directory
cd c:\Users\riswa\Desktop\mybot

# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate.bat

# You should see (venv) in your prompt
```

**Linux / macOS:**
```bash
# Navigate to project directory
cd ~/Desktop/mybot

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# You should see (venv) in your prompt
```

### 4. Install Dependencies

With the virtual environment activated:

```bash
pip install -r requirements.txt
```

**Note**: The bot uses the `ta` library (Technical Analysis Library) which is pure Python and works with Python 3.10 - 3.14+.

### 5. Configure the Bot

Edit `config.yaml` to set your preferred model and settings:

```yaml
ai:
  provider: "ollama_cloud"
  model: "minimax-m2.7"  # Options: minimax-m2.7, glm-4-plus, gpt-120b, deepseek-r1, qwen-max
  base_url: "https://api.ollama.com/v1"
  temperature: 0.3
  max_tokens: 1000
  timeout_seconds: 120
  reasoning_prompt_style: "chain_of_thought"
```

### 6. Run the Bot

With the virtual environment activated:

```bash
python main.py
```

The rich terminal dashboard will display:
- Current portfolio balance and equity
- Open positions with P&L
- Latest AI trading signals
- Recent activity log

### 7. Run the Web Dashboard (Optional)

For a more detailed monitoring experience, run the web-based dashboard:

```bash
# In a separate terminal (with venv activated)
python dashboard/web_ui.py
```

Then open your browser to **http://127.0.0.1:5000**

The web dashboard shows:
- 📊 **Real-time Portfolio** - Total value, cash, buying power, daily P&L
- 📈 **Live Prices** - Current BTC/USD, ETH/USD prices
- ⚙️ **Bot Status** - Running status, loop count, API connections
- 📂 **Open Positions** - Entry price, current price, unrealized P&L
- 🎯 **Recent Signals** - BUY/SELL/HOLD decisions with confidence %
- 📝 **Recent Orders** - Filled orders with prices
- 📋 **System Logs** - Real-time log viewer with filtering

**Note**: You can run both the terminal UI (main.py) and web dashboard (web_ui.py) simultaneously. They share the same data.

### 8. Deactivate Virtual Environment (When Done)

```bash
deactivate
```

## Virtual Environment Quick Reference

| Task | Windows PowerShell | Windows CMD | Linux/macOS |
|------|-------------------|-------------|-------------|
| Create venv | `python -m venv venv` | `python -m venv venv` | `python3 -m venv venv` |
| Activate | `.\venv\Scripts\Activate.ps1` | `venv\Scripts\activate.bat` | `source venv/bin/activate` |
| Deactivate | `deactivate` | `deactivate` | `deactivate` |
| Check Python | `python --version` | `python --version` | `python --version` |

## Architecture

```
ai_trader/
├── .env                    # API keys (never commit this!)
├── config.yaml             # All bot settings
├── requirements.txt        # Python dependencies
├── main.py                 # Entry point
├── core/
│   ├── config_loader.py    # Loads config + environment
│   ├── market_data.py      # Alpaca OHLCV data fetching
│   ├── indicators.py       # Technical indicators (ta library)
│   ├── ai_brain.py         # Ollama Cloud LLM integration
│   ├── signal_engine.py    # Dual-timeframe signal generation
│   ├── risk_manager.py     # Position sizing & risk limits
│   ├── order_executor.py   # Alpaca order placement
│   └── portfolio_tracker.py # Position & P&L tracking
├── dashboard/
│   └── terminal_ui.py      # Rich terminal dashboard
└── logs/
    └── trading.log         # Rotating daily logs
```

## How It Works

### 1. Market Data Collection
- Fetches OHLCV bars from Alpaca for both 10-minute (trend) and 5-minute (entry) timeframes
- Caches data to minimize API calls
- Supports crypto (BTC/USD, ETH/USD, SOL/USD, AVAX/USD) and forex (EUR/USD, GBP/USD)

### 2. Technical Analysis
- Calculates RSI, MACD, EMA (9/21), Bollinger Bands, and ATR on both timeframes
- Determines trend direction from 10-minute data
- Identifies entry opportunities from 5-minute data

### 3. AI Reasoning (Ollama Cloud)
The bot sends a detailed prompt to the Ollama Cloud model including:
- Current price and recent price action
- All technical indicator values
- Market context and trend analysis

The AI model uses **chain-of-thought reasoning** to:
- Analyze the overall market structure
- Evaluate bullish and bearish factors
- Determine optimal action (BUY, SELL, or HOLD)
- Provide a confidence score (0-100%)

### 4. Risk Management
Before any trade:
- Check daily loss limits (halt if exceeded)
- Check maximum concurrent positions
- Calculate position size based on risk percentage
- Set stop loss using ATR-based dynamic sizing
- Set take profit at configured multiplier

### 5. Order Execution
- Paper trades only (never real money)
- Market orders for immediate execution
- Bracket orders with stop loss and take profit
- Automatic order tracking and P&L calculation

## Configuration Reference

### Bot Settings
```yaml
bot:
  name: "AI Crypto Trader"
  mode: "paper"              # Never change to "live"
  base_currency: "USD"
  loop_interval_seconds: 30  # Main loop frequency
  log_level: "INFO"
```

### AI Model Settings
```yaml
ai:
  provider: "ollama_cloud"
  model: "minimax-m2.7"      # Cloud model with thinking capability
  base_url: "https://api.ollama.com/v1"
  temperature: 0.3           # Low = more deterministic
  max_tokens: 1000           # Allow detailed reasoning
  timeout_seconds: 120       # Thinking models need more time
```

### Risk Settings
```yaml
risk:
  max_positions: 3           # Max concurrent trades
  risk_per_trade_pct: 1.5    # % of portfolio per trade
  stop_loss_pct: 2.0         # % below entry
  take_profit_multiplier: 3.0 # 3x the stop loss
  max_daily_loss_pct: 5.0    # Halt if exceeded
  min_signal_confidence: 0.65 # AI must be ≥65% confident
```

## Changing AI Models

To use a different Ollama Cloud model, update `config.yaml`:

```yaml
ai:
  model: "gpt-120b"  # For deeper reasoning
  # or
  model: "glm-4-plus"  # For longer context analysis
  # or
  model: "deepseek-r1"  # For structured chain-of-thought
```

The bot will automatically use the specified model for all trading decisions.

## Troubleshooting

### "API key not valid"
- Verify your `OLLAMA_API_KEY` in `.env` is correct
- Ensure you have an active Ollama Cloud subscription
- Check that the API key has not expired

### "Model not found"
- Verify the model name in `config.yaml` matches Ollama Cloud's available models
- Try using `minimax-m2.7` as it's the recommended default

### "Connection timeout"
- Increase `timeout_seconds` in config (thinking models need more time)
- Check your internet connection
- Verify Ollama Cloud service status

### "Rate limit exceeded"
- Reduce `loop_interval_seconds` to slow down API calls
- Upgrade your Ollama Cloud plan if needed

### "Market data unavailable"
- Verify your Alpaca API keys are correct
- Check that the market is open (crypto trades 24/7, forex has specific hours)
- Ensure you're using paper trading URL

### "Failed to install pandas-ta / numba"
- This project uses the `ta` library (Technical Analysis) instead of `pandas-ta`
- The `ta` library is pure Python and works with Python 3.14+
- Make sure you're using the updated `requirements.txt`

### Virtual Environment Issues
- Make sure you've activated the virtual environment before installing dependencies
- If PowerShell blocks script execution, run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`
- Recreate the venv if corrupted: `Remove-Item -Recurse -Force venv` then create new one

## Logging

All activity is logged to `logs/trading.log`:
- Trading decisions with AI reasoning
- Order placements and fills
- Errors and warnings
- P&L updates

Logs rotate daily to prevent disk space issues.

## Safety Features

1. **Paper Trading Only**: The `paper=True` flag is hardcoded in the Alpaca client
2. **Daily Loss Limits**: Trading halts automatically if daily loss exceeds configured %
3. **Position Limits**: Maximum concurrent positions enforced
4. **Risk Per Trade**: Position sizing limits exposure per trade
5. **Confidence Threshold**: Trades only execute when AI confidence ≥ configured minimum

## Legal Disclaimer

This software is for educational purposes only. Trading cryptocurrencies and forex carries significant risk. Past performance does not guarantee future results. Never trade with money you cannot afford to lose. This bot is designed for paper trading only and should not be used with real funds.


Web Dashboard

URL: http://127.0.0.1:5000
Auto-refreshes every 5 seconds
Shows all trading data in real-time

## License

MIT License - See LICENSE file for details.
