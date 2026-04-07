# core/ai_brain.py
"""
AI Brain module using Ollama Cloud API for trading decisions.
Connects to Ollama Cloud models (MiniMax M2.7, GLM-4-Plus, GPT-120B, etc.)
for market analysis and chain-of-thought reasoning.

Ollama Cloud provides access to powerful thinking models without local GPU requirements.
Uses Ollama's native API format (not OpenAI-compatible).
"""

import json
import logging
import re
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import requests
from ollama import Client as OllamaClient

from .config_loader import get_config, ConfigLoader
from .indicators import IndicatorValues, SymbolIndicators


logger = logging.getLogger(__name__)


# Ollama Cloud model capabilities mapping
# Model names for direct API access (without -cloud suffix)
OLLAMA_CLOUD_MODELS = {
    "minimax-m2.7": {
        "name": "MiniMax M2.7",
        "api_name": "minimax-m2.7",  # For direct API
        "thinking": True,
        "context_length": 80000,
        "description": "Strong reasoning and analysis capabilities",
    },
    "gpt-oss:120b": {
        "name": "GPT OSS 120B",
        "api_name": "gpt-oss:120b",
        "thinking": True,
        "context_length": 32000,
        "description": "Large model with deep reasoning",
    },
    "deepseek-v3.1:671b": {
        "name": "DeepSeek V3.1 671B",
        "api_name": "deepseek-v3.1:671b",
        "thinking": True,
        "context_length": 64000,
        "description": "Excellent for structured analysis",
    },
    "cogito-2.1:671b": {
        "name": "Cogito 2.1 671B",
        "api_name": "cogito-2.1:671b",
        "thinking": True,
        "context_length": 64000,
        "description": "Deep reasoning model",
    },
    "mistral-large-3:675b": {
        "name": "Mistral Large 3 675B",
        "api_name": "mistral-large-3:675b",
        "thinking": True,
        "context_length": 128000,
        "description": "Large Mistral model",
    },
}


@dataclass
class AIDecision:
    """Container for AI trading decision."""
    symbol: str
    action: str  # BUY, SELL, HOLD, CLOSE
    confidence: float  # 0.0 to 1.0
    reasoning: str
    trend: str  # BULLISH, BEARISH, SIDEWAYS
    entry_quality: str  # STRONG, MODERATE, WEAK
    timestamp: str
    raw_response: str = ""
    thinking_content: str = ""  # Chain-of-thought reasoning from model
    is_fallback: bool = False  # True if using rule-based fallback
    model_used: str = ""  # Which model generated this decision


class AIBrain:
    """
    AI reasoning engine using Ollama Cloud API.
    
    Connects to Ollama Cloud to access powerful thinking models like:
    - MiniMax M2.7: Strong reasoning for trading analysis
    - GLM-4-Plus: Advanced analytical capabilities
    - GPT-120B: Large model with deep reasoning
    - DeepSeek R1: Excellent for structured analysis
    
    These models support chain-of-thought reasoning for better trading decisions.
    """
    
    # Default values for fallback
    DEFAULT_ACTION = "HOLD"
    DEFAULT_CONFIDENCE = 0.5
    
    def __init__(self, config: Optional[ConfigLoader] = None) -> None:
        """Initialize the AI Brain with Ollama Cloud configuration."""
        self.config = config or get_config()
        
        self.model = self.config.ai.model
        self.temperature = self.config.ai.temperature
        self.max_tokens = self.config.ai.max_tokens
        self.fallback_models = self.config.ai.fallback_models
        self.timeout = getattr(self.config.ai, 'timeout_seconds', 120)
        
        # Ollama Cloud API key from env
        self.api_key = self.config.env.ollama_api_key
        
        if not self.api_key:
            logger.warning("OLLAMA_API_KEY not set - cloud models may not work")
        
        # Initialize Ollama Cloud client using native library
        # Direct API access to ollama.com (NOT localhost)
        self.ollama_client = None
        if self.api_key:
            try:
                self.ollama_client = OllamaClient(
                    host="https://ollama.com",
                    headers={'Authorization': f'Bearer {self.api_key}'}
                )
                logger.info("Ollama Cloud client initialized with API key")
            except Exception as e:
                logger.warning(f"Failed to initialize Ollama client: {e}")
        
        # Connection status
        self._connected = False
        self._active_model = self.model
        
        # Recent decisions for context
        self._recent_decisions: List[AIDecision] = []
        self._max_history = 10
        
        logger.info(f"AIBrain initialized with Ollama Cloud model: {self.model}")
        logger.info(f"API Host: https://ollama.com")
    
    def check_connection(self) -> Tuple[bool, str]:
        """
        Check if Ollama Cloud API is reachable and authenticate.
        Uses the native Ollama client library.
        
        Returns:
            Tuple of (connected: bool, message: str)
        """
        if not self.api_key:
            self._connected = False
            return False, "OLLAMA_API_KEY not configured in .env file"
        
        if not self.ollama_client:
            self._connected = False
            return False, "Ollama client not initialized"
        
        # Directly verify with a test chat - most reliable method
        return self._verify_with_test_request()
    
    def _verify_with_test_request(self) -> Tuple[bool, str]:
        """Verify connection by making a simple test chat request using Ollama native client."""
        try:
            if not self.ollama_client:
                self._connected = False
                return False, "Ollama client not initialized"
            
            # Use the Ollama native chat API
            messages = [{"role": "user", "content": "Say 'connected' in one word."}]
            
            response = self.ollama_client.chat(
                model=self.model,
                messages=messages,
                options={"temperature": 0, "num_predict": 10}
            )
            
            if response and 'message' in response:
                self._connected = True
                self._active_model = self.model
                model_info = OLLAMA_CLOUD_MODELS.get(self.model, {})
                model_name = model_info.get('name', self.model)
                return True, f"Connected to Ollama Cloud: {model_name}"
            else:
                self._connected = False
                return False, "Invalid response from Ollama Cloud"
                
        except Exception as e:
            self._connected = False
            error_msg = str(e)
            logger.warning(f"Ollama test request failed: {error_msg}")
            return False, f"Connection test failed: {error_msg}"
    
    def get_active_model(self) -> str:
        """Get the currently active model name."""
        return self._active_model
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the active model."""
        return OLLAMA_CLOUD_MODELS.get(self._active_model, {
            "name": self._active_model,
            "thinking": True,
            "description": "Ollama Cloud model",
        })
    
    def is_connected(self) -> bool:
        """Check if AI is connected."""
        return self._connected
    
    def _build_prompt(
        self,
        symbol: str,
        indicators: SymbolIndicators,
        current_position: Optional[Dict[str, Any]] = None,
        portfolio_state: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Build a detailed prompt for the LLM.
        
        Args:
            symbol: Trading pair
            indicators: Calculated indicators for both timeframes
            current_position: Current position data if any
            portfolio_state: Current portfolio state
            
        Returns:
            Formatted prompt string
        """
        # Extract indicator data
        trend = indicators.trend_tf
        entry = indicators.entry_tf
        
        prompt = f"""You are an expert quantitative trading analyst with deep knowledge of technical analysis.
Analyze the following market data for {symbol} and provide a trading decision.

=== CURRENT PRICE ===
{symbol}: ${indicators.current_price:,.4f}

=== 10-MINUTE TREND TIMEFRAME ===
- RSI(14): {trend.rsi:.1f} ({trend.rsi_trend})
- MACD: Line={trend.macd_line:.4f}, Signal={trend.macd_signal:.4f}, Histogram={trend.macd_histogram:.4f}
- MACD Trend: {trend.macd_trend}, Histogram Rising: {trend.macd_histogram_rising}
- EMA(9): ${trend.ema_short:,.4f}, EMA(21): ${trend.ema_long:,.4f}
- EMA Trend: {trend.ema_trend}, Crossover: {trend.ema_crossover}
- Price vs EMA9: {trend.price_vs_ema_short}, vs EMA21: {trend.price_vs_ema_long}
- Bollinger Bands: Lower=${trend.bb_lower:,.4f}, Mid=${trend.bb_middle:,.4f}, Upper=${trend.bb_upper:,.4f}
- %B (Position in BB): {trend.bb_percent:.2f} ({trend.bb_trend})
- ATR(14): ${trend.atr:.4f} ({trend.atr_percent:.2f}% of price)
- Volume Ratio (vs 20-SMA): {trend.volume_ratio:.2f}x
- Overall Trend: {trend.overall_trend} ({trend.trend_strength})

=== 5-MINUTE ENTRY TIMEFRAME ===
- RSI(14): {entry.rsi:.1f} ({entry.rsi_trend})
- MACD: Line={entry.macd_line:.4f}, Signal={entry.macd_signal:.4f}, Histogram={entry.macd_histogram:.4f}
- MACD Trend: {entry.macd_trend}, Histogram Rising: {entry.macd_histogram_rising}
- EMA(9): ${entry.ema_short:,.4f}, EMA(21): ${entry.ema_long:,.4f}
- EMA Trend: {entry.ema_trend}, Crossover: {entry.ema_crossover}
- Bollinger %B: {entry.bb_percent:.2f} ({entry.bb_trend})
- ATR(14): ${entry.atr:.4f} ({entry.atr_percent:.2f}% of price)
- Overall Trend: {entry.overall_trend} ({entry.trend_strength})

=== RECENT PRICE ACTION (5 bars) ===
- High: ${trend.high_5bars:,.4f}
- Low: ${trend.low_5bars:,.4f}
- Range: ${trend.price_range_5bars:,.4f}
"""

        # Add position info if we have one
        if current_position:
            entry_price = current_position.get('entry_price', 0)
            qty = current_position.get('qty', 0)
            unrealized_pnl = current_position.get('unrealized_pnl', 0)
            unrealized_pnl_pct = current_position.get('unrealized_pnl_pct', 0)
            side = current_position.get('side', 'long')
            
            prompt += f"""
=== CURRENT POSITION ===
- Side: {side.upper()}
- Entry Price: ${entry_price:,.4f}
- Quantity: {qty}
- Unrealized P&L: ${unrealized_pnl:,.2f} ({unrealized_pnl_pct:+.2f}%)
"""
        else:
            prompt += """
=== CURRENT POSITION ===
- No open position for this symbol
"""

        # Add portfolio state
        if portfolio_state:
            total_value = portfolio_state.get('total_value', 100000)
            cash = portfolio_state.get('cash', 100000)
            open_positions = portfolio_state.get('open_positions', 0)
            max_positions = self.config.risk.max_positions
            
            prompt += f"""
=== PORTFOLIO STATE ===
- Total Value: ${total_value:,.2f}
- Available Cash: ${cash:,.2f}
- Open Positions: {open_positions} / {max_positions} max
"""

        # Get scalping settings from config
        scalping_mode = self.config.signals.scalping_mode
        quick_profit_pct = self.config.risk.quick_profit_threshold
        min_profit = self.config.risk.min_profit_to_exit
        stop_loss_pct = self.config.risk.stop_loss_pct
        take_profit_mult = self.config.risk.take_profit_multiplier
        min_confidence = self.config.risk.min_signal_confidence
        
        if scalping_mode:
            # Aggressive scalping mode prompt
            prompt += f"""
=== SCALPING MODE (AGGRESSIVE TRADING) ===
You are in SCALPING MODE. Trade frequently for small, quick profits.
Risk-Reward Ratio: {take_profit_mult}:1 (Stop Loss: {stop_loss_pct}%, Take Profit: {stop_loss_pct * take_profit_mult}%)
Quick Profit Target: {quick_profit_pct}% (or ${min_profit} minimum)

=== TRADING RULES (SCALPING) ===
1. BUY SIGNALS - Be aggressive! Enter when:
   - Any bullish momentum on 5-min (MACD crossing up, RSI rising)
   - Price bouncing from lower Bollinger Band
   - RSI is below {self.config.indicators.rsi_overbought} (not overbought)
   - Confidence >= {min_confidence * 100:.0f}%

2. CLOSE POSITIONS - Take profits quickly!
   - IMMEDIATELY recommend CLOSE when position shows {quick_profit_pct}%+ profit
   - Close if profit >= ${min_profit} even if less than {quick_profit_pct}%
   - Close if momentum is fading (MACD histogram shrinking)
   - Close if RSI reaches extreme levels

3. SELL/SHORT - Only in clear downtrends:
   - Strong bearish momentum confirmed
   - Price rejected from upper Bollinger Band

4. HOLD - Minimize holding time:
   - Only HOLD if entry just happened and need more data
   - Never hold for long periods - trade actively!

=== CRITICAL: PROFIT TAKING ===
When we have an open position:
- If profit >= {quick_profit_pct}% → CLOSE (lock in gains!)
- If momentum reversing → CLOSE (don't give back profits)
- If trend weakening → CLOSE (capital preservation)
- NEVER let a winning trade become a loser

=== YOUR TASK ===
Make a quick decision. In scalping:
1. Favor action over waiting
2. Small profits accumulate - take them!
3. Cut losses fast at {stop_loss_pct}%
4. Trade momentum, not conviction

Respond ONLY with a valid JSON object:
{{
    "action": "BUY" | "SELL" | "HOLD" | "CLOSE",
    "confidence": 0.0 to 1.0,
    "reasoning": "Brief explanation of your decision",
    "trend": "BULLISH" | "BEARISH" | "SIDEWAYS",
    "entry_quality": "STRONG" | "MODERATE" | "WEAK"
}}

Respond with ONLY the JSON, no additional text."""
        else:
            # Standard swing trading prompt
            prompt += f"""
=== TRADING RULES (SWING TRADING) ===
1. Only recommend BUY if:
   - 10-min trend is BULLISH or SIDEWAYS with bullish momentum
   - 5-min entry shows good timing (RSI not overbought, MACD positive)
   - No current long position exists
   - Confidence is {min_confidence * 100:.0f}% or higher

2. Only recommend SELL/SHORT if:
   - 10-min trend is BEARISH or SIDEWAYS with bearish momentum
   - 5-min entry shows good timing (RSI not oversold, MACD negative)
   - Confidence is {min_confidence * 100:.0f}% or higher

3. Recommend CLOSE if:
   - We have an open position AND
   - Trend is reversing against us OR
   - Take profit level ({stop_loss_pct * take_profit_mult}%) is near OR
   - Stop loss ({stop_loss_pct}%) is about to be hit

4. Recommend HOLD if:
   - Conditions are unclear
   - Conflicting signals between timeframes
   - Low confidence in any direction

=== YOUR TASK ===
Analyze all the data above using chain-of-thought reasoning. Consider:
1. Is the 10-min trend supporting entry direction?
2. Is the 5-min timing optimal for entry?
3. Are momentum indicators (MACD, RSI) aligned?
4. What is the risk/reward based on ATR and Bollinger Bands?
5. Should we enter, exit, or wait?

Respond ONLY with a valid JSON object in this exact format:
{{
    "action": "BUY" | "SELL" | "HOLD" | "CLOSE",
    "confidence": 0.0 to 1.0,
    "reasoning": "One clear sentence explaining your decision",
    "trend": "BULLISH" | "BEARISH" | "SIDEWAYS",
    "entry_quality": "STRONG" | "MODERATE" | "WEAK"
}}

Respond with ONLY the JSON, no additional text before or after."""

        return prompt
    
    def _call_ollama_cloud_with_model(self, prompt: str, model: str) -> Tuple[Optional[str], str]:
        """
        Call Ollama Cloud API with a specific model.
        
        Args:
            prompt: The prompt to send
            
        Returns:
            Tuple of (response_text, thinking_content)
        """
        if not self.api_key:
            logger.error("OLLAMA_API_KEY not configured")
            return None, ""
        
        if not self.ollama_client:
            logger.error("Ollama client not initialized")
            return None, ""
        
        try:
            def call_chat() -> Any:
                return self.ollama_client.chat(
                    model=model,
                    messages=messages,
                    options={
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens,
                    }
                )

            # Build messages for chat (Ollama native format)
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert quantitative trading analyst. Analyze market data and provide precise trading recommendations in JSON format. Use chain-of-thought reasoning to analyze all indicators before making a decision."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ]
            
            logger.debug(f"Calling Ollama Cloud with model: {model}")
            
            # Use Ollama native client chat method
            response = call_chat()
            
            if response:
                # Extract content from Ollama response format
                content = ""
                thinking = ""
                
                if 'message' in response:
                    message = response['message']
                    content = message.get('content', '')
                    
                    # Some models return thinking in a separate field
                    if 'reasoning' in message:
                        thinking = message.get('reasoning', '')
                    elif 'thinking' in message:
                        thinking = message.get('thinking', '')
                
                # Fallback for direct content in response
                elif 'response' in response:
                    content = response.get('response', '')
                elif 'content' in response:
                    content = response.get('content', '')
                
                logger.debug(f"Ollama Cloud response received: {len(content)} chars")
                return content, thinking
            
            return None, ""
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error calling Ollama Cloud: {error_msg}")
            
            # Check for specific error types
            if "401" in error_msg or "unauthorized" in error_msg.lower():
                logger.error("Ollama Cloud authentication failed - check API key")
            elif "429" in error_msg or "rate" in error_msg.lower():
                logger.warning(f"Ollama Cloud rate limit on {model} - retrying in 2s...")
                time.sleep(2)
                try:
                    response = call_chat()
                    if response and 'message' in response:
                        message = response['message']
                        content = message.get('content', '')
                        thinking = message.get('reasoning', '') or message.get('thinking', '')
                        return content, thinking
                except Exception:
                    pass
            elif "timeout" in error_msg.lower():
                logger.error(f"Ollama Cloud request timed out")
            
            return None, ""

    def _call_ollama_cloud(self, prompt: str) -> Tuple[Optional[str], str]:
        """
        Call Ollama Cloud API with active model and automatic failover models.
        """
        model_candidates = [self._active_model] + [
            m for m in self.fallback_models if m != self._active_model
        ]
        last_model_error = ""

        for model in model_candidates:
            response, thinking = self._call_ollama_cloud_with_model(prompt, model)
            if response:
                if model != self._active_model:
                    logger.warning(f"Switched active model from {self._active_model} to {model}")
                    self._active_model = model
                return response, thinking
            last_model_error = model

        if last_model_error:
            logger.error(f"All Ollama Cloud model attempts failed (last tried: {last_model_error})")
        return None, ""
    
    def _parse_response(self, response: str, symbol: str, thinking: str = "") -> AIDecision:
        """
        Parse the LLM response into an AIDecision.
        
        Args:
            response: Raw LLM response
            symbol: Trading symbol
            thinking: Optional chain-of-thought content from model
            
        Returns:
            AIDecision object
        """
        timestamp = datetime.now().isoformat()
        
        try:
            # Try to extract JSON from response
            # Handle cases where model adds text before/after JSON
            # Support nested JSON and multiline
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
            
            if json_match:
                json_str = json_match.group()
                data = json.loads(json_str)
                
                # Validate and extract fields
                action = data.get("action", "HOLD").upper()
                if action not in ["BUY", "SELL", "HOLD", "CLOSE"]:
                    action = "HOLD"
                
                confidence = float(data.get("confidence", 0.5))
                confidence = max(0.0, min(1.0, confidence))
                
                reasoning = str(data.get("reasoning", "No reasoning provided"))
                
                trend = data.get("trend", "SIDEWAYS").upper()
                if trend not in ["BULLISH", "BEARISH", "SIDEWAYS"]:
                    trend = "SIDEWAYS"
                
                entry_quality = data.get("entry_quality", "WEAK").upper()
                if entry_quality not in ["STRONG", "MODERATE", "WEAK"]:
                    entry_quality = "WEAK"
                
                return AIDecision(
                    symbol=symbol,
                    action=action,
                    confidence=confidence,
                    reasoning=reasoning,
                    trend=trend,
                    entry_quality=entry_quality,
                    timestamp=timestamp,
                    raw_response=response,
                    thinking_content=thinking,
                    is_fallback=False,
                    model_used=self._active_model,
                )
            
            # No valid JSON found
            logger.warning(f"No valid JSON in response: {response[:200]}")
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON: {e}")
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
        
        # Return default HOLD decision
        return AIDecision(
            symbol=symbol,
            action="HOLD",
            confidence=0.5,
            reasoning="Could not parse AI response, defaulting to HOLD",
            trend="SIDEWAYS",
            entry_quality="WEAK",
            timestamp=timestamp,
            raw_response=response,
            thinking_content=thinking,
            is_fallback=True,
            model_used=self._active_model,
        )
    
    def _rule_based_decision(
        self,
        symbol: str,
        indicators: SymbolIndicators,
    ) -> AIDecision:
        """
        Generate a rule-based decision when Ollama is unavailable.
        
        Args:
            symbol: Trading symbol
            indicators: Calculated indicators
            
        Returns:
            AIDecision based on rules
        """
        timestamp = datetime.now().isoformat()
        
        trend = indicators.trend_tf
        entry = indicators.entry_tf
        
        # Default values
        action = "HOLD"
        confidence = 0.5
        reasoning = "Rule-based analysis (AI unavailable)"
        trend_dir = "SIDEWAYS"
        quality = "WEAK"
        
        try:
            # Simple rule-based logic
            bullish_signals = 0
            bearish_signals = 0
            
            # Check trend timeframe
            if trend.overall_trend == "BULLISH":
                bullish_signals += 2
            elif trend.overall_trend == "BEARISH":
                bearish_signals += 2
            
            # Check entry timeframe
            if entry.overall_trend == "BULLISH":
                bullish_signals += 1
            elif entry.overall_trend == "BEARISH":
                bearish_signals += 1
            
            # Check RSI - more aggressive thresholds
            if entry.rsi < 45:  # Was 35
                bullish_signals += 1
            elif entry.rsi > 55:  # Was 65
                bearish_signals += 1
            
            # Check MACD
            if entry.macd_trend == "BULLISH":  # Removed histogram requirement
                bullish_signals += 1
            elif entry.macd_trend == "BEARISH":
                bearish_signals += 1
            
            # Check EMA crossover
            if entry.ema_crossover == "GOLDEN_CROSS":
                bullish_signals += 2
            elif entry.ema_crossover == "DEATH_CROSS":
                bearish_signals += 2
            
            # Log indicator values for debugging
            logger.info(
                f"[{symbol}] RSI={entry.rsi:.1f}, MACD={entry.macd_trend}, "
                f"EMA={entry.ema_crossover}, Trend={trend.overall_trend}, "
                f"Entry={entry.overall_trend} → Bull={bullish_signals} Bear={bearish_signals}"
            )
            
            # Make decision - more aggressive for testing
            # NOTE: Long-only strategy since short selling requires margin approval
            total_signals = bullish_signals + bearish_signals
            signal_diff = bullish_signals - bearish_signals
            
            # More aggressive thresholds for active trading - TESTING MODE
            # Lower thresholds to trigger trades for testing the system
            if signal_diff >= 2:  # Net 2+ bullish signals
                action = "BUY"
                confidence = min(0.85, 0.55 + signal_diff * 0.08)
                trend_dir = "BULLISH"
                quality = "STRONG" if signal_diff >= 4 else "MODERATE"
                reasoning = f"Rule-based BUY: {bullish_signals} bullish vs {bearish_signals} bearish (net +{signal_diff})"
                
            elif bullish_signals >= 1 and bearish_signals == 0:
                # ANY bullish signal with no bearish - BUY for testing
                action = "BUY"
                confidence = 0.55
                trend_dir = "BULLISH"
                quality = "WEAK"
                reasoning = f"Rule-based BUY: {bullish_signals} bullish signals, no bearish resistance"
                
            else:
                # No clear bullish signal, HOLD (not SELL since we're long-only)
                action = "HOLD"
                confidence = 0.45
                trend_dir = trend.overall_trend
                quality = "WEAK"
                reasoning = f"Rule-based HOLD: No clear bullish signal ({bullish_signals} bull, {bearish_signals} bear)"
            
        except Exception as e:
            logger.error(f"Error in rule-based decision: {e}")
        
        return AIDecision(
            symbol=symbol,
            action=action,
            confidence=confidence,
            reasoning=reasoning,
            trend=trend_dir,
            entry_quality=quality,
            timestamp=timestamp,
            raw_response="",
            thinking_content="",
            is_fallback=True,
            model_used="rule_based",
        )
    
    def get_decision(
        self,
        symbol: str,
        indicators: SymbolIndicators,
        current_position: Optional[Dict[str, Any]] = None,
        portfolio_state: Optional[Dict[str, Any]] = None,
    ) -> AIDecision:
        """
        Get an AI decision for a trading symbol using Ollama Cloud.
        
        Args:
            symbol: Trading pair
            indicators: Calculated indicators for both timeframes
            current_position: Current position data if any
            portfolio_state: Current portfolio state
            
        Returns:
            AIDecision with action, confidence, and reasoning
        """
        # Check if we have valid indicator data
        if not indicators.trend_tf.has_data or not indicators.entry_tf.has_data:
            logger.warning(f"Insufficient indicator data for {symbol}")
            return AIDecision(
                symbol=symbol,
                action="HOLD",
                confidence=0.3,
                reasoning="Insufficient data to make a decision",
                trend="SIDEWAYS",
                entry_quality="WEAK",
                timestamp=datetime.now().isoformat(),
                thinking_content="",
                is_fallback=True,
                model_used="none",
            )
        
        # Try to get AI decision from Ollama Cloud
        if self._connected:
            try:
                prompt = self._build_prompt(symbol, indicators, current_position, portfolio_state)
                logger.debug(f"AI prompt for {symbol}:\n{prompt}")
                
                # Call Ollama Cloud API
                response, thinking = self._call_ollama_cloud(prompt)
                
                if response:
                    logger.debug(f"AI response for {symbol}:\n{response}")
                    if thinking:
                        logger.debug(f"AI thinking for {symbol}:\n{thinking}")
                    
                    decision = self._parse_response(response, symbol, thinking)
                    
                    # Store in history
                    self._recent_decisions.append(decision)
                    if len(self._recent_decisions) > self._max_history:
                        self._recent_decisions.pop(0)
                    
                    return decision
                    
            except Exception as e:
                logger.error(f"Error getting AI decision for {symbol}: {e}")
        
        # Fallback to rule-based decision
        logger.info(f"Using rule-based decision for {symbol} (Ollama Cloud unavailable)")
        decision = self._rule_based_decision(symbol, indicators)
        
        # Store in history
        self._recent_decisions.append(decision)
        if len(self._recent_decisions) > self._max_history:
            self._recent_decisions.pop(0)
        
        return decision
    
    def get_recent_decisions(self, limit: int = 5) -> List[AIDecision]:
        """Get recent decisions from history."""
        return self._recent_decisions[-limit:]
    
    def get_decision_for_symbol(self, symbol: str) -> Optional[AIDecision]:
        """Get the most recent decision for a specific symbol."""
        for decision in reversed(self._recent_decisions):
            if decision.symbol == symbol:
                return decision
        return None
