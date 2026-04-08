# core/signal_engine.py
"""
Signal engine that combines indicators and AI decisions into final trading signals.
Implements dual-timeframe confirmation and signal validation.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from .config_loader import get_config, ConfigLoader
from .indicators import IndicatorValues, SymbolIndicators
from .ai_brain import AIDecision


logger = logging.getLogger(__name__)


@dataclass
class SignalResult:
    """Container for the final trading signal."""
    symbol: str
    action: str  # BUY, SELL, HOLD, CLOSE
    confidence: float  # 0.0 to 1.0
    reasoning: str
    
    # Trend information
    trend_direction: str  # BULLISH, BEARISH, SIDEWAYS
    trend_strength: str  # STRONG, MODERATE, WEAK
    entry_quality: str  # STRONG, MODERATE, WEAK
    
    # Timeframe data
    trend_tf_summary: Dict[str, Any] = field(default_factory=dict)
    entry_tf_summary: Dict[str, Any] = field(default_factory=dict)
    
    # Signal validation
    is_confirmed: bool = False  # True if all confirmations pass
    confirmation_details: List[str] = field(default_factory=list)
    rejection_reasons: List[str] = field(default_factory=list)
    
    # Timestamps
    timestamp: str = ""
    
    # Source
    signal_source: str = "AI"  # AI or RULE_BASED
    ai_decision: Optional[AIDecision] = None


class SignalEngine:
    """
    Combines indicators and AI decisions into validated trading signals.
    Implements dual-timeframe confirmation and signal filtering.
    """
    
    def __init__(self, config: Optional[ConfigLoader] = None) -> None:
        """Initialize the signal engine."""
        self.config = config or get_config()
        
        # Get signal settings from config
        self.require_trend_confirmation = self.config.signals.require_trend_confirmation
        self.require_volume_confirmation = self.config.signals.require_volume_confirmation
        self.min_rsi_for_buy = self.config.signals.min_rsi_for_buy
        self.max_rsi_for_sell = self.config.signals.max_rsi_for_sell
        self.min_confidence = self.config.risk.min_signal_confidence
        
        logger.info("SignalEngine initialized")
    
    def generate_signal(
        self,
        symbol: str,
        indicators: SymbolIndicators,
        ai_decision: AIDecision,
        current_position: Optional[Dict[str, Any]] = None,
    ) -> SignalResult:
        """
        Generate a final trading signal by combining indicators and AI decision.
        
        Args:
            symbol: Trading pair
            indicators: Calculated indicators for both timeframes
            ai_decision: Decision from AI brain
            current_position: Current position data if any
            
        Returns:
            SignalResult with validated action and reasoning
        """
        timestamp = datetime.now().isoformat()
        
        # Initialize result
        result = SignalResult(
            symbol=symbol,
            action="HOLD",
            confidence=0.5,
            reasoning="Analyzing...",
            trend_direction=ai_decision.trend,
            trend_strength=indicators.trend_tf.trend_strength,
            entry_quality=ai_decision.entry_quality,
            timestamp=timestamp,
            signal_source="AI" if not ai_decision.is_fallback else "RULE_BASED",
            ai_decision=ai_decision,
        )
        
        # Build timeframe summaries
        result.trend_tf_summary = self._build_tf_summary(indicators.trend_tf, "10-min")
        result.entry_tf_summary = self._build_tf_summary(indicators.entry_tf, "5-min")
        
        # Start with AI decision
        proposed_action = ai_decision.action
        confidence = ai_decision.confidence
        
        # Run all validations
        confirmations = []
        rejections = []
        
        # 1. Confidence check
        if confidence >= self.min_confidence:
            confirmations.append(f"Confidence {confidence:.0%} meets minimum {self.min_confidence:.0%}")
        else:
            rejections.append(f"Confidence {confidence:.0%} below minimum {self.min_confidence:.0%}")
        
        # 2. Trend confirmation (if required)
        if self.require_trend_confirmation and proposed_action in ["BUY", "SELL"]:
            trend_confirms, trend_reason = self._check_trend_confirmation(
                proposed_action, indicators.trend_tf, indicators.entry_tf
            )
            if trend_confirms:
                confirmations.append(trend_reason)
            else:
                rejections.append(trend_reason)
        
        # 3. RSI validation
        rsi_valid, rsi_reason = self._check_rsi_validation(
            proposed_action, indicators.entry_tf
        )
        if rsi_valid:
            confirmations.append(rsi_reason)
        else:
            rejections.append(rsi_reason)
        
        # 4. MACD validation
        macd_valid, macd_reason = self._check_macd_validation(
            proposed_action, indicators.entry_tf
        )
        if macd_valid:
            confirmations.append(macd_reason)
        else:
            rejections.append(macd_reason)

        # Step 4b: ADX gate
        adx = getattr(indicators.entry_tf, 'adx', 25.0)
        if proposed_action in ["BUY", "SELL"] and adx < 20:
            rejections.append(f"ADX={adx:.1f} below 20 — choppy market, no trade")
        elif proposed_action in ["BUY", "SELL"]:
            confirmations.append(f"ADX={adx:.1f} confirms trending market")

        # Step 4c: Volume spike check (two-candle increasing volume)
        # Only allow BUY if last candle volume > prior candle volume (buyers stepping in)
        volume_increasing = getattr(indicators.entry_tf, 'volume_increasing', True)
        if proposed_action == "BUY" and not volume_increasing:
            rejections.append("Volume declining on entry candle — buyers fading")
        elif proposed_action == "BUY":
            confirmations.append("Volume increasing — buyer participation confirmed")
        
        # 5. Volume confirmation (if required)
        if self.require_volume_confirmation and proposed_action in ["BUY", "SELL"]:
            volume_confirms, volume_reason = self._check_volume_confirmation(
                indicators.entry_tf
            )
            if volume_confirms:
                confirmations.append(volume_reason)
            else:
                rejections.append(volume_reason)
        
        # 6. Position-specific checks
        if current_position:
            position_valid, position_reason = self._check_position_context(
                proposed_action, current_position
            )
            if position_valid:
                confirmations.append(position_reason)
            else:
                rejections.append(position_reason)
        
        # Determine final action
        result.confirmation_details = confirmations
        result.rejection_reasons = rejections
        
        # Count critical rejections (confidence, trend, RSI)
        critical_rejections = len([r for r in rejections if any(
            x in r.lower() for x in ['confidence', 'trend', 'rsi', 'adx']
        )])
        
        if proposed_action in ["BUY", "SELL"]:
            if critical_rejections == 0 and len(rejections) <= 1:
                # Signal confirmed
                result.action = proposed_action
                result.confidence = confidence
                result.is_confirmed = True
                result.reasoning = ai_decision.reasoning
            else:
                # Signal rejected, downgrade to HOLD
                result.action = "HOLD"
                result.confidence = max(0.3, confidence - 0.2)
                result.is_confirmed = False
                result.reasoning = f"Signal rejected: {'; '.join(rejections[:2])}"
        
        elif proposed_action == "CLOSE":
            # CLOSE signals are more lenient
            if len(rejections) <= 2:
                result.action = "CLOSE"
                result.confidence = confidence
                result.is_confirmed = True
                result.reasoning = ai_decision.reasoning
            else:
                result.action = "HOLD"
                result.confidence = 0.5
                result.is_confirmed = False
                result.reasoning = "CLOSE signal not confirmed"
        
        else:
            # HOLD
            result.action = "HOLD"
            result.confidence = confidence
            result.is_confirmed = True
            result.reasoning = ai_decision.reasoning
        
        logger.info(
            f"Signal for {symbol}: {result.action} (conf: {result.confidence:.0%}) - "
            f"Confirmed: {result.is_confirmed}"
        )
        
        return result
    
    def _build_tf_summary(self, indicators: IndicatorValues, label: str) -> Dict[str, Any]:
        """Build a summary dictionary for a timeframe."""
        return {
            "label": label,
            "rsi": round(indicators.rsi, 1),
            "rsi_trend": indicators.rsi_trend,
            "macd_trend": indicators.macd_trend,
            "macd_rising": indicators.macd_histogram_rising,
            "ema_trend": indicators.ema_trend,
            "ema_crossover": indicators.ema_crossover,
            "bb_percent": round(indicators.bb_percent, 2),
            "bb_trend": indicators.bb_trend,
            "overall_trend": indicators.overall_trend,
            "trend_strength": indicators.trend_strength,
        }
    
    def _check_trend_confirmation(
        self,
        action: str,
        trend_tf: IndicatorValues,
        entry_tf: IndicatorValues,
    ) -> tuple[bool, str]:
        """
        Check if the trend timeframe confirms the entry direction.
        
        Returns:
            Tuple of (is_confirmed, reason_message)
        """
        if action == "BUY":
            # For BUY, trend should be BULLISH or SIDEWAYS with bullish momentum
            if trend_tf.overall_trend == "BULLISH":
                return True, "10-min trend is BULLISH (confirms BUY)"
            elif trend_tf.overall_trend == "SIDEWAYS":
                adx = getattr(trend_tf, "adx", 0.0)
                if trend_tf.macd_histogram_rising and adx > 22:
                    return True, "10-min trend is SIDEWAYS with bullish momentum"
                return False, "SIDEWAYS + low ADX — insufficient momentum"
            else:
                return False, f"10-min trend is {trend_tf.overall_trend} (contradicts BUY)"
        
        elif action == "SELL":
            # For SELL, trend should be BEARISH or SIDEWAYS with bearish momentum
            if trend_tf.overall_trend == "BEARISH":
                return True, "10-min trend is BEARISH (confirms SELL)"
            elif trend_tf.overall_trend == "SIDEWAYS":
                # Check for bearish momentum in sideways
                if not trend_tf.macd_histogram_rising and trend_tf.ema_trend != "BULLISH":
                    return True, "10-min trend is SIDEWAYS with bearish momentum"
                else:
                    return False, "10-min trend is SIDEWAYS without bearish momentum"
            else:
                return False, f"10-min trend is {trend_tf.overall_trend} (contradicts SELL)"
        
        return True, "No trend confirmation needed"
    
    def _check_rsi_validation(
        self,
        action: str,
        entry_tf: IndicatorValues,
    ) -> tuple[bool, str]:
        """
        Check if RSI conditions are valid for the action.
        
        Returns:
            Tuple of (is_valid, reason_message)
        """
        rsi = entry_tf.rsi
        
        if action == "BUY":
            # For BUY, RSI should not be overbought
            # We also prefer RSI to be in neutral to oversold territory
            if rsi > 70:
                return False, f"RSI at {rsi:.1f} is overbought (>70, risky for BUY)"
            elif rsi <= self.min_rsi_for_buy:
                return True, f"RSI at {rsi:.1f} shows oversold conditions (good for BUY)"
            else:
                return True, f"RSI at {rsi:.1f} is neutral (acceptable for BUY)"
        
        elif action == "SELL":
            # For SELL, RSI should not be oversold
            if rsi < 30:
                return False, f"RSI at {rsi:.1f} is oversold (<30, risky for SELL)"
            elif rsi >= self.max_rsi_for_sell:
                return True, f"RSI at {rsi:.1f} shows overbought conditions (good for SELL)"
            else:
                return True, f"RSI at {rsi:.1f} is neutral (acceptable for SELL)"
        
        return True, f"RSI at {rsi:.1f} (no specific requirement for {action})"
    
    def _check_macd_validation(
        self,
        action: str,
        entry_tf: IndicatorValues,
    ) -> tuple[bool, str]:
        """
        Check if MACD conditions support the action.
        
        Returns:
            Tuple of (is_valid, reason_message)
        """
        macd_trend = entry_tf.macd_trend
        histogram = entry_tf.macd_histogram
        rising = entry_tf.macd_histogram_rising
        
        if action == "BUY":
            # For BUY, MACD should be bullish or turning bullish
            if macd_trend == "BULLISH" and histogram > 0 and rising:
                return True, "MACD is BULLISH with rising histogram (strong BUY)"
            elif macd_trend == "BULLISH" or (histogram > 0 and rising):
                return True, "MACD shows bullish momentum"
            elif histogram < 0 and rising:
                return True, "MACD histogram rising from below (potential reversal)"
            else:
                return False, f"MACD is {macd_trend}, histogram not supportive"
        
        elif action == "SELL":
            # For SELL, MACD should be bearish or turning bearish
            if macd_trend == "BEARISH" and histogram < 0 and not rising:
                return True, "MACD is BEARISH with falling histogram (strong SELL)"
            elif macd_trend == "BEARISH" or (histogram < 0 and not rising):
                return True, "MACD shows bearish momentum"
            elif histogram > 0 and not rising:
                return True, "MACD histogram falling from above (potential reversal)"
            else:
                return False, f"MACD is {macd_trend}, histogram not supportive"
        
        return True, f"MACD is {macd_trend} (no specific requirement for {action})"
    
    def _check_volume_confirmation(
        self,
        entry_tf: IndicatorValues,
    ) -> tuple[bool, str]:
        """
        Check if volume confirms the trade.
        
        Returns:
            Tuple of (is_confirmed, reason_message)
        """
        volume_ratio = entry_tf.volume_ratio
        
        if volume_ratio >= 1.2:
            return True, f"Volume {volume_ratio:.1f}x average (above average confirms move)"
        elif volume_ratio >= 0.8:
            return True, f"Volume {volume_ratio:.1f}x average (normal)"
        else:
            return False, f"Volume {volume_ratio:.1f}x average (low volume, weak signal)"
    
    def _check_position_context(
        self,
        action: str,
        current_position: Dict[str, Any],
    ) -> tuple[bool, str]:
        """
        Check if action makes sense given current position.
        
        Returns:
            Tuple of (is_valid, reason_message)
        """
        side = current_position.get('side', 'long')
        unrealized_pnl_pct = current_position.get('unrealized_pnl_pct', 0)
        
        if action == "BUY":
            if side == 'long':
                return False, f"Already in LONG position (P&L: {unrealized_pnl_pct:+.2f}%)"
            else:
                return True, "No conflicting long position"
        
        elif action == "SELL":
            if side == 'short':
                return False, f"Already in SHORT position (P&L: {unrealized_pnl_pct:+.2f}%)"
            else:
                return True, "No conflicting short position"
        
        elif action == "CLOSE":
            return True, f"Closing existing {side.upper()} position (P&L: {unrealized_pnl_pct:+.2f}%)"
        
        return True, "Position context is valid"

    def _triple_confirmation_check(
        self, symbol: str, indicators: SymbolIndicators, symbol_meta: Dict[str, Any]
    ) -> tuple[bool, str]:
        """
        Returns (passed: bool, reason: str).
        All 3 confirmations must pass for a BUY to proceed.
        """
        del symbol
        trend = indicators.trend_tf
        entry = indicators.entry_tf
        reasons: List[str] = []

        # CONFIRMATION 1: Trend Alignment
        trend_ok = (
            trend.overall_trend == "BULLISH"
            and entry.overall_trend in ["BULLISH", "SIDEWAYS"]
        )
        if not trend_ok:
            reasons.append(
                f"Trend conflict: 10m={trend.overall_trend}, 5m={entry.overall_trend}"
            )

        # CONFIRMATION 2: Momentum
        momentum_ok = (
            entry.rsi < 70 and
            entry.rsi > 30 and
            entry.macd_trend == "BULLISH"
        )
        if not momentum_ok:
            reasons.append(
                f"Momentum weak: RSI={entry.rsi:.1f}, MACD={entry.macd_trend}"
            )

        # CONFIRMATION 3: Volume
        vol_ratio = symbol_meta.get("vol_ratio", 1.0)
        volume_ok = vol_ratio >= 1.2
        if not volume_ok:
            reasons.append(f"Low volume: {vol_ratio:.2f}x (need 1.2x)")

        all_passed = trend_ok and momentum_ok and volume_ok
        fail_reason = " | ".join(reasons) if reasons else "All confirmations passed"
        return all_passed, fail_reason
    
    def quick_signal(
        self,
        indicators: SymbolIndicators,
    ) -> str:
        """
        Get a quick signal summary without full AI analysis.
        Used for dashboard display.
        
        Args:
            indicators: Calculated indicators
            
        Returns:
            Quick signal string: "BUY", "SELL", "HOLD", "WATCH"
        """
        trend = indicators.trend_tf
        entry = indicators.entry_tf
        
        # Simple scoring
        bullish = 0
        bearish = 0
        
        # Trend TF
        if trend.overall_trend == "BULLISH":
            bullish += 2
        elif trend.overall_trend == "BEARISH":
            bearish += 2
        
        # Entry TF
        if entry.overall_trend == "BULLISH":
            bullish += 1
        elif entry.overall_trend == "BEARISH":
            bearish += 1
        
        # RSI
        if entry.rsi < 35:
            bullish += 1
        elif entry.rsi > 65:
            bearish += 1
        
        # MACD
        if entry.macd_trend == "BULLISH" and entry.macd_histogram_rising:
            bullish += 1
        elif entry.macd_trend == "BEARISH" and not entry.macd_histogram_rising:
            bearish += 1
        
        # Determine signal
        if bullish >= 4 and bearish <= 1:
            return "BUY"
        elif bearish >= 4 and bullish <= 1:
            return "SELL"
        elif bullish >= 3 or bearish >= 3:
            return "WATCH"
        else:
            return "HOLD"
    
    def get_macd_arrow(self, indicators: IndicatorValues) -> str:
        """Get MACD trend arrow for display."""
        if indicators.macd_trend == "BULLISH" and indicators.macd_histogram_rising:
            return "↑"
        elif indicators.macd_trend == "BEARISH" and not indicators.macd_histogram_rising:
            return "↓"
        else:
            return "→"
