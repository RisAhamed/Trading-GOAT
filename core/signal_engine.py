# core/signal_engine.py
"""
Signal engine that combines indicators and AI decisions into final trading signals.
Implements dual-timeframe confirmation and signal validation.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .config_loader import get_config, ConfigLoader
from .indicators import IndicatorValues, SymbolIndicators
from .ai_brain import AIDecision

logger = logging.getLogger(__name__)


@dataclass
class SignalResult:
    """Container for the final trading signal."""

    symbol: str
    action: str           # BUY, SELL, HOLD, CLOSE
    confidence: float     # 0.0 to 1.0
    reasoning: str

    # Trend information
    trend_direction: str  # BULLISH, BEARISH, SIDEWAYS
    trend_strength: str   # STRONG, MODERATE, WEAK
    entry_quality: str    # STRONG, MODERATE, WEAK

    # Timeframe data
    trend_tf_summary: Dict[str, Any] = field(default_factory=dict)
    entry_tf_summary: Dict[str, Any] = field(default_factory=dict)

    # Signal validation
    is_confirmed: bool = False
    confirmation_details: List[str] = field(default_factory=list)
    rejection_reasons: List[str] = field(default_factory=list)

    # Timestamps
    timestamp: str = ""

    # Source
    signal_source: str = "AI"           # AI or RULE_BASED
    ai_decision: Optional[AIDecision] = None


class SignalEngine:
    """
    Combines indicators and AI decisions into validated trading signals.
    Implements dual-timeframe confirmation and signal filtering.
    """

    def __init__(self, config: Optional[ConfigLoader] = None) -> None:
        """Initialize the signal engine."""
        self.config = config or get_config()

        # Get signal settings from config with safe getattr fallbacks
        signals_cfg = getattr(self.config, "signals", None)
        risk_cfg    = getattr(self.config, "risk", None)

        self.require_trend_confirmation = getattr(
            signals_cfg, "require_trend_confirmation", False
        )
        self.require_volume_confirmation = getattr(
            signals_cfg, "require_volume_confirmation", False
        )
        self.min_rsi_for_buy  = getattr(signals_cfg, "min_rsi_for_buy",  65)
        self.max_rsi_for_sell = getattr(signals_cfg, "max_rsi_for_sell", 40)
        self.min_confidence   = getattr(risk_cfg,    "min_signal_confidence", 0.25)

        # ADX thresholds — use config values, fall back to safe defaults
        self.adx_min_threshold      = getattr(signals_cfg, "adx_min_threshold", 15)
        self.sideways_adx_threshold = getattr(signals_cfg, "sideways_adx_threshold", 15)

        logger.info("SignalEngine initialized")

    # ------------------------------------------------------------------ #
    #  PUBLIC: generate_signal                                            #
    # ------------------------------------------------------------------ #
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
            symbol:           Trading pair
            indicators:       Calculated indicators for both timeframes
            ai_decision:      Decision from AI brain
            current_position: Current position data if any

        Returns:
            SignalResult with validated action and reasoning
        """
        timestamp = datetime.now().isoformat()

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

        result.trend_tf_summary = self._build_tf_summary(indicators.trend_tf, "10-min")
        result.entry_tf_summary = self._build_tf_summary(indicators.entry_tf,  "5-min")

        proposed_action = ai_decision.action
        confidence      = ai_decision.confidence

        confirmations: List[str] = []
        rejections:    List[str] = []

        # ── 1. Confidence gate ─────────────────────────────────────────────
        if confidence >= self.min_confidence:
            confirmations.append(
                f"Confidence {confidence:.0%} meets minimum {self.min_confidence:.0%}"
            )
        else:
            rejections.append(
                f"Confidence {confidence:.0%} below minimum {self.min_confidence:.0%}"
            )

        # ── 2. Trend confirmation (optional) ─────────────────────────────
        if self.require_trend_confirmation and proposed_action in ("BUY", "SELL"):
            trend_ok, trend_reason = self._check_trend_confirmation(
                proposed_action, indicators.trend_tf, indicators.entry_tf
            )
            (confirmations if trend_ok else rejections).append(trend_reason)

        # ── 3. RSI validation ─────────────────────────────────────────
        rsi_ok, rsi_reason = self._check_rsi_validation(
            proposed_action, indicators.entry_tf
        )
        (confirmations if rsi_ok else rejections).append(rsi_reason)

        # ── 4. MACD validation ─────────────────────────────────────────
        macd_ok, macd_reason = self._check_macd_validation(
            proposed_action, indicators.entry_tf
        )
        (confirmations if macd_ok else rejections).append(macd_reason)

        # ── 4b. ADX gate ────────────────────────────────────────────────
        adx = getattr(indicators.entry_tf, "adx", 25.0)
        if proposed_action in ("BUY", "SELL"):
            if adx < self.adx_min_threshold:
                rejections.append(
                    f"ADX={adx:.1f} below {self.adx_min_threshold} — choppy market, no trade"
                )
            else:
                confirmations.append(f"ADX={adx:.1f} confirms trending market")

        # ── 4c. Volume spike check (two-candle rising volume for BUY) ─────
        volume_increasing = getattr(indicators.entry_tf, "volume_increasing", True)
        if proposed_action == "BUY":
            if not volume_increasing:
                rejections.append("Volume declining on entry candle — buyers fading")
            else:
                confirmations.append("Volume increasing — buyer participation confirmed")

        # ── 5. Volume confirmation (optional) ───────────────────────────
        if self.require_volume_confirmation and proposed_action in ("BUY", "SELL"):
            vol_ok, vol_reason = self._check_volume_confirmation(indicators.entry_tf)
            (confirmations if vol_ok else rejections).append(vol_reason)

        # ── 6. Position context ─────────────────────────────────────────
        if current_position:
            pos_ok, pos_reason = self._check_position_context(
                proposed_action, current_position
            )
            (confirmations if pos_ok else rejections).append(pos_reason)

        # ── Decision ───────────────────────────────────────────────────
        result.confirmation_details = confirmations
        result.rejection_reasons    = rejections

        critical_rejections = sum(
            1 for r in rejections
            if any(x in r.lower() for x in ("confidence", "trend", "rsi", "adx"))
        )

        if proposed_action in ("BUY", "SELL"):
            if critical_rejections == 0 and len(rejections) <= 1:
                result.action       = proposed_action
                result.confidence   = confidence
                result.is_confirmed = True
                result.reasoning    = ai_decision.reasoning
            else:
                result.action       = "HOLD"
                result.confidence   = max(0.3, confidence - 0.2)
                result.is_confirmed = False
                result.reasoning    = f"Signal rejected: {'; '.join(rejections[:2])}"

        elif proposed_action == "CLOSE":
            if len(rejections) <= 2:
                result.action       = "CLOSE"
                result.confidence   = confidence
                result.is_confirmed = True
                result.reasoning    = ai_decision.reasoning
            else:
                result.action       = "HOLD"
                result.confidence   = 0.5
                result.is_confirmed = False
                result.reasoning    = "CLOSE signal not confirmed"

        else:  # HOLD
            result.action       = "HOLD"
            result.confidence   = confidence
            result.is_confirmed = True
            result.reasoning    = ai_decision.reasoning

        logger.info(
            f"Signal for {symbol}: {result.action} (conf: {result.confidence:.0%}) "
            f"— Confirmed: {result.is_confirmed}"
        )
        return result

    # ------------------------------------------------------------------ #
    #  PRIVATE helpers                                                     #
    # ------------------------------------------------------------------ #
    def _build_tf_summary(
        self, indicators: IndicatorValues, label: str
    ) -> Dict[str, Any]:
        """Build a summary dictionary for a timeframe."""
        return {
            "label":         label,
            "rsi":           round(indicators.rsi, 1),
            "rsi_trend":     indicators.rsi_trend,
            "macd_trend":    indicators.macd_trend,
            "macd_rising":   indicators.macd_histogram_rising,
            "ema_trend":     indicators.ema_trend,
            "ema_crossover": indicators.ema_crossover,
            "bb_percent":    round(indicators.bb_percent, 2),
            "bb_trend":      indicators.bb_trend,
            "overall_trend": indicators.overall_trend,
            "trend_strength":indicators.trend_strength,
        }

    def _check_trend_confirmation(
        self,
        action: str,
        trend_tf: IndicatorValues,
        entry_tf: IndicatorValues,
    ) -> Tuple[bool, str]:
        """
        Check if the 10-min trend timeframe confirms the entry direction.

        Returns (is_confirmed, reason_message).
        """
        if action == "BUY":
            if trend_tf.overall_trend == "BULLISH":
                return True, "10-min trend is BULLISH (confirms BUY)"
            elif trend_tf.overall_trend == "SIDEWAYS":
                adx = getattr(trend_tf, "adx", 0.0)
                if trend_tf.macd_histogram_rising and adx > self.sideways_adx_threshold:
                    return True, "10-min trend is SIDEWAYS with bullish momentum"
                return False, f"SIDEWAYS + ADX={adx:.1f} — insufficient momentum"
            else:
                return False, f"10-min trend is {trend_tf.overall_trend} (contradicts BUY)"

        elif action == "SELL":
            if trend_tf.overall_trend == "BEARISH":
                return True, "10-min trend is BEARISH (confirms SELL)"
            elif trend_tf.overall_trend == "SIDEWAYS":
                if (
                    not trend_tf.macd_histogram_rising
                    and trend_tf.ema_trend != "BULLISH"
                ):
                    return True, "10-min trend is SIDEWAYS with bearish momentum"
                return False, "10-min trend is SIDEWAYS without bearish momentum"
            else:
                return False, f"10-min trend is {trend_tf.overall_trend} (contradicts SELL)"

        return True, "No trend confirmation needed"

    def _check_rsi_validation(
        self,
        action: str,
        entry_tf: IndicatorValues,
    ) -> Tuple[bool, str]:
        """
        Validate RSI conditions for the proposed action.

        Returns (is_valid, reason_message).
        """
        rsi = entry_tf.rsi

        if action == "BUY":
            if rsi > 70:
                return False, f"RSI at {rsi:.1f} is overbought (>70, risky for BUY)"
            elif rsi <= self.min_rsi_for_buy:
                return True, f"RSI at {rsi:.1f} shows oversold conditions (good for BUY)"
            else:
                return True, f"RSI at {rsi:.1f} is neutral (acceptable for BUY)"

        elif action == "SELL":
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
    ) -> Tuple[bool, str]:
        """
        Validate MACD conditions for the proposed action.

        Returns (is_valid, reason_message).
        """
        macd_trend = entry_tf.macd_trend
        histogram  = entry_tf.macd_histogram
        rising     = entry_tf.macd_histogram_rising

        if action == "BUY":
            if macd_trend == "BULLISH" and histogram > 0 and rising:
                return True, "MACD is BULLISH with rising histogram (strong BUY)"
            elif macd_trend == "BULLISH" or (histogram > 0 and rising):
                return True, "MACD shows bullish momentum"
            elif histogram < 0 and rising:
                return True, "MACD histogram rising from below (potential reversal)"
            else:
                return False, f"MACD is {macd_trend}, histogram not supportive"

        elif action == "SELL":
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
    ) -> Tuple[bool, str]:
        """
        Confirm volume is adequate.

        Returns (is_confirmed, reason_message).
        """
        volume_ratio = entry_tf.volume_ratio

        if volume_ratio >= 1.2:
            return True,  f"Volume {volume_ratio:.1f}x average (above average confirms move)"
        elif volume_ratio >= 0.8:
            return True,  f"Volume {volume_ratio:.1f}x average (normal)"
        else:
            return False, f"Volume {volume_ratio:.1f}x average (low volume, weak signal)"

    def _check_position_context(
        self,
        action: str,
        current_position: Dict[str, Any],
    ) -> Tuple[bool, str]:
        """
        Verify the proposed action makes sense given the open position.

        Returns (is_valid, reason_message).
        """
        side               = current_position.get("side", "long")
        unrealized_pnl_pct = current_position.get("unrealized_pnl_pct", 0.0)

        if action == "BUY":
            if side == "long":
                return False, (
                    f"Already in LONG position (P&L: {unrealized_pnl_pct:+.2f}%)"
                )
            return True, "No conflicting long position"

        elif action == "SELL":
            if side == "short":
                return False, (
                    f"Already in SHORT position (P&L: {unrealized_pnl_pct:+.2f}%)"
                )
            return True, "No conflicting short position"

        elif action == "CLOSE":
            return True, (
                f"Closing existing {side.upper()} position "
                f"(P&L: {unrealized_pnl_pct:+.2f}%)"
            )

        return True, "Position context is valid"

    # ------------------------------------------------------------------ #
    #  PUBLIC: utility methods                                            #
    # ------------------------------------------------------------------ #
    def quick_signal(self, indicators: SymbolIndicators) -> str:
        """
        Get a quick signal summary without full AI analysis.
        Used for dashboard display.

        Returns "BUY", "SELL", "HOLD", or "WATCH".
        """
        trend = indicators.trend_tf
        entry = indicators.entry_tf

        bullish = 0
        bearish = 0

        if trend.overall_trend == "BULLISH":
            bullish += 2
        elif trend.overall_trend == "BEARISH":
            bearish += 2

        if entry.overall_trend == "BULLISH":
            bullish += 1
        elif entry.overall_trend == "BEARISH":
            bearish += 1

        if entry.rsi < 35:
            bullish += 1
        elif entry.rsi > 65:
            bearish += 1

        if entry.macd_trend == "BULLISH" and entry.macd_histogram_rising:
            bullish += 1
        elif entry.macd_trend == "BEARISH" and not entry.macd_histogram_rising:
            bearish += 1

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
