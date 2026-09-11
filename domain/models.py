"""Typed domain objects for the real-time trading pipeline.

One responsibility: define the state that flows through the cycle so every
layer speaks the same language instead of passing dicts around.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Action(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE = "CLOSE"


class Regime(str, Enum):
    TRENDING_BULLISH = "TRENDING_BULLISH"
    TRENDING_BEARISH = "TRENDING_BEARISH"
    SIDEWAYS = "SIDEWAYS"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    TRANSITION = "TRANSITION"
    UNKNOWN = "UNKNOWN"


class ExitReason(str, Enum):
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    TRAILING_STOP = "TRAILING_STOP"
    SIGNAL_REVERSAL = "SIGNAL_REVERSAL"
    TIMEOUT = "TIMEOUT"
    RISK_KILL_SWITCH = "RISK_KILL_SWITCH"
    MANUAL = "MANUAL"
    NONE = "NONE"


@dataclass
class MarketSnapshot:
    symbol: str
    price: float
    timestamp: datetime
    bar_timestamp: datetime | None = None
    timeframe: str = ""
    bars: int = 0
    stale: bool = False
    issues: list[str] = field(default_factory=list)


@dataclass
class FeatureSnapshot:
    symbol: str
    timeframe: str
    timestamp: datetime
    price: float = 0.0
    rsi: float = 50.0
    macd_line: float = 0.0
    macd_signal: float = 0.0
    macd_histogram: float = 0.0
    ema_short: float = 0.0
    ema_long: float = 0.0
    bb_upper: float = 0.0
    bb_middle: float = 0.0
    bb_lower: float = 0.0
    bb_percent: float = 0.5
    atr: float = 0.0
    atr_percent: float = 0.0
    adx: float = 0.0
    volume_ratio: float = 1.0
    overall_trend: str = "SIDEWAYS"
    # Data-quality flags (explicit, never silent)
    valid: bool = True
    quality_issues: list[str] = field(default_factory=list)
    data_points: int = 0


@dataclass
class TrendContext:
    direction: str = "SIDEWAYS"  # BULLISH | BEARISH | SIDEWAYS
    strength: str = "WEAK"  # STRONG | MODERATE | WEAK
    ema_trend: str = "NEUTRAL"
    macd_trend: str = "NEUTRAL"
    adx: float = 0.0
    reasons: list[str] = field(default_factory=list)


@dataclass
class EntryContext:
    direction: str = "SIDEWAYS"
    rsi: float = 50.0
    rsi_state: str = "NEUTRAL"
    macd_histogram_rising: bool = False
    bb_percent: float = 0.5
    volume_ratio: float = 1.0
    reasons: list[str] = field(default_factory=list)


@dataclass
class MarketContext:
    symbol: str
    price: float
    trend: TrendContext
    entry: EntryContext
    regime: Regime = Regime.UNKNOWN
    atr_percent: float = 0.0
    timestamp: datetime = field(default_factory=utcnow)


@dataclass
class AIAnalysis:
    symbol: str
    action: Action
    confidence: float
    trend: str = "UNKNOWN"
    entry_quality: str = "UNKNOWN"
    reasoning_summary: str = ""
    risk_notes: list[str] = field(default_factory=list)
    model_requested: str = ""
    model_used: str = ""
    fallback_used: bool = False
    latency_ms: float = 0.0
    prompt_version: str = "v1"
    schema_version: str = "1"
    timestamp: datetime = field(default_factory=utcnow)
    raw_invalid: bool = False
    failure_reason: str = ""


@dataclass
class SignalDecision:
    symbol: str
    final_action: Action
    ai_action: Action
    confidence: float
    confirmations: list[str] = field(default_factory=list)
    rejections: list[str] = field(default_factory=list)
    score: float = 0.0
    timestamp: datetime = field(default_factory=utcnow)

    @property
    def accepted(self) -> bool:
        return self.final_action in (Action.BUY, Action.SELL)


@dataclass
class RiskDecision:
    symbol: str
    allowed: bool
    rejection_reason: str = ""
    position_size: float = 0.0  # in units of base asset (or notional/qty per broker)
    notional: float = 0.0
    risk_amount: float = 0.0
    stop_price: float = 0.0
    take_profit_price: float = 0.0
    risk_reward_ratio: float = 0.0
    timestamp: datetime = field(default_factory=utcnow)


@dataclass
class OrderIntent:
    intent_id: str
    cycle_id: str
    signal_id: str
    symbol: str
    side: str  # buy | sell
    qty: float
    stop_price: float = 0.0
    take_profit_price: float = 0.0
    timestamp: datetime = field(default_factory=utcnow)


@dataclass
class ExecutionResult:
    intent_id: str
    symbol: str
    status: str  # FILLED | REJECTED | FAILED | PARTIAL | DRY_RUN
    filled_qty: float = 0.0
    avg_fill_price: float = 0.0
    order_id: str = ""
    message: str = ""
    timestamp: datetime = field(default_factory=utcnow)


@dataclass
class PositionSnapshot:
    symbol: str
    side: str  # long | short | flat
    qty: float = 0.0
    entry_price: float = 0.0
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    stop_price: float = 0.0
    target_price: float = 0.0
    hold_seconds: float = 0.0


@dataclass
class ExitEvent:
    symbol: str
    reason: ExitReason
    price: float
    pnl: float = 0.0
    message: str = ""
    timestamp: datetime = field(default_factory=utcnow)


@dataclass
class TradingCycleResult:
    cycle_id: str
    symbol: str
    started_at: datetime
    market_snapshot: MarketSnapshot | None = None
    features: FeatureSnapshot | None = None
    regime: Regime = Regime.UNKNOWN
    ai: AIAnalysis | None = None
    signal: SignalDecision | None = None
    risk: RiskDecision | None = None
    execution: ExecutionResult | None = None
    exit_event: ExitEvent | None = None
    latency_ms: dict[str, float] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    final_state: str = "HOLD"
    finished_at: datetime = field(default_factory=utcnow)

    def to_event(self) -> dict[str, Any]:
        return {
            "event": "CYCLE_COMPLETED",
            "cycle_id": self.cycle_id,
            "symbol": self.symbol,
            "final_state": self.final_state,
            "latency_ms": self.latency_ms,
            "errors": self.errors,
        }
