"""AI provider abstraction with explicit fallback semantics.

Delegates to REAL legacy methods (verified against core/ai_brain.py):
  primary  -> AIBrain._call_ollama_cloud_with_model(prompt, model)  (cloud only)
  fallback -> AIBrain._call_ollama_local(prompt)                    (local only)

Outcome is always one of: PRIMARY_SUCCESS | FALLBACK_SUCCESS | ALL_FAILED |
INVALID_OUTPUT. Anything but a valid schema -> HOLD. Never trades on failure.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

from domain.models import Action, AIAnalysis, MarketContext
from intelligence.schemas import parse_ai_json

PROMPT_VERSION = "v1"


class AIBrainLike(Protocol):
    def _call_ollama_cloud_with_model(self, prompt: str, model: str): ...
    def _call_ollama_local(self, prompt: str): ...


@dataclass
class ProviderResult:
    text: str
    model_used: str
    provider: str  # "ollama_cloud" | "local_ollama"
    fallback_used: bool
    latency_ms: float
    outcome: str  # PRIMARY_SUCCESS | FALLBACK_SUCCESS | ALL_FAILED
    failure_reason: str = ""


def _call_cloud_only(brain: AIBrainLike, prompt: str, model: str) -> tuple[str, str]:
    """Cloud-only attempt (no silent local switch inside)."""
    try:
        out = brain._call_ollama_cloud_with_model(prompt, model)
        if isinstance(out, tuple) and len(out) == 2:
            return out[0] or "", out[1] or ""
        return "", "unexpected_return_shape"
    except Exception as e:  # provider boundary: record, don't raise
        return "", f"cloud_exception: {e}"[:200]


def _call_local_only(brain: AIBrainLike, prompt: str) -> tuple[str, str]:
    try:
        out = brain._call_ollama_local(prompt)
        if isinstance(out, tuple) and len(out) == 2:
            return out[0] or "", out[1] or ""
        return "", "unexpected_return_shape"
    except Exception as e:  # provider boundary: record, don't raise
        return "", f"local_exception: {e}"[:200]


def build_prompt(ctx: MarketContext, position_desc: str, portfolio_desc: str) -> str:
    from pathlib import Path

    tpl = Path(__file__).parent / "prompts" / "trading_analysis_v1.txt"
    template = tpl.read_text(encoding="utf-8")
    return template.format(
        symbol=ctx.symbol,
        price=f"{ctx.price:,.2f}",
        regime=ctx.regime.value,
        trend_direction=ctx.trend.direction,
        trend_strength=ctx.trend.strength,
        trend_reasons="; ".join(ctx.trend.reasons) or "n/a",
        entry_direction=ctx.entry.direction,
        rsi=f"{ctx.entry.rsi:.1f}",
        rsi_state=ctx.entry.rsi_state,
        bb_percent=f"{ctx.entry.bb_percent:.2f}",
        volume_ratio=f"{ctx.entry.volume_ratio:.2f}",
        atr_percent=f"{ctx.atr_percent:.2f}",
        entry_reasons="; ".join(ctx.entry.reasons) or "n/a",
        position=position_desc,
        portfolio=portfolio_desc,
    )


def decide(
    ctx: MarketContext,
    position_desc: str,
    portfolio_desc: str,
    brain: AIBrainLike,
    model_requested: str,
    local_model: str = "",
    timeout_s: int = 60,  # reserved for future per-call timeout plumbing
) -> AIAnalysis:
    """Request a decision. Any failure or invalid schema -> HOLD (never trade)."""
    _ = timeout_s
    prompt = build_prompt(ctx, position_desc, portfolio_desc)

    t0 = time.perf_counter()
    text, cloud_err = _call_cloud_only(brain, prompt, model_requested)
    cloud_ms = (time.perf_counter() - t0) * 1000.0
    if text:
        parsed = parse_ai_json(text)
        if not parsed.raw_invalid:
            return AIAnalysis(
                symbol=ctx.symbol, action=parsed.action, confidence=parsed.confidence,
                trend=parsed.trend, entry_quality=parsed.entry_quality,
                reasoning_summary=parsed.reasoning_summary, risk_notes=parsed.risk_notes,
                model_requested=model_requested, model_used=model_requested,
                fallback_used=False, latency_ms=cloud_ms, prompt_version=PROMPT_VERSION,
            )
        invalid_reason = parsed.failure_reason
    else:
        invalid_reason = ""

    t1 = time.perf_counter()
    local_text, local_err = _call_local_only(brain, prompt)
    local_ms = (time.perf_counter() - t1) * 1000.0
    if local_text:
        parsed = parse_ai_json(local_text)
        if not parsed.raw_invalid:
            return AIAnalysis(
                symbol=ctx.symbol, action=parsed.action, confidence=parsed.confidence,
                trend=parsed.trend, entry_quality=parsed.entry_quality,
                reasoning_summary=parsed.reasoning_summary, risk_notes=parsed.risk_notes,
                model_requested=model_requested, model_used=local_model or "local",
                fallback_used=True, latency_ms=cloud_ms + local_ms,
                prompt_version=PROMPT_VERSION,
            )
        return AIAnalysis(
            symbol=ctx.symbol, action=Action.HOLD, confidence=0.0,
            model_requested=model_requested, model_used=local_model or "local",
            fallback_used=True, latency_ms=cloud_ms + local_ms,
            prompt_version=PROMPT_VERSION, raw_invalid=True,
            failure_reason=f"fallback_invalid: {parsed.failure_reason}",
        )
    failure = "; ".join(
        p for p in (
            f"cloud:{cloud_err or invalid_reason or 'empty'}",
            f"local:{local_err or 'empty'}",
        ) if p
    )
    return AIAnalysis(
        symbol=ctx.symbol, action=Action.HOLD, confidence=0.0,
        model_requested=model_requested, model_used="",
        fallback_used=True, latency_ms=cloud_ms + local_ms,
        prompt_version=PROMPT_VERSION, raw_invalid=True,
        failure_reason=f"all_failed: {failure}"[:300],
    )


# Backwards-compatible provider classes (same real delegation underneath).
class OllamaCloudProvider:
    name = "ollama_cloud"

    def __init__(self, legacy_brain: AIBrainLike) -> None:
        self._brain = legacy_brain

    def complete(self, prompt: str, model: str) -> ProviderResult:
        t0 = time.perf_counter()
        text, err = _call_cloud_only(self._brain, prompt, model)
        ms = (time.perf_counter() - t0) * 1000.0
        return ProviderResult(
            text=text, model_used=model if text else "", provider="ollama_cloud",
            fallback_used=False, latency_ms=ms,
            outcome="PRIMARY_SUCCESS" if text else "ALL_FAILED",
            failure_reason="" if text else err,
        )


class LocalOllamaProvider:
    name = "local_ollama"

    def __init__(self, legacy_brain: AIBrainLike) -> None:
        self._brain = legacy_brain

    def complete(self, prompt: str, model: str = "") -> ProviderResult:
        t0 = time.perf_counter()
        text, err = _call_local_only(self._brain, prompt)
        ms = (time.perf_counter() - t0) * 1000.0
        return ProviderResult(
            text=text, model_used=model or "local", provider="local_ollama",
            fallback_used=True, latency_ms=ms,
            outcome="FALLBACK_SUCCESS" if text else "ALL_FAILED",
            failure_reason="" if text else err,
        )
