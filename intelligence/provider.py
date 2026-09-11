"""AI provider abstraction: OllamaCloud with local fallback. AI down -> HOLD."""
from __future__ import annotations

import time
from dataclasses import dataclass

from domain.models import Action, AIAnalysis, MarketContext
from intelligence.schemas import parse_ai_json


@dataclass
class ProviderResult:
    text: str
    model_used: str
    fallback_used: bool
    latency_ms: float
    failure_reason: str = ""


class AIProvider:
    name: str = "base"

    def complete(self, prompt: str, timeout_s: int) -> ProviderResult:
        raise NotImplementedError


class OllamaCloudProvider(AIProvider):
    name = "ollama_cloud"

    def __init__(self, legacy_brain) -> None:
        self._brain = legacy_brain  # reuse tested core.ai_brain wiring

    def complete(self, prompt: str, timeout_s: int) -> ProviderResult:
        t0 = time.perf_counter()
        try:
            text = self._brain._call_cloud(prompt, timeout_s)  # reuse cloud call path
            ms = (time.perf_counter() - t0) * 1000.0
            return ProviderResult(text=text or "", model_used=getattr(self._brain, "active_model", "cloud"),
                                  fallback_used=False, latency_ms=ms)
        except Exception as e:
            ms = (time.perf_counter() - t0) * 1000.0
            return ProviderResult(text="", model_used="", fallback_used=False, latency_ms=ms, failure_reason=str(e)[:200])


class LocalOllamaProvider(AIProvider):
    name = "local_ollama"

    def __init__(self, legacy_brain) -> None:
        self._brain = legacy_brain

    def complete(self, prompt: str, timeout_s: int) -> ProviderResult:
        t0 = time.perf_counter()
        try:
            text = self._brain._call_local(prompt, timeout_s)
            ms = (time.perf_counter() - t0) * 1000.0
            return ProviderResult(text=text or "", model_used="local", fallback_used=True, latency_ms=ms)
        except Exception as e:
            ms = (time.perf_counter() - t0) * 1000.0
            return ProviderResult(text="", model_used="", fallback_used=True, latency_ms=ms, failure_reason=str(e)[:200])


PROMPT_VERSION = "v1"


def build_prompt(ctx: MarketContext, position_desc: str, portfolio_desc: str) -> str:
    from pathlib import Path
    tpl = Path(__file__).parent / "prompts" / "trading_analysis_v1.txt"
    template = tpl.read_text(encoding="utf-8")
    return template.format(
        symbol=ctx.symbol, price=f"{ctx.price:,.2f}",
        regime=ctx.regime.value,
        trend_direction=ctx.trend.direction, trend_strength=ctx.trend.strength,
        trend_reasons="; ".join(ctx.trend.reasons) or "n/a",
        entry_direction=ctx.entry.direction, rsi=f"{ctx.entry.rsi:.1f}",
        rsi_state=ctx.entry.rsi_state, bb_percent=f"{ctx.entry.bb_percent:.2f}",
        volume_ratio=f"{ctx.entry.volume_ratio:.2f}", atr_percent=f"{ctx.atr_percent:.2f}",
        entry_reasons="; ".join(ctx.entry.reasons) or "n/a",
        position=position_desc, portfolio=portfolio_desc,
    )


def decide(ctx: MarketContext, position_desc: str, portfolio_desc: str,
           primary: AIProvider, fallback: AIProvider | None,
           timeout_s: int, model_requested: str) -> AIAnalysis:
    """Request decision; any failure or invalid schema -> HOLD (never trade)."""
    prompt = build_prompt(ctx, position_desc, portfolio_desc)
    res = primary.complete(prompt, timeout_s)
    if not res.text and fallback is not None:
        res = fallback.complete(prompt, timeout_s)
    parsed = parse_ai_json(res.text)
    if parsed.raw_invalid:
        return AIAnalysis(symbol=ctx.symbol, action=Action.HOLD, confidence=0.0,
                          reasoning_summary="", model_requested=model_requested,
                          model_used=res.model_used, fallback_used=res.fallback_used,
                          latency_ms=res.latency_ms, prompt_version=PROMPT_VERSION,
                          raw_invalid=True, failure_reason=parsed.failure_reason)
    return AIAnalysis(symbol=ctx.symbol, action=parsed.action, confidence=parsed.confidence,
                      trend=parsed.trend, entry_quality=parsed.entry_quality,
                      reasoning_summary=parsed.reasoning_summary, risk_notes=parsed.risk_notes,
                      model_requested=model_requested, model_used=res.model_used,
                      fallback_used=res.fallback_used, latency_ms=res.latency_ms,
                      prompt_version=PROMPT_VERSION)
