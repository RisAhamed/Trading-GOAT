"""Canonical typed configuration.

Single source of truth for settings. Env vars are ONLY for secrets.
Validates on startup and prints a safe summary (no secrets).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"


@dataclass
class ValidationIssue:
    level: str  # error | warning
    message: str


@dataclass
class CanonicalSettings:
    mode: str = "paper"
    symbols: list[str] = field(default_factory=lambda: ["BTC/USD", "ETH/USD"])
    trend_interval: str = "5Min"
    trend_lookback: int = 30
    entry_interval: str = "1Min"
    entry_lookback: int = 20
    ai_model: str = "gpt-oss:120b"
    ai_timeout_s: int = 60
    max_positions: int = 2
    risk_per_trade_pct: float = 1.0
    stop_loss_pct: float = 0.5
    take_profit_multiplier: float = 2.0
    max_daily_loss_pct: float = 3.0
    min_signal_confidence: float = 0.50
    loop_interval_s: int = 15
    web_host: str = "127.0.0.1"
    web_port: int = 5000
    raw: dict = field(default_factory=dict)

    def safe_summary(self) -> str:
        lines = [
            f"MODE = {self.mode.upper()}",
            f"SYMBOLS = {', '.join(self.symbols)}",
            f"TREND TF = {self.trend_interval} x{self.trend_lookback}",
            f"ENTRY TF = {self.entry_interval} x{self.entry_lookback}",
            f"AI MODEL = {self.ai_model}",
            f"MAX POSITIONS = {self.max_positions}",
            f"RISK PER TRADE = {self.risk_per_trade_pct}%",
            f"STOP = {self.stop_loss_pct}% | R:R = 1:{self.take_profit_multiplier}",
            f"MIN CONFIDENCE = {self.min_signal_confidence}",
        ]
        return "\n".join(lines)


def _get(d: dict, *path: str, default: Any = None) -> Any:
    cur: Any = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def load_canonical(path: str | Path = DEFAULT_CONFIG_PATH) -> tuple[CanonicalSettings, list[ValidationIssue]]:
    """Load config.yaml and collapse overlapping keys into one canonical model."""
    issues: list[ValidationIssue] = []
    with open(path, encoding="utf-8") as f:
        raw: dict = yaml.safe_load(f) or {}

    mode = str(_get(raw, "bot", "mode", default="paper")).lower()

    # Symbols: canonical = markets.*.pairs for enabled sections (not the scanner pool).
    symbols: list[str] = []
    if _get(raw, "markets", "crypto", "enabled", default=True):
        symbols += list(_get(raw, "markets", "crypto", "pairs", default=[]) or [])
    if _get(raw, "markets", "forex", "enabled", default=False):
        symbols += list(_get(raw, "markets", "forex", "pairs", default=[]) or [])
    symbols = [s for s in symbols if isinstance(s, str) and "/" in s]
    if not symbols:
        issues.append(ValidationIssue("error", "No trading symbols enabled in markets.*.pairs"))

    # Detect overlapping/conflicting risk keys (documented, single hierarchy).
    risk = _get(raw, "risk", default={}) or {}
    if "risk_per_trade_pct" in risk and "base_risk_pct" in risk:
        # base_risk_pct is a fraction (0.02 = 2%) while risk_per_trade_pct is a percent (1.0 = 1%).
        # Canonical hierarchy: risk_per_trade_pct wins; base_risk_pct only used if the former missing.
        issues.append(
            ValidationIssue(
                "warning",
                "risk.risk_per_trade_pct and risk.base_risk_pct both set; "
                "canonical uses risk_per_trade_pct (percent). base_risk_pct ignored.",
            )
        )
    exit_risk = _get(raw, "exit_engine", default={}) or {}
    for k in ("min_risk_pct", "max_risk_pct"):
        if k in exit_risk:
            issues.append(
                ValidationIssue(
                    "warning",
                    f"exit_engine.{k} overlaps risk policy; treated as dynamic-sizing clamp only, "
                    "not the base risk.",
                )
            )

    settings = CanonicalSettings(
        mode=mode,
        symbols=symbols or ["BTC/USD", "ETH/USD"],
        trend_interval=str(_get(raw, "timeframes", "trend", "interval", default="5Min")),
        trend_lookback=int(_get(raw, "timeframes", "trend", "lookback_bars", default=30)),
        entry_interval=str(_get(raw, "timeframes", "entry", "interval", default="1Min")),
        entry_lookback=int(_get(raw, "timeframes", "entry", "lookback_bars", default=20)),
        ai_model=str(_get(raw, "ai", "model", default="gpt-oss:120b")),
        ai_timeout_s=int(_get(raw, "ai", "timeout_seconds", default=60)),
        max_positions=int(_get(raw, "risk", "max_positions", default=2)),
        risk_per_trade_pct=float(_get(raw, "risk", "risk_per_trade_pct", default=1.0)),
        stop_loss_pct=float(_get(raw, "risk", "stop_loss_pct", default=0.5)),
        take_profit_multiplier=float(_get(raw, "risk", "take_profit_multiplier", default=2.0)),
        max_daily_loss_pct=float(_get(raw, "risk", "max_daily_loss_pct", default=3.0)),
        min_signal_confidence=float(_get(raw, "risk", "min_signal_confidence", default=0.50)),
        loop_interval_s=int(_get(raw, "bot", "loop_interval_seconds", default=15)),
        web_host=str(_get(raw, "web_ui", "host", default="127.0.0.1")),
        web_port=int(_get(raw, "web_ui", "port", default=5000)),
        raw=raw,
    )
    if settings.mode != "paper":
        issues.append(ValidationIssue("error", f"mode must be 'paper', got '{settings.mode}'"))
    if not (0 < settings.risk_per_trade_pct <= 5):
        issues.append(ValidationIssue("warning", "risk_per_trade_pct outside sane 0-5% band"))
    return settings, issues


def assert_paper_mode(settings: CanonicalSettings) -> None:
    """Fail-closed guard: anything but PAPER refuses to start."""
    if settings.mode.strip().lower() != "paper":
        raise SystemExit(f"FATAL: refusing to run in non-paper mode ({settings.mode!r}).")


def secrets_present() -> dict[str, bool]:
    from dotenv import load_dotenv

    load_dotenv()
    return {
        "ALPACA_API_KEY": bool(os.getenv("ALPACA_API_KEY")),
        "ALPACA_API_SECRET": bool(os.getenv("ALPACA_API_SECRET")),
        "OLLAMA_API_KEY": bool(os.getenv("OLLAMA_API_KEY")),
    }
