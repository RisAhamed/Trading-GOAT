"""Health model: /health /ready /metrics served by dashboard or health_check script."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

_started = time.time()
_checks: dict[str, dict] = {}


@dataclass
class Health:
    status: str = "STARTING"  # STARTING | READY | DEGRADED | NOT_READY
    components: dict[str, str] = field(default_factory=dict)
    uptime_s: float = 0.0
    detail: str = ""


def set_component(name: str, ok: bool, detail: str = "") -> None:
    _checks[name] = {"ok": ok, "detail": detail}


def get_health() -> Health:
    if not _checks:
        return Health(status="STARTING", uptime_s=time.time() - _started)
    bad = [k for k, v in _checks.items() if not v["ok"]]
    # alpaca is critical; ollama degrades to HOLD (still READY)
    critical = [b for b in bad if b in ("config", "alpaca", "market_data", "paper_mode")]
    status = "READY" if not bad else ("DEGRADED" if not critical else "NOT_READY")
    return Health(
        status=status,
        components={k: ("ok" if v["ok"] else f"FAIL: {v['detail']}") for k, v in _checks.items()},
        uptime_s=time.time() - _started,
    )
