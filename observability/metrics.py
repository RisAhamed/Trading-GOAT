"""In-process metrics registry (Prometheus-compatible exposition)."""
from __future__ import annotations

import threading
import time

_lock = threading.Lock()
_counters: dict[str, float] = {}
_gauges: dict[str, float] = {}
_lat_samples: dict[str, list[float]] = {}


def inc(name: str, amount: float = 1.0) -> None:
    with _lock:
        _counters[name] = _counters.get(name, 0.0) + amount


def set_gauge(name: str, value: float) -> None:
    with _lock:
        _gauges[name] = value


def observe_latency(name: str, ms: float) -> None:
    with _lock:
        _lat_samples.setdefault(name, []).append(ms)
        if len(_lat_samples[name]) > 500:
            _lat_samples[name] = _lat_samples[name][-500:]


class Timer:
    def __init__(self, name: str) -> None:
        self.name = name
        self._t0 = 0.0
        self.ms = 0.0

    def __enter__(self) -> Timer:
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.ms = (time.perf_counter() - self._t0) * 1000.0
        observe_latency(self.name, self.ms)


def snapshot() -> dict:
    with _lock:
        lat = {k: {"count": len(v), "avg_ms": sum(v) / len(v) if v else 0.0, "max_ms": max(v) if v else 0.0} for k, v in _lat_samples.items()}
        return {"counters": dict(_counters), "gauges": dict(_gauges), "latency": lat}


def prometheus_text() -> str:
    snap = snapshot()
    lines = []
    for k, v in snap["counters"].items():
        lines.append(f"# TYPE goat_{k} counter\ngoat_{k} {v}")
    for k, v in snap["gauges"].items():
        lines.append(f"# TYPE goat_{k} gauge\ngoat_{k} {v}")
    for k, v in snap["latency"].items():
        lines.append(f"# TYPE goat_{k}_avg_ms gauge\ngoat_{k}_avg_ms {v['avg_ms']:.3f}")
    return "\n".join(lines) + "\n"
