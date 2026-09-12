"""Advisory symbol universe: rank, prioritize, never starve primaries.

Decision (documented): the scanner advises evaluation ORDER, it does not
eliminate the configured strategy universe. Configured primary markets
(BTC/USD, ETH/USD) are always evaluated; open strategy positions are always
evaluated; remaining candidates are evaluated by rank up to capacity.

Each row exposes rank/score/priority/tradeable/reason for the dashboard.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class UniverseRow:
    symbol: str
    rank: int
    score: float
    priority: str  # PRIMARY | OPEN_POSITION | RANKED | BELOW_THRESHOLD
    tradeable: bool
    reason: str


def select_symbols(
    configured: list[str],
    rankings: dict[str, float],
    open_strategy_symbols: set[str],
    max_ranked: int = 3,
    min_score: float = 40.0,
    primaries: tuple[str, ...] = ("BTC/USD", "ETH/USD"),
) -> list[UniverseRow]:
    pool = sorted(set(configured) | set(rankings) | set(open_strategy_symbols))
    ranked = sorted(pool, key=lambda s: (-float(rankings.get(s, 0.0)), s))
    rows: list[UniverseRow] = []
    ranked_added = 0
    for i, sym in enumerate(ranked, start=1):
        score = float(rankings.get(sym, 0.0))
        if sym in open_strategy_symbols:
            rows.append(UniverseRow(sym, i, score, "OPEN_POSITION", True, "open strategy position always evaluated"))
        elif sym in primaries and sym in configured:
            rows.append(UniverseRow(sym, i, score, "PRIMARY", True,
                                    "primary market always evaluated despite score"))
        elif score >= min_score and ranked_added < max_ranked:
            ranked_added += 1
            rows.append(UniverseRow(sym, i, score, "RANKED", True, f"score {score:.0f} >= {min_score:.0f}"))
        else:
            rows.append(UniverseRow(sym, i, score, "BELOW_THRESHOLD", False,
                                    f"score {score:.0f} < {min_score:.0f} or capacity reached"))
    # Evaluation order: open positions, primaries, then ranked tradeables.
    prio = {"OPEN_POSITION": 0, "PRIMARY": 1, "RANKED": 2, "BELOW_THRESHOLD": 3}
    return sorted(rows, key=lambda r: (prio[r.priority], r.rank))
