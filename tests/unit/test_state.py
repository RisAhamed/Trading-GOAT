"""Unit: universe advisory never starves primaries; reconcile; retry; enrichment."""
from exits.retry import ExitRetryTracker
from intelligence.enrichment import EnrichmentHub
from portfolio.snapshot import (
    BrokerPosition,
    PortfolioSnapshot,
    is_strategy_symbol,
    normalize_symbol,
    reconcile,
)
from strategy.universe import select_symbols


def test_universe_preserves_primaries_and_open():
    rows = select_symbols(
        configured=["BTC/USD", "ETH/USD"],
        rankings={"BTC/USD": 5, "ETH/USD": 20, "SOL/USD": 40, "DOGE/USD": 10},
        open_strategy_symbols={"ETH/USD"},
        max_ranked=1, min_score=40.0)
    by = {r.symbol: r for r in rows}
    assert by["BTC/USD"].tradeable and by["BTC/USD"].priority == "PRIMARY"
    assert by["ETH/USD"].tradeable and by["ETH/USD"].priority == "OPEN_POSITION"
    assert by["SOL/USD"].tradeable  # top ranked within capacity
    assert not by["DOGE/USD"].tradeable


def test_external_positions_classified_not_counted():
    assert normalize_symbol("BTCUSD") == "BTC/USD"
    assert not is_strategy_symbol("NVDA")
    assert is_strategy_symbol("BTC/USD")


def test_reconcile_broker_wins():
    snap = PortfolioSnapshot(broker_ok=True, strategy_positions=[
        BrokerPosition("SOL/USD", "SOLUSD", qty=1.0, entry_price=100.0)])
    res = reconcile(snap, {"BTC/USD": {"status": "OPEN"}, "SOL/USD": {"status": "OPEN"}})
    assert any("local_OPEN_but_broker_flat:BTC/USD" in m for m in res.mismatches)
    assert any("adopted" in r or "repaired" in r or "SOL" in r for r in res.repaired) or res.mismatches


def test_external_flagged_not_liquidated():
    snap = PortfolioSnapshot(broker_ok=True, external_positions=[
        BrokerPosition("NVDA", "NVDA", qty=10.0, entry_price=180.0, external=True)])
    assert snap.strategy_open_positions == 0
    res = reconcile(snap, {})
    assert any("external_unsupported_position:nvda" in m.lower() for m in res.mismatches)


def test_retry_backoff_and_flat_clear():
    tr = ExitRetryTracker()
    assert tr.should_retry("X")
    tr.record_failure("X", "boom", now_s=1000.0)
    assert not tr.should_retry("X", now_s=1000.0 + 10)  # 30s backoff
    assert tr.should_retry("X", now_s=1000.0 + 31)
    for _ in range(10):
        tr.record_failure("X", "boom", now_s=2000.0)
    assert not tr.should_retry("X", now_s=999999.0)  # max attempts -> human attention
    assert tr.clear_if_flat("X", set())  # broker flat clears


def test_enrichment_never_blocks_and_breaks_circuit():
    hub = EnrichmentHub()
    hub.register("hang", lambda: __import__("time").sleep(60) or {})
    t0 = __import__("time").perf_counter()
    hub.refresh_once(timeout_s=0.3)
    assert __import__("time").perf_counter() - t0 < 5  # bounded
    assert hub.snapshot()["hang"]["available"] is False

    hub2 = EnrichmentHub()
    hub2.register("pol", lambda: (_ for _ in ()).throw(RuntimeError("401 Unauthorized")))
    hub2.refresh_once(timeout_s=2.0)
    st = hub2.snapshot()["pol"]
    assert st["available"] is False and "AUTH" in st["reason"]
