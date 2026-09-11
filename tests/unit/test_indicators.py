"""Unit: Bollinger must use float std (regression for int() truncation 1.8->1)."""


def test_bollinger_uses_float_std():
    from unittest.mock import MagicMock

    from core.indicators import IndicatorCalculator
    cfg = MagicMock()
    cfg.indicators.rsi_period = 7
    cfg.indicators.rsi_oversold = 35
    cfg.indicators.rsi_overbought = 65
    cfg.indicators.macd_fast = 8
    cfg.indicators.macd_slow = 17
    cfg.indicators.macd_signal = 9
    cfg.indicators.ema_short = 5
    cfg.indicators.ema_long = 13
    cfg.indicators.bb_period = 14
    cfg.indicators.bb_std_dev = 1.8
    cfg.indicators.atr_period = 10
    calc = IndicatorCalculator(config=cfg)
    assert calc.bb_std == 1.8
    import inspect

    import core.indicators as ind_mod
    src = inspect.getsource(ind_mod.IndicatorCalculator._calculate_bollinger)
    assert "int(self.bb_std)" not in src, "Bollinger truncation bug regressed"
    assert "float(self.bb_std)" in src
