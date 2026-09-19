from __future__ import annotations

import numpy as np


def sharpe_ratio(returns: np.ndarray, periods_per_year: float = 24 * 365) -> float:
    r = returns[np.isfinite(returns)]
    if r.std() < 1e-12:
        return 0.0
    return float(np.sqrt(periods_per_year) * r.mean() / r.std())


def max_drawdown(equity_curve: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity_curve)
    dd = (equity_curve - peak) / peak
    return float(dd.min())


def total_return(equity_curve: np.ndarray) -> float:
    return float(equity_curve[-1] / equity_curve[0] - 1.0)


def periods_per_year(timeframe: str) -> float:
    """'4h' -> 2190, '1h' -> 8760, '1d' -> 365。Sharpe 年化必須跟 K 棒週期一致。"""
    n, unit = int(timeframe[:-1]), timeframe[-1]
    return {"m": 525600, "h": 8760, "d": 365}[unit] / n
