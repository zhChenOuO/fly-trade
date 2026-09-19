"""極簡向量化回測引擎。

動作在時間點 t 決定的是「下一根 K 棒」要持有的部位,避免用到未來
資訊(look-ahead bias):position[t] 套用在 price_return[t+1] 上。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metrics import max_drawdown, sharpe_ratio, total_return
from ..network.readout import ACTION_SELL, ACTION_HOLD, ACTION_BUY

ACTION_TO_POSITION = {ACTION_SELL: -1.0, ACTION_HOLD: 0.0, ACTION_BUY: 1.0}


@dataclass
class BacktestResult:
    equity_curve: np.ndarray
    strategy_returns: np.ndarray
    sharpe: float
    max_drawdown: float
    total_return: float
    turnover: float


def run_backtest(
    close_prices: np.ndarray,
    actions: np.ndarray,
    fee_bps: float = 5.0,
    periods_per_year: float = 24 * 365,
) -> BacktestResult:
    """
    close_prices: (T,) 收盤價
    actions: (T,) 每個時間點的離散動作 {0,1,2}
    fee_bps: 每次換倉的手續費(basis points,5 = 0.05%)
    """
    positions = np.array([ACTION_TO_POSITION[a] for a in actions], dtype=np.float64)
    return run_backtest_pos(close_prices, positions, fee_bps, periods_per_year)


def run_backtest_pos(
    close_prices: np.ndarray,
    positions: np.ndarray,
    fee_bps: float = 5.0,
    periods_per_year: float = 24 * 365,
) -> BacktestResult:
    """positions: (T,) 連續部位 [-1,1](例如多個 seed 的部位平均 = 集成)。"""
    price_returns = np.zeros_like(close_prices)
    price_returns[1:] = close_prices[1:] / close_prices[:-1] - 1.0

    # 用「上一步」的部位去乘「這一步」的報酬,避免用到未來資訊
    applied_position = np.roll(positions, 1)
    applied_position[0] = 0.0

    turnover_series = np.abs(np.diff(positions, prepend=0.0))
    fee = turnover_series * (fee_bps / 10000.0)

    strategy_returns = applied_position * price_returns - fee
    equity_curve = np.cumprod(1.0 + strategy_returns)

    return BacktestResult(
        equity_curve=equity_curve,
        strategy_returns=strategy_returns,
        sharpe=sharpe_ratio(strategy_returns, periods_per_year),
        max_drawdown=max_drawdown(equity_curve),
        total_return=total_return(equity_curve),
        turnover=float(turnover_series.mean()),
    )


def run_buy_and_hold(
    close_prices: np.ndarray,
    fee_bps: float = 5.0,
    periods_per_year: float = 24 * 365,
) -> BacktestResult:
    """最簡單的 baseline:一開始就買進、全程不動。

    這個數字的用途是拿來檢查 agent 訓練出來的策略是不是真的學到了
    什麼,還是根本比不上什麼都不做、單純長期持有——如果 agent 在某段
    資料上輸給 buy-and-hold,代表這次訓練/這段市場條件下,方法本身
    (或至少這次的訓練結果)是有問題的,不能只看 agent 自己的 Sharpe
    是不是正的就覺得有用。
    """
    actions = np.full(close_prices.shape[0], ACTION_BUY, dtype=np.int64)
    return run_backtest(close_prices, actions, fee_bps=fee_bps, periods_per_year=periods_per_year)
