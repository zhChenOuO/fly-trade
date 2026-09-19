"""煙霧測試:用很小的規模跑通整條 pipeline,確認環境安裝正確、
程式邏輯沒有明顯錯誤。不是在驗證交易策略是否有效。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.connectome.synthetic import make_synthetic_connectome
from src.data.market_data import make_synthetic_ohlcv
from src.data.features import build_features
from src.train.evolve import ConnectomeTradingAgent, train_cma
from src.backtest.engine import run_backtest


def test_end_to_end_smoke():
    graph = make_synthetic_connectome(
        n_neurons=300, avg_out_degree=6, n_sensory=8, n_motor=4, seed=1
    )
    assert graph.n_neurons == 300
    assert graph.n_synapses > 0

    df = make_synthetic_ohlcv(n_bars=200, seed=1)
    features = build_features(df)
    assert features.shape[0] == len(df)

    agent = ConnectomeTradingAgent(graph, n_features=features.shape[1], motor_window=4)
    theta0 = agent.random_init(seed=1)
    actions = agent.act(features, theta0)
    assert actions.shape[0] == len(df)
    assert set(np.unique(actions)).issubset({0, 1, 2})

    result = run_backtest(df["close"].values, actions, fee_bps=5.0)
    assert np.isfinite(result.sharpe)
    assert result.equity_curve.shape[0] == len(df)

    best_theta, history = train_cma(
        agent, features, df["close"].values, sigma0=0.3, max_iter=2, popsize=6, seed=1, verbose=False
    )
    assert best_theta.shape[0] == agent.layout.n_total
    assert len(history) == 2


if __name__ == "__main__":
    test_end_to_end_smoke()
    print("OK: smoke test passed")
