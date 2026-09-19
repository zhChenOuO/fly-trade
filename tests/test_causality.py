"""因果性測試:只改 t0 以後的輸入,t0 以前的 motor 活動必須完全不變(否則 = look-ahead)。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
from src.connectome.synthetic import make_synthetic_connectome
from src.train.evolve import ConnectomeTradingAgent
from src.data.features import build_features
from src.data.market_data import make_synthetic_ohlcv


def test_no_lookahead():
    g = make_synthetic_connectome(n_neurons=500, n_sensory=24, n_motor=8, seed=0)
    a = ConnectomeTradingAgent(g, n_features=6, motor_window=8)
    theta = a.random_init(1)
    enc_w, _, gain, leak = a.layout.unpack(theta)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 6)); X2 = X.copy(); X2[151:] = rng.normal(size=(149, 6))
    sim = lambda x: a.reservoir.simulate(a.encoder.encode(x, enc_w), gain=gain, leak=leak, record_motor_window=8)
    assert np.array_equal(sim(X)[:151], sim(X2)[:151]), "reservoir 偷看未來"

    df = make_synthetic_ohlcv(2000, seed=3); df2 = df.copy()
    df2.iloc[1500:, :] = df2.iloc[1500:, :] * 1.7
    assert np.array_equal(build_features(df)[:1500], build_features(df2)[:1500]), "features 偷看未來"


if __name__ == "__main__":
    test_no_lookahead(); print("OK: causality test passed")
