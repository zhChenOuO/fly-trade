"""用 CMA-ES(演化策略)訓練「輸入編碼層 + 讀出層」。

為什麼用 CMA-ES 而不是梯度下降/反向傳播?

1. 連接體的遞迴矩陣 W 是固定的,我們不需要對它求梯度。
2. 要訓練的參數(輸入編碼權重 + 讀出層權重 + 兩個全域純量)只有幾百
   到幾千個,規模很適合 CMA-ES 這種不需要梯度、只需要「跑一次模擬拿
   到一個分數」的黑箱優化演算法。
3. 回測的 Sharpe ratio 本身是不可微分的(涉及 argmax 離散動作、換倉
   手續費等非平滑運算),用梯度下降去優化它並不直接,黑箱優化反而
   更自然。
4. 好處是完全不需要在 M1 上做反向傳播/自動微分,計算瓶頸只在於「跑
   一次 reservoir 模擬」這個前向運算,而這一步本來就要做。

代價:CMA-ES 是 population-based 方法,每一代要跑「族群大小」次模擬,
所以參數量不能太多(建議控制在幾千個以內),也因此我們刻意把可訓練
參數限制在輸入層跟讀出層,而不是想辦法訓練全部 3000 萬個連接體權重。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..connectome.schema import ConnectomeGraph
from ..network.reservoir import LeakyReservoir
from ..network.encoding import LinearEncoder
from ..network.readout import LinearReadout
from ..backtest.engine import run_backtest, run_backtest_pos
from ..backtest.position import to_positions


@dataclass
class ParamLayout:
    n_encoder: int
    n_readout: int

    @property
    def n_total(self) -> int:
        return self.n_encoder + self.n_readout + 2  # +2 = gain, leak(pre-sigmoid)

    def unpack(self, theta: np.ndarray):
        enc = theta[: self.n_encoder]
        ro = theta[self.n_encoder : self.n_encoder + self.n_readout]
        gain_raw, leak_raw = theta[-2], theta[-1]
        gain = 0.1 + 2.0 / (1.0 + np.exp(-gain_raw))       # 映射到 (0.1, 2.1)
        leak = 0.05 + 0.9 / (1.0 + np.exp(-leak_raw))      # 映射到 (0.05, 0.95)
        return enc, ro, gain, leak


class ConnectomeTradingAgent:
    def __init__(self, graph: ConnectomeGraph, n_features: int, motor_window: int = 8):
        self.graph = graph
        self.reservoir = LeakyReservoir(graph)
        self.encoder = LinearEncoder(n_features, len(graph.sensory_idx))
        self.readout = LinearReadout(len(graph.motor_idx))
        self.layout = ParamLayout(self.encoder.n_params, self.readout.n_params)
        self.motor_window = motor_window

    def act(self, features: np.ndarray, theta: np.ndarray) -> np.ndarray:
        enc_w, ro_w, gain, leak = self.layout.unpack(theta)
        currents = self.encoder.encode(features, enc_w)
        motor_trace = self.reservoir.simulate(
            currents, gain=gain, leak=leak, record_motor_window=self.motor_window
        )
        return self.readout.actions(motor_trace, ro_w)

    def logits(self, features: np.ndarray, theta: np.ndarray) -> np.ndarray:
        enc_w, ro_w, gain, leak = self.layout.unpack(theta)
        motor = self.reservoir.simulate(self.encoder.encode(features, enc_w), gain=gain, leak=leak,
                                        record_motor_window=self.motor_window)
        return self.readout.logits(motor, ro_w)

    def logits_batch(self, features: np.ndarray, thetas: np.ndarray) -> np.ndarray:
        """thetas: (P, D) -> logits (P, T, 3)。整個 population 一次算。"""
        L = self.layout
        P = thetas.shape[0]
        enc = thetas[:, : L.n_encoder].reshape(P, self.encoder.n_sensory, self.encoder.n_features)
        ro = thetas[:, L.n_encoder : L.n_encoder + L.n_readout].reshape(P, 3, len(self.graph.motor_idx))
        gain = 0.1 + 2.0 / (1.0 + np.exp(-thetas[:, -2]))
        leak = 0.05 + 0.9 / (1.0 + np.exp(-thetas[:, -1]))
        currents = np.einsum("tf,psf->pts", features, enc)
        motor = self.reservoir.simulate_batch(currents, gain, leak, self.motor_window)
        return np.einsum("ptm,pam->pta", motor, ro)

    def act_batch(self, features: np.ndarray, thetas: np.ndarray) -> np.ndarray:
        return np.argmax(self.logits_batch(features, thetas), axis=2)

    def random_init(self, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.normal(0, 0.3, size=self.layout.n_total)


def fitness_fn(agent: ConnectomeTradingAgent, features: np.ndarray, close_prices: np.ndarray, fee_bps: float = 5.0, ppy: float = 24 * 365,
               l2: float = 0.0, turnover_pen: float = 0.0):
    def _fn(theta: np.ndarray) -> float:
        try:
            actions = agent.act(features, theta)
            result = run_backtest(close_prices, actions, fee_bps=fee_bps, periods_per_year=ppy)
            # 懲罰完全不動作的退化解(全部持有/全部同一動作),避免 CMA-ES
            # 學到「什麼都不做」這種平凡但沒意義的高 Sharpe(0/0 情況)
            action_diversity_penalty = 0.0
            if len(np.unique(actions)) == 1:
                action_diversity_penalty = 1.0
            # l2 / turnover_pen 是抗過擬合的正則項(預設 0 = 舊行為)
            return -(result.sharpe) + action_diversity_penalty + l2 * float(np.mean(theta**2)) + turnover_pen * result.turnover
        except Exception:
            return 1e6  # 模擬中發生數值錯誤(如爆炸)時給極差分數

    return _fn


def train_cma(
    agent: ConnectomeTradingAgent,
    features_train: np.ndarray,
    close_prices_train: np.ndarray,
    sigma0: float = 0.3,
    max_iter: int = 60,
    popsize: int | None = None,
    seed: int = 0,
    verbose: bool = True,
    x0: np.ndarray | None = None,
    fee_bps: float = 5.0,
    ppy: float = 24 * 365,
    l2: float = 0.0,
    turnover_pen: float = 0.0,
    expo_floor: float = 0.0,
    extra: list | None = None,
    pcfg: dict | None = None,
):
    """extra: 額外資產的 [(features, close), ...](池化訓練,fitness = 各資產 Sharpe 的平均)。
    pcfg: 部位映射設定,見 backtest/position.py(None = 舊行為)。

    x0: 若提供,CMA-ES 會從這個參數點開始搜尋(「warm start」),而不是
    隨機初始化。run_evolution.py 用這個機制讓每一次執行都能接著上一次
    存下來的 champion 繼續進化,而不是每次從頭亂猜。
    """
    import cma

    x0 = x0 if x0 is not None else agent.random_init(seed=seed)
    sets = [(features_train, close_prices_train)] + list(extra or [])

    def batch_fn(thetas):
        thetas = np.asarray(thetas)
        tot = np.zeros(len(thetas))
        for f, c in sets:
            lg = agent.logits_batch(f, thetas)
            for i in range(len(thetas)):
                pos = to_positions(lg[i], c, pcfg, ppy)
                r = run_backtest_pos(c, pos, fee_bps=fee_bps, periods_per_year=ppy)
                # 曝險下限:平均 |部位| 低於 expo_floor 就重罰,擋掉「99% 空手、賭幾筆單子」的退化解
                pen = 1.0 if np.ptp(pos) < 1e-12 else 0.0
                if expo_floor > 0:
                    pen += 5.0 * max(0.0, expo_floor - float(np.mean(np.abs(pos)))) / expo_floor
                tot[i] += -r.sharpe + pen + turnover_pen * r.turnover
        return list(tot / len(sets) + l2 * np.mean(np.square(thetas), axis=1))

    opts = {"maxiter": max_iter, "seed": seed, "verbose": -9}
    if popsize is not None:
        opts["popsize"] = popsize

    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
    history = []
    while not es.stop():
        solutions = es.ask()
        fitnesses = batch_fn(solutions)
        es.tell(solutions, fitnesses)
        best = -min(fitnesses)
        history.append(best)
        if verbose:
            print(f"[cma] iter={len(history):3d}  best_sharpe_so_far={max(history):.3f}  this_gen_best={best:.3f}")

    return es.result.xbest, history
