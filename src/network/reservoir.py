"""把連接體拓撲當作一個固定不變的「大型遞迴儲備池(reservoir)」。

核心想法(reservoir computing / echo-state network 的做法,套用在
昆蟲連接體上是有文獻先例的,例如用果蠅嗅覺迴路的固定連結做分類器):

- 連接體本身的 3000 萬個突觸權重「完全不訓練」,只當作固定的隨機
  (但有生物結構的)遞迴矩陣,神經動態自然就有豐富、非線性、帶記憶
  的時間動態。
- 真正拿去訓練的參數只有:
    1. 輸入編碼權重(市場特徵 -> 感覺神經元電流),數量 = n_sensory x n_features
    2. 讀出層權重(運動神經元活動 -> 買/賣/持有 logits),數量 = n_motor x n_actions
    3. 兩個全域純量:gain(整體訊號放大倍率)、leak(神經元的漏電/
       時間常數)
  參數量通常只有幾百到幾千個,即使連接體是 13 萬顆神經元的全腦規模,
  訓練這幾百個參數的計算量跟連接體大小基本無關(每一步都只是一次
  稀疏矩陣乘法,不需要對 3000 萬個權重做反向傳播),這就是為什麼這個
  方法在 M1 筆電上可行。

動態方程(leaky-rate 模型,比逐一模擬 spike 更省算力,但仍保留
「有記憶的非線性遞迴系統」這個核心性質):

    x[t+1] = (1 - leak) * x[t] + leak * tanh( gain * (W @ x[t]) + I[t] )

其中 W 是連接體的稀疏權重矩陣,I[t] 是這一步注入到感覺神經元的
外部輸入電流(其他神經元 I=0)。
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from ..connectome.schema import ConnectomeGraph


class LeakyReservoir:
    def __init__(self, graph: ConnectomeGraph):
        self.graph = graph
        self.W: sp.csr_matrix = graph.weights
        self.n = graph.n_neurons
        self.sensory_idx = graph.sensory_idx
        self.motor_idx = graph.motor_idx

    def simulate(
        self,
        input_currents: np.ndarray,  # shape (T, n_sensory)
        gain: float = 1.0,
        leak: float = 0.5,
        x0: np.ndarray | None = None,
        record_motor_window: int | None = None,
    ) -> np.ndarray:
        """跑 T 步遞迴動態,回傳運動神經元的活動軌跡 (T, n_motor)。

        record_motor_window: 若設定,回傳的是每個時間點「過去 N 步」
        運動神經元活動的平均值,而不是單步瞬時值——這樣讀出層決策時
        看到的是一段時間的整合訊號,雜訊較小,也比較符合真實神經系統
        用時間積分做決策的特性。
        """
        T = input_currents.shape[0]
        x = np.zeros(self.n, dtype=np.float64) if x0 is None else x0.copy()

        motor_trace = np.zeros((T, len(self.motor_idx)), dtype=np.float64)

        I = np.zeros(self.n, dtype=np.float64)
        for t in range(T):
            I[:] = 0.0
            I[self.sensory_idx] = input_currents[t]

            rec = self.W.dot(x)  # 稀疏矩陣 x 向量,O(n_synapses) 而非 O(n_neurons^2)
            x = (1.0 - leak) * x + leak * np.tanh(gain * rec + I)

            motor_trace[t] = x[self.motor_idx]

        if record_motor_window and record_motor_window > 1:
            # 只用「過去 N 步」的尾端均值(trailing mean),不能用置中窗口,
            # 否則 t 時刻的決策會看到 t+1..t+N/2 的輸入 = look-ahead。
            w = record_motor_window
            c = np.vstack([np.zeros((1, motor_trace.shape[1])), np.cumsum(motor_trace, axis=0)])
            hi = np.arange(1, T + 1)
            lo = np.maximum(hi - w, 0)
            motor_trace = (c[hi] - c[lo]) / (hi - lo)[:, None]

        return motor_trace

    def simulate_batch(self, currents: np.ndarray, gain: np.ndarray, leak: np.ndarray,
                       record_motor_window: int | None = None) -> np.ndarray:
        """一次模擬 P 個參數不同、連接體相同的 reservoir(CMA-ES 的整個 population)。

        currents: (P, T, n_sensory); gain, leak: (P,)  ->  (P, T, n_motor)
        稀疏矩陣 x 稠密矩陣 (n, P) 比 P 次矩陣 x 向量快很多,也是搬上 GPU 的同一個形狀。
        """
        P, T, _ = currents.shape
        X = np.zeros((self.n, P))
        g, lk = gain[None, :], leak[None, :]
        cur = np.ascontiguousarray(currents.transpose(1, 2, 0))  # (T, n_s, P)
        motor = np.empty((T, len(self.motor_idx), P))
        for t in range(T):
            Z = g * self.W.dot(X)
            Z[self.sensory_idx] += cur[t]
            X = (1.0 - lk) * X + lk * np.tanh(Z)
            motor[t] = X[self.motor_idx]
        motor = motor.transpose(2, 0, 1)  # (P, T, n_m)
        if record_motor_window and record_motor_window > 1:
            w = record_motor_window
            c = np.concatenate([np.zeros((P, 1, motor.shape[2])), np.cumsum(motor, axis=1)], axis=1)
            hi = np.arange(1, T + 1); lo = np.maximum(hi - w, 0)
            motor = (c[:, hi] - c[:, lo]) / (hi - lo)[None, :, None]
        return motor
