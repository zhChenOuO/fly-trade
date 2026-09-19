"""共用的連接體資料結構。

所有 loader(合成資料 / hemibrain / FlyWire)最後都要組出這個物件,
下游的 reservoir 模擬器只依賴這個介面,不管資料是哪裡來的。
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import scipy.sparse as sp


@dataclass
class ConnectomeGraph:
    # 帶正負號的突觸權重矩陣 (n_neurons x n_neurons),W[i, j] = 神經元 j -> i 的權重
    # 正值 = 興奮性 (excitatory),負值 = 抑制性 (inhibitory)
    weights: sp.csr_matrix

    # 每個神經元的顯示用 ID / 名稱(例如 hemibrain bodyId,或合成資料的 "N0001")
    neuron_ids: list[str]

    # 每個神經元的細胞類型標籤(例如 "ORN_DA1", "PN", "KC", "DN_local" ...),不確定時可為 "unknown"
    neuron_types: list[str]

    # 被選為「感覺輸入」的神經元 index(市場特徵會從這裡注入電流)
    sensory_idx: np.ndarray

    # 被選為「運動輸出」的神經元 index(讀出層會讀這些神經元的活動來決定買/賣/持有)
    motor_idx: np.ndarray

    meta: dict = field(default_factory=dict)

    @property
    def n_neurons(self) -> int:
        return self.weights.shape[0]

    @property
    def n_synapses(self) -> int:
        return self.weights.nnz

    def summary(self) -> str:
        return (
            f"ConnectomeGraph(n_neurons={self.n_neurons:,}, "
            f"n_synapses={self.n_synapses:,}, "
            f"n_sensory={len(self.sensory_idx)}, "
            f"n_motor={len(self.motor_idx)}, "
            f"source={self.meta.get('source', 'unknown')})"
        )
