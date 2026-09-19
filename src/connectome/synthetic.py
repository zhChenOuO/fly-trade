"""合成連接體產生器。

真實的 hemibrain / FlyWire 連接體需要申請帳號才能下載(見
hemibrain_loader.py / flywire_loader.py),申請下來之前、或只是想
先跑通整個 pipeline 驗證架構時,可以用這個模組生一個「統計特性
接近真實果蠅腦」的合成圖:

- 神經元數量、每個神經元的平均出/入分支數(degree)可調整,預設抓
  接近 hemibrain 的量級(對於一般 M1 筆電,建議先用幾千顆神經元的
  子集合做原型,完整 13 萬顆神經元 / 3 千萬個突觸的規模只建議在
  「只訓練讀出層、reservoir 權重固定不動」的模式下嘗試)。
- 遵守 Dale's law:每個神經元只能是興奮性或抑制性,不會同一個神經元
  同時輸出正負權重(這是真實神經系統的限制,合成資料也照做以求更
  接近真實動態)。
- 用小世界(Watts–Strogatz 概念的簡化版:局部群聚 + 少量長程捷徑)
  拓撲近似連接體「模組化 + 少數跨腦區長程連結」的特性。
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .schema import ConnectomeGraph


def make_synthetic_connectome(
    n_neurons: int = 5000,
    avg_out_degree: float = 8.0,
    long_range_fraction: float = 0.05,
    frac_inhibitory: float = 0.2,
    n_sensory: int = 64,
    n_motor: int = 16,
    seed: int = 0,
) -> ConnectomeGraph:
    """產生一個小世界拓撲 + Dale's law 的合成連接體。

    Parameters
    ----------
    n_neurons: 神經元總數
    avg_out_degree: 每個神經元平均輸出多少個突觸連結
    long_range_fraction: 有多少比例的連結是「長程」(隨機遠距,而非
        局部群聚),模擬跨腦區投射神經元
    frac_inhibitory: 抑制性神經元佔比(果蠅腦內大約 20-30% 是 GABA
        能/抑制性神經元,這裡用 0.2 當預設)
    n_sensory: 指定多少個神經元當作「感覺輸入」埠(市場特徵從這裡注入)
    n_motor: 指定多少個神經元當作「運動輸出」埠(讀出層從這裡讀活動)
    """
    rng = np.random.default_rng(seed)

    # 1. 決定每個神經元是興奮性 (+1) 還是抑制性 (-1) —— Dale's law
    is_inhibitory = rng.random(n_neurons) < frac_inhibitory
    sign = np.where(is_inhibitory, -1.0, 1.0)

    # 2. 局部群聚連結:把神經元排在一個環上,傾向連到附近的神經元
    #    (近似腦區內部的密集局部連結)
    n_local_edges = int(n_neurons * avg_out_degree * (1 - long_range_fraction))
    local_src = rng.integers(0, n_neurons, size=n_local_edges)
    offset = rng.integers(1, max(2, n_neurons // 20), size=n_local_edges)
    direction = rng.choice([-1, 1], size=n_local_edges)
    local_dst = (local_src + offset * direction) % n_neurons

    # 3. 長程隨機連結(近似投射神經元跨腦區長距離連線)
    n_long_edges = int(n_neurons * avg_out_degree * long_range_fraction)
    long_src = rng.integers(0, n_neurons, size=n_long_edges)
    long_dst = rng.integers(0, n_neurons, size=n_long_edges)

    src = np.concatenate([local_src, long_src])
    dst = np.concatenate([local_dst, long_dst])
    keep = src != dst  # 不要自我連結
    src, dst = src[keep], dst[keep]

    # 4. 權重大小:突觸強度常近似 log-normal 分布(少數強連結,大量弱連結)
    magnitude = rng.lognormal(mean=0.0, sigma=0.7, size=src.shape[0])
    values = magnitude * sign[src]  # 每條邊的正負號由「前突觸神經元」的興奮/抑制性決定

    # W[i, j] = 神經元 j -> i,所以 row=dst(後突觸), col=src(前突觸)
    W = sp.coo_matrix((values, (dst, src)), shape=(n_neurons, n_neurons))
    W = W.tocsr()
    W.sum_duplicates()

    # 5. 隨機挑一批神經元當感覺輸入 / 運動輸出(盡量不重疊)
    perm = rng.permutation(n_neurons)
    sensory_idx = perm[:n_sensory]
    motor_idx = perm[n_sensory : n_sensory + n_motor]

    neuron_ids = [f"SYN{i:06d}" for i in range(n_neurons)]
    neuron_types = [
        ("inhibitory" if is_inhibitory[i] else "excitatory") for i in range(n_neurons)
    ]
    for i in sensory_idx:
        neuron_types[i] += "_sensory"
    for i in motor_idx:
        neuron_types[i] += "_motor"

    return ConnectomeGraph(
        weights=W,
        neuron_ids=neuron_ids,
        neuron_types=neuron_types,
        sensory_idx=sensory_idx,
        motor_idx=motor_idx,
        meta={
            "source": "synthetic",
            "seed": seed,
            "avg_out_degree": avg_out_degree,
            "long_range_fraction": long_range_fraction,
        },
    )
