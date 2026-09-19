"""從 Janelia neuprint 下載 hemibrain 連接體(真實果蠅腦部分連接體,約 2.5 萬顆神經元)。

使用前準備:
1. 到 https://neuprint.janelia.org 註冊帳號並登入
2. 右上角 Account 頁面複製你的 personal auth token
3. 設定環境變數: export NEUPRINT_APPLICATION_CREDENTIALS="你的token"
   (或是呼叫本模組函式時直接傳入 token 參數)

注意規模與效能:
- hemibrain v1.2.1 大約 2.5 萬顆神經元、2000 萬個突觸連結,下載 +
  組成稀疏矩陣在 M1 Mac(16GB 記憶體以上)是可行的,但第一次下載
  會花上幾分鐘到十幾分鐘不等,且需要穩定網路。
- 如果只是想先驗證程式邏輯,建議先用 synthetic.py 產生的小規模合成
  連接體跑通整條 pipeline,再換成這裡的真實資料。
"""
from __future__ import annotations

import os
import numpy as np
import scipy.sparse as sp

from .schema import ConnectomeGraph

# 依「細胞類型名稱」關鍵字,粗略判斷哪些神經元適合當作 sensory / motor 埠。
# hemibrain 的 type 欄位常見前綴:
#   ORN_*  = 嗅覺受器神經元 (olfactory receptor neuron) -> 感覺輸入
#   PN_*   = projection neuron (嗅覺訊號往下傳遞)        -> 也可視為輸入端
#   DNp*/DNa*/DNg* = descending neuron (下行到胸神經節控制行為) -> 運動輸出
SENSORY_TYPE_PREFIXES = ("ORN", "PN")
MOTOR_TYPE_PREFIXES = ("DNp", "DNa", "DNg", "DNb")


def load_hemibrain_connectome(
    dataset: str = "hemibrain:v1.2.1",
    server: str = "neuprint.janelia.org",
    token: str | None = None,
    max_neurons: int | None = None,
) -> ConnectomeGraph:
    """從 neuprint 抓 hemibrain 連接體並組成 ConnectomeGraph。

    max_neurons: 若設定,會先取前 N 顆神經元做子集合(依 bodyId 排序),
        適合在完整下載前先用較小規模測試記憶體/速度。
    """
    try:
        from neuprint import Client, fetch_adjacencies, fetch_neurons, NeuronCriteria as NC
    except ImportError as e:
        raise ImportError(
            "需要先安裝 neuprint-python: pip install neuprint-python"
        ) from e

    token = token or os.environ.get("NEUPRINT_APPLICATION_CREDENTIALS")
    if not token:
        raise RuntimeError(
            "找不到 neuprint API token。請到 https://neuprint.janelia.org "
            "登入後在 Account 頁面複製 token,並設定環境變數 "
            "NEUPRINT_APPLICATION_CREDENTIALS,或是呼叫本函式時傳入 token 參數。"
        )

    client = Client(server, dataset=dataset, token=token)

    # 1. 取得神經元清單與型別標註
    neuron_df, _ = fetch_neurons(NC(status="Traced"), client=client)
    if max_neurons is not None:
        neuron_df = neuron_df.sort_values("bodyId").iloc[:max_neurons]

    body_ids = neuron_df["bodyId"].tolist()
    id_to_idx = {bid: i for i, bid in enumerate(body_ids)}
    n_neurons = len(body_ids)

    # 2. 取得這批神經元之間的連結權重(pre -> post,weight = 突觸數)
    #    hemibrain 沒有直接標示興奮/抑制,這裡用一個簡化假設:
    #    以神經傳導物質(neurotransmitter)欄位判斷,GABA/Glutamate 常見於
    #    抑制性;若欄位缺失則預設興奮性。實務上請依研究需求調整這個假設。
    _, conn_df = fetch_adjacencies(sources=NC(bodyId=body_ids), targets=NC(bodyId=body_ids), client=client)

    rows, cols, vals = [], [], []
    for pre, post, weight in zip(conn_df["bodyId_pre"], conn_df["bodyId_post"], conn_df["weight"]):
        if pre not in id_to_idx or post not in id_to_idx:
            continue
        j = id_to_idx[pre]   # 前突觸(輸入端)
        i = id_to_idx[post]  # 後突觸(輸出端), W[i, j]
        rows.append(i)
        cols.append(j)
        vals.append(float(weight))

    W = sp.coo_matrix((vals, (rows, cols)), shape=(n_neurons, n_neurons)).tocsr()

    types = neuron_df.get("type", neuron_df.get("instance", ["unknown"] * n_neurons)).fillna("unknown").tolist()

    sensory_idx = np.array(
        [i for i, t in enumerate(types) if str(t).startswith(SENSORY_TYPE_PREFIXES)]
    )
    motor_idx = np.array(
        [i for i, t in enumerate(types) if str(t).startswith(MOTOR_TYPE_PREFIXES)]
    )

    if len(sensory_idx) == 0 or len(motor_idx) == 0:
        raise RuntimeError(
            "在這個神經元子集合裡找不到符合命名規則的感覺/運動神經元 "
            f"(sensory={len(sensory_idx)}, motor={len(motor_idx)})。"
            "如果你用 max_neurons 限制了子集合,請放寬或改成隨機挑選 "
            "sensory_idx / motor_idx(可參考 synthetic.py 的做法)。"
        )

    return ConnectomeGraph(
        weights=W,
        neuron_ids=[str(b) for b in body_ids],
        neuron_types=[str(t) for t in types],
        sensory_idx=sensory_idx,
        motor_idx=motor_idx,
        meta={"source": "hemibrain", "dataset": dataset},
    )
