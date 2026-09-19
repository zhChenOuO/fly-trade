"""載入 FlyWire 全腦連接體(約 13-14 萬顆神經元、3000 萬個突觸連結)。

FlyWire 的 Codex(https://codex.flywire.ai)FAQ 明確說明:他們「不」
提供可以一直重複呼叫的大量查詢 API,而是希望使用者用網頁介面或
下載連結取得靜態檔案,再自己本機處理。所以這個 loader 讀的是「你
已經下載好的本機 CSV 檔案」,而不是即時打 API。

取得資料步驟:
1. 到 https://codex.flywire.ai 註冊帳號,登入後在帳號頁面複製 API token
   (若之後只需下載一次靜態檔案,也可以直接從網頁的 Download 頁面手動下載)
2. 下載至少兩個檔案到本機資料夾:
   - connections.csv:欄位通常包含 pre_root_id, post_root_id,
     syn_count(突觸數量), nt_type(神經傳導物質類型)
   - classification.csv 或 cell_stats.csv:欄位包含 root_id 與
     cell_type / super_class / flow(可用來判斷 sensory/motor)
   實際欄位名稱請以你下載當下的檔案為準(FlyWire 資料集仍在持續更新版本)。
3. 把檔案路徑傳給 load_flywire_connectome()。

規模提醒:全量 FlyWire 資料(13萬神經元 x 3000萬連結)用稀疏矩陣儲存
大約需要 1-2GB 記憶體,讀取與建構矩陣可能要幾分鐘,在 16GB 以上的
M1 Mac 上可行,但不建議在讀出層訓練迴圈中重複重建這個矩陣——先建好、
存成 .npz 快取,訓練時每次載入快取即可(見 build_graph.py 的
save_cache/load_cache)。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .schema import ConnectomeGraph

# nt_type 常見標記中,GABA 與 Glutamate 通常視為抑制性,其餘
# (ACh, dopamine, serotonin, octopamine 等)在這裡簡化當作興奮性。
INHIBITORY_NT = {"GABA", "GLUTAMATE"}

SENSORY_KEYWORDS = ("sensory", "photoreceptor", "olfactory", "ORN")
MOTOR_KEYWORDS = ("motor", "descending", "DN")


def load_flywire_connectome(
    connections_csv: str,
    classification_csv: str,
    max_neurons: int | None = None,
) -> ConnectomeGraph:
    conn = pd.read_csv(connections_csv)
    cls = pd.read_csv(classification_csv)

    id_col_candidates = [c for c in cls.columns if c.lower() in ("root_id", "pt_root_id", "id")]
    if not id_col_candidates:
        raise ValueError(
            f"在 {classification_csv} 裡找不到神經元 ID 欄位,"
            f"現有欄位有: {list(cls.columns)}。請確認下載的檔案格式。"
        )
    id_col = id_col_candidates[0]

    if max_neurons is not None:
        cls = cls.iloc[:max_neurons]

    root_ids = cls[id_col].tolist()
    id_to_idx = {rid: i for i, rid in enumerate(root_ids)}
    n_neurons = len(root_ids)

    pre_col = "pre_root_id" if "pre_root_id" in conn.columns else conn.columns[0]
    post_col = "post_root_id" if "post_root_id" in conn.columns else conn.columns[1]
    weight_col = "syn_count" if "syn_count" in conn.columns else conn.columns[2]
    nt_col = "nt_type" if "nt_type" in conn.columns else None

    rows, cols_, vals = [], [], []
    for _, r in conn.iterrows():
        pre, post = r[pre_col], r[post_col]
        if pre not in id_to_idx or post not in id_to_idx:
            continue
        j = id_to_idx[pre]
        i = id_to_idx[post]
        w = float(r[weight_col])
        if nt_col and str(r.get(nt_col, "")).upper() in INHIBITORY_NT:
            w = -w
        rows.append(i)
        cols_.append(j)
        vals.append(w)

    W = sp.coo_matrix((vals, (rows, cols_)), shape=(n_neurons, n_neurons)).tocsr()

    type_col_candidates = [c for c in cls.columns if c.lower() in ("cell_type", "super_class", "class", "flow")]
    type_col = type_col_candidates[0] if type_col_candidates else None
    types = cls[type_col].fillna("unknown").astype(str).tolist() if type_col else ["unknown"] * n_neurons

    def matches(t: str, keywords: tuple[str, ...]) -> bool:
        tl = t.lower()
        return any(k.lower() in tl for k in keywords)

    sensory_idx = np.array([i for i, t in enumerate(types) if matches(t, SENSORY_KEYWORDS)])
    motor_idx = np.array([i for i, t in enumerate(types) if matches(t, MOTOR_KEYWORDS)])

    if len(sensory_idx) == 0 or len(motor_idx) == 0:
        raise RuntimeError(
            "依現有型別欄位找不到 sensory/motor 神經元,請檢查 "
            f"{classification_csv} 的型別欄位內容,或改用隨機挑選的方式 "
            "(可參考 synthetic.py)自行指定 sensory_idx / motor_idx。"
        )

    return ConnectomeGraph(
        weights=W,
        neuron_ids=[str(r) for r in root_ids],
        neuron_types=types,
        sensory_idx=sensory_idx,
        motor_idx=motor_idx,
        meta={"source": "flywire"},
    )
