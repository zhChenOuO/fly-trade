"""logits -> 部位。全部只用 t 時刻(含)以前的資訊(因果)。

pcfg(全部可省略,省略 = 舊行為:argmax -> {-1,0,+1}):
  long_only  : True 時「賣」與「空手」合併成空手,只剩 {0,+1}
  margin     : 遲滯。要換到別的部位,新部位 logit 必須比目前部位高出 margin 以上
  min_hold   : 換倉後至少持有幾根才准再換
  vol_target : 年化目標波動(如 0.4);部位乘 min(1, vol_target / 近期已實現年化波動)
  vol_window : 已實現波動的回看根數(預設 60,用到 t 為止的報酬,因果)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def vol_scale(close: np.ndarray, target: float, window: int, ppy: float) -> np.ndarray:
    r = np.zeros(len(close)); r[1:] = close[1:] / close[:-1] - 1.0
    sd = pd.Series(r).rolling(window, min_periods=max(10, window // 3)).std().values
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.minimum(1.0, target / (sd * np.sqrt(ppy)))
    return np.where(np.isfinite(s), s, 1.0)  # 暖機不足時不縮


def to_positions(logits: np.ndarray, close: np.ndarray, pcfg: dict | None, ppy: float) -> np.ndarray:
    p = pcfg or {}
    if p.get("long_only"):
        lg, lv = np.stack([np.maximum(logits[:, 0], logits[:, 1]), logits[:, 2]], 1), np.array([0.0, 1.0])
    else:
        lg, lv = logits, np.array([-1.0, 0.0, 1.0])
    margin, mh = p.get("margin", 0.0), p.get("min_hold", 1)
    if margin <= 0 and mh <= 1:
        st = lg.argmax(1)
    else:  # 狀態機,起始為空手
        rows, K = lg.tolist(), lg.shape[1]
        cur, held, st = (K - 1) // 2 if K == 3 else 0, mh, np.empty(len(rows), dtype=np.int64)
        for t, row in enumerate(rows):
            b = max(range(K), key=row.__getitem__)
            if b != cur and row[b] - row[cur] > margin and held >= mh:
                cur, held = b, 0
            held += 1
            st[t] = cur
    pos = lv[st]
    if p.get("vol_target"):
        pos = pos * vol_scale(close, p["vol_target"], p.get("vol_window", 60), ppy)
    return pos
