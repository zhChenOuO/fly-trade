"""input_encoding_v2: 去除背景 DC 偏差的輸入編碼。mean 只來自 Train 並凍結。"""
from __future__ import annotations

import numpy as np

MODES = ("mean_sub", "zero_black", "on_off")


def encoder(mode: str, mean_img: np.ndarray):
    """回傳 encode(images uint8 (B,64,64,3)) -> (B, F) float64。mean_img: Train 平均圖 (64,64,3) 0-255。"""
    m = np.asarray(mean_img, dtype=np.float64)
    if mode == "mean_sub":
        return lambda x: ((x.astype(np.float64) - m) / 128.0).reshape(len(x), -1)
    if mode == "zero_black":
        return lambda x: (x.astype(np.float64) / 255.0).reshape(len(x), -1)
    if mode == "on_off":
        def enc(x):
            d = x.astype(np.float64) - m
            return np.concatenate([np.maximum(d, 0), np.maximum(-d, 0)], axis=-1).reshape(len(x), -1) / 128.0
        return enc
    raise ValueError(mode)


def calibrate_baseline(sim_factory, train_imgs: np.ndarray, chunk: int = 500) -> dict:
    """在 Train 圖片上量每神經元平均活動的均值/std(不含 label),凍結後給 z-score 用。"""
    sim = sim_factory(None)  # baseline=None -> 回傳總和,自行換算成每神經元平均
    n_b, n_s = sim._mid, len(sim.graph.motor_idx) - sim._mid
    outs = [sim.run(train_imgs[i:i + chunk]) for i in range(0, len(train_imgs), chunk)]
    b = np.concatenate([o["buy_score"] for o in outs]) / (sim.steps * n_b)
    s = np.concatenate([o["sell_score"] for o in outs]) / (sim.steps * n_s)
    return {"buy_mean": float(b.mean()), "buy_std": float(b.std()), "sell_mean": float(s.mean()), "sell_std": float(s.std())}
