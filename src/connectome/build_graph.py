"""ConnectomeGraph 的存檔/讀取快取工具。

真實連接體(hemibrain / FlyWire)下載跟組矩陣很花時間,建好一次之後
應該存成本機檔案,之後訓練迴圈直接讀快取,不要每次重新下載。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from .schema import ConnectomeGraph


def save_cache(graph: ConnectomeGraph, path: str | Path) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    sp.save_npz(path / "weights.npz", graph.weights)
    np.save(path / "sensory_idx.npy", graph.sensory_idx)
    np.save(path / "motor_idx.npy", graph.motor_idx)
    with open(path / "meta.json", "w") as f:
        json.dump(
            {
                "neuron_ids": graph.neuron_ids,
                "neuron_types": graph.neuron_types,
                "meta": graph.meta,
            },
            f,
        )


def load_cache(path: str | Path) -> ConnectomeGraph:
    path = Path(path)
    weights = sp.load_npz(path / "weights.npz").tocsr()
    sensory_idx = np.load(path / "sensory_idx.npy")
    motor_idx = np.load(path / "motor_idx.npy")
    with open(path / "meta.json") as f:
        meta = json.load(f)
    return ConnectomeGraph(
        weights=weights,
        neuron_ids=meta["neuron_ids"],
        neuron_types=meta["neuron_types"],
        sensory_idx=sensory_idx,
        motor_idx=motor_idx,
        meta=meta["meta"],
    )
