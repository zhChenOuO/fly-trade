"""持久化「目前最好的一組參數」(champion),讓進化可以跨執行接續。

概念很簡單,對應你想要的「有賺錢就留著,下次再進化」:
- 每次執行 run_evolution.py,都是一次新的「世代」嘗試。
- 只有當這一代訓練出來的新解,在「樣本外(validation)資料」上真的
  贏過目前的 champion,才會取代它、存成新的 champion。
- 沒贏的話,舊 champion 原封不動留著,新的嘗試只會被記錄在歷史紀錄
  裡,不會覆蓋掉原本能用的解。

這樣不管你今天跑一次、明天再跑一次、還是完全不管它,champion 永遠
是「目前為止驗證過表現最好的那組參數」,不會因為某次運氣不好的訓練
就被洗掉。
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


class ChampionStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

    @property
    def theta_path(self) -> Path:
        return self.path / "theta.npy"

    @property
    def meta_path(self) -> Path:
        return self.path / "meta.json"

    @property
    def history_path(self) -> Path:
        return self.path / "history.csv"

    @property
    def forward_ledger_path(self) -> Path:
        return self.path / "forward_ledger.csv"

    def exists(self) -> bool:
        return self.theta_path.exists() and self.meta_path.exists()

    def load(self) -> tuple[np.ndarray, dict]:
        theta = np.load(self.theta_path)
        with open(self.meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        return theta, meta

    def save(self, theta: np.ndarray, meta: dict) -> None:
        np.save(self.theta_path, theta)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    def _append_csv(self, path: Path, row: dict) -> None:
        is_new = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if is_new:
                writer.writeheader()
            writer.writerow(row)

    def append_history(self, row: dict) -> None:
        """每次執行 run_evolution.py 的嘗試紀錄(不管有沒有取代 champion)。"""
        self._append_csv(self.history_path, row)

    def append_forward_ledger(self, row: dict) -> None:
        """每次執行 run_forward_test.py 的「樣外/未來」監控紀錄。"""
        self._append_csv(self.forward_ledger_path, row)
