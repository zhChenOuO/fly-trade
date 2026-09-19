"""把市場特徵向量編碼成注入到感覺神經元的電流。

這一層的權重是「會被訓練」的少數參數之一(見 train/evolve.py)。
"""
from __future__ import annotations

import numpy as np


class LinearEncoder:
    def __init__(self, n_features: int, n_sensory: int):
        self.n_features = n_features
        self.n_sensory = n_sensory
        self.n_params = n_features * n_sensory

    def encode(self, features: np.ndarray, weights_flat: np.ndarray) -> np.ndarray:
        """features: (T, n_features) -> currents: (T, n_sensory)"""
        W = weights_flat.reshape(self.n_sensory, self.n_features)
        return features @ W.T
