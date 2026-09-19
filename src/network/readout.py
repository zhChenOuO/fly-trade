"""從運動神經元活動決定買/賣/持有動作。

動作定義(離散,3 類):
    0 = 賣出/放空 (sell)
    1 = 持有/空手 (hold)
    2 = 買進/做多 (buy)
"""
from __future__ import annotations

import numpy as np

ACTION_SELL, ACTION_HOLD, ACTION_BUY = 0, 1, 2
N_ACTIONS = 3


class LinearReadout:
    def __init__(self, n_motor: int):
        self.n_motor = n_motor
        self.n_params = n_motor * N_ACTIONS

    def logits(self, motor_trace: np.ndarray, weights_flat: np.ndarray) -> np.ndarray:
        """motor_trace: (T, n_motor) -> logits: (T, n_actions)"""
        W = weights_flat.reshape(N_ACTIONS, self.n_motor)
        return motor_trace @ W.T

    def actions(self, motor_trace: np.ndarray, weights_flat: np.ndarray) -> np.ndarray:
        logits = self.logits(motor_trace, weights_flat)
        return np.argmax(logits, axis=1)
