"""Action decoder: maps buy and sell activity scores to discrete trading actions.

Contract (from research/SPEC.md & research.md §4 Step 3):
- decode(buy, sell) -> np.ndarray(0=SELL, 1=BUY)
- Tie rule: buy_score > sell_score -> BUY (1), otherwise (buy_score <= sell_score) -> SELL (0)
"""
from __future__ import annotations

import numpy as np

ACTION_SELL = 0
ACTION_BUY = 1


def decode(
    buy: np.ndarray | float | int | list,
    sell: np.ndarray | float | int | list,
) -> np.ndarray:
    """Decode buy and sell scores into discrete actions (0=SELL, 1=BUY).

    Parameters
    ----------
    buy: array-like or float
        Activity score for BUY motor neurons.
    sell: array-like or float
        Activity score for SELL motor neurons.

    Returns
    -------
    np.ndarray
        Array of integer actions: 1 for BUY, 0 for SELL.
        Ties strictly resolve to SELL (0).
    """
    buy_arr = np.asarray(buy, dtype=np.float64)
    sell_arr = np.asarray(sell, dtype=np.float64)

    if buy_arr.shape != sell_arr.shape:
        raise ValueError(
            f"Shape mismatch in action decoder: buy shape {buy_arr.shape} != sell shape {sell_arr.shape}"
        )

    # buy_score > sell_score -> 1 (BUY), otherwise -> 0 (SELL)
    actions = np.where(buy_arr > sell_arr, ACTION_BUY, ACTION_SELL).astype(np.int64)

    # Ensure scalar inputs return at least 1D array per contract
    if actions.ndim == 0:
        actions = np.array([int(actions)], dtype=np.int64)

    return actions


class ActionDecoder:
    """Class wrapper for action decoding."""

    def __init__(self, tie_rule: str = "SELL"):
        if tie_rule != "SELL":
            raise ValueError(f"Only tie_rule='SELL' is supported by current spec, got '{tie_rule}'")
        self.tie_rule = tie_rule

    def decode(
        self,
        buy: np.ndarray | float | int | list,
        sell: np.ndarray | float | int | list,
    ) -> np.ndarray:
        return decode(buy, sell)
