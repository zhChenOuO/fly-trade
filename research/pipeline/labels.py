"""Research v2 label generation pipeline.

Pure mathematical functions for computing multi-horizon continuous returns,
forward volatility, maximum favorable/adverse excursions (MFE/MAE), and
cost-aware discrete action labels (BUY/HOLD/SELL).

Contract (from research_v2.md §4.2, §4.3 and task_v2_batch1.md):
- future_return_{h} = log(close[t+h] / close[t]) for h in {1, 3, 6, 12}.
- future_volatility_6: sample std (ddof=1) of forward 5m bar log returns in (t, t+6].
- maximum_favorable_excursion_6: log(max(high_{(t, t+6]}) / close[t]).
- maximum_adverse_excursion_6: log(min(low_{(t, t+6]}) / close[t]).
- Action labels:
    future_return_6 > +cost_threshold -> BUY
    future_return_6 < -cost_threshold -> SELL
    otherwise                         -> HOLD
  where cost_threshold = 2 * fee_per_side + slippage.
  Strict boundary: returns exactly equal to +/- cost_threshold are HOLD.
- Split boundary rule:
  Samples whose horizon targets cross into the next split or purge area
  (t + h >= split_stop or t + h >= len(raw)) have that horizon's label set to NaN.
  Never fill NaNs across split boundaries.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ACTION_BUY = "BUY"
ACTION_HOLD = "HOLD"
ACTION_SELL = "SELL"


def compute_cost_threshold(cost_fee_per_side: float, cost_slippage: float) -> float:
    """Derive round-trip cost threshold from single-side fee and slippage.

    Formula: cost_threshold = 2 * cost_fee_per_side + cost_slippage
    """
    return float(2.0 * cost_fee_per_side + cost_slippage)


def classify_action(
    returns: float | np.ndarray,
    cost_threshold: float,
) -> str | np.ndarray:
    """Classify future return into BUY, HOLD, or SELL given a cost threshold.

    Parameters
    ----------
    returns: float or np.ndarray
        Future return values (typically future_return_6).
    cost_threshold: float
        Positive round-trip cost threshold.

    Returns
    -------
    str or np.ndarray of object/str
        'BUY' if return > cost_threshold
        'SELL' if return < -cost_threshold
        'HOLD' if -cost_threshold <= return <= cost_threshold
        np.nan if return is NaN
    """
    if np.ndim(returns) == 0:
        val = float(returns)
        if np.isnan(val):
            return np.nan
        if val > cost_threshold:
            return ACTION_BUY
        if val < -cost_threshold:
            return ACTION_SELL
        return ACTION_HOLD

    arr = np.asarray(returns, dtype=np.float64)
    res = np.full(arr.shape, ACTION_HOLD, dtype=object)
    res[arr > cost_threshold] = ACTION_BUY
    res[arr < -cost_threshold] = ACTION_SELL
    res[np.isnan(arr)] = np.nan
    return res


def compute_future_returns(
    closes: np.ndarray,
    positions: np.ndarray,
    horizons: list[int] = (1, 3, 6, 12),
) -> dict[int, np.ndarray]:
    """Compute forward log returns for multiple prediction horizons.

    Parameters
    ----------
    closes: np.ndarray
        1D array of close prices.
    positions: np.ndarray
        1D integer array of index t (last input candle) for each sample.
    horizons: list of int
        Forward candle horizons to evaluate.

    Returns
    -------
    dict[int, np.ndarray]
        Mapping from horizon h to 1D float64 array of log(close[t+h] / close[t]).
    """
    closes = np.asarray(closes, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.int64)
    n_samples = len(positions)
    n_bars = len(closes)

    out = {}
    c_t = closes[positions]

    for h in horizons:
        ret = np.full(n_samples, np.nan, dtype=np.float64)
        valid = (positions + h < n_bars) & (positions >= 0)
        valid_pos = positions[valid]
        ret[valid] = np.log(closes[valid_pos + h] / c_t[valid])
        out[h] = ret

    return out


def compute_future_volatility_6(
    closes: np.ndarray,
    positions: np.ndarray,
    horizon: int = 6,
    ddof: int = 1,
) -> np.ndarray:
    """Compute forward volatility over (t, t+horizon] bars.

    Defined as the sample standard deviation (ddof=1) of the 5-minute
    close-to-close log returns within (t, t+horizon].

    Parameters
    ----------
    closes: np.ndarray
        1D array of close prices.
    positions: np.ndarray
        1D integer array of index t for each sample.
    horizon: int
        Forward window size (default 6 bars).
    ddof: int
        Delta degrees of freedom (default 1 for sample std).

    Returns
    -------
    np.ndarray
        1D float64 array of future volatility values.
    """
    closes = np.asarray(closes, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.int64)
    n_samples = len(positions)
    n_bars = len(closes)

    vol = np.full(n_samples, np.nan, dtype=np.float64)
    valid = (positions + horizon < n_bars) & (positions >= 0)
    if not np.any(valid):
        return vol

    val_pos = positions[valid]
    # Future bars: t+1, ..., t+horizon
    # Forward closes of shape (n_valid, horizon)
    k_bars = np.arange(1, horizon + 1)
    fut_closes = closes[val_pos[:, None] + k_bars]  # (n_valid, horizon)
    entry_closes = closes[val_pos, None]            # (n_valid, 1)

    prev_closes = np.column_stack([entry_closes, fut_closes[:, :-1]])
    bar_returns = np.log(fut_closes / prev_closes)  # (n_valid, horizon)

    vol[valid] = np.std(bar_returns, axis=1, ddof=ddof)
    return vol


def compute_excursions_6(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    positions: np.ndarray,
    horizon: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Maximum Favorable Excursion (MFE) and Maximum Adverse Excursion (MAE).

    Using exclusively bars in (t, t+horizon]:
    MFE_6 = log(max_{k=1..horizon}(high[t+k]) / close[t])
    MAE_6 = log(min_{k=1..horizon}(low[t+k]) / close[t])

    Parameters
    ----------
    highs: np.ndarray
        1D array of high prices.
    lows: np.ndarray
        1D array of low prices.
    closes: np.ndarray
        1D array of close prices.
    positions: np.ndarray
        1D integer array of index t for each sample.
    horizon: int
        Forward window size (default 6 bars).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (mfe_6, mae_6) arrays of shape (n_samples,).
    """
    highs = np.asarray(highs, dtype=np.float64)
    lows = np.asarray(lows, dtype=np.float64)
    closes = np.asarray(closes, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.int64)
    n_samples = len(positions)
    n_bars = len(closes)

    mfe = np.full(n_samples, np.nan, dtype=np.float64)
    mae = np.full(n_samples, np.nan, dtype=np.float64)

    valid = (positions + horizon < n_bars) & (positions >= 0)
    if not np.any(valid):
        return mfe, mae

    val_pos = positions[valid]
    k_bars = np.arange(1, horizon + 1)

    fut_highs = highs[val_pos[:, None] + k_bars]  # (n_valid, horizon)
    fut_lows = lows[val_pos[:, None] + k_bars]    # (n_valid, horizon)
    entry_closes = closes[val_pos]                # (n_valid,)

    max_high = np.max(fut_highs, axis=1)
    min_low = np.min(fut_lows, axis=1)

    mfe[valid] = np.log(max_high / entry_closes)
    mae[valid] = np.log(min_low / entry_closes)

    return mfe, mae


def compute_labels(
    raw: pd.DataFrame,
    positions: np.ndarray,
    split_names: pd.Series | np.ndarray | list[str],
    split_ranges: dict[str, list[int] | tuple[int, int]],
    cost_threshold: float,
    horizons: list[int] = (1, 3, 6, 12),
) -> pd.DataFrame:
    """Pure function generating complete Research v2 labels with boundary purge.

    Parameters
    ----------
    raw: pd.DataFrame
        Cleaned OHLCV DataFrame with columns ['open', 'high', 'low', 'close', 'volume'].
    positions: np.ndarray
        1D integer array of candle indices in `raw` for the last input candle of each sample.
    split_names: array-like
        Split name for each sample (e.g. 'train', 'val', 'dev_test_v1').
    split_ranges: dict
        Mapping from split name (or role) to [raw_start, raw_stop) raw bar index boundaries.
    cost_threshold: float
        Positive round-trip cost threshold.
    horizons: list of int
        Prediction horizons to compute (default [1, 3, 6, 12]).

    Returns
    -------
    pd.DataFrame
        DataFrame containing computed labels:
        - future_return_{h} for each h
        - future_volatility_6
        - maximum_favorable_excursion_6
        - maximum_adverse_excursion_6
        - action
    """
    closes = raw["close"].to_numpy(dtype=np.float64)
    highs = raw["high"].to_numpy(dtype=np.float64)
    lows = raw["low"].to_numpy(dtype=np.float64)
    positions = np.asarray(positions, dtype=np.int64)
    split_names = np.asarray(split_names)

    # 1. Multi-horizon continuous returns
    ret_dict = compute_future_returns(closes, positions, horizons=horizons)

    # 2. 6-bar forward volatility and excursions
    vol_6 = compute_future_volatility_6(closes, positions, horizon=6, ddof=1)
    mfe_6, mae_6 = compute_excursions_6(highs, lows, closes, positions, horizon=6)

    # 3. Apply split boundary / purge rules:
    # Any sample whose target bar reaches >= split_stop or >= len(raw) is set to NaN
    # Build stop lookup per sample
    split_stops = np.full(len(positions), len(closes), dtype=np.int64)
    for s_name, s_range in split_ranges.items():
        mask = split_names == s_name
        split_stops[mask] = s_range[1]

    for h in horizons:
        cross = (positions + h >= split_stops) | (positions + h >= len(closes))
        ret_dict[h][cross] = np.nan

    # 6-bar metrics cross check
    cross_6 = (positions + 6 >= split_stops) | (positions + 6 >= len(closes))
    vol_6[cross_6] = np.nan
    mfe_6[cross_6] = np.nan
    mae_6[cross_6] = np.nan

    # 4. Action labels based on future_return_6
    ret_6 = ret_dict[6]
    action = classify_action(ret_6, cost_threshold=cost_threshold)

    # Assemble output DataFrame
    data = {}
    for h in horizons:
        data[f"future_return_{h}"] = ret_dict[h]
    data["future_volatility_6"] = vol_6
    data["maximum_favorable_excursion_6"] = mfe_6
    data["maximum_adverse_excursion_6"] = mae_6
    data["action"] = action

    return pd.DataFrame(data)
