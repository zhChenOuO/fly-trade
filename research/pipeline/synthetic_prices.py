"""Synthetic price generators for Phase 0 positive/negative control suites.

Implements SPEC_v3 §Phase 0 / 合成價格:
- trend: close = P0 * (1 + k * t) + noise (relative noise 0.02%), k in [-0.01, 0.01]
- vol: GBM with per-bar volatility sigma in {0.05%, 0.15%, 0.4%}
- ar1: log-returns follow AR(1) phi = 0.8
- gbm: zero-drift pure random walk (NC-1 negative control)
- Candle geometry: open, high, low, close expand realistically from close and
  strictly pass render_market.render() checks.
- Deterministic random seeds, with separate seeds for train and test splits.
"""
from __future__ import annotations

import numpy as np

# Fixed volatility levels for PC-2 (0.05%, 0.15%, 0.40%)
VOL_LEVELS = (0.0005, 0.0015, 0.0040)


def make_candles_from_close(
    close: np.ndarray,
    rng: np.random.Generator,
    intra_vol: float = 0.001,
) -> np.ndarray:
    """Expand a 1D close price sequence into an OHLCV window (T, 5) passing render()."""
    T = len(close)
    if T < 2:
        raise ValueError(f"Close price array must have length >= 2, got {T}")
    close = np.asarray(close, dtype=np.float64)
    if np.any(close <= 0) or not np.isfinite(close).all():
        raise ValueError("Close prices must be strictly positive and finite")

    open_p = np.empty(T, dtype=np.float64)
    open_p[0] = close[0] * (1.0 + rng.normal(0.0, intra_vol))
    open_p[1:] = close[:-1]

    m = np.minimum(open_p, close)
    M = np.maximum(open_p, close)
    spread = M - m
    min_spread = close * 0.0001
    spread = np.where(spread < min_spread, min_spread, spread)

    # Expanding high and low ensuring low <= min(open, close) and high >= max(open, close)
    high = M + np.abs(rng.normal(0.0, 1.0, size=T)) * spread + min_spread
    low = m - np.abs(rng.normal(0.0, 1.0, size=T)) * spread - min_spread
    # Volume: lognormal distribution
    volume = np.exp(rng.normal(10.0, 0.5, size=T))

    # Guard bounds
    low = np.maximum(low, close * 0.001)
    high = np.maximum(high, M + 1e-6)

    return np.column_stack([open_p, high, low, close, volume])


def generate_trend_dataset(
    n_samples: int,
    seed: int = 0,
    P0: float = 1000.0,
    window_bars: int = 48,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """PC-1: Linear trend slope k in [-0.01, 0.01] per bar."""
    rng = np.random.default_rng(seed)
    t = np.arange(window_bars, dtype=np.float64)
    k_values = rng.uniform(-0.01, 0.01, size=n_samples)

    windows = np.empty((n_samples, window_bars, 5), dtype=np.float64)
    for i in range(n_samples):
        k = k_values[i]
        # relative noise std = 0.02% = 0.0002
        noise = rng.normal(0.0, 0.0002 * P0, size=window_bars)
        close = P0 * (1.0 + k * t) + noise
        windows[i] = make_candles_from_close(close, rng)

    labels = {
        "k": k_values.astype(np.float64),
        "sign": (k_values >= 0.0).astype(np.int8),
    }
    return windows, labels


def generate_vol_dataset(
    n_samples: int,
    seed: int = 0,
    P0: float = 1000.0,
    window_bars: int = 48,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """PC-2: Volatility levels in 3 tiers: 0.05%, 0.15%, 0.40% per bar."""
    rng = np.random.default_rng(seed)
    n_levels = len(VOL_LEVELS)
    vol_classes = rng.integers(0, n_levels, size=n_samples)
    vol_values = np.array([VOL_LEVELS[c] for c in vol_classes], dtype=np.float64)

    windows = np.empty((n_samples, window_bars, 5), dtype=np.float64)
    for i in range(n_samples):
        sigma = vol_values[i]
        log_ret = rng.normal(-0.5 * sigma**2, sigma, size=window_bars)
        close = P0 * np.exp(np.cumsum(log_ret))
        windows[i] = make_candles_from_close(close, rng, intra_vol=sigma)

    labels = {
        "vol": vol_values,
        "vol_class": vol_classes.astype(np.int8),
    }
    return windows, labels


def generate_ar1_dataset(
    n_samples: int,
    seed: int = 0,
    P0: float = 1000.0,
    phi: float = 0.8,
    sigma_eps: float = 0.002,
    window_bars: int = 48,
    horizon_bars: int = 6,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """PC-3: Log-returns follow AR(1) with phi=0.8. Predicts future 6-bar return direction."""
    rng = np.random.default_rng(seed)
    total_bars = window_bars + horizon_bars

    windows = np.empty((n_samples, window_bars, 5), dtype=np.float64)
    future_returns = np.empty(n_samples, dtype=np.float64)

    for i in range(n_samples):
        r = np.empty(total_bars, dtype=np.float64)
        r[0] = rng.normal(0.0, sigma_eps / np.sqrt(1.0 - phi**2))
        eps = rng.normal(0.0, sigma_eps, size=total_bars - 1)
        for t in range(1, total_bars):
            r[t] = phi * r[t - 1] + eps[t - 1]

        close = P0 * np.exp(np.cumsum(r))
        candles = make_candles_from_close(close, rng, intra_vol=sigma_eps)
        windows[i] = candles[:window_bars]
        # Return from bar (window_bars - 1) to (total_bars - 1)
        ret_future = (close[total_bars - 1] / close[window_bars - 1]) - 1.0
        future_returns[i] = ret_future

    labels = {
        "future_return": future_returns,
        "direction": (future_returns > 0.0).astype(np.int8),
    }
    return windows, labels


def generate_gbm_dataset(
    n_samples: int,
    seed: int = 0,
    P0: float = 1000.0,
    sigma: float = 0.002,
    window_bars: int = 48,
    horizon_bars: int = 6,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """NC-1: Zero-drift pure random walk (GBM). Predicts future 6-bar return direction."""
    rng = np.random.default_rng(seed)
    total_bars = window_bars + horizon_bars

    windows = np.empty((n_samples, window_bars, 5), dtype=np.float64)
    future_returns = np.empty(n_samples, dtype=np.float64)

    for i in range(n_samples):
        log_ret = rng.normal(-0.5 * sigma**2, sigma, size=total_bars)
        close = P0 * np.exp(np.cumsum(log_ret))
        candles = make_candles_from_close(close, rng, intra_vol=sigma)
        windows[i] = candles[:window_bars]
        ret_future = (close[total_bars - 1] / close[window_bars - 1]) - 1.0
        future_returns[i] = ret_future

    labels = {
        "future_return": future_returns,
        "direction": (future_returns > 0.0).astype(np.int8),
    }
    return windows, labels


def generate_phase0_synthetic_suite(
    split: str = "train",
    n_samples: int = 1000,
    seed_offset: int = 0,
    nc1_samples: int | None = None,
) -> dict[str, tuple[np.ndarray, dict[str, np.ndarray]]]:
    """Generate the full synthetic benchmark suite (PC-1, PC-2, PC-3, NC-1) with distinct seeds."""
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got '{split}'")

    base_seed = seed_offset + (1000 if split == "train" else 2000)
    n_nc1 = nc1_samples if nc1_samples is not None else n_samples

    return {
        "pc1_trend": generate_trend_dataset(n_samples, seed=base_seed + 1),
        "pc2_vol": generate_vol_dataset(n_samples, seed=base_seed + 2),
        "pc3_ar1": generate_ar1_dataset(n_samples, seed=base_seed + 3),
        "nc1_gbm": generate_gbm_dataset(n_nc1, seed=base_seed + 4),
    }
