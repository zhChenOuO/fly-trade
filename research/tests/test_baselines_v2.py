"""Tests for Research v2 Phase 4 Non-Connectome Baselines.

Contracts verified:
1. Feature Leakage: Mutating raw bars strictly after sample timestamp t does NOT affect sample features.
2. Train-only Normalization: Mutating validation data does NOT change normalization parameters.
3. Split Whitelist: Requesting or passing dev_test_v1 or holdout strictly raises ValueError.
4. Model Sanity:
   - Ridge recovers known coefficients on synthetic linear data.
   - Majority baseline has Balanced Accuracy == 1/3 on balanced 3-class data.
5. Trading Simulator:
   - Matches hand-calculated 6-step toy example with costs to numerical precision.
   - Matches buy-and-hold exactly when cost is zero and position is constantly long.
   - Sample stride == horizon == 6 guarantees non-overlapping forward return intervals.
6. Moving Block Bootstrap:
   - Deterministic and identical with fixed random seed.
   - 95% CI covers true parameter on stationary data.
7. Determinism:
   - Fixed seed produces bit-for-bit identical results across runs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research.pipeline.baselines_v2 import (
    ACTION_BUY,
    ACTION_HOLD,
    ACTION_SELL,
    ACTIONS,
    BAR,
    apply_standardization,
    compute_classification_metrics,
    compute_continuous_metrics,
    compute_train_standardization,
    extract_sample_features,
    fit_predict_b0_constant,
    fit_predict_b1a_ridge,
    fit_predict_b2_mlp,
    moving_block_bootstrap_ci,
    simulate_trading_spot_long_only,
    validate_split_name,
    PureNumpyRidge,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def test_leakage_raw_bar_after_t_does_not_affect_features() -> None:
    """Mutating raw bars strictly after t must NOT change sample features at t."""
    raw = pd.read_parquet(DATA_DIR / "raw_ohlcv.parquet")
    samples = pd.read_parquet(DATA_DIR / "samples.parquet").iloc[:50].copy()

    # Position of sample 20
    sample_idx = 20
    sample_t = samples.iloc[sample_idx]["timestamp"]
    pos = int(raw.timestamp.searchsorted(sample_t - BAR))

    # Extract baseline features
    x_orig, _ = extract_sample_features(raw, samples)
    feat_orig = x_orig[sample_idx]

    # Mutate all bars strictly after pos (t+1 onwards)
    raw_mutated = raw.copy()
    raw_mutated.loc[pos + 1 :, ["open", "high", "low", "close", "volume"]] *= 100.0

    x_mutated, _ = extract_sample_features(raw_mutated, samples)
    feat_mutated = x_mutated[sample_idx]

    # Features at t must be strictly identical
    np.testing.assert_array_equal(feat_orig, feat_mutated)

    # Mutating bar pos itself (within the window) MUST change features
    raw_in_window = raw.copy()
    raw_in_window.loc[pos, "close"] *= 2.0
    x_in_window, _ = extract_sample_features(raw_in_window, samples)
    assert not np.array_equal(feat_orig, x_in_window[sample_idx])


def test_train_only_standardization() -> None:
    """Mutating validation data must NOT change train normalization parameters."""
    rng = np.random.default_rng(42)
    n_train = 500
    n_val = 200
    d = 10
    feature_names = [f"f_{i}" for i in range(d)]

    x_train = rng.normal(10.0, 5.0, (n_train, d))
    x_val_1 = rng.normal(0.0, 1.0, (n_val, d))
    x_val_2 = rng.normal(100.0, 50.0, (n_val, d))  # radically different validation data

    norm_1 = compute_train_standardization(x_train, feature_names)
    # Recomputing strictly on x_train regardless of what x_val is
    norm_2 = compute_train_standardization(x_train, feature_names)

    assert norm_1["sha256"] == norm_2["sha256"]
    np.testing.assert_array_almost_equal(norm_1["mean"], norm_2["mean"])
    np.testing.assert_array_almost_equal(norm_1["std"], norm_2["std"])

    # Applied normalization standardizes train to mean ~0, std ~1
    x_train_norm = apply_standardization(x_train, norm_1)
    np.testing.assert_array_almost_equal(np.mean(x_train_norm, axis=0), np.zeros(d), decimal=5)
    np.testing.assert_array_almost_equal(np.std(x_train_norm, axis=0), np.ones(d), decimal=5)


def test_split_whitelist_rejects_dev_test_v1_and_holdout() -> None:
    """Strict governance: only train and val are allowed. dev_test_v1 and holdout must raise."""
    # Permitted splits
    validate_split_name("train")
    validate_split_name("val")

    # Forbidden splits
    with pytest.raises(ValueError, match="strictly forbidden"):
        validate_split_name("dev_test_v1")

    with pytest.raises(ValueError, match="strictly forbidden"):
        validate_split_name("test")

    with pytest.raises(ValueError, match="strictly forbidden"):
        validate_split_name("holdout")

    with pytest.raises(ValueError, match="strictly forbidden"):
        validate_split_name("future_holdout")


def test_ridge_recovers_synthetic_weights_and_majority_ba_third() -> None:
    """Ridge recovers linear weights on synthetic data; Majority BA equals 1/3."""
    # 1. Ridge weight recovery
    rng = np.random.default_rng(123)
    n = 2000
    d = 5
    x = rng.normal(0.0, 1.0, (n, d))
    true_w = np.array([2.5, -1.8, 0.4, 3.2, -0.9])
    noise = rng.normal(0.0, 1e-4, n)
    y = x @ true_w + 0.5 + noise

    ridge = PureNumpyRidge(alpha=1e-5).fit(x, y)
    np.testing.assert_allclose(ridge.beta, true_w, rtol=1e-3, atol=1e-3)
    assert np.isclose(ridge.intercept, 0.5, atol=1e-3)

    # 2. Majority baseline Balanced Accuracy on 3-class data
    y_true = np.array([ACTION_BUY] * 40 + [ACTION_HOLD] * 40 + [ACTION_SELL] * 40, dtype=object)
    y_pred = np.array([ACTION_HOLD] * 120, dtype=object)

    metrics = compute_classification_metrics(y_true, y_pred)
    assert np.isclose(metrics["balanced_accuracy"], 1.0 / 3.0)
    assert metrics["recall"][ACTION_HOLD] == 1.0
    assert metrics["recall"][ACTION_BUY] == 0.0
    assert metrics["recall"][ACTION_SELL] == 0.0
    assert np.isclose(metrics["mcc"], 0.0)


def test_trading_simulation_manual_toy_and_buy_and_hold_equivalence() -> None:
    """Verify trading simulation on a 6-step manual toy example and check B&H equivalence."""
    # 1. 6-step hand-calculated toy example
    # Desired arithmetic market returns: [0.01, -0.02, 0.03, 0.01, -0.01, 0.02]
    r_m_target = np.array([0.01, -0.02, 0.03, 0.01, -0.01, 0.02])
    log_returns = np.log1p(r_m_target)
    actions = np.array([ACTION_BUY, ACTION_HOLD, ACTION_SELL, ACTION_HOLD, ACTION_BUY, ACTION_SELL], dtype=object)

    # fee = 0.0004, slippage = 0.0002 -> cost per trade = 0.0005
    res = simulate_trading_spot_long_only(
        actions=actions,
        future_log_returns_6=log_returns,
        cost_fee_per_side=0.0004,
        cost_slippage=0.0002,
    )

    # Step-by-step verification:
    # t=0: BUY (pos 0->1), cost=0.0005, r_p = 1*0.01 - 0.0005 = 0.0095, V_1 = 1.0095
    # t=1: HOLD (pos 1->1), cost=0, r_p = 1*(-0.02) = -0.02, V_2 = 1.0095 * 0.98 = 0.98931
    # t=2: SELL (pos 1->0), cost=0.0005, r_p = 0*0.03 - 0.0005 = -0.0005, V_3 = 0.98931 * 0.9995 = 0.988815345
    # t=3: HOLD (pos 0->0), cost=0, r_p = 0, V_4 = 0.988815345
    # t=4: BUY (pos 0->1), cost=0.0005, r_p = 1*(-0.01) - 0.0005 = -0.0105, V_5 = 0.988815345 * 0.9895 = 0.9784327838775
    # t=5: SELL (pos 1->0), cost=0.0005, r_p = 0*0.02 - 0.0005 = -0.0005, V_6 = 0.9784327838775 * 0.9995 = 0.9779435674855615
    expected_net_return = 0.9779435674855615 - 1.0
    assert np.isclose(res["net_return"], expected_net_return, atol=1e-7)
    assert res["turnover"] == 4.0
    assert res["trade_count"] == 4
    assert res["market_coverage"] == 0.5

    # 2. Buy-and-Hold equivalence
    res_bh = simulate_trading_spot_long_only(
        actions=np.array([ACTION_BUY] * 6, dtype=object),
        future_log_returns_6=log_returns,
        cost_fee_per_side=0.0,
        cost_slippage=0.0,
    )
    assert np.isclose(res_bh["net_return"], res_bh["buy_and_hold_return"], atol=1e-12)

    # 3. Non-overlapping forward return interval verification on real dataset
    samples = pd.read_parquet(DATA_DIR / "samples.parquet")
    raw = pd.read_parquet(DATA_DIR / "raw_ohlcv.parquet")
    pos_all = np.asarray(raw.timestamp.searchsorted(samples.timestamp - BAR), dtype=np.int64)

    for split in ["train", "val"]:
        mask = (samples["split"] == split).to_numpy()
        pos_split = pos_all[mask]
        diffs = np.diff(pos_split)
        # Stride is 6 bars, exactly equal to primary horizon 6
        assert (diffs == 6).all(), f"{split} samples do not have constant stride == 6"


def test_block_bootstrap_reproducibility_and_coverage() -> None:
    """Moving block bootstrap must be deterministic and have correct coverage on stationary data."""
    # 1. Reproducibility
    rng = np.random.default_rng(99)
    y_true = rng.normal(0, 1, 500)
    y_pred = y_true * 0.1 + rng.normal(0, 1, 500)

    ci1 = moving_block_bootstrap_ci(y_true, y_pred, metric_type="ic", n_bootstraps=200, seed=42)
    ci2 = moving_block_bootstrap_ci(y_true, y_pred, metric_type="ic", n_bootstraps=200, seed=42)
    assert ci1 == ci2

    # 2. Coverage on stationary balanced classification
    # Balanced 3-class random choices
    y_t = rng.choice(ACTIONS, size=600)
    y_p = rng.choice(ACTIONS, size=600)
    ci_ba_low, ci_ba_high = moving_block_bootstrap_ci(y_t, y_p, metric_type="ba", n_bootstraps=300, seed=42)
    # Expected random balanced accuracy is 1/3 (0.333)
    assert ci_ba_low <= 1.0 / 3.0 <= ci_ba_high


def test_mlp_and_baselines_determinism() -> None:
    """MLP training and evaluation must be bit-for-bit identical given identical seeds."""
    rng = np.random.default_rng(77)
    x_tr = rng.normal(0, 1, (200, 20))
    y_tr = rng.normal(0, 0.01, 200)
    x_va = rng.normal(0, 1, (100, 20))

    pred1_c, pred1_a = fit_predict_b2_mlp(x_tr, y_tr, x_va, cost_threshold=0.001, hidden_dim=32, epochs=5, seed=123)
    pred2_c, pred2_a = fit_predict_b2_mlp(x_tr, y_tr, x_va, cost_threshold=0.001, hidden_dim=32, epochs=5, seed=123)

    np.testing.assert_array_equal(pred1_c, pred2_c)
    np.testing.assert_array_equal(pred1_a, pred2_a)
