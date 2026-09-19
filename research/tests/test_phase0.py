"""Tests for Phase 0 gates and synthetic price generators.

Verifies:
1. Synthetic prices generators (trend, vol, ar1, gbm) satisfy SPEC_v3 specs,
   pass render(), separate train/test seeds, and maintain proper bounds.
2. Pure NumPy Ridge with 5-fold CV fits and predicts accurately.
3. When structured information is planted into features, S4 passes all positive gates.
4. Negative control NC-1 on pure random walk (GBM) produces direction BA in [0.49, 0.51].
5. Dynamics calibration (spectral radius, g_sat bisection) functions correctly.
"""
from __future__ import annotations

import numpy as np
import pytest

from research.pipeline.phase0 import (
    PureNumpyRidge,
    balanced_accuracy_score,
    bisection_find_gsat,
    compute_spectral_radius,
    r2_score,
    ridge_cv_fit,
)
from research.pipeline.render_market import render
from research.pipeline.synthetic_prices import (
    VOL_LEVELS,
    generate_ar1_dataset,
    generate_gbm_dataset,
    generate_phase0_synthetic_suite,
    generate_trend_dataset,
    generate_vol_dataset,
    make_candles_from_close,
)


def test_make_candles_validity():
    rng = np.random.default_rng(42)
    close = np.linspace(100.0, 150.0, 48)
    candles = make_candles_from_close(close, rng)

    assert candles.shape == (48, 5)
    assert np.all(candles[:, :4] > 0)
    assert np.all(candles[:, 4] >= 0)
    assert np.all(candles[:, 2] <= np.minimum(candles[:, 0], candles[:, 3]))
    assert np.all(candles[:, 1] >= np.maximum(candles[:, 0], candles[:, 3]))

    # Must pass render() without raising ValueError
    img = render(candles)
    assert img.shape == (64, 64, 3)
    assert img.dtype == np.uint8


def test_synthetic_prices_generators():
    # 1. Trend
    w_trend, l_trend = generate_trend_dataset(10, seed=10)
    assert w_trend.shape == (10, 48, 5)
    assert "k" in l_trend and "sign" in l_trend
    assert np.all(l_trend["k"] >= -0.01) and np.all(l_trend["k"] <= 0.01)
    assert set(np.unique(l_trend["sign"])).issubset({0, 1})
    for w in w_trend:
        assert render(w).shape == (64, 64, 3)

    # 2. Vol
    w_vol, l_vol = generate_vol_dataset(12, seed=20)
    assert w_vol.shape == (12, 48, 5)
    assert "vol" in l_vol and "vol_class" in l_vol
    assert set(np.unique(l_vol["vol"])).issubset(set(VOL_LEVELS))
    assert set(np.unique(l_vol["vol_class"])).issubset({0, 1, 2})
    for w in w_vol:
        assert render(w).shape == (64, 64, 3)

    # 3. AR(1)
    w_ar1, l_ar1 = generate_ar1_dataset(10, seed=30)
    assert w_ar1.shape == (10, 48, 5)
    assert "future_return" in l_ar1 and "direction" in l_ar1
    assert set(np.unique(l_ar1["direction"])).issubset({0, 1})
    for w in w_ar1:
        assert render(w).shape == (64, 64, 3)

    # 4. GBM
    w_gbm, l_gbm = generate_gbm_dataset(10, seed=40)
    assert w_gbm.shape == (10, 48, 5)
    assert "future_return" in l_gbm and "direction" in l_gbm
    assert set(np.unique(l_gbm["direction"])).issubset({0, 1})
    for w in w_gbm:
        assert render(w).shape == (64, 64, 3)


def test_synthetic_suite_seed_separation():
    suite_tr1 = generate_phase0_synthetic_suite("train", n_samples=5, seed_offset=0)
    suite_tr2 = generate_phase0_synthetic_suite("train", n_samples=5, seed_offset=0)
    suite_te = generate_phase0_synthetic_suite("test", n_samples=5, seed_offset=0)

    # Determinism: same seed produces identical results
    np.testing.assert_array_equal(suite_tr1["pc1_trend"][0], suite_tr2["pc1_trend"][0])
    np.testing.assert_array_equal(suite_tr1["pc1_trend"][1]["k"], suite_tr2["pc1_trend"][1]["k"])

    # Separation: train and test produce different data
    assert not np.array_equal(suite_tr1["pc1_trend"][0], suite_te["pc1_trend"][0])
    assert not np.array_equal(suite_tr1["pc1_trend"][1]["k"], suite_te["pc1_trend"][1]["k"])


def test_pure_numpy_ridge_and_cv():
    rng = np.random.default_rng(42)
    N, D = 150, 20
    X = rng.normal(0, 1, size=(N, D))
    true_w = rng.normal(0, 1, size=D)
    y = X @ true_w + rng.normal(0, 0.1, size=N)

    # Test CV fit
    model, best_alpha = ridge_cv_fit(X, y, alphas=(1e-3, 1e-2, 1e-1, 1.0, 10.0))
    assert best_alpha > 0
    preds = model.predict(X)
    r2 = r2_score(y, preds)
    assert r2 > 0.95


def test_planted_information_s4_passes():
    """Verify that when information is planted into features, S4 criteria pass."""
    rng = np.random.default_rng(101)
    N_tr, N_te, D = 300, 300, 50

    # Planted PC-1 (Trend slope k)
    k_tr = rng.uniform(-0.01, 0.01, size=N_tr)
    k_te = rng.uniform(-0.01, 0.01, size=N_te)
    X_tr_pc1 = rng.normal(0, 1, (N_tr, D)) + k_tr[:, None] * 100.0
    X_te_pc1 = rng.normal(0, 1, (N_te, D)) + k_te[:, None] * 100.0

    model_pc1, _ = ridge_cv_fit(X_tr_pc1, k_tr)
    pred_k = model_pc1.predict(X_te_pc1)
    r2_pc1 = r2_score(k_te, pred_k)
    ba_pc1 = balanced_accuracy_score((k_te >= 0).astype(int), (pred_k >= 0).astype(int))

    assert r2_pc1 >= 0.5, f"Planted PC-1 R2 failed: {r2_pc1}"
    assert ba_pc1 >= 0.95, f"Planted PC-1 BA failed: {ba_pc1}"

    # Planted PC-2 (Vol level)
    vol_tr = rng.choice(VOL_LEVELS, size=N_tr)
    vol_te = rng.choice(VOL_LEVELS, size=N_te)
    X_tr_pc2 = rng.normal(0, 1, (N_tr, D)) + vol_tr[:, None] * 500.0
    X_te_pc2 = rng.normal(0, 1, (N_te, D)) + vol_te[:, None] * 500.0

    model_pc2, _ = ridge_cv_fit(X_tr_pc2, vol_tr)
    pred_vol = model_pc2.predict(X_te_pc2)
    r2_pc2 = r2_score(vol_te, pred_vol)
    assert r2_pc2 >= 0.5, f"Planted PC-2 R2 failed: {r2_pc2}"

    # Planted PC-3 (AR1 direction)
    ret_tr = rng.normal(0, 0.01, size=N_tr)
    ret_te = rng.normal(0, 0.01, size=N_te)
    X_tr_pc3 = rng.normal(0, 1, (N_tr, D)) + ret_tr[:, None] * 80.0
    X_te_pc3 = rng.normal(0, 1, (N_te, D)) + ret_te[:, None] * 80.0

    model_pc3, _ = ridge_cv_fit(X_tr_pc3, ret_tr)
    pred_ret = model_pc3.predict(X_te_pc3)
    ba_pc3 = balanced_accuracy_score((ret_te > 0).astype(int), (pred_ret > 0).astype(int))
    assert ba_pc3 >= 0.60, f"Planted PC-3 BA failed: {ba_pc3}"


def test_nc1_negative_control_gbm_ba_in_interval():
    """Verify that pure uninformative noise on independent labels lands BA in 95% binomial interval (SPEC_v3 Amendment 1)."""
    rng = np.random.default_rng(202)
    N_tr = 500
    D = 12

    for N_te in [2000, 10000]:
        X_tr = rng.normal(0, 1, (N_tr, D))
        y_tr = rng.normal(0, 0.01, N_tr)

        X_te = rng.normal(0, 1, (N_te, D))
        y_te = rng.normal(0, 0.01, N_te)
        dir_true = (y_te > 0).astype(int)

        model, _ = ridge_cv_fit(X_tr, y_tr)
        pred_y = model.predict(X_te)
        dir_pred = (pred_y > 0).astype(int)

        ba = balanced_accuracy_score(dir_true, dir_pred)
        half_width = 1.96 * 0.5 / np.sqrt(N_te)
        low, high = 0.5 - half_width, 0.5 + half_width
        assert low <= ba <= high, f"NC-1 BA {ba} out of Amendment 1 interval [{low:.4f}, {high:.4f}] for N={N_te}"


def test_bisection_gsat_and_spectral_radius():
    import scipy.sparse as sp

    # Create dummy random sparse matrix
    rng = np.random.default_rng(303)
    W = sp.random(50, 50, density=0.1, random_state=303, data_rvs=rng.standard_normal).tocsr()
    rho = compute_spectral_radius(W)
    assert rho >= 0.0

    sensory_idx = np.arange(5)
    currents = rng.normal(0, 0.1, size=(20, 32, 5))
    gsat = bisection_find_gsat(W, sensory_idx, currents, steps=8)
    assert 0.001 <= gsat <= 10.0
