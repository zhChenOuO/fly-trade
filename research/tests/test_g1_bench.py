"""Unit tests for G1 Synthetic Benchmark Pipeline: NARMA10, Spectral Ridge Readout, and Gates."""
from __future__ import annotations

import numpy as np
import pytest

from research.pipeline.g1_bench import (
    TRAIN_SEEDS,
    VAL_SEEDS,
    TEST_SEEDS,
    NEGATIVE_CONTROL_SEED,
    G1_SEEDS_SHA256,
    ALPHA_GRID,
    generate_narma10_reference,
    generate_narma10_vectorized,
    generate_narma10_sequence,
    generate_g1_splits,
    compute_nmse,
    G1RidgeReadout,
    evaluate_oracle_positive_control,
    evaluate_negative_control,
)


def test_narma10_two_implementations_bitwise_identical() -> None:
    """Reference for-loop and sliding buffer NARMA10 must produce bitwise identical outputs."""
    for seed in [42, 101, 777]:
        rng = np.random.default_rng(seed)
        u = rng.uniform(0.0, 0.5, size=5500)

        u_ref, y_ref, meta_ref = generate_narma10_reference(u)
        u_vec, y_vec, meta_vec = generate_narma10_vectorized(u)

        assert np.array_equal(u_ref, u_vec)
        assert np.array_equal(y_ref, y_vec)
        assert meta_ref["first_valid_lag_step"] == 9
        assert meta_vec["first_valid_target_index"] == 10


def test_split_shapes_and_frozen_seed_hash() -> None:
    """G1 split configurations and seed constants must strictly match SPEC §2."""
    assert len(TRAIN_SEEDS) == 10
    assert len(VAL_SEEDS) == 3
    assert len(TEST_SEEDS) == 10
    assert len(set(TRAIN_SEEDS) | set(VAL_SEEDS) | set(TEST_SEEDS) | {NEGATIVE_CONTROL_SEED}) == 24

    # Verify SHA256 integrity hash is a valid 64-character hex string
    assert len(G1_SEEDS_SHA256) == 64

    splits = generate_g1_splits()
    assert len(splits["train"]) == 10
    assert len(splits["val"]) == 3
    assert len(splits["test"]) == 10

    # Sequence lengths: Train 5,500 (500 washout + 5,000 valid)
    for s in splits["train"]:
        assert s["total_length"] == 5500
        assert s["washout"] == 500
        assert s["valid_length"] == 5000

    # Val 2,500 (500 washout + 2,000 valid)
    for s in splits["val"]:
        assert s["total_length"] == 2500
        assert s["washout"] == 500
        assert s["valid_length"] == 2000

    # Test 5,500 (500 washout + 5,000 valid)
    for s in splits["test"]:
        assert s["total_length"] == 5500
        assert s["washout"] == 500
        assert s["valid_length"] == 5000


def test_nmse_metric_properties() -> None:
    """compute_nmse must equal 0 for perfect predictions and 1 for constant mean predictions."""
    rng = np.random.default_rng(123)
    y = rng.standard_normal(1000)

    # Perfect prediction: NMSE = 0.0
    assert compute_nmse(y, y) == 0.0

    # Constant mean prediction: NMSE = 1.0
    y_mean = np.full_like(y, np.mean(y))
    np.testing.assert_allclose(compute_nmse(y_mean, y), 1.0, atol=1e-12)


def test_ridge_spectral_matches_lstsq_at_alpha_zero() -> None:
    """G1RidgeReadout at alpha=0 must match np.linalg.lstsq within numerical tolerance."""
    rng = np.random.default_rng(42)
    n_samples = 300
    n_features = 20

    X = rng.standard_normal((n_samples, n_features))
    # Make some features correlated and one constant
    X[:, 5] = X[:, 2] + 0.5 * X[:, 3]
    X[:, 8] = 42.0  # Constant feature, must be dropped

    beta_true = rng.standard_normal(n_features)
    beta_true[8] = 0.0
    y = X @ beta_true + 3.14 + 0.01 * rng.standard_normal(n_samples)

    # Wrap in 10 synthetic sequences of 30 samples each
    seq_features = [X[i * 30 : (i + 1) * 30] for i in range(10)]
    seq_targets = [y[i * 30 : (i + 1) * 30] for i in range(10)]

    # Readout with only alpha=0
    model = G1RidgeReadout(alphas=(0.0,))
    model.fit_sequence_group_cv(seq_features, seq_targets)

    # Compare with reference lstsq on the active standardized features
    active = model.active_features
    assert active[8] == False  # Constant feature 8 dropped
    assert int(np.sum(active)) == 19

    X_act = X[:, active]
    mu_x = np.mean(X_act, axis=0)
    std_x = np.std(X_act, axis=0)
    X_std = (X_act - mu_x) / std_x

    X_ones = np.column_stack([np.ones(n_samples), X_std])
    beta_lstsq = np.linalg.lstsq(X_ones, y, rcond=None)[0]

    np.testing.assert_allclose(model.intercept, beta_lstsq[0], atol=1e-8)
    np.testing.assert_allclose(model.weights, beta_lstsq[1:], atol=1e-8)


def test_ridge_recovers_known_linear_relationship() -> None:
    """G1RidgeReadout must recover linear dependencies with high fidelity."""
    rng = np.random.default_rng(99)
    n_seqs = 10
    len_seq = 100
    d = 12

    seq_features = []
    seq_targets = []
    w_true = rng.standard_normal(d)

    for _ in range(n_seqs):
        X_seq = rng.standard_normal((len_seq, d))
        y_seq = X_seq @ w_true + 0.5 + 1e-4 * rng.standard_normal(len_seq)
        seq_features.append(X_seq)
        seq_targets.append(y_seq)

    model = G1RidgeReadout()
    model.fit_sequence_group_cv(seq_features, seq_targets)

    # Predict on unseen sequence
    X_test = rng.standard_normal((200, d))
    y_test = X_test @ w_true + 0.5
    y_pred = model.predict(X_test)

    nmse_test = compute_nmse(y_pred, y_test)
    assert nmse_test < 1e-4


def test_ridge_alpha_boundary_flags() -> None:
    """G1RidgeReadout must flag when best alpha hits grid boundaries (0.0 or 100.0)."""
    rng = np.random.default_rng(101)
    n_seqs = 10
    len_seq = 50
    d = 5

    # 1. Noiseless linear case -> picks alpha = 0.0 (boundary)
    seq_f_clean = [rng.standard_normal((len_seq, d)) for _ in range(n_seqs)]
    w = np.array([1.0, 2.0, -1.0, 0.5, -0.5])
    seq_t_clean = [X @ w + 1.0 for X in seq_f_clean]

    model_clean = G1RidgeReadout()
    model_clean.fit_sequence_group_cv(seq_f_clean, seq_t_clean)
    assert model_clean.best_alpha == 0.0
    assert model_clean.cv_results["alpha_at_boundary"] is True

    # 2. High dimensional noise -> picks alpha = 100.0 (upper boundary)
    d_high = 80
    len_noise = 40
    rng_high = np.random.default_rng(77)
    seq_f_noise = [rng_high.standard_normal((len_noise, d_high)) for _ in range(n_seqs)]
    seq_t_noise = [rng_high.standard_normal(len_noise) for _ in range(n_seqs)]

    model_noise = G1RidgeReadout()
    model_noise.fit_sequence_group_cv(seq_f_noise, seq_t_noise)
    assert model_noise.best_alpha == 100.0
    assert model_noise.cv_results["alpha_at_boundary"] is True

    # 3. Moderate signal + noise -> picks intermediate alpha (not on boundary)
    rng_mid = np.random.default_rng(42)
    seq_f_mid = [rng_mid.standard_normal((100, 5)) for _ in range(n_seqs)]
    w_mid = np.array([0.5, 0.5, 0.2, -0.2, 0.1])
    seq_t_mid = [X @ w_mid + 0.3 * rng_mid.standard_normal(100) for X in seq_f_mid]

    model_mid = G1RidgeReadout()
    model_mid.fit_sequence_group_cv(seq_f_mid, seq_t_mid)
    assert 0.0 < model_mid.best_alpha < 100.0
    assert model_mid.cv_results["alpha_at_boundary"] is False


def test_oracle_positive_control_achieves_low_nmse() -> None:
    """Oracle positive control must achieve NMSE < 1e-6 across all splits."""
    splits = generate_g1_splits()
    res = evaluate_oracle_positive_control(splits)

    assert res["passed"] is True
    for split_name in ["train", "val", "test"]:
        nmse = res["splits"][split_name]["mean_nmse"]
        assert nmse < 1e-6, f"Oracle {split_name} NMSE {nmse} exceeded 1e-6"


def test_negative_control_mismatched_stream() -> None:
    """Mismatched input negative control must not improve over constant mean baseline."""
    rng = np.random.default_rng(42)
    n_samples = 500
    d = 10

    X_train = [rng.standard_normal((50, d)) for _ in range(10)]
    y_train = [rng.standard_normal(50) for _ in range(10)]

    model = G1RidgeReadout()
    model.fit_sequence_group_cv(X_train, y_train)

    # Mismatched independent test stream
    X_mismatched = rng.standard_normal((n_samples, d))
    y_true = rng.standard_normal(n_samples)

    res = evaluate_negative_control(model, X_mismatched, y_true)
    assert res["passed"] is True
    assert res["observed_nmse"] >= 0.99
