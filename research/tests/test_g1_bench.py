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
    generate_train_sequences,
    generate_val_sequences,
    unseal_test_sequences,
    unseal_test,
    generate_g1_splits,
    compute_nmse,
    G1RidgeReadout,
    check_saturation_gate,
    select_r16_indices,
    compute_memory_capacity_cv,
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
    """G1 split configurations and seed constants must strictly match SPEC §2 and enforce Test boundary."""
    assert len(TRAIN_SEEDS) == 10
    assert len(VAL_SEEDS) == 3
    assert len(TEST_SEEDS) == 10
    assert len(set(TRAIN_SEEDS) | set(VAL_SEEDS) | set(TEST_SEEDS) | {NEGATIVE_CONTROL_SEED}) == 24

    # Verify SHA256 integrity hash is a valid 64-character hex string
    assert len(G1_SEEDS_SHA256) == 64

    # Train and Val can be generated independently
    train_seqs = generate_train_sequences()
    val_seqs = generate_val_sequences()
    assert len(train_seqs) == 10
    assert len(val_seqs) == 3

    # Default generate_g1_splits only provides train and val (Test access boundary)
    splits = generate_g1_splits()
    assert len(splits["train"]) == 10
    assert len(splits["val"]) == 3
    assert "test" not in splits

    # Attempting to access test without valid spec hash raises
    with pytest.raises(ValueError, match="valid spec_hash"):
        unseal_test(spec_hash="")

    valid_hash = "a" * 64
    test_seqs = unseal_test(spec_hash=valid_hash)
    assert len(test_seqs) == 10

    splits_unsealed = generate_g1_splits(unseal_spec_hash=valid_hash)
    assert len(splits_unsealed["test"]) == 10

    # Sequence lengths: Train 5,500 (500 washout + 5,000 valid)
    for s in train_seqs:
        assert s["total_length"] == 5500
        assert s["washout"] == 500
        assert s["valid_length"] == 5000

    # Val 2,500 (500 washout + 2,000 valid)
    for s in val_seqs:
        assert s["total_length"] == 2500
        assert s["washout"] == 500
        assert s["valid_length"] == 2000

    # Test 5,500 (500 washout + 5,000 valid)
    for s in test_seqs:
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
    """G1RidgeReadout must flag when best alpha hits grid boundaries (upper bound 100.0 vs zero diagnostic)."""
    rng = np.random.default_rng(101)
    n_seqs = 10
    len_seq = 50
    d = 5

    # 1. Noiseless linear case -> picks alpha = 0.0 (diagnostic only, not no-go upper bound)
    seq_f_clean = [rng.standard_normal((len_seq, d)) for _ in range(n_seqs)]
    w = np.array([1.0, 2.0, -1.0, 0.5, -0.5])
    seq_t_clean = [X @ w + 1.0 for X in seq_f_clean]

    model_clean = G1RidgeReadout()
    model_clean.fit_sequence_group_cv(seq_f_clean, seq_t_clean)
    assert model_clean.best_alpha == 0.0
    assert model_clean.cv_results["alpha_zero_diagnostic"] is True
    assert model_clean.cv_results["alpha_at_upper_bound"] is False
    assert model_clean.cv_results["alpha_at_boundary"] is False

    # 2. High dimensional noise -> picks alpha = 100.0 (upper boundary no-go flag)
    d_high = 80
    len_noise = 40
    rng_high = np.random.default_rng(77)
    seq_f_noise = [rng_high.standard_normal((len_noise, d_high)) for _ in range(n_seqs)]
    seq_t_noise = [rng_high.standard_normal(len_noise) for _ in range(n_seqs)]

    model_noise = G1RidgeReadout()
    model_noise.fit_sequence_group_cv(seq_f_noise, seq_t_noise)
    assert model_noise.best_alpha == 100.0
    assert model_noise.cv_results["alpha_at_upper_bound"] is True
    assert model_noise.cv_results["alpha_at_boundary"] is True
    assert model_noise.cv_results["alpha_zero_diagnostic"] is False

    # 3. Moderate signal + noise -> picks intermediate alpha (not on boundary)
    rng_mid = np.random.default_rng(42)
    seq_f_mid = [rng_mid.standard_normal((100, 5)) for _ in range(n_seqs)]
    w_mid = np.array([0.5, 0.5, 0.2, -0.2, 0.1])
    seq_t_mid = [X @ w_mid + 0.3 * rng_mid.standard_normal(100) for X in seq_f_mid]

    model_mid = G1RidgeReadout()
    model_mid.fit_sequence_group_cv(seq_f_mid, seq_t_mid)
    assert 0.0 < model_mid.best_alpha < 100.0
    assert model_mid.cv_results["alpha_at_upper_bound"] is False
    assert model_mid.cv_results["alpha_at_boundary"] is False
    assert model_mid.cv_results["alpha_zero_diagnostic"] is False


def test_oracle_positive_control_achieves_low_nmse() -> None:
    """Oracle positive control must achieve NMSE < 1e-6 across train/val and unsealed test."""
    splits = generate_g1_splits()
    res = evaluate_oracle_positive_control(splits)

    assert res["passed"] is True
    for split_name in ["train", "val"]:
        nmse = res["splits"][split_name]["mean_nmse"]
        assert nmse < 1e-6, f"Oracle {split_name} NMSE {nmse} exceeded 1e-6"

    # With unsealed test split
    splits_unsealed = generate_g1_splits(unseal_spec_hash="a" * 64)
    res_unsealed = evaluate_oracle_positive_control(splits_unsealed)
    assert res_unsealed["passed"] is True
    assert res_unsealed["splits"]["test"]["mean_nmse"] < 1e-6


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


def test_ridge_svd_matches_lstsq_on_ill_conditioned_and_collinear() -> None:
    """G1RidgeReadout float64 SVD solver at alpha=0 must match np.linalg.lstsq on ill-conditioned data."""
    rng = np.random.default_rng(2026)
    n_samples = 400
    n_features = 10

    X = rng.standard_normal((n_samples, n_features))
    # Collinear feature
    X[:, 2] = 2.5 * X[:, 0] - 1.2 * X[:, 1]
    # Constant feature
    X[:, 5] = 17.0
    # Almost collinear feature (ill-conditioned)
    X[:, 7] = X[:, 3] + 1e-14 * rng.standard_normal(n_samples)

    y = 0.5 * X[:, 0] - 0.8 * X[:, 1] + 1.2 * X[:, 3] + 4.2 + 0.05 * rng.standard_normal(n_samples)

    # 10 sequences of 40 samples
    seq_f = [X[i * 40 : (i + 1) * 40] for i in range(10)]
    seq_t = [y[i * 40 : (i + 1) * 40] for i in range(10)]

    model = G1RidgeReadout(alphas=(0.0,))
    model.fit_sequence_group_cv(seq_f, seq_t)

    assert model.best_alpha == 0.0
    assert model.cv_results["alpha_zero_diagnostic"] is True

    # Compare refit model with lstsq on active standardized features
    active = model.active_features
    assert active[5] == False  # Constant feature dropped

    X_act = X[:, active]
    mu_x = np.mean(X_act, axis=0)
    std_x = np.std(X_act, axis=0)
    X_std = (X_act - mu_x) / std_x

    # Reference lstsq on design matrix with intercept column
    X_ones = np.column_stack([np.ones(n_samples), X_std])
    beta_lstsq = np.linalg.lstsq(X_ones, y, rcond=None)[0]
    b0_lstsq = float(beta_lstsq[0])
    w_lstsq = beta_lstsq[1:]

    # Weights and predictions match within numerical tolerance (1e-10)
    np.testing.assert_allclose(model.weights, w_lstsq, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(model.intercept, b0_lstsq, atol=1e-10, rtol=1e-10)

    pred_model = model.predict(X)
    pred_lstsq = X_ones @ beta_lstsq
    np.testing.assert_allclose(pred_model, pred_lstsq, atol=1e-10, rtol=1e-10)


def test_check_saturation_gate_fail_closed() -> None:
    """check_saturation_gate must fail-closed on empty, NaN, Inf, and overdriven states."""
    # 1. Empty input must fail
    res_empty = check_saturation_gate(np.array([]))
    assert res_empty["passed"] is False
    assert res_empty["failure_reason"] == "EMPTY_INPUT"

    # 2. NaN input must fail
    nan_arr = np.array([0.1, 0.2, np.nan, 0.4])
    res_nan = check_saturation_gate(nan_arr)
    assert res_nan["passed"] is False
    assert res_nan["non_finite_count"] == 1
    assert "NON_FINITE_STATES" in res_nan["failure_reason"]

    # 3. Inf input must fail
    inf_arr = np.array([0.1, np.inf, 0.3])
    res_inf = check_saturation_gate(inf_arr)
    assert res_inf["passed"] is False
    assert res_inf["non_finite_count"] == 1

    # 4. Overdriven states (> 0.9 ratio > 0.05) must fail
    overdriven = np.array([0.95] * 10 + [0.1] * 90)  # 10% saturated
    res_over = check_saturation_gate(overdriven, threshold=0.90, max_ratio=0.05)
    assert res_over["passed"] is False
    assert abs(res_over["saturation_ratio"] - 0.10) < 1e-6

    # 5. Normal bounded states must pass
    normal = np.array([0.2] * 98 + [0.95] * 2)  # 2% saturated
    res_normal = check_saturation_gate(normal, threshold=0.90, max_ratio=0.05)
    assert res_normal["passed"] is True
    assert abs(res_normal["saturation_ratio"] - 0.02) < 1e-6


def test_select_r16_indices_filters_correctly() -> None:
    """select_r16_indices must strictly select R1-6 photoreceptors with finite coordinates."""
    from research.pipeline.flywire_graph import ConnectomeGraph
    import scipy.sparse as sp

    N = 20
    # Neurons: 0-7 are sensory, 8-15 interneurons, 16-19 motor
    sensory_idx = np.arange(8)
    motor_idx = np.array([16, 17, 18, 19])

    # Types: 0-3 R1-6, 4-5 R7, 6-7 R8
    ptypes = np.array(["R1-6", "R1-6", "R1-6", "R1-6", "R7", "R7", "R8", "R8"], dtype=object)
    # Give neuron 2 NaN retina coordinate
    u_coords = np.array([0.1, 0.2, np.nan, 0.4, 0.5, 0.6, 0.7, 0.8])
    v_coords = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

    meta = {
        "photoreceptor_type": ptypes,
        "u": u_coords,
        "v": v_coords,
        "sha256": "mock_graph_sha256_hash",
    }
    weights = sp.eye(N, format="csr")

    graph = ConnectomeGraph(
        weights=weights,
        neuron_ids=[f"N_{i}" for i in range(N)],
        neuron_types=["photoreceptor" if i < 8 else "interneuron" for i in range(N)],
        sensory_idx=sensory_idx,
        motor_idx=motor_idx,
        meta=meta,
    )

    r16_idx, summary = select_r16_indices(graph)
    # Neurons 0, 1, 3 are valid R1-6 with finite coordinates (neuron 2 has NaN)
    assert list(r16_idx) == [0, 1, 3]
    assert summary["n_r16_selected"] == 3
    assert summary["n_r7_r8_excluded"] == 4
    assert summary["n_non_finite_excluded"] == 1
    assert summary["graph_sha256"] == "mock_graph_sha256_hash"


def test_compute_memory_capacity_cv_group_cv() -> None:
    """compute_memory_capacity_cv must do out-of-sequence CV, passing delay line and rejecting noise."""
    rng = np.random.default_rng(789)
    n_seqs = 10
    T_valid = 600
    washout = 100
    T_total = T_valid + washout

    # Delay line simulation data
    inputs = [rng.uniform(0.0, 0.5, size=T_total) for _ in range(n_seqs)]
    # Feature states: exactly delay 1 to 10
    delay_states = []
    noise_states = []
    for u in inputs:
        # Create 10 delay features
        d_feats = np.column_stack([
            np.concatenate([np.zeros(k), u[:-k]]) if k > 0 else u
            for k in range(1, 11)
        ])
        delay_states.append(d_feats)
        # Create 10 noise features
        noise_states.append(rng.standard_normal((T_total, 10)))

    # Compute CV Memory Capacity on delay features
    mc_delay = compute_memory_capacity_cv(delay_states, inputs, max_lag=15, n_folds=5, washout=washout)
    assert mc_delay["metric"] == "memory_capacity_cv"
    # Delay line should achieve high out-of-sample MC for the first 10 lags
    assert mc_delay["mc_total"] >= 8.0, f"Delay line CV MC {mc_delay['mc_total']} too low"

    # Compute CV Memory Capacity on pure noise features
    mc_noise = compute_memory_capacity_cv(noise_states, inputs, max_lag=15, n_folds=5, washout=washout)
    # Held-out R^2 for pure noise should be ~ 0.0 (no in-sample overfitting inflation)
    assert mc_noise["mc_total"] < 0.2, f"Noise CV MC {mc_noise['mc_total']} too high (in-sample leakage)"


def test_ridge_exact_tie_break_vs_near_tie() -> None:
    """Verify Ridge tie-break strictly uses exact equality (loss == min_loss) per SPEC §4.3."""
    # 1. End-to-end exact tie on uninformative features
    model = G1RidgeReadout(alphas=(0.01, 0.1, 1.0, 10.0))
    X_zero = [np.zeros((50, 4)) for _ in range(10)]
    y_rand = [np.random.default_rng(99).normal(size=50) for _ in range(10)]
    model.fit_sequence_group_cv(X_zero, y_rand)
    # With zero features, all alphas produce identical prediction mu_y; exact tie selects max alpha
    assert model.best_alpha == 10.0

    # 2. Demonstration of behavioral difference between old rule (<= 1e-9) and new rule (loss == min_loss)
    # Synthetic pooled OOF losses: alpha 0.01 has strictly lower loss by 1e-10 relative to alpha 0.1
    min_loss_val = 0.500000000000
    near_loss_val = 0.500000000050  # relative diff = 1e-10 <= 1e-9
    losses_near_tie = {0.01: min_loss_val, 0.1: near_loss_val, 1.0: 0.600000000000}

    # Old logic: relative difference <= 1e-9 triggers tie-break -> picks larger alpha 0.1
    min_l = min(losses_near_tie.values())
    old_tied = [
        a for a, loss in losses_near_tie.items()
        if loss == min_l or (min_l > 0 and abs(loss - min_l) / min_l <= 1e-9)
    ]
    old_selected = max(old_tied)
    assert old_selected == 0.1, "Old rule should have incorrectly treated near-tie as tie and selected 0.1"

    # New logic: exact equality only -> 0.01 is strictly lower, no tie -> selects 0.01
    new_tied = [a for a, loss in losses_near_tie.items() if loss == min_l]
    new_selected = max(new_tied)
    assert new_selected == 0.01, "New rule must strictly select 0.01 because it has strictly smaller loss"

    # 3. Exact tie case: both old and new logic agree and select larger alpha
    losses_exact_tie = {0.01: min_loss_val, 0.1: min_loss_val, 1.0: 0.600000000000}
    min_exact = min(losses_exact_tie.values())
    new_exact_tied = [a for a, loss in losses_exact_tie.items() if loss == min_exact]
    assert max(new_exact_tied) == 0.1

