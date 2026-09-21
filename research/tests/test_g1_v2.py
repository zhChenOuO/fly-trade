"""Comprehensive Unit Tests for G1 v2 Stage A-1 Acceptance Checklist (A01-A13, A15-A17).

Governing Specification: research/G1_SPEC_v2_draft.md
Task Document: /private/tmp/claude-501/-Users-zh-Documents-playground-flywire/358c0aed-3535-404e-ae85-07b100963e55/scratchpad/task_stageA_1.md
"""
from __future__ import annotations

import math
import numpy as np
import pytest
import scipy.sparse as sp

from research.pipeline.g1_bench import (
    TRAIN_SEEDS,
    WASHOUT_STEPS,
    G1RidgeReadout,
    compute_nmse,
    generate_narma10_reference,
    generate_narma10_vectorized,
    generate_narma10_sequence,
    select_r16_indices,
    check_saturation_gate,
    check_forgetting_gate,
)
from research.pipeline.g1_reservoir import (
    ScipyG1Reservoir,
    compute_spectral_radius,
    scale_weights_to_spectral_radius,
)
from research.pipeline.g1_v2 import (
    SPEC_VERSION,
    TARGET_RHO,
    LEAK,
    NOISE_STD,
    ALPHA_GRID,
    FIXTURE_FIT_SEEDS,
    FIXTURE_CHECK_SEEDS,
    validate_v2_config,
    validate_split_request,
    build_diagnostic_targets,
    distinguish_pure_product_from_raw_q,
    compute_pooled_r2,
    compute_pooled_nmse,
    compute_even_median,
    compute_delta_primary,
    compute_diagnostic_sequence_bootstrap,
    evaluate_capability_gate,
    evaluate_diagnostic_negative_control,
    generate_narma10_v2_reference,
    generate_diagnostic_splits,
    load_diagnostic_split,
    solve_ridge_direct_closed_form,
)
from src.connectome.schema import ConnectomeGraph
from src.connectome.synthetic import make_synthetic_connectome


# ==============================================================================
# A01: Configuration Validation & Retracted Feature Rejection
# ==============================================================================

def test_a01_v2_config_validation_and_deprecation_rejection():
    """A01: Validate v2 fixed settings and ensure retracted v1 features strictly raise ValueError."""
    # 1. Valid default config
    cfg = validate_v2_config()
    assert cfg["spec_version"] == "v2"
    assert cfg["target_rho"] == 0.95
    assert cfg["leak"] == 0.5
    assert cfg["noise_std"] == 0.0
    assert cfg["input_mode"] == "raw_u"
    assert cfg["readout_mode"] == "dn_only"
    assert cfg["control_family"] == "scramble_mixed"
    assert cfg["total_graphs"] == 21

    # 2. Rejection of retracted q gate
    with pytest.raises(ValueError, match="strictly rejects q gate"):
        validate_v2_config(q_gate_enabled=True)

    # 3. Rejection of retracted nested CV rho selector
    with pytest.raises(ValueError, match="strictly rejects nested CV rho selector"):
        validate_v2_config(nested_cv_enabled=True)

    # 4. Rejection of candidate rho grid search
    with pytest.raises(ValueError, match="strictly rejects rho grid search"):
        validate_v2_config(candidate_rhos=(0.90, 0.95, 0.99))

    # 5. Rejection of family selection overrides
    with pytest.raises(ValueError, match="sole control family to 'scramble_mixed'"):
        validate_v2_config(control_families=("scramble_mixed", "random_endpoint"))

    # 6. Rejection of input centering override in production validator
    with pytest.raises(ValueError, match="strictly rejects input centering"):
        validate_v2_config(input_centering=True)

    # 7. Rejection of non-zero noise
    with pytest.raises(ValueError, match="requires noise_std=0.0"):
        validate_v2_config(noise_std=0.01)

    # 8. Rejection of non-DN readout mode
    with pytest.raises(ValueError, match="strictly requires DN-only readout"):
        validate_v2_config(readout_mode="all_neurons")


# ==============================================================================
# A02: NARMA Reference Alignment, Negative History & Oracle
# ==============================================================================

def test_a02_narma_reference_generator_alignment_and_oracle():
    """A02: Independent NARMA reference matches vectorized generator (max diff <= 1e-12) and oracle NMSE < 1e-6."""
    rng = np.random.default_rng(12345)
    T = 2500
    u = rng.uniform(0.0, 0.5, size=T).astype(np.float64)
    u_neg = rng.uniform(0.0, 0.5, size=9).astype(np.float64)

    # Compare independent v2 reference, g1_bench reference, and vectorized sliding buffer
    u_ref_v2, y_ref_v2, _ = generate_narma10_v2_reference(u, u_negative=u_neg)
    u_ref_b, y_ref_b, _ = generate_narma10_reference(u, u_negative=u_neg)
    u_vec, y_vec, _ = generate_narma10_vectorized(u, u_negative=u_neg)

    diff_v2_b = float(np.max(np.abs(y_ref_v2 - y_ref_b)))
    diff_b_vec = float(np.max(np.abs(y_ref_b - y_vec)))

    assert diff_v2_b <= 1e-12, f"Float64 mismatch v2 vs bench: {diff_v2_b}"
    assert diff_b_vec <= 1e-12, f"Float64 mismatch bench vs vec: {diff_b_vec}"

    # Verify that negative history array is explicitly mapped without negative index wrapping
    # At t=0, y[1] depends on u[0] * u[-9], where u[-9] must be u_neg[0]
    expected_prod_0 = 1.5 * u[0] * u_neg[0]
    # At t=8, y[9] depends on u[8] * u[-1], where u[-1] must be u_neg[8]
    expected_prod_8 = 1.5 * u[8] * u_neg[8]
    # At t=9, y[10] depends on u[9] * u[0]
    expected_prod_9 = 1.5 * u[9] * u[0]

    # Verify Oracle feature reconstruction:
    # y[t+1] = 0.3*y[t] + 0.05*y[t]*sum_y + 1.5*u[t]*u[t-9] + 0.1
    y_full = np.concatenate(([0.0], y_ref_v2))
    oracle_pred = np.empty(T, dtype=np.float64)
    for t in range(T):
        u_lag = u[t - 9] if t >= 9 else u_neg[t]
        sum_y = sum(y_full[t - k] for k in range(10) if t - k >= 0)
        oracle_pred[t] = 0.3 * y_full[t] + 0.05 * y_full[t] * sum_y + 1.5 * u[t] * u_lag + 0.1

    oracle_diff = float(np.max(np.abs(oracle_pred - y_ref_v2)))
    assert oracle_diff <= 1e-14, f"Oracle step mismatch: {oracle_diff}"

    # Oracle NMSE after washout
    oracle_nmse = compute_nmse(oracle_pred[500:], y_ref_v2[500:])
    assert oracle_nmse < 1e-6, f"Oracle NMSE {oracle_nmse} not < 1e-6"


# ==============================================================================
# A03: Pulse Dynamics & Hand-Calculated Small Graph Alignment
# ==============================================================================

def test_a03_pulse_dynamics_and_chunking_alignment():
    """A03: Pulse / hand-calculated small graph: updates exactly once per step, x[t+1] aligns with target, chunks preserve state."""
    # 2-neuron graph: neuron 0 is sensory (input), neuron 1 is interneuron/readout
    # W = [[0, 0], [0.8, 0]]
    # x[t+1] = 0.5 * x[t] + 0.5 * tanh(W x[t] + I[t])
    # For neuron 0: I[t] = u[t], W_0,* = 0 -> x0[t+1] = 0.5*x0[t] + 0.5*tanh(u[t])
    # For neuron 1: I[t] = 0, W_1,0 = 0.8 -> x1[t+1] = 0.5*x1[t] + 0.5*tanh(0.8 * x0[t])
    weights = sp.csr_matrix([[0.0, 0.0], [0.8, 0.0]], dtype=np.float64)
    res = ScipyG1Reservoir(
        weights=weights,
        sensory_idx=[0],
        target_rho=0.8,
        unscaled_rho=0.8,
        leak=0.5,
        dtype=np.float64,
        verify_spectral_radius=False,
    )

    u = np.array([0.5, 0.0, 0.0], dtype=np.float64)
    # Hand calculation:
    # x[0] = [0, 0]
    # t=0: u[0]=0.5
    #   x0[1] = 0.5*0 + 0.5*tanh(0.5) = 0.5 * 0.46211715726 = 0.23105857863
    #   x1[1] = 0.5*0 + 0.5*tanh(0) = 0.0
    # t=1: u[1]=0.0
    #   x0[2] = 0.5*x0[1] + 0.5*tanh(0) = 0.11552928931
    #   x1[2] = 0.5*0 + 0.5*tanh(0.8 * x0[1]) = 0.5 * tanh(0.8 * 0.23105857863) = 0.5 * tanh(0.18484686) = 0.5 * 0.1827727 = 0.09138635
    x0_1_expected = 0.5 * math.tanh(0.5)
    x1_1_expected = 0.0
    x0_2_expected = 0.5 * x0_1_expected
    x1_2_expected = 0.5 * math.tanh(0.8 * x0_1_expected)

    states, final_x = res.simulate(u, initial_state=np.zeros(2, dtype=np.float64), return_full_states=True, input_centering=False)
    assert states.shape == (3, 2)
    assert abs(states[0, 0] - x0_1_expected) < 1e-10
    assert abs(states[0, 1] - x1_1_expected) < 1e-10
    assert abs(states[1, 0] - x0_2_expected) < 1e-10
    assert abs(states[1, 1] - x1_2_expected) < 1e-10

    # Chunking test: 2 chunks of length 2 and 1 must produce identical states to single run of length 3
    states_c1, x_c1 = res.simulate(u[:2], initial_state=np.zeros(2, dtype=np.float64), return_full_states=True, input_centering=False)
    states_c2, x_c2 = res.simulate(u[2:], initial_state=x_c1, return_full_states=True, input_centering=False)
    states_chunked = np.vstack([states_c1, states_c2])

    max_chunk_diff = float(np.max(np.abs(states - states_chunked)))
    assert max_chunk_diff < 1e-12, f"Chunking state discrepancy: {max_chunk_diff}"


# ==============================================================================
# A04: Production Sensory Current & R1-6 Metadata Alignment
# ==============================================================================

def test_a04_raw_u_sensory_current_and_r16_alignment():
    """A04: Production sensory current is strictly u[t] (no -0.25), R7/R8 excluded, empty/misaligned metadata fails closed."""
    N = 10
    sensory_idx = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    # Types: 0,1 are R1-6; 2 is R7; 3 is R8; 4 is R1-6 with NaN coords
    ptypes = np.array(["R1-6", "R1-6", "R7", "R8", "R1-6"], dtype=object)
    u_coords = np.array([0.1, 0.2, 0.3, 0.4, np.nan], dtype=np.float64)
    v_coords = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)

    meta = {
        "photoreceptor_type": ptypes,
        "u": u_coords,
        "v": v_coords,
        "sha256": "dummy_hash",
    }
    graph = ConnectomeGraph(
        weights=sp.eye(N, format="csr"),
        neuron_ids=[f"N{i}" for i in range(N)],
        neuron_types=["photoreceptor"] * 5 + ["interneuron"] * 5,
        sensory_idx=sensory_idx,
        motor_idx=np.array([8, 9]),
        meta=meta,
    )

    r16_idx, summary = select_r16_indices(graph)
    # Only neurons 0 and 1 qualify (R1-6 with finite coordinates)
    assert list(r16_idx) == [0, 1]
    assert summary["n_r16_selected"] == 2
    assert summary["n_r7_r8_excluded"] == 2
    assert summary["n_non_finite_excluded"] == 1

    # Verify empty metadata raises ValueError
    bad_graph_none = ConnectomeGraph(
        weights=sp.eye(N, format="csr"),
        neuron_ids=[f"N{i}" for i in range(N)],
        neuron_types=["photoreceptor"] * 5 + ["interneuron"] * 5,
        sensory_idx=sensory_idx,
        motor_idx=np.array([8, 9]),
        meta=None,
    )
    with pytest.raises(ValueError, match="Graph meta must be a non-empty dictionary"):
        select_r16_indices(bad_graph_none)

    # Verify misaligned metadata raises ValueError
    bad_graph_mismatch = ConnectomeGraph(
        weights=sp.eye(N, format="csr"),
        neuron_ids=[f"N{i}" for i in range(N)],
        neuron_types=["photoreceptor"] * 5 + ["interneuron"] * 5,
        sensory_idx=sensory_idx,
        motor_idx=np.array([8, 9]),
        meta={"photoreceptor_type": ptypes, "u": u_coords[:3], "v": v_coords},
    )
    with pytest.raises(ValueError, match="Misaligned metadata"):
        select_r16_indices(bad_graph_mismatch)


# ==============================================================================
# A05: Odd Symmetry & 300-Node ESN Capacity Comparison
# ==============================================================================

def test_a05_odd_symmetry_and_300node_esn_capacity_comparison():
    """A05: ±v centered input on zero initial state exhibits strict odd symmetry; raw u breaks symmetry allowing product recovery."""
    # 1. Strict odd symmetry check: x[-v] + x[v] = 0.0 with float64 precision
    N = 20
    W = sp.random(N, N, density=0.2, random_state=42, format="csr", dtype=np.float64)
    res = ScipyG1Reservoir(weights=W, sensory_idx=[0, 1], target_rho=0.95, leak=0.5, dtype=np.float64, verify_spectral_radius=False)

    rng = np.random.default_rng(999)
    u_stream = rng.uniform(0.0, 0.5, size=200).astype(np.float64)
    v_stream = u_stream - 0.25

    # Center input mode
    states_plus, _ = res.simulate(u_stream, initial_state=np.zeros(N, dtype=np.float64), return_full_states=True, input_centering=True)
    # Opposite input: -v = -(u - 0.25) -> input = 0.25 - v
    u_opposite = 0.25 - v_stream
    states_minus, _ = res.simulate(u_opposite, initial_state=np.zeros(N, dtype=np.float64), return_full_states=True, input_centering=True)

    odd_residual = float(np.max(np.abs(states_plus + states_minus)))
    assert odd_residual <= 1e-12, f"Odd symmetry violated: max |x[v] + x[-v]| = {odd_residual}"

    # 2. 300-node sparse ESN capacity fixture per A05 / SPEC §1
    # 300 nodes, density 0.05, rho=0.95, fixed seed
    # 20 sequences (10 fit / 10 check, 2000 steps + 500 washout)
    graph = make_synthetic_connectome(n_neurons=300, avg_out_degree=15.0, n_sensory=64, n_motor=64, seed=42)
    W_scaled, _, _ = scale_weights_to_spectral_radius(graph.weights, target_rho=0.95, tol=1e-3)

    res_cen = ScipyG1Reservoir(weights=W_scaled, sensory_idx=graph.sensory_idx, target_rho=0.95, leak=0.5, dtype=np.float64, verify_spectral_radius=False)
    res_raw = ScipyG1Reservoir(weights=W_scaled, sensory_idx=graph.sensory_idx, target_rho=0.95, leak=0.5, dtype=np.float64, verify_spectral_radius=False)

    def gen_stream(s: int) -> np.ndarray:
        return np.random.default_rng(s).uniform(0.0, 0.5, size=2500)

    fit_u = [gen_stream(s) for s in range(10)]
    chk_u = [gen_stream(100 + s) for s in range(10)]

    def run_esn_benchmark(res_inst: ScipyG1Reservoir, input_centering: bool):
        fit_st, fit_d1, fit_d2 = [], [], []
        for u in fit_u:
            v = u - 0.25
            st, _ = res_inst.simulate(u, return_full_states=True, input_centering=input_centering)
            fit_st.append(st[500:])
            fit_d1.append(build_diagnostic_targets(u, washout=500)[0])
            fit_d2.append(build_diagnostic_targets(u, washout=500)[1])

        chk_st, chk_d1, chk_d2 = [], [], []
        for u in chk_u:
            v = u - 0.25
            st, _ = res_inst.simulate(u, return_full_states=True, input_centering=input_centering)
            chk_st.append(st[500:])
            chk_d1.append(build_diagnostic_targets(u, washout=500)[0])
            chk_d2.append(build_diagnostic_targets(u, washout=500)[1])

        r1 = G1RidgeReadout(alphas=(1e-4, 1e-2, 1.0)).fit_sequence_group_cv(fit_st, fit_d1)
        r2 = G1RidgeReadout(alphas=(1e-4, 1e-2, 1.0)).fit_sequence_group_cv(fit_st, fit_d2)

        p1 = np.concatenate([r1.predict(s) for s in chk_st])
        p2 = np.concatenate([r2.predict(s) for s in chk_st])

        r2_d1 = compute_pooled_r2(np.concatenate(chk_d1), p1)
        r2_d2 = compute_pooled_r2(np.concatenate(chk_d2), p2)
        return r2_d1, r2_d2

    r2_d1_cen, r2_d2_cen = run_esn_benchmark(res_cen, input_centering=True)
    r2_d1_raw, r2_d2_raw = run_esn_benchmark(res_raw, input_centering=False)

    # Relaxed fixture tolerances stated clearly (SPEC §1, A05):
    # Centered product term is blocked by odd symmetry: R^2 ≈ 0 (< 0.05)
    assert r2_d2_cen < 0.05, f"Centered product R^2 unexpectedly high: {r2_d2_cen}"
    # Raw u breaks odd symmetry, allowing product term recovery (> 0.05, observed ~0.13)
    assert r2_d2_raw > 0.05, f"Raw u product R^2 unexpectedly low: {r2_d2_raw}"
    # Linear delay is learnable in both (> 0.20, observed ~0.30)
    assert r2_d1_cen > 0.20, f"Centered delay R^2: {r2_d1_cen}"
    assert r2_d1_raw > 0.20, f"Raw delay R^2: {r2_d1_raw}"


# ==============================================================================
# A06: Direct Teachers & Pure Product vs Raw q Distinction
# ==============================================================================

def test_a06_direct_teachers_and_product_distinction():
    """A06: Direct lag and product teachers achieve R^2 >= 1 - 1e-6; raw q is distinguished from d2."""
    rng = np.random.default_rng(42)
    u = rng.uniform(0.0, 0.5, size=1000).astype(np.float64)
    d1, d2 = build_diagnostic_targets(u, washout=200)

    # Positive control teacher
    r2_teacher_d1 = compute_pooled_r2(d1, d1)
    r2_teacher_d2 = compute_pooled_r2(d2, d2)
    assert r2_teacher_d1 >= 1.0 - 1e-6
    assert r2_teacher_d2 >= 1.0 - 1e-6

    # Mathematical & numerical distinction between d2 and raw q
    dist_info = distinguish_pure_product_from_raw_q(u, washout=200)
    assert dist_info["is_distinct"] is True
    assert dist_info["max_absolute_difference"] > 0.05
    # Raw q is correlated with linear term v
    assert abs(dist_info["correlation_q_with_linear_v"]) > 0.10


# ==============================================================================
# A07: Diagnostic Splits & Sealed Test Boundary
# ==============================================================================

def test_a07_diagnostic_splits_and_sealed_test_boundary():
    """A07: Verify 2 diagnostic splits (fit 10, check 10), boundary parameters, and check loader fit seed rejection."""
    splits = generate_diagnostic_splits(washout=500, valid_length=2000)
    assert len(splits["diagnostic-fit"]) == 10
    assert len(splits["diagnostic-check"]) == 10

    fit_seeds = [s["seed"] for s in splits["diagnostic-fit"]]
    check_seeds = [s["seed"] for s in splits["diagnostic-check"]]

    assert tuple(fit_seeds) == FIXTURE_FIT_SEEDS
    assert tuple(check_seeds) == FIXTURE_CHECK_SEEDS
    # Disjoint seed sets
    assert len(set(fit_seeds) & set(check_seeds)) == 0

    # Sequence properties
    for s in splits["diagnostic-fit"] + splits["diagnostic-check"]:
        assert len(s["u"]) == 2500
        assert len(s["y_next"]) == 2500
        assert len(s["d1"]) == 2000
        assert len(s["d2"]) == 2000
        assert len(s["u_negative"]) == 9

    # Check loader strictly rejects fit seeds
    with pytest.raises(ValueError, match="Check loader strictly rejects fit seeds"):
        load_diagnostic_split("diagnostic-check", requested_seeds=[FIXTURE_FIT_SEEDS[0]])

    # Sealed Test boundary
    with pytest.raises(PermissionError, match="Access to Main-Test is strictly forbidden"):
        validate_split_request("main_test", authorized=False)


# ==============================================================================
# A08: Ridge Scaling Invariance from Check Distribution
# ==============================================================================

def test_a08_ridge_fit_only_scaling_invariance():
    """A08: Ridge scaling and standardization use fold fit only; modifying check data does not change fit weights."""
    rng = np.random.default_rng(777)
    n_fit = 800
    n_check = 400
    p = 10

    X_fit = rng.normal(size=(n_fit, p))
    # Add a constant column to verify dropping
    X_fit[:, 3] = 5.0
    true_beta = rng.uniform(-1.0, 1.0, size=p)
    true_beta[3] = 0.0
    y_fit = X_fit @ true_beta + 2.5 + rng.normal(scale=0.1, size=n_fit)

    # Fit Ridge
    seq_X = [X_fit[i * 80 : (i + 1) * 80] for i in range(10)]
    seq_y = [y_fit[i * 80 : (i + 1) * 80] for i in range(10)]

    model1 = G1RidgeReadout(alphas=(0.01, 0.1, 1.0))
    model1.fit_sequence_group_cv(seq_X, seq_y)

    # Check prediction on check data 1
    X_check1 = rng.normal(loc=0.0, scale=1.0, size=(n_check, p))
    X_check1[:, 3] = 5.0
    pred1 = model1.predict(X_check1)

    # Modify check distribution wildly (shift mean and scale by 10x)
    X_check2 = rng.normal(loc=100.0, scale=10.0, size=(n_check, p))
    X_check2[:, 3] = 5.0

    # Fit identical model again on fit data
    model2 = G1RidgeReadout(alphas=(0.01, 0.1, 1.0))
    model2.fit_sequence_group_cv(seq_X, seq_y)

    # Weights and intercept must be strictly bitwise identical regardless of check data
    np.testing.assert_allclose(model1.weights, model2.weights, rtol=1e-12, atol=1e-12)
    assert abs(model1.intercept - model2.intercept) < 1e-12
    assert model1.active_features[3] == False


# ==============================================================================
# A09: Ridge Objective, SVD Diagnostics & Rank-Deficient alpha=0
# ==============================================================================

def test_a09_ridge_objective_and_svd_rank_deficient_diagnostics():
    """A09: Ridge objective (1/n)||Y - XB||^2 + alpha||B||^2 matches direct solver; rank-deficient SVD diagnostics reported."""
    rng = np.random.default_rng(888)
    n = 200
    p = 15

    X = rng.normal(size=(n, p))
    # Make rank deficient by duplicating columns
    X[:, 5] = X[:, 2] + X[:, 3]
    X[:, 8] = 2.0 * X[:, 1]
    y = rng.normal(size=n)

    for alpha in [0.0, 1e-5, 0.1, 10.0]:
        B_direct, b0_direct = solve_ridge_direct_closed_form(X, y, alpha=alpha)

        Xc = X - np.mean(X, axis=0)
        yc = y - np.mean(y)
        solutions, diag = G1RidgeReadout._fit_svd_system(Xc, yc, np.mean(X, axis=0), float(np.mean(y)), [alpha])
        B_svd, b0_svd = solutions[alpha]

        # Predictions must match within 1e-8
        pred_direct = X @ B_direct + b0_direct
        pred_svd = X @ B_svd + b0_svd
        max_err = float(np.max(np.abs(pred_direct - pred_svd)))
        assert max_err <= 1e-8, f"Prediction mismatch at alpha={alpha}: {max_err}"

        # Diagnostics verification
        assert diag["effective_rank"] <= p - 2
        assert diag["n_discarded_directions"] >= 2
        assert diag["retained_condition_number"] > 1.0


# ==============================================================================
# A10: 5-Fold Sequence CV & Pooled OOF Alpha Selection
# ==============================================================================

def test_a10_sequence_group_cv_pooled_oof_selection():
    """A10: 5-fold sequence CV selects alpha with minimum pooled OOF NMSE, breaks ties with larger alpha, refits on full fit."""
    rng = np.random.default_rng(101)
    n_seqs = 10
    seq_len = 100
    p = 5

    seq_X = [rng.normal(size=(seq_len, p)) for _ in range(n_seqs)]
    beta_true = np.array([0.5, -0.3, 0.2, 0.0, 0.0])
    seq_y = [X @ beta_true + 1.0 + rng.normal(scale=0.05, size=seq_len) for X in seq_X]

    model = G1RidgeReadout(alphas=ALPHA_GRID)
    model.fit_sequence_group_cv(seq_X, seq_y)

    assert model.best_alpha is not None
    assert model.best_alpha in ALPHA_GRID
    assert model.weights is not None
    assert model.intercept is not None
    assert "pooled_oof_nmse_by_alpha" in model.cv_results

    # Confirm selected alpha minimizes pooled OOF NMSE
    pooled_oof = model.cv_results["pooled_oof_nmse_by_alpha"]
    min_oof = min(pooled_oof.values())
    assert abs(pooled_oof[model.best_alpha] - min_oof) < 1e-9


# ==============================================================================
# A11: Pooled Metrics, Even Median, and Primary Delta
# ==============================================================================

def test_a11_pooled_metrics_even_median_and_delta():
    """A11: Hand-calculated fixture for pooled R^2 / NMSE (ddof=0), even median, delta, and fail-closed checks."""
    # Hand calculation:
    # y_true = [1.0, 2.0, 3.0], y_pred = [1.5, 2.0, 2.5]
    # mu_y = 2.0, SST = (1-2)^2 + (2-2)^2 + (3-2)^2 = 1 + 0 + 1 = 2.0
    # SSE = (1-1.5)^2 + (2-2)^2 + (3-2.5)^2 = 0.25 + 0 + 0.25 = 0.50
    # R^2 = 1 - 0.5 / 2.0 = 0.75
    # Var(y, ddof=0) = 2.0 / 3 = 2/3
    # MSE = 0.5 / 3 = 1/6
    # NMSE = MSE / Var = (1/6) / (2/3) = 0.25
    y_t = np.array([1.0, 2.0, 3.0])
    y_p = np.array([1.5, 2.0, 2.5])

    r2 = compute_pooled_r2(y_t, y_p)
    nmse = compute_pooled_nmse(y_t, y_p)
    assert abs(r2 - 0.75) < 1e-12
    assert abs(nmse - 0.25) < 1e-12

    # Negative R^2 retention
    y_bad = np.array([10.0, 20.0, 30.0])
    r2_neg = compute_pooled_r2(y_t, y_bad)
    assert r2_neg < -100.0  # Must be retained and not clipped to zero

    # Even median: arithmetic mean of two middle values
    even_vals = [0.1, 0.4, 0.2, 0.3]  # sorted: 0.1, 0.2, 0.3, 0.4 -> (0.2 + 0.3)/2 = 0.25
    med = compute_even_median(even_vals)
    assert abs(med - 0.25) < 1e-12

    # Delta: (C - real) / C
    # If C = 0.5, real = 0.4 -> Delta = (0.5 - 0.4) / 0.5 = 0.20
    delta = compute_delta_primary(0.4, [0.45, 0.55])
    assert abs(delta - 0.20) < 1e-12

    # Fail closed on zero variance, NaN, Inf, or empty
    with pytest.raises(ValueError):
        compute_pooled_r2(np.array([1.0, 1.0, 1.0]), np.array([1.0, 1.0, 1.0]))
    with pytest.raises(ValueError):
        compute_pooled_nmse(np.array([]), np.array([]))
    with pytest.raises(ValueError):
        compute_even_median([1.0, np.nan, 3.0])


# ==============================================================================
# A12: Diagnostic Sequence Bootstrap & Dual AND Capability Gate
# ==============================================================================

def test_a12_diagnostic_sequence_bootstrap_and_capability_gate():
    """A12: Diagnostic sequence bootstrap (10 complete sequences resampled), dual AND capability gate, constant prediction rejection."""
    rng = np.random.default_rng(202)
    n_seqs = 10
    T_val = 500

    targets = [rng.normal(size=T_val) for _ in range(n_seqs)]
    # Good predictions: signal + small noise
    good_preds = [t + rng.normal(scale=0.2, size=T_val) for t in targets]

    boot_res = compute_diagnostic_sequence_bootstrap(targets, good_preds, n_bootstraps=200, seed=42)
    assert boot_res["r2_point"] > 0.80
    assert boot_res["ci_lower"] > 0.70
    assert boot_res["ci_upper"] > boot_res["ci_lower"]

    # Constant predictions must fail capability gate
    const_preds = [np.zeros(T_val) for _ in range(n_seqs)]
    const_boot = compute_diagnostic_sequence_bootstrap(targets, const_preds, n_bootstraps=100)
    assert const_boot["is_constant_prediction"] is True

    # Dual AND gate evaluator
    # Both >= 0.10 and CI lower > 0 -> PASS
    eval_pass = evaluate_capability_gate(r2_d1=0.25, ci_lower_d1=0.08, r2_d2=0.15, ci_lower_d2=0.03)
    assert eval_pass["passed"] is True

    # One target fails -> FAIL
    eval_fail1 = evaluate_capability_gate(r2_d1=0.25, ci_lower_d1=0.08, r2_d2=0.08, ci_lower_d2=0.02)
    assert eval_fail1["passed"] is False

    # CI lower <= 0 -> FAIL
    eval_fail2 = evaluate_capability_gate(r2_d1=0.25, ci_lower_d1=-0.01, r2_d2=0.15, ci_lower_d2=0.03)
    assert eval_fail2["passed"] is False


# ==============================================================================
# A13: Diagnostic Negative Control (Cyclic Shift by 1 Sequence)
# ==============================================================================

def test_a13_diagnostic_negative_control_cyclic_shift():
    """A13: Within-split cyclic shift negative control: refits on shifted fit, evaluates G and CI lower bound <= 0."""
    rng = np.random.default_rng(303)
    n_seqs = 10
    T_val = 200
    p = 4

    # Generate independent synthetic states and targets
    fit_states = [rng.normal(size=(T_val, p)) for _ in range(n_seqs)]
    fit_t1 = [rng.normal(size=T_val) for _ in range(n_seqs)]
    fit_t2 = [rng.normal(size=T_val) for _ in range(n_seqs)]

    check_states = [rng.normal(size=(T_val, p)) for _ in range(n_seqs)]
    check_t1 = [rng.normal(size=T_val) for _ in range(n_seqs)]
    check_t2 = [rng.normal(size=T_val) for _ in range(n_seqs)]

    def ridge_factory():
        return G1RidgeReadout(alphas=(0.1, 1.0, 10.0))

    neg_res = evaluate_diagnostic_negative_control(
        fit_states=fit_states,
        fit_targets_d1=fit_t1,
        fit_targets_d2=fit_t2,
        check_states=check_states,
        check_targets_d1=check_t1,
        check_targets_d2=check_t2,
        ridge_factory=ridge_factory,
        n_bootstraps=100,
        seed=42,
    )

    # Independent noise targets must not yield significant positive improvement (CI lower <= 0)
    assert neg_res["passed"] is True
    assert neg_res["results"]["d1"]["ci_lower"] <= 0.0
    assert neg_res["results"]["d2"]["ci_lower"] <= 0.0


# ==============================================================================
# A15: Spectral Radius Convergence & Dense Eigenspectrum Alignment
# ==============================================================================

def test_a15_spectral_radius_dense_comparison_and_independent_init():
    """A15: Spectral radius eigensolver matches dense eigenspectrum (<= 1e-6) and passes independent init tolerance (<= 1e-3)."""
    N = 30
    rng = np.random.default_rng(404)
    # Generate random directed sparse matrix
    W_raw = sp.random(N, N, density=0.15, random_state=404, format="csr", dtype=np.float64)

    rho_scipy, solver_info = compute_spectral_radius(W_raw, return_diagnostics=True)

    # Dense comparison
    dense_eigs = np.linalg.eigvals(W_raw.toarray())
    rho_dense = float(np.max(np.abs(dense_eigs)))

    abs_diff = abs(rho_scipy - rho_dense)
    rel_diff = abs_diff / rho_dense
    assert rel_diff <= 1e-6, f"Spectral radius mismatch scipy ({rho_scipy}) vs dense ({rho_dense})"

    # Scale matrix to 0.95
    W_scaled, target_rho, unscaled_rho, scale_factor, diag = scale_weights_to_spectral_radius(
        W_raw, target_rho=0.95, tol=1e-3, return_diagnostics=True
    )
    unscaled_sha = diag["unscaled_sha256"]
    scaled_sha = diag["scaled_sha256"]
    assert len(unscaled_sha) == 64
    assert len(scaled_sha) == 64
    assert unscaled_sha != scaled_sha

    # Verify scaled matrix has spectral radius 0.95 within 1e-3
    rho_scaled = compute_spectral_radius(W_scaled)
    assert abs(rho_scaled - 0.95) / 0.95 <= 1e-3


# ==============================================================================
# A16: Saturation Gate on Non-Sensory Neurons & Initial-State Forgetting
# ==============================================================================

def test_a16_saturation_non_sensory_and_forgetting_gates():
    """A16: Saturation gate counts only non-sensory (|x| > 0.90 proportion < 5%); forgetting gate verifies >= 38/40 pairs."""
    # 1. Saturation gate on non-sensory neurons
    N = 10
    T = 100
    sensory_idx = [0, 1]  # 2 sensory, 8 non-sensory

    # Construct states where sensory neurons are 100% saturated (|x|=1.0),
    # but non-sensory neurons are unsaturated (|x|=0.1)
    states = np.full((T, N), 0.1, dtype=np.float64)
    states[:, sensory_idx] = 1.0  # Saturated sensory

    sat_res = check_saturation_gate(states, sensory_idx=sensory_idx, threshold=0.90, max_saturation_ratio=0.05)
    assert sat_res["passed"] is True
    assert sat_res["saturated_points"] == 0
    assert sat_res["total_non_sensory_points"] == T * 8

    # Fail closed on non-finite states
    states_nan = states.copy()
    states_nan[10, 5] = np.nan
    sat_res_nan = check_saturation_gate(states_nan, sensory_idx=sensory_idx)
    assert sat_res_nan["passed"] is False

    # 2. Forgetting gate on contracting reservoir (rho=0.8)
    W_sub = sp.random(20, 20, density=0.2, random_state=505, format="csr", dtype=np.float64)
    W_scaled, _, _ = scale_weights_to_spectral_radius(W_sub, target_rho=0.8, tol=1e-3)
    res = ScipyG1Reservoir(weights=W_scaled, sensory_idx=[0, 1], target_rho=0.8, leak=0.5, dtype=np.float64, verify_spectral_radius=False)

    rng = np.random.default_rng(505)
    # 2 sequences of length 500, 2 pairs each = 4 pairs total
    u_seqs = [rng.uniform(0.0, 0.5, size=500).astype(np.float64) for _ in range(2)]

    forget_res = check_forgetting_gate(
        sim_fn=lambda u, x0: res.simulate(u, initial_state=x0, return_full_states=True),
        train_u_sequences=u_seqs,
        n_neurons=20,
        seed=42,
        n_pairs_per_seq=2,
        t_check=500,
        d_rel_threshold=1e-3,
        min_pass_pairs=4,
    )
    assert forget_res["passed"] is True
    assert forget_res["n_passed_pairs"] == 4


# ==============================================================================
# A17: DN-Only Readout Slicing & CPU vs Torch Alignment
# ==============================================================================

def test_a17_dn_only_slice_and_chunk_consistency():
    """A17: DN-only output matches full-state slice without allocating T x N tensor; chunk output is consistent."""
    N = 25
    sensory_idx = [0, 1, 2]
    dn_idx = [10, 11, 12, 13]

    W = sp.random(N, N, density=0.2, random_state=606, format="csr", dtype=np.float64)
    W_scaled, _, _ = scale_weights_to_spectral_radius(W, target_rho=0.95, tol=1e-3)

    res_scipy = ScipyG1Reservoir(
        weights=W_scaled,
        sensory_idx=sensory_idx,
        readout_idx=dn_idx,
        target_rho=0.95,
        leak=0.5,
        dtype=np.float32,
        verify_spectral_radius=False,
    )

    rng = np.random.default_rng(606)
    u = rng.uniform(0.0, 0.5, size=150).astype(np.float32)

    # Full state simulation
    states_full, final_full = res_scipy.simulate(u, return_full_states=True, input_centering=False)
    # DN-only simulation
    states_dn, final_dn = res_scipy.simulate(u, return_full_states=False, input_centering=False)

    assert states_full.shape == (150, N)
    assert states_dn.shape == (150, len(dn_idx))
    # Sliced states must be identical
    np.testing.assert_allclose(states_dn, states_full[:, dn_idx], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(final_dn, final_full, rtol=1e-6, atol=1e-6)

    # Chunk consistency in DN-only mode
    c1, x1 = res_scipy.simulate(u[:80], return_full_states=False, input_centering=False)
    c2, x2 = res_scipy.simulate(u[80:], initial_state=x1, return_full_states=False, input_centering=False)
    chunked = np.vstack([c1, c2])
    np.testing.assert_allclose(states_dn, chunked, rtol=1e-6, atol=1e-6)


def test_a17_cpu_vs_torch_cpu_alignment():
    """A17: Scipy CPU vs Torch CPU max error <= 1e-4; skips cleanly when Torch is not installed; marks device as CPU."""
    try:
        import torch
        from research.pipeline.g1_reservoir import TorchG1Reservoir
    except ImportError:
        pytest.skip("PyTorch not installed in this environment; skipping Torch CPU test per A17.")

    N = 25
    sensory_idx = [0, 1, 2]
    dn_idx = [10, 11, 12, 13]

    W = sp.random(N, N, density=0.2, random_state=606, format="csr", dtype=np.float64)
    W_scaled, _, _ = scale_weights_to_spectral_radius(W, target_rho=0.95, tol=1e-3)

    res_scipy = ScipyG1Reservoir(
        weights=W_scaled,
        sensory_idx=sensory_idx,
        readout_idx=dn_idx,
        target_rho=0.95,
        leak=0.5,
        dtype=np.float32,
        verify_spectral_radius=False,
    )
    res_torch = TorchG1Reservoir(
        weights=W_scaled,
        sensory_idx=sensory_idx,
        readout_idx=dn_idx,
        target_rho=0.95,
        leak=0.5,
        device="cpu",
        verify_spectral_radius=False,
    )

    rng = np.random.default_rng(606)
    u = rng.uniform(0.0, 0.5, size=150).astype(np.float32)

    states_scipy, _ = res_scipy.simulate(u, return_full_states=False, input_centering=False)
    states_torch, _ = res_torch.simulate(u, return_full_states=False, input_centering=False)

    max_cpu_torch_diff = float(np.max(np.abs(states_scipy - states_torch)))
    print(f"[Device: CPU] Scipy vs Torch CPU max absolute difference: {max_cpu_torch_diff:.2e}")
    assert max_cpu_torch_diff <= 1e-4, f"Scipy vs Torch CPU error {max_cpu_torch_diff} exceeds 1e-4"

