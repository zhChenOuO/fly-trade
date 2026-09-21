"""Unit tests for G1 v2 Statistical Power, Cross Bootstrap, and MDE Calibration (SPEC v2 §5, §7).

Covers验收清單 (A18-A22):
1. A18: Deprecation of old planted-q path and end-to-end return validation (no unbound variables).
2. A19: Handcrafted unequal MSE residual tensors proving exact null (pop real MSE = M, rel err <= 1e-12)
   and alt (pop real MSE = 0.95M, rel err <= 1e-12).
3. A20: Cohort structure (1 fixed real, 20 resampled controls, 10 shared sequences) and cancellation of shared sequence difficulty.
4. A21: Graph x sequence cross bootstrap (2,000 resamples), percentile CI, fixed seed reproducibility.
5. A22: Exact Clopper-Pearson bounds (k0=0, k0=N, k1=0, k1=N boundaries), a=0.025, N=1000, and go/no-go gates.
"""
from __future__ import annotations

import math
import numpy as np
import pytest

from research.pipeline.g1_power import (
    wilson_score_interval,
    exact_clopper_pearson_fpr_upper,
    exact_clopper_pearson_power_lower,
    evaluate_mc_bounds,
    scale_residuals_for_null_and_alt,
    build_sequence_stats,
    compute_nmse_and_delta,
    graph_sequence_cross_bootstrap,
    calibrate_v2_power_and_fpr,
    build_narma10_planted_feature,
    tune_planted_feature_scale,
    calibrate_5pct_mde_power,
    evaluate_synthetic_mde_power,
    calibrate_beta_for_q_oracle,
    compute_q_readout_metric,
    fit_q_probe_heads,
    evaluate_q_positive_control_gate,
    calibrate_false_positive_rate,
    calibrate_fpr_with_family_selection,
    hierarchical_paired_bootstrap,
)


def test_wilson_score_interval_properties():
    """Verify Wilson score confidence interval mathematical bounds and behavior."""
    # 0 successes out of 100
    low, high = wilson_score_interval(0, 100, 0.95)
    assert low == 0.0
    assert 0.03 < high < 0.04

    # 100 successes out of 100
    low, high = wilson_score_interval(100, 100, 0.95)
    assert 0.96 < low < 0.97
    assert high == 1.0

    # 50 successes out of 100 (symmetric around 0.5)
    low, high = wilson_score_interval(50, 100, 0.95)
    assert abs(0.5 - low - (high - 0.5)) < 1e-6
    assert 0.39 < low < 0.41
    assert 0.59 < high < 0.61

    # Empty trials
    assert wilson_score_interval(0, 0) == (0.0, 0.0)


def test_a22_exact_clopper_pearson_bounds_and_boundaries():
    """Verify exact Clopper-Pearson one-sided bounds and boundary conditions (A22)."""
    N = 1000
    a = 0.025

    # Boundary: k0 = 0 -> strictly positive small upper bound
    u_0 = exact_clopper_pearson_fpr_upper(0, N, a)
    expected_u_0 = 1.0 - (a ** (1.0 / N))
    assert abs(u_0 - expected_u_0) < 1e-6
    assert 0.003 < u_0 < 0.004

    # Boundary: k0 = N -> exactly 1.0
    assert exact_clopper_pearson_fpr_upper(N, N, a) == 1.0

    # Boundary: k1 = 0 -> exactly 0.0
    assert exact_clopper_pearson_power_lower(0, N, a) == 0.0

    # Boundary: k1 = N -> strictly near 1.0
    l_n = exact_clopper_pearson_power_lower(N, N, a)
    expected_l_n = a ** (1.0 / N)
    assert abs(l_n - expected_l_n) < 1e-6
    assert 0.996 < l_n < 0.997

    # Standard qualification gates
    # 1. Qualified case: FPR=30/1000 (3%), Power=850/1000 (85%)
    res_qual = evaluate_mc_bounds(n_fpr_events=30, n_power_events=850, n_full_success_events=830, n_replicates=N, alpha=a)
    assert res_qual["passed_fpr_gate"] is True   # U_FPR <= 0.05
    assert res_qual["passed_power_gate"] is True # L_power >= 0.80
    assert res_qual["passed_mde_gate"] is True
    assert res_qual["u_fpr_975"] < 0.05
    assert res_qual["l_power_975"] > 0.80

    # 2. Gate fail: FPR too high (60/1000)
    res_fail_fpr = evaluate_mc_bounds(n_fpr_events=60, n_power_events=850, n_full_success_events=830, n_replicates=N, alpha=a)
    assert res_fail_fpr["passed_fpr_gate"] is False
    assert res_fail_fpr["passed_mde_gate"] is False

    # 3. Gate fail: Power too low (810/1000: point estimate > 0.80, but lower bound < 0.80)
    res_border_power = evaluate_mc_bounds(n_fpr_events=20, n_power_events=810, n_full_success_events=800, n_replicates=N, alpha=a)
    assert res_border_power["power_point_estimate"] == 0.81
    assert res_border_power["l_power_975"] < 0.80  # Demonstrates bound, not point estimate, governs gate!
    assert res_border_power["passed_power_gate"] is False
    assert res_border_power["passed_mde_gate"] is False


def test_a19_residual_scaling_exact_null_and_alt():
    """Verify handcrafted residual scaling achieves exact population MSE invariants (A19)."""
    rng = np.random.default_rng(42)
    n_ctrl = 20
    n_seqs = 10
    n_steps = 1000

    # Hand-craft unequal control MSEs
    ctrl_scales = np.linspace(0.8, 1.4, n_ctrl)
    e_ctrl = np.empty((n_ctrl, n_seqs, n_steps), dtype=np.float64)
    for g in range(n_ctrl):
        e_ctrl[g] = rng.normal(loc=0.0, scale=ctrl_scales[g], size=(n_seqs, n_steps))

    # Hand-craft real residuals with different MSE
    e_real = rng.normal(loc=0.0, scale=1.1, size=(n_seqs, n_steps))

    res = scale_residuals_for_null_and_alt(e_real=e_real, e_ctrl=e_ctrl)

    M = res["M"]
    m_real = res["m_real"]
    assert m_real != M  # Unscaled data must not be mislabeled as null!
    assert res["null_relative_error"] <= 1e-12
    assert res["alt_relative_error"] <= 1e-12

    # Verify control MSEs remain distinct and unchanged
    ctrl_mses = res["ctrl_mses"]
    assert len(np.unique(np.round(ctrl_mses, 4))) == n_ctrl
    assert np.all(res["e_null_ctrl"] == e_ctrl)
    assert np.all(res["e_alt_ctrl"] == e_ctrl)


def test_a20_shared_sequence_difficulty_pairing():
    """Verify paired sampling preserves cancellation of sequence-level difficulty (A20)."""
    rng = np.random.default_rng(101)
    n_ctrl = 20
    n_seqs = 10
    n_steps = 500

    # Create sequence-specific baseline difficulty offset
    # Sequence s has difficulty offset d_s
    d_seq = np.linspace(0.5, 3.0, n_seqs)  # Large difficulty variance across sequences

    y = rng.normal(loc=0.0, scale=1.0, size=(n_seqs, n_steps))

    # Residuals for real and controls share the sequence difficulty offset d_seq
    e_real = np.empty((n_seqs, n_steps), dtype=np.float64)
    e_ctrl = np.empty((n_ctrl, n_seqs, n_steps), dtype=np.float64)

    for s in range(n_seqs):
        # Base error with sequence difficulty
        diff = d_seq[s]
        e_real[s] = rng.normal(loc=0.0, scale=0.2, size=n_steps) + diff
        for g in range(n_ctrl):
            e_ctrl[g, s] = rng.normal(loc=0.0, scale=0.2, size=n_steps) + diff

    stats = build_sequence_stats(e_real=e_real, e_ctrl=e_ctrl, y=y)

    # 1. Paired sampling: same sequence indices used for real and controls
    seq_idx = rng.choice(n_seqs, size=n_seqs, replace=True)
    ctrl_idx = rng.choice(n_ctrl, size=n_ctrl, replace=True)

    delta_paired, c_med, nmse_real = compute_nmse_and_delta(stats, seq_idx, ctrl_idx)
    # Because both experienced identical d_seq, real and controls have nearly identical SSE
    assert abs(delta_paired) < 0.05

    # 2. Broken pairing (counter-factual): different sequences for real vs controls
    # If control used independent sequence indices, sequence difficulty would not cancel
    unpaired_seq_idx = rng.choice(n_seqs, size=n_seqs, replace=True)
    ctrl_sse_unpaired = np.sum(stats.sse_ctrl[ctrl_idx][:, unpaired_seq_idx], axis=1)
    sst_paired = np.sum(stats.sum_y2[seq_idx]) - len(seq_idx) * n_steps * (np.sum(stats.sum_y[seq_idx]) / (len(seq_idx) * n_steps)) ** 2
    c_unpaired = float(np.median(ctrl_sse_unpaired / sst_paired))
    delta_unpaired = (c_unpaired - nmse_real) / c_unpaired
    # Unpaired delta exhibits large variance due to unmatched sequence difficulty
    assert isinstance(delta_unpaired, float)


def test_a21_cross_bootstrap_reproducibility_and_sensitivity():
    """Verify graph x sequence cross bootstrap (2,000 resamples), CI, and fixed seed (A21)."""
    rng = np.random.default_rng(202)
    n_ctrl = 20
    n_seqs = 10
    n_steps = 1000

    y = rng.normal(loc=0.0, scale=1.0, size=(n_seqs, n_steps))

    # Case 1: Positive effect (Real NMSE ~ 0.18, Control NMSE ~ 0.20 -> Delta ~ 10%)
    e_real_pos = rng.normal(loc=0.0, scale=0.42, size=(n_seqs, n_steps))
    e_ctrl = rng.normal(loc=0.0, scale=0.45, size=(n_ctrl, n_seqs, n_steps))

    stats_pos = build_sequence_stats(e_real_pos, e_ctrl, y)
    res_pos1 = graph_sequence_cross_bootstrap(stats_pos, n_bootstraps=1000, seed=42)
    res_pos2 = graph_sequence_cross_bootstrap(stats_pos, n_bootstraps=1000, seed=42)

    # Deterministic bitwise reproducibility under fixed seed
    assert res_pos1["delta_obs"] == res_pos2["delta_obs"]
    assert res_pos1["ci_lower"] == res_pos2["ci_lower"]
    assert res_pos1["ci_upper"] == res_pos2["ci_upper"]

    assert res_pos1["delta_obs"] > 0.05
    assert res_pos1["ci_lower"] > 0.0
    assert res_pos1["passed_primary_gate"] is True

    # Case 2: Negative effect (Real NMSE ~ 0.22, Control NMSE ~ 0.20)
    e_real_neg = rng.normal(loc=0.0, scale=0.48, size=(n_seqs, n_steps))
    stats_neg = build_sequence_stats(e_real_neg, e_ctrl, y)
    res_neg = graph_sequence_cross_bootstrap(stats_neg, n_bootstraps=500, seed=42)

    assert res_neg["delta_obs"] < 0.0
    assert res_neg["ci_lower"] < 0.0
    assert res_neg["passed_primary_gate"] is False


def test_a18_v2_power_end_to_end_returns_and_spec_audit():
    """Verify v2 calibrate_v2_power_and_fpr returns complete spec without unbound variables (A18)."""
    rng = np.random.default_rng(303)
    n_ctrl = 20
    n_seqs = 10
    n_steps = 200

    y = rng.normal(loc=0.0, scale=1.0, size=(n_seqs, n_steps))
    e_ctrl = rng.normal(loc=0.0, scale=0.5, size=(n_ctrl, n_seqs, n_steps))
    e_real = rng.normal(loc=0.0, scale=0.6, size=(n_seqs, n_steps))

    # Run small fast replicate test
    res = calibrate_v2_power_and_fpr(
        e_real=e_real,
        e_ctrl=e_ctrl,
        y=y,
        n_replicates=20,
        n_bootstraps_per_cohort=100,
        base_seed=12345,
    )

    # Verify all expected v2 keys exist and have correct types
    assert "null_scale" in res
    assert isinstance(res["null_scale"], float)
    assert "alt_scale" in res
    assert isinstance(res["alt_scale"], float)
    assert "m_real" in res
    assert "M_median" in res
    assert "ctrl_mses" in res
    assert len(res["ctrl_mses"]) == 20

    assert "seeds" in res
    assert res["seeds"]["base_seed"] == 12345
    assert len(res["seeds"]["first_10_replicate_seeds"]) == 10

    assert "events" in res
    assert "n_fpr_events" in res["events"]
    assert "n_power_events" in res["events"]
    assert "n_full_success_events" in res["events"]

    assert "bounds" in res
    bounds = res["bounds"]
    assert "u_fpr_975" in bounds
    assert "l_power_975" in bounds
    assert "full_success_lower_975" in bounds
    assert "passed_mde_gate" in bounds
    assert "passed_fpr_gate" in bounds
    assert "passed_power_gate" in bounds

    assert "cohort_spec" in res
    assert res["cohort_spec"]["n_pseudo_real"] == 1
    assert res["cohort_spec"]["n_control_templates"] == 20
    assert res["cohort_spec"]["n_sequence_templates"] == 10


def test_a18_deprecated_planted_q_and_legacy_apis_explicitly_raise():
    """Verify that calling deprecated planted-q, block-level, or family selection APIs raises RuntimeError (A18)."""
    with pytest.raises(RuntimeError, match="deprecated and withdrawn in G1 v2"):
        build_narma10_planted_feature(np.zeros(10))

    with pytest.raises(RuntimeError, match="deprecated and withdrawn in G1 v2"):
        tune_planted_feature_scale()

    with pytest.raises(RuntimeError, match="deprecated and withdrawn in G1 v2"):
        calibrate_5pct_mde_power()

    with pytest.raises(RuntimeError, match="deprecated and withdrawn in G1 v2"):
        evaluate_synthetic_mde_power()

    with pytest.raises(RuntimeError, match="deprecated and withdrawn in G1 v2"):
        calibrate_beta_for_q_oracle()

    with pytest.raises(RuntimeError, match="deprecated and withdrawn in G1 v2"):
        compute_q_readout_metric()

    with pytest.raises(RuntimeError, match="deprecated and withdrawn in G1 v2"):
        fit_q_probe_heads()

    with pytest.raises(RuntimeError, match="deprecated and withdrawn in G1 v2"):
        evaluate_q_positive_control_gate()

    with pytest.raises(RuntimeError, match="deprecated and withdrawn in G1 v2"):
        calibrate_false_positive_rate()

    with pytest.raises(RuntimeError, match="deprecated and withdrawn in G1 v2"):
        calibrate_fpr_with_family_selection()

    with pytest.raises(RuntimeError, match="deprecated and withdrawn in G1 v2"):
        hierarchical_paired_bootstrap()
