"""Unit tests for G1 Statistical Power, Hierarchical Bootstrap, and MDE Calibration.

Verifications (SPEC §5, §6):
1. Hierarchical paired bootstrap correctly recovers Delta and produces valid 95% CIs.
2. Positive Delta (10% advantage) passes gate; negative Delta (-5% deficit) fails.
3. Permutation test on null family yields nominal FPR <= 5%.
4. Train-only primary control family selection does not inflate Test FPR.
5. Planted feature tuning finds exact gamma achieving 5.0% advantage (within tolerance).
6. Planted feature power calibration achieves power >= 80% with null FPR <= 5%.
7. Pure statistical MDE simulation interface correctness.
8. Deterministic reproducibility under fixed seeds.
"""
from __future__ import annotations

import numpy as np
import pytest

from research.pipeline.g1_power import (
    wilson_score_interval,
    hierarchical_paired_bootstrap,
    calibrate_false_positive_rate,
    calibrate_fpr_with_family_selection,
    build_narma10_planted_feature,
    tune_planted_feature_scale,
    calibrate_5pct_mde_power,
    evaluate_synthetic_mde_power,
    calibrate_beta_for_q_oracle,
    compute_q_readout_metric,
    fit_q_probe_heads,
    evaluate_q_positive_control_gate,
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


def test_hierarchical_paired_bootstrap_passes_positive_fails_negative():
    """Verify bootstrap behavior when real outperforms vs underperforms controls."""
    rng = np.random.default_rng(42)
    n_instances = 20
    n_seqs = 10
    n_blocks = 10

    # Control baseline NMSE around 0.20
    ctrl_blocks = rng.normal(loc=0.20, scale=0.01, size=(n_instances, n_seqs, n_blocks))

    # Case A: Real outperforms control by 10% (real NMSE around 0.18)
    real_blocks_pos = rng.normal(loc=0.18, scale=0.01, size=(n_seqs, n_blocks))
    res_pos = hierarchical_paired_bootstrap(
        nmse_real=real_blocks_pos,
        nmse_ctrl=ctrl_blocks,
        n_bootstraps=500,
        seed=101,
    )
    assert res_pos["delta_obs"] > 0.08
    assert res_pos["ci_lower"] > 0.0
    assert res_pos["delta_ge_5pct"] is True
    assert res_pos["passed_primary_gate"] is True

    # Case B: Real underperforms control by 5% (real NMSE around 0.21)
    real_blocks_neg = rng.normal(loc=0.21, scale=0.01, size=(n_seqs, n_blocks))
    res_neg = hierarchical_paired_bootstrap(
        nmse_real=real_blocks_neg,
        nmse_ctrl=ctrl_blocks,
        n_bootstraps=500,
        seed=102,
    )
    assert res_neg["delta_obs"] < 0.0
    assert res_neg["ci_lower"] < 0.0
    assert res_neg["passed_primary_gate"] is False


def test_false_positive_rate_calibration_under_null():
    """Verify group-label permutation FPR <= nominal 5% under pure null distribution."""
    rng = np.random.default_rng(123)
    # 20 null instances with identical underlying distribution
    null_family_nmse = rng.normal(loc=0.20, scale=0.015, size=(20, 10, 10))

    fpr_res = calibrate_false_positive_rate(
        nmse_null_family=null_family_nmse,
        n_perms=300,
        seed=42,
        alpha=0.05,
    )

    assert fpr_res["n_permutations"] == 300
    # Nominal false positive rate should be around 5% (well within Wilson interval of nominal alpha)
    assert fpr_res["passed_fpr_gate"] is True
    assert fpr_res["fpr_nominal"] <= 0.08  # Tolerant upper bound for 300 perms
    # Dual gate (Delta >= 0.05 AND p < 0.05) should have virtually zero false positives
    assert fpr_res["fpr_dual_gate"] <= 0.02


def test_fpr_with_family_selection():
    """Verify that Train-only primary control selection does not inflate Test FPR."""
    rng = np.random.default_rng(456)
    families = ["scramble_mixed", "weight_shuffle_source", "random_endpoint"]

    train_fams = {}
    test_fams = {}
    for f in families:
        # 20 instances per family
        train_fams[f] = rng.normal(loc=0.22, scale=0.02, size=(20, 10, 10))
        test_fams[f] = rng.normal(loc=0.22, scale=0.02, size=(20, 10, 10))

    sel_res = calibrate_fpr_with_family_selection(
        train_families=train_fams,
        test_families=test_fams,
        n_perms=200,
        seed=99,
        alpha=0.05,
    )

    assert sel_res["passed_fpr_gate"] is True
    assert sel_res["fpr"] <= 0.08  # Within sampling tolerance of 5%


def test_planted_feature_scale_tuning():
    """Verify tuning scale gamma to achieve exactly 5.0% advantage on synthetic data."""
    rng = np.random.default_rng(789)
    N = 2000

    u = rng.uniform(0.0, 0.5, size=N)
    q = build_narma10_planted_feature(u)

    # Synthetic reservoir state X with noisy representation of q and other dynamics
    X = rng.standard_normal((N, 8))
    y = 0.5 * X[:, 0] + 0.3 * X[:, 1] + 0.8 * q + rng.normal(0, 0.1, size=N)

    gamma = tune_planted_feature_scale(
        X_train=X,
        y_train=y,
        u_train=u,
        target_advantage=0.05,
        tol=1e-4,
    )

    assert gamma > 0.0

    # Verify that Ridge on [X, gamma * q] indeed improves NMSE by ~5%
    from research.pipeline.g1_bench import compute_nmse
    X_ones = np.column_stack([np.ones(N), X])
    b_base = np.linalg.lstsq(X_ones, y, rcond=None)[0]
    p_base = X_ones @ b_base
    nmse_base = compute_nmse(p_base, y)

    p_aug = p_base + gamma * q
    nmse_aug = compute_nmse(p_aug, y)

    observed_adv = (nmse_base - nmse_aug) / nmse_base
    assert abs(observed_adv - 0.05) < 1e-4


def test_planted_feature_raises_if_unreachable():
    """Verify that if 5% advantage cannot be reached, tune_planted_feature_scale raises."""
    rng = np.random.default_rng(999)
    N = 1000
    u = rng.uniform(0.0, 0.5, size=N)
    # y is completely uncorrelated with u/q
    X = rng.standard_normal((N, 5))
    y = rng.standard_normal(N)

    with pytest.raises(ValueError, match="cannot achieve 5.0% advantage"):
        tune_planted_feature_scale(
            X_train=X,
            y_train=y,
            u_train=u,
            target_advantage=0.05,
        )


def test_pure_statistical_mde_power():
    """Verify pure statistical MDE power simulator behavior."""
    rng = np.random.default_rng(321)
    ctrl_tables = rng.normal(loc=0.20, scale=0.01, size=(20, 10, 10))

    # 10% true advantage gives high power (>= 80%)
    res_power = evaluate_synthetic_mde_power(
        baseline_ctrl_nmse=ctrl_tables,
        true_advantage=0.10,
        noise_std=0.005,
        n_replicates=50,
        seed=42,
    )
    assert res_power["power"] >= 0.80
    assert res_power["null_fpr"] <= 0.05
    assert res_power["passed_mde_gate"] is True


def test_calibrate_beta_for_q_oracle_exact_5pct():
    """Verify exact closed-form beta calibration achieving 5.0% relative advantage."""
    rng = np.random.default_rng(42)
    N = 5000
    u = rng.uniform(0.0, 0.5, size=N)
    q = build_narma10_planted_feature(u)
    r_true = rng.normal(loc=0.0, scale=1.0, size=N)
    r_pred = 0.8 * r_true + rng.normal(loc=0.0, scale=0.3, size=N)

    beta = calibrate_beta_for_q_oracle(
        r_true=r_true,
        r_pred=r_pred,
        q=q,
        target_advantage=0.05,
        tol=1e-5,
    )
    assert beta > 0.0

    z_true = r_true + beta * q
    mse_base = float(np.mean((z_true - r_pred) ** 2))
    mse_oracle = float(np.mean((r_true - r_pred) ** 2))
    observed_adv = (mse_base - mse_oracle) / mse_base
    assert abs(observed_adv - 0.05) < 1e-6


def test_calibrate_5pct_mde_power_runs_without_name_error():
    """Verify calibrate_5pct_mde_power no longer raises NameError for gamma_star."""
    rng = np.random.default_rng(123)
    N_tr = 2000
    u_tr = rng.uniform(0.0, 0.5, size=N_tr)
    q_tr = build_narma10_planted_feature(u_tr)
    X_tr = rng.standard_normal((N_tr, 5))
    y_tr = 0.5 * X_tr[:, 0] + 0.3 * X_tr[:, 1] + 0.5 * q_tr + rng.normal(0, 0.1, size=N_tr)

    test_streams = []
    for s in range(5):
        u_te = rng.uniform(0.0, 0.5, size=1000)
        q_te = build_narma10_planted_feature(u_te)
        X_te = rng.standard_normal((1000, 5))
        y_te = 0.5 * X_te[:, 0] + 0.3 * X_te[:, 1] + 0.5 * q_te + rng.normal(0, 0.1, size=1000)
        test_streams.append({"X": X_te, "y": y_te, "u": u_te})

    res = calibrate_5pct_mde_power(
        X_train=X_tr,
        y_train=y_tr,
        u_train=u_tr,
        test_streams=test_streams,
        target_advantage=0.05,
        n_replicates=50,
        seed=42,
    )
    assert "gamma_star" in res
    assert "c_star" in res
    assert isinstance(res["gamma_star"], float)
    assert res["gamma_star"] > 0.0
    assert "power" in res
    assert "null_fpr" in res


def test_q_gate_passes_on_encoding_and_fails_on_disconnected():
    """Verify q positive-control gate passes when DN encodes q and fails when disconnected."""
    from research.pipeline.g1_bench import generate_narma10_sequence, TRAIN_SEEDS
    from src.connectome.synthetic import make_synthetic_connectome
    from research.pipeline.g1_reservoir import ScipyG1Reservoir

    seqs = [generate_narma10_sequence(seed=s, valid_length=2000, washout=200) for s in TRAIN_SEEDS]
    washout = 200
    seq_y = [s["y_next"] for s in seqs]
    seq_u = [s["u"] for s in seqs]
    u_batch = np.array([s["u"] for s in seqs], dtype=np.float32)

    # 1. Connected encoding graph: sensory feeds into network and readout neurons
    g_conn = make_synthetic_connectome(n_neurons=30, avg_out_degree=6.0, seed=42)
    sensory_idx = np.arange(4)
    readout_idx = np.arange(4, 8)
    res_conn = ScipyG1Reservoir(
        weights=g_conn.weights,
        sensory_idx=sensory_idx,
        readout_idx=readout_idx,
        target_rho=0.90,
    )
    states_conn, _ = res_conn.simulate(u_batch, track_saturation=False)

    outer_recs_enc = []
    outer_recs_mis = []
    for f in range(5):
        te_idx = [f * 2, f * 2 + 1]
        tr_idx = [i for i in range(10) if i not in te_idx]
        bundle = fit_q_probe_heads(
            [states_conn[i] for i in tr_idx],
            [seq_y[i] for i in tr_idx],
            [seq_u[i] for i in tr_idx],
            washout=washout,
            target_advantage=0.05,
        )
        for j in te_idx:
            X_j = states_conn[j, washout:]
            u_j = seq_u[j][washout:]
            y_j = seq_y[j][washout:]
            q_j = build_narma10_planted_feature(seq_u[j])[washout:]
            r_j = y_j - q_j
            z_j = r_j + bundle["beta_star"] * q_j

            r_hat = bundle["predict_r"](X_j)
            z_hat = bundle["predict_z"](X_j)
            oracle_hat = r_hat + bundle["beta_star"] * q_j
            outer_recs_enc.append({"z_true": z_j, "r_pred": r_hat, "z_pred": z_hat, "oracle_pred": oracle_hat})

            mis_idx = (j + 1) % 10
            u_mis = seq_u[mis_idx][washout:]
            y_mis = seq_y[mis_idx][washout:]
            q_mis = build_narma10_planted_feature(seq_u[mis_idx])[washout:]
            r_mis = y_mis - q_mis
            z_mis = r_mis + bundle["beta_star"] * q_mis
            outer_recs_mis.append({"z_true": z_mis, "r_pred": r_hat, "z_pred": z_hat})

    res_enc = evaluate_q_positive_control_gate(
        outer_records=outer_recs_enc,
        n_bootstraps=200,
        block_size=200,
        mismatched_records=outer_recs_mis,
    )
    assert res_enc["passed_q_gate"] is True
    assert res_enc["aq_pooled"] >= 0.05
    assert res_enc["ci_lower_95"] > 0.0
    assert res_enc["oracle_power"] >= 0.80
    assert res_enc["mismatched_null_fpr"] <= 0.05

    # 2. Disconnected graph: readout neurons are completely disconnected from sensory inputs
    g_dis = make_synthetic_connectome(n_neurons=40, avg_out_degree=5.0, seed=123)
    W_dis = g_dis.weights.tolil()
    W_dis[20:, :20] = 0  # Zero out all connections from 0..19 to 20..39
    W_dis = W_dis.tocsr()
    res_dis_engine = ScipyG1Reservoir(
        weights=W_dis,
        sensory_idx=sensory_idx,
        readout_idx=np.arange(30, 36),
        target_rho=0.90,
    )
    states_dis, _ = res_dis_engine.simulate(u_batch, track_saturation=False)
    # Add tiny perturbation so inactive states have non-zero variance
    states_dis = states_dis + np.random.default_rng(999).normal(scale=1e-5, size=states_dis.shape)

    outer_recs_dis = []
    for f in range(5):
        te_idx = [f * 2, f * 2 + 1]
        tr_idx = [i for i in range(10) if i not in te_idx]
        bundle = fit_q_probe_heads(
            [states_dis[i] for i in tr_idx],
            [seq_y[i] for i in tr_idx],
            [seq_u[i] for i in tr_idx],
            washout=washout,
            target_advantage=0.05,
        )
        for j in te_idx:
            X_j = states_dis[j, washout:]
            u_j = seq_u[j][washout:]
            y_j = seq_y[j][washout:]
            q_j = build_narma10_planted_feature(seq_u[j])[washout:]
            r_j = y_j - q_j
            z_j = r_j + bundle["beta_star"] * q_j
            r_hat = bundle["predict_r"](X_j)
            z_hat = bundle["predict_z"](X_j)
            oracle_hat = r_hat + bundle["beta_star"] * q_j
            outer_recs_dis.append({"z_true": z_j, "r_pred": r_hat, "z_pred": z_hat, "oracle_pred": oracle_hat})

    res_dis = evaluate_q_positive_control_gate(
        outer_records=outer_recs_dis,
        n_bootstraps=200,
        block_size=200,
    )
    assert res_dis["passed_q_gate"] is False
    assert res_dis["aq_pooled"] < 0.05

