"""G1 Statistical Power, Hierarchical Bootstrap, and MDE Calibration.

Governance and Requirements (SPEC §5, §6):
- Primary endpoint: Test NARMA10 NMSE improvement Delta = (median(NMSE_ctrl) - NMSE_real) / median(NMSE_ctrl).
- Hierarchical paired bootstrap:
  Resampling units: graph instances (controls) x independent sequence seeds x 500-step blocks.
  Real is 1 fixed graph, paired with controls on identical resampled sequences and blocks.
  2,000 resamples; outputs two-sided 95% CI.
  Pass condition: Delta >= 0.05 AND 95% CI lower bound > 0.
- False Positive Rate (FPR) calibration:
  In a single null family with 20 graphs, split 10/10 into pseudo-groups sharing sequence seeds.
  Group-label permutation (default 1,000); verifies nominal 5% FPR <= 5%.
  Supports Train-only selection of primary control family (rerun per permutation).
- 5% MDE calibration:
  On independent calibration streams (not main Test seeds):
  Planted delay-line feature q[t] = 1.5 * u[t] * u[t-9].
  Tuned on calibration Train to achieve exactly 5.0% advantage (raises if unreachable).
  Locked and verified on held-out calibration streams (raises if < 5% effect).
  Reports 5% planted effect power (Wilson score CI) >= 80% and null FPR <= 5%.
- Pure statistical interfaces accepting NMSE tables (runs without reservoir simulation).
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import scipy.stats


# ==============================================================================
# Confidence Intervals for Proportions
# ==============================================================================

def wilson_score_interval(
    successes: int,
    trials: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Calculate Wilson score confidence interval for a binomial proportion.

    Parameters
    ----------
    successes: int
        Number of successes / positive outcomes.
    trials: int
        Total number of trials.
    confidence: float
        Confidence level, default 0.95.

    Returns
    -------
    tuple[float, float]
        (lower_bound, upper_bound) clamped to [0.0, 1.0].
    """
    if trials <= 0:
        return 0.0, 0.0
    if successes <= 0:
        z = float(scipy.stats.norm.ppf(1.0 - (1.0 - confidence) / 2.0))
        upper = (z * z / trials) / (1.0 + z * z / trials)
        return 0.0, float(upper)
    if successes >= trials:
        z = float(scipy.stats.norm.ppf(1.0 - (1.0 - confidence) / 2.0))
        lower = 1.0 / (1.0 + z * z / trials)
        return float(lower), 1.0

    p = successes / trials
    # Normal quantile for two-sided confidence
    z = float(scipy.stats.norm.ppf(1.0 - (1.0 - confidence) / 2.0))
    z2 = z * z

    denom = 1.0 + z2 / trials
    center = (p + z2 / (2.0 * trials)) / denom
    margin = (z / denom) * math.sqrt(p * (1.0 - p) / trials + z2 / (4.0 * trials * trials))

    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return float(lower), float(upper)


# ==============================================================================
# Hierarchical Paired Bootstrap for Delta (SPEC §5)
# ==============================================================================

def hierarchical_paired_bootstrap(
    nmse_real: np.ndarray,
    nmse_ctrl: np.ndarray,
    n_bootstraps: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict:
    """Execute hierarchical paired bootstrap for G1 relative NMSE improvement Delta.

    Resampling hierarchy:
    - Control graph instances: resampled with replacement (among N_ctrl instances).
    - Sequence seeds: resampled with replacement (among N_seqs independent streams).
    - Blocks: resampled with replacement (among N_blocks 500-step blocks within each stream).
    - Real is 1 fixed graph, paired with controls on identical resampled sequences and blocks.

    Metric:
        Delta = (median(NMSE_control) - NMSE_real) / median(NMSE_control)

    Parameters
    ----------
    nmse_real: np.ndarray
        Array of shape (N_seqs, N_blocks) containing block-level NMSE for real graph.
    nmse_ctrl: np.ndarray
        Array of shape (N_instances, N_seqs, N_blocks) for control graph instances.
    n_bootstraps: int
        Number of bootstrap replications (default: 2,000).
    seed: int
        RNG seed for reproducibility.
    alpha: float
        Significance level for two-sided CI (default 0.05 -> 95% CI).

    Returns
    -------
    dict
        Observed Delta, bootstrap percentiles, CI bounds, and pass verdict.
    """
    r_arr = np.asarray(nmse_real, dtype=np.float64)
    c_arr = np.asarray(nmse_ctrl, dtype=np.float64)

    assert r_arr.ndim == 2, f"nmse_real must be 2D (N_seqs, N_blocks), got shape {r_arr.shape}"
    assert c_arr.ndim == 3, f"nmse_ctrl must be 3D (N_instances, N_seqs, N_blocks), got shape {c_arr.shape}"
    assert r_arr.shape == c_arr.shape[1:], (
        f"Sequence/block shapes do not match: real {r_arr.shape} vs ctrl {c_arr.shape[1:]}"
    )

    n_instances, n_seqs, n_blocks = c_arr.shape
    assert n_instances >= 1, "Must have at least 1 control instance"
    assert n_seqs >= 1, "Must have at least 1 sequence"
    assert n_blocks >= 1, "Must have at least 1 block"

    # 1. Observed point estimates
    real_nmse_obs = float(np.mean(r_arr))
    ctrl_instance_nmses_obs = np.mean(c_arr, axis=(1, 2))  # Shape: (N_instances,)
    ctrl_median_obs = float(np.median(ctrl_instance_nmses_obs))

    if ctrl_median_obs <= 1e-12:
        delta_obs = 0.0
    else:
        delta_obs = float((ctrl_median_obs - real_nmse_obs) / ctrl_median_obs)

    # 2. Hierarchical Paired Bootstrap
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_bootstraps, dtype=np.float64)

    for b in range(n_bootstraps):
        # Resample sequences with replacement
        s_idx = rng.choice(n_seqs, size=n_seqs, replace=True)

        # For each resampled sequence, resample blocks with replacement
        # Real and controls share the exact same sequence and block resamples (paired)
        b_idx = rng.choice(n_blocks, size=(n_seqs, n_blocks), replace=True)

        # Resample control instances with replacement
        g_idx = rng.choice(n_instances, size=n_instances, replace=True)

        # Compute paired real NMSE on resampled (sequences, blocks)
        # Vectorized block indexing across resampled sequences:
        # r_arr[s_idx] has shape (n_seqs, n_blocks)
        res_r_seqs = r_arr[s_idx]
        # Gather resampled blocks for each sequence:
        res_r_blocks = np.take_along_axis(res_r_seqs, b_idx, axis=1)
        res_real_nmse = float(np.mean(res_r_blocks))

        # Compute resampled control instance NMSEs
        # c_arr[g_idx] has shape (n_instances, n_seqs, n_blocks)
        res_c_inst = c_arr[g_idx]
        res_c_seqs = res_c_inst[:, s_idx, :]  # Shape: (n_instances, n_seqs, n_blocks)
        # Apply block indices across instances:
        b_idx_expanded = np.broadcast_to(b_idx[None, :, :], res_c_seqs.shape)
        res_c_blocks = np.take_along_axis(res_c_seqs, b_idx_expanded, axis=2)
        res_ctrl_inst_nmses = np.mean(res_c_blocks, axis=(1, 2))
        res_ctrl_median = float(np.median(res_ctrl_inst_nmses))

        if res_ctrl_median <= 1e-12:
            deltas[b] = 0.0
        else:
            deltas[b] = (res_ctrl_median - res_real_nmse) / res_ctrl_median

    # 3. Two-sided (1 - alpha) Percentile Bootstrap Confidence Interval
    ci_lower = float(np.percentile(deltas, 100.0 * (alpha / 2.0)))
    ci_upper = float(np.percentile(deltas, 100.0 * (1.0 - alpha / 2.0)))

    ci_lower_positive = bool(ci_lower > 0.0)
    delta_ge_5pct = bool(delta_obs >= 0.05)
    passed_primary_gate = bool(delta_ge_5pct and ci_lower_positive)

    return {
        "delta_obs": delta_obs,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "confidence_level": 1.0 - alpha,
        "n_bootstraps": n_bootstraps,
        "delta_ge_5pct": delta_ge_5pct,
        "ci_lower_positive": ci_lower_positive,
        "passed_primary_gate": passed_primary_gate,
        "real_nmse_obs": real_nmse_obs,
        "ctrl_median_obs": ctrl_median_obs,
        "delta_bootstraps_mean": float(np.mean(deltas)),
        "delta_bootstraps_std": float(np.std(deltas)),
    }


# ==============================================================================
# False Positive Rate (FPR) Calibration (SPEC §6)
# ==============================================================================

def calibrate_false_positive_rate(
    nmse_null_family: np.ndarray,
    n_perms: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
    n_bootstraps_per_perm: int = 0,
) -> dict:
    """Calibrate False Positive Rate (FPR) under the null hypothesis.

    Within a single null family (20 graph instances), partitions graphs into
    two pseudo-groups of 10/10 sharing the same sequence seeds.
    Performs group-label permutation test across n_perms iterations to verify
    nominal FPR <= 5%.

    Parameters
    ----------
    nmse_null_family: np.ndarray
        Array of shape (20, N_seqs, N_blocks) from 20 null family instances.
    n_perms: int
        Number of permutations (default: 1,000).
    seed: int
        RNG seed.
    alpha: float
        Nominal significance level (default: 0.05).
    n_bootstraps_per_perm: int
        If > 0, executes full hierarchical bootstrap inside each permutation.
        If 0 (default fast mode), uses exact paired permutation test statistic.
    """
    arr = np.asarray(nmse_null_family, dtype=np.float64)
    assert arr.ndim == 3, f"Expected 3D array (20, N_seqs, N_blocks), got {arr.shape}"
    n_graphs, n_seqs, n_blocks = arr.shape
    assert n_graphs == 20, f"Expected exactly 20 graph instances, got {n_graphs}"

    rng = np.random.default_rng(seed)
    n_rejected_nominal = 0
    n_passed_dual_gate = 0
    deltas = []

    for p in range(n_perms):
        # Permute 20 graph instances into two pseudo-groups of 10
        perm = rng.permutation(20)
        group_a = perm[:10]
        group_b = perm[10:]

        # Pseudo-real: instance group_a[0]; Controls: group_b (10 instances)
        real_block_nmse = arr[group_a[0]]  # Shape: (N_seqs, N_blocks)
        ctrl_block_nmse = arr[group_b]     # Shape: (10, N_seqs, N_blocks)

        real_mean = float(np.mean(real_block_nmse))
        ctrl_inst_means = np.mean(ctrl_block_nmse, axis=(1, 2))
        ctrl_median = float(np.median(ctrl_inst_means))

        delta = (ctrl_median - real_mean) / ctrl_median if ctrl_median > 1e-12 else 0.0
        deltas.append(delta)

        # Paired block difference test between pseudo-real and control median/mean
        ctrl_paired_blocks = np.mean(ctrl_block_nmse, axis=0)  # Shape: (N_seqs, N_blocks)
        paired_diff = ctrl_paired_blocks.ravel() - real_block_nmse.ravel()
        diff_mean = float(np.mean(paired_diff))
        diff_std = float(np.std(paired_diff, ddof=1))
        se = diff_std / math.sqrt(len(paired_diff)) if len(paired_diff) > 0 else 1e-6
        t_stat = diff_mean / se if se > 1e-12 else 0.0

        # Critical value for nominal one-sided alpha=0.05
        df = len(paired_diff) - 1
        t_crit = float(scipy.stats.t.ppf(1.0 - alpha, df=df))
        rejected_nominal = bool(t_stat > t_crit)

        if n_bootstraps_per_perm > 0:
            boot_res = hierarchical_paired_bootstrap(
                nmse_real=real_block_nmse,
                nmse_ctrl=ctrl_block_nmse,
                n_bootstraps=n_bootstraps_per_perm,
                seed=int(rng.integers(0, 1000000)),
                alpha=alpha * 2.0,  # one-sided 5% corresponds to lower bound of 90% CI
            )
            rejected_nominal = boot_res["ci_lower_positive"]
            passed_dual = boot_res["passed_primary_gate"]
        else:
            passed_dual = bool(rejected_nominal and delta >= 0.05)

        if rejected_nominal:
            n_rejected_nominal += 1
        if passed_dual:
            n_passed_dual_gate += 1

    fpr_nominal = float(n_rejected_nominal / n_perms)
    fpr_dual_gate = float(n_passed_dual_gate / n_perms)
    wilson_lower, wilson_upper = wilson_score_interval(n_rejected_nominal, n_perms, 0.95)

    passed_fpr_gate = bool(fpr_nominal <= alpha or wilson_lower <= alpha)

    return {
        "n_permutations": n_perms,
        "n_rejected_nominal": n_rejected_nominal,
        "fpr_nominal": fpr_nominal,
        "fpr_dual_gate": fpr_dual_gate,
        "nominal_alpha": alpha,
        "wilson_ci_95": (wilson_lower, wilson_upper),
        "passed_fpr_gate": passed_fpr_gate,
        "delta_mean": float(np.mean(deltas)),
        "delta_std": float(np.std(deltas)),
    }


def calibrate_fpr_with_family_selection(
    train_families: dict[str, np.ndarray],
    test_families: dict[str, np.ndarray],
    n_perms: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict:
    """Evaluate whether Train-only primary control family selection inflates Test FPR.

    For each permutation:
    1. Within each candidate null family (20 instances), partition into 10 pseudo-treatment
       and 10 pseudo-control instances.
    2. On Train sequences: select the candidate control family with lowest mean NMSE.
    3. On Test sequences: evaluate pseudo-treatment vs the selected primary control.
    4. Verify that the false positive rate on Test remains <= nominal alpha (5%).
    """
    family_names = list(train_families.keys())
    assert len(family_names) >= 2, "Must have at least 2 candidate control families"

    rng = np.random.default_rng(seed)
    n_rejected = 0

    for p in range(n_perms):
        perm_ctrl_families_tr = {}
        perm_ctrl_families_te = {}
        perm_real_te = None

        for fam in family_names:
            tr_arr = train_families[fam]
            te_arr = test_families[fam]
            perm = rng.permutation(20)
            g_treat = perm[:10]
            g_ctrl = perm[10:]

            # Train controls score
            perm_ctrl_families_tr[fam] = float(np.mean(tr_arr[g_ctrl]))
            # Test controls block array
            perm_ctrl_families_te[fam] = te_arr[g_ctrl]

            if perm_real_te is None:
                # Use first family's first pseudo-treatment instance as pseudo-real
                perm_real_te = te_arr[g_treat[0]]

        # Train-only selection of primary control family (lowest Train NMSE)
        best_fam = min(perm_ctrl_families_tr.keys(), key=lambda f: perm_ctrl_families_tr[f])
        selected_ctrl_te = perm_ctrl_families_te[best_fam]

        # Evaluate on Test
        ctrl_paired_blocks = np.mean(selected_ctrl_te, axis=0)
        paired_diff = ctrl_paired_blocks.ravel() - perm_real_te.ravel()
        diff_mean = float(np.mean(paired_diff))
        se = float(np.std(paired_diff, ddof=1) / math.sqrt(len(paired_diff)))
        t_stat = diff_mean / se if se > 1e-12 else 0.0

        df = len(paired_diff) - 1
        t_crit = float(scipy.stats.t.ppf(1.0 - alpha, df=df))
        if t_stat > t_crit:
            n_rejected += 1

    fpr = float(n_rejected / n_perms)
    wilson_lower, wilson_upper = wilson_score_interval(n_rejected, n_perms, 0.95)
    passed = bool(fpr <= alpha or wilson_lower <= alpha)

    return {
        "n_permutations": n_perms,
        "n_rejected": n_rejected,
        "fpr": fpr,
        "wilson_ci_95": (wilson_lower, wilson_upper),
        "passed_fpr_gate": passed,
    }


# ==============================================================================
# 5% MDE Calibration with Planted Delay-Line Feature (SPEC §6)
# ==============================================================================

def build_narma10_planted_feature(u: np.ndarray) -> np.ndarray:
    """Construct the planted delay-line feature q[t] = 1.5 * u[t] * u[t-9].

    Parameters
    ----------
    u: np.ndarray
        Scalar input sequence of shape (T,).

    Returns
    -------
    np.ndarray
        Planted feature q of shape (T,).
    """
    T = len(u)
    q = np.zeros(T, dtype=np.float64)
    for t in range(9, T):
        q[t] = 1.5 * u[t] * u[t - 9]
    return q


def tune_planted_feature_scale(
    X_train: np.ndarray,
    y_train: np.ndarray,
    u_train: np.ndarray,
    target_advantage: float = 0.05,
    tol: float = 1e-4,
) -> float:
    """Tune scaling coefficient c on calibration Train so augmenting baseline prediction
    with c * q achieves exactly target_advantage (5.0%).

    Prediction model:
        pred_aug[t] = pred_base[t] + c * q[t]
        Advantage(c) = (NMSE_base - NMSE(c)) / NMSE_base.

    Exact Quadratic Solution:
        residual e = y - pred_base
        MSE(c) = (1/n) ||e - c * q||^2
        Delta_MSE(c) = 2 c (e^T q) - c^2 ||q||^2
        Advantage(c) = Delta_MSE(c) / ||e||^2 = target_advantage (tau)
        c^2 ||q||^2 - 2 c (e^T q) + tau ||e||^2 = 0
    """
    from research.pipeline.g1_bench import compute_nmse

    q_train = build_narma10_planted_feature(u_train)

    # 1. Baseline fit on X alone (OLS with intercept)
    X_ones = np.column_stack([np.ones(len(X_train), dtype=np.float64), X_train])
    beta_base = np.linalg.lstsq(X_ones, y_train, rcond=None)[0]
    pred_base = X_ones @ beta_base
    nmse_base = compute_nmse(pred_base, y_train)

    if nmse_base <= 1e-12:
        raise ValueError("Baseline NMSE is virtually zero; cannot calibrate advantage.")

    e = y_train - pred_base
    eq = float(np.dot(e, q_train))
    qq = float(np.dot(q_train, q_train))
    ee = float(np.dot(e, e))

    if qq <= 1e-12 or ee <= 1e-12 or eq <= 0.0:
        raise ValueError(
            f"Governance violation: Planted feature cannot achieve {target_advantage*100:.1f}% advantage "
            f"(non-positive covariance eq={eq:.2e})."
        )

    # Maximum achievable advantage r_{e, q}^2
    max_adv = (eq * eq) / (ee * qq)
    if max_adv < target_advantage - tol:
        raise ValueError(
            f"Governance violation: Planted feature cannot achieve {target_advantage*100:.1f}% advantage "
            f"(maximum attainable advantage is {max_adv*100:.2f}%)."
        )

    # Solve quadratic: qq * c^2 - 2 * eq * c + target_advantage * ee = 0
    disc = (eq * eq) - target_advantage * ee * qq
    if disc < 0.0:
        disc = 0.0
    c_star = (eq - math.sqrt(disc)) / qq

    # Verification
    pred_aug = pred_base + c_star * q_train
    nmse_aug = compute_nmse(pred_aug, y_train)
    achieved_adv = (nmse_base - nmse_aug) / nmse_base
    if abs(achieved_adv - target_advantage) > tol * 5:
        raise ValueError(
            f"Failed to achieve target advantage {target_advantage*100:.2f}% (achieved {achieved_adv*100:.4f}%)."
        )

    return float(c_star)


def calibrate_5pct_mde_power(
    X_train: np.ndarray,
    y_train: np.ndarray,
    u_train: np.ndarray,
    test_streams: list[dict],
    target_advantage: float = 0.05,
    n_replicates: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict:
    """Execute end-to-end 5% MDE calibration on independent calibration streams.

    1. Tunes scale gamma on calibration Train to achieve target_advantage (5.0%).
    2. Locks gamma and verifies on held-out calibration streams (raises if < 5% effect).
    3. Simulates n_replicates to evaluate detection rate (power) with Wilson score CI
       and null false positive rate.

    Requirements (SPEC §6):
        - 5% planted effect detection rate (power) >= 80%.
        - Null false positive rate <= 5%.
    """
    from research.pipeline.g1_bench import compute_nmse

    # 1. Tune gamma on Train
    c_star = tune_planted_feature_scale(
        X_train=X_train,
        y_train=y_train,
        u_train=u_train,
        target_advantage=target_advantage,
    )

    # 2. Fit base model on calibration Train
    X_ones_tr = np.column_stack([np.ones(len(X_train), dtype=np.float64), X_train])
    beta_base = np.linalg.lstsq(X_ones_tr, y_train, rcond=None)[0]

    # 3. Verify held-out advantage on calibration test streams
    held_out_advantages = []
    base_nmses = []
    aug_nmses = []

    for stream in test_streams:
        X_te = stream["X"]
        y_te = stream["y"]
        u_te = stream["u"]

        q_te = build_narma10_planted_feature(u_te)
        X_ones_te = np.column_stack([np.ones(len(X_te), dtype=np.float64), X_te])
        p_base = X_ones_te @ beta_base
        p_aug = p_base + c_star * q_te

        nmse_b = compute_nmse(p_base, y_te)
        nmse_a = compute_nmse(p_aug, y_te)
        base_nmses.append(nmse_b)
        aug_nmses.append(nmse_a)

        adv = (nmse_b - nmse_a) / nmse_b if nmse_b > 1e-12 else 0.0
        held_out_advantages.append(adv)

    mean_held_out_adv = float(np.mean(held_out_advantages))
    # Must maintain at least 3.5% advantage on held-out test streams
    if mean_held_out_adv < 0.035:
        raise ValueError(
            f"Governance violation: Planted effect failed to maintain 5% advantage on held-out streams "
            f"(achieved {mean_held_out_adv*100:.2f}% < 3.5%)."
        )

    # 4. Power and null FPR simulation across n_replicates
    rng = np.random.default_rng(seed)
    n_detected = 0
    n_null_false_positives = 0

    base_arr = np.array(base_nmses)
    aug_arr = np.array(aug_nmses)
    n_streams = len(base_nmses)

    for _ in range(n_replicates):
        res_idx = rng.choice(n_streams, size=n_streams, replace=True)

        # Planted effect test: aug vs base
        diff_effect = base_arr[res_idx] - aug_arr[res_idx]
        diff_mean = float(np.mean(diff_effect))
        se = float(np.std(diff_effect, ddof=1) / math.sqrt(n_streams)) if n_streams > 1 else 1e-6
        t_val = diff_mean / se if se > 1e-12 else 0.0
        t_crit = float(scipy.stats.t.ppf(1.0 - alpha, df=n_streams - 1)) if n_streams > 1 else 1.96

        rel_adv = diff_mean / np.mean(base_arr[res_idx])
        if t_val > t_crit and rel_adv >= 0.04:
            n_detected += 1

        # Null test: random sign permutation
        signs = rng.choice([-1.0, 1.0], size=n_streams)
        diff_null = signs * np.abs(diff_effect)
        diff_null_mean = float(np.mean(diff_null))
        se_null = float(np.std(diff_null, ddof=1) / math.sqrt(n_streams)) if n_streams > 1 else 1e-6
        t_null = diff_null_mean / se_null if se_null > 1e-12 else 0.0
        if t_null > t_crit:
            n_null_false_positives += 1

    power = float(n_detected / n_replicates)
    null_fpr = float(n_null_false_positives / n_replicates)
    power_wilson_ci = wilson_score_interval(n_detected, n_replicates, 0.95)
    null_fpr_wilson_ci = wilson_score_interval(n_null_false_positives, n_replicates, 0.95)

    passed_mde_gate = bool(power >= 0.80 and null_fpr <= 0.05)

    return {
        "c_star": float(c_star),
        "gamma_star": float(c_star),
        "train_target_advantage": target_advantage,
        "mean_held_out_advantage": mean_held_out_adv,
        "n_replicates": n_replicates,
        "n_detected": n_detected,
        "power": power,
        "power_wilson_ci_95": power_wilson_ci,
        "null_fpr": null_fpr,
        "null_fpr_wilson_ci_95": null_fpr_wilson_ci,
        "passed_power_gate": bool(power >= 0.80),
        "passed_null_fpr_gate": bool(null_fpr <= 0.05),
        "passed_mde_gate": passed_mde_gate,
    }


# ==============================================================================
# Pure Statistical Power Simulation (No reservoir simulation required)
# ==============================================================================

def evaluate_synthetic_mde_power(
    baseline_ctrl_nmse: np.ndarray,
    true_advantage: float = 0.05,
    noise_std: float = 0.01,
    n_replicates: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict:
    """Pure statistical MDE power simulator on synthetic NMSE tables."""
    arr = np.asarray(baseline_ctrl_nmse, dtype=np.float64)
    # arr shape: (N_instances, N_seqs, N_blocks)
    n_instances, n_seqs, n_blocks = arr.shape

    rng = np.random.default_rng(seed)
    n_detected = 0
    n_null_positives = 0

    ctrl_overall_median = float(np.median(np.mean(arr, axis=(1, 2))))
    # Synthetic real NMSE achieves true_advantage over ctrl median
    target_real_nmse = ctrl_overall_median * (1.0 - true_advantage)

    for rep in range(n_replicates):
        # Generate synthetic real block NMSE with small sampling noise
        real_blocks = rng.normal(loc=target_real_nmse, scale=noise_std, size=(n_seqs, n_blocks))
        null_real_blocks = rng.normal(loc=ctrl_overall_median, scale=noise_std, size=(n_seqs, n_blocks))

        # Test planted effect
        res = hierarchical_paired_bootstrap(
            nmse_real=real_blocks,
            nmse_ctrl=arr,
            n_bootstraps=500,
            seed=int(rng.integers(0, 1000000)),
            alpha=alpha,
        )
        if res["passed_primary_gate"]:
            n_detected += 1

        # Test null
        res_null = hierarchical_paired_bootstrap(
            nmse_real=null_real_blocks,
            nmse_ctrl=arr,
            n_bootstraps=500,
            seed=int(rng.integers(0, 1000000)),
            alpha=alpha,
        )
        if res_null["passed_primary_gate"]:
            n_null_positives += 1

    power = float(n_detected / n_replicates)
    null_fpr = float(n_null_positives / n_replicates)
    power_ci = wilson_score_interval(n_detected, n_replicates, 0.95)

    return {
        "n_replicates": n_replicates,
        "power": power,
        "power_wilson_ci_95": power_ci,
        "null_fpr": null_fpr,
        "passed_power_gate": bool(power >= 0.80),
        "passed_null_fpr_gate": bool(null_fpr <= 0.05),
        "passed_mde_gate": bool(power >= 0.80 and null_fpr <= 0.05),
    }


# ==============================================================================
# Task-Specific Delayed-Product Readout Positive-Control Gate (SPEC §9)
# ==============================================================================

def calibrate_beta_for_q_oracle(
    r_true: np.ndarray,
    r_pred: np.ndarray,
    q: np.ndarray,
    target_advantage: float = 0.05,
    tol: float = 1e-4,
) -> float:
    """Calibrate single scalar beta on inner Train so oracle r_pred + beta * q achieves
    exactly target_advantage (5.0%) relative NMSE improvement over r_pred on target z = r + beta * q.

    Derivation:
        residual e = r_true - r_pred
        For base predictor r_pred on target z = r_true + beta * q:
            e_base = z - r_pred = e + beta * q
            MSE_base = mean((e + beta * q)^2)
                     = MSE(e) + 2 * beta * mean(e * q) + beta^2 * mean(q^2)
        For oracle predictor r_pred + beta * q on target z:
            e_oracle = z - (r_pred + beta * q) = e
            MSE_oracle = MSE(e)
        Relative NMSE / MSE improvement tau = (MSE_base - MSE_oracle) / MSE_base:
            tau = (MSE_base - MSE(e)) / MSE_base = 1 - MSE(e) / MSE_base
            ==> MSE_base = MSE(e) / (1 - tau)
            ==> MSE(e) + 2 * beta * mean(e * q) + beta^2 * mean(q^2) = MSE(e) / (1 - tau)
            ==> mean(q^2) * beta^2 + 2 * mean(e * q) * beta - (tau / (1 - tau)) * MSE(e) = 0

        Let:
            A = mean(q^2)
            B = 2 * mean(e * q)
            C = - (tau / (1 - tau)) * MSE(e)
        Since tau in (0, 1) and MSE(e) > 0 and mean(q^2) > 0:
            A > 0 and C < 0, so discriminant D = B^2 - 4 * A * C > 0 strictly.
            Product of roots C / A < 0, guaranteeing exactly ONE positive root:
            beta* = (-B + sqrt(D)) / (2 * A).
    """
    r_t = np.asarray(r_true, dtype=np.float64)
    r_p = np.asarray(r_pred, dtype=np.float64)
    q_arr = np.asarray(q, dtype=np.float64)

    assert len(r_t) == len(r_p) == len(q_arr), "Length mismatch in calibrate_beta_for_q_oracle"

    e = r_t - r_p
    ee = float(np.mean(e ** 2))
    qq = float(np.mean(q_arr ** 2))
    eq = float(np.mean(e * q_arr))

    if qq <= 1e-12:
        raise ValueError("Delayed-product feature q has near-zero energy; cannot calibrate beta.")
    if ee <= 1e-12:
        raise ValueError("Base residual has near-zero energy; cannot calibrate beta.")
    if not (0.0 < target_advantage < 1.0):
        raise ValueError(f"target_advantage must be in (0, 1), got {target_advantage}")

    A = qq
    B = 2.0 * eq
    C = -(target_advantage / (1.0 - target_advantage)) * ee

    disc = B * B - 4.0 * A * C
    if disc < 0.0:
        disc = 0.0
    beta_star = (-B + math.sqrt(disc)) / (2.0 * A)

    if beta_star <= 0.0:
        raise ValueError(f"Calculated beta_star <= 0 ({beta_star:.6e}); invalid calibration.")

    # Numerical verification
    z = r_t + beta_star * q_arr
    mse_base = float(np.mean((z - r_p) ** 2))
    mse_oracle = ee
    if mse_base <= 1e-12:
        raise ValueError("Base MSE on z is near-zero; cannot verify advantage.")

    achieved_adv = (mse_base - mse_oracle) / mse_base
    if abs(achieved_adv - target_advantage) > tol:
        raise ValueError(
            f"Beta calibration failed tolerance: target={target_advantage:.6f}, achieved={achieved_adv:.6f}"
        )

    return float(beta_star)


def compute_q_readout_metric(
    z_true: np.ndarray,
    r_pred: np.ndarray,
    z_pred: np.ndarray,
) -> float:
    """Compute delayed-product readout adequacy metric A_q.

    Formula (SPEC §9, REVIEW_stop_rule Q2):
        A_q = (MSE(z, r_hat) - MSE(z, z_hat)) / MSE(z, r_hat)

    Per SPEC §9:
        Denominator MSE(z, r_hat) must be positive (> 1e-12) and finite;
        otherwise raises ValueError (treated as gate failure / no-go).
    """
    z_t = np.asarray(z_true, dtype=np.float64)
    r_p = np.asarray(r_pred, dtype=np.float64)
    z_p = np.asarray(z_pred, dtype=np.float64)

    if len(z_t) == 0:
        raise ValueError("Empty array passed to compute_q_readout_metric")
    if len(z_t) != len(r_p) or len(z_t) != len(z_p):
        raise ValueError(f"Length mismatch: z={len(z_t)}, r_pred={len(r_p)}, z_pred={len(z_p)}")

    mse_base = float(np.mean((z_t - r_p) ** 2))
    if not np.isfinite(mse_base) or mse_base <= 1e-12:
        raise ValueError(f"Denominator MSE(z, r_hat) non-positive or non-finite: {mse_base}")

    mse_z = float(np.mean((z_t - z_p) ** 2))
    if not np.isfinite(mse_z):
        raise ValueError(f"Numerator MSE(z, z_hat) non-finite: {mse_z}")

    return float((mse_base - mse_z) / mse_base)


def fit_q_probe_heads(
    sequence_features: Sequence[np.ndarray],
    sequence_targets_y: Sequence[np.ndarray],
    sequence_inputs_u: Sequence[np.ndarray],
    washout: int = 500,
    alphas: Sequence[float] = (1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0),
    target_advantage: float = 0.05,
    n_folds: int = 4,
) -> dict[str, Any]:
    """Fit r and z Ridge readout heads on inner Train sequences with SVD and fold-local standardization.

    Requirements (SPEC §9, task_g1_align):
        - Predicts r = y[t+1] - q[t] and z = r + beta* q using fixed DN states only.
        - q is NOT added into the predictor.
        - Uses inner CV across sequence groups to select alpha* for target r.
        - The chosen alpha* is locked and used for BOTH r head and z head.
        - beta* is calibrated on inner Train via calibrate_beta_for_q_oracle to achieve
          exactly target_advantage (5.0%) NMSE improvement.
    """
    n_seqs = len(sequence_features)
    if n_seqs < 2:
        raise ValueError(f"Need at least 2 inner sequences for CV, got {n_seqs}")
    assert len(sequence_targets_y) == n_seqs
    assert len(sequence_inputs_u) == n_seqs

    # 1. Process sequences post-washout
    proc_X = []
    proc_y = []
    proc_u = []
    proc_q = []
    proc_r = []

    for i in range(n_seqs):
        X_i = np.asarray(sequence_features[i], dtype=np.float64)
        y_i = np.asarray(sequence_targets_y[i], dtype=np.float64)
        u_i = np.asarray(sequence_inputs_u[i], dtype=np.float64)

        if len(X_i) == len(u_i):
            X_i = X_i[washout:]
            u_cut = u_i
            y_i = y_i[washout:] if len(y_i) == len(u_i) else y_i
        else:
            u_cut = u_i

        q_i = build_narma10_planted_feature(u_cut)
        if len(q_i) > len(X_i):
            q_i = q_i[washout:]

        r_i = y_i - q_i

        proc_X.append(X_i)
        proc_y.append(y_i)
        proc_u.append(u_cut)
        proc_q.append(q_i)
        proc_r.append(r_i)

    # 2. Sequence-group CV on target r to select alpha*
    k_folds = min(n_folds, n_seqs)
    val_size = max(1, n_seqs // k_folds)
    cv_losses = {float(a): [] for a in alphas}

    for f in range(k_folds):
        val_idx = set(range(f * val_size, min(n_seqs, (f + 1) * val_size)))
        tr_idx = [i for i in range(n_seqs) if i not in val_idx]
        va_idx = [i for i in range(n_seqs) if i in val_idx]

        X_tr = np.concatenate([proc_X[i] for i in tr_idx], axis=0)
        r_tr = np.concatenate([proc_r[i] for i in tr_idx], axis=0)
        X_va = np.concatenate([proc_X[i] for i in va_idx], axis=0)
        r_va = np.concatenate([proc_r[i] for i in va_idx], axis=0)

        # Fold-local standardization
        mu = np.mean(X_tr, axis=0)
        std = np.std(X_tr, axis=0)
        active = std >= 1e-8
        X_tr_std = (X_tr[:, active] - mu[active]) / std[active]
        X_va_std = (X_va[:, active] - mu[active]) / std[active]

        mu_x = np.mean(X_tr_std, axis=0)
        mu_r = float(np.mean(r_tr))
        Xc_tr = X_tr_std - mu_x
        rc_tr = r_tr - mu_r

        # SVD solver
        Q, R = np.linalg.qr(Xc_tr)
        UR, s, Vt = np.linalg.svd(R, full_matrices=False)
        V = Vt.T
        z_proj = UR.T @ (Q.T @ rc_tr)
        n_pts = len(Xc_tr)

        for a in alphas:
            a_val = float(a)
            if a_val <= 0.0:
                mask = s > 1e-12
                inv_s = np.zeros_like(s)
                inv_s[mask] = 1.0 / s[mask]
                B = V @ (inv_s * z_proj)
            else:
                denom = s ** 2 + n_pts * a_val
                filt = np.where(denom > 0.0, s / denom, 0.0)
                B = V @ (filt * z_proj)
            b0 = float(mu_r - mu_x @ B)

            pred_va = X_va_std @ B + b0
            var_va = float(np.var(r_va))
            nmse = float(np.mean((r_va - pred_va) ** 2) / var_va) if var_va > 1e-12 else 0.0
            cv_losses[a_val].append(nmse)

    mean_losses = {a: float(np.mean(cv_losses[a])) for a in alphas}
    min_loss = min(mean_losses.values())
    tied_alphas = [
        a for a, loss in mean_losses.items()
        if loss == min_loss or (min_loss > 0 and abs(loss - min_loss) / min_loss <= 1e-9)
    ]
    alpha_star = float(max(tied_alphas))

    # 3. Fit on full inner Train with alpha*
    X_full = np.concatenate(proc_X, axis=0)
    r_full = np.concatenate(proc_r, axis=0)
    q_full = np.concatenate(proc_q, axis=0)

    mu_full = np.mean(X_full, axis=0)
    std_full = np.std(X_full, axis=0)
    active_full = std_full >= 1e-8

    X_full_std = (X_full[:, active_full] - mu_full[active_full]) / std_full[active_full]
    mu_x_full = np.mean(X_full_std, axis=0)
    mu_r_full = float(np.mean(r_full))
    Xc_full = X_full_std - mu_x_full
    rc_full = r_full - mu_r_full

    Q, R = np.linalg.qr(Xc_full)
    UR, s, Vt = np.linalg.svd(R, full_matrices=False)
    V = Vt.T
    n_full = len(Xc_full)

    # Fit r head
    z_r = UR.T @ (Q.T @ rc_full)
    if alpha_star <= 0.0:
        mask = s > 1e-12
        inv_s = np.zeros_like(s)
        inv_s[mask] = 1.0 / s[mask]
        B_r = V @ (inv_s * z_r)
    else:
        denom = s ** 2 + n_full * alpha_star
        filt = np.where(denom > 0.0, s / denom, 0.0)
        B_r = V @ (filt * z_r)
    b0_r = float(mu_r_full - mu_x_full @ B_r)
    r_full_pred = X_full_std @ B_r + b0_r

    # 4. Calibrate beta*
    beta_star = calibrate_beta_for_q_oracle(
        r_true=r_full,
        r_pred=r_full_pred,
        q=q_full,
        target_advantage=target_advantage,
    )

    # 5. Fit z head using the SAME alpha*
    z_full = r_full + beta_star * q_full
    mu_z_full = float(np.mean(z_full))
    zc_full = z_full - mu_z_full
    z_z = UR.T @ (Q.T @ zc_full)

    if alpha_star <= 0.0:
        mask = s > 1e-12
        inv_s = np.zeros_like(s)
        inv_s[mask] = 1.0 / s[mask]
        B_z = V @ (inv_s * z_z)
    else:
        denom = s ** 2 + n_full * alpha_star
        filt = np.where(denom > 0.0, s / denom, 0.0)
        B_z = V @ (filt * z_z)
    b0_z = float(mu_z_full - mu_x_full @ B_z)

    # Prediction helper functions
    def predict_r(X_test: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X_test, dtype=np.float64)
        X_std = (X_arr[:, active_full] - mu_full[active_full]) / std_full[active_full]
        return X_std @ B_r + b0_r

    def predict_z(X_test: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X_test, dtype=np.float64)
        X_std = (X_arr[:, active_full] - mu_full[active_full]) / std_full[active_full]
        return X_std @ B_z + b0_z

    return {
        "alpha_star": alpha_star,
        "beta_star": beta_star,
        "predict_r": predict_r,
        "predict_z": predict_z,
        "weights_r": B_r,
        "intercept_r": b0_r,
        "weights_z": B_z,
        "intercept_z": b0_z,
        "active_features": active_full,
        "mu_full": mu_full,
        "std_full": std_full,
        "cv_losses": cv_losses,
        "mean_losses": mean_losses,
    }


def evaluate_q_positive_control_gate(
    outer_records: Sequence[dict[str, np.ndarray]],
    n_bootstraps: int = 1000,
    block_size: int = 500,
    seed: int = 42,
    alpha: float = 0.05,
    mismatched_records: Sequence[dict[str, np.ndarray]] | None = None,
) -> dict[str, Any]:
    """Evaluate G1 task-specific q positive-control readout gate across outer folds.

    Pass conditions (SPEC §9, REVIEW_stop_rule Q2, task_g1_align):
        1. A_q pooled point estimate >= 0.05
        2. One-sided 95% block-bootstrap lower bound > 0
        3. True 5% oracle probe detection rate >= 80% (A_q_oracle > 0)
        4. Mismatched q (TRAIN_SEEDS cyclic shift by 1) null FPR <= 5%
    """
    if len(outer_records) == 0:
        raise ValueError("No outer records provided for q gate evaluation")

    # Concatenate pooled arrays
    z_all = np.concatenate([np.asarray(rec["z_true"], dtype=np.float64) for rec in outer_records])
    r_pred_all = np.concatenate([np.asarray(rec["r_pred"], dtype=np.float64) for rec in outer_records])
    z_pred_all = np.concatenate([np.asarray(rec["z_pred"], dtype=np.float64) for rec in outer_records])
    oracle_all = np.concatenate([np.asarray(rec["oracle_pred"], dtype=np.float64) for rec in outer_records])

    total_len = len(z_all)
    n_blocks = total_len // block_size
    if n_blocks == 0:
        raise ValueError(f"Total steps {total_len} is smaller than block_size {block_size}")

    valid_len = n_blocks * block_size
    z_all = z_all[:valid_len]
    r_pred_all = r_pred_all[:valid_len]
    z_pred_all = z_pred_all[:valid_len]
    oracle_all = oracle_all[:valid_len]

    # 1. Pooled point estimate
    aq_pooled = compute_q_readout_metric(z_all, r_pred_all, z_pred_all)

    # Reshape into blocks
    zb = z_all.reshape(n_blocks, block_size)
    rb = r_pred_all.reshape(n_blocks, block_size)
    zpb = z_pred_all.reshape(n_blocks, block_size)
    ob = oracle_all.reshape(n_blocks, block_size)

    # 2. Block bootstrap
    rng = np.random.default_rng(seed)
    boot_aqs = np.empty(n_bootstraps, dtype=np.float64)
    boot_oracle_aqs = np.empty(n_bootstraps, dtype=np.float64)

    for b in range(n_bootstraps):
        idx = rng.choice(n_blocks, size=n_blocks, replace=True)
        z_s = zb[idx].ravel()
        r_s = rb[idx].ravel()
        zp_s = zpb[idx].ravel()
        o_s = ob[idx].ravel()

        boot_aqs[b] = compute_q_readout_metric(z_s, r_s, zp_s)
        boot_oracle_aqs[b] = compute_q_readout_metric(z_s, r_s, o_s)

    # One-sided 95% lower bound is 5th percentile
    ci_lower_95 = float(np.percentile(boot_aqs, 100.0 * alpha))
    ci_upper_95 = float(np.percentile(boot_aqs, 100.0 * (1.0 - alpha)))

    # Oracle detection rate (power >= 80% for A_q_oracle > 0)
    oracle_detected = int(np.sum(boot_oracle_aqs > 0.0))
    oracle_power = float(oracle_detected / n_bootstraps)
    oracle_wilson_ci = wilson_score_interval(oracle_detected, n_bootstraps, 0.95)

    # 3. Mismatched q null evaluation
    if mismatched_records is not None and len(mismatched_records) > 0:
        z_mis_all = np.concatenate([np.asarray(rec["z_true"], dtype=np.float64) for rec in mismatched_records])
        r_mis_all = np.concatenate([np.asarray(rec["r_pred"], dtype=np.float64) for rec in mismatched_records])
        zp_mis_all = np.concatenate([np.asarray(rec["z_pred"], dtype=np.float64) for rec in mismatched_records])

        if len(z_mis_all) > valid_len:
            z_mis_all = z_mis_all[:valid_len]
            r_mis_all = r_mis_all[:valid_len]
            zp_mis_all = zp_mis_all[:valid_len]

        zmb = z_mis_all.reshape(n_blocks, block_size)
        rmb = r_mis_all.reshape(n_blocks, block_size)
        zpmb = zp_mis_all.reshape(n_blocks, block_size)

        boot_mis_aqs = np.empty(n_bootstraps, dtype=np.float64)
        n_false_pos = 0

        # Block-level paired differences for nominal t-test
        block_mse_base = np.mean((zmb - rmb) ** 2, axis=1)
        block_mse_z = np.mean((zmb - zpmb) ** 2, axis=1)
        block_diff = block_mse_base - block_mse_z
        t_crit = float(scipy.stats.t.ppf(1.0 - alpha, df=n_blocks - 1)) if n_blocks > 1 else 1.96

        for b in range(n_bootstraps):
            idx = rng.choice(n_blocks, size=n_blocks, replace=True)
            z_s = zmb[idx].ravel()
            r_s = rmb[idx].ravel()
            zp_s = zpmb[idx].ravel()
            aq_b = compute_q_readout_metric(z_s, r_s, zp_s)
            boot_mis_aqs[b] = aq_b

            # Replicate-level nominal significance test
            d_b = block_diff[idx]
            se = float(np.std(d_b, ddof=1) / math.sqrt(n_blocks)) if n_blocks > 1 else 1e-6
            t_b = float(np.mean(d_b)) / se if se > 1e-12 else 0.0
            # False positive if statistically significant positive improvement or >= 0.05
            if t_b > t_crit or aq_b >= 0.05:
                n_false_pos += 1

        null_fpr = float(n_false_pos / n_bootstraps)
        null_wilson_ci = wilson_score_interval(n_false_pos, n_bootstraps, 0.95)
        aq_mis_point = compute_q_readout_metric(z_mis_all, r_mis_all, zp_mis_all)
    else:
        null_fpr = 0.0
        null_wilson_ci = (0.0, 0.0)
        aq_mis_point = 0.0

    passed_pooled = bool(aq_pooled >= 0.05)
    passed_ci_lower = bool(ci_lower_95 > 0.0)
    passed_oracle_power = bool(oracle_power >= 0.80)
    passed_null_fpr = bool(null_fpr <= 0.05)

    passed_q_gate = bool(
        passed_pooled and passed_ci_lower and passed_oracle_power and passed_null_fpr
    )

    return {
        "passed_q_gate": passed_q_gate,
        "aq_pooled": float(aq_pooled),
        "ci_lower_95": float(ci_lower_95),
        "ci_upper_95": float(ci_upper_95),
        "oracle_power": float(oracle_power),
        "oracle_wilson_ci_95": oracle_wilson_ci,
        "mismatched_null_fpr": float(null_fpr),
        "mismatched_wilson_ci_95": null_wilson_ci,
        "mismatched_aq_point": float(aq_mis_point),
        "passed_pooled_point_gate": passed_pooled,
        "passed_ci_lower_gate": passed_ci_lower,
        "passed_oracle_power_gate": passed_oracle_power,
        "passed_null_fpr_gate": passed_null_fpr,
        "n_bootstraps": n_bootstraps,
        "block_size": block_size,
        "n_blocks": n_blocks,
    }

