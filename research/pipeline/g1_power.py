"""G1 Statistical Power, Hierarchical/Cross Bootstrap, and MDE Calibration (SPEC v2 §5, §7).

Contract:
- §7.2 / A19 Null & Alternative Population Residual Scaling:
  m_real = mean(e_real^2), m_g = mean(e_ctrl[g]^2), M = median(m_1..m_20).
  e_null[real] = e_real * sqrt(M / m_real); controls unchanged.
  e_alt[real] = sqrt(0.95) * e_null[real]; controls unchanged.
  Population real MSE under null == M; under alt == 0.95*M (relative error <= 1e-12).
  Unscaled data cannot be labeled as null.
  Unequal control MSEs are preserved.
- §5 / A20 / A21 Graph x Sequence Cross Bootstrap:
  Cohorts: 1 fixed pseudo-real + 20 resampled controls + 10 resampled full sequences.
  All graphs share identical resampled sequence indices per replicate.
  2,000 bootstrap resamples on resampled full sequence indices and control identities.
  Real identity is fixed (never resampled).
  No within-sequence block resampling (preserves temporal autocorrelation).
  Primary endpoint Delta = (median(NMSE_ctrl) - NMSE_real) / median(NMSE_ctrl).
  Primary gate: Delta_obs >= 0.05 AND 95% CI lower bound > 0.
- §7.4 / A22 Exact Clopper-Pearson Monte Carlo Bounds:
  Fixed N=1,000 outer replicates, tail probability a=0.025.
  U_FPR = BetaQuantile(1-a; k0+1, N-k0) (k0=N -> 1.0).
  L_power = BetaQuantile(a; k1, N-k1+1) (k1=0 -> 0.0).
  Go condition: L_power >= 0.80 AND U_FPR <= 0.05.
  Full success rate (lower bound > 0 and Delta_obs >= 0.05) reported separately (never called power).
  No adaptive replicate addition.
- A18 Deprecation of Old Planted-Q Path:
  Calls to obsolete planted-q and block-level APIs explicitly raise RuntimeError.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import scipy.stats


# ==============================================================================
# 1. Proportional Confidence Intervals & Exact Clopper-Pearson Bounds (SPEC §7.4, A22)
# ==============================================================================

def wilson_score_interval(
    successes: int,
    trials: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Calculate Wilson score confidence interval for a binomial proportion."""
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
    z = float(scipy.stats.norm.ppf(1.0 - (1.0 - confidence) / 2.0))
    z2 = z * z

    denom = 1.0 + z2 / trials
    center = (p + z2 / (2.0 * trials)) / denom
    margin = (z / denom) * math.sqrt(p * (1.0 - p) / trials + z2 / (4.0 * trials * trials))

    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return float(lower), float(upper)


def exact_clopper_pearson_fpr_upper(
    k0: int,
    n: int = 1000,
    alpha: float = 0.025,
) -> float:
    """Calculate exact Clopper-Pearson one-sided upper bound for FPR (SPEC §7.4).

    U_FPR = BetaQuantile(1 - alpha; k0 + 1, n - k0).
    Boundary: k0 == n -> 1.0.
    """
    if n <= 0:
        raise ValueError(f"Number of trials n must be positive, got {n}")
    if k0 < 0 or k0 > n:
        raise ValueError(f"k0 must be in [0, {n}], got {k0}")
    if k0 == n:
        return 1.0
    return float(scipy.stats.beta.ppf(1.0 - alpha, k0 + 1, n - k0))


def exact_clopper_pearson_power_lower(
    k1: int,
    n: int = 1000,
    alpha: float = 0.025,
) -> float:
    """Calculate exact Clopper-Pearson one-sided lower bound for Power (SPEC §7.4).

    L_power = BetaQuantile(alpha; k1, n - k1 + 1).
    Boundary: k1 == 0 -> 0.0.
    """
    if n <= 0:
        raise ValueError(f"Number of trials n must be positive, got {n}")
    if k1 < 0 or k1 > n:
        raise ValueError(f"k1 must be in [0, {n}], got {k1}")
    if k1 == 0:
        return 0.0
    return float(scipy.stats.beta.ppf(alpha, k1, n - k1 + 1))


def evaluate_mc_bounds(
    n_fpr_events: int,
    n_power_events: int,
    n_full_success_events: int,
    n_replicates: int = 1000,
    alpha: float = 0.025,
) -> dict[str, Any]:
    """Evaluate exact Monte Carlo bounds against SPEC §7.4 go/no-go gates."""
    u_fpr = exact_clopper_pearson_fpr_upper(n_fpr_events, n_replicates, alpha=alpha)
    l_power = exact_clopper_pearson_power_lower(n_power_events, n_replicates, alpha=alpha)
    l_full = exact_clopper_pearson_power_lower(n_full_success_events, n_replicates, alpha=alpha)

    passed_fpr_gate = bool(u_fpr <= 0.05)
    passed_power_gate = bool(l_power >= 0.80)
    passed_mde_gate = bool(passed_fpr_gate and passed_power_gate)

    return {
        "n_replicates": n_replicates,
        "alpha_mc": alpha,
        "n_fpr_events": n_fpr_events,
        "fpr_point_estimate": float(n_fpr_events / n_replicates),
        "u_fpr_975": u_fpr,
        "passed_fpr_gate": passed_fpr_gate,
        "n_power_events": n_power_events,
        "power_point_estimate": float(n_power_events / n_replicates),
        "l_power_975": l_power,
        "passed_power_gate": passed_power_gate,
        "n_full_success_events": n_full_success_events,
        "full_success_rate": float(n_full_success_events / n_replicates),
        "full_success_lower_975": l_full,
        "passed_mde_gate": passed_mde_gate,
    }


# ==============================================================================
# 2. Residual Population Scaling for Null and 5% Alternative (SPEC §7.2, §7.3, A19)
# ==============================================================================

def scale_residuals_for_null_and_alt(
    e_real: np.ndarray,
    e_ctrl: np.ndarray,
    tol: float = 1e-12,
) -> dict[str, Any]:
    """Scale real residuals to construct exact empirical null and 5% alternative populations.

    Parameters
    ----------
    e_real: np.ndarray
        Residuals for pseudo-real graph, shape (n_seqs, n_steps) or (1, n_seqs, n_steps).
    e_ctrl: np.ndarray
        Residuals for 20 control graphs, shape (20, n_seqs, n_steps).
    tol: float
        Relative tolerance for verifying population MSE invariants (<= 1e-12).

    Returns
    -------
    dict[str, Any]
        Scaled residual tensors and scaling metadata.
    """
    e_real_arr = np.asarray(e_real, dtype=np.float64)
    e_ctrl_arr = np.asarray(e_ctrl, dtype=np.float64)

    if e_real_arr.ndim == 3 and e_real_arr.shape[0] == 1:
        e_real_arr = e_real_arr[0]
    if e_real_arr.ndim != 2:
        raise ValueError(f"e_real must have 2 dimensions (n_seqs, n_steps), got shape {e_real.shape}")
    if e_ctrl_arr.ndim != 3:
        raise ValueError(f"e_ctrl must have 3 dimensions (20, n_seqs, n_steps), got shape {e_ctrl.shape}")

    n_ctrl = e_ctrl_arr.shape[0]
    if n_ctrl != 20:
        raise ValueError(f"e_ctrl must contain exactly 20 control instances, got {n_ctrl}")
    if e_real_arr.shape != e_ctrl_arr.shape[1:]:
        raise ValueError(
            f"Shape mismatch: e_real {e_real_arr.shape} vs e_ctrl sequences {e_ctrl_arr.shape[1:]}"
        )

    if not np.all(np.isfinite(e_real_arr)):
        raise ValueError("e_real contains NaN or Inf values")
    if not np.all(np.isfinite(e_ctrl_arr)):
        raise ValueError("e_ctrl contains NaN or Inf values")

    # 1. Compute individual graph MSEs across all sequences and time steps
    m_real = float(np.mean(e_real_arr ** 2))
    if m_real <= 0.0:
        raise ValueError(f"m_real must be strictly positive, got {m_real}")

    ctrl_mses = np.array([float(np.mean(e_ctrl_arr[g] ** 2)) for g in range(n_ctrl)], dtype=np.float64)
    if np.any(ctrl_mses <= 0.0):
        raise ValueError("All control graph MSEs must be strictly positive")

    # 2. M = median(m_1..m_20). For even count 20, arithmetic mean of 10th and 11th values
    M = float(np.median(ctrl_mses))
    if M <= 0.0:
        raise ValueError(f"Control MSE median M must be strictly positive, got {M}")

    # 3. Compute scale factors
    null_scale = math.sqrt(M / m_real)
    alt_scale = math.sqrt(0.95) * null_scale

    # 4. Construct scaled real residuals
    e_null_real = e_real_arr * null_scale
    e_alt_real = e_real_arr * alt_scale
    e_null_ctrl = e_ctrl_arr
    e_alt_ctrl = e_ctrl_arr

    # 5. Rigorous mathematical invariant verifications (relative error <= 1e-12)
    pop_null_real_mse = float(np.mean(e_null_real ** 2))
    null_rel_err = abs(pop_null_real_mse - M) / M
    if null_rel_err > tol:
        raise ValueError(
            f"Null scaling invariant violated: pop MSE {pop_null_real_mse:.14f} vs M {M:.14f} "
            f"(rel err {null_rel_err:.2e} > {tol})"
        )

    pop_alt_real_mse = float(np.mean(e_alt_real ** 2))
    expected_alt_mse = 0.95 * M
    alt_rel_err = abs(pop_alt_real_mse - expected_alt_mse) / expected_alt_mse
    if alt_rel_err > tol:
        raise ValueError(
            f"Alt scaling invariant violated: pop MSE {pop_alt_real_mse:.14f} vs 0.95*M {expected_alt_mse:.14f} "
            f"(rel err {alt_rel_err:.2e} > {tol})"
        )

    return {
        "m_real": m_real,
        "ctrl_mses": ctrl_mses,
        "M": M,
        "null_scale": null_scale,
        "alt_scale": alt_scale,
        "null_relative_error": null_rel_err,
        "alt_relative_error": alt_rel_err,
        "e_null_real": e_null_real,
        "e_alt_real": e_alt_real,
        "e_null_ctrl": e_null_ctrl,
        "e_alt_ctrl": e_alt_ctrl,
    }


# ==============================================================================
# 3. Sequence Sufficient Statistics for Vectorized Bootstrapping (SPEC §5, §8, A21)
# ==============================================================================

@dataclass
class SequenceResidualStats:
    """Precomputed sequence-level sufficient statistics for fast exact bootstrapping."""
    sse_real: np.ndarray        # Shape: (n_seqs,)
    sse_ctrl: np.ndarray        # Shape: (n_ctrl, n_seqs)
    sum_y: np.ndarray           # Shape: (n_seqs,)
    sum_y2: np.ndarray          # Shape: (n_seqs,)
    n_steps: np.ndarray         # Shape: (n_seqs,)

    @property
    def n_seqs(self) -> int:
        return len(self.sse_real)

    @property
    def n_ctrl(self) -> int:
        return self.sse_ctrl.shape[0]


def build_sequence_stats(
    e_real: np.ndarray,
    e_ctrl: np.ndarray,
    y: np.ndarray,
) -> SequenceResidualStats:
    """Compute exact sequence-level sufficient statistics from raw residuals and targets."""
    e_r = np.asarray(e_real, dtype=np.float64)
    e_c = np.asarray(e_ctrl, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    if e_r.ndim == 3 and e_r.shape[0] == 1:
        e_r = e_r[0]
    if y_arr.ndim != 2:
        raise ValueError(f"y must have 2 dimensions (n_seqs, n_steps), got shape {y_arr.shape}")

    n_seqs = len(y_arr)
    sse_real = np.sum(e_r ** 2, axis=1)
    sse_ctrl = np.sum(e_c ** 2, axis=2)
    sum_y = np.sum(y_arr, axis=1)
    sum_y2 = np.sum(y_arr ** 2, axis=1)
    n_steps = np.array([len(row) for row in y_arr], dtype=np.int64)

    return SequenceResidualStats(
        sse_real=sse_real,
        sse_ctrl=sse_ctrl,
        sum_y=sum_y,
        sum_y2=sum_y2,
        n_steps=n_steps,
    )


def compute_nmse_and_delta(
    stats: SequenceResidualStats,
    seq_indices: np.ndarray,
    ctrl_indices: np.ndarray,
) -> tuple[float, float, float]:
    """Compute exact pooled NMSE_real, control median C, and Delta for given resampled indices.

    Returns
    -------
    tuple[float, float, float]
        (delta, c_median, nmse_real)
    """
    total_steps = float(np.sum(stats.n_steps[seq_indices]))
    total_y = float(np.sum(stats.sum_y[seq_indices]))
    total_y2 = float(np.sum(stats.sum_y2[seq_indices]))
    mean_y = total_y / total_steps
    sst = total_y2 - total_steps * (mean_y ** 2)

    if sst <= 0.0 or not np.isfinite(sst):
        raise ValueError(f"Invalid SST denominator: {sst}")

    sse_real = float(np.sum(stats.sse_real[seq_indices]))
    nmse_real = sse_real / sst

    # For each control in ctrl_indices, sum over resampled sequence indices
    ctrl_sse = np.sum(stats.sse_ctrl[ctrl_indices][:, seq_indices], axis=1)
    ctrl_nmses = ctrl_sse / sst
    c_median = float(np.median(ctrl_nmses))

    if c_median <= 0.0 or not np.isfinite(c_median):
        raise ValueError(f"Invalid control median NMSE denominator: {c_median}")

    delta = (c_median - nmse_real) / c_median
    return float(delta), float(c_median), float(nmse_real)


# ==============================================================================
# 4. Primary Graph x Sequence Cross Bootstrap (SPEC §5, A20, A21)
# ==============================================================================

def graph_sequence_cross_bootstrap(
    stats: SequenceResidualStats,
    n_bootstraps: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Execute primary graph x sequence cross bootstrap (2,000 resamples).

    Sampling contract (SPEC §5, A20, A21):
    - 10 full sequence templates: resampled with replacement (10 from 10).
      This index is SHARED across real and ALL 20 controls.
    - 20 control identities: resampled with replacement (20 from 20).
    - Real graph identity is FIXED (never resampled into an alternate real specimen).
    - No within-sequence block resampling.
    - Recomputes pooled SST, NMSE_real, NMSE_ctrl, C, and Delta on each resample.
    - Confidence interval: two-sided (1 - alpha) = 95% CI via linear percentile interpolation.
    """
    n_seqs = stats.n_seqs
    n_ctrl = stats.n_ctrl

    # 1. Point estimate on original cohort (identity indices)
    id_seqs = np.arange(n_seqs, dtype=np.int64)
    id_ctrls = np.arange(n_ctrl, dtype=np.int64)
    delta_obs, c_obs, nmse_real_obs = compute_nmse_and_delta(stats, id_seqs, id_ctrls)

    # 2. Vectorized 2,000 resamples
    rng = np.random.default_rng(seed)
    seq_resamples = rng.choice(n_seqs, size=(n_bootstraps, n_seqs), replace=True)
    ctrl_resamples = rng.choice(n_ctrl, size=(n_bootstraps, n_ctrl), replace=True)

    # Vectorized SST and real SSE computation
    tot_steps = np.sum(stats.n_steps[seq_resamples], axis=1).astype(np.float64)
    tot_y = np.sum(stats.sum_y[seq_resamples], axis=1)
    tot_y2 = np.sum(stats.sum_y2[seq_resamples], axis=1)
    mean_y = tot_y / tot_steps
    sst_b = tot_y2 - tot_steps * (mean_y ** 2)

    sse_real_b = np.sum(stats.sse_real[seq_resamples], axis=1)
    nmse_real_b = sse_real_b / sst_b

    # Vectorized control NMSE computation
    deltas = np.empty(n_bootstraps, dtype=np.float64)
    for b in range(n_bootstraps):
        s_idx = seq_resamples[b]
        c_idx = ctrl_resamples[b]
        # sse_ctrl: (20, 10), select rows c_idx and cols s_idx
        ctrl_sse = np.sum(stats.sse_ctrl[c_idx][:, s_idx], axis=1)
        ctrl_nmses = ctrl_sse / sst_b[b]
        c_med = float(np.median(ctrl_nmses))
        deltas[b] = (c_med - nmse_real_b[b]) / c_med

    # 3. Percentiles using linear method
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)
    ci_lower = float(np.percentile(deltas, lower_pct, method="linear"))
    ci_upper = float(np.percentile(deltas, upper_pct, method="linear"))

    passed_primary_gate = bool(delta_obs >= 0.05 and ci_lower > 0.0)
    ci_lower_gt_0 = bool(ci_lower > 0.0)

    return {
        "delta_obs": delta_obs,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "c_median": c_obs,
        "nmse_real": nmse_real_obs,
        "passed_primary_gate": passed_primary_gate,
        "ci_lower_gt_0": ci_lower_gt_0,
        "n_bootstraps": n_bootstraps,
        "alpha": alpha,
    }


# ==============================================================================
# 5. Monte Carlo 1,000-Cohort Calibration for Power & FPR (SPEC §7.4, A18, A20, A22)
# ==============================================================================

def calibrate_v2_power_and_fpr(
    e_real: np.ndarray,
    e_ctrl: np.ndarray,
    y: np.ndarray,
    n_replicates: int = 1000,
    n_bootstraps_per_cohort: int = 2000,
    base_seed: int = 50000,
    alpha_mc: float = 0.025,
    bootstrap_alpha: float = 0.05,
) -> dict[str, Any]:
    """Execute end-to-end Monte Carlo power and FPR calibration across 1,000 cohorts.

    Returns complete audit specification with null/alt scale, seeds, events, bounds,
    and cohort specifications (SPEC §7.4, A18, A22).
    """
    # 1. Perform exact §7.2 / §7.3 population residual scaling
    scale_info = scale_residuals_for_null_and_alt(e_real=e_real, e_ctrl=e_ctrl)
    e_null_real = scale_info["e_null_real"]
    e_alt_real = scale_info["e_alt_real"]
    e_ctrl_arr = scale_info["e_null_ctrl"]

    # 2. Precompute sequence sufficient statistics for both null and alternative
    stats_null = build_sequence_stats(e_null_real, e_ctrl_arr, y)
    stats_alt = build_sequence_stats(e_alt_real, e_ctrl_arr, y)

    n_seqs = stats_null.n_seqs
    n_ctrl = stats_null.n_ctrl

    n_fpr_events = 0
    n_power_events = 0
    n_full_success_events = 0

    replicate_seeds = []
    outer_rng = np.random.default_rng(base_seed)

    for rep in range(n_replicates):
        rep_seed = int(outer_rng.integers(0, 2**31 - 1))
        replicate_seeds.append(rep_seed)
        rep_rng = np.random.default_rng(rep_seed)

        # Draw cohort sequence and control templates (A20: 10 sequences shared, 20 controls)
        cohort_seqs = rep_rng.choice(n_seqs, size=n_seqs, replace=True)
        cohort_ctrls = rep_rng.choice(n_ctrl, size=n_ctrl, replace=True)

        inner_seed = int(rep_rng.integers(0, 2**31 - 1))

        # Build cohort sub-stats for NULL
        cohort_stats_null = SequenceResidualStats(
            sse_real=stats_null.sse_real[cohort_seqs],
            sse_ctrl=stats_null.sse_ctrl[cohort_ctrls][:, cohort_seqs],
            sum_y=stats_null.sum_y[cohort_seqs],
            sum_y2=stats_null.sum_y2[cohort_seqs],
            n_steps=stats_null.n_steps[cohort_seqs],
        )

        res_null = graph_sequence_cross_bootstrap(
            cohort_stats_null,
            n_bootstraps=n_bootstraps_per_cohort,
            seed=inner_seed,
            alpha=bootstrap_alpha,
        )
        if res_null["ci_lower_gt_0"]:
            n_fpr_events += 1

        # Build cohort sub-stats for ALT (shares exact same sequence and control templates!)
        cohort_stats_alt = SequenceResidualStats(
            sse_real=stats_alt.sse_real[cohort_seqs],
            sse_ctrl=stats_alt.sse_ctrl[cohort_ctrls][:, cohort_seqs],
            sum_y=stats_alt.sum_y[cohort_seqs],
            sum_y2=stats_alt.sum_y2[cohort_seqs],
            n_steps=stats_alt.n_steps[cohort_seqs],
        )

        res_alt = graph_sequence_cross_bootstrap(
            cohort_stats_alt,
            n_bootstraps=n_bootstraps_per_cohort,
            seed=inner_seed,
            alpha=bootstrap_alpha,
        )
        if res_alt["ci_lower_gt_0"]:
            n_power_events += 1
        if res_alt["passed_primary_gate"]:
            n_full_success_events += 1

    # 3. Exact Clopper-Pearson Bounds
    bounds = evaluate_mc_bounds(
        n_fpr_events=n_fpr_events,
        n_power_events=n_power_events,
        n_full_success_events=n_full_success_events,
        n_replicates=n_replicates,
        alpha=alpha_mc,
    )

    cohort_spec = {
        "n_pseudo_real": 1,
        "n_control_templates": n_ctrl,
        "n_sequence_templates": n_seqs,
        "sampling_rule": "1 fixed real + 20 resampled controls + 10 resampled full sequences with replacement",
        "pairing_rule": "All graphs share identical resampled sequence indices per replicate cohort",
    }

    return {
        "null_scale": scale_info["null_scale"],
        "alt_scale": scale_info["alt_scale"],
        "m_real": scale_info["m_real"],
        "M_median": scale_info["M"],
        "ctrl_mses": scale_info["ctrl_mses"].tolist(),
        "seeds": {
            "base_seed": base_seed,
            "n_replicates": n_replicates,
            "first_10_replicate_seeds": replicate_seeds[:10],
        },
        "events": {
            "n_fpr_events": n_fpr_events,
            "n_power_events": n_power_events,
            "n_full_success_events": n_full_success_events,
        },
        "bounds": bounds,
        "cohort_spec": cohort_spec,
        "passed_mde_gate": bounds["passed_mde_gate"],
    }


# ==============================================================================
# 6. Deprecated Old APIs with Explicit Raise (SPEC §1, §7, A18)
# ==============================================================================

def build_narma10_planted_feature(*args: Any, **kwargs: Any) -> Any:
    raise RuntimeError(
        "build_narma10_planted_feature is deprecated and withdrawn in G1 v2 per SPEC §1, §7. "
        "Planted-q path removed in favor of §7.2 empirical residual scaling."
    )


def tune_planted_feature_scale(*args: Any, **kwargs: Any) -> Any:
    raise RuntimeError(
        "tune_planted_feature_scale is deprecated and withdrawn in G1 v2 per SPEC §1, §7. "
        "Planted-q path removed in favor of §7.2 empirical residual scaling."
    )


def calibrate_5pct_mde_power(*args: Any, **kwargs: Any) -> Any:
    raise RuntimeError(
        "calibrate_5pct_mde_power is deprecated and withdrawn in G1 v2 per SPEC §1, §7. "
        "Use calibrate_v2_power_and_fpr with empirical residual scaling."
    )


def evaluate_synthetic_mde_power(*args: Any, **kwargs: Any) -> Any:
    raise RuntimeError(
        "evaluate_synthetic_mde_power is deprecated and withdrawn in G1 v2 per SPEC §1, §7. "
        "Independent Gaussian residuals prohibited under SPEC §7.2 / A20."
    )


def calibrate_beta_for_q_oracle(*args: Any, **kwargs: Any) -> Any:
    raise RuntimeError(
        "calibrate_beta_for_q_oracle is deprecated and withdrawn in G1 v2 per SPEC §1, §7."
    )


def compute_q_readout_metric(*args: Any, **kwargs: Any) -> Any:
    raise RuntimeError(
        "compute_q_readout_metric is deprecated and withdrawn in G1 v2 per SPEC §1, §7."
    )


def fit_q_probe_heads(*args: Any, **kwargs: Any) -> Any:
    raise RuntimeError(
        "fit_q_probe_heads is deprecated and withdrawn in G1 v2 per SPEC §1, §7."
    )


def evaluate_q_positive_control_gate(*args: Any, **kwargs: Any) -> Any:
    raise RuntimeError(
        "evaluate_q_positive_control_gate is deprecated and withdrawn in G1 v2 per SPEC §1, §7. "
        "Old q gate removed in favor of §6 direct ability diagnostics (v[t-9], v[t]v[t-9])."
    )


def calibrate_false_positive_rate(*args: Any, **kwargs: Any) -> Any:
    raise RuntimeError(
        "calibrate_false_positive_rate is deprecated and withdrawn in G1 v2 per SPEC §1, §7. "
        "Use calibrate_v2_power_and_fpr per SPEC §7.4."
    )


def calibrate_fpr_with_family_selection(*args: Any, **kwargs: Any) -> Any:
    raise RuntimeError(
        "calibrate_fpr_with_family_selection is deprecated and withdrawn in G1 v2 per SPEC §1, §7. "
        "Family selection removed; sole control is 20 scramble_mixed graphs."
    )


def hierarchical_paired_bootstrap(*args: Any, **kwargs: Any) -> Any:
    raise RuntimeError(
        "hierarchical_paired_bootstrap (block-level) is deprecated and withdrawn in G1 v2 per SPEC §1, §5. "
        "Use graph_sequence_cross_bootstrap with full sequences."
    )
