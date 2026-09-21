"""G1 v2 Specification Configuration, Diagnostic Targets, and Capability Gates.

Governing Specification: research/G1_SPEC_v2_draft.md (SPEC v2 §1, §3, §4, §5, §6, §10)
- Fixed settings: raw u input (no -0.25), rho=0.95, leak=0.5, noise=0, DN-only (1291 DNs),
  sole control family scramble_mixed (20 instances, 21 graphs total).
- Rejects retracted features: q gate, rho grid search, family selection, nested CV rho selector.
- Diagnostic targets: d1[t] = v[t-9] (linear delay), d2[t] = v[t]v[t-9] (pure product),
  where v[t] = u[t] - 0.25 (used only to define targets, never to center reservoir input).
- Diagnostic sequence bootstrap (2,000 complete sequence resamples, 95% two-sided percentile CI).
- Capability gate: Dual AND condition (R^2 >= 0.10 and CI lower > 0 for both d1 and d2).
- Diagnostic negative control: within-split cyclic shift by 1 sequence; constant baseline
  uses shifted fit mean; gate requires 95% CI lower bound of G <= 0.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Sequence

import numpy as np


# ==============================================================================
# G1 v2 Preregistered Constants (§1, §3, §4, §5)
# ==============================================================================

SPEC_VERSION: str = "v2"
TARGET_RHO: float = 0.95
LEAK: float = 0.5
NOISE_STD: float = 0.0
RECURRENCE_GAIN: float = 1.0

INPUT_MODE: str = "raw_u"
READOUT_MODE: str = "dn_only"
N_DN_NEURONS: int = 1291

CONTROL_FAMILY: str = "scramble_mixed"
N_CONTROL_INSTANCES: int = 20
N_REAL_INSTANCES: int = 1
TOTAL_GRAPHS: int = 21

ALPHA_GRID: tuple[float, ...] = (
    0.0,
    1e-8,
    1e-7,
    1e-6,
    1e-5,
    1e-4,
    1e-3,
    1e-2,
    1e-1,
    1.0,
    10.0,
    100.0,
)
UPPER_BOUND_ALPHA: float = 100.0

# Split lengths and sample sizes (§3.2)
DIAGNOSTIC_FIT_SEQS: int = 10
DIAGNOSTIC_FIT_VALID_LENGTH: int = 2000
DIAGNOSTIC_CHECK_SEQS: int = 10
DIAGNOSTIC_CHECK_VALID_LENGTH: int = 2000
DIAGNOSTIC_WASHOUT: int = 500

MAIN_TRAIN_SEQS: int = 10
MAIN_TRAIN_VALID_LENGTH: int = 5000
MAIN_VAL_SEQS: int = 3
MAIN_VAL_VALID_LENGTH: int = 2000
CALIBRATION_SEQS: int = 10
CALIBRATION_VALID_LENGTH: int = 5000
MAIN_TEST_SEQS: int = 10
MAIN_TEST_VALID_LENGTH: int = 5000
MAIN_WASHOUT: int = 500

# Fixture seed namespaces (§3.2: Stage A must NOT use frozen Main seeds)
FIXTURE_FIT_SEEDS: tuple[int, ...] = (10001, 10002, 10003, 10004, 10005, 10006, 10007, 10008, 10009, 10010)
FIXTURE_CHECK_SEEDS: tuple[int, ...] = (20001, 20002, 20003, 20004, 20005, 20006, 20007, 20008, 20009, 20010)


# ==============================================================================
# Configuration Validation & Rejection of Retracted v1 Features (A01)
# ==============================================================================

def validate_v2_config(
    target_rho: float = TARGET_RHO,
    candidate_rhos: Sequence[float] | None = None,
    control_families: Sequence[str] | None = None,
    q_gate_enabled: bool = False,
    nested_cv_enabled: bool = False,
    input_centering: bool = False,
    noise_std: float = NOISE_STD,
    readout_mode: str = READOUT_MODE,
) -> dict[str, Any]:
    """Validate G1 v2 configuration and strictly reject retracted v1 mechanics.

    Per G1_SPEC_v2 §1, §10 (A01):
        - Fixed raw u, rho=0.95, leak=0.5, noise=0, DN-only, scramble_mixed only.
        - Attempts to override candidate rho grid, enable q gate, use family selection,
          use nested CV rho selector, center input, or add noise must raise ValueError.
    """
    if q_gate_enabled:
        raise ValueError("G1 v2 strictly rejects q gate (retracted in SPEC v2 §1).")

    if nested_cv_enabled:
        raise ValueError("G1 v2 strictly rejects nested CV rho selector (retracted in SPEC v2 §1).")

    if candidate_rhos is not None:
        rhos_tuple = tuple(float(r) for r in candidate_rhos)
        if rhos_tuple != (TARGET_RHO,):
            raise ValueError(
                f"G1 v2 fixes target_rho={TARGET_RHO} and strictly rejects rho grid search. Got: {rhos_tuple}"
            )

    if abs(float(target_rho) - TARGET_RHO) > 1e-9:
        raise ValueError(f"G1 v2 requires target_rho={TARGET_RHO}, got {target_rho}")

    if control_families is not None:
        fams = tuple(str(f) for f in control_families)
        if fams != (CONTROL_FAMILY,):
            raise ValueError(
                f"G1 v2 fixes sole control family to '{CONTROL_FAMILY}' (20 instances) and rejects family selection. Got: {fams}"
            )

    if input_centering:
        raise ValueError("G1 v2 uses production raw u input and strictly rejects input centering.")

    if noise_std != 0.0:
        raise ValueError(f"G1 v2 requires noise_std=0.0, got {noise_std}")

    if readout_mode != "dn_only":
        raise ValueError(f"G1 v2 strictly requires DN-only readout, got {readout_mode}")

    return {
        "spec_version": SPEC_VERSION,
        "target_rho": TARGET_RHO,
        "leak": LEAK,
        "noise_std": NOISE_STD,
        "recurrence_gain": RECURRENCE_GAIN,
        "input_mode": INPUT_MODE,
        "readout_mode": READOUT_MODE,
        "control_family": CONTROL_FAMILY,
        "n_control_instances": N_CONTROL_INSTANCES,
        "total_graphs": TOTAL_GRAPHS,
    }


def validate_split_request(split: str, authorized: bool = False) -> None:
    """Enforce sealed Test boundary (§3.2, §8).

    Access to Main-Test sequences before code freeze and gate verification
    strictly raises PermissionError.
    """
    s_clean = str(split).strip().lower()
    if s_clean in ("main_test", "test") and not authorized:
        raise PermissionError(
            "Access to Main-Test is strictly forbidden before code freeze, "
            "simulator integrity, and inference calibration are completed."
        )


# ==============================================================================
# Diagnostic Targets: d1 (Linear Lag-9) and d2 (Pure Product) (A06)
# ==============================================================================

def build_diagnostic_targets(
    u: np.ndarray,
    u_negative: np.ndarray | None = None,
    washout: int = DIAGNOSTIC_WASHOUT,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute G1 v2 diagnostic targets d1[t] = v[t-9] and d2[t] = v[t] * v[t-9].

    Where:
        v[t] = u[t] - 0.25 (used strictly to define targets, never to center reservoir input).
        For negative history t < 0:
            Uses u_negative array of length 9 for u[-9..-1] if provided,
            otherwise zeroes (u[k]=0 -> v[k]=-0.25).
            Explicit index mapping prevents negative indexing into the end of arrays.

    Returns:
        (d1_valid, d2_valid) sliced after washout.
    """
    u_arr = np.asarray(u, dtype=np.float64)
    T = len(u_arr)
    if T <= washout:
        raise ValueError(f"Sequence length {T} must be strictly greater than washout {washout}")

    v = u_arr - 0.25

    if u_negative is not None:
        u_neg = np.asarray(u_negative, dtype=np.float64)
        if len(u_neg) != 9:
            raise ValueError(f"Expected u_negative of length 9 for u[-9..-1], got {len(u_neg)}")
        v_neg = u_neg - 0.25
    else:
        v_neg = np.full(9, -0.25, dtype=np.float64)

    d1 = np.empty(T, dtype=np.float64)
    d2 = np.empty(T, dtype=np.float64)

    for t in range(T):
        if t >= 9:
            v_lag = v[t - 9]
        else:
            # t = 0 -> lag 9 is index 0 of v_neg; t = 8 -> lag 1 is index 8 of v_neg
            v_lag = v_neg[t]

        d1[t] = v_lag
        d2[t] = v[t] * v_lag

    return d1[washout:], d2[washout:]


def distinguish_pure_product_from_raw_q(
    u: np.ndarray,
    washout: int = DIAGNOSTIC_WASHOUT,
) -> dict[str, Any]:
    """Demonstrate mathematical and numerical distinction between d2 and raw q (A06).

    d2[t] = v[t] * v[t-9] = (u[t] - 0.25) * (u[t-9] - 0.25)
          = u[t]*u[t-9] - 0.25*u[t] - 0.25*u[t-9] + 0.0625

    raw q[t] = 1.5 * u[t] * u[t-9]
             = 1.5 * (v[t] + 0.25) * (v[t-9] + 0.25)
             = 1.5 * v[t]*v[t-9] + 0.375*v[t] + 0.375*v[t-9] + 0.09375

    raw q contains linear delay and input terms as well as a constant offset,
    making it impossible to cleanly isolate pure non-linear multiplication capacity.
    """
    u_arr = np.asarray(u, dtype=np.float64)
    T = len(u_arr)
    v = u_arr - 0.25

    d2_full = np.empty(T, dtype=np.float64)
    q_full = np.empty(T, dtype=np.float64)

    for t in range(T):
        u_lag = u_arr[t - 9] if t >= 9 else 0.0
        v_lag = v[t - 9] if t >= 9 else -0.25
        d2_full[t] = v[t] * v_lag
        q_full[t] = 1.5 * u_arr[t] * u_lag

    d2_valid = d2_full[washout:]
    q_valid = q_full[washout:]

    abs_diff = np.abs(q_valid - d2_valid)
    max_diff = float(np.max(abs_diff))
    mean_diff = float(np.mean(abs_diff))

    # Correlation between raw q and linear terms
    corr_q_v = float(np.corrcoef(q_valid, v[washout:])[0, 1])

    return {
        "max_absolute_difference": max_diff,
        "mean_absolute_difference": mean_diff,
        "is_distinct": bool(max_diff > 0.01),
        "correlation_q_with_linear_v": corr_q_v,
    }


# ==============================================================================
# Pooled Metrics, Even Median, and Primary Delta (A11)
# ==============================================================================

def compute_pooled_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute pooled R^2 = 1 - SSE / SST with population variance (ddof=0).

    Per G1_SPEC_v2 §5, §6, §10 (A11):
        - Retains negative R^2 values (never clipped to zero).
        - Zero denominator (SST <= 1e-12), NaN, Inf, or empty arrays raise ValueError.
    """
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)

    if len(y_t) == 0:
        raise ValueError("Empty array passed to compute_pooled_r2")
    if len(y_t) != len(y_p):
        raise ValueError(f"Length mismatch in compute_pooled_r2: true={len(y_t)}, pred={len(y_p)}")
    if not np.all(np.isfinite(y_t)) or not np.all(np.isfinite(y_p)):
        raise ValueError("Non-finite values encountered in compute_pooled_r2")

    mu_y = float(np.mean(y_t))
    sst = float(np.sum((y_t - mu_y) ** 2))
    if sst <= 1e-12:
        raise ValueError(f"SST is non-positive or near zero ({sst:.2e}); cannot compute R^2")

    sse = float(np.sum((y_t - y_p) ** 2))
    if not np.isfinite(sse):
        raise ValueError("Non-finite SSE in compute_pooled_r2")

    return float(1.0 - (sse / sst))


def compute_pooled_nmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute pooled NMSE = MSE / Var(y) with ddof=0 (§5, A11).

    Fails closed on non-positive variance, NaN, Inf, or empty arrays.
    """
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)

    if len(y_t) == 0:
        raise ValueError("Empty array passed to compute_pooled_nmse")
    if len(y_t) != len(y_p):
        raise ValueError(f"Length mismatch: true={len(y_t)}, pred={len(y_p)}")
    if not np.all(np.isfinite(y_t)) or not np.all(np.isfinite(y_p)):
        raise ValueError("Non-finite values encountered in compute_pooled_nmse")

    var_y = float(np.var(y_t, ddof=0))
    if var_y <= 1e-12:
        raise ValueError(f"Target variance non-positive ({var_y:.2e}); cannot compute NMSE")

    mse = float(np.mean((y_p - y_t) ** 2))
    return float(mse / var_y)


def compute_even_median(values: Sequence[float]) -> float:
    """Compute median, taking arithmetic mean of the two middle values for even lengths (§5, A11).

    Fails closed on empty sequence or non-finite values.
    """
    arr = [float(v) for v in values]
    if len(arr) == 0:
        raise ValueError("Empty sequence passed to compute_even_median")
    if not all(math.isfinite(v) for v in arr):
        raise ValueError("Non-finite values encountered in compute_even_median")

    arr.sort()
    n = len(arr)
    mid = n // 2
    if n % 2 == 1:
        return arr[mid]
    return float((arr[mid - 1] + arr[mid]) / 2.0)


def compute_delta_primary(real_nmse: float, control_nmses: Sequence[float]) -> float:
    """Compute primary improvement delta = (C - NMSE_real) / C (§5, A11).

    Where C = median(NMSE_control_1 .. NMSE_control_20) using even median rule.
    Fails closed if C <= 1e-12 or any value is non-finite.
    """
    r_nmse = float(real_nmse)
    if not math.isfinite(r_nmse):
        raise ValueError(f"Non-finite real NMSE: {r_nmse}")

    C = compute_even_median(control_nmses)
    if C <= 1e-12:
        raise ValueError(f"Control median NMSE non-positive ({C:.2e}); cannot compute Delta")

    return float((C - r_nmse) / C)


# ==============================================================================
# Diagnostic Sequence Bootstrap & Capability Gate (A12)
# ==============================================================================

def compute_diagnostic_sequence_bootstrap(
    check_targets: Sequence[np.ndarray],
    check_predictions: Sequence[np.ndarray],
    n_bootstraps: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Perform complete check-sequence bootstrap for diagnostic R^2 (§6, A12).

    Requirements:
        - Resamples complete check sequences with replacement (10 out of 10).
        - Computes two-sided (1 - alpha) percentile CI (2.5% and 97.5% with linear interpolation).
        - Retains full bootstrap distribution.
        - Fails closed on constant states or zero SST.
    """
    n_seqs = len(check_targets)
    if n_seqs != len(check_predictions):
        raise ValueError(f"Mismatch in sequence count: targets={n_seqs}, preds={len(check_predictions)}")
    if n_seqs < 2:
        raise ValueError(f"Need at least 2 sequences for bootstrap, got {n_seqs}")

    # Check for constant predictions across all sequences (loss-of-signal check)
    all_preds = np.concatenate(check_predictions)
    if float(np.std(all_preds)) < 1e-8:
        # Constant predictions must fail closed
        pred_val = float(np.mean(all_preds))
        all_y = np.concatenate(check_targets)
        r2_point = compute_pooled_r2(all_y, all_preds)
        return {
            "r2_point": r2_point,
            "ci_lower": r2_point,
            "ci_upper": r2_point,
            "confidence_level": 1.0 - alpha,
            "n_bootstraps": n_bootstraps,
            "is_constant_prediction": True,
            "bootstraps": np.full(n_bootstraps, r2_point),
        }

    # Point estimate on all check data
    y_full = np.concatenate(check_targets)
    p_full = np.concatenate(check_predictions)
    r2_point = compute_pooled_r2(y_full, p_full)

    rng = np.random.default_rng(seed)
    boot_r2s = np.empty(n_bootstraps, dtype=np.float64)

    for b in range(n_bootstraps):
        seq_idx = rng.choice(n_seqs, size=n_seqs, replace=True)
        y_b = np.concatenate([check_targets[i] for i in seq_idx])
        p_b = np.concatenate([check_predictions[i] for i in seq_idx])
        boot_r2s[b] = compute_pooled_r2(y_b, p_b)

    ci_lower = float(np.percentile(boot_r2s, 100.0 * (alpha / 2.0), method="linear"))
    ci_upper = float(np.percentile(boot_r2s, 100.0 * (1.0 - alpha / 2.0), method="linear"))

    return {
        "r2_point": r2_point,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "confidence_level": 1.0 - alpha,
        "n_bootstraps": n_bootstraps,
        "is_constant_prediction": False,
        "bootstraps": boot_r2s,
    }


def evaluate_capability_gate(
    r2_d1: float,
    ci_lower_d1: float,
    r2_d2: float,
    ci_lower_d2: float,
    min_r2: float = 0.10,
) -> dict[str, Any]:
    """Evaluate dual AND capability gate on Diagnostic-check (§6, A12).

    Pass condition:
        Both d1 (linear delay) AND d2 (pure product) must satisfy:
        1. Point estimate R^2 >= min_r2 (0.10)
        2. Two-sided 95% sequence bootstrap CI lower bound > 0.0
    """
    passed_d1 = bool(r2_d1 >= min_r2 and ci_lower_d1 > 0.0)
    passed_d2 = bool(r2_d2 >= min_r2 and ci_lower_d2 > 0.0)
    passed = bool(passed_d1 and passed_d2)

    return {
        "passed": passed,
        "passed_d1": passed_d1,
        "passed_d2": passed_d2,
        "d1": {"r2_point": float(r2_d1), "ci_lower": float(ci_lower_d1), "min_r2": float(min_r2)},
        "d2": {"r2_d2": float(r2_d2), "ci_lower": float(ci_lower_d2), "min_r2": float(min_r2)},
    }


# ==============================================================================
# Diagnostic Negative Control (A13)
# ==============================================================================

def evaluate_diagnostic_negative_control(
    fit_states: Sequence[np.ndarray],
    fit_targets_d1: Sequence[np.ndarray],
    fit_targets_d2: Sequence[np.ndarray],
    check_states: Sequence[np.ndarray],
    check_targets_d1: Sequence[np.ndarray],
    check_targets_d2: Sequence[np.ndarray],
    ridge_factory: Callable[[], Any],
    n_bootstraps: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Evaluate diagnostic negative control via within-split sequence cyclic shift by 1 (§6, A13).

    Rules:
        - Fit and check are independently circularly shifted by 1 sequence within their own fixed order:
          sequence s DN states paired with sequence (s + 1) mod S target.
        - Two splits never cross-pair.
        - Ridge model re-runs 5-fold sequence CV and refits on shifted fit data.
        - Constant baseline uses the mean of the shifted fit target.
        - Improvement metric: G = (MSE_constant - MSE_model) / MSE_constant.
        - Evaluates two-sided 95% CI on G using complete check sequence bootstrap.
        - Gate: If ANY G's 95% CI lower bound > 0.0 -> FAIL (no-go).
    """
    n_fit = len(fit_states)
    n_check = len(check_states)
    assert n_fit == len(fit_targets_d1) == len(fit_targets_d2)
    assert n_check == len(check_targets_d1) == len(check_targets_d2)

    # Within-split cyclic shift: sequence s states paired with target (s + 1) % N
    fit_t1_mis = [fit_targets_d1[(s + 1) % n_fit] for s in range(n_fit)]
    fit_t2_mis = [fit_targets_d2[(s + 1) % n_fit] for s in range(n_fit)]
    check_t1_mis = [check_targets_d1[(s + 1) % n_check] for s in range(n_check)]
    check_t2_mis = [check_targets_d2[(s + 1) % n_check] for s in range(n_check)]

    results_by_target = {}
    any_failed = False

    for name, f_targets, c_targets in [
        ("d1", fit_t1_mis, check_t1_mis),
        ("d2", fit_t2_mis, check_t2_mis),
    ]:
        # Refit Ridge on shifted fit data
        ridge = ridge_factory()
        ridge.fit_sequence_group_cv(list(fit_states), list(f_targets))

        # Predict on shifted check states
        c_preds = [ridge.predict(cs) for cs in check_states]

        # Constant baseline from shifted fit target mean
        y_fit_full = np.concatenate(f_targets)
        c_base = float(np.mean(y_fit_full))

        # Check sequence bootstrap for metric G
        rng = np.random.default_rng(seed)
        boot_Gs = np.empty(n_bootstraps, dtype=np.float64)

        for b in range(n_bootstraps):
            seq_idx = rng.choice(n_check, size=n_check, replace=True)
            y_b = np.concatenate([c_targets[i] for i in seq_idx])
            p_b = np.concatenate([c_preds[i] for i in seq_idx])

            mse_const = float(np.mean((y_b - c_base) ** 2))
            mse_mod = float(np.mean((y_b - p_b) ** 2))
            boot_Gs[b] = (mse_const - mse_mod) / mse_const if mse_const > 1e-12 else 0.0

        y_check_all = np.concatenate(c_targets)
        p_check_all = np.concatenate(c_preds)
        mse_const_pt = float(np.mean((y_check_all - c_base) ** 2))
        mse_mod_pt = float(np.mean((y_check_all - p_check_all) ** 2))
        G_point = float((mse_const_pt - mse_mod_pt) / mse_const_pt) if mse_const_pt > 1e-12 else 0.0

        ci_lower = float(np.percentile(boot_Gs, 100.0 * (alpha / 2.0), method="linear"))
        ci_upper = float(np.percentile(boot_Gs, 100.0 * (1.0 - alpha / 2.0), method="linear"))

        # Gate passes if CI lower bound <= 0 (no significant positive improvement)
        passed_target = bool(ci_lower <= 0.0)
        if not passed_target:
            any_failed = True

        results_by_target[name] = {
            "G_point": G_point,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "passed": passed_target,
            "best_alpha": float(ridge.best_alpha) if ridge.best_alpha is not None else 0.0,
        }

    return {
        "passed": bool(not any_failed),
        "results": results_by_target,
    }


# ==============================================================================
# Diagnostic Split Loader & Boundary Guard (A07)
# ==============================================================================

def generate_narma10_v2_reference(
    u: np.ndarray,
    u_negative: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Independent step-by-step float64 reference implementation of NARMA10 (§3.1, A02).

    Recurrence:
        y[t+1] = 0.3 * y[t] + 0.05 * y[t] * sum_{k=0..9}(y[t-k]) + 1.5 * u[t] * u[t-9] + 0.1
    Initial state:
        y[-9..0] = 0.0, x[0] = 0.0
        u[-9..-1] generated from explicit u_negative array without wrapping.
    """
    T = len(u)
    y = np.zeros(T + 1, dtype=np.float64)

    if u_negative is not None:
        u_neg = np.asarray(u_negative, dtype=np.float64)
        if len(u_neg) != 9:
            raise ValueError(f"u_negative must have length 9 for u[-9..-1], got {len(u_neg)}")
    else:
        u_neg = np.zeros(9, dtype=np.float64)

    for t in range(T):
        if t >= 9:
            u_lag = float(u[t - 9])
        else:
            u_lag = float(u_neg[t])

        sum_y = 0.0
        for k in range(10):
            if t - k >= 0:
                sum_y += y[t - k]
            # y[-9..0] is 0.0, so negative t-k adds 0.0

        y[t + 1] = 0.3 * y[t] + 0.05 * y[t] * sum_y + 1.5 * float(u[t]) * u_lag + 0.1

    meta = {
        "generator": "independent_step_by_step_v2_reference",
        "washout_start": 0,
        "valid_start": 500,
        "length": T,
    }
    return u.astype(np.float64), y[1:], meta


def generate_diagnostic_splits(
    washout: int = DIAGNOSTIC_WASHOUT,
    valid_length: int = DIAGNOSTIC_FIT_VALID_LENGTH,
) -> dict[str, list[dict]]:
    """Generate the two Stage A/B diagnostic splits (§3.2, A07).

    - Diagnostic-fit: 10 independent sequences using FIXTURE_FIT_SEEDS.
    - Diagnostic-check: 10 independent sequences using FIXTURE_CHECK_SEEDS.
    - Washout = 500, Valid = 2000 steps per sequence.
    - Each sequence generates an independent negative history u[-9..-1] to forbid cross-sequence lag.
    """
    splits: dict[str, list[dict]] = {"diagnostic-fit": [], "diagnostic-check": []}

    for name, seeds in [("diagnostic-fit", FIXTURE_FIT_SEEDS), ("diagnostic-check", FIXTURE_CHECK_SEEDS)]:
        for s in seeds:
            rng = np.random.default_rng(s)
            u_neg = rng.uniform(0.0, 0.5, size=9).astype(np.float64)
            u_main = rng.uniform(0.0, 0.5, size=washout + valid_length).astype(np.float64)

            u_arr, y_next, meta = generate_narma10_v2_reference(u_main, u_negative=u_neg)
            d1, d2 = build_diagnostic_targets(u_arr, u_negative=u_neg, washout=washout)

            splits[name].append({
                "seed": s,
                "split": name,
                "u": u_arr,
                "y_next": y_next,
                "u_negative": u_neg,
                "d1": d1,
                "d2": d2,
                "washout": washout,
                "valid_length": valid_length,
                "total_length": washout + valid_length,
            })

    return splits


def load_diagnostic_split(
    split_name: str,
    requested_seeds: Sequence[int] | None = None,
    authorized: bool = False,
) -> list[dict]:
    """Load diagnostic split sequences, strictly enforcing boundary checks (§3.2, A07).

    - If split_name is 'diagnostic-check', check loader strictly rejects fit seeds.
    - If split_name targets 'test' or 'main-test', raises PermissionError unless authorized.
    """
    validate_split_request(split_name, authorized=authorized)

    clean_name = str(split_name).strip().lower().replace("_", "-")
    all_splits = generate_diagnostic_splits()

    if clean_name not in all_splits:
        raise ValueError(f"Unknown diagnostic split: '{split_name}'. Available: {list(all_splits.keys())}")

    seqs = all_splits[clean_name]

    if requested_seeds is not None:
        req_set = set(int(s) for s in requested_seeds)
        fit_set = set(FIXTURE_FIT_SEEDS)

        if clean_name == "diagnostic-check":
            overlap = req_set & fit_set
            if overlap:
                raise ValueError(
                    f"Check loader strictly rejects fit seeds. Forbidden seeds requested: {overlap}"
                )

        seqs = [seq for seq in seqs if seq["seed"] in req_set]

    return seqs


def solve_ridge_direct_closed_form(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, float]:
    """Independent exact direct solver for (1/n) ||Y - (X B + b0)||^2 + alpha ||B||^2 (A09).

    Derivation:
        With unpenalized intercept, centering gives X_c = X - mean(X), y_c = y - mean(y).
        Normal equations: (X_c^T X_c + n * alpha * I) B = X_c^T y_c
        Intercept: b0 = mean(y) - mean(X) @ B
    """
    X_f = np.asarray(X, dtype=np.float64)
    y_f = np.asarray(y, dtype=np.float64)
    n, p = X_f.shape

    mu_x = np.mean(X_f, axis=0)
    mu_y = float(np.mean(y_f))
    Xc = X_f - mu_x
    yc = y_f - mu_y

    Gram = Xc.T @ Xc
    penalty = (n * alpha) * np.eye(p, dtype=np.float64)
    A = Gram + penalty

    if alpha == 0.0:
        # Thin SVD pseudo-inverse with standard cutoff
        B = np.linalg.pinv(Xc, rcond=np.finfo(np.float64).eps * max(n, p)) @ yc
    else:
        B = np.linalg.solve(A, Xc.T @ yc)

    b0 = float(mu_y - mu_x @ B)
    return B, b0

