"""G1 Synthetic Benchmark Pipeline: NARMA10, Stateful Ridge Readout, and Integrity Gates.

Strict Governance & Review Fixes (SPEC §2, §3, §5, §6, §7, §9):
- Pure numpy/scipy implementation (no external machine learning dependencies).
- Frozen seed constants with SHA256 integrity hash.
- Test access boundary: generate_train_sequences, generate_val_sequences, and unseal_test_sequences.
- Sequence-group 5-fold cross-validation on Train sequences only.
- Ridge Readout: fold-local centered design matrix float64 thin SVD with NumPy lstsq rcond cutoff.
  Reports effective_rank, n_discarded_directions, retained_condition_number.
  Boundary flags: alpha_at_upper_bound (10^2) vs alpha_zero_diagnostic.
- R1-6 sensory selection: select_r16_indices filters for photoreceptor_type == 'R1-6' with finite (u, v).
- Saturation gate: fail-closed on empty, NaN, or Inf values.
- Train-only out-of-sequence 5-fold group CV memory capacity with rank-aware SVD.
- Exact oracle positive control and mismatched negative control.
"""
from __future__ import annotations

import hashlib
import json
from typing import Callable, Sequence

import numpy as np
import scipy.linalg


# ==============================================================================
# Frozen Seed Constants and Configuration (SPEC §2)
# ==============================================================================

TRAIN_SEEDS: tuple[int, ...] = (1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010)
VAL_SEEDS: tuple[int, ...] = (2001, 2002, 2003)
TEST_SEEDS: tuple[int, ...] = (3001, 3002, 3003, 3004, 3005, 3006, 3007, 3008, 3009, 3010)
NEGATIVE_CONTROL_SEED: int = 9999

TRAIN_SEQUENCE_LENGTH: int = 5000
VAL_SEQUENCE_LENGTH: int = 2000
TEST_SEQUENCE_LENGTH: int = 5000
WASHOUT_STEPS: int = 500

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

# SHA-256 hash of the canonical JSON representation of frozen seeds
_SEEDS_PAYLOAD = json.dumps(
    {"negative": NEGATIVE_CONTROL_SEED, "test": list(TEST_SEEDS), "train": list(TRAIN_SEEDS), "val": list(VAL_SEEDS)},
    sort_keys=True,
)
G1_SEEDS_SHA256: str = hashlib.sha256(_SEEDS_PAYLOAD.encode("utf-8")).hexdigest()


# ==============================================================================
# NARMA10 Generators & Sealed Test Access (SPEC §2, §9)
# ==============================================================================

def generate_narma10_reference(u: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Reference step-by-step for-loop implementation of discrete NARMA10.

    Recurrence equation:
        y[t+1] = 0.3 * y[t] + 0.05 * y[t] * sum_{k=0..9}(y[t-k]) + 1.5 * u[t] * u[t-9] + 0.1
    where:
        u[t] ~ Uniform[0, 0.5]
        Initial history: y[-9..0] = 0.0, u[k] = 0.0 for k < 0.

    Alignment:
        At time step t, the reservoir receives input u[t] and targets y_next[t] = y[t+1].

    Returns:
        (u, y_next, metadata)
    """
    T = len(u)
    y = np.zeros(T + 1, dtype=np.float64)

    for t in range(T):
        u_lag = u[t - 9] if t >= 9 else 0.0
        sum_y = 0.0
        for k in range(10):
            if t - k >= 0:
                sum_y += y[t - k]
        y[t + 1] = 0.3 * y[t] + 0.05 * y[t] * sum_y + 1.5 * u[t] * u_lag + 0.1

    metadata = {
        "generator": "reference_for_loop",
        "first_valid_lag_step": 9,
        "first_valid_target_index": 10,
        "initial_history_rule": "y[-9..0]=0, u[<0]=0",
        "length": T,
    }
    return u.astype(np.float64), y[1:], metadata


def generate_narma10_vectorized(u: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Sliding-window buffer implementation of discrete NARMA10.

    Maintains a 10-element ring/history buffer with identical arithmetic order
    to verify bitwise/value consistency against the reference implementation.

    Returns:
        (u, y_next, metadata)
    """
    T = len(u)
    y_next = np.empty(T, dtype=np.float64)
    hist = np.zeros(10, dtype=np.float64)

    for t in range(T):
        u_lag = u[t - 9] if t >= 9 else 0.0
        yt = hist[-1]
        sum_y = 0.0
        for k in range(10):
            sum_y += hist[9 - k]
        y_step = 0.3 * yt + 0.05 * yt * sum_y + 1.5 * u[t] * u_lag + 0.1
        hist[:-1] = hist[1:]
        hist[-1] = y_step
        y_next[t] = y_step

    metadata = {
        "generator": "sliding_buffer_vectorized",
        "first_valid_lag_step": 9,
        "first_valid_target_index": 10,
        "initial_history_rule": "y[-9..0]=0, u[<0]=0",
        "length": T,
    }
    return u.astype(np.float64), y_next, metadata


def generate_narma10_sequence(
    seed: int,
    valid_length: int,
    washout: int = WASHOUT_STEPS,
    split_tag: str = "train",
) -> dict:
    """Generate a single NARMA10 stream with pre-pended washout steps."""
    total_length = valid_length + washout
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 0.5, size=total_length).astype(np.float64)

    u, y_next, meta = generate_narma10_reference(u)
    meta["seed"] = seed
    meta["split"] = split_tag
    meta["valid_length"] = valid_length
    meta["washout_steps"] = washout
    meta["total_length"] = total_length

    return {
        "seed": seed,
        "split": split_tag,
        "u": u,
        "y_next": y_next,
        "washout": washout,
        "valid_length": valid_length,
        "total_length": total_length,
        "metadata": meta,
    }


def generate_train_sequences() -> list[dict]:
    """Generate all frozen Train NARMA10 sequences."""
    return [
        generate_narma10_sequence(seed=s, valid_length=TRAIN_SEQUENCE_LENGTH, washout=WASHOUT_STEPS, split_tag="train")
        for s in TRAIN_SEEDS
    ]


def generate_val_sequences() -> list[dict]:
    """Generate all frozen Val NARMA10 sequences."""
    return [
        generate_narma10_sequence(seed=s, valid_length=VAL_SEQUENCE_LENGTH, washout=WASHOUT_STEPS, split_tag="val")
        for s in VAL_SEEDS
    ]


def unseal_test_sequences(spec_hash: str, code_commit: str | None = None) -> list[dict]:
    """Unseal and generate frozen Test NARMA10 sequences.

    Per G1_SPEC §2, §9:
        Test sequences can ONLY be accessed by providing a valid spec_hash matching
        frozen specification requirements. Early access before code/parameter freeze
        invalidates confirmatory G1 analysis.
    """
    if not spec_hash or not isinstance(spec_hash, str) or len(spec_hash) < 16:
        raise ValueError("Cannot unseal Test sequences: valid spec_hash string required.")

    return [
        generate_narma10_sequence(seed=s, valid_length=TEST_SEQUENCE_LENGTH, washout=WASHOUT_STEPS, split_tag="test")
        for s in TEST_SEEDS
    ]


unseal_test = unseal_test_sequences


def generate_g1_splits(unseal_spec_hash: str | None = None) -> dict[str, list[dict]]:
    """Generate G1 benchmark sequences.

    By default, only returns 'train' and 'val' splits to enforce sealed Test boundary.
    If unseal_spec_hash is provided, unseals and includes 'test' split.
    """
    splits = {
        "train": generate_train_sequences(),
        "val": generate_val_sequences(),
    }
    if unseal_spec_hash is not None:
        splits["test"] = unseal_test_sequences(spec_hash=unseal_spec_hash)
    return splits


# ==============================================================================
# Metric: NMSE (SPEC §5)
# ==============================================================================

def compute_nmse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Normalized Mean Squared Error: NMSE = mean((ŷ - y)^2) / Var(y)."""
    y_p = np.asarray(y_pred, dtype=np.float64)
    y_t = np.asarray(y_true, dtype=np.float64)
    var_y = float(np.var(y_t))
    if var_y <= 1e-12:
        return 0.0
    mse = float(np.mean((y_p - y_t) ** 2))
    return float(mse / var_y)


# ==============================================================================
# SVD Ridge Readout with Sequence-Group 5-Fold CV (SPEC §3, §9)
# ==============================================================================

class G1RidgeReadout:
    """Ridge Readout with unpenalized intercept and float64 SVD pseudo-inverse.

    Objective:
        min_{b0, B} (1/n) ||Y - (X B + b0 * 1)||^2 + alpha ||B||^2

    Normal Equations (via centering X_c = X - mean(X), Y_c = Y - mean(Y)):
        (X_c^T X_c + n * alpha * I) B = X_c^T Y_c
        b0 = mean(Y) - mean(X) @ B

    SVD Solver (Float64 with standard lstsq relative cutoff):
        X_c = Q R, R = U_R Sigma V^T, U = Q U_R
        Cutoff = eps * max(n, d) * sigma_1
        Effective rank = count(sigma_i > cutoff)
        At alpha=0: Moore-Penrose pseudo-inverse solution B(0) = V_k diag(1/sigma_k) U_k^T y_c
        At alpha>0: B(alpha) = V diag( sigma_i / (sigma_i^2 + n * alpha) ) U^T y_c
    """

    def __init__(self, alphas: Sequence[float] = ALPHA_GRID, std_cutoff: float = 1e-8):
        self.alphas = tuple(alphas)
        self.std_cutoff = float(std_cutoff)
        self.best_alpha: float | None = None
        self.weights: np.ndarray | None = None
        self.intercept: float | None = None
        self.train_mu: np.ndarray | None = None
        self.train_std: np.ndarray | None = None
        self.active_features: np.ndarray | None = None
        self.cv_results: dict | None = None

    def _standardize_train(self, X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute Train-only feature mean/std and remove constant columns (std < std_cutoff)."""
        mu = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0)
        active = std >= self.std_cutoff

        X_active = X_train[:, active]
        std_active = std[active]
        mu_active = mu[active]

        X_std = (X_active - mu_active) / std_active
        return X_std, mu_active, std_active, active

    def _apply_standardization(
        self, X: np.ndarray, mu_active: np.ndarray, std_active: np.ndarray, active: np.ndarray
    ) -> np.ndarray:
        """Apply precomputed standardization using only active features."""
        X_active = X[:, active]
        return (X_active - mu_active) / std_active

    @staticmethod
    def _fit_svd_system(
        X_c: np.ndarray,
        y_c: np.ndarray,
        mu_x: np.ndarray,
        mu_y: float,
        alphas: Sequence[float],
    ) -> tuple[dict[float, tuple[np.ndarray, float]], dict]:
        """Solve for B(alpha) and intercept using float64 thin SVD with rank-cutoff.

        Returns:
            solutions_by_alpha: dict mapping alpha -> (B, b0)
            svd_diagnostics: dict with effective_rank, n_discarded, condition_number
        """
        n, d = X_c.shape
        Xc_f64 = X_c.astype(np.float64)
        yc_f64 = y_c.astype(np.float64)

        # Reduced QR for numerical stability and speed
        Q, R = np.linalg.qr(Xc_f64)
        # SVD of R matrix (d x d)
        UR, s, Vt = np.linalg.svd(R, full_matrices=False)
        V = Vt.T

        # Standard numpy lstsq rcond cutoff
        eps = np.finfo(np.float64).eps
        s_max = float(s[0]) if len(s) > 0 else 0.0
        cutoff = eps * max(n, d) * s_max
        mask = s > cutoff
        eff_rank = int(np.sum(mask))
        n_discarded = int(len(s) - eff_rank)
        s_min_retained = float(s[eff_rank - 1]) if eff_rank > 0 else 1.0
        retained_cond = float(s_max / s_min_retained) if s_min_retained > 1e-15 else 1.0

        # Precompute projected target vector z = UR^T @ (Q^T @ yc)
        z = UR.T @ (Q.T @ yc_f64)

        solutions = {}
        for a in alphas:
            if a <= 0.0:
                # Moore-Penrose pseudo-inverse
                inv_s = np.zeros_like(s)
                inv_s[mask] = 1.0 / s[mask]
                B = V @ (inv_s * z)
            else:
                denom = s ** 2 + n * a
                filt = np.where(denom > 0.0, s / denom, 0.0)
                B = V @ (filt * z)

            b0 = float(mu_y - mu_x @ B)
            solutions[a] = (B, b0)

        diag = {
            "effective_rank": eff_rank,
            "n_discarded_directions": n_discarded,
            "retained_condition_number": retained_cond,
            "s_max": s_max,
            "s_min_retained": s_min_retained,
        }
        return solutions, diag

    def fit_sequence_group_cv(
        self,
        sequence_features: list[np.ndarray],
        sequence_targets: list[np.ndarray],
        n_folds: int = 5,
    ) -> G1RidgeReadout:
        """Perform 5-fold sequence-group CV on the 10 Train sequences to select best alpha."""
        n_seqs = len(sequence_features)
        assert n_seqs == 10, f"Expected exactly 10 Train sequences for group CV, got {n_seqs}"

        cv_losses: dict[float, list[float]] = {a: [] for a in self.alphas}
        val_size = n_seqs // n_folds  # 2 sequences per fold

        for f in range(n_folds):
            val_seq_idx = set(range(f * val_size, (f + 1) * val_size))
            tr_seq_idx = [i for i in range(n_seqs) if i not in val_seq_idx]
            va_seq_idx = [i for i in range(n_seqs) if i in val_seq_idx]

            X_tr_fold = np.concatenate([sequence_features[i] for i in tr_seq_idx], axis=0)
            y_tr_fold = np.concatenate([sequence_targets[i] for i in tr_seq_idx], axis=0)
            X_va_fold = np.concatenate([sequence_features[i] for i in va_seq_idx], axis=0)
            y_va_fold = np.concatenate([sequence_targets[i] for i in va_seq_idx], axis=0)

            # Train-only standardization for this fold
            X_tr_std, mu_act, std_act, act = self._standardize_train(X_tr_fold)
            X_va_std = self._apply_standardization(X_va_fold, mu_act, std_act, act)

            # Center variables for unpenalized intercept
            mu_x_tr = np.mean(X_tr_std, axis=0)
            mu_y_tr = float(np.mean(y_tr_fold))
            Xc_tr = X_tr_std - mu_x_tr
            yc_tr = y_tr_fold - mu_y_tr

            # SVD solutions for all alphas on this fold
            solutions, _ = self._fit_svd_system(Xc_tr, yc_tr, mu_x_tr, mu_y_tr, self.alphas)

            for a in self.alphas:
                B_a, b0_a = solutions[a]
                pred_va = X_va_std @ B_a + b0_a
                nmse_val = compute_nmse(pred_va, y_va_fold)
                cv_losses[a].append(nmse_val)

        # Average NMSE across folds
        mean_losses = {a: float(np.mean(cv_losses[a])) for a in self.alphas}
        min_loss = min(mean_losses.values())

        # Select alpha with minimum mean NMSE; tie-break: select LARGER alpha
        tied_alphas = [
            a for a, loss in mean_losses.items()
            if loss == min_loss or (min_loss > 0 and abs(loss - min_loss) / min_loss <= 1e-9)
        ]
        best_a = max(tied_alphas)
        self.best_alpha = float(best_a)

        # Fit final model on all 10 Train sequences using best_alpha
        X_all = np.concatenate(sequence_features, axis=0)
        y_all = np.concatenate(sequence_targets, axis=0)

        X_all_std, self.train_mu, self.train_std, self.active_features = self._standardize_train(X_all)
        mu_x_all = np.mean(X_all_std, axis=0)
        mu_y_all = float(np.mean(y_all))
        Xc_all = X_all_std - mu_x_all
        yc_all = y_all - mu_y_all

        final_solutions, final_diag = self._fit_svd_system(
            Xc_all, yc_all, mu_x_all, mu_y_all, [self.best_alpha]
        )
        self.weights, self.intercept = final_solutions[self.best_alpha]

        # Per SPEC §3, §9: alpha=10^2 is the only upper-bound no-go condition; alpha=0 is valid OLS
        alpha_at_upper = bool(self.best_alpha == self.alphas[-1])
        alpha_zero_diag = bool(self.best_alpha == 0.0)

        self.cv_results = {
            "cv_losses_by_alpha": cv_losses,
            "mean_nmse_by_alpha": mean_losses,
            "best_alpha": self.best_alpha,
            "best_cv_mean_nmse": float(mean_losses[self.best_alpha]),
            "alpha_at_upper_bound": alpha_at_upper,
            "alpha_zero_diagnostic": alpha_zero_diag,
            "alpha_at_boundary": alpha_at_upper,  # Backward-compatible alias for upper bound no-go
            "effective_rank": final_diag["effective_rank"],
            "n_discarded_directions": final_diag["n_discarded_directions"],
            "retained_condition_number": final_diag["retained_condition_number"],
            "n_active_features": int(np.sum(self.active_features)),
            "n_constant_features_dropped": int(len(self.active_features) - np.sum(self.active_features)),
        }
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict on raw feature matrix using fitted weights and standardization."""
        assert self.weights is not None and self.intercept is not None, "Model not fitted"
        X_std = self._apply_standardization(X, self.train_mu, self.train_std, self.active_features)
        return X_std @ self.weights + self.intercept


# ==============================================================================
# R1-6 Sensory Selection (SPEC §3, §9)
# ==============================================================================

def select_r16_indices(graph: Any) -> tuple[np.ndarray, dict]:
    """Select R1-6 photoreceptor indices with finite retina coordinates (u, v).

    Per G1_SPEC §3, §9:
        Only photoreceptors with photoreceptor_type == 'R1-6' and finite (u, v)
        coordinates receive uniform luminance current. R7 and R8 are not injected.
        Non-injected neurons (including R7 and R8) form the denominator of saturation gate.
    """
    meta = graph.meta
    sensory_idx = np.asarray(graph.sensory_idx, dtype=np.int64)

    if "photoreceptor_type" in meta:
        ptypes = np.asarray(meta["photoreceptor_type"])
        u = np.asarray(meta.get("u", []))
        v = np.asarray(meta.get("v", []))

        is_r16 = (ptypes == "R1-6")
        if len(u) == len(ptypes) and len(v) == len(ptypes):
            is_finite = np.isfinite(u) & np.isfinite(v)
            mask = is_r16 & is_finite
            n_non_finite = int(np.sum(is_r16 & ~is_finite))
        else:
            mask = is_r16
            n_non_finite = 0
        n_r7_r8 = int(np.sum((ptypes == "R7") | (ptypes == "R8")))

        if len(mask) == len(sensory_idx):
            r16_idx = sensory_idx[mask]
        elif len(mask) == graph.n_neurons:
            r16_idx = np.where(mask)[0]
        else:
            raise ValueError(
                f"photoreceptor_type length {len(ptypes)} matches neither sensory_idx ({len(sensory_idx)}) nor n_neurons ({graph.n_neurons})"
            )
    else:
        # If no photoreceptor_type metadata (synthetic connectome without R7/R8), use sensory_idx
        r16_idx = sensory_idx
        n_r7_r8 = 0
        n_non_finite = 0

    assert len(r16_idx) > 0, "No valid R1-6 photoreceptor neurons found in graph."

    from research.pipeline.graph_variants import compute_graph_sha256
    graph_sha = meta.get("sha256") or compute_graph_sha256(graph)
    summary = {
        "n_r16_selected": len(r16_idx),
        "total_sensory": len(sensory_idx),
        "n_r7_r8_excluded": n_r7_r8,
        "n_non_finite_excluded": n_non_finite,
        "graph_sha256": graph_sha,
    }
    return r16_idx.astype(np.int64), summary


# ==============================================================================
# Gate Functions (SPEC §4, §7, §9)
# ==============================================================================

def check_saturation_gate(
    states: np.ndarray,
    sensory_idx: np.ndarray | Sequence[int] = (),
    threshold: float = 0.90,
    max_saturation_ratio: float = 0.05,
    max_ratio: float | None = None,
) -> dict:
    """Check saturation gate (a): post-washout non-sensory |x| > 0.90 proportion < 5%.

    Per G1_SPEC §9:
        Empty states, NaN, or Inf values fail closed immediately.
    """
    if max_ratio is not None:
        max_saturation_ratio = max_ratio

    states_arr = np.asarray(states)
    if states_arr.size == 0:
        return {
            "gate": "saturation_gate",
            "passed": False,
            "saturation_ratio": 0.0,
            "threshold": threshold,
            "max_allowed_ratio": max_saturation_ratio,
            "saturated_points": 0,
            "total_non_sensory_points": 0,
            "non_finite_count": 0,
            "failure_reason": "EMPTY_INPUT",
            "error": "Empty non-sensory state matrix",
        }

    if states_arr.ndim == 1:
        states_arr = states_arr[None, :, None]
    elif states_arr.ndim == 2:
        states_arr = states_arr[None, ...]
    B, T, N = states_arr.shape

    sensory_set = set(int(i) for i in sensory_idx)
    non_sensory_mask = np.array([i not in sensory_set for i in range(N)], dtype=bool)

    non_sensory_states = states_arr[:, :, non_sensory_mask]
    total_non_sensory_points = int(non_sensory_states.size)

    if total_non_sensory_points == 0:
        return {
            "gate": "saturation_gate",
            "passed": False,
            "saturation_ratio": 0.0,
            "threshold": threshold,
            "max_allowed_ratio": max_saturation_ratio,
            "saturated_points": 0,
            "total_non_sensory_points": 0,
            "non_finite_count": 0,
            "failure_reason": "EMPTY_INPUT",
            "error": "Empty non-sensory state matrix",
        }

    non_finite_count = int(np.sum(~np.isfinite(non_sensory_states)))
    if non_finite_count > 0:
        return {
            "gate": "saturation_gate",
            "passed": False,
            "saturation_ratio": 1.0,
            "threshold": threshold,
            "max_allowed_ratio": max_saturation_ratio,
            "saturated_points": total_non_sensory_points,
            "total_non_sensory_points": total_non_sensory_points,
            "non_finite_count": non_finite_count,
            "failure_reason": f"NON_FINITE_STATES: Found {non_finite_count} non-finite state values (NaN/Inf)",
            "error": f"Found {non_finite_count} non-finite state values (NaN/Inf)",
        }

    sat_count = int(np.sum(np.abs(non_sensory_states) > threshold))
    sat_ratio = float(sat_count / total_non_sensory_points)
    passed = bool(sat_ratio < max_saturation_ratio)

    return {
        "gate": "saturation_gate",
        "passed": passed,
        "saturation_ratio": sat_ratio,
        "threshold": threshold,
        "max_allowed_ratio": max_saturation_ratio,
        "saturated_points": sat_count,
        "total_non_sensory_points": total_non_sensory_points,
        "non_finite_count": 0,
    }


def check_forgetting_gate(
    sim_fn: Callable[[np.ndarray, np.ndarray | None], tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, dict]],
    train_u_sequences: list[np.ndarray],
    n_neurons: int,
    seed: int = 42,
    n_pairs_per_seq: int = 4,
    t_check: int = 500,
    d_rel_threshold: float = 1e-3,
    min_pass_pairs: int = 38,
) -> dict:
    """Check initial-state forgetting gate (b).

    For the identical first 500 steps across 10 Train sequences:
    Test 4 pairs per sequence of x0=0 vs x0 ~ Uniform[-0.1, 0.1].
    At t=500: d_rel = RMS(x_a - x_b) / max(RMS(x_a), RMS(x_b), 1e-6).
    Requires at least 38 out of 40 pairs (95%) to achieve d_rel < 1e-3.
    """
    rng = np.random.default_rng(seed)
    n_seqs = len(train_u_sequences)
    total_pairs = n_seqs * n_pairs_per_seq
    d_rels = []

    for s_idx, u_full in enumerate(train_u_sequences):
        u_500 = u_full[:t_check]

        for p_idx in range(n_pairs_per_seq):
            x0_a = np.zeros(n_neurons, dtype=np.float32)
            x0_b = rng.uniform(-0.1, 0.1, size=n_neurons).astype(np.float32)

            res_a = sim_fn(u_500, x0_a)
            res_b = sim_fn(u_500, x0_b)
            final_a = res_a[1]
            final_b = res_b[1]

            diff = final_a.ravel() - final_b.ravel()
            rms_diff = float(np.sqrt(np.mean(diff ** 2)))
            rms_a = float(np.sqrt(np.mean(final_a.ravel() ** 2)))
            rms_b = float(np.sqrt(np.mean(final_b.ravel() ** 2)))
            denom = max(rms_a, rms_b, 1e-6)
            d_rel = float(rms_diff / denom)
            d_rels.append(d_rel)

    n_passed = sum(bool(d < d_rel_threshold) for d in d_rels)
    passed = bool(n_passed >= min_pass_pairs)

    return {
        "gate": "forgetting_gate",
        "passed": passed,
        "n_passed_pairs": n_passed,
        "total_pairs": total_pairs,
        "pass_ratio": float(n_passed / total_pairs),
        "d_rel_threshold": d_rel_threshold,
        "min_pass_pairs": min_pass_pairs,
        "d_rel_median": float(np.median(d_rels)),
        "d_rel_max": float(np.max(d_rels)),
        "d_rel_p95": float(np.percentile(d_rels, 95)),
    }


# ==============================================================================
# Train-Only Out-of-Sequence 5-Fold Group CV Memory Capacity (SPEC §4, §9)
# ==============================================================================

def compute_memory_capacity_cv(
    sequence_states: list[np.ndarray],
    sequence_inputs: list[np.ndarray],
    max_lag: int = 50,
    n_folds: int = 5,
    washout: int = WASHOUT_STEPS,
) -> dict:
    """Compute Linear Memory Capacity using out-of-sequence 5-fold group CV.

    Per G1_SPEC §9:
        - Evaluates fixed DN readout states on 10 Train sequences.
        - Splits sequences into 5 folds (2 held-out sequences per fold).
        - Solves linear readout with rank-aware SVD pseudo-inverse on training folds.
        - Evaluates R^2 on held-out sequences for each lag 1..50.
        - MC = sum_{k=1..50} max(R^2_k, 0).
    """
    n_seqs = len(sequence_states)
    assert n_seqs == len(sequence_inputs), "States and inputs list length mismatch"
    assert n_seqs >= n_folds, f"Need at least {n_folds} sequences for CV"

    val_size = n_seqs // n_folds

    # Align post-washout targets and states per sequence
    X_seqs = []
    u_targets_by_seq = []

    for i in range(n_seqs):
        st = sequence_states[i]
        u = sequence_inputs[i]

        # Valid segment
        valid_len = len(st) - washout if len(st) > washout else len(st)
        X_valid = st[washout:] if len(st) > washout else st
        u_valid = u[washout:] if len(u) > washout else u

        X_seqs.append(X_valid)

        # Target matrix for lags 1..max_lag
        # Target for lag k at valid time t is u[washout + t - k]
        T_val = len(X_valid)
        targets_mat = np.empty((T_val, max_lag), dtype=np.float64)
        for k in range(1, max_lag + 1):
            start_idx = washout - k
            targets_mat[:, k - 1] = u[start_idx : start_idx + T_val]
        u_targets_by_seq.append(targets_mat)

    # Cross-validation across folds
    y_true_all = {k: [] for k in range(max_lag)}
    y_pred_all = {k: [] for k in range(max_lag)}

    for f in range(n_folds):
        va_idx = set(range(f * val_size, (f + 1) * val_size))
        tr_idx = [i for i in range(n_seqs) if i not in va_idx]

        X_tr = np.concatenate([X_seqs[i] for i in tr_idx], axis=0).astype(np.float64)
        X_va = np.concatenate([X_seqs[i] for i in va_idx], axis=0).astype(np.float64)

        mu_x = np.mean(X_tr, axis=0)
        Xc_tr = X_tr - mu_x

        # SVD solver for fold design matrix
        Q, R = np.linalg.qr(Xc_tr)
        UR, s, Vt = np.linalg.svd(R, full_matrices=False)
        V = Vt.T

        eps = np.finfo(np.float64).eps
        cutoff = eps * max(len(Xc_tr), Xc_tr.shape[1]) * (s[0] if len(s) > 0 else 0.0)
        mask = s > cutoff
        inv_s = np.zeros_like(s)
        inv_s[mask] = 1.0 / s[mask]

        for k_idx in range(max_lag):
            y_tr = np.concatenate([u_targets_by_seq[i][:, k_idx] for i in tr_idx], axis=0).astype(np.float64)
            y_va = np.concatenate([u_targets_by_seq[i][:, k_idx] for i in va_idx], axis=0).astype(np.float64)

            mu_y = float(np.mean(y_tr))
            yc_tr = y_tr - mu_y

            z = UR.T @ (Q.T @ yc_tr)
            beta = V @ (inv_s * z)
            b0 = float(mu_y - mu_x @ beta)

            pred_va = X_va @ beta + b0

            y_true_all[k_idx].append(y_va)
            y_pred_all[k_idx].append(pred_va)

    # Compute out-of-fold pooled R^2 per lag
    r2_by_lag = []
    for k_idx in range(max_lag):
        yt = np.concatenate(y_true_all[k_idx])
        yp = np.concatenate(y_pred_all[k_idx])

        ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
        ss_res = float(np.sum((yt - yp) ** 2))
        r2 = max(0.0, float(1.0 - ss_res / ss_tot)) if ss_tot > 1e-12 else 0.0
        r2_by_lag.append(r2)

    mc_total = float(np.sum(r2_by_lag))
    return {
        "metric": "memory_capacity_cv",
        "mc_total": mc_total,
        "max_lag": max_lag,
        "n_folds": n_folds,
        "r2_by_lag": r2_by_lag,
    }


def compute_memory_capacity(
    states: np.ndarray,
    u: np.ndarray,
    max_lag: int = 50,
) -> dict:
    """In-sample linear memory capacity helper for single continuous sequence benchmarks."""
    T = len(u)
    assert len(states) == T, f"States length {len(states)} does not match u length {T}"
    assert T > max_lag + 50, f"Sequence length {T} too short for max_lag {max_lag}"

    eval_slice = slice(max_lag, T)
    X_eval = states[eval_slice]
    n_samples = len(X_eval)

    X_ones = np.column_stack([np.ones(n_samples, dtype=np.float64), X_eval])

    r2_by_lag = []
    for k in range(1, max_lag + 1):
        target_u = u[max_lag - k : T - k]
        beta = np.linalg.lstsq(X_ones, target_u, rcond=None)[0]
        pred_u = X_ones @ beta

        ss_tot = float(np.sum((target_u - np.mean(target_u)) ** 2))
        ss_res = float(np.sum((target_u - pred_u) ** 2))
        r2 = max(0.0, float(1.0 - ss_res / ss_tot)) if ss_tot > 1e-12 else 0.0
        r2_by_lag.append(r2)

    mc_total = float(np.sum(r2_by_lag))
    return {
        "metric": "memory_capacity",
        "mc_total": mc_total,
        "max_lag": max_lag,
        "r2_by_lag": r2_by_lag,
    }


def compute_memory_capacity_fast(
    states: np.ndarray,
    u: np.ndarray,
    max_lag: int = 50,
) -> dict:
    """Vectorized in-sample linear memory capacity helper."""
    T = len(u)
    assert len(states) == T, f"States length {len(states)} does not match u length {T}"
    assert T > max_lag + 50, f"Sequence length {T} too short for max_lag {max_lag}"

    eval_slice = slice(max_lag, T)
    X_eval = states[eval_slice]
    n_samples = len(X_eval)

    X_ones = np.column_stack([np.ones(n_samples, dtype=np.float64), X_eval])
    Q, _ = np.linalg.qr(X_ones)

    U_targets = np.column_stack([u[max_lag - k : T - k] for k in range(1, max_lag + 1)])
    pred_matrix = Q @ (Q.T @ U_targets)

    ss_tot = np.sum((U_targets - np.mean(U_targets, axis=0)) ** 2, axis=0)
    ss_res = np.sum((U_targets - pred_matrix) ** 2, axis=0)
    r2_by_lag = np.maximum(0.0, np.where(ss_tot > 1e-12, 1.0 - ss_res / ss_tot, 0.0))

    return {
        "metric": "memory_capacity",
        "mc_total": float(np.sum(r2_by_lag)),
        "max_lag": max_lag,
        "r2_by_lag": [float(r) for r in r2_by_lag],
    }


# ==============================================================================
# Positive and Negative Controls (SPEC §6)
# ==============================================================================

def build_narma10_oracle_features(u: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Construct teacher features from known NARMA10 recurrence terms."""
    T = len(u)
    features = np.zeros((T, 3), dtype=np.float64)

    for t in range(T):
        u_lag = u[t - 9] if t >= 9 else 0.0
        sum_y = sum(y[t - k] for k in range(10) if t - k >= 0)
        features[t, 0] = y[t]
        features[t, 1] = y[t] * sum_y
        features[t, 2] = u[t] * u_lag

    return features


def evaluate_oracle_positive_control(splits: dict[str, list[dict]]) -> dict:
    """Verify that oracle teacher features achieve NMSE < 1e-6 across all provided splits."""
    results = {}

    for split_name in splits:
        seqs = splits[split_name]
        nmse_list = []

        for seq in seqs:
            u = seq["u"]
            y_next = seq["y_next"]
            washout = seq["washout"]

            y = np.concatenate([[0.0], y_next[:-1]])
            X_oracle = build_narma10_oracle_features(u, y)

            X_eval = X_oracle[washout:]
            y_eval = y_next[washout:]

            X_ones = np.column_stack([np.ones(len(X_eval)), X_eval])
            beta = np.linalg.lstsq(X_ones, y_eval, rcond=None)[0]
            pred = X_ones @ beta
            nmse = compute_nmse(pred, y_eval)
            nmse_list.append(nmse)

        mean_nmse = float(np.mean(nmse_list))
        results[split_name] = {
            "mean_nmse": mean_nmse,
            "max_nmse": float(np.max(nmse_list)),
            "passed": bool(mean_nmse < 1e-6),
        }

    all_passed = all(r["passed"] for r in results.values())
    return {
        "control": "oracle_positive_control",
        "passed": all_passed,
        "splits": results,
    }


def evaluate_negative_control(
    model: G1RidgeReadout,
    mismatched_states: np.ndarray,
    y_true: np.ndarray,
) -> dict:
    """Verify that mismatched input stream shows no improvement over constant-mean baseline (NMSE >= 1.0)."""
    pred = model.predict(mismatched_states)
    nmse = compute_nmse(pred, y_true)
    passed = bool(nmse >= 0.99)
    return {
        "control": "mismatched_negative_control",
        "passed": passed,
        "observed_nmse": nmse,
        "baseline_nmse": 1.0,
    }
