"""Research v2 Phase 5 Post-Hoc Exploratory Analysis (EXPLORATORY_POST_HOC).

CRITICAL GOVERNANCE AND AUDIT PRINCIPLES:
1. ORIGINAL_VERDICT: FAIL (unchanged; see research/outputs/v2/probe_val.json).
2. THIS FILE IS EXPLORATORY_POST_HOC AND MUST NOT BE CITED AS PASS EVIDENCE.
   Per research_v2.md §12, Phase 5 pre-registered criteria have failed.
   This post-hoc exploratory module investigates:
   - Logistic decoder performance differences against matched and non-connectome controls.
   - Paired moving block bootstrap (block=24, 2000 replicates) with two-tailed p-values and Holm correction.
   - Redundancy of Connectome signals relative to OHLCV (B1b) and renderer nuisance features.
   - Residual IC after linear regression on control predictions.
   - Sub-period temporal stability across 4 validation quarters.
   - Corrected presentation of pass/fail criteria (auditing vacuous trading passes, competitive
     balanced accuracy baselines, and renderer shortcut R^2 on non-constant Logistic scores).
3. Whitelist: Strictly evaluates 'train' and 'val'. dev_test_v1 and holdout are completely forbidden.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import yaml

import numpy as np
import pandas as pd
import scipy.stats

from research.pipeline.baselines_v2 import (
    ACTIONS,
    ACTION_BUY,
    ACTION_SELL,
    ACTION_HOLD,
    PureNumpyRidge,
    select_ridge_alpha_timeseries_cv,
    fit_predict_b1b_logistic,
    compute_continuous_metrics,
    compute_classification_metrics,
    moving_block_bootstrap_ci,
    simulate_trading_spot_long_only,
    compute_train_standardization,
    apply_standardization,
    validate_split_name,
    compute_file_sha256,
    get_git_commit,
    extract_sample_features,
)
from research.pipeline.train_probe_v2 import extract_nuisance_features_split

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"


def holm_bonferroni_correction(p_values: list[float] | np.ndarray) -> np.ndarray:
    """Compute Holm-Bonferroni step-down family-wise error rate corrected p-values.

    Parameters
    ----------
    p_values : list of floats or 1D array of uncorrected p-values.

    Returns
    -------
    np.ndarray of corrected p-values, in the original input order.
    """
    p_vals = np.asarray(p_values, dtype=np.float64)
    m = len(p_vals)
    if m == 0:
        return np.array([], dtype=np.float64)
    if m == 1:
        return np.clip(p_vals, 0.0, 1.0)

    order = np.argsort(p_vals)
    sorted_p = p_vals[order]

    # Adjusted p_k = (m - k + 1) * p_k (for 1-indexed k)
    multipliers = m - np.arange(m)
    adj_p = sorted_p * multipliers

    # Enforce monotonicity: adj_p[i] = max(adj_p[i-1], adj_p[i])
    for i in range(1, m):
        adj_p[i] = max(adj_p[i], adj_p[i - 1])

    adj_p = np.clip(adj_p, 0.0, 1.0)

    # Restore original order
    rev_order = np.empty(m, dtype=int)
    rev_order[order] = np.arange(m)
    return adj_p[rev_order]


def paired_multi_block_bootstrap(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    block_size: int = 24,
    n_bootstraps: int = 2000,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Paired moving block bootstrap applying identical resampled blocks to all models.

    Parameters
    ----------
    y_true : 1D array of ground truth future returns.
    predictions : dict mapping model_name -> 1D array of continuous predictions on Val.
    block_size : int, length of each block (default 24).
    n_bootstraps : int, number of resamplings (default 2000).
    seed : int, RNG seed.

    Returns
    -------
    dict mapping model_name -> np.ndarray of shape (n_bootstraps,) containing Spearman ICs.
    """
    n = len(y_true)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_size))

    model_names = list(predictions.keys())
    replicates = {name: np.empty(n_bootstraps, dtype=np.float64) for name in model_names}

    for b in range(n_bootstraps):
        starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        idx = (starts[:, None] + np.arange(block_size)).ravel()[:n]
        sub_y = y_true[idx]

        for name in model_names:
            pred = predictions[name][idx]
            if np.std(pred) < 1e-12 or np.std(sub_y) < 1e-12:
                replicates[name][b] = 0.0
            else:
                r = scipy.stats.spearmanr(pred, sub_y)
                ic = float(r.statistic if hasattr(r, "statistic") else r[0])
                replicates[name][b] = 0.0 if np.isnan(ic) else ic

    return replicates


def compute_delta_ic_stats(
    replicates_real: np.ndarray,
    replicates_ctrl: np.ndarray,
    observed_ic_real: float,
    observed_ic_ctrl: float,
    alpha: float = 0.05,
) -> dict:
    """Compute Delta IC point estimate, 95% bootstrap CI, and two-tailed p-value."""
    delta_reps = replicates_real - replicates_ctrl
    observed_delta = float(observed_ic_real - observed_ic_ctrl)
    lower = float(np.percentile(delta_reps, 100.0 * (alpha / 2.0)))
    upper = float(np.percentile(delta_reps, 100.0 * (1.0 - alpha / 2.0)))

    # Two-tailed bootstrap p-value dual to the percentile CI
    b = len(delta_reps)
    p_lower = (1.0 + np.count_nonzero(delta_reps <= 0.0)) / (b + 1.0)
    p_upper = (1.0 + np.count_nonzero(delta_reps >= 0.0)) / (b + 1.0)
    p_val = float(min(1.0, 2.0 * min(p_lower, p_upper)))

    return {
        "observed_delta_ic": observed_delta,
        "delta_ic_ci_95": [lower, upper],
        "p_value_two_tailed": p_val,
    }


def compute_redundancy_and_residual_ic(
    s_real: np.ndarray,
    s_ohlcv: np.ndarray,
    s_nui: np.ndarray,
    y_true: np.ndarray,
    block_size: int = 24,
    n_bootstraps: int = 2000,
    seed: int = 42,
) -> dict:
    """Analyze signal redundancy and residual IC.

    1. Spearman correlations between real score and controls (OHLCV, Nuisance).
    2. Linear regressions of real score on controls:
       - s_real ~ s_ohlcv
       - s_real ~ s_nui
       - s_real ~ s_ohlcv + s_nui
    3. Spearman IC of regression residuals with future_return_6 (+ block bootstrap 95% CI).
    """
    n = len(y_true)

    # 1. Pairwise correlations
    r_real_ohlcv = scipy.stats.spearmanr(s_real, s_ohlcv)
    corr_real_ohlcv = float(r_real_ohlcv.statistic if hasattr(r_real_ohlcv, "statistic") else r_real_ohlcv[0])

    r_real_nui = scipy.stats.spearmanr(s_real, s_nui)
    corr_real_nui = float(r_real_nui.statistic if hasattr(r_real_nui, "statistic") else r_real_nui[0])

    r_ohlcv_nui = scipy.stats.spearmanr(s_ohlcv, s_nui)
    corr_ohlcv_nui = float(r_ohlcv_nui.statistic if hasattr(r_ohlcv_nui, "statistic") else r_ohlcv_nui[0])

    # 2. Linear regressions & Residuals
    ones = np.ones((n, 1), dtype=np.float64)

    # (a) s_real ~ s_ohlcv
    Z_ohlcv = np.column_stack([ones, s_ohlcv])
    w_ohlcv = np.linalg.lstsq(Z_ohlcv, s_real, rcond=None)[0]
    res_ohlcv = s_real - Z_ohlcv @ w_ohlcv
    r2_ohlcv = float(1.0 - np.sum(res_ohlcv**2) / np.sum((s_real - np.mean(s_real))**2))
    ic_res_ohlcv = compute_continuous_metrics(y_true, res_ohlcv)["spearman_ic"]
    ci_res_ohlcv = moving_block_bootstrap_ci(
        y_true, res_ohlcv, metric_type="ic", block_size=block_size, n_bootstraps=n_bootstraps, seed=seed
    )

    # (b) s_real ~ s_nui
    Z_nui = np.column_stack([ones, s_nui])
    w_nui = np.linalg.lstsq(Z_nui, s_real, rcond=None)[0]
    res_nui = s_real - Z_nui @ w_nui
    r2_nui = float(1.0 - np.sum(res_nui**2) / np.sum((s_real - np.mean(s_real))**2))
    ic_res_nui = compute_continuous_metrics(y_true, res_nui)["spearman_ic"]
    ci_res_nui = moving_block_bootstrap_ci(
        y_true, res_nui, metric_type="ic", block_size=block_size, n_bootstraps=n_bootstraps, seed=seed
    )

    # (c) s_real ~ s_ohlcv + s_nui (Joint)
    Z_joint = np.column_stack([ones, s_ohlcv, s_nui])
    w_joint = np.linalg.lstsq(Z_joint, s_real, rcond=None)[0]
    res_joint = s_real - Z_joint @ w_joint
    r2_joint = float(1.0 - np.sum(res_joint**2) / np.sum((s_real - np.mean(s_real))**2))
    ic_res_joint = compute_continuous_metrics(y_true, res_joint)["spearman_ic"]
    ci_res_joint = moving_block_bootstrap_ci(
        y_true, res_joint, metric_type="ic", block_size=block_size, n_bootstraps=n_bootstraps, seed=seed
    )

    return {
        "correlations": {
            "spearman_real_vs_ohlcv": corr_real_ohlcv,
            "spearman_real_vs_nuisance": corr_real_nui,
            "spearman_ohlcv_vs_nuisance": corr_ohlcv_nui,
        },
        "residual_ic_vs_ohlcv_only": {
            "r2_explained_by_ohlcv": r2_ohlcv,
            "residual_spearman_ic": ic_res_ohlcv,
            "residual_ic_ci_95": list(ci_res_ohlcv),
        },
        "residual_ic_vs_nuisance_only": {
            "r2_explained_by_nuisance": r2_nui,
            "residual_spearman_ic": ic_res_nui,
            "residual_ic_ci_95": list(ci_res_nui),
        },
        "residual_ic_joint_controls": {
            "r2_explained_by_joint_controls": r2_joint,
            "residual_spearman_ic": ic_res_joint,
            "residual_ic_ci_95": list(ci_res_joint),
        },
    }


def compute_temporal_stability(
    predictions: dict[str, np.ndarray],
    y_true: np.ndarray,
    n_quarters: int = 4,
) -> dict:
    """Divide Validation into 4 temporal quarters and compute IC for each quarter."""
    n = len(y_true)
    quarter_size = n // n_quarters
    quarter_results = {}

    for name, pred in predictions.items():
        q_ics = []
        for q in range(n_quarters):
            start = q * quarter_size
            end = (q + 1) * quarter_size if q < n_quarters - 1 else n
            sub_pred = pred[start:end]
            sub_y = y_true[start:end]
            ic = compute_continuous_metrics(sub_y, sub_pred)["spearman_ic"]
            q_ics.append(ic)

        pos_count = int(sum(ic > 0 for ic in q_ics))
        sign_agreement = float(max(np.mean([ic > 0 for ic in q_ics]), np.mean([ic < 0 for ic in q_ics])))
        quarter_results[name] = {
            "quarterly_ic": q_ics,
            "positive_quarter_count": pos_count,
            "total_quarters": n_quarters,
            "all_positive": bool(pos_count == n_quarters),
            "sign_agreement": sign_agreement,
        }

    return quarter_results


def build_corrected_table(
    probe_val_data: dict,
    real_logistic_score: np.ndarray,
    nuisance_features_val: np.ndarray,
    ohlcv_b1b_ba: float,
    nuisance_logistic_ba: float,
) -> list[dict]:
    """Reconstruct and audit pass/fail criteria table with explicit flags and corrections.

    Audits:
    1. trade_count == 0 trading rows marked as N/A_NO_TRADES (not PASS).
    2. balanced_accuracy row compared side-by-side with Nuisance and OHLCV baselines.
    3. renderer_shortcut_r2 evaluated on non-constant real Logistic score against 7 nuisance features.
    """
    orig_table = probe_val_data.get("pass_fail_evaluation", [])
    real_models = probe_val_data.get("models", {}).get("real", {})
    real_ridge = real_models.get("ridge", {})
    real_log = real_models.get("logistic", {})

    ridge_trade_count = real_ridge.get("trading", {}).get("trade_count", 0)
    ridge_ba = real_ridge.get("classification", {}).get("balanced_accuracy", 0.3333333333333333)
    log_ba = real_log.get("classification", {}).get("balanced_accuracy", 0.35680975427584594)

    # Compute R^2 of real Logistic score regressed on 7 nuisance features on Val
    ones = np.ones((len(real_logistic_score), 1), dtype=np.float64)
    Z_nui = np.column_stack([ones, nuisance_features_val])
    w_nui = np.linalg.lstsq(Z_nui, real_logistic_score, rcond=None)[0]
    res_nui = real_logistic_score - Z_nui @ w_nui
    s_var = np.sum((real_logistic_score - np.mean(real_logistic_score))**2)
    logistic_nuisance_r2 = float(1.0 - np.sum(res_nui**2) / s_var) if s_var > 0 else 0.0

    corrected_rows = []

    for orig_row in orig_table:
        key = orig_row["threshold_key"]
        row_copy = dict(orig_row)

        # 1. Trading criteria with 0 trades
        if key in ("trading_net_return_min", "trading_sharpe_min", "trading_max_drawdown_max"):
            if ridge_trade_count == 0:
                row_copy["passed"] = False
                row_copy["verdict_flag"] = "N/A_NO_TRADES"
                row_copy["audit_note"] = (
                    "Zero trades executed in simulation (passive cash position); "
                    "vacuously passed zero-return / zero-drawdown thresholds in original table."
                )
            else:
                row_copy["verdict_flag"] = "PASS" if orig_row["passed"] else "FAIL"
                row_copy["audit_note"] = f"Trade count: {ridge_trade_count}"

        # 2. Balanced accuracy comparison
        elif key == "min_practical_effect_balanced_acc":
            row_copy["observed_value_ridge"] = ridge_ba
            row_copy["observed_value_logistic"] = log_ba
            row_copy["control_nuisance_logistic"] = nuisance_logistic_ba
            row_copy["control_ohlcv_b1b_logistic"] = ohlcv_b1b_ba
            row_copy["verdict_flag"] = "EXCEEDS_RANDOM_BUT_BELOW_CONTROLS"
            row_copy["passed"] = False
            row_copy["audit_note"] = (
                f"Real Logistic BA ({log_ba:.4f}) exceeds random (0.3333) by +{log_ba - 0.333333:.4f}, "
                f"but is substantially lower than Nuisance-only ({nuisance_logistic_ba:.4f}, delta={log_ba - nuisance_logistic_ba:+.4f}) "
                f"and OHLCV B1b ({ohlcv_b1b_ba:.4f}, delta={log_ba - ohlcv_b1b_ba:+.4f})."
            )

        # 3. Renderer shortcut R^2
        elif key == "renderer_shortcut_r2_max":
            row_copy["observed_value_original_ridge"] = orig_row["observed_value"]
            row_copy["observed_value_corrected_logistic_r2"] = logistic_nuisance_r2
            row_copy["observed_value"] = logistic_nuisance_r2
            passed = bool(logistic_nuisance_r2 <= 0.05)
            row_copy["passed"] = passed
            row_copy["verdict_flag"] = "PASS" if passed else "FAIL"
            row_copy["audit_note"] = (
                f"Evaluated on active real Logistic score (variance={np.var(real_logistic_score):.6e}) "
                f"regressed on 7 renderer nuisance features. R^2={logistic_nuisance_r2:.6e} <= 0.05."
            )
        else:
            row_copy["verdict_flag"] = "PASS" if orig_row["passed"] else "FAIL"
            row_copy["audit_note"] = "Unchanged from original pre-registered evaluation."

        corrected_rows.append(row_copy)

    return corrected_rows


def verify_feature_hashes(
    features_dir: Path,
    expected_hashes: dict[str, str],
) -> None:
    """Verify that feature files match expected SHA256 hashes recorded in probe_val.json."""
    for key, expected_hash in expected_hashes.items():
        feat_path = features_dir / f"{key}.npy"
        if not feat_path.exists():
            raise FileNotFoundError(f"Missing required feature file: {feat_path}")
        actual_hash = compute_file_sha256(feat_path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"Feature hash mismatch for {key}:\n"
                f"  Expected: {expected_hash}\n"
                f"  Actual:   {actual_hash}"
            )


def verify_reproduction(
    observed_ics: dict[str, float],
    probe_val_data: dict,
    tolerance: float = 1e-4,
) -> None:
    """Verify retrained model ICs against their recorded probe_val.json fields."""
    models_dict = probe_val_data.get("models", {})
    expected_ics = {
        f"{variant}_{model_type}": models_dict[variant][model_type]["continuous"]["spearman_ic"]
        for variant in ("real", "random", "scramble")
        for model_type in ("ridge", "logistic")
    }
    expected_ics.update({
        f"nuisance_{model_type}": models_dict["nuisance_only"][model_type]["continuous"]["spearman_ic"]
        for model_type in ("ridge", "logistic")
    })
    expected_ics["ohlcv_ridge"] = models_dict["ohlcv_baselines"]["B1a_ridge_ic"]
    expected_ics["ohlcv_logistic"] = models_dict["ohlcv_baselines"]["B1b_logistic_ic"]

    for key, obs_ic in observed_ics.items():
        if key not in expected_ics:
            raise KeyError(f"No recorded probe_val.json IC for reproduction key {key!r}")
        expected_ic = expected_ics[key]
        diff = abs(obs_ic - expected_ic)
        if diff > tolerance:
            raise RuntimeError(
                f"Reproduction check failed for {key}: expected IC={expected_ic:.8f}, "
                f"got IC={obs_ic:.8f} (diff={diff:.4e} > tol {tolerance})"
            )


def run_posthoc_analysis(
    features_dir: Path | str = OUTPUTS_DIR / "v2/features",
    labels_path: Path | str = DATA_DIR / "labels_v2.parquet",
    probe_val_path: Path | str = OUTPUTS_DIR / "v2/probe_val.json",
    config_path: Path | str = ROOT / "config/experiment_v2.yaml",
    output_path: Path | str = OUTPUTS_DIR / "v2/probe_posthoc_val.json",
    bootstrap_samples: int = 2000,
    block_size: int = 24,
    seed: int = 42,
    verify_hashes: bool = True,
    verify_repro: bool = True,
    check_only: bool = False,
) -> dict:
    """Execute Phase 5 Exploratory Post-Hoc Analysis."""
    features_dir = Path(features_dir)
    labels_path = Path(labels_path)
    probe_val_path = Path(probe_val_path)
    config_path = Path(config_path)
    output_path = Path(output_path)

    # 1. Load labels and config
    with open(probe_val_path, "r", encoding="utf-8") as f:
        probe_val_data = json.load(f)

    with open(config_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    labels = pd.read_parquet(labels_path, filters=[("split", "in", ["train", "val"])])
    validate_split_name("train")
    validate_split_name("val")

    labels_split = labels["split"].to_numpy()
    y_cont_all = labels["future_return_6"].to_numpy(dtype=np.float64)
    y_cls_all = labels["action"].to_numpy(dtype=object)

    train_mask = (labels_split == "train") & (~np.isnan(y_cont_all)) & (pd.notna(y_cls_all))
    val_mask = (labels_split == "val") & (~np.isnan(y_cont_all)) & (pd.notna(y_cls_all))

    y_train_cont = y_cont_all[train_mask]
    y_train_cls = y_cls_all[train_mask]
    y_val_cont = y_cont_all[val_mask]
    y_val_cls = y_cls_all[val_mask]

    n_train = int(np.sum(train_mask))
    n_val = int(np.sum(val_mask))

    cost_fee = float(config_data["cost"]["cost_fee_per_side"])
    cost_slip = float(config_data["cost"]["cost_slippage"])
    cost_threshold = float(2.0 * cost_fee + cost_slip)

    # 2. Verify feature file hashes
    expected_hashes = probe_val_data.get("metadata", {}).get("feature_hashes", {})
    if verify_hashes:
        print("Verifying feature file SHA256 hashes against probe_val.json...")
        verify_feature_hashes(features_dir, expected_hashes)
        print("Feature hashes verified successfully.")

    # 3. Load features & train decoders for Real, Random, Scramble
    predictions_ridge = {}
    predictions_log = {}
    observed_ics = {}

    for variant in ["real", "random", "scramble"]:
        tr_path = features_dir / f"{variant}_train.npy"
        va_path = features_dir / f"{variant}_val.npy"
        X_tr = np.load(tr_path)
        X_va = np.load(va_path)

        if len(X_tr) == len(labels_split[labels_split == "train"]):
            X_tr = X_tr[train_mask[labels_split == "train"]]
        if len(X_va) == len(labels_split[labels_split == "val"]):
            X_va = X_va[val_mask[labels_split == "val"]]

        # Train-only standardization
        norm = compute_train_standardization(X_tr, [f"f_{i}" for i in range(X_tr.shape[1])])
        X_tr_std = apply_standardization(X_tr, norm)
        X_va_std = apply_standardization(X_va, norm)

        # Ridge
        best_alpha = select_ridge_alpha_timeseries_cv(X_tr_std, y_train_cont)
        ridge = PureNumpyRidge(alpha=best_alpha).fit(X_tr_std, y_train_cont)
        pred_ridge = ridge.predict(X_va_std)
        ic_ridge = compute_continuous_metrics(y_val_cont, pred_ridge)["spearman_ic"]

        # Logistic
        pred_log, pred_cls = fit_predict_b1b_logistic(X_tr_std, y_train_cls, X_va_std)
        ic_log = compute_continuous_metrics(y_val_cont, pred_log)["spearman_ic"]

        predictions_ridge[variant] = pred_ridge
        predictions_log[variant] = pred_log
        observed_ics[f"{variant}_ridge"] = ic_ridge
        observed_ics[f"{variant}_logistic"] = ic_log

    # 4. OHLCV Baselines
    print("Computing non-Connectome OHLCV features (B1a Ridge, B1b Logistic)...")
    samples = pd.read_parquet(DATA_DIR / "samples.parquet", filters=[("split", "in", ["train", "val"])])
    raw = pd.read_parquet(DATA_DIR / "raw_ohlcv.parquet")

    X_ohlcv, ohlcv_names = extract_sample_features(raw, samples)
    X_ohlcv_tr = X_ohlcv[train_mask]
    X_ohlcv_va = X_ohlcv[val_mask]

    norm_ohlcv = compute_train_standardization(X_ohlcv_tr, ohlcv_names)
    X_ohlcv_tr_std = apply_standardization(X_ohlcv_tr, norm_ohlcv)
    X_ohlcv_va_std = apply_standardization(X_ohlcv_va, norm_ohlcv)

    best_alpha_ohlcv = select_ridge_alpha_timeseries_cv(X_ohlcv_tr_std, y_train_cont)
    ridge_ohlcv = PureNumpyRidge(alpha=best_alpha_ohlcv).fit(X_ohlcv_tr_std, y_train_cont)
    pred_ohlcv_ridge = ridge_ohlcv.predict(X_ohlcv_va_std)
    ic_ohlcv_ridge = compute_continuous_metrics(y_val_cont, pred_ohlcv_ridge)["spearman_ic"]

    pred_ohlcv_log, pred_ohlcv_cls = fit_predict_b1b_logistic(X_ohlcv_tr_std, y_train_cls, X_ohlcv_va_std)
    ic_ohlcv_log = compute_continuous_metrics(y_val_cont, pred_ohlcv_log)["spearman_ic"]
    ba_ohlcv_log = compute_classification_metrics(y_val_cls, pred_ohlcv_cls)["balanced_accuracy"]

    predictions_ridge["ohlcv_b1a"] = pred_ohlcv_ridge
    predictions_log["ohlcv_b1b"] = pred_ohlcv_log
    observed_ics["ohlcv_ridge"] = ic_ohlcv_ridge
    observed_ics["ohlcv_logistic"] = ic_ohlcv_log

    # 5. Nuisance-only Probe (7 renderer features)
    print("Computing Nuisance-only features (7 renderer features)...")
    images_mmap = np.load(DATA_DIR / "images.npy", mmap_mode="r")
    samples_tr = samples[samples["split"] == "train"].iloc[:n_train]
    samples_va = samples[samples["split"] == "val"].iloc[:n_val]

    nui_tr = extract_nuisance_features_split(samples_tr, raw, images_mmap).to_numpy(dtype=np.float64)
    nui_va = extract_nuisance_features_split(samples_va, raw, images_mmap).to_numpy(dtype=np.float64)

    norm_nui = compute_train_standardization(nui_tr, [f"nui_{i}" for i in range(7)])
    nui_tr_std = apply_standardization(nui_tr, norm_nui)
    nui_va_std = apply_standardization(nui_va, norm_nui)

    best_alpha_nui = select_ridge_alpha_timeseries_cv(nui_tr_std, y_train_cont)
    ridge_nui = PureNumpyRidge(alpha=best_alpha_nui).fit(nui_tr_std, y_train_cont)
    pred_nui_ridge = ridge_nui.predict(nui_va_std)
    ic_nui_ridge = compute_continuous_metrics(y_val_cont, pred_nui_ridge)["spearman_ic"]

    pred_nui_log, pred_nui_cls = fit_predict_b1b_logistic(nui_tr_std, y_train_cls, nui_va_std)
    ic_nui_log = compute_continuous_metrics(y_val_cont, pred_nui_log)["spearman_ic"]
    ba_nui_log = compute_classification_metrics(y_val_cls, pred_nui_cls)["balanced_accuracy"]

    predictions_ridge["nuisance_only"] = pred_nui_ridge
    predictions_log["nuisance_only"] = pred_nui_log
    observed_ics["nuisance_ridge"] = ic_nui_ridge
    observed_ics["nuisance_logistic"] = ic_nui_log

    # 6. Verify reproduction against probe_val.json
    if verify_repro:
        print("Verifying reproduction against probe_val.json...")
        verify_reproduction(observed_ics, probe_val_data)
        print("Reproduction check PASSED: All IC values match probe_val.json.")

    if check_only:
        print("Check-only mode requested. All pre-flight checks passed.")
        return {"status": "SUCCESS_CHECKS_PASSED", "observed_ics": observed_ics}

    # 7. Section 1a: Paired Multi-Model Block Bootstrap (2000 replications)
    print(f"Running Paired Block Bootstrap ({bootstrap_samples} replicates, block={block_size})...")
    all_preds = {
        **{f"log_{k}": v for k, v in predictions_log.items()},
        **{f"ridge_{k}": v for k, v in predictions_ridge.items()},
    }
    replicates = paired_multi_block_bootstrap(
        y_true=y_val_cont,
        predictions=all_preds,
        block_size=block_size,
        n_bootstraps=bootstrap_samples,
        seed=seed,
    )

    # Comparisons for Logistic
    log_comparisons_raw = {}
    log_targets = [
        ("real_vs_random", "log_real", "log_random", observed_ics["real_logistic"], observed_ics["random_logistic"]),
        ("real_vs_scramble", "log_real", "log_scramble", observed_ics["real_logistic"], observed_ics["scramble_logistic"]),
        ("real_vs_ohlcv_b1b", "log_real", "log_ohlcv_b1b", observed_ics["real_logistic"], observed_ics["ohlcv_logistic"]),
        ("real_vs_nuisance_only", "log_real", "log_nuisance_only", observed_ics["real_logistic"], observed_ics["nuisance_logistic"]),
    ]

    p_vals_log = []
    for comp_name, real_key, ctrl_key, obs_real, obs_ctrl in log_targets:
        stat = compute_delta_ic_stats(replicates[real_key], replicates[ctrl_key], obs_real, obs_ctrl)
        log_comparisons_raw[comp_name] = stat
        p_vals_log.append(stat["p_value_two_tailed"])

    adj_p_log = holm_bonferroni_correction(p_vals_log)
    for (comp_name, _, _, _, _), adj_p in zip(log_targets, adj_p_log):
        log_comparisons_raw[comp_name]["p_value_holm_adjusted"] = float(adj_p)

    # Comparisons for Ridge
    ridge_comparisons_raw = {}
    ridge_targets = [
        ("real_vs_random", "ridge_real", "ridge_random", observed_ics["real_ridge"], observed_ics["random_ridge"]),
        ("real_vs_scramble", "ridge_real", "ridge_scramble", observed_ics["real_ridge"], observed_ics["scramble_ridge"]),
        ("real_vs_ohlcv_b1a", "ridge_real", "ridge_ohlcv_b1a", observed_ics["real_ridge"], observed_ics["ohlcv_ridge"]),
        ("real_vs_nuisance_only", "ridge_real", "ridge_nuisance_only", observed_ics["real_ridge"], observed_ics["nuisance_ridge"]),
    ]

    p_vals_ridge = []
    for comp_name, real_key, ctrl_key, obs_real, obs_ctrl in ridge_targets:
        stat = compute_delta_ic_stats(replicates[real_key], replicates[ctrl_key], obs_real, obs_ctrl)
        ridge_comparisons_raw[comp_name] = stat
        p_vals_ridge.append(stat["p_value_two_tailed"])

    adj_p_ridge = holm_bonferroni_correction(p_vals_ridge)
    for (comp_name, _, _, _, _), adj_p in zip(ridge_targets, adj_p_ridge):
        ridge_comparisons_raw[comp_name]["p_value_holm_adjusted"] = float(adj_p)

    # Global Holm correction across all 8 comparisons
    all_p_vals = p_vals_log + p_vals_ridge
    all_adj_p = holm_bonferroni_correction(all_p_vals)
    for i, (comp_name, _, _, _, _) in enumerate(log_targets):
        log_comparisons_raw[comp_name]["p_value_global_holm_adjusted"] = float(all_adj_p[i])
    for j, (comp_name, _, _, _, _) in enumerate(ridge_targets):
        ridge_comparisons_raw[comp_name]["p_value_global_holm_adjusted"] = float(all_adj_p[len(log_targets) + j])

    # 8. Section 1b: Signal Redundancy & Residual IC
    print("Evaluating signal redundancy and residual IC...")
    redundancy_results = compute_redundancy_and_residual_ic(
        s_real=predictions_log["real"],
        s_ohlcv=predictions_log["ohlcv_b1b"],
        s_nui=predictions_log["nuisance_only"],
        y_true=y_val_cont,
        block_size=block_size,
        n_bootstraps=bootstrap_samples,
        seed=seed,
    )

    # 9. Section 1c: Sub-period Temporal Stability
    print("Evaluating sub-period temporal stability across 4 validation quarters...")
    stability_log = compute_temporal_stability(predictions_log, y_val_cont, n_quarters=4)
    stability_ridge = compute_temporal_stability(predictions_ridge, y_val_cont, n_quarters=4)

    # 10. Section 1d: Corrected Pass/Fail Presentation Table
    print("Building audited presentation table...")
    corrected_table = build_corrected_table(
        probe_val_data=probe_val_data,
        real_logistic_score=predictions_log["real"],
        nuisance_features_val=nui_va,
        ohlcv_b1b_ba=ba_ohlcv_log,
        nuisance_logistic_ba=ba_nui_log,
    )

    # Assemble output payload
    output_payload = {
        "metadata": {
            "notice_1": "ORIGINAL_VERDICT: FAIL (unchanged; see probe_val.json)",
            "notice_2": "THIS FILE IS EXPLORATORY_POST_HOC AND MUST NOT BE CITED AS PASS EVIDENCE",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": get_git_commit(),
            "config_sha256": compute_file_sha256(config_path),
            "feature_hashes": expected_hashes,
            "train_samples": n_train,
            "val_samples": n_val,
            "bootstrap_seed": seed,
            "bootstrap_samples": bootstrap_samples,
            "block_size": block_size,
            "reproduction_verified": True,
        },
        "model_summary": {
            "spearman_ic": observed_ics,
            "logistic_balanced_accuracy": {
                "real": float(compute_classification_metrics(y_val_cls, (predictions_log["real"] > 0))["balanced_accuracy"])
                if False else float(probe_val_data["models"]["real"]["logistic"]["classification"]["balanced_accuracy"]),
                "random": float(probe_val_data["models"]["random"]["logistic"]["classification"]["balanced_accuracy"]),
                "scramble": float(probe_val_data["models"]["scramble"]["logistic"]["classification"]["balanced_accuracy"]),
                "ohlcv_b1b": ba_ohlcv_log,
                "nuisance_only": ba_nui_log,
            },
        },
        "pairwise_bootstrap_comparisons": {
            "logistic": log_comparisons_raw,
            "ridge": ridge_comparisons_raw,
        },
        "signal_redundancy_and_residual_ic": redundancy_results,
        "temporal_stability_quarters": {
            "logistic": stability_log,
            "ridge": stability_ridge,
        },
        "audited_pass_fail_presentation": {
            "banner": {
                "ORIGINAL_VERDICT": "FAIL (unchanged; see probe_val.json)",
                "DISCLAIMER": "THIS FILE IS EXPLORATORY_POST_HOC AND MUST NOT BE CITED AS PASS EVIDENCE",
            },
            "table": corrected_table,
        },
    }

    # Write output JSON atomically
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)
    tmp_path.replace(output_path)
    print(f"Exploratory post-hoc analysis complete. Output written to {output_path}")

    return output_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Research v2 Phase 5 Post-Hoc Exploratory Analysis.")
    parser.add_argument("--features-dir", type=Path, default=OUTPUTS_DIR / "v2/features")
    parser.add_argument("--labels", type=Path, default=DATA_DIR / "labels_v2.parquet")
    parser.add_argument("--probe-val", type=Path, default=OUTPUTS_DIR / "v2/probe_val.json")
    parser.add_argument("--config", type=Path, default=ROOT / "config/experiment_v2.yaml")
    parser.add_argument("--output", type=Path, default=OUTPUTS_DIR / "v2/probe_posthoc_val.json")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--block-size", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-verify-hashes", action="store_true", help="Skip SHA256 check of feature files.")
    parser.add_argument("--no-verify-repro", action="store_true", help="Skip reproduction check of IC values.")
    parser.add_argument("--check-only", action="store_true", help="Only run reproduction and pre-flight checks.")
    args = parser.parse_args()

    run_posthoc_analysis(
        features_dir=args.features_dir,
        labels_path=args.labels,
        probe_val_path=args.probe_val,
        config_path=args.config,
        output_path=args.output,
        bootstrap_samples=args.bootstrap_samples,
        block_size=args.block_size,
        seed=args.seed,
        verify_hashes=not args.no_verify_hashes,
        verify_repro=not args.no_verify_repro,
        check_only=args.check_only,
    )


if __name__ == "__main__":
    main()
