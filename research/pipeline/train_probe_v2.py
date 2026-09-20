"""Research v2 Phase 5: Frozen Connectome Supervised Linear Probe.

Trains Ridge and Multinomial Logistic Decoders on frozen Connectome readout features
and evaluates predictive power on the Validation set against matched controls:
- C0: Real Intact Drosophila Connectome (FlyWire v783)
- C1: Matched Random Graph (permuted weights, random endpoints)
- C2: Degree-Preserved Scramble Graph (double-edge swapped)
- Control (a): Nuisance-only Probe (7 renderer features from v2_health.py)
- Control (b): Non-Connectome OHLCV Baselines (B1a Ridge, B1b Logistic)

Strict Governance:
- Train-only feature standardization (mean/std stored with SHA256).
- Alpha selected strictly by chronological expanding-window CV on Train.
- Split whitelist: only 'train' and 'val'. dev_test_v1 and holdout are rejected.
- Paired moving block bootstrap for Delta IC confidence intervals.
- Circular shift block permutation test for IC significance.
- Threshold pass/fail evaluation against experiment_v2.yaml pre-registered criteria.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import yaml

import numpy as np
import pandas as pd
import scipy.stats

from research.pipeline.baselines_v2 import (
    ACTIONS,
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
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"


def paired_block_bootstrap_delta_ci(
    y_true: np.ndarray,
    y_pred_real: np.ndarray,
    y_pred_ctrl: np.ndarray,
    block_size: int = 24,
    n_bootstraps: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Paired Moving Block Bootstrap for Delta IC = IC(real) - IC(control)."""
    n = len(y_true)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_size))
    delta_replicates = np.empty(n_bootstraps, dtype=np.float64)

    for b in range(n_bootstraps):
        starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        idx = (starts[:, None] + np.arange(block_size)).ravel()[:n]

        sub_y = y_true[idx]
        sub_real = y_pred_real[idx]
        sub_ctrl = y_pred_ctrl[idx]

        r_real = scipy.stats.spearmanr(sub_real, sub_y)
        ic_real = float(r_real.statistic if hasattr(r_real, "statistic") else r_real[0])
        if np.isnan(ic_real):
            ic_real = 0.0

        r_ctrl = scipy.stats.spearmanr(sub_ctrl, sub_y)
        ic_ctrl = float(r_ctrl.statistic if hasattr(r_ctrl, "statistic") else r_ctrl[0])
        if np.isnan(ic_ctrl):
            ic_ctrl = 0.0

        delta_replicates[b] = ic_real - ic_ctrl

    lower = float(np.percentile(delta_replicates, 100.0 * (alpha / 2.0)))
    upper = float(np.percentile(delta_replicates, 100.0 * (1.0 - alpha / 2.0)))
    return lower, upper


def circular_shift_permutation_p_value(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    n_permutations: int = 1000,
    min_shift: int = 24,
    seed: int = 42,
) -> float:
    """Circular shift block permutation test for IC(real) > 0.

    Preserves the full autocorrelation structure of both the predictor and the target
    while breaking the true temporal alignment.
    """
    n = len(y_true)
    res_obs = scipy.stats.spearmanr(y_pred, y_true)
    ic_obs = float(res_obs.statistic if hasattr(res_obs, "statistic") else res_obs[0])
    if np.isnan(ic_obs):
        return 1.0

    rng = np.random.default_rng(seed)
    # Uniformly pick shifts that are at least min_shift away from 0 and n
    eff_min_shift = min(min_shift, max(1, n // 4))
    eff_max_shift = n - eff_min_shift
    if eff_min_shift >= eff_max_shift:
        eff_min_shift = 1
        eff_max_shift = max(2, n)
    shifts = rng.integers(eff_min_shift, eff_max_shift, size=n_permutations)
    count_geq = 0

    for s in shifts:
        y_shifted = np.roll(y_true, s)
        r = scipy.stats.spearmanr(y_pred, y_shifted)
        ic_perm = float(r.statistic if hasattr(r, "statistic") else r[0])
        if not np.isnan(ic_perm) and ic_perm >= ic_obs:
            count_geq += 1

    return float((count_geq + 1) / (n_permutations + 1))


def extract_nuisance_features_split(
    samples_split: pd.DataFrame,
    raw: pd.DataFrame,
    images_mmap: np.ndarray,
    chunk_size: int = 512,
) -> pd.DataFrame:
    """Compute 7 renderer nuisance features for a given sample subset."""
    from research.run_experiment import get_windows
    from research.v2_health import nuisance

    dfs = []
    n = len(samples_split)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        sub_samples = samples_split.iloc[start:end]
        chunk_imgs = np.asarray(images_mmap[sub_samples["image_idx"].to_numpy(dtype=np.int64)])
        chunk_wins = get_windows(sub_samples, raw)
        nui_df = nuisance(chunk_imgs, chunk_wins)
        dfs.append(nui_df)

    return pd.concat(dfs, ignore_index=True)


def train_and_eval_probe(
    X_train_raw: np.ndarray,
    y_train_cont: np.ndarray,
    y_train_cls: np.ndarray,
    X_val_raw: np.ndarray,
    y_val_cont: np.ndarray,
    y_val_cls: np.ndarray,
    cost_threshold: float,
    cost_fee: float,
    cost_slip: float,
    feature_names: list[str] | None = None,
    bootstrap_samples: int = 1000,
) -> dict:
    """Train Ridge and Logistic probes and evaluate on Validation."""
    if feature_names is None:
        feature_names = [f"f_{i}" for i in range(X_train_raw.shape[1])]

    # 1. Train-only standardization
    norm_artifact = compute_train_standardization(X_train_raw, feature_names)
    X_train = apply_standardization(X_train_raw, norm_artifact)
    X_val = apply_standardization(X_val_raw, norm_artifact)

    # 2. Ridge Decoder
    best_alpha = select_ridge_alpha_timeseries_cv(X_train, y_train_cont)
    ridge = PureNumpyRidge(alpha=best_alpha).fit(X_train, y_train_cont)
    pred_cont_ridge = ridge.predict(X_val)

    pred_cls_ridge = np.full(len(pred_cont_ridge), "HOLD", dtype=object)
    pred_cls_ridge[pred_cont_ridge > cost_threshold] = "BUY"
    pred_cls_ridge[pred_cont_ridge < -cost_threshold] = "SELL"

    cont_metrics_ridge = compute_continuous_metrics(y_val_cont, pred_cont_ridge)
    cls_metrics_ridge = compute_classification_metrics(y_val_cls, pred_cls_ridge)
    trade_ridge = simulate_trading_spot_long_only(pred_cls_ridge, y_val_cont, cost_fee, cost_slip)

    ic_ci_ridge = moving_block_bootstrap_ci(
        y_val_cont, pred_cont_ridge, metric_type="ic", n_bootstraps=bootstrap_samples, seed=42
    )
    ba_ci_ridge = moving_block_bootstrap_ci(
        y_val_cls, pred_cls_ridge, metric_type="ba", n_bootstraps=bootstrap_samples, seed=42
    )

    # 3. Logistic Decoder (3-class softmax)
    pred_cont_log, pred_cls_log = fit_predict_b1b_logistic(X_train, y_train_cls, X_val)
    cont_metrics_log = compute_continuous_metrics(y_val_cont, pred_cont_log)
    cls_metrics_log = compute_classification_metrics(y_val_cls, pred_cls_log)
    trade_log = simulate_trading_spot_long_only(pred_cls_log, y_val_cont, cost_fee, cost_slip)

    ic_ci_log = moving_block_bootstrap_ci(
        y_val_cont, pred_cont_log, metric_type="ic", n_bootstraps=bootstrap_samples, seed=42
    )
    ba_ci_log = moving_block_bootstrap_ci(
        y_val_cls, pred_cls_log, metric_type="ba", n_bootstraps=bootstrap_samples, seed=42
    )

    return {
        "normalization_sha256": norm_artifact["sha256"],
        "ridge": {
            "best_alpha": best_alpha,
            "continuous": {**cont_metrics_ridge, "ic_ci_95": list(ic_ci_ridge)},
            "classification": {**cls_metrics_ridge, "balanced_accuracy_ci_95": list(ba_ci_ridge)},
            "trading": trade_ridge,
            "pred_cont": pred_cont_ridge,
            "pred_cls": pred_cls_ridge,
        },
        "logistic": {
            "continuous": {**cont_metrics_log, "ic_ci_95": list(ic_ci_log)},
            "classification": {**cls_metrics_log, "balanced_accuracy_ci_95": list(ba_ci_log)},
            "trading": trade_log,
            "pred_cont": pred_cont_log,
            "pred_cls": pred_cls_log,
        },
    }


def compute_renderer_shortcut_r2_v2(
    X_nui_train: np.ndarray,
    s_real_train: np.ndarray,
    X_nui_val: np.ndarray,
    s_real_val: np.ndarray,
) -> float:
    """Compute out-of-sample R^2 of real score regressed on 7 nuisance features.

    Fits OLS on Train: s_real_train ~ nuisance_train (with intercept).
    Evaluates out-of-sample R^2 on Val: 1 - SS_res / SS_tot (allows negative values).
    """
    n_tr = len(s_real_train)
    n_va = len(s_real_val)
    ones_tr = np.ones((n_tr, 1), dtype=np.float64)
    ones_va = np.ones((n_va, 1), dtype=np.float64)

    Z_tr = np.column_stack([ones_tr, X_nui_train])
    w = np.linalg.lstsq(Z_tr, s_real_train, rcond=None)[0]

    Z_va = np.column_stack([ones_va, X_nui_val])
    s_pred_va = Z_va @ w

    ss_res = float(np.sum((s_real_val - s_pred_va) ** 2))
    ss_tot = float(np.sum((s_real_val - np.mean(s_real_val)) ** 2))
    if ss_tot <= 1e-12:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def evaluate_thresholds_table(
    real_ridge: dict | None,
    real_logistic: dict | None,
    comparisons: dict | None,
    nuisance_r2: float | None,
    thresholds_cfg: dict,
    strict_gates: bool = False,
) -> list[dict]:
    """Generate pass/fail evaluation table for all pre-registered thresholds.

    When strict_gates=False (default, legacy mode):
        Preserves legacy pass/fail semantics and outputs.
    When strict_gates=True (audited strict mode):
        - trade_count == 0 trading rows marked as N/A_NO_TRADES (passed=False).
        - Missing models or comparisons output NOT_EVALUATED (passed=False).
        - Overall evaluation requires explicit conjunction of all required gates.
    """
    if comparisons is None:
        comparisons = {}

    # Strict mode handling when required models are missing
    if strict_gates and (real_ridge is None or real_logistic is None):
        all_keys = [
            ("primary_metric", "Validation set primary target Spearman IC minimum threshold", ">= 0.02", "§8.1"),
            ("min_practical_effect", "Minimum practical effect size (IC)", ">= 0.02", "§8.3, §12"),
            ("min_practical_effect_balanced_acc", "Minimum practical effect for balanced accuracy over majority/random (delta >= 0.02)", ">= 0.02", "§8.3, §12"),
            ("connectome_vs_matched_control_ic_delta_min (vs Random)", "Minimum Spearman IC increment of Connectome over Random control", ">= 0.01", "§6 Phase 5, §8.3, §12"),
            ("connectome_vs_matched_control_ic_delta_min (vs Scramble)", "Minimum Spearman IC increment of Connectome over Scramble control", ">= 0.01", "§6 Phase 5, §8.3, §12"),
            ("connectome_vs_matched_control_p_max", "Maximum p-value from permutation test", "<= 0.01", "§6 Phase 5, §8.3, §12"),
            ("connectome_vs_ohlcv_baseline_ic_delta_min", "Minimum Spearman IC increment of Connectome over OHLCV baseline", ">= 0.01", "§6 Phase 4/5, §12"),
            ("renderer_shortcut_r2_max", "Maximum R^2 of margin explained by mean brightness, nonzero pixels, or renderer features", "<= 0.05", "§5.3, §12"),
            ("trading_net_return_min", "Minimum net return after cost (>= 0.0)", ">= 0.0", "§8.2, §12"),
            ("trading_sharpe_min", "Minimum strategy Sharpe ratio after cost", ">= 0.5", "§8.2, §12"),
            ("trading_max_drawdown_max", "Maximum allowable strategy drawdown (<= 15%)", "<= 0.15", "§8.2, §12"),
        ]
        return [
            {
                "threshold_key": k,
                "description": desc,
                "observed_value": None,
                "required_value": req,
                "status": "NOT_EVALUATED",
                "spec_ref": sref,
                "passed": False,
                "verdict_flag": "NOT_EVALUATED",
                "audit_note": "Required model(s) missing from evaluation",
            }
            for k, desc, req, sref in all_keys
        ]

    if real_ridge is None or real_logistic is None:
        return []

    rows = []

    # 1. primary_metric (Spearman IC)
    ic_val = real_ridge["continuous"]["spearman_ic"]
    th_ic = thresholds_cfg.get("primary_metric", {})
    th_val_ic = th_ic.get("value", 0.02)
    rows.append({
        "threshold_key": "primary_metric",
        "description": "Validation set primary target Spearman IC minimum threshold",
        "observed_value": ic_val,
        "required_value": f">= {th_val_ic}",
        "status": th_ic.get("status", "PROPOSED_NEEDS_USER_CONFIRMATION"),
        "spec_ref": th_ic.get("spec_ref", "§8.1"),
        "passed": bool(ic_val >= th_val_ic),
    })

    # 2. min_practical_effect (IC)
    th_pe = thresholds_cfg.get("min_practical_effect", {})
    th_pe_val = th_pe.get("value", 0.02)
    rows.append({
        "threshold_key": "min_practical_effect",
        "description": "Minimum practical effect size (IC)",
        "observed_value": ic_val,
        "required_value": f">= {th_pe_val}",
        "status": th_pe.get("status", "PROPOSED_NEEDS_USER_CONFIRMATION"),
        "spec_ref": th_pe.get("spec_ref", "§8.3, §12"),
        "passed": bool(ic_val >= th_pe_val),
    })

    # 3. min_practical_effect_balanced_acc
    ba_val = real_logistic["classification"]["balanced_accuracy"]
    ba_delta = ba_val - (1.0 / 3.0)
    th_ba = thresholds_cfg.get("min_practical_effect_balanced_acc", {})
    th_ba_val = th_ba.get("value", 0.02)
    rows.append({
        "threshold_key": "min_practical_effect_balanced_acc",
        "description": "Minimum practical effect for balanced accuracy over majority/random (delta >= 0.02)",
        "baseline_definition": "1/3 (uniform 3-class chance)",
        "observed_value": ba_delta,
        "required_value": f">= {th_ba_val}",
        "status": th_ba.get("status", "PROPOSED_NEEDS_USER_CONFIRMATION"),
        "spec_ref": th_ba.get("spec_ref", "§8.3, §12"),
        "passed": bool(ba_delta >= th_ba_val),
    })

    # 4. connectome_vs_matched_control_ic_delta_min (Random control)
    has_rand = "delta_ic_vs_random" in comparisons
    delta_rand = comparisons.get("delta_ic_vs_random", {}).get("delta_ic", 0.0) if comparisons else 0.0
    th_mc_ic = thresholds_cfg.get("connectome_vs_matched_control_ic_delta_min", {})
    th_mc_val = th_mc_ic.get("value", 0.01)
    if strict_gates and not has_rand:
        rows.append({
            "threshold_key": "connectome_vs_matched_control_ic_delta_min (vs Random)",
            "description": "Minimum Spearman IC increment of Connectome over Random control",
            "observed_value": None,
            "required_value": f">= {th_mc_val}",
            "status": "NOT_EVALUATED",
            "spec_ref": th_mc_ic.get("spec_ref", "§6 Phase 5, §8.3, §12"),
            "passed": False,
            "verdict_flag": "NOT_EVALUATED",
        })
    else:
        rows.append({
            "threshold_key": "connectome_vs_matched_control_ic_delta_min (vs Random)",
            "description": "Minimum Spearman IC increment of Connectome over Random control",
            "observed_value": delta_rand,
            "required_value": f">= {th_mc_val}",
            "status": th_mc_ic.get("status", "PROPOSED_NEEDS_USER_CONFIRMATION"),
            "spec_ref": th_mc_ic.get("spec_ref", "§6 Phase 5, §8.3, §12"),
            "passed": bool(delta_rand >= th_mc_val),
        })

    # 5. connectome_vs_matched_control_ic_delta_min (Scramble control)
    has_scram = "delta_ic_vs_scramble" in comparisons
    delta_scram = comparisons.get("delta_ic_vs_scramble", {}).get("delta_ic", 0.0) if comparisons else 0.0
    if strict_gates and not has_scram:
        rows.append({
            "threshold_key": "connectome_vs_matched_control_ic_delta_min (vs Scramble)",
            "description": "Minimum Spearman IC increment of Connectome over Scramble control",
            "observed_value": None,
            "required_value": f">= {th_mc_val}",
            "status": "NOT_EVALUATED",
            "spec_ref": th_mc_ic.get("spec_ref", "§6 Phase 5, §8.3, §12"),
            "passed": False,
            "verdict_flag": "NOT_EVALUATED",
        })
    else:
        rows.append({
            "threshold_key": "connectome_vs_matched_control_ic_delta_min (vs Scramble)",
            "description": "Minimum Spearman IC increment of Connectome over Scramble control",
            "observed_value": delta_scram,
            "required_value": f">= {th_mc_val}",
            "status": th_mc_ic.get("status", "PROPOSED_NEEDS_USER_CONFIRMATION"),
            "spec_ref": th_mc_ic.get("spec_ref", "§6 Phase 5, §8.3, §12"),
            "passed": bool(delta_scram >= th_mc_val),
        })

    # 6. connectome_vs_matched_control_p_max
    has_p = "real_ic_permutation_p_value" in comparisons
    p_perm = comparisons.get("real_ic_permutation_p_value", 1.0) if comparisons else 1.0
    th_p = thresholds_cfg.get("connectome_vs_matched_control_p_max", {})
    th_p_val = th_p.get("value", 0.01)
    if strict_gates and not has_p:
        rows.append({
            "threshold_key": "connectome_vs_matched_control_p_max",
            "description": "Maximum p-value from permutation test",
            "observed_value": None,
            "required_value": f"<= {th_p_val}",
            "status": "NOT_EVALUATED",
            "spec_ref": th_p.get("spec_ref", "§6 Phase 5, §8.3, §12"),
            "passed": False,
            "verdict_flag": "NOT_EVALUATED",
        })
    else:
        rows.append({
            "threshold_key": "connectome_vs_matched_control_p_max",
            "description": "Maximum p-value from permutation test",
            "observed_value": p_perm,
            "required_value": f"<= {th_p_val}",
            "status": th_p.get("status", "PROPOSED_NEEDS_USER_CONFIRMATION"),
            "spec_ref": th_p.get("spec_ref", "§6 Phase 5, §8.3, §12"),
            "passed": bool(p_perm <= th_p_val),
        })

    # 7. connectome_vs_ohlcv_baseline_ic_delta_min
    has_ob = "delta_ic_vs_ohlcv_baseline" in comparisons
    delta_base = comparisons.get("delta_ic_vs_ohlcv_baseline", {}).get("delta_ic", 0.0) if comparisons else 0.0
    th_ob = thresholds_cfg.get("connectome_vs_ohlcv_baseline_ic_delta_min", {})
    th_ob_val = th_ob.get("value", 0.01)
    if strict_gates and not has_ob:
        rows.append({
            "threshold_key": "connectome_vs_ohlcv_baseline_ic_delta_min",
            "description": "Minimum Spearman IC increment of Connectome over OHLCV baseline",
            "observed_value": None,
            "required_value": f">= {th_ob_val}",
            "status": "NOT_EVALUATED",
            "spec_ref": th_ob.get("spec_ref", "§6 Phase 4/5, §12"),
            "passed": False,
            "verdict_flag": "NOT_EVALUATED",
        })
    else:
        rows.append({
            "threshold_key": "connectome_vs_ohlcv_baseline_ic_delta_min",
            "description": "Minimum Spearman IC increment of Connectome over OHLCV baseline",
            "observed_value": delta_base,
            "required_value": f">= {th_ob_val}",
            "status": th_ob.get("status", "PROPOSED_NEEDS_USER_CONFIRMATION"),
            "spec_ref": th_ob.get("spec_ref", "§6 Phase 4/5, §12"),
            "passed": bool(delta_base >= th_ob_val),
        })

    # 8. renderer_shortcut_r2_max
    th_r2 = thresholds_cfg.get("renderer_shortcut_r2_max", {})
    th_r2_val = th_r2.get("value", 0.05)
    r2_eval = nuisance_r2 if nuisance_r2 is not None else 0.0
    rows.append({
        "threshold_key": "renderer_shortcut_r2_max",
        "description": "Maximum R^2 of margin explained by mean brightness, nonzero pixels, or renderer features",
        "observed_value": r2_eval,
        "required_value": f"<= {th_r2_val}",
        "status": th_r2.get("status", "PROPOSED_NEEDS_USER_CONFIRMATION"),
        "spec_ref": th_r2.get("spec_ref", "§5.3, §12"),
        "passed": bool(r2_eval <= th_r2_val),
    })

    # Trading metrics
    net_ret = real_ridge["trading"]["net_return"]
    sharpe_val = real_ridge["trading"]["annualized_sharpe"]
    dd_val = real_ridge["trading"]["max_drawdown"]
    trade_cnt = real_ridge.get("trading", {}).get("trade_count", 0)

    # 9. trading_net_return_min
    th_tr = thresholds_cfg.get("trading_net_return_min", {})
    th_tr_val = th_tr.get("value", 0.0)
    if strict_gates and trade_cnt == 0:
        rows.append({
            "threshold_key": "trading_net_return_min",
            "description": "Minimum net return after cost (>= 0.0)",
            "observed_value": net_ret,
            "required_value": f">= {th_tr_val}",
            "status": "N/A_NO_TRADES",
            "spec_ref": th_tr.get("spec_ref", "§8.2, §12"),
            "passed": False,
            "verdict_flag": "N/A_NO_TRADES",
            "audit_note": "Zero trades executed; passive hold does not satisfy trading requirement.",
        })
    else:
        rows.append({
            "threshold_key": "trading_net_return_min",
            "description": "Minimum net return after cost (>= 0.0)",
            "observed_value": net_ret,
            "required_value": f">= {th_tr_val}",
            "status": th_tr.get("status", "PROPOSED_NEEDS_USER_CONFIRMATION"),
            "spec_ref": th_tr.get("spec_ref", "§8.2, §12"),
            "passed": bool(net_ret >= th_tr_val),
        })

    # 10. trading_sharpe_min
    th_sh = thresholds_cfg.get("trading_sharpe_min", {})
    th_sh_val = th_sh.get("value", 0.5)
    if strict_gates and trade_cnt == 0:
        rows.append({
            "threshold_key": "trading_sharpe_min",
            "description": "Minimum strategy Sharpe ratio after cost",
            "observed_value": sharpe_val,
            "required_value": f">= {th_sh_val}",
            "status": "N/A_NO_TRADES",
            "spec_ref": th_sh.get("spec_ref", "§8.2, §12"),
            "passed": False,
            "verdict_flag": "N/A_NO_TRADES",
            "audit_note": "Zero trades executed; passive hold does not satisfy trading requirement.",
        })
    else:
        rows.append({
            "threshold_key": "trading_sharpe_min",
            "description": "Minimum strategy Sharpe ratio after cost",
            "observed_value": sharpe_val,
            "required_value": f">= {th_sh_val}",
            "status": th_sh.get("status", "PROPOSED_NEEDS_USER_CONFIRMATION"),
            "spec_ref": th_sh.get("spec_ref", "§8.2, §12"),
            "passed": bool(sharpe_val >= th_sh_val),
        })

    # 11. trading_max_drawdown_max
    th_dd = thresholds_cfg.get("trading_max_drawdown_max", {})
    th_dd_val = th_dd.get("value", 0.15)
    if strict_gates and trade_cnt == 0:
        rows.append({
            "threshold_key": "trading_max_drawdown_max",
            "description": "Maximum allowable strategy drawdown (<= 15%)",
            "observed_value": dd_val,
            "required_value": f"<= {th_dd_val}",
            "status": "N/A_NO_TRADES",
            "spec_ref": th_dd.get("spec_ref", "§8.2, §12"),
            "passed": False,
            "verdict_flag": "N/A_NO_TRADES",
            "audit_note": "Zero trades executed; passive hold does not satisfy trading requirement.",
        })
    else:
        rows.append({
            "threshold_key": "trading_max_drawdown_max",
            "description": "Maximum allowable strategy drawdown (<= 15%)",
            "observed_value": dd_val,
            "required_value": f"<= {th_dd_val}",
            "status": th_dd.get("status", "PROPOSED_NEEDS_USER_CONFIRMATION"),
            "spec_ref": th_dd.get("spec_ref", "§8.2, §12"),
            "passed": bool(dd_val <= th_dd_val),
        })

    return rows


def run_probe_v2(
    features_dir: Path | str = OUTPUTS_DIR / "v2/features",
    labels_path: Path | str = DATA_DIR / "labels_v2.parquet",
    config_path: Path | str = ROOT / "config/experiment_v2.yaml",
    baselines_path: Path | str = OUTPUTS_DIR / "v2/baselines_val.json",
    output_path: Path | str = OUTPUTS_DIR / "v2/probe_val.json",
    bootstrap_samples: int = 1000,
    permutations: int = 1000,
    strict_gates: bool = False,
) -> dict:
    """Execute Phase 5 Frozen Connectome Linear Probe analysis."""
    features_dir = Path(features_dir)
    labels_path = Path(labels_path)
    config_path = Path(config_path)
    baselines_path = Path(baselines_path)
    output_path = Path(output_path)

    # 1. Load config and labels
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    labels = pd.read_parquet(
        labels_path,
        filters=[("split", "in", ["train", "val"])],
    )
    validate_split_name("train")
    validate_split_name("val")

    cost_fee = float(config["cost"]["cost_fee_per_side"])
    cost_slip = float(config["cost"]["cost_slippage"])
    cost_threshold = float(2.0 * cost_fee + cost_slip)

    # Filter train and val, drop NaNs
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

    feature_hashes = {}
    variant_results = {}

    # 2. Evaluate available graph variants (real, random, scramble)
    for variant in ["real", "random", "scramble"]:
        tr_file = features_dir / f"{variant}_train.npy"
        va_file = features_dir / f"{variant}_val.npy"
        if not tr_file.exists() or not va_file.exists():
            print(f"Skipping variant {variant}: missing {tr_file.name} or {va_file.name}")
            continue

        feature_hashes[f"{variant}_train"] = compute_file_sha256(tr_file)
        feature_hashes[f"{variant}_val"] = compute_file_sha256(va_file)

        X_tr = np.load(tr_file)
        X_va = np.load(va_file)

        # Slice to matched valid samples if full split extracted
        if len(X_tr) == len(labels_split[labels_split == "train"]):
            X_tr = X_tr[train_mask[labels_split == "train"]]
        if len(X_va) == len(labels_split[labels_split == "val"]):
            X_va = X_va[val_mask[labels_split == "val"]]

        print(f"Training decoders for variant {variant} (train={len(X_tr)}, val={len(X_va)})...")
        eval_res = train_and_eval_probe(
            X_train_raw=X_tr,
            y_train_cont=y_train_cont[:len(X_tr)],
            y_train_cls=y_train_cls[:len(X_tr)],
            X_val_raw=X_va,
            y_val_cont=y_val_cont[:len(X_va)],
            y_val_cls=y_val_cls[:len(X_va)],
            cost_threshold=cost_threshold,
            cost_fee=cost_fee,
            cost_slip=cost_slip,
            bootstrap_samples=bootstrap_samples,
        )
        variant_results[variant] = eval_res

    # 3. Control (a): Nuisance-only Probe
    print("Evaluating Control (a): Nuisance-only probe (7 renderer features)...")
    samples = pd.read_parquet(
        DATA_DIR / "samples.parquet",
        filters=[("split", "in", ["train", "val"])],
    )
    raw = pd.read_parquet(DATA_DIR / "raw_ohlcv.parquet")
    images_mmap = np.load(DATA_DIR / "images.npy", mmap_mode="r")
    n_tr_used = n_train
    n_va_used = n_val
    if variant_results:
        first_var = next(iter(variant_results))
        if "pred_cont" in variant_results[first_var]["ridge"]:
            n_va_used = len(variant_results[first_var]["ridge"]["pred_cont"])
        sample_tr_file = features_dir / f"{first_var}_train.npy"
        if sample_tr_file.exists():
            n_tr_used = len(np.load(sample_tr_file))

    samples_tr = samples[samples["split"] == "train"].iloc[:n_tr_used]
    samples_va = samples[samples["split"] == "val"].iloc[:n_va_used]

    nui_tr = extract_nuisance_features_split(samples_tr, raw, images_mmap).to_numpy(dtype=np.float64)
    nui_va = extract_nuisance_features_split(samples_va, raw, images_mmap).to_numpy(dtype=np.float64)
    nuisance_res = train_and_eval_probe(
        X_train_raw=nui_tr,
        y_train_cont=y_train_cont[:len(nui_tr)],
        y_train_cls=y_train_cls[:len(nui_tr)],
        X_val_raw=nui_va,
        y_val_cont=y_val_cont[:len(nui_va)],
        y_val_cls=y_val_cls[:len(nui_va)],
        cost_threshold=cost_threshold,
        cost_fee=cost_fee,
        cost_slip=cost_slip,
        feature_names=["mean_brightness", "nonzero_pixel_ratio", "total_pixel_energy",
                       "price_vertical_range", "n_up_candles", "n_down_candles", "volatility"],
        bootstrap_samples=bootstrap_samples,
    )

    # 4. Control (b): OHLCV Baseline Results
    baselines_info = {}
    best_ohlcv_ic = 0.0
    if baselines_path.exists():
        with open(baselines_path, "r", encoding="utf-8") as f:
            base_data = json.load(f)
        b1a_ic = base_data.get("models", {}).get("B1a_ridge", {}).get("continuous", {}).get("spearman_ic", 0.0)
        b1b_ic = base_data.get("models", {}).get("B1b_logistic", {}).get("continuous", {}).get("spearman_ic", 0.0)
        best_ohlcv_ic = max(b1a_ic, b1b_ic)
        baselines_info = {
            "B1a_ridge_ic": b1a_ic,
            "B1b_logistic_ic": b1b_ic,
            "best_ohlcv_ic": best_ohlcv_ic,
        }

    # 5. Comparisons & Permutation Test for real Connectome
    comparisons = {}
    real_ridge = variant_results.get("real", {}).get("ridge")
    real_logistic = variant_results.get("real", {}).get("logistic")

    if real_ridge is not None:
        y_va_eval = y_val_cont[:len(real_ridge["pred_cont"])]
        real_pred_cont = real_ridge["pred_cont"]

        # Circular shift permutation test
        print(f"Running circular shift permutation test ({permutations} iterations)...")
        p_val_perm = circular_shift_permutation_p_value(
            y_pred=real_pred_cont,
            y_true=y_va_eval,
            n_permutations=permutations,
        )
        comparisons["real_ic_permutation_p_value"] = p_val_perm

        # Compare vs Random
        if "random" in variant_results:
            rand_pred = variant_results["random"]["ridge"]["pred_cont"]
            d_ic = float(real_ridge["continuous"]["spearman_ic"] - variant_results["random"]["ridge"]["continuous"]["spearman_ic"])
            d_ci = paired_block_bootstrap_delta_ci(y_va_eval, real_pred_cont, rand_pred, n_bootstraps=bootstrap_samples)
            comparisons["delta_ic_vs_random"] = {"delta_ic": d_ic, "delta_ic_ci_95": list(d_ci)}

        # Compare vs Scramble
        if "scramble" in variant_results:
            scram_pred = variant_results["scramble"]["ridge"]["pred_cont"]
            d_ic = float(real_ridge["continuous"]["spearman_ic"] - variant_results["scramble"]["ridge"]["continuous"]["spearman_ic"])
            d_ci = paired_block_bootstrap_delta_ci(y_va_eval, real_pred_cont, scram_pred, n_bootstraps=bootstrap_samples)
            comparisons["delta_ic_vs_scramble"] = {"delta_ic": d_ic, "delta_ic_ci_95": list(d_ci)}

        # Compare vs Best OHLCV baseline
        d_base = float(real_ridge["continuous"]["spearman_ic"] - best_ohlcv_ic)
        comparisons["delta_ic_vs_ohlcv_baseline"] = {"delta_ic": d_base}

    # R^2 of real margin explained by nuisance features (legacy version)
    nuisance_r2 = float(nuisance_res["ridge"]["continuous"].get("spearman_ic", 0.0)**2)

    # v2: Out-of-sample R^2 of real score regressed on 7 nuisance features
    renderer_r2_v2 = None
    if "real" in variant_results and "pred_cont" in real_ridge:
        s_val_real = real_ridge["pred_cont"]
        ones_va = np.ones((len(s_val_real), 1), dtype=np.float64)
        Z_va = np.column_stack([ones_va, nui_va])
        w_va = np.linalg.lstsq(Z_va, s_val_real, rcond=None)[0]
        res_va = s_val_real - Z_va @ w_va
        s_var_va = np.sum((s_val_real - np.mean(s_val_real))**2)
        renderer_r2_v2 = float(1.0 - np.sum(res_va**2) / s_var_va) if s_var_va > 1e-12 else 0.0

    # 6. Pass / Fail Evaluation Table against thresholds
    pass_fail_table = []
    if real_ridge is not None and real_logistic is not None:
        eval_r2 = renderer_r2_v2 if strict_gates and renderer_r2_v2 is not None else nuisance_r2
        pass_fail_table = evaluate_thresholds_table(
            real_ridge=real_ridge,
            real_logistic=real_logistic,
            comparisons=comparisons,
            nuisance_r2=eval_r2,
            thresholds_cfg=config.get("thresholds", {}),
            strict_gates=strict_gates,
        )

    # Strip prediction arrays before JSON serialization
    serialized_variants = {}
    for v_name, v_res in variant_results.items():
        v_copy = {
            "normalization_sha256": v_res["normalization_sha256"],
            "ridge": {k: v for k, v in v_res["ridge"].items() if not k.startswith("pred_")},
            "logistic": {k: v for k, v in v_res["logistic"].items() if not k.startswith("pred_")},
        }
        serialized_variants[v_name] = v_copy

    nuisance_serialized = {
        "normalization_sha256": nuisance_res["normalization_sha256"],
        "ridge": {k: v for k, v in nuisance_res["ridge"].items() if not k.startswith("pred_")},
        "logistic": {k: v for k, v in nuisance_res["logistic"].items() if not k.startswith("pred_")},
    }

    results = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": get_git_commit(),
            "config_sha256": compute_file_sha256(config_path),
            "feature_hashes": feature_hashes,
            "train_samples": n_train,
            "val_samples": n_val,
            "dropped_nan_train": int(np.sum(labels_split == "train") - n_train),
            "dropped_nan_val": int(np.sum(labels_split == "val") - n_val),
            "cost_fee_per_side": cost_fee,
            "cost_slippage": cost_slip,
            "cost_threshold": cost_threshold,
            "strict_gates": bool(strict_gates),
        },
        "models": {
            **serialized_variants,
            "nuisance_only": nuisance_serialized,
            "ohlcv_baselines": baselines_info,
        },
        "comparisons": comparisons,
        "nuisance_shortcut_r2": nuisance_r2,
        "renderer_shortcut_r2_v2": renderer_r2_v2,
        "pass_fail_evaluation": pass_fail_table,
    }

    # 7. Atomic output write
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=output_path.parent, delete=False) as tmp:
        json.dump(results, tmp, indent=2)
        tmp_name = tmp.name
    Path(tmp_name).replace(output_path)

    print(f"Probe evaluation complete. Results written to {output_path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Research v2 Phase 5 Supervised Linear Probe.")
    parser.add_argument("--features-dir", type=Path, default=OUTPUTS_DIR / "v2/features")
    parser.add_argument("--labels-path", type=Path, default=DATA_DIR / "labels_v2.parquet")
    parser.add_argument("--config-path", type=Path, default=ROOT / "config/experiment_v2.yaml")
    parser.add_argument("--baselines-path", type=Path, default=OUTPUTS_DIR / "v2/baselines_val.json")
    parser.add_argument("--output-path", type=Path, default=OUTPUTS_DIR / "v2/probe_val.json")
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--permutations", type=int, default=500)
    parser.add_argument("--strict-gates", action="store_true", help="Apply strict gate conjunction and audit rules.")
    args = parser.parse_args()

    run_probe_v2(
        features_dir=args.features_dir,
        labels_path=args.labels_path,
        config_path=args.config_path,
        baselines_path=args.baselines_path,
        output_path=args.output_path,
        bootstrap_samples=args.bootstrap_samples,
        permutations=args.permutations,
        strict_gates=args.strict_gates,
    )


if __name__ == "__main__":
    main()
