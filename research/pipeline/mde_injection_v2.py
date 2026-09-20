"""Research v2 Phase 5 Minimum Detectable Effect (MDE) Label Injection Experiment.

EXPLORATORY POST-HOC PROTOCOL:
- Authority: research/PLAN_v3_direction.md
- THIS EXPERIMENT IS EXPLORATORY_POST_HOC AND DOES NOT CHANGE PHASE 5 VERDICT (FAIL).
- Objective: Measure the statistical detection power (MDE) of the frozen Phase 5 pipeline
  (1291-dim DN readout, Train=31,556, Val=10,504, Ridge with expanding-window CV alpha,
  purge embargo, paired block bootstrap CI) against known synthetic signal forms at target ICs:
  {0.0, 0.005, 0.01, 0.02, 0.03, 0.05}.
- Signal forms s_t:
  (i) momentum: recent 6-bar return sum
  (ii) mean-reversion: negative price departure from window mean
  (iii) image-projection: fixed random projection of 64x64x3 renderer image
- Synthetic labels: y_synth = a * z(s_t) + eps_t, where eps_t is real return circularly shifted.
- Reuses existing Phase 5 features (.npy), no simulator re-runs needed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import yaml

import numpy as np
import pandas as pd
import scipy.stats

from research.pipeline.baselines_v2 import (
    PureNumpyRidge,
    select_ridge_alpha_timeseries_cv,
    derive_purge_samples,
    compute_continuous_metrics,
    compute_train_standardization,
    apply_standardization,
    validate_split_name,
    compute_file_sha256,
    get_git_commit,
)
from research.pipeline.probe_posthoc_v2 import verify_feature_hashes

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"

TARGET_ICS = (0.0, 0.005, 0.01, 0.02, 0.03, 0.05)
SIGNAL_FORMS = ("momentum", "mean_reversion", "image_projection")


def wilson_score_interval(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    """Compute Wilson score confidence interval for a binomial proportion."""
    if trials <= 0:
        return 0.0, 0.0
    if successes <= 0:
        z = float(scipy.stats.norm.ppf(1.0 - (1.0 - confidence) / 2.0))
        high = z**2 / (trials + z**2)
        return 0.0, float(min(1.0, high))
    if successes >= trials:
        z = float(scipy.stats.norm.ppf(1.0 - (1.0 - confidence) / 2.0))
        low = trials / (trials + z**2)
        return float(max(0.0, low)), 1.0

    z = float(scipy.stats.norm.ppf(1.0 - (1.0 - confidence) / 2.0))
    p = float(successes / trials)
    denom = 1.0 + (z**2) / trials
    centre = (p + (z**2) / (2.0 * trials)) / denom
    spread = (z * np.sqrt((p * (1.0 - p) + (z**2) / (4.0 * trials)) / trials)) / denom
    return float(max(0.0, centre - spread)), float(min(1.0, centre + spread))


def generate_circular_shifted_noise(
    y_real: np.ndarray,
    seed: int,
    block_size: int = 24,
    min_shift: int = 24,
) -> np.ndarray:
    """Generate circularly shifted noise preserving empirical distribution and autocorrelation."""
    n = len(y_real)
    rng = np.random.default_rng(seed)
    eff_min = min(min_shift, max(1, n // 4))
    eff_max = max(eff_min + 1, n - eff_min)
    shift = int(rng.integers(eff_min, eff_max))
    return np.roll(y_real, shift)


def calibrate_signal_amplitude(
    s_t: np.ndarray,
    eps: np.ndarray,
    target_ic: float,
    tol: float = 0.001,
    max_iters: int = 40,
) -> float:
    """Calibrate amplitude `a` such that Spearman IC(s_t, a * s_t + eps) equals target_ic."""
    if target_ic <= 0.0:
        return 0.0

    # Ensure s_t is standardized
    s_std = (s_t - np.mean(s_t)) / (np.std(s_t) + 1e-12)
    eps_scale = float(np.std(eps)) if np.std(eps) > 1e-12 else 1.0

    # Find bracket [a_low, a_high]
    a_low = 0.0
    a_high = eps_scale * 0.1
    ic_high = compute_continuous_metrics(a_high * s_std + eps, s_std)["spearman_ic"]

    while ic_high < target_ic and a_high < 1e6:
        a_low = a_high
        a_high *= 2.0
        ic_high = compute_continuous_metrics(a_high * s_std + eps, s_std)["spearman_ic"]

    # Binary search
    for _ in range(max_iters):
        a_mid = 0.5 * (a_low + a_high)
        ic_mid = compute_continuous_metrics(a_mid * s_std + eps, s_std)["spearman_ic"]
        if abs(ic_mid - target_ic) <= tol:
            return float(a_mid)
        if ic_mid < target_ic:
            a_low = a_mid
        else:
            a_high = a_mid

    return float(0.5 * (a_low + a_high))


def extract_signal_forms(
    raw: pd.DataFrame,
    samples: pd.DataFrame,
    images_mmap: np.ndarray | None = None,
    projection_seed: int = 42,
    window_bars: int = 48,
) -> dict[str, np.ndarray]:
    """Extract 3 signal forms s_t strictly from data available at or before timestamp t."""
    raw_closes = raw["close"].to_numpy(dtype=np.float64)
    raw_ts = raw["timestamp"]

    bar_step = pd.Timedelta(minutes=5)
    pos = np.asarray(raw_ts.searchsorted(samples["timestamp"] - bar_step), dtype=np.int64)
    idx = pos[:, None] - (window_bars - 1) + np.arange(window_bars)[None, :]
    if (idx < 0).any():
        raise ValueError("Sample window extends prior to the beginning of raw data.")

    c = raw_closes[idx]

    # (i) Momentum: return over the most recent 6 bars in the window
    # c[:, -1] is close at bar 47, c[:, -7] is close at bar 41
    s_mom = np.log(c[:, -1] / c[:, -7])

    # (ii) Mean reversion: negative departure of close price from window mean
    mean_c = np.mean(c, axis=1)
    s_mr = - (c[:, -1] - mean_c) / mean_c

    signals = {
        "momentum": s_mom,
        "mean_reversion": s_mr,
    }

    # (iii) Image projection: fixed random projection of renderer image
    if images_mmap is not None and "image_idx" in samples.columns:
        img_sub = np.asarray(images_mmap[samples["image_idx"].to_numpy(dtype=np.int64)])
        img_flat = img_sub.reshape(len(img_sub), -1).astype(np.float64)
        rng = np.random.default_rng(projection_seed)
        w_proj = rng.standard_normal(img_flat.shape[1]) / np.sqrt(img_flat.shape[1])
        s_proj = img_flat @ w_proj
        signals["image_projection"] = s_proj

    return signals


def paired_block_bootstrap_ic_and_ci(
    y_true: np.ndarray,
    preds_dict: dict[str, np.ndarray],
    block_size: int = 24,
    n_bootstraps: int = 500,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[dict[str, tuple[float, float]], dict[str, tuple[float, float]]]:
    """Compute moving block bootstrap CIs for IC and pairwise Delta IC using common blocks."""
    n = len(y_true)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_size))

    model_names = list(preds_dict.keys())
    reps = {m: np.empty(n_bootstraps, dtype=np.float64) for m in model_names}

    starts = rng.integers(0, n - block_size + 1, size=(n_bootstraps, n_blocks))
    idx_matrix = (starts[:, :, None] + np.arange(block_size)).reshape(n_bootstraps, -1)[:, :n]

    for b in range(n_bootstraps):
        sub_idx = idx_matrix[b]
        sub_y = y_true[sub_idx]
        for m in model_names:
            sub_p = preds_dict[m][sub_idx]
            if np.std(sub_p) < 1e-12 or np.std(sub_y) < 1e-12:
                reps[m][b] = 0.0
            else:
                r = scipy.stats.spearmanr(sub_p, sub_y)
                ic = float(r.statistic if hasattr(r, "statistic") else r[0])
                reps[m][b] = 0.0 if np.isnan(ic) else ic

    # Individual CIs
    q_low = 100.0 * (alpha / 2.0)
    q_high = 100.0 * (1.0 - alpha / 2.0)
    cis = {
        m: (float(np.percentile(reps[m], q_low)), float(np.percentile(reps[m], q_high)))
        for m in model_names
    }

    # Delta CIs relative to real
    delta_cis = {}
    if "real" in reps:
        for m in model_names:
            if m != "real":
                delta_reps = reps["real"] - reps[m]
                delta_cis[f"real_vs_{m}"] = (
                    float(np.percentile(delta_reps, q_low)),
                    float(np.percentile(delta_reps, q_high)),
                )

    return cis, delta_cis


def run_mde_injection_experiment(
    features_dir: Path | str = OUTPUTS_DIR / "v2/features",
    data_dir: Path | str = DATA_DIR,
    config_path: Path | str = ROOT / "config/experiment_v2.yaml",
    probe_val_path: Path | str = OUTPUTS_DIR / "v2/probe_val.json",
    output_path: Path | str = OUTPUTS_DIR / "v2/mde_injection.json",
    repetitions: int = 100,
    bootstrap_samples: int = 500,
    target_ics: tuple[float, ...] = TARGET_ICS,
    signal_forms: tuple[str, ...] = SIGNAL_FORMS,
    seed: int = 42,
    features_dict: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    verify_hashes: bool = True,
) -> dict:
    """Execute MDE label injection simulation experiment across target ICs and signal forms."""
    features_dir = Path(features_dir)
    data_dir = Path(data_dir)
    config_path = Path(config_path)
    output_path = Path(output_path)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 1. Derive purge_samples from YAML specification
    purge_samples = derive_purge_samples(
        input_window_bars=config["data"]["input_window_bars"],
        max_prediction_horizon=max(config["data"]["prediction_horizons"]),
        sample_stride=config["data"]["sample_stride"],
    )

    # 2. Load and validate data
    validate_split_name("train")
    validate_split_name("val")

    split_filters = [("split", "in", ["train", "val"])]
    samples = pd.read_parquet(data_dir / "samples.parquet", filters=split_filters)
    raw = pd.read_parquet(data_dir / "raw_ohlcv.parquet")
    labels = pd.read_parquet(data_dir / "labels_v2.parquet", filters=split_filters)

    img_path = data_dir / "images.npy"
    images_mmap = np.load(img_path, mmap_mode="r") if img_path.exists() else None

    # Filter train/val masks
    labels_split = labels["split"].to_numpy()
    y_cont_all = labels["future_return_6"].to_numpy(dtype=np.float64)
    y_cls_all = labels["action"].to_numpy(dtype=object)

    train_mask = (labels_split == "train") & (~np.isnan(y_cont_all)) & (pd.notna(y_cls_all))
    val_mask = (labels_split == "val") & (~np.isnan(y_cont_all)) & (pd.notna(y_cls_all))

    y_train_real = y_cont_all[train_mask]
    y_val_real = y_cont_all[val_mask]

    n_train = len(y_train_real)
    n_val = len(y_val_real)

    # 3. Extract signal forms
    raw_signals = extract_signal_forms(raw, samples, images_mmap)
    signals = {}
    for s_name in signal_forms:
        if s_name not in raw_signals:
            continue
        s_all = raw_signals[s_name]
        s_tr = s_all[train_mask]
        s_va = s_all[val_mask]
        # Train-only standardization
        mu = float(np.mean(s_tr))
        sigma = float(np.std(s_tr))
        sigma_adj = sigma if sigma > 1e-12 else 1.0
        signals[s_name] = {
            "train": (s_tr - mu) / sigma_adj,
            "val": (s_va - mu) / sigma_adj,
        }

    # 4. Load or receive feature matrices for Real, Random, Scramble
    feature_matrices = {}
    if features_dict is not None:
        feature_matrices = features_dict
    else:
        if probe_val_path.exists() and verify_hashes:
            with open(probe_val_path, "r", encoding="utf-8") as f:
                p_val_data = json.load(f)
            expected_hashes = p_val_data.get("metadata", {}).get("feature_hashes", {})
            verify_feature_hashes(features_dir, expected_hashes)

        for variant in ["real", "random", "scramble"]:
            tr_file = features_dir / f"{variant}_train.npy"
            va_file = features_dir / f"{variant}_val.npy"
            X_tr = np.load(tr_file)
            X_va = np.load(va_file)
            if len(X_tr) == len(labels_split[labels_split == "train"]):
                X_tr = X_tr[train_mask[labels_split == "train"]]
            if len(X_va) == len(labels_split[labels_split == "val"]):
                X_va = X_va[val_mask[labels_split == "val"]]

            norm = compute_train_standardization(X_tr, [f"f_{i}" for i in range(X_tr.shape[1])])
            X_tr_std = apply_standardization(X_tr, norm)
            X_va_std = apply_standardization(X_va, norm)
            feature_matrices[variant] = (X_tr_std, X_va_std)

    # 5. Run simulation across signal forms, target ICs, and repetitions
    print(f"Starting MDE experiment: {len(signals)} signals x {len(target_ics)} target ICs x {repetitions} reps...")

    variants = list(feature_matrices.keys())
    results_tree: dict[str, dict[str, dict]] = {s_name: {} for s_name in signals}
    mde_table: dict[str, dict[str, str | float]] = {s_name: {} for s_name in signals}

    for s_name, s_data in signals.items():
        s_tr = s_data["train"]
        s_va = s_data["val"]

        for target_ic in target_ics:
            ic_str = f"{target_ic:.4f}"
            rep_stats: dict[str, list[dict]] = {v: [] for v in variants}
            delta_stats: dict[str, list[dict]] = {"real_vs_random": [], "real_vs_scramble": []}
            oracle_ics_val = []

            for r in range(repetitions):
                rep_seed = seed + 10000 * r + int(target_ic * 1000)

                # Generate circularly shifted noise
                eps_tr = generate_circular_shifted_noise(y_train_real, seed=rep_seed, block_size=24)
                eps_va = generate_circular_shifted_noise(y_val_real, seed=rep_seed + 1, block_size=24)

                # Calibrate amplitude on Train and Val independently to achieve oracle target IC
                a_tr = calibrate_signal_amplitude(s_tr, eps_tr, target_ic, tol=0.001)
                a_va = calibrate_signal_amplitude(s_va, eps_va, target_ic, tol=0.001)

                y_synth_tr = a_tr * s_tr + eps_tr
                y_synth_va = a_va * s_va + eps_va

                oracle_ic = compute_continuous_metrics(y_synth_va, s_va)["spearman_ic"]
                oracle_ics_val.append(oracle_ic)

                # Train Ridge models on feature matrices
                preds_va = {}
                alphas_chosen = {}
                for v in variants:
                    X_tr_mat, X_va_mat = feature_matrices[v]
                    best_a, diag = select_ridge_alpha_timeseries_cv(
                        X_tr_mat, y_synth_tr, purge_samples=purge_samples, return_diagnostics=True
                    )
                    alphas_chosen[v] = diag
                    ridge = PureNumpyRidge(alpha=best_a).fit(X_tr_mat, y_synth_tr)
                    preds_va[v] = ridge.predict(X_va_mat)

                # Paired block bootstrap for this repetition
                cis, delta_cis = paired_block_bootstrap_ic_and_ci(
                    y_true=y_synth_va,
                    preds_dict=preds_va,
                    block_size=24,
                    n_bootstraps=bootstrap_samples,
                    seed=rep_seed + 2,
                )

                for v in variants:
                    obs_ic = compute_continuous_metrics(y_synth_va, preds_va[v])["spearman_ic"]
                    ci_low, ci_high = cis[v]
                    detected = bool(ci_low > 0.0)
                    rep_stats[v].append({
                        "observed_ic": obs_ic,
                        "ic_ci_95": [ci_low, ci_high],
                        "ci_width": ci_high - ci_low,
                        "detected": detected,
                        "bias": obs_ic - target_ic,
                        "alpha": alphas_chosen[v]["best_alpha"],
                        "alpha_hits_upper": alphas_chosen[v]["best_alpha_hits_upper_bound"],
                    })

                for pair_key in ["real_vs_random", "real_vs_scramble"]:
                    if pair_key in delta_cis:
                        d_low, d_high = delta_cis[pair_key]
                        delta_stats[pair_key].append({
                            "delta_ci_95": [d_low, d_high],
                            "delta_ci_width": d_high - d_low,
                            "detected": bool(d_low > 0.0),
                        })

            # Aggregate over repetitions for this target_ic
            agg_entry = {
                "target_ic": float(target_ic),
                "oracle_ic_val_mean": float(np.mean(oracle_ics_val)),
                "oracle_ic_val_std": float(np.std(oracle_ics_val)),
                "variants": {},
                "pairwise_delta": {},
            }

            for v in variants:
                detections = [s["detected"] for s in rep_stats[v]]
                n_det = sum(detections)
                det_rate = float(n_det / repetitions)
                det_ci = wilson_score_interval(n_det, repetitions)
                widths = [s["ci_width"] for s in rep_stats[v]]
                biases = [s["bias"] for s in rep_stats[v]]
                alpha_upper_hits = [s["alpha_hits_upper"] for s in rep_stats[v]]

                agg_entry["variants"][v] = {
                    "detection_rate": det_rate,
                    "detection_rate_mc_ci_95": list(det_ci),
                    "mean_ic_bias": float(np.mean(biases)),
                    "ci_width_median": float(np.median(widths)),
                    "ci_width_p90": float(np.percentile(widths, 90)),
                    "alpha_hits_upper_bound_ratio": float(np.mean(alpha_upper_hits)),
                }

            for pair_key in ["real_vs_random", "real_vs_scramble"]:
                if delta_stats[pair_key]:
                    d_dets = [s["detected"] for s in delta_stats[pair_key]]
                    d_widths = [s["delta_ci_width"] for s in delta_stats[pair_key]]
                    agg_entry["pairwise_delta"][pair_key] = {
                        "detection_rate": float(np.mean(d_dets)),
                        "delta_ci_width_median": float(np.median(d_widths)),
                        "delta_ci_width_p90": float(np.percentile(d_widths, 90)),
                    }

            results_tree[s_name][ic_str] = agg_entry

        # Compute MDE (first target IC with detection_rate >= 0.80)
        for v in variants:
            mde_val = "NOT_REACHED"
            for target_ic in target_ics:
                ic_str = f"{target_ic:.4f}"
                rate = results_tree[s_name][ic_str]["variants"][v]["detection_rate"]
                if rate >= 0.80:
                    mde_val = float(target_ic)
                    break
            mde_table[s_name][v] = mde_val

    # Assemble final output
    output_payload = {
        "metadata": {
            "notice_1": "EXPLORATORY_POST_HOC",
            "notice_2": "DOES_NOT_CHANGE_PHASE5_VERDICT",
            "description": "Minimum Detectable Effect (MDE) label injection power curve for Phase 5 linear probe.",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": get_git_commit(),
            "config_sha256": compute_file_sha256(config_path),
            "train_samples": n_train,
            "val_samples": n_val,
            "repetitions_per_condition": repetitions,
            "bootstrap_samples": bootstrap_samples,
            "purge_samples": purge_samples,
            "target_ics": list(target_ics),
            "signal_forms": list(signals.keys()),
            "notes": (
                "MDE depends on the linear readout capability of the selected signal form. "
                "Differences among signal forms are an intrinsic part of the empirical findings. "
                "MDE for a specific synthetic signal form cannot be equated to universal market predictability."
            ),
        },
        "mde_summary_by_signal_and_variant": mde_table,
        "detailed_results": results_tree,
    }

    # Write output JSON atomically if output path specified
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)
    tmp_path.replace(output_path)

    print(f"MDE experiment complete. Results written to {output_path}")
    return output_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5 Minimum Detectable Effect (MDE) Label Injection Experiment.")
    parser.add_argument("--features-dir", type=Path, default=OUTPUTS_DIR / "v2/features")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--config", type=Path, default=ROOT / "config/experiment_v2.yaml")
    parser.add_argument("--probe-val", type=Path, default=OUTPUTS_DIR / "v2/probe_val.json")
    parser.add_argument("--output", type=Path, default=OUTPUTS_DIR / "v2/mde_injection.json")
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-verify-hashes", action="store_true")
    args = parser.parse_args()

    run_mde_injection_experiment(
        features_dir=args.features_dir,
        data_dir=args.data_dir,
        config_path=args.config,
        probe_val_path=args.probe_val,
        output_path=args.output,
        repetitions=args.repetitions,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        verify_hashes=not args.no_verify_hashes,
    )


if __name__ == "__main__":
    main()
