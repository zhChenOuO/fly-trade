"""Unit tests for Research v2 Phase 5 Post-Hoc Exploratory Analysis (probe_posthoc_v2).

Tests:
1. Paired multi-model moving block bootstrap: detects positive delta when signal is present,
   null when identical, and reproduces bitwise with fixed seed.
2. Holm-Bonferroni correction: matches hand calculation on standard examples and edge cases.
3. Signal redundancy and residual IC: residual IC vanishes when signal is fully redundant with controls,
   and remains positive when independent signal is present.
4. Corrected pass/fail table audit: flags trade_count==0 as N/A_NO_TRADES, reports competitive
   balanced accuracy baselines, and computes renderer shortcut R^2 on active Logistic scores.
5. Feature hash mismatch: raises ValueError on hash discrepancy or FileNotFoundError on missing files.
6. Temporal stability across quarters.
"""
from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import pytest

from research.pipeline.probe_posthoc_v2 import (
    holm_bonferroni_correction,
    paired_multi_block_bootstrap,
    compute_delta_ic_stats,
    compute_redundancy_and_residual_ic,
    compute_temporal_stability,
    build_corrected_table,
    verify_feature_hashes,
)


def test_holm_bonferroni_correction_hand_calc() -> None:
    """Holm-Bonferroni correction must match hand-calculated step-down multipliers."""
    # Hand calculation example:
    # Raw p-values: [0.01, 0.04, 0.03, 0.005] (m = 4)
    # Sorted:
    # 1: 0.005 * 4 = 0.020
    # 2: 0.010 * 3 = 0.030 -> max(0.020, 0.030) = 0.030
    # 3: 0.030 * 2 = 0.060 -> max(0.030, 0.060) = 0.060
    # 4: 0.040 * 1 = 0.040 -> max(0.060, 0.040) = 0.060
    # Expected in original order: [0.030, 0.060, 0.060, 0.020]
    p_raw = [0.01, 0.04, 0.03, 0.005]
    expected = [0.03, 0.06, 0.06, 0.02]
    adj = holm_bonferroni_correction(p_raw)
    np.testing.assert_allclose(adj, expected, atol=1e-6)

    # Edge cases
    # Single element
    assert np.allclose(holm_bonferroni_correction([0.042]), [0.042])
    # Empty
    assert len(holm_bonferroni_correction([])) == 0
    # Values capped at 1.0
    adj_capped = holm_bonferroni_correction([0.6, 0.8])
    assert np.all(adj_capped <= 1.0)
    assert np.allclose(adj_capped, [1.0, 1.0])


def test_paired_block_bootstrap_signal_detection_and_reproducibility() -> None:
    """Paired block bootstrap must detect true delta, null delta, and be reproducible."""
    rng = np.random.default_rng(123)
    n = 480  # 20 blocks of 24
    t = np.linspace(0, 10 * np.pi, n)
    y_true = np.sin(t) + 0.1 * rng.standard_normal(n)

    # Model A: strong true signal correlated with y_true
    pred_A = np.sin(t) + 0.2 * rng.standard_normal(n)
    # Model B: pure noise
    pred_B = rng.standard_normal(n)
    # Model C: identical to Model A
    pred_C = pred_A.copy()

    predictions = {"A": pred_A, "B": pred_B, "C": pred_C}
    reps1 = paired_multi_block_bootstrap(y_true, predictions, block_size=24, n_bootstraps=200, seed=42)
    reps2 = paired_multi_block_bootstrap(y_true, predictions, block_size=24, n_bootstraps=200, seed=42)

    # 1. Deterministic bitwise reproducibility
    np.testing.assert_array_equal(reps1["A"], reps2["A"])
    np.testing.assert_array_equal(reps1["B"], reps2["B"])

    # 2. Known signal: A vs B has Delta IC > 0, 95% CI > 0, p < 0.05
    ic_A = float(np.mean(reps1["A"]))
    ic_B = float(np.mean(reps1["B"]))
    stat_AB = compute_delta_ic_stats(reps1["A"], reps1["B"], ic_A, ic_B)
    assert stat_AB["observed_delta_ic"] > 0.4
    assert stat_AB["delta_ic_ci_95"][0] > 0.2
    assert stat_AB["delta_ic_ci_95"][1] > stat_AB["delta_ic_ci_95"][0]
    assert stat_AB["p_value_two_tailed"] < 0.05

    # 3. Identical models: A vs C has Delta IC == 0, CI contains 0, p == 1.0
    stat_AC = compute_delta_ic_stats(reps1["A"], reps1["C"], ic_A, ic_A)
    assert np.isclose(stat_AC["observed_delta_ic"], 0.0, atol=1e-8)
    assert stat_AC["delta_ic_ci_95"][0] <= 0.0 <= stat_AC["delta_ic_ci_95"][1]
    assert np.isclose(stat_AC["p_value_two_tailed"], 1.0)


def test_residual_ic_redundant_vs_independent_signal() -> None:
    """Residual IC must vanish when signal is redundant, and remain positive when independent."""
    rng = np.random.default_rng(456)
    n = 480
    t = np.linspace(0, 8 * np.pi, n)
    y_true = np.sin(t) + 0.1 * rng.standard_normal(n)

    # Case 1: Redundant Signal
    # Control has the signal; Real model's signal is entirely inherited from Control
    s_ohlcv_redundant = np.sin(t) + 0.15 * rng.standard_normal(n)
    s_nui_redundant = rng.standard_normal(n)
    s_real_redundant = 1.8 * s_ohlcv_redundant + 0.05 * rng.standard_normal(n)

    res_redundant = compute_redundancy_and_residual_ic(
        s_real=s_real_redundant,
        s_ohlcv=s_ohlcv_redundant,
        s_nui=s_nui_redundant,
        y_true=y_true,
        block_size=24,
        n_bootstraps=100,
        seed=42,
    )
    # R^2 should be high, residual IC should be near zero (CI contains 0)
    assert res_redundant["correlations"]["spearman_real_vs_ohlcv"] > 0.90
    assert res_redundant["residual_ic_joint_controls"]["r2_explained_by_joint_controls"] > 0.90
    ci_red = res_redundant["residual_ic_joint_controls"]["residual_ic_ci_95"]
    assert ci_red[0] <= 0.0 <= ci_red[1] or abs(res_redundant["residual_ic_joint_controls"]["residual_spearman_ic"]) < 0.15

    # Case 2: Independent Signal
    # Control is noise; Real model has genuine predictive signal
    s_ohlcv_indep = rng.standard_normal(n)
    s_nui_indep = rng.standard_normal(n)
    s_real_indep = np.sin(t) + 0.2 * rng.standard_normal(n)

    res_indep = compute_redundancy_and_residual_ic(
        s_real=s_real_indep,
        s_ohlcv=s_ohlcv_indep,
        s_nui=s_nui_indep,
        y_true=y_true,
        block_size=24,
        n_bootstraps=100,
        seed=42,
    )
    # Residual IC should be strongly positive with CI strictly above 0
    assert res_indep["residual_ic_joint_controls"]["residual_spearman_ic"] > 0.40
    ci_indep = res_indep["residual_ic_joint_controls"]["residual_ic_ci_95"]
    assert ci_indep[0] > 0.10


def test_corrected_table_presentation_audit() -> None:
    """Corrected presentation table must flag 0-trade rows, audit BA, and report Logistic R^2."""
    mock_probe_val = {
        "models": {
            "real": {
                "ridge": {
                    "trading": {"trade_count": 0, "net_return": 0.0, "max_drawdown": 0.0, "annualized_sharpe": 0.0},
                    "classification": {"balanced_accuracy": 0.3333333333333333},
                },
                "logistic": {
                    "classification": {"balanced_accuracy": 0.35680975427584594},
                },
            },
        },
        "pass_fail_evaluation": [
            {
                "threshold_key": "trading_net_return_min",
                "description": "Minimum net return after cost",
                "observed_value": 0.0,
                "required_value": ">= 0.0",
                "passed": True,
            },
            {
                "threshold_key": "trading_max_drawdown_max",
                "description": "Maximum allowable drawdown",
                "observed_value": 0.0,
                "required_value": "<= 0.15",
                "passed": True,
            },
            {
                "threshold_key": "min_practical_effect_balanced_acc",
                "description": "Balanced accuracy increment",
                "observed_value": 0.023476,
                "required_value": ">= 0.02",
                "passed": True,
            },
            {
                "threshold_key": "renderer_shortcut_r2_max",
                "description": "Maximum R^2 explained by nuisance features",
                "observed_value": 3.5e-8,
                "required_value": "<= 0.05",
                "passed": True,
            },
        ],
    }

    n = 200
    real_log_score = np.linspace(-0.5, 0.5, n)
    nui_val = np.random.default_rng(42).standard_normal((n, 7))

    table = build_corrected_table(
        probe_val_data=mock_probe_val,
        real_logistic_score=real_log_score,
        nuisance_features_val=nui_val,
        ohlcv_b1b_ba=0.4146,
        nuisance_logistic_ba=0.4118,
    )

    rows_by_key = {r["threshold_key"]: r for r in table}

    # 1. Zero-trade rows flagged as N/A_NO_TRADES and passed is False
    assert rows_by_key["trading_net_return_min"]["verdict_flag"] == "N/A_NO_TRADES"
    assert rows_by_key["trading_net_return_min"]["passed"] is False
    assert rows_by_key["trading_max_drawdown_max"]["verdict_flag"] == "N/A_NO_TRADES"
    assert rows_by_key["trading_max_drawdown_max"]["passed"] is False

    # 2. Balanced accuracy audits competitive controls
    ba_row = rows_by_key["min_practical_effect_balanced_acc"]
    assert ba_row["verdict_flag"] == "EXCEEDS_RANDOM_BUT_BELOW_CONTROLS"
    assert ba_row["control_ohlcv_b1b_logistic"] == 0.4146
    assert ba_row["control_nuisance_logistic"] == 0.4118

    # 3. Renderer shortcut R^2 uses non-constant Logistic score
    r2_row = rows_by_key["renderer_shortcut_r2_max"]
    assert "observed_value_corrected_logistic_r2" in r2_row
    assert r2_row["observed_value"] >= 0.0


def test_feature_hash_mismatch_raises() -> None:
    """verify_feature_hashes must raise ValueError on SHA256 mismatch and FileNotFoundError on missing files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        dummy_file = tmp_path / "real_train.npy"
        np.save(dummy_file, np.zeros((10, 5)))

        # Expected hash differs
        expected_hashes = {"real_train": "deadbeef" * 8}
        with pytest.raises(ValueError, match="Feature hash mismatch"):
            verify_feature_hashes(tmp_path, expected_hashes)

        # Missing file
        missing_expected = {"nonexistent_file": "deadbeef" * 8}
        with pytest.raises(FileNotFoundError, match="Missing required feature file"):
            verify_feature_hashes(tmp_path, missing_expected)


def test_temporal_stability_quarters() -> None:
    """Quarterly stability must partition into 4 equal segments and compute sign agreement."""
    n = 100
    y_true = np.linspace(-1, 1, n)
    pred_all_pos = np.linspace(-1, 1, n) + 0.1  # positively correlated across all quarters
    pred_mixed = np.concatenate([np.linspace(-1, 1, 50), np.linspace(1, -1, 50)])

    res = compute_temporal_stability(
        {"all_pos": pred_all_pos, "mixed": pred_mixed},
        y_true,
        n_quarters=4,
    )
    assert len(res["all_pos"]["quarterly_ic"]) == 4
    assert res["all_pos"]["all_positive"] is True
    assert res["all_pos"]["positive_quarter_count"] == 4
    assert res["all_pos"]["sign_agreement"] == 1.0

    assert res["mixed"]["all_positive"] is False
    assert res["mixed"]["positive_quarter_count"] < 4
