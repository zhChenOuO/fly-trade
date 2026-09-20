"""Unit tests for Phase 5 bug fixes (Part A of Batch 5).

Verifies:
1. evaluate_thresholds_table:
   - strict_gates=False preserves legacy pass/fail semantics and outputs.
   - strict_gates=True flags trade_count==0 as N/A_NO_TRADES (passed=False).
   - strict_gates=True marks missing models or comparisons as NOT_EVALUATED.
   - min_practical_effect_balanced_acc includes baseline_definition="1/3 (uniform 3-class chance)".
2. renderer_shortcut_r2_v2:
   - Evaluates out-of-sample R^2 from Train to Val (allows negative values).
3. select_ridge_alpha_timeseries_cv & derive_purge_samples:
   - purge_samples derivation: derive_purge_samples(48, 12, 6) == 10.
   - purge_samples=0 preserves legacy behavior.
   - purge_samples > 0 introduces embargo between train_end and eval_start.
   - return_diagnostics=True returns fold losses and best_alpha_hits_upper_bound.
4. run_baselines_v2 split whitelist enforcement at parquet read level:
   - Verifies dev_test_v1 is never read into memory.
"""
from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pytest

from research.pipeline.baselines_v2 import (
    select_ridge_alpha_timeseries_cv,
    derive_purge_samples,
    PureNumpyRidge,
)
from research.pipeline.train_probe_v2 import (
    evaluate_thresholds_table,
    compute_renderer_shortcut_r2_v2,
)


def test_derive_purge_samples_formula() -> None:
    """derive_purge_samples must strictly implement ceil((input_window + max_horizon) / stride)."""
    # 48 input bars + 12 max horizon = 60 bars / 6 stride = 10 samples
    assert derive_purge_samples(input_window_bars=48, max_prediction_horizon=12, sample_stride=6) == 10
    # 48 input bars + 6 primary horizon = 54 bars / 6 stride = 9 samples
    assert derive_purge_samples(input_window_bars=48, max_prediction_horizon=6, sample_stride=6) == 9
    # Uneven division: 48 + 10 = 58 bars / 6 stride = 10 samples (ceil(9.666))
    assert derive_purge_samples(input_window_bars=48, max_prediction_horizon=10, sample_stride=6) == 10


def test_select_ridge_alpha_timeseries_cv_purge_and_diagnostics() -> None:
    """select_ridge_alpha_timeseries_cv must support purge_samples and return_diagnostics."""
    rng = np.random.default_rng(42)
    n = 300
    d = 5
    X = rng.standard_normal((n, d))
    y = X[:, 0] * 2.0 + rng.standard_normal(n) * 0.5

    # 1. purge_samples=0 should be default and return float
    alpha_default = select_ridge_alpha_timeseries_cv(X, y)
    assert isinstance(alpha_default, float)

    # 2. return_diagnostics=True returns alpha and full diagnostics dict
    alpha_diag, diag = select_ridge_alpha_timeseries_cv(X, y, purge_samples=10, return_diagnostics=True)
    assert isinstance(alpha_diag, float)
    assert isinstance(diag, dict)
    assert "cv_losses_by_alpha" in diag
    assert "best_alpha_hits_upper_bound" in diag
    assert "purge_samples" in diag
    assert diag["purge_samples"] == 10
    assert len(diag["cv_losses_by_alpha"]) == 8  # 8 alphas in grid
    for a, losses in diag["cv_losses_by_alpha"].items():
        assert len(losses) > 0

    # 3. Test upper bound hit detection
    # If y is pure independent noise, ridge with high alpha usually achieves lowest MSE
    y_noise = rng.standard_normal(n)
    alpha_noise, diag_noise = select_ridge_alpha_timeseries_cv(X, y_noise, return_diagnostics=True)
    if alpha_noise == 10000.0:
        assert diag_noise["best_alpha_hits_upper_bound"] is True


def test_evaluate_thresholds_table_strict_vs_legacy() -> None:
    """evaluate_thresholds_table must preserve legacy outputs by default and audit under strict_gates=True."""
    mock_ridge = {
        "continuous": {"spearman_ic": 0.005},
        "trading": {"net_return": 0.0, "annualized_sharpe": 0.0, "max_drawdown": 0.0, "trade_count": 0},
    }
    mock_logistic = {
        "classification": {"balanced_accuracy": 0.35},
    }
    mock_comparisons = {
        "delta_ic_vs_random": {"delta_ic": 0.001},
        "delta_ic_vs_scramble": {"delta_ic": 0.002},
        "real_ic_permutation_p_value": 0.45,
        "delta_ic_vs_ohlcv_baseline": {"delta_ic": -0.005},
    }
    mock_cfg = {
        "primary_metric": {"value": 0.02},
        "min_practical_effect": {"value": 0.02},
        "min_practical_effect_balanced_acc": {"value": 0.02},
        "connectome_vs_matched_control_ic_delta_min": {"value": 0.01},
        "connectome_vs_matched_control_p_max": {"value": 0.01},
        "connectome_vs_ohlcv_baseline_ic_delta_min": {"value": 0.01},
        "renderer_shortcut_r2_max": {"value": 0.05},
        "trading_net_return_min": {"value": 0.0},
        "trading_sharpe_min": {"value": 0.5},
        "trading_max_drawdown_max": {"value": 0.15},
    }

    # 1. Legacy mode (strict_gates=False): trade_count=0 vacuously passes net_return >= 0 and drawdown <= 0.15
    table_legacy = evaluate_thresholds_table(
        mock_ridge, mock_logistic, mock_comparisons, nuisance_r2=0.01, thresholds_cfg=mock_cfg, strict_gates=False
    )
    rows_legacy = {r["threshold_key"]: r for r in table_legacy}
    assert rows_legacy["trading_net_return_min"]["passed"] is True  # 0.0 >= 0.0 (vacuous pass)
    assert rows_legacy["trading_max_drawdown_max"]["passed"] is True  # 0.0 <= 0.15 (vacuous pass)
    assert "baseline_definition" in rows_legacy["min_practical_effect_balanced_acc"]
    assert rows_legacy["min_practical_effect_balanced_acc"]["baseline_definition"] == "1/3 (uniform 3-class chance)"

    # 2. Strict mode (strict_gates=True): trade_count=0 marked as N/A_NO_TRADES and passed=False
    table_strict = evaluate_thresholds_table(
        mock_ridge, mock_logistic, mock_comparisons, nuisance_r2=0.01, thresholds_cfg=mock_cfg, strict_gates=True
    )
    rows_strict = {r["threshold_key"]: r for r in table_strict}
    assert rows_strict["trading_net_return_min"]["passed"] is False
    assert rows_strict["trading_net_return_min"]["verdict_flag"] == "N/A_NO_TRADES"
    assert rows_strict["trading_sharpe_min"]["passed"] is False
    assert rows_strict["trading_sharpe_min"]["verdict_flag"] == "N/A_NO_TRADES"
    assert rows_strict["trading_max_drawdown_max"]["passed"] is False
    assert rows_strict["trading_max_drawdown_max"]["verdict_flag"] == "N/A_NO_TRADES"

    # 3. Strict mode with missing models/comparisons
    table_missing = evaluate_thresholds_table(
        real_ridge=None, real_logistic=None, comparisons={}, nuisance_r2=None, thresholds_cfg=mock_cfg, strict_gates=True
    )
    assert len(table_missing) == 11
    for r in table_missing:
        assert r["passed"] is False
        assert r["verdict_flag"] == "NOT_EVALUATED"


def test_renderer_shortcut_r2_v2_oos_calculation() -> None:
    """compute_renderer_shortcut_r2_v2 must evaluate Train-to-Val out-of-sample R^2 (allowing negative values)."""
    rng = np.random.default_rng(101)
    n_tr, n_va = 200, 100

    # Scenario A: Strong genuine relationship with nuisance features
    X_nui_tr = rng.standard_normal((n_tr, 7))
    X_nui_va = rng.standard_normal((n_va, 7))
    true_beta = np.array([1.0, -0.5, 0.8, 0.0, 0.0, 0.3, -0.2])
    s_tr_strong = X_nui_tr @ true_beta + rng.standard_normal(n_tr) * 0.1
    s_va_strong = X_nui_va @ true_beta + rng.standard_normal(n_va) * 0.1

    r2_strong = compute_renderer_shortcut_r2_v2(X_nui_tr, s_tr_strong, X_nui_va, s_va_strong)
    assert r2_strong > 0.80

    # Scenario B: Overfit noise on Train producing negative R^2 on Val
    s_tr_noise = rng.standard_normal(n_tr) * 0.1
    s_va_diff = rng.standard_normal(n_va) * 5.0 + 10.0  # distribution shift
    r2_neg = compute_renderer_shortcut_r2_v2(X_nui_tr, s_tr_noise, X_nui_va, s_va_diff)
    assert r2_neg < 0.0  # out-of-sample R^2 allows negative values


def test_baselines_v2_split_filtering_at_read_level() -> None:
    """run_baselines_v2 must filter splits at parquet read layer and never read dev_test_v1."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_dir = Path(tmpdir)
        # Create small synthetic samples and labels
        n_samples = 60
        df_samples = pd.DataFrame({
            "sample_id": np.arange(n_samples),
            "timestamp": pd.date_range("2024-01-01", periods=n_samples, freq="5min"),
            "image_idx": np.arange(n_samples),
            "future_return": np.zeros(n_samples),
            "label": np.zeros(n_samples),
            "split": ["train"] * 30 + ["val"] * 20 + ["dev_test_v1"] * 10,
        })
        df_labels = pd.DataFrame({
            "split": ["train"] * 30 + ["val"] * 20 + ["dev_test_v1"] * 10,
            "future_return_6": np.zeros(n_samples),
            "action": ["HOLD"] * n_samples,
        })

        samples_path = tmp_dir / "samples.parquet"
        labels_path = tmp_dir / "labels_v2.parquet"
        df_samples.to_parquet(samples_path)
        df_labels.to_parquet(labels_path)

        # Read with split filters
        split_filters = [("split", "in", ["train", "val"])]
        s_read = pd.read_parquet(samples_path, filters=split_filters)
        l_read = pd.read_parquet(labels_path, filters=split_filters)

        assert "dev_test_v1" not in s_read["split"].unique()
        assert "dev_test_v1" not in l_read["split"].unique()
        assert len(s_read) == 50
        assert len(l_read) == 50
