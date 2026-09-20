"""Unit tests for Phase 5 Minimum Detectable Effect (MDE) Label Injection Experiment.

Tests:
1. Signal amplitude calibration hits oracle target IC within 0.002 tolerance.
2. Wilson score confidence interval gives valid binomial coverage bounds.
3. Circular shifted noise generation preserves return length and structure.
4. MDE simulation on synthetic features:
   - Low false positive rate at target IC = 0 (<= 15% with R=20).
   - Detection rate is non-decreasing with increasing target IC for signal-carrying features.
5. Determinism: identical seeds produce bitwise identical results.
6. Split governance whitelist rejects dev_test_v1.
"""
from __future__ import annotations

import numpy as np
import pytest

from research.pipeline.baselines_v2 import (
    validate_split_name,
    compute_continuous_metrics,
)
from research.pipeline.mde_injection_v2 import (
    calibrate_signal_amplitude,
    wilson_score_interval,
    generate_circular_shifted_noise,
    paired_block_bootstrap_ic_and_ci,
)


def test_calibrate_signal_amplitude_target_accuracy() -> None:
    """calibrate_signal_amplitude must hit target Spearman IC within 0.002 tolerance."""
    rng = np.random.default_rng(101)
    n = 2000
    s_t = rng.standard_normal(n)
    eps = rng.standard_normal(n) * 0.02

    # IC = 0.0 returns amplitude 0.0
    a_zero = calibrate_signal_amplitude(s_t, eps, target_ic=0.0)
    assert a_zero == 0.0

    # Non-zero target ICs
    for target in [0.005, 0.01, 0.02, 0.03, 0.05]:
        a = calibrate_signal_amplitude(s_t, eps, target_ic=target, tol=0.001)
        y_synth = a * s_t + eps
        obs_ic = compute_continuous_metrics(y_synth, s_t)["spearman_ic"]
        diff = abs(obs_ic - target)
        assert diff <= 0.002, f"Target {target} had observed IC {obs_ic} (diff={diff})"


def test_wilson_score_interval_bounds() -> None:
    """wilson_score_interval must yield valid binomial confidence bounds in [0, 1]."""
    # 50% success
    low, high = wilson_score_interval(successes=50, trials=100)
    assert 0.39 < low < 0.41
    assert 0.59 < high < 0.61

    # 0% success (low bound is 0.0, high is strictly positive)
    low_0, high_0 = wilson_score_interval(successes=0, trials=100)
    assert low_0 == 0.0
    assert 0.03 < high_0 < 0.05

    # 100% success (high bound is 1.0, low is strictly < 1.0)
    low_1, high_1 = wilson_score_interval(successes=100, trials=100)
    assert 0.95 < low_1 < 0.97
    assert high_1 == 1.0


def test_circular_shifted_noise_preserves_length_and_variance() -> None:
    """generate_circular_shifted_noise must preserve distribution while rolling indices."""
    rng = np.random.default_rng(202)
    y_real = rng.standard_normal(500)
    y_shifted = generate_circular_shifted_noise(y_real, seed=42, block_size=24, min_shift=24)

    assert len(y_shifted) == len(y_real)
    np.testing.assert_allclose(np.mean(y_shifted), np.mean(y_real), atol=1e-12)
    np.testing.assert_allclose(np.std(y_shifted), np.std(y_real), atol=1e-12)
    assert not np.array_equal(y_shifted, y_real)


def test_mde_detection_monotonic_and_low_fpr() -> None:
    """MDE detection rate on features with known signal must be non-decreasing with target IC."""
    rng = np.random.default_rng(303)
    n_va = 1200  # 50 blocks of 24
    s_t = rng.standard_normal(n_va)

    # Model Real has access to the true signal + small noise
    pred_real = s_t + 0.05 * rng.standard_normal(n_va)
    # Model Random is pure noise
    pred_rand = rng.standard_normal(n_va)

    preds_dict = {"real": pred_real, "rand": pred_rand}
    eps = rng.standard_normal(n_va) * 0.05

    detection_counts = []
    # Test at target IC = 0.0, 0.04, 0.15 with R=20 repetitions
    R = 20
    for target in [0.0, 0.04, 0.15]:
        detected_cnt = 0
        calib_noise = [generate_circular_shifted_noise(eps, seed=5000 + k) for k in range(50)]
        a = calibrate_signal_amplitude(s_t, calib_noise, target_ic=target, tol=1e-3)
        for r in range(R):
            rep_seed = 1000 * r + int(target * 1000)
            eps_shifted = generate_circular_shifted_noise(eps, seed=rep_seed)
            y_synth = a * s_t + eps_shifted

            cis, _ = paired_block_bootstrap_ic_and_ci(
                y_true=y_synth,
                preds_dict=preds_dict,
                block_size=24,
                n_bootstraps=50,
                seed=rep_seed + 1,
            )
            if cis["real"][0] > 0.0:
                detected_cnt += 1

        detection_counts.append(detected_cnt)

    # 1. At IC=0, false positive rate should be low (<= 2 out of 20 = 10%)
    assert detection_counts[0] <= 2, f"False positive count at IC=0 was {detection_counts[0]}/{R}"

    # 2. Detection rate is non-decreasing with increasing target IC
    assert detection_counts[1] >= detection_counts[0]
    assert detection_counts[2] >= detection_counts[1]
    # Strong signal should be detected in majority of repetitions (>= 16 out of 20 = 80%)
    assert detection_counts[2] >= 16


def test_population_calibration_train_accuracy_and_val_sampling_sd() -> None:
    """Fixed population amplitude achieves Train IC mean within target +- 0.001 and Val SD in [0.005, 0.015]."""
    rng = np.random.default_rng(42)
    n_tr = 20000
    n_va = 10504

    s_tr = rng.standard_normal(n_tr)
    s_tr = (s_tr - np.mean(s_tr)) / np.std(s_tr)

    s_va = rng.standard_normal(n_va)
    s_va = (s_va - np.mean(s_va)) / np.std(s_va)

    y_tr_real = rng.standard_normal(n_tr) * 0.01
    y_va_real = rng.standard_normal(n_va) * 0.01

    calib_noise_tr = [generate_circular_shifted_noise(y_tr_real, seed=100 + k) for k in range(50)]

    for target in [0.005, 0.01, 0.02, 0.03, 0.05]:
        a = calibrate_signal_amplitude(s_tr, calib_noise_tr, target_ic=target, tol=1e-4)

        # 1. Train oracle IC mean across the K=50 noise draws must be within target +- 0.001
        tr_ics = [compute_continuous_metrics(a * s_tr + e, s_tr)["spearman_ic"] for e in calib_noise_tr]
        tr_mean = float(np.mean(tr_ics))
        assert abs(tr_mean - target) <= 0.001, f"Target {target} Train mean {tr_mean} deviated by {abs(tr_mean - target)}"

        # 2. Val oracle IC across independent noise draws has sampling SD in [0.005, 0.015]
        val_noise = [generate_circular_shifted_noise(y_va_real, seed=2000 + r) for r in range(50)]
        va_ics = [compute_continuous_metrics(a * s_va + e, s_va)["spearman_ic"] for e in val_noise]
        va_sd = float(np.std(va_ics))
        assert 0.005 <= va_sd <= 0.015, f"Target {target} Val SD {va_sd} outside [0.005, 0.015]"


def test_mde_injection_synthetic_dry_run() -> None:
    """Dry run of full MDE injection pipeline with R=5 on synthetic features."""
    import tempfile
    from pathlib import Path
    from research.pipeline.mde_injection_v2 import run_mde_injection_experiment

    rng = np.random.default_rng(42)
    d = 8
    n_tr, n_va = 31556, 10504
    synth_features = {
        "real": (rng.standard_normal((n_tr, d)), rng.standard_normal((n_va, d))),
        "random": (rng.standard_normal((n_tr, d)), rng.standard_normal((n_va, d))),
        "scramble": (rng.standard_normal((n_tr, d)), rng.standard_normal((n_va, d))),
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_out = Path(tmpdir) / "mde_dry_run.json"
        res = run_mde_injection_experiment(
            output_path=tmp_out,
            repetitions=5,
            bootstrap_samples=50,
            target_ics=(0.0, 0.01),
            signal_forms=("momentum",),
            features_dict=synth_features,
            verify_hashes=False,
        )

        assert res["EXPLORATORY_POST_HOC"] is True
        assert res["DOES_NOT_CHANGE_PHASE5_VERDICT"] is True
        assert "detection_monotonicity" in res
        mono_real = res["detection_monotonicity"]["momentum"]["real"]
        assert "is_monotonic_with_mc_tol" in mono_real
        assert "detection_rates" in mono_real

        entry = res["detailed_results"]["momentum"]["0.0100"]
        assert "calibrated_amplitude" in entry
        assert "oracle_ic_train_mean" in entry
        assert "oracle_ic_val_mean" in entry
        assert "oracle_ic_val_std" in entry
        assert abs(entry["oracle_ic_train_mean"] - 0.01) <= 0.002
        assert 0.005 <= entry["oracle_ic_val_std"] <= 0.015


def test_split_whitelist_raises_on_dev_test_v1() -> None:
    """validate_split_name must strictly reject dev_test_v1 and holdout."""
    with pytest.raises(ValueError, match="strictly forbidden"):
        validate_split_name("dev_test_v1")
    with pytest.raises(ValueError, match="strictly forbidden"):
        validate_split_name("holdout")
    # Authorized splits must pass
    validate_split_name("train")
    validate_split_name("val")


def test_mde_calibration_determinism() -> None:
    """Identical seeds and inputs must produce bitwise identical noise and calibration."""
    rng1 = np.random.default_rng(999)
    s = rng1.standard_normal(500)
    eps = rng1.standard_normal(500) * 0.02

    a1 = calibrate_signal_amplitude(s, eps, target_ic=0.03)
    a2 = calibrate_signal_amplitude(s, eps, target_ic=0.03)
    assert a1 == a2

    n1 = generate_circular_shifted_noise(eps, seed=123)
    n2 = generate_circular_shifted_noise(eps, seed=123)
    np.testing.assert_array_equal(n1, n2)


def test_cached_spectral_ridge_matches_pure_numpy_ridge() -> None:
    """CachedSpectralRidgeCV must match PureNumpyRidge and select_ridge_alpha within 1e-6 tolerance."""
    from research.pipeline.baselines_v2 import PureNumpyRidge, select_ridge_alpha_timeseries_cv
    from research.pipeline.mde_injection_v2 import CachedSpectralRidgeCV

    rng = np.random.default_rng(42)
    n_tr, n_va, d = 800, 200, 30
    X_tr = rng.standard_normal((n_tr, d))
    X_va = rng.standard_normal((n_va, d))
    y_tr = rng.standard_normal(n_tr)

    purge_samples = 10
    best_a_ref, diag_ref = select_ridge_alpha_timeseries_cv(
        X_tr, y_tr, purge_samples=purge_samples, return_diagnostics=True
    )
    ridge_ref = PureNumpyRidge(alpha=best_a_ref).fit(X_tr, y_tr)
    pred_va_ref = ridge_ref.predict(X_va)

    spectral_model = CachedSpectralRidgeCV(X_tr, X_va, purge_samples=purge_samples)
    best_a_spec, diag_spec, pred_va_spec = spectral_model.select_alpha_and_predict(y_tr)

    assert best_a_spec == best_a_ref
    max_pred_diff = float(np.max(np.abs(pred_va_ref - pred_va_spec)))
    assert max_pred_diff < 1e-6, f"Prediction discrepancy too large: {max_pred_diff}"

    # Diagnostics check
    for alpha in diag_ref["cv_losses_by_alpha"]:
        ref_loss = np.mean(diag_ref["cv_losses_by_alpha"][alpha])
        spec_loss = np.mean(diag_spec["cv_losses_by_alpha"][alpha])
        assert abs(ref_loss - spec_loss) < 1e-6


def test_paired_block_bootstrap_approximation_accuracy() -> None:
    """Fast rank-based Pearson block bootstrap CI must be within 0.003 of exact Spearman bootstrap."""
    rng = np.random.default_rng(777)
    N = 10504
    block_size = 24
    n_bootstraps = 500

    y = rng.standard_normal(N)
    preds = {
        "real": 0.03 * y + rng.standard_normal(N),
        "random": rng.standard_normal(N),
        "scramble": 0.01 * y + rng.standard_normal(N),
    }

    # 1. Exact Spearman
    cis_exact, deltas_exact = paired_block_bootstrap_ic_and_ci(
        y_true=y,
        preds_dict=preds,
        block_size=block_size,
        n_bootstraps=n_bootstraps,
        seed=12345,
        exact_spearman=True,
    )

    # 2. Fast Rank-based Pearson approximation
    cis_fast, deltas_fast = paired_block_bootstrap_ic_and_ci(
        y_true=y,
        preds_dict=preds,
        block_size=block_size,
        n_bootstraps=n_bootstraps,
        seed=12345,
        exact_spearman=False,
    )

    max_ci_diff = 0.0
    for m in preds:
        d_low = abs(cis_exact[m][0] - cis_fast[m][0])
        d_high = abs(cis_exact[m][1] - cis_fast[m][1])
        max_ci_diff = max(max_ci_diff, d_low, d_high)

    for k in deltas_exact:
        d_low = abs(deltas_exact[k][0] - deltas_fast[k][0])
        d_high = abs(deltas_exact[k][1] - deltas_fast[k][1])
        max_ci_diff = max(max_ci_diff, d_low, d_high)

    assert max_ci_diff < 0.003, f"Bootstrap approximation error {max_ci_diff} exceeded 0.003 threshold!"

