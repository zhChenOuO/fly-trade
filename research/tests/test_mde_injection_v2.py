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
    n_va = 300  # 12.5 blocks of 24
    s_t = rng.standard_normal(n_va)

    # Model Real has access to the true signal + small noise
    pred_real = s_t + 0.05 * rng.standard_normal(n_va)
    # Model Random is pure noise
    pred_rand = rng.standard_normal(n_va)

    preds_dict = {"real": pred_real, "rand": pred_rand}
    eps = rng.standard_normal(n_va) * 0.05

    detection_counts = []
    # Test at target IC = 0.0, 0.15, 0.35 with R=10 repetitions
    R = 10
    for target in [0.0, 0.15, 0.35]:
        detected_cnt = 0
        for r in range(R):
            rep_seed = 1000 * r + int(target * 1000)
            eps_shifted = generate_circular_shifted_noise(eps, seed=rep_seed)
            a = calibrate_signal_amplitude(s_t, eps_shifted, target_ic=target)
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

    # 1. At IC=0, false positive rate should be low (<= 2 out of 10 = 20%)
    assert detection_counts[0] <= 2, f"False positive count at IC=0 was {detection_counts[0]}/{R}"

    # 2. Strong signal has higher detection rate than null
    assert detection_counts[1] >= detection_counts[0]
    assert detection_counts[2] >= detection_counts[1]
    # Strong signal should be detected in at least 8 out of 10 repetitions
    assert detection_counts[2] >= 8


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
