import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")  # run_phase1a 在模組層 import torch;無 torch(如 Mac)時跳過

from research.pipeline.render_market import render  # noqa: E402
from research.run_phase1a import (  # noqa: E402
    get_windows,
    market_state,
    preflight_summary,
    reproducibility_summary,
    variant_images,
    verify_render_alignment,
)


def _raw(n=50):
    timestamp = pd.date_range("2025-01-01", periods=n, freq="5min")
    close = 100.0 + np.arange(n) * 0.1
    opening = np.r_[close[0], close[:-1]]
    return pd.DataFrame(
        {
            "timestamp": timestamp,
            "open": opening,
            "high": np.maximum(opening, close) + 1.0,
            "low": np.minimum(opening, close) - 1.0,
            "close": close,
            "volume": np.arange(n, dtype=float) + 1.0,
        }
    )


def _samples(raw):
    return pd.DataFrame(
        {
            "sample_id": [10, 11],
            "timestamp": [raw.timestamp.iloc[47] + pd.Timedelta(minutes=5),
                          raw.timestamp.iloc[48] + pd.Timedelta(minutes=5)],
            "image_idx": [0, 1],
            "split": ["val", "val"],
        }
    )


def test_get_windows_matches_decision_bar_timestamp_rule():
    raw = _raw()
    samples = _samples(raw)

    windows = get_windows(samples, raw)

    columns = ["open", "high", "low", "close", "volume"]
    np.testing.assert_array_equal(windows[0], raw.iloc[0:48][columns].to_numpy())
    np.testing.assert_array_equal(windows[1], raw.iloc[1:49][columns].to_numpy())


def test_full_render_alignment_uses_only_label_free_sample_columns():
    raw = _raw()
    samples = _samples(raw)
    windows = get_windows(samples, raw)
    images = np.stack([render(window) for window in windows])

    checked = verify_render_alignment(samples, raw, images, chunk_size=1)

    assert checked == len(samples)
    broken = images.copy()
    broken[0, 0, 0, 0] = 1
    with pytest.raises(AssertionError, match="images.npy"):
        verify_render_alignment(samples, raw, broken, chunk_size=1)


def test_variant_images_reuses_the_registered_perturbations():
    raw = _raw()
    samples = _samples(raw)
    images = np.stack([render(window) for window in get_windows(samples, raw)])
    sampled = samples.loc[samples.split == "val"].sample(2, random_state=0)

    variants = variant_images(samples, raw, images, n=2, seed=0)

    assert set(variants) == {
        "original", "flip_price", "shuffle_time", "mask_recent25", "black", "mean_image"
    }
    np.testing.assert_array_equal(variants["original"], images[sampled.image_idx.to_numpy()])
    assert not variants["black"].any()
    assert all(value.shape == (2, 64, 64, 3) for value in variants.values())


def test_market_state_is_six_state_series_without_label_columns():
    raw = _raw(60)
    timestamps = [raw.timestamp.iloc[i] + pd.Timedelta(minutes=5) for i in range(47, 53)]
    samples = pd.DataFrame(
        {
            "sample_id": np.arange(6),
            "timestamp": timestamps,
            "split": ["train", "train", "train", "val", "test", "test"],
        }
    )

    states = market_state(samples, raw)

    assert states.index.tolist() == samples.sample_id.tolist()
    assert states.between(0, 5).all()


def test_preflight_requires_minority_actions_and_input_dependent_margin():
    buy = np.r_[np.ones(19), 0.0]
    sell = np.r_[np.zeros(19), 1.0]

    result = preflight_summary(buy, sell)

    assert result["passed"] is True
    assert result["minority_action_ratio"] == pytest.approx(0.05)
    assert result["margin_range"] == pytest.approx(2.0)
    collapsed = preflight_summary(np.ones(20), np.zeros(20))
    assert collapsed["passed"] is False


def test_reproducibility_allows_only_measured_score_roundoff():
    first_buy = np.array([1.0, -1.0])
    first_sell = np.array([-1.0, 1.0])
    second_buy = first_buy + np.array([4.0e-7, 0.0])

    result = reproducibility_summary(first_buy, first_sell, second_buy, first_sell)

    assert result["passed"] is True
    assert result["scores_exact"] is False
    assert result["actions_exact"] is True
    assert result["max_abs_score_diff"] == pytest.approx(4.0e-7)
    assert result["actions_flipped"] == 0


def test_reproducibility_only_allows_action_flips_inside_margin_tolerance():
    tol = 5.0e-7
    first_buy = np.array([0.1 * tol, 1.0, -1.0])
    first_sell = np.array([0.0, 0.0, 0.0])
    second_buy = np.array([-0.2 * tol, 1.0, -1.0])

    result = reproducibility_summary(first_buy, first_sell, second_buy, first_sell)

    assert result["passed"] is True
    assert result["actions_exact"] is False
    assert result["actions_flipped"] == 1
    assert result["allowed_action_flips"] == 1
    assert result["flipped_samples"][0]["abs_reference_margin"] < result["score_tolerance"]


def test_reproducibility_rejects_action_flip_outside_margin_tolerance():
    tol = 5.0e-7
    first_buy = np.array([0.6 * tol, 1.0, -1.0])
    first_sell = np.array([-0.6 * tol, 0.0, 0.0])
    second_buy = np.array([-0.1 * tol, 1.0, -1.0])
    second_sell = np.array([0.1 * tol, 0.0, 0.0])

    result = reproducibility_summary(first_buy, first_sell, second_buy, second_sell)

    assert result["scores_within_tolerance"] is True
    assert result["actions_flipped"] == 1
    assert result["disallowed_action_flips"] == 1
    assert result["passed"] is False


def test_reproducibility_rejects_scores_above_measured_tolerance():
    first_buy = np.array([1.0, -1.0])
    first_sell = np.array([-1.0, 1.0])
    second_buy = first_buy + np.array([8.0e-7, 0.0])

    result = reproducibility_summary(first_buy, first_sell, second_buy, first_sell)

    assert result["scores_within_tolerance"] is False
    assert result["passed"] is False
