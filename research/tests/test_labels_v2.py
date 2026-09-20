"""Tests for Research v2 multi-horizon labels, action mapping, and purge rules.

Covers (per task_v2_batch1.md):
1. Future mutation beyond horizon does not affect label_h; modifying close[t+h] or close[t] does.
2. MFE/MAE exclusively use forward (t, t+6] bars; modifying bars after t+6 has no effect.
3. Action mapping boundaries (exactly +/- cost_threshold is HOLD) and cost threshold derivation.
4. Split boundaries and purge: no overlap between train samples (input/target) and val/dev_test.
5. Real data contract: random audit of 50 samples verifying input max time <= t, targets > t.
6. Manifest SHA256 hashes match actual files on disk.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import yaml

from research.pipeline.labels import (
    ACTION_BUY,
    ACTION_HOLD,
    ACTION_SELL,
    classify_action,
    compute_cost_threshold,
    compute_excursions_6,
    compute_future_returns,
    compute_future_volatility_6,
    compute_labels,
)

BAR = pd.Timedelta(minutes=5)
ROOT = Path(__file__).resolve().parents[1]


def make_synthetic_ohlcv(n_bars: int = 300, seed: int = 123) -> pd.DataFrame:
    """Create deterministic synthetic 5m OHLCV data."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01 00:00:00", periods=n_bars, freq="5min")
    x = np.arange(n_bars, dtype=np.float64)
    close = 100.0 + np.cumsum(rng.normal(0.0, 0.5, size=n_bars))
    opening = close + rng.normal(0.0, 0.2, size=n_bars)
    high = np.maximum(opening, close) + rng.uniform(0.1, 1.0, size=n_bars)
    low = np.minimum(opening, close) - rng.uniform(0.1, 1.0, size=n_bars)
    vol = rng.uniform(10.0, 100.0, size=n_bars)
    return pd.DataFrame({
        "timestamp": ts,
        "open": opening,
        "high": high,
        "low": low,
        "close": close,
        "volume": vol,
    })


def test_future_mutation_beyond_horizon_does_not_affect_label_h():
    """Modifying bars after t+h must leave future_return_h invariant."""
    raw = make_synthetic_ohlcv(150)
    closes = raw["close"].to_numpy()
    t = 50

    for h in [1, 3, 6, 12]:
        base_returns = compute_future_returns(closes, np.array([t]), horizons=[h])[h][0]

        # 1. Modify bars strictly AFTER t+h
        mutated_after = closes.copy()
        mutated_after[t + h + 1 :] *= 2.0
        ret_after = compute_future_returns(mutated_after, np.array([t]), horizons=[h])[h][0]
        assert np.isclose(base_returns, ret_after, atol=1e-12)

        # 2. Modify close AT t+h
        mutated_at = closes.copy()
        mutated_at[t + h] *= 1.02
        ret_at = compute_future_returns(mutated_at, np.array([t]), horizons=[h])[h][0]
        assert not np.isclose(base_returns, ret_at, atol=1e-6)

        # 3. Modify close AT t
        mutated_t = closes.copy()
        mutated_t[t] *= 1.02
        ret_t = compute_future_returns(mutated_t, np.array([t]), horizons=[h])[h][0]
        assert not np.isclose(base_returns, ret_t, atol=1e-6)


def test_mfe_mae_only_use_forward_six_bars():
    """MFE and MAE must only depend on (t, t+6]; bars after t+6 must have no effect."""
    raw = make_synthetic_ohlcv(100)
    highs = raw["high"].to_numpy()
    lows = raw["low"].to_numpy()
    closes = raw["close"].to_numpy()
    t = 40

    base_mfe, base_mae = compute_excursions_6(highs, lows, closes, np.array([t]), horizon=6)
    base_vol = compute_future_volatility_6(closes, np.array([t]), horizon=6)

    # 1. Mutate all bars strictly AFTER t+6
    mut_highs = highs.copy()
    mut_lows = lows.copy()
    mut_closes = closes.copy()
    mut_highs[t + 7 :] *= 100.0
    mut_lows[t + 7 :] *= 0.01
    mut_closes[t + 7 :] *= 50.0

    mut_mfe, mut_mae = compute_excursions_6(mut_highs, mut_lows, mut_closes, np.array([t]), horizon=6)
    mut_vol = compute_future_volatility_6(mut_closes, np.array([t]), horizon=6)

    assert np.isclose(base_mfe[0], mut_mfe[0], atol=1e-12)
    assert np.isclose(base_mae[0], mut_mae[0], atol=1e-12)
    assert np.isclose(base_vol[0], mut_vol[0], atol=1e-12)

    # 2. Mutate high within (t, t+6]
    mut_highs_in = highs.copy()
    mut_highs_in[t + 3] *= 1.5
    in_mfe, _ = compute_excursions_6(mut_highs_in, lows, closes, np.array([t]), horizon=6)
    assert in_mfe[0] > base_mfe[0]

    # 3. Mutate low within (t, t+6]
    mut_lows_in = lows.copy()
    mut_lows_in[t + 3] *= 0.5
    _, in_mae = compute_excursions_6(highs, mut_lows_in, closes, np.array([t]), horizon=6)
    assert in_mae[0] < base_mae[0]

    # 4. Excursion ordering: MAE <= future_return_6 <= MFE
    ret_6 = compute_future_returns(closes, np.array([t]), horizons=[6])[6][0]
    assert base_mae[0] <= ret_6 <= base_mfe[0]


def test_action_mapping_boundaries_and_cost_derivation():
    """Verify cost_threshold = 2*fee + slippage and strict threshold boundary semantics."""
    cost_fee = 0.0004
    cost_slip = 0.0002
    ct = compute_cost_threshold(cost_fee, cost_slip)

    # Exactly 2 * fee + slippage
    assert np.isclose(ct, 0.0010)

    # Strict boundary checks (per task_v2_batch1.md line 31: 恰等於 +/- cost_threshold 為 HOLD)
    assert classify_action(ct, ct) == ACTION_HOLD
    assert classify_action(-ct, ct) == ACTION_HOLD
    assert classify_action(0.0, ct) == ACTION_HOLD

    # Strict inequality checks
    assert classify_action(ct + 1e-9, ct) == ACTION_BUY
    assert classify_action(-ct - 1e-9, ct) == ACTION_SELL
    assert classify_action(0.05, ct) == ACTION_BUY
    assert classify_action(-0.05, ct) == ACTION_SELL

    # NaN handling
    assert np.isnan(classify_action(np.nan, ct))

    # Vectorized check
    returns = np.array([ct, -ct, ct + 1e-5, -ct - 1e-5, 0.0, np.nan])
    expected = np.array([ACTION_HOLD, ACTION_HOLD, ACTION_BUY, ACTION_SELL, ACTION_HOLD, np.nan], dtype=object)
    actual = classify_action(returns, ct)
    for a, e in zip(actual, expected):
        if pd.isna(e):
            assert pd.isna(a)
        else:
            assert a == e


def test_split_boundary_and_purge_isolation():
    """Verify that train input and label intervals (including h=12) never overlap val/dev_test."""
    data_dir = ROOT / "data"
    raw_path = data_dir / "raw_ohlcv.parquet"
    samples_path = data_dir / "samples.parquet"
    splits_path = data_dir / "splits.json"
    labels_path = data_dir / "labels_v2.parquet"

    if not (raw_path.exists() and samples_path.exists() and splits_path.exists() and labels_path.exists()):
        pytest.skip("Data files not present; run build_labels_v2 first.")

    raw = pd.read_parquet(raw_path)
    samples = pd.read_parquet(samples_path)
    labels = pd.read_parquet(labels_path)
    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    positions = np.asarray(raw.timestamp.searchsorted(samples.timestamp - BAR), dtype=np.int64)

    # 1. Train split checks
    train_mask = labels["split"] == "train"
    train_pos = positions[train_mask]
    train_labels = labels[train_mask]
    train_stop = splits["splits"]["train"]["raw_bar_range"][1]  # 189388

    # For any train sample where future_return_12 is NOT NaN, t + 12 < train_stop
    valid_h12 = train_labels["future_return_12"].notna()
    assert np.all(train_pos[valid_h12] + 12 < train_stop)

    # The single boundary sample that crosses train_stop MUST be NaN
    boundary_cross = train_pos + 12 >= train_stop
    assert np.all(train_labels.loc[boundary_cross, "future_return_12"].isna())
    assert boundary_cross.sum() == 1  # exactly 1 sample (the last train sample)

    # 2. Check no overlap with validation input window
    val_mask = labels["split"] == "val"
    val_pos = positions[val_mask]
    val_first_input_start = val_pos[0] - 47  # earliest input bar of val (189397)

    # Max future bar of ANY valid train label (including h=12)
    max_train_label_bar = (train_pos[valid_h12] + 12).max()
    assert max_train_label_bar < val_first_input_start
    # Positive gap ensuring purge requirement
    assert val_first_input_start - max_train_label_bar >= 14

    # 3. Check val and dev_test_v1
    test_mask = labels["split"] == "dev_test_v1"
    test_pos = positions[test_mask]
    test_first_input_start = test_pos[0] - 47

    val_labels = labels[val_mask]
    val_valid_h12 = val_labels["future_return_12"].notna()
    val_stop = splits["splits"]["val"]["raw_bar_range"][1]
    assert np.all(val_pos[val_valid_h12] + 12 < val_stop)
    max_val_label_bar = (val_pos[val_valid_h12] + 12).max()
    assert max_val_label_bar < test_first_input_start


def test_real_dataset_contract_random_samples():
    """Audit 50 random samples: max input time <= t, targets strictly from t onward."""
    data_dir = ROOT / "data"
    raw_path = data_dir / "raw_ohlcv.parquet"
    samples_path = data_dir / "samples.parquet"
    labels_path = data_dir / "labels_v2.parquet"
    cfg_path = ROOT / "config/experiment_v2.yaml"

    if not (raw_path.exists() and samples_path.exists() and labels_path.exists() and cfg_path.exists()):
        pytest.skip("Data files not present; run build_labels_v2 first.")

    raw = pd.read_parquet(raw_path)
    samples = pd.read_parquet(samples_path)
    labels = pd.read_parquet(labels_path)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ct = compute_cost_threshold(cfg["cost"]["cost_fee_per_side"], cfg["cost"]["cost_slippage"])

    positions = np.asarray(raw.timestamp.searchsorted(samples.timestamp - BAR), dtype=np.int64)
    closes = raw["close"].to_numpy()
    highs = raw["high"].to_numpy()
    lows = raw["low"].to_numpy()

    rng = np.random.default_rng(42)
    audit_indices = rng.choice(len(labels), size=50, replace=False)

    for idx in audit_indices:
        t = positions[idx]
        t_close_time = raw.timestamp.iloc[t] + BAR
        sample_time = labels["timestamp"].iloc[idx]

        # 1. Sample timestamp is exactly the close time of candle t
        assert sample_time == t_close_time

        # 2. Input candles [t-47..t]: all open times <= t's open time, all close times <= sample_time
        input_open_times = raw.timestamp.iloc[t - 47 : t + 1]
        assert input_open_times.max() <= raw.timestamp.iloc[t]
        assert input_open_times.max() + BAR == sample_time

        # 3. Continuous future returns
        for h in [1, 3, 6, 12]:
            stored_val = labels[f"future_return_{h}"].iloc[idx]
            if pd.notna(stored_val):
                target_open_time = raw.timestamp.iloc[t + h]
                assert target_open_time > raw.timestamp.iloc[t]
                expected_ret = np.log(closes[t + h] / closes[t])
                assert np.isclose(stored_val, expected_ret, atol=1e-12)

        # 4. 6-bar forward volatility and excursions
        stored_vol = labels["future_volatility_6"].iloc[idx]
        stored_mfe = labels["maximum_favorable_excursion_6"].iloc[idx]
        stored_mae = labels["maximum_adverse_excursion_6"].iloc[idx]

        if pd.notna(stored_vol):
            fut_c = closes[t + 1 : t + 7]
            prev_c = np.concatenate([[closes[t]], fut_c[:-1]])
            bar_rets = np.log(fut_c / prev_c)
            expected_vol = np.std(bar_rets, ddof=1)
            assert np.isclose(stored_vol, expected_vol, atol=1e-12)

            expected_mfe = np.log(np.max(highs[t + 1 : t + 7]) / closes[t])
            expected_mae = np.log(np.min(lows[t + 1 : t + 7]) / closes[t])
            assert np.isclose(stored_mfe, expected_mfe, atol=1e-12)
            assert np.isclose(stored_mae, expected_mae, atol=1e-12)

        # 5. Action label matches cost_threshold
        stored_act = labels["action"].iloc[idx]
        ret_6 = labels["future_return_6"].iloc[idx]
        if pd.notna(ret_6):
            if ret_6 > ct:
                assert stored_act == ACTION_BUY
            elif ret_6 < -ct:
                assert stored_act == ACTION_SELL
            else:
                assert stored_act == ACTION_HOLD


def test_manifest_v2_hashes_match():
    """Verify manifest_v2.json SHA256 digests match actual files on disk."""
    data_dir = ROOT / "data"
    manifest_path = data_dir / "manifest_v2.json"
    cfg_path = ROOT / "config/experiment_v2.yaml"
    labels_path = data_dir / "labels_v2.parquet"
    raw_path = data_dir / "raw_ohlcv.parquet"

    if not manifest_path.exists():
        pytest.skip("manifest_v2.json does not exist; run build_labels_v2 first.")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # 1. Config hash
    actual_cfg_hash = hashlib.sha256(cfg_path.read_bytes()).hexdigest()
    assert actual_cfg_hash == manifest["experiment_config"]["sha256"]

    # 2. Labels hash
    actual_labels_hash = hashlib.sha256(labels_path.read_bytes()).hexdigest()
    assert actual_labels_hash == manifest["labels_v2_parquet"]["sha256"]

    # 3. Raw OHLCV hash
    actual_raw_hash = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    assert actual_raw_hash == manifest["data_period"]["raw_sha256"]

    # 4. Total sample count
    labels = pd.read_parquet(labels_path)
    assert len(labels) == manifest["labels_v2_parquet"]["total_samples"]
    assert len(labels) == 52564
