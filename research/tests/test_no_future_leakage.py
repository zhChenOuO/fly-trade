"""Tiny fixtures are test inputs only; the delivered dataset must be exchange data."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import yaml

from research.pipeline.build_dataset import clean_ohlcv, write_dataset
from research.pipeline.render_market import render


CONFIG = yaml.safe_load((Path(__file__).parents[1] / "config/experiment.yaml").read_text())
FIELDS = ["open", "high", "low", "close", "volume"]
BAR = pd.Timedelta(minutes=5)


def market_fixture():
    # Alternating directions, nonconstant volume, and enough bars for all splits.
    x = np.arange(1200)
    close = 100 + np.sin(x / 7)
    opening = close + 0.1 * np.cos(x)
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=len(x), freq="5min").astype("datetime64[ns]"),
        "open": opening, "high": np.maximum(opening, close) + 1,
        "low": np.minimum(opening, close) - 1, "close": close,
        "volume": 1 + x % 13,
    })


def read_images(directory):
    return np.load(directory / "images.npy", mmap_mode="r")


def test_future_mutation_cannot_change_past_image_bytes(tmp_path):
    raw = market_fixture()
    write_dataset(raw, CONFIG, tmp_path / "original", exchange_id="test-fixture")
    samples = pd.read_parquet(tmp_path / "original/samples.parquet")
    sample = samples.iloc[20]
    t = raw.index[raw.timestamp == sample.timestamp - BAR][0]
    changed = raw.copy()
    # Change *every* OHLCV value from t+1 onward, including the label endpoint.
    changed.loc[t + 1:, FIELDS] *= 1000
    write_dataset(changed, CONFIG, tmp_path / "changed", exchange_id="test-fixture")
    before, after = read_images(tmp_path / "original"), read_images(tmp_path / "changed")
    assert before[:21].tobytes() == after[:21].tobytes()
    assert before[-1].tobytes() != before[20].tobytes()  # A constant renderer must fail.
    np.testing.assert_array_equal(before[20], render(raw.loc[t - 47:t, FIELDS].to_numpy()))
    changed_samples = pd.read_parquet(tmp_path / "changed/samples.parquet")
    assert changed_samples.iloc[20].future_return != sample.future_return


def test_images_exclude_timestamps_and_text(tmp_path):
    raw = market_fixture()
    write_dataset(raw, CONFIG, tmp_path / "first", exchange_id="test-fixture")
    shifted = raw.copy()
    shifted.timestamp += pd.Timedelta(days=123)
    write_dataset(shifted, CONFIG, tmp_path / "shifted", exchange_id="test-fixture")
    assert read_images(tmp_path / "first").tobytes() == read_images(tmp_path / "shifted").tobytes()
    # Hand-derived flat candle geometry: no axes, labels, glyphs, or dates.
    flat = np.tile([100, 100, 100, 100, 0], (48, 1))
    expected = np.zeros((64, 64, 3), dtype=np.uint8)
    expected[25, 8:56] = (46, 204, 113)
    np.testing.assert_array_equal(render(flat), expected)
    for path in (tmp_path / "first/audit").glob("*.png"):
        with Image.open(path) as png:
            assert not png.info
            assert png.mode == "RGB" and png.size == (64, 64)


@pytest.mark.parametrize(("future_close", "expected_return", "label"), [
    (110.0, 0.1, 1), (100.0, 0.0, 0), (90.0, -0.1, 0),
])
def test_label_uses_only_close_t_and_t_plus_six(tmp_path, future_close, expected_return, label):
    raw = market_fixture()
    t = 47
    raw.loc[t, ["open", "high", "low", "close"]] = [100, 101, 99, 100]
    raw.loc[t + 6, ["open", "high", "low", "close"]] = [future_close, 1000, 1, future_close]
    # Extreme intermediate, past, and later prices must not enter the label.
    raw.loc[t + 1:t + 5, ["open", "high", "low", "close"]] = [800, 900, 700, 800]
    raw.loc[t + 7:, ["open", "high", "low", "close"]] *= 2
    write_dataset(raw, CONFIG, tmp_path / "data", exchange_id="test-fixture")
    sample = pd.read_parquet(tmp_path / "data/samples.parquet").iloc[0]
    assert sample.timestamp == raw.timestamp.iloc[t] + BAR
    assert sample.future_return == pytest.approx(expected_return)
    assert sample.label == label


def test_splits_purge_all_input_and_label_overlap_and_leave_embargo(tmp_path):
    raw = market_fixture()
    output = tmp_path / "data"
    write_dataset(raw, CONFIG, output, exchange_id="test-fixture")
    samples = pd.read_parquet(output / "samples.parquet")
    metadata = json.loads((output / "splits.json").read_text())
    np.testing.assert_array_equal(samples.sample_id, np.arange(len(samples)))
    np.testing.assert_array_equal(samples.image_idx, samples.sample_id)
    assert samples.timestamp.is_monotonic_increasing
    assert set(samples.split) == {"train", "val", "test"}
    # 1200 raw bars -> cuts at 720, 960, with 54 raw bars excluded after each cut.
    expected_ranges = {"train": (0, 720), "val": (774, 960), "test": (1014, 1200)}
    previous_last = None
    for name, (start, stop) in expected_ranges.items():
        part = samples[samples.split == name]
        positions = raw.timestamp.searchsorted(part.timestamp - BAR)
        assert np.all(positions - 47 >= start)
        assert np.all(positions + 6 < stop)
        np.testing.assert_array_equal(np.diff(positions), np.full(len(positions) - 1, 6))
        assert np.all((positions - 47) % 6 == 0)
        if previous_last is not None:
            assert positions[0] - 47 - previous_last - 1 >= 54
        previous_last = positions[-1] + 6
        detail = metadata["splits"][name]
        assert detail["count"] == len(part)
        assert detail["sample_id_range"] == [int(part.sample_id.iloc[0]), int(part.sample_id.iloc[-1])]
        returns = raw.close.to_numpy()[positions + 6] / raw.close.to_numpy()[positions] - 1
        np.testing.assert_array_equal(part.future_return, returns)
        np.testing.assert_array_equal(part.label, (returns > 0).astype(int))
    assert metadata["embargo_bars"] == 54
    assert read_images(output).shape == (len(samples), 64, 64, 3)
    assert read_images(output).dtype == np.uint8
    assert len(list((output / "audit").glob("*.png"))) == 100
    assert (output / "data_hash.txt").read_text().strip() == hashlib.sha256((output / "raw_ohlcv.parquet").read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        write_dataset(raw, CONFIG, output, exchange_id="test-fixture")


def test_cleaning_deduplicates_sorts_but_never_fills_missing_bars():
    raw = market_fixture()
    dirty = pd.concat([raw.iloc[::-1], raw.iloc[[20]]], ignore_index=True)
    pd.testing.assert_frame_equal(clean_ohlcv(dirty), raw)
    with pytest.raises(ValueError, match="continu"):
        clean_ohlcv(raw.drop(index=30))
    conflicting = raw.iloc[[20]].copy()
    conflicting.close += 0.01
    with pytest.raises(ValueError, match="conflict"):
        clean_ohlcv(pd.concat([raw, conflicting]))


def test_render_is_pure_and_validates_ohlcv():
    window = market_fixture().iloc[:48][FIELDS].to_numpy()
    original = window.copy()
    a, b = render(window), render(window)
    assert a.tobytes() == b.tobytes()
    np.testing.assert_array_equal(window, original)
    with pytest.raises(ValueError):
        render(np.column_stack([np.arange(48), window]))
    window[0, 3] = np.nan
    with pytest.raises(ValueError):
        render(window)


def test_saved_real_dataset_matches_the_contract():
    directory = Path(__file__).parents[1] / "data"
    if not (directory / "splits.json").exists():
        pytest.skip("real dataset has not been downloaded")
    raw = pd.read_parquet(directory / "raw_ohlcv.parquet")
    samples = pd.read_parquet(directory / "samples.parquet")
    images = read_images(directory)
    metadata = json.loads((directory / "splits.json").read_text())
    assert metadata["exchange"] == "binance" or metadata["exchange"] in {"okx", "kraken", "coinbase", "bybit"}
    assert list(raw.columns) == ["timestamp", *FIELDS]
    assert list(samples.columns) == ["sample_id", "timestamp", "image_idx", "future_return", "label", "split"]
    assert raw.timestamp.dtype == samples.timestamp.dtype == np.dtype("datetime64[ns]")
    assert raw.timestamp.is_unique and raw.timestamp.diff().iloc[1:].eq(BAR).all()
    assert raw.timestamp.iloc[-1] + BAR == raw.timestamp.iloc[0] + pd.DateOffset(years=3)
    np.testing.assert_array_equal(samples.sample_id, np.arange(len(samples)))
    np.testing.assert_array_equal(samples.image_idx, samples.sample_id)
    assert samples.timestamp.is_monotonic_increasing
    assert images.shape == (len(samples), 64, 64, 3) and images.dtype == np.uint8
    positions = raw.timestamp.searchsorted(samples.timestamp - BAR)
    np.testing.assert_array_equal(raw.timestamp.iloc[positions].to_numpy() + BAR, samples.timestamp)
    returns = raw.close.to_numpy()[positions + 6] / raw.close.to_numpy()[positions] - 1
    np.testing.assert_array_equal(samples.future_return, returns)
    np.testing.assert_array_equal(samples.label, (returns > 0).astype(int))
    previous_last = None
    for name in ("train", "val", "test"):
        ends = positions[samples.split == name]
        if previous_last is not None:
            assert ends[0] - 47 - previous_last - 1 >= 54
        previous_last = ends[-1] + 6
        assert np.all(np.diff(ends) == 6)
        assert len(ends) == metadata["splits"][name]["count"]
    assert (samples.split == "test").sum() >= 10_000
    with (directory / "raw_ohlcv.parquet").open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    assert digest == metadata["raw_sha256"] == (directory / "data_hash.txt").read_text().strip()
    audit_paths = sorted((directory / "audit").glob("*.png"))
    assert len(audit_paths) == len(set(metadata["audit_sample_ids"])) == 100
    for path, sample_id in zip(audit_paths, metadata["audit_sample_ids"], strict=True):
        t = positions[sample_id]
        np.testing.assert_array_equal(images[sample_id], render(raw.iloc[t - 47:t + 1][FIELDS].to_numpy()))
        with Image.open(path) as png:
            np.testing.assert_array_equal(np.asarray(png), images[sample_id])
            assert not png.info
