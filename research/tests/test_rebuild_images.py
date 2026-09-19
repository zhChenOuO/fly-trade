import numpy as np
import pandas as pd

from research.pipeline.rebuild_images import window_for_sample


def test_window_for_sample_matches_run_experiment_get_windows_rule():
    timestamps = pd.date_range("2024-01-01", periods=60, freq="5min")
    opening = 100.0 + np.arange(60)
    raw = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": opening,
            "high": opening + 2.0,
            "low": opening - 2.0,
            "close": opening + 1.0,
            "volume": np.arange(60, dtype=np.float64),
        }
    )

    sample_timestamp = timestamps[49] + pd.Timedelta(minutes=5)
    actual = window_for_sample(raw, sample_timestamp)
    expected = raw.iloc[2:50][["open", "high", "low", "close", "volume"]].to_numpy()

    np.testing.assert_array_equal(actual, expected)
