import numpy as np

from research.pipeline.run_phase6 import holm_adjust, paired_block_bootstrap_ic


def test_holm_adjustment_uses_full_declared_family_and_marks_missing_values():
    adjusted = holm_adjust({"a": 0.01, "b": 0.04, "unavailable": None})

    assert adjusted["a"] == 0.03
    assert adjusted["b"] == 0.08
    assert adjusted["unavailable"] is None


def test_paired_block_bootstrap_is_deterministic_and_detects_known_ordering():
    y = np.arange(120, dtype=np.float64)
    predictions = {
        "signal": y + np.sin(y) * 0.01,
        "reverse": -y,
    }
    pairs = {"signal_minus_reverse": ("signal", "reverse")}

    first = paired_block_bootstrap_ic(
        y, predictions, pairs, block_size=24, n_bootstraps=200, seed=17
    )
    second = paired_block_bootstrap_ic(
        y, predictions, pairs, block_size=24, n_bootstraps=200, seed=17
    )

    assert first == second
    result = first["comparisons"]["signal_minus_reverse"]
    assert result["delta_ic"] > 1.5
    assert result["ci_95"][0] > 1.0
    assert result["p_two_sided"] < 0.02
