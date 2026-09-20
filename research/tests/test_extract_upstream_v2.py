import numpy as np
import pytest
import scipy.sparse as sp

from research.pipeline.extract_upstream_v2 import (
    activity_collapse_stats,
    select_random_edges_per_target,
    select_topk_edges,
)
from src.connectome.schema import ConnectomeGraph


def _graph():
    weights = sp.csr_matrix(
        (
            np.array([0.1, -0.7, 0.3, 0.4, -0.2, 0.8], dtype=np.float32),
            (
                np.array([0, 0, 0, 1, 1, 1]),
                np.array([1, 2, 3, 0, 2, 3]),
            ),
        ),
        shape=(4, 4),
    )
    return ConnectomeGraph(
        weights=weights,
        neuron_ids=[str(i) for i in range(4)],
        neuron_types=["unknown"] * 4,
        sensory_idx=np.array([0]),
        motor_idx=np.array([2, 3]),
    )


def test_topk_edges_use_largest_absolute_incoming_weights():
    graph = _graph()

    target, source, weight = select_topk_edges(graph, np.array([0, 1]), k=2)

    np.testing.assert_array_equal(target, [0, 0, 1, 1])
    np.testing.assert_array_equal(source, [2, 3, 0, 3])
    np.testing.assert_allclose(weight, [-0.7, 0.3, 0.4, 0.8])


def test_null_edges_match_per_target_counts_and_seed_deterministically():
    graph = _graph()
    counts = np.array([2, 1], dtype=np.int64)

    first = select_random_edges_per_target(graph, np.array([0, 1]), counts, seed=0)
    second = select_random_edges_per_target(graph, np.array([0, 1]), counts, seed=0)

    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(np.bincount(first[0], minlength=2), counts)
    assert len(set(first[1][:2].tolist())) == 2


def test_activity_collapse_ratio_counts_constant_and_zero_columns():
    values = np.full((100, 10), 0.1, dtype=np.float16)
    values[:, 9] = np.linspace(0, 1, 100, dtype=np.float16)

    result = activity_collapse_stats(values)

    assert result["constant_or_zero_columns"] == 9
    assert result["constant_or_zero_ratio"] == pytest.approx(0.9)
