"""Unit tests for G0 graph controls: scramble mixing diagnostics, well-mixed scramble, and weight shuffle.

Tests:
1. degree_preserved_scramble emits complete mixing diagnostics and preserves degrees.
2. well_mixed_scramble reaches target overlap (<= 0.05) on synthetic graph.
3. weight_shuffled_graph preserves topology identically and preserves weight multisets.
4. Determinism: identical seeds produce bitwise identical graphs.
5. Legacy default path is bitwise reproducible and backward-compatible.
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from src.connectome.schema import ConnectomeGraph
from research.pipeline.graph_variants import (
    degree_preserved_scramble,
    well_mixed_scramble,
    weight_shuffled_graph,
)


@pytest.fixture
def synthetic_graph() -> ConnectomeGraph:
    """Construct a clean sparse synthetic ConnectomeGraph for testing."""
    rng = np.random.default_rng(42)
    n = 100
    E = 150
    src = rng.integers(0, n, size=E)
    dst = (src + rng.integers(1, n, size=E)) % n
    edges = sorted(list(set(zip(src, dst))))
    E_clean = len(edges)

    s_arr = np.array([e[0] for e in edges], dtype=np.int32)
    d_arr = np.array([e[1] for e in edges], dtype=np.int32)
    # Ensure a mix of positive and negative weights
    val = rng.standard_normal(E_clean).astype(np.float32)
    val[val == 0.0] = 0.5

    W = sp.coo_matrix((val, (d_arr, s_arr)), shape=(n, n)).tocsr()
    return ConnectomeGraph(
        weights=W,
        neuron_ids=[f"neuron_{i}" for i in range(n)],
        neuron_types=["type_a" if i % 2 == 0 else "type_b" for i in range(n)],
        sensory_idx=np.array([0, 1, 2], dtype=np.int64),
        motor_idx=np.array([97, 98, 99], dtype=np.int64),
        meta={"source": "synthetic_test"},
    )


def test_degree_preserved_scramble_diagnostics(synthetic_graph: ConnectomeGraph) -> None:
    """Verify that degree_preserved_scramble emits complete mixing diagnostics."""
    scrambled, diag = degree_preserved_scramble(
        synthetic_graph, seed=101, n_swap_multiplier=2.0, return_diagnostics=True
    )

    # Check diagnostic fields
    assert "attempted_swaps" in diag
    assert "successful_swaps" in diag
    assert "overlap_ratio" in diag
    assert "modified_ratio" in diag
    assert "in_degree_preserved" in diag
    assert "out_degree_preserved" in diag
    assert "sign_ratio_preserved" in diag

    assert diag["in_degree_preserved"] is True
    assert diag["out_degree_preserved"] is True
    assert diag["sign_ratio_preserved"] is True
    assert 0.0 <= diag["overlap_ratio"] <= 1.0
    assert np.isclose(diag["overlap_ratio"] + diag["modified_ratio"], 1.0)

    # Meta also contains diagnostics
    assert "scramble_diagnostics" in scrambled.meta
    assert scrambled.meta["scramble_diagnostics"]["successful_swaps"] == diag["successful_swaps"]


def test_scramble_mixed_overlap_target(synthetic_graph: ConnectomeGraph) -> None:
    """well_mixed_scramble must reach target overlap <= 0.05 and preserve degree sequences."""
    mixed_g, diag = well_mixed_scramble(
        synthetic_graph, seed=202, target_overlap=0.05, max_attempts_multiplier=10.0, return_diagnostics=True
    )

    assert diag["target_reached"] is True
    assert diag["overlap_ratio"] <= 0.05
    assert diag["modified_ratio"] >= 0.95
    assert diag["in_degree_preserved"] is True
    assert diag["out_degree_preserved"] is True
    assert diag["sign_ratio_preserved"] is True
    assert mixed_g.meta["variant"] == "scramble_mixed"


def test_weight_shuffled_graph_properties(synthetic_graph: ConnectomeGraph) -> None:
    """weight_shuffled_graph must strictly preserve topology and weight multisets."""
    orig_csr = synthetic_graph.weights.tocsr()
    orig_data = orig_csr.data

    # 1. Separate signs (default)
    shuffled_sep, diag_sep = weight_shuffled_graph(
        synthetic_graph, seed=303, separate_signs=True, return_diagnostics=True
    )
    s_csr = shuffled_sep.weights.tocsr()

    # Topology is 100% elementwise identical
    np.testing.assert_array_equal(orig_csr.indices, s_csr.indices)
    np.testing.assert_array_equal(orig_csr.indptr, s_csr.indptr)
    assert orig_csr.shape == s_csr.shape

    # Weights multiset identical
    np.testing.assert_allclose(np.sort(orig_data), np.sort(s_csr.data))

    # Positive and negative partitions identically preserved
    pos_mask = orig_data > 0
    neg_mask = orig_data < 0
    np.testing.assert_allclose(np.sort(orig_data[pos_mask]), np.sort(s_csr.data[pos_mask]))
    np.testing.assert_allclose(np.sort(orig_data[neg_mask]), np.sort(s_csr.data[neg_mask]))
    assert not np.array_equal(orig_data, s_csr.data)

    assert diag_sep["topology_preserved"] is True
    assert diag_sep["weights_identical_multiset"] is True
    assert diag_sep["positive_count_preserved"] is True
    assert diag_sep["negative_count_preserved"] is True

    # 2. Joint sign shuffle
    shuffled_joint, diag_joint = weight_shuffled_graph(
        synthetic_graph, seed=303, separate_signs=False, return_diagnostics=True
    )
    j_csr = shuffled_joint.weights.tocsr()
    np.testing.assert_array_equal(orig_csr.indices, j_csr.indices)
    np.testing.assert_array_equal(orig_csr.indptr, j_csr.indptr)
    np.testing.assert_allclose(np.sort(orig_data), np.sort(j_csr.data))


def test_reproducibility_with_seed(synthetic_graph: ConnectomeGraph) -> None:
    """Identical seeds must yield bitwise identical output matrices."""
    # Scramble mixed
    m1 = well_mixed_scramble(synthetic_graph, seed=404, target_overlap=0.05)
    m2 = well_mixed_scramble(synthetic_graph, seed=404, target_overlap=0.05)
    m_diff = well_mixed_scramble(synthetic_graph, seed=405, target_overlap=0.05)

    np.testing.assert_array_equal(m1.weights.toarray(), m2.weights.toarray())
    assert not np.array_equal(m1.weights.toarray(), m_diff.weights.toarray())

    # Weight shuffle
    w1 = weight_shuffled_graph(synthetic_graph, seed=505, separate_signs=True)
    w2 = weight_shuffled_graph(synthetic_graph, seed=505, separate_signs=True)
    w_diff = weight_shuffled_graph(synthetic_graph, seed=506, separate_signs=True)

    np.testing.assert_array_equal(w1.weights.toarray(), w2.weights.toarray())
    assert not np.array_equal(w1.weights.toarray(), w_diff.weights.toarray())


def test_legacy_default_path_compatibility(synthetic_graph: ConnectomeGraph) -> None:
    """Calling degree_preserved_scramble with default arguments returns ConnectomeGraph directly."""
    res = degree_preserved_scramble(synthetic_graph, seed=606)
    assert isinstance(res, ConnectomeGraph)
    assert res.meta["variant"] == "degree_preserved_scramble"
    assert "scramble_diagnostics" in res.meta
    assert res.meta["scramble_diagnostics"]["in_degree_preserved"] is True
