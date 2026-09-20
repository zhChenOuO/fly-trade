"""Unit tests for G0c graph controls: unbiased random endpoints, source-wise weight shuffle, and graph provenance.

Covers:
1. random_endpoint_graph:
   - Edge count equals E
   - No self-loops (diagonal entries == 0)
   - No duplicate edges (canonical CSR format, nnz == E)
   - Weights multiset invariant
   - Out-degree distribution across source index deciles is unbiased within statistical tolerance
   - Deterministic and reproducible with seed
2. source_wise_weight_shuffle:
   - Exact topology and edge signs preserved
   - Per-source positive and negative out-strengths strictly preserved
   - Weights multiset invariant
   - Non-trivial shuffle (different from original)
   - Deterministic and reproducible with seed
3. Graph provenance & integrity report:
   - compute_graph_sha256 determinism and sensitivity
   - graph_integrity_report fields and comparative statistics
4. Cache governance in extract_features_v2:
   - Enforces base graph SHA and generator version match for new variants
   - Raises ValueError on tampering or stale cache
   - Backward compatibility for legacy variants
5. Spectral radius normalization:
   - Verifies target rho scaling and spectral_scaling_factor recording across new controls
"""
from __future__ import annotations

import json
import numpy as np
import pytest
import scipy.sparse as sp

from src.connectome.schema import ConnectomeGraph
from research.pipeline.graph_variants import (
    random_endpoint_graph,
    source_wise_weight_shuffle,
    compute_graph_sha256,
    graph_integrity_report,
    normalize_graph_spectral_radius,
    well_mixed_scramble,
)
from research.pipeline.extract_features_v2 import (
    GENERATOR_VERSION,
    load_or_create_variant_graph,
)


@pytest.fixture
def synthetic_graph() -> ConnectomeGraph:
    """Construct a clean sparse synthetic ConnectomeGraph for testing."""
    rng = np.random.default_rng(42)
    n = 100
    E = 300
    # Generate distinct edges without self loops
    src = rng.integers(0, n, size=E * 2)
    dst = (src + rng.integers(1, n, size=E * 2)) % n
    edges = sorted(list(set(zip(src, dst))))[:E]
    E_clean = len(edges)

    s_arr = np.array([e[0] for e in edges], dtype=np.int32)
    d_arr = np.array([e[1] for e in edges], dtype=np.int32)

    # Mixed positive and negative weights
    val = rng.standard_normal(E_clean).astype(np.float32)
    val[val == 0.0] = 0.5

    W = sp.coo_matrix((val, (d_arr, s_arr)), shape=(n, n)).tocsr()
    W.sort_indices()
    return ConnectomeGraph(
        weights=W,
        neuron_ids=[f"neuron_{i}" for i in range(n)],
        neuron_types=["type_a" if i % 2 == 0 else "type_b" for i in range(n)],
        sensory_idx=np.array([0, 1, 2], dtype=np.int64),
        motor_idx=np.array([97, 98, 99], dtype=np.int64),
        meta={"source": "synthetic_test_g0c", "spectral_radius": 1.25},
    )


def test_random_endpoint_graph_properties(synthetic_graph: ConnectomeGraph) -> None:
    """Verify basic structural invariants of random_endpoint_graph."""
    orig_csr = synthetic_graph.weights.tocsr()
    E_orig = orig_csr.nnz

    rand_g, diag = random_endpoint_graph(synthetic_graph, seed=1234, return_diagnostics=True)
    rand_csr = rand_g.weights.tocsr()

    # 1. Edge count equals E
    assert rand_csr.nnz == E_orig
    assert diag["edge_count"] == E_orig

    # 2. No self-loops
    assert np.count_nonzero(rand_csr.diagonal()) == 0
    assert diag["self_loops"] == 0

    # 3. Canonical CSR without duplicates
    assert rand_csr.has_canonical_format is True
    assert rand_csr.has_sorted_indices is True
    assert diag["is_canonical_csr"] is True

    # 4. Weights multiset invariant
    np.testing.assert_allclose(np.sort(orig_csr.data), np.sort(rand_csr.data), rtol=1e-5, atol=1e-6)
    assert diag["weights_multiset_preserved"] is True

    # 5. Metadata and indices preserved
    np.testing.assert_array_equal(rand_g.sensory_idx, synthetic_graph.sensory_idx)
    np.testing.assert_array_equal(rand_g.motor_idx, synthetic_graph.motor_idx)
    assert rand_g.neuron_ids == synthetic_graph.neuron_ids
    assert rand_g.neuron_types == synthetic_graph.neuron_types
    assert rand_g.meta["variant"] == "random_endpoint"
    assert rand_g.meta["seed"] == 1234

    # 6. Reproducibility
    rand_g2 = random_endpoint_graph(synthetic_graph, seed=1234)
    np.testing.assert_array_equal(rand_csr.data, rand_g2.weights.data)
    np.testing.assert_array_equal(rand_csr.indices, rand_g2.weights.indices)
    np.testing.assert_array_equal(rand_csr.indptr, rand_g2.weights.indptr)

    rand_g_diff = random_endpoint_graph(synthetic_graph, seed=5678)
    assert not np.array_equal(rand_csr.indices, rand_g_diff.weights.indices)


def test_random_endpoint_out_degree_decile_uniformity() -> None:
    """Verify that random_endpoint_graph has no source decile sampling bias.

    Specification:
    - Over 5 independent random seeds on N=500, E=10,000 graph.
    - Standard error of a decile proportion is sqrt(p*(1-p)/E) = sqrt(0.1*0.9/10000) = 0.003 (0.3%).
    - Mean across 5 seeds has SE = 0.003 / sqrt(5) = 0.00134 (0.13%).
    - Acceptance criterion: maximum absolute deviation from expected fraction (0.10) is within
      statistical tolerance of 0.015 (1.5%), which represents >10 standard errors.
    """
    n = 500
    E = 10000
    rng = np.random.default_rng(99)
    # Dummy graph with N=500 and E=10000
    dummy_src = rng.integers(0, n, size=E)
    dummy_dst = (dummy_src + rng.integers(1, n, size=E)) % n
    dummy_val = rng.standard_normal(E).astype(np.float32)
    dummy_W = sp.coo_matrix((dummy_val, (dummy_dst, dummy_src)), shape=(n, n)).tocsr()
    base_g = ConnectomeGraph(
        weights=dummy_W,
        neuron_ids=[f"n_{i}" for i in range(n)],
        neuron_types=["t"] * n,
        sensory_idx=np.array([0, 1]),
        motor_idx=np.array([n - 2, n - 1]),
        meta={},
    )

    n_seeds = 5
    decile_counts = np.zeros(10, dtype=np.float64)

    for seed in range(n_seeds):
        g = random_endpoint_graph(base_g, seed=1000 + seed)
        csr = g.weights.tocsr()
        # W[post, pre] -> column is source
        out_degrees = np.bincount(csr.indices, minlength=n)

        edges_per_decile = n // 10
        for d in range(10):
            lo = d * edges_per_decile
            hi = (d + 1) * edges_per_decile if d < 9 else n
            decile_counts[d] += np.sum(out_degrees[lo:hi])

    total_edges = n_seeds * base_g.weights.nnz
    mean_decile_fractions = decile_counts / total_edges
    expected_fraction = 0.10
    tolerance = 0.015  # 1.5% tolerance (>10 SE)

    for d in range(10):
        deviation = abs(mean_decile_fractions[d] - expected_fraction)
        assert deviation <= tolerance, (
            f"Decile {d} fraction {mean_decile_fractions[d]:.4f} deviates by {deviation:.4f} "
            f"from expected {expected_fraction:.4f}, exceeding tolerance {tolerance}"
        )


def test_source_wise_weight_shuffle_properties(synthetic_graph: ConnectomeGraph) -> None:
    """Verify exact invariants of source_wise_weight_shuffle."""
    orig_csr = synthetic_graph.weights.tocsr()
    orig_data = orig_csr.data
    n = synthetic_graph.n_neurons

    shuffled_g, diag = source_wise_weight_shuffle(synthetic_graph, seed=42, return_diagnostics=True)
    shuf_csr = shuffled_g.weights.tocsr()
    shuf_data = shuf_csr.data

    # 1. Topology strictly invariant
    np.testing.assert_array_equal(orig_csr.indices, shuf_csr.indices)
    np.testing.assert_array_equal(orig_csr.indptr, shuf_csr.indptr)
    assert orig_csr.shape == shuf_csr.shape
    assert diag["topology_preserved"] is True

    # 2. Every edge sign strictly preserved
    np.testing.assert_array_equal(orig_data > 0, shuf_data > 0)
    np.testing.assert_array_equal(orig_data < 0, shuf_data < 0)
    assert diag["edge_signs_preserved"] is True

    # 3. Non-trivial shuffle (values differ from original)
    assert not np.array_equal(orig_data, shuf_data)

    # 4. Global weights multiset invariant
    np.testing.assert_allclose(np.sort(orig_data), np.sort(shuf_data), atol=1e-6, rtol=1e-5)
    assert diag["weights_identical_multiset"] is True

    # 5. Per-source positive and negative out-strength strictly equal
    # Source = column of W
    W_pos = sp.csr_matrix((np.maximum(orig_data, 0.0), orig_csr.indices, orig_csr.indptr), shape=orig_csr.shape)
    shuf_pos = sp.csr_matrix((np.maximum(shuf_data, 0.0), shuf_csr.indices, shuf_csr.indptr), shape=shuf_csr.shape)
    pos_out_orig = np.asarray(W_pos.sum(axis=0), dtype=np.float64).ravel()
    pos_out_shuf = np.asarray(shuf_pos.sum(axis=0), dtype=np.float64).ravel()
    np.testing.assert_allclose(pos_out_orig, pos_out_shuf, atol=1e-5, rtol=1e-5)

    W_neg = sp.csr_matrix((np.minimum(orig_data, 0.0), orig_csr.indices, orig_csr.indptr), shape=orig_csr.shape)
    shuf_neg = sp.csr_matrix((np.minimum(shuf_data, 0.0), shuf_csr.indices, shuf_csr.indptr), shape=shuf_csr.shape)
    neg_out_orig = np.asarray(W_neg.sum(axis=0), dtype=np.float64).ravel()
    neg_out_shuf = np.asarray(shuf_neg.sum(axis=0), dtype=np.float64).ravel()
    np.testing.assert_allclose(neg_out_orig, neg_out_shuf, atol=1e-5, rtol=1e-5)

    # 6. Reproducibility
    shuffled_g2 = source_wise_weight_shuffle(synthetic_graph, seed=42)
    np.testing.assert_array_equal(shuf_data, shuffled_g2.weights.data)

    shuffled_g_diff = source_wise_weight_shuffle(synthetic_graph, seed=43)
    assert not np.array_equal(shuf_data, shuffled_g_diff.weights.data)


def test_graph_sha256_and_integrity_report(synthetic_graph: ConnectomeGraph) -> None:
    """Verify compute_graph_sha256 and graph_integrity_report."""
    sha_orig = compute_graph_sha256(synthetic_graph)
    assert len(sha_orig) == 64
    # Determinism
    assert compute_graph_sha256(synthetic_graph) == sha_orig

    # Standalone integrity report
    rep_standalone = graph_integrity_report(synthetic_graph)
    assert rep_standalone["n_neurons"] == synthetic_graph.n_neurons
    assert rep_standalone["nnz"] == synthetic_graph.weights.nnz
    assert rep_standalone["self_loops"] == 0
    assert rep_standalone["is_canonical_csr"] is True
    assert rep_standalone["graph_sha256"] == sha_orig
    assert "base_sha256" not in rep_standalone

    # Comparative report: source_wise_weight_shuffle vs base
    shuffled_g = source_wise_weight_shuffle(synthetic_graph, seed=77)
    rep_shuf = graph_integrity_report(shuffled_g, base=synthetic_graph)
    assert rep_shuf["source_out_strength_equal"] is True
    assert rep_shuf["degree_equal"] is True
    assert rep_shuf["edge_overlap_ratio"] == 1.0
    assert rep_shuf["weights_multiset_equal"] is True
    assert rep_shuf["base_sha256"] == sha_orig
    assert rep_shuf["graph_sha256"] != sha_orig
    assert rep_shuf["max_diff_out_strength"] < 1e-4

    # Comparative report: random_endpoint vs base
    rand_g = random_endpoint_graph(synthetic_graph, seed=88)
    rep_rand = graph_integrity_report(rand_g, base=synthetic_graph)
    assert rep_rand["self_loops"] == 0
    assert rep_rand["weights_multiset_equal"] is True
    assert rep_rand["edge_overlap_ratio"] < 0.20  # Randomized topology


def test_extract_features_cache_governance(tmp_path: pytest.TempPathFactory, synthetic_graph: ConnectomeGraph) -> None:
    """Verify cache validation and tamper detection in load_or_create_variant_graph."""
    cache_dir = tmp_path / "cache_test"
    cache_dir.mkdir(parents=True)

    # 1. Create variant cache for new variant
    g_new = load_or_create_variant_graph(
        "random_endpoint", synthetic_graph, seed=1, cache_dir=cache_dir
    )
    assert g_new.meta["variant"] == "random_endpoint"
    assert g_new.meta["generator_version"] == GENERATOR_VERSION

    # 2. Reload from cache should succeed
    g_reloaded = load_or_create_variant_graph(
        "random_endpoint", synthetic_graph, seed=1, cache_dir=cache_dir
    )
    np.testing.assert_array_equal(g_new.weights.data, g_reloaded.weights.data)

    # 3. Tampered base SHA should raise ValueError
    base_sha = compute_graph_sha256(synthetic_graph)
    cache_file = cache_dir / f"random_endpoint_seed1_{base_sha[:16]}_cache.npz"
    if not cache_file.exists():
        cache_file = cache_dir / "random_endpoint_seed1_cache.npz"

    loaded_raw = dict(np.load(cache_file, allow_pickle=True))
    loaded_raw["base_graph_sha256"] = np.array("tampered_sha256_hash")
    np.savez(cache_file, **loaded_raw)

    with pytest.raises(ValueError, match="cached base_graph_sha256"):
        load_or_create_variant_graph("random_endpoint", synthetic_graph, seed=1, cache_dir=cache_dir)

    # 4. Tampered generator version should raise ValueError
    loaded_raw["base_graph_sha256"] = np.array(base_sha)
    loaded_raw["generator_version"] = np.array("old_v0")
    np.savez(cache_file, **loaded_raw)

    with pytest.raises(ValueError, match="cached generator_version"):
        load_or_create_variant_graph("random_endpoint", synthetic_graph, seed=1, cache_dir=cache_dir)

    # 5. Legacy variants load fine without base_graph_sha256
    legacy_file = cache_dir / "scramble_seed0_cache.npz"
    np.savez(
        legacy_file,
        csr_data=synthetic_graph.weights.data,
        csr_indices=synthetic_graph.weights.indices,
        csr_indptr=synthetic_graph.weights.indptr,
        csr_shape=np.array(synthetic_graph.weights.shape),
        spectral_radius=np.array([1.25], dtype=np.float32),
    )
    g_legacy = load_or_create_variant_graph("scramble", synthetic_graph, seed=0, cache_dir=cache_dir)
    assert g_legacy.meta["variant"] == "scramble"


def test_spectral_radius_normalization_new_variants(synthetic_graph: ConnectomeGraph) -> None:
    """Verify spectral radius normalization and scaling factor recording."""
    target_rho = 0.95

    # 1. random_endpoint
    r_g = random_endpoint_graph(synthetic_graph, seed=10)
    r_norm = normalize_graph_spectral_radius(r_g, target_spectral_radius=target_rho)
    assert np.isclose(r_norm.meta["spectral_radius"], target_rho)
    assert "spectral_scaling_factor" in r_norm.meta
    assert r_norm.meta["spectral_scaling_factor"] > 0

    # 2. source_wise_weight_shuffle
    s_g = source_wise_weight_shuffle(synthetic_graph, seed=20)
    s_norm = normalize_graph_spectral_radius(s_g, target_spectral_radius=target_rho)
    assert np.isclose(s_norm.meta["spectral_radius"], target_rho)
    assert "spectral_scaling_factor" in s_norm.meta

    # 3. scramble_mixed
    m_g = well_mixed_scramble(synthetic_graph, seed=30, target_overlap=0.05)
    m_norm = normalize_graph_spectral_radius(m_g, target_spectral_radius=target_rho)
    assert np.isclose(m_norm.meta["spectral_radius"], target_rho)
    assert "spectral_scaling_factor" in m_norm.meta
