"""Tests for A14: Unscaled scramble_mixed graph variants and invariants (SPEC v2 §5, §10).

Verifications:
1. Degree sequences (in-degree, out-degree) strictly preserved.
2. Source out-strength (signed and absolute) strictly preserved.
3. Weight multiset strictly preserved.
4. No self-loops and no duplicate edges.
5. Edge overlap <= 5% (thorough mixing).
6. Deterministic SHA-256 invariants under fixed seed.
7. Fail-closed behavior on mixing failure: does not switch seed, does not substitute invalid graphs.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.connectome.synthetic import make_synthetic_connectome
from research.pipeline.g1_graphs import (
    generate_unscaled_scramble_mixed,
    verify_unscaled_scramble_invariants,
)
from research.pipeline.graph_variants import compute_graph_sha256


@pytest.fixture
def base_small_graph():
    """Construct a small 300-neuron sparse synthetic graph for fast CPU testing."""
    g = make_synthetic_connectome(
        n_neurons=300,
        avg_out_degree=6.0,
        n_sensory=16,
        n_motor=16,
        seed=42,
    )
    g.meta["sha256"] = compute_graph_sha256(g)
    return g


def test_a14_unscaled_scramble_invariants(base_small_graph):
    """Verify all topological and weight invariants on unscaled scramble_mixed."""
    seed = 1001
    scram = generate_unscaled_scramble_mixed(
        base_graph=base_small_graph,
        seed=seed,
        target_overlap=0.05,
    )

    summary = verify_unscaled_scramble_invariants(
        orig_graph=base_small_graph,
        scram_graph=scram,
        max_overlap=0.05,
    )

    assert summary["passed"] is True
    assert summary["in_degree_preserved"] is True
    assert summary["out_degree_preserved"] is True
    assert summary["source_signed_strength_preserved"] is True
    assert summary["source_abs_strength_preserved"] is True
    assert summary["max_source_strength_diff"] <= 1e-12
    assert summary["weight_multiset_preserved"] is True
    assert summary["max_weight_diff"] <= 1e-12
    assert summary["no_self_loops"] is True
    assert summary["no_duplicate_edges"] is True
    assert summary["overlap_ratio"] <= 0.05
    assert len(summary["scram_sha256"]) == 64


def test_a14_scramble_determinism_and_hash_invariance(base_small_graph):
    """Verify deterministic reproducibility: same seed gives bitwise identical hash."""
    scram1 = generate_unscaled_scramble_mixed(base_small_graph, seed=2026, target_overlap=0.05)
    scram2 = generate_unscaled_scramble_mixed(base_small_graph, seed=2026, target_overlap=0.05)
    scram3 = generate_unscaled_scramble_mixed(base_small_graph, seed=2027, target_overlap=0.05)

    assert scram1.meta["sha256"] == scram2.meta["sha256"]
    assert np.array_equal(scram1.weights.data, scram2.weights.data)
    assert np.array_equal(scram1.weights.indices, scram2.weights.indices)

    # Different seed yields different topology
    assert scram1.meta["sha256"] != scram3.meta["sha256"]


def test_a14_mixing_failure_fails_closed_without_substitution(base_small_graph):
    """Verify that if mixing fails to reach target overlap, it raises without seed retry."""
    with pytest.raises(RuntimeError) as exc_info:
        # Impossibly low overlap requirement with almost zero attempts
        generate_unscaled_scramble_mixed(
            base_graph=base_small_graph,
            seed=9999,
            target_overlap=0.0001,
            max_attempts_multiplier=0.01,
        )

    err_msg = str(exc_info.value)
    assert "Scramble mixing failed to achieve target overlap" in err_msg
    assert "Fail-closed rule prohibits switching seeds" in err_msg
