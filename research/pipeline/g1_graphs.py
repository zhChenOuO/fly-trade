"""G1 Connectome Graph Helpers and Invariant Verification (SPEC v2 §5, A14).

Contract:
- Unscaled scramble_mixed graph generation using graph_variants.well_mixed_scramble.
- Mixing stopping rule: default target_overlap = 0.05, max_attempts_multiplier = 5.0.
- If mixing fails to reach target_overlap, fail closed without switching seed or replacing graph.
- Verification of invariants:
  1. In-degree sequence per node identical.
  2. Out-degree sequence per node identical.
  3. Source out-strength (sum of weights per source) identical.
  4. Source out-abs-strength (sum of abs(weights) per source) identical.
  5. Empirical weight multiset identical.
  6. No self-loops (zero diagonal).
  7. No duplicate edges (strictly simple directed graph).
  8. Edge overlap ratio <= target_overlap (e.g. 0.05).
  9. Deterministic hash invariance for fixed seeds.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from src.connectome.schema import ConnectomeGraph
from research.pipeline.graph_variants import (
    well_mixed_scramble,
    compute_graph_sha256,
)


def generate_unscaled_scramble_mixed(
    base_graph: ConnectomeGraph,
    seed: int,
    target_overlap: float = 0.05,
    max_attempts_multiplier: float = 5.0,
) -> ConnectomeGraph:
    """Generate an unscaled scramble_mixed graph variant.

    Stopping rule (SPEC §5, A14):
        Mixing stops when edge overlap with base_graph <= target_overlap
        or max attempts (max_attempts_multiplier * E) is reached.
        If target overlap is not reached, mixing fails closed: do NOT switch
        seed, do NOT replace graph with another candidate.
    """
    scram_graph, diag = well_mixed_scramble(
        graph=base_graph,
        seed=seed,
        target_overlap=target_overlap,
        max_attempts_multiplier=max_attempts_multiplier,
        return_diagnostics=True,
    )

    if not diag.get("target_reached", False) or diag.get("overlap_ratio", 1.0) > target_overlap:
        raise RuntimeError(
            f"Scramble mixing failed to achieve target overlap <= {target_overlap:.4f} "
            f"for seed {seed} (actual overlap: {diag.get('overlap_ratio', 1.0):.4f}). "
            f"Fail-closed rule prohibits switching seeds or substituting invalid graphs (SPEC §5, A14)."
        )

    # Compute and attach structural SHA-256 hash
    sha = compute_graph_sha256(scram_graph)
    scram_graph.meta["sha256"] = sha
    scram_graph.meta["unscaled_scramble_seed"] = seed
    scram_graph.meta["target_overlap"] = target_overlap
    scram_graph.meta["max_attempts_multiplier"] = max_attempts_multiplier
    return scram_graph


def compute_source_out_strengths(graph: ConnectomeGraph) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-source neuron outgoing signed strength and absolute strength.

    In W[i, j] = synapse from j (col) to i (row):
    Source neuron j has outgoing sum sum_i W[i, j] and sum_i |W[i, j]|.
    """
    csc = graph.weights.tocsc()
    n = graph.n_neurons
    signed_strength = np.zeros(n, dtype=np.float64)
    abs_strength = np.zeros(n, dtype=np.float64)

    for j in range(n):
        start = csc.indptr[j]
        end = csc.indptr[j + 1]
        if start < end:
            col_data = csc.data[start:end]
            sorted_col = np.sort(col_data)
            signed_strength[j] = float(np.sum(sorted_col, dtype=np.float64))
            abs_strength[j] = float(np.sum(np.abs(sorted_col), dtype=np.float64))

    return signed_strength, abs_strength


def verify_unscaled_scramble_invariants(
    orig_graph: ConnectomeGraph,
    scram_graph: ConnectomeGraph,
    max_overlap: float = 0.05,
    tol: float = 1e-12,
) -> dict:
    """Rigorously verify all topological and weight invariants for unscaled scramble (SPEC §5, A14).

    Parameters
    ----------
    orig_graph: ConnectomeGraph
        Reference connectome graph.
    scram_graph: ConnectomeGraph
        Scrambled connectome graph to verify.
    max_overlap: float
        Maximum allowable edge overlap ratio (default 0.05).
    tol: float
        Floating point comparison tolerance.

    Returns
    -------
    dict
        Verification summary with all invariant checks and measurements.

    Raises
    ------
    ValueError
        If any invariant is violated.
    """
    n = orig_graph.n_neurons
    if scram_graph.n_neurons != n:
        raise ValueError(f"Neuron count mismatch: orig={n}, scram={scram_graph.n_neurons}")

    # 1. Sensory and Motor indices
    if not np.array_equal(orig_graph.sensory_idx, scram_graph.sensory_idx):
        raise ValueError("Sensory index mismatch between original and scrambled graph")
    if not np.array_equal(orig_graph.motor_idx, scram_graph.motor_idx):
        raise ValueError("Motor index mismatch between original and scrambled graph")

    orig_csr = orig_graph.weights.tocsr()
    scram_csr = scram_graph.weights.tocsr()

    # 2. In-degree and Out-degree sequences
    # In W[i, j]: row i is destination (in-degree), col j is source (out-degree)
    orig_coo = orig_csr.tocoo()
    scram_coo = scram_csr.tocoo()

    orig_in = np.bincount(orig_coo.row, minlength=n)
    scram_in = np.bincount(scram_coo.row, minlength=n)
    if not np.array_equal(orig_in, scram_in):
        max_diff = int(np.max(np.abs(orig_in - scram_in)))
        raise ValueError(f"In-degree sequence invariant violated (max diff = {max_diff})")

    orig_out = np.bincount(orig_coo.col, minlength=n)
    scram_out = np.bincount(scram_coo.col, minlength=n)
    if not np.array_equal(orig_out, scram_out):
        max_diff = int(np.max(np.abs(orig_out - scram_out)))
        raise ValueError(f"Out-degree sequence invariant violated (max diff = {max_diff})")

    # 3. Source out-strength and out-abs-strength
    orig_signed_str, orig_abs_str = compute_source_out_strengths(orig_graph)
    scram_signed_str, scram_abs_str = compute_source_out_strengths(scram_graph)

    max_signed_diff = float(np.max(np.abs(orig_signed_str - scram_signed_str)))
    if max_signed_diff > tol:
        raise ValueError(f"Source signed out-strength invariant violated (max diff = {max_signed_diff:.2e} > {tol})")

    max_abs_diff = float(np.max(np.abs(orig_abs_str - scram_abs_str)))
    if max_abs_diff > tol:
        raise ValueError(f"Source absolute out-strength invariant violated (max diff = {max_abs_diff:.2e} > {tol})")

    # 4. Weight multiset
    orig_weights_sorted = np.sort(orig_coo.data)
    scram_weights_sorted = np.sort(scram_coo.data)
    if len(orig_weights_sorted) != len(scram_weights_sorted):
        raise ValueError(f"Synapse count mismatch: {len(orig_weights_sorted)} vs {len(scram_weights_sorted)}")
    max_weight_diff = float(np.max(np.abs(orig_weights_sorted - scram_weights_sorted)))
    if max_weight_diff > tol:
        raise ValueError(f"Weight multiset invariant violated (max diff = {max_weight_diff:.2e} > {tol})")

    # 5. No self-loops (zero diagonal)
    diag_elements = scram_csr.diagonal()
    if np.any(diag_elements != 0):
        n_loops = int(np.sum(diag_elements != 0))
        raise ValueError(f"Self-loops detected in scrambled graph: {n_loops} non-zero diagonal entries")

    # 6. No duplicate edges
    # CSR format sum_duplicates check or unique (row, col) pairs
    packed_edges = (scram_coo.col.astype(np.int64) << 32) | scram_coo.row.astype(np.int64)
    if len(packed_edges) != len(np.unique(packed_edges)):
        raise ValueError("Duplicate edges detected in scrambled graph")

    # 7. Edge overlap ratio <= max_overlap
    orig_packed = set((orig_coo.col.astype(np.int64) << 32) | orig_coo.row.astype(np.int64))
    common_edges = sum(1 for e in packed_edges if e in orig_packed)
    overlap_ratio = float(common_edges / len(packed_edges)) if len(packed_edges) > 0 else 0.0

    if overlap_ratio > max_overlap:
        raise ValueError(f"Edge overlap ratio {overlap_ratio:.4f} exceeds threshold {max_overlap:.4f}")

    # 8. Hash computation
    orig_sha = orig_graph.meta.get("sha256") or compute_graph_sha256(orig_graph)
    scram_sha = scram_graph.meta.get("sha256") or compute_graph_sha256(scram_graph)

    return {
        "passed": True,
        "n_neurons": n,
        "n_synapses": len(packed_edges),
        "in_degree_preserved": True,
        "out_degree_preserved": True,
        "source_signed_strength_preserved": True,
        "source_abs_strength_preserved": True,
        "max_source_strength_diff": max_signed_diff,
        "weight_multiset_preserved": True,
        "max_weight_diff": max_weight_diff,
        "no_self_loops": True,
        "no_duplicate_edges": True,
        "overlap_ratio": overlap_ratio,
        "max_overlap_threshold": max_overlap,
        "orig_sha256": orig_sha,
        "scram_sha256": scram_sha,
    }
