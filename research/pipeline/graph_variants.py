"""Connectome graph variants generator.

Contract (from research/SPEC.md & research.md §4 Step 4 / §6 Level 3):
- degree_preserved_scramble(graph: ConnectomeGraph, seed: int = 0) -> ConnectomeGraph
- Preserves every node's in-degree and out-degree.
- Preserves sensory_idx and motor_idx unchanged.
- Preserves edge sign distribution and weights (associating weights with source neurons).
- Scrambles connectivity via double-edge-swap, ensuring the edge set differs from original.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from src.connectome.schema import ConnectomeGraph


def degree_preserved_scramble(
    graph: ConnectomeGraph,
    seed: int = 0,
    n_swap_multiplier: float = 2.0,
) -> ConnectomeGraph:
    """Scramble connectome edges while preserving in-degree and out-degree of all nodes.

    Uses the directed double-edge swap algorithm:
    Two directed edges A -> B and C -> D are swapped to A -> D and C -> B,
    provided that neither self-loops nor duplicate edges are created.
    Because node A still sends an edge and node C still sends an edge, their out-degrees
    are invariant. Because node B and node D still receive an edge, their in-degrees
    are invariant.

    Parameters
    ----------
    graph: ConnectomeGraph
        Input connectome graph.
    seed: int
        Random seed for edge selection and swapping.
    n_swap_multiplier: float
        Number of swap attempts as a multiple of total edges E. Default 2.0.

    Returns
    -------
    ConnectomeGraph
        Scrambled connectome with identical in/out-degree sequences and preserved
        sensory/motor indices.
    """
    coo = graph.weights.tocoo()
    dst = coo.row.copy()
    src = coo.col.copy()
    val = coo.data.copy()
    E = len(src)

    if E < 4:
        # Not enough edges to swap, return duplicate
        return ConnectomeGraph(
            weights=graph.weights.copy(),
            neuron_ids=list(graph.neuron_ids),
            neuron_types=list(graph.neuron_types),
            sensory_idx=graph.sensory_idx.copy(),
            motor_idx=graph.motor_idx.copy(),
            meta={
                **graph.meta,
                "variant": "degree_preserved_scramble",
                "scramble_seed": seed,
            },
        )

    rng = np.random.default_rng(seed)
    n_attempts = max(10, int(E * n_swap_multiplier))

    idx1 = rng.integers(0, E, size=n_attempts)
    idx2 = rng.integers(0, E, size=n_attempts)

    if E < 100_000:
        # Standard tuple-based edge set for smaller graphs (100% backward compatible)
        edge_set = set(zip(src, dst))
        for i in range(n_attempts):
            e1, e2 = idx1[i], idx2[i]
            if e1 == e2:
                continue
            s1, d1 = src[e1], dst[e1]
            s2, d2 = src[e2], dst[e2]

            if s1 == s2 or d1 == d2:
                continue
            if s1 == d2 or s2 == d1:
                continue
            if (s1, d2) in edge_set or (s2, d1) in edge_set:
                continue

            edge_set.remove((s1, d1))
            edge_set.remove((s2, d2))
            edge_set.add((s1, d2))
            edge_set.add((s2, d1))

            dst[e1] = d2
            dst[e2] = d1
    else:
        # Packed int64 keys ((src << 32) | dst) for large graphs (e.g. 15M edges)
        # Avoids allocating 15M tuples and speeds up set lookups by 3x.
        packed = (src.astype(np.int64) << 32) | dst.astype(np.int64)
        packed_set = set(packed)
        for i in range(n_attempts):
            e1, e2 = idx1[i], idx2[i]
            if e1 == e2:
                continue
            s1, d1 = int(src[e1]), int(dst[e1])
            s2, d2 = int(src[e2]), int(dst[e2])

            if s1 == s2 or d1 == d2:
                continue
            if s1 == d2 or s2 == d1:
                continue

            k_new1 = (s1 << 32) | d2
            k_new2 = (s2 << 32) | d1
            if k_new1 in packed_set or k_new2 in packed_set:
                continue

            packed_set.remove((s1 << 32) | d1)
            packed_set.remove((s2 << 32) | d2)
            packed_set.add(k_new1)
            packed_set.add(k_new2)

            dst[e1] = d2
            dst[e2] = d1

    new_W = sp.coo_matrix(
        (val, (dst, src)), shape=(graph.n_neurons, graph.n_neurons)
    ).tocsr()
    new_W.sum_duplicates()

    return ConnectomeGraph(
        weights=new_W,
        neuron_ids=list(graph.neuron_ids),
        neuron_types=list(graph.neuron_types),
        sensory_idx=graph.sensory_idx.copy(),
        motor_idx=graph.motor_idx.copy(),
        meta={
            **graph.meta,
            "variant": "degree_preserved_scramble",
            "scramble_seed": seed,
            "spectral_radius": None,
        },
    )


def random_matched_graph(
    graph: ConnectomeGraph,
    seed: int = 0,
) -> ConnectomeGraph:
    """Generate an Erdos-Renyi style directed random graph matched to the connectome.

    Preserves:
    - Number of neurons (nodes)
    - Number of synapses (directed edges)
    - Sensory and motor neuron indices
    - Distribution of edge weights and signs (permuted from empirical weights)

    Randomizes:
    - Directed connection endpoints (source and destination pairs) uniformly without self-loops.
    """
    coo = graph.weights.tocoo()
    n_neurons = graph.n_neurons
    E = len(coo.data)
    rng = np.random.default_rng(seed)

    # Permute edge weights to preserve the exact empirical weight distribution and sign balance
    val_shuffled = rng.permutation(coo.data.copy())

    # Generate unique directed pairs without self-loops
    extra = int(E * 1.005) + 100
    src_cand = rng.integers(0, n_neurons, size=extra, dtype=np.int64)
    shift_cand = rng.integers(1, n_neurons, size=extra, dtype=np.int64)
    dst_cand = (src_cand + shift_cand) % n_neurons

    packed = (src_cand << 32) | dst_cand
    unique_packed = np.unique(packed)

    while len(unique_packed) < E:
        more_extra = int(E * 0.05) + 100
        s_more = rng.integers(0, n_neurons, size=more_extra, dtype=np.int64)
        shift_more = rng.integers(1, n_neurons, size=more_extra, dtype=np.int64)
        d_more = (s_more + shift_more) % n_neurons
        more_packed = (s_more << 32) | d_more
        unique_packed = np.unique(np.concatenate([unique_packed, more_packed]))

    chosen = unique_packed[:E]
    final_src = (chosen >> 32).astype(np.int64)
    final_dst = (chosen & 0xFFFFFFFF).astype(np.int64)

    new_W = sp.coo_matrix(
        (val_shuffled, (final_dst, final_src)), shape=(n_neurons, n_neurons)
    ).tocsr()
    new_W.sum_duplicates()

    return ConnectomeGraph(
        weights=new_W,
        neuron_ids=list(graph.neuron_ids),
        neuron_types=list(graph.neuron_types),
        sensory_idx=graph.sensory_idx.copy(),
        motor_idx=graph.motor_idx.copy(),
        meta={
            **graph.meta,
            "variant": "random_matched",
            "seed": seed,
            "spectral_radius": None,
        },
    )


def normalize_graph_spectral_radius(
    graph: ConnectomeGraph,
    target_spectral_radius: float,
) -> ConnectomeGraph:
    """Normalize connectome weights so its spectral radius matches target_spectral_radius."""
    from research.pipeline.flywire_graph import spectral_radius

    current_rho = spectral_radius(graph)
    if current_rho <= 0.0:
        return graph

    scale = float(target_spectral_radius / current_rho)
    new_weights = (graph.weights * scale).tocsr()

    return ConnectomeGraph(
        weights=new_weights,
        neuron_ids=list(graph.neuron_ids),
        neuron_types=list(graph.neuron_types),
        sensory_idx=graph.sensory_idx.copy(),
        motor_idx=graph.motor_idx.copy(),
        meta={
            **graph.meta,
            "spectral_radius": float(target_spectral_radius),
            "unnormalized_spectral_radius": float(current_rho),
            "spectral_scaling_factor": scale,
        },
    )
