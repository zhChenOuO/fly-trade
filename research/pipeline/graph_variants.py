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
    edge_set = set(zip(src, dst))
    n_attempts = max(10, int(E * n_swap_multiplier))

    idx1 = rng.integers(0, E, size=n_attempts)
    idx2 = rng.integers(0, E, size=n_attempts)

    for i in range(n_attempts):
        e1, e2 = idx1[i], idx2[i]
        if e1 == e2:
            continue
        s1, d1 = src[e1], dst[e1]
        s2, d2 = src[e2], dst[e2]

        # Cannot swap if same source or same destination
        if s1 == s2 or d1 == d2:
            continue
        # Avoid creating self-loops
        if s1 == d2 or s2 == d1:
            continue
        # Avoid creating duplicate edges
        if (s1, d2) in edge_set or (s2, d1) in edge_set:
            continue

        # Execute valid swap
        edge_set.remove((s1, d1))
        edge_set.remove((s2, d2))
        edge_set.add((s1, d2))
        edge_set.add((s2, d1))

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
        },
    )
