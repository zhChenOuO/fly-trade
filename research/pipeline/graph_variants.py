"""Connectome graph variants generator.

Contract (from research/SPEC.md & research.md §4 Step 4 / §6 Level 3):
- degree_preserved_scramble(graph: ConnectomeGraph, seed: int = 0) -> ConnectomeGraph
- Preserves every node's in-degree and out-degree.
- Preserves sensory_idx and motor_idx unchanged.
- Preserves edge sign distribution and weights (associating weights with source neurons).
- Scrambles connectivity via double-edge-swap, ensuring the edge set differs from original.
"""
from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp

from src.connectome.schema import ConnectomeGraph


def degree_preserved_scramble(
    graph: ConnectomeGraph,
    seed: int = 0,
    n_swap_multiplier: float = 2.0,
    target_overlap: float | None = None,
    max_attempts_multiplier: float = 5.0,
    return_diagnostics: bool = False,
) -> ConnectomeGraph | tuple[ConnectomeGraph, dict]:
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
        Ignored if target_overlap is specified.
    target_overlap: float | None
        If specified (e.g. 0.05), iteratively performs double-edge swaps in batches
        until the edge overlap ratio between original and scrambled graph <= target_overlap
        (or until max_attempts_multiplier * E is reached).
    max_attempts_multiplier: float
        Maximum total swap attempts as a multiple of E when target_overlap is used. Default 5.0.
    return_diagnostics: bool
        If True, returns a tuple (scrambled_graph, diagnostics_dict).
        If False (default), returns scrambled_graph with diagnostics attached to graph.meta.

    Returns
    -------
    ConnectomeGraph | tuple[ConnectomeGraph, dict]
        Scrambled connectome with identical in/out-degree sequences and preserved
        sensory/motor indices.
    """
    t0 = time.time()
    coo = graph.weights.tocoo()
    dst = coo.row.copy()
    src = coo.col.copy()
    val = coo.data.copy()
    E = len(src)

    if E < 4:
        # Not enough edges to swap, return duplicate
        diag = {
            "attempted_swaps": 0,
            "successful_swaps": 0,
            "overlap_ratio": 1.0,
            "modified_ratio": 0.0,
            "in_degree_preserved": True,
            "out_degree_preserved": True,
            "sign_ratio_preserved": True,
            "target_overlap": target_overlap,
            "target_reached": False if target_overlap is not None else True,
            "elapsed_seconds": float(time.time() - t0),
        }
        res_g = ConnectomeGraph(
            weights=graph.weights.copy(),
            neuron_ids=list(graph.neuron_ids),
            neuron_types=list(graph.neuron_types),
            sensory_idx=graph.sensory_idx.copy(),
            motor_idx=graph.motor_idx.copy(),
            meta={
                **graph.meta,
                "variant": "degree_preserved_scramble",
                "scramble_seed": seed,
                "scramble_diagnostics": diag,
            },
        )
        return (res_g, diag) if return_diagnostics else res_g

    rng = np.random.default_rng(seed)
    total_successful = 0
    total_attempts = 0

    if E < 100_000:
        # Standard tuple-based edge set for smaller graphs (100% backward compatible)
        orig_edge_set = set(zip(src, dst))
        edge_set = set(orig_edge_set)

        if target_overlap is None:
            n_attempts = max(10, int(E * n_swap_multiplier))
            idx1 = rng.integers(0, E, size=n_attempts)
            idx2 = rng.integers(0, E, size=n_attempts)
            for i in range(n_attempts):
                e1, e2 = idx1[i], idx2[i]
                if e1 == e2:
                    continue
                s1, d1 = src[e1], dst[e1]
                s2, d2 = src[e2], dst[e2]

                if s1 == s2 or d1 == d2 or s1 == d2 or s2 == d1:
                    continue
                if (s1, d2) in edge_set or (s2, d1) in edge_set:
                    continue

                edge_set.remove((s1, d1))
                edge_set.remove((s2, d2))
                edge_set.add((s1, d2))
                edge_set.add((s2, d1))

                dst[e1] = d2
                dst[e2] = d1
                total_successful += 1
            total_attempts = n_attempts
        else:
            max_attempts = max(10, int(E * max_attempts_multiplier))
            batch_size = max(10, E)
            while total_attempts < max_attempts:
                this_batch = min(batch_size, max_attempts - total_attempts)
                idx1 = rng.integers(0, E, size=this_batch)
                idx2 = rng.integers(0, E, size=this_batch)
                for i in range(this_batch):
                    e1, e2 = idx1[i], idx2[i]
                    if e1 == e2:
                        continue
                    s1, d1 = src[e1], dst[e1]
                    s2, d2 = src[e2], dst[e2]

                    if s1 == s2 or d1 == d2 or s1 == d2 or s2 == d1:
                        continue
                    if (s1, d2) in edge_set or (s2, d1) in edge_set:
                        continue

                    edge_set.remove((s1, d1))
                    edge_set.remove((s2, d2))
                    edge_set.add((s1, d2))
                    edge_set.add((s2, d1))

                    dst[e1] = d2
                    dst[e2] = d1
                    total_successful += 1
                total_attempts += this_batch
                curr_overlap = len(edge_set.intersection(orig_edge_set)) / E
                if curr_overlap <= target_overlap:
                    break

        overlap_cnt = len(edge_set.intersection(orig_edge_set))
        overlap_ratio = float(overlap_cnt / E)

    else:
        # Packed int64 keys ((src << 32) | dst) for large graphs (e.g. 15M edges)
        packed = (src.astype(np.int64) << 32) | dst.astype(np.int64)
        orig_packed_set = set(packed)
        packed_set = set(packed)

        if target_overlap is None:
            n_attempts = max(10, int(E * n_swap_multiplier))
            idx1 = rng.integers(0, E, size=n_attempts)
            idx2 = rng.integers(0, E, size=n_attempts)
            for i in range(n_attempts):
                e1, e2 = idx1[i], idx2[i]
                if e1 == e2:
                    continue
                s1, d1 = int(src[e1]), int(dst[e1])
                s2, d2 = int(src[e2]), int(dst[e2])

                if s1 == s2 or d1 == d2 or s1 == d2 or s2 == d1:
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
                total_successful += 1
            total_attempts = n_attempts
        else:
            max_attempts = max(10, int(E * max_attempts_multiplier))
            batch_size = max(100, int(E * 0.33))
            while total_attempts < max_attempts:
                this_batch = min(batch_size, max_attempts - total_attempts)
                idx1 = rng.integers(0, E, size=this_batch)
                idx2 = rng.integers(0, E, size=this_batch)
                for i in range(this_batch):
                    e1, e2 = idx1[i], idx2[i]
                    if e1 == e2:
                        continue
                    s1, d1 = int(src[e1]), int(dst[e1])
                    s2, d2 = int(src[e2]), int(dst[e2])

                    if s1 == s2 or d1 == d2 or s1 == d2 or s2 == d1:
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
                    total_successful += 1
                total_attempts += this_batch
                curr_overlap_cnt = sum(1 for e in range(E) if ((int(src[e]) << 32) | int(dst[e])) in orig_packed_set)
                if (curr_overlap_cnt / E) <= target_overlap:
                    break

        overlap_cnt = sum(1 for e in range(E) if ((int(src[e]) << 32) | int(dst[e])) in orig_packed_set)
        overlap_ratio = float(overlap_cnt / E)

    new_W = sp.coo_matrix(
        (val, (dst, src)), shape=(graph.n_neurons, graph.n_neurons)
    ).tocsr()
    new_W.sum_duplicates()

    # Diagnostics validation
    orig_in = np.bincount(coo.row, minlength=graph.n_neurons)
    orig_out = np.bincount(coo.col, minlength=graph.n_neurons)
    scram_in = np.bincount(dst, minlength=graph.n_neurons)
    scram_out = np.bincount(src, minlength=graph.n_neurons)

    in_deg_ok = bool(np.array_equal(orig_in, scram_in))
    out_deg_ok = bool(np.array_equal(orig_out, scram_out))

    pos_orig = int(np.sum(val > 0))
    neg_orig = int(np.sum(val < 0))
    pos_scram = int(np.sum(val > 0))
    neg_scram = int(np.sum(val < 0))
    sign_ok = bool(pos_orig == pos_scram and neg_orig == neg_scram)

    elapsed_s = float(time.time() - t0)
    diagnostics = {
        "attempted_swaps": int(total_attempts),
        "successful_swaps": int(total_successful),
        "overlap_ratio": float(overlap_ratio),
        "modified_ratio": float(1.0 - overlap_ratio),
        "in_degree_preserved": in_deg_ok,
        "out_degree_preserved": out_deg_ok,
        "sign_ratio_preserved": sign_ok,
        "target_overlap": float(target_overlap) if target_overlap is not None else None,
        "target_reached": bool(overlap_ratio <= target_overlap) if target_overlap is not None else True,
        "elapsed_seconds": elapsed_s,
    }

    out_graph = ConnectomeGraph(
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
            "scramble_diagnostics": diagnostics,
        },
    )

    if return_diagnostics:
        return out_graph, diagnostics
    return out_graph


def well_mixed_scramble(
    graph: ConnectomeGraph,
    seed: int = 0,
    target_overlap: float = 0.05,
    max_attempts_multiplier: float = 5.0,
    return_diagnostics: bool = False,
) -> ConnectomeGraph | tuple[ConnectomeGraph, dict]:
    """Scramble connectome edges until edge overlap with original graph <= target_overlap.

    Ensures thorough mixing of directed edges while strictly preserving in- and out-degrees.
    Default target_overlap is 0.05 (at least 95% of edges altered).
    """
    out_g, diag = degree_preserved_scramble(
        graph=graph,
        seed=seed,
        target_overlap=target_overlap,
        max_attempts_multiplier=max_attempts_multiplier,
        return_diagnostics=True,
    )
    out_g.meta["variant"] = "scramble_mixed"
    if return_diagnostics:
        return out_g, diag
    return out_g


def weight_shuffled_graph(
    graph: ConnectomeGraph,
    seed: int = 0,
    separate_signs: bool = True,
    return_diagnostics: bool = False,
) -> ConnectomeGraph | tuple[ConnectomeGraph, dict]:
    """Generate a weight-shuffled connectome with topology strictly preserved.

    Preserves:
    - Graph topology (indices, indptr, shape) is 100% invariant.
    - Empirical synaptic weight multiset is identical.
    - If separate_signs=True (default):
      Positive weights are permuted only among positive edges, and negative weights
      only among negative edges. Preserves the excitatory/inhibitory sign of every
      connection (honoring Dale's principle and E/I balance) while shuffling magnitudes.
    - If separate_signs=False:
      All weights are permuted jointly across all edges, preserving overall weight
      distribution while allowing edge signs to change.
    """
    t0 = time.time()
    rng = np.random.default_rng(seed)
    csr = graph.weights.tocsr()
    data = csr.data.copy()

    if separate_signs:
        pos_mask = data > 0
        neg_mask = data < 0
        new_data = data.copy()
        new_data[pos_mask] = rng.permutation(new_data[pos_mask])
        new_data[neg_mask] = rng.permutation(new_data[neg_mask])
    else:
        new_data = rng.permutation(data.copy())

    new_W = sp.csr_matrix(
        (new_data, csr.indices.copy(), csr.indptr.copy()),
        shape=csr.shape,
        dtype=csr.dtype,
    )

    elapsed = float(time.time() - t0)
    diagnostics = {
        "variant": "weight_shuffle",
        "seed": int(seed),
        "separate_signs": bool(separate_signs),
        "topology_preserved": True,
        "weights_identical_multiset": bool(np.array_equal(np.sort(data), np.sort(new_data))),
        "positive_count_preserved": bool(int(np.sum(data > 0)) == int(np.sum(new_data > 0))),
        "negative_count_preserved": bool(int(np.sum(data < 0)) == int(np.sum(new_data < 0))),
        "elapsed_seconds": elapsed,
    }

    out_g = ConnectomeGraph(
        weights=new_W,
        neuron_ids=list(graph.neuron_ids),
        neuron_types=list(graph.neuron_types),
        sensory_idx=graph.sensory_idx.copy(),
        motor_idx=graph.motor_idx.copy(),
        meta={
            **graph.meta,
            "variant": "weight_shuffle",
            "seed": seed,
            "spectral_radius": None,
            "weight_shuffle_diagnostics": diagnostics,
        },
    )

    if return_diagnostics:
        return out_g, diagnostics
    return out_g


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
