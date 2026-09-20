"""Connectome graph variants generator.

Contract (from research/SPEC.md & research.md §4 Step 4 / §6 Level 3):
- degree_preserved_scramble(graph: ConnectomeGraph, seed: int = 0) -> ConnectomeGraph
- Preserves every node's in-degree and out-degree.
- Preserves sensory_idx and motor_idx unchanged.
- Preserves edge sign distribution and weights (associating weights with source neurons).
- Scrambles connectivity via double-edge-swap, ensuring the edge set differs from original.
"""
from __future__ import annotations

import hashlib
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


def compute_graph_sha256(graph: ConnectomeGraph) -> str:
    """Compute deterministic SHA-256 hex digest of CSR indptr, indices, and data arrays."""
    csr = graph.weights.tocsr()
    if not csr.has_sorted_indices:
        csr = csr.copy()
        csr.sort_indices()
    h = hashlib.sha256()
    h.update(csr.indptr.tobytes())
    h.update(csr.indices.tobytes())
    h.update(csr.data.tobytes())
    return h.hexdigest()


def sample_uniform_edges_no_replacement(
    n_neurons: int,
    E: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Uniformly sample E distinct directed edges without self-loops without sorting truncation.

    Uses rejection sampling:
    1. Generates E uniform non-self-loop candidate pairs (src != dst).
    2. Deduplicates candidates.
    3. If collisions occurred, supplementary uniform candidate batches are generated
       and non-colliding edges are accepted in the order they appear until exactly E
       unique edges are collected.
    4. The collected pairs are randomly shuffled, completely avoiding any sorting-induced
       truncation bias on node degrees.
    """
    max_edges = n_neurons * (n_neurons - 1)
    if E > max_edges:
        raise ValueError(
            f"Requested {E} edges exceeds maximum possible non-self-loop edges {max_edges}."
        )
    if E == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)

    # Round 1: Draw E candidate pairs with src != dst
    src = rng.integers(0, n_neurons, size=E, dtype=np.int64)
    shift = rng.integers(1, n_neurons, size=E, dtype=np.int64)
    dst = (src + shift) % n_neurons
    packed = (src << 32) | dst

    # Deduplicate round 1
    u_packed = np.unique(packed)
    collected = [u_packed]
    n_collected = len(u_packed)

    # Rejection sampling to fill the missing edges without sorting truncation bias
    while n_collected < E:
        missing = E - n_collected
        batch_size = max(missing * 2, 1000)
        s_supp = rng.integers(0, n_neurons, size=batch_size, dtype=np.int64)
        shift_supp = rng.integers(1, n_neurons, size=batch_size, dtype=np.int64)
        d_supp = (s_supp + shift_supp) % n_neurons
        p_supp = (s_supp << 32) | d_supp

        # Check against existing sorted u_packed via binary search
        idx = np.searchsorted(u_packed, p_supp)
        idx = np.clip(idx, 0, len(u_packed) - 1)
        not_in_u = (u_packed[idx] != p_supp)

        valid = p_supp[not_in_u]
        if len(valid) == 0:
            continue

        # Deduplicate candidates preserving stream arrival order
        _, first_idx = np.unique(valid, return_index=True)
        valid_u = valid[np.sort(first_idx)]

        needed = valid_u[:missing]
        collected.append(needed)
        n_collected += len(needed)
        if n_collected < E:
            u_packed = np.unique(np.concatenate(collected))

    all_packed = np.concatenate(collected)
    if len(all_packed) > E:
        all_packed = all_packed[:E]

    # Shuffle to eliminate block ordering
    rng.shuffle(all_packed)

    final_src = (all_packed >> 32).astype(np.int64)
    final_dst = (all_packed & 0xFFFFFFFF).astype(np.int64)
    return final_src, final_dst


def random_endpoint_graph(
    graph: ConnectomeGraph,
    seed: int = 0,
    return_diagnostics: bool = False,
) -> ConnectomeGraph | tuple[ConnectomeGraph, dict]:
    """Generate an unbiased uniform random-endpoint graph matched to the connectome.

    Preserves:
    - Number of neurons (nodes)
    - Number of directed edges (synapses)
    - Sensory and motor neuron indices and metadata
    - Exact multiset of signed edge weights (permuted across edges)

    Randomizes:
    - Directed connection endpoints sampled uniformly without replacement from all
      N*(N-1) non-self-loop pairs, without sorting truncation bias.
    """
    t0 = time.time()
    n_neurons = graph.n_neurons
    E = graph.weights.nnz
    rng = np.random.default_rng(seed)

    # Permute edge weights to preserve the exact empirical weight multiset and sign balance
    val_shuffled = rng.permutation(graph.weights.data.copy())

    # Sample endpoints uniformly without replacement and without self-loops
    final_src, final_dst = sample_uniform_edges_no_replacement(n_neurons, E, rng)

    new_W = sp.coo_matrix(
        (val_shuffled, (final_dst, final_src)), shape=(n_neurons, n_neurons)
    ).tocsr()
    new_W.sort_indices()

    elapsed = float(time.time() - t0)
    diag = {
        "variant": "random_endpoint",
        "seed": int(seed),
        "n_neurons": int(n_neurons),
        "edge_count": int(E),
        "self_loops": int(np.count_nonzero(new_W.diagonal())),
        "is_canonical_csr": bool(new_W.has_sorted_indices and new_W.has_canonical_format),
        "weights_multiset_preserved": bool(
            np.array_equal(np.sort(graph.weights.data), np.sort(new_W.data))
        ),
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
            "variant": "random_endpoint",
            "seed": seed,
            "spectral_radius": None,
            "random_endpoint_diagnostics": diag,
        },
    )
    if return_diagnostics:
        return out_g, diag
    return out_g


def source_wise_weight_shuffle(
    graph: ConnectomeGraph,
    seed: int = 0,
    return_diagnostics: bool = False,
) -> ConnectomeGraph | tuple[ConnectomeGraph, dict]:
    """Permute weights on outgoing edges per source neuron, stratified by edge sign.

    Graph convention: W[post, pre], so source = pre = matrix column.
    For each source neuron j (column of W), its outgoing positive edge weights
    are permuted among themselves, and its outgoing negative edge weights are
    permuted among themselves.

    Preserves:
    - Exact graph topology (indices, indptr, shape identical to original).
    - Excitatory/inhibitory sign of every directed edge (Dale's principle).
    - Exact positive out-strength of every source neuron.
    - Exact negative out-strength of every source neuron.
    - Global empirical synaptic weight multiset.
    """
    t0 = time.time()
    rng = np.random.default_rng(seed)
    csc = graph.weights.tocsc()
    data = csc.data.copy()
    indptr = csc.indptr
    n_neurons = graph.n_neurons

    for j in range(n_neurons):
        start, end = indptr[j], indptr[j + 1]
        if end - start <= 1:
            continue
        col_data = data[start:end]
        pos_idx = np.where(col_data > 0)[0]
        if len(pos_idx) > 1:
            col_data[pos_idx] = rng.permutation(col_data[pos_idx])
        neg_idx = np.where(col_data < 0)[0]
        if len(neg_idx) > 1:
            col_data[neg_idx] = rng.permutation(col_data[neg_idx])
        data[start:end] = col_data

    new_W = sp.csc_matrix((data, csc.indices.copy(), indptr.copy()), shape=csc.shape).tocsr()
    new_W.sort_indices()

    elapsed = float(time.time() - t0)
    diag = {
        "variant": "weight_shuffle_source",
        "seed": int(seed),
        "topology_preserved": True,
        "edge_signs_preserved": True,
        "weights_identical_multiset": bool(
            np.array_equal(np.sort(graph.weights.data), np.sort(new_W.data))
        ),
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
            "variant": "weight_shuffle_source",
            "seed": seed,
            "spectral_radius": None,
            "source_wise_shuffle_diagnostics": diag,
        },
    )
    if return_diagnostics:
        return out_g, diag
    return out_g


def graph_integrity_report(
    graph: ConnectomeGraph,
    base: ConnectomeGraph | None = None,
) -> dict:
    """Compute comprehensive graph integrity and provenance statistics.

    Parameters
    ----------
    graph : ConnectomeGraph
        Graph to evaluate.
    base : ConnectomeGraph | None
        Optional reference (base) graph to compare against.

    Returns
    -------
    dict
        Integrity statistics and comparison results.
    """
    csr = graph.weights.tocsr()
    if not csr.has_sorted_indices:
        csr = csr.copy()
        csr.sort_indices()

    N = int(graph.n_neurons)
    nnz = int(csr.nnz)
    diag_vals = csr.diagonal()
    self_loops = int(np.count_nonzero(diag_vals))
    is_canonical = bool(csr.has_sorted_indices and csr.has_canonical_format)

    data = csr.data
    pos_mask = data > 0
    neg_mask = data < 0
    pos_edge_count = int(np.sum(pos_mask))
    neg_edge_count = int(np.sum(neg_mask))
    pos_weight_sum = float(np.sum(data[pos_mask])) if pos_edge_count > 0 else 0.0
    neg_weight_sum = float(np.sum(data[neg_mask])) if neg_edge_count > 0 else 0.0

    rho = None
    if "spectral_radius" in graph.meta and graph.meta["spectral_radius"] is not None:
        rho = float(graph.meta["spectral_radius"])

    g_sha = compute_graph_sha256(graph)

    report = {
        "n_neurons": N,
        "nnz": nnz,
        "self_loops": self_loops,
        "is_canonical_csr": is_canonical,
        "pos_edge_count": pos_edge_count,
        "neg_edge_count": neg_edge_count,
        "pos_weight_sum": pos_weight_sum,
        "neg_weight_sum": neg_weight_sum,
        "spectral_radius": rho,
        "graph_sha256": g_sha,
    }

    if base is not None:
        base_csr = base.weights.tocsr()
        if not base_csr.has_sorted_indices:
            base_csr = base_csr.copy()
            base_csr.sort_indices()

        report["base_sha256"] = compute_graph_sha256(base)

        # 1. In-degree and Out-degree per node
        # Destination = row (in-degree), Source = column (out-degree)
        in_deg = np.diff(csr.indptr)
        base_in_deg = np.diff(base_csr.indptr)
        in_deg_ok = bool(np.array_equal(in_deg, base_in_deg))

        out_deg = np.bincount(csr.indices, minlength=N)
        base_out_deg = np.bincount(base_csr.indices, minlength=N)
        out_deg_ok = bool(np.array_equal(out_deg, base_out_deg))

        report["in_degree_equal"] = in_deg_ok
        report["out_degree_equal"] = out_deg_ok
        report["degree_equal"] = bool(in_deg_ok and out_deg_ok)

        # 2. Source-wise positive and negative out-strength
        # W[post, pre] -> column j is source j
        W_pos = sp.csr_matrix((np.maximum(data, 0.0), csr.indices, csr.indptr), shape=csr.shape)
        base_pos_data = np.maximum(base_csr.data, 0.0)
        base_W_pos = sp.csr_matrix((base_pos_data, base_csr.indices, base_csr.indptr), shape=base_csr.shape)

        pos_out = np.asarray(W_pos.sum(axis=0), dtype=np.float64).ravel()
        base_pos_out = np.asarray(base_W_pos.sum(axis=0), dtype=np.float64).ravel()
        delta_pos = np.abs(pos_out - base_pos_out)
        max_delta_pos = float(np.max(delta_pos)) if len(delta_pos) > 0 else 0.0
        pos_out_ok = bool(np.allclose(pos_out, base_pos_out, atol=1e-5, rtol=1e-5))

        W_neg = sp.csr_matrix((np.minimum(data, 0.0), csr.indices, csr.indptr), shape=csr.shape)
        base_neg_data = np.minimum(base_csr.data, 0.0)
        base_W_neg = sp.csr_matrix((base_neg_data, base_csr.indices, base_csr.indptr), shape=base_csr.shape)

        neg_out = np.asarray(W_neg.sum(axis=0), dtype=np.float64).ravel()
        base_neg_out = np.asarray(base_W_neg.sum(axis=0), dtype=np.float64).ravel()
        delta_neg = np.abs(neg_out - base_neg_out)
        max_delta_neg = float(np.max(delta_neg)) if len(delta_neg) > 0 else 0.0
        neg_out_ok = bool(np.allclose(neg_out, base_neg_out, atol=1e-5, rtol=1e-5))

        report["source_pos_out_strength_equal"] = pos_out_ok
        report["source_neg_out_strength_equal"] = neg_out_ok
        report["source_out_strength_equal"] = bool(pos_out_ok and neg_out_ok)
        report["max_diff_pos_out_strength"] = max_delta_pos
        report["max_diff_neg_out_strength"] = max_delta_neg
        report["max_diff_out_strength"] = max(max_delta_pos, max_delta_neg)

        # 3. Edge overlap ratio
        if np.array_equal(csr.indptr, base_csr.indptr) and np.array_equal(csr.indices, base_csr.indices):
            overlap_count = nnz
            overlap_ratio = 1.0
        else:
            dst = np.repeat(np.arange(N, dtype=np.int64), np.diff(csr.indptr))
            src = csr.indices.astype(np.int64)
            packed = (src << 32) | dst

            base_dst = np.repeat(np.arange(N, dtype=np.int64), np.diff(base_csr.indptr))
            base_src = base_csr.indices.astype(np.int64)
            base_packed = (base_src << 32) | base_dst

            common = np.intersect1d(packed, base_packed, assume_unique=True)
            overlap_count = int(len(common))
            overlap_ratio = float(overlap_count / max(nnz, 1))

        report["edge_overlap_count"] = overlap_count
        report["edge_overlap_ratio"] = overlap_ratio

        # 4. Weights multiset equality
        s_data = np.sort(data)
        s_base_data = np.sort(base_csr.data)
        multiset_equal = bool(
            len(s_data) == len(s_base_data) and np.allclose(s_data, s_base_data, atol=1e-6, rtol=1e-5)
        )
        report["weights_multiset_equal"] = multiset_equal

    return report

