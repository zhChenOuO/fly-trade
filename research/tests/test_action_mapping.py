"""Action mapping, decoder, and graph variants tests.

Verifies:
1. 64x64x3 image -> sensory neuron mapping is fixed, non-learnable, and independent of simulator seed.
2. BUY and SELL motor neuron sets are disjoint subsets of graph.motor_idx.
3. action_decoder.decode:
   - buy > sell -> 1 (BUY)
   - buy < sell -> 0 (SELL)
   - buy == sell (tie) -> 0 (SELL)
   - Handles vectors, scalars, empty arrays.
4. degree_preserved_scramble:
   - Preserves sensory_idx and motor_idx.
   - Preserves exact in-degree and out-degree vectors for every node.
   - Scrambles edges so that the edge set differs from the original.
   - Conserves total number of synapses and avoids self-loops.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.connectome.synthetic import make_synthetic_connectome
from research.pipeline.fly_simulator import FlySimulator, get_fixed_projection
from research.pipeline.action_decoder import decode, ActionDecoder, ACTION_BUY, ACTION_SELL
from research.pipeline.graph_variants import degree_preserved_scramble


@pytest.fixture(scope="module")
def synthetic_graph():
    return make_synthetic_connectome(
        n_neurons=1200,
        avg_out_degree=8.0,
        long_range_fraction=0.05,
        frac_inhibitory=0.2,
        n_sensory=48,
        n_motor=12,
        seed=42,
    )


# --- Action Decoder Tests ---


def test_action_decoder_decisions():
    """Verify basic BUY, SELL, and tie-breaking decisions."""
    # Strict buy > sell -> BUY (1)
    assert np.array_equal(decode([10.0], [5.0]), [ACTION_BUY])
    # Strict buy < sell -> SELL (0)
    assert np.array_equal(decode([5.0], [10.0]), [ACTION_SELL])
    # Tie rule: buy == sell -> SELL (0)
    assert np.array_equal(decode([7.5], [7.5]), [ACTION_SELL])
    assert np.array_equal(decode([0.0], [0.0]), [ACTION_SELL])
    assert np.array_equal(decode([-1.2], [-1.2]), [ACTION_SELL])


def test_action_decoder_vectorized():
    """Verify batch/vectorized decoding and shape preservation."""
    buy = np.array([10.0, 5.0, 7.0, -2.0, 0.0])
    sell = np.array([5.0, 10.0, 7.0, -1.0, 0.0])
    expected = np.array([ACTION_BUY, ACTION_SELL, ACTION_SELL, ACTION_SELL, ACTION_SELL])

    actions = decode(buy, sell)
    assert np.array_equal(actions, expected)
    assert actions.dtype == np.int64


def test_action_decoder_scalar_and_empty():
    """Verify scalar inputs return 1D array and empty inputs return empty array."""
    # Scalar float
    act_scalar = decode(3.0, 2.0)
    assert isinstance(act_scalar, np.ndarray)
    assert act_scalar.shape == (1,)
    assert act_scalar[0] == ACTION_BUY

    # Empty arrays
    act_empty = decode(np.array([]), np.array([]))
    assert isinstance(act_empty, np.ndarray)
    assert len(act_empty) == 0


def test_action_decoder_shape_mismatch():
    """Verify ValueError is raised on mismatched input shapes."""
    with pytest.raises(ValueError):
        decode([1.0, 2.0], [1.0])


def test_action_decoder_class_wrapper():
    """Verify ActionDecoder class behaves identically."""
    ad = ActionDecoder(tie_rule="SELL")
    assert np.array_equal(ad.decode([1.0, 0.0], [0.0, 1.0]), [ACTION_BUY, ACTION_SELL])

    with pytest.raises(ValueError):
        ActionDecoder(tie_rule="BUY")


# --- Sensory and Motor Mapping Tests ---


def test_motor_neuron_disjoint_subsets(synthetic_graph):
    """BUY and SELL motor neurons must be two disjoint non-empty subsets of graph.motor_idx."""
    sim = FlySimulator(synthetic_graph, seed=0)

    buy_set = set(sim.buy_motor_idx)
    sell_set = set(sim.sell_motor_idx)
    all_motor = set(synthetic_graph.motor_idx)

    assert len(buy_set) > 0
    assert len(sell_set) > 0
    assert buy_set.isdisjoint(sell_set)
    assert buy_set.union(sell_set) == all_motor


def test_sensory_projection_is_fixed_and_independent_of_seed(synthetic_graph):
    """Verify that image-to-sensory projection is completely independent of FlySimulator seed."""
    sim1 = FlySimulator(synthetic_graph, seed=1)
    sim2 = FlySimulator(synthetic_graph, seed=999)

    # Directly check underlying projection matrix
    n_features = 64 * 64 * 3
    n_sensory = len(synthetic_graph.sensory_idx)
    w1 = get_fixed_projection(n_features, n_sensory)
    w2 = get_fixed_projection(n_features, n_sensory)

    assert np.array_equal(w1, w2)

    # In deterministic mode (noise_std=0), both simulators give identical sensory processing
    test_img = np.ones((1, 64, 64, 3), dtype=np.uint8) * 100
    res1 = sim1.run(test_img)
    res2 = sim2.run(test_img)

    assert np.array_equal(res1["buy_score"], res2["buy_score"])
    assert np.array_equal(res1["sell_score"], res2["sell_score"])


# --- Degree-Preserved Scramble Tests ---


def test_degree_preserved_scramble(synthetic_graph):
    """Verify that degree_preserved_scramble preserves degree sequences while changing edges."""
    scrambled = degree_preserved_scramble(synthetic_graph, seed=42)

    # 1. Sensory and motor indices must be preserved identically
    assert np.array_equal(synthetic_graph.sensory_idx, scrambled.sensory_idx)
    assert np.array_equal(synthetic_graph.motor_idx, scrambled.motor_idx)
    assert len(synthetic_graph.neuron_ids) == len(scrambled.neuron_ids)
    assert synthetic_graph.neuron_types == scrambled.neuron_types

    # 2. Number of neurons and synapses must match
    assert synthetic_graph.n_neurons == scrambled.n_neurons
    assert synthetic_graph.n_synapses == scrambled.n_synapses

    # 3. Exact in-degree and out-degree vectors must be identical
    orig_coo = synthetic_graph.weights.tocoo()
    scram_coo = scrambled.weights.tocoo()

    orig_in = np.bincount(orig_coo.row, minlength=synthetic_graph.n_neurons)
    orig_out = np.bincount(orig_coo.col, minlength=synthetic_graph.n_neurons)

    scram_in = np.bincount(scram_coo.row, minlength=scrambled.n_neurons)
    scram_out = np.bincount(scram_coo.col, minlength=scrambled.n_neurons)

    assert np.array_equal(orig_in, scram_in), "In-degree vectors do not match!"
    assert np.array_equal(orig_out, scram_out), "Out-degree vectors do not match!"

    # 4. Edge sets must be strictly different
    orig_edges = set(zip(orig_coo.col, orig_coo.row))
    scram_edges = set(zip(scram_coo.col, scram_coo.row))

    assert orig_edges != scram_edges, "Edge set was not modified by scramble!"

    # 5. No self loops created
    scram_self_loops = np.sum(scram_coo.row == scram_coo.col)
    assert scram_self_loops == 0, f"Found {scram_self_loops} self loops in scrambled graph"

    # 6. Scrambling is deterministic given seed
    scrambled_again = degree_preserved_scramble(synthetic_graph, seed=42)
    assert np.array_equal(scrambled.weights.toarray(), scrambled_again.weights.toarray())

    # 7. Scrambling with different seed produces different graph
    scrambled_diff = degree_preserved_scramble(synthetic_graph, seed=43)
    assert not np.array_equal(scrambled.weights.toarray(), scrambled_diff.weights.toarray())
