"""Determinism tests for FlySimulator.

Verifies:
1. noise_std = 0.0 -> completely deterministic, bit-for-bit identical across runs.
2. noise_std = 0.0 -> seed does not alter sensory mapping or noiseless dynamics.
3. noise_std > 0.0 -> same seed yields identical trajectory; different seeds yield different trajectories.
4. Decoded actions are identical whenever scores are bit-for-bit identical.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.connectome.synthetic import make_synthetic_connectome
from research.pipeline.fly_simulator import FlySimulator
from research.pipeline.action_decoder import decode


@pytest.fixture(scope="module")
def synthetic_graph():
    """Create a standard synthetic connectome for testing."""
    return make_synthetic_connectome(
        n_neurons=1000,
        avg_out_degree=8.0,
        long_range_fraction=0.05,
        frac_inhibitory=0.2,
        n_sensory=48,
        n_motor=12,
        seed=0,
    )


@pytest.fixture(scope="module")
def sample_images():
    """Create a batch of 10 diverse test images."""
    rng = np.random.default_rng(12345)
    imgs = []
    # 2 black
    imgs.append(np.zeros((64, 64, 3), dtype=np.uint8))
    imgs.append(np.zeros((64, 64, 3), dtype=np.uint8))
    # 2 white
    imgs.append(np.full((64, 64, 3), 255, dtype=np.uint8))
    imgs.append(np.full((64, 64, 3), 255, dtype=np.uint8))
    # 6 random
    for _ in range(6):
        imgs.append(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8))
    return np.stack(imgs, axis=0)


def test_determinism_noise_zero_repeated_run(synthetic_graph, sample_images):
    """Calling run() repeatedly on the same instance with noise_std=0 gives bit-identical results."""
    sim = FlySimulator(synthetic_graph, seed=42, noise_std=0.0)
    out1 = sim.run(sample_images)
    out2 = sim.run(sample_images)

    assert np.array_equal(out1["buy_score"], out2["buy_score"])
    assert np.array_equal(out1["sell_score"], out2["sell_score"])
    assert np.all(np.isfinite(out1["buy_score"]))
    assert np.all(np.isfinite(out1["sell_score"]))


def test_determinism_noise_zero_different_instances_same_seed(synthetic_graph, sample_images):
    """Two separate FlySimulator instances with the same seed yield bit-identical results."""
    sim1 = FlySimulator(synthetic_graph, seed=100, noise_std=0.0)
    sim2 = FlySimulator(synthetic_graph, seed=100, noise_std=0.0)

    out1 = sim1.run(sample_images)
    out2 = sim2.run(sample_images)

    assert np.array_equal(out1["buy_score"], out2["buy_score"])
    assert np.array_equal(out1["sell_score"], out2["sell_score"])


def test_determinism_noise_zero_different_seeds(synthetic_graph, sample_images):
    """When noise_std=0.0, seed only affects noise, so different seeds produce bit-identical results."""
    sim_seed_a = FlySimulator(synthetic_graph, seed=1, noise_std=0.0)
    sim_seed_b = FlySimulator(synthetic_graph, seed=9999, noise_std=0.0)

    out_a = sim_seed_a.run(sample_images)
    out_b = sim_seed_b.run(sample_images)

    assert np.array_equal(out_a["buy_score"], out_b["buy_score"])
    assert np.array_equal(out_a["sell_score"], out_b["sell_score"])


def test_determinism_noise_positive_same_seed(synthetic_graph, sample_images):
    """When noise_std > 0, identical seeds produce bit-identical noise and output."""
    sim1 = FlySimulator(synthetic_graph, seed=777, noise_std=0.05)
    sim2 = FlySimulator(synthetic_graph, seed=777, noise_std=0.05)

    out1 = sim1.run(sample_images)
    out2 = sim2.run(sample_images)

    assert np.array_equal(out1["buy_score"], out2["buy_score"])
    assert np.array_equal(out1["sell_score"], out2["sell_score"])


def test_determinism_noise_positive_different_seeds_differ(synthetic_graph, sample_images):
    """When noise_std > 0, different seeds produce different outputs."""
    sim1 = FlySimulator(synthetic_graph, seed=111, noise_std=0.05)
    sim2 = FlySimulator(synthetic_graph, seed=222, noise_std=0.05)

    out1 = sim1.run(sample_images)
    out2 = sim2.run(sample_images)

    assert not np.array_equal(out1["buy_score"], out2["buy_score"])
    assert not np.array_equal(out1["sell_score"], out2["sell_score"])


def test_decoded_action_determinism(synthetic_graph, sample_images):
    """Decoded actions from deterministic runs are also bit-identical."""
    sim1 = FlySimulator(synthetic_graph, seed=0, noise_std=0.0)
    sim2 = FlySimulator(synthetic_graph, seed=0, noise_std=0.0)

    out1 = sim1.run(sample_images)
    out2 = sim2.run(sample_images)

    act1 = decode(out1["buy_score"], out1["sell_score"])
    act2 = decode(out2["buy_score"], out2["sell_score"])

    assert np.array_equal(act1, act2)
