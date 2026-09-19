"""Tests for FlyWire v783 connectome graph loading, caching, and inference."""
from __future__ import annotations

import time
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from src.connectome.schema import ConnectomeGraph
from research.pipeline.flywire_graph import (
    load_flywire_graph,
    spectral_radius,
    CACHE_PATH,
)
from research.pipeline.retina import retina_encoder
from research.pipeline.fly_simulator import FlySimulator


def test_spectral_radius_synthetic():
    """Verify spectral_radius function on a known simple matrix."""
    # 2x2 rotation matrix with eigenvalues +- i -> rho = 1.0
    # Or simple scaled diagonal
    data = np.array([3.0, 4.0, 5.0], dtype=np.float32)
    indices = np.array([0, 1, 2], dtype=np.int32)
    indptr = np.array([0, 1, 2, 3], dtype=np.int32)
    W = sp.csr_matrix((data, indices, indptr), shape=(3, 3))

    graph = ConnectomeGraph(
        weights=W,
        neuron_ids=["0", "1", "2"],
        neuron_types=["a", "b", "c"],
        sensory_idx=np.array([0]),
        motor_idx=np.array([1, 2]),
        meta={},
    )

    rho = spectral_radius(graph)
    assert np.isclose(rho, 5.0, atol=1e-4)
    assert graph.meta["spectral_radius"] == rho


def test_load_flywire_graph_properties():
    """Integration test loading real FlyWire connectome graph and verifying properties."""
    graph = load_flywire_graph(use_cache=True)

    # 1. Neuron and synapse counts
    assert graph.n_neurons == 138639
    assert graph.n_synapses == 15091983

    # 2. Weights dtype and sign distribution
    assert graph.weights.dtype == np.float32
    data = graph.weights.data
    pos_edges = np.sum(data > 0)
    neg_edges = np.sum(data < 0)
    assert pos_edges + neg_edges == graph.n_synapses
    pos_ratio = pos_edges / graph.n_synapses
    neg_ratio = neg_edges / graph.n_synapses
    # Excitatory ~60%, Inhibitory ~40%
    assert 0.55 < pos_ratio < 0.65
    assert 0.35 < neg_ratio < 0.45

    # 3. Sensory neurons (R1-6, R7, R8)
    # Total in connectome: 10,582 (R1-6: 7932, R7: 1336, R8: 1314)
    assert len(graph.sensory_idx) == 10582
    # Check that root_id order is strictly ascending
    sensory_root_ids = [int(graph.neuron_ids[idx]) for idx in graph.sensory_idx]
    assert np.all(np.diff(sensory_root_ids) > 0)

    # 4. BUY / SELL motor neuron count
    # Connectome graph has 645 left DN (BUY) and 646 right DN (SELL)
    # (Full annotation catalog has 646/649, but 4 isolated DNs are not in the connectome)
    buy_idx = graph.meta["buy_idx"]
    sell_idx = graph.meta["sell_idx"]
    assert len(buy_idx) in (645, 646)
    assert len(sell_idx) in (646, 649)
    assert set(buy_idx).isdisjoint(set(sell_idx))

    # 5. Metadata verification
    assert "u" in graph.meta and "v" in graph.meta
    u = graph.meta["u"]
    v = graph.meta["v"]
    assert len(u) == len(graph.sensory_idx)
    assert len(v) == len(graph.sensory_idx)
    assert not np.isnan(u).any() and not np.isnan(v).any()
    assert np.all(u >= 0.0) and np.all(u <= 1.0)
    assert np.all(v >= 0.0) and np.all(v <= 1.0)

    # 6. Spectral radius
    rho = spectral_radius(graph)
    assert 2100.0 < rho < 2200.0


def test_flywire_inference_real_train_images():
    """Run inference on real FlyWire graph with <=10 real Train images."""
    data_dir = Path("research/data")
    images_file = data_dir / "images.npy"
    samples_file = data_dir / "samples.parquet"
    mean_img_file = Path("research/outputs/v3/train_mean_image.npy")

    if not (images_file.exists() and samples_file.exists()):
        pytest.skip("research/data/images.npy or samples.parquet not found")

    graph = load_flywire_graph(use_cache=True)

    samples = pd.read_parquet(samples_file)
    train_samples = samples[samples["split"] == "train"]
    img_indices = train_samples["image_idx"].values[:5]  # Test with 5 real train images

    images = np.load(images_file, mmap_mode="r")[img_indices]
    assert images.shape == (5, 64, 64, 3)

    if mean_img_file.exists():
        mean_img = np.load(mean_img_file)
    else:
        mean_img = images.mean(axis=0).astype(np.float32)

    encoder = retina_encoder(graph, mean_img)
    rho = spectral_radius(graph)
    gain = 0.95 / rho

    sim = FlySimulator(
        graph=graph,
        steps=32,
        gain=gain,
        leak=0.5,
        noise_std=0.0,
        encoder=encoder,
        direct_currents=True,
    )

    t0 = time.time()
    res = sim.run(images)
    elapsed = time.time() - t0
    sec_per_img = elapsed / len(images)
    print(f"\nReal graph inference: {len(images)} images in {elapsed:.3f}s ({sec_per_img:.3f}s/img)")

    buy = res["buy_score"]
    sell = res["sell_score"]

    assert len(buy) == 5
    assert len(sell) == 5
    assert not np.isnan(buy).any()
    assert not np.isnan(sell).any()
    assert not np.isinf(buy).any()
    assert not np.isinf(sell).any()

    # Output should vary with inputs
    assert np.std(buy) > 1e-6
    assert np.std(sell) > 1e-6
