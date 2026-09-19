"""Unit tests for retina PCA coordinates and bilinear encoder."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.connectome.schema import ConnectomeGraph
from research.pipeline.retina import (
    compute_retina_coordinates,
    bilinear_sample_retina,
    retina_encoder,
)


@pytest.fixture
def mock_photoreceptors():
    """Create a small mock dataset of photoreceptors for left and right eyes."""
    rng = np.random.default_rng(42)
    rows = []

    # Left eye: 20 R1-6, 5 R7, 5 R8
    for i in range(20):
        # 3D points on a curved surface
        x = rng.uniform(10.0, 50.0)
        y = rng.uniform(20.0, 60.0)
        z = np.sin(x / 10.0) + np.cos(y / 10.0)
        rows.append({"root_id": 1000 + i, "cell_type": "R1-6", "side": "left", "pos_x": x, "pos_y": y, "pos_z": z})

    for i in range(5):
        # Place near first 5 R1-6
        base = rows[i]
        rows.append({"root_id": 2000 + i, "cell_type": "R7", "side": "left", "pos_x": base["pos_x"] + 0.01, "pos_y": base["pos_y"] + 0.01, "pos_z": base["pos_z"] + 0.01})
        rows.append({"root_id": 3000 + i, "cell_type": "R8", "side": "left", "pos_x": base["pos_x"] + 0.02, "pos_y": base["pos_y"] + 0.02, "pos_z": base["pos_z"] + 0.02})

    # Right eye: 20 R1-6, 5 R7, 5 R8
    for i in range(20):
        x = rng.uniform(110.0, 150.0)
        y = rng.uniform(20.0, 60.0)
        z = np.sin(x / 10.0) + np.cos(y / 10.0)
        rows.append({"root_id": 4000 + i, "cell_type": "R1-6", "side": "right", "pos_x": x, "pos_y": y, "pos_z": z})

    for i in range(5):
        base = rows[30 + i]
        rows.append({"root_id": 5000 + i, "cell_type": "R7", "side": "right", "pos_x": base["pos_x"] + 0.01, "pos_y": base["pos_y"] + 0.01, "pos_z": base["pos_z"] + 0.01})
        rows.append({"root_id": 6000 + i, "cell_type": "R8", "side": "right", "pos_x": base["pos_x"] + 0.02, "pos_y": base["pos_y"] + 0.02, "pos_z": base["pos_z"] + 0.02})

    return pd.DataFrame(rows)


def test_compute_retina_coordinates(mock_photoreceptors):
    """Test PCA coordinate generation, sign convention, and nearest-neighbor matching."""
    coords = compute_retina_coordinates(mock_photoreceptors)

    u = coords["u"]
    v = coords["v"]
    uv = coords["uv"]

    assert len(u) == len(mock_photoreceptors)
    assert len(v) == len(mock_photoreceptors)
    assert uv.shape == (len(mock_photoreceptors), 2)

    # 1. Bounds check: all (u, v) must be in [0, 1]
    assert np.all(u >= 0.0) and np.all(u <= 1.0)
    assert np.all(v >= 0.0) and np.all(v <= 1.0)
    assert not np.isnan(u).any() and not np.isnan(v).any()

    # 2. Nearest-neighbor matching for R7 and R8
    # For the left eye, R7 index 20 was placed right next to R1-6 index 0
    # Their (u, v) must match identically
    assert np.isclose(u[20], u[0], atol=1e-5)
    assert np.isclose(v[20], v[0], atol=1e-5)
    # R8 index 21 was placed right next to R1-6 index 0
    assert np.isclose(u[21], u[0], atol=1e-5)
    assert np.isclose(v[21], v[0], atol=1e-5)


def test_bilinear_channel_selectivity():
    """Verify R1-6 samples luminance, R7 samples R channel, R8 samples G channel."""
    B = 2
    # Create image with specific pure channels
    # Image 0: Pure Red (255, 0, 0)
    # Image 1: Pure Green (0, 255, 0)
    imgs = np.zeros((B, 64, 64, 3), dtype=np.float32)
    imgs[0, :, :, 0] = 1.0  # R=1
    imgs[1, :, :, 1] = 1.0  # G=1

    u = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    v = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    types = ["R1-6", "R7", "R8"]

    currents = bilinear_sample_retina(imgs, u, v, types)
    assert currents.shape == (2, 3)

    # Image 0 (Pure Red):
    # R1-6 gets luminance = mean(1, 0, 0) = 1/3
    assert np.isclose(currents[0, 0], 1.0 / 3.0, atol=1e-4)
    # R7 gets R channel = 1.0
    assert np.isclose(currents[0, 1], 1.0, atol=1e-4)
    # R8 gets G channel = 0.0
    assert np.isclose(currents[0, 2], 0.0, atol=1e-4)

    # Image 1 (Pure Green):
    # R1-6 gets luminance = mean(0, 1, 0) = 1/3
    assert np.isclose(currents[1, 0], 1.0 / 3.0, atol=1e-4)
    # R7 gets R channel = 0.0
    assert np.isclose(currents[1, 1], 0.0, atol=1e-4)
    # R8 gets G channel = 1.0
    assert np.isclose(currents[1, 2], 1.0, atol=1e-4)


def test_retina_encoder_mean_subtraction():
    """Verify retina_encoder applies (x - train_mean) / 128.0 and handles shapes."""
    mean_img = np.full((64, 64, 3), 128.0, dtype=np.float32)

    # Create dummy graph with meta
    u = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    v = np.array([0.2, 0.5, 0.8], dtype=np.float32)
    pr_types = ["R1-6", "R7", "R8"]

    import scipy.sparse as sp

    graph = ConnectomeGraph(
        weights=sp.csr_matrix((10, 10), dtype=np.float32),
        neuron_ids=[str(i) for i in range(10)],
        neuron_types=["unknown"] * 10,
        sensory_idx=np.array([0, 1, 2]),
        motor_idx=np.array([8, 9]),
        meta={"u": u, "v": v, "photoreceptor_type": pr_types},
    )

    encode = retina_encoder(graph, mean_img, input_scale=1.0)

    # Test 1: Image exactly equal to mean_img should produce all zeros
    exact_mean = np.full((64, 64, 3), 128, dtype=np.uint8)
    curr_zero = encode(exact_mean)
    assert curr_zero.shape == (1, 3)
    assert np.allclose(curr_zero, 0.0, atol=1e-5)

    # Test 2: Batch of images
    batch = np.zeros((4, 64, 64, 3), dtype=np.uint8)
    curr_batch = encode(batch)
    assert curr_batch.shape == (4, 3)
    assert not np.isnan(curr_batch).any()
