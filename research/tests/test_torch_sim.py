from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.sparse.linalg import eigs

torch = pytest.importorskip("torch")

from research.pipeline.encoding_v2 import encoder
from research.pipeline.fly_simulator import FlySimulator
from research.pipeline.flywire_graph import load_flywire_graph
from research.pipeline.render_market import render
from research.pipeline.retina import retina_encoder
from research.pipeline.synthetic_prices import generate_phase0_synthetic_suite
from src.connectome.synthetic import make_synthetic_connectome

RESEARCH_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = RESEARCH_DIR.parent
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU unavailable")


def _margin(result):
    return result["buy_score"] - result["sell_score"]


def _synthetic_images(n_samples=1000):
    train = generate_phase0_synthetic_suite(split="train", n_samples=n_samples)
    test = generate_phase0_synthetic_suite(split="test", n_samples=n_samples)
    train_images = np.stack([render(window) for window in train["pc1_trend"][0]])
    test_images = np.stack([render(window) for window in test["pc1_trend"][0]])
    train_mean = train_images.astype(np.float64).mean(axis=0)
    return train_images, test_images, train_mean


def test_torch_matches_scipy_on_synthetic_configuration():
    train_images, test_images, train_mean = _synthetic_images()
    graph = make_synthetic_connectome(
        n_neurons=4000,
        avg_out_degree=8.0,
        long_range_fraction=0.05,
        frac_inhibitory=0.2,
        n_sensory=48,
        n_motor=12,
        seed=0,
    )
    rho = float(np.abs(eigs(graph.weights, k=1, which="LM", return_eigenvectors=False)[0]))
    gain = 0.95 / rho
    baseline = {"buy_mean": 0.0, "buy_std": 1.0, "sell_mean": 0.0, "sell_std": 1.0}
    encode = encoder("mean_sub", train_mean)
    scipy_sim = FlySimulator(
        graph, seed=17, steps=32, gain=gain, leak=0.5, encoder=encode, baseline=baseline
    )
    torch_sim = FlySimulator(
        graph, seed=17, steps=32, gain=gain, leak=0.5, encoder=encode, baseline=baseline,
        backend="torch",
    )

    scipy_result = scipy_sim.run(test_images)
    torch_result = torch_sim.run(test_images)
    correlation = float(np.corrcoef(_margin(scipy_result), _margin(torch_result))[0, 1])
    action_agreement = float(
        np.mean((_margin(scipy_result) > 0) == (_margin(torch_result) > 0))
    )
    print(
        f"synthetic equivalence: margin_pearson={correlation:.8f}, "
        f"action_agreement={action_agreement:.6f}"
    )
    assert correlation >= 0.9999
    assert action_agreement >= 0.999

    noisy_scipy = FlySimulator(
        graph, seed=91, steps=32, gain=gain, leak=0.5, noise_std=0.05,
        encoder=encode, baseline=baseline,
    ).run(test_images[:32])
    noisy_torch = FlySimulator(
        graph, seed=91, steps=32, gain=gain, leak=0.5, noise_std=0.05,
        encoder=encode, baseline=baseline, backend="torch",
    ).run(test_images[:32])
    np.testing.assert_allclose(noisy_torch["buy_score"], noisy_scipy["buy_score"], rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(noisy_torch["sell_score"], noisy_scipy["sell_score"], rtol=1e-4, atol=1e-6)


def test_torch_matches_scipy_on_20_real_train_images_with_retina_encoder():
    data_dir = RESEARCH_DIR / "data"
    flywire_dir = data_dir / "flywire"
    required = [
        flywire_dir / "Connectivity_783.parquet",
        flywire_dir / "Completeness_783.csv",
        flywire_dir / "Supplemental_file1_neuron_annotations.tsv",
        data_dir / "samples.parquet",
        data_dir / "images.npy",
    ]
    if not all(path.exists() for path in required):
        pytest.skip("real FlyWire graph or market images are unavailable")

    graph = load_flywire_graph()
    samples = pd.read_parquet(data_dir / "samples.parquet", columns=["image_idx", "split"])
    train_indices = samples.loc[samples.split == "train", "image_idx"].to_numpy()
    mean_indices = train_indices[::10]
    images = np.load(data_dir / "images.npy", mmap_mode="r")
    train_mean = np.asarray(images[mean_indices]).astype(np.float64).mean(axis=0)
    selected = train_indices[np.linspace(0, len(train_indices) - 1, 20, dtype=np.int64)]
    train_images = np.asarray(images[selected])
    encode = retina_encoder(graph, train_mean)
    gain = 0.95 / float(graph.meta["spectral_radius"])

    scipy_result = FlySimulator(
        graph, seed=23, steps=32, gain=gain, leak=0.5, encoder=encode,
        direct_currents=True,
    ).run(train_images)
    torch_result = FlySimulator(
        graph, seed=23, steps=32, gain=gain, leak=0.5, encoder=encode,
        direct_currents=True, backend="torch",
    ).run(train_images)
    correlation = float(np.corrcoef(_margin(scipy_result), _margin(torch_result))[0, 1])
    print(f"real-train equivalence: margin_pearson={correlation:.8f}")
    assert correlation >= 0.999
