"""Unit tests for Research v2 Phase 5 Frozen Connectome Linear Probe.

All tests run on small synthetic graphs or pure synthetic arrays without requiring GPU or large FlyWire data.
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from src.connectome.synthetic import make_synthetic_connectome
from research.pipeline.graph_variants import (
    random_matched_graph,
    degree_preserved_scramble,
    normalize_graph_spectral_radius,
)
from research.pipeline.flywire_graph import spectral_radius
from research.pipeline.fly_simulator import FlySimulator
from research.pipeline.train_probe_v2 import (
    train_and_eval_probe,
    paired_block_bootstrap_delta_ci,
    circular_shift_permutation_p_value,
    evaluate_thresholds_table,
)
from research.pipeline.extract_features_v2 import validate_split_and_variant


@pytest.fixture
def synthetic_graph():
    return make_synthetic_connectome(seed=42)


def test_random_matched_and_scramble_contracts(synthetic_graph) -> None:
    """Verify graph variants contracts: degree preservation, weight distributions, indices."""
    orig_coo = synthetic_graph.weights.tocoo()
    n_neurons = synthetic_graph.n_neurons
    n_synapses = synthetic_graph.n_synapses

    # 1. Random Matched Graph
    g_rand = random_matched_graph(synthetic_graph, seed=101)
    assert g_rand.n_neurons == n_neurons
    assert g_rand.n_synapses == n_synapses
    np.testing.assert_array_equal(g_rand.sensory_idx, synthetic_graph.sensory_idx)
    np.testing.assert_array_equal(g_rand.motor_idx, synthetic_graph.motor_idx)

    # Weight distribution (sorted values) must be identical
    rand_coo = g_rand.weights.tocoo()
    np.testing.assert_array_almost_equal(np.sort(orig_coo.data), np.sort(rand_coo.data))

    # Edges must be scrambled
    orig_edges = set(zip(orig_coo.col, orig_coo.row))
    rand_edges = set(zip(rand_coo.col, rand_coo.row))
    assert orig_edges != rand_edges

    # 2. Degree Preserved Scramble
    g_scram = degree_preserved_scramble(synthetic_graph, seed=202)
    assert g_scram.n_neurons == n_neurons
    assert g_scram.n_synapses == n_synapses
    np.testing.assert_array_equal(g_scram.sensory_idx, synthetic_graph.sensory_idx)
    np.testing.assert_array_equal(g_scram.motor_idx, synthetic_graph.motor_idx)

    scram_coo = g_scram.weights.tocoo()
    orig_in = np.bincount(orig_coo.row, minlength=n_neurons)
    orig_out = np.bincount(orig_coo.col, minlength=n_neurons)
    scram_in = np.bincount(scram_coo.row, minlength=n_neurons)
    scram_out = np.bincount(scram_coo.col, minlength=n_neurons)
    np.testing.assert_array_equal(orig_in, scram_in)
    np.testing.assert_array_equal(orig_out, scram_out)

    # 3. Spectral radius normalization
    target_rho = 12.5
    g_norm = normalize_graph_spectral_radius(synthetic_graph, target_spectral_radius=target_rho)
    assert np.isclose(spectral_radius(g_norm), target_rho, atol=1e-3)


def test_feature_extraction_readout_dimension_and_consistency(synthetic_graph) -> None:
    """Readout dimension must match |buy_idx| + |sell_idx| and sum/average must match scores."""
    sim = FlySimulator(
        graph=synthetic_graph,
        seed=0,
        steps=16,
        gain=0.1,
        leak=0.5,
        noise_std=0.0,
        backend="scipy",
    )

    n_readout_expected = len(sim.buy_motor_idx) + len(sim.sell_motor_idx)
    assert n_readout_expected == len(synthetic_graph.motor_idx)

    imgs = np.zeros((8, 64, 64, 3), dtype=np.uint8)
    imgs[0, 10:20, 10:20] = 200

    out = sim.run(imgs, return_readout=True)
    assert "readout" in out
    readout = out["readout"]
    assert readout.shape == (8, n_readout_expected)
    assert readout.dtype == np.float32
    assert np.isfinite(readout).all()

    # Mathematical consistency without baseline:
    # readout is time-average: mean_t(motor). So sum_neurons(readout) * steps == raw score!
    n_b = len(sim.buy_motor_idx)
    buy_readout = readout[:, :n_b]
    sell_readout = readout[:, n_b:]

    reconstructed_buy = buy_readout.sum(axis=1) * sim.steps
    reconstructed_sell = sell_readout.sum(axis=1) * sim.steps
    np.testing.assert_allclose(reconstructed_buy, out["buy_score"], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(reconstructed_sell, out["sell_score"], rtol=1e-5, atol=1e-5)

    # With baseline:
    bl = {"buy_mean": 0.05, "buy_std": 0.02, "sell_mean": 0.04, "sell_std": 0.015}
    sim_bl = FlySimulator(
        graph=synthetic_graph,
        seed=0,
        steps=16,
        gain=0.1,
        leak=0.5,
        noise_std=0.0,
        baseline=bl,
        backend="scipy",
    )
    out_bl = sim_bl.run(imgs, return_readout=True)
    r_bl = out_bl["readout"]
    b_mean_neuron = r_bl[:, :n_b].mean(axis=1)
    s_mean_neuron = r_bl[:, n_b:].mean(axis=1)

    rec_b_bl = (b_mean_neuron - bl["buy_mean"]) / bl["buy_std"]
    rec_s_bl = (s_mean_neuron - bl["sell_mean"]) / bl["sell_std"]
    np.testing.assert_allclose(rec_b_bl, out_bl["buy_score"], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(rec_s_bl, out_bl["sell_score"], rtol=1e-5, atol=1e-5)


def test_split_whitelist_governance() -> None:
    """Strict governance: only train and val are allowed."""
    validate_split_and_variant("train", "real")
    validate_split_and_variant("val", "random")
    validate_split_and_variant("val", "scramble")

    with pytest.raises(ValueError, match="strictly forbidden"):
        validate_split_and_variant("dev_test_v1", "real")

    with pytest.raises(ValueError, match="strictly forbidden"):
        validate_split_and_variant("holdout", "real")

    with pytest.raises(ValueError, match="invalid"):
        validate_split_and_variant("train", "unknown_variant")


def test_train_probe_signal_detection_and_null_rejection() -> None:
    """Probe pipeline detects planted signals in real features and rejects null signal."""
    rng = np.random.default_rng(42)
    n_train = 400
    n_val = 200
    d = 20

    # 1. Planted signal dataset: target y strongly correlated with feature linear combination
    true_weights = rng.normal(0, 1, d)
    x_tr_signal = rng.normal(0, 1, (n_train, d))
    y_tr_cont = x_tr_signal @ true_weights + rng.normal(0, 0.2, n_train)

    x_va_signal = rng.normal(0, 1, (n_val, d))
    y_va_cont = x_va_signal @ true_weights + rng.normal(0, 0.2, n_val)

    # 3-class actions for signal
    cost_th = 0.5
    y_tr_cls = np.full(n_train, "HOLD", dtype=object)
    y_tr_cls[y_tr_cont > cost_th] = "BUY"
    y_tr_cls[y_tr_cont < -cost_th] = "SELL"

    y_va_cls = np.full(n_val, "HOLD", dtype=object)
    y_va_cls[y_va_cont > cost_th] = "BUY"
    y_va_cls[y_va_cont < -cost_th] = "SELL"

    eval_signal = train_and_eval_probe(
        X_train_raw=x_tr_signal,
        y_train_cont=y_tr_cont,
        y_train_cls=y_tr_cls,
        X_val_raw=x_va_signal,
        y_val_cont=y_va_cont,
        y_val_cls=y_va_cls,
        cost_threshold=cost_th,
        cost_fee=0.0004,
        cost_slip=0.0002,
        bootstrap_samples=100,
    )
    ic_signal = eval_signal["ridge"]["continuous"]["spearman_ic"]
    assert ic_signal > 0.70, f"Planted signal IC should be high, got {ic_signal}"

    p_val_sig = circular_shift_permutation_p_value(
        eval_signal["ridge"]["pred_cont"], y_va_cont, n_permutations=200
    )
    assert p_val_sig < 0.05, f"Signal should be statistically significant, got p={p_val_sig}"

    # 2. Control comparison: random control features with zero correlation
    x_tr_noise = rng.normal(0, 1, (n_train, d))
    x_va_noise = rng.normal(0, 1, (n_val, d))

    eval_noise = train_and_eval_probe(
        X_train_raw=x_tr_noise,
        y_train_cont=y_tr_cont,
        y_train_cls=y_tr_cls,
        X_val_raw=x_va_noise,
        y_val_cont=y_va_cont,
        y_val_cls=y_va_cls,
        cost_threshold=cost_th,
        cost_fee=0.0004,
        cost_slip=0.0002,
        bootstrap_samples=100,
    )
    ic_noise = eval_noise["ridge"]["continuous"]["spearman_ic"]
    assert abs(ic_noise) < 0.20

    # Delta IC comparison
    ci_low, ci_high = paired_block_bootstrap_delta_ci(
        y_va_cont,
        eval_signal["ridge"]["pred_cont"],
        eval_noise["ridge"]["pred_cont"],
        n_bootstraps=200,
    )
    delta_ic = ic_signal - ic_noise
    assert delta_ic > 0.50
    assert ci_low > 0.30

    # 3. Null signal evaluation: independent pure noise
    y_null = rng.normal(0, 1, n_val)
    eval_null = train_and_eval_probe(
        X_train_raw=x_tr_noise,
        y_train_cont=rng.normal(0, 1, n_train),
        y_train_cls=np.random.choice(["BUY", "HOLD", "SELL"], size=n_train),
        X_val_raw=x_va_noise,
        y_val_cont=y_null,
        y_val_cls=np.random.choice(["BUY", "HOLD", "SELL"], size=n_val),
        cost_threshold=cost_th,
        cost_fee=0.0004,
        cost_slip=0.0002,
        bootstrap_samples=50,
    )

    # Verify IC is close to 0
    ic_null = eval_null["ridge"]["continuous"]["spearman_ic"]
    assert abs(ic_null) < 0.15, f"Null IC should be close to 0, got {ic_null}"

    # Permutation p-value under null should not be significant
    p_val_null = circular_shift_permutation_p_value(
        eval_null["ridge"]["pred_cont"], y_null, n_permutations=100
    )
    assert p_val_null > 0.05, f"Null signal must not be statistically significant, got p={p_val_null}"

    thresholds_cfg = {
        "primary_metric": {"value": 0.02, "status": "PROPOSED_NEEDS_USER_CONFIRMATION"},
        "min_practical_effect": {"value": 0.02, "status": "PROPOSED_NEEDS_USER_CONFIRMATION"},
    }
    table = evaluate_thresholds_table(
        real_ridge=eval_null["ridge"],
        real_logistic=eval_null["logistic"],
        comparisons={"real_ic_permutation_p_value": p_val_null},
        nuisance_r2=0.0,
        thresholds_cfg=thresholds_cfg,
    )
    failing_criteria = [r["threshold_key"] for r in table if not r["passed"]]
    assert len(failing_criteria) > 0, "Null signal must fail threshold criteria"
    assert "connectome_vs_matched_control_p_max" in failing_criteria


def test_probe_determinism() -> None:
    """Fixed seed evaluation must yield identical numerical results across runs."""
    rng = np.random.default_rng(999)
    n_tr, n_va, d = 150, 80, 10
    xtr = rng.normal(0, 1, (n_tr, d))
    ytr_c = rng.normal(0, 0.01, n_tr)
    ytr_a = np.random.choice(["BUY", "HOLD", "SELL"], size=n_tr)

    xva = rng.normal(0, 1, (n_va, d))
    yva_c = rng.normal(0, 0.01, n_va)
    yva_a = np.random.choice(["BUY", "HOLD", "SELL"], size=n_va)

    res1 = train_and_eval_probe(
        xtr, ytr_c, ytr_a, xva, yva_c, yva_a, cost_threshold=0.001, cost_fee=0.0004, cost_slip=0.0002, bootstrap_samples=50
    )
    res2 = train_and_eval_probe(
        xtr, ytr_c, ytr_a, xva, yva_c, yva_a, cost_threshold=0.001, cost_fee=0.0004, cost_slip=0.0002, bootstrap_samples=50
    )

    assert res1["ridge"]["continuous"]["spearman_ic"] == res2["ridge"]["continuous"]["spearman_ic"]
    assert res1["ridge"]["best_alpha"] == res2["ridge"]["best_alpha"]
    assert res1["normalization_sha256"] == res2["normalization_sha256"]
