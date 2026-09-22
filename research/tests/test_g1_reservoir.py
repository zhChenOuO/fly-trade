"""Unit tests for G1 Stateful Reservoir Engine, Gates, and Alignment."""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from research.pipeline.g1_bench import (
    check_saturation_gate,
    check_forgetting_gate,
    compute_memory_capacity,
)
from research.pipeline.g1_reservoir import (
    ScipyG1Reservoir,
    TorchG1Reservoir,
    HAS_TORCH,
    compute_spectral_radius,
    scale_weights_to_spectral_radius,
)

if HAS_TORCH:
    import torch


def _create_synthetic_sparse_reservoir(n_neurons: int = 50, density: float = 0.1, seed: int = 42) -> sp.csr_matrix:
    """Helper to create a reproducible small random sparse connectome."""
    rng = np.random.default_rng(seed)
    W = sp.random(
        n_neurons,
        n_neurons,
        density=density,
        random_state=seed,
        format="csr",
        data_rvs=lambda s: rng.standard_normal(s),
    )
    # Exclude self loops
    W.setdiag(0.0)
    W.eliminate_zeros()
    return W


def test_scipy_stateful_chunking_alignment() -> None:
    """Simulating a sequence in chunks with state carryover must produce bitwise identical states."""
    N, B, T = 60, 4, 300
    W = _create_synthetic_sparse_reservoir(n_neurons=N, seed=123)
    sensory_idx = np.array([0, 1, 2, 3])
    readout_idx = np.array([10, 20, 30, 40, 50])

    res = ScipyG1Reservoir(
        weights=W,
        sensory_idx=sensory_idx,
        readout_idx=readout_idx,
        target_rho=0.95,
        leak=0.5,
        dtype=np.float32,
    )

    rng = np.random.default_rng(456)
    u = rng.uniform(0.0, 0.5, size=(B, T)).astype(np.float32)

    # 1. Full run
    states_full, final_full = res.simulate(u)

    # 2. Split run across T1 and T - T1
    T1 = 117
    states_part1, final_part1 = res.simulate(u[:, :T1])
    states_part2, final_part2 = res.simulate(u[:, T1:], initial_state=final_part1)
    states_split = np.concatenate([states_part1, states_part2], axis=1)

    assert np.array_equal(states_full, states_split)
    assert np.array_equal(final_full, final_part2)


def test_batch_independent_sequences_no_crosstalk() -> None:
    """Modifying sequence j in a batch must not alter trajectory of sequence i != j."""
    N, B, T = 50, 3, 200
    W = _create_synthetic_sparse_reservoir(n_neurons=N, seed=789)
    sensory_idx = np.array([0, 5, 10])
    readout_idx = np.array([15, 25, 35, 45])

    res = ScipyG1Reservoir(
        weights=W,
        sensory_idx=sensory_idx,
        readout_idx=readout_idx,
        target_rho=0.95,
        leak=0.5,
        dtype=np.float32,
    )

    rng = np.random.default_rng(101)
    u_batch1 = rng.uniform(0.0, 0.5, size=(B, T)).astype(np.float32)

    # Alter sequence 2 in batch 2
    u_batch2 = u_batch1.copy()
    u_batch2[2] = rng.uniform(0.0, 0.5, size=T).astype(np.float32)

    states1, _ = res.simulate(u_batch1)
    states2, _ = res.simulate(u_batch2)

    # Sequence 0 and 1 must remain bitwise identical
    assert np.array_equal(states1[0], states2[0])
    assert np.array_equal(states1[1], states2[1])
    # Sequence 2 must differ
    assert not np.array_equal(states1[2], states2[2])


def test_shift_register_delay_line_alignment_and_mc() -> None:
    """Shift-register delay line must preserve time alignment and yield expected Memory Capacity."""
    K = 15  # Delay line of length 15
    row = np.arange(1, K)
    col = np.arange(0, K - 1)
    data = np.ones(K - 1, dtype=np.float64)
    W_delay = sp.csr_matrix((data, (row, col)), shape=(K, K))

    sensory_idx = np.array([0])
    readout_idx = np.arange(K)

    # leak=1.0 makes x[t+1] = tanh(W x[t] + I[t])
    # For small inputs, tanh(z) ≈ z, so node k exactly holds u[t-k]
    res = ScipyG1Reservoir(
        weights=W_delay,
        sensory_idx=sensory_idx,
        readout_idx=readout_idx,
        target_rho=0.99,  # dummy scaling
        leak=1.0,
        unscaled_rho=0.99,
        dtype=np.float64,
        verify_spectral_radius=False,
    )
    # Manually ensure shift weight is exactly 1.0
    res.W_eff = W_delay.copy().astype(np.float64)

    rng = np.random.default_rng(999)
    T = 1500
    u = rng.uniform(0.0, 0.5, size=T)

    states, _ = res.simulate(u, return_full_states=True)

    # Verify linear memory capacity on this delay line
    mc_dict = compute_memory_capacity(states, u, max_lag=30)
    # For k <= K-1, R^2 should be close to 1.0; total MC up to lag K-1 should be ~14
    r2_early = sum(mc_dict["r2_by_lag"][: K - 1])
    assert r2_early > 10.0, f"Delay line early MC {r2_early} too low"


def test_saturation_gate_esn_pass_and_overdriven_fail() -> None:
    """Saturation gate must pass for well-scaled ESN and fail when overdriven."""
    N, B, T = 60, 2, 200
    W = _create_synthetic_sparse_reservoir(n_neurons=N, seed=321)
    sensory_idx = np.array([0, 1])

    res = ScipyG1Reservoir(
        weights=W,
        sensory_idx=sensory_idx,
        target_rho=0.90,
        leak=0.5,
        dtype=np.float32,
    )

    rng = np.random.default_rng(555)
    # Normal moderate input
    u_normal = rng.uniform(0.0, 0.5, size=(B, T)).astype(np.float32)
    states_normal, _ = res.simulate(u_normal, return_full_states=True)
    gate_normal = check_saturation_gate(states_normal, sensory_idx)
    assert gate_normal["passed"] is True
    assert gate_normal["saturation_ratio"] < 0.05

    # Artificially overdriven reservoir states
    states_sat = np.ones((B, T, N), dtype=np.float32) * 0.95
    gate_sat = check_saturation_gate(states_sat, sensory_idx)
    assert gate_sat["passed"] is False
    assert gate_sat["saturation_ratio"] > 0.90


def test_forgetting_gate_esn_pass_and_chaotic_fail() -> None:
    """Forgetting gate must pass for stable ESN (rho=0.9) and fail for chaotic reservoir (rho=1.6)."""
    N = 80
    W = _create_synthetic_sparse_reservoir(n_neurons=N, density=0.15, seed=444)
    sensory_idx = np.array([0, 1, 2, 3])

    # 1. Stable ESN (target_rho=0.90)
    res_stable = ScipyG1Reservoir(
        weights=W,
        sensory_idx=sensory_idx,
        target_rho=0.90,
        leak=0.5,
        dtype=np.float32,
    )

    rng = np.random.default_rng(666)
    train_u_seqs = [rng.uniform(0.0, 0.5, size=600).astype(np.float32) for _ in range(10)]

    def sim_stable(u_sub, x0):
        return res_stable.simulate(u_sub, initial_state=x0, return_full_states=True)

    gate_res_stable = check_forgetting_gate(
        sim_fn=sim_stable,
        train_u_sequences=train_u_seqs,
        n_neurons=N,
        seed=42,
        t_check=500,
        d_rel_threshold=1e-3,
        min_pass_pairs=38,
    )
    assert gate_res_stable["passed"] is True
    assert gate_res_stable["n_passed_pairs"] == 40  # All 40 pairs converge

    # 2. Chaotic regime (target_rho=1.6)
    res_chaotic = ScipyG1Reservoir(
        weights=W,
        sensory_idx=sensory_idx,
        target_rho=1.60,
        leak=0.5,
        dtype=np.float32,
    )

    def sim_chaotic(u_sub, x0):
        return res_chaotic.simulate(u_sub, initial_state=x0, return_full_states=True)

    gate_res_chaotic = check_forgetting_gate(
        sim_fn=sim_chaotic,
        train_u_sequences=train_u_seqs,
        n_neurons=N,
        seed=42,
        t_check=500,
        d_rel_threshold=1e-3,
        min_pass_pairs=38,
    )
    assert gate_res_chaotic["passed"] is False


def test_torch_reservoir_graceful_import_or_alignment() -> None:
    """TorchG1Reservoir must raise ImportError without torch, or match Scipy within 1e-4 with torch."""
    W = _create_synthetic_sparse_reservoir(n_neurons=30, seed=12)
    sensory_idx = np.array([0, 1])
    readout_idx = np.array([5, 10, 15])

    if not HAS_TORCH:
        with pytest.raises(ImportError, match="PyTorch is required"):
            TorchG1Reservoir(weights=W, sensory_idx=sensory_idx, readout_idx=readout_idx)
    else:
        # If torch is available (e.g. on GPU machine)
        res_cpu = ScipyG1Reservoir(weights=W, sensory_idx=sensory_idx, readout_idx=readout_idx, target_rho=0.95)
        res_torch = TorchG1Reservoir(weights=W, sensory_idx=sensory_idx, readout_idx=readout_idx, target_rho=0.95, device="cpu")

        rng = np.random.default_rng(888)
        u = rng.uniform(0.0, 0.5, size=(2, 100)).astype(np.float32)

        out_cpu, final_cpu = res_cpu.simulate(u)
        out_torch, final_torch = res_torch.simulate(u)

        max_err = float(np.max(np.abs(out_cpu - out_torch)))
        assert max_err <= 1e-4, f"CPU vs Torch discrepancy {max_err} exceeded 1e-4"


def test_cpu_vs_torch_state_alignment(n_neurons: int = 2000, seed: int = 42) -> None:
    """Explicitly verify CPU Scipy vs GPU Torch step-by-step state agreement on a 2,000-node graph."""
    if not HAS_TORCH:
        pytest.skip("PyTorch not installed in this environment")

    W = _create_synthetic_sparse_reservoir(n_neurons=n_neurons, density=0.01, seed=seed)
    sensory_idx = np.arange(0, min(100, n_neurons))
    readout_idx = np.arange(n_neurons - min(100, n_neurons), n_neurons)

    res_cpu = ScipyG1Reservoir(weights=W, sensory_idx=sensory_idx, readout_idx=readout_idx, target_rho=0.95)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    res_torch = TorchG1Reservoir(weights=W, sensory_idx=sensory_idx, readout_idx=readout_idx, target_rho=0.95, device=device)

    rng = np.random.default_rng(seed + 1)
    B, T = 4, 100
    u = rng.uniform(0.0, 0.5, size=(B, T)).astype(np.float32)

    out_cpu, final_cpu = res_cpu.simulate(u)
    out_torch, final_torch = res_torch.simulate(u)

    max_err = float(np.max(np.abs(out_cpu - out_torch)))
    max_final_err = float(np.max(np.abs(final_cpu - final_torch)))
    print(f"CPU vs Torch max readout error: {max_err:.2e}, max final state error: {max_final_err:.2e}")
    assert max_err <= 1e-4, f"CPU vs Torch discrepancy {max_err} exceeded 1e-4"
    assert max_final_err <= 1e-4, f"Final state discrepancy {max_final_err} exceeded 1e-4"


def test_spectral_radius_rotation_matrix_and_fail_closed() -> None:
    """Spectral radius must be 1.0 on 2D rotation matrix and fail closed on invalid scaling."""
    # 2D 90-degree rotation matrix: eigenvalues are +i and -i, so rho = 1.0.
    # A symmetric Rayleigh quotient v^T W v would be 0 for real v!
    W_rot = sp.csr_matrix([[0.0, -1.0], [1.0, 0.0]])
    rho, diag = compute_spectral_radius(W_rot, return_diagnostics=True)
    np.testing.assert_allclose(rho, 1.0, atol=1e-8)
    assert diag["converged"] is True

    # Scaling to 0.95 must produce verified rho = 0.95
    W_scaled, target, unscaled, factor, sdiag = scale_weights_to_spectral_radius(
        W_rot, target_rho=0.95, verify=True, return_diagnostics=True
    )
    np.testing.assert_allclose(target, 0.95, atol=1e-8)
    np.testing.assert_allclose(unscaled, 1.0, atol=1e-8)
    np.testing.assert_allclose(sdiag["verified_rho"], 0.95, atol=1e-8)

    # Scaling zero matrix must raise ValueError
    W_zero = sp.csr_matrix((4, 4), dtype=np.float64)
    with pytest.raises(ValueError, match="zero spectral radius"):
        scale_weights_to_spectral_radius(W_zero, target_rho=0.95)


def test_streaming_saturation_matches_offline_and_catches_non_finite() -> None:
    """Streaming saturation accumulation must match full-state evaluation and catch NaN."""
    N, B, T = 30, 2, 200
    washout = 50
    W = _create_synthetic_sparse_reservoir(n_neurons=N, seed=555)
    sensory_idx = np.array([0, 1])
    readout_idx = np.array([5, 10])

    res = ScipyG1Reservoir(
        weights=W,
        sensory_idx=sensory_idx,
        readout_idx=readout_idx,
        target_rho=0.95,
        leak=0.5,
    )

    rng = np.random.default_rng(123)
    u = rng.uniform(0.0, 0.5, size=(B, T)).astype(np.float32)

    # 1. Full states offline computation
    full_states, _ = res.simulate(u, return_full_states=True)
    post_washout = full_states[:, washout:, :]
    # Non-sensory neurons: all except sensory_idx
    non_sensory_mask = np.ones(N, dtype=bool)
    non_sensory_mask[sensory_idx] = False

    non_sensory_states = post_washout[:, :, non_sensory_mask]
    offline_sat_count = int(np.sum(np.abs(non_sensory_states) > 0.90))
    offline_total = int(non_sensory_states.size)
    offline_ratio = offline_sat_count / offline_total

    # 2. Streaming saturation computation
    _, _, stream_summary = res.simulate(
        u,
        track_saturation=True,
        washout=washout,
        saturation_threshold=0.90,
    )

    assert stream_summary["saturated_points"] == offline_sat_count
    assert stream_summary["total_non_sensory_points"] == offline_total
    np.testing.assert_allclose(stream_summary["saturation_ratio"], offline_ratio, atol=1e-8)
    assert stream_summary["passed"] == (offline_ratio < 0.05)
    assert stream_summary["non_finite_count"] == 0

    # 3. Non-finite injection must fail streaming gate
    u_nan = u.copy()
    u_nan[0, 100] = np.nan
    _, _, nan_summary = res.simulate(u_nan, track_saturation=True, washout=washout)
    assert nan_summary["passed"] is False
    assert nan_summary["non_finite_count"] > 0


def test_torch_device_dtype_and_require_cuda() -> None:
    """TorchG1Reservoir must record actual_device and actual_dtype, and enforce require_cuda."""
    if not HAS_TORCH:
        pytest.skip("PyTorch not installed")

    import torch

    W = _create_synthetic_sparse_reservoir(n_neurons=20, seed=77)
    sensory_idx = np.array([0, 1])

    # CPU instantiation
    res_cpu = TorchG1Reservoir(weights=W, sensory_idx=sensory_idx, device="cpu")
    assert res_cpu.actual_device == "cpu"
    assert res_cpu.actual_dtype == "torch.float32"

    # require_cuda=True on non-CUDA system must raise RuntimeError
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="require_cuda=True specified but CUDA is not available"):
            TorchG1Reservoir(weights=W, sensory_idx=sensory_idx, require_cuda=True)
    else:
        res_cuda = TorchG1Reservoir(weights=W, sensory_idx=sensory_idx, device="cuda", require_cuda=True)
        assert "cuda" in res_cuda.actual_device


# ==============================================================================
# Spectral Radius Cluster Robustness & Regression Tests (task_fix_eigs)
# ==============================================================================

def test_spectral_radius_cluster_stability_2000_node_fixture() -> None:
    """Regression test for 2000-node fixture scaling stability under near-degenerate eigenvalues (task_fix_eigs)."""
    W = _create_synthetic_sparse_reservoir(n_neurons=2000, density=0.01, seed=42)

    # Scaling must not raise and verified_rho must match target_rho within 1e-3
    W_scaled, target, unscaled, factor, diag = scale_weights_to_spectral_radius(
        W, target_rho=0.95, verify=True, return_diagnostics=True
    )
    rel_diff = abs(diag["verified_rho"] - 0.95) / 0.95
    assert rel_diff <= 1e-3, f"Scaling verification failed: rel_diff={rel_diff:.2e} > 1e-3"

    # ScipyG1Reservoir instantiation must succeed without raising verification failure
    sensory_idx = np.arange(0, 100)
    readout_idx = np.arange(1900, 2000)
    res_cpu = ScipyG1Reservoir(weights=W, sensory_idx=sensory_idx, readout_idx=readout_idx, target_rho=0.95)
    assert abs(res_cpu.target_rho - 0.95) < 1e-9


def test_compute_spectral_radius_multi_seed_v0_stability_2000_nodes() -> None:
    """compute_spectral_radius must yield identical dominant eigenvalue across different v0 initializations."""
    W = _create_synthetic_sparse_reservoir(n_neurons=2000, density=0.01, seed=42)

    rhos = []
    for s in [1, 2, 3, 4, 5]:
        v0 = np.random.default_rng(s).normal(size=2000).astype(np.float64)
        rho = compute_spectral_radius(W, v0=v0)
        rhos.append(rho)

    spread = max(rhos) - min(rhos)
    rel_spread = spread / max(rhos)
    assert rel_spread <= 1e-6, f"Eigensolver unstable across v0 seeds: rel_spread={rel_spread:.2e} > 1e-6"


def test_spectral_radius_cluster_stability_5000_node_medium_graph() -> None:
    """Independent 5000-node random graph must also pass spectral radius scaling and multi-seed stability."""
    n = 5000
    rng = np.random.default_rng(999)
    W = sp.random(n, n, density=0.005, format="csr", dtype=np.float64, random_state=rng)

    W_scaled, target, unscaled, factor, diag = scale_weights_to_spectral_radius(
        W, target_rho=0.95, verify=True, return_diagnostics=True
    )
    rel_diff = abs(diag["verified_rho"] - 0.95) / 0.95
    assert rel_diff <= 1e-3, f"5000-node scaling verification failed: rel_diff={rel_diff:.2e} > 1e-3"

    # Multi-seed test
    rhos = []
    for s in [101, 102, 103, 104, 105]:
        v0 = np.random.default_rng(s).normal(size=n).astype(np.float64)
        rho = compute_spectral_radius(W, v0=v0)
        rhos.append(rho)

    spread = max(rhos) - min(rhos)
    rel_spread = spread / max(rhos)
    assert rel_spread <= 1e-6, f"5000-node eigensolver unstable across v0 seeds: rel_spread={rel_spread:.2e} > 1e-6"

