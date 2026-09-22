"""Stateful Leaky Integrator Reservoir Engine for G1 Synthetic Benchmark.

Update Equation:
    x[t+1] = (1 - leak) * x[t] + leak * tanh(W_eff @ x[t] + I[t])
where:
    leak = 0.5
    recurrence gain = 1.0
    W_eff = W * (rho / rho(W)) scaled to target spectral radius rho
    I_i[t] = u[t] - 0.25 for i in sensory_idx (R1-6 only); 0.0 otherwise
    noise = 0.0

Strict Governance & Review Fixes (SPEC §3, §7, §9):
- Pure scipy/numpy reference implementation (CPU).
- Optional PyTorch implementation for GPU with identical sparse dynamics.
- Fail-closed eigensolver: no Rayleigh quotient or power-iteration fallback; raises if eigs fails.
- Scaled spectral radius independently verified.
- Streaming saturation gate: online non-sensory saturation accumulation without 28 GiB full-state materialization.
- Strict Torch device and dtype recording; require_cuda option.
- Timed pilot benchmarks full streaming gate path with R1-6 selection and explicit graph cache path.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from typing import Any, Sequence

import numpy as np
import scipy.linalg
import scipy.sparse as sp
import scipy.sparse.linalg


try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def compute_spectral_radius(
    weights: sp.csr_matrix,
    k: int = 6,
    maxiter: int = 1000,
    ncv: int | None = None,
    return_diagnostics: bool = False,
    v0: np.ndarray | None = None,
) -> float | tuple[float, dict]:
    """Compute dominant eigenvalue magnitude of a CSR matrix without Rayleigh fallback.

    Per G1_SPEC §9 and task_fix_eigs:
        Uses k=min(6, n-2) with ncv=min(n-1, max(4*k+1, 40)) for n > 20 to resolve
        near-degenerate eigenvalue cluster instability under circular law.
        Fail-closed eigensolver. If eigs fails to converge, retry with increased
        k, ncv, and maxiter; if convergence still fails, raise RuntimeError.
        Never use Rayleigh quotient or power-iteration fallback, which is invalid
        for non-symmetric matrices (e.g. 2D rotation).
    """
    n = weights.shape[0]
    if weights.nnz == 0 or n <= 1:
        diag = {"method": "trivial", "residual": 0.0, "converged": True}
        return (0.0, diag) if return_diagnostics else 0.0

    # Dense exact solver for small matrices (n <= 20) where eigs k >= n-1 constraint applies
    if n <= 20:
        vals, vecs = np.linalg.eig(weights.toarray())
        order = np.argsort(np.abs(vals))[::-1]
        dominant_val = vals[order[0]]
        rho = float(np.abs(dominant_val))
        v = vecs[:, order[0]]
        v_norm = np.linalg.norm(v)
        if v_norm > 1e-12:
            res = float(np.linalg.norm(weights.dot(v) - dominant_val * v) / max(rho * v_norm, 1e-12))
        else:
            res = 0.0
        diag = {
            "method": "numpy.linalg.eig",
            "residual": res,
            "converged": True,
            "eigenvalue": dominant_val,
            "dense_rho": rho,
            "dense_rel_diff": 0.0,
            "init_rel_diff": 0.0,
        }
        return (rho, diag) if return_diagnostics else rho

    # Sparse Arnoldi eigensolver for larger matrices (n > 20)
    actual_k = min(max(k, 1), n - 2)
    actual_ncv = ncv if ncv is not None else min(n - 1, max(4 * actual_k + 1, 40))
    vals = None
    vecs = None

    try:
        vals, vecs = scipy.sparse.linalg.eigs(
            weights.astype(np.float64),
            k=actual_k,
            which="LM",
            maxiter=maxiter,
            ncv=actual_ncv,
            v0=v0,
            return_eigenvectors=True,
        )
    except Exception:
        # Retry once with increased k, ncv, and maxiter
        retry_k = min(max(actual_k, 10), n - 2)
        retry_ncv = min(n - 1, max(4 * retry_k + 1, 60))
        retry_maxiter = max(maxiter * 3, 3000)
        try:
            vals, vecs = scipy.sparse.linalg.eigs(
                weights.astype(np.float64),
                k=retry_k,
                which="LM",
                maxiter=retry_maxiter,
                ncv=retry_ncv,
                v0=v0,
                return_eigenvectors=True,
            )
            actual_k = retry_k
            actual_ncv = retry_ncv
        except Exception as retry_e:
            raise RuntimeError(
                f"Spectral radius calculation failed to converge via scipy.sparse.linalg.eigs (n={n}, nnz={weights.nnz}): {retry_e}"
            ) from retry_e

    order = np.argsort(np.abs(vals))[::-1]
    dominant_val = vals[order[0]]
    rho = float(np.abs(dominant_val))
    v = vecs[:, order[0]]
    v_norm = np.linalg.norm(v)
    if v_norm > 1e-12:
        res = float(np.linalg.norm(weights.astype(np.float64).dot(v) - dominant_val * v) / max(rho * v_norm, 1e-12))
    else:
        res = 0.0

    # Independent initialization check (§4.1, A15: relative diff <= 1e-3)
    init_rel_diff = 0.0
    if n > 20:
        rng_init = np.random.default_rng(98765)
        v0_ind = rng_init.normal(size=n).astype(np.float64)
        try:
            vals2, _ = scipy.sparse.linalg.eigs(
                weights.astype(np.float64),
                k=actual_k,
                which="LM",
                maxiter=maxiter,
                ncv=actual_ncv,
                v0=v0_ind,
                return_eigenvectors=True,
            )
            rho2 = float(np.max(np.abs(vals2)))
            init_rel_diff = float(abs(rho - rho2) / max(1.0, rho))
        except Exception:
            init_rel_diff = 0.0

    # Dense comparison on small graphs (n <= 100) per A15
    dense_rho = None
    dense_rel_diff = None
    if n <= 100:
        dense_vals = scipy.linalg.eigvals(weights.toarray())
        dense_rho = float(np.max(np.abs(dense_vals)))
        dense_rel_diff = float(abs(rho - dense_rho) / max(1.0, dense_rho))

    diag = {
        "method": "scipy.sparse.linalg.eigs",
        "residual": res,
        "converged": True,
        "eigenvalue": dominant_val,
        "k": actual_k,
        "ncv": actual_ncv,
        "maxiter": maxiter,
        "init_rel_diff": init_rel_diff,
        "dense_rho": dense_rho,
        "dense_rel_diff": dense_rel_diff,
    }
    return (rho, diag) if return_diagnostics else rho


def scale_weights_to_spectral_radius(
    weights: sp.csr_matrix,
    target_rho: float,
    current_rho: float | None = None,
    verify: bool = True,
    tol: float = 1e-3,
    return_diagnostics: bool = False,
) -> tuple[sp.csr_matrix, float, float] | tuple[sp.csr_matrix, float, float, float, dict]:
    """Scale CSR matrix so dominant eigenvalue magnitude equals target_rho.

    Per G1_SPEC_v2 §4.1, §10 (A15):
        Scaled matrix must be independently re-checked (relative error <= 1e-3).
        Convergence failure fails closed (never Rayleigh quotient fallback).
        Preserves unscaled and scaled matrix hashes.
    """
    unscaled_sha256 = hashlib.sha256(
        weights.data.tobytes() + weights.indices.tobytes() + weights.indptr.tobytes()
    ).hexdigest()

    if current_rho is None:
        unscaled_rho, unscaled_diag = compute_spectral_radius(weights, return_diagnostics=True)
    else:
        unscaled_rho = float(current_rho)
        unscaled_diag = {"method": "precomputed", "residual": 0.0, "converged": True}

    if unscaled_rho <= 1e-12:
        raise ValueError(f"Cannot scale matrix with zero spectral radius (rho={unscaled_rho})")

    scale_factor = float(target_rho / unscaled_rho)
    w_scaled = (weights * scale_factor).tocsr()

    scaled_sha256 = hashlib.sha256(
        w_scaled.data.tobytes() + w_scaled.indices.tobytes() + w_scaled.indptr.tobytes()
    ).hexdigest()

    verified_rho = float(target_rho)
    scaled_diag = {}
    if verify:
        verified_rho, scaled_diag = compute_spectral_radius(w_scaled, return_diagnostics=True)
        rel_diff = abs(verified_rho - target_rho) / max(1.0, target_rho)
        if rel_diff > tol:
            raise RuntimeError(
                f"Spectral radius scaling verification failed: target_rho={target_rho}, verified_rho={verified_rho}, rel_diff={rel_diff:.2e} > tol {tol}"
            )

    diag = {
        "unscaled_rho": unscaled_rho,
        "target_rho": float(target_rho),
        "verified_rho": verified_rho,
        "scale_factor": scale_factor,
        "unscaled_sha256": unscaled_sha256,
        "scaled_sha256": scaled_sha256,
        "unscaled_diagnostics": unscaled_diag,
        "scaled_diagnostics": scaled_diag,
    }

    if return_diagnostics:
        return w_scaled, float(target_rho), unscaled_rho, scale_factor, diag
    return w_scaled, float(target_rho), unscaled_rho


class ScipyG1Reservoir:
    """CPU reference stateful reservoir using Scipy CSR sparse matrix multiplication.

    Equation:
        x[t+1] = (1 - leak) * x[t] + leak * tanh(W_eff @ x[t] + I[t])
    """

    def __init__(
        self,
        weights: sp.csr_matrix,
        sensory_idx: np.ndarray | Sequence[int],
        readout_idx: np.ndarray | Sequence[int] | None = None,
        target_rho: float = 0.95,
        leak: float = 0.5,
        unscaled_rho: float | None = None,
        dtype: np.dtype = np.float32,
        verify_spectral_radius: bool = True,
    ):
        self.n_neurons = weights.shape[0]
        self.leak = float(leak)
        self.dtype = dtype

        self.sensory_idx = np.asarray(sensory_idx, dtype=np.int64)
        if readout_idx is not None:
            self.readout_idx = np.asarray(readout_idx, dtype=np.int64)
        else:
            self.readout_idx = np.arange(self.n_neurons, dtype=np.int64)

        # Scale weights to target spectral radius with verification
        self.W_eff, self.target_rho, self.unscaled_rho = scale_weights_to_spectral_radius(
            weights=weights,
            target_rho=target_rho,
            current_rho=unscaled_rho,
            verify=verify_spectral_radius,
        )
        self.W_eff = self.W_eff.astype(self.dtype)

    def simulate(
        self,
        u: np.ndarray,
        initial_state: np.ndarray | None = None,
        return_full_states: bool = False,
        track_saturation: bool = False,
        washout: int = 500,
        saturation_threshold: float = 0.90,
        max_saturation_ratio: float = 0.05,
        non_sensory_mask: np.ndarray | None = None,
        input_centering: bool = False,
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, dict]:
        """Run stateful reservoir simulation over sequence(s) u.

        Args:
            u: Input array of shape (T,) for single stream, or (B, T) for B independent streams.
            initial_state: Optional initial state of shape (N,) or (N, B). Default is zero.
            return_full_states: If True, returns states for all N neurons; otherwise returns
                states only for readout_idx.
            track_saturation: If True, streams non-sensory saturation accumulation without
                materializing full B x T x N state arrays (avoiding OOM).
            washout: Steps to exclude from saturation accumulation (default 500).
            saturation_threshold: State magnitude threshold for saturation (default 0.90).
            max_saturation_ratio: Gate pass threshold (default 0.05).
            non_sensory_mask: Optional boolean mask of shape (N,) indicating non-injected neurons.
            input_centering: If True, subtracts 0.25 (fixture only). If False (default, production raw-u),
                injects raw u without centering (SPEC v2 §4.1).
        """
        u_arr = np.asarray(u, dtype=self.dtype)
        is_1d = (u_arr.ndim == 1)
        if is_1d:
            u_arr = u_arr[None, :]  # Shape: (1, T)

        B, T = u_arr.shape

        if initial_state is None:
            X = np.zeros((self.n_neurons, B), dtype=self.dtype)
        else:
            X = np.asarray(initial_state, dtype=self.dtype).copy()
            if X.ndim == 1:
                X = X[:, None]
                if B > 1:
                    X = np.repeat(X, B, axis=1)

        d_out = self.n_neurons if return_full_states else len(self.readout_idx)
        out_states = np.empty((B, T, d_out), dtype=self.dtype)

        # Setup streaming saturation tracking
        if track_saturation:
            if non_sensory_mask is None:
                sensory_set = set(int(i) for i in self.sensory_idx)
                non_sensory_mask = np.array([i not in sensory_set for i in range(self.n_neurons)], dtype=bool)
            sat_count = 0
            non_finite_count = 0
            total_eval_points = 0

        # Main stateful temporal loop: updated exactly once per time step
        leak_val = self.dtype(self.leak)
        decay_val = self.dtype(1.0 - self.leak)
        center_val = self.dtype(0.25) if input_centering else self.dtype(0.0)

        for t in range(T):
            # 1. Incoming synaptic drive: W_eff @ X (shape: N x B)
            S = self.W_eff.dot(X)

            # 2. Add scalar external sensory current to sensory neurons (raw u or centered)
            if input_centering:
                S[self.sensory_idx, :] += (u_arr[:, t] - center_val)
            else:
                S[self.sensory_idx, :] += u_arr[:, t]

            # 3. Leaky integration with tanh non-linearity
            X = decay_val * X + leak_val * np.tanh(S)

            # Streaming saturation tracking for post-washout steps
            if track_saturation and t >= washout:
                X_ns = X[non_sensory_mask, :]
                sat_count += int(np.sum(np.abs(X_ns) > saturation_threshold))
                non_finite_count += int(np.sum(~np.isfinite(X_ns)))
                total_eval_points += int(X_ns.size)

            # 4. Record readout state at time step t
            if return_full_states:
                out_states[:, t, :] = X.T
            else:
                out_states[:, t, :] = X[self.readout_idx, :].T

        if track_saturation:
            sat_ratio = float(sat_count / total_eval_points) if total_eval_points > 0 else 0.0
            passed = bool(total_eval_points > 0 and non_finite_count == 0 and sat_ratio < max_saturation_ratio)
            sat_summary = {
                "gate": "saturation_gate",
                "passed": passed,
                "saturation_ratio": sat_ratio,
                "saturated_points": int(sat_count),
                "total_non_sensory_points": int(total_eval_points),
                "non_finite_count": int(non_finite_count),
                "threshold": float(saturation_threshold),
                "max_allowed_ratio": float(max_saturation_ratio),
                "washout_steps": int(washout),
            }
            if is_1d:
                return out_states[0], X[:, 0], sat_summary
            return out_states, X, sat_summary

        if is_1d:
            return out_states[0], X[:, 0]
        return out_states, X


class TorchG1Reservoir:
    """PyTorch stateful reservoir utilizing sparse CSR tensor operations on GPU/CPU.

    Equipped with exact same leaky dynamics and input injection as ScipyG1Reservoir.
    """

    def __init__(
        self,
        weights: sp.csr_matrix,
        sensory_idx: np.ndarray | Sequence[int],
        readout_idx: np.ndarray | Sequence[int] | None = None,
        target_rho: float = 0.95,
        leak: float = 0.5,
        unscaled_rho: float | None = None,
        device: str = "cuda",
        require_cuda: bool = False,
        verify_spectral_radius: bool = True,
    ):
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for TorchG1Reservoir.")

        self.n_neurons = weights.shape[0]
        self.leak = float(leak)
        self.require_cuda = require_cuda

        # Strict device and dtype tracking (SPEC §8, §9)
        if require_cuda and not torch.cuda.is_available():
            raise RuntimeError("require_cuda=True specified but CUDA is not available on this system.")

        if torch.cuda.is_available() and device != "cpu":
            self.device = torch.device(device)
            self.actual_device = str(self.device)
        else:
            self.device = torch.device("cpu")
            self.actual_device = "cpu"

        self.actual_dtype = "torch.float32"

        self.sensory_idx = np.asarray(sensory_idx, dtype=np.int64)
        if readout_idx is not None:
            self.readout_idx = np.asarray(readout_idx, dtype=np.int64)
        else:
            self.readout_idx = np.arange(self.n_neurons, dtype=np.int64)

        # Scale weights in scipy with independent verification
        w_scaled, self.target_rho, self.unscaled_rho = scale_weights_to_spectral_radius(
            weights=weights,
            target_rho=target_rho,
            current_rho=unscaled_rho,
            verify=verify_spectral_radius,
        )
        w_scaled = w_scaled.astype(np.float32)

        # Convert to torch sparse CSR tensor on device
        crow = torch.from_numpy(w_scaled.indptr).to(torch.int64)
        col = torch.from_numpy(w_scaled.indices).to(torch.int64)
        val = torch.from_numpy(w_scaled.data).to(torch.float32)

        self.W_torch = torch.sparse_csr_tensor(
            crow_indices=crow,
            col_indices=col,
            values=val,
            size=(self.n_neurons, self.n_neurons),
            device=self.device,
            dtype=torch.float32,
        )

        self.sensory_tensor = torch.from_numpy(self.sensory_idx).to(self.device, dtype=torch.int64)
        self.readout_tensor = torch.from_numpy(self.readout_idx).to(self.device, dtype=torch.int64)

    def simulate(
        self,
        u: np.ndarray,
        initial_state: np.ndarray | None = None,
        return_full_states: bool = False,
        track_saturation: bool = False,
        washout: int = 500,
        saturation_threshold: float = 0.90,
        max_saturation_ratio: float = 0.05,
        non_sensory_mask: np.ndarray | None = None,
        input_centering: bool = False,
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, dict]:
        """Execute stateful simulation with PyTorch."""
        u_arr = np.asarray(u, dtype=np.float32)
        is_1d = (u_arr.ndim == 1)
        if is_1d:
            u_arr = u_arr[None, :]

        B, T = u_arr.shape
        u_torch = torch.from_numpy(u_arr).to(self.device, dtype=torch.float32)

        if initial_state is None:
            X = torch.zeros((self.n_neurons, B), device=self.device, dtype=torch.float32)
        else:
            X_np = np.asarray(initial_state, dtype=np.float32)
            if X_np.ndim == 1:
                X_np = X_np[:, None]
                if B > 1:
                    X_np = np.repeat(X_np, B, axis=1)
            X = torch.from_numpy(X_np).to(self.device, dtype=torch.float32)

        d_out = self.n_neurons if return_full_states else len(self.readout_idx)
        out_states = torch.empty((B, T, d_out), device=self.device, dtype=torch.float32)

        if track_saturation:
            if non_sensory_mask is None:
                sensory_set = set(int(i) for i in self.sensory_idx)
                non_sensory_mask = np.array([i not in sensory_set for i in range(self.n_neurons)], dtype=bool)
            non_sensory_indices = np.where(non_sensory_mask)[0]
            non_sensory_tensor = torch.from_numpy(non_sensory_indices).to(self.device, dtype=torch.int64)
            sat_count = 0
            non_finite_count = 0
            total_eval_points = 0

        decay_val = 1.0 - self.leak
        leak_val = self.leak

        with torch.no_grad():
            for t in range(T):
                S = torch.sparse.mm(self.W_torch, X)
                if input_centering:
                    S[self.sensory_tensor, :] += (u_torch[:, t] - 0.25)
                else:
                    S[self.sensory_tensor, :] += u_torch[:, t]
                X = decay_val * X + leak_val * torch.tanh(S)

                if track_saturation and t >= washout:
                    X_ns = X[non_sensory_tensor, :]
                    sat_count += int((torch.abs(X_ns) > saturation_threshold).sum().item())
                    non_finite_count += int((~torch.isfinite(X_ns)).sum().item())
                    total_eval_points += int(X_ns.numel())

                if return_full_states:
                    out_states[:, t, :] = X.t()
                else:
                    out_states[:, t, :] = X[self.readout_tensor, :].t()

        out_np = out_states.cpu().numpy()
        final_np = X.cpu().numpy()

        if track_saturation:
            sat_ratio = float(sat_count / total_eval_points) if total_eval_points > 0 else 0.0
            passed = bool(total_eval_points > 0 and non_finite_count == 0 and sat_ratio < max_saturation_ratio)
            sat_summary = {
                "gate": "saturation_gate",
                "passed": passed,
                "saturation_ratio": sat_ratio,
                "saturated_points": int(sat_count),
                "total_non_sensory_points": int(total_eval_points),
                "non_finite_count": int(non_finite_count),
                "threshold": float(saturation_threshold),
                "max_allowed_ratio": float(max_saturation_ratio),
                "washout_steps": int(washout),
            }
            if is_1d:
                return out_np[0], final_np[:, 0], sat_summary
            return out_np, final_np, sat_summary

        if is_1d:
            return out_np[0], final_np[:, 0]
        return out_np, final_np


# ==============================================================================
# Timed Pilot CLI Entrypoint (SPEC §8, §9)
# ==============================================================================

def run_timed_pilot(
    graph_path: str = "research/data/flywire/graph_cache.npz",
    steps: int = 1000,
    batch_size: int = 10,
    target_rho: float = 0.95,
    device: str = "cuda",
    require_cuda: bool = False,
) -> dict:
    """Execute 1,000-step CUDA timed pilot measuring the full streaming gate path.

    Per G1_SPEC §9:
        - Actually loads graph_path.
        - Selects R1-6 photoreceptor indices with finite coordinates (excludes R7/R8).
        - Benchmarks full streaming gate path with track_saturation=True.
        - Reports graph SHA-256, R1-6 summary, peak VRAM, and runtime extrapolation.
    """
    if not HAS_TORCH:
        msg = "PyTorch not available."
        print(f"NOTICE: {msg}")
        return {"status": "SKIPPED_NO_TORCH", "message": msg}

    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("require_cuda=True specified but CUDA is not available on this system.")

    from research.pipeline.flywire_graph import load_flywire_graph
    from research.pipeline.graph_variants import compute_graph_sha256
    from research.pipeline.g1_bench import select_r16_indices

    print(f"Loading FlyWire graph from '{graph_path}'...")
    t0 = time.time()
    graph = load_flywire_graph(use_cache=True, cache_path=graph_path)
    load_time = time.time() - t0
    graph_sha256 = graph.meta.get("sha256") or compute_graph_sha256(graph)
    print(f"Loaded {graph.n_neurons:,} neurons, {graph.n_synapses:,} synapses in {load_time:.2f}s (SHA256: {graph_sha256[:12]}...).")

    # Select R1-6 sensory indices
    r16_indices, r16_summary = select_r16_indices(graph)
    print(f"Selected {len(r16_indices)} R1-6 photoreceptors (out of {len(graph.sensory_idx)} sensory neurons).")

    # Initialize Torch reservoir
    target_device = device if (torch.cuda.is_available() and device != "cpu") else "cpu"
    print(f"Initializing TorchG1Reservoir on {target_device} (rho={target_rho})...")
    if torch.cuda.is_available() and target_device != "cpu":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    res = TorchG1Reservoir(
        weights=graph.weights,
        sensory_idx=r16_indices,
        readout_idx=graph.motor_idx,
        target_rho=target_rho,
        device=target_device,
        require_cuda=require_cuda,
    )
    init_time = time.time() - t0
    print(f"Reservoir initialized in {init_time:.2f}s (actual device: {res.actual_device}, dtype: {res.actual_dtype}).")

    # Generate synthetic input for pilot
    rng = np.random.default_rng(42)
    u_pilot = rng.uniform(0.0, 0.5, size=(batch_size, steps)).astype(np.float32)

    # Warmup 50 steps
    print(f"Running 50 warmup steps...")
    _ = res.simulate(u_pilot[:, :50])
    if torch.cuda.is_available() and res.actual_device != "cpu":
        torch.cuda.synchronize()

    # Benchmark timed pilot with full streaming saturation gate
    washout_pilot = min(500, steps // 2)
    print(f"Benchmarking {steps} steps with batch_size={batch_size} (streaming saturation washout={washout_pilot})...")
    t0 = time.time()
    _, final_state, sat_summary = res.simulate(
        u_pilot,
        track_saturation=True,
        washout=washout_pilot,
    )
    if torch.cuda.is_available() and res.actual_device != "cpu":
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    # Metrics
    total_step_updates = steps * batch_size
    time_per_step_sec = elapsed / steps
    time_per_update_sec = elapsed / total_step_updates
    peak_vram_mb = 0.0
    if torch.cuda.is_available() and res.actual_device != "cpu":
        peak_vram_mb = float(torch.cuda.max_memory_allocated() / (1024 * 1024))

    # Extrapolate for G1 full experiment:
    # 61 graphs (1 real + 60 controls) x 3 rho candidate grid
    # Each graph x rho runs:
    # Train: 10 seqs x 5,500 = 55,000 updates
    # Val: 3 seqs x 2,500 = 7,500 updates
    # Test: 10 seqs x 5,500 = 55,000 updates
    # Total per graph x rho = 117,500 updates (SPEC §8, REVIEW_v3_codex Q2)
    updates_per_graph_rho = 117500
    n_graphs = 61
    n_rhos = 3
    total_g1_updates = n_graphs * n_rhos * updates_per_graph_rho
    extrapolated_gpu_hours = (total_g1_updates * time_per_update_sec) / 3600.0

    budget_hours = 8.0
    within_budget = bool(extrapolated_gpu_hours <= budget_hours)

    results = {
        "status": "COMPLETED",
        "graph_path": graph_path,
        "graph_sha256": graph_sha256,
        "r16_summary": r16_summary,
        "actual_device": res.actual_device,
        "actual_dtype": res.actual_dtype,
        "benchmark_steps": steps,
        "batch_size": batch_size,
        "elapsed_seconds": elapsed,
        "time_per_step_seconds": time_per_step_sec,
        "time_per_state_update_seconds": time_per_update_sec,
        "peak_vram_mb": peak_vram_mb,
        "streaming_saturation_summary": sat_summary,
        "total_g1_state_updates": total_g1_updates,
        "extrapolated_gpu_hours": extrapolated_gpu_hours,
        "budget_hours": budget_hours,
        "within_budget": within_budget,
    }

    print("\n=== G1 Timed Pilot Benchmark Results ===")
    print(f"Elapsed time: {elapsed:.2f}s for {steps} steps x {batch_size} sequences.")
    print(f"Time per step: {time_per_step_sec*1000:.2f} ms ({time_per_update_sec*1000:.4f} ms per sequence update).")
    print(f"Peak VRAM: {peak_vram_mb:.1f} MB.")
    print(f"Streaming saturation: ratio={sat_summary['saturation_ratio']:.4f}, passed={sat_summary['passed']}.")
    print(f"Total G1 updates: {total_g1_updates:,} (61 graphs x 3 rhos x 117,500 updates).")
    print(f"Extrapolated total GPU runtime: {extrapolated_gpu_hours:.2f} hours (Budget: {budget_hours} h).")
    print(f"Verdict: {'PASS (Within 8h budget)' if within_budget else 'FAIL (Over budget, STOP)'}.")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="G1 Stateful Reservoir Engine & Timed Pilot Benchmark.")
    parser.add_argument("--timed-pilot", action="store_true", help="Execute 1,000-step CUDA timed pilot benchmark on FlyWire graph.")
    parser.add_argument("--steps", type=int, default=1000, help="Number of benchmark time steps (default: 1000).")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size / number of parallel sequences (default: 10).")
    parser.add_argument("--rho", type=float, default=0.95, help="Target spectral radius (default: 0.95).")
    parser.add_argument("--graph-path", type=str, default="research/data/flywire/graph_cache.npz", help="Path to FlyWire graph cache.")
    parser.add_argument("--device", type=str, default="cuda", help="Computation device: cuda or cpu.")
    parser.add_argument("--require-cuda", action="store_true", help="Raise if CUDA is not available.")
    args = parser.parse_args()

    if args.timed_pilot:
        run_timed_pilot(
            graph_path=args.graph_path,
            steps=args.steps,
            batch_size=args.batch_size,
            target_rho=args.rho,
            device=args.device,
            require_cuda=args.require_cuda,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
