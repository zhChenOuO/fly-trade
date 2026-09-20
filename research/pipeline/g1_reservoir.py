"""Stateful Leaky Integrator Reservoir Engine for G1 Synthetic Benchmark.

Update Equation:
    x[t+1] = (1 - leak) * x[t] + leak * tanh(W_eff @ x[t] + I[t])
where:
    leak = 0.5
    recurrence gain = 1.0
    W_eff = W * (rho / rho(W)) scaled to target spectral radius rho
    I_i[t] = u[t] - 0.25 for i in sensory_idx; 0.0 otherwise
    noise = 0.0

Strict Governance:
- Pure scipy/numpy reference implementation (CPU).
- Optional PyTorch implementation for GPU with identical sparse dynamics.
- Stateful: preserves state across steps, updated exactly once per time step.
- No 32-step static replay or windowed state reset.
- Timed pilot CLI entrypoint for GPU runtime and VRAM extrapolation.
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import Sequence

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg


try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def compute_spectral_radius(weights: sp.csr_matrix, k: int = 1, maxiter: int = 1000) -> float:
    """Compute dominant eigenvalue magnitude of a CSR matrix using Arnoldi iteration."""
    if weights.nnz == 0 or weights.shape[0] <= 1:
        return 0.0
    try:
        vals = scipy.sparse.linalg.eigs(weights.astype(np.float64), k=k, which="LM", maxiter=maxiter, return_eigenvectors=False)
        return float(np.max(np.abs(vals)))
    except Exception:
        # Fallback to power iteration for small or ill-conditioned matrices
        n = weights.shape[0]
        v = np.ones(n, dtype=np.float64) / np.sqrt(n)
        for _ in range(50):
            v_next = weights.dot(v)
            norm = np.linalg.norm(v_next)
            if norm < 1e-12:
                return 0.0
            v = v_next / norm
        rayleigh = float(np.abs(v.dot(weights.dot(v))))
        return rayleigh


def scale_weights_to_spectral_radius(
    weights: sp.csr_matrix,
    target_rho: float,
    current_rho: float | None = None,
) -> tuple[sp.csr_matrix, float, float]:
    """Scale CSR matrix so its dominant eigenvalue magnitude equals target_rho.

    Returns:
        (W_eff, target_rho, unscaled_rho)
    """
    if current_rho is None:
        unscaled_rho = compute_spectral_radius(weights)
    else:
        unscaled_rho = float(current_rho)

    if unscaled_rho <= 1e-12:
        raise ValueError(f"Cannot scale matrix with zero spectral radius (rho={unscaled_rho})")

    scale_factor = target_rho / unscaled_rho
    w_scaled = (weights * scale_factor).tocsr()
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
    ):
        self.n_neurons = weights.shape[0]
        self.leak = float(leak)
        self.dtype = dtype

        self.sensory_idx = np.asarray(sensory_idx, dtype=np.int64)
        if readout_idx is not None:
            self.readout_idx = np.asarray(readout_idx, dtype=np.int64)
        else:
            self.readout_idx = np.arange(self.n_neurons, dtype=np.int64)

        # Scale weights to target spectral radius
        self.W_eff, self.target_rho, self.unscaled_rho = scale_weights_to_spectral_radius(
            weights=weights,
            target_rho=target_rho,
            current_rho=unscaled_rho,
        )
        self.W_eff = self.W_eff.astype(self.dtype)

    def simulate(
        self,
        u: np.ndarray,
        initial_state: np.ndarray | None = None,
        return_full_states: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run stateful reservoir simulation over sequence(s) u.

        Args:
            u: Input array of shape (T,) for single stream, or (B, T) for B independent streams.
            initial_state: Optional initial state of shape (N,) or (N, B). Default is zero.
            return_full_states: If True, returns states for all N neurons; otherwise returns
                states only for readout_idx.

        Returns:
            (states, final_state)
            - states: shape (B, T, D_readout) or (B, T, N) if return_full_states=True.
            - final_state: shape (N, B) full reservoir state at final time step.
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

        # Main stateful temporal loop: updated exactly once per time step
        leak_val = self.dtype(self.leak)
        decay_val = self.dtype(1.0 - self.leak)
        center_val = self.dtype(0.25)

        for t in range(T):
            # 1. Incoming synaptic drive: W_eff @ X (shape: N x B)
            S = self.W_eff.dot(X)

            # 2. Add scalar external sensory current u[b, t] - 0.25 to sensory neurons
            S[self.sensory_idx, :] += (u_arr[:, t] - center_val)

            # 3. Leaky integration with tanh non-linearity
            X = decay_val * X + leak_val * np.tanh(S)

            # 4. Record readout state at time step t
            if return_full_states:
                out_states[:, t, :] = X.T
            else:
                out_states[:, t, :] = X[self.readout_idx, :].T

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
    ):
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for TorchG1Reservoir but not installed in this environment.")

        self.n_neurons = weights.shape[0]
        self.leak = float(leak)
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")

        self.sensory_idx = np.asarray(sensory_idx, dtype=np.int64)
        if readout_idx is not None:
            self.readout_idx = np.asarray(readout_idx, dtype=np.int64)
        else:
            self.readout_idx = np.arange(self.n_neurons, dtype=np.int64)

        # Scale weights in scipy
        w_scaled, self.target_rho, self.unscaled_rho = scale_weights_to_spectral_radius(
            weights=weights,
            target_rho=target_rho,
            current_rho=unscaled_rho,
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
    ) -> tuple[np.ndarray, np.ndarray]:
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

        decay_val = 1.0 - self.leak
        leak_val = self.leak

        with torch.no_grad():
            for t in range(T):
                S = torch.sparse.mm(self.W_torch, X)
                S[self.sensory_tensor, :] += (u_torch[:, t] - 0.25)
                X = decay_val * X + leak_val * torch.tanh(S)

                if return_full_states:
                    out_states[:, t, :] = X.t()
                else:
                    out_states[:, t, :] = X[self.readout_tensor, :].t()

        out_np = out_states.cpu().numpy()
        final_np = X.cpu().numpy()

        if is_1d:
            return out_np[0], final_np[:, 0]
        return out_np, final_np


# ==============================================================================
# Timed Pilot CLI Entrypoint (SPEC §8)
# ==============================================================================

def run_timed_pilot(
    graph_path: str = "research/data/flywire/graph_cache.npz",
    steps: int = 1000,
    batch_size: int = 10,
    target_rho: float = 0.95,
    device: str = "cuda",
) -> dict:
    """Execute 1,000-step CUDA timed pilot and extrapolate total GPU time across 61 graphs x 3 rhos."""
    if not HAS_TORCH or not torch.cuda.is_available():
        msg = "CUDA / PyTorch not available. Timed pilot must be executed on a GPU machine with CUDA."
        print(f"NOTICE: {msg}")
        return {
            "status": "SKIPPED_NO_CUDA",
            "message": msg,
        }

    from research.pipeline.flywire_graph import load_flywire_graph

    print(f"Loading FlyWire graph from cache...")
    t0 = time.time()
    graph = load_flywire_graph(use_cache=True)
    load_time = time.time() - t0
    print(f"Loaded {graph.n_neurons:,} neurons, {graph.n_synapses:,} synapses in {load_time:.2f}s.")

    # Initialize Torch reservoir
    print(f"Initializing TorchG1Reservoir on {device} (rho={target_rho})...")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    res = TorchG1Reservoir(
        weights=graph.weights,
        sensory_idx=graph.sensory_idx,
        readout_idx=graph.motor_idx,
        target_rho=target_rho,
        device=device,
    )
    init_time = time.time() - t0
    print(f"Reservoir initialized in {init_time:.2f}s.")

    # Generate synthetic input for pilot
    rng = np.random.default_rng(42)
    u_pilot = rng.uniform(0.0, 0.5, size=(batch_size, steps)).astype(np.float32)

    # Warmup 50 steps
    print(f"Running 50 warmup steps...")
    _ = res.simulate(u_pilot[:, :50])
    torch.cuda.synchronize()

    # Benchmark timed pilot
    print(f"Benchmarking {steps} steps with batch_size={batch_size}...")
    t0 = time.time()
    _, final_state = res.simulate(u_pilot)
    torch.cuda.synchronize()
    elapsed = time.time() - t0

    # Metrics
    total_step_updates = steps * batch_size
    time_per_step_sec = elapsed / steps
    time_per_update_sec = elapsed / total_step_updates
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

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
        "benchmark_steps": steps,
        "batch_size": batch_size,
        "elapsed_seconds": elapsed,
        "time_per_step_seconds": time_per_step_sec,
        "time_per_state_update_seconds": time_per_update_sec,
        "peak_vram_mb": peak_vram_mb,
        "total_g1_state_updates": total_g1_updates,
        "extrapolated_gpu_hours": extrapolated_gpu_hours,
        "budget_hours": budget_hours,
        "within_budget": within_budget,
    }

    print("\n=== G1 Timed Pilot Benchmark Results ===")
    print(f"Elapsed time: {elapsed:.2f}s for {steps} steps x {batch_size} sequences.")
    print(f"Time per step: {time_per_step_sec*1000:.2f} ms ({time_per_update_sec*1000:.4f} ms per sequence update).")
    print(f"Peak VRAM: {peak_vram_mb:.1f} MB.")
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
    args = parser.parse_args()

    if args.timed_pilot:
        run_timed_pilot(
            graph_path=args.graph_path,
            steps=args.steps,
            batch_size=args.batch_size,
            target_rho=args.rho,
            device=args.device,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
