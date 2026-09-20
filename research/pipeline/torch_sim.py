"""Float32 CUDA CSR implementation of the leaky reservoir batch dynamics."""
from __future__ import annotations

from threading import RLock

import numpy as np
import scipy.sparse as sp
import torch

from src.connectome.schema import ConnectomeGraph

_WEIGHT_CACHE: dict[tuple[int, str], tuple[object, torch.Tensor]] = {}
_CACHE_LOCK = RLock()
_VRAM_FRACTION = 0.8
_DENSE_WORKING_COPIES_PER_NEURON = 8


def _cuda_csr(weights: sp.spmatrix, device: torch.device) -> torch.Tensor:
    """Copy one SciPy CSR matrix to the selected CUDA device and cache it."""
    key = (id(weights), str(device))
    with _CACHE_LOCK:
        cached = _WEIGHT_CACHE.get(key)
        if cached is not None and cached[0] is weights:
            return cached[1]

        matrix = sp.csr_matrix(weights, dtype=np.float32, copy=True)
        matrix.sum_duplicates()
        matrix.sort_indices()
        crow = torch.as_tensor(matrix.indptr, dtype=torch.int64, device=device)
        columns = torch.as_tensor(matrix.indices, dtype=torch.int64, device=device)
        values = torch.as_tensor(matrix.data, dtype=torch.float32, device=device)
        tensor = torch.sparse_csr_tensor(
            crow,
            columns,
            values,
            size=matrix.shape,
            dtype=torch.float32,
            device=device,
        )
        _WEIGHT_CACHE[key] = (weights, tensor)
        return tensor


class TorchReservoir:
    """Run independent reservoir inputs as dense columns against a CUDA CSR graph."""

    def __init__(
        self,
        graph: ConnectomeGraph,
        device: str | torch.device | None = None,
        memory_fraction: float = _VRAM_FRACTION,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; backend='torch' requires a CUDA GPU")
        if not 0.0 < memory_fraction <= _VRAM_FRACTION:
            raise ValueError(f"memory_fraction must be in (0, {_VRAM_FRACTION}]")

        self.graph = graph
        self.device = torch.device(device or "cuda")
        if self.device.type != "cuda":
            raise ValueError("TorchReservoir requires a CUDA device")
        self.memory_fraction = float(memory_fraction)
        self.n_neurons = int(graph.n_neurons)
        self.sensory_idx = torch.as_tensor(
            np.asarray(graph.sensory_idx, dtype=np.int64),
            dtype=torch.int64,
            device=self.device,
        )
        self.motor_idx = torch.as_tensor(
            np.asarray(graph.motor_idx, dtype=np.int64),
            dtype=torch.int64,
            device=self.device,
        )
        self.weights = _cuda_csr(graph.weights, self.device)
        self.peak_memory_bytes = 0

    def max_batch_size(
        self,
        steps: int = 32,
        n_sensory: int | None = None,
        n_recorded: int = 0,
    ) -> int:
        """Conservatively cap dense state batches below 80% total VRAM use."""
        if n_sensory is None:
            n_sensory = len(self.graph.sensory_idx)
        torch.cuda.synchronize(self.device)
        free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)
        currently_used = total_bytes - free_bytes
        budget = int(total_bytes * self.memory_fraction) - currently_used
        bytes_per_sample = 4 * (
            self.n_neurons * _DENSE_WORKING_COPIES_PER_NEURON
            + int(steps) * int(n_sensory)
            + int(steps) * len(self.graph.motor_idx)
            + int(n_recorded)
        )
        if budget < bytes_per_sample:
            raise MemoryError(
                "CUDA memory use leaves less than one sample within the 80% VRAM limit"
            )
        return max(1, int(budget // bytes_per_sample))

    @staticmethod
    def _parameter_vector(value, batch_size: int, name: str) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32)
        if array.ndim == 0:
            return np.full(batch_size, array.item(), dtype=np.float32)
        if array.shape != (batch_size,):
            raise ValueError(f"{name} must be scalar or have shape ({batch_size},)")
        return array

    @torch.inference_mode()
    def simulate_batch(
        self,
        currents: np.ndarray,
        gain: np.ndarray | float,
        leak: np.ndarray | float,
        record_motor_window: int | None = None,
        return_final_state: bool = False,
        record_activity_indices: np.ndarray | None = None,
    ) -> np.ndarray | tuple[np.ndarray, ...]:
        """Simulate `(B,T,n_sensory)` currents and return `(B,T,n_motor)`.

        `return_final_state=True` also returns final all-neuron activity with shape
        `(B,n_neurons)`, used by the unchanged S1 saturation criterion.
        `record_activity_indices` additionally returns the time mean for the selected
        neuron indices, avoiding storage of full all-neuron activity traces.
        """
        currents = np.asarray(currents)
        if currents.ndim != 3:
            raise ValueError("currents must have shape (B, T, n_sensory)")
        batch_size, steps, n_sensory = currents.shape
        if n_sensory != len(self.graph.sensory_idx):
            raise ValueError(
                f"expected {len(self.graph.sensory_idx)} sensory currents, got {n_sensory}"
            )
        if batch_size == 0:
            empty_motor = np.empty((0, steps, len(self.graph.motor_idx)), dtype=np.float32)
            empty_parts: list[np.ndarray] = [empty_motor]
            if return_final_state:
                empty_parts.append(np.empty((0, self.n_neurons), dtype=np.float32))
            if record_activity_indices is not None:
                selected = np.asarray(record_activity_indices, dtype=np.int64)
                if selected.ndim != 1 or np.any(selected < 0) or np.any(selected >= self.n_neurons):
                    raise ValueError("record_activity_indices must be valid neuron indices")
                empty_parts.append(np.empty((0, len(selected)), dtype=np.float32))
            return empty_parts[0] if len(empty_parts) == 1 else tuple(empty_parts)

        recorded_idx = None
        recorded_idx_tensor = None
        if record_activity_indices is not None:
            recorded_idx = np.asarray(record_activity_indices, dtype=np.int64)
            if recorded_idx.ndim != 1:
                raise ValueError("record_activity_indices must be one-dimensional")
            if np.any(recorded_idx < 0) or np.any(recorded_idx >= self.n_neurons):
                raise ValueError("record_activity_indices contains an out-of-range neuron")
            recorded_idx_tensor = torch.as_tensor(
                recorded_idx, dtype=torch.int64, device=self.device
            )

        gains = self._parameter_vector(gain, batch_size, "gain")
        leaks = self._parameter_vector(leak, batch_size, "leak")
        chunk_size = self.max_batch_size(
            steps=steps,
            n_sensory=n_sensory,
            n_recorded=0 if recorded_idx is None else len(recorded_idx),
        )
        motor_result = np.empty(
            (batch_size, steps, len(self.graph.motor_idx)), dtype=np.float32
        )
        final_result = (
            np.empty((batch_size, self.n_neurons), dtype=np.float32)
            if return_final_state
            else None
        )
        activity_mean_result = (
            np.empty((batch_size, len(recorded_idx)), dtype=np.float32)
            if recorded_idx is not None
            else None
        )

        for start in range(0, batch_size, chunk_size):
            stop = min(start + chunk_size, batch_size)
            count = stop - start
            current_chunk = np.ascontiguousarray(
                currents[start:stop].transpose(1, 2, 0), dtype=np.float32
            )
            current_tensor = torch.as_tensor(
                current_chunk, dtype=torch.float32, device=self.device
            )
            gain_tensor = torch.as_tensor(
                gains[start:stop], dtype=torch.float32, device=self.device
            ).view(1, count)
            leak_tensor = torch.as_tensor(
                leaks[start:stop], dtype=torch.float32, device=self.device
            ).view(1, count)

            state = torch.zeros(
                (self.n_neurons, count), dtype=torch.float32, device=self.device
            )
            motor_trace = torch.empty(
                (steps, len(self.graph.motor_idx), count),
                dtype=torch.float32,
                device=self.device,
            )
            activity_sum = (
                torch.zeros((count, len(recorded_idx)), dtype=torch.float32, device=self.device)
                if recorded_idx is not None
                else None
            )
            for step in range(steps):
                activity = torch.sparse.mm(self.weights, state)
                activity.mul_(gain_tensor)
                activity.index_add_(0, self.sensory_idx, current_tensor[step])
                state = (1.0 - leak_tensor) * state + leak_tensor * torch.tanh(activity)
                motor_trace[step].copy_(state.index_select(0, self.motor_idx))
                if activity_sum is not None:
                    assert recorded_idx_tensor is not None
                    activity_sum.add_(state.index_select(0, recorded_idx_tensor).T)

            motor_batch = motor_trace.permute(2, 0, 1).contiguous()
            if record_motor_window and record_motor_window > 1:
                width = int(record_motor_window)
                cumulative = torch.cat(
                    [
                        torch.zeros(
                            (count, 1, motor_batch.shape[2]),
                            dtype=torch.float32,
                            device=self.device,
                        ),
                        torch.cumsum(motor_batch, dim=1),
                    ],
                    dim=1,
                )
                high = torch.arange(1, steps + 1, device=self.device)
                low = torch.clamp(high - width, min=0)
                denominator = (high - low).view(1, steps, 1)
                motor_batch = (cumulative[:, high] - cumulative[:, low]) / denominator

            motor_result[start:stop] = motor_batch.cpu().numpy()
            if final_result is not None:
                final_result[start:stop] = state.T.contiguous().cpu().numpy()
            if activity_mean_result is not None:
                assert activity_sum is not None
                activity_mean_result[start:stop] = (
                    activity_sum.div_(steps).cpu().numpy()
                )

        self.peak_memory_bytes = max(
            self.peak_memory_bytes,
            int(torch.cuda.max_memory_allocated(self.device)),
        )
        if return_final_state:
            assert final_result is not None
            if activity_mean_result is not None:
                return motor_result, final_result, activity_mean_result
            return motor_result, final_result
        if activity_mean_result is not None:
            return motor_result, activity_mean_result
        return motor_result
