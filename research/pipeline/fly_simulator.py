"""Fly Simulator: runs Drosophila connectome reservoir simulations on market image inputs.

Contract (from research/SPEC.md & research.md §4 Step 3 & SPEC_v3.md §10-12):
- FlySimulator(graph: ConnectomeGraph, seed: int, steps=32, gain=1.0, leak=0.5, noise_std=0.0)
- .run(images: uint8 (B,64,64,3)) -> dict(buy_score=(B,), sell_score=(B,))
- Weights are fixed, non-learnable.
- 64x64x3 image -> sensory neuron mapping is fixed, non-learnable, independent of seed.
  (Seed only affects Gaussian noise injected during simulation when noise_std > 0).
- BUY and SELL are two disjoint subsets of graph.motor_idx (prioritizing graph.meta['buy_idx']
  and graph.meta['sell_idx'] when present, falling back to even split).
  score = total activity of that subset of neurons within the simulation window.
- direct_currents=True: skips random projection W_in; encoder output (or direct input)
  is already (B, n_sensory) sensory currents.
- Reuses src.network.reservoir.LeakyReservoir without modifying or reimplementing dynamics.
"""
from __future__ import annotations

import numpy as np

from src.connectome.schema import ConnectomeGraph
from src.network.reservoir import LeakyReservoir

# Global cache for fixed visual projection matrices (n_features, n_sensory) -> W_in
_PROJECTION_CACHE: dict[tuple[int, int], np.ndarray] = {}


def get_fixed_projection(n_features: int, n_sensory: int) -> np.ndarray:
    """Generate a fixed, non-learnable projection matrix from image pixels to sensory neurons.

    The projection is generated column-by-column using a fixed deterministic seed sequence,
    ensuring that sensory neuron k's projection is strictly independent of n_sensory and
    any simulator seed.
    """
    key = (n_features, n_sensory)
    if key not in _PROJECTION_CACHE:
        cols = [
            np.random.default_rng(42 + k * 10007).normal(
                0.0, 1.0 / np.sqrt(n_features), size=n_features
            )
            for k in range(n_sensory)
        ]
        _PROJECTION_CACHE[key] = np.column_stack(cols)
    return _PROJECTION_CACHE[key]


class FlySimulator:
    """Connectome reservoir simulator for market image inputs."""

    def __init__(
        self,
        graph: ConnectomeGraph,
        seed: int = 0,
        steps: int = 32,
        gain: float = 1.0,
        leak: float = 0.5,
        noise_std: float = 0.0,
        encoder=None,
        baseline: dict | None = None,
        direct_currents: bool = False,
    ):
        self.graph = graph
        self.seed = int(seed)
        self.steps = int(steps)
        self.gain = float(gain)
        self.leak = float(leak)
        self.noise_std = float(noise_std)
        # v2: encoder(images uint8 (B,64,64,3)) -> (B, F) float; baseline={buy_mean,buy_std,sell_mean,sell_std}
        self.encoder = encoder
        self.baseline = baseline
        self.direct_currents = bool(direct_currents)

        # Reuse existing LeakyReservoir directly
        self.reservoir = LeakyReservoir(graph)

        n_motor = len(graph.motor_idx)
        if n_motor < 2:
            raise ValueError(
                f"Graph must have at least 2 motor neurons for disjoint BUY/SELL sets, got {n_motor}"
            )

        # Disjoint partition of motor neurons into BUY and SELL subsets
        if "buy_idx" in graph.meta and "sell_idx" in graph.meta:
            self.buy_motor_idx = np.asarray(graph.meta["buy_idx"], dtype=np.int64)
            self.sell_motor_idx = np.asarray(graph.meta["sell_idx"], dtype=np.int64)
            motor_to_col = {idx: i for i, idx in enumerate(graph.motor_idx)}
            self._buy_cols = np.array(
                [motor_to_col[idx] for idx in self.buy_motor_idx if idx in motor_to_col],
                dtype=np.int64,
            )
            self._sell_cols = np.array(
                [motor_to_col[idx] for idx in self.sell_motor_idx if idx in motor_to_col],
                dtype=np.int64,
            )
            self._use_meta_motor = True
        else:
            mid = n_motor // 2
            self.buy_motor_idx = graph.motor_idx[:mid]
            self.sell_motor_idx = graph.motor_idx[mid:]
            self._buy_cols = np.arange(mid, dtype=np.int64)
            self._sell_cols = np.arange(mid, n_motor, dtype=np.int64)
            self._use_meta_motor = False
        self._mid = len(self.buy_motor_idx)

        # PRNG initialized with simulator seed (used strictly for noise when noise_std > 0)
        self._rng = np.random.default_rng(self.seed)

    def reset_rng(self) -> None:
        """Reset the internal RNG to the initial seed."""
        self._rng = np.random.default_rng(self.seed)

    def run(self, images: np.ndarray) -> dict[str, np.ndarray]:
        """Run reservoir simulation for a batch of 64x64x3 images or direct sensory currents.

        Parameters
        ----------
        images: np.ndarray
            Input images of shape (B, 64, 64, 3) or (64, 64, 3).
            Can be uint8 [0..255] or float.
            If direct_currents=True and encoder=None, can be sensory currents (B, n_sensory).

        Returns
        -------
        dict[str, np.ndarray]
            dict(buy_score=np.ndarray (B,), sell_score=np.ndarray (B,))
        """
        img_arr = np.asarray(images)

        if img_arr.size == 0 or img_arr.shape[0] == 0:
            return {
                "buy_score": np.zeros(0, dtype=np.float64),
                "sell_score": np.zeros(0, dtype=np.float64),
            }

        n_sensory = len(self.graph.sensory_idx)

        # Check direct_currents mode
        if self.direct_currents:
            if self.encoder is not None:
                # Encoder outputs (B, n_sensory) currents directly
                if img_arr.ndim == 3:
                    img_arr = img_arr[None, ...]
                currents_1d = self.encoder(img_arr)
            else:
                # Input is already sensory currents
                if img_arr.ndim == 1:
                    img_arr = img_arr[None, :]
                if img_arr.shape[1] != n_sensory:
                    raise ValueError(
                        f"Expected direct sensory currents with shape (*, {n_sensory}), got {img_arr.shape}"
                    )
                currents_1d = img_arr
            B = currents_1d.shape[0]
        else:
            if img_arr.ndim == 3:
                img_arr = img_arr[None, ...]

            if img_arr.shape[1:] != (64, 64, 3):
                raise ValueError(
                    f"Expected image shape (*, 64, 64, 3), got {img_arr.shape}"
                )

            B = img_arr.shape[0]

            # 1. Image normalization: center pixel values around zero
            if self.encoder is not None:
                norm_img = None
            elif img_arr.dtype == np.uint8:
                norm_img = (img_arr.astype(np.float64) - 128.0) / 128.0
            elif np.issubdtype(img_arr.dtype, np.floating):
                if img_arr.max() > 1.0:
                    norm_img = (img_arr.astype(np.float64) - 128.0) / 128.0
                else:
                    norm_img = img_arr.astype(np.float64) * 2.0 - 1.0
            else:
                norm_img = (img_arr.astype(np.float64) - 128.0) / 128.0

            flat_img = self.encoder(img_arr) if self.encoder is not None else norm_img.reshape(B, -1)  # (B, F)

            # 2. Fixed, non-learnable projection to sensory neurons
            n_features = flat_img.shape[1]
            W_in = get_fixed_projection(n_features, n_sensory)
            currents_1d = flat_img @ W_in  # (B, n_sensory)

        # 3. Temporal expansion over simulation window
        currents = np.repeat(currents_1d[:, None, :], self.steps, axis=1)  # (B, steps, n_sensory)

        # 4. Inject noise if noise_std > 0
        if self.noise_std > 0.0:
            noise = self._rng.normal(0.0, self.noise_std, size=currents.shape)
            currents = currents + noise

        # 5. Reservoir simulation via simulate_batch
        chunk_size = 500
        if B <= chunk_size:
            gain_arr = np.full(B, self.gain, dtype=np.float64)
            leak_arr = np.full(B, self.leak, dtype=np.float64)
            motor_trace = self.reservoir.simulate_batch(currents, gain_arr, leak_arr)
            buy_score = motor_trace[:, :, self._buy_cols].sum(axis=(1, 2))
            sell_score = motor_trace[:, :, self._sell_cols].sum(axis=(1, 2))
        else:
            buy_scores_list = []
            sell_scores_list = []
            for start_idx in range(0, B, chunk_size):
                end_idx = min(start_idx + chunk_size, B)
                c_chunk = currents[start_idx:end_idx]
                p_chunk = len(c_chunk)
                gain_chunk = np.full(p_chunk, self.gain, dtype=np.float64)
                leak_chunk = np.full(p_chunk, self.leak, dtype=np.float64)
                m_chunk = self.reservoir.simulate_batch(c_chunk, gain_chunk, leak_chunk)
                buy_scores_list.append(m_chunk[:, :, self._buy_cols].sum(axis=(1, 2)))
                sell_scores_list.append(m_chunk[:, :, self._sell_cols].sum(axis=(1, 2)))
            buy_score = np.concatenate(buy_scores_list, axis=0)
            sell_score = np.concatenate(sell_scores_list, axis=0)

        if self.baseline is not None:  # v2: 每神經元平均活動,以 Train 基準 z-score
            bl = self.baseline
            n_b = len(self.buy_motor_idx)
            n_s = len(self.sell_motor_idx)
            buy_score = (buy_score / (self.steps * n_b) - bl["buy_mean"]) / bl["buy_std"]
            sell_score = (sell_score / (self.steps * n_s) - bl["sell_mean"]) / bl["sell_std"]

        return {
            "buy_score": buy_score.astype(np.float64),
            "sell_score": sell_score.astype(np.float64),
        }
