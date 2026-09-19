"""Phase 0 Gate Suite S1--S5.

Implements SPEC_v3 Phase 0:
- S1 Saturation: Non-sensory neurons |x| > 0.9 ratio < 5% (Train 300 images, final step)
- S2 No Collapse: Val 500 images margin all finite, std > 0, minority action ratio >= 0.05
- S3 Semantic Smoothness: Adjacent window (shifted 1 bar) margin corr >= 0.5;
  random independent pair corr < 0.2. Also reports 1% pixel noise delta/SD & corr.
  Uses real Val samples (never reading label/future_return; never touching test split).
- S4 Positive/Negative Controls: Ridge regression with 5-fold CV on DN last-step activity
  - PC-1: trend slope k: sign BA >= 0.95, slope R2 >= 0.5 (zero-shot Spearman reported)
  - PC-2: vol level: R2 >= 0.5
  - PC-3: AR(1) phi=0.8 direction BA >= 0.60
  - NC-1: GBM random walk BA in [0.49, 0.51]
- S5 SNR: 200 Val inputs x 10 repeats with 5% sensory noise, between/within ratio >= 3.0

Outputs:
- outputs/v3/dynamics_remote.json
- outputs/v3/phase0_report_remote.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.linalg import eigs
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent))

from research.pipeline.encoding_v2 import calibrate_baseline, encoder
from research.pipeline.fly_simulator import FlySimulator, get_fixed_projection
from research.pipeline.render_market import render
from research.pipeline.synthetic_prices import (
    generate_phase0_synthetic_suite,
)
from src.connectome.synthetic import make_synthetic_connectome

OUT_V3 = ROOT / "outputs/v3"


# ---------------------------------------------------------------------------
# Pure NumPy Ridge Regression with 5-Fold Cross Validation
# ---------------------------------------------------------------------------
class PureNumpyRidge:
    """Pure NumPy Ridge regression with z-scored X and centered y."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)
        self.mu_X: np.ndarray | None = None
        self.std_X: np.ndarray | None = None
        self.mu_y: float = 0.0
        self.beta: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> PureNumpyRidge:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.mu_X = np.mean(X, axis=0)
        self.std_X = np.std(X, axis=0) + 1e-8
        X_tilde = (X - self.mu_X) / self.std_X
        self.mu_y = float(np.mean(y))
        y_tilde = y - self.mu_y

        N, D = X_tilde.shape
        if N >= D:
            A = X_tilde.T @ X_tilde + self.alpha * np.eye(D)
            self.beta = np.linalg.solve(A, X_tilde.T @ y_tilde)
        else:
            A = X_tilde @ X_tilde.T + self.alpha * np.eye(N)
            self.beta = X_tilde.T @ np.linalg.solve(A, y_tilde)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.beta is None or self.mu_X is None or self.std_X is None:
            raise RuntimeError("Model is not fitted")
        X = np.asarray(X, dtype=np.float64)
        X_tilde = (X - self.mu_X) / self.std_X
        return X_tilde @ self.beta + self.mu_y


def ridge_cv_fit(
    X_train: np.ndarray,
    y_train: np.ndarray,
    alphas: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0),
    n_folds: int = 5,
    seed: int = 0,
) -> tuple[PureNumpyRidge, float]:
    """Select best alpha via n-fold CV on train set, then fit on all train data."""
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    N = len(X_train)

    rng = np.random.default_rng(seed)
    indices = rng.permutation(N)
    fold_sizes = np.full(n_folds, N // n_folds, dtype=int)
    fold_sizes[: N % n_folds] += 1
    folds: list[np.ndarray] = []
    current = 0
    for fs in fold_sizes:
        folds.append(indices[current : current + fs])
        current += fs

    best_alpha = alphas[0]
    best_cv_mse = float("inf")

    for alpha in alphas:
        fold_mses = []
        for f in range(n_folds):
            val_idx = folds[f]
            tr_idx = np.setdiff1d(indices, val_idx)
            model = PureNumpyRidge(alpha=alpha).fit(X_train[tr_idx], y_train[tr_idx])
            pred_val = model.predict(X_train[val_idx])
            mse = float(np.mean((y_train[val_idx] - pred_val) ** 2))
            fold_mses.append(mse)
        mean_mse = float(np.mean(fold_mses))
        if mean_mse < best_cv_mse:
            best_cv_mse = mean_mse
            best_alpha = alpha

    final_model = PureNumpyRidge(alpha=best_alpha).fit(X_train, y_train)
    return final_model, best_alpha


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def balanced_accuracy_score(y_true: np.ndarray, y_pred_binary: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred_binary, dtype=int)
    c1 = (y_true == 1)
    c0 = (y_true == 0)
    rec1 = float((y_pred[c1] == 1).mean()) if np.any(c1) else 0.5
    rec0 = float((y_pred[c0] == 0).mean()) if np.any(c0) else 0.5
    return 0.5 * (rec1 + rec0)


# ---------------------------------------------------------------------------
# Graph & Simulation Wrappers (Defensive against FlyWire availability)
# ---------------------------------------------------------------------------
def compute_spectral_radius(W) -> float:
    """Compute maximum absolute eigenvalue of sparse matrix W."""
    try:
        evals = eigs(W.astype(float), k=6, which="LM", return_eigenvectors=False)
        return float(np.max(np.abs(evals)))
    except Exception:
        # Fallback approximation via power iteration
        v = np.random.default_rng(42).normal(0, 1, W.shape[0])
        v /= np.linalg.norm(v) + 1e-12
        for _ in range(50):
            w = W.dot(v)
            norm = np.linalg.norm(w)
            if norm == 0:
                return 0.0
            v = w / norm
        return float(norm)


def bisection_find_gsat(
    W,
    sensory_idx: np.ndarray,
    currents_train_300: np.ndarray,
    gain_spectral: float = 1.0,
    leak: float = 0.5,
    steps: int = 32,
    sat_thresh: float = 0.9,
    max_sat_ratio: float = 0.05,
    n_iter: int = 6,
    backend: str = "scipy",
    graph=None,
) -> float:
    """Find largest gain in [gain_spectral, 10.0] such that non-sensory |x| > 0.9 ratio < 5%."""
    n_neurons = W.shape[0]
    # Use up to 50 samples for bisection to avoid CPU bottleneck
    sub_currents = currents_train_300[: min(len(currents_train_300), 50)]
    B = sub_currents.shape[0]
    non_sensory = np.setdiff1d(np.arange(n_neurons), sensory_idx)
    torch_reservoir = None
    if backend == "torch":
        if graph is None:
            raise ValueError("graph is required for torch gain calibration")
        from research.pipeline.torch_sim import TorchReservoir

        torch_reservoir = TorchReservoir(graph)
    elif backend != "scipy":
        raise ValueError(f"unknown backend {backend!r}")
    cur_t = (
        np.ascontiguousarray(sub_currents.transpose(1, 2, 0))
        if torch_reservoir is None
        else None
    )  # (steps, n_s, B)

    def eval_gain(g: float) -> float:
        if torch_reservoir is not None:
            _, final_state = torch_reservoir.simulate_batch(
                sub_currents,
                np.full(B, g, dtype=np.float32),
                np.full(B, leak, dtype=np.float32),
                return_final_state=True,
            )
            return float(np.mean(np.abs(final_state[:, non_sensory]) > sat_thresh))
        X = np.zeros((n_neurons, B), dtype=np.float64)
        for t in range(steps):
            Z = g * W.dot(X)
            assert cur_t is not None
            Z[sensory_idx] += cur_t[t]
            X = (1.0 - leak) * X + leak * np.tanh(Z)
        return float(np.mean(np.abs(X[non_sensory]) > sat_thresh))

    low = min(gain_spectral, 0.001)
    high = 1.0
    if eval_gain(high) < max_sat_ratio:
        return high

    for _ in range(n_iter):
        mid = (low + high) / 2.0
        if eval_gain(mid) < max_sat_ratio:
            low = mid
        else:
            high = mid
    return float(low)


def extract_dn_activity(
    sim: FlySimulator,
    graph,
    images: np.ndarray,
    gain: float,
    leak: float = 0.5,
    retina=None,
    chunk_size: int = 500,
) -> np.ndarray:
    """Extract motor neurons (DN) final step activity across images in chunks."""
    B = len(images)
    features_list: list[np.ndarray] = []

    for start in range(0, B, chunk_size):
        end = min(start + chunk_size, B)
        sub_imgs = images[start:end]
        p_chunk = len(sub_imgs)

        # 1. Currents
        if retina is not None:
            cur_1d = retina(sub_imgs)  # (p_chunk, n_sensory)
        elif sim.encoder is not None:
            flat = sim.encoder(sub_imgs)
            W_in = get_fixed_projection(flat.shape[1], len(graph.sensory_idx))
            cur_1d = flat @ W_in
        else:
            norm_img = (sub_imgs.astype(np.float64) - 128.0) / 128.0
            flat = norm_img.reshape(p_chunk, -1)
            W_in = get_fixed_projection(flat.shape[1], len(graph.sensory_idx))
            cur_1d = flat @ W_in

        currents = np.repeat(cur_1d[:, None, :], sim.steps, axis=1)  # (p_chunk, steps, n_s)

        gain_arr = np.full(p_chunk, gain, dtype=np.float64)
        leak_arr = np.full(p_chunk, leak, dtype=np.float64)

        # Simulate batch on the selected backend.
        if sim.backend == "torch":
            assert sim.torch_reservoir is not None
            motor_trace = sim.torch_reservoir.simulate_batch(
                currents, gain_arr, leak_arr
            )
        else:
            motor_trace = sim.reservoir.simulate_batch(currents, gain_arr, leak_arr)
        dn_last = motor_trace[:, -1, :]  # (p_chunk, n_motor)
        features_list.append(dn_last)

    return np.concatenate(features_list, axis=0)


# ---------------------------------------------------------------------------
# Phase 0 Runner
# ---------------------------------------------------------------------------
def run_phase0(
    graph_type: str = "synthetic",
    n_train_s1: int = 300,
    n_val_s2: int = 500,
    n_val_s3: int = 200,
    n_synthetic_s4: int = 500,
    n_nc1_test: int = 2000,
    n_val_s5: int = 200,
    seed: int = 0,
    backend: str = "scipy",
) -> dict:
    if backend not in ("scipy", "torch"):
        raise ValueError(f"backend must be 'scipy' or 'torch', got {backend!r}")
    if backend == "torch":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable for --backend torch")
        torch.cuda.reset_peak_memory_stats()

    OUT_V3.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    # 1. Load data safely (label-free; NEVER touch label/future_return or test split)
    samples = pd.read_parquet(
        ROOT / "data/samples.parquet",
        columns=["sample_id", "timestamp", "image_idx", "split"],
    )
    raw = pd.read_parquet(ROOT / "data/raw_ohlcv.parquet")
    images = np.load(ROOT / "data/images.npy", mmap_mode="r")

    tr_samples = samples[samples.split == "train"].sort_values("sample_id")
    val_samples = samples[samples.split == "val"].sort_values("sample_id")

    # Train mean image for mean_sub encoder
    mean_img_path = OUT_V3 / "train_mean_image.npy"
    if not mean_img_path.exists():
        tr_indices = tr_samples.image_idx.to_numpy()[::10]
        train_mean_img = np.asarray(images[tr_indices]).astype(np.float64).mean(axis=0)
        np.save(mean_img_path, train_mean_img)
    else:
        train_mean_img = np.load(mean_img_path)

    # 2. Setup Graph & Encoder
    retina = None
    if graph_type == "flywire":
        try:
            from research.pipeline.flywire_graph import load_flywire_graph
            from research.pipeline.retina import retina_encoder

            cache_path = ROOT / "data/flywire/graph_cache.npz"
            if not cache_path.exists():
                raise FileNotFoundError(f"FlyWire cache not found at {cache_path}")
            g = load_flywire_graph(cache_path=cache_path)
            retina = retina_encoder(g, train_mean_img)
            enc = None
        except Exception as e:
            raise RuntimeError(f"FlyWire graph or retina loading failed: {e}")
    elif graph_type == "synthetic":
        g = make_synthetic_connectome(
            n_neurons=4000,
            avg_out_degree=8.0,
            long_range_fraction=0.05,
            frac_inhibitory=0.2,
            n_sensory=48,
            n_motor=12,
            seed=seed,
        )
        enc = encoder("mean_sub", train_mean_img)
    else:
        raise ValueError(f"Unknown graph_type: {graph_type}")

    # 3. Compute Dynamics & Determine gain
    rho = compute_spectral_radius(g.weights)
    gain_spectral = 0.95 / rho if rho > 0 else 1.0

    # Train 300 images for S1 and g_sat bisection
    s1_sample_indices = tr_samples.image_idx.to_numpy()[:n_train_s1]
    s1_images = np.asarray(images[s1_sample_indices])

    if retina is not None:
        cur_300 = retina(s1_images)
    else:
        flat = enc(s1_images)
        W_in = get_fixed_projection(flat.shape[1], len(g.sensory_idx))
        cur_300 = flat @ W_in

    currents_train_300 = np.repeat(cur_300[:, None, :], 32, axis=1)

    g_sat = bisection_find_gsat(
        g.weights,
        g.sensory_idx,
        currents_train_300,
        gain_spectral=gain_spectral,
        leak=0.5,
        steps=32,
        backend=backend,
        graph=g,
    )
    chosen_gain = float(min(gain_spectral, g_sat))

    # Sensory current std on Train for noise_std
    sensory_std = float(cur_300.std())
    noise_std_5pct = float(0.05 * sensory_std)

    dynamics_info = {
        "graph": graph_type,
        "spectral_radius": float(rho),
        "gain_spectral": float(gain_spectral),
        "g_sat": float(g_sat),
        "gain": chosen_gain,
        "leak": 0.5,
        "steps": 32,
        "sensory_currents_std": sensory_std,
        "noise_std_robustness": noise_std_5pct,
        "n_neurons": int(g.n_neurons),
        "n_sensory": int(len(g.sensory_idx)),
        "n_motor": int(len(g.motor_idx)),
    }
    active_enc = retina if retina is not None else enc
    use_direct = bool(graph_type == "flywire")

    # Baseline calibration on Train (200 for flywire to avoid CPU latency, 1000 for synthetic)
    n_cal = 200 if graph_type == "flywire" else 1000
    cal_indices = tr_samples.image_idx.to_numpy()[:: max(len(tr_samples) // n_cal, 1)][:n_cal]
    cal_imgs = np.asarray(images[cal_indices])
    bl = calibrate_baseline(
        lambda _: FlySimulator(
            g,
            seed=0,
            steps=32,
            gain=chosen_gain,
            leak=0.5,
            noise_std=0.0,
            encoder=active_enc,
            baseline=None,
            direct_currents=use_direct,
            backend=backend,
        ),
        cal_imgs,
    )

    sim_det = FlySimulator(
        g,
        seed=0,
        steps=32,
        gain=chosen_gain,
        leak=0.5,
        noise_std=0.0,
        encoder=active_enc,
        baseline=bl,
        direct_currents=use_direct,
        backend=backend,
    )

    # -----------------------------------------------------------------------
    # Gate S1: Saturation (< 5%)
    # -----------------------------------------------------------------------
    print(f"[1/5] Running S1 (Saturation) on {len(s1_images)} Train images...", flush=True)
    non_sensory = np.setdiff1d(np.arange(g.n_neurons), g.sensory_idx)
    if backend == "torch":
        assert sim_det.torch_reservoir is not None
        _, final_state = sim_det.torch_reservoir.simulate_batch(
            currents_train_300,
            np.full(len(s1_images), chosen_gain, dtype=np.float32),
            np.full(len(s1_images), 0.5, dtype=np.float32),
            return_final_state=True,
        )
        sat_ratio = float(np.mean(np.abs(final_state[:, non_sensory]) > 0.9))
    else:
        cur_t = np.ascontiguousarray(currents_train_300.transpose(1, 2, 0))
        X_s1 = np.zeros((g.n_neurons, len(s1_images)), dtype=np.float64)
        for t in range(32):
            Z = chosen_gain * g.weights.dot(X_s1)
            Z[g.sensory_idx] += cur_t[t]
            X_s1 = 0.5 * X_s1 + 0.5 * np.tanh(Z)
        sat_ratio = float(np.mean(np.abs(X_s1[non_sensory]) > 0.9))
    s1_pass = bool(sat_ratio < 0.05)
    print(f"[1/5] S1 Result: sat_ratio={sat_ratio:.4f} (pass={s1_pass})", flush=True)

    gate_s1 = {
        "passed": s1_pass,
        "sat_ratio": sat_ratio,
        "threshold": 0.05,
        "n_samples": len(s1_images),
    }

    # -----------------------------------------------------------------------
    # Gate S2: No Collapse (Val 500 images)
    # -----------------------------------------------------------------------
    print(f"[2/5] Running S2 (No Collapse) on {n_val_s2} Val images...", flush=True)
    s2_indices = val_samples.image_idx.to_numpy()[:n_val_s2]
    s2_imgs = np.asarray(images[s2_indices])
    out_s2 = sim_det.run(s2_imgs)
    margin_s2 = out_s2["buy_score"] - out_s2["sell_score"]
    finite_nonconst = bool(np.isfinite(margin_s2).all() and margin_s2.std() > 0)
    minority_ratio = float(min((margin_s2 > 0).mean(), (margin_s2 <= 0).mean()))
    s2_pass = bool(finite_nonconst and minority_ratio >= 0.05)
    print(f"[2/5] S2 Result: minority_ratio={minority_ratio:.4f}, margin_std={margin_s2.std():.4f} (pass={s2_pass})", flush=True)

    gate_s2 = {
        "passed": s2_pass,
        "finite_nonconst": finite_nonconst,
        "minority_action_ratio": minority_ratio,
        "margin_std": float(margin_s2.std()),
        "threshold": 0.05,
    }

    # -----------------------------------------------------------------------
    # Gate S3: Semantic Smoothness (Val Adjacent Window & Baseline)
    # -----------------------------------------------------------------------
    s3_val = val_samples.iloc[:n_val_s3]
    print(f"[3/5] Running S3 (Semantic Smoothness) on {len(s3_val)} Val window pairs...", flush=True)
    end_indices = raw.timestamp.searchsorted(s3_val.timestamp - pd.Timedelta(minutes=5))

    wins_t = [
        raw.iloc[e - 48 + 1 : e + 1][["open", "high", "low", "close", "volume"]].to_numpy()
        for e in end_indices
    ]
    wins_t1 = [
        raw.iloc[e - 48 + 2 : e + 2][["open", "high", "low", "close", "volume"]].to_numpy()
        for e in end_indices
    ]

    imgs_t = np.stack([render(w) for w in wins_t])
    imgs_t1 = np.stack([render(w) for w in wins_t1])

    out_t = sim_det.run(imgs_t)
    out_t1 = sim_det.run(imgs_t1)
    m_t = out_t["buy_score"] - out_t["sell_score"]
    m_t1 = out_t1["buy_score"] - out_t1["sell_score"]

    corr_adjacent = float(np.corrcoef(m_t, m_t1)[0, 1])

    # Random independent pair correlation baseline
    perm = np.random.default_rng(1234).permutation(len(m_t))
    corr_independent = float(np.corrcoef(m_t, m_t[perm])[0, 1])

    # 1% pixel noise report
    rng_noise = np.random.default_rng(5678)
    imgs_noise = np.clip(
        imgs_t.astype(np.float64) + rng_noise.normal(0.0, 0.01 * 255.0, size=imgs_t.shape),
        0.0,
        255.0,
    ).astype(np.uint8)
    out_noise = sim_det.run(imgs_noise)
    m_noise = out_noise["buy_score"] - out_noise["sell_score"]
    corr_noise = float(np.corrcoef(m_t, m_noise)[0, 1])
    delta_sd_noise = float(np.mean(np.abs(m_noise - m_t)) / (m_t.std() + 1e-12))

    s3_pass = bool(corr_adjacent >= 0.5 and corr_independent < 0.2)
    print(f"[3/5] S3 Result: corr_adjacent={corr_adjacent:.4f}, corr_independent={corr_independent:.4f} (pass={s3_pass})", flush=True)

    gate_s3 = {
        "passed": s3_pass,
        "corr_adjacent": corr_adjacent,
        "corr_independent": corr_independent,
        "corr_adjacent_threshold": 0.5,
        "corr_independent_threshold": 0.2,
        "pixel_noise_1pct": {
            "corr": corr_noise,
            "delta_over_sd": delta_sd_noise,
        },
    }

    # -----------------------------------------------------------------------
    # Gate S4: Positive and Negative Controls
    # -----------------------------------------------------------------------
    print(f"[4/5] Running S4 Controls (n_synth={n_synthetic_s4}, n_nc1_te={n_nc1_test})...", flush=True)
    train_synth = generate_phase0_synthetic_suite(
        split="train", n_samples=n_synthetic_s4, nc1_samples=n_synthetic_s4
    )
    test_synth = generate_phase0_synthetic_suite(
        split="test", n_samples=n_synthetic_s4, nc1_samples=n_nc1_test
    )

    # PC-1: Trend
    print("  [4/5] [PC-1 Trend] Rendering & simulating...", flush=True)
    imgs_tr_pc1 = np.stack([render(w) for w in train_synth["pc1_trend"][0]])
    imgs_te_pc1 = np.stack([render(w) for w in test_synth["pc1_trend"][0]])
    X_tr_pc1 = extract_dn_activity(sim_det, g, imgs_tr_pc1, chosen_gain, retina=retina)
    X_te_pc1 = extract_dn_activity(sim_det, g, imgs_te_pc1, chosen_gain, retina=retina)

    y_tr_k = train_synth["pc1_trend"][1]["k"]
    y_te_k = test_synth["pc1_trend"][1]["k"]
    model_pc1, alpha_pc1 = ridge_cv_fit(X_tr_pc1, y_tr_k)
    pred_k = model_pc1.predict(X_te_pc1)
    r2_pc1 = r2_score(y_te_k, pred_k)

    sign_true = (y_te_k >= 0.0).astype(int)
    sign_pred = (pred_k >= 0.0).astype(int)
    ba_pc1 = balanced_accuracy_score(sign_true, sign_pred)

    # Zero-shot margin Spearman correlation on PC-1 (report only)
    out_pc1_te = sim_det.run(imgs_te_pc1)
    m_pc1 = out_pc1_te["buy_score"] - out_pc1_te["sell_score"]
    spearman_pc1 = float(spearmanr(m_pc1, y_te_k).statistic)

    pc1_pass = bool(ba_pc1 >= 0.95 and r2_pc1 >= 0.5)
    print(f"  [4/5] [PC-1 Trend] BA={ba_pc1:.4f}, R2={r2_pc1:.4f}, Spearman={spearman_pc1:.4f} (pass={pc1_pass})", flush=True)

    # PC-2: Vol
    print("  [4/5] [PC-2 Vol] Rendering & simulating...", flush=True)
    imgs_tr_pc2 = np.stack([render(w) for w in train_synth["pc2_vol"][0]])
    imgs_te_pc2 = np.stack([render(w) for w in test_synth["pc2_vol"][0]])
    X_tr_pc2 = extract_dn_activity(sim_det, g, imgs_tr_pc2, chosen_gain, retina=retina)
    X_te_pc2 = extract_dn_activity(sim_det, g, imgs_te_pc2, chosen_gain, retina=retina)

    y_tr_vol = train_synth["pc2_vol"][1]["vol"]
    y_te_vol = test_synth["pc2_vol"][1]["vol"]
    model_pc2, alpha_pc2 = ridge_cv_fit(X_tr_pc2, y_tr_vol)
    pred_vol = model_pc2.predict(X_te_pc2)
    r2_pc2 = r2_score(y_te_vol, pred_vol)
    pc2_pass = bool(r2_pc2 >= 0.5)
    print(f"  [4/5] [PC-2 Vol] R2={r2_pc2:.4f} (pass={pc2_pass})", flush=True)

    # PC-3: AR(1)
    print("  [4/5] [PC-3 AR(1)] Rendering & simulating...", flush=True)
    imgs_tr_pc3 = np.stack([render(w) for w in train_synth["pc3_ar1"][0]])
    imgs_te_pc3 = np.stack([render(w) for w in test_synth["pc3_ar1"][0]])
    X_tr_pc3 = extract_dn_activity(sim_det, g, imgs_tr_pc3, chosen_gain, retina=retina)
    X_te_pc3 = extract_dn_activity(sim_det, g, imgs_te_pc3, chosen_gain, retina=retina)

    y_tr_ret = train_synth["pc3_ar1"][1]["future_return"]
    y_te_ret = test_synth["pc3_ar1"][1]["future_return"]
    model_pc3, alpha_pc3 = ridge_cv_fit(X_tr_pc3, y_tr_ret)
    pred_ret_ar1 = model_pc3.predict(X_te_pc3)
    dir_true_ar1 = (y_te_ret > 0.0).astype(int)
    dir_pred_ar1 = (pred_ret_ar1 > 0.0).astype(int)
    ba_pc3 = balanced_accuracy_score(dir_true_ar1, dir_pred_ar1)
    pc3_pass = bool(ba_pc3 >= 0.60)
    print(f"  [4/5] [PC-3 AR(1)] BA={ba_pc3:.4f} (pass={pc3_pass})", flush=True)

    # NC-1: GBM
    print("  [4/5] [NC-1 GBM] Rendering & simulating...", flush=True)
    imgs_tr_nc1 = np.stack([render(w) for w in train_synth["nc1_gbm"][0]])
    imgs_te_nc1 = np.stack([render(w) for w in test_synth["nc1_gbm"][0]])
    X_tr_nc1 = extract_dn_activity(sim_det, g, imgs_tr_nc1, chosen_gain, retina=retina)
    X_te_nc1 = extract_dn_activity(sim_det, g, imgs_te_nc1, chosen_gain, retina=retina)

    y_tr_gbm = train_synth["nc1_gbm"][1]["future_return"]
    y_te_gbm = test_synth["nc1_gbm"][1]["future_return"]
    model_nc1, alpha_nc1 = ridge_cv_fit(X_tr_nc1, y_tr_gbm)
    pred_ret_gbm = model_nc1.predict(X_te_nc1)
    dir_true_gbm = (y_te_gbm > 0.0).astype(int)
    dir_pred_gbm = (pred_ret_gbm > 0.0).astype(int)
    ba_nc1 = balanced_accuracy_score(dir_true_gbm, dir_pred_gbm)
    # NC-1: GBM (SPEC_v3 Amendment 1: 95% binomial interval 0.5 +/- 1.96*0.5/sqrt(n_test))
    n_te_nc1 = len(y_te_gbm)
    half_width_nc1 = float(1.96 * 0.5 / np.sqrt(n_te_nc1))
    nc1_low = float(0.5 - half_width_nc1)
    nc1_high = float(0.5 + half_width_nc1)
    nc1_pass = bool(nc1_low <= ba_nc1 <= nc1_high)
    print(f"  [4/5] [NC-1 GBM] BA={ba_nc1:.4f} in [{nc1_low:.4f}, {nc1_high:.4f}] (pass={nc1_pass})", flush=True)

    s4_pass = bool(pc1_pass and pc2_pass and pc3_pass and nc1_pass)
    print(f"[4/5] S4 Result: PC1={pc1_pass}, PC2={pc2_pass}, PC3={pc3_pass}, NC1={nc1_pass} (pass={s4_pass})", flush=True)

    gate_s4 = {
        "passed": s4_pass,
        "PC1_trend": {
            "passed": pc1_pass,
            "sign_balanced_accuracy": ba_pc1,
            "slope_r2": r2_pc1,
            "sign_ba_threshold": 0.95,
            "r2_threshold": 0.5,
            "zero_shot_margin_spearman": spearman_pc1,
            "best_alpha": alpha_pc1,
        },
        "PC2_vol": {
            "passed": pc2_pass,
            "r2": r2_pc2,
            "r2_threshold": 0.5,
            "best_alpha": alpha_pc2,
        },
        "PC3_ar1": {
            "passed": pc3_pass,
            "direction_balanced_accuracy": ba_pc3,
            "threshold": 0.60,
            "best_alpha": alpha_pc3,
        },
        "NC1_gbm": {
            "passed": nc1_pass,
            "direction_balanced_accuracy": ba_nc1,
            "target_interval": [nc1_low, nc1_high],
            "n_test_samples": n_te_nc1,
            "best_alpha": alpha_nc1,
            "amendment": "SPEC_v3 Amendment 1 (0.5 +/- 1.96*0.5/sqrt(n_test))",
        },
    }

    # -----------------------------------------------------------------------
    # Gate S5: Signal to Noise Ratio (Between / Within >= 3.0)
    # -----------------------------------------------------------------------
    repeats = 10
    print(f"[5/5] Running S5 (SNR) on {n_val_s5} Val images x {repeats} repeats...", flush=True)
    s5_indices = val_samples.image_idx.to_numpy()[:n_val_s5]
    s5_imgs = np.asarray(images[s5_indices])

    rep_margins = []
    for r in range(repeats):
        sim_noisy = FlySimulator(
            g,
            seed=20000 + r,
            steps=32,
            gain=chosen_gain,
            leak=0.5,
            noise_std=noise_std_5pct,
            encoder=active_enc,
            baseline=bl,
            direct_currents=use_direct,
            backend=backend,
        )
        out_r = sim_noisy.run(s5_imgs)
        rep_margins.append(out_r["buy_score"] - out_r["sell_score"])
        print(f"  [5/5] Repeat {r+1}/{repeats} completed", flush=True)

    rep_mat = np.array(rep_margins).T  # (n_inputs, 10)
    between_var = float(np.var(rep_mat.mean(axis=1), ddof=1))
    within_var = float(np.var(rep_mat, axis=1, ddof=1).mean())
    bw_ratio = float(between_var / within_var) if within_var > 0 else float("inf")
    s5_pass = bool(bw_ratio >= 3.0)
    print(f"[5/5] S5 Result: between/within={bw_ratio:.4f} (pass={s5_pass})", flush=True)

    gate_s5 = {
        "passed": s5_pass,
        "between_within_variance_ratio": bw_ratio,
        "threshold": 3.0,
        "between_var": between_var,
        "within_var": within_var,
        "noise_std": noise_std_5pct,
        "repeats": repeats,
    }

    # -----------------------------------------------------------------------
    # Final Decision (SPEC_v3: S1-S5 all pass -> Go, else Stop)
    # -----------------------------------------------------------------------
    all_passed = bool(s1_pass and s2_pass and s3_pass and s4_pass and s5_pass)
    decision = "GO" if all_passed else "STOP"

    reason_list = []
    if not s1_pass:
        reason_list.append(f"S1 failed: sat_ratio={sat_ratio:.4f} >= 0.05")
    if not s2_pass:
        reason_list.append(f"S2 failed: minority_ratio={minority_ratio:.4f} < 0.05 or non-finite")
    if not s3_pass:
        reason_list.append(f"S3 failed: corr_adj={corr_adjacent:.4f} < 0.5 or corr_indep={corr_independent:.4f} >= 0.2")
    if not s4_pass:
        reason_list.append("S4 failed: pipeline cannot carry structured information")
    if not s5_pass:
        reason_list.append(f"S5 failed: between/within={bw_ratio:.4f} < 3.0")

    if backend == "torch":
        import torch

        torch.cuda.synchronize()
        dynamics_info["torch_peak_memory_allocated_bytes"] = int(
            torch.cuda.max_memory_allocated()
        )
        dynamics_info["torch_peak_memory_reserved_bytes"] = int(
            torch.cuda.max_memory_reserved()
        )
        dynamics_info["torch_total_vram_bytes"] = int(
            torch.cuda.get_device_properties(0).total_memory
        )

    dt = time.perf_counter() - t_start

    report = {
        "graph": graph_type,
        "backend": backend,
        "decision": decision,
        "all_passed": all_passed,
        "reasons": reason_list,
        "runtime_seconds": float(dt),
        "gates": {
            "S1_saturation": gate_s1,
            "S2_no_collapse": gate_s2,
            "S3_semantic_smoothness": gate_s3,
            "S4_positive_negative_controls": gate_s4,
            "S5_snr": gate_s5,
        },
        "dynamics": dynamics_info,
    }

    (OUT_V3 / "dynamics_remote.json").write_text(
        json.dumps(dynamics_info, indent=2)
    )
    (OUT_V3 / "phase0_report_remote.json").write_text(
        json.dumps(report, indent=2)
    )
    return report


def print_summary(report: dict) -> None:
    print("=" * 60)
    print(f" Phase 0 Evaluation Report [{report['graph'].upper()}]")
    print("=" * 60)
    print(f"Decision:       {report['decision']}")
    print(f"Runtime:        {report['runtime_seconds']:.2f}s")
    if report["reasons"]:
        print("Failure causes:")
        for r in report["reasons"]:
            print(f"  - {r}")

    g = report["gates"]
    print("\n--- Gate Status ---")
    print(f"S1 Saturation:       {'PASS' if g['S1_saturation']['passed'] else 'FAIL'} (sat_ratio: {g['S1_saturation']['sat_ratio']:.4f} < 0.05)")
    print(f"S2 No Collapse:      {'PASS' if g['S2_no_collapse']['passed'] else 'FAIL'} (minority: {g['S2_no_collapse']['minority_action_ratio']:.4f} >= 0.05, std: {g['S2_no_collapse']['margin_std']:.4f})")
    print(f"S3 Smoothness:       {'PASS' if g['S3_semantic_smoothness']['passed'] else 'FAIL'} (adj_corr: {g['S3_semantic_smoothness']['corr_adjacent']:.4f} >= 0.5, indep: {g['S3_semantic_smoothness']['corr_independent']:.4f} < 0.2)")
    print(f"  [1% noise report]: corr: {g['S3_semantic_smoothness']['pixel_noise_1pct']['corr']:.4f}, delta/sd: {g['S3_semantic_smoothness']['pixel_noise_1pct']['delta_over_sd']:.4f}")

    s4 = g["S4_positive_negative_controls"]
    print(f"S4 Controls:         {'PASS' if s4['passed'] else 'FAIL'}")
    print(f"  PC-1 Trend:        {'PASS' if s4['PC1_trend']['passed'] else 'FAIL'} (BA: {s4['PC1_trend']['sign_balanced_accuracy']:.4f} >= 0.95, R2: {s4['PC1_trend']['slope_r2']:.4f} >= 0.5, margin_spearman: {s4['PC1_trend']['zero_shot_margin_spearman']:.4f})")
    print(f"  PC-2 Vol:          {'PASS' if s4['PC2_vol']['passed'] else 'FAIL'} (R2: {s4['PC2_vol']['r2']:.4f} >= 0.5)")
    print(f"  PC-3 AR(1):        {'PASS' if s4['PC3_ar1']['passed'] else 'FAIL'} (BA: {s4['PC3_ar1']['direction_balanced_accuracy']:.4f} >= 0.60)")
    nc1_int = s4['NC1_gbm']['target_interval']
    print(f"  NC-1 GBM:          {'PASS' if s4['NC1_gbm']['passed'] else 'FAIL'} (BA: {s4['NC1_gbm']['direction_balanced_accuracy']:.4f} in [{nc1_int[0]:.4f}, {nc1_int[1]:.4f}] [Amendment 1], n={s4['NC1_gbm']['n_test_samples']})")

    s5 = g["S5_snr"]
    print(f"S5 SNR:              {'PASS' if s5['passed'] else 'FAIL'} (between/within: {s5['between_within_variance_ratio']:.2f} >= 3.0)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run Phase 0 gates S1--S5.")
    parser.add_argument(
        "--graph",
        choices=["synthetic", "flywire"],
        default="synthetic",
        help="Connectome graph type to evaluate",
    )
    parser.add_argument(
        "--backend",
        choices=["scipy", "torch"],
        default="scipy",
        help="Reservoir backend (torch requires a CUDA GPU)",
    )
    parser.add_argument("--n-train-s1", type=int, default=300)
    parser.add_argument("--n-val-s2", type=int, default=500)
    parser.add_argument("--n-val-s3", type=int, default=200)
    parser.add_argument("--n-synthetic-s4", type=int, default=500)
    parser.add_argument("--n-nc1-test", type=int, default=2000)
    parser.add_argument("--n-val-s5", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    report = run_phase0(
        graph_type=args.graph,
        n_train_s1=args.n_train_s1,
        n_val_s2=args.n_val_s2,
        n_val_s3=args.n_val_s3,
        n_synthetic_s4=args.n_synthetic_s4,
        n_nc1_test=args.n_nc1_test,
        n_val_s5=args.n_val_s5,
        seed=args.seed,
        backend=args.backend,
    )
    print_summary(report)


if __name__ == "__main__":
    main()
