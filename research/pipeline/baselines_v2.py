"""Research v2 Phase 4: Non-Connectome Baselines.

Evaluates benchmark models on OHLCV features to establish performance floors:
- B0: Constant / Majority baseline
- B1a: Ridge regression (continuous future_return_6)
- B1b: Multinomial Logistic regression (3-class action)
- B2: Simple 1-hidden-layer MLP (Adam optimizer, multi-seed)

Governance Contract:
- Strict split whitelist: only 'train' and 'val' are permitted.
- 'dev_test_v1' (old test) and any holdout are strictly forbidden.
- Standardization parameters (mean/std) are computed exclusively on Train.
- Continuous target: future_return_6.
- Discrete target: action (BUY, HOLD, SELL; spot long-only semantics where SELL = flat).
- Moving block bootstrap for time-series confidence intervals.
- Non-overlapping trading simulation with transaction costs dynamically read from config.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import yaml

import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats

ALLOWED_SPLITS = ("train", "val")
ACTION_BUY = "BUY"
ACTION_HOLD = "HOLD"
ACTION_SELL = "SELL"
ACTIONS = [ACTION_BUY, ACTION_HOLD, ACTION_SELL]

BAR = pd.Timedelta(minutes=5)
ROOT = Path(__file__).resolve().parents[1]


def validate_split_name(split: str) -> None:
    """Validate that split is in the whitelist. Raise ValueError if forbidden."""
    if split not in ALLOWED_SPLITS:
        raise ValueError(
            f"Split {split!r} is strictly forbidden in baseline fitting/evaluation. "
            f"Allowed splits: {ALLOWED_SPLITS}. 'dev_test_v1' and 'holdout' must never be evaluated."
        )


def compute_file_sha256(path: Path | str) -> str:
    """Compute SHA256 hex digest of a file."""
    path = Path(path)
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def get_git_commit() -> str:
    """Get current git HEAD commit hash, or 'UNKNOWN' if unavailable."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT.parent,
            stderr=subprocess.DEVNULL,
        ).decode("utf-8").strip()
        return out
    except Exception:
        return "UNKNOWN"


def extract_sample_features(
    raw: pd.DataFrame,
    samples: pd.DataFrame,
    window_bars: int = 48,
) -> tuple[np.ndarray, list[str]]:
    """Extract OHLCV features strictly using data at and before timestamp t.

    For each sample, uses the 48-bar input window [t - window_bars + 1, t].
    Features:
    1. 48 close log returns:
       r[0] = log(close[0] / open[0])
       r[k] = log(close[k] / close[k-1]) for k = 1..47
    2. 48 volume log: log(1 + volume[k]) for k = 0..47
    3. 48 volume rate of change: log_vol[k] - log_vol[k-1] (0 for k=0)
    4. 1 window realized volatility: sample std of close log returns (ddof=1)
    5. 1 window total return: log(close[47] / close[0])
    6. 4 time features:
       sin/cos of day of week (2pi * dow / 7)
       sin/cos of 5m time of day (2pi * minute_of_day / 1440)

    Total features: 48 + 48 + 48 + 1 + 1 + 2 + 2 = 150 features.
    """
    raw_closes = raw["close"].to_numpy(dtype=np.float64)
    raw_opens = raw["open"].to_numpy(dtype=np.float64)
    raw_volumes = raw["volume"].to_numpy(dtype=np.float64)
    raw_ts = raw["timestamp"]

    # Align each sample to its final input bar index
    pos = np.asarray(raw_ts.searchsorted(samples["timestamp"] - BAR), dtype=np.int64)

    # Matrix of bar indices: shape (N, window_bars)
    idx = pos[:, None] - (window_bars - 1) + np.arange(window_bars)[None, :]
    if (idx < 0).any():
        raise ValueError("Sample window extends prior to the beginning of raw data.")

    c = raw_closes[idx]
    o = raw_opens[idx]
    v = raw_volumes[idx]

    # 1. 48 close log returns
    r = np.empty_like(c)
    r[:, 0] = np.log(c[:, 0] / o[:, 0])
    r[:, 1:] = np.log(c[:, 1:] / c[:, :-1])

    # 2. 48 volume log
    log_v = np.log1p(v)

    # 3. 48 volume rate of change
    diff_v = np.zeros_like(log_v)
    diff_v[:, 1:] = log_v[:, 1:] - log_v[:, :-1]

    # 4. Window realized volatility
    r_vol = np.std(r, axis=1, ddof=1, keepdims=True)

    # 5. Window total return
    tot_r = np.log(c[:, -1:] / c[:, :1])

    # 6. Time features (evaluated at the close of the last input candle t)
    sample_ts = pd.to_datetime(samples["timestamp"]) - BAR
    dow = sample_ts.dt.dayofweek.to_numpy()
    dow_sin = np.sin(2.0 * np.pi * dow / 7.0)[:, None]
    dow_cos = np.cos(2.0 * np.pi * dow / 7.0)[:, None]

    mod = (sample_ts.dt.hour * 60 + sample_ts.dt.minute).to_numpy()
    tod_sin = np.sin(2.0 * np.pi * mod / 1440.0)[:, None]
    tod_cos = np.cos(2.0 * np.pi * mod / 1440.0)[:, None]

    X = np.hstack([r, log_v, diff_v, r_vol, tot_r, dow_sin, dow_cos, tod_sin, tod_cos])

    # Feature names
    names = [f"close_return_bar_{i}" for i in range(window_bars)]
    names.extend([f"log_volume_bar_{i}" for i in range(window_bars)])
    names.extend([f"volume_change_bar_{i}" for i in range(window_bars)])
    names.append("window_realized_volatility")
    names.append("window_total_return")
    names.extend(["day_of_week_sin", "day_of_week_cos", "time_of_day_sin", "time_of_day_cos"])

    return X, names


def compute_train_standardization(
    X_train: np.ndarray,
    feature_names: list[str],
) -> dict:
    """Compute standardization mean and std exclusively on Train data.

    Returns artifact dictionary with SHA256 digest.
    """
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    # Avoid zero division on flat features
    std_adj = np.where(std < 1e-8, 1.0, std)

    payload = {
        "mean": mean.tolist(),
        "std": std_adj.tolist(),
        "feature_names": feature_names,
        "n_samples": int(len(X_train)),
    }
    payload_str = json.dumps(payload, sort_keys=True)
    payload["sha256"] = hashlib.sha256(payload_str.encode("utf-8")).hexdigest()
    return payload


def apply_standardization(X: np.ndarray, norm_artifact: dict) -> np.ndarray:
    """Apply pre-computed standardization to features."""
    mean = np.asarray(norm_artifact["mean"], dtype=np.float64)
    std = np.asarray(norm_artifact["std"], dtype=np.float64)
    return (X - mean) / std


# ---------------------------------------------------------------------------
# Evaluation Metrics
# ---------------------------------------------------------------------------

def compute_continuous_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute Spearman IC and MAE for continuous predictions."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    # Edge case: zero variance prediction (e.g. constant B0)
    if np.std(y_pred) < 1e-12 or np.std(y_true) < 1e-12:
        ic = 0.0
    else:
        res = scipy.stats.spearmanr(y_pred, y_true)
        ic = float(res.statistic if hasattr(res, "statistic") else res[0])
        if np.isnan(ic):
            ic = 0.0

    mae = float(np.mean(np.abs(y_pred - y_true)))
    return {"spearman_ic": ic, "mae": mae}


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: list[str] = ACTIONS,
) -> dict:
    """Compute 3-class classification metrics: Balanced Accuracy, MCC, Precision, Recall."""
    y_true = np.asarray(y_true, dtype=object)
    y_pred = np.asarray(y_pred, dtype=object)
    n_samples = len(y_true)

    k = len(classes)
    c2i = {c: i for i, c in enumerate(classes)}

    conf_matrix = np.zeros((k, k), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if t in c2i and p in c2i:
            conf_matrix[c2i[t], c2i[p]] += 1

    recalls = {}
    precisions = {}
    for c in classes:
        idx = c2i[c]
        true_cnt = np.sum(conf_matrix[idx, :])
        pred_cnt = np.sum(conf_matrix[:, idx])
        tp = conf_matrix[idx, idx]

        recalls[c] = float(tp / true_cnt) if true_cnt > 0 else 0.0
        precisions[c] = float(tp / pred_cnt) if pred_cnt > 0 else 0.0

    # Balanced Accuracy: macro-average of recalls across all present classes
    ba = float(np.mean(list(recalls.values())))

    # Multi-class Matthews Correlation Coefficient (Gorodkin's formula)
    s = float(np.sum(conf_matrix))
    c_sum = float(np.trace(conf_matrix))
    p_sum = np.sum(conf_matrix, axis=0, dtype=np.float64)
    t_sum = np.sum(conf_matrix, axis=1, dtype=np.float64)

    num = c_sum * s - float(np.sum(p_sum * t_sum))
    den = float(np.sqrt(max(s**2 - np.sum(p_sum**2), 0.0)) * np.sqrt(max(s**2 - np.sum(t_sum**2), 0.0)))
    mcc = float(num / den) if den > 0 else 0.0

    counts = {c: int(np.sum(y_pred == c)) for c in classes}
    ratios = {c: float(counts[c] / n_samples) if n_samples > 0 else 0.0 for c in classes}

    return {
        "balanced_accuracy": ba,
        "mcc": mcc,
        "precision": precisions,
        "recall": recalls,
        "predicted_counts": counts,
        "predicted_ratios": ratios,
        "confusion_matrix": conf_matrix.tolist(),
    }


def moving_block_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_type: str = "ic",
    block_size: int = 24,
    n_bootstraps: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Moving Block Bootstrap (MBB) for time series confidence intervals.

    Block Length Selection Rationale:
    Samples are spaced by sample_stride = 6 bars (30 minutes).
    block_size = 24 samples corresponds to 24 * 30 min = 12 hours of market time.
    This window captures intraday volatility clustering, session persistence,
    and autocorrelations without excessively diminishing block variety.

    Parameters
    ----------
    y_true: np.ndarray
        Ground truth targets.
    y_pred: np.ndarray
        Predicted outputs.
    metric_type: str
        'ic' for Spearman IC or 'ba' for Balanced Accuracy.
    block_size: int
        Length of contiguous blocks (default 24).
    n_bootstraps: int
        Number of bootstrap replications.
    alpha: float
        Significance level (e.g. 0.05 for 95% CI).
    seed: int
        Random seed for determinism.

    Returns
    -------
    tuple[float, float]
        (lower_bound, upper_bound)
    """
    n = len(y_true)
    if n <= block_size:
        raise ValueError(f"Sample length {n} must be greater than block size {block_size}")

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_size))
    replicates = np.empty(n_bootstraps, dtype=np.float64)

    for b in range(n_bootstraps):
        starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        idx = (starts[:, None] + np.arange(block_size)).ravel()[:n]

        sub_true = y_true[idx]
        sub_pred = y_pred[idx]

        if metric_type == "ic":
            if np.std(sub_pred) < 1e-12 or np.std(sub_true) < 1e-12:
                replicates[b] = 0.0
            else:
                r = scipy.stats.spearmanr(sub_pred, sub_true)
                val = float(r.statistic if hasattr(r, "statistic") else r[0])
                replicates[b] = 0.0 if np.isnan(val) else val
        elif metric_type == "ba":
            recs = []
            for c in ACTIONS:
                mask = (sub_true == c)
                if np.sum(mask) > 0:
                    recs.append(float(np.mean(sub_pred[mask] == c)))
            replicates[b] = float(np.mean(recs)) if recs else 0.0
        else:
            raise ValueError(f"Unknown metric_type {metric_type}")

    lower = float(np.percentile(replicates, 100.0 * (alpha / 2.0)))
    upper = float(np.percentile(replicates, 100.0 * (1.0 - alpha / 2.0)))
    return lower, upper


# ---------------------------------------------------------------------------
# Trading Simulation (Spot Long-Only)
# ---------------------------------------------------------------------------

def simulate_trading_spot_long_only(
    actions: np.ndarray,
    future_log_returns_6: np.ndarray,
    cost_fee_per_side: float,
    cost_slippage: float,
) -> dict:
    """Spot long-only trading simulation over non-overlapping horizon-6 steps.

    Trading Semantics:
    - BUY: enter/hold 100% long (position = 1.0)
    - SELL: reduce/close position to flat (position = 0.0, NOT short)
    - HOLD: maintain current position (position = previous_position)
    - Cost per trade: cost_fee_per_side + cost_slippage / 2.0 on position change
    - Compounding: portfolio equity V_{t+1} = V_t * (1 + R_{p, t})
      where R_{p, t} = position_t * (exp(future_log_return_6) - 1) - cost_t
    """
    actions = np.asarray(actions, dtype=object)
    returns = np.asarray(future_log_returns_6, dtype=np.float64)
    t_len = len(actions)

    cost_per_trade = float(cost_fee_per_side + cost_slippage / 2.0)

    # Determine position curve
    positions = np.zeros(t_len, dtype=np.float64)
    curr_pos = 0.0
    for t in range(t_len):
        act = actions[t]
        if act == ACTION_BUY:
            curr_pos = 1.0
        elif act == ACTION_SELL:
            curr_pos = 0.0
        # HOLD keeps curr_pos
        positions[t] = curr_pos

    pos_prev = np.empty(t_len, dtype=np.float64)
    pos_prev[0] = 0.0
    pos_prev[1:] = positions[:-1]

    pos_changes = np.abs(positions - pos_prev)
    turnover = float(np.sum(pos_changes))
    trade_count = int(np.sum(pos_changes > 0))
    trade_costs = pos_changes * cost_per_trade

    # Market arithmetic return over each 6-bar step
    r_market = np.expm1(returns)
    r_portfolio = positions * r_market - trade_costs

    equity = np.cumprod(1.0 + r_portfolio)
    net_return = float(equity[-1] - 1.0) if t_len > 0 else 0.0

    bh_equity = np.cumprod(1.0 + r_market)
    bh_return = float(bh_equity[-1] - 1.0) if t_len > 0 else 0.0

    coverage = float(np.mean(positions)) if t_len > 0 else 0.0

    peaks = np.maximum.accumulate(np.insert(equity, 0, 1.0))
    drawdowns = (peaks[1:] - equity) / peaks[1:]
    max_dd = float(np.max(drawdowns)) if t_len > 0 else 0.0

    mu_r = float(np.mean(r_portfolio)) if t_len > 0 else 0.0
    sigma_r = float(np.std(r_portfolio, ddof=1)) if t_len > 1 else 0.0

    # 17,532 steps per year (365.25 days * 24 hrs * 2 steps/hr)
    steps_per_year = 17532
    ann_sharpe = float((mu_r / (sigma_r + 1e-12)) * np.sqrt(steps_per_year)) if sigma_r > 0 else 0.0

    return {
        "net_return": net_return,
        "buy_and_hold_return": bh_return,
        "annualized_sharpe": ann_sharpe,
        "max_drawdown": max_dd,
        "turnover": turnover,
        "trade_count": trade_count,
        "market_coverage": coverage,
        "per_step_mean_return": mu_r,
        "per_step_volatility": sigma_r,
    }


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def fit_predict_b0_constant(
    y_train_cont: np.ndarray,
    y_train_cls: np.ndarray,
    n_val: int,
) -> tuple[np.ndarray, np.ndarray]:
    """B0: Constant continuous mean and Majority classification baseline."""
    mean_val = float(np.mean(y_train_cont))
    pred_cont = np.full(n_val, mean_val, dtype=np.float64)

    # Majority class in Train
    vals, counts = np.unique(y_train_cls, return_counts=True)
    majority_class = str(vals[np.argmax(counts)])
    pred_cls = np.full(n_val, majority_class, dtype=object)

    return pred_cont, pred_cls


class PureNumpyRidge:
    """Pure NumPy Ridge regression with centered X and y (linear model with intercept)."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)
        self.beta: np.ndarray | None = None
        self.mu_x: np.ndarray | None = None
        self.mu_y: float = 0.0
        self.intercept: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> PureNumpyRidge:
        """Fit Ridge weights on X and 1D y."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.mu_x = np.mean(X, axis=0)
        self.mu_y = float(np.mean(y))
        X_c = X - self.mu_x
        y_c = y - self.mu_y

        n, d = X_c.shape
        if n >= d:
            a_mat = X_c.T @ X_c + self.alpha * np.eye(d)
            self.beta = np.linalg.solve(a_mat, X_c.T @ y_c)
        else:
            a_mat = X_c @ X_c.T + self.alpha * np.eye(n)
            self.beta = X_c.T @ np.linalg.solve(a_mat, y_c)
        self.intercept = float(self.mu_y - self.mu_x @ self.beta)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.beta is None or self.mu_x is None:
            raise RuntimeError("Model is not fitted")
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mu_x) @ self.beta + self.mu_y


def select_ridge_alpha_timeseries_cv(
    X_train: np.ndarray,
    y_train: np.ndarray,
    alphas: tuple[float, ...] = (1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0),
    n_folds: int = 5,
) -> float:
    """Select best Ridge regularization alpha via chronological expanding-window CV on Train.

    Validation Rationale:
    CV is conducted exclusively on Train partitions. Expanding-window folds ensure that
    each evaluation fold is strictly in the chronological future of its training fold,
    preventing lookahead bias while keeping the Validation set completely pristine.
    """
    n_samples = len(X_train)
    fold_size = n_samples // (n_folds + 1)

    best_alpha = alphas[0]
    lowest_mse = float("inf")

    for alpha in alphas:
        mses = []
        for f in range(1, n_folds + 1):
            train_end = f * fold_size
            eval_end = (f + 1) * fold_size

            x_tr = X_train[:train_end]
            y_tr = y_train[:train_end]
            x_ev = X_train[train_end:eval_end]
            y_ev = y_train[train_end:eval_end]

            model = PureNumpyRidge(alpha=alpha).fit(x_tr, y_tr)
            pred = model.predict(x_ev)
            mses.append(float(np.mean((pred - y_ev)**2)))

        mean_mse = float(np.mean(mses))
        if mean_mse < lowest_mse:
            lowest_mse = mean_mse
            best_alpha = alpha

    return best_alpha


def fit_predict_b1a_ridge(
    X_train: np.ndarray,
    y_train_cont: np.ndarray,
    X_val: np.ndarray,
    cost_threshold: float,
    alphas: tuple[float, ...] = (1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0),
) -> tuple[np.ndarray, np.ndarray, float]:
    """B1a: Ridge Regression on continuous return with action thresholding."""
    best_alpha = select_ridge_alpha_timeseries_cv(X_train, y_train_cont, alphas=alphas)
    model = PureNumpyRidge(alpha=best_alpha).fit(X_train, y_train_cont)

    pred_cont = model.predict(X_val)

    # Classify into actions using cost_threshold
    pred_cls = np.full(len(pred_cont), ACTION_HOLD, dtype=object)
    pred_cls[pred_cont > cost_threshold] = ACTION_BUY
    pred_cls[pred_cont < -cost_threshold] = ACTION_SELL

    return pred_cont, pred_cls, best_alpha


def fit_predict_b1b_logistic(
    X_train: np.ndarray,
    y_train_cls: np.ndarray,
    X_val: np.ndarray,
    l2_reg: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """B1b: Multinomial Logistic Regression (3-class Softmax with L2 regularization).

    Optimized using L-BFGS-B via scipy.optimize.minimize.
    Returns:
    - directional probability score P(BUY) - P(SELL)
    - argmax action classifications
    """
    classes = ACTIONS
    c2i = {c: i for i, c in enumerate(classes)}
    n_train, d = X_train.shape
    k = len(classes)

    # One-hot encoding
    y_onehot = np.zeros((n_train, k), dtype=np.float64)
    for i, act in enumerate(y_train_cls):
        y_onehot[i, c2i[act]] = 1.0

    def objective_and_grad(theta: np.ndarray) -> tuple[float, np.ndarray]:
        w = theta[:d * k].reshape((d, k))
        b = theta[d * k:]

        logits = X_train @ w + b
        logits -= np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        loss = -float(np.mean(np.sum(y_onehot * np.log(probs + 1e-15), axis=1)))
        loss += 0.5 * l2_reg * float(np.sum(w**2))

        diff = (probs - y_onehot) / float(n_train)
        grad_w = X_train.T @ diff + l2_reg * w
        grad_b = np.sum(diff, axis=0)
        grad = np.concatenate([grad_w.ravel(), grad_b])
        return loss, grad

    theta0 = np.zeros(d * k + k, dtype=np.float64)
    res = scipy.optimize.minimize(
        objective_and_grad,
        theta0,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 100},
    )

    w_opt = res.x[:d * k].reshape((d, k))
    b_opt = res.x[d * k:]

    logits_val = X_val @ w_opt + b_opt
    logits_val -= np.max(logits_val, axis=1, keepdims=True)
    exp_val = np.exp(logits_val)
    probs_val = exp_val / np.sum(exp_val, axis=1, keepdims=True)

    # Classification by argmax
    pred_idx = np.argmax(probs_val, axis=1)
    pred_cls = np.array([classes[idx] for idx in pred_idx], dtype=object)

    # Directional score: P(BUY) - P(SELL)
    pred_cont = probs_val[:, c2i[ACTION_BUY]] - probs_val[:, c2i[ACTION_SELL]]

    return pred_cont, pred_cls


def fit_predict_b2_mlp(
    X_train: np.ndarray,
    y_train_cont: np.ndarray,
    X_val: np.ndarray,
    cost_threshold: float,
    hidden_dim: int = 64,
    epochs: int = 15,
    batch_size: int = 256,
    lr: float = 1e-3,
    l2_reg: float = 1e-4,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """B2: Simple 1-hidden-layer MLP with Adam optimizer in pure NumPy.

    Architecture: Linear(D -> H) -> ReLU -> Linear(H -> 1)
    Trained on continuous future_return_6 MSE loss, mapped to actions via cost_threshold.
    """
    n, d = X_train.shape
    y_train_vec = y_train_cont[:, None]

    rng = np.random.default_rng(seed)
    w1 = rng.normal(0.0, np.sqrt(2.0 / d), (d, hidden_dim))
    b1 = np.zeros(hidden_dim, dtype=np.float64)
    w2 = rng.normal(0.0, np.sqrt(2.0 / hidden_dim), (hidden_dim, 1))
    b2 = np.zeros(1, dtype=np.float64)

    mw1, vw1 = np.zeros_like(w1), np.zeros_like(w1)
    mb1, vb1 = np.zeros_like(b1), np.zeros_like(b1)
    mw2, vw2 = np.zeros_like(w2), np.zeros_like(w2)
    mb2, vb2 = np.zeros_like(b2), np.zeros_like(b2)

    n_batches = int(np.ceil(n / batch_size))
    step = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    for _ in range(epochs):
        perm = rng.permutation(n)
        for b in range(n_batches):
            step += 1
            idx_b = perm[b * batch_size : (b + 1) * batch_size]
            xb, yb = X_train[idx_b], y_train_vec[idx_b]
            nb = len(idx_b)

            # Forward
            z1 = xb @ w1 + b1
            a1 = np.maximum(z1, 0.0)
            y_hat = a1 @ w2 + b2

            # Backward
            dy = 2.0 * (y_hat - yb) / nb
            dw2 = a1.T @ dy + l2_reg * w2
            db2 = np.sum(dy, axis=0)
            da1 = dy @ w2.T
            dz1 = da1 * (z1 > 0.0)
            dw1 = xb.T @ dz1 + l2_reg * w1
            db1 = np.sum(dz1, axis=0)

            # Adam parameter updates
            for p, dp, m, v in [
                (w1, dw1, mw1, vw1),
                (b1, db1, mb1, vb1),
                (w2, dw2, mw2, vw2),
                (b2, db2, mb2, vb2),
            ]:
                m[:] = beta1 * m + (1.0 - beta1) * dp
                v[:] = beta2 * v + (1.0 - beta2) * (dp**2)
                m_hat = m / (1.0 - beta1**step)
                v_hat = v / (1.0 - beta2**step)
                p -= lr * m_hat / (np.sqrt(v_hat) + eps)

    # Validation prediction
    z1_val = X_val @ w1 + b1
    a1_val = np.maximum(z1_val, 0.0)
    pred_cont = (a1_val @ w2 + b2).ravel()

    pred_cls = np.full(len(pred_cont), ACTION_HOLD, dtype=object)
    pred_cls[pred_cont > cost_threshold] = ACTION_BUY
    pred_cls[pred_cont < -cost_threshold] = ACTION_SELL

    return pred_cont, pred_cls


# ---------------------------------------------------------------------------
# Pipeline Execution & Output
# ---------------------------------------------------------------------------

def run_baselines_v2(
    data_dir: Path | str = ROOT / "data",
    config_path: Path | str = ROOT / "config/experiment_v2.yaml",
    output_path: Path | str = ROOT / "outputs/v2/baselines_val.json",
    mlp_seeds: tuple[int, ...] = (42, 43, 44),
    bootstrap_samples: int = 1000,
) -> dict:
    """Execute Phase 4 non-connectome baselines on Train/Validation."""
    data_dir = Path(data_dir)
    config_path = Path(config_path)
    output_path = Path(output_path)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Read inputs
    raw = pd.read_parquet(data_dir / "raw_ohlcv.parquet")
    samples = pd.read_parquet(data_dir / "samples.parquet")
    labels = pd.read_parquet(data_dir / "labels_v2.parquet")

    # Strict governance whitelist enforcement
    splits_present = set(labels["split"].unique())
    # Ensure no forbidden splits are accidentally accessed in evaluation
    validate_split_name("train")
    validate_split_name("val")

    cost_fee = float(config["cost"]["cost_fee_per_side"])
    cost_slip = float(config["cost"]["cost_slippage"])
    cost_threshold = float(2.0 * cost_fee + cost_slip)

    # 1. Extract features for all samples
    X_all, feature_names = extract_sample_features(raw, samples, window_bars=config["data"]["input_window_bars"])

    # 2. Filter splits (strictly whitelist train and val)
    labels_split = labels["split"].to_numpy()
    y_cont_all = labels["future_return_6"].to_numpy(dtype=np.float64)
    y_cls_all = labels["action"].to_numpy(dtype=object)

    train_mask = (labels_split == "train")
    val_mask = (labels_split == "val")

    # Exclude NaNs (recorded for transparency)
    valid_train = train_mask & (~np.isnan(y_cont_all)) & (pd.notna(y_cls_all))
    valid_val = val_mask & (~np.isnan(y_cont_all)) & (pd.notna(y_cls_all))

    dropped_train = int(np.sum(train_mask) - np.sum(valid_train))
    dropped_val = int(np.sum(val_mask) - np.sum(valid_val))

    X_train_raw = X_all[valid_train]
    y_train_cont = y_cont_all[valid_train]
    y_train_cls = y_cls_all[valid_train]

    X_val_raw = X_all[valid_val]
    y_val_cont = y_cont_all[valid_val]
    y_val_cls = y_cls_all[valid_val]

    # 3. Standardization (Train only)
    norm_artifact = compute_train_standardization(X_train_raw, feature_names)
    X_train = apply_standardization(X_train_raw, norm_artifact)
    X_val = apply_standardization(X_val_raw, norm_artifact)

    results: dict = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": get_git_commit(),
            "config_sha256": compute_file_sha256(config_path),
            "normalization_sha256": norm_artifact["sha256"],
            "split_evaluated": "val",
            "train_samples": int(len(y_train_cont)),
            "val_samples": int(len(y_val_cont)),
            "dropped_nan_train": dropped_train,
            "dropped_nan_val": dropped_val,
            "feature_dim": int(X_train.shape[1]),
            "cost_fee_per_side": cost_fee,
            "cost_slippage": cost_slip,
            "cost_threshold": cost_threshold,
        },
        "models": {},
    }

    # Model B0: Constant / Majority
    b0_pred_cont, b0_pred_cls = fit_predict_b0_constant(y_train_cont, y_train_cls, len(y_val_cont))
    b0_cont_metrics = compute_continuous_metrics(y_val_cont, b0_pred_cont)
    b0_cls_metrics = compute_classification_metrics(y_val_cls, b0_pred_cls)
    b0_trade = simulate_trading_spot_long_only(b0_pred_cls, y_val_cont, cost_fee, cost_slip)
    b0_ic_ci = (0.0, 0.0)
    b0_ba_ci = (b0_cls_metrics["balanced_accuracy"], b0_cls_metrics["balanced_accuracy"])

    results["models"]["B0_constant_majority"] = {
        "description": "Train mean for continuous return; Train majority class for action",
        "continuous": {**b0_cont_metrics, "ic_ci_95": list(b0_ic_ci)},
        "classification": {**b0_cls_metrics, "balanced_accuracy_ci_95": list(b0_ba_ci)},
        "trading": b0_trade,
    }

    # Model B1a: Ridge Regression
    b1a_pred_cont, b1a_pred_cls, b1a_alpha = fit_predict_b1a_ridge(
        X_train, y_train_cont, X_val, cost_threshold
    )
    b1a_cont_metrics = compute_continuous_metrics(y_val_cont, b1a_pred_cont)
    b1a_cls_metrics = compute_classification_metrics(y_val_cls, b1a_pred_cls)
    b1a_trade = simulate_trading_spot_long_only(b1a_pred_cls, y_val_cont, cost_fee, cost_slip)
    b1a_ic_ci = moving_block_bootstrap_ci(
        y_val_cont, b1a_pred_cont, metric_type="ic", n_bootstraps=bootstrap_samples, seed=42
    )
    b1a_ba_ci = moving_block_bootstrap_ci(
        y_val_cls, b1a_pred_cls, metric_type="ba", n_bootstraps=bootstrap_samples, seed=42
    )

    results["models"]["B1a_ridge"] = {
        "description": "Ridge regression with alpha chosen by chronological expanding-window CV on Train",
        "best_alpha": b1a_alpha,
        "continuous": {**b1a_cont_metrics, "ic_ci_95": list(b1a_ic_ci)},
        "classification": {**b1a_cls_metrics, "balanced_accuracy_ci_95": list(b1a_ba_ci)},
        "trading": b1a_trade,
    }

    # Model B1b: Logistic Regression
    b1b_pred_cont, b1b_pred_cls = fit_predict_b1b_logistic(X_train, y_train_cls, X_val)
    b1b_cont_metrics = compute_continuous_metrics(y_val_cont, b1b_pred_cont)
    b1b_cls_metrics = compute_classification_metrics(y_val_cls, b1b_pred_cls)
    b1b_trade = simulate_trading_spot_long_only(b1b_pred_cls, y_val_cont, cost_fee, cost_slip)
    b1b_ic_ci = moving_block_bootstrap_ci(
        y_val_cont, b1b_pred_cont, metric_type="ic", n_bootstraps=bootstrap_samples, seed=42
    )
    b1b_ba_ci = moving_block_bootstrap_ci(
        y_val_cls, b1b_pred_cls, metric_type="ba", n_bootstraps=bootstrap_samples, seed=42
    )

    results["models"]["B1b_logistic"] = {
        "description": "Multinomial Logistic regression (3-class softmax with L2)",
        "continuous": {**b1b_cont_metrics, "ic_ci_95": list(b1b_ic_ci)},
        "classification": {**b1b_cls_metrics, "balanced_accuracy_ci_95": list(b1b_ba_ci)},
        "trading": b1b_trade,
    }

    # Model B2: Simple MLP (multi-seed evaluation)
    mlp_seed_runs = []
    for s in mlp_seeds:
        pred_c, pred_a = fit_predict_b2_mlp(
            X_train, y_train_cont, X_val, cost_threshold, hidden_dim=64, epochs=15, seed=s
        )
        c_met = compute_continuous_metrics(y_val_cont, pred_c)
        a_met = compute_classification_metrics(y_val_cls, pred_a)
        t_met = simulate_trading_spot_long_only(pred_a, y_val_cont, cost_fee, cost_slip)
        mlp_seed_runs.append({
            "seed": s,
            "continuous": c_met,
            "classification": a_met,
            "trading": t_met,
            "pred_cont": pred_c,
            "pred_cls": pred_a,
        })

    # Summary statistics across seeds
    ics = [r["continuous"]["spearman_ic"] for r in mlp_seed_runs]
    maes = [r["continuous"]["mae"] for r in mlp_seed_runs]
    bas = [r["classification"]["balanced_accuracy"] for r in mlp_seed_runs]
    mccs = [r["classification"]["mcc"] for r in mlp_seed_runs]
    rets = [r["trading"]["net_return"] for r in mlp_seed_runs]
    sharpes = [r["trading"]["annualized_sharpe"] for r in mlp_seed_runs]
    drawdowns = [r["trading"]["max_drawdown"] for r in mlp_seed_runs]
    turnovers = [r["trading"]["turnover"] for r in mlp_seed_runs]

    # CI computed on seed 42
    mlp_ic_ci = moving_block_bootstrap_ci(
        y_val_cont, mlp_seed_runs[0]["pred_cont"], metric_type="ic", n_bootstraps=bootstrap_samples, seed=42
    )
    mlp_ba_ci = moving_block_bootstrap_ci(
        y_val_cls, mlp_seed_runs[0]["pred_cls"], metric_type="ba", n_bootstraps=bootstrap_samples, seed=42
    )

    results["models"]["B2_mlp"] = {
        "description": "1-hidden-layer MLP (D=150 -> H=64 -> 1) with Adam optimizer, multi-seed",
        "seeds": list(mlp_seeds),
        "continuous": {
            "ic_mean": float(np.mean(ics)),
            "ic_std": float(np.std(ics)),
            "mae_mean": float(np.mean(maes)),
            "mae_std": float(np.std(maes)),
            "ic_ci_95": list(mlp_ic_ci),
        },
        "classification": {
            "balanced_accuracy_mean": float(np.mean(bas)),
            "balanced_accuracy_std": float(np.std(bas)),
            "mcc_mean": float(np.mean(mccs)),
            "mcc_std": float(np.std(mccs)),
            "balanced_accuracy_ci_95": list(mlp_ba_ci),
        },
        "trading": {
            "net_return_mean": float(np.mean(rets)),
            "net_return_std": float(np.std(rets)),
            "annualized_sharpe_mean": float(np.mean(sharpes)),
            "max_drawdown_mean": float(np.mean(drawdowns)),
            "turnover_mean": float(np.mean(turnovers)),
            "buy_and_hold_return": mlp_seed_runs[0]["trading"]["buy_and_hold_return"],
        },
        "seeds_detail": [
            {
                "seed": r["seed"],
                "ic": r["continuous"]["spearman_ic"],
                "balanced_accuracy": r["classification"]["balanced_accuracy"],
                "net_return": r["trading"]["net_return"],
                "annualized_sharpe": r["trading"]["annualized_sharpe"],
            }
            for r in mlp_seed_runs
        ],
    }

    # 4. Atomic output write
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=output_path.parent, delete=False) as tmp:
        json.dump(results, tmp, indent=2)
        tmp_name = tmp.name
    Path(tmp_name).replace(output_path)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 4 Non-Connectome Baselines.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--config-path", type=Path, default=ROOT / "config/experiment_v2.yaml")
    parser.add_argument("--output-path", type=Path, default=ROOT / "outputs/v2/baselines_val.json")
    args = parser.parse_args()

    print(f"Running Phase 4 Non-Connectome Baselines using config {args.config_path}...")
    res = run_baselines_v2(data_dir=args.data_dir, config_path=args.config_path, output_path=args.output_path)
    print(f"Successfully evaluated baselines on Validation set. Results saved to {args.output_path}")


if __name__ == "__main__":
    main()
