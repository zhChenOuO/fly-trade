"""Frozen Phase 6A decoder and Phase 6B sparse plastic readout training."""
from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
from scipy.stats import spearmanr

try:  # Pure split/metric helpers remain importable in CPU-only environments.
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError:  # pragma: no cover - exercised on machines without PyTorch
    torch = None
    nn = None
    F = None


ACTIONS = ("BUY", "HOLD", "SELL")
HYPERPARAMETER_GRID = (
    {"lr": 1e-3, "weight_decay": 1e-4},
    {"lr": 1e-3, "weight_decay": 1e-2},
    {"lr": 3e-4, "weight_decay": 1e-4},
    {"lr": 3e-4, "weight_decay": 1e-2},
)
MAX_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 5
BATCH_SIZE = 1024


def purged_time_split(
    n_samples: int,
    sample_stride_bars: int,
    purge_bars: int = 60,
    train_fraction: float = 0.8,
) -> tuple[np.ndarray, np.ndarray]:
    """Split an ordered Train sequence and remove the configured temporal purge."""
    if n_samples < 2:
        raise ValueError("at least two Train samples are required")
    if sample_stride_bars <= 0 or purge_bars < 0:
        raise ValueError("sample stride must be positive and purge bars non-negative")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between zero and one")
    cutoff = int(math.floor(n_samples * train_fraction))
    purge_samples = int(math.ceil(purge_bars / sample_stride_bars))
    val_start = cutoff + purge_samples
    if cutoff == 0 or val_start >= n_samples:
        raise ValueError("Train sequence is too short for the requested split and purge")
    return np.arange(cutoff, dtype=np.int64), np.arange(val_start, n_samples, dtype=np.int64)


def prune_edge_budget(
    target_index: np.ndarray,
    source_index: np.ndarray,
    weights: np.ndarray,
    budget: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep exactly ``budget`` edges, dropping smallest |weight| with stable ties."""
    target = np.asarray(target_index, dtype=np.int64)
    source = np.asarray(source_index, dtype=np.int64)
    weight = np.asarray(weights, dtype=np.float64)
    if target.ndim != 1 or source.ndim != 1 or weight.ndim != 1:
        raise ValueError("edge arrays must be one-dimensional")
    if not (len(target) == len(source) == len(weight)):
        raise ValueError("edge arrays must have equal lengths")
    if not np.isfinite(weight).all():
        raise FloatingPointError("edge weights contain NaN or Inf")
    if budget < 0 or budget > len(target):
        raise ValueError(f"budget must be in [0, {len(target)}]")
    if len(target) == 0 or budget == len(target):
        keep = np.arange(len(target), dtype=np.int64)
    else:
        # lexsort's last key is primary: |w|, then target, then source.
        drop_order = np.lexsort((source, target, np.abs(weight)))
        drop = drop_order[: len(target) - budget]
        keep_mask = np.ones(len(target), dtype=bool)
        keep_mask[drop] = False
        keep = np.flatnonzero(keep_mask)
    # Canonical output order makes hashes and edge-to-slot maps stable.
    order = np.lexsort((source[keep], target[keep]))
    keep = keep[order]
    return target[keep].copy(), source[keep].copy(), weight[keep].copy()


def spearman_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return finite Spearman IC, treating a constant vector as zero IC."""
    truth = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    if truth.ndim != 1 or pred.ndim != 1 or truth.shape != pred.shape:
        raise ValueError("y_true and y_pred must be equal-length one-dimensional arrays")
    if not np.isfinite(truth).all() or not np.isfinite(pred).all():
        raise FloatingPointError("IC inputs contain NaN or Inf")
    if len(truth) == 0 or np.std(truth) < 1e-12 or np.std(pred) < 1e-12:
        return 0.0
    value = spearmanr(truth, pred).statistic
    return float(value) if np.isfinite(value) else 0.0


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("Phase 6 model training requires PyTorch")


if nn is not None:

    class DecoderMLP(nn.Module):
        """One hidden layer, shared trunk, regression and three-class heads."""

        def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.1):
            super().__init__()
            self.trunk = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.return_head = nn.Linear(hidden_dim, 1)
            self.action_head = nn.Linear(hidden_dim, 3)

        def forward(self, x):
            hidden = self.trunk(x)
            return self.return_head(hidden).squeeze(-1), self.action_head(hidden)


    class SparsePlasticReadout(nn.Module):
        """Train only masked Δ weights and linear output heads over sparse edges."""

        def __init__(self, source_indices, base_weights, active_mask):
            super().__init__()
            source_indices = torch.as_tensor(source_indices, dtype=torch.long)
            base_weights = torch.as_tensor(base_weights, dtype=torch.float32)
            active_mask = torch.as_tensor(active_mask, dtype=torch.bool)
            if source_indices.ndim != 2 or base_weights.shape != source_indices.shape:
                raise ValueError("source_indices and base_weights must share shape (readout, k)")
            if active_mask.shape != source_indices.shape:
                raise ValueError("active_mask must match source_indices")
            if source_indices.shape[0] == 0 or not active_mask.any():
                raise ValueError("sparse readout must contain at least one active edge")
            self.register_buffer("source_indices", source_indices)
            self.register_buffer("base_weights", base_weights)
            self.register_buffer("active_mask", active_mask)
            self.delta = nn.Parameter(torch.zeros_like(base_weights))
            self.return_head = nn.Linear(source_indices.shape[0], 1)
            self.action_head = nn.Linear(source_indices.shape[0], 3)

        def forward(self, x):
            # Inactive padded slots have a safe source index and a zero coefficient.
            source_activity = x[:, self.source_indices]
            coeff = self.base_weights + self.delta * self.active_mask
            z = torch.sum(source_activity * coeff.unsqueeze(0), dim=-1)
            return self.return_head(z).squeeze(-1), self.action_head(z)

        def active_delta_l2(self):
            return torch.sum((self.delta * self.active_mask) ** 2)

        @property
        def effective_edge_parameters(self) -> int:
            return int(self.active_mask.sum().item())


else:  # pragma: no cover - CPU-only imports without the optional torch dependency

    class DecoderMLP:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            _require_torch()


    class SparsePlasticReadout:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            _require_torch()


def _action_indices(actions: np.ndarray) -> np.ndarray:
    mapping = {name: i for i, name in enumerate(ACTIONS)}
    values = np.asarray(actions, dtype=object)
    unknown = sorted(set(values.tolist()) - set(mapping))
    if unknown:
        raise ValueError(f"unknown action labels: {unknown}")
    return np.asarray([mapping[value] for value in values], dtype=np.int64)


def _feature_scaler(X: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    block = np.asarray(X[indices], dtype=np.float64)
    if not np.isfinite(block).all():
        raise FloatingPointError("Train features contain NaN or Inf")
    mean = block.mean(axis=0)
    std = block.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def _target_scaler(y: np.ndarray, indices: np.ndarray) -> tuple[float, float]:
    block = np.asarray(y[indices], dtype=np.float64)
    if not np.isfinite(block).all():
        raise FloatingPointError("Train targets contain NaN or Inf")
    mean = float(block.mean())
    std = float(block.std())
    if not np.isfinite(std) or std < 1e-12:
        raise ValueError("Train continuous target has zero or non-finite variance")
    return mean, std


def _new_model(input_dim: int, sparse_context: Mapping[str, Any] | None, device):
    _require_torch()
    if sparse_context is None:
        return DecoderMLP(input_dim, hidden_dim=64, dropout=0.1).to(device)
    return SparsePlasticReadout(
        sparse_context["source_indices"],
        sparse_context["base_weights"],
        sparse_context["active_mask"],
    ).to(device)


def _batch_inputs(X, indices, start, stop, scaler, device):
    selected = np.asarray(X[indices[start:stop]], dtype=np.float32)
    if scaler is not None:
        mean, std = scaler
        selected = (selected - mean) / std
    if not np.isfinite(selected).all():
        raise FloatingPointError("model input contains NaN or Inf")
    return torch.as_tensor(selected, dtype=torch.float32, device=device)


def _loss_for_batch(model, xb, yb, ab, weight_decay: float):
    pred_return, action_logits = model(xb)
    loss = F.huber_loss(pred_return, yb, delta=1.0) + F.cross_entropy(action_logits, ab)
    if isinstance(model, SparsePlasticReadout):
        penalty = model.active_delta_l2()
    else:
        penalty = sum(parameter.square().sum() for parameter in model.parameters())
    loss = loss + 0.5 * float(weight_decay) * penalty
    return loss


def _train_epochs(
    model,
    X,
    y_cont_std,
    y_action_idx,
    train_indices,
    val_indices,
    scaler,
    *,
    lr: float,
    weight_decay: float,
    seed: int,
    max_epochs: int,
    patience: int | None,
    device,
) -> tuple[int, float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss = math.inf
    best_epoch = 0
    stale = 0
    for epoch in range(max_epochs):
        model.train()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed) + epoch * 1_000_003)
        order = torch.randperm(len(train_indices), generator=generator).numpy()
        ordered_indices = train_indices[order]
        for start in range(0, len(ordered_indices), BATCH_SIZE):
            batch_idx = ordered_indices[start : start + BATCH_SIZE]
            xb = _batch_inputs(X, batch_idx, 0, len(batch_idx), scaler, device)
            yb = torch.as_tensor(
                y_cont_std[batch_idx], dtype=torch.float32, device=device
            )
            ab = torch.as_tensor(y_action_idx[batch_idx], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = _loss_for_batch(model, xb, yb, ab, weight_decay)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite training loss")
            loss.backward()
            optimizer.step()

        if val_indices is None:
            best_epoch = epoch + 1
            continue

        model.eval()
        total = 0.0
        count = 0
        with torch.inference_mode():
            for start in range(0, len(val_indices), BATCH_SIZE):
                batch_idx = val_indices[start : start + BATCH_SIZE]
                xb = _batch_inputs(X, batch_idx, 0, len(batch_idx), scaler, device)
                yb = torch.as_tensor(
                    y_cont_std[batch_idx], dtype=torch.float32, device=device
                )
                ab = torch.as_tensor(y_action_idx[batch_idx], dtype=torch.long, device=device)
                loss = _loss_for_batch(model, xb, yb, ab, weight_decay)
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite internal validation loss")
                total += float(loss.item()) * len(batch_idx)
                count += len(batch_idx)
        val_loss = total / count
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch + 1
            stale = 0
        else:
            stale += 1
            if patience is not None and stale >= patience:
                break
    return best_epoch, float(best_loss)


def select_hyperparameters_train_only(
    X_train: np.ndarray,
    y_cont_train: np.ndarray,
    y_action_train: np.ndarray,
    *,
    sample_stride_bars: int,
    seed: int,
    sparse_context: Mapping[str, Any] | None = None,
    device: str = "cuda",
) -> dict[str, Any]:
    """Select fixed-grid hyperparameters using a purged split inside Train only.

    There are deliberately no validation-set parameters in this API.
    """
    _require_torch()
    if not torch.cuda.is_available() and str(device).startswith("cuda"):
        raise RuntimeError("Phase 6 training requires CUDA")
    X = X_train
    y = np.asarray(y_cont_train, dtype=np.float32)
    actions = np.asarray(y_action_train, dtype=object)
    if X.ndim != 2 or len(X) != len(y) or len(y) != len(actions):
        raise ValueError("Train feature and label lengths do not match")
    if not np.isfinite(y).all():
        raise FloatingPointError("Train continuous targets contain NaN or Inf")
    action_idx = _action_indices(actions)
    inner_train, inner_val = purged_time_split(
        len(y), sample_stride_bars=sample_stride_bars, purge_bars=60, train_fraction=0.8
    )
    if sparse_context is None:
        scaler = _feature_scaler(X, inner_train)
    else:
        scaler = None  # 6B preserves the frozen raw time-mean activity in z_i.
    y_mean, y_std = _target_scaler(y, inner_train)
    y_std_values = (y - y_mean) / y_std
    scores: list[dict[str, Any]] = []
    for candidate in HYPERPARAMETER_GRID:
        torch.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))
        model = _new_model(X.shape[1], sparse_context, device)
        epoch, val_loss = _train_epochs(
            model,
            X,
            y_std_values,
            action_idx,
            inner_train,
            inner_val,
            scaler,
            lr=candidate["lr"],
            weight_decay=candidate["weight_decay"],
            seed=seed,
            max_epochs=MAX_EPOCHS,
            patience=EARLY_STOPPING_PATIENCE,
            device=device,
        )
        scores.append(
            {
                **candidate,
                "best_epoch": epoch,
                "inner_val_loss": val_loss,
            }
        )
        del model
    best = min(scores, key=lambda row: (row["inner_val_loss"], HYPERPARAMETER_GRID.index({"lr": row["lr"], "weight_decay": row["weight_decay"]})))
    return {
        "selected": {"lr": best["lr"], "weight_decay": best["weight_decay"]},
        "best_epoch": int(best["best_epoch"]),
        "inner_val_loss": float(best["inner_val_loss"]),
        "grid_scores": scores,
        "inner_train_samples": int(len(inner_train)),
        "purged_samples": int(inner_val[0] - inner_train[-1] - 1),
        "inner_val_samples": int(len(inner_val)),
        "inner_target_mean": y_mean,
        "inner_target_std": y_std,
    }


def fit_final_model(
    X_train: np.ndarray,
    y_cont_train: np.ndarray,
    y_action_train: np.ndarray,
    *,
    selected: Mapping[str, float],
    epochs: int,
    seed: int,
    sparse_context: Mapping[str, Any] | None = None,
    device: str = "cuda",
):
    """Retrain on the full Train split for the inner split's selected best epoch."""
    _require_torch()
    if not torch.cuda.is_available() and str(device).startswith("cuda"):
        raise RuntimeError("Phase 6 training requires CUDA")
    X = X_train
    y = np.asarray(y_cont_train, dtype=np.float32)
    actions = np.asarray(y_action_train, dtype=object)
    action_idx = _action_indices(actions)
    if not np.isfinite(y).all():
        raise FloatingPointError("Train continuous targets contain NaN or Inf")
    all_indices = np.arange(len(y), dtype=np.int64)
    scaler = None if sparse_context is not None else _feature_scaler(X, all_indices)
    y_mean, y_std = _target_scaler(y, all_indices)
    y_std_values = (y - y_mean) / y_std
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    model = _new_model(X.shape[1], sparse_context, device)
    _train_epochs(
        model,
        X,
        y_std_values,
        action_idx,
        all_indices,
        None,
        scaler,
        lr=float(selected["lr"]),
        weight_decay=float(selected["weight_decay"]),
        seed=seed,
        max_epochs=int(epochs),
        patience=None,
        device=device,
    )
    model.eval()
    return model, {
        "x_mean": scaler[0].tolist() if scaler is not None else None,
        "x_std": scaler[1].tolist() if scaler is not None else None,
        "target_mean": y_mean,
        "target_std": y_std,
    }


def predict_model(
    model,
    X: np.ndarray,
    normalization: Mapping[str, Any],
    *,
    batch_size: int = BATCH_SIZE,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray]:
    """Predict continuous targets and return three-class action logits."""
    _require_torch()
    mean = normalization.get("x_mean")
    std = normalization.get("x_std")
    scaler = (
        (np.asarray(mean, dtype=np.float32), np.asarray(std, dtype=np.float32))
        if mean is not None
        else None
    )
    ret = np.empty(len(X), dtype=np.float32)
    logits_result = np.empty((len(X), 3), dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(X), batch_size):
            idx = np.arange(start, min(start + batch_size, len(X)), dtype=np.int64)
            xb = _batch_inputs(X, idx, 0, len(idx), scaler, device)
            pred, logits = model(xb)
            if not torch.isfinite(pred).all() or not torch.isfinite(logits).all():
                raise FloatingPointError("model predictions contain NaN or Inf")
            ret[start : start + len(idx)] = (
                pred.cpu().numpy() * float(normalization["target_std"])
                + float(normalization["target_mean"])
            )
            logits_result[start : start + len(idx)] = logits.cpu().numpy()
    return ret.astype(np.float64), logits_result.astype(np.float64)
