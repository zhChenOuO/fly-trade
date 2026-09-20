"""Execute the frozen Phase 6A/6B experiment without accessing sealed splits."""
from __future__ import annotations

from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import pearsonr
import yaml

from research.pipeline.baselines_v2 import (
    compute_classification_metrics,
    compute_continuous_metrics,
    simulate_trading_spot_long_only,
)
from research.pipeline.extract_features_v2 import load_or_create_variant_graph
from research.pipeline.extract_upstream_v2 import (
    OUTPUT_DIR as UPSTREAM_DIR,
    activity_collapse_stats,
    extract_upstream_activities,
    select_random_edges_per_target,
    select_topk_edges,
    source_index_hash,
)
from research.pipeline.flywire_graph import load_flywire_graph, spectral_radius
from research.pipeline.phase6_training import (
    ACTIONS,
    DecoderMLP,
    fit_final_model,
    predict_model,
    prune_edge_budget,
    select_hyperparameters_train_only,
    spearman_ic,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_ROOT = PROJECT_ROOT / "research"
DATA_DIR = RESEARCH_ROOT / "data"
OUTPUTS_DIR = RESEARCH_ROOT / "outputs" / "v2"
P5_ROOT = PROJECT_ROOT.parent / "phase5"
P5_FEATURES_DIR = P5_ROOT / "research" / "outputs" / "v2" / "features"
P5_PROBE_PATH = P5_ROOT / "research" / "outputs" / "v2" / "probe_val.json"
P5_BASELINES_PATH = RESEARCH_ROOT / "outputs" / "v2" / "baselines_val.json"
CONFIG_PATH = RESEARCH_ROOT / "config" / "experiment_v2.yaml"
ALLOWED_SPLITS = ("train", "val")
VARIANTS = ("real", "random", "scramble")
MODEL_KEYS = ("6A_real", "6A_random", "6A_scramble", "6B_real", "6B_random", "6B_scramble", "6B_null")
BLOCK_SIZE = 24
BOOTSTRAPS = 2000
BOOTSTRAP_SEED = 42
READOUT_K = 16


class StopConditionError(RuntimeError):
    """Raised when a frozen Phase 6 stop condition is reached."""


def sha256_file(path: Path) -> str:
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def stable_array_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        if value.dtype.kind == "f":
            value = value.astype("<f8", copy=False)
        else:
            value = value.astype("<i8", copy=False)
        digest.update(value.tobytes())
    return digest.hexdigest()


def holm_adjust(p_values: Mapping[str, float | None]) -> dict[str, float | None]:
    """Holm adjustment across the declared family; missing p-values use 1.0.

    Returning None for unavailable comparisons prevents the placeholder from being
    mistaken for an observed p-value. The 1.0 placeholder is conservative for the
    adjusted values of comparisons whose paired bootstrap is available.
    """
    if not p_values:
        return {}
    effective: list[tuple[str, float]] = []
    for name, raw in p_values.items():
        if raw is None:
            value = 1.0
        else:
            value = float(raw)
            if not np.isfinite(value) or value < 0.0 or value > 1.0:
                raise ValueError(f"invalid p-value for {name}: {raw}")
        effective.append((name, value))
    ordered = sorted(effective, key=lambda item: (item[1], item[0]))
    family_size = len(ordered)
    adjusted: dict[str, float | None] = {name: None for name in p_values}
    running = 0.0
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (family_size - rank) * value))
        if p_values[name] is not None:
            adjusted[name] = float(running)
    return adjusted


def paired_block_bootstrap_ic(
    y_true: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    comparison_pairs: Mapping[str, tuple[str, str]],
    *,
    block_size: int = BLOCK_SIZE,
    n_bootstraps: int = BOOTSTRAPS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Bootstrap model IC and paired ΔIC using identical moving blocks."""
    truth = np.asarray(y_true, dtype=np.float64)
    if truth.ndim != 1 or len(truth) <= block_size:
        raise ValueError("y_true must be 1-D and longer than the bootstrap block")
    if n_bootstraps <= 0 or block_size <= 0:
        raise ValueError("bootstrap count and block size must be positive")
    pred = {name: np.asarray(values, dtype=np.float64) for name, values in predictions.items()}
    for name, values in pred.items():
        if values.shape != truth.shape or not np.isfinite(values).all():
            raise ValueError(f"invalid prediction vector for {name}")
    for name, (left, right) in comparison_pairs.items():
        if left not in pred or right not in pred:
            raise KeyError(f"comparison {name} references a missing prediction")

    n = len(truth)
    n_blocks = int(math.ceil(n / block_size))
    starts = np.random.default_rng(seed).integers(
        0, n - block_size + 1, size=(n_bootstraps, n_blocks), dtype=np.int64
    )
    model_draws = {name: np.empty(n_bootstraps, dtype=np.float64) for name in pred}
    pair_draws = {name: np.empty(n_bootstraps, dtype=np.float64) for name in comparison_pairs}
    offsets = np.arange(block_size, dtype=np.int64)
    for draw in range(n_bootstraps):
        indices = (starts[draw, :, None] + offsets[None, :]).reshape(-1)[:n]
        ic_values = {
            name: spearman_ic(truth[indices], values[indices])
            for name, values in pred.items()
        }
        for name, value in ic_values.items():
            model_draws[name][draw] = value
        for name, (left, right) in comparison_pairs.items():
            pair_draws[name][draw] = ic_values[left] - ic_values[right]

    model_summary: dict[str, dict[str, Any]] = {}
    for name, values in pred.items():
        model_summary[name] = {
            "spearman_ic": spearman_ic(truth, values),
            "ci_95": [float(np.percentile(model_draws[name], 2.5)), float(np.percentile(model_draws[name], 97.5))],
        }
    comparison_summary: dict[str, dict[str, Any]] = {}
    for name, (left, right) in comparison_pairs.items():
        delta = pair_draws[name]
        p_lo = (1 + int(np.count_nonzero(delta <= 0.0))) / (n_bootstraps + 1)
        p_hi = (1 + int(np.count_nonzero(delta >= 0.0))) / (n_bootstraps + 1)
        comparison_summary[name] = {
            "left": left,
            "right": right,
            "delta_ic": spearman_ic(truth, pred[left]) - spearman_ic(truth, pred[right]),
            "ci_95": [float(np.percentile(delta, 2.5)), float(np.percentile(delta, 97.5))],
            "p_two_sided": float(min(1.0, 2.0 * min(p_lo, p_hi))),
            "bootstrap_replicates": int(n_bootstraps),
            "block_size": int(block_size),
        }
    return {"models": model_summary, "comparisons": comparison_summary}


def _read_allowed_split(split: str) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, dict]:
    if split not in ALLOWED_SPLITS:
        raise ValueError(f"forbidden split {split!r}; allowed={ALLOWED_SPLITS}")
    samples = pd.read_parquet(
        DATA_DIR / "samples.parquet",
        columns=["sample_id", "image_idx", "split"],
        filters=[("split", "=", split)],
    )
    labels = pd.read_parquet(
        DATA_DIR / "labels_v2.parquet",
        columns=["sample_id", "split", "future_return_6", "action"],
        filters=[("split", "=", split)],
    )
    samples = samples.loc[samples["split"] == split].sort_values("sample_id").reset_index(drop=True)
    labels = labels.loc[labels["split"] == split].sort_values("sample_id").reset_index(drop=True)
    if not np.array_equal(samples["sample_id"].to_numpy(), labels["sample_id"].to_numpy()):
        raise ValueError(f"sample/label IDs differ in allowed split {split}")
    valid = labels["future_return_6"].notna().to_numpy() & labels["action"].notna().to_numpy()
    dropped = int(len(labels) - int(valid.sum()))
    targets = labels.loc[valid, "future_return_6"].to_numpy(dtype=np.float64)
    actions = labels.loc[valid, "action"].to_numpy(dtype=object)
    if not np.isfinite(targets).all():
        raise FloatingPointError(f"{split} target contains NaN or Inf after dropping NaN labels")
    if not set(actions.tolist()).issubset(set(ACTIONS)):
        raise ValueError(f"unexpected action labels in {split}: {sorted(set(actions.tolist()) - set(ACTIONS))}")
    return samples, labels, samples.loc[valid, "sample_id"].to_numpy(dtype=np.int64), targets, actions, {
        "samples_before_nan_drop": int(len(labels)),
        "samples_after_nan_drop": int(valid.sum()),
        "dropped_nan_labels": dropped,
    }


def _validate_feature_files() -> tuple[dict[str, str], dict[str, dict]]:
    if not P5_PROBE_PATH.is_file():
        raise FileNotFoundError(f"missing Phase 5 probe metadata: {P5_PROBE_PATH}")
    probe = json.loads(P5_PROBE_PATH.read_text(encoding="utf-8"))
    expected_hashes = probe.get("metadata", {}).get("feature_hashes", {})
    if len(expected_hashes) != 6:
        raise ValueError("Phase 5 probe metadata does not list all six frozen feature hashes")
    samples_by_split = {split: _read_allowed_split(split) for split in ALLOWED_SPLITS}
    actual_hashes: dict[str, str] = {}
    sidecars: dict[str, dict] = {}
    for variant in VARIANTS:
        for split in ALLOWED_SPLITS:
            name = f"{variant}_{split}"
            source = P5_FEATURES_DIR / f"{name}.npy"
            target = OUTPUTS_DIR / "features" / f"{name}.npy"
            sidecar_path = P5_FEATURES_DIR / f"{name}.json"
            if not source.is_file() or not target.is_file() or not target.is_symlink():
                raise StopConditionError(f"Phase 5 feature must be symlinked from Phase 5: {target}")
            if target.resolve() != source.resolve():
                raise StopConditionError(f"Phase 5 feature symlink points at wrong source: {target}")
            actual = sha256_file(target)
            expected = expected_hashes.get(name)
            if actual != expected:
                raise StopConditionError(
                    f"Phase 5 feature SHA256 mismatch for {name}: expected={expected}, actual={actual}"
                )
            if not sidecar_path.is_file():
                raise FileNotFoundError(f"missing Phase 5 feature metadata: {sidecar_path}")
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            if sidecar.get("variant") != variant or sidecar.get("split") != split:
                raise ValueError(f"Phase 5 metadata identity mismatch for {name}")
            all_ids = samples_by_split[split][0]["sample_id"].to_numpy(dtype=np.int64)
            if len(all_ids) != sidecar.get("n_samples"):
                raise ValueError(f"Phase 5 sample count mismatch for {name}")
            ids_hash = hashlib.sha256(np.ascontiguousarray(all_ids, dtype="<i8").tobytes()).hexdigest()
            if ids_hash != sidecar.get("sample_ids_sha256"):
                raise ValueError(f"Phase 5 sample ID order mismatch for {name}")
            features = np.load(target, mmap_mode="r")
            if features.shape != (len(all_ids), 1291) or features.dtype != np.float32:
                raise ValueError(f"unexpected Phase 5 feature shape/dtype for {name}: {features.shape}/{features.dtype}")
            for start in range(0, len(features), 2048):
                if not np.isfinite(features[start : start + 2048]).all():
                    raise StopConditionError(f"NaN or Inf in Phase 5 feature file {name}")
            actual_hashes[name] = actual
            sidecars[name] = sidecar
            print(f"feature gate {name}: sha256 PASS, shape={features.shape}, dtype={features.dtype}")
    return actual_hashes, sidecars


def _graph_cache_paths(variant: str) -> tuple[Path, Path]:
    name = "graph_cache.npz" if variant == "real" else f"{variant}_seed0_cache.npz"
    phase6 = DATA_DIR / "flywire" / name
    phase5 = P5_ROOT / "research" / "data" / "flywire" / name
    return phase5, phase6


def _load_graph_edge_sets(sidecars: Mapping[str, dict]) -> tuple[dict, dict, dict]:
    base = load_flywire_graph(use_cache=True)
    readout = np.concatenate([base.meta["buy_idx"], base.meta["sell_idx"]]).astype(np.int64)
    if len(readout) != 1291:
        raise ValueError(f"expected 1291 Phase 5 DN readout neurons, got {len(readout)}")
    graphs = {"real": base}
    graph_metadata: dict[str, dict[str, Any]] = {}
    candidates: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for variant in VARIANTS:
        graph = base if variant == "real" else load_or_create_variant_graph(variant, base, seed=0)
        p5_cache, p6_cache = _graph_cache_paths(variant)
        if not p5_cache.is_file() or not p6_cache.is_file():
            raise FileNotFoundError(f"missing Phase 5 graph cache for {variant}")
        p5_sha, p6_sha = sha256_file(p5_cache), sha256_file(p6_cache)
        if p5_sha != p6_sha:
            raise ValueError(f"Phase 5/6 graph cache SHA mismatch for {variant}")
        expected_rho = float(sidecars[f"{variant}_train"]["graph_spectral_radius"])
        actual_rho = spectral_radius(graph)
        if actual_rho != expected_rho:
            raise ValueError(
                f"spectral radius mismatch for {variant}: Phase 5={expected_rho}, Phase 6={actual_rho}"
            )
        target, source, weight = select_topk_edges(graph, readout, k=READOUT_K)
        if len(target) == 0:
            raise StopConditionError(f"no candidate incoming edges for {variant}")
        graphs[variant] = graph
        candidates[variant] = (target, source, weight)
        graph_metadata[variant] = {
            "graph_cache_sha256": p6_sha,
            "phase5_graph_cache_sha256": p5_sha,
            "phase5_graph_sha_in_metadata": sidecars[f"{variant}_train"].get("graph_sha256"),
            "graph_sha_was_present_in_phase5_metadata": "graph_sha256" in sidecars[f"{variant}_train"],
            "spectral_radius_phase5_metadata": expected_rho,
            "spectral_radius_loaded_graph": actual_rho,
            "spectral_radius_matches_phase5": actual_rho == expected_rho,
            "n_neurons": int(graph.n_neurons),
            "n_synapses": int(graph.weights.nnz),
            "candidate_edge_count": int(len(target)),
            "candidate_unique_source_count": int(len(np.unique(source))),
        }
        print(
            f"graph gate {variant}: sha256={p6_sha}; rho={actual_rho:.9f} "
            f"(Phase 5={expected_rho:.9f}); candidate_edges={len(target)}"
        )

    common_budget = min(len(edges[0]) for edges in candidates.values())
    if common_budget <= 0:
        raise StopConditionError("cannot align a positive 6B synapse parameter budget")
    aligned: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for variant, edge_arrays in candidates.items():
        aligned[variant] = prune_edge_budget(*edge_arrays, budget=common_budget)
        if len(aligned[variant][0]) != common_budget:
            raise StopConditionError(f"failed to align 6B edge budget for {variant}")
        graph_metadata[variant]["aligned_edge_count"] = int(len(aligned[variant][0]))

    real_counts = np.bincount(aligned["real"][0], minlength=len(readout))
    null_edges = select_random_edges_per_target(
        base, readout, real_counts, seed=0
    )
    if len(null_edges[0]) != common_budget:
        raise StopConditionError("6B-null could not match the real per-readout edge budget")
    aligned["null"] = null_edges
    graph_metadata["null"] = {
        "graph_cache_sha256": graph_metadata["real"]["graph_cache_sha256"],
        "phase5_graph_cache_sha256": graph_metadata["real"]["phase5_graph_cache_sha256"],
        "spectral_radius_phase5_metadata": graph_metadata["real"]["spectral_radius_phase5_metadata"],
        "spectral_radius_loaded_graph": graph_metadata["real"]["spectral_radius_loaded_graph"],
        "candidate_edge_count": int(len(candidates["real"][0])),
        "aligned_edge_count": int(len(null_edges[0])),
        "selection": "seed-0 random incoming edges; per-target counts match aligned real E_i",
    }
    for variant in VARIANTS:
        graph_metadata[variant]["pre_alignment_edge_count"] = int(len(candidates[variant][0]))
        graph_metadata[variant]["post_alignment_edge_count"] = int(len(aligned[variant][0]))
        graph_metadata[variant]["edge_set_sha256"] = stable_array_hash(*aligned[variant])
    graph_metadata["null"]["pre_alignment_edge_count"] = int(len(candidates["real"][0]))
    graph_metadata["null"]["post_alignment_edge_count"] = int(len(null_edges[0]))
    graph_metadata["null"]["edge_set_sha256"] = stable_array_hash(*null_edges)
    graph_metadata["parameter_budget"] = {
        "common_active_synapse_parameters": int(common_budget),
        "readout_dim": int(len(readout)),
        "k_max_per_readout": READOUT_K,
        "linear_head_parameters": int(4 * (len(readout) + 1)),
        "effective_total_trainable_parameters": int(common_budget + 4 * (len(readout) + 1)),
        "registered_delta_slots_per_model": int(len(readout) * READOUT_K),
        "masked_delta_slots_per_model": int(len(readout) * READOUT_K - common_budget),
        "budget_aligned": True,
    }
    return graphs, aligned, graph_metadata


def _source_nodes_by_variant(
    aligned: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> dict[str, np.ndarray]:
    result = {
        variant: np.unique(aligned[variant][1]).astype(np.int64)
        for variant in VARIANTS
    }
    result["real"] = np.unique(np.concatenate([aligned["real"][1], aligned["null"][1]])).astype(np.int64)
    return result


def _sparse_context(
    edges: tuple[np.ndarray, np.ndarray, np.ndarray],
    source_nodes: np.ndarray,
    n_readout: int = 1291,
) -> dict[str, np.ndarray]:
    target, source, weight = edges
    source_nodes = np.asarray(source_nodes, dtype=np.int64)
    source_idx = np.zeros((n_readout, READOUT_K), dtype=np.int64)
    base_weights = np.zeros((n_readout, READOUT_K), dtype=np.float32)
    mask = np.zeros((n_readout, READOUT_K), dtype=bool)
    slots = np.zeros(n_readout, dtype=np.int64)
    local_sources = np.searchsorted(source_nodes, source)
    if np.any(local_sources >= len(source_nodes)) or not np.array_equal(source_nodes[local_sources], source):
        raise ValueError("source node is missing from upstream feature matrix")
    for t, s, w, local_s in zip(target, source, weight, local_sources):
        slot = int(slots[t])
        if slot >= READOUT_K:
            raise ValueError(f"readout {int(t)} exceeds the frozen k={READOUT_K}")
        source_idx[t, slot] = int(local_s)
        base_weights[t, slot] = float(w)
        mask[t, slot] = True
        slots[t] += 1
    return {
        "source_indices": source_idx,
        "base_weights": base_weights,
        "active_mask": mask,
    }


def _collapse_gate(
    variant: str,
    split_array: np.ndarray,
    source_nodes: np.ndarray,
    model_sources: np.ndarray,
) -> dict:
    local_columns = np.searchsorted(source_nodes, model_sources)
    if np.any(local_columns >= len(source_nodes)) or not np.array_equal(source_nodes[local_columns], model_sources):
        raise ValueError("model source is missing from extracted upstream activities")
    return activity_collapse_stats(split_array, columns=local_columns)


def _extract_upstream(
    aligned: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    sources: Mapping[str, np.ndarray],
    p5_sidecars: Mapping[str, dict],
) -> tuple[dict[str, dict[str, np.memmap]], dict[str, dict]]:
    arrays: dict[str, dict[str, np.memmap]] = {v: {} for v in VARIANTS}
    metadata: dict[str, dict] = {}
    model_sources = {
        "6B_real": np.unique(aligned["real"][1]),
        "6B_null": np.unique(aligned["null"][1]),
        "6B_random": np.unique(aligned["random"][1]),
        "6B_scramble": np.unique(aligned["scramble"][1]),
    }
    for variant in VARIANTS:
        for split in ALLOWED_SPLITS:
            activity, info = extract_upstream_activities(
                variant,
                split,
                sources[variant],
                chunk_size=256,
            )
            expected_ids_hash = p5_sidecars[f"{variant}_{split}"]["sample_ids_sha256"]
            if info["sample_ids_sha256"] != expected_ids_hash:
                raise ValueError(f"upstream sample order differs from Phase 5 for {variant}/{split}")
            arrays[variant][split] = activity
            metadata[f"{variant}_{split}"] = info
            print(
                f"upstream gate {variant}/{split}: rows={info['n_samples']}, "
                f"sources={info['n_source_neurons']}, sha256={info['output_sha256']}"
            )
            if split == "train":
                applicable = [key for key in model_sources if key in (f"6B_{variant}", "6B_real", "6B_null") and (variant == "real" or key == f"6B_{variant}")]
                for model_key in applicable:
                    stats = _collapse_gate(
                        variant,
                        activity,
                        sources[variant],
                        model_sources[model_key],
                    )
                    metadata.setdefault("train_activity_collapse", {})[model_key] = stats
                    print(
                        f"upstream collapse {model_key}: "
                        f"{stats['constant_or_zero_columns']}/{stats['features']} "
                        f"({stats['constant_or_zero_ratio']:.6f})"
                    )
                    if stats["constant_or_zero_ratio"] > 0.90:
                        raise StopConditionError(
                            f"upstream activity collapse >90% for {model_key}: {stats}"
                        )
    return arrays, metadata


def _consistency_check(
    upstream_val: np.ndarray,
    source_nodes: np.ndarray,
    edges: tuple[np.ndarray, np.ndarray, np.ndarray],
    dn_val: np.ndarray,
) -> dict[str, Any]:
    n = min(100, len(upstream_val), len(dn_val))
    target, source, weight = edges
    local = np.searchsorted(source_nodes, source)
    if np.any(local >= len(source_nodes)) or not np.array_equal(source_nodes[local], source):
        raise ValueError("consistency check source index is missing")
    matrix = sp.csr_matrix(
        (weight.astype(np.float32), (target, local)),
        shape=(dn_val.shape[1], len(source_nodes)),
        dtype=np.float32,
    )
    approximate = np.asarray(matrix @ np.asarray(upstream_val[:n], dtype=np.float32).T).T
    reference = np.asarray(dn_val[:n], dtype=np.float32)
    if not np.isfinite(approximate).all() or not np.isfinite(reference).all():
        raise FloatingPointError("NaN or Inf in Phase 6B consistency check")
    if np.std(approximate) == 0.0 or np.std(reference) == 0.0:
        flat_r = None
    else:
        raw_r = float(pearsonr(approximate.reshape(-1), reference.reshape(-1)).statistic)
        flat_r = raw_r if np.isfinite(raw_r) else None
    per_target = []
    for col in range(reference.shape[1]):
        if np.std(approximate[:, col]) > 0 and np.std(reference[:, col]) > 0:
            raw = float(pearsonr(approximate[:, col], reference[:, col]).statistic)
            if np.isfinite(raw):
                per_target.append(raw)
    return {
        "samples": int(n),
        "split": "val",
        "method": "initial W_ij-weighted upstream mean activity versus frozen Phase 5 DN readout; diagnostic only",
        "flattened_pearson_r": flat_r,
        "finite_readout_dimensions": int(len(per_target)),
        "per_dimension_pearson_r_mean": float(np.mean(per_target)) if per_target else None,
        "per_dimension_pearson_r_median": float(np.median(per_target)) if per_target else None,
    }


def _model_predictions(
    model_key: str,
    model_type: str,
    variant: str,
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: np.ndarray,
    action_train: np.ndarray,
    sparse_context: Mapping[str, np.ndarray] | None,
    *,
    sample_stride_bars: int,
    device: str,
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    per_seed: list[dict] = []
    return_predictions: list[np.ndarray] = []
    action_logits: list[np.ndarray] = []
    for seed in range(5):
        selected = select_hyperparameters_train_only(
            X_train,
            y_train,
            action_train,
            sample_stride_bars=sample_stride_bars,
            seed=seed,
            sparse_context=sparse_context,
            device=device,
        )
        model, normalization = fit_final_model(
            X_train,
            y_train,
            action_train,
            selected=selected["selected"],
            epochs=selected["best_epoch"],
            seed=seed,
            sparse_context=sparse_context,
            device=device,
        )
        pred_return, logits = predict_model(model, X_val, normalization, device=device)
        if not np.isfinite(pred_return).all() or not np.isfinite(logits).all():
            raise StopConditionError(f"NaN or Inf predictions from {model_key}, seed={seed}")
        return_predictions.append(pred_return)
        action_logits.append(logits)
        per_seed.append(
            {
                "seed": seed,
                "selected_hyperparameters": selected["selected"],
                "best_epoch": selected["best_epoch"],
                "inner_validation": {
                    "combined_loss": selected["inner_val_loss"],
                    "grid_scores": selected["grid_scores"],
                    "train_samples": selected["inner_train_samples"],
                    "purged_samples": selected["purged_samples"],
                    "validation_samples": selected["inner_val_samples"],
                    "target_mean": selected["inner_target_mean"],
                    "target_std": selected["inner_target_std"],
                },
                "full_train_normalization": normalization,
                "continuous_predictions": pred_return,
                "action_logits": logits,
            }
        )
        print(
            f"trained {model_key} seed={seed}: hparams={selected['selected']} "
            f"best_epoch={selected['best_epoch']} inner_loss={selected['inner_val_loss']:.8g}"
        )
        del model
        try:
            import torch

            torch.cuda.empty_cache()
        except ImportError:
            pass
    return per_seed, np.mean(np.stack(return_predictions), axis=0), np.mean(np.stack(action_logits), axis=0)


def _trading_metrics(actions: np.ndarray, returns: np.ndarray, config: Mapping[str, Any]) -> dict:
    trade = simulate_trading_spot_long_only(
        actions,
        returns,
        float(config["cost"]["cost_fee_per_side"]),
        float(config["cost"]["cost_slippage"]),
    )
    if trade["trade_count"] == 0:
        trade["status"] = "N/A_NO_TRADES"
        for key in (
            "net_return",
            "annualized_sharpe",
            "max_drawdown",
            "turnover",
            "per_step_mean_return",
            "per_step_volatility",
        ):
            trade[key] = "N/A_NO_TRADES"
    else:
        trade["status"] = "COMPUTED"
    return trade


def _evaluate_one(
    y_true: np.ndarray,
    action_true: np.ndarray,
    pred_return: np.ndarray,
    action_logits: np.ndarray,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    if not np.isfinite(pred_return).all() or not np.isfinite(action_logits).all():
        raise StopConditionError("NaN or Inf in final Val predictions")
    action_pred = np.asarray(ACTIONS, dtype=object)[np.argmax(action_logits, axis=1)]
    sorted_logits = np.sort(action_logits, axis=1)
    margins = sorted_logits[:, -1] - sorted_logits[:, -2]
    continuous = compute_continuous_metrics(y_true, pred_return)
    classification = compute_classification_metrics(action_true, action_pred)
    return {
        "continuous": continuous,
        "classification": classification,
        "action_margin": {
            "top_vs_second_logit_std": float(np.std(margins)),
            "top_vs_second_logit_mean": float(np.mean(margins)),
        },
        "prediction_distribution": {
            "counts": classification["predicted_counts"],
            "ratios": classification["predicted_ratios"],
        },
        "trading": _trading_metrics(action_pred, y_true, config),
    }


def _make_threshold_results(
    model_results: Mapping[str, dict],
    comparisons: Mapping[str, dict],
    thresholds: Mapping[str, dict],
) -> dict[str, dict]:
    def value(key: str) -> float:
        return float(thresholds[key]["value"])

    def check(observed, threshold, operator: str = ">=") -> dict:
        passed = observed is not None and np.isfinite(float(observed)) and (
            float(observed) >= threshold if operator == ">=" else float(observed) <= threshold
        )
        return {
            "observed": observed,
            "threshold": threshold,
            "operator": operator,
            "status": "PASS" if passed else "FAIL",
        }

    out: dict[str, dict] = {}
    for model_key, result in model_results.items():
        aggregate = result["aggregate"]
        continuous = aggregate["continuous"]
        classification = aggregate["classification"]
        ratios = aggregate["prediction_distribution"]["ratios"]
        trading = aggregate["trading"]
        seeds = result["seed_summary"]
        checks = {
            "primary_metric_spearman_ic": check(
                continuous["spearman_ic"], value("primary_metric")
            ),
            "balanced_accuracy_min_practical_effect": check(
                classification["balanced_accuracy"],
                0.5 + value("min_practical_effect_balanced_acc"),
            ),
            "seeds_agreement_ratio": check(
                seeds["direction_agreement_ratio"], value("seeds_agreement_ratio")
            ),
            "minority_action_ratio": check(
                min(ratios.values()), value("minority_action_ratio")
            ),
            "action_margin_std": check(
                aggregate["action_margin"]["top_vs_second_logit_std"], value("margin_std_min")
            ),
        }
        if model_key in ("6A_real", "6B_real"):
            comparison_names = (
                [f"6A_real_vs_6A_{control}" for control in ("random", "scramble")]
                if model_key == "6A_real"
                else [f"6B_real_vs_6B_{control}" for control in ("random", "scramble", "null")]
            )
            deltas = [comparisons[name]["delta_ic"] for name in comparison_names]
            checks["min_practical_effect_spearman_ic"] = check(
                min(deltas), value("min_practical_effect")
            )
            matched_passes = []
            for name in comparison_names:
                comp = comparisons[name]
                matched_passes.append(
                    comp["delta_ic"] >= value("connectome_vs_matched_control_ic_delta_min")
                    and comp["ci_95"] is not None
                    and comp["ci_95"][0] > 0.0
                    and comp["p_holm"] is not None
                    and comp["p_holm"] <= value("connectome_vs_matched_control_p_max")
                )
            checks["matched_control_gain_all_required_comparisons"] = {
                "comparisons": comparison_names,
                "minimum_delta_threshold": value("connectome_vs_matched_control_ic_delta_min"),
                "maximum_holm_p_threshold": value("connectome_vs_matched_control_p_max"),
                "all_ci_lower_bounds_gt_zero": all(
                    comparisons[name]["ci_95"] is not None and comparisons[name]["ci_95"][0] > 0.0
                    for name in comparison_names
                ),
                "status": "PASS" if all(matched_passes) else "FAIL",
            }
        else:
            checks["min_practical_effect_spearman_ic"] = {
                "status": "NOT_APPLICABLE_CONTROL_MODEL"
            }
        if model_key in ("6A_real", "6B_real"):
            phase = model_key[:2]
            baseline_names = (f"{phase}_real_vs_B1b", f"{phase}_real_vs_B2")
            checks["ohlcv_baseline_gain"] = {
                "comparisons": {
                    name: {
                        "point_delta_ic": comparisons[name]["delta_ic"],
                        "minimum_delta_threshold": value("connectome_vs_ohlcv_baseline_ic_delta_min"),
                        "paired_ci_95": None,
                        "paired_p_two_sided": None,
                        "status": "FAIL_UNVERIFIED_REQUIRED_PAIRED_BOOTSTRAP",
                    }
                    for name in baseline_names
                },
                "status": "FAIL_UNVERIFIED_REQUIRED_PAIRED_BOOTSTRAP",
            }
        else:
            checks["ohlcv_baseline_gain"] = {"status": "NOT_APPLICABLE_CONTROL_MODEL"}
        for check_key, metric, threshold, operator in (
            ("trading_net_return", "net_return", value("trading_net_return_min"), ">="),
            ("trading_sharpe", "annualized_sharpe", value("trading_sharpe_min"), ">="),
            ("trading_max_drawdown", "max_drawdown", value("trading_max_drawdown_max"), "<="),
        ):
            raw = trading[metric]
            checks[check_key] = (
                check(raw, threshold, operator)
                if isinstance(raw, (int, float))
                else {"observed": raw, "threshold": threshold, "status": "FAIL_NA_NO_TRADES"}
            )
        out[model_key] = checks
    return out


def _git_commit() -> str:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except Exception:
        return "UNKNOWN"


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def _memory_gib() -> float | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return float(line.split()[1]) / 1024**2
    except OSError:
        pass
    return None


def run_phase6() -> dict[str, Any]:
    started_at = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    config_hash = sha256_file(CONFIG_PATH)
    if any(split not in ALLOWED_SPLITS for split in ALLOWED_SPLITS):
        raise AssertionError("invalid split whitelist")
    print("PHASE6_OVERRIDE_EXPLORATORY: NOT CONFIRMATORY EVIDENCE")
    print(f"start={started_at} branch=remote/phase6 base={_git_commit()}")

    feature_hashes, p5_sidecars = _validate_feature_files()
    split_data = {split: _read_allowed_split(split) for split in ALLOWED_SPLITS}
    train_samples, train_labels, train_valid_ids, y_train, action_train, train_counts = split_data["train"]
    val_samples, val_labels, val_valid_ids, y_val, action_val, val_counts = split_data["val"]
    train_valid_mask = train_labels["future_return_6"].notna().to_numpy() & train_labels["action"].notna().to_numpy()
    val_valid_mask = val_labels["future_return_6"].notna().to_numpy() & val_labels["action"].notna().to_numpy()

    features_a: dict[str, dict[str, np.ndarray]] = {v: {} for v in VARIANTS}
    for variant in VARIANTS:
        for split, valid_mask in (("train", train_valid_mask), ("val", val_valid_mask)):
            feature_path = OUTPUTS_DIR / "features" / f"{variant}_{split}.npy"
            all_features = np.load(feature_path, mmap_mode="r")
            features_a[variant][split] = all_features[valid_mask]
            if not np.isfinite(features_a[variant][split]).all():
                raise StopConditionError(f"NaN or Inf after aligning labels for {variant}/{split}")

    graphs, edge_sets, graph_metadata = _load_graph_edge_sets(p5_sidecars)
    source_nodes = _source_nodes_by_variant(edge_sets)
    contexts = {
        "6B_real": _sparse_context(edge_sets["real"], source_nodes["real"]),
        "6B_null": _sparse_context(edge_sets["null"], source_nodes["real"]),
        "6B_random": _sparse_context(edge_sets["random"], source_nodes["random"]),
        "6B_scramble": _sparse_context(edge_sets["scramble"], source_nodes["scramble"]),
    }
    graph_metadata["source_activity"] = {
        variant: {
            "source_count": int(len(source_nodes[variant])),
            "source_indices_sha256": source_index_hash(source_nodes[variant]),
        }
        for variant in VARIANTS
    }

    upstream, upstream_meta = _extract_upstream(edge_sets, source_nodes, p5_sidecars)
    for variant in VARIANTS:
        for split in ALLOWED_SPLITS:
            feature_path = UPSTREAM_DIR / f"{variant}_{split}.npy"
            if feature_path.is_file():
                upstream_meta[f"{variant}_{split}"]["feature_path"] = str(feature_path.relative_to(PROJECT_ROOT))
    check_100 = _consistency_check(
        upstream["real"]["val"],
        source_nodes["real"],
        edge_sets["real"],
        features_a["real"]["val"],
    )
    check_100["sample_id_first"] = int(val_valid_ids[0])
    check_100["sample_id_last"] = int(val_valid_ids[min(99, len(val_valid_ids) - 1)])
    print(
        "6B initial consistency (diagnostic, first 100 Val): "
        f"flattened Pearson r={check_100['flattened_pearson_r']}"
    )

    stride = int(config["data"]["sample_stride"])
    model_inputs: dict[str, tuple[np.ndarray, np.ndarray, Mapping[str, np.ndarray] | None]] = {}
    for variant in VARIANTS:
        model_inputs[f"6A_{variant}"] = (
            features_a[variant]["train"], features_a[variant]["val"], None
        )
        model_inputs[f"6B_{variant}"] = (
            upstream[variant]["train"], upstream[variant]["val"], contexts[f"6B_{variant}"]
        )
    model_inputs["6B_null"] = (
        upstream["real"]["train"], upstream["real"]["val"], contexts["6B_null"]
    )

    all_predictions: dict[str, np.ndarray] = {}
    all_logits: dict[str, np.ndarray] = {}
    fit_records: dict[str, list[dict]] = {}
    for model_key in MODEL_KEYS:
        if model_key == "6B_null":
            variant = "real"
        else:
            variant = model_key.split("_")[1]
        X_train, X_val, sparse_context = model_inputs[model_key]
        model_type = model_key[:2]
        seeds, mean_return, mean_logits = _model_predictions(
            model_key,
            model_type,
            variant,
            X_train,
            X_val,
            y_train,
            action_train,
            sparse_context,
            sample_stride_bars=stride,
            device="cuda",
        )
        fit_records[model_key] = seeds
        all_predictions[model_key] = mean_return
        all_logits[model_key] = mean_logits

    # The only Val reads used for model selection happened after all Train-only fits:
    # aggregate predictions below are final metrics and are never fed back to training.
    model_results: dict[str, dict[str, Any]] = {}
    for model_key in MODEL_KEYS:
        per_seed_results: list[dict] = []
        per_seed_ics: list[float] = []
        for seed_row in fit_records[model_key]:
            seed_actions = np.asarray(ACTIONS, dtype=object)[np.argmax(seed_row["action_logits"], axis=1)]
            metrics = _evaluate_one(
                y_val,
                action_val,
                seed_row["continuous_predictions"],
                seed_row["action_logits"],
                config,
            )
            per_seed_ics.append(metrics["continuous"]["spearman_ic"])
            per_seed_results.append(
                {
                    "seed": seed_row["seed"],
                    "spearman_ic": metrics["continuous"]["spearman_ic"],
                    "mae": metrics["continuous"]["mae"],
                    "balanced_accuracy": metrics["classification"]["balanced_accuracy"],
                    "mcc": metrics["classification"]["mcc"],
                    "precision": metrics["classification"]["precision"],
                    "recall": metrics["classification"]["recall"],
                    "predicted_counts": metrics["prediction_distribution"]["counts"],
                    "predicted_ratios": metrics["prediction_distribution"]["ratios"],
                    "trading": metrics["trading"],
                    "selected_hyperparameters": seed_row["selected_hyperparameters"],
                    "best_epoch": seed_row["best_epoch"],
                    "inner_validation": seed_row["inner_validation"],
                    "full_train_normalization": seed_row["full_train_normalization"],
                }
            )
        aggregate = _evaluate_one(
            y_val,
            action_val,
            all_predictions[model_key],
            all_logits[model_key],
            config,
        )
        model_results[model_key] = {
            "aggregate": aggregate,
            "per_seed": per_seed_results,
            "seed_summary": {
                "spearman_ic_mean": float(np.mean(per_seed_ics)),
                "spearman_ic_sd": float(np.std(per_seed_ics, ddof=1)),
                "direction_agreement_ratio": float(np.mean(np.asarray(per_seed_ics) > 0.0)),
                "direction_agreement_count": int(np.count_nonzero(np.asarray(per_seed_ics) > 0.0)),
                "seed_count": 5,
            },
        }

    comparisons_pairs = {
        "6A_real_vs_6A_random": ("6A_real", "6A_random"),
        "6A_real_vs_6A_scramble": ("6A_real", "6A_scramble"),
        "6B_real_vs_6B_random": ("6B_real", "6B_random"),
        "6B_real_vs_6B_scramble": ("6B_real", "6B_scramble"),
        "6B_real_vs_6B_null": ("6B_real", "6B_null"),
        "6B_real_vs_6A_real": ("6B_real", "6A_real"),
        "6B_random_vs_6A_random": ("6B_random", "6A_random"),
        "6B_scramble_vs_6A_scramble": ("6B_scramble", "6A_scramble"),
    }
    bootstrap = paired_block_bootstrap_ic(
        y_val,
        all_predictions,
        comparisons_pairs,
        block_size=BLOCK_SIZE,
        n_bootstraps=BOOTSTRAPS,
        seed=BOOTSTRAP_SEED,
    )
    comparisons: dict[str, dict] = dict(bootstrap["comparisons"])

    baseline_payload = json.loads(P5_BASELINES_PATH.read_text(encoding="utf-8"))
    if baseline_payload.get("metadata", {}).get("split_evaluated") != "val":
        raise ValueError("OHLCV baseline artifact is not a Val-only result")
    b1b_ic = float(baseline_payload["models"]["B1b_logistic"]["continuous"]["spearman_ic"])
    b2_ic = float(baseline_payload["models"]["B2_mlp"]["continuous"]["ic_mean"])
    baseline_ics = {"B1b": b1b_ic, "B2": b2_ic}
    for phase, model_key in (("6A", "6A_real"), ("6B", "6B_real")):
        for baseline_name, baseline_ic in baseline_ics.items():
            comparisons[f"{phase}_real_vs_{baseline_name}"] = {
                "left": model_key,
                "right": baseline_name,
                "delta_ic": float(bootstrap["models"][model_key]["spearman_ic"] - baseline_ic),
                "ci_95": None,
                "p_two_sided": None,
                "p_holm": None,
                "status": "N/A_MISSING_PAIRED_PREDICTIONS",
                "reason": "baselines_val.json stores aggregate metrics but no per-sample predictions; Phase 6 spec prohibits rerunning B2",
                "baseline_ic_source": "research/outputs/v2/baselines_val.json",
            }
    p_values = {name: result["p_two_sided"] for name, result in comparisons.items()}
    holm_values = holm_adjust(p_values)
    for name, value in holm_values.items():
        comparisons[name]["p_holm"] = value
        if comparisons[name].get("status") is None:
            comparisons[name]["status"] = "COMPUTED_PAIRED_BLOCK_BOOTSTRAP"

    threshold_config = config["thresholds"]
    threshold_results = _make_threshold_results(model_results, comparisons, threshold_config)
    for model_key, checks in threshold_results.items():
        applicable = [row["status"] for row in checks.values() if row.get("status") not in ("NOT_APPLICABLE_CONTROL_MODEL",)]
        model_results[model_key]["pass_fail"] = "PASS" if applicable and all(v == "PASS" for v in applicable) else "FAIL"

    import torch

    graph_parameter_counts = graph_metadata["parameter_budget"]
    decoder_template = DecoderMLP(input_dim=1291, hidden_dim=64, dropout=0.1)
    decoder_parameters = int(sum(parameter.numel() for parameter in decoder_template.parameters()))
    for model_key in MODEL_KEYS:
        if model_key.startswith("6A"):
            graph_parameter_counts[model_key] = {
                "decoder_trainable_parameters": decoder_parameters,
                "effective_total_trainable_parameters": decoder_parameters,
                "input_dim": 1291,
                "hidden_dim": 64,
                "dropout": 0.1,
            }
        elif model_key.startswith("6B"):
            graph_parameter_counts[model_key] = {
                "active_delta_parameters": int(graph_metadata["parameter_budget"]["common_active_synapse_parameters"]),
                "linear_head_parameters": int(graph_metadata["parameter_budget"]["linear_head_parameters"]),
                "effective_total_trainable_parameters": int(graph_metadata["parameter_budget"]["effective_total_trainable_parameters"]),
                "registered_delta_slots": int(graph_metadata["parameter_budget"]["registered_delta_slots_per_model"]),
                "masked_slots_with_zero_grad": int(graph_metadata["parameter_budget"]["masked_delta_slots_per_model"]),
            }
    elapsed = float(time.perf_counter() - start)
    output = {
        "PHASE6_OVERRIDE_EXPLORATORY": "NOT CONFIRMATORY EVIDENCE",
        "governance": {
            "phase5_preregistered_result": "FAIL",
            "override_authorized_by_user_on": "2026-09-20",
            "phase7_entered": False,
            "splits_read_or_evaluated": ["train", "val"],
            "forbidden_splits_accessed": False,
            "hyperparameters_selected_on": "purged chronological 80/20 split within Train only",
            "val_used_for_selection": False,
            "b2_baseline_rerun": False,
        },
        "metadata": {
            "started_at_utc": started_at,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": elapsed,
            "git_commit": _git_commit(),
            "config_sha256": config_hash,
            "feature_hashes": feature_hashes,
            "graph": graph_metadata,
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "gpu_vram_bytes": int(torch.cuda.get_device_properties(0).total_memory),
            "cpu_count": os.cpu_count(),
            "ram_total_gib": _memory_gib(),
            "training": {
                "seeds": [0, 1, 2, 3, 4],
                "epochs_max": 50,
                "early_stopping_patience": 5,
                "hyperparameter_grid": [dict(x) for x in __import__("research.pipeline.phase6_training", fromlist=["HYPERPARAMETER_GRID"]).HYPERPARAMETER_GRID],
                "batch_size": 1024,
                "loss": "Huber(delta=1) + cross_entropy, equal weights",
                "optimizer": "Adam with explicit L2 penalty from the frozen weight_decay grid",
                "phase6a": "hidden=64, ReLU, dropout=0.1, train-only feature/target standardization",
                "phase6b": "raw frozen-dynamics activity means; only active Δ edges and linear heads train; Δ L2",
                "sample_stride_bars": stride,
                "purge_bars": int(config["data"]["purge_bars"]),
                "purged_sample_count": int(math.ceil(config["data"]["purge_bars"] / stride)),
            },
            "phase5_b1b_ic": b1b_ic,
            "phase5_b2_ic_mean": b2_ic,
            "phase5_b2_seeds": baseline_payload["models"]["B2_mlp"].get("seeds"),
            "phase5_b2_paired_predictions_available": False,
        },
        "sample_counts": {"train": train_counts, "val": val_counts},
        "feature_metadata": {
            "phase5_sidecars": {
                name: {
                    "sample_ids_sha256": row["sample_ids_sha256"],
                    "n_samples": row["n_samples"],
                    "graph_spectral_radius": row["graph_spectral_radius"],
                    "dynamics": row["dynamics"],
                }
                for name, row in p5_sidecars.items()
            },
            "phase6b_upstream": upstream_meta,
        },
        "phase6b_consistency_check": check_100,
        "parameter_counts": graph_parameter_counts,
        "thresholds_frozen_from_yaml": threshold_config,
        "models": model_results,
        "aggregate_bootstrap": bootstrap["models"],
        "comparisons": {
            "block_size": BLOCK_SIZE,
            "bootstrap_replicates": BOOTSTRAPS,
            "seed": BOOTSTRAP_SEED,
            "family_size": len(comparisons_pairs) + 4,
            "holm_missing_p_placeholder": 1.0,
            "holm_note": "available paired comparisons were adjusted over the full 12-test family; missing baseline p-values use a conservative 1.0 placeholder and remain N/A",
            "items": comparisons,
        },
        "threshold_results": threshold_results,
        "overall_pass_fail": "PASS" if all(row["pass_fail"] == "PASS" for row in model_results.values()) else "FAIL",
        "reporting": "PHASE6_OVERRIDE_EXPLORATORY: NOT CONFIRMATORY EVIDENCE",
    }
    output_path = OUTPUTS_DIR / "phase6_val.json"
    _write_json_atomic(output_path, output)
    print(f"saved {output_path.relative_to(PROJECT_ROOT)}")
    print(f"overall={output['overall_pass_fail']} elapsed_seconds={elapsed:.2f}")
    return output


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def main() -> int:
    log_path = OUTPUTS_DIR / "phase6_run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        tee = _Tee(sys.stdout, log_file)
        with redirect_stdout(tee):
            try:
                run_phase6()
            except StopConditionError as exc:
                print(f"PHASE6_STOP_CONDITION: {exc}")
                return 2
            except Exception as exc:
                print(f"PHASE6_STEP_FAILURE: {type(exc).__name__}: {exc}")
                raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
