"""Extract frozen-dynamics upstream activity features for Phase 6B."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from research.pipeline.extract_features_v2 import (
    DATA_DIR,
    OUTPUTS_DIR,
    load_or_create_variant_graph,
    validate_split_and_variant,
)
from research.pipeline.fly_simulator import FlySimulator
from research.pipeline.flywire_graph import load_flywire_graph
from research.pipeline.retina import retina_encoder
from src.connectome.schema import ConnectomeGraph


VARIANTS = ("real", "random", "scramble")
SPLITS = ("train", "val")
OUTPUT_DIR = OUTPUTS_DIR / "v2" / "phase6_upstream"


def select_topk_edges(
    graph: ConnectomeGraph,
    readout_neuron_indices: np.ndarray,
    k: int = 16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select up to k largest-|weight| incoming edges for each ordered readout."""
    if k <= 0:
        raise ValueError("k must be positive")
    readout = np.asarray(readout_neuron_indices, dtype=np.int64)
    if readout.ndim != 1 or np.any(readout < 0) or np.any(readout >= graph.n_neurons):
        raise ValueError("readout_neuron_indices must be valid graph indices")
    weights = graph.weights.tocsr()
    targets: list[int] = []
    sources: list[int] = []
    values: list[float] = []
    for target_slot, neuron_index in enumerate(readout):
        start, stop = weights.indptr[neuron_index : neuron_index + 2]
        row_sources = weights.indices[start:stop]
        row_weights = weights.data[start:stop]
        if not np.isfinite(row_weights).all():
            raise FloatingPointError("graph contains NaN or Inf edge weights")
        order = np.lexsort((row_sources, -np.abs(row_weights)))[:k]
        targets.extend([target_slot] * len(order))
        sources.extend(row_sources[order].tolist())
        values.extend(row_weights[order].astype(np.float64).tolist())
    if not values:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
        )
    target = np.asarray(targets, dtype=np.int64)
    source = np.asarray(sources, dtype=np.int64)
    value = np.asarray(values, dtype=np.float64)
    order = np.lexsort((source, target))
    return target[order], source[order], value[order]


def select_random_edges_per_target(
    graph: ConnectomeGraph,
    readout_neuron_indices: np.ndarray,
    target_counts: np.ndarray,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select deterministic random incoming edges matching each target's count."""
    readout = np.asarray(readout_neuron_indices, dtype=np.int64)
    counts = np.asarray(target_counts, dtype=np.int64)
    if readout.ndim != 1 or counts.shape != readout.shape or np.any(counts < 0):
        raise ValueError("target_counts must be non-negative and match readout length")
    weights = graph.weights.tocsr()
    rng = np.random.default_rng(int(seed))
    targets: list[int] = []
    sources: list[int] = []
    values: list[float] = []
    for target_slot, (neuron_index, count) in enumerate(zip(readout, counts)):
        start, stop = weights.indptr[neuron_index : neuron_index + 2]
        row_sources = weights.indices[start:stop]
        row_weights = weights.data[start:stop]
        if count > len(row_sources):
            raise ValueError(
                f"target {int(neuron_index)} has {len(row_sources)} incoming edges, needs {int(count)}"
            )
        if count == 0:
            continue
        chosen = np.sort(rng.choice(len(row_sources), size=int(count), replace=False))
        targets.extend([target_slot] * len(chosen))
        sources.extend(row_sources[chosen].tolist())
        values.extend(row_weights[chosen].astype(np.float64).tolist())
    if not values:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
        )
    target = np.asarray(targets, dtype=np.int64)
    source = np.asarray(sources, dtype=np.int64)
    value = np.asarray(values, dtype=np.float64)
    order = np.lexsort((source, target))
    return target[order], source[order], value[order]


def activity_collapse_stats(
    activities: np.ndarray,
    chunk_size: int = 2048,
    columns: np.ndarray | None = None,
) -> dict[str, float | int]:
    """Measure constant/zero upstream dimensions using bounded-memory moments."""
    if activities.ndim != 2 or len(activities) == 0:
        raise ValueError("activities must be a non-empty sample-by-neuron matrix")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    n_samples = activities.shape[0]
    selected_columns = (
        np.arange(activities.shape[1], dtype=np.int64)
        if columns is None
        else np.asarray(columns, dtype=np.int64)
    )
    if selected_columns.ndim != 1 or len(selected_columns) == 0:
        raise ValueError("columns must select at least one upstream neuron")
    if np.any(selected_columns < 0) or np.any(selected_columns >= activities.shape[1]):
        raise ValueError("columns contains an out-of-range feature index")
    n_features = len(selected_columns)
    all_zero = np.ones(n_features, dtype=bool)
    minima = np.full(n_features, np.inf, dtype=np.float64)
    maxima = np.full(n_features, -np.inf, dtype=np.float64)
    for start in range(0, n_samples, chunk_size):
        block = np.asarray(
            activities[start : start + chunk_size, selected_columns], dtype=np.float64
        )
        if not np.isfinite(block).all():
            raise FloatingPointError("upstream activity contains NaN or Inf")
        all_zero &= np.all(block == 0.0, axis=0)
        minima = np.minimum(minima, block.min(axis=0))
        maxima = np.maximum(maxima, block.max(axis=0))
    # Comparing extrema avoids cancellation falsely giving nonzero variance to
    # constant float16 columns when their mean is not exactly representable.
    constant = minima == maxima
    collapsed = constant | all_zero
    count = int(collapsed.sum())
    return {
        "samples": int(n_samples),
        "features": int(n_features),
        "constant_or_zero_columns": count,
        "constant_or_zero_ratio": float(count / n_features),
        "all_zero_columns": int(all_zero.sum()),
        "zero_variance_columns": int(constant.sum()),
    }


def source_index_hash(source_indices: np.ndarray) -> str:
    """Hash an ordered int64 source-index vector for provenance."""
    vector = np.ascontiguousarray(source_indices, dtype="<i8")
    return hashlib.sha256(vector.tobytes()).hexdigest()


def extract_upstream_activities(
    variant: str,
    split: str,
    source_indices: np.ndarray,
    *,
    chunk_size: int = 256,
    output_path: Path | None = None,
) -> tuple[np.memmap, dict]:
    """Extract time-mean activity on only Train or Val, always with a fresh seed-0 sim."""
    validate_split_and_variant(split, variant)
    if split not in SPLITS:
        raise ValueError(f"forbidden split {split!r}")
    source = np.asarray(source_indices, dtype=np.int64)
    if source.ndim != 1 or len(source) == 0:
        raise ValueError("source_indices must be a non-empty 1-D array")
    if np.any(source < 0) or len(np.unique(source)) != len(source):
        raise ValueError("source_indices must be unique and non-negative")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    samples = pd.read_parquet(
        DATA_DIR / "samples.parquet",
        columns=["sample_id", "image_idx", "split"],
        filters=[("split", "=", split)],
    )
    samples = samples.loc[samples["split"] == split].sort_values("sample_id").reset_index(drop=True)
    n_samples = len(samples)
    if n_samples == 0:
        raise ValueError(f"no samples in allowed split {split!r}")
    image_indices = samples["image_idx"].to_numpy(dtype=np.int64)
    sample_ids = samples["sample_id"].to_numpy(dtype=np.int64)

    image_path = DATA_DIR / "images.npy"
    image_mmap = np.load(image_path, mmap_mode="r")
    graph = load_or_create_variant_graph(variant, load_flywire_graph(use_cache=True), seed=0)
    if np.any(source >= graph.n_neurons):
        raise ValueError("source index exceeds graph neuron count")
    mean_image = np.load(OUTPUTS_DIR / "v3" / "train_mean_image.npy")
    encoder = retina_encoder(graph, mean_image)
    dynamics_path = OUTPUTS_DIR / "v3" / "dynamics_remote.json"
    dynamics = json.loads(dynamics_path.read_text(encoding="utf-8"))
    baseline_path = OUTPUTS_DIR / "v3" / "baseline_remote.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    sim = FlySimulator(
        graph=graph,
        seed=0,
        steps=int(dynamics["steps"]),
        gain=float(dynamics["gain"]),
        leak=float(dynamics["leak"]),
        noise_std=float(dynamics.get("noise_std_robustness", 0.0)),
        encoder=encoder,
        baseline=baseline,
        direct_currents=True,
        backend="torch",
    )

    path = Path(output_path) if output_path is not None else OUTPUT_DIR / f"{variant}_{split}.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite upstream feature file: {path}")
    activities = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float16, shape=(n_samples, len(source))
    )
    started = time.perf_counter()
    for start in range(0, n_samples, chunk_size):
        stop = min(start + chunk_size, n_samples)
        images = np.asarray(image_mmap[image_indices[start:stop]])
        result = sim.run(images, return_activity_indices=source)
        block = result["neuron_activity_mean"]
        if block.shape != (stop - start, len(source)):
            raise RuntimeError("upstream activity extractor returned an unexpected shape")
        if not np.isfinite(block).all():
            raise FloatingPointError(f"NaN or Inf in upstream chunk [{start}:{stop}]")
        stored_block = block.astype(np.float16)
        if not np.isfinite(stored_block).all():
            raise FloatingPointError(f"NaN or Inf after float16 conversion in chunk [{start}:{stop}]")
        activities[start:stop] = stored_block
        if start == 0 or stop == n_samples or stop % (chunk_size * 20) == 0:
            print(
                f"upstream {variant}/{split}: {stop}/{n_samples} samples, "
                f"elapsed={time.perf_counter() - started:.1f}s"
            )
    activities.flush()
    with path.open("rb") as feature_file:
        output_hash = hashlib.file_digest(feature_file, "sha256").hexdigest()
    metadata = {
        "variant": variant,
        "split": split,
        "n_samples": int(n_samples),
        "n_source_neurons": int(len(source)),
        "source_indices_sha256": source_index_hash(source),
        "sample_ids_sha256": hashlib.sha256(
            np.ascontiguousarray(sample_ids, dtype="<i8").tobytes()
        ).hexdigest(),
        "noise_seed": 0,
        "steps": int(dynamics["steps"]),
        "chunk_size": int(chunk_size),
        "dtype": "float16",
        "output_sha256": output_hash,
        "extraction_seconds": float(time.perf_counter() - started),
    }
    return activities, metadata
