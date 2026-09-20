"""Extract Frozen Connectome readout activity vectors for Research v2 Phase 5.

Extracts time-averaged activity across the readout neuron population
(DN neurons: graph.meta['buy_idx'] U graph.meta['sell_idx'], ~1291 dimensions).

Supported Graph Variants:
- 'real': intact Drosophila connectome (FlyWire v783)
- 'random': matched random graph (permuted weights, random endpoints)
- 'scramble': degree-preserved scramble (double-edge swapped)

Strict Governance:
- Whitelist splits: only 'train' and 'val' are allowed.
- dev_test_v1 and holdout splits are strictly rejected.
- Fixed seed (0) for simulation dynamics to match Phase 1a fly_intact.
- Atomic writes of float32 .npy arrays and accompanying metadata .json.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import time
import yaml

import numpy as np
import pandas as pd

from src.connectome.schema import ConnectomeGraph
from research.pipeline.flywire_graph import load_flywire_graph, spectral_radius
from research.pipeline.graph_variants import (
    random_matched_graph,
    degree_preserved_scramble,
    well_mixed_scramble,
    weight_shuffled_graph,
    normalize_graph_spectral_radius,
    random_endpoint_graph,
    source_wise_weight_shuffle,
    compute_graph_sha256,
    graph_integrity_report,
)
from research.pipeline.retina import retina_encoder
from research.pipeline.fly_simulator import FlySimulator

ALLOWED_SPLITS = ("train", "val")
ALLOWED_VARIANTS = (
    "real",
    "random",
    "scramble",
    "scramble_mixed",
    "weight_shuffle",
    "random_endpoint",
    "weight_shuffle_source",
)
GENERATOR_VERSION = "g0c_v1"

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"


def validate_split_and_variant(split: str, variant: str) -> None:
    """Enforce strict whitelist rules."""
    if split not in ALLOWED_SPLITS:
        raise ValueError(
            f"Split {split!r} is strictly forbidden. Allowed splits: {ALLOWED_SPLITS}. "
            f"dev_test_v1 and holdout are not allowed."
        )
    if variant not in ALLOWED_VARIANTS:
        raise ValueError(
            f"Variant {variant!r} is invalid. Allowed variants: {ALLOWED_VARIANTS}."
        )


def compute_file_sha256(path: Path | str) -> str:
    """Compute SHA256 hex digest of a file."""
    path = Path(path)
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def compute_bytes_sha256(data: bytes) -> str:
    """Compute SHA256 hex digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()


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


def load_or_create_variant_graph(
    variant: str,
    base_graph: ConnectomeGraph,
    seed: int = 0,
    cache_dir: Path = DATA_DIR / "flywire",
) -> ConnectomeGraph:
    """Load or generate a spectral-radius normalized graph variant."""
    target_rho = spectral_radius(base_graph)
    base_sha = compute_graph_sha256(base_graph)

    if variant == "real":
        return base_graph

    # Cache file naming: new variants include base graph SHA prefix in primary cache key
    if variant in ("random_endpoint", "weight_shuffle_source"):
        cache_file = cache_dir / f"{variant}_seed{seed}_{base_sha[:16]}_cache.npz"
        if not cache_file.exists():
            alt = cache_dir / f"{variant}_seed{seed}_cache.npz"
            if alt.exists():
                cache_file = alt
    else:
        cache_file = cache_dir / f"{variant}_seed{seed}_cache.npz"

    if cache_file.exists():
        print(f"Loading cached {variant} graph from {cache_file}...")
        loaded = np.load(cache_file, allow_pickle=True)

        # Enforce strict provenance verification on new variants
        if variant in ("random_endpoint", "weight_shuffle_source"):
            cached_base_sha = str(loaded["base_graph_sha256"]) if "base_graph_sha256" in loaded else None
            cached_gen_ver = str(loaded["generator_version"]) if "generator_version" in loaded else None

            if cached_base_sha is None or cached_base_sha != base_sha:
                raise ValueError(
                    f"Cache validation failed for {cache_file}: "
                    f"cached base_graph_sha256 {cached_base_sha!r} does not match current {base_sha!r}"
                )
            if cached_gen_ver is None or cached_gen_ver != GENERATOR_VERSION:
                raise ValueError(
                    f"Cache validation failed for {cache_file}: "
                    f"cached generator_version {cached_gen_ver!r} does not match expected {GENERATOR_VERSION!r}"
                )

        import scipy.sparse as sp
        W = sp.csr_matrix(
            (loaded["csr_data"], loaded["csr_indices"], loaded["csr_indptr"]),
            shape=tuple(loaded["csr_shape"]),
            dtype=np.float32,
        )
        meta_dict = {
            **base_graph.meta,
            "variant": variant,
            "variant_seed": seed,
            "spectral_radius": float(loaded["spectral_radius"][0]),
        }
        if "base_graph_sha256" in loaded:
            meta_dict["base_graph_sha256"] = str(loaded["base_graph_sha256"])
        if "graph_sha256" in loaded:
            meta_dict["graph_sha256"] = str(loaded["graph_sha256"])
        if "generator_version" in loaded:
            meta_dict["generator_version"] = str(loaded["generator_version"])
        if "spectral_scaling_factor" in loaded:
            meta_dict["spectral_scaling_factor"] = float(loaded["spectral_scaling_factor"][0])
        if "generation_params_json" in loaded:
            try:
                meta_dict["generation_params"] = json.loads(str(loaded["generation_params_json"]))
            except Exception:
                pass
        if "integrity_report_json" in loaded:
            try:
                meta_dict["integrity_report"] = json.loads(str(loaded["integrity_report_json"]))
            except Exception:
                pass

        if "diag_json" in loaded:
            try:
                diag_payload = json.loads(str(loaded["diag_json"]))
                if "scramble_diagnostics" in diag_payload:
                    meta_dict["scramble_diagnostics"] = diag_payload["scramble_diagnostics"]
                if "weight_shuffle_diagnostics" in diag_payload:
                    meta_dict["weight_shuffle_diagnostics"] = diag_payload["weight_shuffle_diagnostics"]
                if "random_endpoint_diagnostics" in diag_payload:
                    meta_dict["random_endpoint_diagnostics"] = diag_payload["random_endpoint_diagnostics"]
                if "source_wise_shuffle_diagnostics" in diag_payload:
                    meta_dict["source_wise_shuffle_diagnostics"] = diag_payload["source_wise_shuffle_diagnostics"]
            except Exception:
                pass

        return ConnectomeGraph(
            weights=W,
            neuron_ids=list(base_graph.neuron_ids),
            neuron_types=list(base_graph.neuron_types),
            sensory_idx=base_graph.sensory_idx.copy(),
            motor_idx=base_graph.motor_idx.copy(),
            meta=meta_dict,
        )

    print(f"Generating {variant} graph (seed={seed})...")
    t0 = time.time()
    gen_params = {"seed": int(seed), "target_rho": float(target_rho)}
    if variant == "random":
        g_var = random_matched_graph(base_graph, seed=seed)
    elif variant == "scramble":
        # For large graphs (15M edges), 1M swaps was original Phase 5 behavior
        n_multiplier = 0.1 if base_graph.weights.nnz > 1_000_000 else 2.0
        g_var = degree_preserved_scramble(base_graph, seed=seed, n_swap_multiplier=n_multiplier)
    elif variant == "scramble_mixed":
        g_var = well_mixed_scramble(base_graph, seed=seed, target_overlap=0.05)
    elif variant == "weight_shuffle":
        g_var = weight_shuffled_graph(base_graph, seed=seed, separate_signs=True)
    elif variant == "random_endpoint":
        g_var = random_endpoint_graph(base_graph, seed=seed)
        gen_params["algorithm"] = "uniform_no_replacement_rejection"
        gen_params["exclude_self_loops"] = True
    elif variant == "weight_shuffle_source":
        g_var = source_wise_weight_shuffle(base_graph, seed=seed)
        gen_params["algorithm"] = "column_stratified_sign_shuffle"
        gen_params["separate_signs"] = True
    else:
        raise ValueError(f"Unknown variant {variant}")
    t_gen = time.time() - t0
    print(f"Generated {variant} in {t_gen:.2f}s. Normalizing spectral radius to {target_rho:.4f}...")

    g_norm = normalize_graph_spectral_radius(g_var, target_spectral_radius=target_rho)

    # Save cache
    cache_dir.mkdir(parents=True, exist_ok=True)
    save_kwargs = {
        "csr_data": g_norm.weights.data,
        "csr_indices": g_norm.weights.indices,
        "csr_indptr": g_norm.weights.indptr,
        "csr_shape": np.array(g_norm.weights.shape),
        "spectral_radius": np.array([target_rho], dtype=np.float32),
        "base_graph_sha256": np.array(base_sha),
        "generator_version": np.array(GENERATOR_VERSION),
        "target_rho": np.array([target_rho], dtype=np.float32),
        "seed": np.array([seed], dtype=np.int64),
    }
    if "spectral_scaling_factor" in g_norm.meta:
        save_kwargs["spectral_scaling_factor"] = np.array([g_norm.meta["spectral_scaling_factor"]], dtype=np.float32)

    diag_to_save = {}
    for diag_key in (
        "scramble_diagnostics",
        "weight_shuffle_diagnostics",
        "random_endpoint_diagnostics",
        "source_wise_shuffle_diagnostics",
    ):
        if diag_key in g_var.meta:
            diag_to_save[diag_key] = g_var.meta[diag_key]
            g_norm.meta[diag_key] = g_var.meta[diag_key]

    if diag_to_save:
        save_kwargs["diag_json"] = np.array(json.dumps(diag_to_save))

    if variant in ("random_endpoint", "weight_shuffle_source"):
        integrity_rep = graph_integrity_report(g_norm, base=base_graph)
        g_sha = compute_graph_sha256(g_norm)
        save_kwargs["generation_params_json"] = np.array(json.dumps(gen_params))
        save_kwargs["integrity_report_json"] = np.array(json.dumps(integrity_rep))
        save_kwargs["graph_sha256"] = np.array(g_sha)

        g_norm.meta.update({
            "base_graph_sha256": base_sha,
            "generator_version": GENERATOR_VERSION,
            "generation_params": gen_params,
            "integrity_report": integrity_rep,
            "graph_sha256": g_sha,
        })

    np.savez(cache_file, **save_kwargs)
    print(f"Saved {variant} graph cache to {cache_file}.")
    return g_norm


def extract_features(
    variant: str = "real",
    split: str = "val",
    limit: int | None = None,
    backend: str = "scipy",
    chunk_size: int = 256,
    seed: int = 0,
    output_dir: Path = OUTPUTS_DIR / "v2/features",
) -> tuple[np.ndarray, dict]:
    """Extract readout neural activity vectors for a given variant and split."""
    validate_split_and_variant(split, variant)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load samples and select split
    samples = pd.read_parquet(
        DATA_DIR / "samples.parquet",
        filters=[("split", "=", split)],
    )
    split_samples = samples[samples["split"] == split].sort_values("sample_id").reset_index(drop=True)
    if limit is not None and limit > 0:
        split_samples = split_samples.iloc[:limit].copy()

    n_samples = len(split_samples)
    sample_ids = split_samples["sample_id"].to_numpy(dtype=np.int64)
    image_indices = split_samples["image_idx"].to_numpy(dtype=np.int64)

    # 2. Load images
    images_mmap = np.load(DATA_DIR / "images.npy", mmap_mode="r")

    # 3. Load base FlyWire graph and construct/load variant
    base_graph = load_flywire_graph(use_cache=True)
    graph = load_or_create_variant_graph(variant, base_graph, seed=seed)

    # 4. Load mean image and retina encoder
    mean_image_path = OUTPUTS_DIR / "v3/train_mean_image.npy"
    if not mean_image_path.exists():
        raise FileNotFoundError(f"Missing {mean_image_path}")
    mean_image = np.load(mean_image_path)
    mean_image_sha256 = compute_file_sha256(mean_image_path)
    retina = retina_encoder(graph, mean_image)

    # 5. Load dynamics and baseline
    dynamics_path = OUTPUTS_DIR / "v3/dynamics_remote.json"
    if not dynamics_path.exists():
        dynamics_path = OUTPUTS_DIR / "v3/dynamics.json"
    with open(dynamics_path, "r", encoding="utf-8") as f:
        dynamics = json.load(f)

    baseline_path = OUTPUTS_DIR / "v3/baseline_remote.json"
    baseline = None
    baseline_sha256 = None
    if baseline_path.exists():
        with open(baseline_path, "r", encoding="utf-8") as f:
            baseline = json.load(f)
        baseline_sha256 = compute_file_sha256(baseline_path)

    # Simulator configuration (frozen Phase 1a fly_intact dynamics)
    noise_std = float(dynamics.get("noise_std_robustness", 0.0))
    sim = FlySimulator(
        graph=graph,
        seed=seed,
        steps=int(dynamics["steps"]),
        gain=float(dynamics["gain"]),
        leak=float(dynamics["leak"]),
        noise_std=noise_std,
        encoder=retina,
        baseline=baseline,
        direct_currents=True,
        backend=backend,
    )

    n_readout = len(sim.buy_motor_idx) + len(sim.sell_motor_idx)
    features = np.empty((n_samples, n_readout), dtype=np.float32)

    print(
        f"Extracting features for variant={variant}, split={split} "
        f"({n_samples} samples, {n_readout} dims, backend={backend}, chunk={chunk_size})..."
    )
    t_start = time.perf_counter()
    consistency_passed = True
    max_recon_diff = 0.0

    for start_idx in range(0, n_samples, chunk_size):
        end_idx = min(start_idx + chunk_size, n_samples)
        chunk_img_idx = image_indices[start_idx:end_idx]
        chunk_images = np.asarray(images_mmap[chunk_img_idx])

        # Run simulation with return_readout=True
        sim_out = sim.run(chunk_images, return_readout=True)
        chunk_readout = sim_out["readout"]
        features[start_idx:end_idx] = chunk_readout

        # Verify numerical sanity
        if not np.isfinite(chunk_readout).all():
            raise FloatingPointError(f"NaN or Inf encountered in readout at chunk [{start_idx}:{end_idx}]")

        # Consistency check against buy_score / sell_score
        if baseline is not None:
            n_b = len(sim.buy_motor_idx)
            n_s = len(sim.sell_motor_idx)
            buy_rec = (chunk_readout[:, :n_b].mean(axis=1) - baseline["buy_mean"]) / baseline["buy_std"]
            sell_rec = (chunk_readout[:, n_b:].mean(axis=1) - baseline["sell_mean"]) / baseline["sell_std"]
            diff_b = float(np.max(np.abs(buy_rec - sim_out["buy_score"])))
            diff_s = float(np.max(np.abs(sell_rec - sim_out["sell_score"])))
            chunk_max_diff = max(diff_b, diff_s)
            max_recon_diff = max(max_recon_diff, chunk_max_diff)
            if chunk_max_diff > 1e-4:
                consistency_passed = False
                print(f"WARNING: Consistency check mismatch in chunk [{start_idx}:{end_idx}]: diff={chunk_max_diff}")

        elapsed = time.perf_counter() - t_start
        fps = end_idx / max(elapsed, 1e-5)
        if end_idx % (chunk_size * 4) == 0 or end_idx == n_samples:
            print(f"  Processed {end_idx:,}/{n_samples:,} samples ({fps:.1f} samples/s)")

    total_seconds = time.perf_counter() - t_start

    # Check for activity collapse
    feature_stds = np.std(features, axis=0)
    flat_ratio = float(np.mean(feature_stds < 1e-8))
    if flat_ratio > 0.90:
        raise RuntimeError(f"Activity collapse detected: {flat_ratio:.1%} of readout features have zero variance!")

    # 6. Save atomic float32 npy
    npy_path = output_dir / f"{variant}_{split}.npy"
    with tempfile.NamedTemporaryFile("wb", dir=output_dir, delete=False) as tmp:
        np.save(tmp, features)
        tmp_name = tmp.name
    Path(tmp_name).replace(npy_path)
    npy_sha256 = compute_file_sha256(npy_path)

    # 7. Save metadata JSON
    meta = {
        "variant": variant,
        "split": split,
        "n_samples": n_samples,
        "feature_dim": n_readout,
        "readout_buy_neurons": int(len(sim.buy_motor_idx)),
        "readout_sell_neurons": int(len(sim.sell_motor_idx)),
        "sample_id_first": int(sample_ids[0]) if n_samples > 0 else None,
        "sample_id_last": int(sample_ids[-1]) if n_samples > 0 else None,
        "sample_ids_sha256": compute_bytes_sha256(sample_ids.tobytes()),
        "graph_spectral_radius": float(spectral_radius(graph)),
        "dynamics": {
            "steps": int(dynamics["steps"]),
            "gain": float(dynamics["gain"]),
            "leak": float(dynamics["leak"]),
            "noise_std": noise_std,
            "seed": seed,
        },
        "backend": backend,
        "train_mean_image_sha256": mean_image_sha256,
        "baseline_sha256": baseline_sha256,
        "features_file": npy_path.name,
        "features_sha256": npy_sha256,
        "flat_feature_ratio": flat_ratio,
        "consistency_check_passed": consistency_passed,
        "max_reconstruction_diff": max_recon_diff,
        "extraction_seconds": float(total_seconds),
        "samples_per_second": float(n_samples / max(total_seconds, 1e-5)),
        "git_commit": get_git_commit(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    if "scramble_diagnostics" in graph.meta:
        meta["scramble_diagnostics"] = graph.meta["scramble_diagnostics"]
    if "weight_shuffle_diagnostics" in graph.meta:
        meta["weight_shuffle_diagnostics"] = graph.meta["weight_shuffle_diagnostics"]
    if "random_endpoint_diagnostics" in graph.meta:
        meta["random_endpoint_diagnostics"] = graph.meta["random_endpoint_diagnostics"]
    if "source_wise_shuffle_diagnostics" in graph.meta:
        meta["source_wise_shuffle_diagnostics"] = graph.meta["source_wise_shuffle_diagnostics"]

    # Graph provenance and integrity report fields (required for new variants)
    if "graph_sha256" in graph.meta:
        meta["graph_sha256"] = str(graph.meta["graph_sha256"])
    elif variant in ("random_endpoint", "weight_shuffle_source"):
        meta["graph_sha256"] = compute_graph_sha256(graph)

    if "generator_version" in graph.meta:
        meta["generator_version"] = str(graph.meta["generator_version"])
    elif variant in ("random_endpoint", "weight_shuffle_source"):
        meta["generator_version"] = GENERATOR_VERSION

    if "integrity_report" in graph.meta:
        rep = graph.meta["integrity_report"]
        meta["integrity_report_summary"] = {
            "n_neurons": rep.get("n_neurons"),
            "nnz": rep.get("nnz"),
            "self_loops": rep.get("self_loops"),
            "is_canonical_csr": rep.get("is_canonical_csr"),
            "degree_equal": rep.get("degree_equal"),
            "source_out_strength_equal": rep.get("source_out_strength_equal"),
            "edge_overlap_ratio": rep.get("edge_overlap_ratio"),
            "weights_multiset_equal": rep.get("weights_multiset_equal"),
        }
    elif variant in ("random_endpoint", "weight_shuffle_source"):
        rep = graph_integrity_report(graph, base=base_graph)
        meta["integrity_report_summary"] = {
            "n_neurons": rep.get("n_neurons"),
            "nnz": rep.get("nnz"),
            "self_loops": rep.get("self_loops"),
            "is_canonical_csr": rep.get("is_canonical_csr"),
            "degree_equal": rep.get("degree_equal"),
            "source_out_strength_equal": rep.get("source_out_strength_equal"),
            "edge_overlap_ratio": rep.get("edge_overlap_ratio"),
            "weights_multiset_equal": rep.get("weights_multiset_equal"),
        }

    json_path = output_dir / f"{variant}_{split}.json"
    with tempfile.NamedTemporaryFile("w", dir=output_dir, delete=False) as tmp:
        json.dump(meta, tmp, indent=2)
        tmp_json = tmp.name
    Path(tmp_json).replace(json_path)

    print(
        f"Completed {variant}_{split}: shape={features.shape}, "
        f"elapsed={total_seconds:.2f}s ({n_samples/max(total_seconds, 1e-5):.1f} s/s), "
        f"consistency={'PASS' if consistency_passed else 'FAIL'}"
    )
    return features, meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Frozen Connectome readout features.")
    parser.add_argument("--variant", choices=ALLOWED_VARIANTS, default="real")
    parser.add_argument("--split", choices=ALLOWED_SPLITS, default="val")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples (for smoke test)")
    parser.add_argument("--backend", choices=["scipy", "torch"], default="scipy")
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR / "v2/features")
    args = parser.parse_args()

    extract_features(
        variant=args.variant,
        split=args.split,
        limit=args.limit,
        backend=args.backend,
        chunk_size=args.chunk_size,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
