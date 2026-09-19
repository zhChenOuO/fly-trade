"""FlyWire v783 Connectome Graph loader and caching.

Contract (from research/SPEC_v3.md §8-§10):
- load_flywire_graph(use_cache: bool = True) -> ConnectomeGraph
- W[post, pre] = Excitatory(±1) x Connectivity(突觸數), float32 CSR.
- Neuron order = Completeness_783.csv row order (0..138638).
- neuron_ids = list of str(root_id).
- neuron_types = list of cell_type from annotations.
- sensory_idx = all photoreceptors (R1-6, R7, R8) sorted by root_id ascending.
- motor_idx = descending neurons with side in {'left', 'right'} (center excluded).
- graph.meta contains:
  - buy_idx (left DN indices)
  - sell_idx (right DN indices)
  - photoreceptor_type (array of 'R1-6', 'R7', 'R8')
  - eye (array of 'left', 'right')
  - u, v, uv (retina coordinates in [0, 1]^2, no NaN)
  - spectral_radius (rho(W) float)
- Caching:
  - Saved to research/data/flywire/graph_cache.npz using np.savez to avoid
    re-parsing 15M rows.
- spectral_radius(graph: ConnectomeGraph) -> float using scipy.sparse.linalg.eigs(k=1, which='LM').
"""
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from src.connectome.schema import ConnectomeGraph
from research.pipeline.retina import compute_retina_coordinates

FLYWIRE_DIR = Path(__file__).resolve().parent.parent / "data" / "flywire"
CACHE_PATH = FLYWIRE_DIR / "graph_cache.npz"


def spectral_radius(graph: ConnectomeGraph) -> float:
    """Calculate or retrieve cached spectral radius rho(W) = max |eigenvalue|."""
    if "spectral_radius" in graph.meta and graph.meta["spectral_radius"] is not None:
        return float(graph.meta["spectral_radius"])

    # Compute using scipy.sparse.linalg.eigs
    vals = spla.eigs(graph.weights, k=1, which="LM", return_eigenvectors=False)
    rho = float(np.abs(vals[0]))
    graph.meta["spectral_radius"] = rho
    return rho


def load_flywire_graph(
    data_dir: str | Path | None = None,
    use_cache: bool = True,
    cache_path: str | Path | None = None,
) -> ConnectomeGraph:
    """Load FlyWire v783 connectome graph with retina coordinates and motor indices.

    Parameters
    ----------
    data_dir: str or Path, optional
        Path to directory containing FlyWire CSV/TSV/parquet files.
    use_cache: bool, default True
        Whether to load from pre-built npz cache if available, and save cache if not.
    cache_path: str or Path, optional
        Path to cache npz file. Defaults to research/data/flywire/graph_cache.npz.

    Returns
    -------
    ConnectomeGraph
    """
    if data_dir is None:
        data_dir = FLYWIRE_DIR
    else:
        data_dir = Path(data_dir)

    if cache_path is None:
        cache_path = CACHE_PATH
    else:
        cache_path = Path(cache_path)

    if use_cache and cache_path.exists():
        loaded = np.load(cache_path, allow_pickle=True)
        W = sp.csr_matrix(
            (loaded["csr_data"], loaded["csr_indices"], loaded["csr_indptr"]),
            shape=tuple(loaded["csr_shape"]),
            dtype=np.float32,
        )
        neuron_ids = [str(x) for x in loaded["neuron_ids"]]
        neuron_types = [str(x) for x in loaded["neuron_types"]]
        sensory_idx = loaded["sensory_idx"].astype(np.int64)
        motor_idx = loaded["motor_idx"].astype(np.int64)

        meta = {
            "source": "flywire_v783",
            "buy_idx": loaded["buy_idx"].astype(np.int64),
            "sell_idx": loaded["sell_idx"].astype(np.int64),
            "photoreceptor_type": loaded["photoreceptor_type"].astype(str),
            "eye": loaded["eye"].astype(str),
            "u": loaded["u"].astype(np.float32),
            "v": loaded["v"].astype(np.float32),
            "uv": loaded["uv"].astype(np.float32),
            "spectral_radius": float(loaded["spectral_radius"][0]),
        }

        return ConnectomeGraph(
            weights=W,
            neuron_ids=neuron_ids,
            neuron_types=neuron_types,
            sensory_idx=sensory_idx,
            motor_idx=motor_idx,
            meta=meta,
        )

    # Cache miss or use_cache=False: Parse raw files
    completeness_file = data_dir / "Completeness_783.csv"
    annotations_file = data_dir / "Supplemental_file1_neuron_annotations.tsv"
    connectivity_file = data_dir / "Connectivity_783.parquet"

    if not completeness_file.exists():
        raise FileNotFoundError(f"Missing {completeness_file}")
    if not annotations_file.exists():
        raise FileNotFoundError(f"Missing {annotations_file}")
    if not connectivity_file.exists():
        raise FileNotFoundError(f"Missing {connectivity_file}")

    # 1. Read Completeness to establish neuron indexing 0..N-1
    comp_df = pd.read_csv(completeness_file)
    comp_ids = comp_df.iloc[:, 0].astype("int64").values
    N = len(comp_ids)
    neuron_ids = [str(rid) for rid in comp_ids]

    comp_lookup = pd.DataFrame({"root_id": comp_ids, "comp_idx": np.arange(N, dtype=np.int64)})

    # 2. Read annotations and merge with Completeness order
    anno_df = pd.read_csv(annotations_file, sep="\t", low_memory=False)
    anno_df["root_id"] = anno_df["root_id"].astype("int64")

    merged = pd.merge(comp_lookup, anno_df, on="root_id", how="left")
    neuron_types = merged["cell_type"].fillna("unknown").astype(str).tolist()

    # 3. Sensory neurons: R1-6, R7, R8, sorted strictly by root_id ascending
    pr_mask = merged["cell_type"].isin(["R1-6", "R7", "R8"])
    pr_df = merged[pr_mask].sort_values("root_id").reset_index(drop=True)
    sensory_idx = pr_df["comp_idx"].values.astype(np.int64)

    # 4. Compute retina coordinates (u, v) for photoreceptors
    retina_coords = compute_retina_coordinates(pr_df)

    # 5. Motor neurons: descending neurons on left and right side
    dn_left = merged[(merged["super_class"] == "descending") & (merged["side"] == "left")]
    dn_right = merged[(merged["super_class"] == "descending") & (merged["side"] == "right")]
    buy_idx = dn_left["comp_idx"].values.astype(np.int64)
    sell_idx = dn_right["comp_idx"].values.astype(np.int64)
    motor_idx = np.concatenate([buy_idx, sell_idx]).astype(np.int64)

    # 6. Read connectivity parquet and build float32 CSR
    conn_table = pq.read_table(
        connectivity_file,
        columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"],
    )
    pre = conn_table["Presynaptic_Index"].to_numpy()
    post = conn_table["Postsynaptic_Index"].to_numpy()
    data = conn_table["Excitatory x Connectivity"].to_numpy().astype(np.float32)

    # W[post, pre] = Excitatory * Connectivity
    W = sp.csr_matrix((data, (post, pre)), shape=(N, N), dtype=np.float32)

    # 7. Compute spectral radius
    vals = spla.eigs(W, k=1, which="LM", return_eigenvectors=False)
    rho = float(np.abs(vals[0]))

    meta = {
        "source": "flywire_v783",
        "buy_idx": buy_idx,
        "sell_idx": sell_idx,
        "photoreceptor_type": retina_coords["photoreceptor_type"],
        "eye": retina_coords["eye"],
        "u": retina_coords["u"],
        "v": retina_coords["v"],
        "uv": retina_coords["uv"],
        "spectral_radius": rho,
    }

    graph = ConnectomeGraph(
        weights=W,
        neuron_ids=neuron_ids,
        neuron_types=neuron_types,
        sensory_idx=sensory_idx,
        motor_idx=motor_idx,
        meta=meta,
    )

    # 8. Save cache if enabled
    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache_path,
            csr_data=W.data,
            csr_indices=W.indices,
            csr_indptr=W.indptr,
            csr_shape=np.array(W.shape),
            neuron_ids=np.array(neuron_ids, dtype=object),
            neuron_types=np.array(neuron_types, dtype=object),
            sensory_idx=sensory_idx,
            motor_idx=motor_idx,
            buy_idx=buy_idx,
            sell_idx=sell_idx,
            photoreceptor_type=retina_coords["photoreceptor_type"],
            eye=retina_coords["eye"],
            u=retina_coords["u"],
            v=retina_coords["v"],
            uv=retina_coords["uv"],
            spectral_radius=np.array([rho], dtype=np.float64),
        )

    return graph
