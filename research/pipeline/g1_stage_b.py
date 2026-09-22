"""G1 v2 Stage B Pipeline: Real Single-Graph Capability Diagnostics, Health Gates, and Cost Accounting.

Governing Specification: research/G1_SPEC_v2_draft.md (§3.2, §4, §6, §8, §9)
Review & Task Directives: scratchpad/task_stageB_runner.md, REMOTE_TASK_G1b.md

Core Protocol:
1. Load graph:
   - 'real': load_flywire_graph, select_r16_indices, verify DN readout set is buy_idx U sell_idx
     with exact count == 1291. Provenance mismatch immediately halts with RUN_INVALID.
   - 'fixture': small synthetic connectome marked as fixture.
   - Spectral scaling to target rho=0.95 with fail-closed diagnostics and hash preservation.
2. Generate Diagnostic-fit (10 streams x 2000 valid + 500 washout) and Diagnostic-check (10 streams)
   strictly via per-split manifest API using fresh namespaces. Accessing Main-* or Test raises.
3. Stateful simulation:
   - Scipy (CPU) or PyTorch (GPU/CPU with --device cpu|cuda, --require-cuda).
   - Only materialize DN states (never full T x N state arrays).
   - Streaming dynamics health gates: post-washout non-sensory saturation (|x| > 0.90) < 5%,
     forgetting on fit streams (40 pairs, >= 38 pairs d_rel < 1e-3 at t=500), finite checks.
4. Capability diagnostics:
   - Predictors: DN states x[t+1] post-washout.
   - Targets: d1[t] = v[t-9] (linear lag-9), d2[t] = v[t]*v[t-9] (pure product), where v[t] = u[t] - 0.25.
   - Ridge: float64 thin SVD, fit-only standardization (drop std < 1e-8), 5-fold sequence CV
     over alpha in {0, 1e-8, ..., 1e2}, refit on all 10 fit sequences.
   - Check evaluation: pooled R^2 and 2,000 sequence bootstrap 95% two-sided percentile CI.
   - Capability gate: Dual AND condition (both d1 and d2 have R^2 >= 0.10 and CI lower bound > 0.0).
   - Boundary checks: alpha == 1e2 is no-go (VALID_GATE_FAIL); alpha == 0 is diagnostic OLS.
   - SVD diagnostics: active_p, effective_rank, retained_condition_number, n_discarded_directions.
5. Controls:
   - Negative control: within-split cyclic shift by 1 sequence, refit Ridge on shifted fit,
     evaluate improvement G on check; gate requires CI lower bound <= 0.0 for both targets.
   - Oracle positive control: direct lag/product access achieves R^2 >= 1.0 - 1e-6.
6. Timing, extrapolation, and budget ledger:
   - Measure simulation wall time, time per step, time per sequence, Ridge time, bootstrap time,
     peak RAM (MB), and peak VRAM (MB).
   - Extrapolate Stage C (2,467,500 updates) and Stage D (1,155,000 updates) GPU hours.
   - Separately list projected CPU time for Ridge CV, I/O, and bootstrap.
   - Check hard limits (B <= 1 GPU h, C <= 4, D <= 3, Total <= 8).
7. Output JSON schema:
   - Research path: research/outputs/v3/g1_stage_b_{graph}_{device}.json.
   - Valid gate fail vs run invalid clearly distinguished.
   - Audit hashes: spec_sha256, source_snapshot_sha256, manifest_sha256, graph_sha256 (unscaled/scaled).
   - STRICT PROHIBITION: Contains NO NARMA prediction scores or metrics.
8. Stop rules:
   - Any valid gate failure -> exit code 1 (VALID_GATE_FAIL).
   - Any provenance error or unhandled bug -> exit code 2 (RUN_INVALID).
   - All gates pass -> exit code 0 (STAGE_COMPLETED).
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time
from typing import Any, Callable, Sequence
import warnings

import numpy as np
import scipy.sparse as sp

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from research.pipeline.g1_manifest import (
    G1Manifest,
    create_g1_manifest,
    generate_split_sequences,
    compute_file_sha256,
    compute_source_snapshot_sha,
)
from research.pipeline.g1_runner import (
    BudgetLedger,
    RunInvalidIncident,
    STAGE_BUDGET_LIMITS,
    TOTAL_BUDGET_LIMITS,
)
from research.pipeline.g1_v2 import (
    ALPHA_GRID,
    UPPER_BOUND_ALPHA,
    build_diagnostic_targets,
    compute_diagnostic_sequence_bootstrap,
    compute_pooled_r2,
    compute_pooled_nmse,
    evaluate_capability_gate,
    evaluate_diagnostic_negative_control,
)
from research.pipeline.g1_bench import (
    G1RidgeReadout,
    check_forgetting_gate,
    check_saturation_gate,
    select_r16_indices,
)
from research.pipeline.g1_reservoir import (
    ScipyG1Reservoir,
    TorchG1Reservoir,
    scale_weights_to_spectral_radius,
)
from research.pipeline.graph_variants import compute_graph_sha256
from src.connectome.schema import ConnectomeGraph
from src.connectome.synthetic import make_synthetic_connectome


# ==============================================================================
# Constants & Defaults (§3.2, §4, §8)
# ==============================================================================

EXPECTED_DN_COUNT: int = 1291
DEFAULT_TARGET_RHO: float = 0.95
DEFAULT_LEAK: float = 0.5
STAGE_B_UPDATES: int = 50000          # 20 sequences x 2500 steps
STAGE_C_UPDATES: int = 2467500        # REMOTE_TASK_G1a §3: 21 graphs x 117,500 steps
STAGE_D_UPDATES: int = 1155000        # REMOTE_TASK_G1a §3: 21 graphs x 55,000 steps
TOTAL_PROJECT_UPDATES: int = 3672500  # Total across all stages


# ==============================================================================
# Provenance & Audit Exception Classes
# ==============================================================================

class ProvenanceError(Exception):
    """Raised when connectome provenance, DN count, or metadata integrity checks fail."""
    pass


class SplitAccessError(PermissionError):
    """Raised when unauthorized splits are accessed during Stage B."""
    pass


# ==============================================================================
# Helper Functions: Memory, Split Access, Oracle Controls
# ==============================================================================

def get_peak_ram_mb() -> float:
    """Return peak process resident memory in megabytes."""
    try:
        rusage = resource.getrusage(resource.RUSAGE_SELF)
        if sys.platform == "darwin":
            return float(rusage.ru_maxrss / (1024.0 * 1024.0))
        return float(rusage.ru_maxrss / 1024.0)
    except Exception:
        return 0.0


def get_peak_vram_mb() -> float:
    """Return peak GPU allocated memory in megabytes (if PyTorch and CUDA available)."""
    if HAS_TORCH and torch.cuda.is_available():
        return float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))
    return 0.0


def access_stage_b_split(split_name: str, manifest: G1Manifest) -> list[dict[str, np.ndarray]]:
    """Strictly guard sequence access during Stage B (§3.2, §8).

    Only 'diagnostic-fit' and 'diagnostic-check' are permitted.
    Access to 'main-test', 'main-train', 'main-val', 'calibration' or unknown splits
    raises SplitAccessError immediately.
    """
    clean = str(split_name).strip().lower().replace("_", "-")
    if clean not in ("diagnostic-fit", "diagnostic-check"):
        raise SplitAccessError(
            f"Stage B is strictly prohibited from accessing split '{split_name}'. "
            "Only 'diagnostic-fit' and 'diagnostic-check' are authorized during Stage B (SPEC §3.2, §8)."
        )
    return generate_split_sequences(clean, manifest)


def compute_independent_reference_diagnostic_targets(
    u: Sequence[float] | np.ndarray,
    washout: int = 500,
    u_negative: Sequence[float] | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute reference d1 and d2 directly from input sequence u using an independent implementation.

    This function does NOT call or share code with `g1_v2.py::build_diagnostic_targets`.
    It evaluates the exact mathematical definitions directly in float64:
      v[t] = u[t] - 0.25 for t >= 0
      v[t] = (u_negative[t+9] - 0.25) if u_negative is given else -0.25 for t in [-9..-1]
      d1[t] = v[t - 9]
      d2[t] = v[t] * v[t - 9]

    Returns:
      (ref_d1, ref_d2) sliced after washout: ref_d1[washout:], ref_d2[washout:].
    """
    u_seq = [float(x) for x in u]
    T = len(u_seq)
    if T <= washout:
        raise ValueError(f"Sequence length {T} must be strictly greater than washout {washout}")

    if u_negative is not None:
        neg_vals = [float(x) - 0.25 for x in u_negative]
        if len(neg_vals) != 9:
            raise ValueError(f"Expected u_negative of length 9, got {len(neg_vals)}")
    else:
        neg_vals = [-0.25] * 9

    full_v = neg_vals + [x - 0.25 for x in u_seq]
    # In full_v, index 9 corresponds to t = 0.
    # For any t >= 0:
    # v[t] is at full_v[9 + t]
    # v[t - 9] is at full_v[9 + t - 9] = full_v[t]
    ref_d1 = np.array([full_v[t] for t in range(washout, T)], dtype=np.float64)
    ref_d2 = np.array([full_v[9 + t] * full_v[t] for t in range(washout, T)], dtype=np.float64)
    return ref_d1, ref_d2


def evaluate_stage_b_oracle_controls(
    check_targets_d1: Sequence[np.ndarray],
    check_targets_d2: Sequence[np.ndarray],
    check_sequences: Sequence[dict[str, Any]] | None = None,
    reference_targets_d1: Sequence[np.ndarray] | None = None,
    reference_targets_d2: Sequence[np.ndarray] | None = None,
    rel_tol: float = 1e-10,
    min_r2: float = 1.0 - 1e-6,
) -> dict[str, Any]:
    """Evaluate oracle positive controls for d1 and d2 against independent reference recurrence (§6).

    Replaces deprecated self-comparison identity (R^2(y, y) = 1) with point-by-point comparison
    against independent float64 calculations from raw input u.
    """
    y_d1 = np.concatenate(check_targets_d1)
    y_d2 = np.concatenate(check_targets_d2)

    if check_sequences is not None:
        ref_d1_list = []
        ref_d2_list = []
        for s in check_sequences:
            r1, r2 = compute_independent_reference_diagnostic_targets(
                u=s["u"],
                washout=s.get("washout", 500),
                u_negative=s.get("u_negative", None),
            )
            ref_d1_list.append(r1)
            ref_d2_list.append(r2)
        y_ref_d1 = np.concatenate(ref_d1_list)
        y_ref_d2 = np.concatenate(ref_d2_list)
        mode = "independent_reference"
    elif reference_targets_d1 is not None and reference_targets_d2 is not None:
        y_ref_d1 = np.concatenate(reference_targets_d1)
        y_ref_d2 = np.concatenate(reference_targets_d2)
        mode = "independent_reference"
    else:
        warnings.warn(
            "Self-comparison oracle is deprecated. Pass check_sequences or reference_targets "
            "for independent reference verification.",
            DeprecationWarning,
            stacklevel=2,
        )
        y_ref_d1 = y_d1
        y_ref_d2 = y_d2
        mode = "deprecated_self_comparison"

    # Pointwise absolute and relative differences
    d1_abs_err = np.abs(y_d1 - y_ref_d1)
    d2_abs_err = np.abs(y_d2 - y_ref_d2)
    d1_max_abs = float(np.max(d1_abs_err))
    d2_max_abs = float(np.max(d2_abs_err))

    denom_d1 = np.abs(y_ref_d1)
    denom_d2 = np.abs(y_ref_d2)
    d1_max_rel = float(np.max(d1_abs_err / np.maximum(denom_d1, 1e-12)))
    d2_max_rel = float(np.max(d2_abs_err / np.maximum(denom_d2, 1e-12)))

    # Pooled R^2 against reference
    r2_d1 = compute_pooled_r2(y_ref_d1, y_d1)
    r2_d2 = compute_pooled_r2(y_ref_d2, y_d2)

    passed_d1 = bool(r2_d1 >= min_r2 and d1_max_rel <= rel_tol)
    passed_d2 = bool(r2_d2 >= min_r2 and d2_max_rel <= rel_tol)

    return {
        "passed": bool(passed_d1 and passed_d2),
        "mode": mode,
        "d1_teacher_r2": float(r2_d1),
        "d2_teacher_r2": float(r2_d2),
        "d1_max_abs_diff": d1_max_abs,
        "d2_max_abs_diff": d2_max_abs,
        "d1_max_rel_diff": d1_max_rel,
        "d2_max_rel_diff": d2_max_rel,
        "min_r2": float(min_r2),
        "max_rel_tol": float(rel_tol),
    }


# ==============================================================================
# Synthetic Connectome Fixture Generator
# ==============================================================================

def make_stage_b_fixture_graph(
    n_neurons: int = 100,
    n_sensory: int = 30,
    n_motor: int = 30,
    seed: int = 42,
    mode: str = "go",
) -> ConnectomeGraph:
    """Create a synthetic connectome fixture for Stage B testing and validation.

    Modes:
        - 'go': Sensory and DN configured to pass all capability, health, and negative gates.
        - 'disconnected_dn': DN nodes receive 0 incoming weights -> capability fails.
        - 'high_saturation': Feedforward drive causing non-sensory saturation > 5% -> health fails.
        - 'provenance_fail': Schema mismatch (e.g. DN count != 1291 for real emulation).
    """
    rng = np.random.default_rng(seed)

    if mode == "high_saturation":
        # Create a graph where sensory node projects with huge weight to non-sensory nodes,
        # but rho remains 0.95 via cycle (0, 1)
        W_dense = np.zeros((n_neurons, n_neurons), dtype=np.float32)
        W_dense[0, 1] = 0.95
        W_dense[1, 0] = 0.95
        W_dense[2:n_neurons, 0] = 50.0  # Massive feedforward drive to non-sensory
        sensory_idx = np.array([0], dtype=np.int64)
        dn_idx = np.arange(2, min(n_neurons, 32), dtype=np.int64)
    elif mode == "disconnected_dn":
        W_dense = rng.normal(0, 0.4, size=(n_neurons, n_neurons)).astype(np.float32) * (
            rng.random((n_neurons, n_neurons)) < 0.2
        )
        np.fill_diagonal(W_dense, 0.0)
        sensory_idx = np.arange(n_sensory, dtype=np.int64)
        dn_idx = np.arange(n_sensory, n_sensory + n_motor, dtype=np.int64)
        W_dense[dn_idx, :] = 0.0  # Disconnected DN
    else:  # 'go' or other
        W_dense = rng.normal(0, 0.4, size=(n_neurons, n_neurons)).astype(np.float32) * (
            rng.random((n_neurons, n_neurons)) < 0.2
        )
        np.fill_diagonal(W_dense, 0.0)
        # In go fixture, sensory and DN overlap so both d1 and d2 have sufficient non-linear drive
        sensory_idx = np.arange(n_sensory, dtype=np.int64)
        dn_idx = np.arange(n_motor, dtype=np.int64)

    W = sp.csr_matrix(W_dense, dtype=np.float32)
    n_dn = len(dn_idx)
    buy_idx = dn_idx[: n_dn // 2]
    sell_idx = dn_idx[n_dn // 2 :]

    u_coords = np.zeros(len(sensory_idx), dtype=np.float32)
    v_coords = np.zeros(len(sensory_idx), dtype=np.float32)
    ptypes = np.array(["R1-6"] * len(sensory_idx))

    neuron_ids = [f"SYN_{i:04d}" for i in range(n_neurons)]
    neuron_types = ["interneuron"] * n_neurons
    for s in sensory_idx:
        neuron_types[s] = "photoreceptor"
    for d in dn_idx:
        neuron_types[d] = "descending"

    meta = {
        "source": "fixture",
        "is_fixture": True,
        "seed": seed,
        "mode": mode,
        "buy_idx": buy_idx,
        "sell_idx": sell_idx,
        "u": u_coords,
        "v": v_coords,
        "photoreceptor_type": ptypes,
    }

    if mode == "provenance_fail":
        meta["source"] = "flywire_v783"
        meta["is_fixture"] = False

    return ConnectomeGraph(
        weights=W,
        neuron_ids=neuron_ids,
        neuron_types=neuron_types,
        sensory_idx=sensory_idx,
        motor_idx=dn_idx,
        meta=meta,
    )


# ==============================================================================
# Stage B Execution Engine
# ==============================================================================

def execute_stage_b(
    graph_type: str = "fixture",
    device: str = "cpu",
    require_cuda: bool = False,
    engine: str = "auto",
    repo_root: str | Path | None = None,
    output_path: str | Path | None = None,
    fixture_mode: str = "go",
    n_bootstraps: int = 2000,
    custom_graph: ConnectomeGraph | None = None,
) -> tuple[dict[str, Any], int]:
    """Execute complete Stage B Capability Diagnostic and Cost Ledger Protocol.

    Parameters
    ----------
    graph_type: 'real' or 'fixture'
    device: 'cpu' or 'cuda'
    require_cuda: bool, raise if CUDA is unavailable
    engine: 'torch', 'scipy', or 'auto'
    repo_root: root path of repository
    output_path: custom output JSON destination (if None, writes to research/outputs/v3/...)
    fixture_mode: 'go', 'disconnected_dn', 'high_saturation', or 'provenance_fail'
    n_bootstraps: number of sequence bootstrap iterations (default 2000)
    custom_graph: optional pre-loaded ConnectomeGraph (useful for tests)

    Returns
    -------
    (report_dict, exit_code)
        exit_code 0: STAGE_COMPLETED
        exit_code 1: VALID_GATE_FAIL
        exit_code 2: RUN_INVALID
    """
    t_stage_start = time.perf_counter()
    timestamp_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]

    # Resolve output file path
    if output_path is not None:
        target_json_path = Path(output_path)
    else:
        target_json_path = root / "research" / "outputs" / "v3" / f"g1_stage_b_{graph_type}_{device}.json"

    # Environment details
    env_info = {
        "os": sys.platform,
        "platform": platform.platform(),
        "python_version": sys.version,
        "torch_version": torch.__version__ if HAS_TORCH else None,
        "cuda_available": bool(HAS_TORCH and torch.cuda.is_available()),
        "cuda_device_name": torch.cuda.get_device_name(0) if HAS_TORCH and torch.cuda.is_available() else None,
        "requested_device": device,
        "require_cuda": require_cuda,
    }

    # Manifest initialization & split sealing
    manifest = create_g1_manifest(root)
    manifest.advance_stage("B")

    spec_sha = manifest.spec_sha256
    source_snap_sha = manifest.source_snapshot_sha256
    manifest_sha = manifest.manifest_sha256 or manifest.compute_sha256()

    report: dict[str, Any] = {
        "stage": "B",
        "timestamp": timestamp_iso,
        "status": "RUN_INVALID",
        "conclusion": "RUN_INVALID",
        "exit_code": 2,
        "environment": env_info,
        "hashes": {
            "spec_sha256": spec_sha,
            "source_snapshot_sha256": source_snap_sha,
            "manifest_sha256": manifest_sha,
            "graph_sha256_unscaled": None,
            "graph_sha256_scaled": None,
        },
        "provenance": {},
        "health_gates": {},
        "capability_gates": {},
        "negative_control": {},
        "oracle_controls": {},
        "timing_and_budget": {},
    }

    # --------------------------------------------------------------------------
    # 1. Connectome Loading & Provenance Verification
    # --------------------------------------------------------------------------
    try:
        if custom_graph is not None:
            graph = custom_graph
            is_fixture = bool(getattr(graph, "meta", {}).get("is_fixture", True))
        elif graph_type == "fixture":
            graph = make_stage_b_fixture_graph(mode=fixture_mode)
            is_fixture = bool(graph.meta.get("is_fixture", True))
        elif graph_type == "real":
            # Safety check: On Darwin (macOS), warn or disallow full FlyWire graph execution
            # Per task directives: "Mac 上只用小圖/CPU fixture，不要在真實 FlyWire 全圖上執行"
            if sys.platform == "darwin" and os.environ.get("ALLOW_FLYWIRE_ON_MAC") != "1":
                raise ProvenanceError(
                    "Execution on real full FlyWire graph on macOS is strictly prohibited "
                    "per task directives (GPU / Windows RTX 5070 only). Set ALLOW_FLYWIRE_ON_MAC=1 to bypass."
                )
            from research.pipeline.flywire_graph import load_flywire_graph
            graph = load_flywire_graph(use_cache=True)
            is_fixture = False
        else:
            raise ValueError(f"Unknown graph_type: '{graph_type}'. Must be 'real' or 'fixture'.")

        # Provenance verification of R1-6 sensory selection and DN readout set
        r16_idx, r16_summary = select_r16_indices(graph)
        graph_unscaled_sha = r16_summary["graph_sha256"]
        report["hashes"]["graph_sha256_unscaled"] = graph_unscaled_sha

        buy_idx = graph.meta.get("buy_idx")
        sell_idx = graph.meta.get("sell_idx")
        if buy_idx is None or sell_idx is None:
            raise ProvenanceError("Missing buy_idx or sell_idx in graph metadata.")

        dn_idx = np.union1d(np.asarray(buy_idx, dtype=np.int64), np.asarray(sell_idx, dtype=np.int64))

        if not is_fixture:
            if len(dn_idx) != EXPECTED_DN_COUNT:
                raise ProvenanceError(
                    f"Provenance blocked: DN count {len(dn_idx)} != expected {EXPECTED_DN_COUNT} "
                    "(buy_idx U sell_idx). Node substitution strictly prohibited."
                )

        # Spectral radius scaling (fail-closed)
        w_scaled, target_rho, unscaled_rho, scale_factor, spec_diag = scale_weights_to_spectral_radius(
            weights=graph.weights,
            target_rho=DEFAULT_TARGET_RHO,
            verify=True,
            return_diagnostics=True,
        )
        report["hashes"]["graph_sha256_scaled"] = spec_diag["scaled_sha256"]

        report["provenance"] = {
            "graph_type": "fixture" if is_fixture else "real",
            "n_neurons": int(graph.n_neurons),
            "sensory_indices_count": len(graph.sensory_idx),
            "r16_indices_count": len(r16_idx),
            "r7_r8_excluded": r16_summary.get("n_r7_r8_excluded", 0),
            "dn_indices_count": len(dn_idx),
            "buy_indices_count": len(buy_idx),
            "sell_indices_count": len(sell_idx),
            "spectral_radius_unscaled": float(unscaled_rho),
            "spectral_radius_target": float(target_rho),
            "spectral_radius_verified": float(spec_diag["verified_rho"]),
            "spectral_scale_factor": float(scale_factor),
            "eigensolver_method": spec_diag["unscaled_diagnostics"].get("method", "sparse"),
            "eigensolver_residual": float(spec_diag["unscaled_diagnostics"].get("residual", 0.0)),
        }

    except Exception as e:
        incident = RunInvalidIncident(
            incident_type="provenance_or_split_corruption" if isinstance(e, ProvenanceError) else "code_or_data_defect",
            first_discovered_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            independent_evidence=f"Failed during connectome loading / provenance verification: {e}",
            results_already_viewed=[],
            changed_files=[],
            new_hashes={"manifest": manifest_sha},
            remaining_budget={"remaining_gpu_hours": 8.0, "remaining_person_hours": 32.0},
        )
        report["incident"] = incident.to_dict()
        report["error"] = str(e)
        _write_report(report, target_json_path)
        return report, 2

    # --------------------------------------------------------------------------
    # 2. Sequence Generation (Diagnostic Splits Only)
    # --------------------------------------------------------------------------
    try:
        fit_sequences = access_stage_b_split("diagnostic-fit", manifest)
        check_sequences = access_stage_b_split("diagnostic-check", manifest)
    except Exception as e:
        incident = RunInvalidIncident(
            incident_type="provenance_or_split_corruption",
            first_discovered_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            independent_evidence=f"Failed during split sequence generation: {e}",
            results_already_viewed=[],
            changed_files=[],
            new_hashes={"manifest": manifest_sha},
            remaining_budget={"remaining_gpu_hours": 8.0, "remaining_person_hours": 32.0},
        )
        report["incident"] = incident.to_dict()
        report["error"] = str(e)
        _write_report(report, target_json_path)
        return report, 2

    # --------------------------------------------------------------------------
    # 3. Simulator Setup & Stateful Reservoir Simulation
    # --------------------------------------------------------------------------
    actual_engine: str = ""
    actual_device: str = ""

    if require_cuda and (not HAS_TORCH or not torch.cuda.is_available()):
        incident = RunInvalidIncident(
            incident_type="code_or_data_defect",
            first_discovered_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            independent_evidence="--require-cuda specified but CUDA / PyTorch is not available.",
            results_already_viewed=[],
            changed_files=[],
            new_hashes={},
            remaining_budget={"remaining_gpu_hours": 8.0, "remaining_person_hours": 32.0},
        )
        report["incident"] = incident.to_dict()
        report["error"] = "CUDA is required but unavailable."
        _write_report(report, target_json_path)
        return report, 2

    use_torch = (engine == "torch") or (engine == "auto" and (device == "cuda" or (HAS_TORCH and torch.cuda.is_available())))

    if use_torch:
        if not HAS_TORCH:
            raise ImportError("PyTorch requested but not installed.")
        reservoir = TorchG1Reservoir(
            weights=w_scaled,
            sensory_idx=r16_idx,
            readout_idx=dn_idx,
            target_rho=DEFAULT_TARGET_RHO,
            leak=DEFAULT_LEAK,
            unscaled_rho=DEFAULT_TARGET_RHO,
            device=device,
            require_cuda=require_cuda,
            verify_spectral_radius=False,
        )
        actual_engine = "torch"
        actual_device = reservoir.actual_device
    else:
        reservoir = ScipyG1Reservoir(
            weights=w_scaled,
            sensory_idx=r16_idx,
            readout_idx=dn_idx,
            target_rho=DEFAULT_TARGET_RHO,
            leak=DEFAULT_LEAK,
            unscaled_rho=DEFAULT_TARGET_RHO,
            verify_spectral_radius=False,
        )
        actual_engine = "scipy"
        actual_device = "cpu"

    report["environment"]["actual_engine"] = actual_engine
    report["environment"]["actual_device"] = actual_device

    # Run stateful simulation on fit and check splits
    # Stream saturation gates without full T x N materialization
    t_sim_start = time.perf_counter()
    fit_dn_states: list[np.ndarray] = []
    check_dn_states: list[np.ndarray] = []

    total_saturated_points: int = 0
    total_non_sensory_points: int = 0
    total_non_finite_points: int = 0

    for seq in fit_sequences:
        out_states, final_x, sat_info = reservoir.simulate(
            seq["u"],
            track_saturation=True,
            washout=seq["washout"],
        )
        fit_dn_states.append(out_states[seq["washout"] :])
        total_saturated_points += sat_info["saturated_points"]
        total_non_sensory_points += sat_info["total_non_sensory_points"]
        total_non_finite_points += sat_info["non_finite_count"]

    for seq in check_sequences:
        out_states, final_x, sat_info = reservoir.simulate(
            seq["u"],
            track_saturation=True,
            washout=seq["washout"],
        )
        check_dn_states.append(out_states[seq["washout"] :])
        total_saturated_points += sat_info["saturated_points"]
        total_non_sensory_points += sat_info["total_non_sensory_points"]
        total_non_finite_points += sat_info["non_finite_count"]

    t_sim_end = time.perf_counter()
    sim_wall_time = t_sim_end - t_sim_start

    # Saturation Gate Evaluation
    sat_ratio = float(total_saturated_points / total_non_sensory_points) if total_non_sensory_points > 0 else 0.0
    finite_check_passed = bool(total_non_finite_points == 0)
    sat_passed = bool(finite_check_passed and sat_ratio < 0.05)

    # Initial-state Forgetting Gate Evaluation (on fit sequences)
    t_forget_start = time.perf_counter()
    forgetting_res = check_forgetting_gate(
        sim_fn=lambda u_sub, x0: reservoir.simulate(u_sub, initial_state=x0),
        train_u_sequences=[s["u"] for s in fit_sequences],
        n_neurons=graph.n_neurons,
        seed=42,
    )
    t_forget_end = time.perf_counter()
    forget_wall_time = t_forget_end - t_forget_start

    health_passed = bool(sat_passed and forgetting_res["passed"] and finite_check_passed)
    report["health_gates"] = {
        "passed": health_passed,
        "saturation_gate": {
            "passed": sat_passed,
            "saturation_ratio": sat_ratio,
            "max_allowed_ratio": 0.05,
            "saturated_points": total_saturated_points,
            "total_non_sensory_points": total_non_sensory_points,
            "threshold": 0.90,
        },
        "forgetting_gate": {
            "passed": forgetting_res["passed"],
            "n_passed_pairs": forgetting_res["n_passed_pairs"],
            "total_pairs": forgetting_res["total_pairs"],
            "pass_ratio": forgetting_res["pass_ratio"],
            "min_pass_pairs": forgetting_res["min_pass_pairs"],
            "d_rel_median": forgetting_res["d_rel_median"],
            "d_rel_max": forgetting_res["d_rel_max"],
        },
        "finite_check": {
            "passed": finite_check_passed,
            "non_finite_count": total_non_finite_points,
        },
    }

    # --------------------------------------------------------------------------
    # 4. Capability Diagnostics (d1 and d2 Dual AND Gate)
    # --------------------------------------------------------------------------
    fit_d1 = [build_diagnostic_targets(s["u"], washout=s["washout"])[0] for s in fit_sequences]
    fit_d2 = [build_diagnostic_targets(s["u"], washout=s["washout"])[1] for s in fit_sequences]
    check_d1 = [build_diagnostic_targets(s["u"], washout=s["washout"])[0] for s in check_sequences]
    check_d2 = [build_diagnostic_targets(s["u"], washout=s["washout"])[1] for s in check_sequences]

    t_ridge_start = time.perf_counter()

    # Head 1: Linear delay d1
    ridge_d1 = G1RidgeReadout(alphas=ALPHA_GRID)
    ridge_d1.fit_sequence_group_cv(fit_dn_states, fit_d1, n_folds=5)
    check_preds_d1 = [ridge_d1.predict(cs) for cs in check_dn_states]

    # Head 2: Non-linear pure product d2
    ridge_d2 = G1RidgeReadout(alphas=ALPHA_GRID)
    ridge_d2.fit_sequence_group_cv(fit_dn_states, fit_d2, n_folds=5)
    check_preds_d2 = [ridge_d2.predict(cs) for cs in check_dn_states]

    t_ridge_end = time.perf_counter()
    ridge_wall_time = t_ridge_end - t_ridge_start

    # Sequence bootstrap for check R^2
    t_boot_start = time.perf_counter()
    boot_d1 = compute_diagnostic_sequence_bootstrap(check_d1, check_preds_d1, n_bootstraps=n_bootstraps, seed=42)
    boot_d2 = compute_diagnostic_sequence_bootstrap(check_d2, check_preds_d2, n_bootstraps=n_bootstraps, seed=42)
    t_boot_end = time.perf_counter()
    boot_wall_time = t_boot_end - t_boot_start

    # Boundary rule: alpha == 10^2 -> NO-GO
    alpha_at_upper_d1 = bool(ridge_d1.best_alpha == UPPER_BOUND_ALPHA)
    alpha_at_upper_d2 = bool(ridge_d2.best_alpha == UPPER_BOUND_ALPHA)
    alpha_upper_bound_hit = bool(alpha_at_upper_d1 or alpha_at_upper_d2)

    # Capability Gate Evaluation (Dual AND)
    passed_cap_d1 = bool(boot_d1["r2_point"] >= 0.10 and boot_d1["ci_lower"] > 0.0 and not alpha_at_upper_d1)
    passed_cap_d2 = bool(boot_d2["r2_point"] >= 0.10 and boot_d2["ci_lower"] > 0.0 and not alpha_at_upper_d2)
    capability_passed = bool(passed_cap_d1 and passed_cap_d2)

    report["capability_gates"] = {
        "passed": capability_passed,
        "d1": {
            "passed": passed_cap_d1,
            "target": "v[t-9]",
            "r2_point": float(boot_d1["r2_point"]),
            "ci_lower": float(boot_d1["ci_lower"]),
            "ci_upper": float(boot_d1["ci_upper"]),
            "min_r2": 0.10,
            "best_alpha": float(ridge_d1.best_alpha) if ridge_d1.best_alpha is not None else 0.0,
            "alpha_at_upper_bound": alpha_at_upper_d1,
            "alpha_zero_diagnostic": bool(ridge_d1.cv_results.get("alpha_zero_diagnostic", False)),
            "active_p": int(ridge_d1.cv_results.get("n_active_features", 0)),
            "effective_rank": int(ridge_d1.cv_results.get("effective_rank", 0)),
            "n_discarded_directions": int(ridge_d1.cv_results.get("n_discarded_directions", 0)),
            "retained_condition_number": float(ridge_d1.cv_results.get("retained_condition_number", 1.0)),
        },
        "d2": {
            "passed": passed_cap_d2,
            "target": "v[t]*v[t-9]",
            "r2_point": float(boot_d2["r2_point"]),
            "ci_lower": float(boot_d2["ci_lower"]),
            "ci_upper": float(boot_d2["ci_upper"]),
            "min_r2": 0.10,
            "best_alpha": float(ridge_d2.best_alpha) if ridge_d2.best_alpha is not None else 0.0,
            "alpha_at_upper_bound": alpha_at_upper_d2,
            "alpha_zero_diagnostic": bool(ridge_d2.cv_results.get("alpha_zero_diagnostic", False)),
            "active_p": int(ridge_d2.cv_results.get("n_active_features", 0)),
            "effective_rank": int(ridge_d2.cv_results.get("effective_rank", 0)),
            "n_discarded_directions": int(ridge_d2.cv_results.get("n_discarded_directions", 0)),
            "retained_condition_number": float(ridge_d2.cv_results.get("retained_condition_number", 1.0)),
        },
    }

    # --------------------------------------------------------------------------
    # 5. Negative Control & Oracle Positive Control
    # --------------------------------------------------------------------------
    t_neg_start = time.perf_counter()
    neg_control_res = evaluate_diagnostic_negative_control(
        fit_states=fit_dn_states,
        fit_targets_d1=fit_d1,
        fit_targets_d2=fit_d2,
        check_states=check_dn_states,
        check_targets_d1=check_d1,
        check_targets_d2=check_d2,
        ridge_factory=lambda: G1RidgeReadout(alphas=ALPHA_GRID),
        n_bootstraps=n_bootstraps,
        seed=42,
    )
    t_neg_end = time.perf_counter()
    neg_wall_time = t_neg_end - t_neg_start

    oracle_res = evaluate_stage_b_oracle_controls(
        check_targets_d1=check_d1,
        check_targets_d2=check_d2,
        check_sequences=check_sequences,
    )

    report["negative_control"] = {
        "passed": neg_control_res["passed"],
        "d1": neg_control_res["results"]["d1"],
        "d2": neg_control_res["results"]["d2"],
    }
    report["oracle_controls"] = oracle_res

    # --------------------------------------------------------------------------
    # 6. Timing, Extrapolation & Budget Ledger
    # --------------------------------------------------------------------------
    total_wall_time = time.perf_counter() - t_stage_start
    time_per_sequence = float(sim_wall_time / len(fit_sequences + check_sequences))
    time_per_step = float(sim_wall_time / STAGE_B_UPDATES)

    peak_ram = get_peak_ram_mb()
    peak_vram = get_peak_vram_mb()

    # Actual GPU hours in Stage B
    if actual_device.startswith("cuda"):
        actual_stage_b_gpu_hours = float((sim_wall_time + forget_wall_time) / 3600.0)
    else:
        actual_stage_b_gpu_hours = 0.0

    # Extrapolate Stage C & Stage D GPU hours based on measured throughput
    extrapolated_stage_c_gpu_hours = float((STAGE_C_UPDATES * time_per_step) / 3600.0)
    extrapolated_stage_d_gpu_hours = float((STAGE_D_UPDATES * time_per_step) / 3600.0)
    extrapolated_total_gpu_hours = (
        actual_stage_b_gpu_hours + extrapolated_stage_c_gpu_hours + extrapolated_stage_d_gpu_hours
    )

    # CPU Extrapolations (Ridge, bootstrap, and I/O)
    ridge_time_per_head = float((ridge_wall_time + neg_wall_time) / 4.0)
    # Stage C: 21 graphs x 1 head x 2.5x sample length
    projected_stage_c_ridge_cpu_hours = float((21 * 2.5 * ridge_time_per_head) / 3600.0)
    # Stage C: 1,000 cohorts x 2,000 bootstraps
    bootstrap_rate = float(n_bootstraps / max(boot_wall_time, 1e-4))
    projected_stage_c_calib_cpu_hours = float((1000 * n_bootstraps / bootstrap_rate) / 3600.0)
    projected_cpu_hours_total = projected_stage_c_ridge_cpu_hours + projected_stage_c_calib_cpu_hours

    # Budget Limits Check (§8)
    ledger = BudgetLedger()
    ledger.record_usage("B", person_hours=float(total_wall_time / 3600.0), gpu_hours=actual_stage_b_gpu_hours)

    budget_passed = True
    budget_fail_reasons = []

    if actual_stage_b_gpu_hours > STAGE_BUDGET_LIMITS["B"]["gpu_hours"]:
        budget_passed = False
        budget_fail_reasons.append(
            f"Stage B actual GPU hours {actual_stage_b_gpu_hours:.3f} > limit {STAGE_BUDGET_LIMITS['B']['gpu_hours']}"
        )
    if extrapolated_stage_c_gpu_hours > STAGE_BUDGET_LIMITS["C"]["gpu_hours"]:
        budget_passed = False
        budget_fail_reasons.append(
            f"Stage C extrapolated GPU hours {extrapolated_stage_c_gpu_hours:.3f} > limit {STAGE_BUDGET_LIMITS['C']['gpu_hours']}"
        )
    if extrapolated_stage_d_gpu_hours > STAGE_BUDGET_LIMITS["D"]["gpu_hours"]:
        budget_passed = False
        budget_fail_reasons.append(
            f"Stage D extrapolated GPU hours {extrapolated_stage_d_gpu_hours:.3f} > limit {STAGE_BUDGET_LIMITS['D']['gpu_hours']}"
        )
    if extrapolated_total_gpu_hours > TOTAL_BUDGET_LIMITS["gpu_hours"]:
        budget_passed = False
        budget_fail_reasons.append(
            f"Total extrapolated GPU hours {extrapolated_total_gpu_hours:.3f} > limit {TOTAL_BUDGET_LIMITS['gpu_hours']}"
        )

    report["timing_and_budget"] = {
        "wall_time_simulation_seconds": float(sim_wall_time),
        "wall_time_forgetting_seconds": float(forget_wall_time),
        "wall_time_ridge_seconds": float(ridge_wall_time + neg_wall_time),
        "wall_time_bootstrap_seconds": float(boot_wall_time),
        "wall_time_total_seconds": float(total_wall_time),
        "time_per_sequence_seconds": float(time_per_sequence),
        "time_per_step_seconds": float(time_per_step),
        "peak_ram_mb": float(peak_ram),
        "peak_vram_mb": float(peak_vram),
        "actual_stage_b_gpu_hours": float(actual_stage_b_gpu_hours),
        "extrapolated_stage_c_gpu_hours": float(extrapolated_stage_c_gpu_hours),
        "extrapolated_stage_d_gpu_hours": float(extrapolated_stage_d_gpu_hours),
        "extrapolated_total_gpu_hours": float(extrapolated_total_gpu_hours),
        "projected_cpu_hours_ridge_io_bootstrap": float(projected_cpu_hours_total),
        "budget_limits": {
            "stage_b_gpu_limit": STAGE_BUDGET_LIMITS["B"]["gpu_hours"],
            "stage_c_gpu_limit": STAGE_BUDGET_LIMITS["C"]["gpu_hours"],
            "stage_d_gpu_limit": STAGE_BUDGET_LIMITS["D"]["gpu_hours"],
            "total_gpu_limit": TOTAL_BUDGET_LIMITS["gpu_hours"],
        },
        "budget_passed": budget_passed,
        "budget_fail_reasons": budget_fail_reasons,
    }

    # --------------------------------------------------------------------------
    # 7. Final Stage Decision & Exit Code
    # --------------------------------------------------------------------------
    all_gates_passed = bool(
        health_passed
        and capability_passed
        and neg_control_res["passed"]
        and oracle_res["passed"]
        and budget_passed
    )

    if all_gates_passed:
        report["status"] = "STAGE_COMPLETED"
        report["conclusion"] = "STAGE_COMPLETED"
        report["exit_code"] = 0
    else:
        report["status"] = "VALID_GATE_FAIL"
        report["conclusion"] = "VALID_GATE_FAIL"
        report["exit_code"] = 1

    _write_report(report, target_json_path)
    return report, report["exit_code"]


def _write_report(report: dict[str, Any], path: Path) -> None:
    """Save report JSON to path, ensuring parent directory exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


# ==============================================================================
# CLI Entrypoint
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="G1 v2 Stage B: Real Single-Graph Capability Diagnostics & Cost Accounting"
    )
    parser.add_argument("--graph", choices=["real", "fixture"], default="fixture", help="Graph type to evaluate")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Device for simulation")
    parser.add_argument("--require-cuda", action="store_true", help="Raise if CUDA is unavailable")
    parser.add_argument("--engine", choices=["torch", "scipy", "auto"], default="auto", help="Execution engine")
    parser.add_argument("--output", type=str, default=None, help="Custom output JSON report path")
    parser.add_argument("--repo-root", type=str, default=None, help="Root path of repository")
    parser.add_argument("--fixture-mode", choices=["go", "disconnected_dn", "high_saturation", "provenance_fail"], default="go")
    parser.add_argument("--n-bootstraps", type=int, default=2000, help="Number of bootstrap resamples")

    args = parser.parse_args()

    report, exit_code = execute_stage_b(
        graph_type=args.graph,
        device=args.device,
        require_cuda=args.require_cuda,
        engine=args.engine,
        repo_root=args.repo_root,
        output_path=args.output,
        fixture_mode=args.fixture_mode,
        n_bootstraps=args.n_bootstraps,
    )

    # Print results-only summary
    print(f"STAGE_B_STATUS: {report['status']}")
    print(f"STAGE_B_EXIT_CODE: {exit_code}")
    print(f"HEALTH_GATES_PASSED: {report.get('health_gates', {}).get('passed')}")
    print(f"CAPABILITY_GATES_PASSED: {report.get('capability_gates', {}).get('passed')}")
    print(f"NEGATIVE_CONTROL_PASSED: {report.get('negative_control', {}).get('passed')}")
    print(f"BUDGET_PASSED: {report.get('timing_and_budget', {}).get('budget_passed')}")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
