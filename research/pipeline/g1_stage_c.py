"""G1 v2 Stage C Pipeline: 21-Graph NARMA10 Main-Train/Val/Calibration, Delta, and Power/FPR Calibration.

Governing Specification: research/G1_SPEC_v2_draft.md (§3.2, §4, §5, §7, §8, §9)
Review & Strategy Directives: research/RESULTS_G1_v2.md, research/DISCUSSION_next_target.md, task_stageC_runner.md

Core Protocol:
1. Load graphs:
   - 1 Real FlyWire connectome (v783, 138,639 neurons) or small synthetic fixture.
     Real graph requires R1-6 sensory selection, DN readout (buy_idx U sell_idx == 1,291).
   - 20 scramble_mixed graphs (seeds 5001..5020 from manifest scramble namespace).
     Unscaled invariants verified via verify_unscaled_scramble_invariants (overlap <= 5%, degree, weights, etc.).
   - All graphs independently scaled to spectral radius rho=0.95 with convergence diagnostics.
2. Data streams from manifest:
   - Main-Train: 10 sequences (valid 5,000 + washout 500)
   - Main-Val: 3 sequences (valid 2,000 + washout 500, integrity/finite checks only, no selection)
   - Calibration: 10 sequences (valid 5,000 + washout 500)
   - Main-Test: strictly sealed in Stage C (attempting to generate/read raises PermissionError).
3. Stateful simulation & Dynamics health gates:
   - TorchG1Reservoir (GPU/CPU) or ScipyG1Reservoir (CPU).
   - Streaming saturation tracking (|x| > 0.90 ratio < 5% on non-sensory neurons post-washout).
   - Forgetting gate on Main-Train: 40 pairs, >= 38 pairs normalized RMS < 1e-3 at t=500.
4. Ridge fitting on Main-Train:
   - Predictor: DN states x[t+1] post-washout.
   - Target: NARMA10 y[t+1] post-washout. Timing alignment: x[t+1] -> y[t+1].
   - 5-fold sequence CV over alpha in {0, 1e-8, ..., 1e2}, refit on all 10 Main-Train sequences.
   - Record alpha == 1e2 (upper bound collision, SPEC §4.3 >10% triggers no-go).
   - Compute Main-Train pooled NMSE.
5. Calibration evaluation & Primary Delta:
   - Predict on 10 Calibration sequences, obtain residuals e[g, s, t] and calib NMSE per graph.
   - Primary contrast: C = median(NMSE_ctrl_1..NMSE_ctrl_20) (using even median).
   - Delta = (C - NMSE_real) / C.
   - Primary CI: 2,000 resamples via graph_sequence_cross_bootstrap (10 full sequences x 20 controls).
6. Statistical inference calibration:
   - 1,000-cohort Monte Carlo calibration via calibrate_v2_power_and_fpr.
   - Power lower bound L_power >= 0.80, FPR upper bound U_FPR <= 0.05 (exact Clopper-Pearson 97.5%).
7. Governance & Go / No-Go:
   - Coordinated via G1Runner and BudgetLedger (Stage C <= 4.0 GPU h, total <= 8.0 GPU h).
   - All gates pass -> STAGE_C_GO (exit 0).
   - Any gate fails -> STAGE_C_NO_GO (exit 1).
   - Defect / Provenance failure -> RUN_INVALID (exit 2).
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import sys
import time
from typing import Any, Callable, Sequence

import numpy as np
import scipy.sparse as sp

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from src.connectome.schema import ConnectomeGraph
from src.connectome.synthetic import make_synthetic_connectome
from research.pipeline.g1_manifest import (
    G1Manifest,
    create_g1_manifest,
    generate_split_sequences,
    compute_file_sha256,
    compute_source_snapshot_sha,
)
from research.pipeline.g1_graphs import (
    generate_unscaled_scramble_mixed,
    verify_unscaled_scramble_invariants,
)
from research.pipeline.g1_reservoir import (
    ScipyG1Reservoir,
    TorchG1Reservoir,
    scale_weights_to_spectral_radius,
)
from research.pipeline.g1_bench import (
    ALPHA_GRID,
    G1RidgeReadout,
    check_forgetting_gate,
    check_saturation_gate,
    select_r16_indices,
)
from research.pipeline.g1_v2 import (
    UPPER_BOUND_ALPHA,
    compute_pooled_nmse,
    compute_even_median,
    compute_delta_primary,
)
from research.pipeline.g1_power import (
    build_sequence_stats,
    graph_sequence_cross_bootstrap,
    calibrate_v2_power_and_fpr,
)
from research.pipeline.g1_runner import (
    BudgetLedger,
    G1Runner,
    RunInvalidIncident,
    STAGE_BUDGET_LIMITS,
    TOTAL_BUDGET_LIMITS,
)
from research.pipeline.graph_variants import compute_graph_sha256


# ==============================================================================
# Constants & Defaults (SPEC v2 §3.2, §4, §8)
# ==============================================================================

EXPECTED_DN_COUNT: int = 1291
DEFAULT_TARGET_RHO: float = 0.95
DEFAULT_LEAK: float = 0.5
DEFAULT_N_CONTROLS: int = 20
DEFAULT_MC_REPLICATES: int = 1000
DEFAULT_BOOTSTRAPS: int = 2000


class ProvenanceError(Exception):
    """Raised when connectome provenance, DN count, or split integrity is violated."""


# ==============================================================================
# Fixture Graph Generator for CPU / Unit Testing
# ==============================================================================

def make_stage_c_fixture_graph(
    mode: str = "go",
    seed: int = 42,
    n_neurons: int = 300,
    avg_out_degree: float = 6.0,
    n_sensory: int = 16,
    n_dn: int = 16,
) -> ConnectomeGraph:
    """Generate a clean synthetic connectome for Stage C fast CPU unit testing.

    Uses n_neurons=300 and avg_out_degree=6.0 (density ~0.67%) so that
    unscaled scramble_mixed easily achieves target_overlap <= 0.05.
    """
    rng = np.random.default_rng(seed)

    if mode == "provenance_fail":
        # Missing buy/sell metadata
        base = make_synthetic_connectome(
            n_neurons=n_neurons, avg_out_degree=avg_out_degree,
            n_sensory=n_sensory, n_motor=n_dn, seed=seed,
        )
        base.meta["is_fixture"] = False
        base.meta["source"] = "flywire_v783"
        return base
    else:  # 'go', 'saturation_fail', 'health_fail'
        base = make_synthetic_connectome(
            n_neurons=n_neurons, avg_out_degree=avg_out_degree,
            n_sensory=n_sensory, n_motor=n_dn, seed=seed,
        )
        W = base.weights.astype(np.float32)
        sensory_idx = base.sensory_idx.copy()
        dn_idx = base.motor_idx.copy()

    n_dn_actual = len(dn_idx)
    buy_idx = dn_idx[: n_dn_actual // 2]
    sell_idx = dn_idx[n_dn_actual // 2 :]

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

    g = ConnectomeGraph(
        weights=W,
        neuron_ids=neuron_ids,
        neuron_types=neuron_types,
        sensory_idx=sensory_idx,
        motor_idx=dn_idx,
        meta=meta,
    )
    g.meta["sha256"] = compute_graph_sha256(g)
    return g


# ==============================================================================
# Stage C Main Execution Function
# ==============================================================================

def execute_stage_c(
    graph_source: str = "fixture",
    device: str = "cpu",
    require_cuda: bool = False,
    engine: str = "auto",
    repo_root: str | Path | None = None,
    output_path: str | Path | None = None,
    fixture_mode: str = "go",
    n_controls: int = DEFAULT_N_CONTROLS,
    n_replicates: int = DEFAULT_MC_REPLICATES,
    n_bootstraps: int = DEFAULT_BOOTSTRAPS,
    custom_real_graph: ConnectomeGraph | None = None,
    save_report: bool = True,
) -> tuple[dict[str, Any], int]:
    """Execute complete Stage C 21-graph protocol, primary contrast, and power/FPR calibration.

    Parameters
    ----------
    graph_source: 'real' or 'fixture'
    device: 'cpu' or 'cuda'
    require_cuda: bool, raise if CUDA is unavailable
    engine: 'torch', 'scipy', or 'auto'
    repo_root: root path of repository
    output_path: custom output JSON destination (if None, writes to research/outputs/v3/g1_stage_c_report.json)
    fixture_mode: 'go', 'saturation_fail', or 'provenance_fail'
    n_controls: number of scramble controls to generate and evaluate (default 20)
    n_replicates: number of outer Monte Carlo cohorts for power/FPR (default 1000)
    n_bootstraps: number of cross-bootstrap resamples (default 2000)
    custom_real_graph: optional pre-loaded ConnectomeGraph
    save_report: bool, whether to write report JSON to disk

    Returns
    -------
    (report_dict, exit_code)
        exit_code 0: STAGE_C_GO
        exit_code 1: STAGE_C_NO_GO
        exit_code 2: RUN_INVALID
    """
    t_stage_start = time.perf_counter()
    timestamp_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]

    # Resolve output path
    if output_path is not None:
        target_json_path = Path(output_path)
    else:
        target_json_path = root / "research" / "outputs" / "v3" / "g1_stage_c_report.json"

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

    if require_cuda:
        if not HAS_TORCH or not torch.cuda.is_available():
            raise RuntimeError("--require-cuda was set, but PyTorch CUDA GPU is unavailable.")

    # Determine simulation engine
    if engine == "auto":
        resolved_engine = "torch" if (device == "cuda" or (HAS_TORCH and torch.cuda.is_available())) else "scipy"
    else:
        resolved_engine = engine

    # Initialize manifest and advance to Stage C
    manifest = create_g1_manifest(root)
    manifest.advance_stage("B")
    manifest.advance_stage("C")
    manifest_sha = manifest.compute_sha256()

    report: dict[str, Any] = {
        "protocol": "G1_v2_Stage_C",
        "stage": "C",
        "status": "INITIALIZING",
        "timestamp_utc": timestamp_iso,
        "env": env_info,
        "hashes": {
            "spec_sha256": manifest.spec_sha256,
            "source_snapshot_sha256": manifest.source_snapshot_sha256,
            "manifest_sha256": manifest_sha,
            "simulator_hash": manifest.simulator_hash,
            "readout_hash": manifest.readout_hash,
            "scorer_hash": manifest.scorer_hash,
        },
        "provenance": {},
        "graphs": {},
        "primary_contrast": {},
        "statistical_calibration": {},
        "gates": {},
        "budget_ledger": {},
    }

    ledger = BudgetLedger()
    runner = G1Runner(manifest=manifest, ledger=ledger)

    try:
        # ======================================================================
        # 1. Load and Verify Base Real Graph
        # ======================================================================
        if custom_real_graph is not None:
            base_graph = custom_real_graph
            is_fixture = bool(getattr(base_graph, "meta", {}).get("is_fixture", True))
        elif graph_source == "fixture":
            base_graph = make_stage_c_fixture_graph(mode=fixture_mode)
            is_fixture = True
        elif graph_source == "real":
            if sys.platform == "darwin" and os.environ.get("ALLOW_FLYWIRE_ON_MAC") != "1":
                raise ProvenanceError(
                    "Execution on real full FlyWire graph on macOS is strictly prohibited "
                    "per task directives (GPU / Windows RTX 5070 only). Set ALLOW_FLYWIRE_ON_MAC=1 to bypass."
                )
            from research.pipeline.flywire_graph import load_flywire_graph
            base_graph = load_flywire_graph(use_cache=True)
            is_fixture = False
        else:
            raise ValueError(f"Unknown graph_source '{graph_source}'. Must be 'real' or 'fixture'.")

        r16_idx, r16_summary = select_r16_indices(base_graph)
        real_unscaled_sha = r16_summary["graph_sha256"]
        report["hashes"]["real_graph_sha256_unscaled"] = real_unscaled_sha

        buy_idx = base_graph.meta.get("buy_idx")
        sell_idx = base_graph.meta.get("sell_idx")
        if buy_idx is None or sell_idx is None:
            raise ProvenanceError("Missing buy_idx or sell_idx in base graph metadata.")

        dn_idx = np.union1d(np.asarray(buy_idx, dtype=np.int64), np.asarray(sell_idx, dtype=np.int64))
        if not is_fixture and len(dn_idx) != EXPECTED_DN_COUNT:
            raise ProvenanceError(
                f"Provenance blocked: DN count {len(dn_idx)} != expected {EXPECTED_DN_COUNT} (buy_idx U sell_idx)."
            )

        # Scale base real graph to rho=0.95
        w_scaled, target_rho, unscaled_rho, scale_factor, spec_diag = scale_weights_to_spectral_radius(
            weights=base_graph.weights,
            target_rho=DEFAULT_TARGET_RHO,
            verify=True,
            return_diagnostics=True,
        )
        real_scaled_sha = spec_diag["scaled_sha256"]
        report["hashes"]["real_graph_sha256_scaled"] = real_scaled_sha

        scaled_real_graph = ConnectomeGraph(
            weights=w_scaled,
            neuron_ids=base_graph.neuron_ids,
            neuron_types=base_graph.neuron_types,
            sensory_idx=base_graph.sensory_idx,
            motor_idx=dn_idx,
            meta={**base_graph.meta, "spectral_radius": target_rho, "sha256": real_scaled_sha},
        )

        report["provenance"]["real"] = {
            "graph_source": "fixture" if is_fixture else "real",
            "n_neurons": int(base_graph.n_neurons),
            "r16_indices_count": len(r16_idx),
            "dn_indices_count": len(dn_idx),
            "spectral_radius_unscaled": float(unscaled_rho),
            "spectral_radius_target": float(target_rho),
            "spectral_scale_factor": float(scale_factor),
        }

        # ======================================================================
        # 2. Generate and Verify Scramble Control Graphs
        # ======================================================================
        control_seeds = manifest.seed_namespaces.get("scramble", tuple(range(5001, 5021)))
        chosen_control_seeds = list(control_seeds[:n_controls])

        scaled_control_graphs: list[ConnectomeGraph] = []
        control_scaled_shas: list[str] = []

        for idx, seed in enumerate(chosen_control_seeds):
            scram_unscaled = generate_unscaled_scramble_mixed(
                base_graph=base_graph,
                seed=seed,
                target_overlap=0.05,
            )
            inv_diag = verify_unscaled_scramble_invariants(
                orig_graph=base_graph,
                scram_graph=scram_unscaled,
                max_overlap=0.05,
            )

            # Scale control graph to rho=0.95
            scram_w_scaled, _, _, _, c_spec_diag = scale_weights_to_spectral_radius(
                weights=scram_unscaled.weights,
                target_rho=DEFAULT_TARGET_RHO,
                verify=True,
                return_diagnostics=True,
            )
            c_scaled_sha = c_spec_diag["scaled_sha256"]
            control_scaled_shas.append(c_scaled_sha)

            scram_scaled = ConnectomeGraph(
                weights=scram_w_scaled,
                neuron_ids=scram_unscaled.neuron_ids,
                neuron_types=scram_unscaled.neuron_types,
                sensory_idx=scram_unscaled.sensory_idx,
                motor_idx=dn_idx,
                meta={
                    **scram_unscaled.meta,
                    "spectral_radius": DEFAULT_TARGET_RHO,
                    "sha256": c_scaled_sha,
                    "overlap_ratio": inv_diag["overlap_ratio"],
                },
            )
            scaled_control_graphs.append(scram_scaled)

        report["hashes"]["control_scaled_sha256s"] = control_scaled_shas
        report["provenance"]["n_controls"] = len(scaled_control_graphs)

        # ======================================================================
        # 3. Load Datasets strictly via Manifest (Main-Train, Main-Val, Calibration)
        # ======================================================================
        main_train_seqs = generate_split_sequences("main-train", manifest)
        main_val_seqs = generate_split_sequences("main-val", manifest)
        calib_seqs = generate_split_sequences("calibration", manifest)

        # Security check: Main-Test must NOT be accessed
        if not manifest.is_main_test_sealed:
            raise ProvenanceError("Main-Test split was unsealed prematurely during Stage C.")

        u_train_batch = np.array([seq["u"] for seq in main_train_seqs], dtype=np.float32)
        u_val_batch = np.array([seq["u"] for seq in main_val_seqs], dtype=np.float32)
        u_calib_batch = np.array([seq["u"] for seq in calib_seqs], dtype=np.float32)

        train_targets = [seq["y_next"][500:] for seq in main_train_seqs]
        calib_targets = [seq["y_next"][500:] for seq in calib_seqs]
        calib_targets_arr = np.array(calib_targets, dtype=np.float64)

        # Factory to build reservoir engine for a given graph
        def build_engine(g: ConnectomeGraph):
            if resolved_engine == "torch":
                return TorchG1Reservoir(
                    weights=g.weights,
                    sensory_idx=r16_idx,
                    readout_idx=dn_idx,
                    target_rho=DEFAULT_TARGET_RHO,
                    leak=DEFAULT_LEAK,
                    device=device,
                    dtype=torch.float32,
                    verify_spectral_radius=False,
                )
            else:
                return ScipyG1Reservoir(
                    weights=g.weights,
                    sensory_idx=r16_idx,
                    readout_idx=dn_idx,
                    target_rho=DEFAULT_TARGET_RHO,
                    leak=DEFAULT_LEAK,
                    dtype=np.float32,
                    verify_spectral_radius=False,
                )

        # ======================================================================
        # 4. Simulate & Evaluate All Graphs (1 Real + N Controls)
        # ======================================================================
        all_graphs = [scaled_real_graph] + scaled_control_graphs
        graph_labels = ["real"] + [f"scramble_{s}" for s in chosen_control_seeds]

        graph_records: list[dict[str, Any]] = []
        all_calib_residuals: list[np.ndarray] = []

        total_sim_time = 0.0

        for g_idx, (g, label) in enumerate(zip(all_graphs, graph_labels)):
            t0_g = time.perf_counter()
            engine_inst = build_engine(g)

            # 4a. Main-Train Simulation & Saturation Gate
            states_train, _, sat_train = engine_inst.simulate(
                u_train_batch, track_saturation=True, washout=500, input_centering=False,
            )
            if not np.all(np.isfinite(states_train)):
                raise ValueError(f"Non-finite states in Main-Train simulation for {label}")

            # 4b. Forgetting Gate on Main-Train
            def sim_fn_for_forgetting(u_in: np.ndarray, x0: np.ndarray | None):
                return engine_inst.simulate(u_in, initial_state=x0, track_saturation=False, input_centering=False)

            forget_res = check_forgetting_gate(
                sim_fn=sim_fn_for_forgetting,
                train_u_sequences=[seq["u"] for seq in main_train_seqs],
                n_neurons=g.n_neurons,
            )

            health_passed = bool(sat_train["passed"] and forget_res["passed"])
            if is_fixture and fixture_mode in ["saturation_fail", "health_fail"] and g_idx == 0:
                health_passed = False
                sat_train["passed"] = False
                sat_train["saturation_ratio"] = 0.95

            # 4c. Main-Train Ridge Fit (5-fold sequence CV)
            train_features = [states_train[i, 500:] for i in range(len(main_train_seqs))]
            ridge = G1RidgeReadout(alphas=ALPHA_GRID, std_cutoff=1e-8)
            ridge.fit_sequence_group_cv(
                sequence_features=train_features,
                sequence_targets=train_targets,
                n_folds=5,
            )
            best_alpha = float(ridge.best_alpha)
            is_alpha_upper_bound = bool(abs(best_alpha - UPPER_BOUND_ALPHA) < 1e-12)

            y_tr_pred = [ridge.predict(train_features[i]) for i in range(len(main_train_seqs))]
            nmse_train = compute_pooled_nmse(np.concatenate(train_targets), np.concatenate(y_tr_pred))

            # 4d. Main-Val Simulation (Integrity & finite check only, no selection)
            states_val, _, sat_val = engine_inst.simulate(
                u_val_batch, track_saturation=True, washout=500, input_centering=False,
            )
            if not np.all(np.isfinite(states_val)):
                raise ValueError(f"Non-finite states in Main-Val simulation for {label}")

            # 4e. Calibration Simulation & Prediction
            states_calib, _, sat_calib = engine_inst.simulate(
                u_calib_batch, track_saturation=True, washout=500, input_centering=False,
            )
            if not np.all(np.isfinite(states_calib)):
                raise ValueError(f"Non-finite states in Calibration simulation for {label}")

            calib_features = [states_calib[i, 500:] for i in range(len(calib_seqs))]
            y_calib_pred = [ridge.predict(calib_features[i]) for i in range(len(calib_seqs))]
            nmse_calib = compute_pooled_nmse(np.concatenate(calib_targets), np.concatenate(y_calib_pred))

            # Residuals for calibration: shape (10, 5000)
            e_calib_g = np.array([y_calib_pred[i] - calib_targets[i] for i in range(len(calib_seqs))], dtype=np.float64)
            all_calib_residuals.append(e_calib_g)

            elapsed_g = time.perf_counter() - t0_g
            total_sim_time += elapsed_g

            rec = {
                "label": label,
                "is_real": bool(g_idx == 0),
                "best_alpha": best_alpha,
                "is_alpha_upper_bound": is_alpha_upper_bound,
                "train_nmse": nmse_train,
                "calib_nmse": nmse_calib,
                "saturation_ratio": float(sat_train["saturation_ratio"]),
                "saturation_passed": bool(sat_train["passed"]),
                "forgetting_passed_pairs": int(forget_res["n_passed_pairs"]),
                "forgetting_passed": bool(forget_res["passed"]),
                "health_passed": health_passed,
                "elapsed_seconds": elapsed_g,
            }
            graph_records.append(rec)

        report["graphs"] = {
            "real": graph_records[0],
            "controls": graph_records[1:],
        }

        # ======================================================================
        # 5. Primary Contrast (Delta) and Cross-Bootstrap
        # ======================================================================
        real_rec = graph_records[0]
        ctrl_recs = graph_records[1:]

        real_calib_nmse = real_rec["calib_nmse"]
        ctrl_calib_nmses = [c["calib_nmse"] for c in ctrl_recs]

        control_median = compute_even_median(ctrl_calib_nmses)
        delta_obs = compute_delta_primary(real_calib_nmse, ctrl_calib_nmses)

        # Cross-bootstrap on calibration residuals
        e_real_calib = all_calib_residuals[0]
        e_ctrl_calib = np.stack(all_calib_residuals[1:], axis=0)  # shape: (n_controls, 10, 5000)

        calib_stats = build_sequence_stats(
            e_real=e_real_calib,
            e_ctrl=e_ctrl_calib,
            y=calib_targets_arr,
        )

        boot_res = graph_sequence_cross_bootstrap(
            stats=calib_stats,
            n_bootstraps=n_bootstraps,
            seed=70000,
            alpha=0.05,
        )

        report["primary_contrast"] = {
            "real_calib_nmse": real_calib_nmse,
            "control_median_calib_nmse": control_median,
            "delta": delta_obs,
            "ci_lower_95": boot_res["ci_lower"],
            "ci_upper_95": boot_res["ci_upper"],
            "ci_lower_gt_0": boot_res["ci_lower_gt_0"],
            "delta_ge_5pct": bool(delta_obs >= 0.05),
            "passed_primary_gate": boot_res["passed_primary_gate"],
            "n_bootstraps": n_bootstraps,
        }

        # ======================================================================
        # 6. Monte Carlo Power & FPR Calibration (1,000 Cohorts)
        # ======================================================================
        mc_calib = calibrate_v2_power_and_fpr(
            e_real=e_real_calib,
            e_ctrl=e_ctrl_calib,
            y=calib_targets_arr,
            n_replicates=n_replicates,
            n_bootstraps_per_cohort=n_bootstraps,
            base_seed=60000,
            alpha_mc=0.025,
            expected_n_ctrl=n_controls if n_controls != 20 else 20,
        )

        report["statistical_calibration"] = {
            "null_scale": mc_calib["null_scale"],
            "alt_scale": mc_calib["alt_scale"],
            "m_real": mc_calib["m_real"],
            "M_median": mc_calib["M_median"],
            "events": mc_calib["events"],
            "bounds": mc_calib["bounds"],
            "cohort_spec": mc_calib["cohort_spec"],
            "passed_mde_gate": mc_calib["passed_mde_gate"],
        }

        # ======================================================================
        # 7. Evaluate All Stage C Gates & Decision
        # ======================================================================
        all_passed_health = all(g["health_passed"] for g in graph_records)

        # Alpha boundary check: > 10% hit alpha=1e2 is no-go per SPEC §4.3
        n_alpha_max = sum(1 for g in graph_records if g["is_alpha_upper_bound"])
        alpha_max_ratio = float(n_alpha_max / len(graph_records))
        passed_alpha_boundary = bool(alpha_max_ratio <= 0.10)

        passed_power_gate = bool(mc_calib["bounds"]["passed_power_gate"])
        passed_fpr_gate = bool(mc_calib["bounds"]["passed_fpr_gate"])

        # Cost tracking
        total_wall_time = time.perf_counter() - t_stage_start
        approx_gpu_h = (total_sim_time / 3600.0) if device == "cuda" else 0.0
        approx_person_h = total_wall_time / 3600.0

        ledger.record_usage("C", person_hours=approx_person_h, gpu_hours=approx_gpu_h)
        passed_budget_gate = bool(
            ledger.stage_gpu_hours["C"] <= STAGE_BUDGET_LIMITS["C"]["gpu_hours"]
            and ledger.total_gpu_hours <= TOTAL_BUDGET_LIMITS["gpu_hours"]
        )

        stage_c_go = bool(
            all_passed_health
            and passed_alpha_boundary
            and passed_power_gate
            and passed_fpr_gate
            and passed_budget_gate
        )

        report["gates"] = {
            "passed_health_gates": all_passed_health,
            "passed_alpha_boundary": passed_alpha_boundary,
            "n_alpha_max": n_alpha_max,
            "alpha_max_ratio": alpha_max_ratio,
            "passed_power_gate": passed_power_gate,
            "passed_fpr_gate": passed_fpr_gate,
            "passed_budget_gate": passed_budget_gate,
            "stage_c_go": stage_c_go,
        }

        report["budget_ledger"] = {
            "stage_c_person_hours": approx_person_h,
            "stage_c_gpu_hours": approx_gpu_h,
            "total_person_hours": ledger.total_person_hours,
            "total_gpu_hours": ledger.total_gpu_hours,
            "stage_c_gpu_limit": STAGE_BUDGET_LIMITS["C"]["gpu_hours"],
            "total_gpu_limit": TOTAL_BUDGET_LIMITS["gpu_hours"],
        }

        report["status"] = "STAGE_C_GO" if stage_c_go else "STAGE_C_NO_GO"
        exit_code = 0 if stage_c_go else 1

    except Exception as e:
        incident = RunInvalidIncident(
            incident_type="provenance_or_split_corruption" if isinstance(e, ProvenanceError) else "code_or_data_defect",
            first_discovered_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            independent_evidence=f"Exception in execute_stage_c: {type(e).__name__}: {e}",
            results_already_viewed=[],
            changed_files=[],
            new_hashes={"manifest": manifest_sha},
            remaining_budget=ledger.remaining_budget(),
        )
        report["status"] = "RUN_INVALID"
        report["incident"] = incident.to_dict()
        report["error"] = str(e)
        exit_code = 2

    report["total_wall_seconds"] = time.perf_counter() - t_stage_start

    # Save output if requested
    if save_report and target_json_path is not None:
        target_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    return report, exit_code


# ==============================================================================
# CLI Entrypoint
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="G1 v2 Stage C Execution Engine")
    parser.add_argument("--graph-source", choices=["real", "fixture"], default="fixture",
                        help="Graph source: 'real' (FlyWire connectome) or 'fixture' (synthetic)")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu",
                        help="Execution device: 'cpu' or 'cuda'")
    parser.add_argument("--require-cuda", action="store_true",
                        help="Fail closed if CUDA GPU is not available")
    parser.add_argument("--engine", choices=["auto", "torch", "scipy"], default="auto",
                        help="Simulation engine backend")
    parser.add_argument("--output-path", type=str, default=None,
                        help="Custom path for report JSON output")
    parser.add_argument("--fixture-mode", choices=["go", "saturation_fail", "provenance_fail"], default="go",
                        help="Fixture behavior mode")
    parser.add_argument("--n-controls", type=int, default=DEFAULT_N_CONTROLS,
                        help=f"Number of scramble control graphs (default {DEFAULT_N_CONTROLS})")
    parser.add_argument("--n-replicates", type=int, default=DEFAULT_MC_REPLICATES,
                        help=f"Number of MC cohorts for power/FPR (default {DEFAULT_MC_REPLICATES})")
    parser.add_argument("--n-bootstraps", type=int, default=DEFAULT_BOOTSTRAPS,
                        help=f"Number of bootstrap resamples (default {DEFAULT_BOOTSTRAPS})")

    args = parser.parse_args()

    report, exit_code = execute_stage_c(
        graph_source=args.graph_source,
        device=args.device,
        require_cuda=args.require_cuda,
        engine=args.engine,
        output_path=args.output_path,
        fixture_mode=args.fixture_mode,
        n_controls=args.n_controls,
        n_replicates=args.n_replicates,
        n_bootstraps=args.n_bootstraps,
    )

    print(json.dumps({
        "status": report.get("status"),
        "stage": report.get("stage"),
        "primary_contrast": report.get("primary_contrast", {}),
        "gates": report.get("gates", {}),
        "total_wall_seconds": report.get("total_wall_seconds"),
    }, indent=2))

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
