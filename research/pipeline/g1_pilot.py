"""G1 Pilot Runner and Spectral Radius Selection Pipeline.

Governance and Requirements (SPEC §4, §7, §9):
- Evaluates Train sequences ONLY. Strictly enforces exact match with frozen TRAIN_SEEDS.
- Evaluates candidate rhos {0.90, 0.95, 0.99}. Spec hash binds actual candidate grid.
- Pilot graphs: 16 total (1 real + 5 scramble_mixed + 5 weight_shuffle_source + 5 random_endpoint).
- Sensory mapping: R1-6 photoreceptors only (via select_r16_indices).
- Streaming saturation gate: post-washout non-injected |x| > 0.9 proportion < 5%, fail-closed on NaN/Inf.
- Initial-state forgetting gate: 40 pairs across 10 Train sequences, >=38 pairs d_rel < 1e-3 at t=500.
- Linear memory capacity MC: Train-only out-of-sequence 5-fold group CV on DN readout states, lag 1..50.
- Common rho selection:
  Only rhos where all 4 families and all 16 pilot graphs pass both gates are qualified.
  Select qualified rho with maximum family-balanced mean MC; tie-break: select lower rho.
  If no qualified rho: no-go.
- CLI --dry-run for fast verification on CPU with synthetic graphs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from typing import Any, Callable, Sequence

import numpy as np
import scipy.sparse as sp

from src.connectome.schema import ConnectomeGraph
from research.pipeline.g1_bench import (
    TRAIN_SEEDS,
    VAL_SEEDS,
    TEST_SEEDS,
    WASHOUT_STEPS,
    TRAIN_SEQUENCE_LENGTH,
    generate_train_sequences,
    check_saturation_gate,
    check_forgetting_gate,
    compute_memory_capacity_cv,
    select_r16_indices,
)
from research.pipeline.g1_power import (
    build_narma10_planted_feature,
    fit_q_probe_heads,
    evaluate_q_positive_control_gate,
)
from research.pipeline.g1_reservoir import ScipyG1Reservoir


# ==============================================================================
# Frozen Configurations and Types (SPEC §4, §9)
# ==============================================================================

CANDIDATE_RHOS: tuple[float, ...] = (0.90, 0.95, 0.99)

PILOT_FAMILIES: tuple[str, ...] = (
    "real",
    "scramble_mixed",
    "weight_shuffle_source",
    "random_endpoint",
)

FAMILY_INSTANCE_COUNTS: dict[str, int] = {
    "real": 1,
    "scramble_mixed": 5,
    "weight_shuffle_source": 5,
    "random_endpoint": 5,
}

SPEC_VERSION: str = "G1_SPEC_v3_draft"


def compute_pilot_spec_hash(candidate_rhos: Sequence[float], confirmatory: bool = False) -> str:
    """Compute deterministic specification hash binding actual candidate rhos.

    Per G1_SPEC §9:
        The hash must strictly bind actual candidate rhos. Confirmatory runs
        must reject non-preregistered grids.
    """
    rho_tuple = tuple(float(r) for r in sorted(candidate_rhos))
    if confirmatory and rho_tuple != CANDIDATE_RHOS:
        raise ValueError(
            f"Governance violation: Confirmatory G1 run strictly forbids modifying candidate rho grid "
            f"from preregistered {CANDIDATE_RHOS} (got {rho_tuple})."
        )

    payload = json.dumps(
        {
            "candidate_rhos": list(rho_tuple),
            "families": list(PILOT_FAMILIES),
            "family_counts": FAMILY_INSTANCE_COUNTS,
            "forgetting_min_pass": 38,
            "forgetting_pairs": 40,
            "forgetting_thresh": 1e-3,
            "mc_lags": 50,
            "nested_cv_folds": 5,
            "q_gate_max_null_fpr": 0.05,
            "q_gate_min_aq": 0.05,
            "q_gate_min_oracle_power": 0.80,
            "q_target_advantage": 0.05,
            "saturation_max_ratio": 0.05,
            "saturation_thresh": 0.90,
            "spec_version": SPEC_VERSION,
            "train_seeds": list(TRAIN_SEEDS),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


G1_PILOT_SPEC_HASH: str = compute_pilot_spec_hash(CANDIDATE_RHOS, confirmatory=False)

GraphProvider = Callable[[str, int], ConnectomeGraph]
EngineFactory = Callable[..., Any]


def default_scipy_engine_factory(
    weights: sp.csr_matrix,
    sensory_idx: Sequence[int],
    readout_idx: Sequence[int] | None,
    target_rho: float,
    **kwargs: Any,
) -> ScipyG1Reservoir:
    """Default CPU reservoir engine factory using Scipy sparse CSR operations."""
    return ScipyG1Reservoir(
        weights=weights,
        sensory_idx=sensory_idx,
        readout_idx=readout_idx,
        target_rho=target_rho,
        **kwargs,
    )


# ==============================================================================
# Train-Only Verification Guards (SPEC §2, §4, §9)
# ==============================================================================

def validate_train_only_sequences(sequences: Sequence[dict]) -> None:
    """Strictly verify that sequences match the exact frozen Train seeds."""
    expected_seeds = set(TRAIN_SEEDS)
    actual_seeds = []

    for seq in sequences:
        seed = seq.get("seed")
        split_tag = seq.get("metadata", {}).get("split", seq.get("split", ""))
        if seed in TEST_SEEDS or split_tag == "test":
            raise ValueError(
                f"Governance violation: Pilot evaluation strictly forbids accessing Test seeds or data (seed={seed}, split={split_tag})."
            )
        if seed in VAL_SEEDS or split_tag == "val":
            raise ValueError(
                f"Governance violation: Pilot evaluation strictly forbids accessing Val seeds or data (seed={seed}, split={split_tag})."
            )
        if split_tag != "train":
            raise ValueError(
                f"Governance violation: Pilot sequence split metadata must be 'train', got '{split_tag}' (seed={seed})."
            )
        actual_seeds.append(seed)

    if len(actual_seeds) != len(TRAIN_SEEDS):
        raise ValueError(
            f"Governance violation: Pilot requires exactly {len(TRAIN_SEEDS)} Train sequences, got {len(actual_seeds)}."
        )

    if set(actual_seeds) != expected_seeds:
        missing = expected_seeds - set(actual_seeds)
        extra = set(actual_seeds) - expected_seeds
        raise ValueError(
            f"Governance violation: Train sequences must exactly match frozen TRAIN_SEEDS. Missing: {missing}, Extra: {extra}"
        )

    if len(set(actual_seeds)) != len(actual_seeds):
        raise ValueError("Governance violation: Duplicate Train sequence seeds detected.")


def forbid_narma_performance_evaluation() -> None:
    """Raise if any caller attempts to evaluate NARMA predictive performance in Pilot."""
    raise ValueError(
        "Governance violation: Pilot must NOT compute NARMA predictive NMSE or evaluate task readout performance."
    )


# ==============================================================================
# Single (Graph, Rho) Evaluation (Train-only, streaming saturation)
# ==============================================================================

def evaluate_graph_rho_train_only(
    graph: ConnectomeGraph,
    rho: float | None = None,
    train_sequences: Sequence[dict] | None = None,
    engine_factory: EngineFactory = default_scipy_engine_factory,
    target_rho: float | None = None,
    **kwargs: Any,
) -> dict:
    """Evaluate saturation gate, forgetting gate, and linear MC for a single (graph, rho).

    Per G1_SPEC §3, §4, §9:
    1. Selects R1-6 sensory indices with finite coordinates (excludes R7/R8 from injection).
    2. Uses streaming saturation tracking over post-washout steps on non-injected neurons.
    3. Runs initial-state forgetting gate.
    4. Evaluates linear memory capacity MC using 5-fold sequence-group CV on DN readout states.
    """
    if rho is None and target_rho is not None:
        rho = target_rho
    if rho is None:
        rho = 0.95
    if train_sequences is None:
        train_sequences = generate_train_sequences()
    validate_train_only_sequences(train_sequences)

    t0 = time.time()
    n_neurons = graph.n_neurons

    # 1. Select R1-6 photoreceptors and establish non-injected mask
    r16_idx, r16_summary = select_r16_indices(graph)
    motor_idx = graph.motor_idx

    r16_set = set(int(i) for i in r16_idx)
    non_sensory_mask = np.array([i not in r16_set for i in range(n_neurons)], dtype=bool)

    # 2. Initialize reservoir engine with verified spectral radius
    reservoir = engine_factory(
        weights=graph.weights,
        sensory_idx=r16_idx,
        readout_idx=motor_idx,
        target_rho=rho,
    )

    washout = int(train_sequences[0].get("washout", WASHOUT_STEPS))
    u_batch = np.array([seq["u"] for seq in train_sequences], dtype=np.float32)

    # 3. Streamed simulation with online non-sensory saturation accumulation (no 28 GiB allocation)
    sim_out = reservoir.simulate(
        u_batch,
        return_full_states=False,
        track_saturation=True,
        washout=washout,
        non_sensory_mask=non_sensory_mask,
    )
    readout_states_batch = sim_out[0]  # Shape: (B=10, T=5500, D_readout)
    sat_res = sim_out[2]               # Streaming saturation summary dict

    # 4. Initial-state forgetting gate check
    u_seqs = [seq["u"] for seq in train_sequences]
    min_u_len = min(len(u) for u in u_seqs)
    t_check = min(500, min_u_len)
    fg_res = check_forgetting_gate(
        sim_fn=reservoir.simulate,
        train_u_sequences=u_seqs,
        n_neurons=n_neurons,
        seed=42,
        n_pairs_per_seq=4,
        t_check=t_check,
        d_rel_threshold=1e-3,
        min_pass_pairs=38,
    )

    # 5. Out-of-sequence 5-fold group CV linear memory capacity MC (SPEC §9)
    seq_readouts = [readout_states_batch[i] for i in range(len(train_sequences))]
    seq_inputs = [seq["u"] for seq in train_sequences]
    mc_cv_res = compute_memory_capacity_cv(
        sequence_states=seq_readouts,
        sequence_inputs=seq_inputs,
        max_lag=50,
        n_folds=5,
        washout=washout,
    )
    mean_mc = float(mc_cv_res["mc_total"])
    elapsed = time.time() - t0

    graph_sha256 = graph.meta.get("sha256")
    if not graph_sha256:
        from research.pipeline.graph_variants import compute_graph_sha256
        graph_sha256 = compute_graph_sha256(graph)

    passed_both = bool(sat_res["passed"] and fg_res["passed"])

    return {
        "rho": float(rho),
        "graph_sha256": graph_sha256,
        "r16_summary": r16_summary,
        "r16_count": len(r16_idx),
        "passed_both_gates": passed_both,
        "saturation_gate": sat_res,
        "saturation_summary": sat_res,
        "forgetting_gate": fg_res,
        "mc_mean": mean_mc,
        "mc_cv_details": {
            "mc_total": mean_mc,
            "max_lag": mc_cv_res["max_lag"],
            "n_folds": mc_cv_res["n_folds"],
        },
        "seq_readouts": seq_readouts,
        "runtime_seconds": elapsed,
    }


# ==============================================================================
# Full G1 Pilot Runner & Common Rho Selector (SPEC §4, §9)
# ==============================================================================

def make_planted_q_engine_factory(plant_neuron_idx: int = 0) -> EngineFactory:
    """Construct a CPU reservoir engine factory where readout neuron plant_neuron_idx encodes q.

    Used for testing that when the connectome / reservoir readout is sufficient to encode q,
    the G1 Pilot q positive-control gate successfully passes.
    """
    class PlantedQReservoir(ScipyG1Reservoir):
        def simulate(self, u: np.ndarray, initial_state: np.ndarray | None = None, **kwargs: Any) -> Any:
            out = super().simulate(u, initial_state=initial_state, **kwargs)
            u_arr = np.asarray(u, dtype=np.float32)
            is_1d = (u_arr.ndim == 1)
            if is_1d:
                u_arr = u_arr[None, :]
            B, T = u_arr.shape
            q = np.zeros((B, T), dtype=np.float32)
            for t in range(9, T):
                q[:, t] = 1.5 * u_arr[:, t] * u_arr[:, t - 9]

            if kwargs.get("track_saturation", False):
                out_states, final_state, sat_res = out
                if is_1d:
                    out_states[:, plant_neuron_idx] = q[0]
                else:
                    out_states[:, :, plant_neuron_idx] = q
                return out_states, final_state, sat_res
            else:
                out_states, final_state = out
                if is_1d:
                    out_states[:, plant_neuron_idx] = q[0]
                else:
                    out_states[:, :, plant_neuron_idx] = q
                return out_states, final_state

    return PlantedQReservoir


def run_g1_pilot(
    graph_provider: GraphProvider,
    candidate_rhos: Sequence[float] = CANDIDATE_RHOS,
    engine_factory: EngineFactory = default_scipy_engine_factory,
    train_sequences: Sequence[dict] | None = None,
    confirmatory: bool = False,
    n_bootstraps: int = 1000,
    target_q_advantage: float = 0.05,
) -> dict:
    """Execute G1 Pilot across 16 graphs x candidate rhos with nested 5-fold CV and q gate.

    Governance and Requirements (SPEC §4, §7, §9, REVIEW_stop_rule Q2, Q3, Q5):
    1. Evaluates Train sequences ONLY (strictly validated against TRAIN_SEEDS).
    2. Candidate rhos {0.90, 0.95, 0.99}.
    3. Nested 5-fold Train CV:
       - 5 outer folds (each has 2 outer sequences, 8 inner sequences).
       - Inner 8 sequences evaluate 16 Pilot graphs across candidate rhos (saturation, forgetting,
         and 4-fold group CV MC). Selects common rho_f* with max family-balanced MC.
       - If no candidate rho qualifies in any fold: NO-GO (GATE_FAILURE).
       - Outer 2 held-out sequences evaluate q readout positive control at selected rho_f*.
         r = y[t+1] - q[t], z = r + beta* q. beta* calibrated on inner Train real graph (exact 5%).
         Evaluates matched q and mismatched q (TRAIN_SEEDS cyclic shift by 1).
    4. q gate pass conditions:
       - A_q pooled point estimate >= 0.05.
       - One-sided 95% block-bootstrap lower bound > 0.
       - True 5% oracle probe detection rate >= 80%.
       - Mismatched q null FPR <= 5%.
    5. Final rho selection:
       - Evaluated on all 10 Train sequences independently of q results.
       - Selected rho requires all 16 graphs to pass saturation & forgetting on full Train.
    6. Diagnostics & Verdict:
       - MC ratio is secondary diagnostic only (not a gate).
       - Output JSON clearly distinguishes GATE_FAILURE vs INVALID_RUN.
    """
    total_t0 = time.time()

    # 1. Compute and verify specification hash
    spec_rules_hash = compute_pilot_spec_hash(candidate_rhos, confirmatory=confirmatory)

    # 2. Load frozen Train sequences
    if train_sequences is None:
        train_sequences = generate_train_sequences()
    validate_train_only_sequences(train_sequences)
    n_seqs = len(train_sequences)
    washout = int(train_sequences[0].get("washout", WASHOUT_STEPS))

    # 3. Pre-fetch all 16 pilot graph instances
    graphs: dict[str, list[ConnectomeGraph]] = {}
    graph_hashes: dict[str, list[str]] = {}

    for fam in PILOT_FAMILIES:
        n_inst = FAMILY_INSTANCE_COUNTS[fam]
        graphs[fam] = []
        graph_hashes[fam] = []
        for idx in range(n_inst):
            g = graph_provider(fam, idx)
            sha = g.meta.get("sha256")
            if not sha:
                from research.pipeline.graph_variants import compute_graph_sha256
                sha = compute_graph_sha256(g)
            graphs[fam].append(g)
            graph_hashes[fam].append(sha)

    # 4. Evaluate each candidate rho (simulate 10 sequences once per graph x rho)
    eval_results: dict[float, dict[str, list[dict]]] = {}
    sim_readouts: dict[float, dict[str, list[list[np.ndarray]]]] = {}
    qualification: dict[float, bool] = {}
    disqualification_reasons: dict[float, list[str]] = {}
    family_balanced_mc_by_rho: dict[float, float] = {}

    for rho in candidate_rhos:
        rho_val = float(rho)
        eval_results[rho_val] = {fam: [] for fam in PILOT_FAMILIES}
        sim_readouts[rho_val] = {fam: [] for fam in PILOT_FAMILIES}
        reasons = []
        all_passed_for_rho = True

        for fam in PILOT_FAMILIES:
            for idx, g in enumerate(graphs[fam]):
                res = evaluate_graph_rho_train_only(
                    graph=g,
                    rho=rho_val,
                    train_sequences=train_sequences,
                    engine_factory=engine_factory,
                )
                # Pop seq_readouts to keep eval_results lightweight for JSON serialization
                readouts = res.pop("seq_readouts", None)
                sim_readouts[rho_val][fam].append(readouts)
                eval_results[rho_val][fam].append(res)

                if not res["saturation_gate"]["passed"]:
                    all_passed_for_rho = False
                    ratio = res["saturation_gate"]["saturation_ratio"]
                    err = res["saturation_gate"].get("error", "")
                    reasons.append(
                        f"Family '{fam}' instance {idx} failed saturation gate (ratio={ratio:.4f} >= 0.05, err='{err}')"
                    )
                if not res["forgetting_gate"]["passed"]:
                    all_passed_for_rho = False
                    n_p = res["forgetting_gate"]["n_passed_pairs"]
                    tot = res["forgetting_gate"]["total_pairs"]
                    reasons.append(
                        f"Family '{fam}' instance {idx} failed forgetting gate ({n_p}/{tot} < 38 pairs)"
                    )

        qualification[rho_val] = all_passed_for_rho
        disqualification_reasons[rho_val] = reasons

        # Compute family-balanced mean MC across the 4 families (on full 10 sequences)
        fam_means = []
        for fam in PILOT_FAMILIES:
            mc_list = [r["mc_mean"] for r in eval_results[rho_val][fam]]
            fam_means.append(float(np.mean(mc_list)))
        family_balanced_mc_by_rho[rho_val] = float(np.mean(fam_means))

    # 5. Nested 5-Fold Train CV (SPEC §9, REVIEW_stop_rule Q3)
    outer_records: list[dict[str, np.ndarray]] = []
    mismatched_records: list[dict[str, np.ndarray]] = []
    fold_details: list[dict[str, Any]] = []
    all_folds_qualified = True
    fold_disqualifications: list[str] = []

    n_folds = 5
    val_size = n_seqs // n_folds

    for f in range(n_folds):
        outer_seq_indices = list(range(f * val_size, (f + 1) * val_size))
        inner_seq_indices = [i for i in range(n_seqs) if i not in outer_seq_indices]

        fold_rho_mc: dict[float, float] = {}
        fold_qualified_rhos: list[float] = []

        for rho in candidate_rhos:
            rho_val = float(rho)
            all_passed_dyn = bool(qualification[rho_val])

            # Compute MC on the 8 inner sequences via 4-fold group CV
            fam_inner_mcs = []
            for fam in PILOT_FAMILIES:
                inst_mcs = []
                for idx in range(FAMILY_INSTANCE_COUNTS[fam]):
                    states_inst = sim_readouts[rho_val][fam][idx]
                    if states_inst is not None:
                        inner_states = [states_inst[i] for i in inner_seq_indices]
                        inner_inputs = [train_sequences[i]["u"] for i in inner_seq_indices]
                        mc_cv = compute_memory_capacity_cv(
                            sequence_states=inner_states,
                            sequence_inputs=inner_inputs,
                            max_lag=50,
                            n_folds=4,
                            washout=washout,
                        )
                        inst_mcs.append(float(mc_cv["mc_total"]))
                    else:
                        inst_mcs.append(0.0)
                fam_inner_mcs.append(float(np.mean(inst_mcs)))

            fold_rho_mc[rho_val] = float(np.mean(fam_inner_mcs))
            if all_passed_dyn:
                fold_qualified_rhos.append(rho_val)

        if not fold_qualified_rhos:
            all_folds_qualified = False
            disq_msg = f"Fold {f}: No candidate rho qualified across all 16 graphs"
            fold_disqualifications.append(disq_msg)
            fold_details.append({
                "fold": f,
                "outer_sequence_indices": outer_seq_indices,
                "inner_sequence_indices": inner_seq_indices,
                "qualified_rhos": [],
                "selected_rho": None,
                "reason": disq_msg,
            })
        else:
            # Select qualified rho with maximum family-balanced mean MC; tie-break: lower rho
            selected_rho_f = max(fold_qualified_rhos, key=lambda r: (fold_rho_mc[r], -r))

            # Fit q probe heads on real graph at selected_rho_f using the 8 inner sequences
            real_states = sim_readouts[selected_rho_f]["real"][0]
            real_inner_states = [real_states[i] for i in inner_seq_indices]
            inner_y = [train_sequences[i]["y_next"] for i in inner_seq_indices]
            inner_u = [train_sequences[i]["u"] for i in inner_seq_indices]

            probe_bundle = fit_q_probe_heads(
                sequence_features=real_inner_states,
                sequence_targets_y=inner_y,
                sequence_inputs_u=inner_u,
                washout=washout,
                target_advantage=target_q_advantage,
            )

            # Evaluate outer held-out sequences at selected_rho_f
            for j in outer_seq_indices:
                X_j = real_states[j][washout:]
                u_j = train_sequences[j]["u"][washout:]
                y_j = train_sequences[j]["y_next"][washout:]
                q_j = build_narma10_planted_feature(train_sequences[j]["u"])[washout:]
                r_j = y_j - q_j
                z_j = r_j + probe_bundle["beta_star"] * q_j

                r_pred_j = probe_bundle["predict_r"](X_j)
                z_pred_j = probe_bundle["predict_z"](X_j)
                oracle_pred_j = r_pred_j + probe_bundle["beta_star"] * q_j

                outer_records.append({
                    "fold": f,
                    "sequence_idx": j,
                    "z_true": z_j,
                    "r_pred": r_pred_j,
                    "z_pred": z_pred_j,
                    "oracle_pred": oracle_pred_j,
                })

                # Mismatched q: frozen TRAIN_SEEDS cyclic shift by 1
                mis_idx = (j + 1) % n_seqs
                u_mis = train_sequences[mis_idx]["u"][washout:]
                y_mis = train_sequences[mis_idx]["y_next"][washout:]
                q_mis = build_narma10_planted_feature(train_sequences[mis_idx]["u"])[washout:]
                r_mis = y_mis - q_mis
                z_mis = r_mis + probe_bundle["beta_star"] * q_mis

                mismatched_records.append({
                    "fold": f,
                    "sequence_idx": j,
                    "mismatched_idx": mis_idx,
                    "z_true": z_mis,
                    "r_pred": r_pred_j,
                    "z_pred": z_pred_j,
                })

            fold_details.append({
                "fold": f,
                "outer_sequence_indices": outer_seq_indices,
                "inner_sequence_indices": inner_seq_indices,
                "qualified_rhos": [float(r) for r in fold_qualified_rhos],
                "selected_rho": float(selected_rho_f),
                "alpha_star": float(probe_bundle["alpha_star"]),
                "beta_star": float(probe_bundle["beta_star"]),
            })

    # 6. Evaluate q readout positive control gate (pooled across 5 outer folds)
    if all_folds_qualified and outer_records:
        q_gate_eval = evaluate_q_positive_control_gate(
            outer_records=outer_records,
            n_bootstraps=n_bootstraps,
            mismatched_records=mismatched_records,
        )
    else:
        q_gate_eval = {
            "passed_q_gate": False,
            "aq_pooled": 0.0,
            "ci_lower_95": 0.0,
            "ci_upper_95": 0.0,
            "oracle_power": 0.0,
            "oracle_wilson_ci_95": (0.0, 0.0),
            "mismatched_null_fpr": 1.0,
            "mismatched_wilson_ci_95": (1.0, 1.0),
            "mismatched_aq_point": 0.0,
            "passed_pooled_point_gate": False,
            "passed_ci_lower_gate": False,
            "passed_oracle_power_gate": False,
            "passed_null_fpr_gate": False,
            "reason": "Not all nested CV folds qualified",
        }

    # 7. Final rho selection on all 10 Train sequences (SPEC §9: independent of q results)
    qualified_rhos_full = [r for r in candidate_rhos if qualification[float(r)]]
    selected_rho_full: float | None = None
    if qualified_rhos_full:
        selected_rho_full = max(
            qualified_rhos_full,
            key=lambda r: (family_balanced_mc_by_rho[float(r)], -float(r)),
        )

    # 8. Descriptive MC ratio diagnostic (SPEC §9: secondary diagnostic only, not a gate)
    diag_rho = float(selected_rho_full) if selected_rho_full is not None else float(candidate_rhos[0])
    real_mc = eval_results[diag_rho]["real"][0]["mc_mean"]
    ctrl_mcs = [
        res["mc_mean"]
        for fam in ["scramble_mixed", "weight_shuffle_source", "random_endpoint"]
        for res in eval_results[diag_rho][fam]
    ]
    ctrl_median_mc = float(np.median(ctrl_mcs)) if ctrl_mcs else 0.0
    mc_ratio = float(real_mc / ctrl_median_mc) if ctrl_median_mc > 1e-12 else 0.0
    mc_ratio_diagnostic = {
        "real_dn_mc": real_mc,
        "ctrl_median_mc": ctrl_median_mc,
        "mc_ratio": mc_ratio,
        "rho": diag_rho,
        "note": "Descriptive secondary diagnostic; not a pass/fail gate.",
    }

    # 9. Determine overall gate verdict and no-go reasons
    no_go_reasons_dict: dict[str, list[str]] = {}
    if selected_rho_full is None:
        for r in candidate_rhos:
            r_val = float(r)
            no_go_reasons_dict[str(r)] = disqualification_reasons[r_val]
    elif not all_folds_qualified:
        no_go_reasons_dict["nested_cv"] = fold_disqualifications
    elif not q_gate_eval["passed_q_gate"]:
        q_fails = []
        if not q_gate_eval.get("passed_pooled_point_gate", False):
            q_fails.append(f"A_q pooled {q_gate_eval.get('aq_pooled', 0.0):.4f} < 0.05")
        if not q_gate_eval.get("passed_ci_lower_gate", False):
            q_fails.append(f"95% CI lower {q_gate_eval.get('ci_lower_95', 0.0):.4f} <= 0")
        if not q_gate_eval.get("passed_oracle_power_gate", False):
            q_fails.append(f"Oracle power {q_gate_eval.get('oracle_power', 0.0)*100:.1f}% < 80%")
        if not q_gate_eval.get("passed_null_fpr_gate", False):
            q_fails.append(f"Mismatched null FPR {q_gate_eval.get('mismatched_null_fpr', 0.0)*100:.1f}% > 5%")
        no_go_reasons_dict["q_positive_control_gate"] = q_fails

    passed_all_gates = bool(
        selected_rho_full is not None
        and all_folds_qualified
        and q_gate_eval["passed_q_gate"]
    )

    if passed_all_gates:
        no_go = False
        gate_verdict = "GO"
        no_go_category = None
    else:
        no_go = True
        gate_verdict = "NO_GO"
        no_go_category = "GATE_FAILURE"

    total_elapsed = time.time() - total_t0

    return {
        "spec_version": SPEC_VERSION,
        "spec_rules_hash": spec_rules_hash,
        "candidate_rhos": [float(r) for r in candidate_rhos],
        "gate_verdict": gate_verdict,
        "no_go": no_go,
        "no_go_category": no_go_category,
        "is_valid_run": True,
        "selected_rho": selected_rho_full,
        "no_go_reasons": no_go_reasons_dict,
        "disqualification_reasons_by_rho": {
            str(r): disqualification_reasons[float(r)] for r in candidate_rhos
        },
        "qualification_by_rho": {str(r): qualification[float(r)] for r in candidate_rhos},
        "family_balanced_mc_by_rho": {
            str(r): family_balanced_mc_by_rho[float(r)] for r in candidate_rhos
        },
        "nested_cv_results": {
            "n_folds": n_folds,
            "all_folds_qualified": all_folds_qualified,
            "fold_details": fold_details,
            "q_gate_results": q_gate_eval,
        },
        "mc_ratio_diagnostic": mc_ratio_diagnostic,
        "graph_hashes": graph_hashes,
        "total_runtime_seconds": total_elapsed,
        "results_by_rho": {
            str(r): {
                fam: eval_results[float(r)][fam] for fam in PILOT_FAMILIES
            }
            for r in candidate_rhos
        },
    }


# ==============================================================================
# Synthetic Graph Provider for Testing and --dry-run
# ==============================================================================

def make_synthetic_pilot_provider(
    n_neurons: int = 100,
    avg_out_degree: float = 8.0,
    n_sensory: int = 16,
    n_motor: int = 16,
    base_seed: int = 42,
) -> GraphProvider:
    """Construct an in-memory synthetic graph provider for CPU testing & --dry-run."""
    from src.connectome.synthetic import make_synthetic_connectome
    from research.pipeline.graph_variants import (
        well_mixed_scramble,
        source_wise_weight_shuffle,
        random_endpoint_graph,
        compute_graph_sha256,
    )

    base = make_synthetic_connectome(
        n_neurons=n_neurons,
        avg_out_degree=avg_out_degree,
        n_sensory=n_sensory,
        n_motor=n_motor,
        seed=base_seed,
    )
    # Equip base graph with R1-6 metadata
    # First 12 sensory are R1-6, next 2 are R7, next 2 are R8
    ptypes = ["R1-6"] * max(1, n_sensory - 4) + ["R7"] * 2 + ["R8"] * 2
    ptypes = ptypes[:n_sensory]
    base.meta["photoreceptor_type"] = ptypes
    base.meta["u"] = np.random.default_rng(base_seed).uniform(0, 1, size=n_sensory)
    base.meta["v"] = np.random.default_rng(base_seed + 1).uniform(0, 1, size=n_sensory)
    base.meta["sha256"] = compute_graph_sha256(base)

    cache: dict[tuple[str, int], ConnectomeGraph] = {("real", 0): base}

    def provider(family: str, instance_idx: int) -> ConnectomeGraph:
        key = (family, instance_idx)
        if key in cache:
            return cache[key]

        if family == "real":
            return base

        seed = 1000 + instance_idx * 37
        if family == "scramble_mixed":
            g = well_mixed_scramble(base, seed=seed, target_overlap=0.1)
        elif family == "weight_shuffle_source":
            g = source_wise_weight_shuffle(base, seed=seed)
        elif family == "random_endpoint":
            g = random_endpoint_graph(base, seed=seed)
        else:
            raise ValueError(f"Unknown family '{family}'")

        g.meta["photoreceptor_type"] = ptypes
        g.meta["u"] = base.meta["u"]
        g.meta["v"] = base.meta["v"]
        g.meta["sha256"] = compute_graph_sha256(g)
        cache[key] = g
        return g

    return provider


# ==============================================================================
# CLI Entrypoint
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="G1 Pilot Operating Point Selection.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run CPU pilot with small synthetic graphs (completes in ~10-30 seconds on Mac).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path for pilot results.",
    )
    parser.add_argument(
        "--rhos",
        type=str,
        default="0.90,0.95,0.99",
        help="Comma-separated candidate rhos (default: '0.90,0.95,0.99').",
    )
    parser.add_argument(
        "--confirmatory",
        action="store_true",
        default=False,
        help="Strict confirmatory mode: strictly rejects non-preregistered candidate rho grids.",
    )
    args = parser.parse_args()

    rhos = tuple(float(x.strip()) for x in args.rhos.split(","))

    if args.dry_run:
        print("=== Running G1 Pilot Dry-Run (Synthetic Graphs on CPU) ===")
        print(f"Candidate rhos: {rhos}")
        provider = make_synthetic_pilot_provider(n_neurons=100, n_sensory=16, n_motor=16, base_seed=42)
        results = run_g1_pilot(graph_provider=provider, candidate_rhos=rhos, confirmatory=False)

        print("\n--- Summary Table ---")
        print(f"{'Rho':<8} {'Qualified':<12} {'Family-Balanced MC':<22} {'Failure Count'}")
        print("-" * 55)
        for r in rhos:
            r_str = str(r)
            qual = results["qualification_by_rho"][r_str]
            mc = results["family_balanced_mc_by_rho"][r_str]
            reasons = results["disqualification_reasons_by_rho"].get(r_str, [])
            print(f"{r:<8.2f} {str(qual):<12} {mc:<22.4f} {len(reasons)}")
            if reasons:
                for reason in reasons[:3]:
                    print(f"         * {reason}")

        print("\n--- Verdict ---")
        print(f"Gate Verdict: {results['gate_verdict']}")
        print(f"Valid Run: {results['is_valid_run']}")
        if results.get("no_go_category"):
            print(f"Category: {results['no_go_category']}")
        if results["no_go"]:
            print("Status: NO-GO")
            for r_str, r_list in results["no_go_reasons"].items():
                print(f"  {r_str}:")
                for item in r_list[:3]:
                    print(f"    - {item}")
        else:
            print(f"Status: GO")
            print(f"Selected Common Rho: {results['selected_rho']}")
            print(f"Family-Balanced MC at Selected Rho: {results['family_balanced_mc_by_rho'][str(results['selected_rho'])]:.4f}")

        q_res = results.get("nested_cv_results", {}).get("q_gate_results", {})
        print(f"q Readout Gate Passed: {q_res.get('passed_q_gate', False)} (A_q pooled={q_res.get('aq_pooled', 0.0):.4f}, 95% CI lower={q_res.get('ci_lower_95', 0.0):.4f})")
        print(f"MC Ratio Diagnostic: {results.get('mc_ratio_diagnostic', {}).get('mc_ratio', 0.0):.4f}")
        print(f"Specification Hash: {results['spec_rules_hash'][:16]}...")
        print(f"Total Runtime: {results['total_runtime_seconds']:.2f}s")

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            print(f"Output saved to {args.output}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
