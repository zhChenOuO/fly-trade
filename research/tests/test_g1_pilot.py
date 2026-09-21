"""Unit tests for G1 Pilot Runner and Spectral Radius Selection.

Verifications (SPEC §4, §7):
1. Gate evaluation and common rho selection on synthetic connectome families.
2. Disqualification of rho=0.99 when forgetting gate fails.
3. Selection of candidate rho with highest family-balanced MC.
4. Clean no-go when no candidate rho qualifies.
5. Strict rejection of Val and Test sequences.
6. Deterministic reproducibility with fixed seeds.
"""
from __future__ import annotations

import json
import numpy as np
import pytest

from src.connectome.synthetic import make_synthetic_connectome
from research.pipeline.g1_bench import (
    TRAIN_SEEDS,
    VAL_SEEDS,
    TEST_SEEDS,
    generate_narma10_sequence,
)
from research.pipeline.g1_pilot import (
    run_g1_pilot,
    evaluate_graph_rho_train_only,
    make_synthetic_pilot_provider,
    make_planted_q_engine_factory,
    validate_train_only_sequences,
    PILOT_FAMILIES,
    FAMILY_INSTANCE_COUNTS,
)


def test_pilot_rejects_val_and_test_sequences():
    """Verify that pilot strictly rejects Val and Test sequences with ValueError."""
    # 1. Test sequence with Val seed
    val_seq = generate_narma10_sequence(seed=VAL_SEEDS[0], valid_length=500, washout=50)
    with pytest.raises(ValueError, match="strictly forbids accessing Val"):
        validate_train_only_sequences([val_seq])

    # 2. Test sequence with Test seed
    test_seq = generate_narma10_sequence(seed=TEST_SEEDS[0], valid_length=500, washout=50)
    with pytest.raises(ValueError, match="strictly forbids accessing Test"):
        validate_train_only_sequences([test_seq])

    # 3. Test sequence with metadata split tag 'val'
    tagged_seq = {"u": np.zeros(100), "seed": 9999, "metadata": {"split": "val"}}
    with pytest.raises(ValueError, match="strictly forbids accessing Val"):
        validate_train_only_sequences([tagged_seq])

    # 4. Valid Train sequences must pass validation
    train_seqs = [
        generate_narma10_sequence(seed=s, valid_length=500, washout=50)
        for s in TRAIN_SEEDS
    ]
    validate_train_only_sequences(train_seqs)  # Should not raise


def test_pilot_gate_evaluation_and_common_rho_selection():
    """Verify pilot runner is deprecated and strictly rejected with RuntimeError per SPEC v2 §1, §7."""
    provider = make_synthetic_pilot_provider(
        n_neurons=60,
        avg_out_degree=6.0,
        n_sensory=10,
        n_motor=10,
        base_seed=42,
    )

    fast_train_seqs = [
        generate_narma10_sequence(seed=s, valid_length=1000, washout=200)
        for s in TRAIN_SEEDS
    ]

    with pytest.raises(RuntimeError, match="run_g1_pilot is deprecated and withdrawn in G1 v2"):
        run_g1_pilot(
            graph_provider=provider,
            candidate_rhos=(0.90, 0.95, 0.99),
            train_sequences=fast_train_seqs,
        )


def test_pilot_all_disqualified_leads_to_nogo():
    """Verify that calling deprecated pilot runner raises RuntimeError."""
    base = make_synthetic_connectome(n_neurons=40, avg_out_degree=4.0, n_sensory=5, n_motor=5, seed=123)

    def provider(family: str, instance_idx: int):
        return base

    fast_train_seqs = [
        generate_narma10_sequence(seed=s, valid_length=500, washout=100)
        for s in TRAIN_SEEDS
    ]

    with pytest.raises(RuntimeError, match="run_g1_pilot is deprecated and withdrawn in G1 v2"):
        run_g1_pilot(
            graph_provider=provider,
            candidate_rhos=(1.20, 1.30),
            train_sequences=fast_train_seqs,
        )


def test_pilot_deterministic_reproducibility():
    """Verify that calling deprecated pilot runner raises RuntimeError consistently."""
    provider1 = make_synthetic_pilot_provider(n_neurons=40, avg_out_degree=4.0, n_sensory=8, n_motor=8, base_seed=99)
    fast_train_seqs1 = [generate_narma10_sequence(seed=s, valid_length=500, washout=100) for s in TRAIN_SEEDS]

    with pytest.raises(RuntimeError, match="run_g1_pilot is deprecated and withdrawn in G1 v2"):
        run_g1_pilot(graph_provider=provider1, candidate_rhos=(0.90, 0.95), train_sequences=fast_train_seqs1)


def test_pilot_train_seeds_exact_match_guard():
    """validate_train_only_sequences must strictly require exact set equality with TRAIN_SEEDS."""
    valid_seqs = [
        generate_narma10_sequence(seed=s, valid_length=200, washout=50, split_tag="train")
        for s in TRAIN_SEEDS
    ]
    validate_train_only_sequences(valid_seqs)  # Must pass

    # 1. Missing seed (9 sequences)
    with pytest.raises(ValueError, match="requires exactly 10 Train sequences"):
        validate_train_only_sequences(valid_seqs[:-1])

    # 2. Extra seed (11 sequences)
    extra_seq = generate_narma10_sequence(seed=9999, valid_length=200, washout=50, split_tag="train")
    with pytest.raises(ValueError, match="requires exactly 10 Train sequences"):
        validate_train_only_sequences(valid_seqs + [extra_seq])

    # 3. Duplicate seed (10 sequences, but one duplicated)
    dup_seqs = valid_seqs[:-1] + [valid_seqs[0]]
    with pytest.raises(ValueError, match="Missing|Duplicate"):
        validate_train_only_sequences(dup_seqs)

    # 4. Unknown/unregistered seed replacing one seed
    alien_seq = generate_narma10_sequence(seed=8888, valid_length=200, washout=50, split_tag="train")
    alien_seqs = valid_seqs[:-1] + [alien_seq]
    with pytest.raises(ValueError, match="must exactly match frozen TRAIN_SEEDS"):
        validate_train_only_sequences(alien_seqs)


def test_pilot_confirmatory_rho_grid_guard():
    """Confirmatory G1 pilot must strictly reject any modification to candidate rho grid."""
    from research.pipeline.g1_pilot import compute_pilot_spec_hash

    # Confirmatory mode rejects non-default grid
    with pytest.raises(ValueError, match="Confirmatory G1 run strictly forbids modifying candidate rho grid"):
        compute_pilot_spec_hash(candidate_rhos=(0.90, 0.95), confirmatory=True)

    with pytest.raises(ValueError, match="Confirmatory G1 run strictly forbids modifying candidate rho grid"):
        compute_pilot_spec_hash(candidate_rhos=(0.85, 0.90, 0.95, 0.99), confirmatory=True)

    # Confirmatory mode accepts exact preregistered grid
    hash_conf = compute_pilot_spec_hash(candidate_rhos=(0.90, 0.95, 0.99), confirmatory=True)
    assert len(hash_conf) == 64

    # Exploratory mode generates different hashes for different grids
    hash_sub = compute_pilot_spec_hash(candidate_rhos=(0.90, 0.95), confirmatory=False)
    assert hash_sub != hash_conf


def test_pilot_r16_sensory_selection_and_denominator():
    """Pilot single evaluation must use R1-6 sensory indices and include R7/R8 in saturation denominator."""
    from research.pipeline.g1_pilot import evaluate_graph_rho_train_only
    from research.pipeline.flywire_graph import ConnectomeGraph
    import scipy.sparse as sp

    N = 30
    sensory_idx = np.arange(8)
    motor_idx = np.array([20, 21, 22, 23])

    # 0-3 R1-6, 4-5 R7, 6-7 R8
    ptypes = np.array(["R1-6", "R1-6", "R1-6", "R1-6", "R7", "R7", "R8", "R8"], dtype=object)
    u_coords = np.linspace(0.1, 0.8, 8)
    v_coords = np.linspace(0.1, 0.8, 8)

    meta = {
        "photoreceptor_type": ptypes,
        "u": u_coords,
        "v": v_coords,
        "sha256": "mock_pilot_graph_sha",
    }
    weights = sp.eye(N, format="csr")

    graph = ConnectomeGraph(
        weights=weights,
        neuron_ids=[f"N_{i}" for i in range(N)],
        neuron_types=["photoreceptor" if i < 8 else "interneuron" for i in range(N)],
        sensory_idx=sensory_idx,
        motor_idx=motor_idx,
        meta=meta,
    )

    train_seqs = [
        generate_narma10_sequence(seed=s, valid_length=200, washout=50, split_tag="train")
        for s in TRAIN_SEEDS
    ]

    eval_res = evaluate_graph_rho_train_only(
        graph=graph,
        target_rho=0.95,
        train_sequences=train_seqs,
        washout=50,
        n_forgetting_pairs=2,
    )

    # 4 R1-6 neurons injected -> 30 - 4 = 26 non-sensory neurons in saturation denominator
    # R7 and R8 (4 neurons) are not injected and must be part of denominator
    assert eval_res["r16_count"] == 4
    sat_summary = eval_res["saturation_summary"]
    # Total eval points = 10 sequences * 200 post-washout steps * 26 non-sensory neurons = 52,000
    assert sat_summary["total_non_sensory_points"] == 10 * 200 * (N - 4)


def test_pilot_end_to_end_go_and_gate_failure_scenarios():
    """Verify that calling deprecated pilot runner raises RuntimeError."""
    fast_train_seqs = [
        generate_narma10_sequence(seed=s, valid_length=1000, washout=200)
        for s in TRAIN_SEEDS
    ]
    provider = make_synthetic_pilot_provider(
        n_neurons=30,
        avg_out_degree=4.0,
        n_sensory=6,
        n_motor=6,
        base_seed=42,
    )

    with pytest.raises(RuntimeError, match="run_g1_pilot is deprecated and withdrawn in G1 v2"):
        run_g1_pilot(
            graph_provider=provider,
            candidate_rhos=(0.90,),
            train_sequences=fast_train_seqs,
            engine_factory=make_planted_q_engine_factory(),
            n_bootstraps=100,
        )


def test_nested_cv_non_leakage_and_isolation():
    """Verify that calling deprecated nested CV / pilot runner raises RuntimeError."""
    fast_train_seqs = [
        generate_narma10_sequence(seed=s, valid_length=500, washout=100)
        for s in TRAIN_SEEDS
    ]
    provider = make_synthetic_pilot_provider(
        n_neurons=30,
        avg_out_degree=4.0,
        n_sensory=6,
        n_motor=6,
        base_seed=42,
    )

    with pytest.raises(RuntimeError, match="run_g1_pilot is deprecated and withdrawn in G1 v2"):
        run_g1_pilot(
            graph_provider=provider,
            candidate_rhos=(0.90,),
            train_sequences=fast_train_seqs,
            n_bootstraps=50,
        )

