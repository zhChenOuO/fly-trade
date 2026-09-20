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
    """Verify pilot gate evaluation and rho selection on synthetic graphs."""
    provider = make_synthetic_pilot_provider(
        n_neurons=60,
        avg_out_degree=6.0,
        n_sensory=10,
        n_motor=10,
        base_seed=42,
    )

    # Use shorter Train sequences for fast unit testing (1,000 steps + 200 washout)
    fast_train_seqs = [
        generate_narma10_sequence(seed=s, valid_length=1000, washout=200)
        for s in TRAIN_SEEDS
    ]

    results = run_g1_pilot(
        graph_provider=provider,
        candidate_rhos=(0.90, 0.95, 0.99),
        train_sequences=fast_train_seqs,
    )

    assert "selected_rho" in results
    assert "qualification_by_rho" in results
    assert "family_balanced_mc_by_rho" in results
    assert "spec_rules_hash" in results

    # At least one rho must qualify on this synthetic family
    assert results["selected_rho"] in (0.90, 0.95, 0.99)
    assert not results["no_go"]

    # Verify that selected rho has the highest family-balanced MC among qualified rhos
    qualified = [float(r) for r in (0.90, 0.95, 0.99) if results["qualification_by_rho"][str(r)]]
    assert results["selected_rho"] in qualified

    best_qual_mc = max(results["family_balanced_mc_by_rho"][str(r)] for r in qualified)
    assert abs(results["family_balanced_mc_by_rho"][str(results["selected_rho"])] - best_qual_mc) < 1e-6

    # Verify graph hashes structure
    for fam in PILOT_FAMILIES:
        assert len(results["graph_hashes"][fam]) == FAMILY_INSTANCE_COUNTS[fam]
        for sha in results["graph_hashes"][fam]:
            assert len(sha) == 64  # Valid SHA256 hex string


def test_pilot_all_disqualified_leads_to_nogo():
    """Verify that if all candidate rhos fail a gate, pilot triggers NO-GO."""
    from src.connectome.synthetic import make_synthetic_connectome

    # Graph with zero sensory connections or disconnected motor -> fails gates or forces failure
    base = make_synthetic_connectome(n_neurons=40, avg_out_degree=4.0, n_sensory=5, n_motor=5, seed=123)

    # Mock provider returning base graph
    def provider(family: str, instance_idx: int):
        return base

    fast_train_seqs = [
        generate_narma10_sequence(seed=s, valid_length=500, washout=100)
        for s in TRAIN_SEEDS
    ]

    # Test with candidate rhos that are set very high (e.g. 1.20, 1.30) where forgetting gate fails
    results = run_g1_pilot(
        graph_provider=provider,
        candidate_rhos=(1.20, 1.30),
        train_sequences=fast_train_seqs,
    )

    assert results["no_go"] is True
    assert results["selected_rho"] is None
    assert len(results["no_go_reasons"]) == 2
    for r_str in ["1.2", "1.3"]:
        assert r_str in results["no_go_reasons"]
        assert len(results["no_go_reasons"][r_str]) > 0


def test_pilot_deterministic_reproducibility():
    """Verify that pilot results are bitwise/numerically reproducible."""
    provider1 = make_synthetic_pilot_provider(n_neurons=40, avg_out_degree=4.0, n_sensory=8, n_motor=8, base_seed=99)
    provider2 = make_synthetic_pilot_provider(n_neurons=40, avg_out_degree=4.0, n_sensory=8, n_motor=8, base_seed=99)

    fast_train_seqs1 = [generate_narma10_sequence(seed=s, valid_length=500, washout=100) for s in TRAIN_SEEDS]
    fast_train_seqs2 = [generate_narma10_sequence(seed=s, valid_length=500, washout=100) for s in TRAIN_SEEDS]

    res1 = run_g1_pilot(graph_provider=provider1, candidate_rhos=(0.90, 0.95), train_sequences=fast_train_seqs1)
    res2 = run_g1_pilot(graph_provider=provider2, candidate_rhos=(0.90, 0.95), train_sequences=fast_train_seqs2)

    assert res1["selected_rho"] == res2["selected_rho"]
    assert res1["no_go"] == res2["no_go"]
    assert res1["family_balanced_mc_by_rho"] == res2["family_balanced_mc_by_rho"]


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
    """Verify pilot end-to-end execution for both GO and GATE_FAILURE scenarios with JSON serialization."""
    # Fast sequences for test execution
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

    # 1. Scenario GO: using planted q engine factory to pass q positive-control gate
    res_go = run_g1_pilot(
        graph_provider=provider,
        candidate_rhos=(0.90,),
        train_sequences=fast_train_seqs,
        engine_factory=make_planted_q_engine_factory(),
        n_bootstraps=100,
    )

    assert res_go["gate_verdict"] == "GO"
    assert res_go["no_go"] is False
    assert res_go["no_go_category"] is None
    assert res_go["is_valid_run"] is True
    assert res_go["selected_rho"] == 0.90
    assert len(res_go["spec_rules_hash"]) == 64
    assert res_go["mc_ratio_diagnostic"]["rho"] == 0.90
    assert res_go["nested_cv_results"]["q_gate_results"]["passed_q_gate"] is True

    # Verify JSON serialization roundtrip
    serialized = json.dumps(res_go)
    deserialized = json.loads(serialized)
    assert deserialized["gate_verdict"] == "GO"
    assert deserialized["is_valid_run"] is True

    # 2. Scenario GATE_FAILURE: candidate rhos fail dynamics gates (e.g. forgetting gate failure)
    res_fail = run_g1_pilot(
        graph_provider=provider,
        candidate_rhos=(1.20, 1.30),
        train_sequences=fast_train_seqs,
        n_bootstraps=100,
    )

    assert res_fail["gate_verdict"] == "NO_GO"
    assert res_fail["no_go"] is True
    assert res_fail["no_go_category"] == "GATE_FAILURE"
    assert res_fail["is_valid_run"] is True
    assert res_fail["selected_rho"] is None
    assert len(res_fail["no_go_reasons"]) == 2


def test_nested_cv_non_leakage_and_isolation():
    """Verify that nested 5-fold CV strictly isolates outer sequences from inner CV selection."""
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

    res = run_g1_pilot(
        graph_provider=provider,
        candidate_rhos=(0.90,),
        train_sequences=fast_train_seqs,
        n_bootstraps=50,
    )

    nested_results = res["nested_cv_results"]
    assert nested_results["n_folds"] == 5
    fold_details = nested_results["fold_details"]
    assert len(fold_details) == 5

    all_outer_indices = set()
    for f, details in enumerate(fold_details):
        assert details["fold"] == f
        outer_indices = details["outer_sequence_indices"]
        inner_indices = details["inner_sequence_indices"]

        # Exactly 2 outer sequences, exactly 8 inner sequences
        assert len(outer_indices) == 2
        assert len(inner_indices) == 8
        assert outer_indices == [f * 2, f * 2 + 1]

        # Zero leakage: outer and inner sets are strictly disjoint
        outer_set = set(outer_indices)
        inner_set = set(inner_indices)
        assert len(outer_set & inner_set) == 0, f"Fold {f} leaked outer sequences into inner CV: {outer_set & inner_set}"
        assert outer_set | inner_set == set(range(10))

        all_outer_indices.update(outer_indices)

    # All 10 Train sequences were used across the 5 outer folds exactly once
    assert all_outer_indices == set(range(10))

