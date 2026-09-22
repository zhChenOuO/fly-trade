"""Tests for G1 v2 Stage B Pipeline Runner (research/pipeline/g1_stage_b.py).

Governing Specification: research/G1_SPEC_v2_draft.md (§3.2, §4, §6, §8, §9)
Review & Task Directives: scratchpad/task_stageB_runner.md, REMOTE_TASK_G1b.md

Test Coverage:
1. End-to-end fixture execution: full pipeline passes all gates (status STAGE_COMPLETED, exit 0).
2. Failure scenarios:
   - Disconnected DN -> capability gate fails (VALID_GATE_FAIL, exit 1).
   - High saturation -> health gate fails (VALID_GATE_FAIL, exit 1).
   - Provenance mismatch (DN count != 1291 for real) -> run invalid (RUN_INVALID, exit 2).
3. Boundary & split access guards:
   - Stage B loader strictly forbids main-test, main-train, main-val, and calibration.
4. Determinism: fixed seed produces identical metrics and CI bounds.
5. Zero NARMA metrics: output report JSON contains no NARMA prediction metrics.
6. Device & Torch handling: device honestly recorded; require-cuda fails closed when unavailable.
7. Budget & timing extrapolation: matches REMOTE_TASK_G1a §3 update counts and SPEC §8 limits.
8. Oracle positive controls: verify direct lag/product teachers achieve R^2 >= 1.0 - 1e-6.
"""
from __future__ import annotations

import json
from pathlib import Path
import pytest
import numpy as np

from research.pipeline.g1_manifest import create_g1_manifest
from research.pipeline.g1_stage_b import (
    EXPECTED_DN_COUNT,
    STAGE_B_UPDATES,
    STAGE_C_UPDATES,
    STAGE_D_UPDATES,
    TOTAL_PROJECT_UPDATES,
    ProvenanceError,
    SplitAccessError,
    access_stage_b_split,
    evaluate_stage_b_oracle_controls,
    execute_stage_b,
    make_stage_b_fixture_graph,
)
from research.pipeline.g1_runner import STAGE_BUDGET_LIMITS, TOTAL_BUDGET_LIMITS


# ==============================================================================
# 1. End-to-End Fixture Execution (Go Scenario)
# ==============================================================================

def test_stage_b_fixture_go_end_to_end(tmp_path: Path):
    """Verify full Stage B execution passes all gates on fixture graph with exit code 0."""
    out_json = tmp_path / "stage_b_report_go.json"
    report, exit_code = execute_stage_b(
        graph_type="fixture",
        fixture_mode="go",
        device="cpu",
        engine="scipy",
        output_path=out_json,
        n_bootstraps=200,
    )

    assert exit_code == 0
    assert report["status"] == "STAGE_COMPLETED"
    assert report["conclusion"] == "STAGE_COMPLETED"
    assert report["exit_code"] == 0

    # Environment
    assert report["environment"]["requested_device"] == "cpu"
    assert report["environment"]["actual_engine"] == "scipy"

    # Hashes
    assert len(report["hashes"]["spec_sha256"]) == 64
    assert len(report["hashes"]["manifest_sha256"]) == 64
    assert len(report["hashes"]["graph_sha256_unscaled"]) == 64
    assert len(report["hashes"]["graph_sha256_scaled"]) == 64

    # Provenance
    assert report["provenance"]["graph_type"] == "fixture"
    assert report["provenance"]["spectral_radius_target"] == 0.95
    assert abs(report["provenance"]["spectral_radius_verified"] - 0.95) < 1e-3

    # Health Gates
    health = report["health_gates"]
    assert health["passed"] is True
    assert health["saturation_gate"]["passed"] is True
    assert health["saturation_gate"]["saturation_ratio"] < 0.05
    assert health["forgetting_gate"]["passed"] is True
    assert health["forgetting_gate"]["n_passed_pairs"] >= 38
    assert health["finite_check"]["passed"] is True

    # Capability Gates (Dual AND)
    cap = report["capability_gates"]
    assert cap["passed"] is True
    assert cap["d1"]["passed"] is True
    assert cap["d1"]["r2_point"] >= 0.10
    assert cap["d1"]["ci_lower"] > 0.0
    assert cap["d1"]["alpha_at_upper_bound"] is False
    assert cap["d2"]["passed"] is True
    assert cap["d2"]["r2_point"] >= 0.10
    assert cap["d2"]["ci_lower"] > 0.0
    assert cap["d2"]["alpha_at_upper_bound"] is False

    # Negative Control
    neg = report["negative_control"]
    assert neg["passed"] is True
    assert neg["d1"]["ci_lower"] <= 0.0
    assert neg["d2"]["ci_lower"] <= 0.0

    # Oracle Control
    oracle = report["oracle_controls"]
    assert oracle["passed"] is True
    assert oracle["d1_teacher_r2"] >= 1.0 - 1e-6
    assert oracle["d2_teacher_r2"] >= 1.0 - 1e-6

    # Budget & Timing
    budget = report["timing_and_budget"]
    assert budget["budget_passed"] is True
    assert budget["wall_time_simulation_seconds"] > 0.0
    assert budget["wall_time_ridge_seconds"] > 0.0
    assert budget["peak_ram_mb"] > 0.0

    # File output verification
    assert out_json.is_file()
    with open(out_json, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["status"] == "STAGE_COMPLETED"


# ==============================================================================
# 2. Failure Scenarios (Valid Gate Fail & Run Invalid)
# ==============================================================================

def test_stage_b_disconnected_dn_capability_gate_fail(tmp_path: Path):
    """Verify disconnected DN leads to capability gate failure (VALID_GATE_FAIL, exit 1)."""
    out_json = tmp_path / "stage_b_disc.json"
    report, exit_code = execute_stage_b(
        graph_type="fixture",
        fixture_mode="disconnected_dn",
        device="cpu",
        output_path=out_json,
        n_bootstraps=100,
    )

    assert exit_code == 1
    assert report["status"] == "VALID_GATE_FAIL"
    assert report["conclusion"] == "VALID_GATE_FAIL"
    assert report["capability_gates"]["passed"] is False
    assert report["health_gates"]["passed"] is True


def test_stage_b_high_saturation_health_gate_fail(tmp_path: Path):
    """Verify high saturation causes health gate failure (VALID_GATE_FAIL, exit 1)."""
    out_json = tmp_path / "stage_b_sat.json"
    report, exit_code = execute_stage_b(
        graph_type="fixture",
        fixture_mode="high_saturation",
        device="cpu",
        output_path=out_json,
        n_bootstraps=50,
    )

    assert exit_code == 1
    assert report["status"] == "VALID_GATE_FAIL"
    assert report["conclusion"] == "VALID_GATE_FAIL"
    assert report["health_gates"]["passed"] is False
    assert report["health_gates"]["saturation_gate"]["passed"] is False


def test_stage_b_provenance_mismatch_run_invalid(tmp_path: Path):
    """Verify DN count mismatch under real schema causes provenance blocking (RUN_INVALID, exit 2)."""
    out_json = tmp_path / "stage_b_prov.json"
    report, exit_code = execute_stage_b(
        graph_type="fixture",
        fixture_mode="provenance_fail",
        device="cpu",
        output_path=out_json,
    )

    assert exit_code == 2
    assert report["status"] == "RUN_INVALID"
    assert report["conclusion"] == "RUN_INVALID"
    assert "incident" in report
    assert report["incident"]["incident_type"] == "provenance_or_split_corruption"
    assert "remaining_budget" in report["incident"]


# ==============================================================================
# 3. Boundary & Split Access Guards
# ==============================================================================

def test_stage_b_split_access_guards():
    """Verify Stage B loader strictly forbids main-test, main-train, main-val, and calibration."""
    manifest = create_g1_manifest(".")
    manifest.advance_stage("B")

    # Authorized diagnostic splits succeed
    fit_seqs = access_stage_b_split("diagnostic-fit", manifest)
    assert len(fit_seqs) == 10
    check_seqs = access_stage_b_split("diagnostic-check", manifest)
    assert len(check_seqs) == 10

    # Unauthorized splits must raise SplitAccessError
    forbidden_splits = ["main-test", "main-train", "main-val", "calibration", "test", "all"]
    for split_name in forbidden_splits:
        with pytest.raises(SplitAccessError) as exc_info:
            access_stage_b_split(split_name, manifest)
        assert "strictly prohibited" in str(exc_info.value)


# ==============================================================================
# 4. Zero NARMA Metrics in Output JSON
# ==============================================================================

def test_stage_b_zero_narma_metrics_in_output(tmp_path: Path):
    """Verify output JSON contains absolutely NO NARMA prediction scores (SPEC §6, §8)."""
    out_json = tmp_path / "stage_b_check_narma.json"
    report, exit_code = execute_stage_b(
        graph_type="fixture",
        fixture_mode="go",
        device="cpu",
        output_path=out_json,
        n_bootstraps=50,
    )

    json_str = out_json.read_text(encoding="utf-8")
    loaded = json.loads(json_str)

    def _check_keys_recursive(obj, prefix=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                lower_k = k.lower()
                assert "narma" not in lower_k, f"Forbidden key containing 'narma' found: {prefix}.{k}"
                _check_keys_recursive(v, f"{prefix}.{k}")
        elif isinstance(obj, list):
            for i, elem in enumerate(obj):
                _check_keys_recursive(elem, f"{prefix}[{i}]")

    _check_keys_recursive(loaded)


# ==============================================================================
# 5. Determinism Check
# ==============================================================================

def test_stage_b_determinism(tmp_path: Path):
    """Verify fixed seeds produce bitwise reproducible R^2 and bootstrap CI results."""
    out1 = tmp_path / "rep1.json"
    out2 = tmp_path / "rep2.json"

    rep1, code1 = execute_stage_b(
        graph_type="fixture",
        fixture_mode="go",
        device="cpu",
        output_path=out1,
        n_bootstraps=100,
    )
    rep2, code2 = execute_stage_b(
        graph_type="fixture",
        fixture_mode="go",
        device="cpu",
        output_path=out2,
        n_bootstraps=100,
    )

    assert code1 == code2 == 0
    # Capability d1 and d2 points and CI lower bounds must match exactly
    assert rep1["capability_gates"]["d1"]["r2_point"] == rep2["capability_gates"]["d1"]["r2_point"]
    assert rep1["capability_gates"]["d1"]["ci_lower"] == rep2["capability_gates"]["d1"]["ci_lower"]
    assert rep1["capability_gates"]["d2"]["r2_point"] == rep2["capability_gates"]["d2"]["r2_point"]
    assert rep1["capability_gates"]["d2"]["ci_lower"] == rep2["capability_gates"]["d2"]["ci_lower"]


# ==============================================================================
# 6. Device & Require-CUDA Handling
# ==============================================================================

def test_stage_b_require_cuda_fails_closed_when_unavailable(tmp_path: Path):
    """Verify --require-cuda fails closed with RUN_INVALID if CUDA is unavailable."""
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False

    if not has_cuda:
        out_json = tmp_path / "cuda_fail.json"
        report, exit_code = execute_stage_b(
            graph_type="fixture",
            device="cuda",
            require_cuda=True,
            output_path=out_json,
        )
        assert exit_code == 2
        assert report["status"] == "RUN_INVALID"
        assert "CUDA" in report["error"]


# ==============================================================================
# 7. Budget & Timing Extrapolation Verification
# ==============================================================================

def test_stage_b_budget_extrapolation_formulas(tmp_path: Path):
    """Verify extrapolation arithmetic matches REMOTE_TASK_G1a §3 update counts and SPEC §8 limits."""
    out_json = tmp_path / "budget.json"
    report, exit_code = execute_stage_b(
        graph_type="fixture",
        fixture_mode="go",
        device="cpu",
        output_path=out_json,
        n_bootstraps=50,
    )

    budget = report["timing_and_budget"]
    t_step = budget["time_per_step_seconds"]

    # Verify update counts
    assert STAGE_B_UPDATES == 50000
    assert STAGE_C_UPDATES == 2467500
    assert STAGE_D_UPDATES == 1155000
    assert TOTAL_PROJECT_UPDATES == 3672500

    # Verify mathematical extrapolation
    expected_c_h = (STAGE_C_UPDATES * t_step) / 3600.0
    expected_d_h = (STAGE_D_UPDATES * t_step) / 3600.0
    assert abs(budget["extrapolated_stage_c_gpu_hours"] - expected_c_h) < 1e-9
    assert abs(budget["extrapolated_stage_d_gpu_hours"] - expected_d_h) < 1e-9

    # Verify limits
    assert budget["budget_limits"]["stage_b_gpu_limit"] == STAGE_BUDGET_LIMITS["B"]["gpu_hours"]
    assert budget["budget_limits"]["stage_c_gpu_limit"] == STAGE_BUDGET_LIMITS["C"]["gpu_hours"]
    assert budget["budget_limits"]["stage_d_gpu_limit"] == STAGE_BUDGET_LIMITS["D"]["gpu_hours"]
    assert budget["budget_limits"]["total_gpu_limit"] == TOTAL_BUDGET_LIMITS["gpu_hours"]


# ==============================================================================
# 8. Oracle Positive Controls Verification (Independent Reference & Error Injection)
# ==============================================================================

def test_stage_b_oracle_positive_controls_with_independent_reference():
    """Verify independent recurrence calculation matches runner targets (R^2 >= 1.0 - 1e-6, rel err <= 1e-10)."""
    from research.pipeline.g1_v2 import build_diagnostic_targets
    from research.pipeline.g1_stage_b import compute_independent_reference_diagnostic_targets

    rng = np.random.default_rng(123)
    n_seqs = 10
    washout = 500
    T = 2500

    check_sequences = []
    targets_d1 = []
    targets_d2 = []

    for _ in range(n_seqs):
        u = rng.uniform(0.0, 0.5, size=T).astype(np.float64)
        check_sequences.append({"u": u, "washout": washout})
        d1, d2 = build_diagnostic_targets(u, washout=washout)
        targets_d1.append(d1)
        targets_d2.append(d2)

    res = evaluate_stage_b_oracle_controls(
        check_targets_d1=targets_d1,
        check_targets_d2=targets_d2,
        check_sequences=check_sequences,
    )
    assert res["passed"] is True
    assert res["mode"] == "independent_reference"
    assert res["d1_teacher_r2"] >= 1.0 - 1e-6
    assert res["d2_teacher_r2"] >= 1.0 - 1e-6
    assert res["d1_max_rel_diff"] <= 1e-10
    assert res["d2_max_rel_diff"] <= 1e-10
    assert res["d1_max_abs_diff"] <= 1e-12
    assert res["d2_max_abs_diff"] <= 1e-12


def test_stage_b_oracle_offset_by_one_injection_fails():
    """Deliberate offset-by-one error injection must cause oracle check to FAIL with R^2 << 1.0."""
    from research.pipeline.g1_v2 import build_diagnostic_targets

    rng = np.random.default_rng(456)
    n_seqs = 5
    washout = 500
    T = 1500

    check_sequences = []
    targets_d1 = []
    targets_d2 = []

    for _ in range(n_seqs):
        u = rng.uniform(0.0, 0.5, size=T).astype(np.float64)
        check_sequences.append({"u": u, "washout": washout})
        d1, d2 = build_diagnostic_targets(u, washout=washout)
        targets_d1.append(d1)
        targets_d2.append(d2)

    # Inject offset-by-one into targets: slice targets shifted by 1 vs reference
    shifted_d1 = [arr[1:] for arr in targets_d1]
    # Corresponding check_sequences truncated to match length
    truncated_seqs = [{"u": s["u"][:-1], "washout": washout} for s in check_sequences]

    res = evaluate_stage_b_oracle_controls(
        check_targets_d1=shifted_d1,
        check_targets_d2=[arr[1:] for arr in targets_d2],
        check_sequences=truncated_seqs,
    )
    # The check MUST fail because i.i.d. noise shifted by 1 has ~0 correlation
    assert res["passed"] is False
    assert res["d1_teacher_r2"] < 0.5
    assert res["d1_max_rel_diff"] > 0.1


def test_independent_reference_diagnostic_targets_multi_seed_match():
    """compute_independent_reference_diagnostic_targets must match build_diagnostic_targets across multiple seeds."""
    from research.pipeline.g1_v2 import build_diagnostic_targets
    from research.pipeline.g1_stage_b import compute_independent_reference_diagnostic_targets

    for seed in [11, 22, 33, 44, 55]:
        rng = np.random.default_rng(seed)
        u = rng.uniform(0.0, 0.5, size=2000).astype(np.float64)
        u_neg = rng.uniform(0.0, 0.5, size=9).astype(np.float64)

        # Case 1: with u_negative
        ref_d1, ref_d2 = compute_independent_reference_diagnostic_targets(u, washout=500, u_negative=u_neg)
        prod_d1, prod_d2 = build_diagnostic_targets(u, washout=500, u_negative=u_neg)
        assert np.array_equal(ref_d1, prod_d1)
        assert np.array_equal(ref_d2, prod_d2)

        # Case 2: without u_negative (default zeros -> -0.25)
        ref_d1_0, ref_d2_0 = compute_independent_reference_diagnostic_targets(u, washout=500, u_negative=None)
        prod_d1_0, prod_d2_0 = build_diagnostic_targets(u, washout=500, u_negative=None)
        assert np.array_equal(ref_d1_0, prod_d1_0)
        assert np.array_equal(ref_d2_0, prod_d2_0)


def test_stage_b_oracle_deprecated_fallback_emits_warning():
    """Calling evaluate_stage_b_oracle_controls without reference or sequences emits DeprecationWarning."""
    rng = np.random.default_rng(789)
    targets_d1 = [rng.normal(size=100) for _ in range(2)]
    targets_d2 = [rng.normal(size=100) for _ in range(2)]

    with pytest.deprecated_call():
        res = evaluate_stage_b_oracle_controls(targets_d1, targets_d2)
    assert res["passed"] is True
    assert res["mode"] == "deprecated_self_comparison"

