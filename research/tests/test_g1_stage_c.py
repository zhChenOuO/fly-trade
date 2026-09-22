"""Unit tests for Stage C Runner (SPEC v2 §3.2, §4, §5, §7, §8, §9).

Verifications:
1. End-to-end Stage C fixture execution (1 real + 4 scramble controls, CPU):
   generates complete report JSON, verifies all keys, primary Delta, CI, and MC bounds.
2. Dynamics health gate failure leads to STAGE_C_NO_GO (exit 1).
3. Main-Test split access is strictly prohibited and raises PermissionError in Stage C.
4. Deterministic reproducibility: fixed seeds yield identical Delta and CI.
5. Budget ledger enforcement: exceeding Stage C GPU limit (4.0 h) or total (8.0 h) halts.
"""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from research.pipeline.g1_manifest import create_g1_manifest, generate_split_sequences
from research.pipeline.g1_stage_c import execute_stage_c, make_stage_c_fixture_graph
from research.pipeline.g1_runner import BudgetLedger, STAGE_BUDGET_LIMITS, TOTAL_BUDGET_LIMITS

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_stage_c_end_to_end_fixture_execution(tmp_path: Path):
    """Verify complete Stage C end-to-end execution on small fixture connectome."""
    out_json = tmp_path / "stage_c_fixture_report.json"

    report, exit_code = execute_stage_c(
        graph_source="fixture",
        device="cpu",
        engine="scipy",
        repo_root=REPO_ROOT,
        output_path=out_json,
        fixture_mode="go",
        n_controls=4,
        n_replicates=20,
        n_bootstraps=100,
        save_report=True,
    )

    # 1. Output file written to target path (and NOT to research/outputs/v3/)
    assert out_json.is_file()
    assert exit_code in [0, 1]  # 0 for GO, 1 for NO_GO
    assert report["status"] in ["STAGE_C_GO", "STAGE_C_NO_GO"]
    assert report["stage"] == "C"

    # 2. Check report structure and schema
    assert "hashes" in report
    assert len(report["hashes"]["manifest_sha256"]) == 64
    assert len(report["hashes"]["real_graph_sha256_scaled"]) == 64
    assert len(report["hashes"]["control_scaled_sha256s"]) == 4

    # 3. Check graph records
    assert "real" in report["graphs"]
    assert report["graphs"]["real"]["is_real"] is True
    assert "train_nmse" in report["graphs"]["real"]
    assert "calib_nmse" in report["graphs"]["real"]
    assert report["graphs"]["real"]["health_passed"] is True

    assert "controls" in report["graphs"]
    assert len(report["graphs"]["controls"]) == 4
    for ctrl in report["graphs"]["controls"]:
        assert ctrl["is_real"] is False
        assert "calib_nmse" in ctrl
        assert ctrl["health_passed"] is True

    # 4. Check primary contrast
    pc = report["primary_contrast"]
    assert "real_calib_nmse" in pc
    assert "control_median_calib_nmse" in pc
    assert "delta" in pc
    assert "ci_lower_95" in pc
    assert "ci_upper_95" in pc
    assert pc["ci_lower_95"] <= pc["ci_upper_95"]

    # 5. Check statistical calibration
    sc = report["statistical_calibration"]
    assert "null_scale" in sc
    assert "alt_scale" in sc
    assert "bounds" in sc
    assert "l_power_975" in sc["bounds"]
    assert "u_fpr_975" in sc["bounds"]

    # 6. Check gates
    gates = report["gates"]
    assert gates["passed_health_gates"] is True
    assert gates["passed_alpha_boundary"] is True
    assert "passed_power_gate" in gates
    assert "passed_fpr_gate" in gates
    assert "passed_budget_gate" in gates


def test_stage_c_dynamics_gate_failure_nogo(tmp_path: Path):
    """Verify that a dynamics health gate failure correctly triggers STAGE_C_NO_GO."""
    out_json = tmp_path / "stage_c_sat_fail.json"

    report, exit_code = execute_stage_c(
        graph_source="fixture",
        device="cpu",
        engine="scipy",
        repo_root=REPO_ROOT,
        output_path=out_json,
        fixture_mode="saturation_fail",
        n_controls=2,
        n_replicates=10,
        n_bootstraps=50,
        save_report=True,
    )

    assert exit_code == 1
    assert report["status"] == "STAGE_C_NO_GO"
    assert report["gates"]["passed_health_gates"] is False
    assert report["gates"]["stage_c_go"] is False


def test_stage_c_main_test_access_strictly_prohibited():
    """Verify that attempting to generate or unseal Main-Test in Stage C raises PermissionError."""
    manifest = create_g1_manifest(REPO_ROOT)
    manifest.advance_stage("B")
    manifest.advance_stage("C")

    # Generating Main-Test raises PermissionError
    with pytest.raises(PermissionError, match="Main-Test split is sealed in stage 'C'"):
        generate_split_sequences("main-test", manifest)

    # Attempting to unseal in Stage C raises PermissionError
    with pytest.raises(PermissionError, match="Main-Test split cannot be unsealed in stage 'C'"):
        manifest.unseal_main_test("arbitrary_hash")


def test_stage_c_deterministic_reproducibility(tmp_path: Path):
    """Verify that running Stage C with fixed seeds produces reproducible Delta and CI."""
    out1 = tmp_path / "rep1.json"
    out2 = tmp_path / "rep2.json"

    # Pre-generate custom fixture graph to ensure identical topology
    fixed_graph = make_stage_c_fixture_graph(mode="go", seed=999)

    rep1, code1 = execute_stage_c(
        graph_source="fixture",
        device="cpu",
        engine="scipy",
        repo_root=REPO_ROOT,
        output_path=out1,
        custom_real_graph=fixed_graph,
        n_controls=3,
        n_replicates=10,
        n_bootstraps=50,
        save_report=False,
    )

    rep2, code2 = execute_stage_c(
        graph_source="fixture",
        device="cpu",
        engine="scipy",
        repo_root=REPO_ROOT,
        output_path=out2,
        custom_real_graph=fixed_graph,
        n_controls=3,
        n_replicates=10,
        n_bootstraps=50,
        save_report=False,
    )

    assert code1 == code2
    assert rep1["primary_contrast"]["delta"] == rep2["primary_contrast"]["delta"]
    assert rep1["primary_contrast"]["ci_lower_95"] == rep2["primary_contrast"]["ci_lower_95"]
    assert rep1["primary_contrast"]["ci_upper_95"] == rep2["primary_contrast"]["ci_upper_95"]


def test_stage_c_budget_ledger_enforcement():
    """Verify that exceeding Stage C GPU limit or total GPU limit raises budget error."""
    ledger = BudgetLedger()

    # Stage C GPU limit is 4.0 hours
    limit_c = STAGE_BUDGET_LIMITS["C"]["gpu_hours"]
    assert limit_c == 4.0

    # Within budget passes
    ledger.check_limits("C", projected_gpu=3.5, projected_person=4.0)

    # Exceeding Stage C GPU limit raises
    with pytest.raises(RuntimeError, match="Budget exceeded for Stage C"):
        ledger.check_limits("C", projected_gpu=4.5)

    # Exceeding total GPU limit (8.0 hours) raises
    ledger.record_usage("A", person_hours=2.0, gpu_hours=3.5)
    ledger.record_usage("B", person_hours=2.0, gpu_hours=3.0)
    # Total so far = 6.5 hours. Stage C limit is 4.0. Projected 2.0 is <= 4.0, but total 6.5 + 2.0 = 8.5 > 8.0.
    with pytest.raises(RuntimeError, match="Total budget exceeded"):
        ledger.check_limits("C", projected_gpu=2.0)
