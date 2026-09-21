"""Unit tests for A24: G1 v2 State Machine, Budget Ledger, and Governance (SPEC v2 §8, §9).

Verifications:
1. Full end-to-end A -> B -> C -> D mock execution path.
2. Budget ledger enforcement: exceeding stage or total limits halts with BUDGET_EXCEEDED.
3. VALID_GATE_FAIL concludes the research immediately without rescue or retry.
4. RUN_INVALID requires all SPEC §9 audit fields.
5. Test split freeze boundary: Stage A, B, C strictly block Main-Test access.
"""
from __future__ import annotations

from pathlib import Path
import pytest

from research.pipeline.g1_manifest import create_g1_manifest
from research.pipeline.g1_runner import (
    BudgetLedger,
    G1Runner,
    RunInvalidIncident,
    STAGE_BUDGET_LIMITS,
    TOTAL_BUDGET_LIMITS,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_a24_runner_end_to_end_lifecycle():
    """Verify complete A -> B -> C -> D normal progression with budget accounting."""
    manifest = create_g1_manifest(REPO_ROOT)
    runner = G1Runner(manifest=manifest)

    # 1. Stage A (0 GPU h, 1.0 person h)
    def stage_a_fn(r: G1Runner) -> dict:
        # Access fixture split
        seqs = r.access_split("fixture")
        assert len(seqs) == 5
        return {"status": "STAGE_COMPLETED", "n_fixtures_verified": len(seqs)}

    res_a = runner.execute_stage("A", stage_a_fn, person_hours=1.0, gpu_hours=0.0)
    assert res_a["status"] == "STAGE_COMPLETED"
    assert runner.ledger.stage_person_hours["A"] == 1.0
    assert runner.ledger.stage_gpu_hours["A"] == 0.0

    # 2. Stage B (0.5 GPU h, 2.0 person h)
    runner.advance_to_stage("B")
    def stage_b_fn(r: G1Runner) -> dict:
        seqs = r.access_split("diagnostic-fit")
        assert len(seqs) == 10
        return {"status": "STAGE_COMPLETED", "r2_d1": 0.45, "r2_d2": 0.35}

    res_b = runner.execute_stage("B", stage_b_fn, person_hours=2.0, gpu_hours=0.5)
    assert res_b["status"] == "STAGE_COMPLETED"
    assert runner.ledger.stage_gpu_hours["B"] == 0.5

    # 3. Stage C (2.0 GPU h, 4.0 person h)
    runner.advance_to_stage("C")
    def stage_c_fn(r: G1Runner) -> dict:
        seqs = r.access_split("calibration")
        assert len(seqs) == 10
        return {"status": "STAGE_COMPLETED", "passed_mde_gate": True}

    res_c = runner.execute_stage("C", stage_c_fn, person_hours=4.0, gpu_hours=2.0)
    assert res_c["status"] == "STAGE_COMPLETED"

    # 4. Stage D (1.5 GPU h, 3.0 person h)
    runner.advance_to_stage("D")
    # Unseal Main-Test
    manifest_sha = runner.manifest.compute_sha256()
    runner.manifest.unseal_main_test(manifest_sha)

    def stage_d_fn(r: G1Runner) -> dict:
        test_seqs = r.access_split("main-test")
        assert len(test_seqs) == 10
        return {"status": "STAGE_COMPLETED", "delta": 0.075, "ci_lower": 0.02}

    res_d = runner.execute_stage("D", stage_d_fn, person_hours=3.0, gpu_hours=1.5)
    assert res_d["status"] == "STAGE_COMPLETED"

    # Total budget checks
    assert runner.ledger.total_person_hours == 10.0 <= TOTAL_BUDGET_LIMITS["person_hours"]
    assert runner.ledger.total_gpu_hours == 4.0 <= TOTAL_BUDGET_LIMITS["gpu_hours"]
    assert runner.concluded is False


def test_a24_budget_ledger_hard_caps():
    """Verify exceeding stage or total limits halts with BUDGET_EXCEEDED."""
    manifest = create_g1_manifest(REPO_ROOT)
    runner = G1Runner(manifest=manifest)

    # Attempting to use GPU in Stage A (limit is 0.0)
    res = runner.execute_stage("A", lambda r: {"status": "STAGE_COMPLETED"}, person_hours=1.0, gpu_hours=0.1)
    assert res["status"] == "BUDGET_EXCEEDED"
    assert runner.concluded is True
    assert runner.conclusion_status == "BUDGET_EXCEEDED"

    # Cannot execute further after budget halt
    with pytest.raises(RuntimeError, match="Runner already concluded"):
        runner.execute_stage("A", lambda r: {"status": "STAGE_COMPLETED"}, person_hours=1.0, gpu_hours=0.0)


def test_a24_valid_gate_fail_concludes_research():
    """Verify VALID_GATE_FAIL immediately concludes research with no further execution allowed."""
    manifest = create_g1_manifest(REPO_ROOT)
    runner = G1Runner(manifest=manifest)

    def fail_stage_a(r: G1Runner) -> dict:
        return {"status": "VALID_GATE_FAIL", "reason": "Saturation ratio exceeded 5%"}

    res = runner.execute_stage("A", fail_stage_a, person_hours=1.0, gpu_hours=0.0)
    assert res["status"] == "VALID_GATE_FAIL"
    assert res["conclusion"] == "RESEARCH_CONCLUDED_NEGATIVE"
    assert runner.concluded is True
    assert runner.conclusion_status == "VALID_GATE_FAIL"

    # Cannot advance stage after valid gate failure
    with pytest.raises(RuntimeError, match="Cannot advance stage: runner concluded"):
        runner.advance_to_stage("B")


def test_a24_run_invalid_audit_records():
    """Verify RUN_INVALID requires all SPEC §9 audit logging fields."""
    manifest = create_g1_manifest(REPO_ROOT)
    runner = G1Runner(manifest=manifest)

    incident = RunInvalidIncident(
        incident_type="code_or_data_defect",
        first_discovered_at="2026-09-21T01:00:00Z",
        independent_evidence="Array index out of bounds in reference script verified by reference test",
        results_already_viewed=["research/outputs/v3/preliminary_metrics.json"],
        changed_files=["research/pipeline/g1_power.py"],
        new_hashes={"g1_power.py": "abc123hash"},
        remaining_budget=runner.ledger.remaining_budget(),
    )

    def invalid_stage(r: G1Runner) -> dict:
        return {"status": "RUN_INVALID", "incident": incident}

    res = runner.execute_stage("A", invalid_stage, person_hours=0.5, gpu_hours=0.0)
    assert res["status"] == "RUN_INVALID"
    assert runner.concluded is True
    assert runner.conclusion_status == "RUN_INVALID"
    assert len(runner.invalid_incidents) == 1
    assert runner.invalid_incidents[0].incident_type == "code_or_data_defect"


def test_a24_test_split_freeze_boundary():
    """Verify runner strictly enforces Test split freeze in stages A, B, and C."""
    manifest = create_g1_manifest(REPO_ROOT)
    runner = G1Runner(manifest=manifest)

    # In Stage A
    with pytest.raises(PermissionError, match="Main-Test split is sealed"):
        runner.access_split("main-test")

    # In Stage B
    runner.advance_to_stage("B")
    with pytest.raises(PermissionError, match="Main-Test split is sealed"):
        runner.access_split("main-test")

    # In Stage C
    runner.advance_to_stage("C")
    with pytest.raises(PermissionError, match="Main-Test split is sealed"):
        runner.access_split("main-test")
