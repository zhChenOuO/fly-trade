"""G1 v2 State Machine, Budget Ledger, and Governance Runner (SPEC v2 §8, §9, A24).

Contract:
- Stages: A -> B -> C -> D.
- Budget Ledger (SPEC §8):
  Total hard caps: 4.0 work days (32.0 person-hours), 8.0 GPU hours across all stages.
  Stage A: <= 4.0 person-hours (0.5 days), 0.0 GPU hours.
  Stage B: <= 8.0 person-hours (1.0 day), <= 1.0 GPU hours.
  Stage C: <= 8.0 person-hours (1.0 day), <= 4.0 GPU hours.
  Stage D: <= 12.0 person-hours (1.5 days), <= 3.0 GPU hours.
  Exceeding stage or total limits halts immediately with BUDGET_EXCEEDED.
- Outcome Classifications (SPEC §9):
  1. VALID_GATE_FAIL: Valid scientific failure (e.g. saturation, forgetting, capability R^2,
     power MDE bounds, test Delta < 0.05). Concludes the research without rescue.
  2. RUN_INVALID: Exceptional execution failure (bug, crash, split tampering, spec contradiction).
     Requires logging all §9 audit fields: incident_type, first_discovered_at, independent_evidence,
     results_already_viewed, changed_files, new_hashes, remaining_budget.
  3. STAGE_COMPLETED: Passed all requirements and eligible to proceed to next stage.
- Test Freeze Boundary:
  Access to Main-Test split strictly prohibited before Stage D.
"""
from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Any, Callable

from research.pipeline.g1_manifest import G1Manifest, generate_split_sequences


# ==============================================================================
# 1. Budget Ledger & Hard Limits (SPEC §8, A24)
# ==============================================================================

STAGE_BUDGET_LIMITS: dict[str, dict[str, float]] = {
    "A": {"person_hours": 4.0, "gpu_hours": 0.0},
    "B": {"person_hours": 8.0, "gpu_hours": 1.0},
    "C": {"person_hours": 8.0, "gpu_hours": 4.0},
    "D": {"person_hours": 12.0, "gpu_hours": 3.0},
}

TOTAL_BUDGET_LIMITS = {
    "person_hours": 32.0,  # 4 work days = 32 person-hours
    "gpu_hours": 8.0,      # 8 GPU hours total
}


@dataclass
class BudgetLedger:
    stage_person_hours: dict[str, float] = field(
        default_factory=lambda: {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0}
    )
    stage_gpu_hours: dict[str, float] = field(
        default_factory=lambda: {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0}
    )

    @property
    def total_person_hours(self) -> float:
        return sum(self.stage_person_hours.values())

    @property
    def total_gpu_hours(self) -> float:
        return sum(self.stage_gpu_hours.values())

    def record_usage(self, stage: str, person_hours: float, gpu_hours: float) -> None:
        """Record consumed hours for a stage and check hard limits."""
        if stage not in self.stage_person_hours:
            raise ValueError(f"Unknown stage: {stage}")
        self.stage_person_hours[stage] += float(person_hours)
        self.stage_gpu_hours[stage] += float(gpu_hours)

    def check_limits(self, stage: str, projected_person: float = 0.0, projected_gpu: float = 0.0) -> None:
        """Verify projected or current consumption does not exceed stage or total limits.

        Raises
        ------
        RuntimeError
            If stage or total limit would be exceeded.
        """
        curr_p = self.stage_person_hours[stage] + projected_person
        curr_g = self.stage_gpu_hours[stage] + projected_gpu
        limit_p = STAGE_BUDGET_LIMITS[stage]["person_hours"]
        limit_g = STAGE_BUDGET_LIMITS[stage]["gpu_hours"]

        if curr_p > limit_p:
            raise RuntimeError(
                f"Budget exceeded for Stage {stage}: person-hours {curr_p:.2f} > limit {limit_p:.2f}"
            )
        if curr_g > limit_g:
            raise RuntimeError(
                f"Budget exceeded for Stage {stage}: GPU-hours {curr_g:.2f} > limit {limit_g:.2f}"
            )

        tot_p = self.total_person_hours + projected_person
        tot_g = self.total_gpu_hours + projected_gpu
        if tot_p > TOTAL_BUDGET_LIMITS["person_hours"]:
            raise RuntimeError(
                f"Total budget exceeded: person-hours {tot_p:.2f} > limit {TOTAL_BUDGET_LIMITS['person_hours']:.2f}"
            )
        if tot_g > TOTAL_BUDGET_LIMITS["gpu_hours"]:
            raise RuntimeError(
                f"Total budget exceeded: GPU-hours {tot_g:.2f} > limit {TOTAL_BUDGET_LIMITS['gpu_hours']:.2f}"
            )

    def remaining_budget(self) -> dict[str, float]:
        """Compute remaining person and GPU hours."""
        return {
            "remaining_person_hours": max(0.0, TOTAL_BUDGET_LIMITS["person_hours"] - self.total_person_hours),
            "remaining_gpu_hours": max(0.0, TOTAL_BUDGET_LIMITS["gpu_hours"] - self.total_gpu_hours),
        }


# ==============================================================================
# 2. Run Invalid Incident Audit Record (SPEC §9)
# ==============================================================================

VALID_INCIDENT_TYPES = {
    "code_or_data_defect",
    "external_interruption",
    "provenance_or_split_corruption",
    "specification_contradiction",
}


@dataclass
class RunInvalidIncident:
    incident_type: str
    first_discovered_at: str
    independent_evidence: str
    results_already_viewed: list[str]
    changed_files: list[str]
    new_hashes: dict[str, str]
    remaining_budget: dict[str, float]

    def __post_init__(self):
        if self.incident_type not in VALID_INCIDENT_TYPES:
            raise ValueError(
                f"Invalid incident_type '{self.incident_type}'. Must be one of: {VALID_INCIDENT_TYPES}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "incident_type": self.incident_type,
            "first_discovered_at": self.first_discovered_at,
            "independent_evidence": self.independent_evidence,
            "results_already_viewed": self.results_already_viewed,
            "changed_files": self.changed_files,
            "new_hashes": self.new_hashes,
            "remaining_budget": self.remaining_budget,
        }


# ==============================================================================
# 3. G1 Runner & State Machine (A -> B -> C -> D)
# ==============================================================================

class G1Runner:
    """State machine coordinator managing A -> B -> C -> D transitions, budgets, and gates."""

    def __init__(self, manifest: G1Manifest, ledger: BudgetLedger | None = None):
        self.manifest = manifest
        self.ledger = ledger or BudgetLedger()
        self.history: list[dict[str, Any]] = []
        self.invalid_incidents: list[RunInvalidIncident] = []
        self.concluded: bool = False
        self.conclusion_status: str | None = None

    @property
    def current_stage(self) -> str:
        return self.manifest.stage

    def access_split(self, split_name: str) -> list[dict[str, Any]]:
        """Access sequence data for a split while enforcing Test split freeze."""
        return generate_split_sequences(split_name, self.manifest)

    def execute_stage(
        self,
        stage: str,
        stage_fn: Callable[[G1Runner], dict[str, Any]],
        person_hours: float,
        gpu_hours: float,
    ) -> dict[str, Any]:
        """Execute a pipeline stage under budget monitoring and outcome classification."""
        if self.concluded:
            raise RuntimeError(f"Runner already concluded with status '{self.conclusion_status}'")
        if stage != self.manifest.stage:
            raise ValueError(f"Runner is in stage '{self.manifest.stage}', cannot execute '{stage}'")

        # 1. Budget projection check
        try:
            self.ledger.check_limits(stage, projected_person=person_hours, projected_gpu=gpu_hours)
        except RuntimeError as e:
            self.concluded = True
            self.conclusion_status = "BUDGET_EXCEEDED"
            record = {
                "stage": stage,
                "status": "BUDGET_EXCEEDED",
                "error": str(e),
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
            self.history.append(record)
            return record

        # 2. Record resource usage
        self.ledger.record_usage(stage, person_hours, gpu_hours)

        # 3. Run stage function
        try:
            result = stage_fn(self)
        except Exception as e:
            # Uncaught error classified as potential invalid or unexpected exception
            record = {
                "stage": stage,
                "status": "EXECUTION_EXCEPTION",
                "error": str(e),
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
            self.history.append(record)
            raise

        status = result.get("status")
        if status == "VALID_GATE_FAIL":
            self.concluded = True
            self.conclusion_status = "VALID_GATE_FAIL"
            result["conclusion"] = "RESEARCH_CONCLUDED_NEGATIVE"
        elif status == "RUN_INVALID":
            incident = result.get("incident")
            if isinstance(incident, RunInvalidIncident):
                self.invalid_incidents.append(incident)
            self.concluded = True
            self.conclusion_status = "RUN_INVALID"
        elif status == "STAGE_COMPLETED":
            # Ready for next stage
            pass
        else:
            raise ValueError(f"Stage function must return valid status in result dict, got '{status}'")

        result["stage"] = stage
        result["timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.history.append(result)
        return result

    def advance_to_stage(self, next_stage: str) -> None:
        """Advance runner stage machine to next stage."""
        if self.concluded:
            raise RuntimeError(f"Cannot advance stage: runner concluded with '{self.conclusion_status}'")
        self.manifest.advance_stage(next_stage)
