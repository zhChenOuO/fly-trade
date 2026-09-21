"""G1 v2 Manifest, Provenance, Seed Namespaces, and Test Split Sealing (SPEC v2 §3.2, §8, A23).

Contract:
- Full enumeration of mutually exclusive seed namespaces:
  fixture, diagnostic-fit, diagnostic-check, main-train, main-val, calibration, main-test, scramble.
  Namespace sets are strictly disjoint (no overlapping seeds). Unknown seed/split fail closed.
- Provenance hashes:
  spec_sha256, source_snapshot_sha256 (hashes working tree source files, not just git HEAD),
  simulator_hash, readout_hash, scorer_hash, version ("2.0.0").
- Test split sealing:
  Main-Test sequences CANNOT be created or loaded during stages A, B, or C.
  Unseal requires stage == "D" AND an exact manifest SHA-256 match.
  Hash mismatch, missing files, or tampering fail closed.
- Sequence generation API:
  Must generate sequences strictly on a per-split basis.
  Generating all splits at once is strictly prohibited.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np


# ==============================================================================
# 1. Seed Namespaces Specification (SPEC v2 §3.2, A23)
# ==============================================================================

SEED_NAMESPACES: dict[str, tuple[int, ...]] = {
    # Formal namespaces use fresh ranges never used by earlier sanity runs or legacy g1_bench streams
    # (SPEC v2 §3.2: no reuse of streams whose performance was already seen).
    "fixture": tuple(range(9001, 9021)),                 # 20 fixture seeds for testing
    "diagnostic-fit": tuple(range(31001, 31011)),        # 10 sequences (valid 2000, washout 500)
    "diagnostic-check": tuple(range(31101, 31111)),      # 10 sequences (valid 2000, washout 500)
    "main-train": tuple(range(32001, 32011)),            # 10 sequences (valid 5000, washout 500)
    "main-val": tuple(range(32101, 32104)),              # 3 sequences (valid 2000, washout 500)
    "calibration": tuple(range(33001, 33011)),           # 10 sequences (valid 5000, washout 500)
    "main-test": tuple(range(34001, 34011)),             # 10 sequences (valid 5000, washout 500)
    "scramble": tuple(range(35001, 35021)),              # 20 scramble_mixed graph seeds
}

# Seeds already used by legacy pre-v2 g1_bench streams and v2 fixtures; formal namespaces must not reuse them.
LEGACY_AND_FIXTURE_SEEDS: frozenset[int] = frozenset(
    set(range(1001, 1011)) | set(range(2001, 2004)) | set(range(3001, 3011)) | {9999}
    | set(range(10001, 10011)) | set(range(20001, 20011))
)

SPLIT_SPECS: dict[str, dict[str, int]] = {
    "fixture": {"n_sequences": 5, "valid_length": 1000, "washout": 200},
    "diagnostic-fit": {"n_sequences": 10, "valid_length": 2000, "washout": 500},
    "diagnostic-check": {"n_sequences": 10, "valid_length": 2000, "washout": 500},
    "main-train": {"n_sequences": 10, "valid_length": 5000, "washout": 500},
    "main-val": {"n_sequences": 3, "valid_length": 2000, "washout": 500},
    "calibration": {"n_sequences": 10, "valid_length": 5000, "washout": 500},
    "main-test": {"n_sequences": 10, "valid_length": 5000, "washout": 500},
}


def verify_seed_namespaces_disjoint() -> None:
    """Verify that all seed namespaces are strictly disjoint and do not reuse legacy/fixture streams."""
    seen_seeds: dict[int, str] = {}
    for ns_name, seeds in SEED_NAMESPACES.items():
        for s in seeds:
            if s in seen_seeds:
                raise ValueError(
                    f"Seed collision detected: seed {s} appears in both "
                    f"'{seen_seeds[s]}' and '{ns_name}'"
                )
            if ns_name != "fixture" and s in LEGACY_AND_FIXTURE_SEEDS:
                raise ValueError(f"Seed {s} in '{ns_name}' reuses a legacy/fixture stream")
            seen_seeds[s] = ns_name


# Verify at import time
verify_seed_namespaces_disjoint()


# ==============================================================================
# 2. Source Snapshot & File Hashing
# ==============================================================================

def compute_file_sha256(filepath: str | Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    p = Path(filepath)
    if not p.is_file():
        raise FileNotFoundError(f"File not found for hashing: {p}")
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_source_snapshot_sha(repo_root: str | Path) -> str:
    """Compute working-tree source snapshot SHA-256 (including uncommitted edits).

    Scans and hashes Python pipeline and source files in sorted order so that any
    uncommitted modification in the working tree directly alters the snapshot SHA.
    """
    root = Path(repo_root)
    source_files: list[Path] = []

    # Include key directories
    for sub in ["research/pipeline", "src"]:
        d = root / sub
        if d.is_dir():
            for p in sorted(d.glob("**/*.py")):
                if "__pycache__" not in p.parts:
                    source_files.append(p)

    h = hashlib.sha256()
    for sf in source_files:
        rel_path = sf.relative_to(root).as_posix().encode("utf-8")
        h.update(rel_path)
        with open(sf, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
    return h.hexdigest()


# ==============================================================================
# 3. G1 v2 Manifest Data Structure & Seal/Unseal (SPEC §8, A23)
# ==============================================================================

@dataclass
class G1Manifest:
    version: str
    spec_sha256: str
    source_snapshot_sha256: str
    simulator_hash: str
    readout_hash: str
    scorer_hash: str
    seed_namespaces: dict[str, list[int]]
    split_specs: dict[str, dict[str, int]]
    stage: str = "A"
    is_main_test_sealed: bool = True
    manifest_sha256: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def compute_sha256(self) -> str:
        """Compute structural SHA-256 of the manifest contents (excluding manifest_sha256)."""
        d = self.to_dict()
        d.pop("manifest_sha256", None)
        canonical_json = json.dumps(d, sort_keys=True, indent=2).encode("utf-8")
        return hashlib.sha256(canonical_json).hexdigest()

    def advance_stage(self, target_stage: str) -> None:
        """Advance state machine stage (A -> B -> C -> D)."""
        valid_transitions = {
            "A": ["B"],
            "B": ["C"],
            "C": ["D"],
            "D": [],
        }
        if target_stage not in valid_transitions.get(self.stage, []):
            raise ValueError(
                f"Invalid stage transition from '{self.stage}' to '{target_stage}'. "
                f"Allowed next stages: {valid_transitions.get(self.stage, [])}"
            )
        self.stage = target_stage
        self.manifest_sha256 = self.compute_sha256()

    def unseal_main_test(self, provided_manifest_sha: str) -> None:
        """Unseal Main-Test split strictly during Stage D with verified manifest hash.

        Raises
        ------
        PermissionError
            If unseal is attempted before Stage D.
        ValueError
            If provided manifest hash does not match expected manifest SHA-256.
        """
        if self.stage != "D":
            raise PermissionError(
                f"Main-Test split cannot be unsealed in stage '{self.stage}'. "
                f"Unsealing is strictly restricted to Stage D confirmation per SPEC §3.2, §8."
            )
        current_sha = self.compute_sha256()
        if provided_manifest_sha != current_sha:
            raise ValueError(
                f"Manifest SHA-256 mismatch during unseal attempt: "
                f"expected '{current_sha}', provided '{provided_manifest_sha}'"
            )
        self.is_main_test_sealed = False


def create_g1_manifest(repo_root: str | Path) -> G1Manifest:
    """Create a fully populated G1 v2 manifest with hashes and disjoint seed namespaces."""
    root = Path(repo_root)
    spec_path = root / "research" / "G1_SPEC_v2_draft.md"
    sim_path = root / "research" / "pipeline" / "g1_reservoir.py"
    readout_path = root / "research" / "pipeline" / "g1_bench.py"
    scorer_path = root / "research" / "pipeline" / "g1_power.py"

    spec_sha = compute_file_sha256(spec_path)
    sim_sha = compute_file_sha256(sim_path)
    readout_sha = compute_file_sha256(readout_path)
    scorer_sha = compute_file_sha256(scorer_path)
    source_snap_sha = compute_source_snapshot_sha(root)

    manifest = G1Manifest(
        version="2.0.0",
        spec_sha256=spec_sha,
        source_snapshot_sha256=source_snap_sha,
        simulator_hash=sim_sha,
        readout_hash=readout_sha,
        scorer_hash=scorer_sha,
        seed_namespaces={k: list(v) for k, v in SEED_NAMESPACES.items()},
        split_specs=SPLIT_SPECS,
        stage="A",
        is_main_test_sealed=True,
    )
    manifest.manifest_sha256 = manifest.compute_sha256()
    return manifest


def verify_manifest_integrity(manifest: G1Manifest, repo_root: str | Path) -> dict[str, Any]:
    """Rigorously verify manifest integrity against the working tree (fail-closed)."""
    root = Path(repo_root)

    # 1. Verify seed namespaces are disjoint
    seen: set[int] = set()
    for ns, seeds in manifest.seed_namespaces.items():
        for s in seeds:
            if s in seen:
                raise ValueError(f"Integrity check failed: duplicate seed {s} in namespace '{ns}'")
            seen.add(s)

    # 2. Verify file hashes
    spec_path = root / "research" / "G1_SPEC_v2_draft.md"
    if compute_file_sha256(spec_path) != manifest.spec_sha256:
        raise ValueError("Integrity check failed: spec_sha256 mismatch")

    sim_path = root / "research" / "pipeline" / "g1_reservoir.py"
    if compute_file_sha256(sim_path) != manifest.simulator_hash:
        raise ValueError("Integrity check failed: simulator_hash mismatch")

    readout_path = root / "research" / "pipeline" / "g1_bench.py"
    if compute_file_sha256(readout_path) != manifest.readout_hash:
        raise ValueError("Integrity check failed: readout_hash mismatch")

    scorer_path = root / "research" / "pipeline" / "g1_power.py"
    if compute_file_sha256(scorer_path) != manifest.scorer_hash:
        raise ValueError("Integrity check failed: scorer_hash mismatch")

    # 3. Check manifest self-hash
    computed_sha = manifest.compute_sha256()
    if manifest.manifest_sha256 and manifest.manifest_sha256 != computed_sha:
        raise ValueError(
            f"Integrity check failed: manifest_sha256 mismatch ({manifest.manifest_sha256} vs {computed_sha})"
        )

    return {
        "verified": True,
        "version": manifest.version,
        "stage": manifest.stage,
        "is_main_test_sealed": manifest.is_main_test_sealed,
        "manifest_sha256": computed_sha,
    }


# ==============================================================================
# 4. Sequence Generation API per Split (SPEC §3.1, §3.2, A23)
# ==============================================================================

def generate_narma10_raw_stream(
    seed: int,
    valid_length: int,
    washout: int = 500,
) -> dict[str, np.ndarray]:
    """Generate discrete NARMA10 stream with raw u[t] ~ Uniform[0, 0.5] (SPEC §3.1).

    Recurrence:
        y[t+1] = 0.3 * y[t] + 0.05 * y[t] * sum_{k=0..9} y[t-k] + 1.5 * u[t] * u[t-9] + 0.1
    Initialization:
        y[-9..0] = 0.0, u[<0] = 0.0.
        t=0..499: washout.
        t=500..T-1: valid evaluation points.
    """
    T = washout + valid_length
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 0.5, size=T).astype(np.float64)

    y = np.zeros(T + 1, dtype=np.float64)
    for t in range(T):
        u_lag = u[t - 9] if t >= 9 else 0.0
        sum_y = 0.0
        for k in range(10):
            if t - k >= 0:
                sum_y += y[t - k]
        y[t + 1] = 0.3 * y[t] + 0.05 * y[t] * sum_y + 1.5 * u[t] * u_lag + 0.1

    y_next = y[1:]  # y_next[t] = y[t+1]
    return {
        "u": u,
        "y_next": y_next,
        "seed": seed,
        "valid_length": valid_length,
        "washout": washout,
    }


def generate_split_sequences(
    split_name: str,
    manifest: G1Manifest,
) -> list[dict[str, np.ndarray]]:
    """Generate sequences strictly for a single designated split (SPEC §3.2, A23).

    Prohibitions & Security Guards:
    - Attempting to generate all splits at once raises ValueError.
    - Unknown split name raises ValueError.
    - Generating Main-Test while sealed raises PermissionError.
    """
    if split_name.lower() in ["all", "*", "every"]:
        raise ValueError(
            "Generating all splits at once is strictly prohibited per SPEC §3.2 / A23. "
            "Sequences must be generated on a single-split basis."
        )

    if split_name not in SPLIT_SPECS or split_name not in manifest.seed_namespaces:
        raise ValueError(
            f"Unknown split '{split_name}'. Allowed splits: {list(SPLIT_SPECS.keys())}"
        )

    if split_name == "main-test" and manifest.is_main_test_sealed:
        raise PermissionError(
            f"Main-Test split is sealed in stage '{manifest.stage}'. "
            f"Access strictly prohibited before Stage D unseal (SPEC §3.2, §8)."
        )

    spec = SPLIT_SPECS[split_name]
    seeds = manifest.seed_namespaces[split_name]
    n_seqs = spec["n_sequences"]
    valid_len = spec["valid_length"]
    washout = spec["washout"]

    sequences = []
    for s in seeds[:n_seqs]:
        seq = generate_narma10_raw_stream(
            seed=s,
            valid_length=valid_len,
            washout=washout,
        )
        sequences.append(seq)

    return sequences
