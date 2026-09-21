"""Unit tests for A23: G1 v2 Manifest, Seed Namespaces, and Test Sealing (SPEC v2 §3.2, §8).

Verifications:
1. Seed namespaces are strictly disjoint with zero overlap.
2. Manifest creation captures exact SHA-256 hashes for spec, source snapshot, simulator, readout, and scorer.
3. Test split sealing: Stage A, B, C reject Main-Test creation and reject unseal attempts.
4. Stage D unseal requires exact manifest SHA-256 match; wrong hash fails closed.
5. Single-split sequence API: attempting to generate all splits at once raises ValueError.
6. Unknown split names and tampered manifest hashes fail closed.
"""
from __future__ import annotations

from pathlib import Path
import pytest

from research.pipeline.g1_manifest import (
    SEED_NAMESPACES,
    SPLIT_SPECS,
    G1Manifest,
    create_g1_manifest,
    verify_manifest_integrity,
    generate_split_sequences,
    verify_seed_namespaces_disjoint,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_a23_seed_namespaces_strictly_disjoint():
    """Verify that all seed namespaces are pairwise disjoint with zero overlap."""
    verify_seed_namespaces_disjoint()

    # Explicit pair-wise intersection check
    namespaces = list(SEED_NAMESPACES.keys())
    for i in range(len(namespaces)):
        for j in range(i + 1, len(namespaces)):
            set_i = set(SEED_NAMESPACES[namespaces[i]])
            set_j = set(SEED_NAMESPACES[namespaces[j]])
            overlap = set_i.intersection(set_j)
            assert len(overlap) == 0, f"Overlap between '{namespaces[i]}' and '{namespaces[j]}': {overlap}"


def test_a23_manifest_creation_and_integrity():
    """Verify manifest creation, hashing, and verification against disk."""
    manifest = create_g1_manifest(REPO_ROOT)

    assert manifest.version == "2.0.0"
    assert manifest.stage == "A"
    assert manifest.is_main_test_sealed is True
    assert len(manifest.spec_sha256) == 64
    assert len(manifest.source_snapshot_sha256) == 64
    assert len(manifest.simulator_hash) == 64
    assert len(manifest.readout_hash) == 64
    assert len(manifest.scorer_hash) == 64
    assert len(manifest.manifest_sha256) == 64

    # Verify integrity against repo
    res = verify_manifest_integrity(manifest, REPO_ROOT)
    assert res["verified"] is True
    assert res["is_main_test_sealed"] is True


def test_a23_main_test_sealing_stages_a_to_c():
    """Verify that Main-Test cannot be generated or unsealed during stages A, B, or C."""
    manifest = create_g1_manifest(REPO_ROOT)

    # In stage A
    with pytest.raises(PermissionError, match="Main-Test split is sealed in stage 'A'"):
        generate_split_sequences("main-test", manifest)

    with pytest.raises(PermissionError, match="Main-Test split cannot be unsealed in stage 'A'"):
        manifest.unseal_main_test("any_hash")

    # Advance to B
    manifest.advance_stage("B")
    with pytest.raises(PermissionError, match="Main-Test split is sealed in stage 'B'"):
        generate_split_sequences("main-test", manifest)

    with pytest.raises(PermissionError, match="Main-Test split cannot be unsealed in stage 'B'"):
        manifest.unseal_main_test("any_hash")

    # Advance to C
    manifest.advance_stage("C")
    with pytest.raises(PermissionError, match="Main-Test split is sealed in stage 'C'"):
        generate_split_sequences("main-test", manifest)

    with pytest.raises(PermissionError, match="Main-Test split cannot be unsealed in stage 'C'"):
        manifest.unseal_main_test("any_hash")


def test_a23_stage_d_unseal_and_generation():
    """Verify that Stage D unseal requires exact manifest hash match before generating Main-Test."""
    manifest = create_g1_manifest(REPO_ROOT)
    manifest.advance_stage("B")
    manifest.advance_stage("C")
    manifest.advance_stage("D")

    # Wrong hash fails closed
    with pytest.raises(ValueError, match="Manifest SHA-256 mismatch"):
        manifest.unseal_main_test("wrong_tampered_hash_value")

    assert manifest.is_main_test_sealed is True

    # Correct hash succeeds
    correct_sha = manifest.compute_sha256()
    manifest.unseal_main_test(correct_sha)
    assert manifest.is_main_test_sealed is False

    # Now generation succeeds strictly on single-split basis
    test_seqs = generate_split_sequences("main-test", manifest)
    assert len(test_seqs) == SPLIT_SPECS["main-test"]["n_sequences"]
    assert test_seqs[0]["valid_length"] == 5000
    assert test_seqs[0]["washout"] == 500


def test_a23_split_generation_prohibitions_and_fail_closed():
    """Verify that attempting to generate all splits or unknown splits raises ValueError."""
    manifest = create_g1_manifest(REPO_ROOT)

    # Prohibit generating all splits at once
    with pytest.raises(ValueError, match="Generating all splits at once is strictly prohibited"):
        generate_split_sequences("all", manifest)

    with pytest.raises(ValueError, match="Generating all splits at once is strictly prohibited"):
        generate_split_sequences("*", manifest)

    # Unknown split raises
    with pytest.raises(ValueError, match="Unknown split 'unknown_split'"):
        generate_split_sequences("unknown_split", manifest)

    # Tampered manifest hash fails closed
    manifest.spec_sha256 = "0" * 64
    with pytest.raises(ValueError, match="spec_sha256 mismatch"):
        verify_manifest_integrity(manifest, REPO_ROOT)


def test_formal_seed_namespaces_do_not_reuse_legacy_or_fixture_streams():
    from research.pipeline.g1_manifest import LEGACY_AND_FIXTURE_SEEDS, SEED_NAMESPACES

    for name, seeds in SEED_NAMESPACES.items():
        if name == "fixture":
            continue
        assert not (set(seeds) & LEGACY_AND_FIXTURE_SEEDS), name
