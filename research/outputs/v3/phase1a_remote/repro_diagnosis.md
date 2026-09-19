# Phase 1a reproducibility diagnosis

## Finding

The original seed reproducibility gate failed because it required bitwise-identical GPU scores. Repeated runs on the RTX 5070 produce tiny score differences even with noise disabled. Actions remained identical in every measured comparison. The failure is consistent with GPU numerical nondeterminism (H2), not a reused noise generator (H1) or a material batch-composition effect (H3). No simulator or encoder logic bug was found, and neither was changed.

The original run log recorded only `fixed-seed A decisions are not reproducible for seed 0`; it did not persist the compared score arrays. The measurements below reproduce that same seed-0 check and then exercise all 30 configured seeds.

## Original gate code

This was the check before diagnosis instrumentation was added:

```python
for seed in range(int(CFG["seeds"])):
    first = _simulate_chunks(
        _new_simulator(graph, retina, baseline, dynamics, seed, dynamics["noise_std_robustness"]),
        deterministic_images,
        category="seed_reproducibility",
        tracker=tracker,
        description=f"seed {seed} reproducibility first pass",
    )
    second = _simulate_chunks(
        _new_simulator(graph, retina, baseline, dynamics, seed, dynamics["noise_std_robustness"]),
        deterministic_images,
        category="seed_reproducibility",
        tracker=tracker,
        description=f"seed {seed} reproducibility repeat",
    )
    actions_match = np.array_equal(decode(first[0], first[1]), decode(second[0], second[1]))
    scores_match = np.array_equal(first[0], second[0]) and np.array_equal(first[1], second[1])
    deterministic_results.append({"seed": seed, "actions_match": bool(actions_match), "scores_match": bool(scores_match)})
    if not actions_match or not scores_match:
        raise RuntimeError(f"fixed-seed A decisions are not reproducible for seed {seed}")
```

The production gate now calls `reproducibility_summary()` for measured absolute/relative score differences and margin-aware action flips. Its absolute tolerance is `min(4.76837158203125e-7, 1e-5 * pooled reference score SD)`. Each BUY and SELL score must meet that limit. Any action flip is accepted only when the reference sample has `abs(margin) < score_tolerance`; all such flips are counted and their margins are stored. No Phase 1a statistic, experiment threshold, or other check was changed.

## H1 — reused noise RNG

**Result: ruled out as the cause of this failure.**

- The check creates a fresh `FlySimulator` via `_new_simulator(...)` for both passes, with the same seed and noise level.
- Main group A creates one fresh simulator inside each seed iteration. Group E does the same; each sensitivity repeat/variant also starts with a fresh simulator for that repeat seed.
- Two independently initialized NumPy generators with seed 0 produced bitwise-identical noise arrays for the same `(32, 32, n_sensory)` shape.
- As a statefulness control, one simulator run twice without resetting its RNG changed scores by up to `0.0324093`; calling `reset_rng()` restored the same noise sequence, with the remaining GPU score delta at most `1.49012e-7` in that pair. Thus `FlySimulator._rng` is stateful, but the failing check and seeded main paths already reconstruct it per seed.

Seed-0 reproduction using two fresh simulator instances, 32 images, and `noise_std=0.012225364148616791`:

| Measurement | Result |
| --- | ---: |
| Scores bitwise equal | No |
| Actions equal | Yes |
| Maximum absolute score difference | `2.384185791015625e-7` |
| Maximum relative difference¹ | `4.495711458265668e-6` |
| BUY maximum absolute difference | `1.7881393432617188e-7` |
| SELL maximum absolute difference | `2.384185791015625e-7` |
| Action flips | `0` |
| Flipped-sample margins | None; there were no flips |

¹ Relative difference is `abs(a-b) / max(abs(a), abs(b))`; it is sensitive to scores near zero.

## H2 — GPU nondeterminism

**Result: confirmed.** With noise disabled, the same 32-image batch was run 10 times in each mode. “Exact comparisons” counts bitwise-equal pairs among the 9 comparisons to the first run.

| Mode | Exact comparisons | Maximum absolute difference | Median steady seconds/call | Relative speed |
| --- | ---: | ---: | ---: | ---: |
| Default, no CUBLAS setting | `0/9` | `2.384185791015625e-7` | `0.111073` | `1.000x` |
| `torch.use_deterministic_algorithms(True)` | `0/9` | `2.384185791015625e-7` | `0.112298` | `1.011x` |
| Deterministic algorithms plus `CUBLAS_WORKSPACE_CONFIG=:4096:8` set before Python starts | `0/9` | `2.384185791015625e-7` | `0.108505` | `0.993x` within that process |

The CUBLAS-configured process measured `0.109264` seconds/call in its default mode, so enabling deterministic algorithms measured `0.993x` there. Timing variation was about 1%; neither setting improved bitwise reproducibility. The requested deterministic options were tested and did not resolve this GPU operation, so the runner does not claim bitwise reproducibility or alter the frozen simulation path.

## H3 — batch composition

**Result: no material batch-composition effect detected; `SIM_CHUNK` remains 256.** With noise disabled, four repetitions compared the first 32 images alone against those same images at the start of a 1000-image input processed in chunks of 256 (`256, 256, 256, 232`).

- Same-shape repeat maximum absolute difference: `2.980232238769531e-7` for the 32-image runs and `2.980232238769531e-7` for the first 32 outputs of 1000-image runs.
- Across 16 comparisons between the two batch shapes, maximum absolute difference was `2.980232238769531e-7` and minimum was `1.7881393432617188e-7`.
- No comparison changed an action.

Cross-shape differences stayed within the observed same-shape GPU repeat variation. The measurements do not support changing chunking or padding batches.

## Tolerance calibration and repair

A direct survey repeated the exact 32-image check with two fresh simulators for each of seeds `0..29`, using the configured robustness noise. Across 30 seed pairs:

- Scores were not bitwise equal in any pair; actions were equal in all 30 pairs.
- Maximum absolute score difference was `4.76837158203125e-7`.
- Maximum relative difference was `1.3039437932573902e-4`, caused by a near-zero score denominator.
- Total action flips: `0`; no flipped-sample margins exist. The smallest reference `abs(margin)` across the 30 comparisons was `0.0505705`.
- In the seed-0 reproduction, the pooled reference score SD was `0.855042`; the measured maximum absolute difference was `2.79e-7` of that SD. The selected empirical cap `4.76837158203125e-7` is below `1e-5` of that score SD (`8.55e-6`).

The runner uses the largest absolute delta observed in this 30-seed survey as a strict empirical ceiling, further capped at `1e-5` of each comparison's pooled reference score SD. It still fails if any individual score exceeds the limit. Action changes are separately constrained to samples whose reference margin is strictly inside that same limit, and the health output records score/action exactness, differences, tolerance, and flip counts. Unit tests cover score drift, allowed near-zero flips, disallowed outside-margin flips, and score differences above the measured ceiling.

Only the reproducibility comparison was repaired. `research/config/experiment.yaml`, Phase 1a thresholds, sample counts, group definitions, chunk size, and all other checks remain unchanged.

## Full Phase 1a run after the repair

The full run completed on Python `3.14.6`, PyTorch `2.14.0+cu130`, CUDA `13.0`, and the NVIDIA GeForce RTX 5070 (11.50 GiB reported by PyTorch):

- Final 30-seed reproducibility check: 0/30 score pairs bitwise equal, 30/30 action pairs equal, maximum absolute score difference `4.76837158203125e-7`, maximum relative difference `7.826928308056224e-5`, maximum absolute difference / score SD `5.589594765582567e-7`, and 0 action flips. The largest per-seed tolerance was `4.76837158203125e-7`.
- Full render alignment: all `52,564` images passed exact equality.
- Train baseline and 20-image Val preflight passed; preflight minority-action ratio was `0.10`.
- Full A/B/C/E evaluation completed for 30 seeds and `21,008` selected Val+development-test samples per seed. The ignored `decisions.parquet` has `2,520,960` rows.
- The frozen global gate, Level 1, and Level 2 passed. Level 3 is reported `passed: false` with the reason that degree-scramble D is reserved for Phase 1b. Level 4 is reported `passed: false` with the reason that `development_test_v1` is not sealed holdout evidence. Those levels contain only pass/fail and reason; no additional Level 3/4 evidence is claimed.
- Total run time was `10,395.8` seconds. The final run outputs are in this directory; the decisions parquet remains ignored and is not part of delivery.
