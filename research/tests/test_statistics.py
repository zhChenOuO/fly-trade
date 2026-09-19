"""Synthetic checks only: no market data, simulator, or filesystem fixtures."""

import json

import numpy as np
import pandas as pd
import pytest

from research.pipeline.baselines import (
    compare_constant_input,
    constant_input_profile,
    matched_random,
    matched_random_from_validation,
)
from research.pipeline.statistics import (
    balanced_accuracy,
    between_within_variance_ratio,
    bootstrap_ci,
    buy_ratio,
    consistency,
    delta_margin_effect,
    empirical_p,
    evaluate_levels,
    input_shuffle_permutation_test,
    longest_run,
    mcc,
    mutual_information,
    precision_by_side,
    seed_direction_agreement,
)


def synthetic_decisions(kind, *, start="2024-01-01", rng_seed=17, n=400):
    """Each seed sees the same samples; market_state uses no future labels."""
    rng = np.random.default_rng(rng_seed)
    state = rng.integers(0, 2, 2 * n)
    label = state ^ (rng.random(2 * n) < 0.12)
    samples = pd.DataFrame({
        "sample_id": np.arange(2 * n),
        "timestamp": pd.date_range(start, periods=2 * n, freq="30min", tz="UTC"),
        "image_idx": np.arange(2 * n),
        "future_return": np.where(label, 0.01, -0.01),
        "label": label.astype(int),
        "split": np.repeat(["val", "test"], n),
    })
    frames = []
    for seed in range(5):
        if kind == "signal":
            action = state.copy()
        elif kind == "bias":
            action = np.ones(2 * n, dtype=int)
        elif kind == "buy_bias":
            action = (rng.random(2 * n) < 0.95).astype(int)
        else:
            action = rng.integers(0, 2, 2 * n)
        controls = {
            "fly_intact": action,
            "matched_random": matched_random(action[:n].mean(), 2 * n, seed),
            "input_shuffled": rng.permutation(action),
            "degree_scramble": rng.integers(0, 2, 2 * n),
            "constant_input": np.ones(2 * n, dtype=int),
        }
        for group, actions in controls.items():
            frames.append(pd.DataFrame({
                "sample_id": samples.sample_id,
                "seed": seed,
                "group": group,
                "buy_score": actions.astype(float),
                "sell_score": 1.0 - actions,
                "action": actions,
                "latency_ms": 0.1,
            }))
    # Even deterministic random-looking outputs must fail input dependence.
    repeats = np.repeat(frames[0].action.to_numpy()[:200, None], 5, axis=1)
    return pd.concat(frames, ignore_index=True), samples, repeats, pd.Series(state, index=samples.sample_id)


def evaluate(data, **kwargs):
    decisions, samples, repeats, states = data
    margins = 2.0 * repeats - 1.0
    evidence = dict(
        repeat_actions=repeats, market_state=states, repeat_margins=margins,
        perturbation_margins={"price_flip": -margins.mean(axis=1)},
        train_margin_sd=1.0, nuisance_r2=0.1,
        n_permutations=99, n_bootstrap=149, seed=81,
    )
    evidence.update(kwargs)
    return evaluate_levels(decisions, samples, **evidence)


def test_metrics_have_known_values_and_explicit_degenerate_behavior():
    y, pred = [1, 1, 0, 0], [1, 0, 1, 0]
    assert balanced_accuracy(y, pred) == 0.5
    assert mcc(y, pred) == 0.0
    assert mcc(y, y) == 1.0
    assert mcc(y, 1 - np.array(y)) == -1.0
    assert mcc(y, [1] * 4) == 0.0
    assert balanced_accuracy([1, 1, 1, 0], [1] * 4) == 0.5
    assert precision_by_side(y, pred) == {"BUY": 0.5, "SELL": 0.5}
    assert precision_by_side(y, [1] * 4) == {"BUY": 0.5, "SELL": None}
    assert buy_ratio(pred) == 0.5
    assert longest_run([1, 1, 0, 0, 0, 1]) == 3
    assert longest_run([]) == 0
    assert consistency([[1, 1, 1, 0], [0, 0, 0, 0]]) == 0.875
    assert mutual_information([0, 0, 1, 1], [0, 0, 1, 1]) == pytest.approx(np.log(2))
    assert mutual_information([0, 0, 1, 1], [0, 1, 0, 1]) == 0.0
    assert mutual_information([0, 1, 2, 3], [1] * 4) == 0.0
    assert seed_direction_agreement([1, 2, -1, 0, 1]) == 0.6
    assert seed_direction_agreement([-1, -2, -3]) == 1.0
    assert seed_direction_agreement([0, 0]) == 0.0


@pytest.mark.parametrize("fn,args", [
    (balanced_accuracy, ([0, 1], [1])),
    (balanced_accuracy, ([1, 1], [1, 0])),
    (mcc, ([0, 1], [0, 2])),
    (buy_ratio, ([] ,)),
    (consistency, ([0, 1],)),
    (empirical_p, (1, [])),
    (empirical_p, (np.nan, [0, 1])),
    (seed_direction_agreement, ([np.nan],)),
    (matched_random, (1.1, 10, 0)),
    (matched_random, (0.5, -1, 0)),
])
def test_invalid_inputs_are_rejected(fn, args):
    with pytest.raises(ValueError):
        fn(*args)


def test_empirical_p_counts_ties_and_never_returns_zero():
    assert empirical_p(3, [0, 1, 2]) == 0.25
    assert empirical_p(2, [0, 1, 2]) == 0.5
    assert empirical_p(0, [0, 0, 0]) == 1.0
    assert empirical_p(10, np.zeros(1000)) == 1 / 1001


def test_shuffle_preserves_bias_is_deterministic_and_detects_signal():
    y = np.tile([0, 1], 100)
    a = input_shuffle_permutation_test(balanced_accuracy, y, y, 199, 42)
    b = input_shuffle_permutation_test(balanced_accuracy, y, y, 199, 42)
    np.testing.assert_array_equal(a["null_samples"], b["null_samples"])
    assert a["statistic"] == 1.0
    assert a["p_value"] == 1 / 200
    fixed = input_shuffle_permutation_test(balanced_accuracy, y, np.ones(200), 199, 42)
    assert fixed["p_value"] == 1.0
    assert np.all(fixed["null_samples"] == 0.5)


def test_null_p_values_are_approximately_uniform():
    rng = np.random.default_rng(33)
    ps = [input_shuffle_permutation_test(
        balanced_accuracy, np.tile([0, 1], 100), rng.integers(0, 2, 200), 199, i,
    )["p_value"] for i in range(160)]
    assert 0.4 < np.mean(ps) < 0.62
    assert np.mean(np.array(ps) < 0.05) <= 0.10
    assert 0.35 < np.mean(np.array(ps) <= 0.5) < 0.65


def test_bootstrap_is_paired_reproducible_and_has_reasonable_coverage():
    rng = np.random.default_rng(102)
    y = np.tile([0, 1], 150)
    covered = 0
    for seed in range(80):
        pred = y ^ (rng.random(len(y)) > 0.7)
        low, high = bootstrap_ci(balanced_accuracy, y, pred, 199, seed)
        covered += low <= 0.7 <= high
    assert 0.85 <= covered / 80 <= 1.0
    assert bootstrap_ci(balanced_accuracy, y, y, 99, 9) == (1.0, 1.0)
    predictions = np.column_stack([pred, pred])
    delta = lambda truth, p: balanced_accuracy(truth, p[:, 0]) - balanced_accuracy(truth, p[:, 1])
    assert bootstrap_ci(delta, y, predictions, 99, 9) == (0.0, 0.0)
    assert bootstrap_ci(balanced_accuracy, y, pred, 99, 9) == bootstrap_ci(balanced_accuracy, y, pred, 99, 9)


def test_matched_random_uses_only_fly_intact_validation_and_local_rng():
    decisions, samples, _, _ = synthetic_decisions("signal", n=100)
    val_ids = samples.loc[samples.split == "val", "sample_id"]
    mask = (decisions.group == "fly_intact") & decisions.sample_id.isin(val_ids)
    decisions.loc[mask, "action"] = np.tile(np.r_[np.ones(80), np.zeros(20)], 5)
    before = decisions.copy(deep=True)
    actual = matched_random_from_validation(decisions, samples, 20000, 12)
    np.testing.assert_array_equal(actual, matched_random(0.8, 20000, 12))
    assert actual.mean() == pytest.approx(0.8, abs=0.02)
    decisions.loc[(decisions.group == "fly_intact") & ~decisions.sample_id.isin(val_ids), "action"] = 0
    np.testing.assert_array_equal(actual, matched_random_from_validation(decisions, samples, 20000, 12))
    assert before.loc[mask, "action"].equals(decisions.loc[mask, "action"])
    with pytest.raises(ValueError, match="validation"):
        matched_random_from_validation(decisions, samples[samples.split == "test"], 10, 0)
    assert np.all(matched_random(0, 10, 1) == 0)
    assert np.all(matched_random(1, 10, 1) == 1)


def test_constant_input_comparison_includes_score_distribution():
    assert constant_input_profile([1, 1, 0])["longest_run"] == 2
    same = compare_constant_input([1] * 100, [1] * 100)
    assert not same["different"]
    different = compare_constant_input(np.tile([0, 1], 100), np.ones(200))
    assert different["different"]
    scores = compare_constant_input(
        np.ones(100), np.ones(100), real_buy_score=np.arange(100) + 2,
        real_sell_score=np.zeros(100), constant_buy_score=np.ones(100),
        constant_sell_score=np.zeros(100),
    )
    assert scores["different"]


@pytest.mark.parametrize("kind", ["random", "bias", "buy_bias"])
def test_no_signal_and_fixed_buy_cannot_pass_level_two_or_above(kind):
    result = evaluate(synthetic_decisions(kind), connectome_source="flywire")
    assert not result["level_2"]["passed"]
    assert not result["level_3"]["passed"]
    assert not result["level_4"]["passed"]
    assert not result["learning_allowed"]
    json.dumps(result, allow_nan=False)


def test_planted_signal_passes_all_levels_with_real_topology_and_replication():
    data = synthetic_decisions("signal")
    replication = synthetic_decisions("signal", start="2025-01-01", rng_seed=29)
    before = data[0].copy(deep=True)
    result = evaluate(
        data, connectome_source="flywire",
        replication_decisions=replication[0], replication_samples=replication[1],
    )
    assert all(result[f"level_{i}"]["passed"] for i in range(1, 5))
    assert result["highest_level"] == 4
    assert result["learning_allowed"]
    assert result["level_2"]["criteria"]["input_shuffle"]["p_value"] < 0.05
    assert result["level_3"]["criteria"]["bootstrap_ci"]["value"][0] > 0
    assert result["level_4"]["criteria"]["matched_random"]["effect"] > 0.2
    assert result["metrics"]["fly_intact"]["balanced_accuracy"] > 0.8
    assert result["metrics"]["fly_intact"]["mcc"] > 0.6
    pd.testing.assert_frame_equal(data[0], before)
    json.dumps(result, allow_nan=False)


def test_missing_evidence_and_synthetic_topology_are_explicit_failures():
    data = synthetic_decisions("signal")
    result = evaluate(data)
    assert result["level_2"]["passed"]
    assert not result["level_3"]["criteria"]["real_connectome"]["passed"]
    assert not result["level_4"]["criteria"]["independent_period"]["passed"]
    missing = evaluate_levels(data[0], data[1], n_permutations=19, n_bootstrap=29)
    assert not missing["level_1"]["passed"]
    assert not missing["level_2"]["passed"]
    assert missing["level_2"]["criteria"]["input_shuffle"]["p_value"] is None


def test_level_four_rejects_reused_period_and_opposite_replication():
    data = synthetic_decisions("signal")
    reused = evaluate(data, connectome_source="flywire", replication_decisions=data[0], replication_samples=data[1])
    assert not reused["level_4"]["criteria"]["independent_period"]["passed"]
    replication = synthetic_decisions("signal", start="2025-01-01")
    replication[1]["label"] = 1 - replication[1].label
    reversed_result = evaluate(data, connectome_source="flywire", replication_decisions=replication[0], replication_samples=replication[1])
    assert not reversed_result["level_4"]["passed"]


def test_evaluation_rejects_duplicate_or_unpaired_decisions():
    data = synthetic_decisions("signal")
    with pytest.raises(ValueError, match="duplicate"):
        evaluate((pd.concat([data[0], data[0].iloc[:1]]), *data[1:]))
    incomplete = data[0].drop(data[0].query("group == 'degree_scramble'").index[0])
    with pytest.raises(ValueError, match="paired"):
        evaluate((incomplete, *data[1:]))


def test_evaluation_preserves_row_order_independence_and_seed_dependence():
    data = synthetic_decisions("signal", n=100)
    result = evaluate(data)
    shuffled = (data[0].sample(frac=1, random_state=2), data[1].sample(frac=1, random_state=3), *data[2:])
    assert evaluate(shuffled) == result
    # Repeating identical deterministic seeds must not tighten the sample CI.
    single = data[0].loc[data[0].seed == 0]
    copies = pd.concat([single.assign(seed=i) for i in range(5)], ignore_index=True)
    one = evaluate((single, *data[1:]))
    many = evaluate((copies, *data[1:]))
    assert one["metrics"]["fly_intact"]["ci95"] == many["metrics"]["fly_intact"]["ci95"]


def test_constant_input_single_observation_has_finite_conservative_p_value():
    result = compare_constant_input(
        [1], [1], real_buy_score=[2], real_sell_score=[0],
        constant_buy_score=[1], constant_sell_score=[0],
    )
    assert result["p_value"] == 1.0
    assert not result["different"]
    json.dumps(result, allow_nan=False)


def test_topology_seed_agreement_must_favor_intact_not_merely_agree():
    data = synthetic_decisions("signal")
    decisions, samples = data[:2]
    scramble = decisions.group == "degree_scramble"
    truth = decisions.loc[scramble, "sample_id"].map(samples.set_index("sample_id").label)
    action = np.where(decisions.loc[scramble, "seed"] < 4, truth, 1 - truth)
    decisions.loc[scramble, "action"] = action
    decisions.loc[scramble, "buy_score"] = action
    decisions.loc[scramble, "sell_score"] = 1 - action
    result = evaluate(data, connectome_source="flywire")
    criteria = result["level_3"]["criteria"]
    assert criteria["degree_scramble"]["effect"] > 0
    assert criteria["seed_direction"]["value"] == 0.2
    assert not result["level_3"]["passed"]


def test_overlapping_future_return_windows_are_not_an_independent_period():
    data = synthetic_decisions("signal")
    replication = synthetic_decisions("signal", start="2025-01-01")
    last_test = data[1].loc[data[1].split == "test", "timestamp"].max()
    first_rep_test = replication[1].loc[replication[1].split == "test", "timestamp"].min()
    replication[1]["timestamp"] += last_test + pd.Timedelta(minutes=5) - first_rep_test
    result = evaluate(data, connectome_source="flywire", replication_decisions=replication[0], replication_samples=replication[1])
    assert not result["level_4"]["criteria"]["independent_period"]["passed"]


def test_continuous_margin_statistics_and_degenerate_repeats():
    # Between variance = var([0, 2], ddof=1) = 2; mean within variance = 2.
    assert between_within_variance_ratio([[-1, 1], [1, 3]]) == 1.0
    assert between_within_variance_ratio([[0, 0], [2, 2]]) == np.inf
    assert between_within_variance_ratio([[1, 1], [1, 1]]) == 0.0
    assert delta_margin_effect([1, -1], [0, 1], 2.0) == 0.75
    for margins in ([[1, 2]], [[1], [2]], [[1, np.nan], [2, 3]]):
        with pytest.raises(ValueError):
            between_within_variance_ratio(margins)
    for sd in (0, -1, np.nan, np.inf):
        with pytest.raises(ValueError):
            delta_margin_effect([1, 2], [2, 3], sd)
    with pytest.raises(ValueError):
        delta_margin_effect([1, 2], [[1], [2]], 1)


@pytest.mark.parametrize("split", ["val", "test"])
def test_three_buys_action_collapse_blocks_every_level_and_subcriterion(split):
    data = synthetic_decisions("signal", n=10000)
    decisions, samples = data[:2]
    ids = samples.loc[samples.split == split, "sample_id"]
    # Exactly three BUY rows in the affected split, all remaining rows SELL.
    mask = (decisions.group == "fly_intact") & decisions.sample_id.isin(ids)
    decisions.loc[mask, ["action", "buy_score", "sell_score"]] = [0, 0.0, 1.0]
    decisions.loc[decisions.index[mask][:3], ["action", "buy_score", "sell_score"]] = [1, 1.0, 0.0]
    result = evaluate(data)
    assert result["reason"] == "action_collapse"
    assert result["highest_level"] == 0
    assert not result["global_gate"]["passed"]
    assert not result["learning_allowed"]
    for i in range(1, 5):
        assert not result[f"level_{i}"]["passed"]
        assert result[f"level_{i}"]["reason"] == "action_collapse"
        assert all(not c["passed"] for c in result[f"level_{i}"]["criteria"].values())
    assert not result["level_4"]["criteria"]["matched_random"]["passed"]
    assert not result["level_4"]["criteria"]["permutation"]["passed"]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("score,reason", [(1.0, "constant_margin"), (np.nan, "nonfinite_margin"), (np.inf, "nonfinite_margin")])
def test_invalid_intact_margins_fail_the_global_gate(score, reason):
    data = synthetic_decisions("signal")
    intact = data[0].group == "fly_intact"
    data[0].loc[intact, "buy_score"] = score
    data[0].loc[intact, "sell_score"] = 0.0
    result = evaluate(data)
    assert result["reason"] == reason
    assert result["highest_level"] == 0
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("kwargs,level,criterion", [
    ({"repeat_margins": None}, 1, "between_within_variance"),
    ({"repeat_margins": np.ones((200, 5))}, 1, "between_within_variance"),
    ({"perturbation_margins": None}, 1, "perturbation_effect"),
    ({"perturbation_margins": {}}, 1, "perturbation_effect"),
    ({"train_margin_sd": None}, 1, "perturbation_effect"),
    ({"train_margin_sd": 0.0}, 1, "perturbation_effect"),
    ({"nuisance_r2": None}, 2, "nuisance_robustness"),
    ({"nuisance_r2": np.nan}, 2, "nuisance_robustness"),
    ({"nuisance_r2": 0.5001}, 2, "nuisance_robustness"),
])
def test_new_evidence_is_required_and_never_defaults_to_pass(kwargs, level, criterion):
    result = evaluate(synthetic_decisions("signal"), **kwargs)
    assert not result[f"level_{level}"]["criteria"][criterion]["passed"]
    assert not result[f"level_{level}"]["passed"]
    assert not result["level_4"]["passed"]
    json.dumps(result, allow_nan=False)


def test_registered_thresholds_are_returned_and_overrides_are_applied():
    data = synthetic_decisions("signal")
    result = evaluate(data, nuisance_r2=0.5)
    assert result["thresholds"] == {
        "min_minority_action_ratio": 0.05, "min_margin_range": 0.0,
        "min_consistency": 0.8, "min_between_within_ratio": 3.0,
        "min_delta_margin_effect": 0.2, "max_nuisance_r2": 0.5,
        "min_seed_agreement": 0.8, "min_effect": 0.005,
        "min_ba_ci_lower": 0.5, "min_mcc_ci_lower": 0.0,
        "alpha": 0.05, "confidence": 0.95,
    }
    assert result["level_2"]["criteria"]["nuisance_robustness"]["passed"]
    stricter = evaluate(data, min_delta_margin_effect=2.1, max_nuisance_r2=0.05)
    assert stricter["thresholds"]["min_delta_margin_effect"] == 2.1
    assert not stricter["level_1"]["criteria"]["perturbation_effect"]["passed"]
    assert not stricter["level_2"]["criteria"]["nuisance_robustness"]["passed"]
    with pytest.raises(ValueError):
        evaluate(data, min_effect=np.nan)


def test_high_consistency_without_enough_margin_separation_fails_level_one():
    data = synthetic_decisions("signal")
    margins = (2.0 * data[2] - 1) + np.array([-1.2, -0.6, 0, 0.6, 1.2])
    result = evaluate(data, repeat_margins=margins, repeat_actions=(margins > 0).astype(int))
    criteria = result["level_1"]["criteria"]
    assert criteria["consistency"]["passed"]
    assert criteria["between_within_variance"]["value"] < 3
    assert not criteria["between_within_variance"]["passed"]
    assert not result["level_1"]["passed"]


@pytest.mark.parametrize("score", [np.nan, 1.0])
def test_invalid_replication_margins_cannot_support_level_four(score):
    data = synthetic_decisions("signal")
    replication = synthetic_decisions("signal", start="2025-01-01")
    mask = replication[0].group == "fly_intact"
    replication[0].loc[mask, "buy_score"] = score
    replication[0].loc[mask, "sell_score"] = 0.0
    result = evaluate(data, replication_decisions=replication[0], replication_samples=replication[1])
    assert result["level_2"]["passed"]
    assert not result["level_4"]["criteria"]["independent_period"]["passed"]
    assert not result["level_4"]["passed"]


def test_level_four_ci_and_seed_thresholds_are_enforced_on_signal():
    data = synthetic_decisions("signal")
    replication = synthetic_decisions("signal", start="2025-01-01", rng_seed=29)
    result = evaluate(data, connectome_source="flywire", replication_decisions=replication[0], replication_samples=replication[1])
    assert result["level_4"]["passed"]
    criteria = result["level_4"]["criteria"]
    assert criteria["ba_ci"]["value"][0] > 0.5
    assert criteria["mcc_ci"]["value"][0] > 0.0
    assert criteria["seed_direction"]["value"] >= 0.8
    assert result["metrics"]["fly_intact"]["mcc_ci95"] == criteria["mcc_ci"]["value"]
    strict = evaluate(data, min_ba_ci_lower=criteria["ba_ci"]["value"][0], min_mcc_ci_lower=criteria["mcc_ci"]["value"][0])
    assert not strict["level_4"]["criteria"]["ba_ci"]["passed"]
    assert not strict["level_4"]["criteria"]["mcc_ci"]["passed"]
    # Most seeds favor the control, even though a large outlier lifts the mean.
    mask = data[0].group == "matched_random"
    truth = data[0].loc[mask, "sample_id"].map(data[1].set_index("sample_id").label)
    actions = np.where(data[0].loc[mask, "seed"] < 4, truth, 1 - truth)
    data[0].loc[mask, "action"] = actions
    bad_seeds = evaluate(data)
    assert bad_seeds["level_4"]["criteria"]["matched_random"]["effect"] > 0.005
    assert not bad_seeds["level_4"]["criteria"]["seed_direction"]["passed"]


def test_balanced_accuracy_05004_cannot_pass_level_four_on_large_sample():
    n = 100000
    rng = np.random.default_rng(77)
    # Exact class balance and 50.04% recall for both classes, not action collapse.
    y = np.repeat([0, 1], n // 2)
    pred = y.copy()
    wrong = int((n // 2) * (1 - 0.5004))
    pred[:wrong] = 1
    pred[n // 2:n // 2 + wrong] = 0
    order = rng.permutation(n)
    y, pred = y[order], pred[order]
    samples = pd.DataFrame({
        "sample_id": np.arange(2 * n), "label": np.tile(y, 2),
        "split": np.repeat(["val", "test"], n),
    })
    frames = []
    for seed in range(2):
        # Exactly chance matched control, so tiny effect cannot hide in noise.
        control = pred.copy()
        for label in (0, 1):
            correct = np.flatnonzero((y == label) & (control == label))[:20]
            control[correct] = 1 - label
        for group, action in (("fly_intact", pred), ("matched_random", control), ("constant_input", np.ones(n))):
            actions = np.tile(action, 2)
            frames.append(pd.DataFrame({"sample_id": samples.sample_id, "seed": seed, "group": group,
                                        "action": actions, "buy_score": actions.astype(float),
                                        "sell_score": 1.0 - actions, "latency_ms": 0.0}))
    repeats = np.repeat(pred[:200, None], 5, axis=1)
    states = pd.Series(np.tile(pred, 2), index=samples.sample_id)
    result = evaluate((pd.concat(frames, ignore_index=True), samples, repeats, states), n_permutations=39, n_bootstrap=59)
    assert result["global_gate"]["passed"]
    assert result["level_2"]["passed"]
    assert result["metrics"]["fly_intact"]["balanced_accuracy"] == pytest.approx(0.5004)
    assert result["level_4"]["criteria"]["permutation"]["p_value"] < 0.05
    effect = result["level_4"]["criteria"]["matched_random"]
    assert effect["effect"] == pytest.approx(0.0004)
    assert not effect["passed"]
    assert not result["level_4"]["passed"]
