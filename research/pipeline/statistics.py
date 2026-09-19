"""Pure statistics for research.md sections 5--7; no fitting or file access.

Actions are 0=SELL, 1=BUY. Larger statistics mean stronger evidence. MI is
in nats. Zero effects never count toward seed direction agreement.
"""

import numpy as np
import pandas as pd


def _binary(values, name="actions", *, allow_empty=False):
    values = np.asarray(values)
    if values.ndim != 1 or (not allow_empty and not values.size):
        raise ValueError(f"{name} must be a nonempty one-dimensional array")
    if not np.isin(values, [0, 1]).all():
        raise ValueError(f"{name} must contain only 0 and 1")
    return values.astype(np.int8)


def _paired(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    if y.ndim != 1 or pred.ndim < 1 or not y.size or len(y) != len(pred):
        raise ValueError("y and pred must have the same nonempty sample axis")
    if pd.isna(y).any() or pd.isna(pred).any():
        raise ValueError("y and pred must not contain missing values")
    return y, pred


def _count(n):
    if not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("n must be a positive integer")


def _binary_pair(y, pred):
    y, pred = _paired(_binary(y, "y"), _binary(pred, "pred"))
    return y, pred


def consistency(actions_matrix):
    """Mean modal-action fraction; rows are inputs, columns are repeats."""
    actions = np.asarray(actions_matrix)
    if actions.ndim != 2 or 0 in actions.shape:
        raise ValueError("actions_matrix must have nonempty input and repeat axes")
    _binary(actions.ravel())
    p = actions.mean(axis=1)
    return float(np.maximum(p, 1 - p).mean())


def between_within_variance_ratio(margins):
    """Var(input means, ddof=1) / mean(within-input variance, ddof=1).

    Rows are inputs and columns independent repeats (at least two of each).
    Deterministic differing inputs give infinity; entirely constant data give
    zero. Scaling first avoids overflow without changing the variance ratio.
    """
    margins = np.asarray(margins, dtype=float)
    if margins.ndim != 2 or min(margins.shape) < 2 or not np.isfinite(margins).all():
        raise ValueError("margins must be finite with at least two inputs and repeats")
    scale = np.max(np.abs(margins))
    if scale == 0:
        return 0.0
    margins = margins / scale
    between = np.var(margins.mean(axis=1), ddof=1)
    within = np.var(margins, axis=1, ddof=1).mean()
    if within == 0:
        return float("inf") if between > 0 else 0.0
    return float(between / within)


def delta_margin_effect(base_margin, perturbed_margin, train_sd):
    """Mean absolute paired margin change, standardized ONLY by train SD."""
    base, perturbed = np.asarray(base_margin, dtype=float), np.asarray(perturbed_margin, dtype=float)
    if base.shape != perturbed.shape or base.ndim < 1 or not base.size:
        raise ValueError("base and perturbed margins must have equal nonempty shapes")
    if not np.isfinite(base).all() or not np.isfinite(perturbed).all():
        raise ValueError("margins must be finite")
    if not np.isscalar(train_sd) or not np.isfinite(train_sd) or train_sd <= 0:
        raise ValueError("train_sd must be finite and strictly positive")
    return float(np.mean(np.abs(perturbed - base)) / train_sd)


def mutual_information(x_discrete, action):
    """Plug-in mutual information for discrete states and binary actions (nats)."""
    x, action = _paired(x_discrete, _binary(action))
    _, codes = np.unique(x, return_inverse=True)
    counts = np.bincount(codes * 2 + action, minlength=2 * (codes.max() + 1)).reshape(-1, 2)
    joint = counts / len(x)
    independent = joint.sum(axis=1, keepdims=True) * joint.sum(axis=0, keepdims=True)
    nonzero = joint > 0
    return float(np.maximum(0, np.sum(joint[nonzero] * np.log(joint[nonzero] / independent[nonzero]))))


def balanced_accuracy(y, pred):
    """Mean recall of DOWN and UP; undefined one-class labels are rejected."""
    y, pred = _binary_pair(y, pred)
    if np.unique(y).size != 2:
        raise ValueError("balanced accuracy requires both label classes")
    return float(((pred[y == 0] == 0).mean() + (pred[y == 1] == 1).mean()) / 2)


def mcc(y, pred):
    """Matthews correlation coefficient; a zero denominator returns 0."""
    y, pred = _binary_pair(y, pred)
    tp = float(np.sum((y == 1) & (pred == 1)))
    tn = float(np.sum((y == 0) & (pred == 0)))
    fp = float(np.sum((y == 0) & (pred == 1)))
    fn = float(np.sum((y == 1) & (pred == 0)))
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float((tp * tn - fp * fn) / denominator) if denominator else 0.0


def precision_by_side(y, pred):
    """P(UP|BUY), P(DOWN|SELL); an unselected side has precision None."""
    y, pred = _binary_pair(y, pred)
    return {name: float((y[pred == side] == side).mean()) if np.any(pred == side) else None
            for name, side in (("BUY", 1), ("SELL", 0))}


def longest_run(actions):
    actions = _binary(actions, allow_empty=True)
    if not actions.size:
        return 0
    boundaries = np.r_[0, np.flatnonzero(actions[1:] != actions[:-1]) + 1, len(actions)]
    return int(np.diff(boundaries).max())


def buy_ratio(actions):
    return float(_binary(actions).mean())


def bootstrap_ci(stat_fn, y, pred, n=2000, seed=0, *, confidence=0.95):
    """Percentile CI (default 95%), paired and stratified by discrete label.

    Resample the first (sample) axis together, preserving class counts and any
    seed/model axes in pred. This conditions on observed label prevalence.
    Assumes independent/low-overlap samples, not arbitrary overlapping bars.
    """
    _count(n)
    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be strictly between zero and one")
    y, pred = _paired(y, pred)
    rng = np.random.default_rng(seed)
    strata = [np.flatnonzero(y == label) for label in np.unique(y)]
    estimates = np.empty(n)
    for i in range(n):
        indices = np.concatenate([rng.choice(group, size=len(group), replace=True) for group in strata])
        estimates[i] = stat_fn(y[indices], pred[indices])
    if not np.isfinite(estimates).all():
        raise ValueError("bootstrap statistic must be finite")
    tail = (1 - confidence) / 2
    return tuple(float(value) for value in np.quantile(estimates, [tail, 1 - tail]))


def empirical_p(real, null_samples):
    null = np.asarray(null_samples, dtype=float)
    if null.ndim != 1 or not null.size or not np.isfinite(null).all() or not np.isfinite(real):
        raise ValueError("real and nonempty null_samples must be finite")
    return float((1 + np.count_nonzero(null >= real)) / (1 + len(null)))


def input_shuffle_permutation_test(stat_fn, y, pred, n=1000, seed=0):
    """Shuffle prediction rows against states/labels, preserving BUY bias.

    Valid for fixed, stateless per-input outputs under sample exchangeability;
    a stateful simulator needs freshly rerun shuffled-input decisions instead.
    """
    _count(n)
    y, pred = _paired(y, pred)
    rng = np.random.default_rng(seed)
    real = float(stat_fn(y, pred))
    null = np.array([stat_fn(y, pred[rng.permutation(len(y))]) for _ in range(n)], dtype=float)
    return {"statistic": real, "p_value": empirical_p(real, null), "null_samples": null}


def seed_direction_agreement(effects):
    """Fraction in the majority strict direction; zeros are nonagreements."""
    effects = np.asarray(effects, dtype=float)
    if effects.ndim != 1 or not effects.size or not np.isfinite(effects).all():
        raise ValueError("effects must be a nonempty finite vector")
    return float(max(np.mean(effects > 0), np.mean(effects < 0)))


def _joined(decisions, samples):
    required = {"sample_id", "seed", "group", "buy_score", "sell_score", "action", "latency_ms"}
    if not required.issubset(decisions.columns) or not {"sample_id", "label", "split"}.issubset(samples.columns):
        raise ValueError("decisions/samples are missing SPEC columns")
    if samples.sample_id.isna().any() or samples.sample_id.duplicated().any():
        raise ValueError("duplicate or missing sample_id in samples")
    if decisions[["sample_id", "seed", "group"]].isna().any().any():
        raise ValueError("decision keys must not be missing")
    if decisions.duplicated(["sample_id", "seed", "group"]).any():
        raise ValueError("duplicate decisions for sample_id, seed, group")
    _binary(samples.label, "label")
    _binary(decisions.action)
    if not samples.split.isin(["train", "val", "test"]).all():
        raise ValueError("unknown sample split")
    if not decisions.group.isin(["fly_intact", "matched_random", "input_shuffled", "degree_scramble", "constant_input"]).all():
        raise ValueError("unknown decision group")
    if not np.isfinite(decisions.latency_ms.to_numpy(dtype=float)).all():
        raise ValueError("latency must be finite")
    # Intact score failures are reported by evaluate_levels' global gate.
    controls = decisions.loc[decisions.group != "fly_intact", ["buy_score", "sell_score"]]
    if not np.isfinite(controls.to_numpy(dtype=float)).all():
        raise ValueError("control scores must be finite")
    if (decisions.latency_ms < 0).any():
        raise ValueError("latency must be nonnegative")
    # Join labels by ID, never by incidental dataframe row order.
    columns = ["sample_id", "label", "split"]
    result = decisions[list(sorted(required))].merge(samples[columns], on="sample_id", how="left", validate="many_to_one")
    if result.label.isna().any():
        raise ValueError("decision sample_id is missing from samples")
    return result


def _panel(joined, group, ids, seeds, column="action"):
    rows = joined.loc[joined.group == group]
    if rows.empty:
        return None
    panel = rows.pivot(index="sample_id", columns="seed", values=column).reindex(index=ids, columns=seeds)
    if panel.isna().any().any() or set(rows.seed.unique()) != set(seeds):
        raise ValueError(f"{group} must be paired on all sample IDs and seeds")
    return panel.to_numpy()


def _ba_columns(y, predictions):
    if np.unique(y).size != 2:
        raise ValueError("balanced accuracy requires both label classes")
    return ((predictions[y == 0] == 0).mean(axis=0) + (predictions[y == 1] == 1).mean(axis=0)) / 2


def _mean_ba(y, predictions):
    return float(_ba_columns(y, predictions).mean())


def _mean_mcc(y, predictions):
    return float(np.mean([mcc(y, predictions[:, i]) for i in range(predictions.shape[1])]))


def _delta_ba(y, predictions):
    return _mean_ba(y, predictions[..., 0]) - _mean_ba(y, predictions[..., 1])


def _comparison(y, real, control, n_bootstrap, n_permutations, seed, confidence=0.95):
    effects = _ba_columns(y, real) - _ba_columns(y, control)
    paired = np.stack([real, control], axis=-1)
    ci = bootstrap_ci(_delta_ba, y, paired, n_bootstrap, seed, confidence=confidence)
    # Paired model-assignment randomization: one swap per sample, shared by
    # all seeds. Repeated seeds are not independent market observations.
    contribution = ((real == y[:, None]).mean(axis=1) - (control == y[:, None]).mean(axis=1))
    contribution /= 2 * np.where(y == 1, np.sum(y == 1), np.sum(y == 0))
    rng = np.random.default_rng(seed)
    null = np.array([np.dot(contribution, rng.choice([-1, 1], len(y))) for _ in range(n_permutations)])
    return {"effect": float(effects.mean()), "ci95": list(ci),
            "p_value": empirical_p(float(contribution.sum()), null),
            "effects_by_seed": effects.tolist(),
            "direction_agreement": seed_direction_agreement(effects),
            "positive_seed_fraction": float(np.mean(effects > 0))}


def _criterion(passed, **values):
    return {"passed": bool(passed), **values}


def evaluate_levels(
    decisions, samples, *, repeat_actions=None, market_state=None,
    repeat_margins=None, perturbation_margins=None, train_margin_sd=None,
    nuisance_r2=None,
    connectome_source="synthetic", replication_decisions=None,
    replication_samples=None, n_permutations=1000, n_bootstrap=2000, seed=0,
    min_minority_action_ratio=0.05, min_margin_range=0.0,
    min_consistency=0.8, min_between_within_ratio=3.0,
    min_delta_margin_effect=0.2, max_nuisance_r2=0.5,
    min_seed_agreement=0.8, min_effect=0.005,
    min_ba_ci_lower=0.5, min_mcc_ci_lower=0.0,
    alpha=0.05, confidence=0.95,
):
    """Evaluate preregistered thresholds with explicit missing-evidence failures.

    `market_state` is a discrete Series indexed by sample_id, computed ONLY
    from past/current inputs with bins fixed before test. Level 2 uses val;
    future labels are never a fallback. Repeat rows are inputs, columns runs.
    Level 3/4 use test and the locked primary metric, balanced accuracy.

    `repeat_margins` has shape (inputs, repeats). Each named controlled
    perturbation has the same input order and either one mean margin per input
    or that same matrix shape; repeated margins are averaged per input before
    computing delta. `train_margin_sd` must come from train, never val/test.
    `nuisance_r2` is the supplied linear nuisance-model R-squared. Missing or
    invalid new evidence fails its criterion; no values are inferred for it.
    All numeric decision thresholds are explicit below and echoed in result.
    Legacy `ci95` keys use the requested `confidence` (default 0.95).

    Matched-random nulls are calibrated exclusively on Fly-Intact validation.
    Supplied matched_random decisions must have been generated with that same
    probability (use baselines.matched_random_from_validation). Comparisons
    use supplied paired controls and also check the calibrated random null.

    CIs resample samples together across seeds; per-seed effects are reported
    separately. Level 2 requires Level 1; Levels 3 and 4 require Level 2.
    Level 4 does not require a topology advantage, which is a distinct claim.
    `connectome_source` is caller-provided provenance, not verified here.
    Replication periods include the locked 6 x 5-minute forward-label horizon.
    Defaults match experiment.yaml; smaller resampling counts are for tests.
    """
    from .baselines import compare_constant_input, matched_random, validation_buy_ratio

    _count(n_permutations)
    _count(n_bootstrap)
    thresholds = dict(
        min_minority_action_ratio=min_minority_action_ratio, min_margin_range=min_margin_range,
        min_consistency=min_consistency, min_between_within_ratio=min_between_within_ratio,
        min_delta_margin_effect=min_delta_margin_effect, max_nuisance_r2=max_nuisance_r2,
        min_seed_agreement=min_seed_agreement, min_effect=min_effect,
        min_ba_ci_lower=min_ba_ci_lower, min_mcc_ci_lower=min_mcc_ci_lower,
        alpha=alpha, confidence=confidence,
    )
    if not all(np.isscalar(value) and np.isfinite(value) for value in thresholds.values()):
        raise ValueError("thresholds must be finite scalars")
    if (not 0 <= min_minority_action_ratio <= 0.5 or min_margin_range < 0
            or min_between_within_ratio < 0 or min_delta_margin_effect < 0
            or not 0 <= min_effect <= 1 or not -1 <= min_mcc_ci_lower <= 1
            or not 0 < alpha < 1 or not 0 < confidence < 1
            or not all(0 <= value <= 1 for value in (min_consistency, max_nuisance_r2, min_seed_agreement, min_ba_ci_lower))):
        raise ValueError("thresholds are outside their valid ranges")
    if connectome_source not in {"synthetic", "flywire", "hemibrain"}:
        raise ValueError("unknown connectome_source")
    joined = _joined(decisions, samples)
    seeds = sorted(joined.loc[joined.group == "fly_intact", "seed"].unique().tolist())
    if not seeds:
        raise ValueError("Fly-Intact decisions are required")
    p_buy = validation_buy_ratio(decisions, samples)
    panels, labels, split_samples = {}, {}, {}
    for split in ("val", "test"):
        selected = samples.loc[samples.split == split].sort_values("sample_id")
        if selected.empty:
            raise ValueError(f"{split} samples are required")
        split_samples[split] = selected
        labels[split] = selected.label.to_numpy()
        subset = joined.loc[joined.split == split]
        panels[split] = {group: _panel(subset, group, selected.sample_id, seeds)
                         for group in ("fly_intact", "matched_random", "input_shuffled", "degree_scramble", "constant_input")}
        if panels[split]["fly_intact"] is None:
            raise ValueError(f"Fly-Intact {split} decisions are required")

    gate_criteria = {}
    for split in ("val", "test"):
        actions = panels[split]["fly_intact"]
        minority = float(min(np.count_nonzero(actions == 1), np.count_nonzero(actions == 0)) / actions.size)
        intact = joined.loc[(joined.group == "fly_intact") & (joined.split == split)]
        with np.errstate(over="ignore", invalid="ignore"):
            margin = intact.buy_score.to_numpy(dtype=float) - intact.sell_score.to_numpy(dtype=float)
            finite = bool(np.isfinite(margin).all())
            span = float(np.ptp(margin)) if finite else None
        gate_criteria[f"{split}_minority_action_ratio"] = _criterion(
            minority >= min_minority_action_ratio, value=minority, threshold=min_minority_action_ratio)
        gate_criteria[f"{split}_margin"] = _criterion(
            finite and span > min_margin_range, finite=finite,
            value=span if span is not None and np.isfinite(span) else None, threshold=min_margin_range)
    global_gate = {"passed": all(c["passed"] for c in gate_criteria.values()), "criteria": gate_criteria}
    result = dict(thresholds=thresholds, global_gate=global_gate, metrics={},
                  seeds=seeds, validation_buy_ratio=p_buy, primary_metric="balanced_accuracy",
                  sample_counts={key: len(value) for key, value in labels.items()})
    if not global_gate["passed"]:
        if any(not gate_criteria[f"{split}_minority_action_ratio"]["passed"] for split in ("val", "test")):
            reason = "action_collapse"
        elif any(not gate_criteria[f"{split}_margin"]["finite"] for split in ("val", "test")):
            reason = "nonfinite_margin"
        else:
            reason = "constant_margin"
        global_gate["reason"] = reason
        # Skip inference entirely; even individual criteria cannot appear to
        # endorse an experiment rejected by the global prerequisite.
        names = (
            ("global_gate", "consistency", "input_variation", "between_within_variance", "perturbation_effect", "constant_input"),
            ("level_1", "input_shuffle", "seed_direction", "beyond_fixed_bias", "nuisance_robustness"),
            ("level_2", "real_connectome", "degree_scramble", "bootstrap_ci", "seed_direction"),
            ("level_2", "matched_random", "permutation", "ba_ci", "mcc_ci", "seed_direction", "independent_period"),
        )
        result.update({f"level_{i}": {"passed": False, "reason": reason,
                                     "criteria": {name: _criterion(False, value=None, reason=reason) for name in criteria}}
                       for i, criteria in enumerate(names, start=1)})
        result.update(highest_level=0, learning_allowed=False, reason=reason)
        return result

    val = panels["val"]
    val_rows = joined.loc[joined.split == "val"]
    margins = {}
    for group in ("fly_intact", "constant_input"):
        if val[group] is not None:
            margins[group] = (_panel(val_rows, group, split_samples["val"].sample_id, seeds, "buy_score")
                              - _panel(val_rows, group, split_samples["val"].sample_id, seeds, "sell_score"))
    consistency_value = consistency(repeat_actions) if repeat_actions is not None else None
    score_range = float(np.ptp(margins["fly_intact"], axis=0).max())
    action_varies = bool(np.any(np.ptp(val["fly_intact"], axis=0) > 0))
    variance_ratio, base_margin = None, None
    if repeat_margins is not None:
        try:
            variance_ratio = between_within_variance_ratio(repeat_margins)
            base_margin = np.asarray(repeat_margins, dtype=float).mean(axis=1)
        except ValueError:
            pass  # Invalid supplied evidence fails below, just as missing evidence does.
    perturbations = {}
    if perturbation_margins is not None:
        if not isinstance(perturbation_margins, dict):
            raise ValueError("perturbation_margins must map names to aligned margin arrays")
        for name, values in perturbation_margins.items():
            effect = None
            if base_margin is not None and train_margin_sd is not None:
                try:
                    perturbed = np.asarray(values, dtype=float)
                    if perturbed.shape == np.shape(repeat_margins):
                        perturbed = perturbed.mean(axis=1)
                    effect = delta_margin_effect(base_margin, perturbed, train_margin_sd)
                    if not np.isfinite(effect):
                        effect = None
                except ValueError:
                    pass
            perturbations[str(name)] = _criterion(effect is not None and effect >= min_delta_margin_effect, value=effect)
    nuisance = float(nuisance_r2) if nuisance_r2 is not None else None
    if nuisance is not None and not np.isfinite(nuisance):
        nuisance = None
    constant = None
    if "constant_input" in margins:
        # One observation per sample; average scores across simulator seeds.
        constant = compare_constant_input(
            val["fly_intact"][:, 0], val["constant_input"][:, 0],
            real_buy_score=margins["fly_intact"].mean(axis=1), real_sell_score=np.zeros(len(labels["val"])),
            constant_buy_score=margins["constant_input"].mean(axis=1), constant_sell_score=np.zeros(len(labels["val"])),
        )
    l1 = {
        "global_gate": _criterion(global_gate["passed"]),
        "consistency": _criterion(consistency_value is not None and consistency_value >= min_consistency, value=consistency_value, threshold=min_consistency),
        "input_variation": _criterion(score_range > min_margin_range, score_range=score_range, action_varies=action_varies, threshold=min_margin_range),
        "between_within_variance": _criterion(
            variance_ratio is not None and variance_ratio >= min_between_within_ratio,
            value=variance_ratio if variance_ratio is not None and np.isfinite(variance_ratio) else None,
            unbounded=variance_ratio == float("inf"), threshold=min_between_within_ratio),
        "perturbation_effect": _criterion(any(p["passed"] for p in perturbations.values()),
                                           effects=perturbations, threshold=min_delta_margin_effect),
        "constant_input": _criterion(constant is not None and constant["p_value"] < alpha, p_value=None if constant is None else constant["p_value"], alpha=alpha),
    }
    level1 = all(item["passed"] for item in l1.values())

    shuffle_p, mi_effect, agreement, positive, effects = None, None, None, None, []
    if market_state is not None:
        if not isinstance(market_state, pd.Series) or not market_state.index.is_unique:
            raise ValueError("market_state must be a Series uniquely indexed by sample_id")
        states = market_state.reindex(split_samples["val"].sample_id).to_numpy()
        if pd.isna(states).any():
            raise ValueError("market_state is missing validation sample IDs")
        # Reset to the same seed so every seed's null uses the same row shuffle.
        tests = [input_shuffle_permutation_test(mutual_information, states, val["fly_intact"][:, i], n_permutations, seed)
                 for i in range(len(seeds))]
        real = np.array([test["statistic"] for test in tests])
        null = np.column_stack([test["null_samples"] for test in tests])
        effects_array = real - null.mean(axis=0)
        effects = effects_array.tolist()
        mi_effect = float(effects_array.mean())
        agreement = seed_direction_agreement(effects_array)
        positive = float(np.mean(effects_array > 0))
        shuffle_p = empirical_p(float(real.mean()), null.mean(axis=1))
    l2 = {
        "level_1": _criterion(level1),
        "input_shuffle": _criterion(shuffle_p is not None and shuffle_p < alpha, p_value=shuffle_p, effect=mi_effect, alpha=alpha),
        "seed_direction": _criterion(positive is not None and positive >= min_seed_agreement, value=positive, agreement=agreement, effects_by_seed=effects, threshold=min_seed_agreement),
        # Shuffling retains the exact action frequencies, including fixed bias.
        "beyond_fixed_bias": _criterion(action_varies and mi_effect is not None and mi_effect > 0, validation_buy_ratio=p_buy, bias_preserved_by_shuffle=True),
        "nuisance_robustness": _criterion(nuisance is not None and nuisance <= max_nuisance_r2, value=nuisance, threshold=max_nuisance_r2),
    }
    level2 = all(item["passed"] for item in l2.values())

    test, y = panels["test"], labels["test"]
    metrics = {}
    for group, actions in test.items():
        if actions is None:
            continue
        per_seed = [{"seed": s, "balanced_accuracy": balanced_accuracy(y, actions[:, i]),
                     "mcc": mcc(y, actions[:, i]), "precision_by_side": precision_by_side(y, actions[:, i]),
                     "buy_ratio": buy_ratio(actions[:, i]), "longest_run": longest_run(actions[:, i])}
                    for i, s in enumerate(seeds)]
        metrics[group] = {
            "balanced_accuracy": _mean_ba(y, actions), "mcc": float(np.mean([row["mcc"] for row in per_seed])),
            "precision_by_side": precision_by_side(np.repeat(y, len(seeds)), actions.ravel()),
            "buy_ratio": float(actions.mean()), "longest_run": max(row["longest_run"] for row in per_seed),
            "ci95": list(bootstrap_ci(_mean_ba, y, actions, n_bootstrap, seed, confidence=confidence)),
            "confidence": confidence, "per_seed": per_seed,
        }
    metrics["fly_intact"]["mcc_ci95"] = list(bootstrap_ci(_mean_mcc, y, test["fly_intact"], n_bootstrap, seed, confidence=confidence))
    topology = None if test["degree_scramble"] is None else _comparison(y, test["fly_intact"], test["degree_scramble"], n_bootstrap, n_permutations, seed, confidence)
    l3 = {
        "level_2": _criterion(level2),
        "real_connectome": _criterion(connectome_source in {"flywire", "hemibrain"}, value=connectome_source),
        "degree_scramble": _criterion(topology is not None and topology["effect"] > 0, effect=None if topology is None else topology["effect"]),
        "bootstrap_ci": _criterion(topology is not None and topology["ci95"][0] > 0, value=None if topology is None else topology["ci95"], confidence=confidence),
        "seed_direction": _criterion(topology is not None and topology["positive_seed_fraction"] >= min_seed_agreement,
                                     value=None if topology is None else topology["positive_seed_fraction"],
                                     effects_by_seed=[] if topology is None else topology["effects_by_seed"], threshold=min_seed_agreement),
    }
    matched = None if test["matched_random"] is None else _comparison(y, test["fly_intact"], test["matched_random"], n_bootstrap, n_permutations, seed, confidence)
    # A calibrated null also prevents a malformed supplied baseline (e.g. an
    # anticorrelated control) from manufacturing a predictive-signal claim.
    rng = np.random.default_rng(seed)
    null = np.array([balanced_accuracy(y, matched_random(p_buy, len(y), int(rng.integers(2**32))))
                     for _ in range(n_permutations)])
    calibrated_p = empirical_p(metrics["fly_intact"]["balanced_accuracy"], null)
    replication_effect, disjoint, replication_valid = None, False, False
    if (replication_decisions is None) != (replication_samples is None):
        raise ValueError("both replication_decisions and replication_samples are required")
    if replication_decisions is not None:
        rep = _joined(replication_decisions, replication_samples)
        rs = replication_samples.loc[replication_samples.split == "test"].sort_values("sample_id")
        if rs.empty:
            raise ValueError("replication test samples are required")
        rep = rep.loc[rep.split == "test"]
        ra = _panel(rep, "fly_intact", rs.sample_id, seeds)
        rb = _panel(rep, "matched_random", rs.sample_id, seeds)
        if ra is not None and rb is not None:
            replication_effect = _mean_ba(rs.label.to_numpy(), ra) - _mean_ba(rs.label.to_numpy(), rb)
            intact = rep.loc[rep.group == "fly_intact"]
            with np.errstate(over="ignore", invalid="ignore"):
                rep_margin = intact.buy_score.to_numpy(dtype=float) - intact.sell_score.to_numpy(dtype=float)
                replication_valid = bool(np.isfinite(rep_margin).all() and np.ptp(rep_margin) > min_margin_range
                                         and min(np.count_nonzero(ra == 0), np.count_nonzero(ra == 1)) / ra.size >= min_minority_action_ratio)
        if "timestamp" in rs and "timestamp" in split_samples["test"]:
            first = pd.to_datetime(split_samples["test"].timestamp, utc=True, errors="raise")
            second = pd.to_datetime(rs.timestamp, utc=True, errors="raise")
            horizon = pd.Timedelta(minutes=30)  # Locked experiment.yaml: 6 bars x 5 minutes.
            disjoint = bool(not first.isna().any() and not second.isna().any()
                            and (first.max() + horizon <= second.min() or second.max() + horizon <= first.min()))
    l4 = {
        "level_2": _criterion(level2),
        "matched_random": _criterion(matched is not None and matched["effect"] > 0 and matched["effect"] >= min_effect, effect=None if matched is None else matched["effect"],
                                     ci95=None if matched is None else matched["ci95"], validation_buy_ratio=p_buy, threshold=min_effect),
        "permutation": _criterion(matched is not None and matched["p_value"] < alpha and calibrated_p < alpha,
                                  p_value=None if matched is None else matched["p_value"], calibrated_null_p_value=calibrated_p, alpha=alpha),
        "ba_ci": _criterion(metrics["fly_intact"]["ci95"][0] > min_ba_ci_lower,
                            value=metrics["fly_intact"]["ci95"], threshold=min_ba_ci_lower, confidence=confidence),
        "mcc_ci": _criterion(metrics["fly_intact"]["mcc_ci95"][0] > min_mcc_ci_lower,
                             value=metrics["fly_intact"]["mcc_ci95"], threshold=min_mcc_ci_lower, confidence=confidence),
        "seed_direction": _criterion(matched is not None and matched["positive_seed_fraction"] >= min_seed_agreement,
                                     value=None if matched is None else matched["positive_seed_fraction"],
                                     effects_by_seed=[] if matched is None else matched["effects_by_seed"], threshold=min_seed_agreement),
        "independent_period": _criterion(disjoint and replication_valid and replication_effect is not None and replication_effect > 0,
                                         disjoint=disjoint, valid_behavior=replication_valid, effect=replication_effect),
    }
    result.update({f"level_{i}": {"passed": all(c["passed"] for c in criteria.values()), "criteria": criteria}
                   for i, criteria in enumerate((l1, l2, l3, l4), start=1)})
    result.update(highest_level=max([0] + [i for i in range(1, 5) if result[f"level_{i}"]["passed"]]),
                  learning_allowed=level2, primary_metric="balanced_accuracy", metrics=metrics,
                  seeds=seeds, validation_buy_ratio=p_buy,
                  sample_counts={key: len(value) for key, value in labels.items()})
    return result
