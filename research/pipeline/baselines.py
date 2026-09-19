"""Stateless, validation-calibrated controls; no filesystem or global RNG."""

import numpy as np
from scipy.stats import fisher_exact, ks_2samp

from .statistics import _binary, buy_ratio, longest_run


def matched_random(p_buy, n, seed):
    """Bernoulli BUY decisions; p_buy must be fixed from intact validation."""
    if not np.isfinite(p_buy) or not 0 <= p_buy <= 1:
        raise ValueError("p_buy must be between 0 and 1")
    if not isinstance(n, (int, np.integer)) or n < 0:
        raise ValueError("n must be a nonnegative integer")
    return (np.random.default_rng(seed).random(n) < p_buy).astype(np.int8)


def validation_buy_ratio(decisions, samples):
    """Pool only Fly-Intact validation actions across a complete seed grid."""
    ids = samples.loc[samples.split == "val", "sample_id"]
    if ids.empty or ids.isna().any() or ids.duplicated().any():
        raise ValueError("unique validation samples are required")
    rows = decisions.loc[(decisions.group == "fly_intact") & decisions.sample_id.isin(ids)]
    if rows.empty or rows.seed.isna().any() or rows.duplicated(["sample_id", "seed"]).any():
        raise ValueError("unique Fly-Intact validation decisions are required")
    if len(rows) != len(ids) * rows.seed.nunique():
        raise ValueError("Fly-Intact validation decisions must be paired across seeds")
    return buy_ratio(rows.action.to_numpy())


def matched_random_from_validation(decisions, samples, n, seed):
    """Convenience boundary that cannot accidentally calibrate on test."""
    return matched_random(validation_buy_ratio(decisions, samples), n, seed)


def constant_input_profile(actions, buy_score=None, sell_score=None):
    actions = _binary(actions)
    profile = {"n": len(actions), "buy_ratio": buy_ratio(actions), "longest_run": longest_run(actions)}
    if (buy_score is None) != (sell_score is None):
        raise ValueError("both buy_score and sell_score are required")
    if buy_score is not None:
        buy, sell = np.asarray(buy_score, dtype=float), np.asarray(sell_score, dtype=float)
        if buy.shape != actions.shape or sell.shape != actions.shape or not np.isfinite(buy).all() or not np.isfinite(sell).all():
            raise ValueError("scores must be finite vectors aligned with actions")
        margin = buy - sell
        profile.update(score_mean=float(margin.mean()), score_std=float(margin.std()), score_range=float(np.ptp(margin)))
    return profile


def compare_constant_input(
    real_actions, constant_actions, *, real_buy_score=None, real_sell_score=None,
    constant_buy_score=None, constant_sell_score=None,
):
    """Compare score-margin distributions if provided, otherwise BUY rates.

    The preselected two-sided test is KS for scores, Fisher exact for actions;
    do not choose whichever test happens to yield a smaller p-value.
    """
    real = constant_input_profile(real_actions, real_buy_score, real_sell_score)
    constant = constant_input_profile(constant_actions, constant_buy_score, constant_sell_score)
    if (real_buy_score is None) != (constant_buy_score is None):
        raise ValueError("both groups require scores, or neither group")
    if real_buy_score is not None:
        a = np.asarray(real_buy_score) - np.asarray(real_sell_score)
        b = np.asarray(constant_buy_score) - np.asarray(constant_sell_score)
        test = ks_2samp(a, b, alternative="two-sided", method="exact" if min(len(a), len(b)) < 2 else "asymp")
        effect, p_value, method = float(test.statistic), float(test.pvalue), "score_ks"
    else:
        a, b = _binary(real_actions), _binary(constant_actions)
        table = [[int(a.sum()), int(len(a) - a.sum())], [int(b.sum()), int(len(b) - b.sum())]]
        p_value = float(fisher_exact(table, alternative="two-sided").pvalue)
        effect, method = real["buy_ratio"] - constant["buy_ratio"], "action_fisher"
    return {"real": real, "constant": constant, "effect": effect, "p_value": p_value,
            "different": bool(p_value < 0.05), "method": method}
