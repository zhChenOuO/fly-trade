"""Smoke test for FlySimulator connectome trading inference.

Contract (from research/SPEC.md & research.md §4 Step 4):
Verifies:
1. Complete inference on 100 synthetic market images (black, white, uptrend, downtrend, random).
2. No NaN, no overflow, no zero activity.
3. Different inputs cause measurable differences in activity scores.
4. Action outputs are not degenerate (does NOT output >=99% the same action).
5. Prints BUY ratio and score distribution statistics.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import yaml

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.connectome.synthetic import make_synthetic_connectome
from research.pipeline.fly_simulator import FlySimulator
from research.pipeline.action_decoder import decode, ACTION_BUY, ACTION_SELL


def generate_synthetic_dataset(
    n_per_category: int = 20,
    seed: int = 42,
) -> tuple[np.ndarray, list[str]]:
    """Generate synthetic 64x64x3 images across 5 distinct categories without external data.

    Categories:
    - all-black (zeros)
    - all-white (255)
    - uptrend (diagonal price line / candlesticks rising bottom-left to top-right)
    - downtrend (diagonal price line / candlesticks falling top-left to bottom-right)
    - random (uniform random noise)
    """
    rng = np.random.default_rng(seed)
    images = []
    labels = []

    # 1. All-black images
    for _ in range(n_per_category):
        images.append(np.zeros((64, 64, 3), dtype=np.uint8))
        labels.append("black")

    # 2. All-white images
    for _ in range(n_per_category):
        images.append(np.full((64, 64, 3), 255, dtype=np.uint8))
        labels.append("white")

    # 3. Uptrend images (rising green candles)
    for _ in range(n_per_category):
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        start_y = rng.integers(42, 60)
        end_y = rng.integers(5, 22)
        for col in range(64):
            frac = col / 63.0
            y = int(start_y + (end_y - start_y) * frac + rng.integers(-2, 3))
            y = int(np.clip(y, 2, 61))
            # Green candle body
            img[max(0, y - 2) : min(64, y + 3), col, 1] = rng.integers(180, 255)
            # High-low wick
            wick_low = min(63, y + rng.integers(2, 6))
            wick_high = max(0, y - rng.integers(2, 6))
            img[wick_high : wick_low + 1, col, 1] = rng.integers(100, 180)
        images.append(img)
        labels.append("uptrend")

    # 4. Downtrend images (falling red candles)
    for _ in range(n_per_category):
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        start_y = rng.integers(5, 22)
        end_y = rng.integers(42, 60)
        for col in range(64):
            frac = col / 63.0
            y = int(start_y + (end_y - start_y) * frac + rng.integers(-2, 3))
            y = int(np.clip(y, 2, 61))
            # Red candle body
            img[max(0, y - 2) : min(64, y + 3), col, 0] = rng.integers(180, 255)
            # High-low wick
            wick_low = min(63, y + rng.integers(2, 6))
            wick_high = max(0, y - rng.integers(2, 6))
            img[wick_high : wick_low + 1, col, 0] = rng.integers(100, 180)
        images.append(img)
        labels.append("downtrend")

    # 5. Random noise images
    for _ in range(n_per_category):
        images.append(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8))
        labels.append("random")

    return np.stack(images, axis=0), labels


def print_distribution_summary(name: str, values: np.ndarray) -> None:
    """Print standard 5-number summary and distribution metrics."""
    q25, q50, q75 = np.percentile(values, [25, 50, 75])
    print(
        f"  {name:<15s}: mean={values.mean():+8.3f}, std={values.std():8.3f}, "
        f"min={values.min():+8.3f}, 25%={q25:+8.3f}, 50%={q50:+8.3f}, 75%={q75:+8.3f}, max={values.max():+8.3f}"
    )


def run_smoke_test(config_path: str | Path | None = None) -> dict:
    """Execute smoke test according to research/SPEC.md Step 4.

    Raises AssertionError if any validation criterion fails.
    """
    # 1. Load experiment config
    if config_path is None:
        config_path = PROJECT_ROOT / "research" / "config" / "experiment.yaml"

    synth_cfg = {
        "n_neurons": 4000,
        "avg_out_degree": 8.0,
        "long_range_fraction": 0.05,
        "frac_inhibitory": 0.2,
        "n_sensory": 48,
        "n_motor": 12,
        "seed": 0,
    }
    sim_cfg = {"steps": 32, "gain": 1.0, "leak": 0.5, "noise_std": 0.0}

    if Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            if "connectome" in cfg and "synthetic" in cfg["connectome"]:
                synth_cfg.update(cfg["connectome"]["synthetic"])
            if "sim" in cfg:
                sim_cfg.update(cfg["sim"])

    print("=" * 70)
    print("FLYWIRE SIMULATOR SMOKE TEST (Step 4)")
    print("=" * 70)
    print(f"Connectome config: {synth_cfg}")
    print(f"Simulator config:  {sim_cfg}")

    # 2. Build synthetic connectome
    graph = make_synthetic_connectome(**synth_cfg)
    print(f"Graph: {graph.summary()}")

    # 3. Create synthetic test set (100 images: 20 per class)
    images, labels = generate_synthetic_dataset(n_per_category=20, seed=42)
    total_samples = len(images)
    print(f"Generated {total_samples} synthetic images of shape {images.shape[1:]}.")

    # 4. Initialize simulator
    sim = FlySimulator(
        graph=graph,
        seed=synth_cfg.get("seed", 0),
        steps=sim_cfg.get("steps", 32),
        gain=sim_cfg.get("gain", 1.0),
        leak=sim_cfg.get("leak", 0.5),
        noise_std=sim_cfg.get("noise_std", 0.0),
    )

    # 5. Run inference
    scores = sim.run(images)
    buy_score = scores["buy_score"]
    sell_score = scores["sell_score"]
    score_diff = buy_score - sell_score

    # 6. Decode actions
    actions = decode(buy_score, sell_score)
    buy_count = int(np.sum(actions == ACTION_BUY))
    sell_count = int(np.sum(actions == ACTION_SELL))
    buy_ratio = buy_count / total_samples

    print("\n--- Validation Assertions ---")

    # Criterion 1: No NaN
    assert not np.isnan(buy_score).any(), "Found NaN in buy_score!"
    assert not np.isnan(sell_score).any(), "Found NaN in sell_score!"
    print("✓ [PASS] No NaN in scores")

    # Criterion 2: No Overflow / Infinite values
    assert not np.isinf(buy_score).any(), "Found infinite value in buy_score!"
    assert not np.isinf(sell_score).any(), "Found infinite value in sell_score!"
    print("✓ [PASS] No overflow or infinite values")

    # Criterion 3: No Zero Activity
    assert not np.all(buy_score == 0.0), "All buy_scores are zero (no activity)!"
    assert not np.all(sell_score == 0.0), "All sell_scores are zero (no activity)!"
    assert not np.any(buy_score == 0.0), "Found sample with zero buy_score activity!"
    assert not np.any(sell_score == 0.0), "Found sample with zero sell_score activity!"
    print("✓ [PASS] No zero activity (all samples active)")

    # Criterion 4: Score variance / sensitivity to inputs
    assert np.std(buy_score) > 1e-4, f"buy_score std too low: {np.std(buy_score)}"
    assert np.std(sell_score) > 1e-4, f"sell_score std too low: {np.std(sell_score)}"
    assert len(np.unique(buy_score)) > 1, "Scores are identical across different inputs!"
    print(f"✓ [PASS] Input sensitivity confirmed (score std: buy={np.std(buy_score):.2f}, sell={np.std(sell_score):.2f})")

    # Criterion 5: Non-degenerate action distribution (<99% single action)
    assert buy_ratio < 0.99, f"Degenerate policy: BUY ratio is {buy_ratio:.1%} (>= 99%)!"
    assert buy_ratio > 0.01, f"Degenerate policy: BUY ratio is {buy_ratio:.1%} (<= 1%)!"
    print(f"✓ [PASS] Non-degenerate action distribution (BUY ratio = {buy_ratio:.1%}, between 1% and 99%)")

    # 7. Print Category Breakdown
    print("\n--- Per-Category Breakdown ---")
    print(f"{'Category':<12s} {'Count':>6s} {'BUY':>6s} {'SELL':>6s} {'BUY%':>8s} {'Mean Buy':>10s} {'Mean Sell':>10s} {'Mean Diff':>10s}")
    print("-" * 72)
    categories = ["black", "white", "uptrend", "downtrend", "random"]
    for cat in categories:
        cat_idx = [i for i, lbl in enumerate(labels) if lbl == cat]
        c_acts = actions[cat_idx]
        c_buy = buy_score[cat_idx]
        c_sell = sell_score[cat_idx]
        c_diff = score_diff[cat_idx]
        c_b_cnt = int(np.sum(c_acts == ACTION_BUY))
        c_s_cnt = int(np.sum(c_acts == ACTION_SELL))
        c_ratio = c_b_cnt / len(cat_idx)
        print(
            f"{cat:<12s} {len(cat_idx):>6d} {c_b_cnt:>6d} {c_s_cnt:>6d} {c_ratio:>7.1%} "
            f"{c_buy.mean():>+10.2f} {c_sell.mean():>+10.2f} {c_diff.mean():>+10.2f}"
        )
    print("-" * 72)
    print(f"{'TOTAL':<12s} {total_samples:>6d} {buy_count:>6d} {sell_count:>6d} {buy_ratio:>7.1%} "
          f"{buy_score.mean():>+10.2f} {sell_score.mean():>+10.2f} {score_diff.mean():>+10.2f}")

    # 8. Print Score Distributions
    print("\n--- Score Distribution Statistics ---")
    print_distribution_summary("buy_score", buy_score)
    print_distribution_summary("sell_score", sell_score)
    print_distribution_summary("diff (B - S)", score_diff)
    print("=" * 70)
    print("SMOKE TEST PASSED SUCCESSFULLY")
    print("=" * 70)

    return {
        "total_samples": total_samples,
        "buy_count": buy_count,
        "sell_count": sell_count,
        "buy_ratio": buy_ratio,
        "buy_score_mean": float(buy_score.mean()),
        "sell_score_mean": float(sell_score.mean()),
        "buy_score_std": float(buy_score.std()),
        "sell_score_std": float(sell_score.std()),
    }


if __name__ == "__main__":
    run_smoke_test()
